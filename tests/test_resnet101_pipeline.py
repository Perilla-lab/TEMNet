"""Hardware-accelerated integration test for RCNN training and inference.

The test uses the local full dataset and released HDF5 checkpoint, so it is
opt-in rather than part of the fast unit-test suite:

    TEMNET_RUN_INTEGRATION=1 TEMNET_REQUIRE_GPU=1 \
    TEMNET_TEST_BACKBONE=resnet101 \
        python -m unittest tests.test_resnet101_pipeline -v

Set TEMNET_TEST_BACKBONE to "temnet", "resnet101v2", or
"inception_resnetv2" to exercise that backbone.
"""

import json
import gc
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
RCNN_SCRIPTS = ROOT / "scripts" / "rcnn"
if str(RCNN_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(RCNN_SCRIPTS))

from config import Config, Dataset
import input_pipeline as I
from model import RCNN


RUN_INTEGRATION = os.environ.get(
    "TEMNET_RUN_INTEGRATION",
    os.environ.get("TEMNET_RUN_RESNET101_INTEGRATION", "0"),
) == "1"
REQUIRE_GPU = os.environ.get("TEMNET_REQUIRE_GPU") == "1"
BACKBONE = os.environ.get("TEMNET_TEST_BACKBONE", "resnet101")
WEIGHT_FILES = {
    "inception_resnetv2": "rcnn_inception_resnetv2_weights_res512.hdf5",
    "resnet101": "rcnn_resnet101_weights_res512.hdf5",
    "resnet101v2": "rcnn_resnet101v2_weights_full_res512.hdf5",
    "temnet": "rcnn_temnet_weights_gn_res512.hdf5",
}
if BACKBONE not in WEIGHT_FILES:
    raise ValueError(
        f"unsupported TEMNET_TEST_BACKBONE={BACKBONE!r}; "
        f"choose one of {sorted(WEIGHT_FILES)}")
WEIGHTS = ROOT / "weights" / WEIGHT_FILES[BACKBONE]
TRAIN_PATH = ROOT / "dataset" / "rcnn_dataset_full" / "train"
VAL_PATH = ROOT / "dataset" / "rcnn_dataset_full" / "val"
RESULTS_DIR = Path(os.environ.get(
    "TEMNET_TEST_RESULTS_DIR",
    ROOT / "tests" / "results",
))


def save_prediction_image(image, prediction, config, output_path):
    """Draw detected boxes, class labels, and scores on a PIL image."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    class_names = {
        class_info["id"]: class_info["name"]
        for class_info in config.CLASS_INFO
    }
    colors = {1: "yellow", 2: "lime", 3: "cyan"}
    width, height = annotated.size

    for box, class_id, score in zip(
            prediction["rois"],
            prediction["class_ids"],
            prediction["scores"]):
        y1, x1, y2, x2 = (int(value) for value in box)
        x1 = min(max(x1, 0), width - 1)
        x2 = min(max(x2, 0), width - 1)
        y1 = min(max(y1, 0), height - 1)
        y2 = min(max(y2, 0), height - 1)
        color = colors.get(int(class_id), "red")
        label = f"{class_names.get(int(class_id), class_id)} {score:.3f}"
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        text_box = draw.textbbox((x1 + 2, y1 + 2), label)
        draw.rectangle(text_box, fill=color)
        draw.text((x1 + 2, y1 + 2), label, fill="black")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(output_path)
    return output_path


class DetectorTestConfig(Config):
    TRAIN_PATH = str(TRAIN_PATH)
    VAL_PATH = str(VAL_PATH)
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    BATCH_SIZE = 1
    EPOCHS = 1
    DATASET_IMAGE_SIZE = (2620, 4000)
    MAX_GT_INSTANCES = 100


@unittest.skipUnless(
    RUN_INTEGRATION,
    "set TEMNET_RUN_INTEGRATION=1 to run the real-data test",
)
class DetectorPipelineTest(unittest.TestCase):
    """Exercises one real crop inference and one real optimizer step."""

    @classmethod
    def setUpClass(cls):
        cls.logical_gpus = tf.config.list_logical_devices("GPU")
        if REQUIRE_GPU and not cls.logical_gpus:
            raise RuntimeError("TEMNET_REQUIRE_GPU=1 but TensorFlow found no GPU")
        print("RUNTIME " + json.dumps({"tensorflow": tf.__version__,
              "backbone": BACKBONE, "weights": str(WEIGHTS),
              "logical_gpus": [gpu.name for gpu in cls.logical_gpus]}))
        missing = [path for path in (WEIGHTS, TRAIN_PATH, VAL_PATH) if not path.exists()]
        if missing:
            raise unittest.SkipTest(
                "missing local integration-test assets: "
                + ", ".join(str(path) for path in missing)
            )
        tf.keras.utils.set_random_seed(20261006)

    def tearDown(self):
        tf.keras.backend.clear_session()
        gc.collect()

    def test_inference_and_training(self):
        config = DetectorTestConfig(BACKBONE)

        inference_model = RCNN(config, "inference")
        inference_model.load_weights(str(WEIGHTS), by_name=True)
        inference_variable = inference_model.keras_model.trainable_variables[0]
        inference_device = inference_variable.value.device
        if REQUIRE_GPU:
            self.assertIn("GPU", inference_device.upper())

        image_id = "0131002"
        image_path = VAL_PATH / image_id / f"{image_id}.png"
        with Image.open(image_path) as source:
            crop = source.convert("RGB").crop((0, 0, 1024, 1024))
            resized = crop.resize((512, 512), Image.Resampling.NEAREST)
            image = np.asarray(resized, dtype=np.float32)[None, ...]

        image_data = np.asarray(
            [I.compose_image_data(image_id, (1024, 1024), (512, 512))],
            dtype=np.float32,
        )
        prediction = inference_model.predict_batch(image, image_data)[0]
        detection_count = len(prediction["scores"])

        for name, values in prediction.items():
            self.assertTrue(np.isfinite(values).all(), f"non-finite {name}")
        self.assertEqual(prediction["rois"].shape, (detection_count, 4))
        self.assertEqual(prediction["class_ids"].shape, (detection_count,))
        self.assertEqual(prediction["scores"].shape, (detection_count,))
        prediction_path = RESULTS_DIR / (
            f"{BACKBONE}_prediction_{image_id}_crop0.png")
        save_prediction_image(
            resized, prediction, config, prediction_path)
        self.assertTrue(prediction_path.is_file())
        print("INFERENCE_RESULT " + json.dumps({
            "backbone": BACKBONE,
            "device": inference_device,
            "detections": detection_count,
            "max_score": (float(np.max(prediction["scores"]))
                          if detection_count else None),
            "min_confidence": config.DETECTION_MIN_CONFIDENCE,
            "prediction_image": str(prediction_path),
        }))

        del inference_model
        tf.keras.backend.clear_session()
        gc.collect()

        with tempfile.TemporaryDirectory(prefix=f"temnet-{BACKBONE}-test-") as temp_dir:
            config.WEIGHT_PATH = temp_dir
            training_model = RCNN(config, "train")
            training_model.load_weights(str(WEIGHTS), by_name=True)

            train_data = Dataset(str(TRAIN_PATH), config, "train")
            train_data.image_ids = ["0131001"]
            inputs, targets = train_data[0]
            self.assertIsInstance(inputs, tuple)
            self.assertIsInstance(targets, tuple)
            self.assertGreater(float(np.max(inputs[0])), 1.0)

            kernel = training_model.keras_model.get_layer(
                "rcnn_class_logits"
            ).weights[0]
            kernel_device = kernel.value.device
            if REQUIRE_GPU:
                self.assertIn("GPU", kernel_device.upper())
            before = kernel.numpy().copy()
            history = training_model.keras_model.fit(
                train_data,
                steps_per_epoch=1,
                epochs=1,
                verbose=0,
            )

            loss = np.asarray(history.history["loss"], dtype=np.float64)
            max_update = float(np.max(np.abs(kernel.numpy() - before)))
            optimizer_iterations = int(
                training_model.keras_model.optimizer.iterations.numpy())
            self.assertTrue(np.isfinite(loss).all(), f"non-finite loss: {loss}")
            self.assertGreater(
                max_update,
                0.0,
                "classifier weights did not update",
            )
            self.assertEqual(
                optimizer_iterations,
                1,
            )
            gpu_memory = (tf.config.experimental.get_memory_info("GPU:0")
                          if self.logical_gpus else {})
            print("TRAINING_RESULT " + json.dumps({
                "backbone": BACKBONE,
                "device": kernel_device,
                "loss": float(loss[-1]),
                "classifier_kernel_max_update": max_update,
                "optimizer_iterations": optimizer_iterations,
                "gpu_memory_bytes": gpu_memory,
            }))


if __name__ == "__main__":
    unittest.main()
