import os, time, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #Reduce Tensorflow messages
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1,2,3')
import tensorflow as tf
import visualize as V
from model import RCNN
from config import Config, Dataset

# ********* SETUP *********
print("TensorFlow Version", tf.__version__)
#tf.debugging.set_log_device_placement(True) #Enable for device debugging
mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice(reduce_to_device="cpu:0"))
print('Number of devices recognized by Mirror Strategy: ', mirrored_strategy.num_replicas_in_sync)

"""
compound_training: Run the training procedures multiple times consecutively
Inputs:
    model, the training Keras sequential models
    dataset, the dataset built by Dataset object
    iterations, the number of overall training procedures
"""
def compound_training(model, dataset, iterations):
    train_losses = []
    val_losses = []
    for i in range((iterations)):
        if(i != 0):
            model.keras_model.load_weights(model.config.WEIGHT_SET, by_name=True)
        print("Compound weights loaded!")
        hist, lrm = model.train(dataset)
        train_losses.append(hist.history['loss'])
        val_losses.append(hist.history['val_loss'])
    V.visualize_compound_training(train_losses, val_losses)

def save_inference_compatible_weights(model, filepath):
    """Write named legacy HDF5 weights loadable across train/inference graphs."""
    import h5py
    from keras.src.legacy.saving.legacy_h5_format import (
        save_weights_to_hdf5_group,
    )

    with h5py.File(filepath, "w") as handle:
        save_weights_to_hdf5_group(handle, model)


"""
train_model: Run a single training procedure for RPN model
"""
def train_model(backbone='temnet', weights_path=None, n_gpu='0', train_path=None, val_path=None, epochs=None, batch_size=None, dataset_image_size=None, output_path=None):
    if train_path is not None:
        Config.TRAIN_PATH = os.path.abspath(train_path)
    if val_path is not None:
        Config.VAL_PATH = os.path.abspath(val_path)
    with(tf.device('/GPU:'+n_gpu)):
    #with mirrored_strategy.scope():
        config = Config(backbone=backbone)
        config.GPU_COUNT = 1
        if epochs is not None:
            config.EPOCHS = epochs
        if batch_size is not None:
            config.BATCH_SIZE = batch_size
        if dataset_image_size is not None:
            config.DATASET_IMAGE_SIZE = tuple(dataset_image_size)
        if output_path is not None:
            config.WEIGHT_PATH = os.path.abspath(output_path)
        os.makedirs(config.WEIGHT_PATH, exist_ok=True)
        print(f"Training configuration: train={config.TRAIN_PATH}, val={config.VAL_PATH}, epochs={config.EPOCHS}, batch_size={config.BATCH_SIZE}, dataset_image_size={config.DATASET_IMAGE_SIZE}, output={config.WEIGHT_PATH}")
        print(f"Training for RPN: {config.TRAIN_ONLY_RPN}")
        dataset = {"train": Dataset(config.TRAIN_PATH, config, "train"), "validation": Dataset(config.VAL_PATH, config, "validation")}
        rcnn = RCNN(config, 'train')
        if config.TRAIN_ONLY_RPN:
            print("--------------------Training RPN model ------------------")
        else:
            print("--------------------Training RCNN model ------------------")
        if weights_path != None:
            print(f"Reading weights from {weights_path} ...")
            try:
                rcnn.load_weights(weights_path, by_name=True)
            except:
                print(f"Could not load weights, resorting back to imagenet pretrained weights ...")
                weights_path = rcnn.get_imagenet_weights(backbone=backbone)
                rcnn.load_weights(weights_path, by_name=True)
        else:
            print("No weights loaded.")
        hist, lrm = rcnn.train(dataset)
        inference_weights_path = os.path.join(
            config.WEIGHT_PATH,
            f"rcnn_{config.BACKBONE}_trained_inference_compatible.hdf5",
        )
        save_inference_compatible_weights(
            rcnn.keras_model, inference_weights_path)
        print(f"Inference-compatible weights saved to {inference_weights_path}")
        print("--------------------RCNN model trained---------------------")
        V.visualize_benchmarks(hist.history['loss'], hist.history['val_loss'], config)
        V.visualize_learning_rate(lrm.lrates, config)
        print(f"TRAIN_SCRIPT_RESULT train_images={len(dataset['train'].image_ids)} validation_images={len(dataset['validation'].image_ids)} train_steps={len(dataset['train'])} validation_steps={len(dataset['validation'])} loss={hist.history['loss'][-1]} val_loss={hist.history['val_loss'][-1]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--backbone", help="Backbone to train, options are \'temnet\', \'resnet101\' or \'resnet101v2\', mind weights are different for each model", default='temnet')
    parser.add_argument("-g", "--gpu", help="Number of the GPU to use for training", default='0')
    parser.add_argument("-w", "--weights", help="Path to starting weights to use for training", default=None)
    parser.add_argument("--train-path", help="Training dataset directory", default=None)
    parser.add_argument("--val-path", help="Validation dataset directory", default=None)
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=None)
    parser.add_argument("--batch-size", help="Images per training batch", type=int, default=None)
    parser.add_argument("--dataset-image-size", help="Original image height and width", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"), default=None)
    parser.add_argument("--output-path", help="Directory for training checkpoints", default=None)
    args = parser.parse_args()
    start=time.perf_counter()
    train_model(args.backbone, args.weights, args.gpu, args.train_path, args.val_path, args.epochs, args.batch_size, args.dataset_image_size, args.output_path)
    finish=time.perf_counter()
    print(f"Finished in {round(finish-start,2)} seconds")
