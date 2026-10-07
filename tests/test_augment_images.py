"""Fast regression tests for ``dataset/augment-images.py``."""

import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'dataset' / 'augment-images.py'
SPEC = importlib.util.spec_from_file_location('augment_images', SCRIPT)
AUGMENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUGMENT)


class AugmentImagesTest(unittest.TestCase):
    def test_constant_image_loads_without_nan_or_intensity_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / 'constant.tif'
            Image.fromarray(np.full((5, 7), 42, dtype=np.uint16)).save(image_path)
            image = AUGMENT.load_image_safe(image_path)
        self.assertEqual(image.shape, (5, 7, 3))
        self.assertEqual(image.dtype, np.uint8)
        self.assertTrue(np.all(image == 42))

    def test_annotation_round_trip_filters_invalid_boxes(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / 'region_data_sample.csv'
            written = AUGMENT.write_region_data(
                csv_path,
                np.array(['7', '8']), np.array(['mature', 'eccentric']),
                np.array([2, 9]), np.array([3, 9]),
                np.array([4, 4]), np.array([5, 4]),
                max_height=12, max_width=12,
                image_filename='sample.png')
            parsed = AUGMENT.parse_region_data(csv_path)
            with csv_path.open(newline='') as stream:
                row = next(csv.DictReader(stream))
        self.assertEqual(written, 1)
        self.assertEqual(parsed, (['7'], ['mature'], [2], [3], [4], [5]))
        self.assertEqual(row['filename'], 'sample.png')
        self.assertEqual(row['region_count'], '1')

    def test_legacy_unquoted_class_label_is_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / 'legacy.csv'
            with csv_path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=AUGMENT.CSV_FIELDS)
                writer.writeheader()
                writer.writerow({
                    'filename': 'legacy.png', 'file_size': 'irrelevant',
                    'file_attributes': '{}', 'region_count': 1,
                    'region_id': 0,
                    'region_shape_attributes':
                        '{"name":"rect","x":1,"y":2,"width":3,"height":4}',
                    'region_attributes': '{"particle_class":immature}',
                })
            parsed = AUGMENT.parse_region_data(csv_path)
        self.assertEqual(parsed[1], ['immature'])

    def test_flips_rotation_and_noise_preserve_valid_geometry(self):
        image = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
        box = tuple(np.array([value], dtype=np.int32) for value in (1, 1, 2, 2))

        horizontal = AUGMENT.augment(image, *box, 'horizontal-flip')
        vertical = AUGMENT.augment(image, *box, 'vertical-flip')
        rotated = AUGMENT.augment(image, *box, '180-rotation')
        noisy_a = AUGMENT.augment(
            image, *box, 'gaussian-noise', rng=np.random.default_rng(17))
        noisy_b = AUGMENT.augment(
            image, *box, 'gaussian-noise', rng=np.random.default_rng(17))

        np.testing.assert_array_equal(horizontal[0], image[:, ::-1])
        np.testing.assert_array_equal(horizontal[1], [3])
        np.testing.assert_array_equal(vertical[0], image[::-1])
        np.testing.assert_array_equal(vertical[2], [1])
        np.testing.assert_array_equal(rotated[0], image[::-1, ::-1])
        np.testing.assert_array_equal(rotated[1], [3])
        np.testing.assert_array_equal(rotated[2], [1])
        self.assertEqual(noisy_a[0].dtype, np.uint8)
        np.testing.assert_array_equal(noisy_a[0], noisy_b[0])
        self.assertFalse(np.array_equal(noisy_a[0], image))

    def test_translation_clips_boxes_to_image(self):
        image = np.ones((8, 10, 3), dtype=np.uint8)
        result = AUGMENT.augment(
            image, np.array([1]), np.array([2]),
            np.array([6]), np.array([4]), 'translate-left')
        np.testing.assert_array_equal(result[1], [0])
        np.testing.assert_array_equal(result[3], [2])
        self.assertTrue(np.all(result[0][:, 5:] == 0))

    def test_multicrop_has_unique_edges_and_in_bounds_boxes(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        results = AUGMENT.multicrop_image(
            image, (6, 6), (4, 4),
            np.array(['0']), np.array(['mature']),
            np.array([4]), np.array([4]), np.array([2]), np.array([2]))
        self.assertEqual(len(results[0]), 4)
        for cropped, xs, ys, widths, heights in zip(
                results[0], results[3], results[4], results[5], results[6]):
            self.assertEqual(cropped.shape, (6, 6, 3))
            self.assertTrue(np.all(xs >= 0))
            self.assertTrue(np.all(ys >= 0))
            self.assertTrue(np.all(xs + widths <= 6))
            self.assertTrue(np.all(ys + heights <= 6))

    def test_small_dataset_crop_and_augment_workflow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / 'source' / 'sample'
            source_dir.mkdir(parents=True)
            image = np.full((12, 12, 3), 100, dtype=np.uint8)
            Image.fromarray(image).save(source_dir / 'sample.png')
            AUGMENT.write_region_data(
                source_dir / 'region_data_sample.csv', ['0'], ['mature'],
                [4], [4], [4], [4], 12, 12, image_filename='sample.png')

            crop_count = AUGMENT.expand_images_crops(
                (8, 8), (4, 4), root / 'source', root / 'output')
            augmented_count = AUGMENT.expand_images(
                root / 'output', augmentations=('horizontal-flip',), seed=3)
            generated = sorted(path.name for path in (root / 'output').iterdir())

        self.assertEqual(crop_count, 4)
        self.assertEqual(augmented_count, 4)
        self.assertEqual(len(generated), 8)


if __name__ == '__main__':
    unittest.main()
