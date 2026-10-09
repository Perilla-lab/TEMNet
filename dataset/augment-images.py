#!/usr/bin/env python3
"""Crop and augment annotated TEM images while preserving bounding boxes."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


CSV_FIELDS = [
    'filename', 'file_size', 'file_attributes', 'region_count', 'region_id',
    'region_shape_attributes', 'region_attributes',
]
DEFAULT_AUGMENTATIONS = (
    'horizontal-flip', 'vertical-flip', '180-rotation', 'gaussian-noise',
)
SUPPORTED_AUGMENTATIONS = DEFAULT_AUGMENTATIONS + (
    'salt-pepper', 'translate-up', 'translate-down', 'translate-right',
    'translate-left',
)

def load_image_safe(imgname, target_size=None, verbose=False):
    """Load an image as a three-channel ``uint8`` NumPy array."""
    image_path = Path(imgname)
    with Image.open(image_path) as source:
        if verbose:
            print(f'Loading {image_path} with mode {source.mode}')
        if target_size is not None:
            if verbose:
                print(f'Resizing to: {target_size}')
            source = source.resize(
                tuple(target_size), resample=Image.Resampling.NEAREST)
        image = np.asarray(source)

    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]
    if image.ndim not in (2, 3):
        raise ValueError(f'unsupported image shape {image.shape} for {image_path}')
    if verbose:
        print(f'Loaded dtype={image.dtype}, shape={image.shape}, '
              f'range=({image.min()}, {image.max()})')

    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        image_min = float(image.min())
        image_max = float(image.max())
        if image_max > image_min:
            image = np.rint(
                (image - image_min) * (255.0 / (image_max - image_min)))
        else:
            image = np.full(
                image.shape, np.clip(image_min, 0, 255), dtype=np.float32)
        image = image.astype(np.uint8)

    if image.ndim == 2:
        image = np.repeat(image[..., np.newaxis], 3, axis=-1)
    elif image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] != 3:
        raise ValueError(
            f'unsupported channel count {image.shape[-1]} for {image_path}')
    return image


def _json_object(value, field_name, csv_path):
    try:
        parsed = json.loads(value or '{}')
    except json.JSONDecodeError as error:
        # Older TEMNet CSVs stored the class value without JSON quotes, for
        # example {"particle_class":immature}. Accept only that known form.
        prefix = '{"particle_class":'
        if field_name == 'region_attributes' and value.startswith(prefix) \
                and value.endswith('}'):
            label = value[len(prefix):-1].strip().strip('"')
            if label and not any(character in label for character in '{}:,'):
                return {'particle_class': label}
        raise ValueError(
            f'invalid {field_name} JSON in {csv_path}: {value!r}') from error
    if not isinstance(parsed, dict):
        raise ValueError(f'{field_name} must be a JSON object in {csv_path}')
    return parsed

def parse_region_data(csvname):
    """Read VIA rectangle annotations and return IDs, labels, x, y, w, h."""
    csv_path = Path(csvname)
    idx, lab, x, y, w, h = [], [], [], [], [], []
    with csv_path.open(newline='', encoding='utf-8-sig') as labels:
        reader = csv.DictReader(labels)
        present = set(reader.fieldnames or ())
        missing = set(CSV_FIELDS[1:]) - present
        if not ({'filename', '#filename'} & present):
            missing.add('filename')
        if missing:
            raise ValueError(f'{csv_path} is missing columns: {sorted(missing)}')
        for line_number, row in enumerate(reader, start=2):
            shape = _json_object(
                row['region_shape_attributes'],
                'region_shape_attributes', csv_path)
            attributes = _json_object(
                row['region_attributes'], 'region_attributes', csv_path)
            if shape.get('name') != 'rect':
                raise ValueError(
                    f"unsupported region shape {shape.get('name')!r} "
                    f'in {csv_path}:{line_number}')
            try:
                x.append(int(shape['x']))
                y.append(int(shape['y']))
                w.append(int(shape['width']))
                h.append(int(shape['height']))
                lab.append(str(attributes['particle_class']))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f'invalid rectangle in {csv_path}:{line_number}') from error
            idx.append(row['region_id'])
    return idx, lab, x, y, w, h


def write_region_data(
        csvname, idx, lab, x, y, w, h, max_height, max_width,
        image_filename=None):
    """Write valid rectangle annotations in VIA-compatible CSV format."""
    arrays = [np.asarray(values) for values in (idx, lab, x, y, w, h)]
    if len({len(values) for values in arrays}) != 1:
        raise ValueError('annotation arrays must all have the same length')
    indices, labels = arrays[:2]
    xs, ys, widths, heights = [
        values.astype(np.int64) for values in arrays[2:]]
    valid = (
        (xs >= 0) & (ys >= 0) & (widths > 0) & (heights > 0)
        & (xs + widths <= int(max_width))
        & (ys + heights <= int(max_height)))
    indices, labels, xs, ys, widths, heights = [
        values[valid]
        for values in (indices, labels, xs, ys, widths, heights)]

    csv_path = Path(csvname)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    filename = image_filename or (
        csv_path.name.removeprefix('region_data_').replace('.csv', '.png'))
    with csv_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for region_id, label, box_x, box_y, box_w, box_h in zip(
                indices, labels, xs, ys, widths, heights):
            writer.writerow({
                'filename': filename,
                'file_size': 'irrelevant',
                'file_attributes': '{}',
                'region_count': len(xs),
                'region_id': region_id,
                'region_shape_attributes': json.dumps({
                    'name': 'rect', 'x': int(box_x), 'y': int(box_y),
                    'width': int(box_w), 'height': int(box_h),
                }, separators=(',', ':')),
                'region_attributes': json.dumps(
                    {'particle_class': str(label)}, separators=(',', ':')),
            })
    return len(xs)


def _clip_boxes(x, y, w, h, image_height, image_width):
    xs, ys, widths, heights = [
        np.asarray(values, dtype=np.int32) for values in (x, y, w, h)]
    x1 = np.clip(xs, 0, image_width)
    y1 = np.clip(ys, 0, image_height)
    x2 = np.clip(xs + widths, 0, image_width)
    y2 = np.clip(ys + heights, 0, image_height)
    return x1, y1, np.maximum(x2 - x1, 0), np.maximum(y2 - y1, 0)

def augment(
        image, x, y, w, h, atype=None, rng=None, noise_stddev=0.05):
    """Transform an image and its ``(x, y, width, height)`` boxes."""
    if atype not in (None,) + SUPPORTED_AUGMENTATIONS:
        raise ValueError(
            f'unsupported augmentation {atype!r}; '
            f'choose from {SUPPORTED_AUGMENTATIONS}')
    image = np.asarray(image)
    xs, ys, widths, heights = [
        np.asarray(values, dtype=np.int32).copy()
        for values in (x, y, w, h)]
    if len({len(values) for values in (xs, ys, widths, heights)}) != 1:
        raise ValueError('box coordinate arrays must all have the same length')

    image_height, image_width = image.shape[:2]
    if atype is None:
        augmented = image.copy()
    elif atype == 'horizontal-flip':
        augmented = cv2.flip(image, 1)
        xs = image_width - (xs + widths)
    elif atype == 'vertical-flip':
        augmented = cv2.flip(image, 0)
        ys = image_height - (ys + heights)
    elif atype == '180-rotation':
        augmented = cv2.flip(image, -1)
        xs = image_width - (xs + widths)
        ys = image_height - (ys + heights)
    elif atype in ('gaussian-noise', 'salt-pepper'):
        # ``salt-pepper`` is retained as an alias because the legacy function
        # used that name for Gaussian noise.
        generator = rng if rng is not None else np.random.default_rng()
        noise = generator.normal(0.0, noise_stddev * 255.0, image.shape)
        augmented = np.clip(
            image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    else:
        shift_y = {
            'translate-up': -(image_height // 2),
            'translate-down': image_height // 2,
        }.get(atype, 0)
        shift_x = {
            'translate-left': -(image_width // 2),
            'translate-right': image_width // 2,
        }.get(atype, 0)
        transform = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        augmented = cv2.warpAffine(
            image, transform, (image_width, image_height),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
            borderValue=0)
        xs += shift_x
        ys += shift_y

    xs, ys, widths, heights = _clip_boxes(
        xs, ys, widths, heights, image_height, image_width)
    return (
        np.asarray(augmented, dtype=np.uint8),
        xs, ys, widths, heights)

def crop_image_center(image, crop_size, starting_point, idx, lab, x, y, w, h):
    """Crop an image, retaining boxes whose centres lie inside the crop."""
    crop_height, crop_width = map(int, crop_size)
    start_y, start_x = map(int, starting_point)
    xs, ys, widths, heights = [
        np.asarray(values, dtype=np.int32) for values in (x, y, w, h)]
    centre_x = xs + widths / 2.0
    centre_y = ys + heights / 2.0
    keep = (
        (centre_x >= start_x) & (centre_x < start_x + crop_width)
        & (centre_y >= start_y) & (centre_y < start_y + crop_height))
    cropped_x, cropped_y, cropped_w, cropped_h = _clip_boxes(
        xs[keep] - start_x, ys[keep] - start_y,
        widths[keep], heights[keep], crop_height, crop_width)
    return (
        image[start_y:start_y + crop_height,
              start_x:start_x + crop_width].copy(),
        np.asarray(idx)[keep], np.asarray(lab)[keep],
        cropped_x, cropped_y, cropped_w, cropped_h)

def calculate_iou_matrix(anchors, gt_boxes):
    """Return pairwise IoU for ``[x1, y1, x2, y2]`` boxes."""
    anchors = np.asarray(anchors, dtype=np.float32).reshape(-1, 4)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4)
    if not len(anchors) or not len(gt_boxes):
        return np.zeros((len(anchors), len(gt_boxes)), dtype=np.float32)
    x1 = np.maximum(anchors[:, None, 0], gt_boxes[None, :, 0])
    y1 = np.maximum(anchors[:, None, 1], gt_boxes[None, :, 1])
    x2 = np.minimum(anchors[:, None, 2], gt_boxes[None, :, 2])
    y2 = np.minimum(anchors[:, None, 3], gt_boxes[None, :, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    anchor_area = np.maximum(anchors[:, 2] - anchors[:, 0], 0) * np.maximum(
        anchors[:, 3] - anchors[:, 1], 0)
    gt_area = np.maximum(gt_boxes[:, 2] - gt_boxes[:, 0], 0) * np.maximum(
        gt_boxes[:, 3] - gt_boxes[:, 1], 0)
    union = anchor_area[:, None] + gt_area[None, :] - intersection
    return np.divide(
        intersection, union, out=np.zeros_like(intersection), where=union > 0)

def crop_image(
        image, crop_size, starting_point, idx, lab, x, y, w, h,
        iou_threshold=0.75):
    """Crop an image and retain boxes at least ``iou_threshold`` visible."""
    crop_height, crop_width = map(int, crop_size)
    start_y, start_x = map(int, starting_point)
    image_height, image_width = image.shape[:2]
    if crop_height <= 0 or crop_width <= 0:
        raise ValueError('crop dimensions must be positive')
    if crop_height > image_height or crop_width > image_width:
        raise ValueError(
            f'crop size {(crop_height, crop_width)} exceeds image shape '
            f'{image.shape[:2]}')
    if not (0 <= start_y <= image_height - crop_height
            and 0 <= start_x <= image_width - crop_width):
        raise ValueError(f'invalid crop origin {(start_y, start_x)}')

    indices, labels = np.asarray(idx), np.asarray(lab)
    xs, ys, widths, heights = [
        np.asarray(values, dtype=np.int32) for values in (x, y, w, h)]
    if len({len(values) for values in (
            indices, labels, xs, ys, widths, heights)}) != 1:
        raise ValueError('annotation arrays must all have the same length')

    end_x, end_y = start_x + crop_width, start_y + crop_height
    clipped_x1 = np.maximum(xs, start_x)
    clipped_y1 = np.maximum(ys, start_y)
    clipped_x2 = np.minimum(xs + widths, end_x)
    clipped_y2 = np.minimum(ys + heights, end_y)
    clipped_w = np.maximum(clipped_x2 - clipped_x1, 0)
    clipped_h = np.maximum(clipped_y2 - clipped_y1, 0)
    original_area = widths.astype(np.float64) * heights.astype(np.float64)
    retained_fraction = np.divide(
        clipped_w * clipped_h, original_area,
        out=np.zeros_like(original_area), where=original_area > 0)
    keep = (
        (clipped_w > 0) & (clipped_h > 0)
        & (retained_fraction > iou_threshold))

    cropped = image[
        start_y:start_y + crop_height,
        start_x:start_x + crop_width].copy()
    return (
        cropped, indices[keep], labels[keep],
        clipped_x1[keep] - start_x, clipped_y1[keep] - start_y,
        clipped_w[keep], clipped_h[keep])

def _crop_origins(length, crop_length, step):
    if step <= 0:
        raise ValueError('crop step dimensions must be positive')
    if crop_length > length:
        raise ValueError(
            f'crop dimension {crop_length} exceeds image dimension {length}')
    last = length - crop_length
    # Preserve the legacy grid exactly: iterate ceil(length / step) times and
    # clamp windows that cross the far edge back to the last valid origin.
    # This deliberately retains repeated edge positions. For a 2620x4000
    # image, 1024x1024 crops, and a 500x500 step, that is a 6x8 grid (48
    # candidates), even though only 5x7=35 origins are spatially unique.
    return [
        min(index * step, last)
        for index in range((length + step - 1) // step)
    ]

def multicrop_image(image, crop_size, crop_step, idx, lab, x, y, w, h):
    """Create the legacy overlapping crop grid, omitting empty crops."""
    y_origins = _crop_origins(
        image.shape[0], int(crop_size[0]), int(crop_step[0]))
    x_origins = _crop_origins(
        image.shape[1], int(crop_size[1]), int(crop_step[1]))
    outputs = [[] for _ in range(7)]
    # Match the original numbering: walk down Y for each X column.
    for start_x in x_origins:
        for start_y in y_origins:
            result = crop_image(
                image, crop_size, (start_y, start_x),
                idx, lab, x, y, w, h)
            if len(result[1]):
                for output, value in zip(outputs, result):
                    output.append(value)
    return tuple(outputs)


def _image_ids(read_path, excluded_terms=()):
    root = Path(read_path)
    if not root.is_dir():
        raise FileNotFoundError(f'dataset directory does not exist: {root}')
    return sorted(
        path.name for path in root.iterdir()
        if path.is_dir()
        and not any(term in path.name for term in excluded_terms))

def _source_paths(read_path, image_id):
    image_dir = Path(read_path) / image_id
    candidates = [
        image_dir / f'{image_id}{extension}'
        for extension in ('.tif', '.tiff', '.png')]
    image_path = next((path for path in candidates if path.is_file()), None)
    if image_path is None:
        raise FileNotFoundError(
            f'no .tif, .tiff, or .png image found for {image_id}')
    csv_path = image_dir / f'region_data_{image_id}.csv'
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    return image_path, csv_path

def _load_annotations(csv_path):
    values = parse_region_data(csv_path)
    return (
        np.asarray(values[0]), np.asarray(values[1]),
        *(np.asarray(value, dtype=np.int32) for value in values[2:]))

def _save_dataset_item(output_root, item_id, image, annotations, rewrite):
    output_dir = Path(output_root) / item_id
    image_path = output_dir / f'{item_id}.png'
    csv_path = output_dir / f'region_data_{item_id}.csv'
    if output_dir.exists() and not rewrite:
        return False
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(image_path)
    idx, lab, x, y, w, h = annotations
    write_region_data(
        csv_path, idx, lab, x, y, w, h,
        image.shape[0], image.shape[1], image_filename=image_path.name)
    return True

def expand_images(
        read_path, rewrite=True, augmentations=DEFAULT_AUGMENTATIONS,
        seed=None):
    """Apply selected augmentations to every non-augmented dataset item."""
    augmentations = tuple(augmentations)
    unsupported = set(augmentations) - set(SUPPORTED_AUGMENTATIONS)
    if unsupported:
        raise ValueError(f'unsupported augmentations: {sorted(unsupported)}')
    image_ids = _image_ids(read_path, SUPPORTED_AUGMENTATIONS)
    generator = np.random.default_rng(seed)
    saved = 0
    for image_id in image_ids:
        image_path, csv_path = _source_paths(read_path, image_id)
        image = load_image_safe(image_path)
        idx, lab, x, y, w, h = _load_annotations(csv_path)
        for augmentation_name in augmentations:
            transformed = augment(
                image, x, y, w, h, augmentation_name, rng=generator)
            item_id = f'{image_id}-{augmentation_name}'
            if _save_dataset_item(
                    read_path, item_id, transformed[0],
                    (idx, lab, *transformed[1:]), rewrite):
                saved += 1
    print(f'Saved {saved} augmented dataset items to {read_path}')
    return saved

def expand_images_crops(
        crop_size, step_size, read_path, write_path, rewrite=True):
    """Crop each dataset image and save crops containing retained boxes."""
    saved = 0
    for image_id in _image_ids(read_path):
        image_path, csv_path = _source_paths(read_path, image_id)
        image = load_image_safe(image_path)
        annotations = _load_annotations(csv_path)
        crops = multicrop_image(
            image, crop_size, step_size, *annotations)
        for crop_number, values in enumerate(zip(*crops)):
            cropped_image, idx, lab, x, y, w, h = values
            item_id = f'{image_id}-crop{crop_number}'
            if _save_dataset_item(
                    write_path, item_id, cropped_image,
                    (idx, lab, x, y, w, h), rewrite):
                saved += 1
    print(f'Saved {saved} cropped dataset items to {write_path}')
    return saved

def _pair(value):
    try:
        first, second = (int(item) for item in value.split(','))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            'expected two integers formatted as HEIGHT,WIDTH') from error
    if first <= 0 or second <= 0:
        raise argparse.ArgumentTypeError('dimensions must be positive')
    return first, second

def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--mode', choices=('all', 'crop', 'augment'), default='all',
        help='operation to run (default: %(default)s)')
    parser.add_argument(
        '--dataset-root', type=Path, default=Path('rcnn_dataset_full'),
        help='root containing train/ and val/ (default: %(default)s)')
    parser.add_argument(
        '--output-root', type=Path, default=Path('rcnn_dataset_augmented'),
        help='root for cropped train/ and val/ (default: %(default)s)')
    parser.add_argument(
        '--crop-size', type=_pair, default=(1024, 1024), metavar='H,W')
    parser.add_argument(
        '--step-size', type=_pair, default=(500, 500), metavar='Y,X')
    parser.add_argument(
        '--augmentations', nargs='+', choices=SUPPORTED_AUGMENTATIONS,
        default=list(DEFAULT_AUGMENTATIONS))
    parser.add_argument('--seed', type=int, default=0, help='noise RNG seed')
    parser.add_argument(
        '--rewrite', action=argparse.BooleanOptionalAction, default=True,
        help='replace existing output files')
    return parser

def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    train_input = args.dataset_root / 'train'
    val_input = args.dataset_root / 'val'
    train_output = args.output_root / 'train'
    val_output = args.output_root / 'val'
    if args.mode in ('all', 'crop'):
        expand_images_crops(
            args.crop_size, args.step_size,
            train_input, train_output, args.rewrite)
        expand_images_crops(
            args.crop_size, args.step_size,
            val_input, val_output, args.rewrite)
    if args.mode in ('all', 'augment'):
        expand_images(
            train_output, args.rewrite, args.augmentations, seed=args.seed)


if __name__ == '__main__':
    main()
