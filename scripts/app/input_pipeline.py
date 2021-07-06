from __future__ import absolute_import, division, print_function, unicode_literals
import re, csv, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend
#from config import Stopwatch

"""
parse_region_data: Open a CSV for an annotated dataset and parse information to generate RoIs to validate against
Inputs: 
  csvname, the filename of the CSV you wish to parse
Outputs: idx, lab, x, y, w, h, the respective metadata contained within the CSV
"""
def parse_region_data(csvname):
  # print("input_pypeline: parsing data from file", csvname)
  idx = []
  x   = []
  y   = []
  w   = []
  h   = []
  lab = []
  with open(csvname, newline='') as labels:
    fields = ['#filename',
              'file_size',
              'file_attributes',
              'region_count',
              'region_id',
              'region_shape_attributes',
              'region_attributes']
    reader = csv.DictReader(labels,
      fieldnames=fields,
      dialect='excel',
      quoting=csv.QUOTE_MINIMAL)
    for row in reader:
      lt = row['region_attributes']
      lt = str(re.sub(r'([{}"])','',lt))
      lt = lt.split(':')
      if len(lt) == 2:
        lab.append(lt[1])
      idx.append(row['region_id'])
      tmp = row['region_shape_attributes']
      tmp = str(re.sub(r'([{}"])','',tmp))
      tmp = tmp.split(',')
      for t in tmp:
        tt = t.split(':')
        if len(tt) == 1:
          continue
        else:
          if tt[0] == 'x':
            x.append(int(tt[1]))
          elif tt[0] == 'y':
            y.append(int(tt[1]))
          elif tt[0] == 'height':
            h.append(int(tt[1]))
          elif tt[0] == 'width':
            w.append(int(tt[1]))
  idx.remove('region_id')
  # print("input_pypeling: parsing region data with length (number of GT boxes) ", len(x))
  return idx, lab, x, y, w, h

def change_label_to_num(classes_labels, classes_info):
  """
  Changes classes labels from string to a integer
  Inputs:
    classes_labels: List of labels (strings) to be changed to ints
    classes_info: List of dictionaries with the classes info, following the strucure [{"id":int, "name":str},...]
  Outputs:
    classes_ids: Returns a list of ids corresponding to the input labels
  """
  classes_ids=[]
  #Create a label:id dictionary for all classes in this dataset
  classes_label_id_dict={class_["name"]:class_["id"] for class_ in classes_info}
  #Convert labels to ids
  for label in classes_labels:
    if label in classes_label_id_dict.keys():
      classes_ids.append(classes_label_id_dict[label])
    else:
      raise Exception(f"Label {label} not native to the dataset classes defined, please add it to the class info dict.")
  return np.array(classes_ids).astype('int32')
"""
generate_anchors: Generate all possible anchors within an image
Inputs:
    anchor_sizes, a list of pixel-level lengths for proposed anchors
    anchor_ratios, a list of width:height ratios for proposed anchors
    feature_shapes, list of backbone feature map shapes
    feature_strides, downscaling factors used to calculate feature map shapes
    anchor_stride, stride to use when creating anchor combinations, usually 1
Outputs:
    anchors, a list of anchors of shape [N * len(anchor_size), (y1, x1, y2, x2)]
"""
def generate_anchors(anchor_sizes, anchor_ratios, feature_shapes, feature_strides, anchor_stride, img_shape):
    # print("input_pypeline: generate_anchors")
    anchors = []
    for idx in range(len(anchor_sizes)):
        # All combinations of sizes and ratios
        sizes, ratios = np.meshgrid(anchor_sizes[idx], anchor_ratios)
        sizes = sizes.flatten()
        ratios = ratios.flatten()

        # All combinations of height and width enumerated from scales and ratios
        heights = sizes / np.sqrt(ratios)
        widths = sizes * np.sqrt(ratios)

        # All combinations of shifts in the feature space
        shifts_x = np.arange(0, feature_shapes[idx][1], anchor_stride) * feature_strides[idx]
        shifts_y = np.arange(0, feature_shapes[idx][0], anchor_stride) * feature_strides[idx]
        shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

        # All combinations of shifts, widths and heights to form boxes on the image
        box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
        box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

        # Reshape shifts and shapes to get a list of (y,x) and (h,w) pairs
        box_centers = np.stack([box_centers_y,box_centers_x],axis=2).reshape([-1,2])
        box_sizes = np.stack([box_heights,box_widths],axis=2).reshape([-1,2])
        #And  restructure them to [y1, x1, y2, x2]
        boxes = np.concatenate((box_centers-0.5*box_sizes,
                                box_centers+0.5*box_sizes), axis=1)
        #Adjust boxes so they don't get outside the image
        boxes[boxes<0.] = 0.
        boxes[boxes>img_shape[0]] = img_shape[0]
        anchors.append(boxes)
        """
        x = x.flatten().reshape((-1, 1))
        y = y.flatten().reshape((-1, 1))

        width = width.flatten().reshape((-1, 1))
        height = height.flatten().reshape((-1, 1))
        print("coords for iter#{}: x = {},\ny = {},\nwidth = {},\nheight = {}".format(idx, x, y, width, height))

        # Create the centers coordinates and shapes for the anchors
        bbox_centers = np.concatenate((y, x), axis=1)
        print("bbox_center #{}:\n{}".format(idx, bbox_centers))
        bbox_shapes = np.concatenate((height, width), axis=1)
        print("bbox_shapes #{}:\n{}".format(idx, bbox_shapes))
        # Restructure as [y1, x1, y2, x2]
        bboxes = np.concatenate((bbox_centers - bbox_shapes / 2, bbox_centers + bbox_shapes / 2), axis=1)
        bboxes[bboxes<0.] = 0.
        bboxes[bboxes>img_shape[0]] = img_shape[0]
        print("bbox #{}:\n{}".format(idx, bboxes))
        print(f"anchor size for iteration #{idx}: {anchor_sizes[idx]}")
        print(f"generate_anchors min anchor x for iteration #{idx}: {np.min(bboxes[:,1])}")
        print(f"generate_anchors min anchor y for iteration #{idx}: {np.min(bboxes[:,0])}")
        # Anchors are created for each feature map
        anchors.append(bboxes)
        """
    # print("input_pipeline: generate_anchors done")
    return np.concatenate(anchors, axis=0)

"""
cnn_input_shapes:
Computes the width and height of each stage of the backbone convolutional network
Inputs:
    image_shape: [H,W] The shape of the input image (pixels)
    strides: A list of stride sizes
Outputs:
    shapes: [N, (height, width)] shapes for the backbone network of N stages
"""
def cnn_input_shapes(image_shape, strides):
    # print("input_pipeline: cnn_input_shapes")
    shapes = np.array([[int(np.ceil(image_shape[0] / stride)), int(np.ceil(image_shape[1] / stride))] for stride in strides])
    return shapes

def generate_random_rois(image_shape, count, gt_class_ids, gt_boxes):
    """Generates ROI proposals similar to what the RPN would generate.
    image_shape: [Height, Width, Depth]
    count: Number of ROIs to generate
    gt_class_ids: [N] Integer ground truth class IDs
    gt_boxes: [N, (y1, x1, y2, x2)] Ground truth boxes in pixels.

    Returns: [count, (y1, x1, y2, x2)] ROI boxes in pixels.
    """
    # placeholder
    rois = np.zeros((count, 4), dtype=np.int32)

    # Generate random ROIs around GT boxes (90% of count)
    rois_per_box = int(0.9 * count / gt_boxes.shape[0])
    for i in range(gt_boxes.shape[0]):
        gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[i]
        h = gt_y2 - gt_y1
        w = gt_x2 - gt_x1
        # random boundaries
        r_y1 = max(gt_y1 - h, 0)
        r_y2 = min(gt_y2 + h, image_shape[0])
        r_x1 = max(gt_x1 - w, 0)
        r_x2 = min(gt_x2 + w, image_shape[1])

        # To avoid generating boxes with zero area, we generate double what
        # we need and filter out the extra. If we get fewer valid boxes
        # than we need, we loop and try again.
        while True:
            y1y2 = np.random.randint(r_y1, r_y2, (rois_per_box * 2, 2))
            x1x2 = np.random.randint(r_x1, r_x2, (rois_per_box * 2, 2))
            # Filter out zero area boxes
            threshold = 1
            y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >=
                        threshold][:rois_per_box]
            x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >=
                        threshold][:rois_per_box]
            if y1y2.shape[0] == rois_per_box and x1x2.shape[0] == rois_per_box:
                break

        # Sort on axis 1 to ensure x1 <= x2 and y1 <= y2 and then reshape
        # into x1, y1, x2, y2 order
        x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
        y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
        box_rois = np.hstack([y1, x1, y2, x2])
        rois[rois_per_box * i:rois_per_box * (i + 1)] = box_rois

    # Generate random ROIs anywhere in the image (10% of count)
    remaining_count = count - (rois_per_box * gt_boxes.shape[0])
    # To avoid generating boxes with zero area, we generate double what
    # we need and filter out the extra. If we get fewer valid boxes
    # than we need, we loop and try again.
    while True:
        y1y2 = np.random.randint(0, image_shape[0], (remaining_count * 2, 2))
        x1x2 = np.random.randint(0, image_shape[1], (remaining_count * 2, 2))
        # Filter out zero area boxes
        threshold = 1
        y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >=
                    threshold][:remaining_count]
        x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >=
                    threshold][:remaining_count]
        if y1y2.shape[0] == remaining_count and x1x2.shape[0] == remaining_count:
            break

    # Sort on axis 1 to ensure x1 <= x2 and y1 <= y2 and then reshape
    # into x1, y1, x2, y2 order
    x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
    y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
    global_rois = np.hstack([y1, x1, y2, x2])
    rois[-remaining_count:] = global_rois
    return rois

def build_rpn_targets(anchors, gt_boxes, config):
    """
    build_rpn_targets: Identify the positive and negative anchors and calculate their deltas so they match the corresponding ground truth boxes
    Inputs:
      anchors: [NUM_ANCHORS, (y1,x1,y2,x2)] coordinates of the anchors from generate_anchors
      gt_boxes: [NUM_GT_BOXES, (y1,x1,y2,x2)] coordinates of ground truth boxes parsed from dataset csv files
      config: instance of the Config class
    Outputs:
    rpn_match: [NUM_ANCHORS] matches between anchors and groun truth boxes following the convention:
            1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_bbox: [len(positive_anchors), (dy, dx, log(dh), log(dw))] Deltas for the anchor boxes to fit the GT boxes
    """
    # print("input_pipeline: build_rpn_targets")
    # Prepping array sizes
    rpn_match = np.zeros(anchors.shape[0], dtype=np.int32) #Used to be float but it's not necessary
    rpn_bbox = np.zeros((config.ANCHORS_PER_IMG, 4), dtype=np.float32)

    # Generate IoU values for all anchors and bboxes
    # print("input_pipeline: rpn_targets, calculating IoU scores")
    IoU = calculate_iou_matrix(anchors, gt_boxes) # Shape of [#anchors, #boxes]
    # print(f"Best IoU value {np.max(IoU)} at {np.argmax(IoU)}")
    # print(f"Average IoU value {np.mean(IoU)}")
    # Matching anchors to Ground Truth Boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.3).
    #
    #1. Set negative anchors for anchors where the IoU is less than 0.3 for every GT
    # print("Calculating best boxes and anchors from IoU values")
    anchor_IoU_argmax = np.argmax(IoU, axis = 1) # Size of (#anchors)
    anchor_IoU_max = IoU[np.arange(IoU.shape[0]), anchor_IoU_argmax] # The max IoU for each anchor
    rpn_match[anchor_IoU_max < 0.3] = -1 #Negative anchor designation
    #2. Set an anchor for each GT box (regardless of IoU value)
    # If multiple  anchors have the same IoU match all of them, this means there will be duplicates
    gt_IoU_argmax = np.argwhere(IoU == np.max(IoU, axis=0))[:,0]#Size of (# boxes)
    rpn_match[gt_IoU_argmax] = 1 #Will have duplicates
    #3. Set anchors with high overlap as positive.
    rpn_match[anchor_IoU_max > 0.9] = 1 #Positive anchor designation

    """
    #This is our old implementation, somehow this doesn't need non max supression, still on testing area
    print("Calculating best boxes and anchors from IoU values")
    # Return best bounding box for each anchor
    best_boxes = np.argmax(IoU, axis = 1) # Size of (#anchors)
    # Returns best anchor for every bounding box
    best_anchors = np.argmax(IoU, axis = 0) # Size of (#boxes)
    anchor_IoU = IoU[np.arange(0, IoU.shape[0]), best_boxes] # Shape of #anchors # IS THERE WHERE ITS GOING WRONG
    print("Boxes and anchors:")
    print("Best Boxes: ", best_boxes)
    print("Best Anchors: ", best_anchors)
    print("Best Anchors IoU: ", anchor_IoU)
    print("Average best anchors IoU: ", np.mean(anchor_IoU))
    print("Best IoU for all anchors: ", np.max(IoU))

    rpn_match[anchor_IoU < 0.3] = -1 # Negative anchor designation
    rpn_match[anchor_IoU > 0.9] = 1  # Positive anchor designation
    rpn_match[best_anchors] = 1      # Will have duplicates
    """

    # Subsample to balance positive and negative anchors
    # The positive anchors shouldn't be more than half the anchors
    positive_anchors = np.where(rpn_match==1)[0]
    if len(positive_anchors) > config.ANCHORS_PER_IMG // 2:
        # Remove duplicate positive anchors and set them to neutral
        set_to_zero = np.random.choice(positive_anchors, len(positive_anchors) - config.ANCHORS_PER_IMG //2, replace=False)
        rpn_match[set_to_zero] = 0
        # Reset positive anchors
        positive_anchors = np.where(rpn_match == 1)[0]

    # Remove duplicates for negative proposals so negative_anchors + positive_anchors <= ANCHORS_PER_IMG
    negative_anchors = np.where(rpn_match == -1)[0]
    if len(negative_anchors) > config.ANCHORS_PER_IMG - len(positive_anchors):
        set_to_zero = np.random.choice(negative_anchors, len(negative_anchors) - (config.ANCHORS_PER_IMG - len(positive_anchors)), replace=False)
        rpn_match[set_to_zero] = 0
        # Reset negative anchors
        negative_anchors = np.where(rpn_match == -1)[0]

    #Now target the positive anchors and calculate the shifts and scales needed to transform them to match the corresponging GT boxes, that is, the bboxes (these are the anchor deltas)
    target_anchors = anchors[positive_anchors]

    # TODO: Change this so we use calculate_box_shifts instead
    for idx in range(target_anchors.shape[0]):
        gt_box = gt_boxes[anchor_IoU_argmax[positive_anchors[idx]]]
        anchor = target_anchors[idx]

        # GT Box dimensions and centroids
        # gt_height = abs(gt_box[2] - gt_box[0])
        # gt_width = abs(gt_box[3] - gt_box[1])
        #These should be positive if there a negative value there's something wrong in the csv files where the gt_boxes are parsed from
        gt_height = gt_box[2] - gt_box[0]
        gt_width = gt_box[3] - gt_box[1]
        gt_center_y = np.mean([gt_box[2], gt_box[0]])
        gt_center_x = np.mean([gt_box[3], gt_box[1]])

        # Anchor dimensions and centroids
        anchor_height = anchor[2] - anchor[0]
        anchor_width = anchor[3] - anchor[1]
        anchor_center_y = np.mean([anchor[2], anchor[0]])
        anchor_center_x = np.mean([anchor[3], anchor[1]])

        # Adjustment in normalized coordinates
        # Calculate the bbox (box deltas) that the RPN should predict, these are the adjustements to make the positive anchors fit their respective GT boxes
        # NOTE: INVALID VALUE ENCOUNTER IN LOG RUNTIME WARNING FOR np.log CALLS
        # ^This note was left by Hagan on the previous implementation, which makes me think the error comes from a height or width being negative due to bad csv information, this information is done manually so it's possible
        # Yep, there were some badly labeled data. I've fixed them since, ask me if you need the fixed datasets :D
        #print("BBOX HEIGHT: {}  ANCHOR HEIGHT: {}    BBOX_WIDTH: {}    ANCHOR_WIDTH: {}".format(bbox_height, anchor_height, bbox_width, anchor_width))
        anchor_deltas = np.array([(gt_center_y - anchor_center_y) / anchor_height,
                               (gt_center_x - anchor_center_x) / anchor_width,
                               np.log(gt_height / anchor_height),
                               np.log(gt_width / anchor_width)])
        #print(adjustment)
        #Normalize adjustment using the std dev, THIS DEPENDS ON THE NETWORK USED
        anchor_deltas /= config.RPN_BBOX_STD_DEV
        rpn_bbox[idx] = anchor_deltas
    # print("rpn_match", rpn_match.shape) # shape = #of anchors
    # print("rpn_bbox", rpn_bbox.shape) # shape = config #of anchors per image, 4
    # print("input_pipeline: rpn_targets done")
    return rpn_match, rpn_bbox

def build_detection_targets(rpn_rois, gt_class_ids, gt_boxes, config):
    """Generate targets for training Stage 2 classifier and mask heads.
    This is not used in normal training. It's useful for debugging or to train
    the RCNN heads without using the RPN head.

    Inputs:
    rpn_rois: [N, (y1, x1, y2, x2)] proposal boxes.
    gt_class_ids: [instance count] Integer class IDs
    gt_boxes: [instance count, (y1, x1, y2, x2)]

    Returns:
    rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)]
    class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    bboxes: [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (y, x, log(h), log(w))]. Class-specific bbox refinements.
    """
    assert rpn_rois.shape[0] > 0
    assert gt_class_ids.dtype == np.int32, "Expected int but got {}".format(gt_class_ids.dtype)
    assert gt_boxes.dtype == np.int32, "Expected int but got {}".format(gt_boxes.dtype)

    # Trim empty padding in gt_boxes and gt_masks parts
    instance_ids = np.where(gt_class_ids > 0)[0]
    assert instance_ids.shape[0] > 0, "Image must contain instances."
    gt_class_ids = gt_class_ids[instance_ids]
    gt_boxes = gt_boxes[instance_ids]

    IoU = calculate_iou_matrix(rpn_rois, gt_boxes)
    # Compute areas of ROIs and ground truth boxes.
    # rpn_roi_area = (rpn_rois[:, 2] - rpn_rois[:, 0]) * \
    #     (rpn_rois[:, 3] - rpn_rois[:, 1])
    # gt_box_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * \
    #     (gt_boxes[:, 3] - gt_boxes[:, 1])

    # Compute overlaps [rpn_rois, gt_boxes]
    # overlaps = np.zeros((rpn_rois.shape[0], gt_boxes.shape[0]))
    # for i in range(overlaps.shape[1]):
    #     gt = gt_boxes[i]
    #     overlaps[:, i] = utils.compute_iou(
    #         gt, rpn_rois, gt_box_area[i], rpn_roi_area)

    # Assign ROIs to GT boxes
    rpn_roi_iou_argmax = np.argmax(IoU, axis=1)
    rpn_roi_iou_max = IoU[np.arange(IoU.shape[0]), rpn_roi_iou_argmax]
    # GT box assigned to each ROI
    rpn_roi_gt_boxes = gt_boxes[rpn_roi_iou_argmax]
    rpn_roi_gt_class_ids = gt_class_ids[rpn_roi_iou_argmax]

    # Positive ROIs are those with >= 0.5 IoU with a GT box.
    fg_ids = np.where(rpn_roi_iou_max >= 0.5)[0]

    # Negative ROIs are those with max IoU <0.5 (hard example mining)
    bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]

    # Subsample ROIs. Aim for 33% foreground.
    # FG
    fg_roi_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
    if fg_ids.shape[0] > fg_roi_count:
        keep_fg_ids = np.random.choice(fg_ids, fg_roi_count, replace=False)
    else:
        keep_fg_ids = fg_ids
    # BG
    remaining = config.TRAIN_ROIS_PER_IMAGE - keep_fg_ids.shape[0]
    if bg_ids.shape[0] > remaining:
        keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
    else:
        keep_bg_ids = bg_ids
    # Combine indices of ROIs to keep
    keep = np.concatenate([keep_fg_ids, keep_bg_ids])
    # Need more?
    remaining = config.TRAIN_ROIS_PER_IMAGE - keep.shape[0]
    if remaining > 0:
        # Looks like we don't have enough samples to maintain the desired
        # balance. Reduce requirements and fill in the rest. This is
        # likely different from the Mask RCNN paper.

        # There is a small chance we have neither fg nor bg samples.
        if keep.shape[0] == 0:
            # Pick bg regions with easier IoU threshold
            bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]
            assert bg_ids.shape[0] >= remaining
            keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
            assert keep_bg_ids.shape[0] == remaining
            keep = np.concatenate([keep, keep_bg_ids])
        else:
            # Fill the rest with repeated bg rois.
            keep_extra_ids = np.random.choice(
                keep_bg_ids, remaining, replace=True)
            keep = np.concatenate([keep, keep_extra_ids])
    assert keep.shape[0] == config.TRAIN_ROIS_PER_IMAGE, \
        "keep doesn't match ROI batch size {}, {}".format(
            keep.shape[0], config.TRAIN_ROIS_PER_IMAGE)

    # Reset the gt boxes assigned to BG ROIs.
    rpn_roi_gt_boxes[keep_bg_ids, :] = 0
    rpn_roi_gt_class_ids[keep_bg_ids] = 0

    # For each kept ROI, assign a class_id, and for FG ROIs also add bbox refinement.
    rois = rpn_rois[keep]
    roi_gt_boxes = rpn_roi_gt_boxes[keep]
    roi_gt_class_ids = rpn_roi_gt_class_ids[keep]
    roi_gt_assignment = rpn_roi_iou_argmax[keep]

    # Class-aware bbox deltas. [y, x, log(h), log(w)]
    bboxes = np.zeros((config.TRAIN_ROIS_PER_IMAGE,
                       config.NUM_CLASSES, 4), dtype=np.float32)
    pos_ids = np.where(roi_gt_class_ids > 0)[0]
    bboxes[pos_ids, roi_gt_class_ids[pos_ids]] = calculate_box_shifts(rois[pos_ids], roi_gt_boxes[pos_ids, :4])
    # Normalize bbox refinements
    bboxes /= config.RPN_BBOX_STD_DEV

    return rois, roi_gt_class_ids, bboxes

def calculate_iou_matrix(anchors, gt_boxes):
    """
    calculate_iou_matrix: Creates a jaccard index matrix for each anchor and ground truth bounding box
    Inputs:
      anchors, an array of anchors in the form of bounding box coords
      gt_boxes, the coords of ground truth bounding boxes
    Ouputs:
      iou matrix: [len(anchors),len(gt_boxes)]
    """

    # stopwatch=Stopwatch("iou_matrix")
    y1_bbox, y1_anchor = np.meshgrid(gt_boxes[:, 1], anchors[:, 1])  # higher y value
    x1_bbox, x1_anchor = np.meshgrid(gt_boxes[:, 0], anchors[:, 0])  # lower x value
    y2_bbox, y2_anchor = np.meshgrid(gt_boxes[:, 3], anchors[:, 3])  # lower y value
    x2_bbox, x2_anchor = np.meshgrid(gt_boxes[:, 2], anchors[:, 2])  # higher x value

    boxArea = (x2_bbox - x1_bbox) * (y2_bbox - y1_bbox)
    anchorArea = (x2_anchor - x1_anchor) * (y2_anchor - y1_anchor)

    x1 = np.maximum(x1_bbox, x1_anchor)
    x2 = np.minimum(x2_bbox, x2_anchor)
    y1 = np.maximum(y1_bbox, y1_anchor)
    y2 = np.minimum(y2_anchor, y2_bbox)
    intersection = np.maximum(0, y2-y1) * np.maximum(0, x2-x1)

    union = (boxArea + anchorArea) - intersection
    return intersection / union

def calculate_iou_matrix_tf(anchors, gt_boxes):
    """
    calculate_iou_matrix_tf: Creates a jaccard index matrix for each anchor and ground truth bounding box, using tf function to avoid symbolic Tensor to np array conversion errors
    Inputs:
      anchors, an array of anchors in the form of bounding box coords
      gt_boxes, the coords of ground truth bounding boxes
    Ouputs:
      iou matrix: [len(anchors),len(gt_boxes)]
    """

    # stopwatch=Stopwatch("calculate_iou_matrix_tf")
    y1_bbox, y1_anchor = tf.meshgrid(gt_boxes[:, 1], anchors[:, 1])  # higher y value
    x1_bbox, x1_anchor = tf.meshgrid(gt_boxes[:, 0], anchors[:, 0])  # lower x value
    y2_bbox, y2_anchor = tf.meshgrid(gt_boxes[:, 3], anchors[:, 3])  # lower y value
    x2_bbox, x2_anchor = tf.meshgrid(gt_boxes[:, 2], anchors[:, 2])  # higher x value

    boxArea = (x2_bbox - x1_bbox) * (y2_bbox - y1_bbox)
    anchorArea = (x2_anchor - x1_anchor) * (y2_anchor - y1_anchor)

    x1 = tf.maximum(x1_bbox, x1_anchor)
    x2 = tf.minimum(x2_bbox, x2_anchor)
    y1 = tf.maximum(y1_bbox, y1_anchor)
    y2 = tf.minimum(y2_anchor, y2_bbox)
    intersection = tf.maximum(0., y2-y1) * tf.maximum(0., x2-x1)

    union = (boxArea + anchorArea) - intersection
    return intersection / union

def calculate_box_shifts(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)]. (y2, x2) is
    assumed to be outside the box.
    """
    box = box.astype(np.float32)
    gt_box = gt_box.astype(np.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = np.log(gt_height / height)
    dw = np.log(gt_width / width)

    return np.stack([dy, dx, dh, dw], axis=1)

def calculate_box_shifts_tf(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)]
    """
    box = tf.cast(box, tf.float32)
    gt_box = tf.cast(gt_box, tf.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = tf.math.log(gt_height / height)
    dw = tf.math.log(gt_width / width)

    result = tf.stack([dy, dx, dh, dw], axis=1)
    return result

def shift_bboxes(bboxes, shifts):
    """
    shift_bboxes: Regresses anchors based on predicted bounding box shifts
    Inputs:
      bboxes, array of bounding box coordinates
      shifts, array of bounding box shifts
    Outputs:
      shifted bounding box coords
    """

    # print("input_pipeline: shift_bboxes")
    height = bboxes[:, 2] - bboxes[:, 0]
    width = bboxes[:, 3] - bboxes[:, 1]

    center_y = bboxes[:, 0] + 0.5*height
    center_x = bboxes[:, 1] + 0.5*width

    center_y = shifts[:, 0] * height + center_y
    center_x = shifts[:, 1] * width + center_x

    height = np.exp(shifts[:, 2]) * height
    width = np.exp(shifts[:, 3]) * width

    y1 = center_y - 0.5*height
    y2 = center_y + 0.5*height

    x1 = center_x - 0.5*width
    x2 = center_x + 0.5*width

    return np.stack([y1, x1, y2, x2], axis=1)

def find_matches(gt_boxes, gt_class_ids, pred_boxes, pred_class_ids,pred_scores, iou_threshold=0.5, score_threshold=0.0):
    """
    Finds the matches between given ground truth and predictions taking into account both boxes coordinates and class ids
    Inputs:
      gt_boxes: [MAX_GT_INSTANCES, 4] coordinates of gt instances
      gt_class_ids: [MAX_GT_INSTANCES,1]: integer indexes representing the classes
      pred_boxes: [N, 4] coordinates of predicted boxes
      pred_class_ids: [N,1] ids of predicted classes
      pred_scores: [N,1] rcnn prediction probabilities for classes
      iou_threshold: threshold above which to identify a box coordinate pred to gt match
      score_threshold: theshold above which to identify a class pred to gt match
    Returns:
      gt_match: [MAX_GT_INSTANCES] for each GT contains the index of the best matched pred_box.
      pred_match: [N] for each pred_box contains the index of the best matched GT box.
      iou: [pred_boxes, gt_boxes] IoU scores for each GT, pred_box pair.
    """
    #Trim zeros from padding
    gt_boxes=gt_boxes[~np.all(gt_boxes == 0, axis=1)]
    pred_boxes=pred_boxes[~np.all(pred_boxes == 0, axis=1)]
    #Sorting the predictions by score
    idx = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[idx]
    pred_class_ids = pred_class_ids[idx]
    pred_scores = pred_scores[idx]

    #Calculate the IoU scores between predicted and gt boxes
    IoU = calculate_iou_matrix(pred_boxes, gt_boxes)

    #Loop through pred_boxes and find the matching gt_boxes
    pred_match = -1*np.ones([pred_boxes.shape[0]])
    gt_match = -1*np.ones([gt_boxes.shape[0]])
    for pred_i in range(len(pred_boxes)):
        #For every pred_box find the best matching gt_box
        #1st sort the IoUs by score from high to low
        sorted_idx = np.argsort(IoU[pred_i])[::-1]
        #2nd keep only the predictions above our score_threshold
        sorted_idx = sorted_idx[np.where(IoU[pred_i,sorted_idx]>score_threshold)]
        #3rd find the pred and gt matches
        for gt_j in sorted_idx:
            #If the gt_box already has a match go to the next one
            if gt_match[gt_j] > -1:
                continue
            #And break the loop if we ever go below our iou threshold
            if IoU[pred_i,gt_j] < iou_threshold:
                break
            #Save the pred and gt if their classes match
            if pred_class_ids[pred_i] == gt_class_ids[gt_j]:
                gt_match[gt_j]=pred_i
                pred_match[pred_i]=gt_j
                break
    return gt_match, pred_match, IoU

def compute_mAP(gt_boxes, gt_class_ids, pred_boxes, pred_class_ids,
                pred_scores, iou_threshold=0.5):
    """
    Computes the mean Average Precision for a given IoU threshold (default is 0.5 as it is the standard for COCO dataset)
    Inputs:
        gt_boxes: [MAX_GT_INSTANCES, 4] coordinates of gt instances
        gt_class_ids: [MAX_GT_INSTANCES,1]: integer indexes representing the classes
        pred_boxes: [N, 4] coordinates of predicted boxes
        pred_class_ids: [N,1] ids of predicted classes
        pred_scores: [N,1] rcnn prediction probabilities for classes
        iou_threshold: threshold above which to identify a box coordinate pred to gt match
    Returns:
        mAP: double, mean Average Precision
        precisions: List of precisions at different class score thresholds
        recalls: List of recalls at different class score thresholds (see https://en.wikipedia.org/wiki/Precision_and_recall for reference)
        iou: [pred_boxes, gt_boxes] IoU scores for each GT, pred_box pair.
    """
    #Compute matches and IoU scores
    gt_match, pred_match, IoU = find_matches(gt_boxes, gt_class_ids,
        pred_boxes, pred_class_ids, pred_scores,iou_threshold)
    # Compute precision and recall at each prediction box step
    #pred_match stores the indices of the gt_boxes that best match each pred_box
    #When there are no such matches there's a -1, so to calculate the precisions treat these as false positives
    fp_count = 0
    if len(pred_match) != 0:
        fp_count = np.cumsum(pred_match == -1)[-1]
    #Remember
    #precision=true_positives/(true_positives+false_positives)
    #recall=true_positives/(true_positives+false_negatives)
    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

    # Pad with start and end values to simplify the math in the next step
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    # read https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173 for reference
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1]) #This is choosing the max to the right

    # Compute mean AP over recall range
    # Note we skip the first and last values to avoid the padding
    # And then we add 1 to make sure our indices match the padded array
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    # Remember we calculate the AP from the precision recall curve
    #  AP = \int_0^1 precision(recall) drecall
    # Since the AP is the integral (the area under the curve) 
    # and we made sureour curve is composed of blocks
    # it can be calculated simply as the area of the rectangle (recall_i+1 -recall_i)*precision_i
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])

    return mAP, precisions, recalls, IoU, fp_count

def norm_boxes_tf(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [N, (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in normalized coordinates
    """
    boxes = tf.cast(boxes, dtype=tf.float32)
    # print(f"->norm_boxes_tf shape: {shape}")
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    #tf.print("->norm_boxes_tf h, w:", h, w, output_stream="file:///home/07655/jsreyl/hivclass/scripts/rcnnv3/norm.out")
    #h=backend.print_tensor(h, message="->norm_boxes_tf h:")
    #w=backend.print_tensor(w, message="->norm_boxes_tf w:")
    # print(f"->norm_boxes_tf h: {h} , w: {w}")
    scale = tf.concat([h, w, h, w], axis=-1) # - tf.constant(1.0)
    # shift = tf.constant([0., 0., 1., 1.])
    shift = tf.constant([0., 0., 0., 0.])
    #scale=backend.print_tensor(scale, message="->norm_boxes_tf scale:")
    # print(f"->norm_boxes_tf scale: {scale}")
    #tf.print("->norm_boxes_tf shift:", shift, output_stream="file:///home/07655/jsreyl/hivclass/scripts/rcnnv3/norm.out")
    # shift=backend.print_tensor(shift, message="->norm_boxes_tf shift:")
    # print(f"->norm_boxes_tf shift: {shift}")
    return tf.divide(boxes - shift, scale)

def denorm_boxes_tf(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [N, (y1, x1, y2, x2)] in normalized coordinates
    shape: [..., (height, width)] in pixels

    Returns:
        [N, (y1, x1, y2, x2)] in pixel coordinates
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) # - tf.constant(1.0)
    shift = tf.constant([0., 0., 0., 0.])
    return tf.cast(tf.round(tf.multiply(boxes, scale) + shift), tf.int32)

def norm_boxes(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [N, (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in normalized coordinates
    """
    # print(f"shape: {shape}")
    h, w = shape
    # print(f"h: {h}, w: {w}")
    #scale = np.array([h - 1, w - 1, h - 1, w - 1])
    #shift = np.array([0, 0, 1, 1])
    scale = np.array([h, w, h, w])
    shift = np.array([0, 0, 0, 0])
    return np.divide((boxes - shift), scale).astype(np.float32)

def denorm_boxes(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [N, (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in normalized coordinates
    """
    h, w = shape
    # scale = np.array([h - 1, w - 1, h - 1, w - 1])
    # shift = np.array([0, 0, 1, 1])
    scale = np.array([h, w, h, w])
    shift = np.array([0, 0, 0, 0])
    return np.around(np.multiply(boxes, scale) + shift).astype(np.int32)


# ## Batch Slicing
# Some custom layers support a batch size of 1 only, and require a lot of work
# to support batches greater than 1. This function slices an input tensor
# across the batch dimension and feeds batches of size 1. Effectively,
# an easy way to support batches > 1 quickly with little code modification.
# In the long run, it's more efficient to modify the code to support large
# batches and getting rid of this function. Consider this a temporary solution
def batch_slice(inputs, graph_fn, batch_size, names=None):
    """Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results. It allows you to run a
    graph on a batch of inputs even if the graph is written to support one
    instance only.

    inputs: list of tensors. All must have the same first dimension length
    graph_fn: A function that returns
    a TF tensor that's part of a graph.
    batch_size: number of slices to divide the data into.
    names: If provided, assigns names to the resulting tensors.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    # Change outputs from a list of slices where each is
    # a list of outputs to a list of outputs and each has
    # a list of slices
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]

    return result

def compose_image_data(image_name, original_image_shape, image_shape):
    """
    compose_image_data: Build an array with image metadata
    Inputs:
      image_name: string, name of the image to process for example 'img_num-crops##-aug-name.png'
      original_image_shape: original shape of the dataset image loaded
      image_shape: shape when loaded as an input for the model
    Outputs:
      data: array containing image name, original image shape, processing image shape (the shape inputted to the classifier, i.e. IMAGE_SHAPE in Config), augmentation id and crop id.
    """
    crop_id = 0
    aug_id = 0
    image_id = image_name
    image_id = int(image_name) if image_name.isnumeric() else ord(image_name[-1])
    metadata = image_name.split('-')
    if len(metadata)>=2:
        image_id = int(metadata[0]) if metadata[0].isnumeric() else ord(metadata[0][-1])
        crop_id = ''.join(filter(lambda i: i.isdigit(), metadata[1])) #get the crop number from string 'crop##'
        # print(crop_id)
        if len(metadata)>3:
            #augmentation can be horizontal-flip, vertical-flip, 180-rotation and salt-pepper
            aug_string = '-'.join(metadata[2:])
            if aug_string == 'horizontal-flip':
                aug_id = 1
            elif aug_string == 'vertical-flip':
                aug_id = 2
            elif aug_string == '180-rotation':
                aug_id = 3
            elif aug_string == 'salt-pepper':
                aug_id = 4
            else:
                print(f"compose_image_data: parsing image name with unknown augmentation: {image_name}")
                aug_id = 5
    data = np.array(
        [image_id]+                     # size=1
        list(original_image_shape)+     # size=2
        list(image_shape)+              # size=2
        [crop_id]+
        [aug_id]
    )
    return data

def parse_image_data(data):
    """Parses an array that contains image attributes to its components. See compose_image_data() for more details.
    Inputs:
     data: [batch, data length] where data length is IMAGE_DATA_SIZE
    Outputs:
      Returns a dict of the parsed array.
    """
    image_id = data[:, 0]
    original_image_shape = data[:, 1:3]
    image_shape = data[:, 3:5]
    crop_id = data[:, 5]
    aug_id = data[:, 6]
    return {
        "image_id": image_id,
        "original_image_shape": original_image_shape.astype(np.int32),
        "image_shape": image_shape.astype(np.int32),
        "crop_id": crop_id.astype(np.int32),
        "aug_id": aug_id.astype(np.int32),
    }

def parse_image_data_tf(data):
    """Parses a tensor that contains image attributes to its components.
    See compose_image_data() for more details.

    meta: [batch, data length] where data length depends on NUM_CLASSES

    Returns a dict of the parsed tensors.
    """
    image_id = data[:, 0]
    original_image_shape = data[:, 1:3]
    image_shape = data[:, 3:5]
    crop_id = data[:, 5]
    aug_id = data[:, 6]
    return {
        "image_id": image_id,
        "original_image_shape": original_image_shape,
        "image_shape": image_shape,
        "crop_id": crop_id,
        "aug_id": aug_id,
    }

def build_image_name(data):
  """
  build_image_name: Generates a describing name from a 1-sized batch metadata
  Inputs:
    data: batch metadata of a given image containing info of crop number and augmentation (see compose_image_data)
  Outputs:
    imgName: image name for the metadata in the batch
  """
  imgName=''
  imgData = parse_image_data(data)
  #imgName+=data[:,0]
  #crop_id = data[:,5]
  #aug_id = data[:,6]
  imgName+=str(int(imgData["image_id"]))
  crop_id = int(imgData["crop_id"])
  aug_id = int(imgData["aug_id"])
  # print(aug_id)
  if crop_id !=0:
    imgName+='-crop'+str(crop_id)
  if aug_id !=0:
      if aug_id ==1:
        aug_string = 'horizontal-flip'
      elif aug_id == 2:
        aug_string = 'vertical-flip'
      elif aug_id == 3:
        aug_string = '180-rotation'
      elif aug_id == 4:
        aug_string = 'salt-pepper'
      else:
        print(f"build_image_name: parsing image name with unknown augmentation: {imgName}")
        aug_string = 'unknown'
      imgName+='-'+aug_string
  return imgName

def build_image_names(batch_data):
  """
  build_image_names: Generates an array of describing names from batch metadata
  Inputs:
    batch_data: batch metadata of a given image containing info of crop number and augmentation (see compose_image_data)
  Outputs:
    imgNames: array of image names for the corresponding batch
  """
  imgNames=[]
  imgData = parse_image_data(batch_data)
  for i in range(len(imgData["image_id"])):
    imgName=''
    #imgName+=data[:,0]
    #crop_id = data[:,5]
    #aug_id = data[:,6]
    imgName+=str(int(imgData["image_id"][i]))
    crop_id = int(imgData["crop_id"][i])
    aug_id = int(imgData["aug_id"][i])
    # print(aug_id)
    if crop_id !=0:
      imgName+='-crop'+str(crop_id)
    if aug_id !=0:
        if aug_id ==1:
          aug_string = 'horizontal-flip'
        elif aug_id == 2:
          aug_string = 'vertical-flip'
        elif aug_id == 3:
          aug_string = '180-rotation'
        elif aug_id == 4:
          aug_string = 'salt-pepper'
        else:
          print(f"build_image_names: parsing image name with unknown augmentation: {imgName}")
          aug_string = 'unknown'
        imgName+='-'+aug_string
    imgNames.append(imgName)
  return imgNames

def nms(boxes, scores, class_ids, nms_treshold=0.5, class_id=None):
  """
  nms: Perform non max supression between boxes1 and boxes2
  Inputs:
    boxes: [N, (y1,x1,y2,x2)] numpy array containing the coordinate information of the boxes
    scores: [N] numpy array containing the best score for each of the boxes
    class_ids: [N] numpy array containing the class_id of each box
    nms_treshold: float, value between 0 to 1 to consider an IoU to remove boxes due to overlapping
    class_id: Which class id to perform the nms to
  Outputs:
    nms_idx: indexes of the boxes, scores, class_ids that are valid after nms
  """
  # stopwatch = Stopwatch("nms")
  #Define an array to store the valid box indexes
  nms_idx = []
  if class_id != None:
    local_boxes = np.array(boxes[np.where(class_ids == class_id)])
  else:
    local_boxes = np.array(boxes) #Cheap way to copy an array without using copy.deepcopy
  IoU = calculate_iou_matrix(local_boxes,local_boxes)
  for b_i in range(len(local_boxes)):
    discard = False
    for b_j in range(len(local_boxes)):
      if b_i == b_j: continue
      if IoU[b_i,b_j] > nms_treshold: #If boxes i and j are overlapping
        if scores[b_j] > scores[b_i]: #And the jth box is a better prediction than the ith one
          discard = True #Discard the ith box
        if scores[b_j] == scores[b_i]:#If two boxes have the same score discard only one if there's another already counted in the indexes
          discard = b_j in nms_idx
    if not discard: #If the ith box hasn't been discarded after comparing it with it's brethren add it
      nms_idx.append(b_i)
  return nms_idx

def nms2(boxes, scores, class_ids, nms_treshold=0.5, class_id=None):
  """
  nms2: Perform non max supression between boxes1 and boxes2
  Inputs:
    boxes: [N, (y1,x1,y2,x2)] numpy array containing the coordinate information of the boxes
    scores: [N] numpy array containing the best score for each of the boxes
    class_ids: [N] numpy array containing the class_id of each box
    nms_treshold: float, value between 0 to 1 to consider an IoU to remove boxes due to overlapping
    class_id: Which class id to perform the nms to
  Outputs:
    nms_idx: indexes of the boxes, scores, class_ids that are valid after nms
  """
  # stopwatch = Stopwatch("nms2")
  #Define an array to store the valid box indexes
  nms_idx = []
  if class_id != None:
    local_boxes = np.array(boxes[np.where(class_ids == class_id)])
  else:
    local_boxes = np.array(boxes) #Cheap way to copy an array without using copy.deepcopy
  IoU = calculate_iou_matrix(local_boxes,local_boxes)
  for b_i in range(len(local_boxes)):#For every box
    overlapped_idx = np.where(IoU[b_i]>nms_treshold)[0]#Get those boxes that overlap this one, this includes this b_i box itself
    max_score_idx = np.argmax(scores[overlapped_idx])#Now get the position where the max score between them is found
    max_idx = overlapped_idx[max_score_idx]#Finally get the position of that box in the global array
    #if max_idx not in nms_idx:#And add it if it's not there already
    nms_idx.append(max_idx)
  #Another way is to add them all and then do np unique at the end to avoid making the comparison every step
  nms_idx = np.unique(nms_idx)
  return nms_idx

def nms3(boxes, scores, class_ids, nms_treshold=0.5, class_id=None):
  """
  nms3: Perform non max supression between on boxes based on best score boxes and solves ties by largest area
  Inputs:
    boxes: [N, (y1,x1,y2,x2)] numpy array containing the coordinate information of the boxes
    scores: [N] numpy array containing the best score for each of the boxes
    class_ids: [N] numpy array containing the class_id of each box
    nms_treshold: float, value between 0 to 1 to consider an IoU to remove boxes due to overlapping
    class_id: Which class id to perform the nms to
  Outputs:
    nms_idx: indexes of the boxes, scores, class_ids that are valid after nms
  """
  # stopwatch=Stopwatch("nms3")
  #Define an array to store the valid box indexes
  nms_idx = []
  if len(boxes) == 0: #If there are no boxes to begin with return an empty list
    return nms_idx
  if class_id != None:
    local_boxes = np.array(boxes[np.where(class_ids == class_id)])
  else:
    local_boxes = np.array(boxes) #Cheap way to copy an array without using copy.deepcopy
  #First calculate the IoU of the boxes to see which overlap
  IoU = calculate_iou_matrix(local_boxes,local_boxes)
  for b_i in range(len(local_boxes)):
    overlapped_idx = np.where(IoU[b_i]>nms_treshold)[0]#For those that overlap more than the treshold do the nms
    max_score = np.max(scores[overlapped_idx]) #Find the best score between them
    #max_score_idx = np.argmax(scores[overlapped_idx])
    max_score_idxs = np.argwhere(scores[overlapped_idx] == max_score).flatten() #And where it's located in the array
    if len(max_score_idxs) > 1 : # If more than one box has the best score use the one with the biggest area
      max_boxes = local_boxes[overlapped_idx[max_score_idxs]]
      #print(f'max_boxes:\n {max_boxes}')
      y1_box = max_boxes[:, 0] # higher y value
      x1_box = max_boxes[:, 1] # lower x value
      y2_box = max_boxes[:, 2] # lower y value
      x2_box = max_boxes[:, 3] # higher x value
      boxArea = (x2_box - x1_box) * (y2_box - y1_box)
      #print(f'boxArea:\n {boxArea}')
      max_box_idx = np.argmax(boxArea) #Find the box with the largest area
      max_idx = overlapped_idx[max_score_idxs[max_box_idx]] #And return it's loaction on the global array
    else:
      max_idx = overlapped_idx[max_score_idxs[0]]
    if max_idx not in nms_idx: #Somehow doing individual checks is faster than using np.unique to remove duplicated indexes
      nms_idx.append(max_idx)
  #nms_idx = np.unique(nms_idx)
  return nms_idx
