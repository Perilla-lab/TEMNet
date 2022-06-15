import os, sys, argparse
import numpy as np
import tensorflow as tf
from tensorflow.python.eager import context #To check whether we're executing eagerly or not
#tf.compat.v1.enable_eager_execution()
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
import tensorflow.keras.optimizers as KO
import tensorflow.keras.regularizers as KR
from tensorflow.keras import backend
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ReduceLROnPlateau

import helpers as H
import input_pipeline as I

from config import Config, Dataset
from addons import GroupNormalization
#tf.compat.v1.disable_eager_execution()


def trim_zeros(boxes, name='trim_zeros'):
    """Often boxes are represented with matrices of shape [N, 4] and
    are padded with zeros. This removes zero boxes.

    boxes: [N, 4] matrix of boxes.
    non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """
    non_zeros = tf.cast(tf.reduce_sum(input_tensor=tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(tensor=boxes, mask=non_zeros, name=name)
    return boxes, non_zeros

############################################################
#  Learning Rate Trackers
############################################################

# monitor the learning rate
class LearningRateMonitor(Callback):
	# start of training
	def on_train_begin(self, logs={}):
		self.lrates = list()

	# end of each training epoch
	def on_epoch_end(self, epoch, logs={}):
		# get and store the learning rate
		optimizer = self.model.optimizer
		lrate = float(backend.get_value(self.model.optimizer.lr))
		self.lrates.append(lrate)



############################################################
#  Proposal Layer
############################################################

# This is the same as the shift bboxes in input_pipeline, but this version uses tf.
# Use this for now but check if the np version works aswell
def apply_box_deltas(boxes, deltas):
    """Applies the given deltas (bboxes) to the given boxes. So they fit gt_boxes coordinates better
    boxes: [N, (y1, x1, y2, x2)] boxes to update
    deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    return result

class ProposalLayer(KL.Layer):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinement deltas (bboxes) to anchors.

    Inputs:
        rpn_probs: [batch, num_anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, num_anchors, (dy, dx, log(dh), log(dw))]
        anchors: [batch, num_anchors, (y1, x1, y2, x2)] anchors in normalized coordinates

    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """

    def __init__(self, proposal_count, nms_threshold, config=None, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.config = config
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold

    def get_config(self):
        # Add proposal_count and nms_threshold as attributes to our config as a dictionary
        config = super(ProposalLayer, self).get_config()
        config["config"] = self.config.to_dict()
        config["proposal_count"] = self.proposal_count
        config["nms_threshold"] = self.nms_threshold
        return config

    def call(self, inputs):
        # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
        scores = inputs[0][:, :, 1]
        # Box deltas [batch, num_rois, 4]
        deltas = inputs[1]
        #Remember we normalized the bboxes dividing by the RPN std dev, 
        # so multiply to go back to the non normalized bboxes
        deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV, [1, 1, 4])
        # Anchors
        anchors = inputs[2]

        # Improve performance by trimming to top anchors by score
        # and doing the rest on the smaller subset.
        pre_nms_limit = tf.minimum(self.config.PRE_NMS_LIMIT, tf.shape(input=anchors)[1])
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True,
                         name="top_anchors").indices
        scores = I.batch_slice([scores, ix], lambda x, y: tf.gather(x, y),
                                   self.config.BATCH_SIZE)
        deltas = I.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y),
                                   self.config.BATCH_SIZE)
        pre_nms_anchors = I.batch_slice([anchors, ix], lambda a, x: tf.gather(a, x),
                                    self.config.BATCH_SIZE,
                                    names=["pre_nms_anchors"])

        # Apply deltas to anchors to get refined anchors.
        # [batch, N, (y1, x1, y2, x2)]
        boxes = I.batch_slice([pre_nms_anchors, deltas],
                                  lambda x, y: apply_box_deltas(x, y),
                                  self.config.BATCH_SIZE,
                                  names=["refined_anchors"])

        # Non-max suppression from tensorflow.image
        def nms(boxes, scores):
            indices = tf.image.non_max_suppression(
                boxes, scores, self.proposal_count,
                self.nms_threshold, name="rpn_non_max_suppression")
            proposals = tf.gather(boxes, indices)
            # Pad if needed
            padding = tf.maximum(self.proposal_count - tf.shape(input=proposals)[0], 0)
            proposals = tf.pad(tensor=proposals, paddings=[(0, padding), (0, 0)])
            return proposals
        proposals = I.batch_slice([boxes, scores], nms,
                                      self.config.BATCH_SIZE)

        if not context.executing_eagerly():
            # Infer the static output shape:
            out_shape = self.compute_output_shape(None)
            proposals.set_shape(out_shape)
        return proposals

    def compute_output_shape(self, input_shape):
        return None, self.proposal_count, 4

############################################################
#  Detection Target Layer
############################################################

#This is already defined in calculate_iou_matrix in input-pipeline, just uncomment in case we need a tf implementation instead of np
def calculate_iou_matrix_tf2(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    """
    # 1. Tile boxes2 and repeat boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeat() so simulate it
    # using tf.tile() and tf.reshape.
    
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                            [1, 1, tf.shape(input=boxes2)[0]]), [-1, 4])
    b2 = tf.tile(boxes2, [tf.shape(input=boxes1)[0], 1])
    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection
    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(input=boxes1)[0], tf.shape(input=boxes2)[0]])
    return overlaps
    


def detection_targets(proposals, gt_class_ids, gt_boxes, config):
    """Generates detection targets for one image. Subsamples proposals and
    generates target class IDs and bounding box deltas for each.

    Inputs:
    proposals: [POST_NMS_ROIS_TRAINING, (y1, x1, y2, x2)] in normalized coordinates. Might be zero padded if there are not enough proposals.
    gt_class_ids: [MAX_GT_INSTANCES] int class IDs
    gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coordinates.

    Returns: Target ROIs and corresponding class IDs and bounding box shifts.
    rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
    class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
    deltas: [TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw))]
    
    Note: Returned arrays might be zero padded if not enough target ROIs.
    This is because input proposals are zeri padded if POST_NMS_ROIS_TRAINING is larger than the number of post nms rois
    """
    # Assertions
    asserts = [
        tf.Assert(tf.greater(tf.shape(input=proposals)[0], 0), [proposals],
                  name="roi_assertion"),
    ]
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals) 
    #proposals=tf.identity(proposals)
    # print(f"Eager execution?: {tf.executing_eagerly()}")
    # print("DetectTargetsLayer: Proposals start")
    # print(f"{tf.identity(proposals)}")
    # print("DetectTargetsLayer: Proposals end")
    # print("DetectTargetsLayer: GT_BOXES start")
    # print(f"{tf.identity(gt_boxes)}")
    # print("DetectTargetsLayer: GT_BOXES end")
    # print("DetectTargetsLayer: GT_CLASS_IDS start")
    # print(f"{tf.identity(gt_class_ids)}")
    # print("DetectTargetsLayer: GT_CLASS_IDS end")
    # Remove zero padding
    if config.DETECTION_EXCLUDE_BACKGROUND:
        proposals, _ = trim_zeros(proposals, name="trim_proposals")
        gt_boxes, non_zeros = trim_zeros(gt_boxes, name="trim_gt_boxes")
        gt_class_ids = tf.boolean_mask(tensor=gt_class_ids, mask=non_zeros,
                                       name="trim_gt_class_ids")
    # print("DetectTargetsLayer: Proposals start")
    # print(f"{tf.identity(proposals)}")
    # print("DetectTargetsLayer: Proposals end")
    # print("DetectTargetsLayer: GT_BOXES start")
    # print(f"{tf.identity(gt_boxes)}")
    # print("DetectTargetsLayer: GT_BOXES end")
    # print("DetectTargetsLayer: GT_CLASS_IDS start")
    # print(f"{tf.identity(gt_class_ids)}")
    # print("DetectTargetsLayer: GT_CLASS_IDS end")

    # Just for safekeeping kill all values where gt_class_ids is 0
    if config.DETECTION_EXCLUDE_BACKGROUND:
        non_zero_ix = tf.compat.v1.where(gt_class_ids>0)[:,0]
    else:
        non_zero_ix = tf.compat.v1.where(gt_class_ids>=0)[:,0]
    gt_class_ids = tf.gather(gt_class_ids, non_zero_ix)
    gt_boxes = tf.gather(gt_boxes, non_zero_ix)
    # Compute overlaps matrix [proposals, gt_boxes]
    IoU = calculate_iou_matrix_tf2(proposals, gt_boxes)
    #IoU = I.calculate_iou_matrix_tf(proposals, gt_boxes)

    # Determine positive and negative ROIs
    roi_iou_max = tf.reduce_max(input_tensor=IoU, axis=1)
    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = (roi_iou_max >= 0.5)
    positive_indices = tf.compat.v1.where(positive_roi_bool)[:, 0]
    # 2. Negative ROIs are those with < 0.5 with every GT box.
    negative_indices = tf.compat.v1.where(roi_iou_max < 0.5)[:, 0]

    # Subsample ROIs. Aim for 33% positive
    # Positive ROIs
    positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
                         config.ROI_POSITIVE_RATIO)
    positive_indices = tf.random.shuffle(positive_indices)[:positive_count]
    positive_count = tf.shape(input=positive_indices)[0]
    # Negative ROIs. Add enough to maintain positive:negative ratio.
    r = 1.0 / config.ROI_POSITIVE_RATIO
    negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    negative_indices = tf.random.shuffle(negative_indices)[:negative_count]
    # Gather selected ROIs
    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)

    # Assign positive ROIs to GT boxes.
    positive_IoUs = tf.gather(IoU, positive_indices)
    roi_gt_box_assignment = tf.cond(
        pred=tf.greater(tf.shape(input=positive_IoUs)[1], 0),
        true_fn=lambda: tf.argmax(input=positive_IoUs, axis=1),
        false_fn=lambda: tf.cast(tf.constant([]), tf.int64)
    )
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)
    # print(f"detection_targets: rot_gt_class_ids: {roi_gt_class_ids}")
    # print("DetectTargetsLayer: POSITIVE_ROIS start")
    # print(f"{tf.identity(positive_rois)}")
    # print("DetectTargetsLayer: POSITIVE_ROIS end")
    # print("DetectTargetsLayer: ROI_GT_BOXES start")
    # print(f"{tf.identity(roi_gt_boxes)}")
    # print("DetectTargetsLayer: ROI_GT_BOXES end")
    # print("DetectTargetsLayer: ROI_GT_CLASS_IDS start")
    # print(f"{tf.identity(roi_gt_class_ids)}")
    # print("DetectTargetsLayer: ROI_GT_CLASS_IDS end")

    # Compute bbox refinement for positive ROIs
    deltas = I.calculate_box_shifts_tf(positive_rois, roi_gt_boxes) # This is dy, dx, log(dh), log(dw) NOT COORDINATES
    deltas /= config.RPN_BBOX_STD_DEV

    # Append negative ROIs and pad bbox deltas that
    # are not used for negative ROIs with zeros.
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    N = tf.shape(input=negative_rois)[0]
    P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(input=rois)[0], 0)
    rois = tf.pad(tensor=rois, paddings=[(0, P), (0, 0)])
    roi_gt_boxes = tf.pad(tensor=roi_gt_boxes, paddings=[(0, N + P), (0, 0)])
    roi_gt_class_ids = tf.pad(tensor=roi_gt_class_ids, paddings=[(0, N + P)])
    deltas = tf.pad(tensor=deltas, paddings=[(0, N + P), (0, 0)])
    # print(f"detection_targets: rot_gt_class_ids: {roi_gt_class_ids}")

    # print("DetectTargetsLayer: ROIS start")
    # print(f"{tf.identity(rois)}")
    # print("DetectTargetsLayer: ROIS end")
    # print("DetectTargetsLayer: ROI_GT_BOXES start")
    # print(f"{tf.identity(roi_gt_boxes)}")
    # print("DetectTargetsLayer: ROI_GT_BOXES end")
    # print("DetectTargetsLayer: ROI_GT_CLASS_IDS start")
    # print(f"{tf.identity(roi_gt_class_ids)}")
    # print("DetectTargetsLayer: ROI_GT_CLASS_IDS end")
    return rois, roi_gt_class_ids, deltas


class DetectionTargetLayer(KL.Layer):
    """
    Subsamples proposals and generates target box refinement and class_ids for each.

    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coordinates.

    Returns: Target ROIs and corresponding class IDs and bounding box shifts
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw)]

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """

    def __init__(self, config, **kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)
        self.config = config

    def get_config(self):
        config = super(DetectionTargetLayer, self).get_config()
        config["config"] = self.config.to_dict()
        return config

    def call(self, inputs):
        proposals = inputs[0]
        gt_class_ids = inputs[1]
        gt_boxes = inputs[2]

        # Slice the batch and run a graph for each slice
        names = ["rois", "target_class_ids", "target_bbox"] #NOTE : used to be target_bbox instead of target_deltas
        outputs = I.batch_slice(
            [proposals, gt_class_ids, gt_boxes],
            lambda w, x, y: detection_targets(
                w, x, y, self.config),
            self.config.BATCH_SIZE, names=names)
        # print(f"DetectionTargetLayers: gt_class_ids: {outputs[1]}")
        # outputs are rois (y1,x1,y2,x2), target_class_ids (1-3) and target_deltas (dy,dx,log(dh),log(dw))
        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # rois
            (None, self.config.TRAIN_ROIS_PER_IMAGE),  # class_ids
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4)  # deltas
        ]

############################################################
#  Detection Layer
############################################################

def refine_detections(rois, probs, deltas, config):
    """
    Refine classified proposals and filter overlaps and return final
    detections.

    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.

    Returns detections shaped: [num_detections, (y1, x1, y2, x2, class_id, score)] where
        coordinates are normalized.
    """
    # Class IDs per ROI
    class_ids = tf.argmax(input=probs, axis=1, output_type=tf.int32)
    # Class probability of the top class of each ROI
    indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
    class_scores = tf.gather_nd(probs, indices)
    # Class-specific bounding box deltas
    deltas_specific = tf.gather_nd(deltas, indices)
    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    refined_rois = apply_box_deltas(
        rois, deltas_specific * config.RPN_BBOX_STD_DEV)

    if config.DETECTION_EXCLUDE_BACKGROUND:
        # Filter out background boxes
        keep = tf.compat.v1.where(class_ids > 0)[:, 0]
    else:
        keep = tf.compat.v1.where(class_ids >= 0)[:, 0]
    # Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE:
        conf_keep = tf.compat.v1.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[:, 0]
        keep = tf.sets.intersection(tf.expand_dims(keep, 0),
                                        tf.expand_dims(conf_keep, 0))
        keep = tf.sparse.to_dense(keep)[0]

    # Apply per-class NMS
    # 1. Prepare variables
    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(class_scores, keep)
    pre_nms_rois = tf.gather(refined_rois,   keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    def nms_keep_map(class_id):
        """Apply Non-Maximum Suppression on ROIs of the given class."""
        # Indices of ROIs of the given class
        ixs = tf.compat.v1.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
        # Apply NMS
        class_keep = tf.image.non_max_suppression(
                tf.gather(pre_nms_rois, ixs),
                tf.gather(pre_nms_scores, ixs),
                max_output_size=config.DETECTION_MAX_INSTANCES,
                iou_threshold=config.DETECTION_NMS_THRESHOLD)
        # Map indices
        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
        # Pad with -1 so returned tensors have the same shape
        gap = config.DETECTION_MAX_INSTANCES - tf.shape(input=class_keep)[0]
        class_keep = tf.pad(tensor=class_keep, paddings=[(0, gap)],
                            mode='CONSTANT', constant_values=-1)
        # Set shape so map_fn() can infer result shape
        class_keep.set_shape([config.DETECTION_MAX_INSTANCES])
        return class_keep

    # 2. Map over class IDs
    nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids,
                         dtype=tf.int64)
    # 3. Merge results into one list, and remove -1 padding
    nms_keep = tf.reshape(nms_keep, [-1])
    nms_keep = tf.gather(nms_keep, tf.compat.v1.where(nms_keep > -1)[:, 0])
    # 4. Compute intersection between keep and nms_keep
    keep = tf.sets.intersection(tf.expand_dims(keep, 0),
                                    tf.expand_dims(nms_keep, 0))
    keep = tf.sparse.to_dense(keep)[0]
    # Keep top detections
    roi_count = config.DETECTION_MAX_INSTANCES
    class_scores_keep = tf.gather(class_scores, keep)
    num_keep = tf.minimum(tf.shape(input=class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    keep = tf.gather(keep, top_ids)

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are normalized.
    detections = tf.concat([
        tf.gather(refined_rois, keep),
        tf.dtypes.cast(tf.gather(class_ids, keep), tf.float32)[..., tf.newaxis],
        tf.gather(class_scores, keep)[..., tf.newaxis]
        ], axis=1)

    # Pad with zeros if detections < DETECTION_MAX_INSTANCES
    gap = config.DETECTION_MAX_INSTANCES - tf.shape(input=detections)[0]
    detections = tf.pad(tensor=detections, paddings=[(0, gap), (0, 0)], mode="CONSTANT")
    return detections


class DetectionLayer(KL.Layer):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] where coordinates are normalized.
    """

    def __init__(self, config=None, **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        self.config = config

    def get_config(self):
        config = super(DetectionLayer, self).get_config()
        config["config"] = self.config.to_dict()
        return config

    def call(self, inputs):
        rois = inputs[0]
        rcnn_class = inputs[1]
        rcnn_bbox = inputs[2]

        # Run detection refinement graph on each item in the batch
        detections_batch = I.batch_slice(
            [rois, rcnn_class, rcnn_bbox],
            lambda x, y, w: refine_detections(x, y, w, self.config),
            self.config.BATCH_SIZE)

        # Reshape output
        # [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] in
        # normalized coordinates
        return tf.reshape(
            detections_batch,
            [self.config.BATCH_SIZE, self.config.DETECTION_MAX_INSTANCES, 6])

    def compute_output_shape(self, input_shape):
        return (None, self.config.DETECTION_MAX_INSTANCES, 6)

############################################################
#  ROIAlign Layer
############################################################

def log2_graph(x):
    """Implementation of Log2. TF doesn't have a native implementation."""
    return tf.math.log(x) / tf.math.log(2.0)


class PyramidROIAlign(KL.Layer):
    """
    Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - pool_shape: [pool_height, pool_width] of the output pooled regions. Usually [7, 7]

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - image_data: [batch, [image_id, original_size, image_size]] image_id original size of the image used as input to the network and actual image size
    - feature_maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, height, width, channels]

    Output:
    Pooled regions in the shape: [batch, num_boxes, pool_height, pool_width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    def __init__(self, pool_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)

    def get_config(self):
        config = super(PyramidROIAlign, self).get_config()
        config['pool_shape'] = self.pool_shape
        return config

    def call(self, inputs):
        # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
        boxes = inputs[0]

        # Image shape, that is the size of the input image
        image_data = inputs[1]

        # Feature Maps. List of feature maps from different level of the
        # feature pyramid. Each is [batch, height, width, channels]
        feature_maps = inputs[2:]

        # Assign each ROI to a level in the pyramid based on the ROI area.
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1

        # Image size of the input image (original size is dataset size the input image to the network is resized)
        image_shape = I.parse_image_data_tf(image_data)['image_shape'][0]
        # Equation 1 in the Feature Pyramid Networks paper. 
        # https://arxiv.org/pdf/1612.03144.pdf
        # "A RoI of width w and height h (on the input image of the network)
        # is assigned to a level P_k of the feature pyramid. where
        # k = k_0 + log2(\sqrt(wh)/224)
        # 224 is the canonical ImageNet pre-training size, 
        # and k_0 is the target level on which an RoI withw×h= 2242
        # should  be  mapped  into."
        # For resnet that uses C4 as the single scale feature map k_0=4
        # Account for the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        #roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        roi_level = log2_graph(tf.sqrt(h * w) / (2620.0 / tf.sqrt(image_area))) #Change to use 512 since that is our training image size
        roi_level = tf.minimum(1+len(feature_maps), tf.maximum(
            2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        roi_level = tf.squeeze(roi_level, 2)

        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 2+len(feature_maps))):
            ix = tf.compat.v1.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix) #Get the boxes in the ith level

            # Box indices for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so
            # that we can evaluate either max or average pooling. In fact,
            # interpolating only a single value at each bin center (without
            # pooling) is nearly as effective."
            #
            # Here we use the simplified approach of a single value per bin,
            # which is how it's done in tf.crop_and_resize()
            # Result: [batch * num_boxes, pool_height, pool_width, channels]
            pooled.append(tf.image.crop_and_resize(
                feature_maps[i], level_boxes, box_indices, self.pool_shape,
                method="bilinear"))

        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0)

        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(input=box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
            input=box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        # Re-add the batch dimension
        shape = tf.concat([tf.shape(input=boxes)[:2], tf.shape(input=pooled)[1:]], axis=0)
        pooled = tf.reshape(pooled, shape)
        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1], )



#####################################################################
#                      RCNN
#####################################################################

class RCNN(object):
    """
    Encapsulate the RCNN model in a class
    the actual Keras model is in the keras_model property
    """
    def __init__(self, config, mode='train'):
        assert mode in ['train', 'inference']
        self.config = config
        self.mode = mode
        
        # Build the model
        self.keras_model = self.build_entire_model(mode)
        self.keras_model.metrics_tensors = []
        # print(self.keras_model.summary())

        # Compile in training mode
        if mode == 'train':
            self.compile()

    @staticmethod
    def build_backbone(input_tensor, architecture, stage5=False, train_bn=None):
        """Build a ResNet model. We could change this for another backbone

        Arguments
        ----------
        input_tensor: Keras Input layer
            Tensor for image input
        architecture: str, "resnet50" or "resnet101"
            Architecture to use
        stage5: bool
            If False, stage5 of the network is not created
        train_bn: bool.
            Train or freeze Batch Normalization layers

        Returns
        -------
        list
            Backbone layers of ResNet 50 or 101

        """

        # Code adopted from:
        # https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

        def identity_block(tensor, kernel_size, filters, stage, block, use_bias=True):
            """The identity_block is the block that has no convolution layer at shortcut

            Arguments
            --------
            tensor: Keras Layer
                The tensor to connect to this block.
            kernel_size: int
                The kernel size of the convolutional layer
            filters: list
                List of integers indicating how many filters to use for each convolution layer
            stage: int
                Current stage label for generating layer names
            block: str
                Current block label for generating layer names
            use_bias: bool
                To use or not use a bias in conv layers.

            Returns
            -------
            y: Keras Layer
                Output of the Resnet identity block
            """

            nb_filter1, nb_filter2, nb_filter3 = filters
            conv_name_base = 'res' + str(stage) + block + '_branch'
            bn_name_base = 'bn' + str(stage) + block + '_branch'

            y = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', use_bias=use_bias)(tensor)
            y = KL.BatchNormalization(name=bn_name_base + '2a')(y, training=train_bn)
            y = KL.Activation('relu')(y)

            y = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b',
                          use_bias=use_bias)(y)
            y = KL.BatchNormalization(name=bn_name_base + '2b')(y, training=train_bn)
            y = KL.Activation('relu')(y)

            y = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=use_bias)(y)
            y = KL.BatchNormalization(name=bn_name_base + '2c')(y, training=train_bn)

            y = KL.Add()([y, tensor])
            y = KL.Activation('relu', name='res' + str(stage) + block + '_out')(y)
            return y

        def conv_block(tensor, kernel_size, filters, stage, block, strides=(2, 2), use_bias=True):

            """conv_block is the block that has a conv layer at shortcut

            Arguments
            ---------
            tensor: Keras Layer
                The tensor to connect to this block.
            kernel_size: int
                The kernel size of the convolutional layer
            filters: list
                List of integers indicating how many filters to use for each convolution layer
            stage: int
                Current stage label for generating layer names
            block: str
                Current block label for generating layer names
            strides: tuple
                A tuple of integers indicating the strides to make during convolution.
            use_bias: bool
                To use or not use a bias in conv layers.

            Returns
            -------
            y: Keras Layer
                Output layer of Resnet conv block

            """
            nb_filter1, nb_filter2, nb_filter3 = filters
            conv_name_base = 'res' + str(stage) + block + '_branch'
            bn_name_base = 'bn' + str(stage) + block + '_branch'

            y = KL.Conv2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a', use_bias=use_bias)(
                tensor)
            y = KL.BatchNormalization(name=bn_name_base + '2a')(y, training=train_bn)
            y = KL.Activation('relu')(y)

            y = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b',
                          use_bias=use_bias)(y)
            y = KL.BatchNormalization(name=bn_name_base + '2b')(y, training=train_bn)
            y = KL.Activation('relu')(y)

            y = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=use_bias)(y)
            y = KL.BatchNormalization(name=bn_name_base + '2c')(y, training=train_bn)

            shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides, name=conv_name_base + '1', use_bias=use_bias)(
                tensor)
            shortcut = KL.BatchNormalization(name=bn_name_base + '1')(shortcut, training=train_bn)

            y = KL.Add()([y, shortcut])
            y = KL.Activation('relu', name='res' + str(stage) + block + '_out')(y)
            return y

        def block2(x, filters, kernel_size=3, stride=1, conv_shortcut=False, name=None):
            """A residual block. This is effectively a way to write conv_block and identity_block in one just function
            
            Arguments:
            x: input tensor.
            filters: integer, filters of the bottleneck layer.
            kernel_size: default 3, kernel size of the bottleneck layer.
            stride: default 1, stride of the first layer.
            conv_shortcut: default False, use convolution shortcut if True,
            otherwise identity shortcut.
            name: string, block label.
            
            Returns:
            Output tensor for the residual block.
            """
            preact = KL.BatchNormalization(epsilon=1.001e-5, name=name + '_preact_bn')(x, training=train_bn)
            preact = KL.Activation('relu', name=name + '_preact_relu')(preact)

            if conv_shortcut:
                shortcut = KL.Conv2D(
                    4 * filters, 1, strides=stride, name=name + '_0_conv')(preact)
            else:
                shortcut = KL.MaxPooling2D(1, strides=stride)(x) if stride > 1 else x
                
            x = KL.Conv2D(
                filters, 1, strides=1, use_bias=False, name=name + '_1_conv')(preact)
            x = KL.BatchNormalization(epsilon=1.001e-5, name=name + '_1_bn')(x, training=train_bn)
            x = KL.Activation('relu', name=name + '_1_relu')(x)
            
            x = KL.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
            x = KL.Conv2D(
                filters,
                kernel_size,
                strides=stride,
                use_bias=False,
            name=name + '_2_conv')(x)
            x = KL.BatchNormalization(epsilon=1.001e-5, name=name + '_2_bn')(x, training=train_bn)
            x = KL.Activation('relu', name=name + '_2_relu')(x)
            
            x = KL.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
            x = KL.Add(name=name + '_out')([shortcut, x])
            return x


        def stack2(x, filters, blocks, stride1=2, name=None):
            """A set of stacked residual blocks. This is an effective way of writing multiple implementations of ResNet at once since the only thing different is the number of residual blocks stacked
            
            Arguments:
            x: input tensor.
            filters: integer, filters of the bottleneck layer in a block.
            blocks: integer, blocks in the stacked blocks.
            stride1: default 2, stride of the first layer in the first block.
            name: string, stack label.
            
            Returns:
            Output tensor for the stacked blocks.
            """
            x = block2(x, filters, conv_shortcut=True, name=name + '_block1')
            for i in range(2, blocks):
                x = block2(x, filters, name=name + '_block' + str(i))
            x = block2(x, filters, stride=stride1, name=name + '_block' + str(blocks))
            return x


        assert architecture in ["resnet50", "resnet101", "resnet152","resnet50v2", "resnet101v2", "resnet152v2", "temnet", "vgg16", "vgg19", "inception_resnetv2"]
        if architecture in ['resnet50', 'resnet101', 'resnet152']:
            # Stage 1
            x = KL.ZeroPadding2D((3, 3))(input_tensor)
            x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
            x = KL.BatchNormalization(name='bn_conv1')(x, training=train_bn)
            x = KL.Activation('relu')(x)
            C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
            # Stage 2
            x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
            x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
            C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
            # Stage 3
            x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
            x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
            x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
            C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
            # Stage 4
            x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
            block_count = {"resnet50": 5, "resnet101": 22}[architecture]
            for i in range(block_count):
                x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i))
            C4 = x
            # Stage 5
            if stage5:
                x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
                x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
                C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
            else:
                C5 = None
        elif architecture in ['resnet50v2', 'resnet101v2', 'resnet152v2']:
            #Stage 1 (256,256,64)
            x = KL.ZeroPadding2D(
                padding=((3, 3), (3, 3)), name='conv1_pad')(input_tensor)
            C1 = x = KL.Conv2D(64, 7, strides=2, use_bias=True, name='conv1_conv')(x)
            #Stage 2 (128,128,64)
            x = KL.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
            C2 = x = KL.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

            #Stage 3 Remember there's a pre activation on each residual block so there's no activation on these C## stages unlike the ResNetv1 counterpart (64,64,256)
            C3 = x = stack2(x, 64, 3, name='conv2')
            #Stage 4 (32,32,512)
            block_count_c4 = {"resnet50v2": 4, "resnet101v2": 4, "resnet152v2": 8}[architecture]
            C4 = x = stack2(x, 128, block_count_c4, name='conv3')
            #Stage 5 (16,16,2048)
            block_count_c5 = {"resnet50v2": 6, "resnet101v2": 23, "resnet152v2": 36}[architecture]
            x = stack2(x, 256, block_count_c5, name='conv4')
            #Stage 5 still, since we use stride1=1 there is no change to the tensor size
            if stage5:
                x = stack2(x, 512, 3, stride1=1, name='conv5')
                x = KL.BatchNormalization(epsilon=1.001e-5, name='post_bn')(x)
                C5 = x = KL.Activation('relu', name='post_relu')(x)
            else:
                C5 = None
        elif architecture == "inception_resnetv2":
            def conv2d_bn(x,
                        filters,
                        kernel_size,
                        strides=1,
                        padding='same',
                        activation='relu',
                        use_bias=False,
                        name=None):
                """Utility function to apply conv + BN.

                Arguments:
                x: input tensor.
                filters: filters in `Conv2D`.
                kernel_size: kernel size as in `Conv2D`.
                strides: strides in `Conv2D`.
                padding: padding mode in `Conv2D`.
                activation: activation in `Conv2D`.
                use_bias: whether to use a bias in `Conv2D`.
                name: name of the ops; will become `name + '_ac'` for the activation
                    and `name + '_bn'` for the batch norm layer.

                Returns:
                Output tensor after applying `Conv2D` and `BatchNormalization`.
                """
                x = KL.Conv2D(
                    filters,
                    kernel_size,
                    strides=strides,
                    padding=padding,
                    use_bias=use_bias,
                    name=name)(
                        x)
                if not use_bias:
                    bn_axis = 1 if backend.image_data_format() == 'channels_first' else 3
                    bn_name = None if name is None else name + '_bn'
                    x = KL.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x, training=train_bn)
                if activation is not None:
                    ac_name = None if name is None else name + '_ac'
                    x = KL.Activation(activation, name=ac_name)(x)
                return x


            def inception_resnet_block(x, scale, block_type, block_idx, activation='relu'):
                """Adds an Inception-ResNet block.

                This function builds 3 types of Inception-ResNet blocks mentioned
                in the paper, controlled by the `block_type` argument (which is the
                block name used in the official TF-slim implementation):
                - Inception-ResNet-A: `block_type='block35'`
                - Inception-ResNet-B: `block_type='block17'`
                - Inception-ResNet-C: `block_type='block8'`

                Arguments:
                x: input tensor.
                scale: scaling factor to scale the residuals (i.e., the output of passing
                `x` through an inception module) before adding them to the shortcut
                branch. Let `r` be the output from the residual branch, the output of this
                block will be `x + scale * r`.
                block_type: `'block35'`, `'block17'` or `'block8'`, determines the network
                structure in the residual branch.
                block_idx: an `int` used for generating layer names. The Inception-ResNet
                blocks are repeated many times in this network. We use `block_idx` to
                identify each of the repetitions. For example, the first
                Inception-ResNet-A block will have `block_type='block35', block_idx=0`,
                and the layer names will have a common prefix `'block35_0'`.
                activation: activation function to use at the end of the block (see
                [activations](../activations.md)). When `activation=None`, no activation
                is applied
                (i.e., "linear" activation: `a(x) = x`).

                Returns:
                Output tensor for the block.

                Raises:
                ValueError: if `block_type` is not one of `'block35'`,
                `'block17'` or `'block8'`.
                """
                if block_type == 'block35':
                    branch_0 = conv2d_bn(x, 32, 1)
                    branch_1 = conv2d_bn(x, 32, 1)
                    branch_1 = conv2d_bn(branch_1, 32, 3)
                    branch_2 = conv2d_bn(x, 32, 1)
                    branch_2 = conv2d_bn(branch_2, 48, 3)
                    branch_2 = conv2d_bn(branch_2, 64, 3)
                    branches = [branch_0, branch_1, branch_2]
                elif block_type == 'block17':
                    branch_0 = conv2d_bn(x, 192, 1)
                    branch_1 = conv2d_bn(x, 128, 1)
                    branch_1 = conv2d_bn(branch_1, 160, [1, 7])
                    branch_1 = conv2d_bn(branch_1, 192, [7, 1])
                    branches = [branch_0, branch_1]
                elif block_type == 'block8':
                    branch_0 = conv2d_bn(x, 192, 1)
                    branch_1 = conv2d_bn(x, 192, 1)
                    branch_1 = conv2d_bn(branch_1, 224, [1, 3])
                    branch_1 = conv2d_bn(branch_1, 256, [3, 1])
                    branches = [branch_0, branch_1]
                else:
                    raise ValueError('Unknown Inception-ResNet block type. '
                                    'Expects "block35", "block17" or "block8", '
                                    'but got: ' + str(block_type))

                block_name = block_type + '_' + str(block_idx)
                channel_axis = 1 if backend.image_data_format() == 'channels_first' else 3
                mixed = KL.Concatenate(
                    axis=channel_axis, name=block_name + '_mixed')(
                        branches)
                up = conv2d_bn(
                    mixed,
                    backend.int_shape(x)[channel_axis],
                    1,
                    activation=None,
                    use_bias=True,
                    name=block_name + '_conv')

                x = KL.Lambda(
                    lambda inputs, scale: inputs[0] + inputs[1] * scale,
                    output_shape=backend.int_shape(x)[1:],
                    arguments={'scale': scale},
                    name=block_name)([x, up])
                if activation is not None:
                    x = KL.Activation(activation, name=block_name + '_ac')(x)
                return x
            #Default input size for Inception is 299x299px
            # Our default size is 512x512px so the sizes change slightly
            # We indicate both sizes as: default_inception_size_feature_maps / our_size_feature_maps
            # In order to use Inception with FPN we nedd to add paddings in order to make sure 2D Upsampling from later feature maps match with feature maps the tensor size of previous feature maps
            """
            input_shape = imagenet_utils.obtain_input_shape(
                input_shape,
                default_size=299,
                min_size=75,
                data_format=backend.image_data_format(),
                require_flatten=include_top,
            weights=weights)
            """
            input_shape = (512,512,3)
            #input_shape = (self.config.IMAGE_SHAPE[0],self.config.IMAGE_SHAPE[0],3)
            if input_tensor is None:
                img_input = KL.Input(shape=input_shape)
            else:
                if not backend.is_keras_tensor(input_tensor):
                    img_input = KL.Input(tensor=input_tensor, shape=input_shape)
                else:
                    img_input = input_tensor

            align_feature_maps = True
            padding = 'SAME' if align_feature_maps else 'valid'

            #Stage 1
            # Stem block: 35 x 35 x 192
            C1 = x = conv2d_bn(img_input, 32, 3, strides=2, padding=padding)#  / 255x255x32 tensor
            # C1 = KL.ZeroPadding2D(((0,1),(0,1)))(x) #  / 256x256x32 tensor

            #Stage 2
            x = conv2d_bn(x, 32, 3, padding=padding) # / 253x253x32
            x = conv2d_bn(x, 64, 3)
            C2 = x = KL.MaxPooling2D(3, strides=2, padding=padding)(x) # / 126x126x64
            # C2 = KL.ZeroPadding2D(1)(x) #  / 128x128x64 tensor

            # Stage 3
            x = conv2d_bn(x, 80, 1, padding=padding)
            x = conv2d_bn(x, 192, 3, padding=padding)
            # x = KL.MaxPooling2D(3, strides=2)(x) # 35x35x192 / 61x61x192 tensor
            x = KL.MaxPooling2D(3, strides=2, padding=padding)(x) # 35x35x192 / 61x61x192 tensor

            # Mixed 5b (Inception-A block): 35 x 35 x 320 / 61 x 61 x 256
            branch_0 = conv2d_bn(x, 96, 1)
            branch_1 = conv2d_bn(x, 48, 1)
            branch_1 = conv2d_bn(branch_1, 64, 5)
            branch_2 = conv2d_bn(x, 64, 1)
            branch_2 = conv2d_bn(branch_2, 96, 3)
            branch_2 = conv2d_bn(branch_2, 96, 3)
            branch_pool = KL.AveragePooling2D(3, strides=1, padding='SAME')(x)
            branch_pool = conv2d_bn(branch_pool, 64, 1)
            branches = [branch_0, branch_1, branch_2, branch_pool]
            channel_axis = 1 if backend.image_data_format() == 'channels_first' else 3
            x = KL.Concatenate(axis=channel_axis, name='mixed_5b')(branches)

            # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
            for block_idx in range(1, 11):
                x = inception_resnet_block(
                    x, scale=0.17, block_type='block35', block_idx=block_idx)
            # C3 = KL.ZeroPadding2D(((1,2),(1,2)))(x) # 38x38x320 / 64x64x320 tensor
            C3 = x # 38x38x320 / 64x64x320 tensor

            # Stage 4
            # Mixed 6a (Reduction-A block): 17 x 17 x 1088 / 30 x 30 x 1088
            branch_0 = conv2d_bn(x, 384, 3, strides=2, padding=padding)
            branch_1 = conv2d_bn(x, 256, 1)
            branch_1 = conv2d_bn(branch_1, 256, 3)
            branch_1 = conv2d_bn(branch_1, 384, 3, strides=2, padding=padding)
            branch_pool = KL.MaxPooling2D(3, strides=2, padding=padding)(x)
            branches = [branch_0, branch_1, branch_pool]
            x = KL.Concatenate(axis=channel_axis, name='mixed_6a')(branches)

            # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
            for block_idx in range(1, 21):
                x = inception_resnet_block(
                    x, scale=0.1, block_type='block17', block_idx=block_idx)
            # C4 = KL.ZeroPadding2D(1)(x) # 19x19x768 / 32x32x768 tensor
            C4 = x # 19x19x768 / 32x32x768 tensor

            # Stage 5
            # Mixed 7a (Reduction-B block): 8 x 8 x 2080 / 14 x 14 x 2080
            branch_0 = conv2d_bn(x, 256, 1)
            branch_0 = conv2d_bn(branch_0, 384, 3, strides=2, padding=padding)
            branch_1 = conv2d_bn(x, 256, 1)
            branch_1 = conv2d_bn(branch_1, 288, 3, strides=2, padding=padding)
            branch_2 = conv2d_bn(x, 256, 1)
            branch_2 = conv2d_bn(branch_2, 288, 3)
            branch_2 = conv2d_bn(branch_2, 320, 3, strides=2, padding=padding)
            branch_pool = KL.MaxPooling2D(3, strides=2, padding=padding)(x)
            branches = [branch_0, branch_1, branch_2, branch_pool]
            x = KL.Concatenate(axis=channel_axis, name='mixed_7a')(branches)

            # 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
            for block_idx in range(1, 10):
                x = inception_resnet_block(
                    x, scale=0.2, block_type='block8', block_idx=block_idx)
            x = inception_resnet_block(
                x, scale=1., activation=None, block_type='block8', block_idx=10)

            # Final convolution block: 8 x 8 x 1536
            x = conv2d_bn(x, 1536, 1, name='conv_7b')
            if stage5:
                # C5 = KL.ZeroPadding2D(1)(x) # 10x10x1536 / 16x16x1536 tensor
                C5 = x # 10x10x1536 / 16x16x1536 tensor
            else:
                C5 = None
        elif "vgg" in architecture:
            #VGG is cool because it only uses 3x3 Convolutions and 2x2 MaxPooling layers so it's pretty easy to implement, however the number of parameters is 135M which is insane
            # Also since all Convolutions use same padding and all MaxPooling are 2x2 it's automatically compatible with FPN's architecture based on 1x1 Convolutions and 2x2 Upsamplings
            # Block 1
            x = KL.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_tensor)
            x = KL.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
            C1 = x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

            # Block 2
            x = KL.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
            x = KL.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
            C2 = x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

            # Block 3
            x = KL.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
            x = KL.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
            x = KL.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
            if architecture == "vgg19":
                x = KL.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
            C3 = x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

            # Block 4
            x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
            x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
            x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
            if architecture == "vgg19":
                x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
            C4 = x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

            if stage5:
                # Block 5
                x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
                x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
                x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
                if architecture == "vgg19":
                    x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
                C5 = x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
            else:
                C5 = None
        elif architecture == "temnet":
            #Implement our TEMNet architecture
            #Stage 1:
            #x = KL.ZeroPadding2D((3, 3))(input_tensor)
            x = KL.Conv2D(8, (13, 13), padding="same", name='conv1', use_bias=True)(input_tensor)
            #x = KL.BatchNormalization(name="bn_conv1")(x, training=train_bn)
            x = GroupNormalization(groups=8, axis=3, name="gn_conv1")(x)
            x = KL.Activation('relu')(x)
            x = KL.MaxPooling2D((2, 2), strides=(2, 2), name="pool1")(x)
            C1 = x = KL.GaussianNoise(0.1, name="noise")(x)
            #Stage 2:
            x = KL.Conv2D(16, (9, 9), padding="same", name="conv2")(x)
            #x = KL.BatchNormalization(name="bn_conv2")(x, training=train_bn)
            x = GroupNormalization(groups=8, axis=3, name="gn_conv2")(x)
            x = KL.Activation('relu')(x)
            C2 = x = KL.MaxPooling2D((2, 2), strides=2, name = "pool2")(x)
            #Stage 3:
            x = KL.Conv2D(32, (7, 7), padding="same", name="conv3")(x)
            #x = KL.BatchNormalization(name="bn_conv3")(x, training=train_bn)
            x = GroupNormalization(groups=8, axis=3, name="gn_conv3")(x)
            x = KL.Activation('relu')(x)
            C3 = x = KL.MaxPooling2D((2, 2), strides=2, name = "pool3")(x)
            #Stage 4:
            x = KL.Conv2D(64, (5, 5), padding="same", name="conv4")(x)
            #x = KL.BatchNormalization(name="bn_conv4")(x, training=train_bn)
            x = GroupNormalization(groups=8, axis=3, name="gn_conv4")(x)
            x = KL.Activation('relu')(x)
            C4 = x = KL.MaxPooling2D((2, 2), strides=2, name = "pool4")(x)
            C5 = None

        return [C1, C2, C3, C4, C5]

    def build_feature_maps(self, input_tensor):

        """Build the feature maps for the feature pyramid network.

        Arguments
        ---------
        input_tensor: Keras Input layer [height, width, channels]

        Returns
        -------
        list
            Pyramid layers

        """

        # Don't create the head (stage 5), so we pick the 4th item in the list.
        _, C2, C3, C4, C5 = self.build_backbone(input_tensor, self.config.BACKBONE, stage5=True,
                                                train_bn=self.config.TRAIN_BATCH_NORMALIZATION)

        if C5 != None:
            # Top-down Layers
            P5 = KL.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5)
            P4 = KL.Add(name="fpn_p4add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
            KL.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)])
            P3 = KL.Add(name="fpn_p3add")([         #Trying to add incompatible shapes?
                KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
                KL.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)]) # ISSUE HERE
            P2 = KL.Add(name="fpn_p2add")([
                KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
                KL.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)])
            
            # Attach 3x3 conv to all P layers to get the final feature maps.
            P2 = KL.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")(P2)
            P3 = KL.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")(P3)
            P4 = KL.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")(P4)
            P5 = KL.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")(P5)
            
            # P6 is used for the 5th anchor scale in RPN. Generated by sub-sampling from P5 with stride of 2.
            P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)
        else: #TEMNet's model doesn't have a C5 layer so we only build P5 down to P2
            # Top-down Layers
            P4 = KL.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)
            P3 = KL.Add(name="fpn_p3add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
            KL.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)])
            P2 = KL.Add(name="fpn_p2add")([         #Trying to add incompatible shapes?
                KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
                KL.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)]) # ISSUE HERE
            
            # Attach 3x3 conv to all P layers to get the final feature maps.
            P2 = KL.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")(P2)
            P3 = KL.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")(P3)
            P4 = KL.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")(P4)
            P5 = None #Our TEMNet model has no C5 layer therefore P5 is not calculated

            # P6 is used for the 5th anchor scale in RPN. Generated by sub-sampling from P4 with stride of 2.
            P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P4)
        # Note that P6 is used in RPN, but not in the classifier heads.
        # That is the rcnn feature maps are [P2, P3, P4, P5]
        return [P2, P3, P4, P5, P6]
    
    #  Feature Pyramid Network Heads
    #@staticmethod
    def build_fpn_classifier(self, rois, feature_maps, image_data,
                       pool_size, num_classes, train_bn=True,
                       fc_layers_size=1024):
        """
        Builds the computation graph of the feature pyramid network classifier
        and regressor heads.
    
        rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
              coordinates.
        feature_maps: List of feature maps from different layers of the pyramid,
                      [P2, P3, P4, P5]. Each has a different resolution.
        image_data: [batch, [img_id, original_size, img_size]] Image height and width in pixels
        pool_size: The width of the square feature map generated from ROI Pooling.
        num_classes: number of classes, which determines the depth of the results
        train_bn: Boolean. Train or freeze Batch Norm layers
        fc_layers_size: Size of the 2 FC layers
    
        Returns:
            logits: [batch, num_rois, NUM_CLASSES] classifier logits (before softmax)
            probs: [batch, num_rois, NUM_CLASSES] classifier probabilities
            bbox_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))] Deltas to apply to
                         proposal boxes
        """
        # ROI Pooling
        # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
        x = PyramidROIAlign([pool_size, pool_size],
                            name="roi_align_classifier")([rois, image_data] + feature_maps)
        # Two 1024 FC layers (implemented with Conv2D for consistency)
        x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid"),
                               name="rcnn_class_conv1")(x)
        x = KL.TimeDistributed(KL.BatchNormalization(), name='rcnn_class_bn1')(x, training=train_bn)
        x = KL.Activation('relu')(x)
        x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (1, 1)),
                               name="rcnn_class_conv2")(x)
        x = KL.TimeDistributed(KL.BatchNormalization(), name='rcnn_class_bn2')(x, training=train_bn)
        x = KL.Activation('relu')(x)
    
        shared = KL.Lambda(lambda x: backend.squeeze(backend.squeeze(x, 3), 2),
                           name="pool_squeeze")(x)
    
        # Classifier head
        rcnn_class_logits = KL.TimeDistributed(KL.Dense(num_classes),
                                                name='rcnn_class_logits')(shared)
        rcnn_probs = KL.TimeDistributed(KL.Activation("softmax"),
                                         name="rcnn_class")(rcnn_class_logits)
    
        # BBox head
        # [batch, num_rois, NUM_CLASSES * (dy, dx, log(dh), log(dw))]
        x = KL.TimeDistributed(KL.Dense(num_classes * 4, activation='linear'),
                               name='rcnn_bbox_fc')(shared)
        # Reshape to [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
        init_shape = backend.int_shape(x)
        if init_shape[1] is None:
            rcnn_bbox = KL.Reshape((-1, num_classes, 4), name="rcnn_bbox")(x)
        else:
            rcnn_bbox = KL.Reshape((init_shape[1], num_classes, 4), name="rcnn_bbox")(x)
    
        return rcnn_class_logits, rcnn_probs, rcnn_bbox

    @staticmethod
    def build_rpn_model(anchor_stride, anchors_per_location, depth):
        """Builds a Keras model of the Region Proposal Network.

        Arguments
        ---------
        anchor_stride: int
        Controls the density of anchors. Typically 1 (anchors for every pixel in the feature map), or 2.
        anchors_per_location: int
            Number of anchors per pixel in the feature map. Equivalent to length of anchor ratios.
        depth: int,
            Depth of the backbone feature map. Same as TOP_DOWN_PYRAMID_SIZE

        Returns
        -------
        Keras Model

        The model outputs, when called, are:
            rpn_class_logits: [batch, H * W * anchors_per_location, 2]
                Anchor classifier logits (before softmax)
            rpn_probs: [batch, H * W * anchors_per_location, 2]
                Anchor classifier probabilities.
            rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))]
                Deltas to be applied to anchors.

        """

        input_feature_map = KL.Input(shape=[None, None, depth], name="input_rpn_feature_map")

        # Shared convolutional base of the RPN
        shared = KL.Conv2D(512, (3, 3), padding='same', activation='relu', strides=anchor_stride,
                           name='rpn_conv_shared')(input_feature_map)

        # Anchor Score. [batch, height, width, anchors per location * 2].
        x = KL.Conv2D(2 * anchors_per_location, (1, 1), padding='valid', activation='linear',
                      name='rpn_class_raw')(shared)

        # Reshape to [batch, anchors, 2]
        rpn_class_logits = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)

        # Softmax on last dimension of BG/FG.
        rpn_probs = KL.Activation("softmax", name="rpn_class_xxx")(rpn_class_logits)

        # Bounding box refinement. [batch, H, W, anchors per location * depth]
        # where depth is [x, y, log(w), log(h)]
        x = KL.Conv2D(anchors_per_location * 4, (1, 1), padding="valid", activation='linear', name='rpn_bbox_pred')(
            shared)

        # Reshape to [batch, anchors, 4]
        rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)

        outputs = [rpn_class_logits, rpn_probs, rpn_bbox]
        return KM.Model([input_feature_map], outputs, name="rpn_model")

    def build_entire_model(self, mode='train'):
        """
        Build the RCNN architecture
        
        Parameters
        ----------
        mode : Either 'train' or 'validation'. 
        The outputs of the model change according to this
        """
        assert mode in ['train', 'inference'], "The model mode must be either \"train\" or \"inference\""
        #In order to be able to use the 6 layers of the feature pyramid
        #since these need to be able to upscale and downscale the input
        #through convolutions. The image must be divisible by 2**6
        #So choose a size like 256, 512, 1024, etc
        h, w = self.config.IMAGE_SHAPE[:2]
        if h/2**6 != int(h/2**6) or w/2**6 != int(w/2**6):
            raise Exception("Image must be divisable by 2**6 for FPN processing"
                            "Use for example 256, 320, 384, 448, 512, etc.")
        
        # Input image
        #input_image = KL.Input(shape=[None, None,
        #                               self.config.NUM_CHANNELS], name="input_image")
        input_image = KL.Input(shape=[h, w,
                                       self.config.NUM_CHANNELS], name="input_image")
        input_image_data = KL.Input(shape=[self.config.IMAGE_DATA_SIZE], name="input_image_meta")
        if mode == "train":
            # RPN GT
            input_rpn_match = KL.Input(
                shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
            input_rpn_bbox = KL.Input(
                shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

            # Detection GT (class IDs, bounding boxes, and masks)
            # 1. GT Class IDs (zero padded)
            input_gt_class_ids = KL.Input(
                shape=[None], name="input_gt_class_ids", dtype=tf.int32)
            # 2. GT Boxes in pixels (zero padded)
            # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
            input_gt_boxes = KL.Input(
                shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
            #print(f"input_gt_boxes: {input_gt_boxes}")
            # Normalize coordinates
            #print(f"input_image: {input_image}")
            #print(f"input_image shape: {backend.shape(input_image)}")
            #print(f"input_image shape[1:3]: {backend.shape(input_image[1:3])}")
            #Either normalize here or normalize on the dataset inputs
            # gt_boxes = KL.Lambda(lambda x: I.norm_boxes_tf(
            #     x, backend.shape(input_image)[1:3]))(input_gt_boxes)
            gt_boxes = KL.Lambda(lambda x: I.norm_boxes_tf(
                x, tf.convert_to_tensor(list(self.config.IMAGE_SHAPE))))(input_gt_boxes)
        elif mode == "inference":
            # Anchors in normalized coordinates
            input_anchors = KL.Input(shape=[None, 4], name="input_anchors")

        # RPN feature maps
        P2, P3, P4, P5, P6 = self.build_feature_maps(input_image)
        if P5 != None:
            rpn_feature_maps = [P2, P3, P4, P5, P6]
            #RCNN feature maps don't use the last layer (P6) of the RPN
            rcnn_feature_maps = [P2, P3, P4, P5]
        else: #For a model with no 5th convolutional stage
            rpn_feature_maps = [P2, P3, P4, P6]
            #RCNN feature maps don't use the last layer (P6) of the RPN
            rcnn_feature_maps = [P2, P3, P4]

        # Anchors
        if mode == "train":
            anchors = self.get_anchors(self.config.IMAGE_SHAPE)
            # print(f"->build_entire_model anchors from get_anchors: {anchors}")
            # print(f"->build_entire_model anchors shape from get_anchors: {anchors.shape}")
            # Duplicate across the batch dimension because Keras requires it
            # TODO: can this be optimized to avoid duplicating the anchors?
            anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)
            # print(f"->build_entire_model anchors from np_broadcast: {anchors}")
            # print(f"->build_entire_model anchors shape from np_broadcast: {anchors.shape}")
            # A hack to get around Keras's bad support for constants
            # This class returns a constant layer
            class ConstLayer(KL.Layer):
                def __init__(self, x, name=None):
                    super(ConstLayer, self).__init__(name=name)
                    #self.x = tf.Variable(x) # THIS IS THE CULPRIT #KERAS LAYERS SHOULD RECEIVE TF TENSORS AND OUTPUT TF TENSORS NOT TF VARIABLES THEY ARE NOT THE SAME TYPE AND THIS WILL GENERATE REALLY HARD TO TRACK ERRORS
                    self.x = tf.convert_to_tensor(x)

                def call(self, input):
                    return self.x

            anchors = ConstLayer(anchors, name="anchors")(input_image)
            # print(f"->build_entire_model anchors from ConstLayer: {anchors}")
        else:
            anchors = input_anchors


        # RPN Network
        rpn = self.build_rpn_model(self.config.RPN_ANCHOR_STRIDE, len(self.config.RPN_ANCHOR_RATIOS),
                                   self.config.TOP_DOWN_PYRAMID_SIZE)

        # Restructures [[a1, b1, c1], [a2, b2, c2]] -> [[a1, a2], [b1, b2], [c1, c2]]
        layer_outputs = []
        # print(f"-> RCNN: rpn ([P2]): {rpn([P2])}")
        for layer in rpn_feature_maps:
            layer_outputs.append(rpn([layer]))
        # print(f"-> RCNN: layer_outputs: {layer_outputs}")
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        rpn_outputs = list(zip(*layer_outputs))
        # print(f"-> RCNN: rpn_outputs before concatenate: {rpn_outputs}")
        rpn_outputs = [KL.Concatenate(axis=1, name=n)(list(o)) 
                       for o, n in zip(rpn_outputs, output_names)]
        # print(f"-> RCNN: rpn_outputs[0] : {rpn_outputs[0]}")

        # Outputs of RPN
        rpn_class_logits, rpn_class, rpn_bbox = rpn_outputs
        # print(f"-> RCNN: rpn_class_logits : {rpn_class_logits.get_shape()}")
        # print(f"-> RCNN: rpn_class_logits : {rpn_class_logits}")

        # Generate proposals for the rcnn (RoIs)
        # Propsals are [batch, N, (y1, x1, y2,x2)]
        # Make proposals by using Non-max supression on the anchors to eliminate duplicates
        max_roi_proposals = self.config.POST_NMS_ROIS_TRAINING if mode == "train" else self.config.POST_NMS_ROIS_VALIDATION
        #Build the rpn RoIs out of the anchors, deltas (bbox) and class probabilities
        #This returns us a fraction of the anchors that fit with gt_boxes
        #These are the ones we're interested in passing to the classifier
        rpn_rois = ProposalLayer(proposal_count = max_roi_proposals,
                                 nms_threshold = self.config.RPN_NMS_THRESHOLD,
                                 name ="RPN_RoIs",
                                 config=self.config)([rpn_class,rpn_bbox, anchors])
        #Instead of using the ProposalLayer for RoI generation we could use the positive anchors
        # This can be done since for our model there doesn't seem to be overlaps so we need no NMS
        # Using our inputs rpn_match, rpn_bbox and anchors we can find our positive anchors 
        # And use these as RoIs for the classifier heads
        # positive_indices = np.where(np.argmax(input_rpn_match, axis=1)==1)[0]
        # positive_anchors = I.shift_bboxes(anchors[positive_indices], input_rpn_bbox[positive_indices])
        # rpn_rois = positive_anchors
        
        #Set targets rois for training and validation
        if mode == "train":
            if self.config.USE_RPN_ROIS:
                target_rois = rpn_rois
            else:
                #Get RoIs from a provided input and disregard RPN predictions
                input_rois = KL.Input(shape=[self.config.POST_NMS_ROIS_TRAINING, 4],
                                      name="input_roi", dtype=np.int32)
                # Normalize coordinates
                # target_rois = KL.Lambda(lambda x: I.norm_boxes_tf(
                #     x, backend.shape(input_image)[1:3]))(input_rois)
                target_rois = KL.Lambda(lambda x: I.norm_boxes_tf(
                    x, tf.convert_to_tensor(list(self.config.IMAGE_SHAPE))))(input_rois)

            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            rois, target_class_ids, target_bbox=DetectionTargetLayer(self.config, name="proposal_targets")([target_rois, input_gt_class_ids, gt_boxes]) #Use input_gt_boxes if those are already normalized

            # Network Heads
            # print(f"->build_entire_model config DATASET_IMAGE_SIZE: {self.config.DATASET_IMAGE_SIZE}")
            #input_img_size = KL.Lambda(lambda x : 1* x, name = "input_img_size")(tf.constant([self.config.DATASET_IMAGE_SIZE[0], self.config.DATASET_IMAGE_SIZE[1]]))
            #input_img_size = KL.Lambda(lambda x : 1* x, name = "input_img_size")(tf.constant([self.config.DATASET_IMAGE_SIZE[0], self.config.DATASET_IMAGE_SIZE[1]]))
            #print(f"->build_entire_model input_img_size: {input_img_size}")
            rcnn_class_logits, rcnn_class, rcnn_bbox =\
                self.build_fpn_classifier(rois, rcnn_feature_maps, input_image_data,
                                     self.config.POOL_SIZE, self.config.NUM_CLASSES,
                                     train_bn=self.config.TRAIN_BATCH_NORMALIZATION,
                                     fc_layers_size=self.config.FPN_CLASSIF_FC_LAYERS_SIZE)

            # TODO: clean up (use tf.identify if necessary)
            output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)
            
            # Loss functions
            # GT inputs to RPN
            rpn_class_loss = KL.Lambda(lambda x: H.rpn_match_loss(*x), name="rpn_class_loss")(
                                       [input_rpn_match, rpn_class_logits])
            """print("model input_rpn match", input_rpn_match.shape)
            print("model input_rpn_bbox", input_rpn_bbox.shape)
            print("model rpn_bbox", rpn_bbox.shape)"""
            rpn_bbox_loss = KL.Lambda(lambda x: H.rpn_bbox_loss(self.config, *x), name="rpn_bbox_loss")(
                                      [input_rpn_match, input_rpn_bbox, rpn_bbox])
            if not self.config.TRAIN_ONLY_RPN:
                class_loss = KL.Lambda(lambda x: H.rcnn_class_loss(*x), name="rcnn_class_loss")([target_class_ids, rcnn_class_logits])
                # print("->model class_loss calculated succesfully", class_loss)
                bbox_loss = KL.Lambda(lambda x: H.rcnn_bbox_loss(self.config, *x), name="rcnn_bbox_loss")([target_bbox, target_class_ids, rcnn_bbox])
                #mAP_accuracy = KL.Lambda(lambda x: H.mAP_accuracy(target_bbox, target_class_ids, rcnn_bbox, rcnn_class, rcnn_class_logits))
                #Inputs and outputs of the model
                inputs = [input_image, input_image_data, input_rpn_match, input_rpn_bbox,
                          input_gt_class_ids, input_gt_boxes]
                if not self.config.USE_RPN_ROIS:
                    inputs.append(input_rois)
                outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                           rcnn_class_logits, rcnn_class, rcnn_bbox,
                           rpn_rois, output_rois,
                           rpn_class_loss, rpn_bbox_loss,
                           class_loss, bbox_loss]
                #print("output shapes: \nrpn_class_logits {} \nrpn_class {} \nrpn_bbox {} \nrpn_class_loss {} \nrpn_bbox_loss {}".format(rpn_class_logits.shape, rpn_class.shape, rpn_bbox.shape, rpn_class_loss.shape, rpn_bbox_loss.shape))
            else: #Inputs/Outputs for training only RPN
                inputs = [input_image, input_rpn_match, input_rpn_bbox]
                outputs = [rpn_class_logits, rpn_class, rpn_bbox, rpn_class_loss, rpn_bbox_loss]

        else: #mode =="inference"
            # Network Heads
            # input_img_size = KL.Lambda(lambda x : 1* x, name = "input_img_size")(tf.constant([self.config.DATASET_IMAGE_SIZE[0], self.config.DATASET_IMAGE_SIZE[1]]))
            # Proposal classifier and BBox regressor heads
            rcnn_class_logits, rcnn_class, rcnn_bbox =\
                self.build_fpn_classifier(rpn_rois, rcnn_feature_maps, input_image_data,
                                     self.config.POOL_SIZE, self.config.NUM_CLASSES,
                                     train_bn=self.config.TRAIN_BATCH_NORMALIZATION,
                                     fc_layers_size=self.config.FPN_CLASSIF_FC_LAYERS_SIZE)

            # Detections
            # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in
            # normalized coordinates
            detections = DetectionLayer(self.config, name="rcnn_detection")([rpn_rois, rcnn_class, rcnn_bbox])
            
            #Inputs and outputs of the model
            if not self.config.TRAIN_ONLY_RPN:
                inputs= [input_image, input_image_data, input_anchors]
                outputs=[detections, rcnn_class, rcnn_bbox,
                         rpn_rois, rpn_class, rpn_bbox]
            else:# Train just the RPN
                inputs = [input_image]
                outputs = [rpn_class, rpn_bbox]

        # Set the model attribute
        model = KM.Model(inputs, outputs, name='rcnn')
        #TODO : Add multi-GPU support
        
        return model

# TODO : Add support for weighting

    def get_anchors(self, image_shape):
        """Returns anchor pyramid for the given image size."""
        # backbone_shapes = I.cnn_input_shapes(self.config, image_shape)
        backbone_shapes = I.cnn_input_shapes(image_shape, self.config.BACKBONE_STRIDES)
        # Cache anchors and reuse if image shape is the same
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            # Generate Anchors
            a = I.generate_anchors(
                self.config.RPN_ANCHOR_SCALES,
                self.config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE,
                self.config.IMAGE_SHAPE)
            # Keep a copy of the latest anchors in pixel coordinates because
            # it's used in inspect_model notebooks.
            # TODO: Remove this after the notebook are refactored to not use it
            self.anchors = a
            # Normalize coordinates
            self._anchor_cache[tuple(image_shape)] = I.norm_boxes(a, image_shape[:2])
        return self._anchor_cache[tuple(image_shape)]


    #@tf.function
    def compile(self):
        """
        Ready the model for training. Add losses, regularizations and metrics.
        Then calls the Keras compile() function.
        """
        #tf.compat.v1.enable_eager_execution()
        # Create the optimizer
        optimizer = KO.SGD(lr=self.config.LEARNING_RATE, momentum=self.config.LEARNING_MOMENTUM,
                           clipnorm=self.config.GRADIENT_CLIP_NORM)

        # Add Losses
        #self.model._losses = []
        #self.model._per_input_losses = {}
        if not self.config.TRAIN_ONLY_RPN:
            loss_names = ["rpn_class_loss", "rpn_bbox_loss",
                          "rcnn_class_loss", "rcnn_bbox_loss"]
        else:
            loss_names = ["rpn_class_loss", "rpn_bbox_loss"]
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            #if layer.output in self.keras_model.losses:
            #    continue
            loss = (tf.reduce_mean(input_tensor=layer.output, keepdims=True) * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.add_loss(loss)
            
        #Add L2 Regularization to avoid overfitting
        reg_losses = [
            KR.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(input=w), tf.float32)
            for w in self.keras_model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        #Compile the model
        # self.keras_model.compile(optimizer=optimizer, loss=[None] * len(self.keras_model.outputs), metrics = ['accuracy'])
        self.keras_model.compile(optimizer=optimizer, loss=[None] * len(self.keras_model.outputs))

        # Add metrics for losses
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            loss = (tf.reduce_mean(input_tensor=layer.output, keepdims=True) * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.metrics_tensors.append(loss) # TODO: WHERE IS METRICS_TENSORS?
            #self.keras_model.add_metric(loss, name=name, aggregation='mean')

    def train(self, dataset):
        """
        Train the rcnn using training and validation data

        Arguments
        ---------
        dataset: dict
            Dictionary with 'train' and 'validation' keys that hold custom instances of a DataSequence in data_utils.py
            that is dataset dependent.

        """

        # Create the training directories
        #self.config.create_training_directory()

        # Create a callback for saving weights
        if self.config.TRAIN_ONLY_RPN:
            filename = "rpn_"+self.config.BACKBONE+"_weights.{epoch:02d}.hdf5"
        else:
            filename = "rcnn_"+self.config.BACKBONE+"_weights.{epoch:02d}.hdf5"
        rlrp = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_delta=1E-7)
        lrm = LearningRateMonitor()
        callbacks = [ModelCheckpoint(os.path.join(self.config.WEIGHT_PATH, filename), save_weights_only=True), rlrp, lrm]
        # callbacks = [ModelCheckpoint(os.path.join(self.config.WEIGHT_PATH, filename), save_weights_only=True), lrm]

        """# Create a callback for logging training information
        callbacks.append(CSVLogger(os.path.join(self.config.LOGS, self.config.NAME,
                                                self.config.TIME_STAMP, 'training.csv')))"""

        # Train the model
        #history = self.keras_model.fit_generator(dataset["train"], len(dataset["train"]), epochs=self.config.EPOCHS, callbacks=callbacks,
        #                         validation_data=dataset["validation"], validation_steps=len(dataset["validation"]))
        history = self.keras_model.fit(dataset["train"], steps_per_epoch=len(dataset["train"]), epochs=self.config.EPOCHS, callbacks=callbacks,
                                 validation_data=dataset["validation"], validation_steps=len(dataset["validation"]))
        #history = self.keras_model.fit(dataset["train"], steps_per_epoch=len(dataset["train"]), epochs=self.config.EPOCHS, callbacks=callbacks)
        return history, lrm

    def load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the corresponding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exclude: list of layer names to exclude
        """
        import h5py
        from tensorflow.python.keras.saving import hdf5_format

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        with h5py.File(filepath, mode='r') as f:
            if 'layer_names' not in f.attrs and 'model_weights' in f:
                f = f['model_weights']

            # In multi-GPU training, we wrap the model. Get layers
            # of the inner model because they have the weights.
            keras_model = self.keras_model
            layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
                else keras_model.layers

            # Exclude some layers
            if exclude:
                layers = filter(lambda l: l.name not in exclude, layers)

            if by_name:
                hdf5_format.load_weights_from_hdf5_group_by_name(f, layers)
            else:
                hdf5_format.load_weights_from_hdf5_group(f, layers)

        # Update the log directory
        # self.set_log_dir(filepath)

    def load_weights_by_name(self, filepath, verbose=False):
        """Modified version of the corresponding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        """
        import h5py
        def load_model_weights(cmodel, weights):
            for layer in cmodel.layers:
                print(layer.name)
                if hasattr(layer, 'layers'):
                    load_model_weights(layer, weights[layer.name])
                else:
                    for w in layer.weights:
                        _, name = w.name.split('/')
                        if verbose:
                            print(w.name)
                        try:
                            w.assign(weights[layer.name][name][()])
                        except:
                            w.assign(weights[layer.name][layer.name][name][()])

        with h5py.File(filepath, 'r') as f:
            load_model_weights(self.keras_model, f)

    def get_imagenet_weights(self, backbone):
        """Downloads ImageNet trained weights from Keras.
        Returns path to weights file.
        """
        from keras.utils.data_utils import get_file
        if 'inception' in backbone:
            # To Download Inception-ResNet-v2 weights pretrained in ImageNet
            TF_WEIGHTS_PATH_NO_TOP = ('https://storage.googleapis.com/tensorflow/'
                               'keras-applications/inception_resnet_v2/'
                               'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5')

            weights_path = get_file(
                'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5',
                TF_WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                file_hash='d19885ff4a710c122648d3b5c3b684e4')
        elif 'resnet' in backbone:
            # To Download ResNet50 weights pretrained in ImageNet
            TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/'\
                                     'releases/download/v0.2/'\
                                     'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    TF_WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    md5_hash='a268eb855778b3df3c7506639542a6af')
        else:
            raise NameError('No ImageNet weights could be found!')
        return weights_path


    def predict_batch(self, images, image_metas):
        """
        Runs predictions on an image with given image meta

        Arguments
        ---------------
        images: np array represenation of an image from the dataset, this is first input form a Dataset class instance
        image_metas: array containing info dictionaries for the given image, built on a Dataset class instance

        Returns: A list of dictionaries contaiing the predictions for the image:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes coordinates
        class_ids: [N] integer class IDs
        scores: [N] float score probabilities corresponding to the class_ids
        """

        assert self.mode == "inference", "Prediction is only available on inference mode model"

        # Our model needs three tensors as inputs (see the build_entire_model function):
        # Image tensor representation, tensor of image meta and anchors
        # Image tensor and meta are given from the Dataset class and used as inputs for this function
        # Create the anchors
        anchors = self.get_anchors(images[0].shape)
        #Duplicate images so they are the same shape as the images given
        anchors = np.broadcast_to(anchors, (images.shape[0],)+anchors.shape)

        #Now run keras_model's object detection from our arguments
        #outputs are detections, rcnn_class, rcnn_bbox, rpn_rois, rpn_class and rpn_bbox we need only the first
        detections, rcnn_class, rcnn_bbox, rpn_rois, rpn_class, rpn_bbox = self.keras_model.predict([images, image_metas, anchors])
        rcnn_class = np.squeeze(rcnn_class)
        # print(f"RCNN-> predict_batch rcnn_class: {rcnn_class}")
        rcnn_bbox = np.squeeze(rcnn_bbox)
        # print(f"RCNN-> predict_batch rcnn_bbox: {rcnn_bbox}")
        rpn_rois = np.squeeze(rpn_rois)
        # print(f"RCNN-> predict_batch rpn_rois: {rpn_rois}")
        rpn_class = np.squeeze(rcnn_class)
        # print(f"RCNN-> predict_batch rpn_class: {rpn_class}")
        rpn_bbox = np.squeeze(rpn_bbox)
        # print(f"RCNN-> predict_batch rpn_bbox: {rpn_bbox}")
        #Now divide the detections into their components
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores = self.process_detections(detections[i], images[i].shape)
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores":final_scores
            })
        return results

    def predict_batch_rpn(self, images, image_metas):
        """
        Runs predictions on an image with given image meta

        Arguments
        ---------------
        images: np array represenation of an image from the dataset, this is first input form a Dataset class instance
        image_metas: array containing info dictionaries for the given image, built on a Dataset class instance

        Returns: A list of dictionaries contaiing the predictions for the image:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes coordinates
        class_ids: [N] integer class IDs for RPN: 0 for no particle 1 for particle
        scores: [N] float score probabilities corresponding to the class_ids
        """

        assert self.mode == "inference", "Prediction is only available on inference mode model"

        # Our model needs three tensors as inputs (see the build_entire_model function):
        # Image tensor representation, tensor of image meta and anchors
        # Image tensor and meta are given from the Dataset class and used as inputs for this function
        # Create the anchors
        anchors = self.get_anchors(images[0].shape)
        #Duplicate images so they are the same shape as the images given
        anchors = np.broadcast_to(anchors, (images.shape[0],)+anchors.shape)

        #Now run keras_model's object detection from our arguments
        #outputs are detections, rcnn_class, rcnn_bbox, rpn_rois, rpn_class and rpn_bbox we need only the first
        detections, rcnn_class, rcnn_bbox, rpn_rois, rpn_class, rpn_bbox = self.keras_model.predict([images, image_metas, anchors])
        rcnn_class = np.squeeze(rcnn_class)
        # print(f"RCNN-> predict_batch rcnn_class: {rcnn_class}")
        rcnn_bbox = np.squeeze(rcnn_bbox)
        # print(f"RCNN-> predict_batch rcnn_bbox: {rcnn_bbox}")
        rpn_rois = np.squeeze(rpn_rois)
        # print(f"RCNN-> predict_batch rpn_rois: {rpn_rois}")
        rpn_class = np.squeeze(rcnn_class)
        # print(f"RCNN-> predict_batch rpn_class: {rpn_class}")
        rpn_bbox = np.squeeze(rpn_bbox)
        # print(f"RCNN-> predict_batch rpn_bbox: {rpn_bbox}")
        #Now divide the detections into their components
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores = self.process_detections(detections[i], images[i].shape)
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores":final_scores
            })
        return results

    def process_detections(self, detections, image_shape):
        """
        Processes detections to return boxes corresponding to image_shape
        Arguments:
        --------------
        detections: [N, (y1,x1,y2,x2, class_id, score)] in normalized coordinates
        image_shape: [H,W] shape of the input image for detections
        Returns:
        -----------
        boxes: [N, (y1,x1,y2,x2)] Bounding boxes in pixel coordinates
        class_ids: [N] Integer class ids for each box
        scores: [N] classification probability scores for each box
        """
        #Since the detections are zero padded (see DetectionLayer) we can filter out those null detections of our arrays
        # print(f"RCNN model-> process_detections: input detections: {detections}")
        # print(f"RCNN model-> process_detections: input image shape: {image_shape}")
        # print(f"RCNN model-> process_detections: input detection boxes: {detections[:,:4]}")
        zero_ix = np.where(detections[:,4]==0)[0]
        N = zero_ix[0] if zero_ix.shape[0]>0 else detections.shape[0]
        # print(f"RCNN model-> process_detections: Number of detections: {N}")
        #Now get only the important boxes, class_ids and scores
        boxes = detections[:N,:4]
        class_ids = detections[:N,4].astype(np.int32)
        scores = detections[:N, 5]
        #Since boxes are in normalized coordinates we have to rescale them
        # scale = np.array([image_shape[0]-1, image_shape[1]-1,image_shape[0]-1, image_shape[1]-1])
        # shift = np.array([0,0,1,1])
        scale = np.array([image_shape[0], image_shape[1],image_shape[0], image_shape[1]])
        shift = np.array([0,0,0,0])
        # print(f"RCNN prediction processing scale: {scale}")
        boxes = np.around(np.multiply(boxes,scale)+shift).astype(np.int32)
        return boxes, class_ids, scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--backbone", help="Backbone to use for prediction, options are \'temnet\', \'resnet101\' or \'resnet101v2\', mind weights are different for each model", default='temnet')
    args = parser.parse_args()
    config = Config(backbone=args.backbone)
    print(f"BATCH_SIZE: {config.BATCH_SIZE}")
    #Get the info from our dataset, since this comes in batches we only take the first image of the first batch for testing
    dataset = Dataset(config.TRAIN_PATH, config, "train")
    print(f"dataset size: {len(dataset)}");
    #get the input numpy arrays out of the dataset
    # dataset is called on batches dataset[i] corresponds to the ith batch of config.BATCH_SIZE images
    #inputs are dataset[i][0][#] and outputs are dataset[i][1][#]
    inputs, outputs = dataset[0][0], dataset[0][1]
    #print(f"dataset 0", dataset[0])
    #print(f"dataset 1", dataset[1])
    batch_imgs = inputs[0]
    batch_img_data = inputs[1]
    batch_rpn_match = inputs[2]
    batch_rpn_bbox = inputs[3]
    batch_gt_class_ids = inputs[4]
    batch_gt_boxes = inputs[5]
    if config.RANDOM_ROIS:
        batch_rpn_rois = inputs[6]
        if config.GENERATE_DETECTION_TARGETS:
            batch_rois = inputs[7]
            batch_rcnn_class_ids = outputs[0]
            batch_rcnn_bbox = outputs[1]
    print(f"batch_imgs size {batch_imgs.size}")
    print(f"batch_imgs type {batch_imgs.dtype}")
    print(f"batch_img_data size {batch_img_data.size}")
    print(f"batch_img_data type {batch_img_data.dtype}")
    print(f"batch_rpn_match size {batch_rpn_match.size}")
    print(f"batch_rpn_match type {batch_rpn_match.dtype}")
    print(f"batch_rpn_bbox size {batch_rpn_bbox.size}")
    print(f"batch_rpn_bbox type {batch_rpn_bbox.dtype}")
    print(f"Eager evaluation? {tf.executing_eagerly()}")
    #Convert all input np arrays to tf tensors
    batch_imgs = tf.convert_to_tensor(batch_imgs, dtype=tf.float32)
    batch_img_data = tf.convert_to_tensor(batch_img_data, dtype=tf.float32)
    batch_rpn_match = tf.convert_to_tensor(batch_rpn_match, dtype=tf.int32)
    batch_rpn_bbox = tf.convert_to_tensor(batch_rpn_bbox, dtype=tf.float32)
    batch_gt_class_ids = tf.convert_to_tensor(batch_gt_class_ids, dtype=tf.int32)
    batch_gt_boxes = tf.convert_to_tensor(batch_gt_boxes, dtype=tf.float32)
    print(f"tf batch_imgs {batch_imgs}")
    print(f"Eager evaluation? {tf.executing_eagerly()}")
    x = KL.ZeroPadding2D((3,3))(batch_imgs)
    #tf.print(x, output_stream=sys.stdout)
    print(x)
    print(f"Eager evaluation? {tf.executing_eagerly()}")
