#Losses for the RPN and RCNN models
#import numpy as np
#import os, csv, re, sys, cv2
import tensorflow as tf
from tensorflow.keras import backend
import tensorflow.keras.layers as KL

import input_pipeline as I

# ********** HELPER FUNCTIONS FOR REGION PROPOSAL NETWORK **********

# ************ LOSS FUNCTIONS ************
def rpn_bbox_loss(config, rpn_match_gt, rpn_bbox_gt, rpn_bbox):
    """
    RPN_bbox_loss: loss function for bounding box output, i.e. the bounding box delta regressions
    Inputs:
      config, a Config class instance (may be modified by user)
      rpn_match_gt: [batch, anchors, 1] the ground truth match set for anchor types (-1, 0, 1) for negative, neutral and positive respectively
      rpn_bbox_gt: [batch, MAX_POSITIVE_ANCHORS, (dy, dx, ln(dh), ln(dw))] the ground truth for bounding box dimensions/coordinates (actually these are the deltas that must be applied to the anchors to fit the ground boxes)
      rpn_bbox: [batch, anchors, (dy, dx, ln(dh), ln(dw))] the output of the RPN prediction
    Ouputs:
      loss: calculated loss for bbox regression
    """
    loss = []
    #for batch_idx in range(config.BATCH_SIZE):
    for batch_idx in range(config.IMAGES_PER_GPU):
        #According to https://arxiv.org/pdf/1504.08083.pdf page 3 there's no ground truth box notion for negative anchors so we must ignore them aswell as neutral anchors
        # Find indices for positive anchors
        match = backend.squeeze(rpn_match_gt[batch_idx], -1)
        positive_idxs = tf.where(backend.equal(match, 1)) #NOTE: may have compatibility issues, if problems are present change it for tf.compat.v1.where(...)

        # Select positive predicted bbox shifts
        bbox = tf.gather_nd(rpn_bbox[batch_idx], positive_idxs)

        # Trim target bounding box deltas to the same length as rpn_bbox
        target_bbox = rpn_bbox_gt[batch_idx, :backend.shape(positive_idxs)[0]]

        # Calculate the loss for the batch
        loss.append(smooth_L1_loss(target_bbox, bbox))
        #print("RPN_BBOX_LOSS STAGE {}:  {}".format(batch_idx, smooth_l1_loss(target_bbox, bbox)))
    return backend.mean(backend.concatenate(loss, 0))

def smooth_L1_loss(yTrue, yPredicted):
    """
    smooth_L1_loss: calculates smooth L1 loss for predicted bounding box shifts
    As implemented in the fast rcnn paper: https://arxiv.org/pdf/1504.08083.pdf
    Inputs:
      yTrue, ground truth bounding box shifts
      yPredicted, predicted bounding box shifts
    Outputs:
      loss, value of smooth L1 loss
"""
    absDiff = backend.abs(yTrue - yPredicted)                             # Calculate absolute value of difference between ground truth BB and predicted BB
    mask = backend.cast(backend.less(absDiff, 1.0), "float32")            # Find indices of values less than 1 (if absolute difference is less than 1, cast as float 32)
    loss = (mask * ((absDiff ** 2) * 0.5) + (absDiff - 0.5)*(1 - mask))   # L1 loss equation (only valid for indices of less than 1)
    #print("L1 Loss: {}".format(loss))
    return loss

def rpn_match_loss(rpn_match_true, rpn_class_logits):
    """
    RPN_loss: loss function for RPN anchor matches
    A match corresponds to the type of an anchor (positive, negative, neutral)
    Usual policy is to have 1 for positive anchor, 0 for neutral anchor, -1 for negative anchor
    Inputs:
      rpn_match_true: [BATCH, NUM_ANCHORS, 1] the ground truth for anchor match type [-1, 0, 1]
      rpn_class_logits: [BATCH, NUM_ANCHORS, 2] the 'logit' function for foreground/background in the RPN
    Outputs:
      loss: the loss value for the ground truth and logits
    """
    rpn_match_true = tf.squeeze(rpn_match_true, -1)                                           # Squeeze last dimension from ground truth matches to simplify
    anchor_class = backend.cast(backend.equal(rpn_match_true, 1), tf.int32)               # Convert to anchor class 0 or 1 for negative or positive anchor (from -1 and 1)
    #Positive and negative anchors contribute to the loss calculation but neutral values don't so let's localize them and exclude them
    indices = tf.where(backend.not_equal(rpn_match_true, 0))                             # Find indices of positive and negative anchors #NOTE: There might be an incompatibility with TF2 here, if this generates an error change it by tf.compat.v1.where(...)

    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)                                # Find only rows that contribute to loss
    anchor_class = tf.gather_nd(anchor_class, indices)                                # Find only rows that contribute to loss

    # Set our own definition of loss function to pass to model fit, this is a categorical crossentropy as per the Fast RCNN paper
    loss = backend.sparse_categorical_crossentropy(target = anchor_class,
                                                   from_logits = True,
                                                   output = rpn_class_logits)
    # Pre-process loss
    loss = backend.switch(tf.size(input=loss) > 0, backend.mean(loss), tf.constant(0.0))
    #print("RPN MATCH LOSS: {}".format(loss))
    return loss

def rcnn_class_loss(target_class_ids, pred_class_logits):
    """
    Loss for the classifier heads of RCNN
    Inputs:
      target_class_ids: [batch, num_rois] Ground truth class_ids for the rois
      pred_class_ids: [batch, num_rois, num_classes] Predicted class_ids for the rois by the network
    Outputs:
      loss: Returns crossentropy loss for the predictions, this is consistent with the Fast RCNN paper
    """
    # print(f"H.rcnn_class_loss: pred_class_logits shape: {tf.shape(pred_class_logits)}")
    # print(f"H.rcnn_class_loss: target_class_ids shape: {tf.shape(target_class_ids)}")
    # print(f"H.rcnn_class_loss: pred_class_logits: {pred_class_logits}")
    # print(f"H.rcnn_class_loss: target_class_ids shape: {target_class_ids}")
    target_class_ids = tf.cast(target_class_ids, 'int64')
    # pred_class_logits = tf.argmax(input=pred_class_logits, axis=2)
    # Loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_class_ids, logits=pred_class_logits)
    # print(f"H.rcnn_class_loss: loss shape: {tf.shape(loss)}")

    # Computer loss mean. Use only predictions that contribute
    # to the loss to get a correct mean.
    # loss = tf.reduce_sum(input_tensor=loss) / tf.reduce_sum(input_tensor=pred_active)
    # loss = backend.mean(loss)
    loss = tf.reduce_mean(loss)
    return loss

def rcnn_bbox_loss(config, target_bbox, target_class_ids, pred_bbox):
    """
    Loss for the box delta regression of the RCNN, this is L_{loc} in https://arxiv.org/pdf/1504.08083.pdf
    Inputs:
      target_bbox: [batch, num_rois, (dy, dx, ln(dh), ln(dw))] GT box deltas for rois
      target_class_ids: [batch, num_rois] Integer class IDs of the rois
      pred_bbox: [batch, num_rois, num_classes, (dy, dx, ln(dh), ln(dw))] predicted box deltas for rois
    """
    # print(f"H.rcnn_bbox_loss: target_bbox: {target_bbox}")
    # print(f"H.rcnn_bbox_loss: target_class_ids: {target_class_ids}")
    # print(f"H.rcnn_bbox_loss: pred_bbox: {pred_bbox}")
    """
    loss = []
    for batch_idx in range(config.BATCH_SIZE):
        # Reshape to merge batch and roi dimensions to simplify
        tg_class_ids = backend.reshape(target_class_ids[batch_idx],(-1))
        tg_bbox = backend.reshape(target_bbox[batch_idx],(-1,4))
        pd_bbox = backend.reshape(pred_bbox[batch_idx],(-1, backend.int_shape(pred_bbox)[2],4))

        #For loss calculation only the positive rois contribute and only the right class:id for each of them
        positive_roi_idx = tf.where(tg_class_ids > 0)[:, 0]
        positive_roi_class_ids = tf.cast(tf.gather(tg_class_ids, positive_roi_idx),tf.int64)
        indices = tf.stack([positive_roi_idx, positive_roi_class_ids],axis=1)
        #Gather the pred and target deltas that contribute to the loss
        tg_bbox = tf.gather(tg_bbox, positive_roi_idx)
        pd_bbox = tf.gather_nd(pd_bbox, indices)
        #Calculate the loss for the batch
        loss.append(smooth_L1_loss(tg_bbox, pd_bbox))
        print(f"RCNN_BBOX_LOSS BATCH {batch_idx}: {loss[-1]}")
    return backend.mean(backend.concatenate(loss,0))
    """
    #Simplify merging batch and roi dimensions
    target_class_ids = backend.reshape(target_class_ids,(-1,))
    target_bbox = backend.reshape(target_bbox,(-1,4))
    pred_bbox = backend.reshape(pred_bbox,(-1, backend.int_shape(pred_bbox)[2],4))
    
    #Get the indices of the positive ROIs since only these contribute to the loss
    positive_roi_idx = tf.where(target_class_ids > 0)[:,0]
    positive_roi_class_ids = tf.cast(tf.gather(target_class_ids, positive_roi_idx), tf.int64)
    indices = tf.stack([positive_roi_idx, positive_roi_class_ids], axis=1)

    # Gather the bbox deltas that contribute to the loss
    target_bbox = tf.gather(target_bbox, positive_roi_idx)
    pred_bbox = tf.gather_nd(pred_bbox, indices)
    # Add the L1 loss
    loss = backend.switch(tf.size(input=target_bbox)>0, smooth_L1_loss(target_bbox, pred_bbox), tf.constant(0.0))
    return backend.mean(loss)

#**************************CONVERT LOSSES TO LAYERS************
"""
# Regularizer class from keras documentation as reference
class MyActivityRegularizer(Layer):
  #Layer that creates an activity sparsity regularization loss.

  def __init__(self, rate=1e-2):
    super(MyActivityRegularizer, self).__init__()
    self.rate = rate

  def call(self, inputs):
    # We use `add_loss` to create a regularization loss
    # that depends on the inputs.
    self.add_loss(self.rate * tf.reduce_sum(tf.square(inputs)))
    return inputs
# These definitions replace the KL.Lambda layers and add loss in the layers so they are automatically found at compile time
rpn_class_loss = KL.Lambda(lambda x: H.rpn_match_loss(*x), name="rpn_class_loss")([input_rpn_match, rpn_class_logits])
rpn_bbox_loss = KL.Lambda(lambda x: H.rpn_bbox_loss(self.config, *x), name="rpn_bbox_loss")([input_rpn_match, input_rpn_bbox, rpn_bbox])
class_loss = KL.Lambda(lambda x: H.rcnn_class_loss(*x), name="rcnn_class_loss")([target_class_ids, rcnn_class_logits])
bbox_loss = KL.Lambda(lambda x: H.rcnn_bbox_loss(self.config, *x), name="rcnn_bbox_loss")([target_bbox, target_class_ids, rcnn_bbox])

"""
class RPNBBoxLoss(KL.Layer):
  """Layer that creates an activity sparsity regularization loss."""
  #Gonn be called like rpn_bbox_loss = KL.Lambda(lambda x: H.rpn_bbox_loss(self.config, *x), name="rpn_bbox_loss")([input_rpn_match, input_rpn_bbox, rpn_bbox])
  def __init__(self, config, **kwargs):
    super(RPNBBoxLoss, self).__init__(**kwargs)
    self.config = config
    #self.name = name

  def call(self, inputs):
    input_rpn_match = inputs[0]
    input_rpn_bbox = inputs[1]
    rpn_bbox = inputs[2]
    rpn_bbox_loss_output = rpn_bbox_loss(self.config, input_rpn_match, input_rpn_bbox, rpn_bbox)
    # We use `add_loss` to create a regularization loss
    # that depends on the inputs.
    #self.add_loss(self.rate * tf.reduce_sum(tf.square(inputs)))
    loss = tf.reduce_mean(input_tensor=rpn_bbox_loss_output, keepdims=True) * self.config.LOSS_WEIGHTS.get(self.name, 1.)
    self.add_loss(loss)
    self.add_metric(loss, name=self.name, aggregation='mean')
    return rpn_bbox_loss_output

class RPNClassLoss(KL.Layer):
  """Layer that creates an activity sparsity regularization loss."""
  def __init__(self, config, **kwargs):
    super(RPNClassLoss, self).__init__(**kwargs)
    self.config = config
    #self.name = name

  def call(self, inputs):
    input_rpn_match = inputs[0]
    rpn_class_logits = inputs[1]
    rpn_class_loss_output = rpn_match_loss(input_rpn_match, rpn_class_logits)
    loss = tf.reduce_mean(input_tensor=rpn_class_loss_output, keepdims=True) * self.config.LOSS_WEIGHTS.get(self.name, 1.)
    self.add_loss(loss)
    self.add_metric(loss, name=self.name, aggregation='mean')
    return rpn_class_loss_output

class RCNNClassLoss(KL.Layer):
  """Layer that creates an activity sparsity regularization loss."""
  def __init__(self, config, **kwargs):
    super(RCNNClassLoss, self).__init__(**kwargs)
    self.config = config
    #self.name = name

  def call(self, inputs):
    target_class_ids = inputs[0]
    rcnn_class_logits = inputs[1]
    rcnn_class_loss_output = rcnn_class_loss(target_class_ids, rcnn_class_logits)
    loss = tf.reduce_mean(input_tensor=rcnn_class_loss_output, keepdims=True) * self.config.LOSS_WEIGHTS.get(self.name, 1.)
    self.add_loss(loss)
    self.add_metric(loss, name=self.name, aggregation='mean')
    return rcnn_class_loss_output

class RCNNBBoxLoss(KL.Layer):
  """Layer that creates an activity sparsity regularization loss."""
  def __init__(self, config, **kwargs):
    super(RCNNBBoxLoss, self).__init__(**kwargs)
    self.config = config
    #self.name = name

  def call(self, inputs):
    target_bbox = inputs[0]
    target_class_ids = inputs[1]
    rcnn_bbox = inputs[2]
    rcnn_bbox_loss_output = rcnn_bbox_loss(self.config, target_bbox, target_class_ids, rcnn_bbox)
    loss = tf.reduce_mean(input_tensor=rcnn_bbox_loss_output, keepdims=True) * self.config.LOSS_WEIGHTS.get(self.name, 1.)
    self.add_loss(loss)
    self.add_metric(loss, name=self.name, aggregation='mean')
    return rcnn_bbox_loss_output

#**************************ACCURACY METRICS ******************************
#TODO: mAP doesn't seem to be working when appended to the metrics for training, possible solution might be changing it to use only tf functions and tf tensors
def find_matches_tf(gt_boxes, gt_class_ids, pred_boxes, pred_class_ids,pred_scores, iou_threshold=0.5, score_threshold=0.0):
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
    #Sorting the predictions by score
    idx = tf.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[idx]
    pred_class_ids = pred_class_ids[idx]
    pred_scores = pred_scores[idx]

    #Calculate the IoU scores between predicted and gt boxes
    IoU = I.calculate_iou_matrix_tf(pred_boxes, gt_boxes)

    #Loop through pred_boxes and find the matching gt_boxes
    pred_match = -1*backend.ones([backend.shape(pred_boxes)[0]])
    gt_match = -1*backend.ones([backend.shape(gt_boxes.shape)[0]])
    for pred_i in range(len(backend.shape(pred_boxes)[0])):
        #For every pred_box find the best matching gt_box
        #1st sort the IoUs by score from high to low
        sorted_idx = tf.argsort(IoU[pred_i])[::-1]
        #2nd keep only the predictions above our score_threshold
        sorted_idx = sorted_idx[tf.where(IoU[pred_i,sorted_idx]>score_threshold)]
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

def mAP_accuracy(gt_boxes, gt_class_ids, pred_boxes, pred_class_ids,
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
    #Remember
    #precision=true_positives/(true_positives+false_positives)
    #recall=true_positives/(true_positives+false_negatives)
    len_pred_match = len(backend.shape(pred_match)[0])
    len_gt_match = len(backend.shape(gt_match)[0])
    precisions = backend.cumsum(pred_match > -1) / (tf.range(len_pred_match) + 1)
    recalls = backend.cumsum(pred_match > -1).astype(np.float32) / len_gt_match

    # Pad with start and end values to simplify the math in the next step
    precisions = tf.concat([[0], precisions, [0]])
    recalls = tf.concat([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    # read https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173 for reference
    for i in range(len(backend.shape(precisions)) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1]) #This is choosing the max to the right

    # Compute mean AP over recall range
    # Note we skip the first and last values to avoid the padding
    # And then we add 1 to make sure our indices match the padded array
    indices = tf.where(recalls[:-1] != recalls[1:])[0] + 1
    # Remember we calculate the AP from the precision recall curve
    #  AP = \int_0^1 precision(recall) drecall
    # Since the AP is the integral (the area under the curve) 
    # and we made sureour curve is composed of blocks
    # it can be calculated simply as the area of the rectangle (recall_i+1 -recall_i)*precision_i
    mAP = backend.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])

    return mAP, precisions, recalls, IoU


