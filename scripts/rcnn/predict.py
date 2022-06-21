
import os, argparse, csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

from model import RCNN
#import helpers as H
import input_pipeline as I
import visualize as V
from config import Config, Dataset, Image

print("TensorFlow Version ",tf.__version__)

IMG_FORMAT = ('.tif','.png','.jpg','.jpeg','.bpm','.eps')


def get_anchors(image_shape, config):
  """Returns anchor pyramid for the given image size."""
  # backbone_shapes = I.cnn_input_shapes(self.config, image_shape)
  backbone_shapes = I.cnn_input_shapes(image_shape, config.BACKBONE_STRIDES)
  # Cache anchors and reuse if image shape is the same
  _anchor_cache = {}
  if not tuple(image_shape) in _anchor_cache:
    # Generate Anchors
    a = I.generate_anchors(
      config.RPN_ANCHOR_SCALES,
      config.RPN_ANCHOR_RATIOS,
      backbone_shapes,
      config.BACKBONE_STRIDES,
      config.RPN_ANCHOR_STRIDE,
      config.IMAGE_SHAPE)
    # Keep a copy of the latest anchors in pixel coordinates because
    # it's used in inspect_model notebooks.
    # TODO: Remove this after the notebook are refactored to not use it
    # self.anchors = a
    # Normalize coordinates
    _anchor_cache[tuple(image_shape)] = I.norm_boxes(a, image_shape[:2])
  return _anchor_cache[tuple(image_shape)]

def predict_batch(images, image_metas, config, keras_model):
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
  # Our model needs three tensors as inputs (see the build_entire_model function):
  # Image tensor representation, tensor of image meta and anchors
  # Image tensor and meta are given from the Dataset class and used as inputs for this function
  # Create the anchors
  anchors = get_anchors(images[0].shape, config)
  #Duplicate images so they are the same shape as the images given
  anchors = np.broadcast_to(anchors, (images.shape[0],)+anchors.shape)
  
  #Now run keras_model's object detection from our arguments
  #outputs are detections, rcnn_class, rcnn_bbox, rpn_rois, rpn_class and rpn_bbox we need only the first
  detections, rcnn_class, rcnn_bbox, rpn_rois, rpn_class, rpn_bbox = keras_model.predict([images, image_metas, anchors])
  rcnn_class = np.squeeze(rcnn_class)
  rcnn_bbox = np.squeeze(rcnn_bbox)
  rpn_rois = np.squeeze(rpn_rois)
  rpn_class = np.squeeze(rpn_class)
  rpn_bbox = np.squeeze(rpn_bbox)
  #Now divide the detections into their components
  results = []
  for i, image in enumerate(images):
    final_rois, final_class_ids, final_scores = process_detections(detections[i], images[i].shape)
    results.append({
      "rois": final_rois,
      "class_ids": final_class_ids,
      "scores":final_scores
    })
  return results

def process_detections(detections, image_shape):
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
  zero_ix = np.where(detections[:,4]==0)[0]
  N = zero_ix[0] if zero_ix.shape[0]>0 else detections.shape[0]
  #Now get only the important boxes, class_ids and scores
  boxes = detections[:N,:4]
  class_ids = detections[:N,4].astype(np.int32)
  scores = detections[:N, 5]
  #Since boxes are in normalized coordinates we have to rescale them
  scale = np.array([image_shape[0], image_shape[1],image_shape[0], image_shape[1]])
  shift = np.array([0,0,0,0])
  boxes = np.around(np.multiply(boxes,scale)+shift).astype(np.int32)
  return boxes, class_ids, scores

def visualize(dataset, config, imgName = 'test'):
  """
  visualize: create graphs of predicted bounding boxes and classes for labeled data
  Inputs:
  dataset, dataset object
  config, config object
  imgName, the name of the image to be visualized
  """
  inputs = dataset[0][0]
  images_gt = inputs[0]      # batch_img in __getitem__
  images_meta = inputs[1]    # batch_img_data in __getitem__
  rpn_match_gt = inputs[2]   # batch_rpn_match in __getitem__
  rpn_bbox = inputs[3]       # batch_rpn_bbox in __getitem__
  gt_class_ids = inputs[4]   # batch_gt_class_ids in __getitem__
  gt_boxes = inputs[5]       # batch_gt_boxes in __getitem__
  # print(f"visualize: images_gt.shape {images_gt.shape}")
  # print(f"visualize: images_meta.shape {images_meta.shape}")
  # print(f"visualize: gt_boxes.shape {gt_boxes.shape}")
  # print(f"visualize: gt_class_ids.shape {gt_class_ids.shape}")
  #anchors = dataset.anchors
  #positive_anchors = anchors[np.where(rpn_match_gt == 1)[0]]
  #negative_anchors = anchors[np.where(rpn_match_gt == -1)[0]]
  #V.visualize_training_anchors(positive_anchors, negative_anchors, np.uint8(image_gt), imgName)
  # predict(config, imgName, image_gt, anchors)
  # print(f"images_meta[0] : {images_meta[0]}")
  #imgName = str(int(images_meta[0][0]))
  imgName = I.build_image_names(images_meta)[0]
  print(f"Predicting on image: {imgName}")
  rcnn = RCNN(config, 'inference')
  rcnn.keras_model.load_weights(config.WEIGHT_SET, by_name=True)
  # predictions = rcnn.predict_batch(np.expand_dims(images_gt,0), np.expand_dims(images_meta,0))
  predictions = rcnn.predict_batch(images_gt, images_meta)
  pred_0 = predictions[0]
  # print(f"visualize: images_gt[0].shape[0] : {images_gt[0].shape[0]}")
  pred_boxes = pred_0["rois"]
  # print(f"visualize: pred_0[\"rois\"] : {pred_boxes}")
  #Scale prediction rois to the image scale and limit values to fit under 512 and over 512
  # pred_boxes = I.denorm_boxes(pred_boxes, images_gt.shape[0:2])
  pred_boxes[pred_boxes < 0.] = 0.
  pred_boxes[pred_boxes>images_gt[0].shape[0]] = images_gt[0].shape[0]
  # print(f"visualize: pred_boxes : {pred_boxes}")
  V.visualize_rcnn_predictions_multiclass(np.uint8(images_gt[0]), pred_boxes, pred_0["class_ids"], pred_0["scores"], config.BACKBONE+'_'+imgName, config)
  mAP = I.compute_mAP(gt_boxes[0], gt_class_ids[0], pred_boxes, pred_0["class_ids"], pred_0["scores"])
  print(f"### mAP obtained for predictions on image {imgName} : {mAP}")
  #pred_boxes, pred_class_ids, pred_scores = predict_rcnn(config, imgName, image_gt, image_meta, anchors)

def predict_all_rcnn(dataset, config):
  """
  predict_all_rcnn: create graphs of predicted bounding boxes and classes for all labeled data aswell as calculating the average mAP for the dataset
  Inputs:
  dataset, dataset object
  config, config object
  """
  mAPs = []
  rcnn = RCNN(config, 'inference')
  rcnn.keras_model.load_weights(config.WEIGHT_SET, by_name=True)
  for i in range(len(dataset)): #For every batch in the dataset
    inputs = dataset[i][0]
    images_gt = inputs[0]      # batch_img in __getitem__
    images_meta = inputs[1]    # batch_img_data in __getitem__
    rpn_match_gt = inputs[2]   # batch_rpn_match in __getitem__
    rpn_bbox = inputs[3]       # batch_rpn_bbox in __getitem__
    gt_class_ids = inputs[4]   # batch_gt_class_ids in __getitem__
    gt_boxes = inputs[5]       # batch_gt_boxes in __getitem__
    #print(f"predict_all_rcnn: images_gt.shape {images_gt.shape}")
    #print(f"predict_all_rcnn: images_meta.shape {images_meta.shape}")
    #print(f"predict_all_rcnn: gt_boxes.shape {gt_boxes.shape}")
    #print(f"predict_all_rcnn: gt_class_ids.shape {gt_class_ids.shape}")
    #print(f"predict_all_rcnn: images_meta[0] : {images_meta[0]}")
    imgNames = I.build_image_names(images_meta)
    predictions = rcnn.predict_batch(images_gt, images_meta)
    for j, pred in enumerate(predictions): #For every prediction in the batch
      print(f"Predicting on image: {imgNames[j]} ...")
      pred_boxes = pred["rois"]
      # print(f"predict_all_rcnn: pred[\"rois\"] : {pred_boxes}")
      #Scale prediction rois to the image scale and limit values to fit under 512 and over 512
      # pred_boxes = I.denorm_boxes(pred_boxes, images_gt.shape[0:2])
      pred_boxes[pred_boxes < 0.] = 0.
      pred_boxes[pred_boxes>images_gt[j].shape[0]] = images_gt[j].shape[0]
      # print(f"predict_all_rcnn: pred_boxes : {pred_boxes}")
      V.visualize_rcnn_predictions_multiclass(np.uint8(images_gt[j]), pred_boxes, pred["class_ids"], pred["scores"], config.BACKBONE+'_'+imgNames[j], config)
      mAP, _, _, _,_ = I.compute_mAP(gt_boxes[0], gt_class_ids[0], pred_boxes, pred["class_ids"], pred["scores"])
      print(f"### mAP obtained for predictions on image {imgNames[j]} : {mAP}")
      mAPs.append(mAP)
  avg_mAP = np.mean(mAPs)
  print(f"### Average mAP for dataset: {avg_mAP}")
  return avg_mAP, mAPs

"""
predict: Predict on a new dataset using weight set, visualize results every step of the way
Inputs:
  dataset, a Dataset object
  config, a Config object
  weights, the path to the weight set to be loaded
  imgName, the semantic label for the image to be predicted upon
Outputs:
  regions, an array of coordinates corresponding to predicted regions of interest
"""
# TODO: ADD SUPPORT FOR LOADING IMAGE HERE
### OLD IMPLEMENTATION FOR RPN, WILL NOT WORK UNLESS MODIFIED, use predict_uncropped_image, visualize or predict_all_rcnn
def predict(config, imgName, image_gt, image_meta, image_anchors):
  rcnn = RCNN(config, 'inference')
  rcnn.keras_model.load_weights(config.WEIGHT_SET, by_name=True)

  predictions, rcnn_class, rcnn_bbox, rpn_rois, rpn_match, rpn_bbox = rpn.model.predict(np.expand_dims(image_gt, 0),np.expand_dims(image_meta, 0),np.expand_dims(image_anchors, 0))
  rcnn_class = np.squeeze(rcnn_class)
  rcnn_bbox = np.squeeze(rcnn_bbox)
  rpn_rois = np.squeeze(rpn_rois)
  rpn_match = np.squeeze(rpn_match)
  rpn_bbox = np.squeeze(rpn_bbox) #* config.RPN_BBOX_STD_DEV

  # Find where positive predictions took place
  positive_idxs = np.where(np.argmax(rpn_match, axis=1) == 1)[0]

  # Get the predicted anchors for the positive anchors
  # print(f"Anchors shape: {np.shape(image_anchors)}")
  # print(f"RPN Bbox shape: {np.shape(rpn_bbox)}")
  # print(f"Positive Anchors shape: {np.shape(anchors[positive_idxs])}")
  # print(f"Positive RPN Bbox shape: {np.shape(rpn_bbox[positive_idxs])}")
  predicted_anchors = I.shift_bboxes(image_anchors[positive_idxs], rpn_bbox[positive_idxs])
  V.visualize_bboxes(np.uint8(image_gt), predicted_anchors, imgName)

  # Sort predicted class by strength of prediction
  argsort = np.flip(np.argsort(rpn_match[positive_idxs, 1]), axis=0)
  sorted_anchors = predicted_anchors[argsort]
  sorted_anchors = sorted_anchors[:min(100, sorted_anchors.shape[0])]
  V.visualize_rpn_predictions(np.uint8(image_gt), sorted_anchors, imgName)
  return sorted_anchors

def predict_one_image(img_paths, config, save_fig):
  """
  predict_one_image: Predicts on a single image passed through a path
  Inputs:
    img_paths: Array of paths to the images to predict
    config: Config class instance containing batch size, weight set and image dimensions
  Outputs:
    None, uses matplotlib to generate graphs of the predictions (see visualization.py)
  """
  #First build the image as a batch for predictions using rcnn.predict_batch, this is done through the Image class
  #This is pretty much a replication of what we do in the Dataset class but built for just one image
  #Since prediction only requires an array of images and an array of image_metadata we only build these.
  #This would be the equivalent of using a batch size of 1 and a dataset of 1 image, the difference is that we don't need ground truth to generate the predictions, this also means that since we're not comparing to any ground thruth we can't calculate mAP
  #Load image from path
  rcnn = RCNN(config, 'inference')
  rcnn.keras_model.load_weights(config.WEIGHT_SET, by_name=True)
  images = Image(img_paths, config)
  overall_class_ids = []
  overall_scores = []
  overall_boxes = []
  pred_imgs = []
  for i in range(len(images)):#For every image read from the parsed paths
    inputs = images[i]
    image_batch = inputs[0]
    image_batch_meta = inputs[1]
    imgNames = I.build_image_names(image_batch_meta)
    predictions = rcnn.predict_batch(image_batch, image_batch_meta)
    for j, pred in enumerate(predictions): #For every prediction in the batch
      print(f"Predicting on image: {imgNames[j]} ...")
      pred_boxes = pred["rois"]
      # print(f"predict_all_rcnn: pred[\"rois\"] : {pred_boxes}")
      #Scale prediction rois to the image scale and limit values to fit under 512 and over 512
      # pred_boxes = I.denorm_boxes(pred_boxes, images_gt.shape[0:2])
      pred_boxes[pred_boxes < 0.] = 0.
      pred_boxes[pred_boxes>image_batch[j].shape[0]] = image_batch[j].shape[0]
      # print(f"predict_all_rcnn: pred_boxes : {pred_boxes}")
      pred_img = V.visualize_rcnn_predictions_multiclass(np.uint8(image_batch[j]), pred_boxes, pred["class_ids"], pred["scores"], config.BACKBONE+'_'+imgNames[j], config, save_fig)
      pred_imgs.append(pred_img)
      overall_class_ids += list(pred["class_ids"])
      overall_scores += list(pred["scores"])
      overall_boxes += list(pred_boxes)
  return overall_boxes, overall_class_ids, overall_scores, pred_imgs

#TODO: modify predict_uncropped_image so the window/crops data are generated in a class instead of inside this function, use Image class implementation as a guide
def predict_uncropped_image(img_path, crop_size, crop_step, config, rcnn, save_fig=True):
  """
  predict_uncropped_image: Build predictions on an uncropped image by creating a set of cropped images to represent it, predicting on those and then uniting all predictions onto the original image coordinates
  INPUTS:
      img_path: path to read the image from
      config: Config class instance with the batch_size and dataset image size
  OUTPUTS:
      if save_img == True:
         None, uses matplotlib to save an image with the preditions drawn on top
      else:
         pred_img, the image with the predictions

  """
  #Load image from path
  img_name = img_path.split('/')[-1].split('.')[0] #Take only the number part, that is '/path/to/img/133433.png' -> '133433'
  print(f"Loading image from {img_path}")
  source_image = np.uint8(img_to_array(load_img(img_path)))
  #If the image is bigger than the DATASET_SIZE in config proced to crop into smaller images proper to the model
  h, w = source_image.shape[:2]
  if h > config.DATASET_IMAGE_SIZE[0] or w > config.DATASET_IMAGE_SIZE[1]:
    # crop_size = (1024, 1024)#ALL DIMENSIONS SHOULD BE FORMATED H,W; that is y, x else the cropping algorithm will skip parts of the image and make dupliactes of others
    # crop_step = (500,500)
    cropped_images, cropped_images_shifts = multicrop_input_image(source_image, crop_size, crop_step) # Get the cropped images as well as their position in the original image
    #Now order them in batches for predictions using rcnn.predict_batch
    #This is pretty much a replication of what we do in the Dataset class but since prediction only requires an array of images and an array of image_metadata we only build these.
    #TODO: make this into a class, use the Image class in config.py as a guide for implementation
    #Number of batches to split the cropped images into
    len_dataset = int(np.ceil(len(cropped_images)/config.BATCH_SIZE))
    dataset=[]
    for idx in range(len_dataset):#Build each batch
      batch_cropped_images = np.array(cropped_images[idx * config.BATCH_SIZE : (idx +1) * config.BATCH_SIZE]) #cropped images to be processed int the idx batch
      batch_cropped_images_shifts = np.array(cropped_images_shifts[idx * config.BATCH_SIZE : (idx +1) * config.BATCH_SIZE])#Keep these to track the shifts corresponding to each batch on prediction
      # Pixel level information of images in the batch with the correct input size
      #batch_imgs = tf.image.resize(batch_cropped_images, list(config.DATASET_IMAGE_SIZE))
      batch_imgs = np.zeros((config.BATCH_SIZE,
                             config.IMAGE_SHAPE[0],
                             config.IMAGE_SHAPE[1],
                             config.NUM_CHANNELS), dtype=np.float32)
      # Info parsed from the images
      #batch_img_data = np.array([I.compose_image_data(str(idx)+str(batch_id)+img_name, config.DATASET_IMAGE_SIZE, config.IMAGE_SHAPE) for batch_id in range(len(batch_cropped_images))])
      batch_img_data = np.zeros((config.BATCH_SIZE, config.IMAGE_DATA_SIZE)) #Img id, og size and size after resizing
      for batch_id in range(len(batch_cropped_images)):#For each image in the batch
        img = tf.image.resize(batch_cropped_images[batch_id], list(config.IMAGE_SHAPE))
        img_data = I.compose_image_data(str(idx)+str(batch_id)+img_name, config.DATASET_IMAGE_SIZE, config.IMAGE_SHAPE)
        #Add image and data to the batch arrays
        batch_imgs[batch_id] = img
        batch_img_data[batch_id] = img_data
      inputs=[batch_imgs, batch_img_data, batch_cropped_images_shifts]
      dataset.append(inputs)
    #NOTE: ^The previous block of code can be converted into a class and it would look prettier

    #Dataset should now be built, now we can start with the predictions and storing them in arrays
    all_pred_boxes = []
    all_pred_class_ids = []
    all_pred_scores = []
    for i in range(len(dataset)):
      inputs = dataset[i]
      image_batch = inputs[0]
      image_batch_meta = inputs[1]
      image_batch_shifts = inputs[2]
      imgNames = I.build_image_names(image_batch_meta)
      predictions = []
      try:
        predictions = rcnn.predict_batch(image_batch, image_batch_meta)
      except AttributeError:
        print('RCNN from script failed. Using internal Keras model for backbone')
        predictions = predict_batch(image_batch, image_batch_meta, config, rcnn)

      for j, pred in enumerate(predictions): #For every prediction in the batch
        print(f"Predicting on image: {imgNames[j]} ...")
        pred_boxes = pred["rois"]
        pred_class_ids = pred["class_ids"]
        pred_scores = pred["scores"]
        # print(f"predict_all_rcnn: pred[\"rois\"] : {pred_boxes}")
        #Limit values to fit under 512 and over 512 and scale prediction rois to the image scale
        pred_boxes[pred_boxes < 0.] = 0.
        pred_boxes[pred_boxes>image_batch[j].shape[0]] = image_batch[j].shape[0]
        # print(f"predict_all_rcnn: pred_boxes : {pred_boxes}")
        scale_y = crop_size[0]/config.IMAGE_SHAPE[0]
        scale_x = crop_size[1]/config.IMAGE_SHAPE[1]
        #pred_boxes = I.denorm_boxes(pred_boxes, config.DATASET_IMAGE_SIZE)
        pred_boxes = np.multiply(pred_boxes, np.array([scale_y, scale_x, scale_y, scale_x]))
        # print(f"predict_all_rcnn: pred_boxes : {pred_boxes}")
        #Now shift those predictions to their real place in the source image
        if j < len(image_batch_shifts):
          shift_y, shift_x = image_batch_shifts[j]
        else:
          shift_y, shift_x = 0 , 0 #There might be zero paddings on the batches so this is a safety measure
        pred_boxes = pred_boxes+np.array([shift_y,shift_x,shift_y,shift_x])
        #Add predictions to our global arrays
        for k in range(len(pred_boxes)):
          all_pred_boxes.append(pred_boxes[k])
          all_pred_class_ids.append(pred_class_ids[k])
          all_pred_scores.append(pred_scores[k])
    all_pred_boxes = np.array(all_pred_boxes)
    all_pred_class_ids = np.array(all_pred_class_ids)
    all_pred_scores = np.array(all_pred_scores)
    # print(f"all_pred_boxes before nms: {all_pred_boxes}")
    nms_idx = I.nms3(all_pred_boxes, all_pred_scores, all_pred_class_ids, nms_treshold=0.25)
    # print(f"nms_idx: \n {nms_idx}")
    nms_pred_boxes = all_pred_boxes[nms_idx]
    nms_pred_class_ids = all_pred_class_ids[nms_idx]
    nms_pred_scores = all_pred_scores[nms_idx]
    print(f"### Finished Predicting on cropped images ###")
    print(f"Final number of detections: {len(all_pred_boxes)}")
    print(f"Final number of detections after nms: {len(nms_pred_boxes)}")
    # V.visualize_rcnn_predictions(source_image, all_pred_boxes, all_pred_class_ids, all_pred_scores, img_name)
    pred_image = V.visualize_rcnn_predictions_multiclass(source_image, nms_pred_boxes, nms_pred_class_ids, nms_pred_scores, config.BACKBONE+'_'+img_name, config, save_fig)
    V.visualize_predictions_count_multiclass(nms_pred_class_ids, nms_pred_scores, config.BACKBONE+'_'+img_name, config, config.DETECTION_EXCLUDE_BACKGROUND)
    #V.visualize_score_histograms(nms_pred_class_ids, nms_pred_scores, img_name)
    return nms_pred_boxes, nms_pred_class_ids, nms_pred_scores, pred_image
  else:
    one_boxes, one_class_ids, one_scores, one_img = predict_one_image([img_path], config, save_fig)
    return np.array(one_boxes), np.array(one_class_ids), np.array(one_scores), np.array(one_img)

def crop_image(image, crop_size, starting_point):
  """
  crop_image: Crop an image to a region determined by a given size and starting point
  Inputs:
      image: numpy matrix representation of the source image
      crop_size: (height, width) dimensions of the cropped image
      starting_point: (y,x) coordinate of the original image in pixels where the zero of the cropped image is located
  Outputs:
      aug_image: pixel level information of the image in the cropped region
  """
  start_y = starting_point[0]
  start_x = starting_point[1]
  end_y = starting_point[0]+crop_size[0]
  end_x = starting_point[1]+crop_size[1]
  aug_image = image[start_y:end_y, start_x:end_x]
  return aug_image


def multicrop_input_image(image, crop_size, crop_step):
  """
  multicrop_input_image: Convers an image into multiple small cropped images
  Inputs:
      image: numpy matrix representation of source image to crop
      crop_size: (height, width) dimensions of the resulting cropped images
      crop_step: (shift_y, shift_x) size of the step in pixels to displace the cropping region
  Outputs:
      crop_imgs: List of croped images, essentially a list of numpy matrixes
      crop_shifts: List of the pixel coordinates where each of the crop_images belong in the source image
  """
  crop_imgs = []
  crop_shifts = []
  image_height, image_width = image.shape[:2] #Define this here to avoid multiple calls to .shape
  if image_width//crop_step[1] == image_width/crop_step[1]:
    nx = image_width//crop_step[1]
  else:
    nx = 1+image_width//crop_step[1]
  if image_height//crop_step[0] == image_height/crop_step[0]:
    ny = image_height//crop_step[0]
  else:
    ny = 1+image_height//crop_step[0]
  #print(f"nx: {nx}, ny: {ny}")
  for i in range(nx):
    for j in range(ny):
      starting_point = (j*crop_step[0], i*crop_step[1]) #That is y , x
      if starting_point[0]+crop_size[0] > image_height:#If the cropped image would go beyond image height, move it back inside
        starting_point = (image_height-crop_size[0],starting_point[1])
      if starting_point[1]+crop_size[1] > image_width :#If the cropped image would go beyond image width, move it back inside
        starting_point = (starting_point[0],image_width-crop_size[1])
      #print(f"starting point: {starting_point} for i: {i}, j: {j}")  
      crop_img = crop_image(image, crop_size, starting_point)
      crop_imgs.append(crop_img)
      crop_shifts.append(list(starting_point))
  print(f"### Number of images generated: {len(crop_imgs)}")
  return crop_imgs, crop_shifts

def predict_all_uncropped(img_paths, csv_paths, crop_size, crop_step, config, rcnn):
  """
  predict_all_uncropped:
  """
  mAPs = []
  all_class_ids = []
  all_scores = []
  all_pred_counts = []
  all_gt_counts = []
  all_img_names = []
  all_fp_counts = []
  for img_path, csv_path in zip(img_paths, csv_paths):
    _, gt_labs, x, y, w, h = I.parse_region_data(csv_path)
    #Format gt_labs and boxes so they can be compared against prediction outputs
    gt_boxes=[]
    for i in range(len(x)):
      gt_boxes.append([y[i],x[i], y[i]+h[i], x[i]+w[i]])
    gt_boxes = np.array(gt_boxes).astype('int32')
    gt_class_ids=I.change_label_to_num(gt_labs, config.CLASS_INFO)
    #Now predict
    pred_boxes, pred_class_ids, pred_scores, _ = predict_uncropped_image(img_path, crop_size, crop_step, config, rcnn)
    #And calculate the mAP between gt and pred
    if len(pred_boxes) != 0:
      mAP, _, _, _, fp_count = I.compute_mAP(gt_boxes, gt_class_ids, pred_boxes, pred_class_ids, pred_scores)
    else:
      mAP = 0.
      fp_count = 0
    img_name = img_path.split('/')[-1]
    #Keep the counts to write a csv file
    #classes = ['eccentric','mature','immature']
    class_counts = np.zeros(3) #Start at zero
    present_class_id, counts = np.unique(pred_class_ids, return_counts=True) #And count those present
    if len(counts) != 0:
      class_counts[present_class_id-1]=counts #Since the class ids are 1,2,3 and the counts have indices 0,1,2 we have to subtract 1
    all_pred_counts.append(class_counts)
    class_counts = np.zeros(3) #Start at zero
    present_class_id, counts = np.unique(gt_class_ids, return_counts=True) #And count those present
    class_counts[present_class_id-1]=counts #Since the class ids are 1,2,3 and the counts have indices 0,1,2 we have to subtract 1
    all_gt_counts.append(class_counts)
    all_img_names.append(img_name)
    all_fp_counts.append(fp_count)
    print(f"Class counts on image {img_name}: {class_counts}")
    print(f"#mAP obtained for predictions on image {img_name} : {mAP}")
    mAPs.append(mAP)
    all_class_ids += list(pred_class_ids)
    all_scores += list(pred_scores)
  avg_mAP = np.mean(mAPs)
  print(f"### Average mAP for dataset: {avg_mAP}")
  return avg_mAP, mAPs, np.array(all_class_ids).astype('int32'), np.array(all_scores), all_pred_counts, all_gt_counts, all_img_names, all_fp_counts

def predict_all_uncropped_metrics(img_paths, csv_paths, crop_size, crop_step, config, rcnn, iou_threshold=0.5):
  """
  predict_all_uncropped_metrics:
  """
  #Metrics
  mAPs = []
  precisions = []
  recalls = []
  f1s = []

  #All matches so precision and recall can be calculated for the whole dataset
  all_gt_match = np.array([])
  all_pred_match = np.array([])

  # Class ids, scores, counts and names for writing files
  all_class_ids = []
  all_scores = []
  all_pred_counts = []
  all_gt_counts = []
  all_img_names = []
  for img_path, csv_path in zip(img_paths, csv_paths):
    _, gt_labs, x, y, w, h = I.parse_region_data(csv_path)
    #Format gt_labs and boxes so they can be compared against prediction outputs
    gt_boxes=[]
    for i in range(len(x)):
      gt_boxes.append([y[i],x[i], y[i]+h[i], x[i]+w[i]])
    gt_boxes = np.array(gt_boxes).astype('int32')
    gt_class_ids=I.change_label_to_num(gt_labs, config.CLASS_INFO)
    #Now predict
    pred_boxes, pred_class_ids, pred_scores, _ = predict_uncropped_image(img_path, crop_size, crop_step, config, rcnn)
    #And find the matches between between gt and pred
    if len(pred_boxes) != 0:
      gt_match, pred_match,  _ = I.find_matches(gt_boxes, gt_class_ids, pred_boxes, pred_class_ids, pred_scores, iou_threshold)
      precision_one, recall_one, f1_one, mAP_one, _, _= I.compute_metrics(gt_match, pred_match)
      # mAP, _, _, _, fp_count = I.compute_mAP(gt_boxes, gt_class_ids, pred_boxes, pred_class_ids, pred_scores)
    else:
      gt_match, pred_match = -1*np.ones(1), -1*np.ones(1)
      precision_one, recall_one, f1_one, mAP_one = 0., 0., 0., 0.
    img_name = img_path.split('/')[-1]
    #Keep the counts to write a csv file
    #classes = ['eccentric','mature','immature']
    class_counts = np.zeros(3) #Start at zero
    present_class_id, counts = np.unique(pred_class_ids, return_counts=True) #And count those present
    if len(counts) != 0:
      class_counts[present_class_id-1]=counts #Since the class ids are 1,2,3 and the counts have indices 0,1,2 we have to subtract 1
    all_pred_counts.append(class_counts)
    class_counts = np.zeros(3) #Start at zero
    present_class_id, counts = np.unique(gt_class_ids, return_counts=True) #And count those present
    class_counts[present_class_id-1]=counts #Since the class ids are 1,2,3 and the counts have indices 0,1,2 we have to subtract 1
    all_gt_counts.append(class_counts)
    all_img_names.append(img_name)
    print(f"Class counts on image {img_name}: {class_counts}")
    print(f"#mAP obtained for predictions on image {img_name} : {mAP_one}")
    all_class_ids += list(pred_class_ids)
    all_scores += list(pred_scores)
    
    mAPs.append(mAP_one)
    recalls.append(recall_one)
    precisions.append(precision_one)
    f1s.append(f1_one)
    all_gt_match = np.append(all_gt_match, gt_match)
    all_pred_match = np.append(all_pred_match, pred_match)
  avg_mAP = np.mean(mAPs)
  print(f"### Average mAP for dataset: {avg_mAP}")
  return avg_mAP, mAPs, recalls, precisions, f1s, all_gt_match, all_pred_match, np.array(all_class_ids).astype('int32'), np.array(all_scores), all_pred_counts, all_gt_counts, all_img_names

def write_counts_csv(csvname, img_names, gt_counts, pred_counts, fp_counts):
  with open(csvname, mode='w', newline='') as labels:
    fields = ['#filename',
              'eccentric_gt',
              'mature_gt',
              'immature_gt',
              'eccentric_pred',
              'mature_pred',
              'immature_pred',
              'false_positives']

    writer = csv.DictWriter(labels,
                            fieldnames=fields,
                            dialect='excel',
    quoting=csv.QUOTE_MINIMAL)
  
    writer.writeheader()
    for i in range(len(gt_counts)):
      writer.writerow({'#filename': img_names[i],
                       'eccentric_gt': gt_counts[i][0],
                       'mature_gt': gt_counts[i][1],
                       'immature_gt': gt_counts[i][2],
                       'eccentric_pred': pred_counts[i][0],
                       'mature_pred': pred_counts[i][1],
                       'immature_pred': pred_counts[i][2],
                       'false_positives': fp_counts[i]
      })
      
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("-d", "--data", help="\'single\', \'multiple\' or \'test dataset\' image prediction", default='multiple')
  parser.add_argument("-p", "--path", help="Path to the image to predict or directory containing images for multiple image prediction", default='')
  parser.add_argument("-b", "--backbone", help="Backbone to use for prediction, options are \'temnet\', \'resnet101\' or \'resnet101v2\', mind weights are different for each model", default='temnet')
  parser.add_argument("-m", "--magnification", help="Magnification of the input image for prediction", default=30000, type=int)
  args = parser.parse_args()
  if(args.path == ''):
      print('\n Please provide a valid path for image prediction.')
      print('Options:\n')
      print("-d", "\t --data", "\t \'single\' or \'multiple\' image prediction")
      print("-p", "\t --path", "\t Path to the image to predict or directory containing images for multiple image prediction")
      print("-b", "\t --backbone", "\t Backbone to use for prediction, options are \'temnet\', \'resnet101\' or \'resnet101v2\', mind weights are different for each model")
      print("-m", "\t --magnification", "\t Magnification of the input image for prediction")
  config = Config(backbone=args.backbone)
  print(f"Prediction mode: {args.data}")
  print(f"Predicting from: {args.path}")
  print(f"Magnification of input TEM micrographs: {args.magnification}")
  print(f"Model for prediction: {config.BACKBONE}")
  print(f"Reading weights from: {config.WEIGHT_SET}")
  magnification = args.magnification
  base_magnification = config.BASE_MAGNIFICATION
  base_crop_size = config.BASE_CROP_SIZE
  base_crop_step = config.BASE_CROP_STEP
  new_crop_size = int(magnification * (base_crop_size/base_magnification) )
  new_crop_step = int(magnification * (base_crop_step/base_magnification) )
  crop_size = (new_crop_size, new_crop_size)
  crop_step = (new_crop_step, new_crop_step)
  # Only load weights once
  rcnn = RCNN(config, 'inference')
  rcnn.keras_model.load_weights(config.WEIGHT_SET, by_name=True)

  if(args.data == 'dataset_all'):
    datasets = {"train": Dataset(config.TRAIN_PATH, config, "train"), "validation": Dataset(config.VAL_PATH, config, "validation")}
    avg_mAP_val, mAPs_val = predict_all_rcnn(datasets["validation"], config)
    avg_mAP_train, mAPs_train = predict_all_rcnn(datasets["train"], config)
    print("\n")
    print(77*"#")
    print(f"####### PREDICTIONS LOG #######")
    print(f"VALIDATION avg mAP: {avg_mAP_val}")
    print(f"VALIDATION max mAP: {np.max(mAPs_val)}")
    print(f"VALIDATION min mAP: {np.min(mAPs_val)}")
    print(f"TRAINING avg mAP: {avg_mAP_train}")
    print(f"TRAINING max mAP: {np.max(mAPs_train)}")
    print(f"TRAINING min mAP: {np.min(mAPs_train)}")
    print(f"COMPLETE avg mAP: {np.mean(mAPs_train+mAPs_val)}")
  elif(args.data=='dataset_random'):
    dataset = Dataset(config.VAL_PATH, config, "validation")
    visualize(dataset, config, args.data)
  elif(args.data == 'dataset_noaug'):
    #Build image paths
    # TRAIN_PATH = '/scratch/07655/jsreyl/imgs/rcnn_dataset_full/train'
    # VAL_PATH = '/scratch/07655/jsreyl/imgs/rcnn_dataset_full/val'
    TRAIN_PATH = config.TRAIN_PATH_NOAUG
    VAL_PATH = config.VAL_PATH_NOAUG
    # Read images from train and validation:
    train_ids = next(os.walk(TRAIN_PATH))[1]#All folder names in TRAIN_PATH
    val_ids = next(os.walk(VAL_PATH))[1]#All folder names in TRAIN_PATH
    image_paths_train = [os.path.join(TRAIN_PATH, img_name, img_name+'.png') for img_name in train_ids]
    image_paths_val = [os.path.join(VAL_PATH, img_name, img_name+'.png') for img_name in val_ids]
    csv_paths_train = [os.path.join(TRAIN_PATH, img_name, 'region_data_'+img_name+'.csv') for img_name in train_ids]
    csv_paths_val = [os.path.join(VAL_PATH, img_name, 'region_data_'+img_name+'.csv') for img_name in val_ids]
    #Sequentially predict on the uncropped images and store their class ids
    #Predict for both Train and Validation sets
    avg_mAP_train, mAPs_train, class_ids_train, scores_train, pred_counts_train, gt_counts_train, img_names_train, fp_counts_train = predict_all_uncropped(image_paths_train, csv_paths_train, crop_size, crop_step, config)
    write_counts_csv(config.LOGS+f'training_counts_{config.BACKBONE}_window_{crop_size[0]}.csv',img_names_train, gt_counts_train, pred_counts_train, fp_counts_train)
    avg_mAP_val, mAPs_val, class_ids_val, scores_val, pred_counts_val, gt_counts_val, img_names_val, fp_counts_val = predict_all_uncropped(image_paths_val, csv_paths_val, crop_size, crop_step, config)
    #V.visualize_predictions_count(class_ids_val, scores_val, f'val_dataset_window_{crop_size[0]}')
    #V.visualize_score_histograms(np.array(class_ids_val), np.array(scores_val), f'val_dataset_window_{crop_size[0]}')
    write_counts_csv(config.LOGS+f'validation_counts_{config.BACKBONE}_window_{crop_size[0]}.csv',img_names_val, gt_counts_val, pred_counts_val, fp_counts_val)
    write_counts_csv(config.LOGS+f'full_counts_{config.BACKBONE}_window_{crop_size[0]}.csv',img_names_train+img_names_val, gt_counts_train+gt_counts_val, pred_counts_train+pred_counts_val, fp_counts_train+fp_counts_val)
    #V.visualize_score_histograms(np.array(list(class_ids_train)+list(class_ids_val)), np.array(list(scores_train)+list(scores_val)), f'full_dataset_window_{crop_size[0]}')

    print("\n")
    print(77*"#")
    print(f"####### PREDICTIONS LOG FOR UNCROPPED DATASET #######")
    print(f"VALIDATION avg mAP: {avg_mAP_val}")
    print(f"VALIDATION max mAP: {np.max(mAPs_val)}")
    print(f"VALIDATION min mAP: {np.min(mAPs_val)}")
    print(f"TRAINING avg mAP: {avg_mAP_train}")
    print(f"TRAINING max mAP: {np.max(mAPs_train)}")
    print(f"TRAINING min mAP: {np.min(mAPs_train)}")
    print(f"COMPLETE avg mAP: {np.mean(mAPs_train+mAPs_val)}")
  elif(args.data == 'dataset_noaug_metrics'):
    #Build image paths
    # TRAIN_PATH = '/scratch/07655/jsreyl/imgs/rcnn_dataset_full/train'
    # VAL_PATH = '/scratch/07655/jsreyl/imgs/rcnn_dataset_full/val'
    TRAIN_PATH = config.TRAIN_PATH_NOAUG
    VAL_PATH = config.VAL_PATH_NOAUG
    # Read images from train and validation:
    train_ids = next(os.walk(TRAIN_PATH))[1]#[:1]#All folder names in TRAIN_PATH
    val_ids = next(os.walk(VAL_PATH))[1][:1]#All folder names in TRAIN_PATH
    image_paths_train = [os.path.join(TRAIN_PATH, img_name, img_name+'.png') for img_name in train_ids]
    image_paths_val = [os.path.join(VAL_PATH, img_name, img_name+'.png') for img_name in val_ids]
    csv_paths_train = [os.path.join(TRAIN_PATH, img_name, 'region_data_'+img_name+'.csv') for img_name in train_ids]
    csv_paths_val = [os.path.join(VAL_PATH, img_name, 'region_data_'+img_name+'.csv') for img_name in val_ids]
    #Sequentially predict on the uncropped images and store their class ids
    #Predict for both Train and Validation sets
    avg_mAP_train, mAPs_train, recalls_train, precisions_train, f1s_train, gt_match_train, pred_match_train, class_ids_train, scores_train, pred_counts_train, gt_counts_train, img_names_train = predict_all_uncropped_metrics(image_paths_train, csv_paths_train, crop_size, crop_step, config, rcnn)
    precision_train, recall_train, f1_train, mAP_train, p_train, r_train = I.compute_metrics(gt_match_train, pred_match_train)
    write_counts_csv(config.LOGS+f'training_counts_{config.BACKBONE}_window_{crop_size[0]}.csv',img_names_train, gt_counts_train, pred_counts_train, -1*np.ones_like(pred_counts_train))
    avg_mAP_val, mAPs_val, recalls_val, precisions_val, f1s_val, gt_match_val, pred_match_val, class_ids_val, scores_val, pred_counts_val, gt_counts_val, img_names_val = predict_all_uncropped_metrics(image_paths_val, csv_paths_val, crop_size, crop_step, config, rcnn)
    precision_val, recall_val, f1_val, mAP_val, p_val, r_val = I.compute_metrics(gt_match_val, pred_match_val)
    write_counts_csv(config.LOGS+f'validation_counts_{config.BACKBONE}_window_{crop_size[0]}.csv',img_names_val, gt_counts_val, pred_counts_val, -1*np.ones_like(pred_counts_val))
    write_counts_csv(config.LOGS+f'full_counts_{config.BACKBONE}_window_{crop_size[0]}.csv',img_names_train+img_names_val, gt_counts_train+gt_counts_val, pred_counts_train+pred_counts_val, -1*np.ones_like(pred_counts_train+pred_counts_val))
    #V.visualize_score_histograms(np.array(list(class_ids_train)+list(class_ids_val)), np.array(list(scores_train)+list(scores_val)), f'full_dataset_window_{crop_size[0]}')

    print("\n")
    print(77*"#")
    print(f"####### PREDICTIONS LOG FOR UNCROPPED DATASET #######")
    print("VALIDATION:\n")
    print(f"mAP: (avg) {avg_mAP_val},    (min) {np.min(mAPs_val)},    (max) {np.max(mAPs_val)},    (full) {mAP_val}    ")
    print(f"precision: (avg) {np.mean(precisions_val)},    (min) {np.min(precisions_val)},    (max) {np.max(precisions_val)},    (full) {precision_val}    ")
    print(f"recall: (avg) {np.mean(recalls_val)},    (min) {np.min(recalls_val)},    (max) {np.max(recalls_val)},    (full) {recall_val}    ")
    print(f"f1: (avg) {np.mean(f1s_val)},    (min) {np.min(f1s_val)},    (max) {np.max(f1s_val)},    (full) {f1_val}    ")
    print("TRAINING:\n")
    print(f"mAP: (avg) {avg_mAP_train},    (min) {np.min(mAPs_train)},    (max) {np.max(mAPs_train)},    (full) {mAP_train}    ")
    print(f"precision: (avg) {np.mean(precisions_train)},    (min) {np.min(precisions_train)},    (max) {np.max(precisions_train)},    (full) {precision_train}    ")
    print(f"recall: (avg) {np.mean(recalls_train)},    (min) {np.min(recalls_train)},    (max) {np.max(recalls_train)},    (full) {recall_train}    ")
    print(f"f1: (avg) {np.mean(f1s_train)},    (min) {np.min(f1s_train)},    (max) {np.max(f1s_train)},    (full) {f1_train}    ")
    V.visualize_pr_curve(p_train,r_train,mAP_train, p_val, r_val, mAP_val, f"{config.BACKBONE}_pr_curve")
    print(f"PR curve saved to: {config.BACKBONE+'_pr_curve.png'}")
  elif(args.data == 'multiple'):
    #Build image paths
    # IMAGES_PATH = '/scratch/07655/jsreyl/imgs/nov_dataset'
    IMAGES_PATH = args.path
    print(IMAGES_PATH)
    # Read images from train and validation:
    images_ids = next(os.walk(IMAGES_PATH))[2]#All file names in IMAGES_PATH
    image_paths_train = [os.path.join(IMAGES_PATH, img_name) for img_name in images_ids if img_name.endswith(IMG_FORMAT)]
    if(len(image_paths_train)==0):#No images found? Search directory-wise, i.e. /path/07655/07655.png
      print('No images found, searching directory wise')
      images_ids = next(os.walk(IMAGES_PATH))[1]#All folder names in IMAGES_PATH
      print(f"Folders found: {images_ids}")
      # for img_fmt in IMG_FORMAT:
      # image_paths_train += [os.path.join(IMAGES_PATH, img_name, img_name+img_fmt) for img_name in images_ids if img_name.endswith(img_fmt)]
      image_paths_train = [os.path.join(IMAGES_PATH, img_name, img_name+'.png') for img_name in images_ids]
    print(f"Image paths: {image_paths_train}")

    #Predict for both Train and Validation sets
    for image_path in image_paths_train:
        imgName = image_path.split('/')[-1].split('.')[0]
        print(f'Predicting for image {imgName}')
        #img_names.append(imgName)
        _, pred_class_ids, pred_scores, _ = predict_uncropped_image(image_path, crop_size, crop_step, config, rcnn)
  elif(args.data == 'single'):
    imgName = args.path.split('/')[-1].split('.')[0]
    print(f"Predicting with crop size: {crop_size} and crop step {crop_step}")
    _, pred_class_ids, pred_scores, _ = predict_uncropped_image(args.path, crop_size, crop_step, config, rcnn)
    #V.visualize_score_histograms(np.array(pred_class_ids), np.array(pred_scores), f'{imgName}_window_{crop_size[0]}')
  elif(args.data == 'save_model'):
    rcnn = RCNN(config, 'inference')
    rcnn.keras_model.load_weights(config.WEIGHT_SET, by_name=True)
    print(f'Saving model for backbone {args.backbone}')
    rcnn.keras_model.save(str(args.backbone))
  elif(args.data == 'test_saved_model'):
    import inspect

    rcnn_loaded = tf.keras.models.load_model(os.path.join(os.getcwd(),args.backbone))
    rcnn_loaded.summary()
    inspect.getmembers(rcnn_loaded, predicate = inspect.isfunction)

    imgName = args.path.split('/')[-1].split('.')[0]
    print(f"Predicting with crop size: {crop_size} and crop step {crop_step}")
    _, pred_class_ids, pred_scores, _ = predict_uncropped_image(args.path, crop_size, crop_step, config, rcnn_loaded)
