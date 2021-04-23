from __future__ import absolute_import, division, print_function, unicode_literals
""" HIV Cell Image Classifier
    Alex Bryer & Hagan Beatson
    Perilla Labs, University of Delaware """
import os, logging, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #shut up, tf
import tensorflow as tf
from tensorflow import keras
import cv2
import csv

# Local Scripts
import database as D
import models as M
import input_pipeline as I
import training as T 
import graphing as G
from model_resnet101 import ResNet101

from rpn.predict import predict 
from classes import Config, PerillaNet

# END TO END
"""
temnet_predict_new_data: overall function to predict new data using TEMNet
Inputs:
  imgName, the name of the image file to predict upon
  config, user config file
  weights, the name of the weight collection to load
Outputs:
  None
"""
def temnet_predict_new_data(imgName, config, weights):
  # Build Model & Load Weights for Prediction
  model, modelName = M.classification_model_TEMNet(config.RESOLUTION, config.RESOLUTION)
  print("Weight file path:", config.CHECKPOINT_PATH + str(weights))
  model.load_weights(tf.train.latest_checkpoint(config.CHECKPOINT_PATH + str(weights)))

  # Open Dataset
  path = config.IMAGE_PATH + imgName 
  img = keras.preprocessing.image.img_to_array(keras.preprocessing.image.load_img(path))
  img = img.astype("float32") 

  # Build prediction arrays on new data
  regions = preprocess_detection_output(config, img, imgName, path) # TODO: Verify that datatypes are consistent
  
  # Make Predictions
  predictions = model.predict(regions, batch_size = config.BATCH_SIZE)
  newPreds = complete_predictions(predictions) # Round to most confident prediction

  # Export Info
  log_predictions(imgName, newPreds, regions)
  G.plot_prediction(img, imgName, newPreds, regions) # TODO: regions need to be resized before being plotted on new image
 
def resnet_predict_new_data(imgName, config, weights):
  # Build Model & Load Weights for Prediction
  input_tens = tf.keras.layers.Input(shape=(config.RESOLUTION, config.RESOLUTION, 3))
  RN101 = ResNet101()
  model = RN101.build_backbone(input_tens, "resnet101", 3, True)
  model = M.finetuneNetwork(model, "resnet101") #Unset the final softmax layer and put a new one with 3 categorical classes
  print("Weight file path:", str(weights))
  RN101.load_weights((str(weights)), model, by_name=True)

  # Open Dataset
  path = config.IMAGE_PATH + imgName 
  img = keras.preprocessing.image.img_to_array(keras.preprocessing.image.load_img(path))
  img = img.astype("float32") 

  # Build prediction arrays on new data
  regions = preprocess_detection_output(config, img, imgName, path) # TODO: Verify that datatypes are consistent
  
  # Make Predictions
  predictions = model.predict(regions, batch_size = config.BATCH_SIZE)
  newPreds = complete_predictions(predictions) # Round to most confident prediction

  # Export Info
  log_predictions(imgName, newPreds, regions)
  G.plot_prediction(img, imgName, newPreds, regions) # TODO: regions need to be resized before being plotted on new image

"""
log_predictions: Export a log file containing prediction information for TEMNet prediction
Inputs:
  img, the name of the image that was originally predicted on
  predictions, an array of predictions
  regions, an array of initial regions proposed from detection network
Outputs:
  log file containing region information (region #, region coords, region classification) & general demographics
TODO: Add support for prediction confidence (pass in original predictions)
"""
def log_predictions(imgName, predictions, regions):
  print("Prediction information for image", imgName)
  classification = ''
  log = 'log_' + imgName + '.csv'
  with open(log, 'w') as f:
    writer = csv.writer(f)
    titles = ['Box Number', 'Classification', 'Region Coordinates']
    writer.writerow(titles)
    if(len(predictions) == len(regions)):
      for i in range(len(predictions)):
        if(predictions[i] == 0):
          classification = 'Eccentric'
        elif(predictions[i] == 1):
          classification = 'Immature'
        elif(predictions[i] == 2):
          classification = 'Mature'
        print("Box #{}: Classification {} at coordinates {}".format(i, classification, regions[i]))
        data = [i, classification, regions[i]]
        writer.writerow(data)
    else:
      print("ERROR WITH PREDICTION DIMENSIONALITY")
  
"""
preprocess_detection_output: Pre-process region of interest data from RPN
Inputs:
  config, user config class
  imgName, the image to have its regions preprocessed
Outputs:
  trueRegions, an array of 400x400 samples resized by pull_image_regions
"""
def preprocess_detection_output(config, image_gt, imgName, path):
  # TODO: ACCOMODATE FOR LACK OF ANCHORS FOR NEW DATA
  _ = None # THIS SHOULD BE ANCHORS
  regions = predict(config, imgName, image_gt, _)
  trueRegions = pull_image_regions(path, regions, config)
  return trueRegions

"""
pull_image_regions: Based on prediction coordinates, get image samples and resize to prediction resolution
Inputs:
  imgName, the name of the image to be parsed
  regions, an array of coordinates to pull
  config, user config class
Outputs:
  newRegions, an array of resized regions of interest
"""
def pull_image_regions(imgName, regions, config):
  # TODO: NEED TO FIGURE OUT COORDINATE ORDER OF REGION COORDINATE LISTS
  img = cv2.imread(imgName)
  newRegions = []
  for i in regions:
    newRegions.append(img[i[0]:i[2], i[1]:i[3]]) # TODO: THESE VALUES ARE WRONG, SHOULD BE [y1:y2, x1:x2]
  for j in range(len(newRegions)):
    newRegions[i] = cv2.resize(newRegions[i], (config.RESOLUTION, config.RESOLUTION)) # PREDICT RESOLUTIONS
  return newRegions


# ************** PREDICTION FUNCTIONS **************
"""
temnet_prediction: Use our novel network to make predictions on data
Inputs: 
  weights, the name of the directory of checkpoints to be loaded for prediction
  config, user config class
Outputs:
  None
"""
def temnet_prediction(weights, config):
  os.chdir(config.IMAGE_PATH)
  shape = config.RESOLUTION, config.RESOLUTION
  imgs = I.build_image_input_array(config.IMAGE_PATH, shape)   
  labels = I.build_label_input_array(config.IMAGE_PATH)
  model, modelName = M.classification_model_TEMNet(config.RESOLUTION, config.RESOLUTION) 
  print("Weight file path:", config.CHECKPOINT_PATH + str(weights))
  model.load_weights(tf.train.latest_checkpoint(config.CHECKPOINT_PATH + str(weights)))

  predictions = model.predict(imgs, batch_size = config.BATCH_SIZE)
  newPreds = complete_predictions(predictions)
  e, i, m = I.get_demographics(labels)
  print("\nTEMNet model predictions:")
  compare_results(newPreds, labels, e, i ,m)
  G.plot_prediction(newPreds, modelName)
  
"""
resnet_prediction: Use the resnet101 network to make predictions on data
Inputs: 
  weights, the name of the directory of checkpoints to be loaded for prediction
  config, user config class
Outputs:
  None
"""
def resnet_prediction(weights, config):
  os.chdir(config.IMAGE_PATH)
  shape = config.RESOLUTION, config.RESOLUTION
  imgs = I.build_image_input_array(config.IMAGE_PATH, shape)   
  labels = I.build_label_input_array(config.IMAGE_PATH)
  input_tens = tf.keras.layers.Input(shape=(config.RESOLUTION, config.RESOLUTION, 3))
  RN101 = ResNet101()
  model = RN101.build_backbone(input_tens, "resnet101", 3, True)
  model = M.finetuneNetwork(model, "resnet101") #Unset the final softmax layer and put a new one with 3 categorical classes
  print("Weight file path:", str(weights))
  #model.load_weights(tf.train.latest_checkpoint(str(weights)))
  RN101.load_weights((str(weights)), model, by_name=True)

  preds = model.predict(imgs, batch_size = config.BATCH_SIZE)
  newPreds = complete_predictions(preds)
  e, i, m = I.get_demographics(labels)
  print("\nResNet101 model predictions:")
  compare_results(newPreds, labels, e, i ,m)
  G.plot_prediction(newPreds, "resnet101")

# ************** HELPER FUNCTIONS **************

"""
Compare prediction array from complete_predictions to true label array, return # of correct results
Inputs: predictions and labels, two arrays corresponding to the predictions of a network and the dataset labels
Outputs: e, i, and m, the total count of correct occurences
"""
def compare_results(predictions, labels, eCount, iCount, mCount):
  e, i, m = 0, 0, 0
  for h in range(len(predictions)):
    if(predictions[h] == labels[h] == 0):
      e += 1
    elif(predictions[h] == labels[h] == 1):
      i += 1
    elif(predictions[h] == labels[h] == 2):
      m += 1
  totalCorrect = e + i + m
  print("pred length: {}".format(len(predictions)))
  print("pred e: {}, pred i: {}, pred m: {}".format(e, i, m))
  print("count e: {}, count i: {}, count m: {}".format(eCount, iCount, mCount))
  print("Overall prediction accuracy: {:5.2f}%".format(100*(totalCorrect/len(predictions))))
  print("Eccentric prediction accuracy: {:5.2f}%".format(100*(e/eCount)))
  print("Immature prediction accuracy: {:5.2f}%".format(100*(i/iCount)))
  print("Mature prediction accuracy: {:5.2f}%".format(100*(m/mCount)))
  return e, i ,m, totalCorrect

"""
complete_predictions: turn output array of thruples into list with singly-classified indices
Inputs: predictions, the output of model.predict call
Outputs: newPreds, a list of classifications corresponding to each sample in a dataset
"""
def complete_predictions(predictions):
  newPreds = []
  for h in range(len(predictions)):
    maxPred = max(predictions[h])
    if(predictions[h][0] == maxPred):
      newPreds.append(0)
    elif(predictions[h][1] == maxPred):
      newPreds.append(1)
    elif(predictions[h][2] == maxPred):
      newPreds.append(2)
  return newPreds
    

# ************** MAIN FUNCTION ************** 
if __name__ == "__main__":
  #parser = argparse.ArgumentParser()
  #parser.add_argument("-m", "--model", help="Name of model directory to be loaded for TEMNet network")
  #parser.add_argument("-i", "--image", help="Name of image to be predicted upon")
  #args = parser.parse_args()

  config = Config()
  #temnet_prediction('90run', config)
  temnet_predict_new_data('7825005.png', config, '90run')
