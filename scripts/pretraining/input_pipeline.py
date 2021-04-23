from __future__ import absolute_import, division, print_function, unicode_literals
import os, random, cv2
import tensorflow as tf
from tensorflow import image, keras
import numpy as np
from keras import utils, models, layers, optimizers
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from classes import Image
#from linetimer import CodeTimer #Used for benchmarking purposes

"""
build_training_arrays: Handles all pre-training array creation and demographics for phase 2 network
Inputs: path, width, height, the metrics needed as inputs for all the helper functions this utilizes. percentTrain is a float used in split_input_arrays
Outputs: ft_imgs, t_cat_labs, fv_imgs, v_cat_labs, the four arrays required for training
"""
def build_training_arrays(path, width, height, percentTrain):
  shape = width, height
  init_img_arr = build_image_input_array(path, shape)   # Create initial image array from raw TIFF images
  init_label_arr = build_label_input_array(path)        # Create initial label array from raw TIFF images
  train_imgs, train_labels, val_imgs, val_labels = split_input_arrays(init_img_arr, init_label_arr, percentTrain) # Seperate arrays into training arrays and validation arrays                                  # Use TensorFlow tensor augmentation functions to generate more data to reduce overfitting
  train_imgs, train_labels = transform_images(train_imgs, train_labels)
  return train_imgs, to_categorical(train_labels), val_imgs, to_categorical(val_labels)

""" 
build_image_input_array: Build input array with images in path, normalize and resize all to 'shape' (max dims for a given set of imgs) 
Inputs: path, the directory where the input images lie, and shape, the desired dimensions of all the images to be resized
Outputs: test_img_arr, a numpy array of all input images
Currently only supports TIFF images.
"""
def build_image_input_array(path, shape):
  print('Preparing input tensors with images at path:',path)
  print('Resizing to shape', shape, '...')
  imgs = []
  with tf.device('/device:GPU:0'):
    for im in os.listdir(path):
      if(im.endswith('.tif')):
        img = Image(im)
        imgs.append(cv2.resize(img.data, shape))
    return np.array(imgs)

""" 
build_label_input_array: Build categorical label array for images in 'path' for phase2 classification network 
Inputs: path, the directory containing the training images
Outputs: labels, an array with values corresponding to the categorical classification of training images in path
Assumes that data is categorized through file names. 
"""
def build_label_input_array(path):
  print('Creating categorical label array...')
  filenames = []
  labels = [] 
  with tf.device('/device:GPU:0'):
    for fp in os.listdir(path):
      if(fp.endswith('.tif')):
        filenames.append(fp)
    for fn in filenames:
      if fn.startswith('e'):
        labels.append(0)
      elif fn.startswith('i'):
        labels.append(1)
      elif fn.startswith('m'):
        labels.append(2)
    return np.array(labels)

"""
get_demographics: Return the total number of categorical classification in an input array, to be used during training analysis
Inputs: labels, an array of training labels
Outputs: e, i, m; the respective counts for # of eccentric particles, # of immature particles, and # of mature particles
"""
def get_demographics(labels):
  e = 0
  i = 0
  m = 0
  for h in range(len(labels)):
    if(labels[h] == 0): 
      e += 1
    elif(labels[h] == 1):
      i += 1
    elif(labels[h] == 2):
      m += 1
  print("\nDataset Demographics:")
  print("# of Total Particles in Dataset:", e+i+m)
  print("# of Eccentric Particles in Dataset:", str(e))
  print("# of Immature Particles in Dataset:", str(i))
  print("# of Mature Particles in Dataset:", str(m))
  return e,i,m

"""
transform_images: Function that randomly transforms images in order to artificially increase amount of training data
Inputs: X_imgs, the original, unrotated images; labs, their corresponding labels, height and width, the desired dimensions of the rotated images
Outputs: X_flip, an array of the newlyrotated images, and new_labels, an array of their corresponding labels
"""
def transform_images(imgs, labels):
  augImgs = []
  newLabs = []
  for i in range(len(imgs)):
    with tf.device('/device:CPU:0'):
      tens = tf.convert_to_tensor(imgs[i]) 
      for f in range(5):
        newLabs.append(labels[i])
    with tf.device('/device:GPU:0'):
      augImgs.append(tf.image.flip_left_right(tens))
      augImgs.append(tf.image.flip_up_down(tens))
      augImgs.append(tf.image.random_contrast(tens, lower = 0.1, upper = 0.3))
      augImgs.append(tf.image.random_saturation(tens, lower = 0.1, upper = 0.3))
      augImgs.append(tf.image.random_brightness(tens, max_delta = 0.2))
  augImgs = tf.convert_to_tensor(augImgs)
  return np.squeeze(augImgs), np.array(newLabs)

"""
split_input_arrays: Divide the total arrays into training and validation arrays
Inputs: imgArr, labArr, and percentTrain. The first two are self explanatory, percent train represents the proportion
(0-1) to reserve for training. Usually kept between 0.7 and 0.8.
Outputs: val_labs, val_imgs, train_labs, train_imgs, the seperated datasets with corresponding labels.
Was replaced by a "validation_split" argument in the model_fit call, but this now throws errors with TF's eager execution. Keep this for now.
"""
def split_input_arrays(imgArr, labArr, percentTrain):
  with tf.device('/device:GPU:0'):
    val_labs = []
    val_imgs = []
    train_labs = []
    train_imgs = []
    numTotal = len(imgArr)
    numTrain = int(round(percentTrain*numTotal))
    for i in range(0, numTrain, 1):
      train_imgs.append(imgArr[i])
      train_labs.append(labArr[i])
    for j in range(numTrain,numTotal,1):
      val_imgs.append(imgArr[j])
      val_labs.append(labArr[j])
    return np.array(train_imgs), np.array(train_labs), np.array(val_imgs), np.array(val_labs) 

"""
build_generators: Create the ImageDataGenerators for training and validation to be used with pre-trained networks for fine tuning
Inputs: train_path and val_path are directory paths for specially built training and validation images. I copied the whole dataset and did a 80/20 split
Outputs: train_generator and val_generator, the two datagens used in the model.fit_from_generator call in pre-built training procedures
"""
def build_generators(train_path, val_path, batch_size, res):
  with tf.device('/device:GPU:0'):
    train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest'
    )
    validation_datagen = ImageDataGenerator(rescale=1./255) 
    train_generator = train_datagen.flow_from_directory(
      train_path,
      target_size=(res, res),
      batch_size=batch_size,
      class_mode='categorical'
    )
    val_generator = validation_datagen.flow_from_directory(
      val_path,
      target_size= (res, res),
      batch_size = batch_size,
      class_mode='categorical',
      shuffle=False
    )
    return train_generator, val_generator
