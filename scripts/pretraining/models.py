from __future__ import absolute_import, division, print_function, unicode_literals
""" HIV Cell Image Classifier
    Alex Bryer & Hagan Beatson
    Perilla Labs, University of Delaware """
import tensorflow as tf
import os
from tensorflow import keras
from keras import models, layers, optimizers
from keras.models import Model, Sequential
#import kerastuner as kt
import graphing as G
import input_pipeline as I
from addons import GroupNormalization

# ********************* TEMNet NETWORK MODELS *********************
""" 
  MODEL BUILDING
    Each convolutional layer activates via ReLU and is followed by a 
    max pooling before a total flatten and dense ReLU layer 
    followed by a softmax output layer. Output shape is 1x3, where 
    each index in range [0,2] correponds to a categorical label
"""

def classification_model_TEMNet(height, width):
	# Suggestions made by Wayne to reduce dense layers, add dropout layer, and add L2 regularizers
	# Current model being used in training
  # TODO: Add L2 regularization
  name = "class_test3"
  model = tf.keras.Sequential([
    #Add batch normalization layers so we can train it from scratch for the rcnn backbone
    #According to Object Detection from Scratch with Deep Supervision https://arxiv.org/pdf/1809.09294.pdf a Conv-BN-ReLU layer gives better accuracy than BN-ReLU-Conv, so we separate the activation and convolution layers
		# tf.keras.layers.Conv2D(8, (13,13), activation=tf.nn.relu, padding="same", input_shape=(height, width, 3), name="conv1"),
    tf.keras.layers.Conv2D(8, (13,13), padding="same", input_shape=(height, width, 3), name="conv1"),
    # tf.keras.layers.BatchNormalization(name="bn_conv1"),
    GroupNormalization(groups=4, axis=3, name="gn_conv1"),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2, name ="pool1"),
    tf.keras.layers.GaussianNoise(0.1, name="noise"),
    # tf.keras.layers.Conv2D(16, (9, 9), activation=tf.nn.relu, padding="same", name="conv2"),
    tf.keras.layers.Conv2D(16, (9, 9), padding="same", name="conv2"),
    # tf.keras.layers.BatchNormalization(name="bn_conv2"),
    GroupNormalization(groups=4, axis=3, name="gn_conv2"),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2, name = "pool2"),
    tf.keras.layers.Conv2D(32, (7, 7), padding="same", name="conv3"),
    # tf.keras.layers.BatchNormalization(name="bn_conv3"),
    GroupNormalization(groups=4, axis=3, name="gn_conv3"),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2, name = "pool3"),
    tf.keras.layers.Conv2D(64, (5, 5), padding="same", name="conv4"),
    # tf.keras.layers.BatchNormalization(name="bn_conv4"),
    GroupNormalization(groups=4, axis=3, name="gn_conv4"),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2, name = "pool4"),
    tf.keras.layers.Flatten(name="flat"),
    tf.keras.layers.Dropout(0.3, name="dropout"),
    tf.keras.layers.Dense(64, activation=tf.nn.relu, name="dense1"),
    tf.keras.layers.Dense(32, activation=tf.nn.relu, name="dense2"),
    tf.keras.layers.Dense(3, activation=tf.nn.softmax, name="visualized_layer")
	])
  model.summary()
  return model, name

"""
hypertune_temnet_network: use Keras Tuner library to optimize hyperparameters for our model
"""
"""def hypertune_temnet_network(hp):
  hp_convSize = hp.Int('convSize', min_value = 8, max_value = 128, step = 8)
  hp_denseSize = hp.Int('denseSize', min_value = 64, max_value = 512, step = 64)
  hp_denseSize2 = hp.Int('denseSize2', min_value = 32, max_value = 256, step = 32)
  hp_learningRate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
  hp_dropout = hp.Float('dropout', min_value = 0.1, max_value = 0.4, step = 0.05)
  hp_kernSize = hp.Int('kernSize', min_value = 8, max_value = 13, step = 1)
  hp_stepSize = hp.Int('stepSize', min_value = 2, max_value = 4, step = 1)
  hp_noise = hp.Float('noise', min_value = 0.1, max_value = 0.5, step = 0.05)
  model = tf.keras.Sequential([
		tf.keras.layers.Conv2D(hp_convSize, (hp_kernSize,hp_kernSize), activation=tf.nn.relu, input_shape=(400, 400, 3)),
    tf.keras.layers.MaxPooling2D((hp_stepSize, hp_stepSize), strides=2),
  	tf.keras.layers.GaussianNoise(hp_noise),
    tf.keras.layers.Conv2D(16, (9, 9), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((hp_stepSize, hp_stepSize), strides=2),
    tf.keras.layers.Conv2D(32, (7, 7), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((hp_stepSize, hp_stepSize), strides=2),
    tf.keras.layers.Conv2D(64, (5, 5), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((hp_stepSize, hp_stepSize), strides=2),
    tf.keras.layers.Flatten(),	
   	tf.keras.layers.Dropout(hp_dropout),
   	tf.keras.layers.Dense(hp_denseSize, activation=tf.nn.relu),
   	tf.keras.layers.Dense(hp_denseSize2, activation=tf.nn.relu),
    tf.keras.layers.Dense(3, activation=tf.nn.softmax)
	])
  model.compile(optimizer=tf.keras.optimizers.Adam(lr=hp_learningRate),
                loss='categorical_crossentropy', 
                metrics=['accuracy'])    
  return model"""

"""
optimize: utilize KerasTuner on our TEMNet network and find the optimal paramaters

"""
"""def optimize_temnet_hyperparams(t_imgs, t_labs, v_imgs, v_labs):
  print("Optimizing TEMNet network hyperparameters...")
  tuner = kt.RandomSearch(hypertune_temnet_network,
                          objective = 'val_accuracy',
                          max_trials = 100)
  tuner.search_space_summary()
  tuner.search(t_imgs, t_labs, epochs = 10, validation_data=(v_imgs, v_labs))
  tuner.results_summary()
  best_hps = tuner.get_best_hyperparameters(num_trials = 10)[0]
  print(f"Best hyperparameters from tuning:\n {best_hps}")
  model = tuner.hypermodel.build(best_hps)
  model.summary()
  hist = model.fit(t_imgs, t_labs, epochs = 50, validation_data=(v_imgs, v_labs))
  G.graph_results(hist, 0, 50, "Adam", "HyperTune_model3", 400)
  return hist"""



# ********************* PRE-TRAINED NETWORK MODELS *********************
"""
create_application_model: Use keras.applications module to create an instance of a pretrained model
Inputs: modelName, the name of the desired model to be created, and inputTens, the shape of the input tensor
Outputs: The actual Keras model corresponding to the input

Currently supported model configurations:
  ResNet50
  ResNet101
  ResNet152
  ResNet50V2
  ResNet101V2
  ResNet152V2
  VGG16
  VGG19
  InceptionV3
"""
def create_application_model(modelName, input_tens):
  if(modelName == 'ResNet50' or modelName == 'resnet50'):
    return keras.applications.resnet.ResNet50(include_top=False, input_tensor=input_tens, weights=None)
  elif(modelName == 'ResNet101' or modelName == 'resnet101'):
    return keras.applications.resnet.ResNet101(include_top=False, input_tensor=input_tens, weights=None)
  elif(modelName == 'ResNet152' or modelName == 'resnet152'):
    return keras.applications.resnet.ResNet152(include_top=False, input_tensor=input_tens, weights=None)
  elif(modelName == 'ResNet50V2' or modelName == 'resnet50v2'):
    return keras.applications.resnet_v2.ResNet50V2(include_top=False, input_tensor=input_tens, weights=None)
  elif(modelName == 'ResNet101V2' or modelName == 'resnet101v2'):
    return keras.applications.resnet_v2.ResNet101V2(include_top=False, input_tensor=input_tens, weights=None)
  elif(modelName == 'ResNet152V2' or modelName == 'resnet152v2'):
    return keras.applications.resnet_v2.ResNet152V2(include_top=False, input_tensor=input_tens, weights=None)
  elif(modelName == 'VGG16' or modelName == 'vgg16'):
    return keras.applications.vgg16.VGG16(include_top=False, input_tensor=input_tens, weights=None)
  elif(modelName == 'VGG19' or modelName == 'vgg19'):
    return keras.applications.vgg19.VGG19(include_top=False, input_tensor=input_tens, weights=None)
  elif(modelName == 'InceptionV3' or modelName == 'inceptionv3'):
    return keras.applications.inception_v3.InceptionV3(include_top=False, input_tensor=input_tens, weights='imagenet')
  else:
    print("Model configuration not recognized...")


"""
finetuneNetwork: function to fine-tune a pre trained network to be suited for our pipeline
Inputs: model, the model you wish to fine-tune
Outputs: updatedModel, a trainable model
Currently supported model configurations:
  ResNet50
  ResNet101
  ResNet152
  ResNet50V2
  ResNet101V2
  ResNet152V2
  VGG16
  VGG19
  InceptionV3
"""
def finetuneNetwork(model, modelName):
  if('ResNet' in modelName or 'resnet' in modelName or 'Resnet' in modelName):
    for layer in model.layers[:-1]:
      layer.trainable=False
    out = model.output
    x = tf.keras.layers.Flatten() (out)
    x = tf.keras.layers.Dense(3, activation=tf.nn.softmax) (x)
    return keras.models.Model(model.input, x)
  # TODO: Remove final FC layer from VGG style networks, replace with dense
  elif('VGG' in modelName or 'vgg' in modelName):
    for layer in model.layers[:-1]:
      layer.trainable=False
    out = model.output
    x = tf.keras.layers.Dense(3, activation=tf.nn.softmax) (out)
    return keras.models.Model(model.input, x)
  elif('Inception' in modelName or 'inception' in modelName):
    for layer in model.layers[:-1]:
      layer.trainable=False
    out = model.output
    x = tf.keras.layers.Flatten() (out)
    x = tf.keras.layers.Dense(3, activation=tf.nn.softmax) (x)
    return keras.models.Model(model.input, x)
  else:
    print("Model type not recognized by finetuneNetwork, please check supported networks")
