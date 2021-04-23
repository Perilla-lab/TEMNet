import os
import numpy as np
import tensorflow as tf
import cv2
import models as M
import input_pipeline as I

class Config(object):
    NAME = None

    # Hyper Params for TEMNet Network
    LEARNING_RATE = 0.00015                 # Learning rate for network optimizer
    BATCH_SIZE = 50                         # Training batch size
    EPOCHS = 100                             # Number of training epochs
    RESOLUTION = 400                        # Resolution size of training images
    PERCENT_TRAIN = 0.90                    # Percentage of training data to reserve for training

    # Paths
    # TEMNet network paths
    CHECKPOINT_PATH = '/scratch/07655/jsreyl/hivclass/checkpoints/'   # Path for training checkpoints
    MODEL_PATH = '/scratch/07655/jsreyl/hivclass/models/myModel'      # Path to save model
    IMAGE_PATH = '/scratch/07655/jsreyl/imgs/class_imgs'              # Path for input images

    # Pre-trained network paths
    TRAINING_PATH = '/scratch/07655/jsreyl/imgs/train/'                         # Path for training data
    VALIDATION_PATH = '/scratch/07655/jsreyl/imgs/val/'               # Path for validation data
    PRETRAINED_MODEL_PATH = '/scratch/07655/jsreyl/hivclass/models/pretrained/' # Path to save model

    # Etc
    GRAPH_PATH = '/scratch/07655/jsreyl/hivclass/graphs'

class Image:
    def __init__(self, path_to_img):
        self.name = path_to_img
        self.data = cv2.imread(self.name)
        self.shape = (self.data).shape

class TEMNet:
    # TODO: ADD SUPPORT FOR WEIGHTING
    def __init__(self, config, weights):
        self.resolution = config.RESOLUTION
        self.model, self.name = M.classification_model_TEMNet(self.resolution, self.resolution)
        #if (weights != None):
        #    self.model.load_weights(tf.train.latest_checkpoint(config.CHECKPOINT_PATH + weights)
    
    def compile(self, config):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=config.LEARNING_RATE),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def save_weights(self, path):
        self.model.save_weights(path)
#class Dataset:
    #def __init__(self, path_to_imgs):
        
