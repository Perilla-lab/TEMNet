from __future__ import absolute_import, division, print_function, unicode_literals
import os, logging, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow import keras
from keras import optimizers
import matplotlib.pyplot as plt


# Local Scripts
import database as D
import models as M
import input_pipeline as I
import graphing as G
from classes import Config, TEMNet

""" 
HIV Cell Image Classifier
Alex Bryer & Hagan Beatson
Perilla Labs, University of Delaware 
"""

# ********* SETUP *********
print("TensorFlow Version ",tf.__version__)
#tf.debugging.set_log_device_placement(True) #Enable for device debugging
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice(reduce_to_device="cpu:0"))
print('Number of devices recognized by Mirrored Strategy: ', mirrored_strategy.num_replicas_in_sync)

# ********* NETWORK TRAINING AND VALIDATION FUNCTIONS *********
# *************************************************************

"""
The training procedure to be used for our TEMNet network
"""
def temnet_network(config, weights, hypertune):
  with mirrored_strategy.scope():
    os.chdir(config.IMAGE_PATH)
    #model, modelName = M.classification_model_test3(config.RESOLUTION, config.RESOLUTION)
    #Declare the pNet model, access the model using pnet.model.keras_funtion()
    pnet = TEMNet(config, weights)
    t_imgs, t_cat_labs, v_imgs, v_cat_labs = I.build_training_arrays(config.IMAGE_PATH, config.RESOLUTION, config.RESOLUTION, config.PERCENT_TRAIN)
    if(hypertune == 1):
      M.optimize_temnet_hyperparams(t_imgs, t_cat_labs, v_imgs, v_cat_labs)
    else:
      pnet.compile(config)
      # Checkpointing saves the weight set at every 5th epoch in ../models/checkpoints
      cp_callback = keras.callbacks.ModelCheckpoint(filepath=config.CHECKPOINT_PATH + 'temnet/temnet-weights-{epoch:02d}.hdf5',
                                                    save_weights_only=True, monitor="val_accuracy", mode="max", save_best_only=False,
                                                    verbose=1)
      # If you are training on a pre-existing weight set
      if(weights != None):
        if(weights == 'latest'):
          print("loading latest weight set...")
          pnet.model.load_weights(tf.train.latest_checkpoint(config.CHECKPOINT_PATH + 'temnet'))
        else:
          print("Weights loaded from file", weights)
          weightPath = config.CHECKPOINT_PATH + str(weights)
          print("Weight file path:", weightPath)
          #pnet.model.load_weights(tf.train.latest_checkpoint(weightPath))
          pnet.model.load_weights(weightPath, by_name=True)
      else:
        print("No weight set specified...")
      pnet.model.save_weights(config.CHECKPOINT_PATH.format(epoch=0))
      hist = pnet.model.fit(t_imgs,
                       t_cat_labs, 
                       batch_size=config.BATCH_SIZE, 
                       epochs=config.EPOCHS,
                       shuffle=True, 
                       callbacks=[cp_callback],
                       validation_data=(v_imgs, v_cat_labs))
      pnet.model.save_weights(config.PRETRAINED_MODEL_PATH + 'temnet_weights_batch_norm.h5')
      pnet.model.save(config.MODEL_PATH) # Using batch normalization and mirrored strategy with tensorflow 2.1 can be problematic, comment this line if that's the case. This bug is fixed in tensorflow 2.2
      G.graph_results(config.GRAPH_PATH, hist, config.LEARNING_RATE, config.BATCH_SIZE, "Adam", pnet.name, config.RESOLUTION)
      loss, acc = pnet.model.evaluate(v_imgs, v_cat_labs, verbose = 2)
      print("Final model accuracy: {:5.2f}%".format(100*acc))
      print("Final model loss: {:5.2f}".format(loss))
      return hist.history['val_accuracy']

"""
The training procedure to be used when using a pre-trained network.
fit_generator is the only way I could get the pre-trained networks to be compatible, as ResNet does not follow a Sequential architecture
train_gen and val_gen are ImageDataGenerators that are dependent on the structure of the directories.
In order for them to read data properly, you need a seperate training and validation directory, with both containing 3 subdirectories for each classification.
"""
def pretrained_network(modelName, config):
  with mirrored_strategy.scope():
    train_gen, val_gen = I.build_generators(config.TRAINING_PATH, config.VALIDATION_PATH, config.BATCH_SIZE, config.RESOLUTION)
    input_tens = tf.keras.layers.Input(shape=(config.RESOLUTION, config.RESOLUTION, 3))
    model = M.create_application_model(modelName, input_tens)
    model = M.finetuneNetwork(model, modelName)
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adadelta(lr=config.LEARNING_RATE),
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    history = model.fit(train_gen,
                        shuffle=True,
                        epochs=config.EPOCHS,
                        validation_data=val_gen,
                        verbose=1)
    model.save(config.PRETRAINED_MODEL_PATH + modelName)
    G.graph_results(config.GRAPH_PATH, history, config.LEARNING_RATE, config.BATCH_SIZE, "Adadelta", modelName, config.RESOLUTION)
    return history.history['val_accuracy']

"""
model_ensemble: To be used to fine-tune/train networks en masse
                After training, models need to be exported to respective directories to be restored for prediction
List of models being trained for voting system:
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
def model_ensemble(config):
  train_gen, val_gen = I.build_generators(config.TRAINING_PATH, config.VALIDATION_PATH, config.BATCH_SIZE, config.RESOLUTION)
  models = []
  history = []
  counter = 0
  input_tens = tf.keras.layers.Input(shape=(config.RESOLUTION, config.RESOLUTION, 3))
  names = ['ResNet50', 'ResNet101', 'ResNet152', 'ResNet50V2', 'ResNet101V2', 'ResNet152V2', 'VGG16', 'VGG19', 'InceptionV3']
  with mirrored_strategy.scope():
    for f in range(len(names)):
      models.append(M.finetuneNetwork(M.create_application_model(names[f], input_tens), names[f]))
    for i in models:
      i.compile(optimizer=tf.keras.optimizers.Adadelta(lr=config.LEARNING_RATE),
                loss='categorical_crossentropy', 
                metrics=['accuracy'])
      hist = i.fit(train_gen,
                   shuffle=True,
                   epochs=config.EPOCHS,
                   validation_data=val_gen,
                   verbose=1)
      history.append(hist.history['val_accuracy'])
      i.save(config.PRETRAINED_MODEL_PATH + str(names[counter]))
      counter += 1
    G.plot_ensemble(config.GRAPH_PATH, history, config.LEARNING_RATE, config.BATCH_SIZE, "Adadelta", config.RESOLUTION)
    

# ********* MAIN FUNCTION ********* 
# *********************************
# TODO: Remove command line arg parsing in future versions, replace with fully fledged config file
if __name__ == "__main__":
  config = Config()
  parser = argparse.ArgumentParser()
  parser.add_argument("-m", "--model", help="Enter model name for training, leave blank if you want TEMNet network, 'Ensemble' for group training", default = None)
  parser.add_argument("-w", "--weights", help="Filename of weights to be loaded onto network. Enter 'latest' to train on most recent checkpoint. If not specified, train completely fresh")
  parser.add_argument("-t", "--tune", help="Enable for Hypertuning model", type=int, default = 0)
  args = parser.parse_args()
  print("*****************\nInput Configuration:\nLearning Rate: " + str(config.LEARNING_RATE), 
        "\nBatch Size: " + str(config.BATCH_SIZE), 
        "\nEpochs: " + str(config.EPOCHS),
        "\nImage resolution: " + str(config.RESOLUTION) + "x" + str(config.RESOLUTION),
        "\nHypertune Enable: " + str(args.tune),
        "\nWeight set: " + str(args.weights), # TODO: Add config support for weight path
        "\n*****************\n")
  if(args.model == 'temnet' or args.model == 'TEMNet' or args.model == None):
    temnet_network(config, args.weights, args.tune)
  elif(args.model == 'ensemble' or args.model == 'Ensemble'):
    model_ensemble(config)
  else:
    pretrained_network(args.model, config) 
