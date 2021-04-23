import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import numpy as np
import os
#from vis.visualization import visualize_saliency
#from vis.utils import utils
import cv2

import training as T
import models as M
from classes import PerillaNet, Config, Image



"""
graph_results:  Plots the accuracy and loss of a models training run and saves them as an image
Inputs: hist, lr, bs, en, optim, and mod, the various hyperparameters of the model
"""
def graph_results(path, hist, lr, bs, optim, mod, res):
  os.chdir(path)
  f, (acc, loss) = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))

  f.tight_layout(pad=5.0)
  acc.set_title('Model Accuracy')
  acc.grid()
  acc.set(xlabel='epoch', ylabel='accuracy')
  acc.axis([0, 100, 0.7, 1])
  acc.plot(hist.history['accuracy'])
  acc.plot(hist.history['val_accuracy'])
  acc.text(2, 0.75, 'Model: ' + mod + ' Optimizer: ' + optim + '\nLearning rate = ' + str(lr) + ' Batch Size = ' + str(bs) + ' Resolution = ' + str(res) + "x" + str(res), fontsize=9)
  acc.legend(['Training', 'Validation'], loc='lower right')

  loss.set_title('Model Loss')
  loss.grid()
  loss.set(xlabel='epoch', ylabel='loss')
  loss.axis([0, 100, 0, 1])
  loss.plot(hist.history['loss'])
  loss.plot(hist.history['val_loss'])
  loss.text(2, 0.8, 'Model Skeleton: ' + mod + ' Optimizer: ' + optim + '\nLearning rate = ' + str(lr) + ' Batch Size = ' + str(bs), fontsize=9)
  loss.legend(['Training', 'Validation'], loc='upper right')
  f.savefig(mod + '_' + optim + '_' + str(lr) + '_' + str(bs) +'.png')

def plot_ensemble(path, history, lr, bs, optim, res):
  os.chdir(path)
  names = ['ResNet50', 'ResNet101', 'ResNet152', 'ResNet50_V2', 'ResNet101_V2', 'ResNet152_V2', 'VGG16', 'VGG19', 'INCEPTION_V3']
  avgs = []
  std_dev = []
  for f in range(len(history)):
    ave = 0
    for h in history[f]:
      ave += h
    avgs.append(ave/len(history[f]))
    std_dev.append(np.std(history[f]))  
  plt.clf()
  plt.errorbar(names, avgs, yerr=std_dev, linestyle='None', color='m', marker='^', capsize=3, ecolor='c', elinewidth=1.5, linewidth=2.5, fontsize = 30)
  plt.ylabel("Average Validation Accuracy", fontsize = 40)
  plt.xlabel("")
  plt.title("Average Validation Accuracy Per Network Configuration", fontsize = 40)
  plt.savefig("ensemble_avgs.png")

def different_resolutions():
  history = []
  avgs = []
  std_dev = []
  for i in range(100, 1300, 100):
    history.append(T.novel_network(0.00015, 50, None, 0, i, 0, 50))
  for f in range(len(history)):
    ave = 0
    for h in history[f]:
      ave += h
    avgs.append(ave/len(history[f]))
    std_dev.append(np.std(history[f]))  
  os.chdir('/home/07049/tg863484/scratch/hivclass/graphs')
  resolutions = ['100x100', '200x200', '300x300', '400x400', '500x500', '600x600', '700x700', '800x800', '900x900', '1000x1000', '1100x1100', '1200x1200']
  plt.clf()
  plt.ylim(0, 1)
  plt.grid()
  plt.errorbar(resolutions, avgs, color='m', yerr=std_dev, marker='^', capsize=5, ecolor='r', elinewidth=1.5, linewidth=2.5)
  plt.xlabel("Resolution (px)", fontsize = 16)
  plt.ylabel("Average Validation Accuracy", fontsize = 16)
  plt.title("Average Validation Accuracy Per Input Image Resolution Size", fontsize = 20)
  plt.savefig("avg_val_res.png")

"""def plot_saliency_map():
  config = Config()
  model = PerillaNet(config, '90run')
  layer_index = utils.find_layer_idx(model.model, 'visualized_layer')
  model.model.load_weights(tf.train.latest_checkpoint(config.CHECKPOINT_PATH + '90run'))
  model.model.layers[layer_index].activation = activations.linear
  model.model = utils.apply_modifications(model.model)
  input_image = Image('/scratch/07049/tg863484/imgs/class_imgs/m45_0134008.tif')
  input_image = cv2.resize(input_image.data, (400, 400))
  classification = 'mature'
  classNum = 2
  fig, axes = plt.subplots(1, 2)
  visualization = visualize_saliency(model.model, layer_index, filter_indices=classNum, seed_input=input_image)
  axes[0].imshow(input_image[..., 0])
  axes[0].set_title('Original Image')
  axes[1].imshow(visualization)
  axes[1].set_title('Saliency Map')
  fig.savefig('SALIENCY.png')"""


"""
plot_prediction: used to visualize prediction performance of a given netwok

"""
def plot_prediction(image, imgName, predictions, regions):
  fig, axes = plt.subplots(ncols=1, figsize=(20, 13))
  axes.imshow(image)
  counter = 0
  for i in regions:
    color = ''
    if(predictions[counter] == 0):
      color = 'y'
    elif(predictions[counter] == 1):
      color = 'b'
    elif(predictions[counter] == 2):
      color = 'r'
    rect = patches.Rectangle((i[3], i[0]), i[1]-i[3], i[2]-i[0], linewidth = 1.5, edgecolor=color, facecolor='none')
    axes.add_patch(rect)
    # Add classification text
    counter += 1
  # Add color legend
  fig.savefig(imgName + "_prediction_outcome.png", bbox_inches='tight', pad_inches = 0.5)

if __name__ == "__main__":
  print("OI")