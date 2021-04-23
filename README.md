# TEMNet. HIV-1 Particle Scan Classifier Network
##### University of Delaware Apr 2021
##### Perilla Labs Research Group
##### Juan Perilla, Alex Bryer, Hagan Beatson, Juan Rey

## Introduction
TEMNet is a CNN backbone designed for viral particle detection from TEM micrographs. TEMNet works as the backbone for a Faster RCNN implementation used for viral instance detection and classification.
![Faster RCNN and TEMNet architectures](/graphs/RCNN_TEMNet.png)

## Getting Started
TEMNet is built using **Tensorflow** and **Keras** version 2.1, **OpenCV** is an optional dependency used for image augmentation.
Important scripts for model definition, training and inference procedures are stored in the **'scripts'** directory. 
TEMNet and other convolutional backbones (ResNet, Inception, VGG) can be pretrained using the **pretraining** scripts. Be sure to change the Config class in classes.py to your preferences.

## Running Training Procedures
Faster RCNN using a TEMNet backbone (as well as other backbones) can be trained using the training script in the **/scripts/rcnn/** directory as 
```
python3 train.py
```
to train other architectures make sure to change the _BACKBONE_ in config.py and provide proper pretraining weights. Weights for every epoch are stored on **/weights/**

## Running Prediction Procedures
Prediction requires trained weights for a given backbone. We have provided weights for TEMNet, ResNet101 and ResNet101v2 which can be downloaded using the script in the **/weights/** directory.
The predict.py script in **/scripts/rcnn/** handles prediction for individual images
```
python3 predict.py -d '/path/to/image.png'
```
or batches of images
```
python3 predict.py -d 'uncropped_dataset3'
```
where the path to the directory containing the images must be specified (editing the code for now).
Output prediction images and count histograms are stored in the **/graphs/** directory.
