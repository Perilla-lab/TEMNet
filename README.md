# TEMNet. HIV-1 Particle Scan Classifier Network
##### University of Delaware Apr 2021
##### Perilla Labs Research Group
##### Juan Perilla, Alex Bryer, Hagan Beatson, Juan Rey

## Introduction
TEMNet is a CNN backbone designed for viral particle detection from TEM micrographs. TEMNet works as the backbone for a Faster RCNN implementation used for viral instance detection and classification.
![Faster RCNN and TEMNet architectures](/graphs/RCNN_TEMNet.png)

![HIV-1 TEM micrograph](/graphs/samples/1797001.png)
![HIV-1 predicted TEM micrograph](/graphs/rcnn/RCNN_PREDS_1797001_05_24_10_16_.png)
![HIV-1 predicted histogram](/graphs/rcnn/RCNN_COUNTS_1797001_05_24_10_16.png)

## Getting Started
TEMNet is built using **Tensorflow** and **Keras** version 2.1, **OpenCV** is an optional dependency used for image augmentation.
Important scripts for model definition, training and inference procedures are stored in the **'scripts'** directory. 
TEMNet and other convolutional backbones (ResNet, Inception, VGG) can be pretrained using the **pretraining** scripts. Be sure to change the Config class in classes.py to your preferences.

## Installation
1. (Recommended) Create a python virtual environment for installing this project's dependencies
```
python3 -m venv --system-site-packages ./temnet-env
```
and activate it
```
source ./temnet-env/bin/activate
```
your shell prompt now should look like
```
(temnet-env) user@host $
```
2. Install the python dependencies
```
pip install -r requirements.txt
```
these will be installed to the environment directory **/temnet-env/** so no need to worry about breaking your system :) .

## Downloading the dataset
You can train the network with any dataset you like. However we provide a dataset of 59 HIV-1 TEM micrographs for training and validating your data. You can download it with the script we provide 
```
bash ./dataset/download_dataset.sh
```
or manually [here!](https://drive.google.com/drive/folders/1lklUSswSsQAaZCZfJPfc5qT6fNGCJ4xj?usp=sharing) and run the python script
```
python3 augment-images.py
```
to augment the dataset into ~10k overlapping cropped images (this might take ~1 hour depending on your hardware so feel free to go for a cup of coffee and listen your favourite music while you wait). After augmentation the dataset should be 19GB in size.

## Running Training Procedures
Faster RCNN using a TEMNet backbone (as well as other backbones) can be trained using the training script in the **/scripts/rcnn/** directory as 
```
python3 train.py -b [backbone_name]
```
to train other architectures please provide the proper pretraining weights. Weights for every epoch are stored on **/weights/**.

## Running Prediction Procedures
Prediction requires trained weights for a given backbone. We have provided weights for TEMNet, ResNet101 and ResNet101v2 which can be downloaded using the script in the **/weights/** directory.
The predict.py script in **/scripts/rcnn/** handles prediction for individual images
```
python3 predict.py -d 'single' -p '/path/to/image.png' -b [backbone_name]
```
or batches of images stored in a directory
```
python3 predict.py -d 'multiple' -p '/path/to/imgs/' -b [backbone_name]
```
More options for multi-magnification predictions can be explored with the -h or --help flag.
Output prediction images and count histograms are stored in the **/graphs/** directory.
