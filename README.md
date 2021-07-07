# TEMNet. HIV-1 Particle Scan Classifier Network
##### University of Delaware Apr 2021
##### Perilla Labs Research Group
##### Juan S. Rey, Hagan Beatson, Christian Lantz, Alex Bryer, Juan Perilla

## Introduction
TEMNet is a CNN backbone designed for viral particle detection from TEM micrographs. TEMNet works as the backbone for a Faster RCNN implementation used for viral instance detection and classification.
![Faster RCNN and TEMNet architectures](/graphs/RCNN_TEMNet.png)

![TEMNet procedure](/graphs/TEMNet_procedure.png)

## Getting Started
TEMNet is built using **Tensorflow** and **Keras** version 2.1, **OpenCV** is an optional dependency used for image augmentation.

Important scripts for model definition, training and inference procedures are stored in the **'scripts'** directory. 

The GUI app implementation can be found under **'scripts/app'** and built using pyinstaller.

TEMNet and other convolutional backbones (ResNet, Inception, VGG) can be pretrained using the **pretraining** scripts. Be sure to change the Config class in classes.py to your preferences.

## TEMNet GUI
![TEMNet GUI](/graphs/TEMNet_GUI.png)
TEMNet now offers a User Interface, just upload your TEM micrographs, click predict and save your results without having to open a terminal or go through the code.
Download the GUI from (click the logos):

 - Linux: [<img src="https://linuxfoundation.org/wp-content/uploads/linux.svg" alt="Linux Download!" width="100"/>](https://drive.google.com/uc?export=download&confirm=SlTA&id=1mCACQs_RszHeo21-IGCn_sD7CXNGSvGb)
 - Windows: [<img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Windows_10_Logo.svg" alt="Windows Download!" width="100"/>](https://drive.google.com/uc?export=download&confirm=QAVe&id=16K15fO53NTx76OBWWQBHvHHri0qWfmN2)
 - Mac: _Mac binaries incoming!_

Simply download the file, uncompress it and run the executable **./TEMNet** (Linux), **TEMNet.exe**(Windows)!


## Installation
1. (Recommended) Create a python virtual environment for installing this project's dependencies
```
python3 -m venv ./temnet-env
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
Prediction requires trained weights for a given backbone. We have provided weights and precompiled models for TEMNet, ResNet101 and ResNet101v2 which can be downloaded using the script in the **/weights/** directory.

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
