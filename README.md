# TEMNet. HIV-1 Particle Scan Classifier Network

##### University of Delaware Apr 2021

##### Perilla Labs Research Group

##### Juan S. Rey, Gage Schuster, Hagan Beatson, Christian Lantz, Alex Bryer, Juan Perilla

## Introduction

TEMNet is a CNN backbone designed for viral particle detection from TEM micrographs. TEMNet works as the backbone for a Faster RCNN implementation used for viral instance detection and classification. Read the paper at https://doi.org/10.1016/j.csbj.2021.10.001 .
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

Installation can be done either from a Singularity container (recommended) or locally into a python environment.

### Installing From A Singularity Container

Singularity is a container platform that allows users to create and run containerized applications, making software dependencies portable and easily reproducable.

If your system does not use singularity you can contact your admin or follow the installation steps [here](https://docs.sylabs.io/guides/3.10/admin-guide/installation.html#installation-on-linux)

To begin, make sure that Singularity and Python are active in your environment.
Then, in order to download our container file, run:

```
pip install gdown
gdown 1hp4YwbEO_4mdnW570WqWMQH4wptFjPCR
```

Next, clone our github repository into a location of your choosing

```
git clone https://github.com/Perilla-lab/TEMNet.git
```

Start a singularity container with access to GPUs, and access to the TEMNet repository folder.
Make sure the TEMNet.sif file is in the current directory

```
singularity run --nv --bind [path/to/TEMNet-folder]:/mnt TEMNet.sif
cd /mnt
```

### Installing locally

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
pip install -r requirements_tf21.txt
```

these will be installed to the environment directory **/temnet-env/** so no need to worry about breaking your system :) .
The training pipeline is currently restricted to tensorflow 2.1, if you wish to train the network you should use requirements_tf21.txt. For inference any tensorflow version >=2.1 works as specified in requirements.txt.

## Downloading the dataset

You can train the network with any dataset you like. However we provide a dataset of 59 HIV-1 TEM micrographs for training and validating your data. You can download it with the script we provide in the **/dataset/** directory

```
cd TEMNet/dataset
bash download_dataset.sh
unzip '*.zip'
```

or download manually [here!](https://drive.google.com/drive/folders/1lklUSswSsQAaZCZfJPfc5qT6fNGCJ4xj?usp=sharing). After unzipping files you should have **backbone_dataset/** and **rcnn_dataset_full/** folders containing the data to train an instance classifier and the RCNN.

Then run the python script:

```
python3 augment-images.py
```

to augment the dataset into ~10k overlapping cropped images (this might take ~1 hour depending on your hardware so feel free to go for a cup of coffee and listen to your favourite music while you wait). After augmentation the dataset should be 19GB in size.

## Running Training Procedures

Faster RCNN using a TEMNet backbone (as well as other backbones) can be trained using the training script in the **/scripts/rcnn/** directory as

```
cd ../scripts/rcnn
python3 train.py -b [backbone_name]
```
provided _backbone_name_ options are temnet, resnet101, resnet101v2, inception_resnetv2. To train other architectures and ensure convergence, please provide the proper pretraining weights with the flag _-w '/path/to/weights'_. Weights for every epoch are stored on **/weights/**.

## Running Prediction Procedures

Prediction requires trained weights for a given backbone. We have provided weights and precompiled models for TEMNet, ResNet101 and ResNet101v2 which can be downloaded using the script in the **/weights/** directory.

```
cd ../../weights/
bash download_weights.sh
cd ../scripts/rcnn
```

The predict.py script in **/scripts/rcnn/** handles prediction for individual images

```
python3 predict.py -d 'single' -p '/path/to/image.png' -b [backbone_name]
```

or batches of images stored in a directory

```
python3 predict.py -d 'multiple' -p '/path/to/imgs/' -b [backbone_name]
```

More options for multi-magnification predictions or calculating object detection metrics can be explored with the -h or --help flag.

Output prediction images and count histograms are stored in the **/graphs/** directory.
