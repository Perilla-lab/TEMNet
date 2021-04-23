import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #Reduce the number of messages by tensorflow
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import load_img, img_to_array, array_to_img, save_img
import numpy as np
import concurrent.futures #for processing in parallel
import time

from colorama import Fore, Back, Style
WARNING=Fore.YELLOW + '# WARNING: ' + Style.RESET_ALL
INFO=Fore.GREEN + '# INFO: ' + Style.RESET_ALL
ERROR=Fore.RED + '# ERROR: ' + Style.RESET_ALL

#Make infomessages clear to the eyes
class INFOPRINT:
    def __init__(self, message):
        self.message_ = message
        print(INFO + "START -> " + self.message_)
    def __del__(self):
        print(INFO + "DONE  -> " + self.message_ + "\n")

class Config(object):
    """
    Contains specific user info to be inputed in the functions
    """
    #paths for loading training and validation images
    #PATH_PREFIX='/home/gorzy/Documents/UDEL/hivclassification/hivclass/scripts/rpn/'
    PATH_PREFIX='/home/gorzy/Documents/UDEL/hivclassification/imgs/'
    TRAIN_PATH=PATH_PREFIX+'rpn/train/'
    VAL_PATH=PATH_PREFIX+'rpn/val/'

    RESIZE_DIMENSIONS=[2620,4000] #dimensions for image resizing
    COPY_CSVS=True #Whether or not to copy csv files in image folders and place them in the resized image folders
    REWRITE=True #Whether or not to rewrite images if there's already a folder named after them

def resize_images(read_path, write_path, config):
    """
    This function resizes images in separate folders in a path read_path and writes them on a path write path
    INPUTS:
      read_path: path to read images from, they should be each on a separate folder named the same as the .png image, i.e. read_path/img_name/img_name.png
      write_path: path to write images in
      config: member of the Config class specifying the dimensions to resize and whether to copy csvs in that folder
    OUTPUT:
      None
    """
    #Get all the images
    image_ids=next(os.walk(read_path))[1]#All directory names in read_path stored in an array
    print(image_ids)
    #Load every image
    print(INFO+f"Number of images to process: {len(image_ids)}")
    for i,img_name in enumerate(image_ids):
        print(77*"#")
        print(INFO+f"Processing image #{i}\n")
        LOAD_PATH=os.path.join(read_path,img_name,img_name+'.png')
        print(f"Loading image from {LOAD_PATH}")
        img = load_img(LOAD_PATH)
        img_array=img_to_array(img)
        #Create folders for the rescaled images
        SAVE_PATH=os.path.join(write_path,img_name)
        print(f"Saving folder: {SAVE_PATH}")
        if os.path.exists(SAVE_PATH):
            print("Saving folder found")
            if not config.REWRITE:
                print("Not rewriting that")
                continue
        if not os.path.exists(SAVE_PATH):
            print(f"{SAVE_PATH} doesn't exist, creating it")
            os.makedirs(SAVE_PATH)
        print(f"Rescaling image {img_name} to dimensions {config.RESIZE_DIMENSIONS}")
        rescaled_img=tf.image.resize(img_array, config.RESIZE_DIMENSIONS)
        print(f"RESCALED IMAGE DIMENSIONS: {rescaled_img.shape}")
        rescaled_img_name=os.path.join(SAVE_PATH,img_name+'.png')
        print(f"Saving image to {rescaled_img_name}")
        save_img(rescaled_img_name,rescaled_img)
        if config.COPY_CSVS:
            print("Copying existent csvs")
            os.system('cp '+os.path.join(read_path,img_name,'*.csv')+' '+SAVE_PATH)#execute the shell command

def resize_one_image(img_name,read_path, write_path, config):
    """
    This function resizes one image in a folders in a path read_path and writes them on a path write path
    INPUTS:
      img_name: the filename of the image
      read_path: path to read images from, they should be each on a separate folder named the same as the .png image, i.e. read_path/img_name/img_name.png
      write_path: path to write images in
      config: member of the Config class specifying the dimensions to resize and whether to copy csvs in that folder
    OUTPUT:
      None
    """
    #Load every image
    #print(77*"#")
    #print(INFO+f"Processing image {img_name}\n")
    LOAD_PATH=os.path.join(read_path,img_name,img_name+'.png')
    #print(f"Loading image from {LOAD_PATH}")
    img = load_img(LOAD_PATH)
    img_array=img_to_array(img)
    #Create folders for the rescaled images
    SAVE_PATH=os.path.join(write_path,img_name)
    #print(f"Saving folder: {SAVE_PATH}")
    if os.path.exists(SAVE_PATH):
        #print("Saving folder found")
        if not config.REWRITE:
            print("Not rewriting that")
            return None
    if not os.path.exists(SAVE_PATH):
        #print(f"{SAVE_PATH} doesn't exist, creating it")
        os.makedirs(SAVE_PATH)
    #print(f"Rescaling image {img_name} to dimensions {config.RESIZE_DIMENSIONS}")
    rescaled_img=tf.image.resize(img_array, config.RESIZE_DIMENSIONS)
    #print(f"RESCALED IMAGE DIMENSIONS: {rescaled_img.shape}")
    rescaled_img_name=os.path.join(SAVE_PATH,img_name+'.png')
    #print(f"Saving image to {rescaled_img_name}")
    save_img(rescaled_img_name,rescaled_img)
    if config.COPY_CSVS:
        #print("Copying existent csvs")
        os.system('cp '+os.path.join(read_path,img_name,'*.csv')+' '+SAVE_PATH)#execute the shell command
    print(f"Image {img_name} resized to {rescaled_img.shape}")

def parallel_resize_images(read_path, write_path, config):
    """
    This function resizes images in separate folders in a path read_path and writes them on a path write path
    INPUTS:
      read_path: path to read images from, they should be each on a separate folder named the same as the .png image, i.e. read_path/img_name/img_name.png
      write_path: path to write images in
      config: member of the Config class specifying the dimensions to resize and whether to copy csvs in that folder
    OUTPUT:
      None
    """
    #Get all the images
    image_ids=next(os.walk(read_path))[1]#All directory names in read_path stored in an array
    print(image_ids)
    #Load every image
    print(INFO+f"Number of images to process: {len(image_ids)}")
    print(77*"#")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for img_name in image_ids:
            executor.submit(resize_one_image, img_name,read_path,write_path,config)


if __name__=="__main__":
    #Resize in parallel if you can afford it, do it in serial to debug
    config = Config()
    #Resize training and validation images
    path_rescaled=os.path.join(config.PATH_PREFIX,'rescaled_rpn')
    start=time.perf_counter()
    # resize_images(config.TRAIN_PATH, os.path.join(path_rescaled,'train'),config)
    # resize_images(config.VAL_PATH, os.path.join(path_rescaled,'val'),config)

    #resize in parallel
    parallel_resize_images(config.TRAIN_PATH, os.path.join(path_rescaled,'train'),config)
    parallel_resize_images(config.VAL_PATH, os.path.join(path_rescaled,'val'),config)
    finish=time.perf_counter()
    print(f"Finished in {round(finish-start,2)} seconds")
