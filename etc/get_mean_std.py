import os
import numpy as np
from PIL import Image

from scripts.app.app import IMG_FORMAT #Tensorflow's load image uses PIL.Image anyways lol
#from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img, save_img

def load_image_safe(imgname, target_size=None, verbose = False):
    """
    Loads a .tif or .png image and converts it to a int8 bit representation
    INPUTS:
        imgname: path of the image to be loaded
    OUTPUTS
        np_img: np.uint8 array containing the 8bit representation of the image (This is the way it would be seen in fiji)
    """
    #Open TIF image using PIL
    im = Image.open(imgname)
    # Load the data into a flat numpy array of the correct type and reshape
    if verbose: print(f'Loading {imgname} with type {im.mode}')
    dtype = {'F': np.float32, 'L': np.uint8, 'I;16': np.uint16, 'I': np.uint32}[im.mode] 
    if verbose: print(f"Image minmax: {(im.min(), im.max())}")
    if target_size!=None:
        if verbose: print(f'Resizing to: {target_size}')
        im = im.resize(size=target_size, resample=Image.NEAREST)
    try:
        np_img = np.array(im, dtype=dtype)
    except:
        #Let numpy decide which dtype to use, we're gonna send this to np.uint8 anyways lol
        np_img = np.array(im)
    if verbose: print(f"Loaded into numpy array of type {np_img.dtype} and shape {np_img.shape}")
    if len(np_img.shape > 2):  print(f"WARNING: Image shape: {np_img.shape} contains more than one channel")
    #w, h = im.size
    #np_img = np.image.reshape((h, w, np_img.size // (w * h)))
    #Normalize the image 
    np_img = (np_img - np_img.min())/(np_img.max() -np_img.max())
    # Copy the data into each RGB channel for visualization purposes
    np_img = np.stack([np_img, np_img, np_img], axis=2)
    # Scale the image to get a 8bit representation
    np_img =  (256*np_img).astype(np.uint8)
    #np_img = np_img[:,:,:,0]
    return np_img

if __name__ == '__main__':
    #IMG_FORMAT = ('.tif','.png','.jpg','.jpeg','.bpm','.eps')
    TRAIN_PATH='../dataset/rcnn_multiviral_dataset_full/train'
    image_ids=next(os.walk(TRAIN_PATH))[1]#All directory names in read_path stored in an array
    print(image_ids)
    #Load every image
    print(f"Number of images to process: {len(image_ids)}")
    means = []
    stds = []
    for i,img_name in enumerate(image_ids):
        print(77*"#")
        print(f"Processing image #{i}\n")
        #Images should be .tif or .png, we could also loop and search for images ending in an image format
        try:
            LOAD_PATH=os.path.join(TRAIN_PATH,img_name,img_name+'.tif')
            print(f"Loading image from {LOAD_PATH}")
            image = load_image_safe(LOAD_PATH, verbose=True)
        except:
            LOAD_PATH=os.path.join(TRAIN_PATH,img_name,img_name+'.png')
            print(f"Loading image from {LOAD_PATH}")
            image = load_image_safe(LOAD_PATH)
        means.append(np.mean(image, axis=(0,1)))
        stds.append(np.std(image, axis=(0,1)))
    full_mean = np.array(means).mean(axis=0)
    full_std = np.array(stds).mean(axis=0)
    print()


