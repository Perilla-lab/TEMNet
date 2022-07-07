#This code is supposed to work as a stand alone to generate our datasets from the input images, that's why I'm copying some functions from or input_pipeline.py ~ JR
import matplotlib.pyplot as plt
import numpy as np
import copy
from PIL import Image
import tensorflow as tf
#from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img, save_img
from tensorflow.keras.preprocessing.image import array_to_img, save_img
import cv2, re, csv, os

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
    if target_size!=None:
        if verbose: print(f'Resizing to: {target_size}')
        im = im.resize(size=target_size, resample=Image.NEAREST)
    try:
        dtype = {'F': np.float32, 'L': np.uint8, 'I;16': np.uint16, 'I': np.uint32}[im.mode] 
        np_img = np.array(im, dtype=dtype)
    except:
        #Let numpy decide which dtype to use, we're gonna send this to np.uint8 anyways lol
        np_img = np.array(im)
    if verbose: print(f"Loaded into numpy array of type {np_img.dtype} and shape {np_img.shape}")
    if verbose: print(f"Image minmax: {(np_img.min(), np_img.max())}")
    if len(np_img.shape) < 3:  print(f"WARNING: Image shape: {np_img.shape} contains one channel, stacking to make three channels...")
    #w, h = im.size
    #np_img = np.image.reshape((h, w, np_img.size // (w * h)))
    #Normalize the image 
    np_img = (np_img - np_img.min())/(np_img.max() - np_img.min())
    # Copy the data into each RGB channel for visualization purposes
    if len(np_img.shape) == 2: np_img = np.stack([np_img, np_img, np_img], axis=2)
    # Scale the image to get a 8bit representation
    np_img =  (255*np_img).astype(np.uint8)
    #np_img = np_img[:,:,:,0]
    return np_img


def parse_region_data(csvname):
  """
  parse_region_data: Open a CSV for an annotated dataset and parse information to generate RoIs to validate against
  Inputs:
    csvname, the filename of the CSV you wish to parse
  Outputs: idx, lab, x, y, w, h, the respective metadata contained within the CSV
  """
  print("input_pypeline: parsing data from file", csvname)
  idx = []
  x   = []
  y   = []
  w   = []
  h   = []
  lab = []
  with open(csvname, newline='') as labels:
    fields = ['#filename',
              'file_size',
              'file_attributes',
              'region_count',
              'region_id',
              'region_shape_attributes',
              'region_attributes']
    reader = csv.DictReader(labels,
      fieldnames=fields,
      dialect='excel',
      quoting=csv.QUOTE_MINIMAL)
    for row in reader:
      lt = row['region_attributes']
      lt = str(re.sub(r'([{}"])','',lt))
      lt = lt.split(':')
      if len(lt) == 2:
        lab.append(lt[1])
      idx.append(row['region_id'])
      tmp = row['region_shape_attributes']
      tmp = str(re.sub(r'([{}"])','',tmp))
      tmp = tmp.split(',')
      for t in tmp:
        tt = t.split(':')
        if len(tt) == 1:
          continue
        else:
          if tt[0] == 'x':
            x.append(int(tt[1]))
          elif tt[0] == 'y':
            y.append(int(tt[1]))
          elif tt[0] == 'height':
            h.append(int(tt[1]))
          elif tt[0] == 'width':
            w.append(int(tt[1]))
  idx.remove('region_id')
  print("input_pypeling: parsing region data with length (number of GT boxes) ", len(x))
  return idx, lab, x, y, w, h

def write_region_data(csvname, idx, lab, x, y, w, h, max_height, max_width):
  """
  write_region_data: Open a CSV and write information equivalent to that of the dataset to generate RoIs to validate against. Essentially the reverse of parse_region_data
  Inputs:
    csvname: the filename of the CSV you wish to write
    idx, lab, x, y, w, h: numpy arrays containing the RoI info to be written in the csv
    max_height, max_width: dimensions of the source image where the RoIs should exist, this is for validation that no RoI coordinate goes outside the image
  Outputs:
    None, writes a CSV with the aforementioned data that can be parsed with parse_region_data
  """
  #Set all boxes outside our image to zero, this allows us to delete them afterwards
  local_w=copy.deepcopy(w)
  local_h=copy.deepcopy(h)
  local_w[x+w>max_width]=0
  local_w[x<0]=0
  local_h[y+h>max_height]=0
  local_h[y<0]=0
  print("input_pypeline: writing data to file", csvname)
  with open(csvname, mode='w', newline='') as labels:
    fields = ['#filename',
              'file_size',
              'file_attributes',
              'region_count',
              'region_id',
              'region_shape_attributes',
              'region_attributes']
    writer = csv.DictWriter(labels,
      fieldnames=fields,
      dialect='excel',
      quoting=csv.QUOTE_MINIMAL)
    
    writer.writeheader()
    n_region=0
    for i in range(len(x)):
      if (local_h[i]!=0 and local_w[i]!=0):
        n_region+=1
        writer.writerow({'#filename': csvname,
                       'file_size': 'irrelevant', 
                       'file_attributes': '{}',
                       'region_count': len(x),
                       'region_id': idx[i],
                       'region_shape_attributes': '{"name":"rect","x":'+str(x[i])+',"y":'+str(y[i])+',"width":'+str(w[i])+',"height":'+str(h[i])+'}',
                       'region_attributes':'{"particle_class":'+lab[i]+'}'
                       })
  print(f"input_pypeling: done writing region data with length (number of GT boxes) {n_region} to {csvname}")

def augment(image, x, y, w, h, atype=None):
  """
  augment: Transform an image according to a given atype of transformation and applies it, also transform the box coordinates so they match the transformed image
  Inputs:
    image: numpy matrix representation of the source image to transform
    x, y, w, h: numpy arrays containing the coordinates of the RoI boxes corresponding to image
   atype: augmentation type: None, 'horizontal-flip', 'vertical-flip', 'rotation-180', 'salt-pepper' (also known as gaussian noise), 'translate-up', 'translate-down', 'translate-left', 'translate-rigth'
  Outputs:
    aug_image, aug_x, aug_y, aug_w, aug_h: augmented image as well as the modified box coordinates
  """
  img_height, img_width = image.shape[:2]
  #print("img h:", img_height)
  #print("img w:", img_width)
  height = np.full(len(x), img_height)
  #print("len h:", len(height))
  width = np.full(len(x), img_width)
  #print("len w:", len(width))
  aug_image = image
  aug_x, aug_y, aug_w, aug_h = x, y, w, h
  if atype==None:
    aug_image = image
    aug_x, aug_y, aug_w, aug_h = x, y, w, h
  elif atype=='horizontal-flip':
    aug_image = cv2.flip(image,1)
    aug_x, aug_y, aug_w, aug_h = width - (x+w), y, w, h
  elif atype=='vertical-flip':
    aug_image = cv2.flip(image,0)
    aug_x, aug_y, aug_w, aug_h = x, height - (y+h), w, h
  elif atype=='180-rotation':
    aug_image = cv2.flip(image,-1)
    aug_x, aug_y, aug_w, aug_h = width - (x+w), height- (y+h), w, h
  elif atype=='salt-pepper':
    #Add gaussian noise of mean 0 and stddev 1
    aug_image=tf.cast(image/255, dtype = tf.float32)
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=1.0, dtype=tf.float32)
    aug_image = tf.add(aug_image, 0.05*noise)
    aug_image=tf.cast(aug_image*255,dtype=tf.uint8)
    aug_x, aug_y, aug_w, aug_h = x, y, w, h
  elif atype=='translate-up':
    #Pad the image to move it and then crop it, displace the box coordinates accordingly
    pad_top, pad_bottom, pad_left, pad_right = 0, img_height//2, 0, 0
    aug_image = tf.image.pad_to_bounding_box(image, pad_top, pad_left, img_height + pad_bottom + pad_top, img_width + pad_right + pad_left)
    aug_image = tf.image.crop_to_bounding_box(aug_image, pad_bottom, pad_right, img_height, img_width)
    aug_x, aug_y, aug_w, aug_h = x -(pad_right - pad_left)*np.ones(len(x), dtype='int32'), y -np.full(len(x),(pad_bottom - pad_top)), w, h
  elif atype=='translate-down':
    pad_top, pad_bottom, pad_left, pad_right = img_height//2, 0, 0, 0
    aug_image = tf.image.pad_to_bounding_box(image, pad_top, pad_left, img_height + pad_bottom + pad_top, img_width + pad_right + pad_left)
    aug_image = tf.image.crop_to_bounding_box(aug_image, pad_bottom, pad_right, img_height, img_width)
    aug_x, aug_y, aug_w, aug_h = x -(pad_right - pad_left)*np.ones(len(x), dtype='int32'), y -np.full(len(x),pad_bottom - pad_top), w, h
  elif atype=='translate-right':
    pad_top, pad_bottom, pad_left, pad_right = 0, 0, img_width//2, 0
    aug_image = tf.image.pad_to_bounding_box(image, pad_top, pad_left, img_height + pad_bottom + pad_top, img_width + pad_right + pad_left)
    aug_image = tf.image.crop_to_bounding_box(aug_image, pad_bottom, pad_right, img_height, img_width)
    aug_x, aug_y, aug_w, aug_h = x -(pad_right - pad_left)*np.ones(len(x), dtype='int32'), y -np.full(len(x),(pad_bottom - pad_top)), w, h
  elif atype=='translate-left':
    pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, img_width//2
    aug_image = tf.image.pad_to_bounding_box(image, pad_top, pad_left, img_height + pad_bottom + pad_top, img_width + pad_right + pad_left)
    aug_image = tf.image.crop_to_bounding_box(aug_image, pad_bottom, pad_right, img_height, img_width)
    aug_x, aug_y, aug_w, aug_h = x -(pad_right - pad_left)*np.ones(len(x), dtype='int32'), y -np.full(len(x),(pad_bottom - pad_top)), w, h

  return aug_image, aug_x, aug_y, aug_w, aug_h

def crop_image_center(image, crop_size, starting_point, idx, lab, x, y, w, h):
  """
  crop_image: Crop an image to a restricted region defined by a starting point and a cropping size and return the boxes inside that region
  Inputs:
    image: numpy matrix representation of the source image to transform
    crop_size: [H,W] size of the cropping region in pixels
    starting_point: [Y,X] coordinates of the zero point for the cropped image (pixels)
    idx, lab, x, y, w, h: numpy arrays containing the index, labels and coordinates of the RoI boxes corresponding to image
  Outputs:
    aug_image, aug_idx, aug_lab, aug_x, aug_y, aug_w, aug_h: augmented image as well as the index, label and modified box coordinates of the RoIs inside the cropping region
  """
  start_y = starting_point[0]
  start_x = starting_point[1]
  end_y = starting_point[0]+crop_size[0]
  end_x = starting_point[1]+crop_size[1]
  aug_image = image[start_y:end_y, start_x:end_x]
  #Now limit the boxes to those between our coordinates
  #If the center of the box is in the cropped section, consider it as valid
  centre_x = x+0.5*w
  centre_y = y+0.5*h
  masks = [centre_x >= start_x, centre_x <= end_x, centre_y >= start_y, centre_y <= end_y]
  mask = masks[0] & masks[1] & masks[2] & masks[3]
  # print(f"mask: {mask}")
  aug_idx = idx[mask]
  aug_lab = lab[mask]
  aug_x = x[mask] - start_x #Adjust coordinates to the new image 0
  aug_y = y[mask] - start_y #Adjust coordinates to the new image 0
  aug_w = w[mask]
  aug_h = h[mask]
  #There might be boxes with negative coordinates or with coordiantes outside the cropped section, fix those
  aug_w[aug_x<0]=aug_w[aug_x<0]+aug_x[aug_x<0]
  aug_w[aug_x+aug_w>crop_size[1]]=crop_size[1]-aug_x[aug_x+aug_w>crop_size[1]]
  aug_x[aug_x<0]=0
  aug_h[aug_y<0]=aug_h[aug_y<0]+aug_y[aug_y<0]
  aug_h[aug_y+aug_h>crop_size[0]]=crop_size[0]-aug_y[aug_y+aug_h>crop_size[0]]
  aug_y[aug_y<0]=0
  # print(f"aug_x: {aug_x}")
  # print(f"aug_y: {aug_y}")
  return aug_image, aug_idx, aug_lab, aug_x, aug_y, aug_w, aug_h

def calculate_iou_matrix(anchors, gt_boxes):
    """
    calculate_iou_matrix: Creates a jaccard index matrix for each anchor and ground truth bounding box
    Inputs:
      anchors, an array of anchors in the form of bounding box coords
      gt_boxes, the coords of ground truth bounding boxes
    Ouputs:
      iou matrix: [len(anchors),len(gt_boxes)]
    """

    #stopwatch=Stopwatch("iou_matrix")
    y1_bbox, y1_anchor = np.meshgrid(gt_boxes[:, 1], anchors[:, 1])  # higher y value
    x1_bbox, x1_anchor = np.meshgrid(gt_boxes[:, 0], anchors[:, 0])  # lower x value
    y2_bbox, y2_anchor = np.meshgrid(gt_boxes[:, 3], anchors[:, 3])  # lower y value
    x2_bbox, x2_anchor = np.meshgrid(gt_boxes[:, 2], anchors[:, 2])  # higher x value

    boxArea = (x2_bbox - x1_bbox) * (y2_bbox - y1_bbox)
    anchorArea = (x2_anchor - x1_anchor) * (y2_anchor - y1_anchor)

    x1 = np.maximum(x1_bbox, x1_anchor)
    x2 = np.minimum(x2_bbox, x2_anchor)
    y1 = np.maximum(y1_bbox, y1_anchor)
    y2 = np.minimum(y2_anchor, y2_bbox)
    intersection = np.maximum(0, y2-y1) * np.maximum(0, x2-x1)

    union = (boxArea + anchorArea) - intersection
    return intersection / union

def crop_image(image, crop_size, starting_point, idx, lab, x, y, w, h):
  """
  crop_image: Crop an image to a restricted region defined by a starting point and a cropping size and return the boxes inside that region
  Inputs:
    image: numpy matrix representation of the source image to transform
    crop_size: [H,W] size of the cropping region in pixels
    starting_point: [Y,X] coordinates of the zero point for the cropped image (pixels)
    idx, lab, x, y, w, h: numpy arrays containing the index, labels and coordinates of the RoI boxes corresponding to image
  Outputs:
    aug_image, aug_idx, aug_lab, aug_x, aug_y, aug_w, aug_h: augmented image as well as the index, label and modified box coordinates of the RoIs inside the cropping region
  """
  IoU_treshold = 0.75
  start_y = starting_point[0]
  start_x = starting_point[1]
  end_y = starting_point[0]+crop_size[0]
  end_x = starting_point[1]+crop_size[1]
  aug_image = image[start_y:end_y, start_x:end_x]
  #Now limit the boxes to those between our coordinates
  # Pre selection coordinates
  pre_x = np.array(x)
  pre_y = np.array(y)
  pre_w = np.array(w)
  pre_h = np.array(h)
  #Modify coordinates outside the crop region so they have either 0 height or width
  pre_w[pre_x<start_x]=np.maximum(pre_w[pre_x<start_x]+pre_x[pre_x<start_x]-start_x, 0)
  pre_w[pre_x+pre_w>end_x]=np.maximum(end_x-pre_x[pre_x+pre_w>end_x], 0)
  pre_x[pre_x<start_x]=start_x
  #print(pre_w)
  #print(pre_x)
  pre_h[pre_y<start_y]=np.maximum(pre_h[pre_y<start_y]+pre_y[pre_y<start_y]-start_y, 0)
  pre_h[pre_y+pre_h>end_y]=np.maximum(end_y-pre_y[pre_y+pre_h>end_y], 0)
  pre_y[pre_y<start_y]=start_y
  #print(pre_h)
  #print(pre_y)
  masks = [pre_w>0, pre_h>0] #Kill those with 0 width/height since their area is 0
  mask = masks[0] & masks[1]
  pre_x = pre_x[mask]
  pre_y = pre_y[mask]
  pre_w = pre_w[mask]
  pre_h = pre_h[mask]
  pre_lab = lab[mask]
  pre_idx = idx[mask]
  #Now we have preselected the boxes inside the cropping region
  #Do a final selection of only those with IoU greater than 0.75, that is that at least 75% of their area is in the cropped region
  IoU = calculate_iou_matrix(np.stack((pre_x,pre_y,pre_x+pre_w,pre_y+pre_h), axis=1), np.stack((x,y,x+w,y+h), axis=1))
  #print("IoU shape:", IoU.shape)
  #print(IoU)
  pre_IoU_argmax = np.argmax(IoU, axis = 1) # Size of (#pre_x)
  pre_IoU_max = IoU[np.arange(IoU.shape[0]), pre_IoU_argmax] # The max IoU for each preselected box
  #print(pre_IoU_max)
  mask = pre_IoU_max > IoU_treshold # Leave only those with IoU > 0.75
  aug_x = pre_x[mask] - start_x #Adjust coordinates to the new image 0
  aug_y = pre_y[mask] - start_y #Adjust coordinates to the new image 0
  aug_w = pre_w[mask]
  aug_h = pre_h[mask]
  aug_lab = pre_lab[mask]
  aug_idx = pre_idx[mask]
  #There might be boxes with negative coordinates or with coordiantes outside the cropped section, fix those
  aug_w[aug_x<0]=aug_w[aug_x<0]+aug_x[aug_x<0]
  aug_w[aug_x+aug_w>crop_size[1]]=crop_size[1]-aug_x[aug_x+aug_w>crop_size[1]]
  aug_x[aug_x<0]=0
  aug_h[aug_y<0]=aug_h[aug_y<0]+aug_y[aug_y<0]
  aug_h[aug_y+aug_h>crop_size[0]]=crop_size[0]-aug_y[aug_y+aug_h>crop_size[0]]
  aug_y[aug_y<0]=0

  # print(f"aug_x: {aug_x}")
  # print(f"aug_y: {aug_y}")
  return aug_image, aug_idx, aug_lab, aug_x, aug_y, aug_w, aug_h

def multicrop_image(image, crop_size, crop_step, idx, lab, x, y, w, h):
  """
  multicrop_image: Sequentially crop an image to restricted regions defined by a starting point and a cropping size and return the boxes inside those regions
  Inputs:
    image: numpy matrix representation of the source image to transform
    crop_size: [H,W] size of the cropping region in pixels
    crop_step: [Y,X] size in pixels of the step to move the cropping region
    idx, lab, x, y, w, h: numpy arrays containing the index, labels and coordinates of the RoI boxes corresponding to image
  Outputs:
    crop_imgs, crop_idxs, crop_labs, crop_xs, crop_ys, crop_ws, crop_hs: arrays of cropped images as well as the corresponding indices, labels and modified box coordinates for the RoIs inside those cropperd images
  """
  crop_imgs = []
  crop_idxs = []
  crop_labs = []
  crop_xs = []
  crop_ys = []
  crop_ws = []
  crop_hs = []
  if image.shape[1]//crop_step[1] == image.shape[1]/crop_step[1]:
    nx = image.shape[1]//crop_step[1]
  else:
    nx = 1+image.shape[1]//crop_step[1]
  if image.shape[0]//crop_step[0] == image.shape[0]/crop_step[0]:
    ny = image.shape[0]//crop_step[0]
  else:
    ny = 1+image.shape[0]//crop_step[0]
  #print(f"nx: {nx}, ny: {ny}")
  for i in range(nx):
    for j in range(ny):
      starting_point = (j*crop_step[0], i*crop_step[1])
      if starting_point[0]+crop_size[0] > image.shape[0]:
        starting_point = (image.shape[0]-crop_size[0],starting_point[1])
      if starting_point[1]+crop_size[1] > image.shape[1] :
        starting_point = (starting_point[0],image.shape[1]-crop_size[1])
      #print(f"starting point: {starting_point} for i: {i}, j: {j}")  
      crop_img, crop_idx, crop_lab, crop_x, crop_y, crop_w, crop_h = crop_image(image, crop_size, starting_point, idx, lab, x, y, w, h)
      #Use only images that have boxes in them
      if len(crop_idx) < 1 :
        continue
      crop_imgs.append(crop_img)
      crop_idxs.append(crop_idx)
      crop_labs.append(crop_lab)
      crop_xs.append(crop_x)
      crop_ys.append(crop_y)
      crop_ws.append(crop_w)
      crop_hs.append(crop_h)
  print(f"### Number of images generated: {len(crop_imgs)}")
  return crop_imgs, crop_idxs, crop_labs, crop_xs, crop_ys, crop_ws, crop_hs

def expand_images(read_path, rewrite=True):
    """
    expand_images: Sequentially apply a list of augmentations to images in a path and save them
    Inputs:
      read_path: path to read images from, should be structured as /read_path/image_name/image_name.png
    Outputs:
      None, saves augmented images to folders /read_path/image_name-augmentation_name/image_name-augmentation_name.png
    """
    image_ids_all=next(os.walk(read_path))[1] #All directory names in read_path stored in an array
    #Filter out folders that are already augmented
    forbidden_words = ['horizontal-flip', 'vertical-flip', '180-rotation','salt-pepper']
    image_ids = image_ids_all
    for f in range(len(forbidden_words)):
        image_ids = [i for i in image_ids if forbidden_words[f] not in i]
    print(image_ids)
    #Load every image
    print("Number of images to process: {len(image_ids)}")
    for i,img_name in enumerate(image_ids):
        augmentations = ['horizontal-flip', 'vertical-flip', '180-rotation','salt-pepper']
        print(77*"#")
        print(f"Processing image #{i}\n")
        try:
            LOAD_PATH=os.path.join(read_path,img_name,img_name+'.tif')
            print(f"Loading image from {LOAD_PATH}")
            image = load_image_safe(LOAD_PATH, verbose=True)
        except:
            LOAD_PATH=os.path.join(read_path,img_name,img_name+'.png')
            print(f"Loading image from {LOAD_PATH}")
            image = load_image_safe(LOAD_PATH)
        #LOAD_PATH=os.path.join(read_path,img_name,img_name+'.png')
        #print(f"Loading image from {LOAD_PATH}")
        #img = load_img(LOAD_PATH)
        #img_array=img_to_array(img)
        #image = np.uint8(img_to_array(load_img(LOAD_PATH)))
        max_height, max_width = image.shape[:2]
        idx, lab, x, y, w, h = parse_region_data(os.path.join(read_path,img_name,'region_data_'+img_name+'.csv')) #'region_data_7826001.csv'
        #Augment, change to np arrays for easy matrix manipulation
        x = np.array(x, dtype='int32')
        w = np.array(w, dtype='int32')
        y = np.array(y, dtype='int32')
        h = np.array(h, dtype='int32')
        #Fix bad coordinates
        x[x<0]=0
        y[y<0]=0
        x[x+w>max_width]=max_width
        y[y+h>max_height]=max_height
        #augmentations = ['horizontal-flip', 'vertical-flip', '180-rotation','salt-pepper','translate-up','translate-down','translate-right','translate-left']
        for augName in augmentations:
          #Create folders for the augmented images
          SAVE_PATH=os.path.join(read_path,img_name+'-'+augName)#/imgs/train/7826001-horizontal-flip/
          print(f"Saving folder: {SAVE_PATH}")
          if os.path.exists(SAVE_PATH):
              print("Saving folder found")
              if not rewrite:
                  print("Not rewriting that")
                  continue
          if not os.path.exists(SAVE_PATH):
              print(f"{SAVE_PATH} doesn't exist, creating it")
              os.makedirs(SAVE_PATH)
          aug_image, aug_x, aug_y, aug_w, aug_h = augment(image, x, y, w, h, augName)
          #Save image and write csv
          save_img(os.path.join(SAVE_PATH,img_name+'-'+augName+'.png'), aug_image)
          write_region_data(os.path.join(SAVE_PATH,'region_data_'+img_name+'-'+augName+'.csv'), idx, lab, aug_x, aug_y, aug_w, aug_h, max_height, max_width)

def expand_images_crops(crop_size, step_size, read_path, write_path, rewrite=True):
  """
  expand_images: Sequentially crop images and save the cropped images as well as their csvs
  Inputs:
    crop_size: [H,W] size of the cropping regions
    step_size: [Y,X] step size for moving the cropping regions
    read_path: path to read images from, should be structured as /read_path/image_name/image_name.png
    write_path: path to write the cropped images to
    rewrite: wether to rewrite files if they already exist
  Outputs:
    None, saves augmented images to folders /write_path/image_name-crops##/image_name-crops##.png
  """
  image_ids=next(os.walk(read_path))[1]#All directory names in read_path stored in an array
  print(image_ids)
  #Load every image
  print(f"Number of images to process: {len(image_ids)}")
  for i,img_name in enumerate(image_ids):
    print(77*"#")
    print(f"Processing image #{i}\n")
    try:
      LOAD_PATH=os.path.join(read_path,img_name,img_name+'.tif')
      print(f"Loading image from {LOAD_PATH}")
      image = load_image_safe(LOAD_PATH, verbose=True)
    except:
      LOAD_PATH=os.path.join(read_path,img_name,img_name+'.png')
      print(f"Loading image from {LOAD_PATH}")
      image = load_image_safe(LOAD_PATH)
    #LOAD_PATH=os.path.join(read_path,img_name,img_name+'.png')
    #LOAD_PATH=os.path.join(read_path,img_name,img_name+'.tif')
    #print(f"Loading image from {LOAD_PATH}")
    #img = load_img(LOAD_PATH)
    #img_array=img_to_array(img)
    #image = np.uint8(img_to_array(load_img(LOAD_PATH)))
    max_height, max_width = image.shape[:2]
    idx, lab, x, y, w, h = parse_region_data(os.path.join(read_path,img_name,'region_data_'+img_name+'.csv')) #'region_data_7826001.csv'
    #Augment, change to np arrays for easy matrix manipulation
    idx = np.array(idx, dtype='int32')
    lab = np.array(lab)
    x = np.array(x, dtype='int32')
    w = np.array(w, dtype='int32')
    y = np.array(y, dtype='int32')
    h = np.array(h, dtype='int32')
    #Fix bad coordinates
    x[x<0]=0
    y[y<0]=0
    x[x+w>max_width]=max_width
    y[y+h>max_height]=max_height
    crop_imgs, idxs, labs, xs, ys, ws, hs = multicrop_image(image, crop_size, step_size, idx, lab, x, y, w, h)
    for j in range(len(crop_imgs)):
      SAVE_PATH=os.path.join(write_path,img_name+'-crop'+str(j))#/imgs/train/7826001-horizontal-flip/
      print(f"Saving folder: {SAVE_PATH}")
      if os.path.exists(SAVE_PATH):
        print("Saving folder found")
        if not rewrite:
          print("Not rewriting that")
          continue
      if not os.path.exists(SAVE_PATH):
        print(f"{SAVE_PATH} doesn't exist, creating it")
        os.makedirs(SAVE_PATH)
      #Save image and write csv
      save_img(os.path.join(SAVE_PATH,img_name+'-crop'+str(j)+'.png'), crop_imgs[j])
      write_region_data(os.path.join(SAVE_PATH,'region_data_'+img_name+'-crop'+str(j)+'.csv'), idxs[j], labs[j], xs[j], ys[j], ws[j], hs[j], crop_size[0], crop_size[1])

if __name__ == '__main__':
    # Paths to search for dataset images
    TRAIN_PATH='./rcnn_dataset_full/train'
    VAL_PATH='./rcnn_dataset_full/val'

    #First crop the images into overlapping regions
    crop_size = (1024, 1024)
    step_size = (500,500)
    expand_images_crops(crop_size, step_size, TRAIN_PATH, './rcnn_dataset_augmented/train', rewrite= False)
    expand_images_crops(crop_size, step_size, VAL_PATH, './rcnn_dataset_augmented/val', rewrite = True)
    # And further expand the training dataset by rotating and adding gaussian noise
    expand_images('./rcnn_dataset_augmented/train', rewrite= True)
    # expand_images(VAL_PATH, rewrite = False)
