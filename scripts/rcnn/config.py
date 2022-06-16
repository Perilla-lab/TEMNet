"""
PerillaNet RCNN
Base Configuration, Dataset and utility classes

Developed by Hagan Beatson, Alex Brier and Juan Rey @ Perillalab University of Delaware (2020)
"""

import cv2, copy, os, argparse
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import Sequence
import numpy as np
import time

import input_pipeline as I

"""
A Stopwatch to measure the time a given function takes to process
"""
class Stopwatch:
    def __init__(self, message):
        """
        Receives a message to be printed alongside the time spent by the given function
        """
        self.tick_ = time.perf_counter()
        self.message_ = message
    def __del__(self):
        self.tock_ = time.perf_counter()
        print(f"Function {self.message_} finished in {self.tock_-self.tick_}s")

"""
Config class for RCNN
based on matterports mrcnn implementation (ported to tensorflow 2 by akTwelve)
https://github.com/akTwelve/Mask_RCNN
"""

class Config(object):
    #SYSTEM SPECIFIC PATH PARAMETERS
    #CHANGE THESE TO SUIT YOUR SYSTEMS DIRECTORY
    # PATH_PREFIX = '/scratch/07655/jsreyl/'
    PATH_PREFIX = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir)) # your_path/TEMNet/

    LOGS = os.path.join(PATH_PREFIX, 'logs')   # SAVE PATH FOR LOGS
    WEIGHT_PATH = os.path.join(PATH_PREFIX, 'weights')                 # SAVE PATH FOR CHECKPOINTS
    TRAIN_PATH = os.path.join(PATH_PREFIX, 'dataset','rcnn_multiviral_dataset_augmented','train')                             # PATH FOR TRAINING DATA DIRECTORY
    VAL_PATH = os.path.join(PATH_PREFIX, 'dataset','rcnn_multiviral_dataset_augmented','val')                             # PATH FOR VALIDATION DATA DIRECTORY
    TRAIN_PATH_NOAUG = os.path.join(PATH_PREFIX, 'dataset','rcnn_multiviral_dataset_full','train')                             # PATH FOR TRAINING DATA BEFORE AUGMENTATIONS
    VAL_PATH_NOAUG = os.path.join(PATH_PREFIX, 'dataset','rcnn_multiviral_dataset_full','val')                                 # PATH FOR VALIDATION DATA BEFORE_AUGMENTATIONS
    IMAGE_PATH = os.path.join(PATH_PREFIX, 'graphs','rcnn_multiviral')                     # PATH FOR GENERAL IMG SETS
    #Weight sets for different backbones
    WEIGHT_SET_DICT = {
        'temnet': WEIGHT_PATH + '/rcnn_temnet_weights_gn_res512.hdf5',
        'resnet101': WEIGHT_PATH + '/rcnn_resnet101_weights_res512.hdf5',
        'resnet101v2': WEIGHT_PATH + '/rcnn_resnet101v2_weights_res512.hdf5',
        'inception_resnetv2': WEIGHT_PATH+'/rcnn_inception_resnetv2_weights_res512.hdf5'
    }
    # General hyperparams
    #Name of the configuration, this can be overridden in Config instances
    #This is useful to create experiments with different hyperparameters and identify them
    NAME = "HIV-1"

    #Number of GPUs to run on.
    GPU_COUNT = 4

    IMAGES_PER_GPU = 2 # A 12GB GPU normally can handle 2 1024x10124px images with no problem
    #BATCH_SIZE = 5 #Number of images per batch on training
    BATCH_SIZE = GPU_COUNT * IMAGES_PER_GPU # Number of images per batch on training

    #Number of epochs for training
    EPOCHS = 50 #100

    #Backbone convolutional network to use
    #Implemented architectures: temnet, resnet50, resnet101, resnet152, resnet50v2, resnet101v2, resnet152v2, vgg, inception_resnetv2.
    BACKBONE = "temnet"

    #Strides for the feature map shapes (this is used to calculate the cnn shapes in cnn_input_shapes and to generate anchors in generate_anchors)
    BACKBONE_STRIDES = [4,8,16,32,64]
    #Standard deviations to gt_box sizes, these correspond to [h, h , w, w] and are used in regressing box deltas from predicted anchors to the actual gt_box dimensions. These box deltas are refered to as bbox
    #These values are based on a Resnet101 backbone and may be adjusted for other backbones
    RPN_BBOX_STD_DEV = np.array([0.1,0.1,0.2,0.2])

    #Feature Pyramid Network (FPN) Parameters
    #This is used to upscale and downscale the image when generating Regions of interest so our classifier doesn't depend on the scale of the objects
    #Size of top-down layers used to build the feature pyramid
    TOP_DOWN_PYRAMID_SIZE = 256

    #Number of classification classes (background + classes)
    #In our case background, eccentric, mature, immature HIV
    #And: adenovirus, astrovirus, cchf, cowpox, dengue, ebola, guanarito, influenza, lcm, lassa, machupo, marburg, nipah, virus, norovirus, orf, papilloma, pseudocowpox, rift, valley, rotavirus, sapovirus, tbe, westnile
    NUM_CLASSES = 1+3+22
    #Dictionary containing class id and name info
    CLASS_INFO=[{"id":0,"name":"BG"},
                {"id":1, "name": "eccentric"},
                {"id":2,"name":"mature"},
                {"id":3,"name":"immature"},
                {"id":4,"name":"adenovirus"},
                {"id":5,"name":"astrovirus"},
                {"id":6,"name":"cchf"},
                {"id":7,"name":"cowpox"},
                {"id":8,"name":"dengue"},
                {"id":9,"name":"ebola"},
                {"id":10,"name":"guanarito"},
                {"id":11,"name":"influenza"},
                {"id":12,"name":"lcm"},
                {"id":13,"name":"lassa"},
                {"id":14,"name":"machupo"},
                {"id":15,"name":"marburg"},
                {"id":16,"name":"nipah_virus"},
                {"id":17,"name":"norovirus"},
                {"id":18,"name":"orf"},
                {"id":19,"name":"papilloma"},
                {"id":20,"name":"pseudocowpox"},
                {"id":21,"name":"rift_valley"},
                {"id":22,"name":"rotavirus"},
                {"id":23,"name":"sapovirus"},
                {"id":24,"name":"tbe"},
                {"id":25,"name":"westnile"}]

    # Size of the image data array, this contains
    # image_id (size=1), original_image_size (size=2), image_size (size=2), crop_id (size=1), aug_id (size=1)
    IMAGE_DATA_SIZE=7

    #Region Proposal Anchor parameters
    # Length of square anchor side in pixels
    #RPN_ANCHOR_SCALES = (32,64,96,128,152)
    RPN_ANCHOR_SCALES = (32,64,128,256,512)
    # Scale ratios (width/height) for the anchors, 1 is a square anchors and 0.66 is a wide anchor
    #RPN_ANCHOR_RATIOS = [0.66,0.75,1,1.25,1.33]
    RPN_ANCHOR_RATIOS = [0.5,0.75,1,1.25,1.5]
    #Stride length for anchors
    RPN_ANCHOR_STRIDE = 2
    #Maximum number of proposed anchors per image
    ANCHORS_PER_IMG = 500
    #Non max supression threshold to eliminate duplicate Regions of interest for a given ground truth box
    RPN_NMS_THRESHOLD = 0.7
    #Max number of Regions of Interest after Non-max supression
    POST_NMS_ROIS_TRAINING = 100
    POST_NMS_ROIS_VALIDATION = 100

    # ROIs kept after tf.nn.top_k and before Non-max supression
    PRE_NMS_LIMIT = 6000
    
    # Max number of final detections
    DETECTION_MAX_INSTANCES = 20 #100

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3

    # Whether to report background instances or not
    DETECTION_EXCLUDE_BACKGROUND = False
    
    #Whether to use ROIs from RPN predictions or from an external input
    # This is useful if we're training only the classifier heads so we can disregard the RPN predictions
    USE_RPN_ROIS = True #False #True
    
    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 50 #200

    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.5    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14
    
    # Size of the fully-connected layers in the classification graph
    FPN_CLASSIF_FC_LAYERS_SIZE_TEMNET = 64
    FPN_CLASSIF_FC_LAYERS_SIZE_RESNET = 1024

    #Validation Data parameters
    #Maximum number of ground truth in each image
    MAX_GT_INSTANCES = 20 #100

    #Input image shape for the RPN, all images are rescaled to this size on image load
    IMAGE_SHAPE = (512, 512)
    #Size of the dataset images, change here for your dataset's specific needs, inputing diferent sized images is slower so it's not recommended
    DATASET_IMAGE_SIZE=(1024,1024) #For the cropped dataset
    #Number of channels for the input image, the rcnn implementation only receives color images
    #If your images are greyscale these are automatically transformed to color by our load_img
    #If your images have an alpha channel it is removed before training so the number of channels is 3 instead of 4
    NUM_CHANNELS = 3

    # Specific parameters for the training data used
    BASE_MAGNIFICATION = 30000 # Magnification of TEM micrographs used for training
    BASE_CROP_SIZE = 1024 # pixels, size of the cropped micrographs used for training
    BASE_CROP_STEP = 500 # pixels, size of the steps for the overlapping cropped micrographs used for training
    MEAN_PIXEL = np.array([131.51, 131.51, 131.51])
    STD_PIXEL = np.array([47.11, 47.11, 47.11])

    #Learning rate for model optimizer, since the model momentum is high we need to keep this down to stop ourselves from overstepping the minima
    #LEARNING_RATE = 0.000000000000001
    LEARNING_RATE = 0.01
    #Learning momentum, this is a "jump" so the stochastic gradient descent doesn't get stuck on a local minima
    LEARNING_MOMENTUM = 0.9
    #Gradient clipping, this works as an upper bound to avoid exploiding gradients. That is if the gradient (g) is greater than the clip norm (c) was, normalize it and multiply it by clip norm.
    #if (|g|>c) g=c*g/|g|
    GRADIENT_CLIP_NORM = 5.0

    #Weight decay regularization for the model optimizer
    WEIGHT_DECAY = 0.0001

    #Weight loss for more precise optimization, this can be used in the R-CNN training setup
    #Note we DON'T use a mask_loss as in mrcnn
    LOSS_WEIGHTS = {
        "rpn_match_loss": 1.,
        "rpn_bbox_loss": 1.,
        "rcnn_class_loss":1.,
        "rcnn_bbox_loss":1.
    }

    #Whether to train the batch normalization layers in network area
    TRAIN_BATCH_NORMALIZATION = False

    #Online image augmentation parameters, set these to False if you're using offline agumentation
    USE_HORIZONTAL_FLIPS = False #True
    USE_VERTICAL_FLIPS = False #True
    USE_ROTATION_180 = False #True
    USE_GAUSSIAN_NOISE = False #True

    #Optional parameters for training specific parts of the model
    RANDOM_ROIS = 0 # How many random roi regions to generate, this is useful for training only the classifier heads without having to go through the RPN RoI proposals
    GENERATE_DETECTION_TARGETS = False #True # Whether to create detection targets as outputs of the dataset class or to get them from the RPN proposals
    TRAIN_ONLY_RPN = False #Whether to train only the RPN and not the classifier heads

    def __init__(self, backbone='temnet'):
        """
        backbone: Backbone convolutional archiitecture to use for training and inference
        """
        #Verify that train and validation paths exist
        assert os.path.exists(self.TRAIN_PATH), "Train path cannot be verified"
        assert os.path.exists(self.VAL_PATH), "Validation path cannot be verified"
        #Tune specific network parameters depending on the backbone
        assert(backbone in ['temnet', 'resnet101', 'resnet101v2', 'inception_resnetv2'], 'Backbone not implemented, options are \'temnet\', \'resnet101\' or \'resnet101v2\'')
        self.BACKBONE = backbone
        self.WEIGHT_SET = self.WEIGHT_SET_DICT[self.BACKBONE]
        if(self.BACKBONE == 'temnet'):
            self.FPN_CLASSIF_FC_LAYERS_SIZE = self.FPN_CLASSIF_FC_LAYERS_SIZE_TEMNET
        else:
            self.FPN_CLASSIF_FC_LAYERS_SIZE = self.FPN_CLASSIF_FC_LAYERS_SIZE_RESNET
 

    def to_dict(self):
        """Returns a dictionary with all the attributes of the config class"""
        return {a: getattr(self,a)
                for a in sorted(dir(self))
                if not a.startswith("__") and not callable(getattr(self,a))}

    def display(self):
        """Display the config values"""
        stopwatch = Stopwatch("config.display")
        print("Configuration values:")
        for key, val in self.to_dict().items():
            print(f"{key:30} {val}")
        print("\n")


"""
Dataset
Generic class for dataset in both training and inference procedures
Inputs:
    path: where to read images from
    conifg: Config class instance specifying parameters for the newrork
    mode: either "train" or "validation" to augment or just detect images
Returns:
    A python iterable from class keras.utils.Sequence. __getitem__() is overridden so that when it's called it returns two lists: inputs and outputs
    inputs: [images, anchors, bbox, class_id] where
        images -> [batch,H,W,C] np.array representing pixel level information of each image
        anchors->[y1,x1,y2,x2] pixel level position of the generated anchors
        bbox->[batch,N,(dy,dx,ln(dh),ln(dw))] deltas to be applied to anchors to better fit the ground truth boxes
        class_id->[batch, MAS_GT_INSTANCES] id corresponding to each anchor representing the closest ground truth box class_id
    outputs:
        Empty list on usual training,
"""
class Dataset(Sequence):
    def __init__(self, path, config, mode):
        """
        path: to read images from
        config: Config class instance for training parameters
        mode: "train" or "validation"
        """
        self.path = path
        print("classes:Dataset: reading data from ", self.path)
        self.image_ids = next(os.walk(self.path))[1]#All the folders in self.path
        # print("classes:Dataset: image_ids ", self.image_ids)
        np.random.shuffle(self.image_ids)
        self.config = config
        self.mode = mode #Either train or validation
        self.random_rois = self.config.RANDOM_ROIS
        self.generate_detection_targets = self.config.GENERATE_DETECTION_TARGETS
        self.train_only_rpn = self.config.TRAIN_ONLY_RPN
        assert mode in ["train","validation"],"Dataset mode must be either train or validation"
        #Generate anchors on input_pipeline
        self.anchors = I.generate_anchors(self.config.RPN_ANCHOR_SCALES,
                                          self.config.RPN_ANCHOR_RATIOS,
                                          I.cnn_input_shapes(self.config.IMAGE_SHAPE, self.config.BACKBONE_STRIDES),
                                          self.config.BACKBONE_STRIDES,
                                          self.config.RPN_ANCHOR_STRIDE,
                                          self.config.IMAGE_SHAPE)

    def __len__(self):
        return int(np.ceil(len(self.image_ids)/self.config.BATCH_SIZE))

    def __getitem__(self,idx):
        """
        __getitem__: Overriding method to generate objects
        Inputs:
          idx: id of the dateset sample, this is an integer that represents the number of the dataset in image_ids
        Outputs:
        Returns a python iterable from class keras.utils.Sequence. __getitem__() is overridden so that when it's called it returns two lists: inputs and outputs
          inputs: [images, anchors, bbox, class_id] where
                 images -> [batch,H,W,C] np.array representing pixel level information of each image
                 anchors->[y1,x1,y2,x2] pixel level position of the generated anchors
                 bbox->[batch,N,(dy,dx,ln(dh),ln(dw))] deltas to be applied to anchors to better fit the ground truth boxes
                 class_id->[batch, MAS_GT_INSTANCES] id corresponding to each anchor representing the closest ground truth box class_id
          outputs:
             Empty list on usual training,
        TODO: To test training on just RCNN heads create a variable detection_targets, when detection_targets is true we return a list of target class_ids and bbox_deltas, this is for testing the dataset generator without actually training the model
        """
        # print("classes:Dataset: reading images to BATCH SIZE with starting index", idx)
        #The images processed on a batch step
        image_ids = self.image_ids[idx * self.config.BATCH_SIZE: (idx + 1) * self.config.BATCH_SIZE]
        #The pixel level information for the images in the batch
        batch_imgs = np.zeros((self.config.BATCH_SIZE,
                               self.config.IMAGE_SHAPE[0],
                               self.config.IMAGE_SHAPE[1],
                               self.config.NUM_CHANNELS), dtype=np.float32)
        # Info parsed from the images
        batch_img_data = np.zeros((self.config.BATCH_SIZE, self.config.IMAGE_DATA_SIZE)) #Img id, og size and size after resizing
        #Which anchors are positive, negative and neutral
        batch_rpn_match = np.zeros((self.config.BATCH_SIZE, self.anchors.shape[0], 1))
        #Deltas for the anchors
        batch_rpn_bbox = np.zeros((self.config.BATCH_SIZE, self.config.ANCHORS_PER_IMG, 4))
        #The class for each of the positive anchors (those that match with a GT Box)
        batch_gt_class_ids = np.zeros((self.config.BATCH_SIZE, self.config.MAX_GT_INSTANCES), dtype=np.int32) # Was (batch_size, max_gt_instances, 1)
        #Coordinates of GT boxes
        batch_gt_boxes = np.zeros((self.config.BATCH_SIZE, self.config.MAX_GT_INSTANCES, 4), dtype=np.int32)

        for batch_id in range(len(image_ids)):# range(self.config.BATCH_SIZE): #For each image in the batch
            #Prepare the dataset, here we get the image, ground truth tboxes and ground truth labels of each image
            # print("classes:Dataset: get_bboxes for image ", image_ids[batch_id])
            imgToLoad = image_ids[batch_id].replace('/',' ')
            img = self.load_image(imgToLoad)
            img_data = I.compose_image_data(image_ids[batch_id], self.config.DATASET_IMAGE_SIZE,self.config.IMAGE_SHAPE)
            gt_boxes, gt_class_labs = self.get_bboxes(image_ids[batch_id],self.config.DATASET_IMAGE_SIZE)
            gt_class_ids = self.change_label_to_num(gt_class_labs, self.config.CLASS_INFO)
            # print(f"Dataset generation: gt_boxes before augmentation:\n {gt_boxes}")

            #Augment data for training
            if self.mode=="train":
                img, gt_boxes=self.augment(img, gt_boxes, self.config)
            # print(f"Dataset generation: gt_boxes after augmentation:\n {gt_boxes}")
            # if gt_boxes.shape[0] > self.config.MAX_GT_INSTANCES:
                # gt_boxes = gt_boxes[:self.config.MAX_GT_INSTANCES]

            #Generate RPN Targets
            # print("classes:Dataset: getting matches and targets from rpn_targets ")
            rpn_match, rpn_bbox = I.build_rpn_targets(self.anchors, gt_boxes, self.config)

            # Generate RCNN Targets if necessary
            if self.random_rois:
                rpn_rois = I.generate_random_rois(img.shape, self.random_rois, gt_class_ids, gt_boxes)
                if batch_id == 0:
                    batch_rpn_rois = np.zeros((self.config.BATCH_SIZE, rpn_rois.shape[0], 4), dtype=rpn_rois.dtype)
                if self.generate_detection_targets:
                    rois, rcnn_class_ids, rcnn_bbox = I.build_detection_targets(rpn_rois, gt_class_ids, gt_boxes, self.config)
                    if batch_id == 0:
                        batch_rois = np.zeros((self.config.BATCH_SIZE,) + rois.shape, dtype=rois.dtype)
                        batch_rcnn_class_ids = np.zeros((self.config.BATCH_SIZE,) + rcnn_class_ids.shape, dtype=rcnn_class_ids.dtype)
                        batch_rcnn_bbox = np.zeros((self.config.BATCH_SIZE,) + rcnn_bbox.shape, dtype=rcnn_bbox.dtype)

            # If there are more GT instances than intended, subsample
            if gt_boxes.shape[0] > self.config.MAX_GT_INSTANCES:
                sample = np.random.choice(np.arange(gt_boxes.shape[0]), self.config.MAX_GT_INSTANCES, replace=False)
                # gt_class_labs = gt_class_labs[sample]
                gt_class_ids = gt_class_ids[sample]
                gt_boxes = gt_boxes[sample]
                #Normalize boxes now so we don't have to in the classifier
                #On second thought don't so we can use these inputs to calculate the mAP
                #gt_boxes = I.norm_boxes(gt_boxes, self.config.IMAGE_SHAPE)
            # gt_class_ids = self.change_label_to_num(gt_class_labs, self.config.CLASS_INFO)

            #And add the data to the batch arrays
            batch_imgs[batch_id]=self.preprocess_image(img)
            batch_img_data[batch_id]=img_data
            batch_rpn_match[batch_id]=np.expand_dims(rpn_match, axis=1)
            batch_rpn_bbox[batch_id]=rpn_bbox
            batch_gt_class_ids[batch_id,:gt_class_ids.shape[0]]=gt_class_ids
            batch_gt_boxes[batch_id,:gt_boxes.shape[0]]=gt_boxes
            if self.random_rois:
                batch_rpn_rois[batch_id]=rpn_rois
                if self.generate_detection_targets:
                    batch_rois[batch_id] = rois
                    batch_rcnn_class_ids[batch_id] = rcnn_class_ids
                    batch_rcnn_bbox[batch_id] = rcnn_bbox

        # print(f"# FINAL INPUT SHAPES-> batch_imgs: {batch_imgs.shape}, batch_img_data.shape: {batch_img_data.shape}, batch_rpn_match.shape: {batch_rpn_match.shape}, batch_rpn_bbox: {batch_rpn_bbox.shape}, batch_gt_class_ids: {batch_gt_class_ids.shape}, batch_gt_boxes: {batch_gt_boxes.shape}")
        # print(f"-> Dataset: batch_gt_class_ids: \n {batch_gt_class_ids.shape}")
        if self.train_only_rpn:
            inputs = [batch_imgs, batch_rpn_match, batch_rpn_bbox]
            outputs = []
        else:
            inputs = [batch_imgs, batch_img_data, batch_rpn_match, batch_rpn_bbox,
                      batch_gt_class_ids, batch_gt_boxes]
            outputs = []

        # Add inputs and outputs according to the training procedure
        if self.random_rois:
            inputs.extend([batch_rpn_rois])
            if self.generate_detection_targets:
                inputs.extend([batch_rois])
                # Keras requires that output and targets have the same number of dimensions
                batch_rcnn_class_ids = np.expand_dims(
                    batch_rcnn_class_ids, -1)
                outputs.extend(
                    [batch_rcnn_class_ids, batch_rcnn_bbox])

        # print(f"# FINAL INPUT/OUTPUT LENGTH-> inputs: {len(inputs)}, outputs: {len(outputs)}")
        return inputs, outputs


    def load_image(self, _id):
        """
        load_image: Loads an image for dateset processing
        Inputs:
          _id: the id of the image to load, this is the name of the file and the folder containing it (e.g. an image in train/10234/ would be called 10243.png)
        Outputs:
          [H,W,3] np.array containing the pixel level information of the image
        """
        filename = os.path.join(self.path,_id,_id+'.png')
        # print (f"classes:Dataset: load_image {filename}")
        #load_img automatically tranforms the image into a RGB PIL image, so no need to worry about converting grayscale or alpha
        return img_to_array(load_img(filename, target_size=self.config.IMAGE_SHAPE))

    def get_bboxes(self, _id, datasetImgShape):
        """
        get_bboxes: Generate bounding boxes based on csv label parsing
        Inputs:
          _id: the ID/name of the image to be getting bounding boxes for
          datasetImgShape: (H,W) a tuple corresponding to the dimensions of the original image in the dataset
        Outputs:
          boxes: [NUM_INSTANCES,(y1,x1,y2,x2)] a numpy array with index containing the four corodinates of the ground truth boxes found in the image
          class_labels: [NUM_INSTANCES] an array containing the class labels of each graund truth box in the image (e.g. "eccentric","mature","immature)
        """
        boxes = []
        # class_labels = []
        # filename = self.path + '/' + _id + '/region_data_' + _id + '.csv'
        #Change this to os.path.join so it doesn't depend on the language and OS (if you're using Windows you should change to Linux but I'l leaving this here for safekeeping)
        filename = os.path.join(self.path,_id,'region_data_' + _id + '.csv')
        assert os.path.exists(filename), f"CSV file {filename} not found."
        #Find the ground truth box coordinates in our image coordinates
        #Remember the original image size isn't necessarily the same as the input image size for the network so we must rescale our coordinates aswell
        #img shape is (og_height, og_width) so we must use index 0 for y and index 1 for x
        x_ratio = self.config.IMAGE_SHAPE[1] / datasetImgShape[1]
        y_ratio = self.config.IMAGE_SHAPE[0] / datasetImgShape[0]
        # print("classes:Dataset: get_bboxes, parsing region data from ",filename)
        _, class_labs, x, y, w, h = I.parse_region_data(filename)
        # print("classes:Dataset: get_bboxes, pared region data from ",filename)
        for i in range(len(x)):
            yMax = round(y[i] * y_ratio)                                   # Y Coord of top left box corner
            xMin = round(x[i] * x_ratio)                                   # X coord of top left box corner
            yMin = round((y[i] + h[i]) * y_ratio)                          # Y coord of bottom right box corner
            xMax = round((x[i] + w[i]) * x_ratio)                          # X coord of bottom right corner
            boxes.append([yMax, xMin, yMin, xMax])
            # class_labels.append(class_labs)
        # print("classes:Dataset: get_bboxes, boxes array dimentions ", np.shape(boxes))
        # print("classes:Dataset: get_bboxes, class_labs array dimentions ", np.shape(class_labs))
        # Fix negative values found on the csvs
        boxes = np.array(boxes).astype('int32')
        boxes[boxes<0]=0
        boxes[boxes>self.config.IMAGE_SHAPE[0]]=self.config.IMAGE_SHAPE[0]
        return boxes, class_labs

    def preprocess_image(self, image):
        """
        preprocess_image: Normalize pixel value in an image
        Inputs:
          image, the image to be preprocessed
        Outputs:
          The image with the average pixel value subtracted
        """

        # print("classes:Dataset: preprocessing image")
        return (image.astype(np.float32) - self.config.MEAN_PIXEL ) / self.config.STD_PIXEL
        #return image.astype(np.float32)# - self.config.MEAN_PIXEL 
    # In the original mask-rpn code the masks were black and white images so this was necessary to erase outliers and make sure the maks pixels were truth to bounding boxes.-JR

    def augment(self, img, bboxes_coords, config):
        """
        This function takes the parsed data from an existing image in the  dataset and returns and augmented image and the corresponding bbox coordinates via horizontal and vertical reflection
        INPUTS: img, pixel information of loaded image (img_to_array(load_img(filename)))
        bboxes_coords: [y1,x1,y2,x2] coordinates of the bounding boxes of the original image
        config: Config class instance specifying which type of augmentation to use
        """
        aug_bboxes_coords=copy.deepcopy(bboxes_coords)
        aug_img=copy.deepcopy(img)

        if config.USE_HORIZONTAL_FLIPS and np.random.randint(0,2):#randomly select whether or not augmentation should be used
            aug_img = cv2.flip(img,1)
            for idx,bbox in enumerate(bboxes_coords):
                aug_bboxes_coords[idx][1]=config.IMAGE_SHAPE[1] - bbox[3] #width -x2
                aug_bboxes_coords[idx][3]=config.IMAGE_SHAPE[1] - bbox[1] #width -x1
            # print(f"Augmentation: Horizontal flip")

        if config.USE_VERTICAL_FLIPS and np.random.randint(0,2):
            aug_img = cv2.flip(img,0)
            for idx,bbox in enumerate(bboxes_coords):
                aug_bboxes_coords[idx][0]=config.IMAGE_SHAPE[0] - bbox[2] #height -y2
                aug_bboxes_coords[idx][2]=config.IMAGE_SHAPE[0] - bbox[0] #height -y1
            # print(f"Augmentation: Vertical flip")

        if config.USE_ROTATION_180 and np.random.randint(0,2):
            aug_img = cv2.flip(img,-1)
            for idx,bbox in enumerate(bboxes_coords):
                aug_bboxes_coords[idx][1]=config.IMAGE_SHAPE[1] - bbox[3] #width -x2
                aug_bboxes_coords[idx][3]=config.IMAGE_SHAPE[1] - bbox[1] #width -x1
                aug_bboxes_coords[idx][0]=config.IMAGE_SHAPE[0] - bbox[2] #height -y2
                aug_bboxes_coords[idx][2]=config.IMAGE_SHAPE[0] - bbox[0] #height -y1
            # print(f"Augmentation: 180 rotation")

        if config.USE_GAUSSIAN_NOISE and np.random.randint(0,2):
            mean=0
            var=100
            sigma=np.sqrt(var)
            gauss=np.random.normal(mean,sigma,(config.IMAGE_SHAPE[0],config.IMAGE_SHAPE[1],3))#that is (h,w,3) where 3 is the num channels
            gauss=gauss.reshape((config.IMAGE_SHAPE[0],config.IMAGE_SHAPE[1],3))
            aug_img=aug_img+np.uint8(gauss)
            # print(f"Augmentation: Gaussian noise")

        return aug_img, aug_bboxes_coords

    def change_label_to_num(self, classes_labels, classes_info):
        """
        Changes classes labels from string to a integer
        Inputs:
          classes_labels: List of labels (strings) to be changed to ints
          classes_info: List of dictionaries with the classes info, following the strucure [{"id":int, "name":str},...]
        Outputs:
          classes_ids: Returns a list of ids corresponding to the input labels
        """
        classes_ids=[]
        #print(f"change_label_to_num classes_labels: {classes_labels}")
        #print(f"->change_label_to_num classes_labels length: {len(classes_labels)}")
        #for i in range(len(classes_labels)):
        #    print(f"->change_label_to_num classes_labels {i} length: {len(classes_labels[i])}")
        #Create a label:id dictionary for all classes in this dataset
        classes_label_id_dict={class_["name"]:class_["id"] for class_ in classes_info}
        # print(class_label_id_dict)
        #Convert labels to ids
        for label in classes_labels:
            if label in classes_label_id_dict.keys():
                classes_ids.append(classes_label_id_dict[label])
            else:
                raise Exception(f"Label {label} not native to the dataset classes defined, please add it to the class info dict.")
        return np.array(classes_ids).astype('int32')

"""
Image class:
Class to build an image for prediction as the RCNN expects them
Inputs:
  paths: paths to read the images from
  config: config class instance with input format for the classifier, specifically batch size and input image size
Outputs:
  images: batch of images on a matrix representation to predict on
  images_data: batch of image metadata to provide to the RCNN
"""
class Image(Sequence):
    def __init__(self, paths, config):
        """
        path: path to read the image from
        """
        self.paths = paths
        print(f"classes: Image: reading image from {self.paths}")
        self.config = config
    def __len__(self):
        return int(np.ceil(len(self.paths)/self.config.BATCH_SIZE))

    def __getitem__(self,idx):
        """
        __getitem__: Overriding method to generate objects
        Inputs:
          idx: id of the image sample, this is an integer that represents the number of the image batch
        Outputs:
        Returns a python iterable from class keras.utils.Sequence. __getitem__() is overridden so that when it's called it returns two lists: inputs and outputs
          inputs: [images, images_meta] where
                 images -> [batch,H,W,C] np.array representing pixel level information of each image
                 images_meta -> [image_id, (original_shape), (image_shape),crop_id, aug_id] where image_id is the image name (a number), original_shape is the original shape in pixels and image_shape is the resized shape of the image, crop_id and aug_id are ID integers for registeing specific augmentations or crops from the image file name
        """
        #The images processed on a batch step
        image_paths = self.paths[idx * self.config.BATCH_SIZE: (idx + 1) * self.config.BATCH_SIZE]
        # Pixel level information of images in the batch
        batch_imgs = np.zeros((self.config.BATCH_SIZE,
                               self.config.IMAGE_SHAPE[0],
                               self.config.IMAGE_SHAPE[1],
                               self.config.NUM_CHANNELS), dtype=np.float32)
        # Info parsed from the images
        batch_img_data = np.zeros((self.config.BATCH_SIZE, self.config.IMAGE_DATA_SIZE)) #Img id, og size and size after resizing
        for batch_id in range(len(image_paths)):#For each image in the batch
            #Use keras' load_img to get the original width and height of the image
            img_path = image_paths[batch_id]
            img_name = img_path.split('/')[-1].split('.')[0] #Take only the number part, that is '/path/to/img/133433.png' -> '133433'
            # print(f"Loading image from {img_path}")
            w, h = load_img(img_path).size
            # print(f"Original width and heigth: {w} x {h}")
            img = self.load_image(img_path)
            img_data = I.compose_image_data(img_name, (h,w), self.config.IMAGE_SHAPE)
            #Add image and data to the batch arrays
            batch_imgs[batch_id] = self.preprocess_image(img)
            batch_img_data[batch_id] = img_data
        inputs=[batch_imgs, batch_img_data]
        return inputs

    def load_image(self, filename):
        """
        load_image: Loads an image for dateset processing
        Inputs:
          _id: the id of the image to load, this is the name of the file and the folder containing it (e.g. an image in train/10234/ would be called 10243.png)
        Outputs:
          [H,W,3] np.array containing the pixel level information of the image
        """
        # print (f"classes:Dataset: load_image {filename}")
        #load_img automatically tranforms the image into a RGB PIL image, so no need to worry about converting grayscale or alpha
        return img_to_array(load_img(filename, target_size=self.config.IMAGE_SHAPE))

    def preprocess_image(self, image):
        """
        preprocess_image: Normalize pixel value in an image
        Inputs:
          image, the image to be preprocessed
        Outputs:
          The image with the average pixel value subtracted
        """
        # print("classes:Dataset: preprocessing image")
        return image.astype(np.float32)# - self.config.MEAN_PIXEL


#Test config implementation
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--backbone", help="Backbone to use for prediction, options are \'temnet\', \'resnet101\' or \'resnet101v2\', mind weights are different for each model", default='temnet')
    args = parser.parse_args()
    config = Config(backbone=args.backbone)
    config.display()
    dataset=Dataset(config.TRAIN_PATH, config, "train")
    for i in range(len(dataset)):# Every batch
        inputs=dataset[i][0]
        images_gt=inputs[0] # A batch of images
        images_data=inputs[1]
        imgNames = I.build_image_names(images_data)
        for j in range(len(images_gt)):#Every image in a batch
            image_gt = np.uint8(images_gt[j])
            # img_data = inputs[1][j]
            # imgName = I.build_image_name(img_data)
            imgName = imgNames[j]
            rpn_match = inputs[2][j]
            rpn_bbox = inputs[3][j]
            gt_class_ids = inputs[4][j]
            print(f"gt_class_ids from dataset {imgName}: {gt_class_ids}")
            gt_boxes = inputs[5][j]
            print(f"gt_boxes from dataset {imgName}: {gt_boxes}")
        # print(f"dataset 0:\n {dataset[0]}")
        # print(f"dataset outputs:\n {dataset[0][1]}")
        # print(f"dataset inputs img_data:\n {dataset[0][0][1]}")
        # print(f"dataset inputs img_classes:\n {dataset[0][0][4]}")
