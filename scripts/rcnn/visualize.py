from __future__ import absolute_import, division, print_function, unicode_literals
import os, datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import input_pipeline as I
from config import Config, Dataset
from keras.preprocessing.image import img_to_array, load_img

# Defining global path prefix so we stop saving stuff on /home/ and actually use scratch for its intended purpose
PATH_PREFIX='/scratch/07655/jsreyl/hivclass/'

"""
visualize_rpn_predictions: Self explanatory
Inputs:
    image, the image to be display
    rpn_match, array with positivity/negativity of given anchor
    rpn_bbox, array of bounding box shifts
    anchors, array of anchors given in the form of box coords
    top_n, the total number of anchors to visualize
Outputs:
    None
"""
def visualize_rpn_predictions(image, sorted_anchors, imgName):
    x = datetime.datetime.now()
    date = x.strftime("%m")+"_"+x.strftime("%d") + "_" + x.strftime("%I") + "_" + x.strftime("%M")
    fig, axes = plt.subplots(ncols=1, figsize=(20, 13))
    axes.imshow(image)
    axes.set_title("Top 100 Region Proposal Network predictions")
    for i in sorted_anchors:
        #remember anchor coordinates are [y1,x1,y2,x2] where (x1,y1) corresponds to the upper left corner and (0,0) is the upper left of the dataset image
        #since patches.Rectangle receives the lower left corner we need to enter (x1,y2) and calculate the height and width accordingly
        rect = patches.Rectangle((i[1], i[2]), i[3] - i[1], i[0] - i[2], linewidth=1, edgecolor='r', facecolor='none', linestyle='-')
        axes.add_patch(rect)
    fig.savefig(PATH_PREFIX+"graphs/rcnn/RCNN_PREDS_" + imgName+ "_" + date + ".png", bbox_inches = 'tight', pad_inches = 0.5)

"""
visualize_training_anchors: Visualize the proposed negative and positive anchors while training RPN
Inputs:
    anchors, array of anchors with bounding box coords
    rpn_match, array inidicating positivity or negativity of given anchor
    image, the image to display
Outputs:
    None
"""
def visualize_training_anchors(positive_anchors, negative_anchors, image, imgName):
    x = datetime.datetime.now()
    date = x.strftime("%m")+"_"+x.strftime("%d") + "_" + x.strftime("%I") + "_" + x.strftime("%M")
    fig, axes = plt.subplots(ncols=2, figsize = (20, 10))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Positive Anchors : {}".format(len(positive_anchors)))
    axes[1].imshow(image, cmap='gray')
    axes[1].set_title("Negative Anchors : {}".format(len(negative_anchors)))
    # Positive anchors
    for i in positive_anchors:
        #rect = patches.Rectangle((i[3], i[0]), i[1]-i[3], i[2]-i[0], linewidth=1.5, edgecolor='b', facecolor='none', linestyle=':')
        #Changing the coordinates for correct visualization, no idea why it was like this here and not like the coordinates used in visualize bboxes
        #anchors coordinates are [y1,x1,y2,x2] with (x1,y1) the upper left corner
        #format should be (x1,y2), width=x2-x1,height=y1-y2
        rect = patches.Rectangle((i[1], i[2]), i[3]-i[1], i[0]-i[2], linewidth=1.5, edgecolor='b', facecolor='none', linestyle=':')
        axes[0].add_patch(rect)
    # Negative anchors
    for h in negative_anchors:
        #rect = patches.Rectangle((h[3], h[0]), h[1]-h[3], h[2]-h[0], linewidth=1.5, edgecolor='r', facecolor='none', linestyle=':')
        rect = patches.Rectangle((h[1], h[2]), h[3]-h[1], h[0]-h[2], linewidth=1.5, edgecolor='r', facecolor='none', linestyle=':')
        axes[1].add_patch(rect)
    fig.savefig(PATH_PREFIX+"graphs/rcnn/ANCHORS_" + imgName + "_" + date + "_.png", bbox_inches = 'tight', pad_inches = 0.5)

"""
visualize_bboxes: Visualize bounding boxes in an image to be saved
Inputs: 
    image, the image to be displayed
    bboxes, array of bounding box coordinates
Outputs:
    None
"""
def visualize_bboxes(image, bboxes, imgName):
    x = datetime.datetime.now()
    date = x.strftime("%m")+"_"+x.strftime("%d") + "_" + x.strftime("%I") + "_" + x.strftime("%M")
    fig, axes = plt.subplots()
    plt.axis('off')
    axes.imshow(image)
    axes.set_title("Bounding Boxes")
    for i in bboxes:
        rect = patches.Rectangle((i[1], i[2]), i[3]-i[1], i[0]-i[2], linewidth=1, edgecolor='r', facecolor='none', linestyle='-')
        axes.add_patch(rect)
    fig.savefig(PATH_PREFIX+"graphs/rcnn/BBOXES_" + imgName + "_" + date + "_.png", bbox_inches = 'tight', pad_inches = 0.5)

"""
visualize_benchmarks: Visualize loss and training time incurred per epoch
Inputs: 
    train_loss, a list of loss values
    val_loss, a list of validation loss values
Ouputs:
    None
"""
def visualize_benchmarks(train_loss, val_loss, config):
    x = datetime.datetime.now()
    date = x.strftime("%m")+"_"+x.strftime("%d") + "_" + x.strftime("%I") + "_" + x.strftime("%M")
    fig, axes = plt.subplots(nrows=2, figsize=(15, 10))
    fig.tight_layout(pad=5.0)

    train_loss.insert(0, 0)
    val_loss.insert(0, 0)

    maxVals = [max(train_loss), max(val_loss)]
    maxVal = max(maxVals) * 1.1
   
    axes[0].set_title("Loss Incurred Per Epoch")
    axes[0].grid()
    axes[0].set(xlabel='Epoch', ylabel='Loss')
    axes[0].axis([1, config.EPOCHS, 0, maxVal])
    axes[0].set_xticks(np.arange(1,config.EPOCHS))

    axes[1].set_title("Training Time Per Epoch (Seconds)")
    axes[1].grid()
    axes[1].set(xlabel='Epoch', ylabel='Time (s)')
    axes[1].axis([1, config.EPOCHS, 0, 35])
    axes[1].set_xticks(np.arange(1, config.EPOCHS))

    times = [0, 19, 16, 14, 15, 14, 16, 15, 15, 15, 14] #This is hardoced since the old RPN implementation Hagan had, would be cool to actually measure the time spent on each epoch from the code

    axes[1].plot(times, marker='o')
    axes[0].plot(train_loss, marker='o')
    axes[0].plot(val_loss, marker='o')
    axes[0].legend(['Training Loss', 'Validation Loss'], loc='upper right')
    lossfile_path = PATH_PREFIX+"graphs/rcnn/RCNN_OUR_BENCHMARKS_"+config.BACKBONE+"_losses.txt"
    with open(lossfile_path, 'w') as lossfile:
        for i in range(len(train_loss)):
            lossfile.write("{} {} {}".format(i, train_loss[i], val_loss[i]))
    fig.savefig(PATH_PREFIX+"graphs/rcnn/RCNN_OUR_BENCHMARKS_"+config.BACKBONE+"_"+date)

"""
visualize_learning_rate: Visualize learning rate progression across the training
Inputs:
    lr, a list of learning rates
    config, an instance of the Config class containing the number of epochs for training
Ouputs:
    None
"""
def visualize_learning_rate(lr, config):
    x = datetime.datetime.now()
    date = x.strftime("%m")+"_"+x.strftime("%d") + "_" + x.strftime("%I") + "_" + x.strftime("%M")
    fig, axes = plt.subplots( figsize=(15, 10))
    fig.tight_layout(pad=5.0)

    maxVal = max(lr) * 1.1
   
    axes.set_title("Learning rate Per Epoch")
    axes.grid()
    axes.set(xlabel='Epoch', ylabel='Loss')
    axes.axis([1, config.EPOCHS, 0, maxVal])
    axes.set_xticks(np.arange(1,config.EPOCHS))
 
    axes.plot(lr, marker='o')
    axes.legend(['Learning rate'], loc='upper right')
    lossfile_path = PATH_PREFIX+"graphs/rcnn/RCNN_OUR_BENCHMARKS_"+config.BACKBONE+"_lr.txt"
    with open(lossfile_path, 'w') as lossfile:
        for i in range(len(lr)):
            lossfile.write("{} {}".format(i, lr[i]))
    fig.savefig(PATH_PREFIX+"graphs/rcnn/RCNN_OUR_BENCHMARKS_lr_"+date)


"""
visualize_parsed_bboxes: Visualize the directly parsed bounding boxing information from CSV files to ensure accuracy
Inputs:
    None
Outputs:
    None
"""
def visualize_parsed_bboxes(train_path, val_path): 
    #VAL_PATH = '/scratch/07049/tg863484/imgs/rpn/train/'
    #TRAIN_PATH = '/scratch/07049/tg863484/imgs/rpn/val/'
    image_ids_val = next(os.walk(val_path))[1] 
    image_ids_train = next(os.walk(train_path))[1]

    for i in range(len(image_ids_val)):
        _id, lab, x, y, w, h = I.parse_region_data(val_path +'/'+ image_ids_val[i] + '/region_data_' + image_ids_val[i] + '.csv')
        fig, axes = plt.subplots(dpi = 100, figsize=(20, 13))
        plt.axis('off')
        imgData = np.uint8(img_to_array(load_img(val_path +'/'+ image_ids_val[i] + '/' + image_ids_val[i] + '.png')))
        axes.imshow(imgData)
        boxcount = 0
        for n, h, z, t, caption in zip(x, y, w, h, lab):
            axes.text(n, h, '{}'.format(caption), fontsize=15, color='w', backgroundcolor='none')
            if caption=='mature':
                color = 'r'
            elif caption=='immature':
                color='y'
            else:
                color ='g'
            rect = patches.Rectangle((n, h), z, t, linewidth=3, alpha=0.7, linestyle='dashed',edgecolor=color,facecolor='none')
            axes.add_patch(rect)
            boxcount += 1
        fig.savefig(PATH_PREFIX+"graphs/rcnn/BBOXES_VIS_" + image_ids_val[i] + ".png", bbox_inches = 'tight', pad_inches = 0.5)
        plt.close(fig)

    for i in range(len(image_ids_train)):
        _id, lab, x, y, w, h = I.parse_region_data(train_path +'/'+ image_ids_train[i] + '/region_data_' + image_ids_train[i] + '.csv')
        fig, axes = plt.subplots(dpi = 100, figsize=(20, 13))
        plt.axis('off')
        imgData = np.uint8(img_to_array(load_img(train_path +'/'+ image_ids_train[i] + '/' + image_ids_train[i] + '.png')))
        axes.imshow(imgData)
        boxcount = 0
        for n, h, z, t in zip(x, y, w, h):
            axes.text(n, h, '({}, {}), box #{}'.format(n, h, boxcount), fontsize=10)
            rect = patches.Rectangle((n, h), z, t, linewidth=1.5, edgecolor='r',facecolor='none')
            axes.add_patch(rect)
            boxcount += 1
        fig.savefig(PATH_PREFIX+"graphs/rcnn/BBOXES_VIS_nolabel_" + image_ids_train[i] + ".png", bbox_inches = 'tight', pad_inches = 0.5)
        plt.close(fig)

"""
visualize_compound_training: Visualizing the various effects of compound training procedures
Inputs:
    train_losses, a list of loss values
    val_losses, a list of loss values
Output:
    None
"""
def visualize_compound_training(train_losses, val_losses):
    fig, axes = plt.subplots(2, 2, figsize = (20, 20))

    # Training Loss/Epoch
    axes[0, 0].set_title("Training Loss Per Epoch")
    axes[0, 0].grid()
    axes[0, 0].set(xlabel='Epoch', ylabel='Loss')
    axes[0, 0].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # Validation Loss/Epoch
    axes[0, 1].set_title("Validation Loss Per Epoch")
    axes[0, 1].grid()
    axes[0, 1].set(xlabel='Epoch', ylabel='Loss')
    axes[0, 1].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # Avg Train Loss
    axes[1, 0].set_title("Average Training Loss Per Iteration")
    axes[1, 0].grid()
    axes[1, 0].set(xlabel='Iteration', ylabel='Average Loss')
    axes[1, 0].set_xticks([1, 2, 3])

    # Avg Val Loss
    axes[1, 1].set_title("Average Validation Loss Per Iteration")
    axes[1, 1].grid()
    axes[1, 1].set(xlabel='Iteration', ylabel='Average Loss')

    # Build Avg/Stddev arrays
    train_aves = []
    train_stdvs = []
    val_aves = []
    val_stdvs = []
    vals = []
    for i in range(len(train_losses)):
        train_aves.append(sum(train_losses[i])/len(train_losses[i]))
        train_stdvs.append(np.std(train_losses[i]))
        val_aves.append(sum(val_losses[i])/len(val_losses[i]))
        val_stdvs.append(np.std(val_losses[i]))
        vals.append(i+1)

    axes[1, 0].set_xticks(vals)
    axes[1, 1].set_xticks(vals)
    axes[1, 0].set_ylim(bottom=(min(train_aves)-2), top=(max(train_aves) + 2))
    axes[1, 1].set_ylim(bottom=(min(train_aves)-2), top=(max(val_aves) + 2))
    axes[1, 0].errorbar(vals, train_aves, yerr=train_stdvs, marker='^', capsize=5, linewidth=2.5, elinewidth=1.5)
    axes[1, 1].errorbar(vals, val_aves, yerr=val_stdvs, marker='^', capsize=5, linewidth=2.5, elinewidth=1.5)

    runs = []
    for i in range(len(train_losses)):
        #train_losses[i].insert(0, 0)
        #val_losses[i].insert(0, 0)
        axes[0, 0].plot(train_losses[i], marker='o')
        axes[0, 1].plot(val_losses[i], marker = 'o')
        runs.append('Run #' + str(i))

    axes[0, 0].legend(runs, loc='upper right')
    axes[0, 1].legend(runs, loc='upper right')
    fig.savefig(PATH_PREFIX+"graphs/rcnn/compound.png", bbox_inches = 'tight', pad_inches = 0.5)

def visualize_rcnn_predictions(image, boxes, class_ids, scores, imgName):
    """
    Saves an image of the rcnn predictions displaying the boxes along with their classes and scores
    Inputs:
      image: source image to display
      boxes: array of boxes coordinates to display on the image
      class_ids: array of predicted classes for the boxes
      scores: array of predicted probabilities for the box classification
      imgName: name of the image
    Outputs:
      None, uses matplotlib to save an image
    """
    # print(f"Image: {image}")
    # print(f"Boxes: {boxes}")
    # print(f"Class_ids: {class_ids}")
    # print(f"Scores: {scores}")
    print(f"ImgName: {imgName}")
    if boxes.shape[0]==0:
        print(f"##### NO INSTANCES TO DISPLAY FOR IMAGE {imgName}")
    else:
        assert boxes.shape[0]==class_ids.shape[0]==scores.shape[0], f"Number of boxes {boxes.shape[0]} and classes {class_ids.shape[0]} or scores {scores.shape[0]} don't match"
    time = datetime.datetime.now()
    date = time.strftime("%m")+"_"+time.strftime("%d") + "_" + time.strftime("%I") + "_" + time.strftime("%M")
    fig, axes = plt.subplots()
    #plt.tight_layout()
    plt.axis('off')
    axes.imshow(image)
    axes.set_title("RCNN predictions")
    print("Source image displayed without errors")
    print(f"Boxes length: {boxes.shape}")
    for i, box, class_id, score in zip(range(boxes.shape[0]), boxes, class_ids, scores):
        # Draw the boxes
        y1, x1, y2, x2 = box
        if class_id == 1:
            label = 'e'
            color = 'g'
        elif class_id == 2:
            label = 'm'
            color = 'orange'
        elif class_id == 3:
            label = 'i'
            color = 'b'
        caption = "{} : {:.2f}".format(label, score) if score else label
        axes.text(x1, y1+8, caption, color='b', size=8, backgroundcolor='none')
        rect = patches.Rectangle((x1, y2), x2-x1, y1-y2, linewidth=2, edgecolor=color, facecolor='none', linestyle='-')
        axes.add_patch(rect)
    fig.savefig(PATH_PREFIX+"graphs/rcnn/RCNN_PREDS_" + imgName + "_" + date + "_.png", bbox_inches = 'tight', pad_inches = 0.5)
    plt.close()


def visualize_predictions_count(class_ids, scores, imgName):
    """
    Saves an bar plot of the number of particles according to their classes by the rcnn predictions
    Inputs:
      class_ids: array of predicted classes for the boxes
      scores: array of predicted probabilities for the box classification
      imgName: name of the image
    Outputs:
      None, uses matplotlib to save an image
    """
    time = datetime.datetime.now()
    date = time.strftime("%m")+"_"+time.strftime("%d") + "_" + time.strftime("%I") + "_" + time.strftime("%M")
    fig, ax = plt.subplots()
    #Generate arrays for counting the number of particles
    classes = ['eccentric','mature','immature']
    class_counts = np.zeros(3) #Start at zero
    present_class_id, counts = np.unique(class_ids, return_counts=True) #And count those present
    if len(counts) != 0:
        class_counts[present_class_id-1]=counts #Since the class ids are 1,2,3 and the counts have indices 0,1,2 we have to subtract 1
    print(f"Class counts on image {imgName}: {class_counts}")
    # Keep track of the mean score for each class to use as labels
    lab_scores = np.zeros(3)
    for i in present_class_id:
        lab_scores[i-1]=np.mean(scores[np.where(class_ids==i)])
    #Now use them for a plot bar
    rects1 = ax.bar(np.arange(len(classes)), class_counts, width = 0.8)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_ylabel('Prediction counts')
    ax.set_title(f'Prediction counts for image {imgName}')
    def autolabel(rects, labels):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for lab, rect in zip(labels, rects):
            height = rect.get_height()
            ax.annotate('{:.2f}'.format(lab),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1, class_counts)

    fig.savefig(PATH_PREFIX+"graphs/rcnn/RCNN_COUNTS_" + imgName + "_" + date + ".png", bbox_inches = 'tight', pad_inches = 0.5)
    plt.close()

def visualize_dataset(dataset):
    """
    Generates images of the bounding boxes and classes used as input for training
    Inputs:
      dataset: Dataset class instance
    Outputs:
      None, uses matplotlib to save images of each of the images in the dataset
    """
    for i in range(len(dataset)):# Every batch
        inputs=dataset[i][0]
        images_gt=inputs[0] # A batch of images
        images_data=inputs[1]
        imgNames = I.build_image_names(images_data)
        for j in range(len(images_gt)):#Every image in a batch
            image_gt = np.uint8(images_gt[j])
            #img_data = inputs[1][j]
            #imgName = str(int(img_data[0]))
            #imgName = I.build_image_name(img_data)
            imgName = imgNames[j]
            rpn_match = inputs[2][j]
            rpn_bbox = inputs[3][j]
            gt_class_ids = inputs[4][j]
            gt_boxes = inputs[5][j]
            scores = np.ones(len(gt_class_ids))
            fig, axes = plt.subplots()
            plt.tight_layout()
            plt.axis('off')
            axes.imshow(image_gt)
            axes.set_title("RCNN inputs")
            for i, box, class_id, score in zip(range(gt_boxes.shape[0]), gt_boxes, gt_class_ids, scores):
                # Draw the boxes
                y1, x1, y2, x2 = box
                #If boxes are empty, this is just so we don't confuse the zero values in gt_boxes to make the array fit MAX_GT_INSTANCES
                label = 'empty'
                color = 'y'
                if class_id == 1:
                    label = 'eccentric'
                    color = 'g'
                elif class_id == 2:
                    label = 'mature'
                    color = 'orange'
                elif class_id == 3:
                    label = 'immature'
                    color = 'b'
                caption = "{} : {:.3f}".format(label, score) if score else label
                axes.text(x1, y1+8, caption, color='w', size=11, backgroundcolor='none')
                rect = patches.Rectangle((x1, y2), x2-x1, y1-y2, linewidth=2, edgecolor=color, facecolor='none', linestyle='-')
                axes.add_patch(rect)
            fig.savefig(PATH_PREFIX+"graphs/rcnn/RCNN_INPUTS_" + imgName +".png", bbox_inches = 'tight', pad_inches = 0.5)
            plt.close()

def visualize_score_histograms(_class_ids, _scores, imgName, nbins=10):
    """
    visualize_score_histograms : Visualize the statistic distribution of scores by class ids for given predictions
    Inputs:
      scores: np array of scores for predictions
      class_ids: np array of the class ids for the specific scores
    Outputs:
      None, saves an image with the different histograms
    """
    if len(_class_ids) == 0:
        class_ids = [0.]
        scores = [0.]
    else:
        class_ids = np.array(_class_ids)
        scores = np.array(_scores)
    time = datetime.datetime.now()
    date = time.strftime("%m")+"_"+time.strftime("%d") + "_" + time.strftime("%I") + "_" + time.strftime("%M")
    fig, (ax0,ax1,ax2,ax3) = plt.subplots(1,4, figsize=(24,6))
    #Scores for all particles
    mu = np.mean(scores)
    sigma = np.std(scores)
    n,bins,patches=ax0.hist(scores,bins=nbins, alpha=0.5,rwidth=0.85, label=rf"$\mu: {round(mu,2)}, \sigma$: {round(sigma,2)}")
    ax0.axvline(mu,ls='--')
    x=[0.5*(bins[i+1]+bins[i]) for i in range(len(bins)-1)]
    ax0.plot(x,max(n)*np.exp(-0.5*((x-mu)/sigma)**2),ls='--')
    ax0.set_ylabel('Counts')
    ax0.set_xlabel('Score')
    ax0.set_title('All')
    ax0.legend(loc='upper right')

    #Scores per class
    #Eccentric
    e_scores = scores[class_ids == 1]
    mu = np.mean(e_scores)
    sigma = np.std(e_scores)
    n,bins,patches=ax1.hist(e_scores,bins=nbins, alpha=0.5,rwidth=0.85, label=rf"$\mu: {round(mu,2)}, \sigma$: {round(sigma,2)}")
    ax1.axvline(mu,ls='--')
    x=[0.5*(bins[i+1]+bins[i]) for i in range(len(bins)-1)]
    ax1.plot(x,max(n)*np.exp(-0.5*((x-mu)/sigma)**2),ls='--')
    ax1.set_xlabel('Score')
    ax1.set_title('Eccentric')
    ax1.legend(loc='upper right')
    #Mature
    m_scores = scores[class_ids == 2]
    mu = np.mean(m_scores)
    sigma = np.std(m_scores)
    n,bins,patches=ax2.hist(m_scores,bins=nbins, alpha=0.5,rwidth=0.85, label=rf"$\mu: {round(mu,2)}, \sigma$: {round(sigma,2)}")
    ax2.axvline(mu,ls='--')
    x=[0.5*(bins[i+1]+bins[i]) for i in range(len(bins)-1)]
    ax2.plot(x,max(n)*np.exp(-0.5*((x-mu)/sigma)**2),ls='--')
    ax2.set_xlabel('Score')
    ax2.set_title('Mature')
    ax2.legend(loc='upper right')
    #Immature
    i_scores = scores[class_ids == 3]
    mu = np.mean(i_scores)
    sigma = np.std(i_scores)
    n,bins,patches=ax3.hist(i_scores,bins=nbins, alpha=0.5,rwidth=0.85, label=rf"$\mu: {round(mu,2)}, \sigma$: {round(sigma,2)}")
    ax3.axvline(mu,ls='--')
    x=[0.5*(bins[i+1]+bins[i]) for i in range(len(bins)-1)]
    ax3.plot(x,max(n)*np.exp(-0.5*((x-mu)/sigma)**2),ls='--')
    ax3.set_xlabel('Score')
    ax3.set_title('Immature')
    ax3.legend(loc='upper right')

    fig.savefig(PATH_PREFIX+"graphs/rcnn/RCNN_STATS_" + imgName + "_"+ date + ".png", bbox_inches = 'tight', pad_inches = 0.5)
    plt.close()

if __name__ == "__main__":
    print("VISUALIZE")
    config = Config()
    # dataset = Dataset(config.TRAIN_PATH, config, "train")
    # visualize_dataset(dataset)
    config.TRAIN_PATH = '/scratch/07655/jsreyl/imgs/rcnn_dataset_full/train'
    config.VAL_PATH = '/scratch/07655/jsreyl/imgs/rcnn_dataset_full/val'
    train_ids = next(os.walk(config.TRAIN_PATH))[1]
    val_ids = next(os.walk(config.VAL_PATH))[1]
    image_paths = [os.path.join(config.TRAIN_PATH, img_name, img_name+'.png') for img_name in train_ids]
    csv_paths = [os.path.join(config.TRAIN_PATH, img_name, 'region_data_'+img_name+'.csv') for img_name in train_ids]
    image_paths += [os.path.join(config.VAL_PATH, img_name, img_name+'.png') for img_name in val_ids]
    csv_paths += [os.path.join(config.VAL_PATH, img_name, 'region_data_'+img_name+'.csv') for img_name in val_ids]
    #Now sequentially read the data of the images
    full_class_ids = []
    for image_path, csv_path in zip(image_paths, csv_paths):
        img_name = image_path.split('/')[-1].split('.')[0]
        print(f"Loading image {img_name} from path {csv_path}")
        _, lab, x, y, w, h = I.parse_region_data(csv_path)
        image = np.uint8(img_to_array(load_img(image_path)))
        #Format xywh to (y1,x1,y2,x2)
        #Format labs to class_ids
        boxes = []
        class_ids = np.zeros(len(lab)).astype('int32')
        for i in range(len(x)):
            yMax = y[i]
            xMin = x[i]
            yMin = y[i]+h[i]
            xMax = x[i]+w[i]
            if lab[i] == 'eccentric':
                class_ids[i] = 1
            elif lab[i] == 'mature':
                class_ids[i] = 2
            elif lab[i] == 'immature':
                class_ids[i] = 3
            boxes.append([yMax,xMin,yMin,xMax])
        boxes = np.array(boxes)
        visualize_rcnn_predictions(image, boxes, class_ids, np.ones(len(class_ids)), img_name)
        visualize_predictions_count(class_ids, np.ones(len(class_ids)), img_name)
        full_class_ids+=list(class_ids)
    visualize_predictions_count(np.array(full_class_ids), np.ones(len(full_class_ids)), 'Validation_counts')
    # inputs = dataset[0][0]
    # image_gt = inputs[0][0]
    # image_meta = inputs[1][0]
    # print(f"image_meta: {image_meta[0]}")
    # gt_class_ids = inputs[4][0]
    # print(f"gt_class_ids: {gt_class_ids[:10]}")
    # gt_boxes = inputs[5][0]
    # print(f"gt_boxes:\n {gt_boxes[:10]}")
    
    # visualize_rcnn_predictions(np.uint8(image_gt), gt_boxes, gt_class_ids, np.ones(len(gt_class_ids)), str(int(image_meta[0])))
    # print(f"image_gt shape: {image_gt.shape[0:2]}")
    # norm_gt_boxes = I.norm_boxes(gt_boxes, config.IMAGE_SHAPE)
    # print(f"norm_gt_boxes:\n {norm_gt_boxes[:10]}")
    # denorm_gt_boxes = I.denorm_boxes(norm_gt_boxes, config.IMAGE_SHAPE)
    # print(f"denorm_gt_boxes:\n {denorm_gt_boxes[:10]}")
    # visualize_rcnn_predictions(np.uint8(image_gt), norm_gt_boxes, gt_class_ids, np.ones(len(gt_class_ids)), str(int(image_meta[0])))
    # visualize_parsed_bboxes(config.TRAIN_PATH,config.VAL_PATH)
