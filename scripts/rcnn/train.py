import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #Reduce Tensorflow messages
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import tensorflow as tf
import time
import visualize as V
from model import RCNN
from config import Config, Dataset

# ********* SETUP *********
print("TensorFlow Version", tf.__version__)
#tf.debugging.set_log_device_placement(True) #Enable for device debugging
mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice(reduce_to_device="cpu:0"))
print('Number of devices recognized by Mirror Strategy: ', mirrored_strategy.num_replicas_in_sync)

"""
compound_training: Run the training procedures multiple times consecutively
Inputs:
    model, the training Keras sequential models
    dataset, the dataset built by Dataset object
    iterations, the number of overall training procedures
"""
def compound_training(model, dataset, iterations):
    train_losses = []
    val_losses = []
    for i in range((iterations)):
        if(i != 0):
            model.keras_model.load_weights(model.config.WEIGHT_SET, by_name=True)
        print("Compound weights loaded!")
        hist, lrm = model.train(dataset)
        train_losses.append(hist.history['loss'])
        val_losses.append(hist.history['val_loss'])
    V.visualize_compound_training(train_losses, val_losses)

"""
train_model: Run a single training procedure for RPN model
"""
def train_model():
    with(tf.device('/GPU:0')):
    #with mirrored_strategy.scope():
        config = Config()
        print(f"Training for RPN: {config.TRAIN_ONLY_RPN}")
        dataset = {"train": Dataset(config.TRAIN_PATH, config, "train"), "validation": Dataset(config.VAL_PATH, config, "validation")}
        rcnn = RCNN(config, 'train')
        # rcnn.keras_model.load_weights("/scratch/07655/jsreyl/hivclass/checkpoints/resnet101_weights_tf.h5", by_name=True)
        # resnet_weights_path = rcnn.get_imagenet_weights()
        if config.TRAIN_ONLY_RPN:
            print("--------------------Training RPN model ------------------")
            if "resnet" in config.BACKBONE:
                weights_path = '/scratch/07655/jsreyl/hivclass/models/pretrained/resnet101_weights.h5'
            elif config.BACKBONE == "novel":
                # weights_path = '/scratch/07655/jsreyl/hivclass/models/pretrained/novel-weights-95_acc8564.hdf5' #Use this if you're not using BN or GN
                weights_path = '/scratch/07655/jsreyl/hivclass/models/pretrained/novel-weights-64_acc8785_bn.hdf5' #Use this if you're using BN
                # weights_path = '/scratch/07655/jsreyl/hivclass/models/pretrained/novel-weights-23_acc8619_gn.hdf5' #Use this if you're using GN
        else:
            print("--------------------Training RCNN model ------------------")
            if "resnet" in config.BACKBONE:
                #resnet_weights_path = tf.train.latest_checkpoint('/scratch/07655/jsreyl/hivclass/checkpoints/rcnn/')
                weights_path = '/scratch/07655/jsreyl/hivclass/checkpoints/rcnn/rcnn_weights.50.hdf5'
                # weights_path = config.WEIGHT_SET
            elif config.BACKBONE == "novel":
                #weights_path = '/scratch/07655/jsreyl/hivclass/checkpoints/rcnn/saved_weights/rpn_novel_weights.99_aug_bn.hdf5'
                weights_path = '/scratch/07655/jsreyl/hivclass/checkpoints/rcnn/saved_weights/rpn_novel_weights.50_aug_gn.hdf5'
                # weights_path = '/scratch/07655/jsreyl/hivclass/models/pretrained/novel-weights-23_acc8619_gn.hdf5' #Use this if you're using GN
        print(f"Reading weights from {weights_path} ...")
        rcnn.load_weights(weights_path, by_name=True)
        hist, lrm = rcnn.train(dataset)
        print("--------------------RCNN model trained---------------------")
        V.visualize_benchmarks(hist.history['loss'], hist.history['val_loss'], config)
        V.visualize_learning_rate(lrm.lrates, config)

if __name__ == "__main__":
    start=time.perf_counter()
    train_model()
    finish=time.perf_counter()
    print(f"Finished in {round(finish-start,2)} seconds")
