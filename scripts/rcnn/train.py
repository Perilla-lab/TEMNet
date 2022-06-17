import os, time, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #Reduce Tensorflow messages
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import tensorflow as tf
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
def train_model(backbone='temnet', weights_path=None, n_gpu='0'):
    with(tf.device('/GPU:'+n_gpu)):
    #with mirrored_strategy.scope():
        config = Config(backbone=backbone)
        print(f"Training for RPN: {config.TRAIN_ONLY_RPN}")
        dataset = {"train": Dataset(config.TRAIN_PATH, config, "train"), "validation": Dataset(config.VAL_PATH, config, "validation")}
        rcnn = RCNN(config, 'train')
        if config.TRAIN_ONLY_RPN:
            print("--------------------Training RPN model ------------------")
        else:
            print("--------------------Training RCNN model ------------------")
        if weights_path != None:
            print(f"Reading weights from {weights_path} ...")
            try:
                rcnn.load_weights(weights_path, by_name=True)
            except:
                print(f"Could not load weights, resorting back to imagenet pretrained weights ...")
                weights_path = rcnn.get_imagenet_weights(backbone=backbone)
                rcnn.load_weights(weights_path, by_name=True)
        else:
            print("No weights loaded.")
        hist, lrm = rcnn.train(dataset)
        print("--------------------RCNN model trained---------------------")
        #Save losses on a file in case automated visualization fails
        with open(f'RCNN_BENCHMARKS_losses_{backbone}.csv', 'w') as lossfile:
            for tl,vl in zip(hist.history['loss'], hist.history['val_loss']):
                lossfile.write("{},{}".format(tl, vl))
        with open(f'RCNN_BENCHMARKS_lr_{backbone}.csv', 'w') as lrfile:
            for i,lr in enumerate(lrm.lrates):
                lrfile.write("{},{}".format(i, lr))
        V.visualize_benchmarks(hist.history['loss'], hist.history['val_loss'], config)
        V.visualize_learning_rate(lrm.lrates, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--backbone", help="Backbone to train, options are \'temnet\', \'resnet101\' or \'resnet101v2\', mind weights are different for each model", default='temnet')
    parser.add_argument("-g", "--gpu", help="Number of the GPU to use for training", default='0')
    parser.add_argument("-w", "--weights", help="Path to starting weights to use for training", default=None)
    args = parser.parse_args()
    start=time.perf_counter()
    train_model(args.backbone, args.weights, args.gpu)
    finish=time.perf_counter()
    print(f"Finished in {round(finish-start,2)} seconds")
