import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
import tensorflow.keras.optimizers as KO

import database as D
import models as M
import input_pipeline as I
import graphing as G
from classes import Config, PerillaNet

class VGG19(object):
    def get_imagenet_weights(self):
        """Downloads ImageNet trained weights from Keras.
        Returns path to weights file.
        """
        from keras.utils.data_utils import get_file
        TF_WEIGHTS_PATH_NO_TOP = ('https://storage.googleapis.com/tensorflow/'
                                  'keras-applications/vgg19/'
                                  'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
        weights_path = get_file(
            'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
            TF_WEIGHTS_PATH_NO_TOP,
            cache_subdir='models',
            file_hash='253f8cb515780f3b799900260a226db6')
        return weights_path

    def load_weights(self, filepath, model, by_name=True, exclude=None):
        """Modified version of the corresponding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exclude: list of layer names to exclude
        """
        import h5py
        from tensorflow.python.keras.saving import hdf5_format

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        with h5py.File(filepath, mode='r') as f:
            if 'layer_names' not in f.attrs and 'model_weights' in f:
                f = f['model_weights']

            # In multi-GPU training, we wrap the model. Get layers
            # of the inner model because they have the weights.
            layers = model.inner_model.layers if hasattr(model, "inner_model")\
                else model.layers

            # Exclude some layers
            if exclude:
                layers = filter(lambda l: l.name not in exclude, layers)

            if by_name:
                hdf5_format.load_weights_from_hdf5_group_by_name(f, layers)
            else:
                hdf5_format.load_weights_from_hdf5_group(f, layers)

    def build_backbone(self, input_tensor, architecture, num_classes, stage5=False, train_bn=None):
        """Build a VGG model.

        Arguments
        ----------
        input_tensor: Keras Input layer
            Tensor for image input
        architecture: str, "resnet50" or "resnet101"
            Architecture to use
        stage5: bool
            If False, stage5 of the network is not created
        train_bn: bool.
            Train or freeze Batch Normalization layers

        Returns
        -------
        list
            Backbone layers of ResNet 50 or 101

        """

        # Code adopted from:
        # https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

        assert architecture in ["vgg16", "vgg19"]
        # Block 1
        x = KL.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_tensor)
        x = KL.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        C1 = x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
        
        # Block 2
        x = KL.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = KL.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        C2 = x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
        
        # Block 3
        x = KL.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = KL.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = KL.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        if architecture == "vgg19":
            x = KL.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
        C3 = x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
        
        # Block 4
        x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        if architecture == "vgg19":
            x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
        C4 = x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
        
        if stage5:
            # Block 5
            x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
            x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
            x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
            if architecture == "vgg19":
                x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
            C5 = x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
        else:
            C5 = None

        # Classification block
        x_fc = KL.Flatten(name='flatten')(x)
        x_fc = KL.Dense(4096, activation='relu', name='fc1')(x)
        x_fc = KL.Dense(4096, activation='relu', name='fc2')(x)
        # from tensorflow.python.keras.applications import imagenet_utils
        # imagenet_utils.validate_activation(classifier_activation, weights)
        x_fc = KL.Dense(1000, activation='softmax',name='predictions')(x)
        model = KM.Model(input_tensor, x_fc)
        return model

if __name__ == '__main__':
    config = Config()
    train_gen, val_gen = I.build_generators(config.TRAINING_PATH, config.VALIDATION_PATH, config.BATCH_SIZE, config.RESOLUTION)
    input_tens = tf.keras.layers.Input(shape=(config.RESOLUTION, config.RESOLUTION, 3))
    # model = M.create_application_model(modelName, input_tens)
    # model = M.finetuneNetwork(model, modelName)
    VGG19 = VGG19()
    model = VGG19.build_backbone(input_tensor=input_tens, architecture="vgg19", num_classes=3, stage5=True)
    weights_path = VGG19.get_imagenet_weights()
    VGG19.load_weights(weights_path, model, by_name=True)
    model = M.finetuneNetwork(model, "vgg19") #Unset the final softmax layer and put a new one with 3 categorical classes
    model.summary()
    sgd = KO.SGD(config.LEARNING_RATE, momentum=0.9, clipnorm=5.0)
    model.compile(optimizer=tf.keras.optimizers.Adadelta(lr=config.LEARNING_RATE),
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    history = model.fit(train_gen,
                        shuffle=True,
                        epochs=config.EPOCHS,
                        validation_data=val_gen,
                        verbose=1)
    model.save_weights(config.PRETRAINED_MODEL_PATH + "vgg19_weights.h5")
    model.save(config.PRETRAINED_MODEL_PATH + "vgg19.h5")
    G.graph_results(config.GRAPH_PATH, history, config.LEARNING_RATE, config.BATCH_SIZE, "Adadelta", "vgg19", config.RESOLUTION)
 
