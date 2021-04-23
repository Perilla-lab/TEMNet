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

class ResNet101(object):
    def get_imagenet_weights(self):
        """Downloads ImageNet trained weights from Keras.
        Returns path to weights file.
        """
        from keras.utils.data_utils import get_file
        TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/'\
                                 'releases/download/v0.2/'\
                                 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                TF_WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models',
                                md5_hash='a268eb855778b3df3c7506639542a6af')
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
        """Build a ResNet model. We could change this for another backbone

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

        def identity_block(tensor, kernel_size, filters, stage, block, use_bias=True):
            """The identity_block is the block that has no convolution layer at shortcut

            Arguments
            --------
            tensor: Keras Layer
                The tensor to connect to this block.
            kernel_size: int
                The kernel size of the convolutional layer
            filters: list
                List of integers indicating how many filters to use for each convolution layer
            stage: int
                Current stage label for generating layer names
            block: str
                Current block label for generating layer names
            use_bias: bool
                To use or not use a bias in conv layers.

            Returns
            -------
            y: Keras Layer
                Output of the Resnet identity block
            """

            nb_filter1, nb_filter2, nb_filter3 = filters
            conv_name_base = 'res' + str(stage) + block + '_branch'
            bn_name_base = 'bn' + str(stage) + block + '_branch'

            y = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', use_bias=use_bias)(tensor)
            y = KL.BatchNormalization(name=bn_name_base + '2a')(y, training=train_bn)
            y = KL.Activation('relu')(y)

            y = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b',
                          use_bias=use_bias)(y)
            y = KL.BatchNormalization(name=bn_name_base + '2b')(y, training=train_bn)
            y = KL.Activation('relu')(y)

            y = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=use_bias)(y)
            y = KL.BatchNormalization(name=bn_name_base + '2c')(y, training=train_bn)

            y = KL.Add()([y, tensor])
            y = KL.Activation('relu', name='res' + str(stage) + block + '_out')(y)
            return y

        def conv_block(tensor, kernel_size, filters, stage, block, strides=(2, 2), use_bias=True):

            """conv_block is the block that has a conv layer at shortcut

            Arguments
            ---------
            tensor: Keras Layer
                The tensor to connect to this block.
            kernel_size: int
                The kernel size of the convolutional layer
            filters: list
                List of integers indicating how many filters to use for each convolution layer
            stage: int
                Current stage label for generating layer names
            block: str
                Current block label for generating layer names
            strides: tuple
                A tuple of integers indicating the strides to make during convolution.
            use_bias: bool
                To use or not use a bias in conv layers.

            Returns
            -------
            y: Keras Layer
                Output layer of Resnet conv block

            """
            nb_filter1, nb_filter2, nb_filter3 = filters
            conv_name_base = 'res' + str(stage) + block + '_branch'
            bn_name_base = 'bn' + str(stage) + block + '_branch'

            y = KL.Conv2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a', use_bias=use_bias)(
                tensor)
            y = KL.BatchNormalization(name=bn_name_base + '2a')(y, training=train_bn)
            y = KL.Activation('relu')(y)

            y = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b',
                          use_bias=use_bias)(y)
            y = KL.BatchNormalization(name=bn_name_base + '2b')(y, training=train_bn)
            y = KL.Activation('relu')(y)

            y = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=use_bias)(y)
            y = KL.BatchNormalization(name=bn_name_base + '2c')(y, training=train_bn)

            shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides, name=conv_name_base + '1', use_bias=use_bias)(
                tensor)
            shortcut = KL.BatchNormalization(name=bn_name_base + '1')(shortcut, training=train_bn)

            y = KL.Add()([y, shortcut])
            y = KL.Activation('relu', name='res' + str(stage) + block + '_out')(y)
            return y

        assert architecture in ["resnet50", "resnet101"]
        # Stage 1
        x = KL.ZeroPadding2D((3, 3))(input_tensor)
        x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
        x = KL.BatchNormalization(name='bn_conv1')(x, training=train_bn)
        x = KL.Activation('relu')(x)
        C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
        # Stage 2
        x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
        # Stage 3
        x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
        # Stage 4
        x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        block_count = {"resnet50": 5, "resnet101": 22}[architecture]
        for i in range(block_count):
            x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i))
        C4 = x
        # Stage 5
        if stage5:
            x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
            x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
            C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
        else:
            C5 = None
        # Fully connected layers
        x_fc = KL.AveragePooling2D((7,7), name='avg_pool')(x)
        x_fc = KL.Flatten()(x_fc)
        x_fc = KL.Dense(1000, activation='softmax', name='fc1000')(x_fc)
        model = KM.Model(input_tensor, x_fc)
        # resnet_weights_path = this.get_imagenet_weights()
        # this.load_weights(resnet_weights_path, model, by_name=True)

        # x_newfc = KL.AveragePooling2D((7, 7), name='avg_pool')(x)
        # x_newfc = KL.Flatten()(x_newfc)
        # x_newfc = KL.Dense(num_classes, activation='softmax', name='fc8')(x_newfc)

        # model = KM.Model(img_input, x_newfc)
        return model
        # return [C1, C2, C3, C4, C5]

if __name__ == '__main__':
    config = Config()
    train_gen, val_gen = I.build_generators(config.TRAINING_PATH, config.VALIDATION_PATH, config.BATCH_SIZE, config.RESOLUTION)
    input_tens = tf.keras.layers.Input(shape=(config.RESOLUTION, config.RESOLUTION, 3))
    # model = M.create_application_model(modelName, input_tens)
    # model = M.finetuneNetwork(model, modelName)
    RN101 = ResNet101()
    model = RN101.build_backbone(input_tensor=input_tens, architecture="resnet101", num_classes=3, stage5=True)
    resnet_weights_path = RN101.get_imagenet_weights()
    # RN101.load_weights(resnet_weights_path, model, by_name=True)
    model = M.finetuneNetwork(model, "resnet101") #Unset the final softmax layer and put a new one with 3 categorical classes
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
    model.save_weights(config.PRETRAINED_MODEL_PATH + "resnet101_weights_nopretrain.h5")
    model.save(config.PRETRAINED_MODEL_PATH + "resnet101_nopretrain.h5")
    G.graph_results(config.GRAPH_PATH, history, config.LEARNING_RATE, config.BATCH_SIZE, "Adadelta", "resnet101_nopretrain", config.RESOLUTION)
 
