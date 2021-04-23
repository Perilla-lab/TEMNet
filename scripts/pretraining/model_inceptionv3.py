import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.models as KM
#import tensorflow.keras.layers as KL
from tensorflow.python.keras.layers import VersionAwareLayers
KL = VersionAwareLayers()
import tensorflow.keras.optimizers as KO
from tensorflow.python.keras import backend
from tensorflow.python.keras.applications import imagenet_utils

import database as D
import models as M
import input_pipeline as I
import graphing as G
from classes import Config, PerillaNet

class InceptionV3(object):
    def get_imagenet_weights(self):
        """Downloads ImageNet trained weights from Keras.
        Returns path to weights file.
        """
        from keras.utils.data_utils import get_file
        TF_WEIGHTS_PATH_NO_TOP = (
            'https://storage.googleapis.com/tensorflow/keras-applications/'
            'inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
        weights_path = get_file(
            'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
            TF_WEIGHTS_PATH_NO_TOP,
            cache_subdir='models',
            md5_hash='bcbd6486424b2319ff4ef7d526e38f63')
        return weights_path
        # TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/'\
        #                          'releases/download/v0.2/'\
        #                          'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        # weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
        #                         TF_WEIGHTS_PATH_NO_TOP,
        #                         cache_subdir='models',
        #                         md5_hash='a268eb855778b3df3c7506639542a6af')
        # return weights_path

    def load_weights2(self, filepath, model, by_name=True, exclude=None):
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

    def build_backbone(self, input_tensor, architecture, num_classes, input_shape=None, stage5=False, train_bn=None):
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

        def conv2d_bn(x,
                      filters,
                      num_row,
                      num_col,
                      padding='same',
                      strides=(1, 1),
                      name=None):
            """Utility function to apply conv + BN.
            
            Arguments:
            x: input tensor.
            filters: filters in `Conv2D`.
            num_row: height of the convolution kernel.
            num_col: width of the convolution kernel.
            padding: padding mode in `Conv2D`.
            strides: strides in `Conv2D`.
            name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
            
            Returns:
            Output tensor after applying `Conv2D` and `BatchNormalization`.
            """
            if name is not None:
                bn_name = name + '_bn'
                conv_name = name + '_conv'
            else:
                bn_name = None
                conv_name = None
            if backend.image_data_format() == 'channels_first':
                bn_axis = 1
            else:
                bn_axis = 3
            x = KL.Conv2D(
                filters, (num_row, num_col),
                strides=strides,
                padding=padding,
                use_bias=False,
                name=conv_name)(
                x)
            x = KL.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
            x = KL.Activation('relu', name=name)(x)
            return x

        #Default input size for Inception is 299x299px
        # Our default size is 512x512px so the sizes change slightly
        # We indicate both sizes as: default_inception_size_feature_maps / our_size_feature_maps
        # In order to use Inception with FPN we nedd to add paddings in order to make sure 2D Upsampling from later feature maps match with feature maps the tensor size of previous feature maps
        """
        input_shape = imagenet_utils.obtain_input_shape(
            input_shape,
            default_size=299,
            min_size=75,
            data_format=backend.image_data_format(),
            require_flatten=include_top,
        weights=weights)
        """
        #input_shape = (400, 400, 3)
        input_shape = (299, 299, 3)
        if input_tensor is None:
            img_input = KL.Input(shape=input_shape)
        else:
            if not backend.is_keras_tensor(input_tensor):
                img_input = KL.Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor
        
        if backend.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3

        #Stage 1
        x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid') #  / 255x255x32 tensor
        C1 = KL.ZeroPadding2D(((0,1),(0,1)))(x) #  / 256x256x32 tensor

        #Stage 2
        x = conv2d_bn(x, 32, 3, 3, padding='valid') # / 253x253x32
        x = conv2d_bn(x, 64, 3, 3) # same
        x = KL.MaxPooling2D((3, 3), strides=(2, 2))(x) # / 126x126x64
        C2 = KL.ZeroPadding2D(1)(x) #  / 128x128x64 tensor

        # Stage 3
        x = conv2d_bn(x, 80, 1, 1, padding='valid')
        x = conv2d_bn(x, 192, 3, 3, padding='valid')
        x = KL.MaxPooling2D((3, 3), strides=(2, 2))(x) # 35x35x192 / 61x61x192 tensor

        # mixed 0: 35 x 35 x 256 / 61 x 61 x 256
        branch1x1 = conv2d_bn(x, 64, 1, 1)

        branch5x5 = conv2d_bn(x, 48, 1, 1)
        branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = KL.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
        x = KL.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                                axis=channel_axis,
                                name='mixed0') # 35 x 35 x 256 / 61 x 61 x 256

        # mixed 1: 35 x 35 x 288 / 61 x 61 x 288
        branch1x1 = conv2d_bn(x, 64, 1, 1)

        branch5x5 = conv2d_bn(x, 48, 1, 1)
        branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = KL.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
        x = KL.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                                axis=channel_axis,
                                name='mixed1') # 35 x 35 x 288 / 61 x 61 x 288

        # mixed 2: 35 x 35 x 288 / 61 x 61 x 288
        branch1x1 = conv2d_bn(x, 64, 1, 1)

        branch5x5 = conv2d_bn(x, 48, 1, 1)
        branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = KL.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
        x = KL.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                                axis=channel_axis,
                                name='mixed2') # 35 x 35 x 288 / 61 x 61 x 288
        C3 = KL.ZeroPadding2D(((1,2),(1,2)))(x) # 38x38x288 / 64x64x288 tensor

        # Stage 4
        # mixed 3: 17 x 17 x 768 / 30 x 30 x 768
        branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

        branch3x3dbl = conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = conv2d_bn(
            branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

        branch_pool = KL.MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = KL.concatenate([branch3x3, branch3x3dbl, branch_pool],
                                axis=channel_axis,
                                name='mixed3')

        # mixed 4: 17 x 17 x 768 / 30 x 30 x 768
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 128, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 128, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = KL.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = KL.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                                axis=channel_axis,
                                name='mixed4')

        # mixed 5, 6: 17 x 17 x 768 / 30 x 30 x 768
        for i in range(2):
            branch1x1 = conv2d_bn(x, 192, 1, 1)

            branch7x7 = conv2d_bn(x, 160, 1, 1)
            branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
            branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

            branch7x7dbl = conv2d_bn(x, 160, 1, 1)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

            branch_pool = KL.AveragePooling2D((3, 3),
                                                strides=(1, 1),
                                                padding='same')(
                                                    x)
            branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
            x = KL.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                                axis=channel_axis,
                                name='mixed' + str(5 + i))

        # mixed 7: 17 x 17 x 768 / 30 x 30 x 768
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 192, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 192, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = KL.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = KL.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                                axis=channel_axis,
                                name='mixed7')# 17 x 17 x 768 / 30 x 30 x 768
        C4 = KL.ZeroPadding2D(1)(x) # 19x19x768 / 32x32x768 tensor

        # Stage 5
        # mixed 8: 8 x 8 x 1280 / 14 x 14 x 1280
        branch3x3 = conv2d_bn(x, 192, 1, 1)
        branch3x3 = conv2d_bn(branch3x3, 320, 3, 3, strides=(2, 2), padding='valid')

        branch7x7x3 = conv2d_bn(x, 192, 1, 1)
        branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
        branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
        branch7x7x3 = conv2d_bn(
            branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

        branch_pool = KL.MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = KL.concatenate([branch3x3, branch7x7x3, branch_pool],
                                axis=channel_axis,
                                name='mixed8')

        # mixed 9,10: 8 x 8 x 2048 / 14 x 14 x 2048
        for i in range(2):
            branch1x1 = conv2d_bn(x, 320, 1, 1)

            branch3x3 = conv2d_bn(x, 384, 1, 1)
            branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
            branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
            branch3x3 = KL.concatenate([branch3x3_1, branch3x3_2],
                                        axis=channel_axis,
                                        name='mixed9_' + str(i))

            branch3x3dbl = conv2d_bn(x, 448, 1, 1)
            branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
            branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
            branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
            branch3x3dbl = KL.concatenate([branch3x3dbl_1, branch3x3dbl_2],
                                            axis=channel_axis)

            branch_pool = KL.AveragePooling2D((3, 3),
                                                strides=(1, 1),
                                                padding='same')(
                                                    x)
            branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
            x = KL.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool],
                                axis=channel_axis,
                                name='mixed' + str(9 + i))# 8 x 8 x 2048 / 14 x 14 x 2048
        C5 = KL.ZeroPadding2D(1)(x) # 10x10x2048 / 16x16x2048 tensor
        x_fc = KL.GlobalAveragePooling2D(name='avg_pool')(x)
        x_fc = KL.Dense(num_classes, activation='softmax',
                     name='predictions')(x_fc)

        # x_fc = KL.AveragePooling2D((7,7), name='avg_pool')(x)
        # x_fc = KL.Flatten()(x_fc)
        # x_fc = KL.Dense(1000, activation='softmax', name='fc1000')(x_fc)
        model = KM.Model(input_tensor, x_fc)
        # resnet_weights_path = this.get_imagenet_weights()
        # this.load_weights(resnet_weights_path, model, by_name=True)

        # x_newfc = KL.AveragePooling2D((7, 7), name='avg_pool')(x)
        # x_newfc = KL.Flatten()(x_newfc)
        # x_newfc = KL.Dense(num_classes, activation='softmax', name='fc8')(x_newfc)

        # model = KM.Model(img_input, x_newfc)
        return model
        # return [C1, C2, C3, C4, C5]

def preprocess_input(x, data_format=None):
    return imagenet_utils.preprocess_input(x, data_format=data_format, mode='tf')

if __name__ == '__main__':
    config = Config()
    config.RESOLUTION = 299
    train_gen, val_gen = I.build_generators(config.TRAINING_PATH, config.VALIDATION_PATH, config.BATCH_SIZE, config.RESOLUTION)
    input_tens = tf.keras.layers.Input(shape=(config.RESOLUTION, config.RESOLUTION, 3))
    input_tens = preprocess_input(input_tens)
    modelName="inceptionv3"
    #model = M.create_application_model(modelName, input_tens)
    #model = M.finetuneNetwork(model, modelName)
    
    IV3 = InceptionV3()
    model = IV3.build_backbone(input_tensor=input_tens, architecture="inceptionV3", num_classes=3, stage5=True)
    inception_weights_path = IV3.get_imagenet_weights()
    #IV3.load_weights2(inception_weights_path, model, by_name=True)
    model.load_weights(inception_weights_path, by_name=True)
    model = M.finetuneNetwork(model, "inceptionv3") #Unset the final softmax layer and put a new one with 3 categorical classes
    sgd = KO.SGD(config.LEARNING_RATE, momentum=0.9, clipnorm=5.0)
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adadelta(lr=config.LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(train_gen,
                        shuffle=True,
                        epochs=config.EPOCHS,
                        validation_data=val_gen,
                        verbose=1)
    model.save_weights(config.PRETRAINED_MODEL_PATH + "inceptionv3_weights.h5")
    model.save(config.PRETRAINED_MODEL_PATH + "ineptionv3.h5")
    G.graph_results(config.GRAPH_PATH, history, config.LEARNING_RATE, config.BATCH_SIZE, "Adadelta", "inceptionv3", config.RESOLUTION)
 
