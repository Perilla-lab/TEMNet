import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
import tensorflow.keras.optimizers as KO
from tensorflow.python.keras import backend
from tensorflow.python.keras.applications import imagenet_utils

import database as D
import models as M
import input_pipeline as I
import graphing as G
from classes import Config, PerillaNet

class InceptionResNetV2(object):
    def get_imagenet_weights(self):
        """Downloads ImageNet trained weights from Keras.
        Returns path to weights file.
        """
        from keras.utils.data_utils import get_file
        TF_WEIGHTS_PATH_NO_TOP = ('https://storage.googleapis.com/tensorflow/'
                           'keras-applications/inception_resnet_v2/'
                           'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5')

        weights_path = get_file(
            'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5',
            TF_WEIGHTS_PATH_NO_TOP,
            cache_subdir='models',
            file_hash='d19885ff4a710c122648d3b5c3b684e4')
        return weights_path
        # TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/'\
        #                          'releases/download/v0.2/'\
        #                          'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        # weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
        #                         TF_WEIGHTS_PATH_NO_TOP,
        #                         cache_subdir='models',
        #                         md5_hash='a268eb855778b3df3c7506639542a6af')
        # return weights_path

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
                    kernel_size,
                    strides=1,
                    padding='same',
                    activation='relu',
                    use_bias=False,
                    name=None):
            """Utility function to apply conv + BN.
            
            Arguments:
            x: input tensor.
            filters: filters in `Conv2D`.
            kernel_size: kernel size as in `Conv2D`.
            strides: strides in `Conv2D`.
            padding: padding mode in `Conv2D`.
            activation: activation in `Conv2D`.
            use_bias: whether to use a bias in `Conv2D`.
            name: name of the ops; will become `name + '_ac'` for the activation
                and `name + '_bn'` for the batch norm layer.
            
            Returns:
            Output tensor after applying `Conv2D` and `BatchNormalization`.
            """
            x = KL.Conv2D(
                filters,
                kernel_size,
                strides=strides,
                padding=padding,
                use_bias=use_bias,
                name=name)(
                    x)
            if not use_bias:
                bn_axis = 1 if backend.image_data_format() == 'channels_first' else 3
                bn_name = None if name is None else name + '_bn'
                x = KL.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
            if activation is not None:
                ac_name = None if name is None else name + '_ac'
                x = KL.Activation(activation, name=ac_name)(x)
            return x


        def inception_resnet_block(x, scale, block_type, block_idx, activation='relu'):
            """Adds an Inception-ResNet block.
            
            This function builds 3 types of Inception-ResNet blocks mentioned
            in the paper, controlled by the `block_type` argument (which is the
            block name used in the official TF-slim implementation):
            - Inception-ResNet-A: `block_type='block35'`
            - Inception-ResNet-B: `block_type='block17'`
            - Inception-ResNet-C: `block_type='block8'`
            
            Arguments:
            x: input tensor.
            scale: scaling factor to scale the residuals (i.e., the output of passing
            `x` through an inception module) before adding them to the shortcut
            branch. Let `r` be the output from the residual branch, the output of this
            block will be `x + scale * r`.
            block_type: `'block35'`, `'block17'` or `'block8'`, determines the network
            structure in the residual branch.
            block_idx: an `int` used for generating layer names. The Inception-ResNet
            blocks are repeated many times in this network. We use `block_idx` to
            identify each of the repetitions. For example, the first
            Inception-ResNet-A block will have `block_type='block35', block_idx=0`,
            and the layer names will have a common prefix `'block35_0'`.
            activation: activation function to use at the end of the block (see
            [activations](../activations.md)). When `activation=None`, no activation
            is applied
            (i.e., "linear" activation: `a(x) = x`).

            Returns:
            Output tensor for the block.
            
            Raises:
            ValueError: if `block_type` is not one of `'block35'`,
            `'block17'` or `'block8'`.
            """
            if block_type == 'block35':
                branch_0 = conv2d_bn(x, 32, 1)
                branch_1 = conv2d_bn(x, 32, 1)
                branch_1 = conv2d_bn(branch_1, 32, 3)
                branch_2 = conv2d_bn(x, 32, 1)
                branch_2 = conv2d_bn(branch_2, 48, 3)
                branch_2 = conv2d_bn(branch_2, 64, 3)
                branches = [branch_0, branch_1, branch_2]
            elif block_type == 'block17':
                branch_0 = conv2d_bn(x, 192, 1)
                branch_1 = conv2d_bn(x, 128, 1)
                branch_1 = conv2d_bn(branch_1, 160, [1, 7])
                branch_1 = conv2d_bn(branch_1, 192, [7, 1])
                branches = [branch_0, branch_1]
            elif block_type == 'block8':
                branch_0 = conv2d_bn(x, 192, 1)
                branch_1 = conv2d_bn(x, 192, 1)
                branch_1 = conv2d_bn(branch_1, 224, [1, 3])
                branch_1 = conv2d_bn(branch_1, 256, [3, 1])
                branches = [branch_0, branch_1]
            else:
                raise ValueError('Unknown Inception-ResNet block type. '
                                'Expects "block35", "block17" or "block8", '
                                'but got: ' + str(block_type))

            block_name = block_type + '_' + str(block_idx)
            channel_axis = 1 if backend.image_data_format() == 'channels_first' else 3
            mixed = KL.Concatenate(
                axis=channel_axis, name=block_name + '_mixed')(
                    branches)
            up = conv2d_bn(
                mixed,
                backend.int_shape(x)[channel_axis],
                1,
                activation=None,
                use_bias=True,
                name=block_name + '_conv')

            x = KL.Lambda(
                lambda inputs, scale: inputs[0] + inputs[1] * scale,
                output_shape=backend.int_shape(x)[1:],
                arguments={'scale': scale},
                name=block_name)([x, up])
            if activation is not None:
                x = KL.Activation(activation, name=block_name + '_ac')(x)
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
        input_shape = (400,400,3)
        if input_tensor is None:
            img_input = KL.Input(shape=input_shape)
        else:
            if not backend.is_keras_tensor(input_tensor):
                img_input = KL.Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor


        #Stage 1
        # Stem block: 35 x 35 x 192
        x = conv2d_bn(img_input, 32, 3, strides=2, padding='valid')#  / 255x255x32 tensor
        C1 = KL.ZeroPadding2D(((0,1),(0,1)))(x) #  / 256x256x32 tensor

        #Stage 2
        x = conv2d_bn(x, 32, 3, padding='valid') # / 253x253x32
        x = conv2d_bn(x, 64, 3)
        x = KL.MaxPooling2D(3, strides=2)(x) # / 126x126x64
        C2 = KL.ZeroPadding2D(1)(x) #  / 128x128x64 tensor

        # Stage 3
        x = conv2d_bn(x, 80, 1, padding='valid')
        x = conv2d_bn(x, 192, 3, padding='valid')
        x = KL.MaxPooling2D(3, strides=2)(x) # 35x35x192 / 61x61x192 tensor

        # Mixed 5b (Inception-A block): 35 x 35 x 320 / 61 x 61 x 256
        branch_0 = conv2d_bn(x, 96, 1)
        branch_1 = conv2d_bn(x, 48, 1)
        branch_1 = conv2d_bn(branch_1, 64, 5)
        branch_2 = conv2d_bn(x, 64, 1)
        branch_2 = conv2d_bn(branch_2, 96, 3)
        branch_2 = conv2d_bn(branch_2, 96, 3)
        branch_pool = KL.AveragePooling2D(3, strides=1, padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 64, 1)
        branches = [branch_0, branch_1, branch_2, branch_pool]
        channel_axis = 1 if backend.image_data_format() == 'channels_first' else 3
        x = KL.Concatenate(axis=channel_axis, name='mixed_5b')(branches)

        # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
        for block_idx in range(1, 11):
            x = inception_resnet_block(
                x, scale=0.17, block_type='block35', block_idx=block_idx)
        C3 = KL.ZeroPadding2D(((1,2),(1,2)))(x) # 38x38x320 / 64x64x320 tensor

        # Stage 4
        # Mixed 6a (Reduction-A block): 17 x 17 x 1088 / 30 x 30 x 1088
        branch_0 = conv2d_bn(x, 384, 3, strides=2, padding='valid')
        branch_1 = conv2d_bn(x, 256, 1)
        branch_1 = conv2d_bn(branch_1, 256, 3)
        branch_1 = conv2d_bn(branch_1, 384, 3, strides=2, padding='valid')
        branch_pool = KL.MaxPooling2D(3, strides=2, padding='valid')(x)
        branches = [branch_0, branch_1, branch_pool]
        x = KL.Concatenate(axis=channel_axis, name='mixed_6a')(branches)

        # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
        for block_idx in range(1, 21):
            x = inception_resnet_block(
                x, scale=0.1, block_type='block17', block_idx=block_idx)
        C4 = KL.ZeroPadding2D(1)(x) # 19x19x768 / 32x32x768 tensor

        # Stage 5
        # Mixed 7a (Reduction-B block): 8 x 8 x 2080 / 14 x 14 x 2080
        branch_0 = conv2d_bn(x, 256, 1)
        branch_0 = conv2d_bn(branch_0, 384, 3, strides=2, padding='valid')
        branch_1 = conv2d_bn(x, 256, 1)
        branch_1 = conv2d_bn(branch_1, 288, 3, strides=2, padding='valid')
        branch_2 = conv2d_bn(x, 256, 1)
        branch_2 = conv2d_bn(branch_2, 288, 3)
        branch_2 = conv2d_bn(branch_2, 320, 3, strides=2, padding='valid')
        branch_pool = KL.MaxPooling2D(3, strides=2, padding='valid')(x)
        branches = [branch_0, branch_1, branch_2, branch_pool]
        x = KL.Concatenate(axis=channel_axis, name='mixed_7a')(branches)

        # 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
        for block_idx in range(1, 10):
            x = inception_resnet_block(
                x, scale=0.2, block_type='block8', block_idx=block_idx)
        x = inception_resnet_block(
            x, scale=1., activation=None, block_type='block8', block_idx=10)

        # Final convolution block: 8 x 8 x 1536
        x = conv2d_bn(x, 1536, 1, name='conv_7b')
        if stage5:
            C5 = KL.ZeroPadding2D(1)(x) # 10x10x1536 / 16x16x1536 tensor

        # Classification block
        x_fc = KL.GlobalAveragePooling2D(name='avg_pool')(x)
        #imagenet_utils.validate_activation(classifier_activation, weights)
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
    train_gen, val_gen = I.build_generators(config.TRAINING_PATH, config.VALIDATION_PATH, config.BATCH_SIZE, config.RESOLUTION)
    input_tens = tf.keras.layers.Input(shape=(config.RESOLUTION, config.RESOLUTION, 3))
    #input_tens = process_input(input_tens)
    # model = M.create_application_model(modelName, input_tens)
    # model = M.finetuneNetwork(model, modelName)
    IV3 = InceptionResNetV2()
    model = IV3.build_backbone(input_tensor=input_tens, architecture="inception_resnetv2", num_classes=3, stage5=True)
    inception_weights_path = IV3.get_imagenet_weights()
    IV3.load_weights(inception_weights_path, model, by_name=True)
    model = M.finetuneNetwork(model, "inceptionv3") #Unset the final softmax layer and put a new one with 3 categorical classes
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
    model.save_weights(config.PRETRAINED_MODEL_PATH + "inception_resnetv2_weights.h5")
    model.save(config.PRETRAINED_MODEL_PATH + "ineption_resnetv2.h5")
    G.graph_results(config.GRAPH_PATH, history, config.LEARNING_RATE, config.BATCH_SIZE, "Adadelta", "inception_resnetv2", config.RESOLUTION)
