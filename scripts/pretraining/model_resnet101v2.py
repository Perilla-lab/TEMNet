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

class ResNet101v2(object):
    def get_imagenet_weights(self):
        """Downloads ImageNet trained weights from Keras.
        Returns path to weights file.
        """
        from keras.utils.data_utils import get_file
        TF_WEIGHTS_PATH_NO_TOP = ('https://storage.googleapis.com/tensorflow/'
                                  'keras-applications/resnet/'
                                  'resnet101v2_weights_tf_dim_ordering_tf_kernels_notop.h5')
        weights_path = get_file(
            'resnet101v2_weights_tf_dim_ordering_tf_kernels_notop.h5',
            TF_WEIGHTS_PATH_NO_TOP,
            cache_subdir='models',
            file_hash=('6343647c601c52e1368623803854d971',
                    'c0ed64b8031c3730f411d2eb4eea35b5')[1]
        )
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

        def block2(x, filters, kernel_size=3, stride=1, conv_shortcut=False, name=None):
            """A residual block.
            
            Arguments:
            x: input tensor.
            filters: integer, filters of the bottleneck layer.
            kernel_size: default 3, kernel size of the bottleneck layer.
            stride: default 1, stride of the first layer.
            conv_shortcut: default False, use convolution shortcut if True,
            otherwise identity shortcut.
            name: string, block label.
            
            Returns:
            Output tensor for the residual block.
            """
            preact = KL.BatchNormalization(epsilon=1.001e-5, name=name + '_preact_bn')(x)
            preact = KL.Activation('relu', name=name + '_preact_relu')(preact)

            if conv_shortcut:
                shortcut = KL.Conv2D(
                    4 * filters, 1, strides=stride, name=name + '_0_conv')(preact)
            else:
                shortcut = KL.MaxPooling2D(1, strides=stride)(x) if stride > 1 else x
                
            x = KL.Conv2D(
                filters, 1, strides=1, use_bias=False, name=name + '_1_conv')(preact)
            x = KL.BatchNormalization(epsilon=1.001e-5, name=name + '_1_bn')(x)
            x = KL.Activation('relu', name=name + '_1_relu')(x)
            
            x = KL.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
            x = KL.Conv2D(
                filters,
                kernel_size,
                strides=stride,
                use_bias=False,
            name=name + '_2_conv')(x)
            x = KL.BatchNormalization(epsilon=1.001e-5, name=name + '_2_bn')(x)
            x = KL.Activation('relu', name=name + '_2_relu')(x)
            
            x = KL.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
            x = KL.Add(name=name + '_out')([shortcut, x])
            return x


        def stack2(x, filters, blocks, stride1=2, name=None):
            """A set of stacked residual blocks.
            
            Arguments:
            x: input tensor.
            filters: integer, filters of the bottleneck layer in a block.
            blocks: integer, blocks in the stacked blocks.
            stride1: default 2, stride of the first layer in the first block.
            name: string, stack label.
            
            Returns:
            Output tensor for the stacked blocks.
            """
            x = block2(x, filters, conv_shortcut=True, name=name + '_block1')
            for i in range(2, blocks):
                x = block2(x, filters, name=name + '_block' + str(i))
            x = block2(x, filters, stride=stride1, name=name + '_block' + str(blocks))
            return x

        assert architecture in ["resnet50v2", "resnet101v2", "resnet152v2"]

        #Stage 1
        x = KL.ZeroPadding2D(
            padding=((3, 3), (3, 3)), name='conv1_pad')(input_tensor)
        x = KL.Conv2D(64, 7, strides=2, use_bias=True, name='conv1_conv')(x)

        x = KL.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
        C1 = x = KL.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

        #Stage 2 Remember there's a pre activation on each residual block so there's no activation on these C## stages unlike the ResNetv1 counterpart
        #x = stack_fn(x)
        C2 = x = stack2(x, 64, 3, name='conv2')
        #Stage 3
        block_count_c3 = {"resnet50v2": 4, "resnet101v2": 4, "resnet152v2": 8}[architecture]
        C3 = x = stack2(x, 128, block_count_c3, name='conv3')
        #Stage 4
        block_count_c4 = {"resnet50v2": 6, "resnet101v2": 23, "resnet152v2": 36}[architecture]
        C4 = x = stack2(x, 256, block_count_c4, name='conv4')
        #Stage 5
        x = stack2(x, 512, 3, stride1=1, name='conv5')

        x = KL.BatchNormalization(epsilon=1.001e-5, name='post_bn')(x)
        C5 = x = KL.Activation('relu', name='post_relu')(x)

        x_fc = KL.GlobalAveragePooling2D(name='avg_pool')(x)
        x_fc = KL.Dense(1000, activation='softmax', name='predictions')(x_fc)
        #x_fc = KL.AveragePooling2D((7,7), name='avg_pool')(x)
        #x_fc = KL.Flatten()(x_fc)
        #x_fc = KL.Dense(1000, activation='softmax', name='fc1000')(x_fc)
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
    RN101 = ResNet101v2()
    model = RN101.build_backbone(input_tensor=input_tens, architecture="resnet101v2", num_classes=3, stage5=True)
    resnet_weights_path = RN101.get_imagenet_weights()
    RN101.load_weights(resnet_weights_path, model, by_name=True)
    model = M.finetuneNetwork(model, "resnet101v2") #Unset the final softmax layer and put a new one with 3 categorical classes
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
    model.save_weights(config.PRETRAINED_MODEL_PATH + "resnet101v2_weights.h5")
    model.save(config.PRETRAINED_MODEL_PATH + "resnet101v2.h5")
    G.graph_results(config.GRAPH_PATH, history, config.LEARNING_RATE, config.BATCH_SIZE, "Adadelta", "resnet101v2", config.RESOLUTION)
 
