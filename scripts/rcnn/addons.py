import logging
import tensorflow as tf
import keras
# from typeguard import typechecked

"""Types for function signatures"""
from typing import Union, Callable, List

Initializer = Union[None, dict, str, Callable]
Regularizer = Union[None, dict, str, Callable]
Constraint = Union[None, dict, str, Callable]

"""
Add Group Normalization function as in https://arxiv.org/pdf/1803.08494.pdf
This should give better performance than Batch Normalization for Object detection networks since those use small batch sizes due to memory constraints
Group Normalization applies a transformation that maintains the mean close to 0 and the standard deviation close to 1, however instead of normalizing across the number of images in the batch you normalize across the number of channels across a group of images
INPUTS:
    x: input tensor to normalize of shape [N, H, W, C] (channels_last is default on the Conv layers) or [N,C,H,W] if you're using channels_first as the data_format
    gamma, beta: scale and offset with shape [1,1,1,C] or [1,1,C,1] if data_format is channels_first, the output is y_i = gamma*mean(x_i)+beta
    G: Number of groups for GN, default is 32 as per the paper
"""
def GroupNormalization_user(x, gamma_initializer="ones", beta_initializer="zeros", G=32, eps=1e-5):
  N, H, W, C = x.shape
  x = tf.reshape(x, [N, G, H, W, C // G])
  mean, var = tf.nn.moments(x, [2, 3, 4], keepdims=True)
  x = (x - mean) / tf.sqrt(var + eps)
  x = tf.reshape(x, [N, H, W, C])

  if gamma_initializer == "ones":
    gamma = tf.ones(shape=[1,1,1,C])
  if beta_initializer == "zeros":
    beta = tf.zeros(shape=[1,1,1,C])

  return x*gamma + beta

class GroupNormalization(keras.layers.Layer):
    """Group normalization layer. from Tensorflow addons

    Source: "Group Normalization" (Yuxin Wu & Kaiming He, 2018)
    https://arxiv.org/abs/1803.08494

    Group Normalization divides the channels into groups and computes
    within each group the mean and variance for normalization.
    Empirically, its accuracy is more stable than batch norm in a wide
    range of small batch sizes, if learning rate is adjusted linearly
    with batch sizes.

    Relation to Layer Normalization:
    If the number of groups is set to 1, then this operation becomes identical
    to Layer Normalization.

    Relation to Instance Normalization:
    If the number of groups is set to the
    input dimension (number of groups is equal
    to number of channels), then this operation becomes
    identical to Instance Normalization.

    Args:
        groups: Integer, the number of groups for Group Normalization.
            Can be in the range [1, N] where N is the input dimension.
            The input dimension must be divisible by the number of groups.
            Defaults to 32.
        axis: Integer, the axis that should be normalized.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.

    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    Output shape:
        Same shape as input.
    """


    # @typechecked
    def __init__(
        self,
        groups: int = 32,
        axis: int = -1,
        epsilon: float = 1e-3,
        center: bool = True,
        scale: bool = True,
        beta_initializer: Initializer = "zeros",
        gamma_initializer: Initializer = "ones",
        beta_regularizer: Regularizer = None,
        gamma_regularizer: Regularizer = None,
        beta_constraint: Constraint = None,
        gamma_constraint: Constraint = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = keras.initializers.get(beta_initializer)
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_regularizer = keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = keras.constraints.get(beta_constraint)
        self.gamma_constraint = keras.constraints.get(gamma_constraint)
        self._check_axis()


    def build(self, input_shape):


        self._check_if_input_shape_is_none(input_shape)
        self._set_number_of_groups_for_instance_norm(input_shape)
        self._check_size_of_dimensions(input_shape)
        self._create_input_spec(input_shape)


        self._add_gamma_weight(input_shape)
        self._add_beta_weight(input_shape)
        self.built = True
        super().build(input_shape)


    def call(self, inputs):


        input_shape = keras.backend.int_shape(inputs)
        tensor_input_shape = tf.shape(inputs)


        reshaped_inputs, group_shape = self._reshape_into_groups(
            inputs, input_shape, tensor_input_shape
        )


        normalized_inputs = self._apply_normalization(reshaped_inputs, input_shape)


        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            outputs = tf.reshape(normalized_inputs, tensor_input_shape)
        else:
            outputs = normalized_inputs


        return outputs


    def get_config(self):
        config = {
            "groups": self.groups,
            "axis": self.axis,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "beta_initializer": keras.initializers.serialize(self.beta_initializer),
            "gamma_initializer": keras.initializers.serialize(
                self.gamma_initializer
            ),
            "beta_regularizer": keras.regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": keras.regularizers.serialize(
                self.gamma_regularizer
            ),
            "beta_constraint": keras.constraints.serialize(self.beta_constraint),
            "gamma_constraint": keras.constraints.serialize(self.gamma_constraint),
        }
        base_config = super().get_config()
        return {**base_config, **config}


    def compute_output_shape(self, input_shape):
        return input_shape


    def _reshape_into_groups(self, inputs, input_shape, tensor_input_shape):


        group_shape = [tensor_input_shape[i] for i in range(len(input_shape))]
        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            group_shape[self.axis] = input_shape[self.axis] // self.groups
            group_shape.insert(self.axis, self.groups)
            group_shape = tf.stack(group_shape)
            reshaped_inputs = tf.reshape(inputs, group_shape)
            return reshaped_inputs, group_shape
        else:
            return inputs, group_shape


    def _apply_normalization(self, reshaped_inputs, input_shape):


        group_shape = keras.backend.int_shape(reshaped_inputs)
        group_reduction_axes = list(range(1, len(group_shape)))
        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            axis = -2 if self.axis == -1 else self.axis - 1
        else:
            axis = -1 if self.axis == -1 else self.axis - 1
        group_reduction_axes.pop(axis)


        mean, variance = tf.nn.moments(
            reshaped_inputs, group_reduction_axes, keepdims=True
        )


        gamma, beta = self._get_reshaped_weights(input_shape)
        normalized_inputs = tf.nn.batch_normalization(
            reshaped_inputs,
            mean=mean,
            variance=variance,
            scale=gamma,
            offset=beta,
            variance_epsilon=self.epsilon,
        )
        return normalized_inputs


    def _get_reshaped_weights(self, input_shape):
        broadcast_shape = self._create_broadcast_shape(input_shape)
        gamma = None
        beta = None
        if self.scale:
            gamma = tf.reshape(self.gamma, broadcast_shape)


        if self.center:
            beta = tf.reshape(self.beta, broadcast_shape)
        return gamma, beta


    def _check_if_input_shape_is_none(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError(
                "Axis " + str(self.axis) + " of "
                "input tensor should have a defined dimension "
                "but the layer received an input with shape " + str(input_shape) + "."
            )


    def _set_number_of_groups_for_instance_norm(self, input_shape):
        dim = input_shape[self.axis]


        if self.groups == -1:
            self.groups = dim


    def _check_size_of_dimensions(self, input_shape):


        dim = input_shape[self.axis]
        if dim < self.groups:
            raise ValueError(
                "Number of groups (" + str(self.groups) + ") cannot be "
                "more than the number of channels (" + str(dim) + ")."
            )


        if dim % self.groups != 0:
            raise ValueError(
                "Number of groups (" + str(self.groups) + ") must be a "
                "multiple of the number of channels (" + str(dim) + ")."
            )


    def _check_axis(self):


        if self.axis == 0:
            raise ValueError(
                "You are trying to normalize your batch axis. Do you want to "
                "use tf.layer.batch_normalization instead"
            )


    def _create_input_spec(self, input_shape):


        dim = input_shape[self.axis]
        self.input_spec = keras.layers.InputSpec(
            ndim=len(input_shape), axes={self.axis: dim}
        )


    def _add_gamma_weight(self, input_shape):


        dim = input_shape[self.axis]
        shape = (dim,)


        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                name="gamma",
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
            )
        else:
            self.gamma = None


    def _add_beta_weight(self, input_shape):


        dim = input_shape[self.axis]
        shape = (dim,)


        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                name="beta",
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
            )
        else:
            self.beta = None


    def _create_broadcast_shape(self, input_shape):
        broadcast_shape = [1] * len(input_shape)
        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
            broadcast_shape.insert(self.axis, self.groups)
        else:
            broadcast_shape[self.axis] = self.groups
        return broadcast_shape
