from tensorflow import pad

from dataset_iterator.keras_layers import InferenceLayer
from ..utils.helpers import ensure_multiplicity
import tensorflow as tf
import numpy as np


class InferenceAwareSelector(InferenceLayer, tf.keras.layers.Layer):
    def __init__(self, inference_idx, name: str= "SelectFeature", **kwargs):
        self.inference_idx=inference_idx
        super().__init__(name=name, **kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({"inference_idx":self.inference_idx})
        return config

    def call(self, input): # (N x B, Y, X, F), N x (B, Y, X, F)
        input_concat, input_split = input
        if self.inference_mode: # only produce one output
            if isinstance(self.inference_idx, (tuple, list)):
                items = [input_split[idx] for idx in self.inference_idx]
                return tf.concat(items, axis = 0)
            else:
                return input_split[self.inference_idx]
        else:
            if input_concat is None:
                return tf.concat(input_split, axis = 0)
            else:
                return input_concat


class InferenceAwareBatchSelector(InferenceLayer, tf.keras.layers.Layer):
    def __init__(self, inference_idx:list, train_idx:list=None, merge_batch_dim:bool=True, name: str= "SelectFeature2", **kwargs):
        self.train_idx=train_idx
        self.inference_idx=inference_idx
        self.merge_batch_dim=merge_batch_dim
        super().__init__(name=name, **kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({"train_idx":self.train_idx, "inference_idx":self.inference_idx, "merge_batch_dim":self.merge_batch_dim})
        return config

    def call(self, input): # (N, B, Y, X, F)
        shape = tf.shape(input)
        if self.inference_mode:
            if self.inference_idx is None:
                return tf.reshape(input, [-1, shape[2], shape[3], shape[4]]) if self.merge_batch_dim else input
            elif isinstance(self.inference_idx, (tuple, list)):
                items = tf.gather(input, tf.constant(self.inference_idx, tf.int32), axis=0)
                return tf.reshape(items, [-1, shape[2], shape[3], shape[4]]) if self.merge_batch_dim else items
            else:
                return input[self.inference_idx] if self.merge_batch_dim else input[self.inference_idx:self.inference_idx+1]
        else:
            if self.train_idx is None:
                return tf.reshape(input, [-1, shape[2], shape[3], shape[4]]) if self.merge_batch_dim else input
            elif isinstance(self.train_idx, (tuple, list)):
                items = tf.gather(input, tf.constant(self.train_idx, tf.int32), axis=0)
                return tf.reshape(items, [-1, shape[2], shape[3], shape[4]]) if self.merge_batch_dim else items
            else:
                return input[self.train_idx] if self.merge_batch_dim else input[self.train_idx:self.train_idx+1]


class ChannelToBatch(tf.keras.layers.Layer):
    def __init__(self, compensate_gradient:bool = False, add_channel_axis:bool = True, name: str="ChannelToBatch", **kwargs):
        self.compensate_gradient=compensate_gradient
        self.add_channel_axis=add_channel_axis
        super().__init__(name=name, **kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({"compensate_gradient": self.compensate_gradient, "add_channel_axis":self.add_channel_axis})
        return config

    def build(self, input_shape): # B, Y, X, C
        try:
            input_shape = input_shape.as_list()
        except:
            pass
        self.rank = len(input_shape)
        self.perm = [self.rank-1, 0] + [i for i in range(1, self.rank-1)]
        if self.compensate_gradient:
            self.grad_fun = get_grad_weight_fun(float(input_shape[-1]))
        super().build(input_shape)

    def call(self, input): # (B, Y, X, C)
        shape = tf.shape(input)
        target_shape  = tf.concat([[-1], [shape[i] for i in range(1, self.rank-1)]], 0) if self.rank>2 else [-1]
        input = tf.transpose(input, perm=self.perm) # (C, B, Y, X)
        input = tf.reshape(input, target_shape) # (C x B, Y, X)
        if self.add_channel_axis:
            input = tf.expand_dims(input, -1) # (C x B, Y, X, 1)
        if self.compensate_gradient:
            input = self.grad_fun(input)
        return input


class SplitBatch(tf.keras.layers.Layer):
    def __init__(self, n_splits:int, compensate_gradient:bool = False, return_list:bool=True, name:str="SplitBatch2D", **kwargs):
        self.n_splits=n_splits
        self.compensate_gradient=compensate_gradient
        self.return_list=return_list
        super().__init__(name=name, **kwargs)

    def get_config(self):
      config = super().get_config().copy()
      config.update({"n_splits": self.n_splits, "compensate_gradient":self.compensate_gradient, "return_list":self.return_list})
      return config

    def build(self, input_shape):
        try:
            input_shape = input_shape.as_list()
        except:
            pass
        self.rank = len(input_shape)
        if self.compensate_gradient:
            self.grad_fun = get_grad_weight_fun(1./self.n_splits)
        super().build(input_shape)

    def call(self, input): #(N x B, Y, X, C)
        shape = tf.shape(input)
        target_shape  = tf.concat([[self.n_splits, -1], [shape[i] for i in range(1, self.rank)]], 0)
        if self.compensate_gradient:
            input = self.grad_fun(input) # so that gradient are averaged over N (number of frames)
        input = tf.reshape(input, target_shape) # (N, B, Y, X, C)
        if self.return_list:
            splits = tf.split(input, num_or_size_splits = self.n_splits, axis=0) # N x (1, B, Y, X, C)
            return [tf.squeeze(s, 0) for s in splits] # N x (B, Y, X, C)
        else :
            return input


class BatchToChannel(InferenceLayer, tf.keras.layers.Layer):
    def __init__(self, n_splits:int, n_splits_inference:int=1, inference_idx=None, compensate_gradient:bool = False, name:str= "BatchToChannel", **kwargs):
        self.n_splits=n_splits
        self.compensate_gradient = compensate_gradient
        self.n_splits_inference=n_splits_inference
        if inference_idx is not None:
            if isinstance(inference_idx, int):
                inference_idx=[inference_idx]
            assert isinstance(inference_idx, list), f"inference_idx must be a list of indices: {inference_idx}"
            assert np.all(np.asarray(inference_idx)<n_splits_inference), f"all inference_idx must be lower than {n_splits_inference} got {inference_idx}"
        self.inference_idx = inference_idx
        super().__init__(name=name, autocast=False, **kwargs)

    def get_config(self):
      config = super().get_config().copy()
      config.update({"n_splits": self.n_splits, "compensate_gradient":self.compensate_gradient, "n_splits_inference":self.n_splits_inference, "inference_idx":self.inference_idx})
      return config

    def build(self, input_shape):
        try:
            input_shape = input_shape.as_list()
        except:
            pass
        self.rank = len(input_shape)
        self.perm = [1] + [i+1 for i in range(1, self.rank-1)] + [0, self.rank] # (B, [DIMS], N, F)
        if self.compensate_gradient:
            self.grad_fun = get_grad_weight_fun(1./self.n_splits)
        super().build(input_shape)

    def call(self, input): #(N x B, Y, X, C)
        if self.inference_mode and self.n_splits_inference==1:
            return input
        if self.compensate_gradient and not self.inference_mode:
            input = self.grad_fun(input)
        n_splits = self.n_splits if not self.inference_mode else self.n_splits_inference
        n_splits_out = self.n_splits if not self.inference_mode else (self.n_splits_inference if self.inference_idx is None else len(self.inference_idx))
        shape = tf.shape(input)
        dims = [shape[i] for i in range(1, self.rank-1)] if self.rank>2 else []
        target_shape1 = tf.concat([[n_splits, -1], dims, shape[-1:]], 0) if self.rank>2 else tf.concat([[n_splits, -1], shape[-1:]], 0)
        target_shape2 = tf.concat([[-1], dims, [n_splits_out * shape[-1]]], 0) if self.rank>2 else [-1, n_splits_out * shape[-1]]
        input = tf.reshape(input, shape = target_shape1) # (N, B, Y, X, F)
        if self.inference_mode and self.inference_idx is not None:
            input = tf.gather(input, axis=0, indices=self.inference_idx) # (N, B, Y, X, F)
        input = tf.transpose(input, perm=self.perm) # (B, Y, X, N, F)
        return tf.reshape(input, target_shape2) # (B, Y, X, N x F)


class Stack(tf.keras.layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super(Stack, self).__init__(**kwargs)
        self.axis = axis
        self.supports_masking = True

    def call(self, inputs):
        return tf.stack(inputs, axis=self.axis)

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, (list, tuple)):
            raise ValueError("StackLayer expects a list of input tensors.")

        first_shape = tf.TensorShape(input_shape[0]).as_list()
        for shape in input_shape[1:]:
            current_shape = tf.TensorShape(shape).as_list()
            if first_shape != current_shape:
                raise ValueError("All input shapes must be the same for stacking.")

        lim = self.axis if self.axis >= 0 else self.axis + 1
        output_shape = first_shape[:lim] + [len(input_shape)] + first_shape[lim:]
        return tf.TensorShape(output_shape)

    def get_config(self):
        config = super(Stack, self).get_config()
        config.update({'axis': self.axis})
        return config


class Combine(tf.keras.layers.Layer):
    def __init__(
            self,
            filters: int = 0,
            kernel_size: int=1,
            activation: str="relu",
            compensate_gradient:bool = False,
            l2_reg: float=0,
            output_dtype:str = None,
            name: str="Combine",
            **kwargs
        ):
        self.activation = activation
        self.filters= filters
        self.kernel_size=kernel_size
        self.compensate_gradient = compensate_gradient
        self.l2_reg=l2_reg
        self.output_dtype=output_dtype
        super().__init__(name=name, **kwargs)

    def get_config(self):
      config = super().get_config().copy()
      config.update({"activation": self.activation, "filters":self.filters, "kernel_size":self.kernel_size, "compensate_gradient":self.compensate_gradient, "l2_reg":self.l2_reg, "output_dtype":self.output_dtype})
      return config

    def build(self, input_shape):
        if self.filters <=0:
            f = [s[-1] for s in input_shape]
            filters = f[0]
            assert np.all(np.array(f) == f[0]), "when filter is not provided, all input tensor must have same filter number"
        else:
            filters = self.filters
        self.concat = tf.keras.layers.Concatenate(axis=-1, name = self.name+"_concat")
        self.combine_conv = Conv2DWithDtype(
            filters=filters,
            kernel_size=self.kernel_size,
            dtype=self.dtype_policy,
            padding='same',
            activation=self.activation,
            l2_reg=self.l2_reg,
            output_dtype=self.output_dtype,
            name=self.name+"_conv1x1")
        if self.compensate_gradient:
            self.grad_fun = get_grad_weight_fun(1./len(input_shape))
        super().build(input_shape)

    def call(self, input):
        # for i in range(len(input)):
            # input[i] = get_print_grad_fun(f"{self.name} before combine {i}")(input[i])
        x = self.concat(input)
        # x = get_print_grad_fun(f"{self.name} before before concat")(x)
        if self.compensate_gradient:
            x = self.grad_fun(x)
        x = self.combine_conv(x)
        # x = get_print_grad_fun(f"{self.name} before before conv")(x)
        return x



class NConvToBatch2D(InferenceLayer, tf.keras.layers.Layer):
    def __init__(self, n_conv:int, inference_idx, filters:int, compensate_gradient:bool = False, activation="relu", name: str= "NConvToBatch2D", l2_reg=None, **kwargs):
        self.n_conv = n_conv
        self.filters = filters
        self.activation = activation
        self.inference_idx=inference_idx
        self.compensate_gradient=compensate_gradient
        self.l2_reg=l2_reg
        super().__init__(name=name, **kwargs)
        print(f"{self.name} activation={self.activation}")

    def get_config(self):
        config = super().get_config().copy()
        config.update({"n_conv": self.n_conv, "filters":self.filters, "compensate_gradient":self.compensate_gradient, "activation":self.activation, "inference_idx":self.inference_idx, "l2_reg":self.l2_reg})
        return config

    def build(self, input_shape):
        self.convs = [
            tf.keras.layers.Conv2D(
                filters=self.filters,
                kernel_size=1,
                padding='same',
                activation=self.activation,
                kernel_initializer=get_kernel_initializer(self.activation),
                dtype=self.dtype_policy,
                kernel_regularizer=HybridThresholdL2Regularizer(directional_strength=self.l2_reg * 10, elementwise_strength=self.l2_reg) if self.l2_reg > 0 else None,
                bias_regularizer=HybridThresholdL2Regularizer(directional_strength=0, elementwise_strength=self.l2_reg) if self.l2_reg > 0 else None,
                kernel_constraint=ClipMaxValue(),
                bias_constraint=ClipMaxValue(),
                name=f"Conv_{i}"
            )
        for i in range(self.n_conv)]

        if self.compensate_gradient and self.n_conv>1:
            self.grad_fun = get_grad_weight_fun(float(self.n_conv))
            self.grad_fun_inv = get_grad_weight_fun(1./self.n_conv)
        else:
            self.grad_fun = None
            self.grad_fun_inv = None
        super().build(input_shape)

    def call(self, input): # (B, Y, X, F)
        if self.inference_mode and self.inference_idx is not None:
            if isinstance(self.inference_idx, (tuple, list)):
                items = [self.convs[idx](input) for idx in self.inference_idx]
                return tf.concat(items, axis=0)
            else:
                return self.convs[self.inference_idx](input)
        # input = get_print_grad_fun(f"{self.name} before split")(input)
        if self.grad_fun_inv is not None and not self.inference_mode:
            input = self.grad_fun_inv(input)

        inputs = [conv(input) for conv in self.convs] # N x (B, Y, X, F)
        # inputs[0] = get_print_grad_fun(f"{self.name} before concat")(inputs[0])
        output = tf.concat(inputs, axis = 0) # (N x B, Y, X, F)
        if self.grad_fun is not None and not self.inference_mode:
            output = self.grad_fun(output) # compensate gradients to have same level in
        # output = get_print_grad_fun(f"{self.name} after concat")(output)
        return output


class ResConv2D(tf.keras.layers.Layer):
    def __init__(
            self,
            kernel_size: int=3,
            dilation: int = 1,
            weighted_sum : bool = False,
            dropout_rate : float = 0.3,
            batch_norm : bool = True,
            activation:str = "relu",
            l2_reg:float = 0,
            output_dtype=None,
            name: str="ResConv2D",
            **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.activation=activation
        self.dropout_rate=dropout_rate
        self.batch_norm=batch_norm
        self.weighted_sum = weighted_sum
        self.l2_reg = l2_reg
        self.output_dtype=output_dtype

    def get_config(self):
      config = super().get_config().copy()
      config.update({"activation": self.activation, "kernel_size":self.kernel_size, "dilation":self.dilation, "dropout_rate":self.dropout_rate, "batch_norm":self.batch_norm, "weighted_sum":self.weighted_sum, "l2_reg":self.l2_reg, "output_dtype":self.output_dtype})
      return config

    def build(self, input_shape):
        input_channels = int(input_shape[-1])
        conv_fun = tf.keras.layers.Conv2D

        self.conv1 = conv_fun(
            filters=input_channels,
            kernel_size=self.kernel_size,
            strides=1,
            padding='same',
            name=f"Conv1_{ker_size_to_string(self.kernel_size)}",
            dtype=self.dtype_policy,
            activation="linear",
            kernel_initializer=get_kernel_initializer(self.activation),
            kernel_regularizer=HybridThresholdL2Regularizer(directional_strength=self.l2_reg * 10, elementwise_strength=self.l2_reg) if self.l2_reg > 0 else None,
            bias_regularizer=HybridThresholdL2Regularizer(directional_strength=0, elementwise_strength=self.l2_reg) if self.l2_reg > 0 else None,
            kernel_constraint=ClipMaxValue(),
            bias_constraint = ClipMaxValue()
        )
        self.conv2 = conv_fun(
            filters=input_channels,
            kernel_size=self.kernel_size,
            dilation_rate = self.dilation,
            strides=1,
            padding='same',
            dtype=self.dtype_policy,
            name=f"Conv2_{ker_size_to_string(self.kernel_size)}",
            activation="linear",
            kernel_initializer=get_kernel_initializer(self.activation),
            kernel_regularizer=HybridThresholdL2Regularizer(directional_strength=self.l2_reg * 10, elementwise_strength=self.l2_reg) if self.l2_reg > 0 else None,
            bias_regularizer=HybridThresholdL2Regularizer(directional_strength=0, elementwise_strength=self.l2_reg) if self.l2_reg > 0 else None,
            kernel_constraint=ClipMaxValue(),
            bias_constraint = ClipMaxValue()
        )
        self.activation_layer = tf.keras.activations.get(self.activation)
        if self.dropout_rate>0:
            self.drop = tf.keras.layers.SpatialDropout2D(self.dropout_rate)
        if self.batch_norm:
            self.bn1 = tf.keras.layers.BatchNormalization(dtype='mixed_float16' if self.compute_dtype=='float16' else 'float32')
            self.bn2 = tf.keras.layers.BatchNormalization(dtype='mixed_float16' if self.compute_dtype=='float16' else 'float32')
        if self.weighted_sum:
            self.ws = WeightedSum(per_channel=True)
        super().build(input_shape)

    def call(self, input, training=None):
        x = self.conv1(input)
        if self.batch_norm:
            x = self.bn1(x, training = training)
        x = self.activation_layer(x)
        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn2(x, training = training)
        if self.dropout_rate>0:
            x = self.drop(x, training = training)
        if self.output_dtype is not None:
            x = tf.cast(x, dtype=self.output_dtype)
            input = tf.cast(input, dtype=self.output_dtype)
        else:
            input = tf.cast(input, dtype=x.dtype)
        if self.weighted_sum:
            return self.activation_layer(self.ws([input, x]))
        else:
            return self.activation_layer(input + x)


class Conv2DBNDrop(tf.keras.layers.Layer):
    def __init__(
            self,
            filters:int,
            kernel_size: int=3,
            dilation: int = 1,
            strides: int = 1,
            dropout_rate:float = 0.2,
            batch_norm : bool = True,
            activation:str = "relu",
            l2_reg:float = 0,
            output_dtype=None,
            name: str="ConvBNDrop",
            **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.activation=activation
        self.dropout_rate=dropout_rate
        self.batch_norm=batch_norm
        self.strides=strides
        self.l2_reg = l2_reg
        self.output_dtype=output_dtype

    def get_config(self):
      config = super().get_config().copy()
      config.update({"filters":self.filters, "activation": self.activation, "kernel_size":self.kernel_size, "dilation":self.dilation, "dropout_rate":self.dropout_rate, "batch_norm":self.batch_norm, "strides":self.strides, "l2_reg":self.l2_reg, "output_dtype":self.output_dtype})
      return config

    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            dilation_rate = self.dilation,
            strides=self.strides,
            padding='same',
            dtype=self.dtype_policy,
            name=f"Conv",
            activation="linear",
            kernel_initializer = get_kernel_initializer(self.activation),
            kernel_regularizer=HybridThresholdL2Regularizer(directional_strength=self.l2_reg * 10, elementwise_strength=self.l2_reg) if self.l2_reg > 0 else None,
            bias_regularizer=HybridThresholdL2Regularizer(directional_strength=0, elementwise_strength=self.l2_reg) if self.l2_reg > 0 else None,
            kernel_constraint=ClipMaxValue(),
            bias_constraint = ClipMaxValue()
        )
        self.activation_layer = tf.keras.activations.get(self.activation)
        if self.dropout_rate>0:
            self.drop = tf.keras.layers.SpatialDropout2D(self.dropout_rate)
        if self.batch_norm:
            self.bn = tf.keras.layers.BatchNormalization(dtype='mixed_float16' if self.compute_dtype=='float16' else 'float32')
        super().build(input_shape)

    def call(self, input, training=None):
        x = self.conv(input)
        if self.batch_norm:
            x = self.bn(x, training = training)
        if self.dropout_rate>0:
            x = self.drop(x, training = training)
        if self.output_dtype is not None:
            x = tf.cast(x, dtype=self.output_dtype)
        return self.activation_layer(x)


class Conv2DWithDtype(tf.keras.layers.Conv2D):
    def __init__(self, *args, l2_reg:float=0, output_dtype:str=None, **kwargs):
        self._activation = kwargs.pop('activation', None)
        self.l2_reg = l2_reg
        if "kernel_initializer" not in kwargs:
            kwargs["kernel_initializer"] = get_kernel_initializer(self._activation)
        kernel_regularizer = HybridThresholdL2Regularizer(directional_strength=self.l2_reg * 10, elementwise_strength=self.l2_reg) if self.l2_reg > 0 else kwargs.pop('kernel_regularizer', None)
        bias_regularizer = HybridThresholdL2Regularizer(directional_strength=0, elementwise_strength=self.l2_reg) if self.l2_reg > 0 else kwargs.pop('bias_regularizer', None)
        kernel_constraint = ClipMaxValue()
        bias_constraint = ClipMaxValue()
        super().__init__(*args, activation=None, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint, **kwargs)
        self.output_dtype = output_dtype
        self.activation = None  # Will be set in build()

    def build(self, input_shape):
        super().build(input_shape)
        if self._activation is not None:
            self.activation = tf.keras.activations.get(self._activation)

    def call(self, inputs):
        output = super().call(inputs)
        if self.output_dtype is not None:
            output = tf.cast(output, dtype=self.output_dtype)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        config = super().get_config()
        config.pop("kernel_regularizer", None)
        config.pop("bias_regularizer", None)
        config.pop("kernel_constraint", None)
        config.pop("bias_constraint", None)
        config.update({
            'output_dtype': self.output_dtype,
            'l2_reg':self.l2_reg,
            'activation': self._activation if isinstance(self._activation, str) else tf.keras.activations.serialize(self._activation)
        })
        return config


class Conv2DTransposeBNDrop(tf.keras.layers.Layer):
    def __init__(
            self,
            filters:int,
            kernel_size: int=4,
            strides: int = 2,
            dropout_rate:float = 0,
            batch_norm : bool = False,
            activation:str = "relu",
            l2_reg:float = 0,
            output_dtype=None,
            name: str="ResConv2DTransposeBNDrop",
            **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation=activation
        self.dropout_rate=dropout_rate
        self.batch_norm=batch_norm
        self.strides=strides
        self.l2_reg=l2_reg
        self.output_dtype=output_dtype

    def get_config(self):
      config = super().get_config().copy()
      config.update({"filters":self.filters, "activation": self.activation, "kernel_size":self.kernel_size, "dropout_rate":self.dropout_rate, "batch_norm":self.batch_norm, "strides":self.strides, "l2_reg":self.l2_reg, "output_dtype":self.output_dtype})
      return config

    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv2DTranspose(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding='same',
            dtype=self.dtype_policy,
            activation="linear",
            kernel_initializer=get_kernel_initializer(self.activation),
            kernel_regularizer=HybridThresholdL2Regularizer(directional_strength=self.l2_reg * 10, elementwise_strength=self.l2_reg) if self.l2_reg > 0 else None,
            bias_regularizer = HybridThresholdL2Regularizer(directional_strength=0, elementwise_strength=self.l2_reg) if self.l2_reg > 0 else None,
            kernel_constraint=ClipMaxValue(),
            bias_constraint = ClipMaxValue(),
            name=f"tConv{ker_size_to_string(self.kernel_size)}",
        )
        self.activation_layer = tf.keras.activations.get(self.activation)
        if self.dropout_rate>0:
            self.drop = tf.keras.layers.SpatialDropout2D(self.dropout_rate, name=f"Dropout")
        if self.batch_norm:
            #self.bn = MockBatchNormalization(name = f"/MockBatchNormalization")
            self.bn = tf.keras.layers.BatchNormalization(name = f"BatchNormalization", dtype='mixed_float16' if self.compute_dtype=='float16' else 'float32')
        super().build(input_shape)

    def call(self, input, training=None):
        x = self.conv(input)
        if self.batch_norm:
            x = self.bn(x, training = training)
        if self.dropout_rate>0:
            x = self.drop(x, training = training)
        if self.output_dtype is not None:
            x = tf.cast(x, dtype=self.output_dtype)
        return self.activation_layer(x)


class Conv2DTransposeWithDtype(tf.keras.layers.Conv2DTranspose):
    def __init__(self, *args, output_dtype=None, l2_reg:float=0, **kwargs):
        self._activation = kwargs.pop('activation', None)
        self.l2_reg = l2_reg
        if "kernel_initializer" not in kwargs:
            kwargs["kernel_initializer"] = get_kernel_initializer(self._activation)
        kernel_regularizer = HybridThresholdL2Regularizer(directional_strength=self.l2_reg * 10, elementwise_strength=self.l2_reg) if self.l2_reg > 0 else kwargs.pop('kernel_regularizer', None)
        bias_regularizer = HybridThresholdL2Regularizer(directional_strength=0, elementwise_strength=self.l2_reg) if self.l2_reg > 0 else kwargs.pop('bias_regularizer', None)
        kernel_constraint = ClipMaxValue()
        bias_constraint = ClipMaxValue()
        super().__init__(*args, activation=None, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint, **kwargs)
        self.output_dtype = output_dtype
        self.activation = None  # Will be set in build()

    def build(self, input_shape):
        super().build(input_shape)
        if self._activation is not None:
            self.activation = tf.keras.activations.get(self._activation)

    def call(self, inputs):
        output = super().call(inputs)
        if self.output_dtype is not None:
            output = tf.cast(output, dtype=self.output_dtype)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        config = super().get_config()
        config.pop("kernel_regularizer", None)
        config.pop("bias_regularizer", None)
        config.pop("kernel_constraint", None)
        config.pop("bias_constraint", None)
        config.update({
            'output_dtype': self.output_dtype,
            'l2_reg':self.l2_reg,
            'activation': self._activation if isinstance(self._activation, str) else tf.keras.activations.serialize(self._activation)
        })
        return config


class UpSampling2DWithDtype(tf.keras.layers.UpSampling2D):
    def __init__(self, *args, output_dtype=None, **kwargs):
        super(UpSampling2DWithDtype, self).__init__(*args, **kwargs)
        self.output_dtype = output_dtype

    def call(self, inputs):
        output = super(UpSampling2DWithDtype, self).call(inputs)
        if self.output_dtype is not None:
            output = tf.cast(output, dtype=self.output_dtype)
        return output

    def get_config(self):
        config = super(UpSampling2DWithDtype, self).get_config()
        config.update({'output_dtype': self.output_dtype})
        return config


class WeightedSum(tf.keras.layers.Layer):
    def __init__(self, per_channel=True, **kwargs):
        super().__init__(**kwargs)
        self.per_channel = per_channel

    def build(self, input_shape):
        assert isinstance(input_shape, (list, tuple)), "input should be a list or tuple of tensor"
        for i, shape in enumerate(input_shape[1:]):
            assert len(shape)==len(input_shape[0]), "ranks differ"
            for j in range(1, len(input_shape[0])):
                assert shape[j] == input_shape[0][j], f"all shape should be equal. Shape at {i+2}/{len(input_shape)} is {shape} which differs from {input_shape[0]}"
        super().build(input_shape)
        self.weight = self.add_weight(
            name="weight",
            shape=(input_shape[0][-1], len(input_shape)) if self.per_channel else (len(input_shape),),
            dtype=self.dtype_policy,
            initializer=tf.constant_initializer(1./len(input_shape)), # "ones"
            trainable=True
        )

    def get_config(self):
        config = super().get_config().copy()
        config.update({"per_channel":self.per_channel})
        return config

    def call(self, inputs):
        weights = self.weight[tf.newaxis, tf.newaxis, tf.newaxis]
        if not self.per_channel:
            weights = weights[tf.newaxis]
        mul = tf.math.multiply(tf.stack(inputs, axis = -1), weights)
        return tf.math.reduce_sum(mul, axis=-1, keepdims = False)


def ker_size_to_string(ker_size, dims=2):
    if not isinstance(ker_size, (list, tuple)):
        ker_size = ensure_multiplicity(dims, ker_size)
    return 'x'.join([str(k) for k in ker_size])


def _standardize_weight(weight, gain, eps):
    mean = tf.math.reduce_mean(weight, axis=(0, 1, 2), keepdims=True)
    var = tf.math.reduce_mean(tf.math.square(weight-mean), axis=(0, 1, 2), keepdims=True)
    fan_in = np.prod(weight.shape[:-1])
    #scale = tf.math.sqrt(tf.math.maximum(var * fan_in, eps))
    scale = tf.math.sqrt(var * fan_in + eps)
    weight = (weight - mean) / scale
    if gain is not None:
        weight = weight * gain
    return weight


def get_gamma(activation):
    if activation.lower()=="relu":
        return 1.
        return 1. / ( (0.5 * (1 - 1 / np.pi)) ** 0.5)
    elif activation.lower()=="sliu":
        return .5595
    elif activation.lower()=="linear":
        return 1.
    else:
        return 1.
        #raise ValueError(f"activation {activation} not supported yet")


class ScheduledDropout(tf.keras.layers.Layer):
    """
    (Spatial) dropout layer with scheduled rate based on training progress.
    """

    def __init__(self,
                 rate: float,  # target rate (min_rate)
                 max_rate: float,  # rate at training start
                 min_progress: float = 0.0,
                 max_progress: float = 1.0,  # When this layer reaches rate
                 spatial: bool = True,  # Use spatial dropout for images
                 power_law: float = 1.0,
                 seed=None,  # Optional: for reproducibility
                 **kwargs):
        super().__init__(autocast=False, **kwargs)
        self.min_rate = rate
        self.max_rate = max_rate
        self.min_progress = min_progress
        self.max_progress = max_progress
        self.spatial = spatial
        self.power_law = power_law
        self.seed = seed

        # Training progress variable [0, 1] - set by callback
        self.progress = None

    def build(self, input_shape):
        super().build(input_shape)

        # Create progress variable (updated by callback)
        self.progress = self.add_weight(
            name='progress',
            shape=(),
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=False,
            dtype=tf.float32
        )
        self.is_3D = self.spatial and len(input_shape) == 5
        # Validate input shape for spatial dropout
        if self.spatial and len(input_shape) not in [4, 5]:
            raise ValueError(
                f"Spatial dropout requires 4D (batch, height, width, channels) or "
                f"5D (batch, depth, height, width, channels) input, got shape {input_shape}"
            )

    def call(self, inputs, training=None):
        if not training:
            return inputs

        current_rate = self.get_current_rate()
        current_rate = tf.cast(current_rate, inputs.dtype)
        if self.spatial:
            # Use tf.nn.dropout with spatial noise_shape for spatial dropout
            input_shape = tf.shape(inputs)

            if not self.is_3D:
                # 2D convolutions: drop (batch, 1, 1, channels)
                noise_shape = [input_shape[0], 1, 1, input_shape[3]]
            else:
                # 3D convolutions: drop (batch, 1, 1, 1, channels)
                noise_shape = [input_shape[0], 1, 1, 1, input_shape[4]]

            return tf.nn.dropout(
                inputs,
                rate=current_rate,
                noise_shape=noise_shape,
                seed=self.seed
            ) * (1 - current_rate) # unscaled dropout

        else:
            # Regular dropout (no noise_shape = element-wise dropout)
            return tf.nn.dropout(
                inputs,
                rate=current_rate,
                seed=self.seed
            ) * (1 - current_rate)


    def set_progress(self, progress_value):
        """Set global training progress [0, 1] - called by callback"""
        self.progress.assign(tf.clip_by_value(progress_value, 0.0, 1.0))

    def get_current_rate(self):
        """Get current dropout rate"""
        if self.progress is None:
            return self.min_rate

        # Calculate layer-specific progress
        # Cast to float32 explicitly to handle mixed precision
        progress_val = tf.cast(self.progress, tf.float32) - tf.cast(self.min_progress, tf.float32)
        progress_norm = tf.cast(self.max_progress - self.min_progress, tf.float32)
        layer_progress = tf.maximum(0.0, tf.minimum(1.0, progress_val / progress_norm))

        # Interpolate from max_rate to min_rate
        max_rate_val = tf.cast(self.max_rate, tf.float32)
        min_rate_val = tf.cast(self.min_rate, tf.float32)
        current_rate = max_rate_val - (max_rate_val - min_rate_val) * tf.math.pow(layer_progress, self.power_law)

        return current_rate

    def get_config(self):
        config = super().get_config()
        config.update({
            "rate": float(self.min_rate),
            "max_rate": float(self.max_rate),
            "min_progress": float(self.min_progress),
            "max_progress": float(self.max_progress),
            "spatial": self.spatial,
            "power_law": self.power_law,
            "seed": self.seed
        })
        return config


## Embeddings
class FrameDistanceEmbedding(tf.keras.layers.Layer):
    def __init__(self, input_dim:int, output_dim:int, frame_prev_idx:list, frame_next_idx:list, offset:int = 0, l2_reg:float=1e-5 , name:str="FrameDistanceEmbedding", **kwargs):
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.frame_next_idx=frame_next_idx
        self.frame_prev_idx=frame_prev_idx
        self.offset=offset
        self.l2_reg=l2_reg
        assert len(frame_prev_idx) == len(frame_next_idx)
        self.embedding=None
        super().__init__(name=name, **kwargs)

    def get_config(self):
      config = super().get_config().copy()
      config.update({"input_dim": self.input_dim, "output_dim":self.output_dim, "frame_prev_idx":self.frame_prev_idx, "frame_next_idx":self.frame_next_idx, "offset":self.offset, "l2_reg":self.l2_reg})
      return config

    def build(self, input_shape):
        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            embeddings_regularizer=HybridThresholdL2Regularizer(directional_strength=self.l2_reg * 10, elementwise_strength=self.l2_reg, axis=1) if self.l2_reg > 0 else None,
            embeddings_constraint=ClipMaxValue(),
            dtype=self.dtype_policy
        )
        super().build(input_shape)

    def call(self, frame_index): # (B, 1, 1, FW)
        offset = tf.cast(self.offset, tf.int32)
        frame_distance = tf.cast( tf.gather(frame_index[:, 0, 0], self.frame_next_idx, axis=-1) - tf.gather(frame_index[:, 0, 0], self.frame_prev_idx, axis=-1), tf.int32 ) + offset # (B, N)
        frame_distance_emb = self.embedding(frame_distance) # (B, N, C)
        frame_distance_emb = tf.transpose(frame_distance_emb, perm=[1, 0, 2]) # (N, B, C)
        frame_distance_emb = tf.reshape(frame_distance_emb, [-1, 1, 1, self.output_dim]) # ( N x B, 1, 1, C )
        return frame_distance_emb


def sinusoidal_temporal_encoding(distances, embedding_dim, dtype="float32"):
    """
    Compute sinusoidal encoding for temporal distances.
    Similar to Transformer positional encoding but for temporal distance.

    Args:
        distances: [..., T] tensor of temporal distances from center
        embedding_dim: dimension of encoding

    Returns:
        encoding: [..., T, embedding_dim]
    """
    distances = tf.cast(distances, tf.float32)
    distances = tf.expand_dims(distances, -1)  # [..., T, 1]

    # Position encoding dimensions
    dim_idx = tf.range(embedding_dim, dtype=tf.float32)
    dim_idx = dim_idx[None, None, :]  # [1, 1, embedding_dim]

    # Compute frequencies
    freq = 1.0 / tf.pow(10000.0, (2 * (dim_idx // 2)) / embedding_dim)

    # Compute encoding
    angle = distances * freq  # [..., T, embedding_dim]

    # Apply sin to even indices, cos to odd
    encoding = tf.where(
        tf.equal(dim_idx % 2, 0),
        tf.sin(angle),
        tf.cos(angle)
    )

    return tf.cast(encoding, dtype=dtype)


class RelativeTemporalEmbedding(tf.keras.layers.Layer):
    def __init__(self, embedding_dim:int, hidden_dim:int=[128, 128], l2_reg=1e-5, **kwargs):
        self.embedding_dim = embedding_dim
        if not isinstance(hidden_dim, (tuple, list)):
            hidden_dim = [hidden_dim]
        self.hidden_dims = hidden_dim
        self.l2_reg = l2_reg
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.n_frames = input_shape[-1]
        layers = []
        for i, h in enumerate(self.hidden_dims):
            activation = "tanh" # TODO : 'silu' if i==0 else "tanh"
            layers.append(tf.keras.layers.Dense(
                units=h,
                activation=activation,
                dtype=self.dtype_policy,
                kernel_initializer=get_kernel_initializer(activation),
                bias_initializer=get_kernel_initializer(activation),
                kernel_regularizer=HybridThresholdL2Regularizer(directional_strength=self.l2_reg * 10, elementwise_strength=self.l2_reg) if self.l2_reg > 0 else None,
                bias_regularizer=HybridThresholdL2Regularizer(directional_strength=0, elementwise_strength=self.l2_reg) if self.l2_reg > 0 else None,
                kernel_constraint=ClipMaxValue(),
                bias_constraint=ClipMaxValue(),
                name=f'{self.name}/hidden{i}'
            ))
        layers.append(tf.keras.layers.Dense(
            units=self.embedding_dim,
            activation=None,
            dtype=self.dtype_policy,
            kernel_initializer='glorot_uniform',
            bias_initializer='glorot_uniform',
            kernel_regularizer=HybridThresholdL2Regularizer(directional_strength=self.l2_reg * 10, elementwise_strength=self.l2_reg) if self.l2_reg > 0 else None,
            bias_regularizer=HybridThresholdL2Regularizer(directional_strength=0,  elementwise_strength=self.l2_reg) if self.l2_reg > 0 else None,
            kernel_constraint=ClipMaxValue(),
            bias_constraint=ClipMaxValue(),
            name=f'{self.name}/add'
        ))
        self.add_embedding = tf.keras.Sequential(layers, name="embedding")
        super().build(input_shape)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "hidden_dims": self.hidden_dims,
            "embedding_dim": self.embedding_dim,
            "l2_reg": self.l2_reg
        })
        return config

    def call(self, distances):
        """
        Args:
            distances: (B, T) - temporal distances (can be negative for backward)

        Returns:
            additive embedding (B, T, D)
        """
        distances = tf.reshape(tf.cast(distances, self.compute_dtype), shape=[-1, self.n_frames, 1]) # B, T, 1
        return self.add_embedding(distances)


class ConcatenateWithDtype(InferenceLayer, tf.keras.layers.Concatenate):
    def __init__(self, *args, inference_idx=None, output_dtype:str=None, autocast=False, **kwargs):
        self.output_dtype = output_dtype
        self.inference_idx = inference_idx
        super().__init__(*args, autocast=autocast, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        if self.inference_mode:
            if isinstance(self.inference_idx, int):
                output = inputs[self.inference_idx]
            elif self.inference_idx is not None:
                inputs = [inputs[i] for i in self.inference_idx]
                output = super().call(inputs)
            else:
                output = super().call(inputs)
        else:
            output = super().call(inputs)
        if self.output_dtype is not None:
            output = tf.cast(output, dtype=self.output_dtype)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({'output_dtype': self.output_dtype, "inference_idx": self.inference_idx})
        return config


# Gradient manipulation
class ResidualGradientLimiter(tf.keras.layers.Layer):
    """
    Limits skip path gradients relative to main path gradients.
    Only reduces skip gradients (never amplifies them).

    Usage:
        limiter = ResidualGradientLimiter()
        limited_skip, main = limiter(x, training=True)
    """

    def __init__(self,
                 max_ratio: float = 1.0,
                 epsilon: float = 1e-5,
                 **kwargs):
        super().__init__(autocast=False, **kwargs)
        self.max_ratio = max_ratio
        self.epsilon = epsilon

    @tf.custom_gradient
    def _limit_gradients(self, x):
        skip, main = x

        def grad(dy_skip, dy_main):
            main_grad_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.cast(dy_main, tf.float32) ** 2), self.epsilon))
            if dy_skip is None:
                return dy_skip, dy_main
            skip_grad_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.cast(dy_skip, tf.float32) ** 2), self.epsilon))

            scale_factor = tf.minimum(
                self.max_ratio * main_grad_norm / tf.maximum(skip_grad_norm, self.epsilon),
                1.0
            )
            return dy_skip * tf.cast(scale_factor, dy_skip.dtype), dy_main

        return [skip, main], grad

    def call(self, inputs, training=None):
        """
        Args:
            inputs: single tensor to be split into skip and main paths
            training: whether in training mode

        Returns:
            list of [limited_skip, main]
        """
        # Split the input into two paths using tf.identity
        res = tf.identity(inputs, name=f"{self.name}_residual")
        x = inputs

        if not training:
            return [res, x]

        return self._limit_gradients([res, x])

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_ratio": float(self.max_ratio),
            "epsilon": float(self.epsilon),
        })
        return config


def _global_norm(g, epsilon):
    return tf.sqrt(tf.maximum(
        tf.reduce_sum(tf.cast(g, tf.float32) ** 2),
        epsilon,
    ))


def _unitwise_norm(g, epsilon):
    """
    Norm computed per output neuron/filter, matching the axis convention of
    AGC's unitwise_norm so the two are consistent.

    Shape conventions:
      <= 1D  (bias / scalar)   : global norm, axis=None
      2D     (Dense IO)        : norm over input dim, axis=0  -> shape [O]
      3D     (multihead linear): norm over input dims, axis=0 -> shape [H, O]
      4D     (Conv2D HWIO)     : norm over spatial+input, axis=[0,1,2] -> [O]
      5D     (Conv3D HWDIO)    : norm over spatial+input, axis=[0,1,2,3] -> [O]

    Returns a tensor broadcastable back onto g.
    """
    g = tf.cast(g, tf.float32)
    shape = g.shape.as_list()
    rank = len(shape)
    if rank <= 1:
        return _global_norm(g, epsilon)
    elif rank == 2:
        axis, keepdims = 0, True          # -> [1, O]
    elif rank == 3:
        axis, keepdims = 0, True          # -> [1, H, O]
    elif rank == 4:
        axis, keepdims = [0, 1, 2], True  # -> [1, 1, 1, O]
    elif rank == 5:
        axis, keepdims = [0, 1, 2, 3], True  # -> [1, 1, 1, 1, O]
    else:
        raise ValueError(f"Unsupported gradient rank {rank}, shape {shape}")
    return tf.sqrt(tf.maximum(
        tf.reduce_sum(g ** 2, axis=axis, keepdims=keepdims),
        epsilon,
    ))


class MultiHeadGradientLimiter(tf.keras.layers.Layer):
    """
    Balances gradients from multiple task heads flowing into a shared encoder.

    During the forward pass this is a pure identity on all branches — no effect
    on activations or inference.

    In the backward pass it works as an amplification cap:
       If the norm of the gradient sum exceeds the norm
       of the strongest individual head gradient, the sum is scaled down to
       that maximum norm.
       When unitwise=True this is applied per output neuron/filter, so heads
       that update disjoint neurons are not incorrectly penalised.
       When unitwise=False a single scalar norm is used (cheaper, sufficient
       when heads share most neurons).

    Args:
        num_heads:   Number of task heads (1 to N). With a single head the
                     layer is a no-op.
        unitwise:    If True, the amplification cap is computed per output
                     neuron/filter (more accurate when heads are specialised
                     on disjoint neurons, but slightly more expensive).
                     If False, a single global scalar norm is used.
        epsilon:     Floor for norm computations to avoid division by zero.
        monitor:     If True, tf.print per-head cosine similarities and the
                     amplification scale factor each step.

    Usage:
        balancer = MultiHeadGradientBalancer(num_heads=3, unitwise=True)
        branches = balancer(encoder_output, training=training)
        feat_seg, feat_track = branches[0], branches[1]
    """

    def __init__(
        self,
        num_heads: int,
        unitwise: bool = False,
        epsilon: float = 1e-6,
        monitor: bool = False,
        **kwargs,
    ):
        super().__init__(autocast=False, **kwargs)
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")
        self.num_heads = num_heads
        self.unitwise = unitwise
        self.epsilon = epsilon
        self.monitor = monitor

    @tf.custom_gradient
    def _gradient_fn(self, *branches):
        def grad(*upstream_grads):
            grads = list(upstream_grads)
            epsilon = self.epsilon
            unitwise = self.unitwise
            monitor = self.monitor

            if self.num_heads == 1:
                return grads

            # Collect non-None gradients; sanitise non-finite values in-graph.
            valid = [ (i, tf.cast(g, tf.float32))  for i, g in enumerate(grads)  if g is not None ]
            if len(valid) <= 1:
                return grads

            indices, f_grads = zip(*valid)
            f_grads = [ tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)) for g in f_grads ]
            f_grads = list(f_grads)

            # ── 1. Natural sum — reference direction ───────────────────────
            g_sum = tf.add_n(f_grads)

            # ── 2. Amplification cap ───────────────────────────────────────
            if unitwise:
                # Per output-neuron/filter cap.
                # For each position, the sum norm must not exceed the maximum
                # individual head norm at that position.
                # head_norms_uw shape: broadcastable onto g (e.g. [1,1,1,C_out])
                head_norms_uw = [_unitwise_norm(g, epsilon) for g in f_grads]
                g_sum_norm_uw = _unitwise_norm(g_sum, epsilon)
                m_cap_uw = tf.reduce_max( tf.stack(head_norms_uw, axis=0), axis=0 )  # same shape as head_norms_uw[i]
                # Elementwise scale, clipped to [0, 1] — never amplify.
                scale = tf.minimum(m_cap_uw / g_sum_norm_uw, 1.0)
                if monitor:
                    tf.print(
                        "[MultiHeadGradientBalancer] unitwise amp_scale min/max:",
                        tf.reduce_min(scale), tf.reduce_max(scale),
                    )
            else:
                # Single scalar cap.
                g_sum_norm_global = _global_norm(g_sum, epsilon)
                head_norms_global = [_global_norm(g, epsilon) for g in f_grads]
                m_cap = tf.reduce_max(tf.stack(head_norms_global))
                scale = tf.minimum(m_cap / g_sum_norm_global, 1.0)
                if monitor:
                    tf.print(
                        "[MultiHeadGradientBalancer] amp_scale:", scale,
                        "g_sum_norm:", g_sum_norm_global,
                        "m_cap:", m_cap,
                    )

            # ── 4. Apply scale and cast back to original dtype ─────────────
            out_grads = list(grads)
            for i, g_new in zip(indices, f_grads):
                out_grads[i] = tf.cast(g_new * scale, grads[i].dtype)

            return out_grads

        return list(branches), grad

    def call(self, inputs, training=None):
        """
        Args:
            inputs:   Single tensor — the shared encoder output.
            training: If False, returns plain tf.identity copies (no-op).

        Returns:
            Plain Python list of `num_heads` tensors equal to `inputs`.
            Always index explicitly: branches[0], branches[1], ...
        """
        branches = [tf.identity(inputs) for _ in range(self.num_heads)]
        if self.num_heads == 1 or not training:
            return branches
        return self._gradient_fn(*branches)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "unitwise":  self.unitwise,
            "epsilon":   float(self.epsilon),
            "monitor":   self.monitor,
        })
        return config


class ScheduledGradientWeight(tf.keras.layers.Layer):
    """
    Layer that applies scheduled gradient weighting to skip connections.
    Forward pass: unchanged (information flows normally)
    Backward pass: gradients are scaled by a scheduled weight
    """

    def __init__(self,
                 min_weight: float = 0.0,  # Initial gradient weight (training start)
                 max_weight: float = 1.0,  # Final gradient weight (target)
                 min_progress: float = 0.0,
                 max_progress: float = 1.0,  # When this layer reaches max_weight
                 power_law: float = 1.0,
                 **kwargs):
        super().__init__(autocast=False, **kwargs)
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.min_progress = min_progress
        self.max_progress = max_progress
        self.power_law=power_law

        # Training progress variable [0, 1] - set by callback
        self.progress = None

    def build(self, input_shape):
        super().build(input_shape)

        # Create progress variable (updated by callback)
        # Initialize to 0.0 (training start)
        self.progress = self.add_weight(
            name='progress',
            shape=(),
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=False,
            dtype=tf.float32
        )

    @tf.custom_gradient
    def _weight_gradient(self, x, weight):
        """
        Forward pass: return input unchanged
        Backward pass: scale gradients by weight
        """

        def grad(dy):
            # Handle different gradient types
            if isinstance(dy, tuple):
                return tuple(y * tf.cast(weight, y.dtype) for y in dy), None
            elif isinstance(dy, list):
                return [y * tf.cast(weight, y.dtype) for y in dy], None
            else:
                return dy * tf.cast(weight, dy.dtype), None

        return x, grad

    def call(self, inputs, training=None):
        if not training:
            return inputs
        current_weight = self.get_current_weight()
        return self._weight_gradient(inputs, current_weight)

    def set_progress(self, progress_value):
        """Set global training progress [0, 1] - called by callback"""
        self.progress.assign(tf.clip_by_value(progress_value, 0.0, 1.0))

    def get_current_weight(self):
        """Get current gradient weight"""
        if self.progress is None:
            return self.max_weight

        # Calculate layer-specific progress
        # Cast to float32 explicitly to handle mixed precision
        progress_val = tf.cast(self.progress, tf.float32) -  tf.cast(self.min_progress, tf.float32)
        progress_norm = tf.cast(self.max_progress - self.min_progress, tf.float32)
        layer_progress = tf.maximum(0.0, tf.minimum(1.0, progress_val / progress_norm))

        # Interpolate from min_weight to max_weight (inverse of dropout)
        min_weight_val = tf.cast(self.min_weight, tf.float32)
        max_weight_val = tf.cast(self.max_weight, tf.float32)
        current_weight = min_weight_val + (max_weight_val - min_weight_val) * tf.math.pow(layer_progress, self.power_law)

        return current_weight

    def get_config(self):
        config = super().get_config()
        config.update({
            "min_weight": float(self.min_weight),
            "max_weight": float(self.max_weight),
            "min_progress": float(self.min_progress),
            "max_progress": float(self.max_progress),
            "power_law": float(self.power_law)
        })
        return config


class StopGradient(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, input, training=None):
        if not training:
            return input
        return tf.stop_gradient( input, name=self.name )


class WeigthedGradient(tf.keras.layers.Layer):
    def __init__(self, weight, **kwargs):
        super().__init__(**kwargs)
        self.weight = weight

    def call(self, x):
        return self.op(x)

    @tf.custom_gradient
    def op(self, x):
        def grad(*dy):
            if isinstance(dy, tuple): #and len(dy)>1
                return (y * self.weight for y in dy)
            else:
                return dy * self.weight
        return x, grad



def get_grad_weight_fun(weight):
    @tf.custom_gradient
    def wgrad(x):
        def grad(dy):
            if isinstance(dy, tuple): #and len(dy)>1
                #print(f"gradient is tuple of length: {len(dy)}")
                return (y * weight for y in dy)
            elif isinstance(dy, list):
                #print(f"gradient is list of length: {len(dy)}")
                return [y * weight for y in dy]
            else:
                return dy * weight
        return x, grad
    return wgrad


def get_print_grad_fun(message):
    @tf.custom_gradient
    def op(x):
        def grad(dy):
                g_flat = tf.reshape(tf.math.abs(dy), [-1])
                g_flat = tf.boolean_mask(g_flat, tf.greater(g_flat, 0))
                print(f"{message} gradient shape: {dy.shape}, non-null: {100 * g_flat.shape[0]/tf.size(dy)}%, value: {tf.math.reduce_mean(g_flat)}")
                return dy
        return x, grad
    return op


class LogGradientMagnitude(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(autocast=False, **kwargs)
        self.grad_accum = None

    def build(self, input_shape):
        self.grad_accum = self.add_weight(
            name='grad_accum',
            shape=(),
            initializer=tf.keras.initializers.Constant(0.),
            trainable=False,
            dtype=tf.float32
        )

    def call(self, x):
        return self.op(x)

    @tf.custom_gradient
    def op(self, x):
        def grad(dy):
            self.grad_accum.assign_add(tf.sqrt(tf.reduce_sum(tf.square(tf.cast(dy, tf.float32)))))
            return dy
        return x, grad

    def get_value(self):
        value = float(self.grad_accum.numpy())
        self.grad_accum.assign(0.0)
        return value

# util class to avoid that a tf.Variable is saved into the model weights
class HideVariableWrapper:
    def __init__(self, value:tf.Variable):
        self.value = value

    def assign(self, value, **kwargs):
        return self.value.assign(value, **kwargs)

    def assign_add(self, delta, **kwargs):
        return self.value.assign_add(delta, **kwargs)

    def assign_sub(self, delta, **kwargs):
        return self.value.assign_sub(delta, **kwargs)

    def __getitem__(self, item):
        return self.value[item]

    def get_shape(self):
        return self.value.get_shape()

    def __len__(self):
        return self.shape[0]

    def __tensor__(self, dtype=None):
        if dtype is not None:
            return tf.cast(self.value, dtype)
        else:
            return self.value

    @property
    def shape(self):
        return self.value.shape

    @property
    def dtype(self):
        return self.value.dtype

    def numpy(self):
        return self.value.numpy()

    def __array__(self):
        return self.value.numpy()


@tf.keras.utils.register_keras_serializable(package='Custom', name='HybridThresholdL2Regularizer')
class HybridThresholdL2Regularizer(tf.keras.regularizers.Regularizer):
    def __init__(self,
                 directional_threshold:float=2.,
                 directional_strength:float=1e-3,
                 elementwise_threshold:float=10.0,
                 elementwise_strength:float=1e-4,
                 axis=None):
        self.directional_threshold = directional_threshold
        self.directional_strength = directional_strength
        self.elementwise_threshold = elementwise_threshold
        self.elementwise_strength = elementwise_strength
        self.axis=axis

    def __call__(self, weights):
        norm_axis = tuple(range(weights.shape.rank - 1)) if self.axis is None else (self.axis if isinstance(self.axis, (tuple, list)) else (self.axis,) )
        if self.directional_strength > 0 and len(norm_axis) > 0:
            norms = tf.sqrt(tf.reduce_sum(tf.square(weights), axis=norm_axis))
            directional_excess = tf.nn.relu(norms - self.directional_threshold)
            directional_penalty = tf.reduce_sum(tf.square(directional_excess))
        else:
            directional_penalty = 0
        if self.elementwise_strength > 0:
            abs_weights = tf.abs(weights)
            elementwise_excess = tf.nn.relu(abs_weights - self.elementwise_threshold)
            elementwise_penalty = tf.reduce_sum(tf.square(elementwise_excess))
        else:
            elementwise_penalty = 0
        return self.directional_strength * directional_penalty + self.elementwise_strength * elementwise_penalty

    def get_config(self):
        return {
            "directional_threshold":self.directional_threshold,
            "directional_strength":self.directional_strength,
            "elementwise_threshold":self.elementwise_threshold,
            "elementwise_strength":self.elementwise_strength,
            "axis":self.axis
        }

@tf.keras.utils.register_keras_serializable(package='Custom', name='ClipMaxValue')
class ClipMaxValue(tf.keras.constraints.Constraint):
    """Clips the weights to not exceed a maximum value (typically lower than 65504.0 for FP16 compatibility)."""

    def __init__(self, max_value=60000.0):
        self.max_value = max_value

    def __call__(self, w):
        return tf.clip_by_value(w, -self.max_value, self.max_value)

    def get_config(self):
        return {'max_value': self.max_value}


def get_kernel_initializer(activation:str):
    if activation is None:
        return "glorot_uniform"
    activation = activation.lower()
    if "elu" in activation or "silu" in activation or activation == "mish":
        return "he_normal"
    else:
        return "glorot_uniform"