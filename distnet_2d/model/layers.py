from tensorflow import pad
from ..utils.helpers import ensure_multiplicity
import tensorflow as tf
import numpy as np

class StopGradient(tf.keras.layers.Layer):
    def __init__(self, name:str="StopGradient"):
        super().__init__(name=name)

    def call(self, input):
        return tf.stop_gradient( input, name=self.name )

class NConvToBatch2D(tf.keras.layers.Layer):
    def __init__(self, n_conv:int, inference_conv_idx:int, filters:int, compensate_gradient:bool = False, name: str="NConvToBatch2D"):
        super().__init__(name=name)
        self.n_conv = n_conv
        self.filters = filters
        self.inference_mode=False
        self.inference_conv_idx=inference_conv_idx
        self.compensate_gradient=compensate_gradient

    def get_config(self):
        config = super().get_config().copy()
        config.update({"n_conv": self.n_conv, "filters":self.filters, "compensate_gradient":self.compensate_gradient, "inference_conv_idx":self.inference_conv_idx})
        return config

    def build(self, input_shape):
        self.convs = [
            tf.keras.layers.Conv2D(filters=self.filters, kernel_size=1, padding='same', activation="relu", name=f"Conv_{i}")
        for i in range(self.n_conv)]

        if self.compensate_gradient and self.n_conv>1:
            self.grad_fun = get_grad_weight_fun(float(self.n_conv))
            self.grad_fun_inv = get_grad_weight_fun(1./self.n_conv)
        else:
            self.grad_fun = None
            self.grad_fun_inv = None
        super().build(input_shape)

    def call(self, input): # (B, Y, X, F)
        if self.inference_mode: # only produce one output
            return self.convs[self.inference_conv_idx](input)
        # input = get_print_grad_fun(f"{self.name} before split")(input)
        if self.grad_fun_inv is not None:
            input = self.grad_fun_inv(input)

        inputs = [conv(input) for conv in self.convs] # N x (B, Y, X, F)
        # inputs[0] = get_print_grad_fun(f"{self.name} before concat")(inputs[0])
        output = tf.concat(inputs, axis = 0) # (N x B, Y, X, F)
        if self.grad_fun is not None:
            output = self.grad_fun(output) # compensate gradients to have same level in
        # output = get_print_grad_fun(f"{self.name} after concat")(output)
        return output

class SelectFeature(tf.keras.layers.Layer):
    def __init__(self, inference_conv_idx:int, name: str="SelectFeature"):
        super().__init__(name=name)
        self.inference_mode=False
        self.inference_conv_idx=inference_conv_idx

    def get_config(self):
        config = super().get_config().copy()
        config.update({"inference_conv_idx":self.inference_conv_idx})
        return config

    def call(self, input): # (N x B, Y, X, F), N x (B, Y, X, F)
        input_concat, input_split = input
        if self.inference_mode: # only produce one output
            return input_split[self.inference_conv_idx]
        else:
            return input_concat

class ChannelToBatch(tf.keras.layers.Layer):
    def __init__(self, compensate_gradient:bool = False, name: str="ChannelToBatch"):
        self.compensate_gradient=compensate_gradient
        super().__init__(name=name)

    def get_config(self):
        config = super().get_config().copy()
        config.update({"compensate_gradient": self.compensate_gradient})
        return config

    def build(self, input_shape):
        self.rank = len(input_shape.as_list())
        self.perm = [self.rank-1, 0] + [i for i in range(1, self.rank-1)]
        if self.compensate_gradient:
            self.grad_fun = get_grad_weight_fun(float(input_shape[-1]))
        super().build(input_shape)

    def call(self, input): # (B, Y, X, C)
        shape = tf.shape(input)
        target_shape  = tf.concat([[-1], [shape[i] for i in range(1, self.rank-1)]], 0) if self.rank>2 else [-1]
        input = tf.transpose(input, perm=self.perm) # (C, B, Y, X)
        input = tf.reshape(input, target_shape) # (C x B, Y, X)
        input = tf.expand_dims(input, -1) # (C x B, Y, X, 1)
        if self.compensate_gradient:
            input = self.grad_fun(input)
        return input

class SplitBatch(tf.keras.layers.Layer):
    def __init__(self, n_splits:int, compensate_gradient:bool = False, name:str="SplitBatch2D"):
        self.n_splits=n_splits
        self.compensate_gradient=compensate_gradient
        super().__init__(name=name)

    def get_config(self):
      config = super().get_config().copy()
      config.update({"n_splits": self.n_splits, "compensate_gradient":self.compensate_gradient})
      return config

    def build(self, input_shape):
        self.rank = len(input_shape.as_list())
        if self.compensate_gradient:
            self.grad_fun = get_grad_weight_fun(1./self.n_splits)
        super().build(input_shape)

    def call(self, input): #(N x B, Y, X, C)
        shape = tf.shape(input)
        target_shape  = tf.concat([[self.n_splits, -1], [shape[i] for i in range(1, self.rank)]], 0)
        if self.compensate_gradient:
            input = self.grad_fun(input) # so that gradient are averaged over N (number of frames)
        input = tf.reshape(input, target_shape) # (N, B, Y, X, C)
        splits = tf.split(input, num_or_size_splits = self.n_splits, axis=0) # N x (1, B, Y, X, C)
        return [tf.squeeze(s, 0) for s in splits] # N x (B, Y, X, C)

class BatchToChannel(tf.keras.layers.Layer):
    def __init__(self, n_splits:int, compensate_gradient:bool = False, name:str="BatchToChannel"):
        self.n_splits=n_splits
        self.compensate_gradient = compensate_gradient
        self.inference_mode=False
        super().__init__(name=name)

    def get_config(self):
      config = super().get_config().copy()
      config.update({"n_splits": self.n_splits, "compensate_gradient":self.compensate_gradient})
      return config

    def build(self, input_shape):
        input_shape = input_shape.as_list()
        self.rank = len(input_shape)
        self.perm = [1] + [i+1 for i in range(1, self.rank-1)] + [0, self.rank] # (B, [DIMS], N, F)
        if self.compensate_gradient:
            self.grad_fun = get_grad_weight_fun(1./self.n_splits)
        super().build(input_shape)

    def call(self, input): #(N x B, Y, X, C)
        if self.inference_mode:
            return input
        if self.compensate_gradient:
            input = self.grad_fun(input)
        shape = tf.shape(input)
        dims = [shape[i] for i in range(1, self.rank-1)] if self.rank>2 else []
        target_shape1 = tf.concat([[self.n_splits, -1], dims, shape[-1:]], 0) if self.rank>2 else tf.concat([[self.n_splits, -1], shape[-1:]], 0)
        target_shape2 = tf.concat([[-1], dims, [self.n_splits * shape[-1]]], 0) if self.rank>2 else [-1, self.n_splits * shape[-1]]
        input = tf.reshape(input, shape = target_shape1) # (N, B, Y, X, F)
        input = tf.transpose(input, perm=self.perm) # (B, Y, X, N, F)
        return tf.reshape(input, target_shape2) # (B, Y, X, N x F)

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
    def wgrad(x):
        def grad(dy):
                g_flat = tf.reshape(tf.math.abs(dy), [-1])
                g_flat = tf.boolean_mask(g_flat, tf.greater(g_flat, 0))
                print(f"{message} gradient shape: {dy.shape}, non-null: {100 * g_flat.shape[0]/tf.size(dy)}%, value: {tf.math.reduce_mean(g_flat)}")
                return dy
        return x, grad
    return wgrad

class Combine(tf.keras.layers.Layer):
    def __init__(
            self,
            filters: int,
            kernel_size: int=1,
            weight_scaled:bool = False,
            activation: str="relu",
            compensate_gradient:bool = False,
            l2_reg: float=0,
            name: str="Combine",
        ):
        self.activation = activation
        self.filters= filters
        self.kernel_size=kernel_size
        self.weight_scaled = weight_scaled
        self.compensate_gradient = compensate_gradient
        self.l2_reg=l2_reg
        super().__init__(name=name)

    def get_config(self):
      config = super().get_config().copy()
      config.update({"activation": self.activation, "filters":self.filters, "kernel_size":self.kernel_size, "weight_scaled":self.weight_scaled, "compensate_gradient":self.compensate_gradient, "l2_reg":self.l2_reg})
      return config

    def build(self, input_shape):
        self.concat = tf.keras.layers.Concatenate(axis=-1, name = "Concat")
        if self.weight_scaled:
            self.combine_conv = WSConv2D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                padding='same',
                activation=self.activation,
                kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg>0 else None,
                name="Conv1x1")
        else:
            self.combine_conv = tf.keras.layers.Conv2D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                padding='same',
                activation=self.activation,
                kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg>0 else None,
                name="Conv1x1")
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

class WeigthedGradient(tf.keras.layers.Layer):
    def __init__(self, weight, name: str="WeigthedGradient"):
        super().__init__()
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

class ResConv2D(tf.keras.layers.Layer):
    def __init__(
            self,
            kernel_size: int=3,
            dilation: int = 1,
            weighted_sum : bool = False,
            dropout_rate : float = 0.3,
            batch_norm : bool = True,
            weight_scaled:bool = False,
            activation:str = "relu",
            l2_reg:float = 0,
            split_conv:bool = False,
            name: str="ResConv2D",
    ):
        super().__init__(name=name)
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.activation=activation
        self.dropout_rate=dropout_rate
        self.batch_norm=batch_norm
        self.weight_scaled = weight_scaled
        self.weighted_sum = weighted_sum
        self.l2_reg = l2_reg
        self.split_conv=split_conv
    def get_config(self):
      config = super().get_config().copy()
      config.update({"activation": self.activation, "kernel_size":self.kernel_size, "dilation":self.dilation, "dropout_rate":self.dropout_rate, "batch_norm":self.batch_norm, "weight_scaled":self.weight_scaled, "weighted_sum":self.weighted_sum, "l2_reg":self.l2_reg, "split_conv":self.split_conv})
      return config

    def build(self, input_shape):
        input_channels = int(input_shape[-1])
        conv_fun = WSConv2D if self.weight_scaled else (SplitConv2D if self.split_conv else tf.keras.layers.Conv2D)
        self.conv1 = conv_fun(
            filters=input_channels,
            kernel_size=self.kernel_size,
            strides=1,
            padding='same',
            name=f"{self.name}/1_{ker_size_to_string(self.kernel_size)}",
            activation="linear",
            #kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg>0 else None
        )
        self.conv2 = conv_fun(
            filters=input_channels,
            kernel_size=self.kernel_size,
            dilation_rate = self.dilation,
            strides=1,
            padding='same',
            name=f"{self.name}/2_{ker_size_to_string(self.kernel_size)}",
            activation="linear",
            #kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg>0 else None
        )
        self.gamma = get_gamma(self.activation) if self.weight_scaled else 1.
        self.activation_layer = tf.keras.activations.get(self.activation)
        if self.dropout_rate>0:
            self.drop = tf.keras.layers.SpatialDropout2D(self.dropout_rate)
        if self.batch_norm:
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.bn2 = tf.keras.layers.BatchNormalization()
        if self.weighted_sum:
            self.ws = WeightedSum(per_channel=True)
        super().build(input_shape)

    def call(self, input, training=None):
        x = self.conv1(input)
        if self.batch_norm:
            x = self.bn1(x, training = training)
        x = self.activation_layer(x) #* self.gamma
        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn2(x, training = training)
        if self.dropout_rate>0:
            x = self.drop(x, training = training)
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
            dropout_rate:float = 0.3,
            batch_norm : bool = True,
            activation:str = "relu",
            l2_reg:float = 0,
            name: str="ConvBNDrop",
    ):
        super().__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.activation=activation
        self.dropout_rate=dropout_rate
        self.batch_norm=batch_norm
        self.strides=strides
        self.l2_reg = l2_reg

    def get_config(self):
      config = super().get_config().copy()
      config.update({"filters":self.filters, "activation": self.activation, "kernel_size":self.kernel_size, "dilation":self.dilation, "dropout_rate":self.dropout_rate, "batch_norm":self.batch_norm, "strides":self.strides, "l2_reg":self.l2_reg})
      return config

    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            dilation_rate = self.dilation,
            strides=self.strides,
            padding='same',
            name=f"{self.name}/Conv",
            activation="linear",
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg>0 else None
        )
        self.activation_layer = tf.keras.activations.get(self.activation)
        if self.dropout_rate>0:
            self.drop = tf.keras.layers.SpatialDropout2D(self.dropout_rate)
        if self.batch_norm:
            self.bn = tf.keras.layers.BatchNormalization()
        super().build(input_shape)

    def call(self, input, training=None):
        x = self.conv(input)
        if self.batch_norm:
            x = self.bn(x, training = training)
        if self.dropout_rate>0:
            x = self.drop(x, training = training)
        return self.activation_layer(x)

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
            name: str="ResConv2DTransposeBNDrop",
    ):
        super().__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation=activation
        self.dropout_rate=dropout_rate
        self.batch_norm=batch_norm
        self.strides=strides
        self.l2_reg=l2_reg

    def get_config(self):
      config = super().get_config().copy()
      config.update({"filters":self.filters, "activation": self.activation, "kernel_size":self.kernel_size, "dropout_rate":self.dropout_rate, "batch_norm":self.batch_norm, "strides":self.strides, "l2_reg":self.l2_reg})
      return config

    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv2DTranspose(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding='same',
            activation="linear",
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg>0 else None,
            name=f"{self.name}/tConv{ker_size_to_string(self.kernel_size)}",
        )
        self.activation_layer = tf.keras.activations.get(self.activation)
        if self.dropout_rate>0:
            self.drop = tf.keras.layers.SpatialDropout2D(self.dropout_rate, name=f"{self.name}/Dropout")
        if self.batch_norm:
            #self.bn = MockBatchNormalization(name = f"{self.name}/MockBatchNormalization")
            self.bn = tf.keras.layers.BatchNormalization(name = f"{self.name}/BatchNormalization")
        super().build(input_shape)

    def call(self, input, training=None):
        x = self.conv(input)
        if self.batch_norm:
            x = self.bn(x, training = training)
        if self.dropout_rate>0:
            x = self.drop(x, training = training)
        return self.activation_layer(x)

class SplitConv2D(tf.keras.layers.Layer):
    def __init__(
            self,
            filters:int=0,
            kernel_size: int=3,
            dilation_rate: int = 1,
            strides: int = 1,
            dropout_rate:float = 0,
            batch_norm : bool = False,
            activation:str = "relu",
            padding:str = "same",
            kernel_regularizer = None,
            name: str="SplitConv2D",
    ):
        super().__init__(name=name)
        self.kernel_size = kernel_size
        self.dilation = dilation_rate
        self.activation=activation
        self.dropout_rate=dropout_rate
        self.batch_norm=batch_norm
        self.strides=strides
        self.padding=padding
        self.kernel_regularizer = kernel_regularizer

    def get_config(self):
      config = super().get_config().copy()
      config.update({"activation": self.activation, "kernel_size":self.kernel_size, "dilation":self.dilation, "dropout_rate":self.dropout_rate, "batch_norm":self.batch_norm, "strides":self.strides, "padding":self.padding, "l2_reg":self.l2_reg})
      return config

    def build(self, input_shape):
        input_filters = input_shape.as_list()[-1]
        assert input_filters % 3==0, f"number of filters must be divisible by 3"
        self.filters = input_filters // 3
        conv_fun = lambda name: tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            dilation_rate = self.dilation,
            strides=self.strides,
            padding=self.padding,
            name=name,
            activation="linear",
            kernel_regularizer=self.kernel_regularizer
        )
        self.convs = [conv_fun(f"{self.name}/Conv_{i}") for i in range(3)]
        self.activation_layer = tf.keras.activations.get(self.activation)
        if self.dropout_rate>0:
            self.drop = tf.keras.layers.SpatialDropout2D(self.dropout_rate)
        if self.batch_norm:
            self.bn = tf.keras.layers.BatchNormalization()
        super().build(input_shape)

    def call(self, input, training=None):
        inputs = tf.split(input, 3, axis=-1)
        x = tf.concat([ self.convs[i](tf.concat([inputs[i], inputs[(i+1)%3]], -1)) for i in range(3) ], -1)
        if self.batch_norm:
            x = self.bn(x, training = training)
        if self.dropout_rate>0:
            x = self.drop(x, training = training)
        return self.activation_layer(x)


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

class WSConv2D(tf.keras.layers.Conv2D):
    def __init__(self, *args, eps=1e-4, use_gain=True, dropout_rate = 0, kernel_initializer="he_normal", **kwargs):
        activation = kwargs.pop("activation", "linear") # bypass activation
        gamma = kwargs.pop("gamma", get_gamma(activation if isinstance(activation, str) else tf.keras.activations.serialize(activation)))
        super().__init__(kernel_initializer=kernel_initializer, *args, **kwargs)
        self.eps = eps
        self.use_gain = use_gain
        self.activation_layer = tf.keras.activations.get(activation)
        self.dropout_rate = dropout_rate
        self.gamma = gamma

    def build(self, input_shape):
        super().build(input_shape)
        if self.use_gain:
            self.gain = self.add_weight(
                name="gain",
                shape=(self.kernel.shape[-1],),
                initializer="ones",
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self.gain = None
        if self.dropout_rate>0:
            self.dropout = tf.keras.layers.SpatialDropout2D(self.dropout_rate)

    def convolution_op(self, inputs, kernel): # original code modified to used standardized weights
        if self.padding == "causal":
            tf_padding = "VALID"  # Causal padding handled in `call`.
        elif isinstance(self.padding, str):
            tf_padding = self.padding.upper()
        else:
            tf_padding = self.padding

        return tf.nn.convolution(
            inputs,
            _standardize_weight(kernel, self.gain, self.eps),
            strides=list(self.strides),
            padding=tf_padding,
            dilations=list(self.dilation_rate),
            data_format=self._tf_data_format,
            name=self.__class__.__name__,
        )

    def get_config(self):
        config = super().get_config().copy()
        config.update({"eps":self.eps, "use_gain": self.use_gain, "activation_layer":tf.keras.activations.serialize(self.activation_layer), "dropout_rate":self.dropout_rate, "gamma":self.gamma})
        return config

    def call(self, input, training=None):
        x = super().call(input)
        if self.dropout_rate>0:
            x = self.dropout(x, training = training)
        if self.activation_layer is not None:
            x = self.activation_layer(x) #* self.gamma
        return x

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
            dtype=self.dtype,
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
