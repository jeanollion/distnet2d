from tensorflow import pad
from tensorflow.keras.layers import Layer, GlobalAveragePooling2D, Reshape, Conv2D, Multiply, Conv3D, UpSampling2D
from tensorflow.keras import Model, backend
from ..utils.helpers import ensure_multiplicity, get_nd_gaussian_kernel
from tensorflow.python.keras.engine.input_spec import InputSpec
import tensorflow as tf
import numpy as np
from keras.utils import conv_utils

class ReflectionPadding2D(Layer):
  def __init__(self, paddingYX=(1, 1), **kwargs):
    paddingYX = ensure_multiplicity(2, paddingYX)
    self.padding = tuple(paddingYX)
    super().__init__(**kwargs)

  def compute_output_shape(self, input_shape):
    return (input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3])

  def call(self, input_tensor, mask=None):
    padding_height, padding_width = self.padding
    return pad(input_tensor, [[0,0], [padding_height, padding_height], [padding_width, padding_width], [0,0] ], 'REFLECT')

  def get_config(self):
      config = super().get_config().copy()
      config.update({"padding": self.padding})
      return config

class ConstantConvolution2D(Layer):
  def __init__(self, kernelYX, reflection_padding=False, **kwargs):
    assert len(kernelYX.shape)==2
    for ax in [0, 1]:
      assert kernelYX.shape[ax]>=1 and kernelYX.shape[ax]%2==1, "invalid kernel size along axis: {}".format(ax)
    self.kernelYX = kernelYX[...,np.newaxis, np.newaxis]
    self.reflection_padding=reflection_padding
    self.padL = ReflectionPadding2D([(dim-1)//2 for dim in self.kernelYX.shape[:-2]] ) if self.reflection_padding else None
    self.n_chan = kwargs.pop("n_chan", None)
    super().__init__(**kwargs)

  def build(self, input_shape):
    n_chan = input_shape[-1] if input_shape[-1] is not None else self.n_chan
    if n_chan is None:
        self.kernel=None
        return
    kernel = tf.constant(self.kernelYX, dtype=tf.float32)
    if n_chan>1:
      self.kernel = tf.tile(kernel, [1, 1, n_chan, 1])
    else:
      self.kernel = kernel
    self.pointwise_filter = tf.eye(n_chan, batch_shape=[1, 1])

  def compute_output_shape(self, input_shape):
    if self.reflection_padding:
        return input_shape
    radY = (self.kernelYX.shape[0] - 1) // 2
    radX = (self.kernelYX.shape[1] - 1) // 2
    return (input_shape[0], input_shape[1] - radY * 2, input_shape[2] - radX * 2, input_shape[3])

  def call(self, input_tensor, mask=None):
    if self.kernel is None: #build was initiated with None shape
        return input_tensor
    if self.padL is not None:
      input_tensor = self.padL(input_tensor)
    return tf.nn.separable_conv2d(input_tensor, self.kernel, self.pointwise_filter, strides=[1, 1, 1, 1], padding='VALID')

  def get_config(self):
    config = super().get_config().copy()
    config.update({"kernelYX": self.kernelYX, "reflection_padding":reflection_padding})
    return config

class Gaussian2D(ConstantConvolution2D):
  def __init__(self, radius=1, **kwargs):
    gauss_ker = get_nd_gaussian_kernel(radius=self.radius, ndim=2)[...,np.newaxis, np.newaxis]
    super().__init__(kernelYX = gauss_ker, **kwargs)

def channel_attention(n_filters, activation='relu'): # TODO TEST + make layer or model + set name to layers
  def ca_fun(input):
    gap = GlobalAveragePooling2D()(input)
    gap = Reshape((1, 1, n_filters))(gap) # or use dense layers and reshape afterwards
    conv1 = Conv2D(kernel_size=1, filters = n_filters, activation=activation)(gap)
    key = Conv2D(kernel_size=1, filters = n_filters, activation='sigmoid')(conv1)
    return Multiply()([key, input])
  return ca_fun

############## TEST #####################
from tensorflow.python.framework import tensor_shape
class SplitContextCenterConv2D(Layer):
    def __init__(self, filters, kernelYX, padding="same", **kwargs): #REFLECT
        kernelYX=ensure_multiplicity(2, kernelYX)
        for k in kernelYX:
            assert k%2==1, "kernel size must be uneven on all spatial axis"
        if padding=="same":
            padding = "CONSTANT"
        name = kwargs.pop('name', None)
        self.padding_constant_value = kwargs.pop('constant_values', 0)
        self.convL = Conv3D(filters=filters, kernel_size = (kernelYX[0], kernelYX[1], 2), padding="valid",  name = name+"conv" if name is not None else None, **kwargs)
        self.input_spec = InputSpec(ndim=4)
        self.padding = padding
        self.ker_center = [(k-1)//2 for k in kernelYX]
        super().__init__(name)

    def compute_output_shape(self, input_shape):
        if self.padding=="valid":
            return (input_shape[0], input_shape[1] - self.convL.kernel_size[0] + 1 , input_shape[2] - self.convL.kernel_size[1] + 1, self.filters)
        else:
            return (input_shape[0], input_shape[1], input_shape[2], self.filters)

    def build(self, input_shape):
        super().build(input_shape)
        input_shape = tensor_shape.TensorShape(input_shape)
        self.n_channels = int(input_shape[-1])//2
        conv_input_shape = input_shape[:-1] + (2, self.n_channels)
        self.convL.build(conv_input_shape)

        kernel_mask = np.ones(shape=(self.convL.kernel_size)+( self.n_channels, self.convL.filters )) # broadcasting
        kernel_mask[self.ker_center[0],self.ker_center[1],0]=0
        kernel_mask[:,:self.ker_center[1],1]=0
        kernel_mask[:,(self.ker_center[1]+1):,1]=0
        kernel_mask[:self.ker_center[0],self.ker_center[1],1]=0
        kernel_mask[(self.ker_center[0]+1):,self.ker_center[1],1]=0
        self.kernel_mask = tf.convert_to_tensor(kernel_mask, dtype=tf.bool)

    def call(self, input_tensor, mask=None):
        if self.padding!="valid": # we set explicitely the padding because convolution is performed with valid padding
            padding_height, padding_width = self.ker_center
            input_tensor = pad(input_tensor, [[0,0], [padding_height, padding_height], [padding_width, padding_width], [0,0] ], mode = self.padding, constant_values=self.padding_constant_value, name = self.name+"pad" if self.name is not None else None)
        # convert to 5D tensor -> split in channel dimension to create a new spatial dimension. assumes channel last
        # in : BYX[2xC], out: BYX2C
        if self.n_channels==1:
            conv_in = input_tensor[...,tf.newaxis]
        else:
            context, center = tf.split(input_tensor, 2, axis=-1)
            conv_in = tf.concat([context[...,tf.newaxis, :], center[...,tf.newaxis, :]], axis=-2)
        self.convL.kernel.assign(tf.where(self.kernel_mask, self.convL.kernel, tf.zeros_like(self.convL.kernel))) # set explicitely the unused weights to zero
        conv = self.convL(conv_in) # BYX1F (valid padding on last conv axis -> size 1)
        return conv[:, :, :, 0, :]

# TODO add periodic padding so that each color has access to the 2 other ones. test to perform the whole net in 4D. see https://stackoverflow.com/questions/39088489/tensorflow-periodic-padding
class Conv3DYXC(Layer):
    def __init__(self, filters, kernelYX, padding="same", **kwargs): #padding can also be REFLECT
        self.kernelYX=tuple(ensure_multiplicity(2, kernelYX))
        for k in kernelYX:
            assert k%2==1, "kernel size must be uneven on all spatial axis"
        self.ker_center = [(k-1)//2 for k in kernelYX]
        if padding=="same":
            padding = "CONSTANT"
        self._name = kwargs.pop('name', "Conv3DYXC")
        self.padding_constant_value = kwargs.pop('constant_values', 0)
        self.input_spec = InputSpec(ndim=4)
        self.padding = padding
        self.filters=filters
        self.conv_args = kwargs
        super().__init__(self._name)

    def compute_output_shape(self, input_shape):
        if self.padding=="valid":
            return (input_shape[0], input_shape[1] - self.convL.kernel_size[0] + 1 , input_shape[2] - self.convL.kernel_size[1] + 1, self.filters)
        else:
            return (input_shape[0], input_shape[1], input_shape[2], self.filters)

    def build(self, input_shape):
        super().build(input_shape)
        input_shape = tensor_shape.TensorShape(input_shape)
        n_channels = int(input_shape[-1])
        self.convL = Conv3D(filters=self.filters, kernel_size = self.kernelYX + (n_channels,), padding="valid",  name = self._name+"conv" if self._name is not None else None, **self.conv_args)
        conv_input_shape = input_shape + (1,)
        self.convL.build(conv_input_shape)

    def call(self, input_tensor, mask=None):
        if self.padding!="valid": # we set explicitely the padding because convolution is performed with valid padding
            padding_height, padding_width = self.ker_center
            input_tensor = pad(input_tensor, [[0,0], [padding_height, padding_height], [padding_width, padding_width], [0,0] ], mode = self.padding, constant_values=self.padding_constant_value, name = self.name+"pad" if self.name is not None else None)
        conv_in = input_tensor[...,tf.newaxis] #BYXC1 (convert to 5D tensor)
        conv = self.convL(conv_in) # BYX1F (valid padding on last conv axis -> size 1)
        return conv[:, :, :, 0, :] # BYXF

# TODO get_config -> attributes of convL ?

class Downsampling2D(Layer):
    def __init__(
            self,
            filters: int,
            kernel_size: int=2,
            conv_kernel_size:int=3,
            mode:str="stride", # maxpool, stride
            activation: str="relu",
            l2_reg: float=1e-5,
            use_bias:bool = True,
            name: str="Downsampling2D",
        ):
        super().__init__(name=name)

        if mode=="stride":
            _available_activation = {
                "relu": tf.keras.layers.ReLU(),
                "relu6": ReLU6(),
                "hswish": HardSwish(),
                "hsigmoid": HardSigmoid(),
                "softmax": tf.keras.layers.Softmax(),
            }
            act = get_layer(activation, _available_activation, Identity())
            self.operation = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=conv_kernel_size,
                strides=kernel_size,
                padding='same',
                name=f"_DownConv{kernel_size}x{kernel_size}",
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                use_bias=use_bias,
                activation=act
            )
        else:
            self.operation = MaxPool2D(pool_size=kernel_size, name=f"Maxpool{kernel_size}x{kernel_size}")

    def call(self, input):
        return self.operation(input)

class UpSamplingLayer2D(Layer):
    def __init__(
            self,
            filters: int,
            kernel_size: int=2,
            mode:str="tconv", # tconv, up_nn, up_bilinera
            norm_layer:str=None,
            activation: str="relu",
            l2_reg: float=1e-5,
            use_bias:bool = True,
            name: str="Upsampling2D",
        ):
        super().__init__(name=name)
        if mode=="tconv":
            self.upsample = tf.keras.layers.Conv2DTranspose(
                filters,
                kernel_size=kernel_size,
                strides=kernel_size,
                padding='same',
                activation=activation,
                use_bias=use_bias,
                # kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                name=f"tConv{kernel_size}x{kernel_size}",
            )
            self.conv=None
        else:
            interpolation = "nearest" if mode=="up_nn" else 'bilinear'
            self.upsample = UpSampling2D(size=kernel_size, interpolation=interpolation, name = f"Upsample_{interpolation}")
            self.conv = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=1,
                padding='same',
                name=f"Conv{kernel_size}x{kernel_size}",
                # kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                use_bias=use_bias,
                activation=activation
            )

        _available_normalization = {
            "bn": BatchNormalization(name),
        }
        self.norm = get_layer(norm_layer, _available_normalization, Identity(name))

        _available_activation = {
            "relu": tf.keras.layers.ReLU(name="ReLU"),
            "relu6": ReLU6(),
            "hswish": HardSwish(),
            "hsigmoid": HardSigmoid(),
            "softmax": tf.keras.layers.Softmax(name="Softmax"),
        }
        self.act = get_layer(activation, _available_activation, Identity(name))

    def call(self, input):
        x = self.upsample(input)
        if self.conv is not None:
            x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class StopGradient(Layer):
    def __init__(self, name:str="StopGradient"):
        super().__init__(name=name)

    def call(self, input):
        return tf.stop_gradient( input, name=self.name )

class ApplyChannelWise(Layer):
    def __init__(self, op:Layer, name:str="ApplyChannelWise"):
        self.op = op
        self.concat = tf.keras.layers.Concatenate(axis=-1, name = "Concat")
        super().__init__(name=self.n)

        def build(self, input_shape):
            self.n_chan = input_shape[-1]
            super().build(input_shape)

        def call(self, input):
            channels = tf.split( value, self.n_chan, axis=0, num=None, name="Split" )
            channels = [self.op(c) for c in channels]
            return self.concat(channels)

class NConvToBatch2D(Layer):
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
            Conv2D(filters=self.filters, kernel_size=1, padding='same', activation="relu", name=f"Conv_{i}")
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

class SelectFeature(Layer):
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

class ChannelToBatch2D(Layer):
    def __init__(self, compensate_gradient:bool = False, name: str="ChannelToBatch2D"):
        self.compensate_gradient=compensate_gradient
        super().__init__(name=name)

    def get_config(self):
        config = super().get_config().copy()
        config.update({"compensate_gradient": self.compensate_gradient})
        return config

    def build(self, input_shape):
        self.target_shape = [-1, input_shape[1], input_shape[2]]
        if self.compensate_gradient:
            self.grad_fun = get_grad_weight_fun(float(input_shape[-1]))
        super().build(input_shape)

    def call(self, input): # (B, Y, X, C)
        input = tf.transpose(input, perm=[3, 0, 1, 2]) # (C, B, Y, X)
        input = tf.reshape(input, self.target_shape) # (C x B, Y, X)
        if self.compensate_gradient:
            input = self.grad_fun(input)
        return tf.expand_dims(input, -1) # (C x B, Y, X, 1)

class SplitBatch2D(Layer):
    def __init__(self, n_splits:int, compensate_gradient:bool = False, name:str="SplitBatch2D"):
        self.n_splits=n_splits
        self.compensate_gradient=compensate_gradient
        super().__init__(name=name)

    def get_config(self):
      config = super().get_config().copy()
      config.update({"n_splits": self.n_splits, "compensate_gradient":self.compensate_gradient})
      return config

    def build(self, input_shape):
        self.target_shape = [self.n_splits, -1, input_shape[1], input_shape[2], input_shape[3]]
        if self.compensate_gradient:
            self.grad_fun = get_grad_weight_fun(1./self.n_splits)
        super().build(input_shape)

    def call(self, input): #(N x B, Y, X, C)
        #input = get_print_grad_fun(f"{self.name} before split")(input)
        if self.compensate_gradient:
            input = self.grad_fun(input) # so that gradient are averaged over N (number of frames)
        input = tf.reshape(input, self.target_shape) # (N, B, Y, X, C)
        splits = tf.split(input, num_or_size_splits = self.n_splits, axis=0) # N x (1, B, Y, X, C)
        #splits[0] = get_print_grad_fun(f"{self.name} after split")(splits[0])
        return [tf.squeeze(s, 0) for s in splits] # N x (B, Y, X, C)

class BatchToChannel2D(Layer):
    def __init__(self, n_splits:int, compensate_gradient:bool = False, name:str="BatchToChannel2D"):
        self.n_splits=n_splits
        self.compensate_gradient = compensate_gradient
        self.inference_mode=False
        super().__init__(name=name)

    def get_config(self):
      config = super().get_config().copy()
      config.update({"n_splits": self.n_splits, "compensate_gradient":self.compensate_gradient})
      return config

    def build(self, input_shape):
        self.target_shape1 = [self.n_splits, -1, input_shape[1], input_shape[2], input_shape[3]]
        self.target_shape2 = [-1, input_shape[1], input_shape[2], self.n_splits * input_shape[-1]]
        if self.compensate_gradient:
            self.grad_fun = get_grad_weight_fun(1./self.n_splits)
        super().build(input_shape)

    def call(self, input): #(N x B, Y, X, C)
        if self.inference_mode:
            return input
        if self.compensate_gradient:
            input = self.grad_fun(input)
        input = tf.reshape(input, shape = self.target_shape1) # (N, B, Y, X, F)
        input = tf.transpose(input, perm=[1, 2, 3, 0, 4]) # (B, Y, X, N, F)
        return tf.reshape(input, self.target_shape2) # (B, Y, X, N x F)

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

class Combine(Layer):
    def __init__(
            self,
            filters: int,
            kernel_size: int=1,
            weight_scaled:bool = False,
            activation: str="relu",
            compensate_gradient:bool = False,
            name: str="Combine",
        ):
        self.activation = activation
        self.filters= filters
        self.kernel_size=kernel_size
        self.weight_scaled = weight_scaled
        self.compensate_gradient = compensate_gradient
        super().__init__(name=name)

    def get_config(self):
      config = super().get_config().copy()
      config.update({"activation": self.activation, "filters":self.filters, "kernel_size":self.kernel_size, "weight_scaled":self.weight_scaled, "compensate_gradient":self.compensate_gradient})
      return config

    def build(self, input_shape):
        self.concat = tf.keras.layers.Concatenate(axis=-1, name = "Concat")
        if self.weight_scaled:
            self.combine_conv = WSConv2D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                padding='same',
                activation=self.activation,
                name="Conv1x1")
        else:
            self.combine_conv = Conv2D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                padding='same',
                activation=self.activation,
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


class ResConv1D(Layer): # Non-bottleneck-1D from ERFNet
    def __init__(
            self,
            kernel_size: int=3,
            dilation: int = 1,
            weighted_sum : bool = False,
            dropout_rate : float = 0.3,
            weight_scaled : bool = False,
            batch_norm : bool = True,
            activation:str = "relu",
            name: str="ResConv1D",
    ):
        super().__init__(name=name)
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.activation=activation
        self.dropout_rate=dropout_rate
        self.batch_norm=batch_norm
        self.weight_scaled=weight_scaled
        self.weighted_sum=weighted_sum

    def get_config(self):
      config = super().get_config().copy()
      config.update({"activation": self.activation, "kernel_size":self.kernel_size, "dilation":self.dilation, "dropout_rate":self.dropout_rate, "batch_norm":self.batch_norm, "weight_scaled":self.weight_scaled, "weighted_sum":self.weighted_sum})
      return config

    def build(self, input_shape):
        input_channels = int(input_shape[-1])
        conv_fun = WSConv2D if self.weight_scaled else tf.keras.layers.Conv2D
        self.convY1 = conv_fun(
            filters=input_channels,
            kernel_size=(self.kernel_size, 1),
            strides=1,
            padding='same',
            name=f"{self.name}/1_{self.kernel_size}x1",
            activation=self.activation
        )
        self.convX1 = conv_fun(
            filters=input_channels,
            kernel_size=(1,self.kernel_size),
            strides=1,
            padding='same',
            name=f"{self.name}/1_1x{self.kernel_size}",
            activation="linear"
        )
        self.convY2 = conv_fun(
            filters=input_channels,
            kernel_size=(self.kernel_size, 1),
            dilation_rate = (self.dilation, 1),
            strides=1,
            padding='same',
            name=f"{self.name}/2_{self.kernel_size}x1",
            activation=self.activation
        )
        self.convX2 = conv_fun(
            filters=input_channels,
            kernel_size=(1,self.kernel_size),
            dilation_rate = (1, self.dilation),
            strides=1,
            padding='same',
            name=f"{self.name}/2_1x{self.kernel_size}",
            activation="linear"
        )
        self.activation_layer = tf.keras.activations.get(self.activation)
        self.gamma = get_gamma(self.activation) if self.weight_scaled else 1.
        if self.dropout_rate>0:
            self.drop = tf.keras.layers.SpatialDropout2D(self.dropout_rate)
        if self.batch_norm:
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.bn2 = tf.keras.layers.BatchNormalization()
        if self.weighted_sum:
            self.ws = WeightedSum(per_channel=True)
        super().build(input_shape)

    def call(self, input, training=None):
        x = self.convY1(input)
        x = self.convX1(x)
        if self.batch_norm:
            x = self.bn1(x, training = training)
        x = self.activation_layer(x) #* self.gamma
        x = self.convY2(x)
        x = self.convX2(x)
        if self.batch_norm:
            x = self.bn2(x, training = training)
        if self.dropout_rate>0:
            x = self.drop(x, training = training)
        if self.weighted_sum:
            return self.activation_layer(self.ws([input, x]))
        else:
            return self.activation_layer(input + x)

class ResConv2D(Layer):
    def __init__(
            self,
            kernel_size: int=3,
            dilation: int = 1,
            weighted_sum : bool = False,
            dropout_rate : float = 0.3,
            batch_norm : bool = True,
            weight_scaled:bool = False,
            activation:str = "relu",
            name: str="ResConv1D",
    ):
        super().__init__(name=name)
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.activation=activation
        self.dropout_rate=dropout_rate
        self.batch_norm=batch_norm
        self.weight_scaled = weight_scaled
        self.weighted_sum = weighted_sum

    def get_config(self):
      config = super().get_config().copy()
      config.update({"activation": self.activation, "kernel_size":self.kernel_size, "dilation":self.dilation, "dropout_rate":self.dropout_rate, "batch_norm":self.batch_norm, "weight_scaled":self.weight_scaled, "weighted_sum":self.weighted_sum})
      return config

    def build(self, input_shape):
        input_channels = int(input_shape[-1])
        conv_fun = WSConv2D if self.weight_scaled else tf.keras.layers.Conv2D
        self.conv1 = conv_fun(
            filters=input_channels,
            kernel_size=self.kernel_size,
            strides=1,
            padding='same',
            name=f"{self.name}/1_{self.kernel_size}x{self.kernel_size}",
            activation="linear"
        )
        self.conv2 = conv_fun(
            filters=input_channels,
            kernel_size=self.kernel_size,
            dilation_rate = self.dilation,
            strides=1,
            padding='same',
            name=f"{self.name}/2_{self.kernel_size}x{self.kernel_size}",
            activation="linear"
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

class Conv2DBNDrop(Layer):
    def __init__(
            self,
            filters:int,
            kernel_size: int=3,
            dilation: int = 1,
            strides: int = 1,
            dropout_rate:float = 0.3,
            batch_norm : bool = True,
            activation:str = "relu",
            name: str="ResConv1D",
    ):
        super().__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.activation=activation
        self.dropout_rate=dropout_rate
        self.batch_norm=batch_norm
        self.strides=strides

    def get_config(self):
      config = super().get_config().copy()
      config.update({"filters":self.filters, "activation": self.activation, "kernel_size":self.kernel_size, "dilation":self.dilation, "dropout_rate":self.dropout_rate, "batch_norm":self.batch_norm, "strides":self.strides})
      return config

    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            dilation_rate = self.dilation,
            strides=self.strides,
            padding='same',
            name=f"{self.name}/{self.kernel_size}",
            activation="linear"
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

class Conv2DTransposeBNDrop(Layer):
    def __init__(
            self,
            filters:int,
            kernel_size: int=4,
            strides: int = 2,
            dropout_rate:float = 0,
            batch_norm : bool = False,
            activation:str = "relu",
            name: str="ResConv2D",
    ):
        super().__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation=activation
        self.dropout_rate=dropout_rate
        self.batch_norm=batch_norm
        self.strides=strides

    def get_config(self):
      config = super().get_config().copy()
      config.update({"filters":self.filters, "activation": self.activation, "kernel_size":self.kernel_size, "dropout_rate":self.dropout_rate, "batch_norm":self.batch_norm, "strides":self.strides})
      return config

    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv2DTranspose(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding='same',
            activation="linear",
            name=f"{self.name}/tConv{self.kernel_size}x{self.kernel_size}",
        )
        self.activation_layer = tf.keras.activations.get(self.activation)
        if self.dropout_rate>0:
            self.drop = tf.keras.layers.SpatialDropout2D(self.dropout_rate, name=f"{self.name}/Dropout")
        if self.batch_norm == "trainablenormalization":
            self.bn = TrainableNormalization(name = f"{self.name}/TrainableNormalization")
        elif self.batch_norm:
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

class TrainableNormalization(Layer):
    def __init__(
            self,
            scale:bool = True,
            center:bool = True,
            name: str="BatchNormalization",
    ):
        super().__init__(name=name)
        self.scale = scale
        self.center = center

    def get_config(self):
      config = super().get_config().copy()
      config.update({"scale":self.scale, "center": self.center})
      return config

    def build(self, input_shape):
        if self.scale:
            self.gamma = self.add_weight(
                name="gamma",
                shape=input_shape[-1:],
                dtype=self.dtype,
                initializer="ones",
                trainable=True,
                experimental_autocast=False,
            )
        else:
            self.gamma = None

        if self.center:
            self.beta = self.add_weight(
                name="beta",
                shape=input_shape[-1:],
                dtype=self.dtype,
                initializer="zeros",
                trainable=True,
                experimental_autocast=False,
            )
        else:
            self.beta = None
        super().build(input_shape)

    def call(self, input):
        if self.scale:
            input = input * self.gamma
        if self.center:
            input = input + self.beta
        return input

class MockBatchNormalization(Layer):
    def __init__(
            self,
            scale:bool = True,
            center:bool = True,
            name: str="BatchNormalization",
    ):
        super().__init__(name=name)
        self.scale = scale
        self.center = center

    def get_config(self):
      config = super().get_config().copy()
      config.update({"scale":self.scale, "center": self.center})
      return config

    def build(self, input_shape):
        if self.scale:
            self.gamma = self.add_weight(
                name="gamma",
                shape=input_shape[-1:],
                dtype=tf.float32,
                trainable=False,
                experimental_autocast=False,
            )
        else:
            self.gamma = None

        if self.center:
            self.beta = self.add_weight(
                name="beta",
                shape=input_shape[-1:],
                dtype=tf.float32,
                trainable=False,
                experimental_autocast=False,
            )
        else:
            self.beta = None
        self.moving_mean = self.add_weight(
            name="moving_mean",
            shape=input_shape[-1:],
            dtype=tf.float32,
            trainable=False,
            experimental_autocast=False,
        )

        self.moving_variance = self.add_weight(
            name="moving_variance",
            shape=input_shape[-1:],
            dtype=tf.float32,
            trainable=False,
            experimental_autocast=False,
        )
        super().build(input_shape)

    def _reshape(self, variable):
        return variable[tf.newaxis, tf.newaxis, tf.newaxis]

    def call(self, input):
        x = (input - self._reshape(self.moving_mean)) / tf.math.sqrt(self._reshape(self.moving_variance)+1e-3)
        if self.scale:
            x = x * self._reshape(self.gamma)
        if self.center:
            x = x + self._reshape(self.beta)
        return x

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

class WSConv2DTranspose(tf.keras.layers.Conv2DTranspose):
    def __init__(self, *args, eps=1e-4, use_gain=True, dropout_rate = 0, **kwargs):
        activation = kwargs.get("activation", "linear")
        gamma = kwargs.pop("gamma", get_gamma(activation if isinstance(activation, str) else tf.keras.activations.serialize(activation)))
        super().__init__(kernel_initializer="he_normal", *args, **kwargs)
        self.eps = eps
        self.use_gain = use_gain
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

    def get_config(self):
        config = super().get_config().copy()
        config.update({"eps":self.eps, "use_gain": self.use_gain, "dropout_rate":self.dropout_rate, "gamma":self.gamma})
        return config

    def call(self, inputs, training=None): # code from TF2.11 modified to use standardized weights + apply dropout: https://github.com/keras-team/keras/blob/v2.11.0/keras/layers/convolutional/conv2d_transpose.py#L34-L362
        inputs_shape = tf.shape(inputs)
        batch_size = inputs_shape[0]
        if self.data_format == "channels_first":
            h_axis, w_axis = 2, 3
        else:
            h_axis, w_axis = 1, 2

        height, width = None, None
        if inputs.shape.rank is not None:
            dims = inputs.shape.as_list()
            height = dims[h_axis]
            width = dims[w_axis]
        height = height if height is not None else inputs_shape[h_axis]
        width = width if width is not None else inputs_shape[w_axis]

        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides

        if self.output_padding is None:
            out_pad_h = out_pad_w = None
        else:
            out_pad_h, out_pad_w = self.output_padding

        # Infer the dynamic output shape:
        out_height = conv_utils.deconv_output_length(
            height,
            kernel_h,
            padding=self.padding,
            output_padding=out_pad_h,
            stride=stride_h,
            dilation=self.dilation_rate[0],
        )
        out_width = conv_utils.deconv_output_length(
            width,
            kernel_w,
            padding=self.padding,
            output_padding=out_pad_w,
            stride=stride_w,
            dilation=self.dilation_rate[1],
        )
        if self.data_format == "channels_first":
            output_shape = (batch_size, self.filters, out_height, out_width)
        else:
            output_shape = (batch_size, out_height, out_width, self.filters)

        output_shape_tensor = tf.stack(output_shape)
        outputs = backend.conv2d_transpose(
            inputs,
            _standardize_weight(self.kernel, self.gain, self.eps),
            output_shape_tensor,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )

        if not tf.executing_eagerly() and inputs.shape.rank:
            # Infer the static output shape:
            out_shape = self.compute_output_shape(inputs.shape)
            outputs.set_shape(out_shape)

        if self.use_bias:
            outputs = tf.nn.bias_add(
                outputs,
                self.bias,
                data_format=conv_utils.convert_data_format(
                    self.data_format, ndim=4
                ),
            )

        if self.dropout_rate>0:
            x = self.dropout(x, training = training)
        if self.activation is not None:
            return self.activation(outputs) #* self.gamma
        return outputs

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

############### MOBILE NET LAYERS ############################################################
############### FROM https://github.com/Bisonai/mobilenetv3-tensorflow/blob/master/layers.py
from .utils import get_layer


class Identity(Layer):
    def __init__(self, name="Identity"):
        super().__init__(name=name)

    def call(self, input):
        return input


class ReLU6(Layer):
    def __init__(self, name="ReLU6"):
        super().__init__(name=name)
        self.relu6 = tf.keras.layers.ReLU(max_value=6, name=name)

    def call(self, input):
        return self.relu6(input)


class HardSigmoid(Layer):
    def __init__(self, name="HardSigmoid"):
        super().__init__(name=name)
        self.relu6 = ReLU6(name)

    def call(self, input):
        return self.relu6(input + 3.0) / 6.0


class HardSwish(Layer):
    def __init__(self, name="HardSwish"):
        super().__init__(name=name)
        self.hard_sigmoid = HardSigmoid(name)

    def call(self, input):
        return input * self.hard_sigmoid(input)



class Squeeze(Layer):
    """Squeeze the second and third dimensions of given tensor.
    (batch, 1, 1, channels) -> (batch, channels)
    """
    def __init__(self, name="Squeeze"):
        super().__init__(name=name)

    def call(self, input):
        x = tf.keras.backend.squeeze(input, 1)
        x = tf.keras.backend.squeeze(x, 1)
        return x


class GlobalAveragePooling2D(Layer):
    """Return tensor of output shape (batch_size, rows, cols, channels)
    where rows and cols are equal to 1. Output shape of
    `tf.keras.layer.GlobalAveragePooling2D` is (batch_size, channels),
    """
    def __init__(self, name="GlobalAveragePooling2D"):
        super().__init__(name=name)

    def build(self, input_shape):
        pool_size = tuple(map(int, input_shape[1:3]))
        self.gap = tf.keras.layers.AveragePooling2D(
            pool_size=pool_size,
            name=f"AvgPool{pool_size[0]}x{pool_size[1]}",
        )

        super().build(input_shape)

    def call(self, input):
        return self.gap(input)


class BatchNormalization(Layer):
    """Searching fo MobileNetV3: All our convolutional layers
    use batch-normalization layers with average decay of 0.99.
    """
    def __init__( self, name="BatchNormalization", momentum: float=0.99):
        super().__init__(name=name)
        self.bn = tf.keras.layers.BatchNormalization( momentum=0.99, name=name)

    def call(self, input):
        return self.bn(input)

class ConvNormAct(Layer):
    def __init__(
            self,
            filters: int,
            kernel_size: int=3,
            stride: int=1,
            padding: int=-1,
            norm_layer: str=None,
            act_layer: str="relu",
            use_bias: bool=True,
            l2_reg: float=1e-5,
            name: str="ConvNormAct",
    ):
        super().__init__(name=name)
        if padding<0:
            padding = (kernel_size - 1)//2
        if padding > 0:
            self.pad = tf.keras.layers.ZeroPadding2D(
                padding=padding,
                name=f"Padding{padding}x{padding}",
            )
        else:
            self.pad = None

        self.conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=stride,
            name=f"Conv{kernel_size}x{kernel_size}",
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg) if l2_reg>0 else None,
            use_bias=use_bias,
        )

        _available_normalization = {
            "bn": BatchNormalization(name),
        }
        self.norm = get_layer(norm_layer, _available_normalization, Identity())

        _available_activation = {
            "relu": tf.keras.layers.ReLU(name="ReLU"),
            "relu6": ReLU6(),
            "hswish": HardSwish(),
            "hsigmoid": HardSigmoid(),
            "softmax": tf.keras.layers.Softmax(name="Softmax"),
        }
        self.act = get_layer(act_layer, _available_activation, Identity())

    def call(self, input):
        if self.pad is not None:
            x = self.pad(input)
        else:
            x = input
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class Bneck(Layer):
    def __init__(
            self,
            out_channels: int,
            exp_channels: int,
            kernel_size: int,
            stride: int,
            use_se: bool,
            act_layer: str,
            l2_reg: float=1e-5,
            batch_norm:bool=False,
            skip:bool = True,
            name:str = "Bneck"
    ):
        super().__init__(name=name)
        self.out_channels=out_channels
        self.exp_channels=exp_channels
        self.kernel_size=kernel_size
        self.stride = stride
        self.use_se = use_se
        self.act_layer=act_layer
        self.l2_reg=l2_reg
        self.batch_norm=batch_norm
        self.skip = skip

        def get_config(self):
            config = super().get_config()
            config.update({"out_channels": self.out_channels})
            config.update({"exp_channels": self.exp_channels})
            config.update({"kernel_size": self.kernel_size})
            config.update({"stride": self.stride})
            config.update({"use_se": self.use_se})
            config.update({"act_layer": self.act_layer})
            config.update({"l2_reg": self.l2_reg})
            config.update({"batch_norm": self.batch_norm})
            config.update({"skip": self.skip})
            return config

    def build(self, input_shape):
        self.in_channels = int(input_shape[-1])
        # Expand
        self.expand = ConvNormAct(
            self.exp_channels,
            kernel_size=1,
            norm_layer="bn" if self.batch_norm else None,
            act_layer=self.act_layer,
            use_bias=False,
            l2_reg=self.l2_reg,
            name="Expand",
        )
        # Depthwise
        dw_padding = (self.kernel_size - 1) // 2
        self.pad = tf.keras.layers.ZeroPadding2D(
            padding=dw_padding,
            name=f"Depthwise/Padding{dw_padding}x{dw_padding}",
        )
        self.depthwise = tf.keras.layers.DepthwiseConv2D(
            kernel_size=self.kernel_size,
            strides=self.stride,
            name=f"Depthwise/DWConv{self.kernel_size}x{self.kernel_size}",
            depthwise_regularizer=tf.keras.regularizers.l2(self.l2_reg),
            use_bias=False,
        )
        self.bn = BatchNormalization("Depthwise/BN") if self.batch_norm else None
        if self.use_se:
            self.se = SEBottleneck( l2_reg=self.l2_reg, name="Depthwise/SEBottleneck",)
        _available_activation = {
            "relu": tf.keras.layers.ReLU(name="Depthwise/ReLU"),
            "hard_sigmoid": HardSigmoid(name="Depthwise/HardSigmoid"),
            "hswish": HardSwish(name="Depthwise/HardSwish"),
        }
        self.act = get_layer(self.act_layer, _available_activation, Identity())

        # Project
        self.project = ConvNormAct(
            self.out_channels,
            kernel_size=1,
            norm_layer="bn" if self.batch_norm else None,
            act_layer=None,
            use_bias=False,
            l2_reg=self.l2_reg,
            name="Project",
        )
        super().build(input_shape)

    def call(self, input):
        x = self.expand(input)
        x = self.pad(x)
        x = self.depthwise(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.use_se:
            x = self.se(x)
        x = self.act(x)
        x = self.project(x)

        if self.skip and self.stride == 1 and self.in_channels == self.out_channels:
            return input + x
        else:
            return x

class SEBottleneck(Layer):
    def __init__(
            self,
            reduction: int=4,
            l2_reg: float=0.01,
            name: str="SEBottleneck",
    ):
        super().__init__(name=name)
        self.reduction = reduction
        self.l2_reg = l2_reg

    def build(self, input_shape):
        input_channels = int(input_shape[3])
        self.gap = GlobalAveragePooling2D(self.name)
        self.conv1 = ConvNormAct(
            input_channels // self.reduction,
            kernel_size=1,
            norm_layer=None,
            act_layer="relu",
            use_bias=False,
            l2_reg=self.l2_reg,
            name="Squeeze",
        )
        self.conv2 = ConvNormAct(
            input_channels,
            kernel_size=1,
            norm_layer=None,
            act_layer="hsigmoid",
            use_bias=False,
            l2_reg=self.l2_reg,
            name="Excite",
        )
        super().build(input_shape)

    def call(self, input):
        x = self.gap(input)
        x = self.conv1(x)
        x = self.conv2(x)
        return input * x
