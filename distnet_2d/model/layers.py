from tensorflow import pad
from tensorflow.keras.layers import Layer, GlobalAveragePooling2D, Reshape, Conv2D, Multiply, Conv3D, UpSampling2D
from tensorflow.keras import Model
from ..utils.helpers import ensure_multiplicity, get_nd_gaussian_kernel
from tensorflow.python.keras.engine.input_spec import InputSpec
import tensorflow as tf
import numpy as np

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

class Combine(Layer):
    def __init__(
            self,
            filters: int,
            activation: str="relu",
            #l2_reg: float=1e-5,
            use_bias:bool = True,
            name: str="Combine",
        ):
        super().__init__(name=name)
        self.concat = tf.keras.layers.Concatenate(axis=-1, name = "Concat")
        self.combine_conv = Conv2D(
            filters=filters,
            kernel_size=1,
            padding='same',
            activation=activation,
            use_bias=use_bias,
            # kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
            name="Conv1x1")

    def call(self, input):
        x = self.concat(input)
        x = self.combine_conv(x)
        return x


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
