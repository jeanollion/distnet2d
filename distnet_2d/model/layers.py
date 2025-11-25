from tensorflow import pad
from ..utils.helpers import ensure_multiplicity
import tensorflow as tf
import numpy as np

class StopGradient(tf.keras.layers.Layer):
    def __init__(self, name:str="StopGradient"):
        super().__init__(name=name)

    def call(self, input):
        return tf.stop_gradient( input, name=self.name )

class InferenceLayer:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inference_mode = False


class NConvToBatch2D(InferenceLayer, tf.keras.layers.Layer):
    def __init__(self, n_conv:int, inference_idx, filters:int, compensate_gradient:bool = False, name: str= "NConvToBatch2D"):
        super().__init__(name=name)
        self.n_conv = n_conv
        self.filters = filters
        self.inference_idx=inference_idx
        self.compensate_gradient=compensate_gradient

    def get_config(self):
        config = super().get_config().copy()
        config.update({"n_conv": self.n_conv, "filters":self.filters, "compensate_gradient":self.compensate_gradient, "inference_idx":self.inference_idx})
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
            if isinstance(self.inference_idx, (tuple, list)):
                items = [self.convs[idx](input) for idx in self.inference_idx]
                return tf.concat(items, axis=0)
            else:
                return self.convs[self.inference_idx](input)
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

class SplitNConvToBatch2D(InferenceLayer, tf.keras.layers.Layer):
    def __init__(self, n_conv:int, inference_idx, filters:int, kernel, compensate_gradient:bool = False, name: str= "SplitNConvToBatch2D"):
        super().__init__(name=name)
        self.n_conv = n_conv
        self.filters = filters
        self.kernel = kernel
        self.inference_idx=inference_idx if isinstance(inference_idx, (list, tuple)) else [inference_idx]
        self.compensate_gradient=compensate_gradient
        self.convs = None
        self.split_layer = None
        self.inference_split_layer = None

    def get_config(self):
        config = super().get_config().copy()
        config.update({"n_conv": self.n_conv, "filters":self.filters, "kernel":self.kernel, "compensate_gradient":self.compensate_gradient, "inference_idx":self.inference_idx})
        return config

    def build(self, input_shape):
        self.convs = [
            tf.keras.layers.Conv2D(filters=self.filters, kernel_size=1, padding='same', activation="relu", name=f"Conv_{i}")
        for i in range(self.n_conv)]
        self.split_layer = SplitBatch(n_splits=self.n_conv)
        self.inference_split_layer = SplitBatch(n_splits=len(self.inference_idx))

        if self.compensate_gradient and self.n_conv>1:
            self.grad_fun = get_grad_weight_fun(float(self.n_conv))
            self.grad_fun_inv = get_grad_weight_fun(1./self.n_conv)
        else:
            self.grad_fun = None
            self.grad_fun_inv = None
        super().build(input_shape)

    def call(self, input): # (B, Y, X, F)
        if self.inference_mode: # only produce one output
            input_split = self.inference_split_layer(input)
            items = [self.convs[idx](input_split[i]) for i, idx in enumerate(self.inference_idx)]
            if len(self.inference_idx)>1:
                return tf.concat(items, axis=0)
            else:
                return items[0]
        # input = get_print_grad_fun(f"{self.name} before split")(input)
        if self.grad_fun_inv is not None:
            input = self.grad_fun_inv(input)
        input_split = self.split_layer(input)
        inputs = [conv(input_split[i]) for i, conv in enumerate(self.convs)] # N x (B, Y, X, F)
        # inputs[0] = get_print_grad_fun(f"{self.name} before concat")(inputs[0])
        output = tf.concat(inputs, axis = 0) # (N x B, Y, X, F)
        if self.grad_fun is not None:
            output = self.grad_fun(output) # compensate gradients to have same level in
        # output = get_print_grad_fun(f"{self.name} after concat")(output)
        return output

class InferenceAwareSelector(InferenceLayer, tf.keras.layers.Layer):
    def __init__(self, inference_idx, name: str= "SelectFeature"):
        super().__init__(name=name)
        self.inference_idx=inference_idx

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
    def __init__(self, inference_idx:list, train_idx:list=None, name: str= "SelectFeature2"):
        super().__init__(name=name)
        self.train_idx=train_idx
        self.inference_idx=inference_idx

    def get_config(self):
        config = super().get_config().copy()
        config.update({"train_idx":self.train_idx, "inference_idx":self.inference_idx})
        return config

    def call(self, input): # (N, B, Y, X, F)
        shape = tf.shape(input)
        if self.inference_mode: # only produce one output
            if isinstance(self.inference_idx, (tuple, list)):
                items = tf.gather(input, tf.constant(self.inference_idx, tf.int32), axis=0)
                return tf.reshape(items, [-1, shape[2], shape[3], shape[4]])
            else:
                return input[self.inference_idx]
        else:
            if self.train_idx is None:
                return tf.reshape(input, [-1, shape[2], shape[3], shape[4]])
            elif isinstance(self.train_idx, (tuple, list)):
                items = tf.gather(input, tf.constant(self.train_idx, tf.int32), axis=0)
                return tf.reshape(items, [-1, shape[2], shape[3], shape[4]])
            else:
                return input[self.train_idx]

class ChannelToBatch(tf.keras.layers.Layer):
    def __init__(self, compensate_gradient:bool = False, add_channel_axis:bool = True, name: str="ChannelToBatch"):
        self.compensate_gradient=compensate_gradient
        self.add_channel_axis=add_channel_axis
        super().__init__(name=name)

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
    def __init__(self, n_splits:int, compensate_gradient:bool = False, name:str="SplitBatch2D"):
        self.n_splits=n_splits
        self.compensate_gradient=compensate_gradient
        super().__init__(name=name)

    def get_config(self):
      config = super().get_config().copy()
      config.update({"n_splits": self.n_splits, "compensate_gradient":self.compensate_gradient})
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
        splits = tf.split(input, num_or_size_splits = self.n_splits, axis=0) # N x (1, B, Y, X, C)
        return [tf.squeeze(s, 0) for s in splits] # N x (B, Y, X, C)

class SplitReplaceConcatBatch(InferenceLayer, tf.keras.layers.Layer):
    def __init__(self, n_splits:int, replace_idx:int, compensate_gradient:bool = False, name:str="SplitReplaceMergeBatch2D"):
        self.n_splits=n_splits
        self.replace_idx = replace_idx
        self.compensate_gradient=compensate_gradient
        self.split_layer = None
        super().__init__(name=name)

    def get_config(self):
      config = super().get_config().copy()
      config.update({"n_splits": self.n_splits, "compensate_gradient":self.compensate_gradient, "replace_idx":self.replace_idx})
      return config

    def build(self, input_shape):
        relace_shape, concat_shape = input_shape
        self.split_layer = SplitBatch(n_splits=self.n_splits, compensate_gradient=self.compensate_gradient)
        self.split_layer.build(concat_shape)
        super().build(input_shape)

    def call(self, input): # (N x B, Y, X, C)
        replace, concat = input
        if self.inference_mode: # N = 1
            return replace
        else:
            input_list = self.split_layer(concat)  # N x (B, Y, X, C)
            input_list[self.replace_idx] = replace
        return tf.concat(input_list, 0) # (N x B, Y, X, C)

class BatchToChannel(InferenceLayer, tf.keras.layers.Layer):
    def __init__(self, n_splits:int, n_splits_inference:int=1, inference_idx=None, compensate_gradient:bool = False, name:str= "BatchToChannel"):
        self.n_splits=n_splits
        self.compensate_gradient = compensate_gradient
        self.n_splits_inference=n_splits_inference
        if inference_idx is not None:
            if isinstance(inference_idx, int):
                inference_idx=[inference_idx]
            assert isinstance(inference_idx, list), f"inference_idx must be a list of indices: {inference_idx}"
            assert np.all(np.asarray(inference_idx)<n_splits_inference), f"all inference_idx must be lower than {n_splits_inference} got {inference_idx}"
        self.inference_idx = inference_idx
        super().__init__(name=name, autocast=False)

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

class FrameDistanceEmbedding(tf.keras.layers.Layer):
    def __init__(self, input_dim:int, output_dim:int, frame_prev_idx:list, frame_next_idx:list , name:str="FrameDistanceEmbedding"):
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.frame_next_idx=frame_next_idx
        self.frame_prev_idx=frame_prev_idx
        assert len(frame_prev_idx) == len(frame_next_idx)
        self.embedding=None
        super().__init__(name=name)

    def get_config(self):
      config = super().get_config().copy()
      config.update({"input_dim": self.input_dim, "output_dim":self.output_dim, "frame_prev_idx":self.frame_prev_idx, "frame_next_idx":self.frame_next_idx})
      return config

    def build(self, input_shape):
        self.embedding = tf.keras.layers.Embedding(input_dim=self.input_dim, output_dim=self.output_dim)
        super().build(input_shape)

    def call(self, frame_index): # (B, 1, 1, FW)
        frame_distance = tf.cast( tf.gather(frame_index[:, 0, 0], self.frame_next_idx, axis=-1) - tf.gather(frame_index[:, 0, 0], self.frame_prev_idx, axis=-1), tf.int32 ) # (B, N)
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
    """Combines learned embeddings with sinusoidal encoding."""

    def __init__(self, max_distance, embedding_dim, l2_reg=None, **kwargs):
        super().__init__(**kwargs)
        self.max_distance = max_distance
        self.embedding_dim = embedding_dim
        self.vocab_size = 2 * max_distance + 1
        self.l2_reg=l2_reg

    def build(self, input_shape):
        input_shape, central_frame = input_shape
        # Learned component (half dimensions)
        self.learned_embedding = tf.keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim // 2,
            embeddings_regularizer=tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None,
            name="learned_temporal"
        )
        super().build(input_shape)

    def get_config(self):
      config = super().get_config().copy()
      config.update({"max_distance": self.max_distance, "embedding_dim":self.embedding_dim, "l2_reg":self.l2_reg})
      return config

    def call(self, distances):

        # Learned part
        indices = tf.clip_by_value(distances + self.max_distance, 0, self.vocab_size - 1)
        learned = self.learned_embedding(indices)  # (T, embedding_dim//2)

        # Sinusoidal part
        sinusoidal = sinusoidal_temporal_encoding(distances, self.embedding_dim // 2, dtype=learned.dtype)

        # Concatenate
        return tf.concat([learned, sinusoidal], axis=-1)  # (T, embedding_dim)


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

class Stack(tf.keras.layers.Layer):
    def __init__(self, axis, **kwargs):
        super(Stack, self).__init__(**kwargs)
        self.axis = axis
        self.supports_masking = True

    def call(self, inputs):
        return tf.stack(inputs, axis=self.axis)

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, (list, tuple)):
            raise ValueError("StackLayer expects a list of input tensors.")

        # Check that all input shapes are compatible for stacking
        first_shape = tf.TensorShape(input_shape[0]).as_list()
        for shape in input_shape[1:]:
            current_shape = tf.TensorShape(shape).as_list()
            if first_shape != current_shape:
                raise ValueError("All input shapes must be the same for stacking.")

        # Compute the output shape
        output_shape = first_shape[:self.axis+1] + [len(input_shape)] + first_shape[self.axis+1:]
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
            weight_scaled:bool = False,
            activation: str="relu",
            compensate_gradient:bool = False,
            l2_reg: float=0,
            output_dtype:str = None,
            name: str="Combine",
        ):
        self.activation = activation
        self.filters= filters
        self.kernel_size=kernel_size
        self.weight_scaled = weight_scaled
        self.compensate_gradient = compensate_gradient
        self.l2_reg=l2_reg
        self.output_dtype=output_dtype
        super().__init__(name=name)

    def get_config(self):
      config = super().get_config().copy()
      config.update({"activation": self.activation, "filters":self.filters, "kernel_size":self.kernel_size, "weight_scaled":self.weight_scaled, "compensate_gradient":self.compensate_gradient, "l2_reg":self.l2_reg, "output_dtype":self.output_dtype})
      return config

    def build(self, input_shape):
        if self.filters <=0:
            f = [s[-1] for s in input_shape]
            filters = f[0]
            assert np.all(np.array(f) == f[0]), "when filter is not provided, all input tensor must have same filter number"
        else:
            filters = self.filters
        self.concat = tf.keras.layers.Concatenate(axis=-1, name = self.name+"_concat")
        if self.weight_scaled:
            self.combine_conv = WSConv2D(
                filters=filters,
                kernel_size=self.kernel_size,
                padding='same',
                activation=self.activation,
                kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg>0 else None,
                name=self.name+"_conv1x1")
        else:
            self.combine_conv = Conv2DWithDtype(
                filters=filters,
                kernel_size=self.kernel_size,
                padding='same',
                activation=self.activation,
                kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg>0 else None,
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
            output_dtype=None,
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
        self.output_dtype=output_dtype
        self.split_conv=split_conv

    def get_config(self):
      config = super().get_config().copy()
      config.update({"activation": self.activation, "kernel_size":self.kernel_size, "dilation":self.dilation, "dropout_rate":self.dropout_rate, "batch_norm":self.batch_norm, "weight_scaled":self.weight_scaled, "weighted_sum":self.weighted_sum, "l2_reg":self.l2_reg, "output_dtype":self.output_dtype, "split_conv":self.split_conv})
      return config

    def build(self, input_shape):
        input_channels = int(input_shape[-1])
        conv_fun = WSConv2D if self.weight_scaled else (SplitConv2D if self.split_conv else tf.keras.layers.Conv2D)
        self.conv1 = conv_fun(
            filters=input_channels,
            kernel_size=self.kernel_size,
            strides=1,
            padding='same',
            name=f"Conv1_{ker_size_to_string(self.kernel_size)}",
            activation="linear",
            #kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg>0 else None
        )
        self.conv2 = conv_fun(
            filters=input_channels,
            kernel_size=self.kernel_size,
            dilation_rate = self.dilation,
            strides=1,
            padding='same',
            name=f"Conv2_{ker_size_to_string(self.kernel_size)}",
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
        if self.output_dtype is not None:
            x = tf.cast(x, dtype=self.output_dtype)
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
            output_dtype=None,
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
            name=f"Conv",
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
        if self.output_dtype is not None:
            x = tf.cast(x, dtype=self.output_dtype)
        if self.batch_norm:
            x = self.bn(x, training = training)
        if self.dropout_rate>0:
            x = self.drop(x, training = training)
        return self.activation_layer(x)


class Conv2DWithDtype(tf.keras.layers.Conv2D):
    def __init__(self, *args, output_dtype:str=None, **kwargs):
        self._activation = kwargs.pop('activation', None)
        super(Conv2DWithDtype, self).__init__(*args, activation=None, **kwargs)
        self.output_dtype = output_dtype
        self.activation = None  # Will be set in build()

    def build(self, input_shape):
        super(Conv2DWithDtype, self).build(input_shape)
        if self._activation is not None:
            self.activation = tf.keras.activations.get(self._activation)

    def call(self, inputs):
        output = super(Conv2DWithDtype, self).call(inputs)
        if self.output_dtype is not None:
            output = tf.cast(output, dtype=self.output_dtype)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        config = super(Conv2DWithDtype, self).get_config()
        config.update({
            'output_dtype': self.output_dtype,
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
    ):
        super().__init__(name=name)
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
            activation="linear",
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg>0 else None,
            name=f"tConv{ker_size_to_string(self.kernel_size)}",
        )
        self.activation_layer = tf.keras.activations.get(self.activation)
        if self.dropout_rate>0:
            self.drop = tf.keras.layers.SpatialDropout2D(self.dropout_rate, name=f"Dropout")
        if self.batch_norm:
            #self.bn = MockBatchNormalization(name = f"/MockBatchNormalization")
            self.bn = tf.keras.layers.BatchNormalization(name = f"BatchNormalization")
        super().build(input_shape)

    def call(self, input, training=None):
        x = self.conv(input)
        if self.output_dtype is not None:
            x = tf.cast(x, dtype=self.output_dtype)
        if self.batch_norm:
            x = self.bn(x, training = training)
        if self.dropout_rate>0:
            x = self.drop(x, training = training)
        return self.activation_layer(x)


class Conv2DTransposeWithDtype(tf.keras.layers.Conv2DTranspose):
    def __init__(self, *args, output_dtype=None, **kwargs):
        self._activation = kwargs.pop('activation', None)
        super(Conv2DTransposeWithDtype, self).__init__(*args, activation=None, **kwargs)
        self.output_dtype = output_dtype
        self.activation = None  # Will be set in build()

    def build(self, input_shape):
        super(Conv2DTransposeWithDtype, self).build(input_shape)
        if self._activation is not None:
            self.activation = tf.keras.activations.get(self._activation)

    def call(self, inputs):
        output = super(Conv2DTransposeWithDtype, self).call(inputs)
        if self.output_dtype is not None:
            output = tf.cast(output, dtype=self.output_dtype)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        config = super(Conv2DTransposeWithDtype, self).get_config()
        config.update({
            'output_dtype': self.output_dtype,
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
        config.update({'output_dtype': self.output_dtype.name})
        return config


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
        try:
            input_shape = input_shape.as_list()
        except:
            pass
        input_filters = input_shape[-1]
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
        self.convs = [conv_fun(f"Conv{i}") for i in range(3)]
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
    def __init__(self, *args, eps=1e-4, use_gain=True, dropout_rate = 0, kernel_initializer="he_normal", output_dtype:str=None, **kwargs):
        activation = kwargs.pop("activation", "linear") # bypass activation
        gamma = kwargs.pop("gamma", get_gamma(activation if isinstance(activation, str) else tf.keras.activations.serialize(activation)))
        super().__init__(kernel_initializer=kernel_initializer, *args, **kwargs)
        self.eps = eps
        self.use_gain = use_gain
        self.activation_layer = tf.keras.activations.get(activation)
        self.dropout_rate = dropout_rate
        self.gamma = gamma
        self.output_dtype=output_dtype

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
        config.update({"eps":self.eps, "use_gain": self.use_gain, "output_dtype":self.output_dtype, "activation_layer":tf.keras.activations.serialize(self.activation_layer), "dropout_rate":self.dropout_rate, "gamma":self.gamma})
        return config

    def call(self, input, training=None):
        x = super().call(input)
        if self.dropout_rate>0:
            x = self.dropout(x, training = training)
        if self.output_dtype is not None:
            x = tf.cast(x, dtype=self.output_dtype)
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
