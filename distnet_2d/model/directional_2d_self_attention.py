import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Reshape, Embedding, Concatenate, Conv2D
from tensorflow.keras.models import Model
from .layers import Combine
import numpy as np

class Directional2DSelfAttention(Layer):
    def __init__(self, positional_encoding:bool=True, combine_xy:bool=True, combine_filters:int=0, filters:int=0, return_attention:bool=False, name:str="DirectionalSelfAttention"):
        '''
            filters : number of output channels (0: use input channel)
            spatial_dim : spatial dimensions of input tensor (x , y)
            if positional_encoding: depth must correspond to input channel number
            adapted from: https://www.tensorflow.org/tutorials/text/transformer
        '''
        super().__init__(name=name)
        self.filters = filters
        self.combine_filters=combine_filters
        self.combine_xy=combine_xy
        self.positional_encoding=positional_encoding
        self.return_attention=return_attention

    def build(self, input_shape):
        spatial_dims=input_shape[1:-1]
        assert len(spatial_dims) == 2 and spatial_dims[0]==spatial_dims[1], "only available for 2D squared images"
        self.spatial_dim = spatial_dims[0]
        if self.filters<=0:
            self.filters = input_shape[-1]//self.spatial_dim
            print(f"number of filters: {self.filters} input shape: {input_shape[-1]}, spatial_dim: {self.spatial_dim}")
        if self.combine_filters<=0:
            self.combine_filters = input_shape[-1]
        self.wq = Dense(self.filters * self.spatial_dim, name="Q")
        self.wk = Dense(self.filters * self.spatial_dim, name="K")
        self.wv = Dense(self.filters * self.spatial_dim, name="W")
        if self.positional_encoding:
            self.pos_embedding = Embedding(self.spatial_dim, input_shape[-1], name="PosEnc")
        if self.combine_xy:
            assert self.combine_filters>0
            self.output_depth=self.combine_filters
            self.combine_xy_op = Combine(filters = self.combine_filters, name="CombineXY")
        else:
            self.combine_xy_op = None
            self.output_depth = self.filters
        super().build(input_shape)

    def call(self, input):
        '''
            input : tensor with shape (batch_size, y, x, channels)
        '''
        shape = tf.shape(input)
        batch_size = shape[0]
        channels = shape[3]
        x = input # (batch size, spa_dim, spa_dim, channels)
        y = input
        if self.positional_encoding:
            spa_index = tf.range(self.spatial_dim, dtype=tf.int32)
            pos_emb = self.pos_embedding(spa_index) # (spa_dim, channels)
            x = x + tf.reshape(pos_emb, (1, self.spatial_dim, channels)) # broadcast
            y = y + tf.reshape(pos_emb, (self.spatial_dim, 1, channels)) # broadcast

        y = tf.reshape(y, (batch_size, -1, channels * self.spatial_dim))
        x = tf.reshape(tf.transpose(x, [0, 2, 1, 3]), (batch_size, self.spatial_dim, channels * self.spatial_dim))

        x, wx = self.compute_attention(x)
        x = tf.reshape(x, (batch_size, self.spatial_dim, self.spatial_dim, self.filters) )
        x = tf.transpose(x, [0, 2, 1, 3])
        y, wy = self.compute_attention(y)
        y = tf.reshape(y, (batch_size, self.spatial_dim, self.spatial_dim, self.filters) )
        if self.combine_xy:
            xy = self.combine_xy_op([y, x])
            if self.return_attention:
                return xy, wy, wx
            else:
                return xy
        else:
            if self.return_attention:
                return y, x, wy, wx
            else:
                return y, x

    def compute_attention(self, input):
        q = self.wq(input)  # (batch_size, spa_dim, self.filters * spa_dim)
        k = self.wk(input)  # (batch_size, spa_dim, self.filters * spa_dim)
        v = self.wv(input)  # (batch_size, spa_dim, self.filters * spa_dim)
        # scaled_attention.shape == (batch_size, spa_dims, depth)
        # attention_weights.shape == (batch_size, spa_dims, spa_dims)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, self.name)
        return scaled_attention, attention_weights

    def compute_output_shape(self, input_shape):
        if self.combine_xy:
            if self.return_attention:
                return input_shape[:-1]+(self.output_depth,), (input_shape[0],self.spatial_dim,self.spatial_dim), (input_shape[0],self.spatial_dim,self.spatial_dim)
            else:
                return input_shape[:-1]+(self.output_depth,)
        else:
            if self.return_attention:
                return input_shape[:-1]+(self.output_depth,), input_shape[:-1]+(self.output_depth,), (input_shape[0],self.spatial_dim,self.spatial_dim), (input_shape[0],self.spatial_dim,self.spatial_dim)
            else:
                return input_shape[:-1]+(self.output_depth,), input_shape[:-1]+(self.output_depth,)

def scaled_dot_product_attention(q, k, v, name=""):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)

    Returns:
    output, attention_weights

    from : https://www.tensorflow.org/tutorials/text/transformer
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1, name=name+"_attention_weights")  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights
