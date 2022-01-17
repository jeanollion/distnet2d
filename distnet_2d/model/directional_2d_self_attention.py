import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Reshape, Embedding, Concatenate, Conv2D
from tensorflow.keras.models import Model
import numpy as np

class Directional2DSelfAttention(Model):
    def __init__(self, d_model, spatial_dim, combine_filters, positional_encoding=True, combine_xy=True, return_attention=False, name="self_attention"):
        '''
            d_model : number of output channels
            spatial_dim : spatial dimensions of input tensor (x , y)
            if positional_encoding: depth must correspond to input channel number
            adapted from: https://www.tensorflow.org/tutorials/text/transformer
        '''
        super().__init__(name=name)
        self.d_model = d_model
        if isinstance(spatial_dim, (tuple, list)):
            assert len(spatial_dim) == 2 and spatial_dim[0]==spatial_dim[1], "only available for 2D squared images"
            self.spatial_dim=spatial_dim[0]
        else:
            self.spatial_dim=spatial_dim
        self.wq = Dense(self.d_model * self.spatial_dim, name=name+"_q")
        self.wk = Dense(self.d_model * self.spatial_dim, name=name+"_k")
        self.wv = Dense(self.d_model * self.spatial_dim, name=name+"_w")
        self.positional_encoding=positional_encoding
        if positional_encoding:
            self.pos_embedding = Embedding(self.spatial_dim, d_model * self.spatial_dim, name=name+"pos_enc")
        if combine_xy:
            assert combine_filters>0
            self.output_depth=combine_filters
            self.concat_xy = Concatenate(axis=3, name = name+"_concat_xy")
            self.combine_xy = Conv2D(kernel_size=1, filters = combine_filters, activation='relu', name=name+"_conmbine_xy")
        else:
            self.combine_xy = None
            self.output_depth = d_model
        self.return_attention=return_attention

    def call(self, input):
        '''
            input : tensor with shape (batch_size, y, x, channels)
        '''
        shape = tf.shape(input)
        batch_size = shape[0]
        depth_dim = shape[3]
        x = tf.reshape(input, (batch_size, -1, depth_dim * self.spatial_dim))
        y = tf.reshape(tf.transpose(input, [0, 2, 1, 3]), (batch_size, -1, depth_dim * self.spatial_dim))

        if self.positional_encoding:
            x_index = tf.range(self.spatial_dim, dtype=tf.int32)
            pos_emb = self.pos_embedding(x_index) # (spa_dim, d_model)
            x = x + pos_emb # broadcast
            y = y + pos_emb # broadcast

        x, wx = self.compute_attention(x)
        x = tf.reshape(x, (batch_size, self.spatial_dim, self.spatial_dim, self.d_model) )
        x = tf.transpose(x, [0, 2, 1, 3])
        y, wy = self.compute_attention(y)
        y = tf.reshape(y, (batch_size, self.spatial_dim, self.spatial_dim, self.d_model) )
        if self.combine_xy is not None:
            xy = self.concat_xy([y, x])
            xy = self.combine_xy(xy)
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
        q = self.wq(input)  # (batch_size, spa_dim, d_model * spa_dim)
        k = self.wk(input)  # (batch_size, spa_dim, d_model * spa_dim)
        v = self.wv(input)  # (batch_size, spa_dim, d_model * spa_dim)
        # scaled_attention.shape == (batch_size, spa_dims, depth)
        # attention_weights.shape == (batch_size, spa_dims, spa_dims)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, self.name)
        return scaled_attention, attention_weights

    def compute_output_shape(self, input_shape):
        if self.combine_xy is not None:
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
