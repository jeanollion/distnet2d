import tensorflow as tf
import numpy as np

class SpatialAttention2D(tf.keras.layers.Layer):
    def __init__(self, positional_encoding:bool=True, filters:int=0, return_attention:bool=False, l2_reg:float=0, name="Attention"):
        '''
            filters : number of output channels
            if positional_encoding: filters must correspond to input channel number
            adapted from: https://www.tensorflow.org/tutorials/text/transformer
        '''
        super().__init__(name=name)
        self.positional_encoding=positional_encoding
        self.filters = filters
        self.return_attention=return_attention
        self.l2_reg=l2_reg

    def get_config(self):
      config = super().get_config().copy()
      config.update({"positional_encoding": self.positional_encoding, "filters":self.filters, "return_attention":self.return_attention, "l2_reg":self.l2_reg})
      return config

    def build(self, input_shape):
        input_shape_, input_shape = input_shape
        self.spatial_dims=input_shape[1:-1]
        self.spatial_dim = np.prod(self.spatial_dims)
        if self.filters<=0:
            self.filters = input_shape[-1]
        self.wq = tf.keras.layers.Dense(self.filters, name="Q", kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg>0 else None)
        self.wk = tf.keras.layers.Dense(self.filters, name="K", kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg>0 else None)
        self.wv = tf.keras.layers.Dense(self.filters, name="V", kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg>0 else None)
        if self.positional_encoding=="2D":
            self.pos_embedding_y = tf.keras.layers.Embedding(self.spatial_dims[0], input_shape[-1], embeddings_regularizer=tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg>0 else None, name="PosEncY")
            self.pos_embedding_x = tf.keras.layers.Embedding(self.spatial_dims[1], input_shape[-1], embeddings_regularizer=tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg>0 else None, name="PosEncX")
        elif self.positional_encoding:
            self.pos_embedding = tf.keras.layers.Embedding(self.spatial_dim, input_shape[-1], embeddings_regularizer=tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg>0 else None, name="PosEnc")
        super().build(input_shape)

    def call(self, x):
        '''
            x : tensor with shape (batch_size, y, x, channels)
        '''
        [input, output] = x
        shape = tf.shape(output)
        batch_size = shape[0]
        #spatial_dims = shape[1:-1]
        #spatial_dim = tf.reduce_prod(spatial_dims)
        depth_dim = shape[3]
        if self.positional_encoding=="2D":
            y_index = tf.range(self.spatial_dims[0], dtype=tf.int32)
            pos_emb_y = self.pos_embedding_y(y_index) # (y, self.filters)
            pos_emb_y = tf.reshape(pos_emb_y, (self.spatial_dims[0], 1, self.filters)) #(y, 1, self.filters)
            x_index = tf.range(self.spatial_dims[1], dtype=tf.int32)
            pos_emb_x = self.pos_embedding_x(x_index) # (x, self.filters)
            pos_emb_x = tf.reshape(pos_emb_x, (1, self.spatial_dims[1], self.filters)) #(1, x, self.filters)
            pos_emb = pos_emb_x + pos_emb_y # broadcast to (y, x, self.filters)
            #pos_emb_y = tf.transpose(pos_emb_y, [2, 0, 1]) #(self.filters, y, 1)
            #pos_emb_x = tf.transpose(pos_emb_x, [2, 0, 1]) #(self.filters, 1, x)
            #pos_emb = tf.matmul(pos_emb_y, pos_emb_x, transpose_b=False, name=self.name+"pos_enc") #(self.filters, y, x) // TODO either scale or simply add the two vectors with broadcast
            #pos_emb = tf.transpose(pos_emb, [1, 2, 0]) #(y, x, self.filters)
            output = output + pos_emb # broadcast
            input = input + pos_emb # broadcast
        elif self.positional_encoding:
            x_index = tf.range(self.spatial_dim, dtype=tf.int32)
            pos_emb = self.pos_embedding(x_index) # (spa_dim, self.filters)
            pos_emb = tf.reshape(pos_emb, (self.spatial_dims[0], self.spatial_dims[1], self.filters)) #for broadcasting purpose
            output = output + pos_emb # broadcast
            input = input + pos_emb # broadcast

        q = self.wq(output)  # (batch_size, *spa_dims, self.filters)
        k = self.wk(input)  # (batch_size, *spa_dims, self.filters)
        v = self.wv(input)  # (batch_size, *spa_dims, self.filters)

        q = tf.reshape(q, (batch_size, -1, depth_dim)) # (batch_size, spa_dim, self.filters)
        k = tf.reshape(k, (batch_size, -1, depth_dim))
        v = tf.reshape(v, (batch_size, -1, depth_dim))

        # scaled_attention.shape == (batch_size, spa_dims, depth)
        # attention_weights.shape == (batch_size, spa_dims, spa_dims)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v)
        output = tf.reshape(scaled_attention, (batch_size, self.spatial_dims[0], self.spatial_dims[1], self.filters))
        if self.return_attention:
            return output, attention_weights
        else:
            return output

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return input_shape[:-1]+(self.filters,), (input_shape[0],self.spatial_dim,self.spatial_dim)
        else:
            return input_shape[:-1]+(self.filters,)

def scaled_dot_product_attention(q, k, v):
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
    dk = tf.cast(tf.shape(k)[-1], matmul_qk.dtype)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1, name="AttentionWeights")  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights
