from math import ceil

import tensorflow as tf
import numpy as np

from distnet_2d.model.layers import InferenceLayer, RelativeTemporalEmbedding


class TemporalCrossAttention(tf.keras.layers.Layer):
    def __init__(self,
                 num_heads:int, attention_filters:int=0,
                 dropout:float=0.1, l2_reg:float=0, embedding_l2_reg:float=1e-5, layer_normalization:bool = True,
                 skip_connection:bool = True,  name="TemporalCrossAttention"):
        '''
            filters : number of output channels
            if positional_encoding: filters must correspond to input channel number
            adapted from: https://www.tensorflow.org/tutorials/text/transformer
        '''
        super().__init__(name=name)
        self.attention_layer = None
        self.num_heads=num_heads
        self.attention_filters = attention_filters
        self.filters = None
        self.skip_connection = skip_connection
        self.dropout=dropout
        self.l2_reg=l2_reg
        self.embedding_l2_reg=embedding_l2_reg
        self.layer_normalization=layer_normalization
        self.temporal_dim=None


    def get_config(self):
      config = super().get_config().copy()
      config.update({"num_heads": self.num_heads, "dropout":self.dropout, "filters":self.filters, "skip_connection":self.skip_connection, "layer_normalization":self.layer_normalization, "l2_reg":self.l2_reg, "embedding_l2_reg":self.embedding_l2_reg})
      return config

    def build(self, input_shape): # (Tq, B, Y, X, C), (T, B, Y, X, C)
        q_input_shape, input_shape = input_shape
        try:
            input_shape = input_shape.as_list()
            q_input_shape = q_input_shape.as_list()
        except:
            pass
        for i, (q, k) in enumerate(zip(q_input_shape, input_shape[1:])):
            assert q==k, f"invalid q/k dim on axis: {i}"

        self.spatial_dims=input_shape[2:-1]
        self.temporal_dim = input_shape[0]
        self.filters = input_shape[-1]

        if self.attention_filters is None or self.attention_filters<=0:
            self.attention_filters = int(ceil(self.filters / self.num_heads))

        self.attention_layer=tf.keras.layers.MultiHeadAttention(
            self.num_heads,
            key_dim=self.attention_filters,
            dropout=self.dropout,
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None,
            name="MultiHeadAttention")

        self.temp_embedding = tf.keras.layers.Embedding(
            input_dim = self.temporal_dim, output_dim=self.filters,
            embeddings_regularizer=tf.keras.regularizers.l2(self.embedding_l2_reg) if self.embedding_l2_reg > 0 else None
        )
        if self.layer_normalization:
            self.ln_q = tf.keras.layers.LayerNormalization()
            self.ln_k = tf.keras.layers.LayerNormalization() # input + emb
            self.ln_v = tf.keras.layers.LayerNormalization() # v is only input
        super().build(input_shape)

    def call(self, x, training:bool=None):
        '''
            x : [query, key] (B, Y, X, C), (T, B, Y, X, C)
        '''
        input_query, input_frames = x

        frame_shape = tf.shape(input_frames)[1:] # B, Y, X, C
        frames = tf.transpose(input_frames, [1, 2, 3, 0, 4]) # B, Y, X, T, C
        B, Y, X, _ = tf.unstack(frame_shape)
        T = self.temporal_dim
        C = self.filters

        t_emb = self.temp_embedding(tf.range(self.temporal_dim, dtype=tf.int32))  # (1, T, C)
        t_emb = tf.reshape(t_emb, (1, 1, 1, T, C))
        key = frames + t_emb
        if self.layer_normalization:
            key = self.ln_k(key)
            value = self.ln_v(frames)
            query = self.ln_q(input_query)
        else:
            value = frames
            query = input_query

        # Flatten spatial dimensions for efficiency: (B*Y*X, T, C)
        key_flat = tf.reshape(key, [-1, T, C])
        value_flat = tf.reshape(value, [-1, T, C])
        query_flat = tf.reshape(query, [-1, 1, C])
        out = self.attention_layer( query=query_flat, value=value_flat, key=key_flat, training=training ) # BxYxX, 1, C
        out = tf.reshape(out, [B, Y, X, C])
        if self.skip_connection:
            out = out + input_query
        return out
