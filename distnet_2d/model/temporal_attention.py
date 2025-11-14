from math import ceil

import tensorflow as tf
import numpy as np

from distnet_2d.model.layers import InferenceLayer


class TemporalAttention(InferenceLayer, tf.keras.layers.Layer):
    def __init__(self, num_heads:int=1, attention_filters:int=0, intra_mode:bool=True, inference_idx:int=None, return_list:bool = False, dropout:float=0.1, l2_reg:float=0., name="TemporalAttention"):
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
        self.dropout=dropout
        self.l2_reg=l2_reg
        self.intra_mode = intra_mode # if true: input is (idx, array), otherwise input is (tensor, array)
        self.temporal_dim=None
        self.inference_idx = inference_idx
        if self.intra_mode:
            assert inference_idx is not None and ( min(inference_idx) >= 0 if isinstance(inference_idx, (list, tuple )) else inference_idx >= 0 )
        self.return_list=return_list

    def get_config(self):
      config = super().get_config().copy()
      config.update({"num_heads": self.num_heads, "dropout":self.dropout, "filters":self.filters, "l2_reg":self.l2_reg, "return_list":self.return_list, "intra_mode":self.intra_mode, "inference_idx":self.inference_idx})
      return config

    def build(self, input_shape):
        if self.intra_mode:
            input_shapes = input_shape
        else:
            query_shapes, input_shapes = input_shape
        try:
            input_shapes = [s.as_list() for s in input_shapes]
        except:
            pass

        input_shape = input_shapes[0]
        for s in input_shapes[1:]:
            assert len(s)==len(input_shape) and all(i==j for i,j in zip(input_shape, s)), f"all tensors must have same input shape: {input_shape} != {s}"

        if not self.intra_mode:
            query_shape = query_shapes[0]
            if len(query_shapes)>1:
                for s in query_shapes[1:]:
                    assert len(s) == len(query_shape) and all(i == j for i, j in zip(query_shape, s)), f"all query tensors must have same input shape: {query_shape} != {s}"
            try:
                query_shape = query_shape[0].as_list()
            except:
                pass
            assert len(query_shape)==len(input_shape) and all(i==j for i,j in zip(input_shape[:-1], query_shape[:-1])), f"query tensor tensors must have same input shape: {input_shape} != {query_shapes}"

        self.spatial_dims=input_shape[1:-1]
        self.temporal_dim = len(input_shapes)
        self.filters = input_shape[-1]
        if self.attention_filters is None or self.attention_filters<=0:
            self.attention_filters = int(ceil(self.filters / self.num_heads))

        self.attention_layer=tf.keras.layers.MultiHeadAttention(self.num_heads, key_dim=self.attention_filters, dropout=self.dropout, name="MultiHeadAttention")

        key_shape = (None, self.temporal_dim, self.filters)
        if self.intra_mode:
            query_shape = (None, 1, self.filters)
        else:
            query_shape = (None, 1, query_shape[-1])
        self.attention_layer._build_from_signature(query=query_shape, value=key_shape, key=key_shape)
        self.temp_embedding = tf.keras.layers.Embedding(self.temporal_dim, self.filters, embeddings_regularizer=tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None, name="TempEnc")
        super().build(input_shape)

    def call(self, x, training:bool=None):
        '''
            x : [idx, all_values] each tensor with shape (batch_size, y, x, channels)
        '''
        if self.intra_mode:
            all_values = x
        else:
            query_list, all_values = x
        C = self.filters
        T = self.temporal_dim
        t_index = tf.range(T, dtype=tf.int32)
        t_emb = self.temp_embedding(t_index)  # (temps_dim, C)
        t_emb = tf.reshape(t_emb, (1, 1, 1, T, C))
        shape = tf.shape(all_values[0]) if self.intra_mode else tf.shape(query_list[0])
        key = tf.stack(all_values, axis=3)  # (B, Y, X, T, C)
        key = key + t_emb
        if self.intra_mode:
            if self.inference_mode:
                idx_list = [self.inference_idx] if not isinstance(self.inference_idx, (list, tuple)) else self.inference_idx
            else:
                idx_list = range(T)
            query_list = [all_values[i] + t_emb[:, :, :, i] for i in idx_list]
        # Flatten spatial dimensions for efficiency: (B*H*W, T, C)
        key_flat = tf.reshape(key, [-1, T, C])
        query_list_flat = [tf.reshape(query, [-1, 1, C]) for query in query_list]
        value_flat = key_flat  # same as key

        attention_output_list = [self.attention_layer(
            query=query_flat,
            value=value_flat,
            key=key_flat,
            training=training
        ) for query_flat in query_list_flat]
        attention_output_list = [tf.reshape(attention_output, shape) for attention_output in attention_output_list]
        if self.return_list:
            return attention_output_list
        if self.inference_mode and not isinstance(self.inference_idx, (list, tuple)):
            return attention_output_list[0]
        else:
            return tf.concat(attention_output_list, 0)
