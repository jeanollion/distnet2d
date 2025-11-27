from math import ceil

import tensorflow as tf
import numpy as np

from distnet_2d.model.layers import InferenceLayer, RelativeTemporalEmbedding


class TemporalAttention(InferenceLayer, tf.keras.layers.Layer):
    def __init__(self, num_heads:int, attention_filters:int=0, training_query_idx:list=None, inference_query_idx:list=None, frame_aware: bool = False, frame_max_distance:int=0, dropout:float=0.1, l2_reg:float=0., name="TemporalAttention"):
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
        self.temporal_dim=None
        self.training_query_idx = training_query_idx
        self.inference_query_idx = inference_query_idx
        self.frame_aware=frame_aware
        if self.frame_aware:
            assert frame_max_distance > 0, "in frame_aware mode frame max distance must be provided"
        self.frame_max_distance = frame_max_distance

    def get_config(self):
      config = super().get_config().copy()
      config.update({"num_heads": self.num_heads, "dropout":self.dropout, "filters":self.filters, "l2_reg":self.l2_reg, "frame_aware":self.frame_aware, "frame_max_distance":self.frame_max_distance, "training_query_idx":self.training_query_idx, "inference_query_idx":self.inference_query_idx})
      return config

    def build(self, input_shape):
        if self.frame_aware:
            input_shapes, t_index_shape = input_shape
        else:
            input_shapes = input_shape
        try:
            input_shapes = [s.as_list() for s in input_shapes]
        except:
            pass

        input_shape = input_shapes[0]
        for s in input_shapes[1:]:
            assert len(s)==len(input_shape) and all(i==j for i,j in zip(input_shape, s)), f"all tensors must have same input shape: {input_shape} != {s}"

        self.spatial_dims=input_shape[1:-1]
        self.temporal_dim = len(input_shapes)

        if self.training_query_idx is None:
            self.training_query_idx = list(range(self.temporal_dim))
        elif not isinstance(self.training_query_idx, (list, tuple)):
            self.training_query_idx = [self.training_query_idx]
        if self.inference_query_idx is None:
            self.inference_query_idx = self.training_query_idx
        elif not isinstance(self.inference_query_idx, (list, tuple)):
            self.inference_query_idx = [self.inference_query_idx]

        self.filters = input_shape[-1]
        if self.attention_filters is None or self.attention_filters<=0:
            self.attention_filters = int(ceil(self.filters / self.num_heads))

        self.attention_layer=tf.keras.layers.MultiHeadAttention(self.num_heads, key_dim=self.attention_filters, dropout=self.dropout, name="MultiHeadAttention")

        self.temp_embedding = tf.keras.layers.Embedding(
            self.temporal_dim,
            self.filters,
            embeddings_regularizer=tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None,
            name="TempEnc"
        ) if not self.frame_aware else RelativeTemporalEmbedding(
            max(self.temporal_dim, self.frame_max_distance),
            self.filters,
            l2_reg=self.l2_reg,
            name="TempEnc"
        )
        super().build(input_shape)

    def call(self, x, training:bool=None):
        '''
            x : [frame_list, t_index] if frame_aware or frame_list each tensor with shape (batch_size, y, x, channels)
        '''
        if self.frame_aware:
            frame_list, t_index = x
        else:
            frame_list = x
            t_index = tf.range(self.temporal_dim, dtype=tf.int32)

        shape = tf.shape(frame_list[0])
        B = shape[0]
        C = self.filters
        T = self.temporal_dim

        t_emb = self.temp_embedding(t_index)  # (T, C) or (B, T, C) if frame_aware
        t_emb = tf.reshape(t_emb, (1, 1, 1, T, C)) if not self.frame_aware else tf.reshape(t_emb, (B, 1, 1, T, C))

        frame_stacked = tf.stack(frame_list, axis=3)  # (B, Y, X, T, C)
        key = frame_stacked + t_emb
        value = frame_stacked

        idx_list = self.training_query_idx if not self.inference_mode else self.inference_query_idx

        query_list = [frame_list[i] + t_emb[:, :, :, i] for i in idx_list]
        # Flatten spatial dimensions for efficiency: (B*H*W, T, C)
        key_flat = tf.reshape(key, [-1, T, C])
        value_flat = tf.reshape(value, [-1, T, C])
        query_list_flat = [tf.reshape(query, [-1, 1, C]) for query in query_list]

        attention_output_list = [self.attention_layer(
            query=query_flat,
            value=value_flat,
            key=key_flat,
            training=training
        ) for query_flat in query_list_flat]
        attention_output_list = [tf.reshape(attention_output, shape) for attention_output in attention_output_list]
        return tf.stack(attention_output_list, 0)
