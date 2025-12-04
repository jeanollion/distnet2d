from math import ceil

import tensorflow as tf
import numpy as np

from distnet_2d.model.layers import InferenceLayer, RelativeTemporalEmbedding


class TemporalAttention(InferenceLayer, tf.keras.layers.Layer):
    def __init__(self,
                 num_heads:int, attention_filters:int=0,
                 training_query_idx:list=None, inference_query_idx:list=None,
                 frame_aware: bool = False,
                 dropout:float=0.1, l2_reg:float=0, embedding_l2_reg:float=1e-5, relative_temporal_embedding:bool=True,  multiplicative_embedding:bool=False, layer_normalization:bool = False,
                 memory_mode:bool=True,
                 skip_connection:bool = True,  name="TemporalAttention"):
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
        self.multiplicative_embedding=multiplicative_embedding
        self.layer_normalization=layer_normalization
        self.temporal_dim=None
        self.training_query_idx = training_query_idx
        self.inference_query_idx = inference_query_idx
        self.frame_aware=frame_aware
        self.relative_temporal_embedding=relative_temporal_embedding
        if self.frame_aware:
            assert relative_temporal_embedding
        self.memory_mode=memory_mode

    def get_config(self):
      config = super().get_config().copy()
      config.update({"num_heads": self.num_heads, "dropout":self.dropout, "filters":self.filters, "skip_connection":self.skip_connection, "layer_normalization":self.layer_normalization, "l2_reg":self.l2_reg, "embedding_l2_reg":self.embedding_l2_reg, "relative_temporal_embedding":self.relative_temporal_embedding, "multiplicative_embedding":self.multiplicative_embedding, "frame_aware":self.frame_aware, "training_query_idx":self.training_query_idx, "inference_query_idx":self.inference_query_idx, "memory_mode":self.memory_mode})
      return config

    def build(self, input_shape): # (T, B, Y, X, C) + (B, T) if frame_aware
        if self.frame_aware:
            input_shape, t_index_shape = input_shape
        try:
            input_shape = input_shape.as_list()
        except:
            pass


        self.spatial_dims=input_shape[2:-1]
        self.temporal_dim = input_shape[0]
        self.filters = input_shape[-1]

        if self.training_query_idx is None:
            self.training_query_idx = list(range(self.temporal_dim))
        elif not isinstance(self.training_query_idx, (list, tuple)):
            self.training_query_idx = [self.training_query_idx]
        if self.inference_query_idx is None:
            self.inference_query_idx = self.training_query_idx
        elif not isinstance(self.inference_query_idx, (list, tuple)):
            self.inference_query_idx = [self.inference_query_idx]

        if self.attention_filters is None or self.attention_filters<=0:
            self.attention_filters = int(ceil(self.filters / self.num_heads))

        self.attention_layer=tf.keras.layers.MultiHeadAttention(
            self.num_heads,
            key_dim=self.attention_filters,
            dropout=self.dropout,
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None,
            name="MultiHeadAttention")

        self.temp_embedding = RelativeTemporalEmbedding(
            self.filters,
            256,
            multiplicative=self.multiplicative_embedding,
            l2_reg=self.embedding_l2_reg,
            name="TempEnc"
        ) if self.relative_temporal_embedding else tf.keras.layers.Embedding(
            input_dim = self.temporal_dim, output_dim=self.filters,
            embeddings_regularizer=tf.keras.regularizers.l2(self.embedding_l2_reg) if self.embedding_l2_reg > 0 else None
        )
        if self.layer_normalization:
            self.ln_qk = tf.keras.layers.LayerNormalization() # qk are input + emb
            self.ln_v = tf.keras.layers.LayerNormalization() # v is only input
        super().build(input_shape)

    def call(self, x, training:bool=None):
        '''
            x : [frames, t_index] if frame_aware or frames each tensor with frames with shape (T, B, Y, X, C)
        '''
        if self.frame_aware:
            input_frames, t_index = x
        else:
            input_frames = x
            if self.relative_temporal_embedding:
                t_index = tf.range(self.temporal_dim, dtype=tf.int32) - tf.cast((self.temporal_dim - 1) // 2, tf.int32) # index is relative to central feature
            else:
                t_index = tf.range(self.temporal_dim, dtype=tf.int32)

        frame_shape = tf.shape(input_frames)[1:] # B, Y, X, C
        frames = tf.transpose(input_frames, [1, 2, 3, 0, 4]) # B, Y, X, T, C
        B, Y, X, _ = tf.unstack(frame_shape)
        T = self.temporal_dim
        C = self.filters

        t_emb = self.temp_embedding(t_index)  # (1, T, C) or (B, T, C) if frame_aware
        if self.multiplicative_embedding:
            t_emb_mul, t_emb = t_emb
            t_emb_mul = tf.reshape(t_emb_mul, (1, 1, 1, T, C)) if not self.frame_aware else tf.reshape(t_emb_mul, (B, 1, 1, T, C))
        t_emb = tf.reshape(t_emb, (1, 1, 1, T, C)) if not self.frame_aware else tf.reshape(t_emb, (B, 1, 1, T, C))

        if self.multiplicative_embedding:
            frame_emb = frames * t_emb_mul + t_emb
        else:
            frame_emb = frames + t_emb
        if self.layer_normalization:
            frame_emb = self.ln_qk(frame_emb)
            frames = self.ln_v(frames)
        key = frame_emb
        value = frames
        idx_list = tf.constant(self.training_query_idx if not self.inference_mode else self.inference_query_idx, dtype = tf.int32)
        query = tf.gather(frame_emb,  idx_list , axis=3)

        # Flatten spatial dimensions for efficiency: (B*Y*X, T, C)
        key_flat = tf.reshape(key, [-1, T, C])
        value_flat = tf.reshape(value, [-1, T, C])
        query_flat = tf.reshape(query, [-1, tf.shape(idx_list)[0], C])
        if not self.memory_mode: # parallel (higher mem footprint but fater)
            out = self.attention_layer(
                query=query_flat,
                value=value_flat,
                key=key_flat,
                training=training
            ) # BxYxX, Q, C
            out = tf.transpose(out, [1, 0, 2]) # Q, BxYxX, C
        else: # sequential (lower memory footprint but slower)
            out = tf.stack([self.attention_layer(
                query=query_flat[:, t:t+1],
                value=value_flat,
                key=key_flat,
                training=training
            ) for t in range(len(idx_list))], 0)
        out = tf.reshape(out, [tf.shape(idx_list)[0], B, Y, X, C])
        if self.skip_connection:
            out = out + tf.gather(input_frames, idx_list, axis=0)
        return out
