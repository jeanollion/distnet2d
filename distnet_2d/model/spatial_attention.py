import tensorflow as tf
import numpy as np

class SpatialAttention2D(tf.keras.layers.Layer):
    def __init__(self, num_heads:int=1, positional_encoding:str= "2d", filters:int=0, return_attention:bool=False, dropout:float=0.1, l2_reg:float=0., name="Attention"):
        '''
            filters : number of output channels
            if positional_encoding: filters must correspond to input channel number
            adapted from: https://www.tensorflow.org/tutorials/text/transformer
        '''
        super().__init__(name=name)
        self.attention_layer = None
        self.num_heads=num_heads
        self.positional_encoding=positional_encoding.lower()
        self.filters = filters
        self.return_attention=return_attention
        self.dropout=dropout
        self.l2_reg=l2_reg

    def get_config(self):
      config = super().get_config().copy()
      config.update({"num_heads": self.num_heads, "positional_encoding": self.positional_encoding, "dropout":self.dropout, "filters":self.filters, "return_attention":self.return_attention, "l2_reg":self.l2_reg})
      return config

    def build(self, input_shape):
        input_shape_, input_shape = input_shape
        try:
            input_shape = input_shape.as_list()
        except:
            pass
        try:
            input_shape_ = input_shape_.as_list()
        except:
            pass
        if isinstance(input_shape, tuple):
            input_shape = list(input_shape)

        assert len(input_shape_)==len(input_shape) and all(i==j for i,j in zip(input_shape_, input_shape)), f"both tensors must have same input shape: {input_shape_} != {input_shape}"
        self.spatial_dims=input_shape[1:-1]
        self.spatial_dim = np.prod(self.spatial_dims)
        print(f"attention spatial dims: {self.spatial_dims}")
        if self.filters is None or self.filters<=0:
            self.filters = input_shape[-1]
        self.attention_layer=tf.keras.layers.MultiHeadAttention(self.num_heads, key_dim=self.filters, value_dim=self.filters, attention_axes=[1, 2], dropout=self.dropout, name="MultiHeadAttention")
        tensor_shape = input_shape[:-1] + [self.filters]
        self.attention_layer._build_from_signature(query=tensor_shape, value=tensor_shape, key=tensor_shape)

        # positional encoding
        if "sine" in self.positional_encoding:
            if "2d" in self.positional_encoding:
                y_index = tf.range(self.spatial_dims[0], dtype=tf.float32)
                x_index = tf.range(self.spatial_dims[1], dtype=tf.float32)
                filter_index = tf.range(0, input_shape[-1], 4, dtype=tf.float32)
                y_grid, x_grid = tf.meshgrid(y_index, x_index, indexing='ij')

                div_term = tf.exp(filter_index * -(tf.math.log(10000.0) / input_shape[-1]))

                # Compute sine and cosine for y and x components
                sin_y = tf.sin(y_grid[..., tf.newaxis] * div_term)  # Shape: (y, x, d_model//4)
                cos_y = tf.cos(y_grid[..., tf.newaxis] * div_term)  # Shape: (y, x, d_model//4)
                sin_x = tf.sin(x_grid[..., tf.newaxis] * div_term)  # Shape: (y, x, d_model//4)
                cos_x = tf.cos(x_grid[..., tf.newaxis] * div_term)  # Shape: (y, x, d_model//4)

                # Stack and reshape to interleave y and x components
                pe = tf.stack([sin_y, cos_y, sin_x, cos_x], axis=-1)  # Shape: (y, x, d_model//4, 4)
                self.pos_enc = tf.reshape(pe, (self.spatial_dims[0], self.spatial_dims[1], -1))[..., :input_shape[-1]]

            elif False and "2d" in self.positional_encoding: # alternative version with sum of x and y components, // TODO : compare to 2D
                y_index = tf.range(self.spatial_dims[0], dtype=tf.float32)
                x_index = tf.range(self.spatial_dims[1], dtype=tf.float32)
                filter_index = tf.range(0, input_shape[-1], 2, dtype=tf.float32)

                div_term = tf.exp( filter_index  * -(tf.math.log(10000.0) / input_shape[-1]))

                # Compute positional encoding for y dimension
                sin_y = tf.sin(y_index[:, tf.newaxis] * div_term)
                cos_y = tf.cos(y_index[:, tf.newaxis] * div_term)
                pe_y = tf.stack([sin_y, cos_y], axis=-1)
                pe_y = tf.reshape(pe_y, (self.spatial_dims[0], -1))[..., :input_shape[-1]]
                pe_y = tf.reshape(pe_y, (self.spatial_dims[0], 1, input_shape[-1]))  # (y, 1, filters)

                # Compute positional encoding for x dimension
                sin_x = tf.sin(x_index[:, tf.newaxis] * div_term)
                cos_x = tf.cos(x_index[:, tf.newaxis] * div_term)
                pe_x = tf.stack([sin_x, cos_x], axis=-1)
                pe_x = tf.reshape(pe_x, (self.spatial_dims[1], -1))[..., :input_shape[-1]]
                pe_x = tf.reshape(pe_x, (1, self.spatial_dims[1], input_shape[-1]))  # (1, x, filters)

                self.pos_enc = pe_y + pe_x  # broadcast to (y, x, filters)
            else: # 1d
                spa_index = tf.range(self.spatial_dim, dtype=tf.float32)
                filter_index = tf.range(0, input_shape[-1], 2, dtype=tf.float32)

                div_term = tf.exp( filter_index * -(tf.math.log(10000.0) / input_shape[-1]))

                # Compute positional encoding
                sin_x = tf.sin(spa_index[:, tf.newaxis] * div_term)
                cos_x = tf.cos(spa_index[:, tf.newaxis] * div_term)
                pe = tf.stack([sin_x, cos_x], axis=-1)
                pe = tf.reshape(pe, (self.spatial_dim, -1))[..., :input_shape[-1]]
                self.pos_enc = tf.reshape(pe, (self.spatial_dims[0], self.spatial_dims[1],  input_shape[-1]))  # for broadcasting purpose

        elif "rotary" in self.positional_encoding or "rope" in self.positional_encoding:
            assert input_shape[-1] % 2 == 0, "Attention filters must be divisible by two for RoPE mode"
            if "2d" in self.positional_encoding:
                y_index = tf.range(self.spatial_dims[0], dtype=tf.float32)
                x_index = tf.range(self.spatial_dims[1], dtype=tf.float32)
                freq_index = tf.range(0, input_shape[-1], 4, dtype=tf.float32)

                div_term = tf.exp(freq_index * -(tf.math.log(10000.0) / input_shape[-1]))

                # Compute rotation angles for y and x positions
                y_angles = tf.einsum('i,j->ij', y_index, div_term)  # Shape: (y, d_model//4)
                y_angles = tf.expand_dims(y_angles, axis=1)  # Shape: (y, 1, d_model//4)

                x_angles = tf.einsum('i,j->ij', x_index, div_term)  # Shape: (x, d_model//4)
                x_angles = tf.expand_dims(x_angles, axis=0)  # Shape: (1, x, d_model//4)

                # Broadcast y_angles and x_angles to (y, x, d_model//2)
                y_angles = tf.tile(y_angles, [1, self.spatial_dims[1], 1])  # Shape: (y, x, d_model//4)
                x_angles = tf.tile(x_angles, [self.spatial_dims[0], 1, 1])  # Shape: (y, x, d_model//4)

                # Combine y and x angles
                angles = tf.stack([y_angles, x_angles], axis=-1)  # Shape: (y, x, d_model//4, 2)
                angles = tf.reshape(angles, (self.spatial_dims[0], self.spatial_dims[1], -1))[..., :input_shape[-1] // 2]  # Shape: (y, x, d_model//2)
                # Create cosine and sine tensors
                self.pos_enc_cos = tf.cos(angles)  # Shape: (y, x, d_model//2)
                self.pos_enc_sin = tf.sin(angles)  # Shape: (y, x, d_model//2)
            else:
                spa_index = tf.range(self.spatial_dim, dtype=tf.float32)
                freq_index = tf.range(0, input_shape[-1], 2, dtype=tf.float32)

                # Compute inverse frequencies
                div_term = tf.exp(freq_index * -(tf.math.log(10000.0) / input_shape[-1]))

                # Compute rotation angles for y and x positions
                angles = tf.einsum('i,j->ij', spa_index, div_term)  # Shape: (yx, d_model//2)
                angles = tf.reshape(angles, (self.spatial_dims[0], self.spatial_dims[1], -1))

                self.pos_enc_cos = tf.cos(angles)  # Shape: (y, x, d_model//2)
                self.pos_enc_sin = tf.sin(angles)  # Shape: y, x, d_model//2)

        elif self.positional_encoding is not None: # embedding
            if "2d" in self.positional_encoding:
                self.pos_embedding_y = tf.keras.layers.Embedding(self.spatial_dims[0], input_shape[-1], embeddings_regularizer=tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg>0 else None, name="PosEncY")
                self.pos_embedding_x = tf.keras.layers.Embedding(self.spatial_dims[1], input_shape[-1], embeddings_regularizer=tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg>0 else None, name="PosEncX")
            else:
                self.pos_embedding = tf.keras.layers.Embedding(self.spatial_dim, input_shape[-1], embeddings_regularizer=tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg>0 else None, name="PosEnc")
        super().build(input_shape)

    def call(self, x, training:bool=None):
        '''
            x : tensor with shape (batch_size, y, x, channels)
        '''
        [input, output] = x
        shape = tf.shape(output)
        batch_size = shape[0]
        #spatial_dims = shape[1:-1]
        #spatial_dim = tf.reduce_prod(spatial_dims)
        depth_dim = shape[3]
        if "sine" in self.positional_encoding:
            key = input + self.pos_enc  # broadcast
            query = output + self.pos_enc
        elif "rotary" in self.positional_encoding or "rope" in self.positional_encoding:
            # Split query and key into two parts
            q1, q2 = tf.split(output, 2, axis=-1)  # Each shape: (batch_size, num_heads, y, x, d_model//2)
            k1, k2 = tf.split(input, 2, axis=-1)  # Each shape: (batch_size, num_heads, y, x, d_model//2)
            #print(f"q1: {q1.shape} q2 {q2.shape}, sin: {self.pos_enc_sin.shape} cos: {self.pos_enc_cos.shape}")
            # Apply rotation to query and key
            query = tf.concat([ q1 * self.pos_enc_cos - q2 * self.pos_enc_sin,  q1 * self.pos_enc_sin + q2 * self.pos_enc_cos ], axis=-1)
            key = tf.concat([  k1 * self.pos_enc_cos - k2 * self.pos_enc_sin,  k1 * self.pos_enc_sin + k2 * self.pos_enc_cos ], axis=-1)

        elif self.positional_encoding is not None:
            if "2d" in self.positional_encoding:
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
                query = output + pos_emb # broadcast
                key = input + pos_emb # broadcast
            else:
                x_index = tf.range(self.spatial_dim, dtype=tf.int32)
                pos_emb = self.pos_embedding(x_index) # (spa_dim, self.filters)
                pos_emb = tf.reshape(pos_emb, (self.spatial_dims[0], self.spatial_dims[1], self.filters)) #for broadcasting purpose
                query = output + pos_emb # broadcast
                key = input + pos_emb # broadcast
        # TODO Legacy mode : add pos_emb to values
        attention_output = self.attention_layer(query=query, value=input, key=key, training=training, return_attention_scores=self.return_attention)
        return attention_output

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return input_shape[:-1]+(self.filters,), (input_shape[0], self.spatial_dim, self.spatial_dim)
        else:
            return input_shape[:-1]+(self.filters,)