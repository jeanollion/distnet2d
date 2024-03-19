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
        self.positional_encoding=positional_encoding
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
        assert len(input_shape_)==len(input_shape) and all(i==j for i,j in zip(input_shape_, input_shape)), f"both tensors must have same input shape: {input_shape_} != {input_shape}"
        self.spatial_dims=input_shape[1:-1]
        self.spatial_dim = np.prod(self.spatial_dims)
        if self.filters is None or self.filters<=0:
            self.filters = input_shape[-1]
        self.attention_layer=tf.keras.layers.MultiHeadAttention(self.num_heads, key_dim=self.filters, value_dim=self.filters, attention_axes=[1, 2], dropout=self.dropout, name="MultiHeadAttention")
        tensor_shape = input_shape[:-1] + [self.filters]
        self.attention_layer._build_from_signature(query=tensor_shape, value=tensor_shape, key=tensor_shape)
        if self.positional_encoding.lower()=="2d":
            self.pos_embedding_y = tf.keras.layers.Embedding(self.spatial_dims[0], input_shape[-1], embeddings_regularizer=tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg>0 else None, name="PosEncY")
            self.pos_embedding_x = tf.keras.layers.Embedding(self.spatial_dims[1], input_shape[-1], embeddings_regularizer=tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg>0 else None, name="PosEncX")
        elif self.positional_encoding.lower()=="1d":
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
        if self.positional_encoding.lower()=="2d":
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
        elif self.positional_encoding.lower()=="1d":
            x_index = tf.range(self.spatial_dim, dtype=tf.int32)
            pos_emb = self.pos_embedding(x_index) # (spa_dim, self.filters)
            pos_emb = tf.reshape(pos_emb, (self.spatial_dims[0], self.spatial_dims[1], self.filters)) #for broadcasting purpose
            output = output + pos_emb # broadcast
            input = input + pos_emb # broadcast

        attention_output = self.attention_layer(query=output, value=input, key=input, training=training, return_attention_scores=self.return_attention)
        return attention_output

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return input_shape[:-1]+(self.filters,), (input_shape[0], self.spatial_dim, self.spatial_dim)
        else:
            return input_shape[:-1]+(self.filters,)