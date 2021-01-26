import tensorflow as tf
import numpy as np
from ops import *
from attention import MultiHeadAttention


class ImageEncoder(tf.keras.layers.Layer):
    def __init__(self, img_size=256, img_ch=3, max_conv_dim=512):
        super(ImageEncoder, self).__init__()
        self.img_size = img_size
        self.img_ch = img_ch
        self.channels = 2 ** 14 // self.img_size
        self.max_conv_dim = max_conv_dim
        self.repeat_num = ing(np.log2(self.img_size)) - 4
        self.from_rgb = tf.keras.layers.Conv2D(filters=self.channels, kernel_size=1, strides=1, padding='valid')
        self.blocks = build_blocks()
    
    def build_blocks(self):
        blocks = []

        ch_in = self.channels
        ch_out = self.channels

        for i in range(self.repeat_num):
            ch_out = min(ch_in * 2, self.max_conv_dim)
            blocks.append(ResBLK(ch_in, ch_out, normalize=True, down_sample=True)) 
            ch_in = ch_out
        for i in range(2):
            blocks.append(ResBLK(ch_out, ch_out, normalize=True))
        
        return blocks

    def call(self, x):
        x = self.from_rgb(x)
        for block in self.blocks:
            x = block(x)
        return x


class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, n_head, dff, d_k, d_v):
        self.self_attention = MultiHeadAttention(d_model, n_head, d_k, d_v)
        self.feed_forward = PointWiseFeedForward(d_model, dff)
        self.dropout1 = tf.keras.layers.Dropout(rate=dropout)
        self.dropout2 = tf.keras.layers.Dropout(rate=dropout)
        self.layer_norm1 = tf.keras.layers.LayerNormalization()
        self.layer_norm2 = tf.keras.layers.LayerNormalization()
        
    def call(self, x):
        attn_output = self.self_attention(x, x, x)
        attn_output = self.dropout1(attn_output)

        out1 = self.layer_norm1(x + attn_output)

        ffn_output = self.feed_forward(out1)
        ffn_output = self.dropout2(ffn_output)

        output = self.layer_norm2(out1 + ffn_output)
        return output


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, n_head, dff, d_k, d_v, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.n_head = n_head
        self.dff = dff
        self.input_vocab_size = input_vocab_size
        self.maximum_position_encoding = maximum_position_encoding

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        self.layers = [TransformerEncoderLayer(self.d_model, self.n_head, dff, d_k, d_v) for _ in range(self.num_layers)]
        self.drop_out = tf.keras.layers.Dropout(rate=rate)

    def call(self, x):
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)
        
        return x


class StarganGenerator(tf.keras.Model):
    def __init__(self, d_k, img_size=256, img_ch=3, max_conv_dim=512):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.img_ch = img_ch
        self.channels = 2**14 // self.img_size
        self.max_conv_dim = max_conv_dim
        self.d_k = d_k
        self.repeat_num = int(np.log2(self.img_size)) - 4
        self.from_rgb = tf.keras.layers.Conv2D(filters=self.channels, kernel_size=1, strides=1, padding='valid')
        self.encoder ,self.decoder = self.build_layers()

    def build_layers(self):
        encoder = []
        decoder = []

        ch_in = self.channels
        ch_in = self.channels

        for i in range(self.repeat_num):
            ch_out = min(ch_in * 2, self.max_conv_dim)
            encoder.append(ResBLK(ch_in, ch_out, normalize=True, down_sample=True))
            decoder.insert(0, ACANResBlock(ch_in, ch_out, self.d_k, up_sample=True)
            ch_in = ch_out
        
        for i in range(2):
            encoder.append(ResBLK(ch_out, ch_out, normalize=True))
            decoder.insert(0, ACANResBlock(ch_out, ch_out, self.d_k))
        return encoder, decoder

    def call(self, x):
        
        