import tensorflow as tf
from normalization import InstanceNormalization, ACAN
import math
import numpy as np


class ResBLK(tf.keras.layers.Layer):
    def __init__(self, channels_in, channels_out, normalize=False, down_sample=False):
        super(ResBLK, self).__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.normalize = normalize
        self.down_sample = down_sample

        self.l_relu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.conv_0 = tf.keras.layers.Conv2D(filters=self.channels_out, kernel_size=3, strides=1, padding='same', use_bias=self.use_bias)
        self.conv_1 = tf.keras.layers.Conv2D(filters=self.channels_out, kernel_size=3, strides=1, padding='same', use_bias=self.use_bias)
    
        self.avg_pool = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, padding='valid')
        self.i_norm = InstanceNormalization()

    def residual(self, x):
        if self.normalize:
            x = self.i_norm(x)
        x = self.l_relu(x)
        x = self.conv_0(x)

        if self.down_sample:
            x = self.avg_pool(x)
        if self.normalize:
            x = self.i_norm(x)
        
        x = self.l_relu(x)
        x = self.conv_1(x)
        return x
    
    def shortcut(self, x):
        if self.down_sample:
            x = self.avg_pool(x)
        return x
    
    def call(self, x):
        x = self.residual(x) + self.shortcut(x)
        return x / math.sqrt(2)


class ACANResBlock(tf.keras.layers.Layer):
    def __init__(self, channels_in, channels_out, d_k, up_sample=False):
        super(ACANResBlock, self).__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.d_k = d_k
        self.up_sample = up_sample

        self.conv_0 = tf.keras.layers.Conv2D(filters=self.channels_out, kernel_size=3, strides=1, padding='same')
        self.acan0 = ACAN(self.channels_out, self.d_k)
        self.conv_1 = tf.keras.layers.Conv2D(filters=self.channels_out, kernel_size=3, strides=1, padding='same')
        self.acan1 = ACAN(self.channels_out, self.d_k)
        self.l_relu = tf.keras.layers.LeakyReLU(alpha=0.2)

    def residual(self, x, t):
        x = self.acan0([x,t])
        x = self.l_relu(x)
        if self.up_sample:
            x = nearest_up_sample(x, scale_factor=2)
        x = self.conv_0(x)
        x = self.acan1([x,t])
        x = self.l_relu(x)
        x = self.conv_1(x)
        return x

    def shortcut(self, x):
        if self.up_sample:
            x = nearest_up_sample(x, scale_factor=2)
        return x

    def call(self, x):
        x, t = x
        x = self.residual(x) + self.shortcut(x)
        return x / math.sqrt(2)




class PointWiseFeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super(PointWiseFeedForward, self).__init__()
        self.fc1 = tf.keras.layers.Dense(units=diff, activation='relu')
        self.fc2 = tf.keras.layers.Dense(units=d_model)

    def call(self, x):
        return self.fc2(self.fc1(x))


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

def nearest_up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize(x, size=new_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


