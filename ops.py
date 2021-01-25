import tensorflow as tf
from normalization import InstanceNormalization
import math


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


