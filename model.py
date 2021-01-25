import tensorflow as tf
from ops import *


class ImageEncoder(tf.keras.layers.Layer):
    def __init__(self):
        super(ImageEncoder, self).__init__()
    
    def build_blocks(self):
        blocks = []
        for i in range(6):
            ResBLK()

    def call(self, x):
        
        