import tensorflow as tf
import math


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, n_head, d_k, d_v):
        super(MultiHeadAttention, self).__init__()
        self.n_head
        self.d_k
        self.d_v
        self.W_q = tf.keras.layers.Dense(units=d_k*n_head)
        self.W_k = tf.keras.layers.Dense(units=d_k*n_head)
        self.W_v = tf.keras.layers.Dense(units=d_v*n_head)
        self.softmax = tf.keras.layers.Softmax()
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.fc = tf.keras.layers.Dense(d_model)

    def residual(self, x):
        return x
    
    def attention(self, q, k, v):
        q = self.W_q(q)
        k = self.W_k(k)
        v = self.W_v(v)
        
        len_q, len_k, len_v = tf.shape(q)[1], tf.shape(k)[1], tf.shape(v)[1]

        q = tf.reshape(q, [-1, len_q, self.n_head, self.d_k])
        k = tf.reshape(k, [-1, len_k, self.n_head, self.d_k])
        v = tf.reshape(v, [-1, len_v, self.n_head, self.d_v])

        q = tf.transpose(q, perm=[0, 2, 1, 3])
        k = tf.transpose(k, perm=[0, 2, 1, 3])
        v = tf.transpose(v, perm=[0, 2, 1, 3])

        scores = tf.matmul(q, tf.tranpose(k, perm=[0, 1, 3, 2]))
        scores = self.softmax(scores)

        attention = tf.matmul(scores, v)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        attention = tf.reshape(attention, [-1, len_q, self.d_v * self.n_head])
        output = self.fc(attention)
        return output

    def call(self, x):
        q, k, v = x
        x = self.residual(q) + self.attention(output)
        self.layer_norm(x)
        return x
