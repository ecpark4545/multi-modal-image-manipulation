import tensorflow as tf


class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self):
        super(InstanceNormalization, self).__init__()
    
    def call(self, x):
        x_mean, x_var = tf.nn.moments(x, axes=[1,2], keepdims=True)
        x_std = tf.sqrt(x_var + self.epsilon)
        x_norm = (x - x_mean) / x_std
        return x_norm


class CrossAdaptiveNormalization(tf.keras.layers.Layer):
    def __init__(self, c_in, d_k):
        super(CrossAdaptiveNormalization, self).__init__()
        self.c_in = c_in
        self.d_k = d_k
        self.conv = tf.keras.layers.Conv2D(filters=self.d_k, kernel_size=3, padding='same')
        self.W_k = tf.keras.layers.Dense(units=self.d_k)
        self.W_mu = tf.keras.layers.Dense(units=self.c_in)
        self.W_simga = tf.keras.layers.Dense(units=self.c_in)
        self.i_norm = InstanceNormalization()
        self.softmax = tf.keras.layers.Softmax()

    def call(self, x):
        x, t = x
        n, h, w, c = tf.shape(x)
        len_k = tf.shape(t)[1]

        x_norm self.i_norm(x)

        x_q = self.conv(x)
        x_q = tf.reshape(x_q, [n, h*w, self.d_k])
        
        k = self.W_k(t)
        mu = self.W_mu(t)
        sigma = self.W_simga(t)

        scores = tf.matmul(x_q, tf.transpose(k, perm=[0,2,1]))
        scores = self.softmax(scores)

        weighted_mu = tf.matmul(scores, mu)
        weighted_sigma = tf.matmul(scores, sigma)

        weighted_mu = tf.reshape(weighted_mu, [n, h, w, c])
        weighted_sigma = tf.reshape(weighted_sigma, [n, h, w, c])
        
        x = (1 + weighted_sigma) * x_norm + weighted_mu
        return x


