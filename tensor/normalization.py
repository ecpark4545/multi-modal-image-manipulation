import tensorflow as tf


class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon
    
    def call(self, x):
        x_mean, x_var = tf.nn.moments(x, axes=[1,2], keepdims=True)
        x_std = tf.sqrt(x_var + self.epsilon)
        x_norm = (x - x_mean) / x_std
        return x_norm


class ACAN(tf.keras.layers.Layer):
    def __init__(self, c_in, d_k):
        super(ACAN, self).__init__()
        self.c_in = c_in
        self.d_k = d_k
        self.conv_q = tf.keras.layers.Conv2D(filters=self.d_k, kernel_size=1, padding='valid')
        self.conv_k = tf.keras.layers.Conv2D(filters=self.d_k, kernel_size=1, padding='valid')
        self.W_k = tf.keras.layers.Dense(units=self.d_k)
        self.W_mu = tf.keras.layers.Dense(units=self.c_in)
        self.W_simga = tf.keras.layers.Dense(units=self.c_in)
        self.i_norm = InstanceNormalization()
        self.softmax = tf.keras.layers.Softmax()

    def call(self, inputs):
        x, t = inputs
        n, h, w, c = tf.shape(x)
        len_k = tf.shape(t)[1]

        x_norm = self.i_norm(x)

        x_q = self.conv_q(x)
        x_k = self.conv_k(x)

        x_q = tf.reshape(x_q, [n, h*w, self.d_k])
        x_k = tf.reshape(x_k, [n, h*w, self.d_k])

        scores_x = x_q * x_k
        scores_x = tf.math.reduce_sum(scores_x, axis=-1, keepdims=True)

        t_k = self.W_k(t)
        mu_t = self.W_mu(t)
        sigma_t = self.W_simga(t)

        scores_t = tf.matmul(x_q, tf.transpose(t_k, perm=[0,2,1]))
        scores = tf.concat([scores_t, scores_x], axis=-1)
        scores = self.softmax(scores)

        scores_t, scores_x = tf.split(scores, [-1, 1], -1)

        scores_x = tf.reshape(scores_x, [n, h, w, 1])

        weighted_mu_t = tf.matmul(scores_t, mu_t)
        weighted_sigma_t = tf.matmul(scores_t, sigma_t)

        weighted_mu_t = tf.reshape(weighted_mu_t, [n, h, w, c])
        weighted_sigma_t = tf.reshape(weighted_sigma_t, [n, h, w, c])
        
        x = (1 + weighted_sigma_t) * x_norm + weighted_mu_t + scores_x * x
        return x