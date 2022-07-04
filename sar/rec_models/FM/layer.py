import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers, regularizers

class FMLayer(layers.Layer):
    def __init__(self, k, w_reg, v_reg):
        super(FMLayer, self).__init__()
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg
    
    def build(self, input_shape):
        self.w0 = self.add_weight("w0", shape=(1,),
                                initializer=tf.zeros_initializer(),
                                trainable=True)
        self.w = self.add_weight("w", shape=(input_shape[-1], 1),
                                initializer=tf.random_normal_initializer(),
                                trainable=True,
                                regularizer=regularizers.L2(self.w_reg))
        self.v = self.add_weight("v", shape=(input_shape[-1], self.k),
                                initializer=tf.random_normal_initializer(),
                                trainable=True,
                                regularizer=regularizers.L2(self.v_reg))
    
    def call(self, inputs):
        if K.ndim(inputs) != 2:
            raise ValueError("Unexpected inputs dimension %d, expect to be 2 dimensions" % (K.ndim(inputs)))
        
        linear_part = tf.matmul(inputs, self.w) + self.w0
        inter_part1 = tf.pow(tf.matmul(inputs, self.v), 2)
        inter_part2 = tf.matmul(tf.pow(inputs, 2), tf.pow(self.v, 2))
        inter_part = 0.5 * tf.reduce_sum(inter_part1 - inter_part2, axis=-1, keepdims=True)
        output = linear_part + inter_part
        return output
