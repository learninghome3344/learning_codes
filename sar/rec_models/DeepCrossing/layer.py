import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers, regularizers

class FFMLayer(layers.Layer):
    def __init__(self, feature_columns, k, w_reg, v_reg):
        super(FFMLayer, self).__init__()
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.feature_num = sum([feat['feat_onehot_dim'] for feat in self.sparse_feature_columns]) \
                            + len(self.dense_feature_columns)
        self.field_num = len(self.dense_feature_columns) + len(self.sparse_feature_columns)
    
    def build(self, input_shape):
        self.w0 = self.add_weight("w0", shape=(1,),
                                initializer=tf.zeros_initializer(),
                                trainable=True)
        self.w = self.add_weight("w", shape=(self.feature_num, 1),
                                initializer=tf.random_normal_initializer(),
                                trainable=True,
                                regularizer=regularizers.L2(self.w_reg))
        # for FM, every feature has only one hidden vector
        # but for FFM, every feature has `field_num` hidden vectors
        self.v = self.add_weight("v", shape=(self.feature_num, self.field_num, self.k),
                                initializer=tf.random_normal_initializer(),
                                trainable=True,
                                regularizer=regularizers.L2(self.v_reg))
    
    def call(self, inputs):
        dense_inputs = inputs[:, :13]
        sparse_inputs = inputs[:, 13:]

        # onehot-encoding
        x = tf.cast(dense_inputs, dtype=tf.float32)
        for i in range(sparse_inputs.shape[1]):
            x = tf.concat([x, tf.one_hot(tf.cast(sparse_inputs[:, i], dtype=tf.int32),
                                      depth=self.sparse_feature_columns[i]['feat_onehot_dim'])], axis=1)
        # for i in range(sparse_inputs.shape[1]):
        #     x = tf.concat(
        #         [x, tf.one_hot(tf.cast(sparse_inputs[:, i], dtype=tf.int32),
        #                            depth=self.sparse_feature_columns[i]['feat_onehot_dim'])], axis=1)
        
        linear_part = tf.matmul(x, self.w) + self.w0
        inter_part = 0
        # 1. calculate VX
            # process of tensordot: https://blog.csdn.net/tjh1998/article/details/123563159
            # for b in batch_size:
            #     tmp = x[b, :] matmul v = [1, 2291] matmul [2291, 39, 8] = [1, 39, 8]
            # res = concat all tmp
        field_f = tf.tensordot(x, self.v, axes=1) # [batch_size, 2291] x [2291, 39, 8] = [batch_size, 39, 8]
        # 2. calculate <V_i_fj, V_j_fi> * x_i * x_j for every (i, j)
        for i in range(self.field_num):
            for j in range(i + 1, self.field_num):
                inter_part += tf.reduce_sum(
                    tf.multiply(field_f[:, i], field_f[:, j]), # [None, 8]
                    axis=1, keepdims=True
                )
        
        output = linear_part + inter_part
        return output
