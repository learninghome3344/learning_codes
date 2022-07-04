import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from layer import FFMLayer

# https://zhuanlan.zhihu.com/p/342803984
class FFM(Model):
    def __init__(self, feature_columns, k, w_reg=1e-4, v_reg=1e-4) -> None:
        super().__init__()
        self.ffm = FFMLayer(feature_columns, k, w_reg, v_reg)
    
    def call(self, inputs):
        output = self.ffm(inputs)
        output = tf.nn.sigmoid(output)
        return output
