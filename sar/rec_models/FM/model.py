import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from layer import FMLayer

# https://zhuanlan.zhihu.com/p/342803984
class FM(Model):
    def __init__(self, k, w_reg=1e-4, v_reg=1e-4) -> None:
        super().__init__()
        self.fm = FMLayer(k, w_reg, v_reg)
    
    def call(self, inputs):
        output = self.fm(inputs)
        output = tf.nn.sigmoid(output)
        return output
