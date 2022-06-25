import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from layer import FMLayer

class FM(Model):
    def __init__(self, k, w_reg=1e-4, v_reg=1e-4) -> None:
        super().__init__()
        self.fm = FMLayer(k, w_reg, v_reg)
    
    def call(self, inputs):
        output = self.fm(inputs)
        return output
