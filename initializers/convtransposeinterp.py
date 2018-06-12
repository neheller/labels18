import numpy as np
import keras.backend as K
from keras.initializers import Initializer

class ConvTransposeInterp(Initializer):
    """Initializer that generates the identity matrix.
    Only use for square 2D matrices.
    # Arguments
        gain: Multiplicative factor to apply to the identity matrix.
    """

    def __init__(self):
        pass

    def __call__(self, shape, dtype=None):
        return K.ones(shape, dtype=dtype)

    def get_config(self):
        return {}
