import keras.backend as K
from keras import losses
from keras.layers import multiply
from keras.layers import Reshape
from keras.layers import Lambda
import numpy as np

# Sampled binary cross-entropy weighted by lis weights (tuned)
def sampled_bce(y_true, y_pred, neg_thresh):
    neg = Lambda(lambda x: x[:,:,1])(y_true)
    thresh = K.cast(neg_thresh*neg, "float32")
    rnd = np.random.random((20, 512*512)).astype(np.float32)
    weights = K.cast(K.greater(rnd, thresh), "float32")
    return -1*K.sum(
        weights*K.sum(y_true*K.log(y_pred+1e-10), axis=2)
    )/(K.sum(weights)+1e-2)
