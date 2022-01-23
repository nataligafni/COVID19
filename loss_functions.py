from keras import backend as K
from tensorflow.keras.losses import MeanSquaredError as mse
import tensorflow as tf
from tensorflow import keras

import numpy as np

def dice_coef(y_true, y_pred, epsilon=1e-6):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    """

    def dice_func():
        intersection = K.sum(K.abs(y_true * y_pred))
        return (2 * intersection + epsilon) / (K.sum(K.abs(y_true) + K.abs(y_pred)) + epsilon)
    def no_seg():
        return tf.constant(1,dtype='float32')

    return tf.cond(tf.less(tf.reduce_sum(y_true),-1), no_seg,dice_func)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def mse_cutout(y_true, y_pred):
    y_pred = tf.where(y_true == 0, 0, y_pred)
    num_non_zero_pixels = tf.math.count_nonzero(y_true, dtype=tf.dtypes.float32) + 1e-8
    mse = tf.divide(tf.reduce_sum(tf.square(y_true-y_pred)),num_non_zero_pixels)
    return mse

    # mseObject = keras.losses.MeanSquaredError()
    # mseTensor = mseObject(y_true, y_pred)
    # return mseTensor

