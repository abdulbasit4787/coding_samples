import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
from keras import backend as K


def dice_loss_with_CE(cls_weights, beta=1, smooth = 1e-5):
    cls_weights = np.reshape(cls_weights, [1, 1, 1, -1])
    def _dice_loss_with_CE(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        CE_loss = - y_true[...,:-1] * K.log(y_pred) * cls_weights
        CE_loss = K.mean(K.sum(CE_loss, axis = -1))

        tp = K.sum(y_true[...,:-1] * y_pred, axis=[0,1,2])
        fp = K.sum(y_pred         , axis=[0,1,2]) - tp
        fn = K.sum(y_true[...,:-1], axis=[0,1,2]) - tp

        score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
        score = tf.reduce_mean(score)
        dice_loss = 1 - score
        # dice_loss = tf.Print(dice_loss, [dice_loss, CE_loss])
        return CE_loss + dice_loss
    return _dice_loss_with_CE

def CE(cls_weights):
    cls_weights = np.reshape(cls_weights, [1, 1, 1, -1])
    def _CE(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        CE_loss = - y_true[...,:-1] * K.log(y_pred) * cls_weights
        CE_loss = K.mean(K.sum(CE_loss, axis = -1))
        # dice_loss = tf.Print(CE_loss, [CE_loss])
        return CE_loss
    return _CE

def dice_loss_with_Focal_Loss(cls_weights, beta=1, smooth = 1e-5, alpha=0.5, gamma=2):
    cls_weights = np.reshape(cls_weights, [1, 1, 1, -1])
    def _dice_loss_with_Focal_Loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        logpt = - y_true[...,:-1] * K.log(y_pred) * cls_weights
        logpt = - K.sum(logpt, axis = -1)

        pt = tf.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        CE_loss = -((1 - pt) ** gamma) * logpt
        CE_loss = K.mean(CE_loss)

        tp = K.sum(y_true[...,:-1] * y_pred, axis=[0,1,2])
        fp = K.sum(y_pred         , axis=[0,1,2]) - tp
        fn = K.sum(y_true[...,:-1], axis=[0,1,2]) - tp

        score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
        score = tf.reduce_mean(score)
        dice_loss = 1 - score
        # dice_loss = tf.Print(dice_loss, [dice_loss, CE_loss])
        return CE_loss + dice_loss
    return _dice_loss_with_Focal_Loss

def focal_loss(classes_num, gamma=2., alpha=.25, e=0.1):
     # classes_num contains sample number of each classes
     def focal_loss_fixed(target_tensor, prediction_tensor):
         '''
         prediction_tensor is the output tensor with shape [None, 100], where 100 is the number of classes
         target_tensor is the label tensor, same shape as predcition_tensor
         '''
         # import tensorflow as tf
         # from tensorflow.python.ops import array_ops
         # from keras import backend as K

         #1# get focal loss with no balanced weight which presented in paper function (4)
         zeros = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)
         one_minus_p = array_ops.where(tf.greater(target_tensor,zeros), target_tensor - prediction_tensor, zeros)
         FT = -1 * (one_minus_p ** gamma) * tf.log(tf.clip_by_value(prediction_tensor, 1e-8, 1.0))

         #2# get balanced weight alpha
         classes_weight = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)

         total_num = float(sum(classes_num))
         classes_w_t1 = [ total_num / ff for ff in classes_num ]
         sum_ = sum(classes_w_t1)
         classes_w_t2 = [ ff/sum_ for ff in classes_w_t1 ]   #scale
         classes_w_tensor = tf.convert_to_tensor(classes_w_t2, dtype=prediction_tensor.dtype)
         classes_weight += classes_w_tensor

         alpha = array_ops.where(tf.greater(target_tensor, zeros), classes_weight, zeros)

         #3# get balanced focal loss
         balanced_fl = alpha * FT
         balanced_fl = tf.reduce_mean(balanced_fl)

         #4# add other op to prevent overfit
         # reference : https://spaces.ac.cn/archives/4493
         nb_classes = len(classes_num)
         fianal_loss = (1-e) * balanced_fl + e * K.categorical_crossentropy(K.ones_like(prediction_tensor)/nb_classes, prediction_tensor)

         return fianal_loss
     return focal_loss_fixed

def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    smooth=0.5
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)
def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)
def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.7
    return K.pow((1-pt_1), gamma)


# def focal_loss(gamma=2., alpha=.25):
# 	def focal_loss_fixed(y_true, y_pred):
# 	    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
#         pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
#         return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
# 	return focal_loss_fixed

def Focal_Loss(cls_weights, alpha=0.8, gamma=3):
    cls_weights = np.reshape(cls_weights, [1, 1, 1, -1])
    print(cls_weights)
    # def _Focal_Loss(y_true, y_pred):
    def _Focal_Loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        logpt = - y_true[...,:-1] * K.log(y_pred) * cls_weights
        # logpt = - y_true * K.log(y_pred) * cls_weights
        logpt = - K.sum(logpt, axis = -1)

        pt = tf.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        CE_loss = -((1 - pt) ** gamma) * logpt
        CE_loss = K.mean(CE_loss)
        return CE_loss
    return _Focal_Loss
