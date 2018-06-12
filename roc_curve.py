import os
from scipy import interp
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import socket

from keras.optimizers import Adam
from keras.models import load_model
from sklearn.metrics import roc_curve, auc

from Models.UNet import create_unet
from Models.SegNet import create_segnet
from Models.FCN32 import create_fcn32
from Models.metrics import *
from DataGenerator import DataGenerator
from callbacks.vizpreds import VizPreds
from callbacks.topnsaver import TopNSaver
from losses.sampledbce import sampled_bce
from initializers.convtransposeinterp import ConvTransposeInterp

def get_performance(y_preds, y_true):
    y_preds = np.greater(y_preds, 0.5)
    y_true = np.greater(y_true, 0.5)
    num_tp = np.sum(np.logical_and(y_preds, y_true))
    num_fp = np.sum(np.logical_and(y_preds, np.logical_not(y_true)))
    num_fn = np.sum(np.logical_and(np.logical_not(y_preds), y_true))
    return num_tp, num_fp, num_fn

from keras.backend import manual_variable_initialization


# probability of OMITTING a pixel
LIS_TVL_SAMPLING_PROB = 1.0 - 0.0716433
PSD_TVL_SAMPLING_PROB = 1.0 - 0.0044771

if __name__ == '__main__':
    # Limit the program to only one GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    filename = '/media/storage/labels18_models/encore0_unet_20_lis_control_2018-05-30_00-39-28/weights__epoch-050_val_dice_sorensen-0.921_val_loss-0.020.h5'
    model = load_model(
        filename, custom_objects={
            'precision': precision,
            'recall': recall,
            'dice_sorensen': dice_sorensen,
            '<lambda>': lambda ytrue, ypreds: sampled_bce(
                ytrue, ypreds, LIS_TVL_SAMPLING_PROB
            ),
            'ConvTransposeInterp': ConvTransposeInterp
        }
    )

    x_begin = "/home/helle246/data/LiS/validation/x-ntl-"
    y_begin = "/home/helle246/data/LiS/validation/y-ntl-lo-"
    end = ".npy"

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for i in range(2):
        x_str = x_begin + str(i) + end
        y_str = y_begin + str(i) + end
        x = np.load(x_str)
        y = np.load(y_str)
        y_pred = model.predict(x)
        np.save("x.npy", x)
        np.save("pred.npy", np.greater(np.reshape(y_pred[:,:,0],(20,512,512,1)), 0.5))
        np.save("true.npy", y[:,:,:,[0]])
        print(y_pred.shape)
        y = np.reshape(y[:,:,:,0], (20*512*512))
        y_pred = np.reshape(y_pred[:,:,0], (20*512*512))
        fpr, tpr, thresholds = roc_curve(y, y_pred, pos_label = 1)
        roc_auc = auc(fpr, tpr)

        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(roc_auc)
        # plt.plot(fpr, tpr, label='ROC fold %d (AUC = %0.4f)' %(i, roc_auc), alpha=0.4)
        tp, fp, fn = get_performance(y_pred, y)

        prec = tp/(tp + fp + 1e-5)
        rec = tp/(tp + fn + 1e-5)
        print(np.sum(y), prec, rec, 2*prec*rec/(prec + rec + 1e-5))

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='k', label='Mean ROC (AUC = %0.4f)' % (mean_auc), alpha=0.8)
    # plt.boxplot(aucs, vert=True, whis="range")
    # plt.title("Box and Whisker AUC of UNet")
    # plt.show()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("UNet ROC")
    plt.legend(loc='lower right')
    plt.show()
