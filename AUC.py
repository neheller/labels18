from sklearn.metrics import roc_curve, auc
from scipy import interp
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import load_model
from Models.metrics import *
from losses.sampledbce import *
import os
import random
from Models.Mylayers import *
from DataGenerator import DataGenerator
from initializers.convtransposeinterp import ConvTransposeInterp
import sys
from pathlib import Path
from keras import backend as K

#base_dirs = ["/media/storage/labels18_models"]
base_dirs = [
    "/home/helle246/data/labels18_models/jinx",
    "/home/helle246/data/labels18_models/jupiter"
]

# probability of OMITTING a pixel
LIS_TVL_SAMPLING_PROB = 1.0 - 0.0716433
PSD_TVL_SAMPLING_PROB = 1.0 - 0.0044771

# x_begin = "/home/helle246/data/LiS/training/x-ntl-"
# y_begin = "/home/helle246/data/LiS/training/y-ntl-lo-"
x_begin = "/home/helle246/data/LiS/validation/x-ntl-"
y_begin = "/home/helle246/data/LiS/validation/y-ntl-lo-"
end = ".npy"
iterations = 10

#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
for base_dir in base_dirs:
    for model_dir in os.listdir(base_dir):

        dirpath = Path(model_dir)
        ds = dirpath.name.split('_')[3]
        #PSD broke, but all LIS done
        #if ds != "lis":
        #    continue


        max_dice = 0
        index = 0
        models = os.listdir(base_dir + '/' + model_dir)
        if "results.txt" in models:
            continue

        #if "tpr.npy" in models:
        #    continue
        if len(models) == 0:
            continue
        for i, model in enumerate(models):
            if model.startswith("weights"):
                dice_val = float(model[37:42])
                if dice_val > max_dice:
                    max_dice = dice_val
                    index = i

        if not models[index].startswith('weights'):
            continue
        print(models[index])



        model = load_model(base_dir + '/' + model_dir + '/' + models[index], custom_objects={'precision': precision, 'recall': recall, 'dice_sorensen': dice_sorensen,
                                                                                             '<lambda>': lambda y_true, y_pred: sampled_bce(y_true, y_pred, LIS_TVL_SAMPLING_PROB),
                                                                                             "MaxPoolingWithArgmax2D": MaxPoolingWithArgmax2D, "MaxUnpooling2D": MaxUnpooling2D,
                                                                                             "ConvTransposeInterp": ConvTransposeInterp})

        dices = []
        recalls = []
        precisions = []
        # aucs = []
        # tprs = []
        # mean_fpr = np.linspace(0, 1, 100)

        #################TRAINING####################
        '''
        train_generator = DataGenerator(
            directory="/home/helle246/data/LiS/training", shape=(512, 512),
            img_channels=1, lbl_channels=2,
            flat_labels=True, batch_size=20, tvl=False)
        results = model.fit_generator(
            generator=train_generator, steps_per_epoch=10, epochs=1,
            use_multiprocessing=False, workers=0)
            '''
        #############################################

        dirpath = Path(model_dir)
        ds = dirpath.name.split('_')[3]
        if ds == "lis":
            print("Already done")
            continue
            validation_generator = DataGenerator(
                directory="/home/helle246/data/LiS/testing", shape=(512, 512),
                img_channels=1, lbl_channels=2,
                flat_labels=True, batch_size=20, tvl=False, ds='lis')
            validation_generator.on_epoch_end()

        else:
            validation_generator = DataGenerator(
                directory="/home/helle246/data/pancreas/testing", shape=(512, 512),
                img_channels=1, lbl_channels=2,
                flat_labels=True, batch_size=20, tvl=False, ds='psd')
            validation_generator.on_epoch_end()


        for i in range(iterations):
            x, y = validation_generator.__getitem__(i)
            eval_results = model.evaluate(x, y)
            #print(model.metrics_names)
            #print(eval_results)

            dices.append(eval_results[3])
            recalls.append(eval_results[2])
            precisions.append(eval_results[1])

            y_pred = model.predict(x)

            # np.save("x.npy", x)
            # np.save("pred.npy", np.greater(np.reshape(y_pred[:,:,0],(20,512,512,1)), 0.5))
            # np.save("true.npy", np.reshape(y[:,:,[0]], (20,512,512,1)))


            y = y.flatten()
            y_pred = y_pred.flatten()
            # fpr, tpr, thresholds = roc_curve(y, y_pred, pos_label=1)
            # roc_auc = auc(fpr, tpr)
            # tprs.append(interp(mean_fpr, fpr, tpr))
            # tprs[-1][0] = 0.0
            # aucs.append(roc_auc)


            #print(model.metrics_names)
            #print(eval_results)
        mean_dice = np.mean(dices)
        mean_recall = np.mean(recalls)
        mean_precision = np.mean(precisions)
        print("dice = " + str(mean_dice))
        print("recall = " + str(mean_recall))
        print("precision = " + str(mean_precision))

        # mean_tpr = np.mean(tprs, axis=0)
        # mean_tpr[-1] = 1.0
        # mean_auc = auc(mean_fpr, mean_tpr)
        # print("auc = " + str(mean_auc))
        # print("test auc = " + str(np.mean(aucs)))

        #TODO store mean_auc, and metrics for data
        output_file = open(base_dir + '/'  + model_dir + '/' + "results.txt", 'w')
        # output_file.write("Mean Auc -> " + str(mean_auc) + '\n')
        output_file.write("Mean Dice -> " + str(mean_dice) + '\n')
        output_file.write("Mean Precision -> " + str(mean_precision) + '\n')
        output_file.write("Mean Recall -> " + str(mean_recall) + '\n')
        output_file.close()

        # np.save(base_dir + '/' + model_dir + '/' + "tpr.npy", tprs)

        del model
        K.clear_session()
