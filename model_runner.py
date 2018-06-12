import argparse
import sys
import numpy as np
from pathlib import Path
import signal, os
from time import gmtime, strftime

import keras.backend as K
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger

from Models.UNet import create_unet
from Models.SegNet import create_segnet
from Models.FCN32 import create_fcn32
from Models.metrics import *
from DataGenerator import DataGenerator
from callbacks.vizpreds import VizPreds
from callbacks.topnsaver import TopNSaver
from losses.sampledbce import sampled_bce
import socket

IMG_SIZE = 512
IMG_CHANNELS = 1
LBL_CHANNELS = 2
NUM_SLICES_PER_FILE = 20

# Prior weights of cost for each class
LIS_CLASS_WEIGHTS = [1, 1]
PSD_CLASS_WEIGHTS = [1, 1]

# probability of OMITTING a pixel
LIS_TVL_SAMPLING_PROB = 1.0 - 0.0716433
PSD_TVL_SAMPLING_PROB = 1.0 - 0.0044771

EPOCHS = 100
STEPS_PER_EPOCH = 35
VAL_STEPS = 10


""" Directories """
cwd = Path(sys.argv[0])
model_dir = cwd / 'model_files'

hostname = socket.gethostname()
if hostname.startswith('ju'):
    lis_dir = Path("/media/storage/LiS")
else:
    lis_dir = Path("/home/helle246/data/LiS")
psd_dir = Path("/home/helle246/data/PSD")
model_dir = Path("/media/storage/labels18_models")


""" Model Builders """

'''
Fully Visualized and Working ->
UNet
FCN32
Segnet

Saving for a Journal Paper Later ->
FCN8
DeeplabV3
Mask R-CNN
'''

def get_fcn32(args):
    if args.existing == None:
        # Build feed-forward network, compile with trainer
        return create_fcn32(
            IMG_SIZE, IMG_CHANNELS, initial_filters=6
        )
    #Load old model
    return load_model(str(args.existing), custom_objects={'precision': precision, 'recall': recall, 'dice_sorensen': dice_sorensen})

def get_unet(args):
    if args.existing == None:
        # Build feed-forward network, compile with trainer
        return create_unet(
            IMG_SIZE, IMG_CHANNELS, filter_size=3, transpose_filter_size=2, depth=5,
            initial_filters=6
        )

    # Load old model
    return load_model(str(args.existing), custom_objects={'precision': precision, 'recall': recall, 'dice_sorensen': dice_sorensen})

def get_segnet(args):
    if args.existing == None:
        # Build feed-forward network, compile with trainer
        return create_segnet((IMG_SIZE, IMG_SIZE, IMG_CHANNELS), LBL_CHANNELS)

    # Load old model
    return load_model(str(args.existing), custom_objects={'precision': precision, 'recall': recall, 'dice_sorensen': dice_sorensen})


""" Ancillary Functions to Parse Arguments and Prep Training """

def get_time():
    return strftime("%Y-%m-%d_%H-%M-%S", gmtime())

def get_run_dir(args):
    if (args.new is not None):
        given_name = args.new
    elif (args.existing is not None):
        run_dir = Path(args.existing)
        run_dir = run_dir.parent
        return run_dir
    else:
        print("Please provide a name for the training run [--new|--existing]")
        sys.exit()

    run_info = [given_name, args.model.lower(), args.batch_size, args.dataset,
                args.perturbations, get_time()]

    run_dir = model_dir / "_".join(map(str, run_info))
    run_dir.mkdir()
    return run_dir

def get_callbacks(args):
    tensorboard = TensorBoard(log_dir=str(args.run_dir), histogram_freq=0,
        write_graph=False)
    logging_dir_str = str(args.run_dir / "training.csv")
    csv_logging = CSVLogger(logging_dir_str, append=True)
    print("Logging at: %s" % logging_dir_str)
    print()
    model_saver = TopNSaver(args.run_dir, 5, mode="min")
    callbacks = [tensorboard, csv_logging, model_saver]
    if (args.visualize):
        vizcallback = VizPreds("./viz/" + args.dataset.lower(), 5)
        callbacks = callbacks + [vizcallback]

    return callbacks


""" Runner """

if __name__ =='__main__':
    os.setpgrp()
    parser = argparse.ArgumentParser(
        description="Run a training or testing round for our Labels 2018 Submission")
    parser.add_argument('-n', '--new', nargs=1, dest="new",
        default=None,
        help='A name for a new training session from random initialization')
    parser.add_argument('-e', '--existing', nargs=1, dest='existing',
        default=None,
        help='A name for an existing training session to continue') #TODO (later)
    parser.add_argument('-b', '--batch_size', nargs=1, dest='batch_size',
        default=[20], type=int,
        help='The batch size to use during training')
    parser.add_argument('-d', '--dataset', nargs=1, dest='dataset',
        default=["lis"],
        help='The dataset to train/test on either lis or psd')
    parser.add_argument('-p', '--perturbations', nargs=1, dest='perturbations',
        default=["control"],
        help='The type of perturbations to do to the training data')
    parser.add_argument('-g', '--gpu', nargs=1, dest='gpu_index',
        default=["0"],
        help='The GPU to limit this training run to.')
    parser.add_argument('-m', '--model', nargs=1, dest='model',
        default=["unet"],
        help='The architecture to use for this round')
    parser.add_argument('-t', '--testing', dest="testing", nargs='?', const=[True],
        help='Whether this round will be a testing round') # TODO (later)
    parser.add_argument('-v', '--visualize', dest="visualize", nargs='?', const=[True],
        help='Whether to visualize')

    print()
    parser.print_help()
    print()
    args = parser.parse_args()

    """ I don't really get why this is necessary but I don't have the patience
    for a better solution right now """
    for a in args.__dict__:
        if args.__dict__[a] is not None:
            args.__dict__[a] = args.__dict__[a][0]

    # Limit the program to only one GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index

    # Set directories
    ds = args.dataset.lower()
    if (ds == "lis"):
        data_dir = lis_dir
        class_weights = LIS_CLASS_WEIGHTS
        objective = lambda ytrue, ypreds: sampled_bce(
            ytrue, ypreds, LIS_TVL_SAMPLING_PROB
        )
    elif (ds == "psd"):
        data_dir = psd_dir
        class_weights = PSD_CLASS_WEIGHTS
        objective = lambda ytrue, ypreds: sampled_bce(
            ytrue, ypreds, PSD_TVL_SAMPLING_PROB
        )
    else:
        print("Dataset: %s is not supported." % ds)
        raise NotImplementedError

    args.training_dir = data_dir / "training"
    args.validation_dir = data_dir / "validation"
    args.testing_dir = data_dir / "testing"

    # Make directory for logging and saving checkpoints
    args.run_dir = get_run_dir(args)

    # Handle testing
    if (args.testing):
        test_generator = DataGenerator(
            directory=str(args.testing_dir), shape=(IMG_SIZE, IMG_SIZE),
            img_channels=IMG_CHANNELS, lbl_channels=LBL_CHANNELS,
            flat_labels=True, perturbations=args.perturbations, ds=ds,
            batch_size=args.batch_size, tvl=False)
        if (args.model.lower() == "unet"):
            model = get_unet(args)
        elif (args.model.lower() == "fcn"):
            model = get_fcn32(args)
        elif (args.model.lower() == "segnet"):
            model = get_segnet(args)
        else:
            print('Model: "%s" is not supported.' % args.model)
            raise NotImplementedError

        results = model.evaluate_generator(test_generator)
        for el in results:
            print(el)
    else:

        train_generator = DataGenerator(
            directory=str(args.training_dir), shape=(IMG_SIZE, IMG_SIZE),
            img_channels=IMG_CHANNELS, lbl_channels=LBL_CHANNELS,
            flat_labels=True, perturbations=args.perturbations, ds=ds,
            batch_size=args.batch_size, tvl=False)
        validation_generator = DataGenerator(
            directory=str(args.validation_dir), shape=(IMG_SIZE, IMG_SIZE),
            img_channels=IMG_CHANNELS, lbl_channels=LBL_CHANNELS, ds=ds,
            flat_labels=True, batch_size=args.batch_size, tvl=False)

        # define callbacks for training
        callbacks = get_callbacks(args)

        # Build the model specified by the args
        if (args.model.lower() == "unet"):
            model = get_unet(args)
        elif (args.model.lower() == "fcn"):
            model = get_fcn32(args)
        elif (args.model.lower() == "segnet"):
            model = get_segnet(args)
        else:
            print('Model: "%s" is not supported.' % args.model)
            raise NotImplementedError

        # Compile the model
        model.compile(optimizer=Adam(), loss=objective,
            metrics=[precision, recall, dice_sorensen, 'accuracy'])

        # Print out summary of feed-foward graph
        model.summary()


        # Train the model
        results = model.fit_generator(
            generator=train_generator, validation_data=validation_generator,
            validation_steps=VAL_STEPS, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS,
            use_multiprocessing=False, workers=0, class_weight=class_weights,
            callbacks=callbacks)

        # Save when done
        model.save(str(args.run_dir / "final_weights.h5"))





    # Stop execution for fuck's sake
    os.killpg(0, signal.SIGKILL)
    sys.exit()
