import numpy as np
import keras
import tensorflow as tf
import os
from pathlib import Path

from preprocessing.tools import preprocess, perturb

SLICES_PER_NPY = 20
#Generates data from numpy arrays
class DataGenerator(keras.utils.Sequence):

    def __init__(self, directory, shape, img_channels, lbl_channels, batch_size,
                    flat_labels=False, perturbations="control", shuffle=True,
                    tvl=False, val=False, ds="lis"):
        self.pth = Path(directory)
        self.shape = shape
        self.img_channels = img_channels
        self.lbl_channels = lbl_channels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.tvl = tvl
        if ds == "lis":
            self.glob = "y*lo*.npy"
            self.xpfx = "x"
            self.sparse = False
        else:
            self.glob = "Y*.npy"
            self.xpfx = "X"
            self.sparse = True
        self.ds = ds
        self.data, self.labels = self.train_filenames()
        self.on_epoch_end() #make sure indices is initialized
        self.flat_labels = flat_labels
        self.perturbations = perturbations

        self.cache_size = int(np.ceil(self.batch_size/SLICES_PER_NPY))
        self.cache = np.zeros([self.cache_size*SLICES_PER_NPY,512,512,3])
        self.last_loaded = 0

        self.num_indices = len(self.data)

        self.no_shuffle = True # Don't shuffle at the moment
        for i in range(0, self.cache_size):
            self._load_next_npy(i)
        self.no_shuffle = False

    def train_filenames(self):
        data = []
        labels = []
        for f in self.pth.glob(self.glob):
            if (("ntl" in f.name) or (self.tvl)):
                typ = f.name.split('-')[1]
                index = int(f.name.split('-')[-1].split('.')[0])
                data.append(
                    str(self.pth / ('%s-%s-%d.npy' % (self.xpfx, typ, index)))
                )
                labels.append(str(f))

        data = np.array(data)
        labels = np.array(labels)
        return data, labels

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _load_next_npy(self, index):
        self.last_loaded = self.last_loaded + 1
        if (self.last_loaded >= self.cache_size):
            self.last_loaded = 0
        start = self.last_loaded*SLICES_PER_NPY

        self.cache[start:start+SLICES_PER_NPY,:,:,[0]] = preprocess(
            np.load(self.data[index])
        )
        labels_name = self.labels[index]
        if ("tvl" in labels_name):
            self.cache[start:start+SLICES_PER_NPY,:,:,1] = np.zeros([512,512])
            self.cache[start:start+SLICES_PER_NPY,:,:,2] = np.ones([512,512])
        elif (not self.sparse):
            self.cache[start:start+SLICES_PER_NPY,:,:,[1,2]] = perturb(
                np.load(labels_name), self.perturbations, self.ds
            )
        else:
            tby = np.zeros((20,512,512,2))
            tby[:,:,:,[0]] = np.load(labels_name)
            tby[:,:,:,[1]] = 1 - tby[:,:,:,[0]]
            self.cache[start:start+SLICES_PER_NPY,:,:,[1,2]] = perturb(
                tby, self.perturbations, self.ds
            )

        if not self.no_shuffle:
            np.random.shuffle(self.cache)

    def __getitem__(self, index):
        self._load_next_npy(index)
        X, Y = (
            self.cache[0:self.batch_size,:,:,[0]],
            self.cache[0:self.batch_size,:,:,[1,2]]
        )
        if self.flat_labels:
            Y = Y.reshape((self.batch_size, np.prod(self.shape), self.lbl_channels))
        return X, Y

    def __len__(self):
        return len(self.data)
