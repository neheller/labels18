import keras
from pathlib import Path
import matplotlib.pyplot as plt
import scipy
import numpy as np

from preprocessing.tools import preprocess


class VizPreds(keras.callbacks.Callback):

    def __init__(self, vizdir, numviz):
        """Construct the callback object

        vizdir: directory in which to find files to visualize
        numviz: number of slices from these files in which to visualize
        """
        # Load the files from specified directory
        viz_path = Path(vizdir)
        self.X = np.zeros((numviz,512,512,1))
        self.Y = np.zeros((numviz,512,512,2))
        self.numviz = numviz
        for fn in list(viz_path.glob("*.npy")):
            if (fn.name.lower().startswith('x')):
                X = np.load(str(fn))
            if (fn.name.lower().startswith('y')):
                Y = np.load(str(fn))
        # Store only the number of slices specified
        for i in range(0, min(numviz, Y.shape[0])):
            self.X[i,:,:,:] = preprocess(X[i,:,:,:])
            self.Y[i,:,:,:] = Y[i,:,:,:]

    def refresh_plot(self):
        """Refresh plots with predictions from updated model"""
        P = self.model.predict(x=self.X)
        P = np.reshape(P, [-1, 512, 512, 2])
        for i in range(0,self.numviz):
            self.axes[i,1].imshow(P[i,:,:,0],cmap='gray')
            plt.pause(0.0001)

    def on_train_begin(self, logs={}):
        """Instantiate plots in such a way that they persist"""
        fig, axes = plt.subplots(self.numviz, 3)
        P = np.zeros(self.Y.shape)
        for i in range(0,self.numviz):
            axes[i,0].imshow(self.X[i,:,:,0],cmap='gray')
            plt.pause(0.0001)
            axes[i,1].imshow(P[i,:,:,0],cmap='gray')
            plt.pause(0.0001)
            axes[i,2].imshow(self.Y[i,:,:,0],cmap='gray')
            plt.pause(0.0001)
            for j in range(0,3):
                axes[i,j].xaxis.set_visible(False)
                axes[i,j].yaxis.set_visible(False)
        plt.tight_layout()
        plt.ion()
        plt.show()
        self.axes = axes
        self.fig = fig

    def on_epoch_end(self, batch, logs={}):
        """Refresh the plots at the end of each epoch"""
        self.refresh_plot()
