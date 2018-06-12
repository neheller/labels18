import keras
from pathlib import Path
import numpy as np
import math

class TopNSaver(keras.callbacks.Callback):

    def __init__(self, model_dir, n, metric="val_loss", mode="max"):
        """Instantiate the TopNSaver object

        model_dir: directory in which to store the models (Path object)
        n: number of models to store at a time
        metric: thing in the logs to check the performance of a model
        mode: "max" if high is good, "min" if low is good
        """

        self.model_dir = model_dir
        self.n = n
        self.metric = metric
        self.mode = mode

        if (mode == "max"):
            self.top_n_metrics = -1*math.inf*np.ones(n)
        else:
            self.top_n_metrics = math.inf*np.ones(n)
        self.top_n_files = [model_dir for i in range(0,n)]

    def get_filepath(self, epoch, metrics, logs):
        """Get the file in which to store the model (Path object)

        epoch: the index of this epoch
        metrics: a list of keys for the logs to include in the filename
        logs: the object with all the info for this epoch
        """
        return self.model_dir / (
            ("weights__epoch-%03d_"
            + "_".join([m + "-%0.3f" for m in metrics]) + ".h5") %
            tuple([epoch] + [logs.get(m) for m in metrics])
        )

    def on_epoch_end(self, epoch, logs={}):
        """After each epoch, save and delete and existing one if called for"""
        if (self.mode == "max"):
            cur_min_ind = np.argmin(self.top_n_metrics)
            if (logs.get(self.metric) > self.top_n_metrics[cur_min_ind]):
                self.top_n_metrics[cur_min_ind] = logs.get(self.metric)
                old_file = self.top_n_files[cur_min_ind]
                self.top_n_files[cur_min_ind] = self.get_filename(
                    epoch, logs.get("val_dice_sorensen")
                )
                if not old_file.is_dir():
                    old_file.unlink()
                self.model.save(str(self.top_n_files[cur_min_ind]), overwrite=True)
        else:
            cur_max_ind = np.argmax(self.top_n_metrics)
            if (logs.get(self.metric) < self.top_n_metrics[cur_max_ind]):
                self.top_n_metrics[cur_max_ind] = logs.get(self.metric)
                old_file = self.top_n_files[cur_max_ind]
                self.top_n_files[cur_max_ind] = self.get_filepath(
                    epoch, ["val_dice_sorensen", self.metric], logs
                )
                if not old_file.is_dir():
                    old_file.unlink()
                self.model.save(str(self.top_n_files[cur_max_ind]), overwrite=True)
