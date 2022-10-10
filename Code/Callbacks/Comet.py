import numpy as np
from tensorflow.keras.callbacks import Callback


class Comet(Callback):

    """Custom Callback to sync with CometML"""

    def __init__(self, experiment):

        """Construct where to store train and validation set"""
        super().__init__()
        self.experiment = experiment

    def on_epoch_end(self, epoch, logs=None):

        # Log losses
        self.experiment.log_metric("val_loss", logs.get("val_loss"), step=epoch)
        self.experiment.log_metric("train_loss", logs.get("loss"), step=epoch)
        self.experiment.log_metric("val_mse", logs.get("val_mse"), step=epoch)
