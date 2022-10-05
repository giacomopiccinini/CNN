import numpy as np
from tensorflow.keras.callbacks import Callback

class Comet(Callback):

    """ Custom Callback to sync with CometML """

    def __init__(self, train, validation, experiment):
        """ Construct where to store train and validation set"""
        super().__init__()
        self.validation = validation
        self.train = train
        self.experiment = experiment

    def on_epoch_end(self, logs=None):

        logs['MAE_train'] = float('inf')

        # Load training data
        X_train, y_train = self.train[0], self.train[1]

        # Compute predictions
        y_pred = self.model.predict(X_train)
        train_score = np.abs(y_train - y_pred).mean()  

        logs['MAE_val'] = float('inf')   

        # Load training data
        X_val, y_val = self.validation[0], self.validation[1]

        # Compute predictions
        y_pred = self.model.predict(X_val)
        val_score = np.abs(y_val - y_pred).mean() 

        self.experiment.log_metric("MAE_train", train_score)
        self.experiment.log_metric("MAE_val", val_score)
