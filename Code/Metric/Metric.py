import numpy as np


class Metric:

    """Class for evaluating results of predictions in regression
    or classification"""

    def __init__(self, y_truth, y_pred, regress):

        # Store ground truths and predictions
        self.y_truth = y_truth
        self.y_pred = y_pred

        # Store if we are in a classification or regression problem
        self.regress = regress

    def compute_mae(self):

        """Compute mean absolute error"""
        self.mae = np.mean(np.abs(self.y_truth - self.y_pred))

    def compute_mse(self):

        """Compute mean squared error"""
        self.mse = np.mean(np.square(self.y_truth - self.y_pred))

    def compute_r2(self):

        """Compute the coefficient of determination R2"""

        # Compute the average
        y_avg = np.mean(self.y_truth)

        # Compute sum of squares of residuals
        SS_res = np.sum(np.square(self.y_truth - self.y_pred))

        # Compute total sum of square
        SS_tot = np.sum(np.square(self.y_truth - y_avg))

        # Compute r2
        self.r2 = 1 - SS_res / SS_tot

    def evaluate(self):

        """Evaluate all metrics"""

        # Compute all metrics
        self.compute_mae()
        self.compute_mse()
        self.compute_r2()

        # Store all metrics in a dictionary
        metrics = {}

        metrics["mae"] = self.mae
        metrics["mse"] = self.mse
        metrics["r2"] = self.r2

        return metrics
