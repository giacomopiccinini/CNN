from tensorflow.keras.losses import MeanAbsolutePercentageError
import logging

def loss(**kwargs):

    """ Create Mean Absolute Percentage Error (MAPE) loss """

    try:
        # Create loss if correct keywords are passed
        Loss = MeanAbsolutePercentageError(**kwargs)

        # Log the loading
        logging.info("Mean Absolute Percentage Error loss has been loaded")

        # Return the optimizer
        return Loss

    except Exception as e:
        # Raise an exception if wrong keywords are passed
        raise e
