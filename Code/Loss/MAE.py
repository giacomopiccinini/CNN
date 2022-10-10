from tensorflow.keras.losses import MeanAbsoluteError
import logging


def loss(**kwargs):

    """Create Mean Absolute Error (MAE) loss"""

    try:
        # Create loss if correct keywords are passed
        Loss = MeanAbsoluteError(**kwargs)

        # Log the loading
        logging.info("Mean Absolute Error loss has been loaded")

        # Return the optimizer
        return Loss

    except Exception as e:
        # Raise an exception if wrong keywords are passed
        raise e
