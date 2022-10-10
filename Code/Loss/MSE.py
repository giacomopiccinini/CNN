from tensorflow.keras.losses import MeanSquaredError
import logging


def loss(**kwargs):

    """Create Mean Squared Error (MSE) loss"""

    try:
        # Create loss if correct keywords are passed
        Loss = MeanSquaredError(**kwargs)

        # Log the loading
        logging.info("Mean Squared Error loss has been loaded")

        # Return the optimizer
        return Loss

    except Exception as e:
        # Raise an exception if wrong keywords are passed
        raise e
