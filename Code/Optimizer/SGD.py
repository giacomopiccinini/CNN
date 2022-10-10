import logging
from tensorflow.keras.optimizers import SGD


def optimizer(**kwargs):

    """Create SGD optimizer"""

    try:
        # Create Adam optimizer if correct keywords are passed
        Optimizer = SGD(**kwargs)

        # Log the loading
        logging.info("SGD optimizer has been loaded")

        # Return the optimizer
        return Optimizer

    except Exception as e:
        # Raise an exception if wrong keywords are passed
        raise e
