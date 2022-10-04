from tensorflow.keras.optimizers import Adam

def optimizer(**kwargs):

    """ Create Adam optimizer """

    try:
        # Create Adam optimizer if correct keywords are passed
        Optimizer   = Adam(**kwargs)

        # Return the optimizer
        return Optimizer

    except Exception as e:
        # Raise an exception if wrong keywords are passed
        raise e

