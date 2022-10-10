import os
import tensorflow as tf
import numpy as np

from Code.CometML.track_experiment import track
from Code.Callbacks.callback import list_callbacks


def train(CNN, train_set, validation_set, args):

    """Train CNN"""
    # Ensure GPU is visible
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # TensorFlow routine
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # Load experiment
    experiment = track(args)

    # Load callbacks
    callbacks = list_callbacks(experiment)

    # Train CNN
    CNN.fit(
        train_set,
        validation_data=validation_set,
        epochs=args["Train"].epochs,
        callbacks=callbacks,
    )

    return experiment
