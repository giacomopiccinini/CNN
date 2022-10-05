from Code.CometML.track_experiment import track
from Code.Callbacks.callback import list_callbacks

def train(CNN, train_set, validation_set, args):

    """ Train CNN """

    # Load experiment
    experiment = track(args)

    # Load callbacks
    callbacks = list_callbacks(experiment, train_set, validation_set)

    # Train CNN
    CNN.fit(train_set, validation_data=validation_set, epochs=args["Train"].epochs, callbacks=callbacks)

