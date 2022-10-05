from Code.CometML.track_experiment import track

def train(CNN, train_set, validation_set, args):

    """ Train CNN """

    experiment = track(args)

    # Declare training environment
    with experiment.context_manager("train"):

        # Train CNN
        CNN.fit(train_set,validation_data=validation_set, epochs=args["Train"].epochs)

