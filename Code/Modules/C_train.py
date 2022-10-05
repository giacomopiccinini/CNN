def train(CNN, train_set, validation_set, args):

    """ Train CNN """

    # Train CNN
    CNN.fit(train_set,validation_data=validation_set, epochs=args.epochs)

