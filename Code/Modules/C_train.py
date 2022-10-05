def train(CNN, train_set, labels_train, validation_set, labels_validation, args):

    """ Train CNN """

    # Train CNN
    CNN.fit(train_set, labels_train, validation_data=(validation_set, labels_validation), epochs=args.epochs, batch_size=args.batch)

