import logging

from Code.Parser.parser     import parse

from Code.Modules.A_split   import split
from Code.Modules.B_prepare import prepare
from Code.Modules.C_train   import train

if __name__ == '__main__':

    logging.info("Parsing requests")
    args = parse()

    logging.info("Loading datasets")
    (train_set, labels_train), (validation_set, labels_validation), (test_set, labels_test) = split(args["Split"])
    shape = train_set.shape

    logging.info("Preparing network")
    CNN = prepare(args=args["Prepare"], shape=shape)

    logging.info("Training network")
    train(CNN, train_set, labels_train, validation_set, labels_validation, args["Train"])


