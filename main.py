import logging

from Code.Parser.parser import parse

from Code.Modules.A_split import split
from Code.Modules.B_prepare import prepare
from Code.Modules.C_train import train
from Code.Modules.D_test import test

if __name__ == "__main__":

    logging.basicConfig(level=logging.NOTSET)

    logging.info("Parsing requests")
    args = parse()

    logging.info("Loading datasets")
    train_set, validation_set, test_set = split(args["Split"], feature=args["Project"].feature)
    shape = train_set.shape

    logging.info("Preparing network")
    CNN = prepare(args=args, shape=shape)

    logging.info("Training network")
    experiment = train(CNN, train_set, validation_set, args)

    logging.info("Testing network")
    test(CNN, experiment, test_set, feature=args["Project"].feature)
