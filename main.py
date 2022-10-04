from Code.Parser.parser     import parse
from Code.Modules.A_split   import split
from Code.Modules.B_prepare import prepare

import logging

if __name__ == '__main__':

    logging.info("Parsing requests")
    args = parse()

    logging.info("Loading datasets")
    #(train_set, labels_train), (validation_set, labels_validation), (test_set, labels_test) = split(args["Split"])

    logging.info("Preparing network")
    prepare(args["Train"])


