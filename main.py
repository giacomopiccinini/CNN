import logging

from Code.Parser.parser     import parse

from Code.Modules.A_split   import split
from Code.Modules.B_prepare import prepare
from Code.Modules.C_train   import train

if __name__ == '__main__':

    logging.info("Parsing requests")
    args = parse()
    print(args)

    logging.info("Loading datasets")
    #train_set, validation_set, test_set = split(args["Split"])
    #shape = train_set.shape

    logging.info("Preparing network")
    #CNN = prepare(args=args["Prepare"], shape=shape)

    logging.info("Training network")
    #train(CNN, train_set, validation_set, args["Train"])


