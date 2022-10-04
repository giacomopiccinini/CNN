from Code.Modules.A_split import split

import logging

if __name__ == '__main__':

    logging.info("Parsing requests")

    logging.info("Loading datasets")
    (train_set, labels_train), (validation_set, labels_validation), (test_set, labels_test) = split()

    logging.info("Preparing network")


