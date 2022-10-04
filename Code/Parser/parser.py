import argparse
from argparse import ArgumentParser

def parse():

    """ Parse command line arguments """

    # Initiate argparser
    parser = ArgumentParser()

    # Add group for splitting
    split_group = parser.add_argument_group("Split", "Arguments for splitting options")

    # Add arguments
    split_group.add_argument("--test_size", const=0.2, default=0.2, nargs='?', type=float, help="Float in (0,1) representing the percentage of sample to be kept for testing")
    split_group.add_argument("--validation_size", const=0.2, default=0.2, nargs='?', type=float, help="Float in (0,1) representing the percentage of sample to be kept for validation")
    split_group.add_argument("--seed", const=42, default=42, nargs='?', type=int, help="Integer representing the seed used in train/test/validation splitting")

    # Add group for train
    train_group = parser.add_argument_group("Train", "Arguments for training options")

    # Add arguments
    train_group.add_argument("--loss", const="MAPE", default="MAPE", nargs='?', type=str, choices=["MAPE"], help="Loss function to be used at training time")
    train_group.add_argument("--optimizer", const="Adam", default="Adam", nargs='?', type=str, choices=["Adam", "SGD"], help="Optimizer to be used at training time")
    train_group.add_argument("--lr", const=0.1, default=0.1, nargs='?', type=float, help="Learning rate")

    # Parse arguments
    args = parser.parse_args()

    # Initialise dictionary for groups of arguments
    arg_groups={}

    # Add arguments to relevant group
    for group in parser._action_groups:
        group_dict={a.dest:getattr(args,a.dest,None) for a in group._group_actions}
        arg_groups[group.title]=argparse.Namespace(**group_dict)

    return arg_groups

