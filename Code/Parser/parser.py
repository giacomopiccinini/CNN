import argparse
from argparse import ArgumentParser

def parse():

    """ Parse command line arguments """

    # Initiate argparser
    parser = ArgumentParser()

    # Add group for general info on the project
    project_group = parser.add_argument_group("Project", "Arguments for project options")

    # Add arguments
    project_group.add_argument("--project", const="Rivertrace", default="Rivertrace", nargs='?', type=str, help="String representing the name of the project (for logging purposes")
    project_group.add_argument("--feature", const="Ferric Oxide PPM", default="Ferric Oxide PPM", nargs='?', type=str, help="String representing the feature to be considered")

    # Add group for splitting
    split_group = parser.add_argument_group("Split", "Arguments for splitting options")

    # Add arguments
    split_group.add_argument("--test_size", const=0.2, default=0.2, nargs='?', type=float, help="Float in (0,1) representing the percentage of sample to be kept for testing")
    split_group.add_argument("--validation_size", const=0.2, default=0.2, nargs='?', type=float, help="Float in (0,1) representing the percentage of sample to be kept for validation")
    split_group.add_argument("--seed", const=42, default=42, nargs='?', type=int, help="Integer representing the seed used in train/test/validation splitting")
    split_group.add_argument("--batch",  const=32,  default=32,  nargs='?', type=int, help="Integer for batch size")

    # Add group for train
    prepare_group = parser.add_argument_group("Prepare", "Arguments for network preparation options")

    # Add arguments
    prepare_group.add_argument("--loss", const="MAPE", default="MAPE", nargs='?', type=str, choices=["MAPE"], help="Loss function to be used at training time")
    prepare_group.add_argument("--optimizer", const="Adam", default="Adam", nargs='?', type=str, choices=["Adam", "SGD"], help="Optimizer to be used at training time")
    prepare_group.add_argument("--learning_rate", const=0.1, default=0.1, nargs='?', type=float, help="Learning rate")
    prepare_group.add_argument("--filters", const=(4, 8, 16, 32, 64, 128, 256), default=(4, 8, 16, 32, 64, 128, 256), nargs='?', type=tuple, help="Filters to apply in CNN")
    prepare_group.add_argument("--regress", const=True, default=True, nargs='?', type=bool, help="Whether to apply or not regression as last layer")

    # Add group for train
    train_group = parser.add_argument_group("Train", "Arguments for training options")

    # Add arguments
    train_group.add_argument("--epochs", const=200, default=200, nargs='?', type=int, help="Integer for training epochs")

    # Parse arguments
    args = parser.parse_args()

    # Initialise dictionary for groups of arguments
    arg_groups={}

    # Add arguments to relevant group
    for group in parser._action_groups:
        group_dict={a.dest:getattr(args,a.dest,None) for a in group._group_actions}
        arg_groups[group.title]=argparse.Namespace(**group_dict)

    return arg_groups

