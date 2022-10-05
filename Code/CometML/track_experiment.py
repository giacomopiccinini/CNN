from comet_ml import Experiment

def track(args):

    """ Track experiments on CometML """

    # Initialise parameters dictionary
    hyperparameters = {}
    project = {}

    # Merge args in a single dictionary
    for key, value in args.items():

        # Save hyperparameters
        if key in ["Split", "Prepare", "Train", "Test"]:
            hyperparameters = {**hyperparameters, **vars(value)}
        # Save project details
        elif key in ["Project"]:
            project = {**project, **vars(value)}
        else:
            continue

    # Instantiate experiment
    experiment = Experiment(project_name=args["Project"].project)

    # Log hyperparameters
    experiment.log_parameters(hyperparameters)

    # Log project details
    experiment.log_others(project)

    return experiment
