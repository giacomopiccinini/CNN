from Code.Selector.Selector import Selector

def prepare(args):

    # Load selectors of Loss function and Optimizer
    loss_selector      = Selector("loss").select(args.loss)
    optimizer_selector = Selector("optimizer").select(args.optimizer)