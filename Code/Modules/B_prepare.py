from Code.Selector.Selector import Selector

def prepare(args):

    # Load selectors of Loss function and Optimizer
    Loss      = Selector("loss").select(args.loss)()
    Optimizer = Selector("optimizer").select(args.optimizer)(learning_rate=args.lr)

    