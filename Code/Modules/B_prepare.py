from Code.Selector.Selector      import Selector
from Code.Networks.RegressionCNN import RegressionCNN

def prepare(args):

    # Load selectors of Loss function and Optimizer
    Loss      = Selector("loss").select(args.loss)()
    Optimizer = Selector("optimizer").select(args.optimizer)(learning_rate=args.learning_rate)

    # Load Regression CNN
    CNN = RegressionCNN(filters=args.filters, regress=args.regress)

