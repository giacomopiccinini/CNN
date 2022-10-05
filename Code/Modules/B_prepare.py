from Code.Selector.Selector      import Selector
from Code.Networks.RegressionCNN import RegressionCNN

def prepare(args, shape):

    # Load selectors of Loss function and Optimizer
    Loss      = Selector("loss").select(args.loss)()
    Optimizer = Selector("optimizer").select(args.optimizer)(learning_rate=args.learning_rate)

    if len(shape) == 2:
        new_shape = (shape[0], shape[1], 1)
    else:
        new_shape = shape

    # Load Regression CNN
    CNN = RegressionCNN(shape=new_shape, filters=args.filters, regress=args.regress)

    # Compile and summarise model
    CNN.compile(optimizer = Optimizer, loss = Loss, metrics=["mse"])
    CNN.summary()

    return CNN