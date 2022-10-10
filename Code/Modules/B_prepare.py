from Code.Selector.Selector import Selector
from Code.Networks.RegressionCNN import RegressionCNN
from Code.Metric.R2 import r2


def prepare(args, shape):

    # Load selectors of Loss function and Optimizer
    Loss = Selector("loss").select(args["Prepare"].loss)()
    Optimizer = Selector("optimizer").select(args["Prepare"].optimizer)(
        learning_rate=args["Prepare"].learning_rate,
        **vars(args[args["Prepare"].optimizer])
    )

    if len(shape) == 2:
        new_shape = (shape[0], shape[1], 1)
    else:
        new_shape = shape

    # Load Regression CNN
    CNN = RegressionCNN(
        shape=new_shape,
        filters=args["Prepare"].filters,
        regress=args["Prepare"].regress,
    )

    # Compile and summarise model
    CNN.compile(optimizer=Optimizer, loss=Loss, metrics=["mae", "mse", r2])
    CNN.summary()

    return CNN
