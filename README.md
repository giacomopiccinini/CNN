# Convolutional Neural Network
This is a template implementation of a convolutional neural network for regression *or* classification tasks. When calling the script it is possible to specify which case we are interested in and the network will adapt accordingly. 

## Comet ML

When running the script, every (hyper)parameter is synced on Comet ML, and so are the training/validation curves. In order to be able to see the results, all you have to do is insert you API key in `comet.config` in the space I have provided for you. Also, please add the name *project name* as this is important when logging the experiments. 

## Input organisation

Place in `Input/Images` the images you wish to classify or apply a regression upon. Similarly, in `Input/Labels` put either the label or the quantitative value corresponding to the image at hand. **Notice**: image and label must have the same name across directories, e.g. `Input/Images/foo.jpg` ans `Input/Labels/foo.yaml`. If this condition is not met, it will not be possible to associate each image with the corresponding label. 

## Running the script

In order to run the script it is sufficient to type

`python main.py`

However, there are a number of (hyper) parameters that can be passed to specify a few details. 

```
[-h] # Print Help

[--project [PROJECT]] # String representing the name of the project (for logging purposes)

[--feature [FEATURE]] # String representing the feature to be considered (in the case of labels in yaml file)

[--test_size [TEST_SIZE]] # Float in (0,1) representing the percentage to be kept for testing

[--validation_size [VALIDATION_SIZE]] # Float in (0,1) representing the percentageto be kept for validation

[--seed [SEED]] # Integer representing the seed used in train/test/validation splitting

[--batch [BATCH]] # Integer for batch size

[--loss [{MAE,MSE,MAPE}]] # Loss function to be used at training time

[--optimizer [{Adam,SGD}]] # Optimizer to be used at training time

[--learning_rate [LEARNING_RATE]] # Learning rate for all types of optimizer

[--filters [FILTERS]] # Tuple with dimension of filters to apply in the CNN

[--regress [REGRESS]] # Whether to apply or not regression as last layer

[--beta_1 [BETA_1]] # Beta_1 parameter for Adam

[--beta_2 [BETA_2]] # Beta_1 parameter for Adam

[--epsilon [EPSILON]] # Epsilon parameter for Adam

[--momentum [MOMENTUM]] # Momentum parameter for SGD

[--epochs [EPOCHS]] # Integer for training epochs

```

