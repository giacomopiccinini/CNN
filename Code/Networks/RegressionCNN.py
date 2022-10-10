from xml.etree.ElementTree import TreeBuilder
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
)
from tensorflow.keras.models import Model


def RegressionCNN(shape, filters=(4, 8, 16, 32, 64, 128, 256, 512), regress=True):

    """Construct CNN for Regression.

    Parameters:

    width: width of input image
    height: height of input image
    depth: number of channels of input image (RGB = 3, greyscale = 1)
    filters: tuple indicating dimensions of filters to be applied to image during convolution
    regress : boolean indicating whether or not a fully-connected linear activation layer will be appended to the CNN for regression purposes.

    """

    # Instantiate a Keras tensor (it should be a n_xpixel x n_ypixel x n_channel tensor)
    input_image = Input(shape=shape)

    # At first, the tensor is simply the input
    x = input_image

    # Loop over the number of filters to construct CNN recursively
    for f in filters:

        # Apply 2D convolution with ReLU activation function
        x = Conv2D(f, kernel_size=(3, 3), activation="relu", padding="same")(x)

        # Apply batch normalisation (specify axis = -1 if assuming TensorFlow/channels-last ordering)
        x = BatchNormalization(axis=-1)(x)

        # Apply Max Pooling
        x = MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten the tensor before applying the dense layer
    x = Flatten()(x)

    # Apply dense layer (units = dimensionality of the output layer)
    x = Dense(units=16, activation="relu")(x)

    # Apply batch normalisation
    x = BatchNormalization(axis=-1)(x)

    # Add drop-out layer
    x = Dropout(0.2)(x)

    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    # x = Dense(4, activation="relu")(x)

    # Add regression layer if needed
    if regress:
        x = Dense(1, activation="linear")(x)

    # Construct the CNN
    model = Model(input_image, x)

    return model
