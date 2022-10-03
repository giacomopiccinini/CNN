import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.ndimage import label

from Code.Techniques.Segmentation.adaptive import adaptive_threshold

class Image():


    # Constructor 
    def __init__(self, tensor, grey, path, name):

        """ Initialise image """

        # Store image as tensor
        self.tensor = tensor
        self.grey   = grey
        self.path   = path
        self.name   = name
        self.shape  = self.tensor.shape 
        

    # Initialise from path
    @classmethod
    def from_path(cls, path):

        """ Initialise image from path """

        # Read image
        image = cv2.imread(path, -1)

        # Determine if greyscale
        tensor, grey = cls.is_grey(cls, image)

        # Initialise class
        return cls(tensor, grey, Path(path), Path(path).name)


    # Initialise from tensor
    @classmethod
    def from_tensor(cls, image, path=None, name=None):

        """ Initialise image from tensor """

        # Determine if greyscale
        tensor, grey = cls.is_grey(cls, image)

        # Initialise without path
        return cls(tensor, grey, path, name)



    def is_grey(self, image): 

        """ Determine if an image is grey scale """

        # Image is already greyscale and just has width x height
        if len(image.shape) < 3: 

            grey   = True
            tensor = image
            
        # Image is of the form (height, width, 1)
        elif image.shape[2]  == 1:
            image = image.reshape((image.shape[0], image.shape[1]))
            grey   = True
            tensor = image

        # Image has 3 channels
        else:
            # Extract BGR value
            b,g,r = image[:,:,0], image[:,:,1], image[:,:,2]

            # If all channels are equal is greyscale
            if (b==g).all() and (b==r).all():
                image = image[:, :, :1]
                image = image.reshape((image.shape[0], image.shape[1]))
                grey   = True
                tensor = image

            # RGB image
            else:
                grey   = False
                tensor = image

        return tensor, grey


    def unique(self, return_counts=False):

        """ Return unique values in an image """

        return np.unique(self.tensor, return_counts=return_counts)


    @classmethod
    def adaptive_thresholding(cls, image):

        """ Segment image using adaptive threshold. Return an instance
        of the Image class. """

        # Segment the tensor
        segmented = adaptive_threshold(image.tensor)

        return cls.from_tensor(segmented, name="segmented")


    @classmethod
    def label(cls, image):

        """ Label an already segmented image """

        # If image is not segmented raise an error
        if image.name != "segmented":
            raise ValueError("Image is not segmented!")

        # Label image
        labelled, _ = label(image.tensor)

        return cls.from_tensor(labelled)


    def __getitem__(self, item):

        """ Define __getitem__ to make the class subscriptable """
        return self.tensor[item]

    
    def dataframe(self, cutoff=0):

        """ Get dataframe corresponding with area of unique pixel values"""

        # Find unique values and their number
        values, areas = self.unique(return_counts=True)

        # Construct dataframe
        df = pd.DataFrame(columns = ["Value", "Area"])
        df["Value"] = values
        df["Area"]  = areas

        # Filter dataframe to areas >= cutoff
        df.query("Area >= @cutoff", inplace=True)

        # Reset index
        df.reset_index(drop=True, inplace=True)

        return df


    def plot(self, y_min=None, y_max=None, x_min=None, x_max=None):

        """ Plot image """

        # Reset extrema
        y_min = 0 if y_min is None else y_min
        y_max = self.shape[0] if y_max is None else y_max

        x_min = 0 if x_min is None else x_min
        x_max = self.shape[1] if x_max is None else x_max

        # Create figure
        figsize = (10, 10)

        # Plot figure
        plt.figure(figsize=figsize)
        plt.imshow(self.tensor[y_min: y_max, x_min: x_max])
        plt.axis("off")
        plt.show()


    @classmethod
    def normalise(cls, image, maximum, minimum):

        # Load tensor from image
        tensor = image.tensor

        # Find the span
        span = float(maximum - minimum)

        # Convert to float
        tensor = tensor.astype('float32')

        # Compute normalised
        normalised = (tensor - minimum) / span

        return cls.from_tensor(normalised)