import numpy as np
import multiprocessing as mp
import yaml
from pathlib import Path
from tensorflow.keras.utils import Sequence
from Code.Classes.Image import Image

class ImageLoader(Sequence):
    
    """ Read images from path, storing them in Image class"""
    
    def __init__(self, paths, set_type="train", batch_size=32, maximum=None, minimum=None):
        
        """ Constructor for ImageLoader class"""

        # Initiate pool for multiprocessing (for speeding up reading of images)
        pool = mp.Pool(mp.cpu_count())

        # Load images
        data = pool.map(Image.from_path, paths)

        # Close pool
        pool.close()

        # Store details
        self.paths      = paths
        self.batch_size = batch_size
        self.data       = data
        self.number     = len(paths)
        self.names      = [Path(path).name for path in self.paths]
        self.set_type   = set_type
        self.shape      = self.check_shape()

        # Set maximum and minimum if not fixed
        if maximum==None or minimum==None:
            self.maximum, self.minimum = self.get_extrema()
        else:
            self.maximum, self.minimum = maximum, minimum

    def __len__(self):
        
        """ Return the number of batches needed for completing the dataset """
        
        return int(np.ceil(self.number / float(self.batch_size)))

    def __getitem__(self, i):
        
        """ Return the images in the i-th batch """
        
        # Find paths for the i-th batch
        batch = self.data[i*self.batch_size : (i + 1)*self.batch_size]

        # Normalise images
        normalised_images = [Image.normalise(image, self.maximum, self.minimum) for image in batch]

        # Retrieve images in the form of tensor from the batch
        batch = np.array([normalised_image.tensor for normalised_image in normalised_images])
                
        return batch, batch  

    def check_shape(self):

        """ Check that all images in dataset have the same shape. If not, raise an error"""

        # Retrieve shapes
        shapes = np.array([tensor.shape for tensor in self.data])

        # Check unicity
        shape = np.unique(shapes)

        # If more than one value
        if len(shape) > 1:
            
            raise ValueError(f"Images in {self.set_type} set do not have the same shape")

        return shape



    def get_extrema(self):

        """ Get maximum and minimum for all images in dataset"""

        # Get maximum and minimum
        maximum = np.max(np.array([image.tensor for image in self.data]))
        minimum = np.min(np.array([image.tensor for image in self.data]))

        return maximum, minimum

    def save_split(self):

        """ Save the splitting of data """

        with open(f"Split/{self.set_type}.yaml", "w") as file:

            split = {"files" : self.names}

            yaml.dump(split, file)

    


    
    
