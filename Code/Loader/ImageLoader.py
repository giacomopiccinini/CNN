import numpy as np
import multiprocessing as mp
from tensorflow.keras.utils import Sequence
from glob import glob
from Code.Classes.Image import Image

class ImageLoader(Sequence):
    
    """ Read images from path, storing them in Image class"""
    
    def __init__(self, directory, batch_size, extension="png", maximum=None, minimum=None):
        
        """ Constructor for ImageLoader class"""

        # Find path of all images
        paths = glob(f"{directory}/*.{extension}")

        # Initiate pool for multiprocessing (for speeding up reading of images)
        pool = mp.Pool(mp.cpu_count())

        # Load images
        data = mp.map(Image.from_path, paths)

        # Close pool
        pool.close()

        # Store details
        self.paths      = paths
        self.batch_size = batch_size
        self.data       = data
        self.number     = len(paths)

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

    def get_extrema(self):

        """ Get maximum and minimum for all images in dataset"""

        # Get maximum and minimum
        maximum = np.max(np.array(self.data))
        minimum = np.min(np.array(self.data))

        return maximum, minimum


    
    
