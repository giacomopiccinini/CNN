import numpy as np
import multiprocessing as mp
from tensorflow.keras.utils import Sequence
from glob import glob
from Code.Classes.Image import Image

class ImageLoader(Sequence):
    
    """ Read images from path, storing them in Image class"""
    
    def __init__(self, directory, batch_size, extension="png"):
        
        """ Constructor for ImageLoader class"""

        # Find path of all images
        images = glob(f"{directory}/*.{extension}")

        # Initiate pool for multiprocessing (for speeding up reading of images)
        pool = mp.Pool(mp.cpu_count())

        # Load images
        data = mp.map(Image.from_path, images)

        self.directory = directory
        
        self.filenames     = filenames
        self.batch_size    = batch_size
        self.is_grayscale  = is_grayscale
        self.minimum       = minimum
        self.maximum       = maximum

    def __len__(self):
        
        """ Return the number of batches needed for completing the dataset """
        
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))

    def __getitem__(self, i):
        
        """ Return the images in the i-th batch """
        
        batch = self.filenames[i*self.batch_size : (i + 1)*self.batch_size]
        
        batch = np.array([read_normalized_image(filename, self.is_grayscale, self.minimum, self.maximum) for filename in batch])
        
        return batch, batch  
    
    
