import numpy as np
import cv2
import pyclesperanto_prototype as cle


def otsu_voronoi(image):

    """Segment image using Voronoi-Otsu"""

    # Segment image using pyclesperanto implementation of Voronoi-Otsu algorithm
    # Parameter might need fixing depending on the case at hand
    segmented = cle.voronoi_otsu_labeling(image, spot_sigma=10, outline_sigma=1)

    # Ensure the segmented image is NumPy array
    segmented = np.array(segmented)

    # Ensure the image is consistent with data-type of masks
    segmented = segmented.astype("uint8")

    # Define kernel to be used for dilation
    kernel = np.ones((3, 3), np.uint8)

    # Dilate masks so as to fill possible gaps
    segmented = cv2.dilate(segmented, kernel, iterations=3)

    return segmented
