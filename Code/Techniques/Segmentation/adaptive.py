import cv2


def adaptive_threshold(
    tensor,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
    thresholdType=cv2.THRESH_BINARY_INV,
    blockSize=23,
    C=6,
):

    """Apply adaptive thresholding to image.

    maxValue: Value to be set to part that are segmented (set it to one if [0,1] output is needed)

    adaptiveMethod: Method to be used when performing thresholding. Options: [cv2.ADAPTIVE_THRESH_MEAN_C, cv2.ADAPTIVE_THRESH_GAUSSIAN_C]

    thresholdType: How thresholding is performed, e.g. if larger than threshold set to maxValue. Options: [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV]

    blockSize: 	Size of a pixel neighborhood that is used to calculate a threshold value for the pixel. Options: integer

    C: Constant subtracted from the mean or weighted mean. Normally, it is positive but may be zero or negative as well."""

    thresholded = cv2.adaptiveThreshold(
        tensor,
        maxValue=1,
        adaptiveMethod=adaptiveMethod,
        thresholdType=thresholdType,
        blockSize=blockSize,
        C=C,
    )

    return thresholded
