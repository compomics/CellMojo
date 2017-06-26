import numpy as np
from pylab import arange, array, uint8

import cv2


# import ImageEnhance (currently not used)
# from PIL import Image


def brightening(image):
    """Brightening the image by increasing pixel intensity"""
    # Parameters for manipulating image data
    phi, theta, maxIntensity = 1, 1, 255.0

    brighten_image = (maxIntensity / phi) * (image / (maxIntensity / theta)) ** 2.5
    brighten_image = array(brighten_image, dtype=uint8)

    return brighten_image


def darkening(image):
    """ Darkening image intensity """

    # Parameters for manipulating image data
    phi, theta, maxIntensity = 1, 1, 255.0

    darkening_image = (maxIntensity / phi) * (image / (maxIntensity / theta)) ** 2
    out = array(darkening_image, dtype=uint8)

    return out


def denoising(image):
    """improve image quality by remove unimportant details"""
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    return denoised


def histEqualize(image):
    """increase image contrast"""
    if len(image.shape) < 3:
        equalizedImage = cv2.equalizeHist(image)
        improvedIMg = cv2.cvtColor(equalizedImage, cv2.COLOR_GRAY2BGR)
    else:
        equalizedImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        equalizedImage = cv2.equalizeHist(equalizedImage)
        improvedIMg = cv2.cvtColor(equalizedImage, cv2.COLOR_GRAY2BGR)
    return improvedIMg


def GaussianBlurring(image):
    """ blur image to remove noisy info, edges might be blurred"""
    image = cv2.GaussianBlur(image, (3, 3), 0, 1, 0)
    return image


def binaryThresholding(image):
    """ threshold image based on the intensity color """
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, thresholded = cv2.threshold(image, 0, 255,
                                       cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    else:
        _, thresholded = cv2.threshold(image, 0, 255,
                                       cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return thresholded


def sharpening(image):
    """ sharpen image edges using different kernel"""

    # generating the kernels
    kernel_sharpen = np.array([[-1, -1, -1, -1, -1],
                               [-1, 2, 2, 2, -1],
                               [-1, 2, 8, 2, -1],
                               [-1, 2, 2, 2, -1],
                               [-1, -1, -1, -1, -1]]) / 5.0

    output_3 = cv2.filter2D(image, -1, kernel_sharpen)
    return output_3
