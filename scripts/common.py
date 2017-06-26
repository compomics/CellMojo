"""
Contains common functionality that is used by different classes
(e.g. saving images, writing to files)

The idea of this class is to reduce importing the same packages over and over in different classes.
Methods will be added to this class as more code gets reformatted
"""

# imports
import os.path

import mahotas
import csv
import cv2
import scripts.preprocessing.preprocessing as preprocessing
import scripts.segmentation.segmentation as segmentation

try:
    import tkinter as tk  # for python 3
except:
    import Tkinter as tk  # for python 2

# take care of the path for different OS
os_path = str(os.path)
if 'posix' in os_path:
    import posixpath as path
elif 'nt' in os_path:
    import ntpath as path


def join_path(pathpart1, pathpart2):
    """
    Join the path parts using the intelligent joiner from the imported module
    """

    return path.join(pathpart1, pathpart2)


def concatenateList(root):
    if isinstance(root, (list, tuple)):
        for element in root:
            for e in concatenateList(element):
                yield e
    else:
        yield root


def save_image(pathpart1, pathpart2, imagefile):
    """
    Method for GUI display, save an image somewhere
    :param path: where to save the image
    :param imagefile: the actual image
    """

    joined_path = join_path(pathpart1, pathpart2)
    mahotas.imsave(joined_path, imagefile)


def read_image(pathpart1, pathpart2):
    """
    OpenCV method, read image from path
    """
    return cv2.imread(join_path(pathpart1, pathpart2))


def write_image(pathpart1, pathpart2, imagefile):
    """
    OpenCV method, write image to path
    """
    joined_path = join_path(pathpart1, pathpart2)
    cv2.imwrite(joined_path, imagefile)


def resize_image(imagefile, dimension):
    """
    OpenCV method, return resized image.
    :param dimension: requested dimension of the new image
    """
    return cv2.resize(imagefile, dimension, interpolation=cv2.INTER_AREA)


def tkinter_photoimage(filepath):
    """
    Tkinter method, return image
    """
    return tk.PhotoImage(file=filepath)


def csv_writer(openedfile):
    """
    Return a csv writer for a file
    """
    return csv.writer(openedfile, lineterminator='\n')


def call_preprocessing(image, preprocessingMethod):
    """
    Execute a preprocessing method on an image
    """

    if preprocessingMethod == 1:
        preprocessedImage = preprocessing.histEqualize(image)

    elif preprocessingMethod == 2:
        preprocessedImage = preprocessing.brightening(image)

    elif preprocessingMethod == 3:
        preprocessedImage = preprocessing.GaussianBlurring(image)

    elif preprocessingMethod == 4:
        preprocessedImage = preprocessing.darkening(image)

    elif preprocessingMethod == 5:
        preprocessedImage = preprocessing.denoising(image)

    elif preprocessingMethod == 6:
        preprocessedImage = preprocessing.binaryThresholding(image)

    elif preprocessingMethod == 8:
        preprocessedImage = preprocessing.sharpening(image)

    if preprocessingMethod == 7:

        preprocessedImage = image

    return preprocessedImage


def call_segmentation(segMeth, preImage, rawImg, minAreaSize, maxAreaSize, fixscale, minDistance, cellEstimate, color, thre):
    """ Call segmentation methods
    :param segMeth: segmentation methods
    :param processedImage1: input image to segment
    :param rawImg: raw image without preprocessing
    :param minAreaSize: estimated minimum area size of the cell
    :param maxAreaSize: estimated maximum area size of the cell
    :param fixscale: pixel intensity from 0.1-1.0
    :param minDistance: the minimum distance between the cells
    :param cellEstimate: minimum estimated number of cells per image
    :param color: color cell path """

    initialpoints, boxes, maskIMage, Image, CellMorph, processedImage = [], [], [], [], [], []
    processedImage = preImage

    if segMeth == 1:
        initialpoints, boxes, maskIMage, mage = segmentation.blob_seg(
            processedImage)

    if segMeth == 2:
        if color == 1:
            initialpoints, boxes, maskIMage, Image, CellMorph = segmentation.black_background(
                processedImage, rawImg, minAreaSize, maxAreaSize)

        if color == 2:
            initialpoints, boxes, maskIMage, Image, CellMorph = segmentation.white_background(
                processedImage, rawImg, minAreaSize, maxAreaSize)

    if segMeth == 3:
        initialpoints, boxes, Image = segmentation.harris_corner(
            processedImage, int(cellEstimate), float(fixscale), int(minDistance))

    if segMeth == 4:
        initialpoints, boxes, Image = segmentation.shi_tomasi(
            processedImage, int(cellEstimate), float(fixscale), int(minDistance))

    if segMeth == 5:
        initialpoints, boxes, maskIMage, Image, CellMorph = segmentation.kmeansSegment(
            processedImage, rawImg, 1, minAreaSize, maxAreaSize)

    if segMeth == 6:
        initialpoints, boxes, maskIMage, Image, CellMorph = segmentation.graphSegmentation(
            processedImage, rawImg, minAreaSize, minAreaSize, maxAreaSize)

    if segMeth == 7:
        initialpoints, boxes, maskIMage, Image, CellMorph = segmentation.meanshif(
            processedImage, rawImg, minAreaSize, maxAreaSize, int(fixscale * 100))

    if segMeth == 8:
        initialpoints, boxes, maskIMage, Image, CellMorph = segmentation.sheetSegment(
            processedImage, rawImg, minAreaSize, maxAreaSize)

    if segMeth == 9:
        initialpoints, boxes, maskIMage, Image, CellMorph = segmentation.findContour(
            processedImage, rawImg, minAreaSize, maxAreaSize)

    if segMeth == 10:
        initialpoints, boxes, maskIMage, Image, CellMorph = segmentation.threshold(
            processedImage, rawImg, minAreaSize, maxAreaSize)

    if segMeth == 11:
        initialpoints, boxes, maskIMage, Image, CellMorph = segmentation.overlapped_seg(
            processedImage, rawImg, minAreaSize, maxAreaSize)

    if segMeth == 12:
        initialpoints, boxes, maskIMage, Image, CellMorph = segmentation.gredientSeg(processedImage, rawImg,
                                                                                     minAreaSize, maxAreaSize, thre)

    return initialpoints, boxes, maskIMage, Image, CellMorph
