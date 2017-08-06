"""
Contains common functionality that is used by different classes
(e.g. saving images, writing to files)

The idea of this class is to reduce importing the same packages over and over in different classes.
Methods will be added to this class as more code gets reformatted
"""

# imports
import os.path
import sys
import mahotas
import csv
import cv2
sys.path.append('./preprocessing/')
sys.path.append('./segmentation/')
import preprocessing
import segmentation
import numpy as np
from itertools import groupby
from operator import itemgetter
import preprocessing.preprocessing as preprocessing
import segmentation.segmentation as segmentation
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

def drawStr(dst, target, s):
    """ Puts text onto an image
    :param dst: the image on which to put a string
    :param target: a tuple designating the image coordinates of the desired text
    :param s: the string to draw on the image
    """
    x, y = target
    cv2.putText(dst, s, (x + 1, y + 1), cv2.FONT_HERSHEY_PLAIN,
                1.0, (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN,
                1.0, (255, 255, 255), lineType=cv2.LINE_AA)


def resizeImage(image):
    """ Resize image to a desirable size for graph segmentation
    :param image: a grey level or rgb image
    :return resized: a resized image
    """
    if image.shape[0] or image.shape[1] > 500:
        r = 500.0 / image.shape[1]
        dimension = (500, int(image.shape[0] * r))

        resized = cv2.resize(image, dimension, interpolation=cv2.INTER_AREA)

    return resized


def missingFrames(frameList):
    """ Find missing framenumbers in a list
    :param frameList: a  list of frame numbers """
    first, last = frameList[0], frameList[-1]
    rangeSet = set(range(first, last + 1))
    return rangeSet - set(frameList)


def deleteLostTracks(trackNr, frameIdx, currentFrameIdx):
    """ Remove tracks that disappear in a few frames
    :param trackNr: the track label/number
    :param frameIdx: number of frames indices the track appear
    :param currentFrameIdx: the current frame ids
    :returns deleteTrack: the index of the track to be remove from the track list"""

    # set max number of missing frames and initiate return value in case no
    # tracks are to be deleted
    invisibleForTooLong, deleteTrack = 15, []  # no of frames

    # check if the current track is not featured in the current frame
    if currentFrameIdx not in frameIdx:
        # add the current frame to the frame array
        frameIdx = np.hstack([frameIdx, currentFrameIdx])
    # look up the indices of the frames still missing
    missingFrameIdx = missingFrames(frameIdx)
    invisible = [int(i) for i in sorted(missingFrameIdx)]
    print invisible

    # find a missing number in a subset
    for k, g in groupby(enumerate(invisible), (lambda (i, x):(i - x))):
        seq = map(itemgetter(1), g)
        if len(seq) > invisibleForTooLong:
            deleteTrack = trackNr
            print deleteTrack

    return deleteTrack


def displayCoordinates(self,objectLabel, a, b, tm):
    """ Display coordinates of the cell to the panel"""

    if objectLabel == 0:
        self.label_10.configure(text=int(tm))
        self.label_43.configure(text=a)
        self.label_63.configure(text=b)
    if objectLabel == 1:
        self.label_11.configure(text=int(tm))
        self.label_44.configure(text=a)
        self.label_64.configure(text=b)

    if objectLabel == 2:
        self.label_12.configure(text=int(tm))
        self.label_45.configure(text=a)
        self.label_65.configure(text=b)

    if objectLabel == 3:
        self.label_13.configure(text=int(tm))
        self.label_46.configure(text=a)
        self.label_66.configure(text=b)

    if objectLabel == 4:
        self.label_14.configure(text=int(tm))
        self.label_47.configure(text=a)
        self.label_67.configure(text=b)

    if objectLabel == 5:
        self.label_15.configure(text=int(tm))
        self.label_48.configure(text=a)
        self.label_68.configure(text=b)
    if objectLabel == 6:
        self.label_16.configure(text=int(tm))
        self.label_49.configure(text=a)
        self.label_69.configure(text=b)
    if objectLabel == 7:
        self.label_17.configure(text=int(tm))
        self.label_50.configure(text=a)
        self.label_70.configure(text=b)

    if objectLabel == 8:
        self.label_18.configure(text=int(tm))
        self.label_51.configure(text=a)
        self.label_71.configure(text=b)

    if objectLabel == 9:
        self.label_19.configure(text=int(tm))
        self.label_52.configure(text=a)
        self.label_72.configure(text=b)


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
