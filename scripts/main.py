# menu.py
# -*- coding: utf-8 -*-


# import operator
import os
# import shutil
# import sys
import time
# import zipfile
from itertools import groupby
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import scipy
# from pylab import tk
from scipy.ndimage import gaussian_filter1d
from sklearn.neighbors import NearestNeighbors

import application
import call_back_preprocessing
import call_back_segmentation
import cv2
import extra_modules

# import pymeanshift as pms
# import report
# from libtiff import TIFF

try:
    import tkinter as tk  # for python 3
except:
    import Tkinter as tk  # for python 2


current_time = time.localtime()
current_time = time.strftime('%a, %d %b %Y %H:%M:%S GMT', current_time)
global prev_image, thre


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

    # find a missing number in a subset
    for k, g in groupby(enumerate(invisible), (lambda i, x: i - x)):
        seq = map(itemgetter(1), g)
        if len(seq) > invisibleForTooLong:
            deleteTrack = trackNr

    return deleteTrack


def displayCoordinates(self, objectLabel, a, b, tm):
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


if __name__ == '__main__':
    root = tk.Tk()
    menu = tk.Menu(root)
    root.config(menu=menu)

    file = tk.Menu(menu)
    file.add_command(label='Exit', command='')

    menu.add_cascade(label='File', menu=file)

    edit = tk.Menu(menu)

    # adds a command to the menu option, calling it exit, and the
    # command it runs on event is client_exit

    # added "file" to our menu
    menu.add_command(label="Open", command='')
    menu.add_command(label="Save", command='')
    menu.add_separator()
    menu.add_cascade(label="Edit", menu=edit)

    # root.geometry('1500x1000')
    # img = Image("photo", file="multimot2.ico")
    # root.tk.call('wm', 'iconphoto', root._w, img)
    root.title("CellMojo: Cell Segmentation and Tracking")
    app = application.Application(root)
    root.mainloop()
