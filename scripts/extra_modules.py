
import os
import re

import numpy as np

import cv2


def sortFiles(ListImages):
    ListImages.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    return ListImages


def make_video(images, outvid=None, fps=5, size=None,
               is_color=True, format="XVID"):
    """
    Create a video from a list of images.

    @param      outvid      output video
    @param      images      list of images to use in the video
    @param      fps         frame per second
    @param      size        size of each frame
    @param      is_color    color
    @param      format      see http://www.fourcc.org/codecs.php
    @return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

    The function relies on http://opencv-python-tutroals.readthedocs.org/en/latest/.
    By default, the video will have the size of the first image.
    It will resize every image to this size before adding them to the video.
    """
    images = sortFiles(images)
    print(images)
    fourcc = cv2.VideoWriter_fourcc(*format)
    vid = None
    for image in images:
        if not os.path.exists(image):
            continue
            # raise FileNotFoundError(image)

        img = cv2.imread(image)

        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = cv2.VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = cv2.resize(img, size)
        vid.write(img)
    vid.release()
    return vid


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def find_if_similiar(cnt1, cnt2):

    row1, row2 = cnt1.shape[0], cnt2.shape[0]
    for i in range(row1):
        for j in range(row2):
            similarityDistance = np.linalg.norm(cnt1[i]-cnt2[j])
            if abs(similarityDistance) < 15:
                return True
            elif i == row1 - 1 and j == row2 - 1:
                return False


# find contour
def duplicationRemoval(contours, minAreaSize, maxAreaSize):

    numContour = len(contours)
    status = np.zeros((numContour, 1))

    for i, cnt1 in enumerate(contours):
        x = i
        if i != numContour - 1:
            for j, cnt2 in enumerate(contours[i + 1:]):
                x = x + 1
                similarityValue = find_if_similiar(cnt1, cnt2)
                if similarityValue is True:
                    minValue = min(status[i], status[x])
                    status[x] = status[i] = minValue
                else:
                    if status[x] == status[i]:
                        status[x] = i + 1

    refinedCont = []
    maxValue = int(status.max()) + 1

    for i in range(maxValue):
        pos = np.where(status == i)[0]
        if pos.size != 0:
            cont = np.vstack(contours[i] for i in pos)
            hull = cv2.convexHull(cont)
            refinedCont.append(hull)

    pts, conts = [], []
    for (ii, cnt) in enumerate(refinedCont):
        # get the centroid of each detected contour
        ((x, y), _) = cv2.minEnclosingCircle(cnt)
        area = cv2.contourArea(cnt)
        if area < minAreaSize or area > maxAreaSize:
            del contours[ii]
            continue
        if ii == 0:
            pts = np.hstack(np.float32([[x, y]]))
            conts.append(cnt)
        else:
            centre = np.hstack(np.float32([[x, y]]))
            pts = np.vstack([pts, centre])
            conts.append(cnt)

    return pts, refinedCont
