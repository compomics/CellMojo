from __future__ import division, print_function

from math import sqrt

import numpy as np
from scipy import ndimage
from skimage import exposure
from skimage.color import rgb2gray
from skimage.feature import blob_log, peak_local_max
from skimage.filter import rank
from skimage.morphology import disk, watershed
from skimage.util import img_as_ubyte
from numpy import NaN
import cv2
import eGraphBasedSegment
import extra_modules
import pymeanshift as pms

image, sm_img, raw_image, raw_im, im = [], [], [], [], []


def mergeOverSegmentedContours(contours, targetCont):
    simValues, cnts, cntsIndices = [], [], []
    for i, cnt1 in enumerate(contours):
        ((xTarget, yTarget), radius) = cv2.minEnclosingCircle(targetCont)
        ((xCnt1, yCnt1), radius) = cv2.minEnclosingCircle(cnt1)
        similarityValue = np.sqrt(np.square(xCnt1 - xTarget) + np.square(yCnt1 - yTarget))

        if similarityValue < 0.0 and float(similarityValue) != 0.0:
            print(float(similarityValue))
            simValues.append([similarityValue])
            cnts.append(cnt1)
            cntsIndices.append(i)

    if simValues:
        minDistance = min(simValues)
        index = simValues.index(minDistance)
        tmp_cnt = cnts[index]
        cntID = cntsIndices[index]
        return cntID, np.vstack([tmp_cnt, targetCont])
    else:
        return None, None


def getCellIntensityModule(Objectcontour, image):
    morph_properties = []

    if len(image.shape) != 2:
        image = cv2.cvtColor(image,  cv2.COLOR_RGB2GRAY)

    mask = np.zeros_like(image)
    area = cv2.contourArea(Objectcontour)
    rect = cv2.boundingRect(Objectcontour)
    hull = cv2.convexHull(Objectcontour)
    hullArea = cv2.contourArea(hull)
    perimeter = cv2.arcLength(Objectcontour, True)
    (x1, y1, w1, h1) = rect

    if len(Objectcontour) > 4:
        (_, _), axisSize, _ = cv2.fitEllipse(Objectcontour)
        minorAxisLengthEllipse = min(axisSize)
        majorAxisLengthEllipse = max(axisSize)
        eccentricityEllipse = np.sqrt(1 - np.square(minorAxisLengthEllipse / majorAxisLengthEllipse))
    else:
        majorAxisLengthEllipse, minorAxisLengthEllipse = NaN, NaN
        eccentricityEllipse = NaN

    moments = cv2.moments(Objectcontour)

    majorAxisLengthMoment = np.sqrt(0.5 * (moments['mu20'] + moments['mu02']) + np.sqrt(4 * np.square(moments['mu11']) - np.square(moments['mu20'] - moments['mu02'])))
    minorAxisLengthMoment = np.sqrt(0.5 * (moments['mu20'] + moments['mu02']) - np.sqrt(4 * np.square(moments['mu11']) - np.square(moments['mu20'] - moments['mu02'])))

    aspect_ratio = float(w1) / h1
    rectAarea = w1 * h1
    areaPerimeterRatio = float(area) / float(perimeter)
    extent = float(area) / rectAarea
    solidity = float(area) / hullArea
    equivalentCellDiameter = np.sqrt(4 * area / np.pi)
    circularity = (4 * np.pi * area) / np.square(perimeter)  # check
    roundnessEllipse = (4 * area) / (np.pi * np.square(majorAxisLengthEllipse))
    roundnessMoment = (4 * area) / (np.pi * np.square(majorAxisLengthMoment))
    eccentricityMoment = np.sqrt(1 - np.square(minorAxisLengthMoment / majorAxisLengthMoment))

    cv2.drawContours(mask, [Objectcontour], -1, 255, -1)
    contourPixels = np.transpose(np.where(mask == 255))
    integratedIntensity = np.sum(contourPixels)
    MeanIntensity = np.average(contourPixels)
    StdIntensity = np.std(contourPixels)
    maxIntensity = np.max(contourPixels)
    minIntensity = np.min(contourPixels)

    morph_properties.append([aspect_ratio, extent, solidity, equivalentCellDiameter,  integratedIntensity, MeanIntensity, StdIntensity, maxIntensity, minIntensity, majorAxisLengthEllipse, minorAxisLengthEllipse, area, hullArea,
                        perimeter, eccentricityEllipse, roundnessEllipse,  circularity, areaPerimeterRatio, majorAxisLengthMoment, minorAxisLengthMoment, eccentricityMoment, roundnessMoment])

    return morph_properties


def blob_seg(image):
    """ identify blob in images"""

    if len(image.shape) == 3:
        image_gray = rgb2gray(image)

    blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.1)

    # Compute radii in the 3rd column.
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

    color = np.random.randint(0, 255, (5000, 3))
    mask_colored = np.zeros_like(image, )
    pts, rectang = [], []
    for ii, blob in enumerate(blobs_log):
        y, x, stddev = blob
        cv2.circle(mask_colored, (int(x), int(y)), int(stddev), color[ii].tolist(), thickness=cv2.FILLED)
        cv2.circle(image, (int(x), int(y)), int(stddev), (0, 255, 0), 1)
        rectang.append((x, y, 3, 3))
        pts.append([x, y])

    return pts, rectang, mask_colored, image


def graphSegmentation(sm_img, raw_im, minSize, minAreaSize, maxAreaSize):
    """ Use efficient graph based segmentation methods for segmenting and contour analysis"""

    if sm_img.shape[0] or sm_img.shape[1] > 500:
        r = 500.0 / sm_img.shape[1]
        dim = (500, int(sm_img.shape[0] * r))

        sm_img = cv2.resize(sm_img, dim, interpolation=cv2.INTER_AREA)

        r = 500.0 / raw_im.shape[1]
        dim = (500, int(raw_im.shape[0] * r))

        raw_im = cv2.resize(raw_im, dim, interpolation=cv2.INTER_AREA)

    if len(raw_im.shape) > 2:
        gray2 = cv2.cvtColor(raw_im, cv2.COLOR_RGB2GRAY)
    else:
        gray2 = raw_im

    # normalize the image

    # sm_img = rank.median(sm_img, disk(2))
    image = exposure.equalize_hist(sm_img)
    # check if the image is 3 channels
    if len(sm_img.shape) > 2:
        height, width, d = sm_img.shape
    else:
        height, width = sm_img.shape

    Imgthreshold = 1900  # threshold1200
    Size = int(int(minSize))
    egbs = eGraphBasedSegment(width, height, threshold=Imgthreshold, minSize=Size)
    egbs.segmentImage(image)
    egbs.mergeSmall()

    labels, edges = egbs.getSegmentEdges()

    mask = np.zeros_like(gray2,)

    for label in range(len(unique(labels))):
        if label == 0:
            continue
        mask[labels == label] = 255

    mask = cv2.bitwise_and(gray2, mask)
    contours = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]

    pts, conts, rectang, cellFeatures = [], [], [], []
    color = np.random.randint(0, 255, (5000, 3))
    mask_colored = np.zeros_like(raw_im,)
    for (ii, cnt) in enumerate(contours):
        # get the centroid of each detected contour
        ((x, y), _) = cv2.minEnclosingCircle(cnt)
        area = cv2.contourArea(cnt)
        rect = cv2.boundingRect(cnt)
        (x1, y1, w1, h1) = rect
        if area < minAreaSize or area > maxAreaSize:
            del contours[ii]
            continue
        else:
            cellMorph = getCellIntensityModule(cnt, raw_im)
            cellFeatures.append(cellMorph)
            cv2.drawContours(mask_colored, [cnt], -1, color[ii].tolist(), thickness=cv2.FILLED)
            cv2.drawContours(raw_im, [cnt], -1, (0, 255, 0), 1)
            rectang.append((x1, y1, w1, h1))
            pts.append([x, y])

    return pts, rectang, mask_colored, raw_im, cellFeatures


def meanshif(image, raw_image, minAreaSize, maxAreaSize, minDensity):
    """ perform segmentation using meanshift modules. THIS REQUIRES
        https://github.com/fjean/pymeanshift"""

    if (len(image.shape) > 3):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mask = np.zeros_like(image, )
    (segmented_image, labels_image, number_regions) = pms.segment(image, spatial_radius=6,
                                                                  range_radius=4.5, min_density=int(minDensity))

    # marked indeentifed objects
    for label in range(len(np.unique(labels_image))):
        if label == 0:
            continue
        mask[labels_image == label] = 255
    mask = cv2.bitwise_and(image, mask)

    if len(mask.shape) > 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    _, contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    color = np.random.randint(0, 255, (5000, 3))
    mask_colored = np.zeros_like(raw_image, )
    pts, rectang, cellFeatures = [], [], []
    for (ii, cnt) in enumerate(contours):
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)

        area = cv2.contourArea(cnt)
        rect = cv2.boundingRect(cnt)
        (x1, y1, w1, h1) = rect

        if area < minAreaSize or area > maxAreaSize or hierarchy[0, ii, 3] != -1:
            del contours[ii]
        else:
            cellMorph = getCellIntensityModule(cnt, image)
            cellFeatures.append(cellMorph)
            cv2.drawContours(mask_colored, [cnt], -1, color[ii].tolist(), thickness=cv2.FILLED)
            cv2.drawContours(raw_image, [cnt], -1, (0, 255, 0), 1)
            rectang.append((x1, y1, w1, h1))
            pts.append([x, y])

    return pts, rectang, mask_colored, image, cellFeatures


def black_background(image, raw_image, minAreaSize, maxAreaSize):
    """ Watershed segmentation for image with white/gray background """
    raw_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2GRAY)
    im = cv2.threshold(image, 173, 255, cv2.THRESH_BINARY)
    im = im[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilation = cv2.dilate(im, kernel, iterations=1)
    gradient = cv2.morphologyEx(dilation, cv2.MORPH_GRADIENT, kernel)
    closing = cv2.morphologyEx(gradient, cv2.MORPH_CLOSE, kernel)

    shifted = cv2.pyrMeanShiftFiltering(closing, 10, 20)
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=10,
                              labels=thresh)

    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)
    # create a mask
    mask = np.zeros(raw_image.shape, dtype="uint8")
    #  loop over the unique labels returned by the Watershed  algorithm for
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background' so simply ignore it
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask[labels == label] = 255
        # close gaps

    mask = cv2.bitwise_and(raw_image, mask)
    _, contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    color = np.random.randint(0, 255, (5000, 3))
    mask_colored = np.zeros_like(image,)
    pts, rectang, cellFeatures = [], [], []

    for (ii, cnt) in enumerate(contours):
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)

        area = cv2.contourArea(cnt)
        rect = cv2.boundingRect(cnt)
        (x1, y1, w1, h1) = rect

        if area < minAreaSize or area > maxAreaSize or hierarchy[0, ii, 3] != -1:
            del contours[ii]
        else:
            cellMorph = getCellIntensityModule(cnt, image)
            cellFeatures.append(cellMorph)
            cv2.drawContours(mask_colored, [cnt], -1, color[ii].tolist(), thickness=cv2.FILLED)
            cv2.drawContours(image, [cnt], -1, (0, 255, 0), 1)
            rectang.append((x1, y1, w1, h1))
            pts.append([x, y])

    return pts, rectang, mask_colored, image, cellFeatures


def kmeansSegment(image, raw_img, bestLabel, minAreaSize, maxAreaSize):

    pixel_list = image.reshape(-1, 3)
    pixel_list = np.float32(pixel_list)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    _, label, centeroid = cv2.kmeans(pixel_list, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    result = centeroid[label.flatten()]
    # segmented_image = result.reshape((image.shape))
    # segmented_image = segmented_image.astype(np.uint8)
    image_labels = label.reshape((image.shape[0], image.shape[1]))

    num_cluster = np.zeros(raw_img.shape, np.uint8)
    num_cluster[image_labels == bestLabel] = image[image_labels == bestLabel]

    num_cluster = cv2.bitwise_and(raw_img, num_cluster)
    num_cluster = cv2.cvtColor(num_cluster, cv2.COLOR_RGB2GRAY)
    _, contours, hierarchy = cv2.findContours(num_cluster.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    color = np.random.randint(0, 255, (len(contours), 4))
    mask_colored = np.zeros_like(raw_img, )
    pts, rectang, cellFeatures = [], [], []

    for (ii, cnt) in enumerate(contours):

        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        area = cv2.contourArea(cnt)
        rect = cv2.boundingRect(cnt)
        (x1, y1, w1, h1) = rect

        if area < minAreaSize or area > maxAreaSize or hierarchy[0, ii, 3] != -1:
            del contours[ii]
            continue
        else:
            cellMorph = getCellIntensityModule(cnt, image)
            cellFeatures.append(cellMorph)
            cv2.drawContours(mask_colored, [cnt], -1, color[ii].tolist(), thickness=cv2.FILLED)
            cv2.drawContours(image, [cnt], -1, (0, 255, 0), 1)
            rectang.append((x1, y1, w1, h1))
            pts.append([x, y])

    return pts, rectang, mask_colored, image, cellFeatures


def white_backgroundMof(image, raw_image, minAreaSize, maxAreaSize):
    """ use watershed to to segment image with black background"""

    # parameters
    THRESHOLD = 55
    # convert to gray image
    image = image.mean(axis=-1)
    # find peaks
    peak = peak_local_max(image, threshold_rel=0.5, min_distance=17)
    # make an image with peaks at 1
    peak_im = np.zeros_like(image)
    for p in peak:
        peak_im[p[0], p[1]] = 1
    # label peaks
    peak_label, _ = ndimage.label(peak_im)
    # propagate peak labels with watershed
    labels = watershed(255 - image, peak_label)
    # limit watershed labels to area where the image is intense enough
    results = labels * (image > THRESHOLD)
    mask = np.zeros(image.shape, dtype="uint8")

    for result in np.unique(results):

        # if the label is zero, we are examining the 'background' so simply ignore it
        if result == 0:
            continue

        # otherwise, allocate memory for the label region and draw it on the mask
        mask[results == result] = 255
    mask = cv2.erode(mask, None, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    res = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Copy the thresholded image.
    im_floodfill = mask.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = mask.shape[:2]
    mask2 = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask2, (0, 0), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = mask | im_floodfill_inv

    # if len(image.shape) < 2:
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # else:
    #     gray = image
    #
    # thresh = cv2.threshold(gray, 0, 255,
    #                        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # thresh = cv2.dilate(cv2.erode(thresh, kernel), kernel)
    #
    # distance = ndi.distance_transform_edt(thresh)
    # local_maxi = peak_local_max(distance, indices=False, min_distance=10,
    #                             labels=gray)
    # # use markers to avoid over-segmentation
    # markers = ndi.label(local_maxi, structure=np.ones((3, 3)))[0]
    # labels = watershed(-distance, markers, mask=thresh)
    #
    # mask = np.zeros_like(gray, )
    #
    # #  loop over the unique labels returned by the Watershed  algorithm for
    # for label in np.unique(labels):
    #     # if the label is zero, we are examining the 'background' so simply ignore it
    #     if label == 0:
    #         continue
    #     # otherwise, allocate memory for the label region and draw
    #     # it on the mask
    #     mask[labels == label] = 255
    # # close gaps
    # mask = cv2.dilate(cv2.erode(mask, kernel), kernel)
    #
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # mask = cv2.bitwise_and(gray, mask)
    # find contours in the the mask
    _, contours, hierarchy = cv2.findContours(im_out.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    color = np.random.randint(0, 255, (len(contours), 4))
    mask_colored = np.zeros_like(raw_image, )
    pts, rectang, cellFeatures = [], [], []

    for (ii, cnt) in enumerate(contours):
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)

        area = cv2.contourArea(cnt)
        rect = cv2.boundingRect(cnt)
        (x1, y1, w1, h1) = rect

        if area < minAreaSize or area > maxAreaSize or hierarchy[0, ii, 3] != -1:
            del contours[ii]
        else:
            cellMorph = getCellIntensityModule(cnt, image)
            cellFeatures.append(cellMorph)
            cv2.drawContours(mask_colored, [cnt], -1, color[ii].tolist(), thickness=cv2.FILLED)
            cv2.drawContours(raw_image, [cnt], -1, (0, 255, 0), 1)
            rectang.append((x1, y1, w1, h1))
            pts.append([x, y])

    return pts, rectang, mask_colored, raw_image, cellFeatures


def sheetSegment(image, raw_image, minAreaSize, maxAreaSize):
    """ use this method to segment cell collectively"""

    # image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    overlayedMask = cv2.cvtColor(raw_image, cv2.COLOR_RGB2GRAY)
    shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    """remove small objects like dots"""
    cv2.imshow('sss', shifted)

    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)  # 4 INSTEAD OF 2

    """enclose area of the contour"""
    kernel = np.ones((7, 7), dtype='uint8')
    image_close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    image_close = cv2.bitwise_and(overlayedMask, image_close)
    contours = cv2.findContours(image_close.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    color = np.random.randint(0, 255, (len(contours), 4))
    mask_colored = np.zeros_like(overlayedMask,)
    # loop over the contours
    pts, rectang, cellFeatures = [], [], []

    for (ii, cnt) in enumerate(contours):
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)

        area = cv2.contourArea(cnt)
        rect = cv2.boundingRect(cnt)
        (x1, y1, w1, h1) = rect

        if area < minAreaSize or area > maxAreaSize:
            del contours[ii]
            continue
        else:
            cellMorph = getCellIntensityModule(cnt, image)
            cellFeatures.append(cellMorph)
            cv2.drawContours(mask_colored, [cnt], -1, color[ii].tolist(), thickness=cv2.FILLED)
            cv2.drawContours(image, [cnt], -1, (0, 255, 0), 1)
            rectang.append((x1, y1, w1, h1))
            pts.append([x, y])

    return pts, rectang, mask_colored, image, cellFeatures


def overlapped_seg(im, raw_image, minAreaSize, maxAreaSize):
    """ segment images using HSV color system"""
    overlayedMask = cv2.cvtColor(raw_image, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    th, bw = cv2.threshold(hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    dist = cv2.distanceTransform(morph, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    borderSize = 15
    distBorder = cv2.copyMakeBorder(dist, borderSize, borderSize, borderSize, borderSize,
                                    cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
    gap = 5
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (borderSize - gap) + 1, 2 * (borderSize - gap) + 1))
    kernel2 = cv2.copyMakeBorder(kernel2, gap, gap, gap, gap,
                                 cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
    distTempl = cv2.distanceTransform(kernel2, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    nCorrelation = cv2.matchTemplate(distBorder, distTempl, cv2.TM_CCOEFF_NORMED)
    mn, mx, _, _ = cv2.minMaxLoc(nCorrelation)
    th, peaks = cv2.threshold(nCorrelation, mx * 0.1, 255, cv2.THRESH_BINARY)
    finalPeaks = cv2.convertScaleAbs(peaks)
    finalPeaks = cv2.bitwise_and(overlayedMask, finalPeaks)
    _, contours, hierarchy = cv2.findContours(finalPeaks, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    color = np.random.randint(0, 255, (len(contours), 4))
    mask_colored = np.zeros_like(im,)
    pts, conts, rectang, cellFeatures = [], [], [], []
    for (ii, cnt) in enumerate(contours):
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        area = cv2.contourArea(cnt)
        rect = cv2.boundingRect(cnt)
        (x1, y1, w1, h1) = rect
        x, y, w, h = cv2.boundingRect(cnt)
        if area < minAreaSize or area > maxAreaSize or hierarchy[0, ii, 3] != -1:
            continue
        else:
            _, mx, _, mxloc = cv2.minMaxLoc(dist[y:y + h, x:x + w], finalPeaks[y:y + h, x:x + w])
            cv2.circle(im, (int(mxloc[0] + x), int(mxloc[1] + y)), int(mx), (0, 255, 0), 1)
            cv2.circle(mask_colored, (int(mxloc[0] + x), int(mxloc[1] + y)), int(mx), color[ii].tolist(),
                       thickness=cv2.FILLED)
            cellMorph = getCellIntensityModule(cnt, im)
            cellFeatures.append(cellMorph)
            rectang.append((x1, y1, w1, h1))
            pts.append([x, y])

    return pts, rectang, mask_colored, im, cellFeatures


def findContour(image, raw_image, minAreaSize, maxAreaSize):

    # Check image dimension
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 3.Thresholds
    flag, tmp_img = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    # 4.Erodes the Thresholded Image
    Kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    newImg = cv2.erode(tmp_img, Kernel)
    newImg = cv2.bitwise_and(image, newImg)
    contours = cv2.findContours(newImg.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]

    color = np.random.randint(0, 255, (len(contours), 4))
    mask_colored = np.zeros_like(raw_image, )
    pts, rectang, cellFeatures = [], [], []

    for (ii, cnt) in enumerate(contours):
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        area = cv2.contourArea(cnt)
        rect = cv2.boundingRect(cnt)
        (x1, y1, w1, h1) = rect
        if area < minAreaSize or area > maxAreaSize:
            del contours[ii]
            continue
        else:
            cellMorph = getCellIntensityModule(cnt, raw_image)
            cellFeatures.append(cellMorph)
            cv2.drawContours(mask_colored, [cnt], -1, color[ii].tolist(), thickness=cv2.FILLED)
            cv2.drawContours(raw_image, [cnt], -1, (0, 255, 0), 1)
            rectang.append((x1, y1, w1, h1))
            pts.append([x, y])

    return pts, rectang, mask_colored, raw_image, cellFeatures


def shi_tomasi(image, maxCorner, qualityLevel, MinDistance):
    # detect corners in the image
    corners, rectang = [], []
    if len(image.shape) == 3:
        img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        img = cv2.equalizeHist(img)
    else:
        img = image

    corners = cv2.goodFeaturesToTrack(img,
                                      maxCorner,
                                      qualityLevel,
                                      MinDistance,
                                      mask=None,
                                      blockSize=7)

    if corners.any():
        for corner in corners:
            x, y = corner[0]
            cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)
            rectang.append((x, y, 2, 3))

    return corners, rectang, image


def harris_corner(image, maxCorner, qualityLevel, minDistance):
    corners, rectang = [], []
    if len(image.shape) == 3:
        img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        img = cv2.equalizeHist(img)
    else:
        img = image

    corners = cv2.goodFeaturesToTrack(img,  # img
                                      maxCorner,  # maxCorners
                                      qualityLevel,  # qualityLevel
                                      minDistance,  # minDistance
                                      None,  # corners,
                                      None,  # mask,
                                      7,  # blockSize,
                                      useHarrisDetector=True,  # useHarrisDetector,
                                      k=0.05  # k
                                      )

    if corners.any():
        for corner in corners:
            x, y = corner[0]
            cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)
            rectang.append((x, y, 2, 3))

    return corners, rectang, image


def threshold(image, raw_image, minAreaSize, maxAreaSize):
    """
    :param image: a greyscale image
    :param raw_image: and rgb image
    :param minAreaSize: the minimum area size of the desired cell
    :param maxAreaSize: the maximum area size of the desired cell
    :returns cell centroids, bounding box, mask, segmented cells, cell morphology
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    eroding = cv2.erode(opening, kernel)
    eroding = cv2.bitwise_and(gray, eroding)
    _, contours, hierarchy = cv2.findContours(eroding.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    color = np.random.randint(0, 255, (len(contours), 4))
    mask_colored = np.zeros_like(gray, )
    pts, rectang, cellFeatures = [], [], []

    for (ii, cnt) in enumerate(contours):
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        centre = (x, y)
        area = cv2.contourArea(cnt)
        rect = cv2.boundingRect(cnt)
        (x1, y1, w1, h1) = rect

        if area < minAreaSize or area > maxAreaSize or hierarchy[0, ii, 3] != -1:
            del contours[ii]
            continue
        else:
            cellMorph = getCellIntensityModule(cnt, raw_image)
            cellFeatures.append(cellMorph)
            cv2.drawContours(mask_colored, [cnt], -1, color[ii].tolist(), thickness=cv2.FILLED)
            cv2.drawContours(raw_image, [cnt], -1, (0, 255, 0), 1)
            rectang.append((x1, y1, w1, h1))
            pts.append([x, y])

    return pts, rectang, mask_colored, gray, cellFeatures


def gredientSeg(preprocessedImage1, raw_image, minAreaSize, maxAreaSize, thre):
    """
    :param preprocessedImage: a greyscale image
    :param raw_image: a raw image for highlighting segmented cells
    :param minAreaSize: the minimum area size of the cell
    :param maxAreaSize: maximum area size of the cell

    :return:
    """
    preprocessedImage, denoised, gradientIm, thresh, imgDisplay, contours = [], [], [], [], [], []
    imgDisplay = raw_image
    preprocessedImage = preprocessedImage1.astype(np.uint8)

    if len(preprocessedImage.shape) > 2:
        preprocessedImage = cv2.cvtColor(preprocessedImage, cv2.COLOR_BGR2GRAY)

    preprocessedImage = img_as_ubyte(preprocessedImage)  # denoise image
    denoised = rank.median(preprocessedImage, disk(2))
    # local ingredient
    gradientIm = rank.gradient(denoised, disk(thre))

    thresh = cv2.threshold(gradientIm, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # fused the the mask and the original image for extracting morph features
    thresh = cv2.bitwise_and(thresh, preprocessedImage)

    # find contours in the the mask
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    color = np.random.randint(0, 255, (len(contours), 3))
    mask_colored = np.zeros_like(raw_image, )
    pts, rectang, cellFeatures = [], [], []

    for (ii, cnt) in enumerate(contours):
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        tmp_perim = cv2.arcLength(cnt, True)
        tmp_approx = cv2.approxPolyDP(cnt, 0.04 * tmp_perim, True)
        (x, y, tmp_w, tmp_h) = cv2.boundingRect(tmp_approx)
        ar = tmp_w / float(tmp_h)

        # if ar >= 0.95 and ar <= 1.05:
        #     "sss"
        # else:
        #     del cnt
        #     continue

        area = cv2.contourArea(cnt)
        rect = cv2.boundingRect(cnt)
        (x1, y1, w1, h1) = rect
        cntIndex, mergedCnt = mergeOverSegmentedContours(contours, cnt)
        if mergedCnt is not None:
            cnt = []
            cnt = mergedCnt
            del contours[cntIndex]
        if area < minAreaSize or area > maxAreaSize or hierarchy[0, ii, 3] != -1:
            # del contours[ii]
            continue
        else:
            cellMorph = getCellIntensityModule(cnt, imgDisplay)
            cellFeatures.append(cellMorph)
            cv2.drawContours(mask_colored, [cnt], -1, color[ii].tolist(), thickness=cv2.FILLED,)
            cv2.drawContours(imgDisplay, [cnt], -1, (0, 255, 0), 1)
            rectang.append((x1, y1, w1, h1))
            pts.append([x, y])
    del thresh,  raw_image, preprocessedImage, preprocessedImage1, denoised, gradientIm
    return pts, rectang, mask_colored, imgDisplay, cellFeatures


def white_background(preprocessedImage1, raw_image, minAreaSize, maxAreaSize):

    # img = cv2.fastNlMeansDenoisingColored(img, None, 7, 7, 7, 41)
    if len(preprocessedImage1.shape) >2:
        preprocessedImage1 = cv2.cvtColor(preprocessedImage1, cv2.COLOR_BGR2GRAY)
    preprocessedImage = img_as_ubyte(preprocessedImage1)  # denoise image
    denoised = rank.median(preprocessedImage, disk(2))
    # local ingredient
    gray = rank.gradient(denoised, disk(2))

    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=2)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.05 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(raw_image, markers)
    mask = np.zeros_like(gray, )

    for label in np.unique(markers):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue

        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask[markers == -1] = 255

    mergedImage = cv2.bitwise_and(mask, preprocessedImage1)
    mergedImage = cv2.dilate(mergedImage, None)
    # detect contours in the mask and grab the largest one
    _, cnts, hierarchy = cv2.findContours(mergedImage.copy(), cv2.RETR_CCOMP,
                                          cv2.CHAIN_APPROX_SIMPLE)
    color = np.random.randint(0, 255, (len(cnts), 3))
    mask_colored = np.zeros_like(raw_image, )
    pts, rectang, cellFeatures = [], [], []

    for (ii, cnt) in enumerate(cnts):
        area = cv2.contourArea(cnt)
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        rect = cv2.boundingRect(cnt)
        (x1, y1, w1, h1) = rect

        if area < 150 or area > 20000 or (hierarchy[0, ii, 3] != -1 or hierarchy[0, ii, 3] == 0):
            continue
        else:
            cellMorph = getCellIntensityModule(cnt, raw_image)
            cellFeatures.append(cellMorph)
            cv2.drawContours(mask_colored, [cnt], -1, color[ii].tolist(), thickness=cv2.FILLED, )
            cv2.drawContours(raw_image, [cnt], -1, (0, 255, 0), 1)
            rectang.append((x1, y1, w1, h1))
            pts.append([x, y])
    del thresh, preprocessedImage, preprocessedImage1, denoised
    return pts, rectang, mask_colored, raw_image, cellFeatures



def maz_seg(image, rawIm, minAreaSize, maxAreaSize):

    for gamma in np.arange(0.0, 0.5, 0.5):
        # ignore when gamma is 1 (there will be no change to the image)
        if gamma == 1:
            continue

        # apply gamma correction
        gamma = gamma if gamma > 0 else 0.1

        adjusted = extra_modules.adjust_gamma(image, gamma=gamma)

    gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
    _, contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    color = np.random.randint(0, 255, (len(contours), 4))
    rawIm = cv2.cvtColor(rawIm, cv2.COLOR_BGR2GRAY)
    mask_colored = np.zeros_like(rawIm, )
    contour_list, pts = [], []
    ii = 0
    for contour in contours:
        (x, y), r = cv2.minEnclosingCircle(contour)
        approx = cv2.approxPolyDP(contour, 0.5 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)
        if ((len(approx) < minAreaSize) & (area < minAreaSize)):
            contour_list.append(contour)
            pts.append([x, y])
            cv2.drawContours(mask_colored, [contour], -1, color[ii].tolist(), thickness=cv2.FILLED)
        ii += 1
    cv2.drawContours(rawIm, contour_list, -1, (0, 0, 255), 1)

    return pts, contours, mask_colored, rawIm
