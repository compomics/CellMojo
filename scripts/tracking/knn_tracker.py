#imports
from extra import common
import time
import csv,cv2, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.neighbors import NearestNeighbors
import pandas as pd
os_path = str(os.path)
if 'posix' in os_path:
    import posixpath as path
elif 'nt' in os_path:
    import ntpath as path

try:
    import tkinter as tk  # for python 3
except:
    import Tkinter as tk  # for python 2

def KNNTracker(self,frames, firstImage, smoothingmethod, segMeth, exp_parameter, updateconvax, progessbar, timelapse, tmp_dir, thre):
    """
    K-Nearest Neighbor tracker
    """

    startProcessingTime = time.clock()

    # initialize some variabiles
    trajectoriesX, trajectoriesY, cellIDs, frameID, t, track_history, CellMorph, ControlledVocabulary, \
        track_age, consecutiveInvisibleCount, totalVisibleCount, lostTracks, frameTime = [
        ], [], [], [], [], [], [], [], [], [], [], [], []

    # do preprocessing followed by segmentation
    old_gray = common.call_preprocessing(firstImage, smoothingmethod)
    initialpoints, boxes, _, _, CellInfo = common.call_segmentation(segMeth, preImage=old_gray,
                                                                                    rawImg=firstImage,
                                                                                    minAreaSize=exp_parameter[2],
                                                                                    maxAreaSize=exp_parameter[3],
                                                                                    fixscale=exp_parameter[4],
                                                                                    minDistance=exp_parameter[5],
                                                                                    cellEstimate=exp_parameter[1],
                                                                                    color=int(exp_parameter[6]),
                                                                                    thre=thre)

    # if initialpoints.shape != (len(initialpoints), 1, 2):
    initialpoints = np.vstack(initialpoints)
    initialpoints = initialpoints.reshape(len(initialpoints), 1, 2)

    Initialtime = int(timelapse)
    noFrames = len(frames)

    # initialize track ids that corresponds to a detection/track
    firstDetections, updatedTrackIdx, updateDetections, old_trackIdx = [], [], [], []
    for indi, row in enumerate(initialpoints):
        g, d = row.ravel()
        firstDetections.append([g, d])
        updatedTrackIdx.append(indi)
        old_trackIdx.append(indi)

    # training a knn model for with K == 1
    firstDetections = np.vstack(firstDetections)
    updateDetections = firstDetections
    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(firstDetections)

    for i, frame in enumerate(frames):
        try:
            frameProcessingTime = time.clock()
            # images should be resized if graph-based segmentation method is
            # chosen
            if segMeth == 6:
                frame = resizeImage(frame)
            imagePlot = frame.copy()
            oldFrame = frame.copy()
            # show the progress bar
            progessbar.step(i * 2)

            good_new, morphImage, nextFrame = [], [], []
            nextFrame = common.call_preprocessing(frame, smoothingmethod)
            p1, boxes, _, morphImage, CellInfo = common.call_segmentation(segMeth, preImage=nextFrame,
                                                                                          rawImg=oldFrame,
                                                                                          minAreaSize=exp_parameter[2],
                                                                                          maxAreaSize=exp_parameter[3],
                                                                                          fixscale=exp_parameter[4],
                                                                                          minDistance=exp_parameter[5],
                                                                                          cellEstimate=exp_parameter[1],
                                                                                          color=int(exp_parameter[6]),
                                                                                          thre=int(thre))

            # different algorithms produce data with different shape and
            # formats
            p1 = np.array(p1)
            if p1.shape != (len(p1), 1, 2):
                if len(p1) > 1:
                    p1 = np.vstack(p1)
                    p1 = p1.reshape(len(p1), 1, 2)

            for _, row in enumerate(p1):
                C2, D2 = row.ravel()
                good_new.append([C2, D2])
            # format a detection matrix to easy access
            good_new = np.vstack(good_new)

            # remove lost tracks
            if lostTracks:
                for lost in lostTracks:
                    updateDetections[lost] = np.hstack([0, 0])

            secondDetections = []
            secondDetections = updateDetections
            neigh = NearestNeighbors(n_neighbors=2)
            neigh.fit(updateDetections)

            updatedTrackIdx = []
            for tt, row in enumerate(good_new):

                z, y = row.ravel()
                test = np.hstack([z, y])
                # find the closet point in the training data for a given data
                # points
                nearestpoint = neigh.kneighbors(np.array([test]))
                trackID = int(nearestpoint[1][0][0])
                distance = nearestpoint[0][0][0]
                distance = np.float32(distance)

                if distance > int(exp_parameter[0]):
                    new_idx = old_trackIdx[-1] + 1
                    updatedTrackIdx.append(new_idx)
                    old_trackIdx.append(new_idx)
                    updateDetections = np.vstack([updateDetections, test])
                else:
                    updatedTrackIdx.append(trackID)
                    updateDetections[trackID] = np.hstack(test)

            secondDetections = np.int32(np.vstack(secondDetections))
            for ii, (new, old) in enumerate(zip(good_new, secondDetections)):
                cellIdx = int(updatedTrackIdx[ii])
                a, b = new.ravel()
                track_history.append([i, cellIdx, a, b, Initialtime])

                # find a track age, remove tracks that has been lost for more
                # than 15 frames
                if CellInfo:
                    tmp_inf = CellInfo[ii]
                    tmpList = list(common.concatenateList([i, int(cellIdx), tmp_inf]))
                    CellMorph.append(tmpList)
                # display some info to the user interface
                common.displayCoordinates(self,ii, a, b, Initialtime)

            dataFrame = pd.DataFrame(track_history, columns=[
                'frame_idx', 'track_no', 'x', 'y', 'time'])

            # review tracking
            common.drawStr(imagePlot, (20, 20), 'track count: %d' % len(good_new))
            common.drawStr(morphImage, (20, 20), 'track count: %d' % len(good_new))
            if dataFrame is not None:
                index_Values = dataFrame["track_no"]
                x_Values = dataFrame["x"]
                y_values = dataFrame["y"]
                frameIDx = dataFrame["frame_idx"]
                timeSeries = dataFrame["time"]
                # create a figure
                fig = plt.figure()

                plt.imshow(cv2.cvtColor(imagePlot, cv2.COLOR_BGR2RGB))

                for _, value in enumerate(np.unique(index_Values)):
                    tr_index = dataFrame.track_no[dataFrame.track_no == int(value)].index.tolist()
                    xCoord = x_Values[tr_index]
                    yCoord = y_values[tr_index]
                    tmpFrameID = frameIDx[tr_index]
                    timeStamp = timeSeries[tr_index]
                    timeStamp = np.int32(timeStamp)
                    tmpFrameID = np.int32(tmpFrameID)
                    tmp_x = np.int32(xCoord)
                    tmp_y = np.int32(yCoord)

                    # smooth trajectories using gaussian filter
                    sigma = 4
                    tmp_x = gaussian_filter1d(tmp_x, sigma)
                    tmp_y = gaussian_filter1d(tmp_y, sigma)
                    xx = tmp_x[-1]
                    yy = tmp_y[-1]
                    # remove tracks with that only appear a few times in the
                    # entire dataset
                    if i == noFrames - 1 and int(tmp_x.shape[0]) < 10:
                        del tmp_x
                        del tmp_y
                    else:
                        # plt.contour(secondlargestcontour, (0,), colors='g', linewidths=2)
                        plt.text(xx, yy, "[%d]" % int(value), fontsize=5, color='yellow')
                        cv2.putText(morphImage, "%d" % int(value), (int(xx) - 10, int(yy)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 3)
                        plt.plot(tmp_x, tmp_y, 'r-', linewidth=1)
                        if i == noFrames - 1 or i == noFrames:
                            for _, (xx, yy, idx, tmx) in enumerate(zip(tmp_x, tmp_y, tmpFrameID, timeStamp)):
                                trajectoriesX.append(xx)
                                trajectoriesY.append(yy)
                                cellIDs.append(value)
                                frameID.append(idx)
                                t.append(tmx)

                    # check  for lost tracks
                    if i > 6:
                        delTrack = common.deleteLostTracks(value, tmpFrameID, i)
                        if delTrack:
                            lostTracks.append(delTrack)
            plt.axis('off')
            mng = plt.get_current_fig_manager()
            mng.full_screen_toggle()
            tmp_img = path.join(str(tmp_dir[1]), 'frame{}.png'.format(i))
            cv2.imwrite(path.join(str(tmp_dir[1]), 'morph{}.png'.format(i)), morphImage)
            fig.savefig(tmp_img, bbox_inches='tight')
            if i == noFrames - 1 or i == noFrames:
                fig.savefig(path.join(str(tmp_dir[0]), 'frame{}.png'.format(i)), bbox_inches='tight')
                cv2.imwrite(path.join(str(tmp_dir[1]), 'morph{}.png'.format(i)), morphImage)
            del fig

            # Now update the previous frame and previous points
            old_gray = nextFrame.copy()
            # handle image in the displace panel
            img = cv2.imread(tmp_img)
            r = 600.0 / img.shape[1]
            dim = (600, int(img.shape[0] * r))

            # perform the actual resizing of the image and display it to the
            # panel
            resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            common.save_image(tmp_dir[3], '%d.gif' % i, resized)

            displayImage = tk.PhotoImage(file=str(path.join(tmp_dir[3], '%d.gif' % i)))
            updateconvax.displayImage = displayImage
            imagesprite = updateconvax.create_image(
                266, 189, image=displayImage)
            updateconvax.update_idletasks()  # Force redraw
            updateconvax.delete(imagesprite)

            if i == noFrames - 1 or i == noFrames:
                displayImage = tk.PhotoImage(file=str(path.join(tmp_dir[3], '%d.gif' % i)))
                updateconvax.displayImage = displayImage
                imagesprite = updateconvax.create_image(
                    263, 187, image=displayImage)

            ControlledVocabulary.append(CellMorph)
            Initialtime += int(timelapse)

            # time computation
            frameEndTime = time.clock()
            endTime = frameEndTime - frameProcessingTime
            frameTime.append([i, endTime])
        except EOFError:
            continue
            # timelapse += Initialtime

    unpacked = zip(frameID, cellIDs, trajectoriesX, trajectoriesY, t)
    
    with open(path.join(tmp_dir[2], 'tracks.csv'), 'wt') as f1:
        writer = csv.writer(f1, lineterminator='\n')
        writer.writerow(('frameID', 'track_no', 'x', "y", "t",))
        for value in unpacked:
            writer.writerow(value)

    f1.close()

    with open(path.join(tmp_dir[2], 'MorphFeatures.csv'), 'wt') as f2:
        writer = csv.writer(f2, lineterminator='\n')
        writer.writerow(
            ('frameID', 'detection_no', 'aspectRatio', 'extent', 'solidity', 'equivalentCellDiameter',  'integratedIntensity', 'meanIntensity', 'stdIntensity', 'maxIntensity', 'minIntensity', 'majorAxisLengthEllipse', 'minorAxisLengthEllipse', 'area', 'hullArea',
                        'perimeter', 'eccentricityEllipse', 'roundnessEllipse',  'circularity', 'areaPerimeterRatio', 'majorAxisLengthMoment', 'minorAxisLengthMoment', 'eccentricityMoment', 'roundnessMoment'))
        for value in CellMorph:
            writer.writerow(value)
    f2.close()

    tmp_endProcessingTime = time.time()
    endProcessingTime = tmp_endProcessingTime - startProcessingTime

    unpacked = zip(frameTime)
    with open(path.join(tmp_dir[2], 'timePerFrame.csv'), 'wt') as f3:
        writer = csv.writer(f3, lineterminator='\n')
        writer.writerow(('frameID', 'time',))
        for value in frameTime:
            writer.writerow(value)
    f3.close()

    # write the feature extraction and tracking time lapse

    # opens file with name of "totalProcessingTime.txt"
    f = open("totalProcessingTime.txt", "w")
    f.write(str(endProcessingTime))
    f.close()
