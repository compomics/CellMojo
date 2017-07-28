# imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.neighbors import NearestNeighbors

from .. import common

def OptflowTracker(self, frames, firstImage, smoothingmethod, segMeth, exp_parameter, updateconvax, progessbar, timelapse, tmp_dir):
    """
    Optical Flow-based tracker
    """
    old_gray = common.call_preprocessing(
        firstImage, smoothingmethod)
    initialpoints, boundingBox, _, _, CellInfo = common.call_segmentation(segMeth, preimage=old_gray,
                                                                           rawimage=firstImage,
                                                                           min_areasize=exp_parameter[2],
                                                                           max_areasize=exp_parameter[3],
                                                                           fixscale=exp_parameter[4],
                                                                           min_distance=exp_parameter[5],
                                                                           cell_estimate=exp_parameter[1],
                                                                                          color=int(exp_parameter[6]),
                                                                                          thre=int(exp_parameter[7]))

    # if initialpoints.shape != (len(initialpoints),1,2):
    initialpoints = np.vstack(initialpoints)
    initialpoints = initialpoints.reshape(len(initialpoints), 1, 2)

    Initialtime = timelapse
    detect_interval = 5

    noFrames = len(frames)

    # training a knn model
    firstDetections, updatedTrackIdx, updateDetections, old_trackIdx = [], [], [], []
    for indice, row in enumerate(initialpoints):
        g, d = row.ravel()
        firstDetections.append([g, d])
        updatedTrackIdx.append(indice)
        old_trackIdx.append(indice)

    firstDetections = np.vstack(firstDetections)
    updateDetections = firstDetections
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(firstDetections)

    trajectoriesX, trajectoriesY, cellIDs, frameID, t, track_history, tracks = [
    ], [], [], [], [], [], []

    for i, frame in enumerate(frames):

        try:
            if segMeth == 6:
                if frame.shape[0] or frame.shape[1] > 500:
                    r = 500.0 / frame.shape[1]
                    dim = (500, int(frame.shape[0] * r))

                    frame = common.resize_image(frame, dim)

            # make a copy of the frame for ploting reasons
            imagePlot = frame.copy()

            # show the progress bar
            progessbar.step(i * 2)

            im = common.call_preprocessing(frame, smoothingmethod)

            # Parameters for lucas kanade optical flow
            lk_params = dict(winSize=(20, 20), maxLevel=3,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.5))

            # do some calculations
            newPoints, state, errorRate = cv2.calcOpticalFlowPyrLK(
                old_gray, im, initialpoints, None, **lk_params)

            # Select good points
            p1 = newPoints[state == 1]
            good_new = []

            for _, row in enumerate(p1):
                C2, D2 = row.ravel()
                good_new.append([C2, D2])
            # make the given detection matrix

            if not good_new:
                continue
            good_new = np.vstack(good_new)

            secondDetections = []
            secondDetections = updateDetections

            neigh = NearestNeighbors(n_neighbors=1)
            neigh.fit(updateDetections)

            updatedTrackIdx = []
            for tt, row in enumerate(good_new):
                z, y = row.ravel()
                test = np.hstack([z, y])
                # find the closet point to the testing  in the training data
                nearestpoint = neigh.kneighbors(np.array([test]))
                trackID = int(nearestpoint[1][0])
                distance = nearestpoint[0][0]
                distance = np.float32(distance[0])

                if distance > int(exp_parameter[0]):
                    new_idx = old_trackIdx[-1] + 1
                    updatedTrackIdx.append(new_idx)
                    old_trackIdx.append(new_idx)
                    updateDetections = np.vstack([updateDetections, test])
                    secondDetections = np.vstack([secondDetections, test])

                else:
                    updatedTrackIdx.append(trackID)
                    updateDetections[trackID] = np.hstack(test)

            secondDetections = np.int32(np.vstack(secondDetections))

            for ii, (new, old) in enumerate(zip(good_new, secondDetections)):
                cellId = updatedTrackIdx[ii]
                a, b = new.ravel()
                track_history.append([i, cellId, a, b, Initialtime])
                if CellInfo:
                    tmp_inf = CellInfo[ii]
                    tmp_inf = tmp_inf[1:]
                    tmpList = list(common.concatenate_list(
                        [i, int(cellId), tmp_inf]))
                    CellMorph.append(tmpList)

                # manage the displaying label
                common.displaycoordinates(self, ii, a, b, Initialtime)

            dataFrame = pd.DataFrame(track_history, columns=[
                'frame_idx', 'track_no', 'x', 'y'])

            # review tracking
            common.draw_str(imagePlot, (20, 20), 'track count: %d' % len(good_new))

            if dataFrame is not None:
                index_Values = dataFrame["track_no"]
                x_Values = dataFrame["x"]
                y_values = dataFrame["y"]
                frameIDx = dataFrame["frame_idx"]
                # timeSeries =dataFrame["time"]

                fig = plt.figure()
                plt.imshow(cv2.cvtColor(imagePlot, cv2.COLOR_BGR2RGB))

                for _, value in enumerate(np.unique(index_Values)):
                    tr_index = dataFrame.track_no[dataFrame.track_no == int(
                        value)].index.tolist()

                    xCoord = x_Values[tr_index]
                    yCoord = y_values[tr_index]
                    tmpFrameID = frameIDx[tr_index]
                    # timeStamp = timeSeries[tr_index]
                    tmpFrameID = np.int32(tmpFrameID)
                    tmp_x = np.int32(xCoord)
                    tmp_y = np.int32(yCoord)
                    sigma = 4
                    tmp_x = gaussian_filter1d(tmp_x, sigma)
                    tmp_y = gaussian_filter1d(tmp_y, sigma)

                    xx = tmp_x[-1]
                    yy = tmp_y[-1]

                    if i == noFrames - 1 and int(tmp_x.shape[0]) < 5:
                        # remove tracks with only appear only a few times
                        del tmp_x
                        del tmp_y
                    else:
                        # plt.contour(secondlargestcontour, (0,), colors='g',
                        # linewidths=2)
                        plt.text(xx, yy, "[%d]" % int(value),
                                 fontsize=5, color='yellow')
                        plt.plot(tmp_x, tmp_y, 'b-', linewidth=1)

                        if i == noFrames - 1 or i == noFrames:

                            for _, (xx, yy, idx) in enumerate(zip(tmp_x, tmp_y, tmpFrameID)):
                                trajectoriesX.append(xx)
                                trajectoriesY.append(yy)
                                cellIDs.append(value)
                                frameID.append(idx)

            plt.axis('off')
            mng = plt.get_current_fig_manager()
            mng.full_screen_toggle()

            fig.savefig(common.join_path(str(tmp_dir[1]), 'frame{}.png'.format(i)))

            if i == noFrames - 1:
                fig.savefig(
                    common.join_path(str(tmp_dir[0]), 'frame{}.png'.format(i)))
            del fig

            # Now update the previous frame and previous points
            old_gray = im.copy()
            initialpoints = p1.reshape(-1, 1, 2)

            # handle image in the displace panel
            img = common.read_image(str(tmp_dir[1]), 'frame{}.png'.format(i))

            r = 600.0 / img.shape[1]
            dim = (600, int(img.shape[0] * r))

            # perform the actual resizing of the image and display it to the
            # panel
            resized = common.resize_image(img, dim)
            common.save_image(tmp_dir[3], '%d.gif' % i, resized)

            displayImage = tk.PhotoImage(
                file=str(common.join_path(tmp_dir[3], '%d.gif' % i)))
            common.display_image(displayImage)
            imagesprite = updateconvax.create_image(
                263, 187, image=displayImage)
            updateconvax.update_idletasks()  # Force redraw
            updateconvax.delete(imagesprite)

            if i == noFrames - 1:

                displayImage = tk.PhotoImage(
                    file=str(common.join_path(tmp_dir[3], '%d.gif' % i)))
                common.display_image(displayImage)
                imagesprite = updateconvax.create_image(
                    263, 187, image=displayImage)

        except EOFError:
            continue
        # timelapse += Initialtime

    unpacked = zip(frameID, cellIDs, trajectoriesX, trajectoriesY)
    with open(common.join_path(tmp_dir[2],  'data.csv'), 'wt') as f1:
        writer = common.csv_writer(f1)
        writer.writerow(('frameID', 'track_no', 'x', "y",))
        for value in unpacked:
            writer.writerow(value)