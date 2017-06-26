# imports
from .. import common

def KCFTrack(self, frames, firstImage, smoothingmethod, segMeth, exp_parameter, updateconvax, progessbar, timelapse, tmp_dir):
    """
    Perfom tracking using Kernalized correlation filters
    """
    
    tracker = cv2.MultiTracker("KCF")
    init_once = False
    old_gray = call_back_preprocessing.call_preprocessing(
        firstImage, smoothingmethod)
    initialpoints, boundingBox, _, _, CellInfo = call_back_segmentation.call_segmentation(segMeth, processedImage=old_gray,
                                                                                    rawImg=firstImage,
                                                                                    minAreaSize=exp_parameter[2],
                                                                                    maxAreaSize=exp_parameter[3],
                                                                                    fixscale=exp_parameter[4],
                                                                                    minDistance=exp_parameter[5],
                                                                                    cellEstimate=exp_parameter[1],
                                                                                          color=int(exp_parameter[6]),
                                                                                          thre=exp_parameter[7])

    trajectoriesX, trajectoriesY, cellIDs, frameID, t = [], [], [], [], []

    # for better performance, the algorithm requires an image to resized to a its average size
    if segMeth == 6:
        if firstImage.shape[0] or firstImage.shape[1] > 500:
            r = 500.0 / firstImage.shape[1]
            dim = (500, int(firstImage.shape[0] * r))

            firstImage = cv2.resize(firstImage, dim, interpolation=cv2.INTER_AREA)

    noFrames = len(frames)
    masks = np.zeros_like(firstImage,)
    for ii, image in enumerate(frames[1:]):
        try:

            if segMeth == 6:
                if image.shape[0] or image.shape[1] > 500:
                    r = 500.0 / image.shape[1]
                    dim = (500, int(image.shape[0] * r))

                    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

            IM = image
            old_gray = call_back_preprocessing.call_preprocessing(image, smoothingmethod)
            # check if its two dim or more
            if len(old_gray.shape) > 3:
                old_gray = cv2.cvtColor(old_gray, cv2.COLOR_GRAY2BGR)

            if not init_once:
                # add a list of boxes:
                ok = tracker.add(old_gray, boundingBox)
                init_once = True

            ok, boxes = tracker.update(old_gray)
            old_points = []
            old_points2 = []
            c = 0

            for newbox in boxes:
                coord = (newbox[0], newbox[1], newbox[2], newbox[3])
                cX, cY = (coord[0] + (coord[2] / 2), coord[1] + (coord[3] / 2))
                points = (int(cX), int(cY))

                if ii == 0:
                    cv2.line(masks, (int(cX), int(cY)), (int(cX), int(cY)), (255, 0, 0), 2)
                    cv2.putText(IM, '[%d]' % c, points, cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1, cv2.LINE_AA)

                else:
                    old_p = old_points2[c]
                    cv2.line(masks, old_p, (int(cX), int(cY)), (255, 0, 0), 2)
                    cv2.putText(
                        IM, '[%d]' % c, points, cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1, cv2.LINE_AA)
                c += 1
                old_points.append(points)

                trajectoriesX.append(cX)
                trajectoriesY.append(cY)
                cellIDs.append(int(c))
                frameID.append(ii)

                # display to the panel the location
                displayCoordinates(self, c, cX, cY)

            old_points2 = old_points

            IM = np.add(IM, masks)

            tmp_img = path.join(str(tmp_dir[1]), 'frame{}.png'.format(ii))
            save_path = path.join(str(tmp_dir[0]), 'frame{}.png'.format(ii))
            cv2.imwrite(save_path, IM)

            if ii == noFrames - 1 or ii == noFrames:
                common.save_image(str(tmp_dir[0]), 'frame{}.png'.format(ii), IM)

            # handle image in the displace panel

            img = cv2.imread(save_path)

            r = 600.0 / img.shape[1]
            dim = (600, int(img.shape[0] * r))

            # perform the actual resizing of the image and display it to the panel
            resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            common.save_image(tmp_dir[3], '%d.gif' % ii, resized)

            displayImage = tk.PhotoImage(file=str(path.join(tmp_dir[3], '%d.gif' % ii)))
            root.displayImage = displayImage
            imagesprite = updateconvax.create_image(
                263, 187, image=displayImage)
            updateconvax.update_idletasks()  # Force redraw
            updateconvax.delete(imagesprite)

            if ii == noFrames - 1 or ii == noFrames:
                displayImage = tk.PhotoImage(
                    file=str(path.join(tmp_dir[3], '%d.gif' % ii)))
                root.displayImage = displayImage
                imagesprite = updateconvax.create_image(
                    263, 187, image=displayImage)
        except EOFError:
            continue
        # timelapse += Initialtime

    unpacked = zip(frameID, cellIDs, trajectoriesX, trajectoriesY)
    with open(path.join(tmp_dir[2], 'data.csv'), 'wt') as f1:
        writer = csv.writer(f1, lineterminator='\n')
        writer.writerow(('frameID', 'track_no', 'x', "y",))
        for value in unpacked:
            writer.writerow(value)
