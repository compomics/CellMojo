"""
This module handles extraction of morphological features
"""
from extra import common

# unused original third argument: firstImage (beteen frames and smoothingmethod)
# unused original third- and second to last argument: progressbar, timelapse (before tmp_dir)
def morph_extraction(frames, smoothingmethod, seg_method,
                    exp_parameter, updateconvax, tmp_dir):
    """
    Perform morphology extraction
    """

    amount_frames = len(frames)

    displayed_image, imagesprite = [], []

    cell_morphology, detections = [], []

    if frames:
        for i, frame in enumerate(frames):
            old_gray = common.call_preprocessing(frame, smoothingmethod)
            # unused: bounding_box and mask_image
            initialpoints, _, _, displayed_image, cell_info = common.call_segmentation(seg_method, preImage=old_gray,
                                                                                       rawImg=frame,
                                                                                       minAreaSize=exp_parameter[2],
                                                                                       maxAreaSize=exp_parameter[3],
                                                                                       fixscale=exp_parameter[4],
                                                                                       minDistance=exp_parameter[5],
                                                                                       cellEstimate=exp_parameter[1],
                                                                                       color=int(exp_parameter[6]),
                                                                                       thre=int(exp_parameter[7]))
            for ii, tmp_inf in enumerate(cell_info):

                if tmp_inf:
                    tmp_inf = tmp_inf[1:]
                    temp_list = list(common.concatenateList([i, int(ii), tmp_inf]))
                    cell_morphology.append(temp_list)

            # cell centroids
            for centre in initialpoints:
                x, y = centre
                detections.append([i, x, y])
            # tmp_img = path.join(str(tmp_dir[1]), 'frame{}.png'.format(i))
            common.write_image(str(tmp_dir[0]), 'frame{}.png'.format(i), displayed_image)

            if i == amount_frames - 1 or i == amount_frames:
                common.save_image(str(tmp_dir[0]), 'frame{}.png'.format(i), displayed_image)

            # handle image in the displace panel

            img = common.read_image(str(tmp_dir[0]), 'frame{}.png'.format(i))

            r = 600.0 / img.shape[1]
            dim = (600, int(img.shape[0] * r))

            # perform the actual resizing of the image and display it to the panel
            resized = common.resize_image(img, dim)
            common.save_image(tmp_dir[3], '%d.gif' % i, resized)

            display_image = common.tkinter_photoimage(str(common.join_path(tmp_dir[3], '%d.gif' % i)))
            updateconvax.displayImage = display_image
            imagesprite = updateconvax.create_image(
                263, 187, image=display_image)
            updateconvax.update_idletasks()  # Force redraw
            updateconvax.delete(imagesprite)

            if i == amount_frames - 1 or i == amount_frames:
                display_image = common.tkinter_photoimage(str(common.join_path(tmp_dir[3], '%d.gif' % i)))
                updateconvax.displayImage = display_image
                imagesprite = updateconvax.create_image(
                    263, 187, image=display_image)

    with open(common.join_path(tmp_dir[2], 'MorphFeatures.csv'), 'wt') as f2:
        writer = common.csv_writer(f2)
        arguments = ("frameID", "detection_no", "aspect_ratio", "extent", "solidity", "celldiameter", "IntegritedInensity",
                     "MeanIntensity", " StdIntensity", "maxIntensity", "minIntensity", " MajorAxisLength_ellipse",
                     "MinorAxisLength_ellipse", "MajorAxisLength_moment", "MinorAxisLength_moment", "area", "hull_area", "perimeter",
                     "eccentricity_ellipse", "eccentricity_moment", "roundness_ellipse", "roundness_moment", "circularity", "ratio")
        writer.writerow(arguments)
        for value in cell_morphology:
            writer.writerow(value)

    with open(common.join_path(tmp_dir[2], 'detections.csv'), 'wt') as f3:
        writer = common.csv_writer(f3)
        writer.writerow(('frameID', 'x', 'y'))
        for obj in detections:
            writer.writerow(obj)
