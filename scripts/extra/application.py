import os
import re
import sys
sys.path.append('./morphology')
os_path = str(os.path)
if 'posix' in os_path:
    import posixpath as path
elif 'nt' in os_path:
    import ntpath as path
import time, zipfile
import matplotlib.animation as animation
from PIL import Image as img2
from PIL import ImageSequence
import matplotlib.pyplot as plt
import common
import preprocessing.preprocessing as preprocessing
import segmentation.segmentation as segmentation
from morphology.morph_extraction import *
from common import call_preprocessing, call_segmentation

from tracking.opt_flow_tracker import *
from tracking.knn_tracker import *
import cv2
import Filesprocessor
import mahotas
import pygubu

import tkMessageBox
import ttk
from Tkinter import NE, E, N, W

try:
    import tkinter as tk  # for python 3
except:
    import Tkinter as tk  # for python 2

from tkFileDialog import askopenfilename, asksaveasfilename

class popupWindow(object):
    """
    This class manages the GUI of a popup window
    """

    def __init__(self, app):
        self.top = tk.Toplevel(app)

        self.l = tk.Label(self.top, text="Parameter settings")
        self.l.pack()
        panel1 = tk.Frame(self.top)
        panel1.pack(fill=tk.X)

        threshLabel = tk.Label(panel1, text="threshold (1~7)")
        threshLabel.pack(side=tk.LEFT, padx=5, pady=5)

        self.thr = tk.Entry(panel1, width=4)
        self.thr.pack(fill=tk.X, padx=2, pady=2)

        panel2 = tk.Frame(self.top)
        panel2.pack(fill=tk.X)
        randomFrameID = tk.Label(panel2, text="Frame number ")
        randomFrameID.pack(side=tk.LEFT, padx=5, pady=5)
        self.frameID = tk.Entry(panel2, width=2)
        self.frameID.pack(fill=tk.X, padx=1)

        panel3 = tk.Frame(self.top)
        panel3.pack(fill=tk.BOTH)
        clusterID = tk.Label(panel3, text="Cluster (1~2) ")
        clusterID.pack(side=tk.LEFT, padx=5, pady=5)
        self.classLabel = tk.Entry(panel3, width=4)
        self.classLabel.pack(fill=tk.X, padx=2, pady=2)

        self.b = tk.Button(self.top, text='Ok', command=self.cleanupVariables)
        self.b.pack(side=tk.BOTTOM)

    def cleanupVariables(self):
        self.thr = self.thr.get()
        self.frameID = self.frameID.get()
        self.classLabel = self.classLabel.get()
        self.top.destroy()


class Application:
    """
    This class contains CellMojo's application methods
    and parameters on the application level
    """

    def __init__(self, master):

        # 1: Create a builder

        self.builder = builder = pygubu.Builder()

        # 2: Load an ui file
        builder.add_from_file('./gui/celltracker_html.ui')

        # 3: Create the widget using a master as parent
        self.mainwindow = builder.get_object('mainwindow', master)

        # 4: Get the labeled frame
        self.labelframe1 = builder.get_object("Labelframe_19")

        # 5:  Get the filename or path
        self.pathchooserinput_3 = builder.get_object("pathchooserinput_3")

        # 6: Read the files
        self.button = builder.get_object("Button_10")

        # 7: Create a progress bar
        self.progressdialog = ttk.Progressbar(
            self.labelframe1, mode='indeterminate', value=0)
        self.progressdialog.grid(row=2, column=0, sticky=N + E + W)

        self.Labelframe_22 = builder.get_object("Labelframe_22")
        self.progressdialog2 = ttk.Progressbar(
            self.Labelframe_22, mode='indeterminate', value=0)
        self.progressdialog2.grid(
            row=3, column=0, columnspan=5, sticky=N + E + W)

        # 8:  Segmentation parameters

        self.labelframe2 = builder.get_object("Labelframe_12")

        # self.convax1 = tk.Canvas(self.labelframe2, width=180, height=100)
        # self.convax1.grid(row=7,column=0)

        # 8.1: Scale label
        self.label = tk.Label(self.labelframe2)
        self.label.grid(row=1, column=5, sticky=W)
        self.fixscale = 0.5
        self.label.configure(text=self.fixscale)

        self.labelTrack = builder.get_object("Labelframe_22")

        # 8.2: Scale2 label
        self.Speed = tk.Label(self.labelTrack)
        self.Speed.grid(row=2, column=3, sticky=W)
        self.AverageSpeed = 30
        self.Speed.configure(text=self.AverageSpeed)

        # get the display label of the coordinates and time lapse
        self.label_10 = self.builder.get_object("Label_10")
        self.label_11 = self.builder.get_object("Label_11")
        self.label_12 = self.builder.get_object("Label_12")
        self.label_13 = self.builder.get_object("Label_13")
        self.label_14 = self.builder.get_object("Label_14")
        self.label_15 = self.builder.get_object("Label_15")
        self.label_16 = self.builder.get_object("Label_16")
        self.label_17 = self.builder.get_object("Label_17")
        self.label_18 = self.builder.get_object("Label_18")
        self.label_19 = self.builder.get_object("Label_19")
        self.label_43 = self.builder.get_object("Label_43")
        self.label_44 = self.builder.get_object("Label_44")
        self.label_45 = self.builder.get_object("Label_45")
        self.label_46 = self.builder.get_object("Label_46")
        self.label_47 = self.builder.get_object("Label_47")
        self.label_48 = self.builder.get_object("Label_48")
        self.label_49 = self.builder.get_object("Label_49")
        self.label_50 = self.builder.get_object("Label_50")
        self.label_51 = self.builder.get_object("Label_51")
        self.label_52 = self.builder.get_object("Label_52")
        self.label_63 = self.builder.get_object("Label_63")
        self.label_64 = self.builder.get_object("Label_64")
        self.label_65 = self.builder.get_object("Label_65")
        self.label_66 = self.builder.get_object("Label_66")
        self.label_67 = self.builder.get_object("Label_67")
        self.label_68 = self.builder.get_object("Label_68")
        self.label_69 = self.builder.get_object("Label_69")
        self.label_70 = self.builder.get_object("Label_70")
        self.label_71 = self.builder.get_object("Label_71")
        self.label_72 = self.builder.get_object("Label_72")

        # 8.2: Entry
        self.cellEstimate = 200
        self.minDistance = 40
        self.minSize = 10
        self.maxAreaSize = 300000
        self.minAreaSize = 2

        # 9: Perform segmentation

        self.preview = builder.get_object("Button_1")

        self.convax1 = builder.get_object("Canvas_4")

        self.preprocesing = self.builder.get_variable("preprocessing")

        self.segmentation = self.builder.get_variable("seg")
        self.color = self.builder.get_variable("background")

        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # 10: Create a tracking labels

        self.track = builder.get_variable("track")

        self.trackconvax = builder.get_object("Canvas_2")

        # 11: Create a file mining

        self.generatefile = builder.get_object("Button_3")

        self.savefile2 = builder.get_object("Button_4")

        # 12: about the toolbox

        self.clear = builder.get_object("Button_11")

        builder.connect_callbacks(self)

        # : set global variable
        self.frames, self.timestamp = [], []

        # call segmentation convax
        self.preconvax = self.builder.get_object("Canvas_5")

    def readfile_process_on_click(self):
        # Get the path choosed by the user""

        self.path = self.pathchooserinput_3.cget('path')

        # show the path
        if self.path:
            tkMessageBox.showinfo('You choosed', str(self.path))

            # get the file name and exclude the rest of the path
            matchedpattern = [x.start()
                              for x in re.finditer(str(os.sep), self.path)]
            matchedpattern1 = [x.start()
                               for x in re.finditer(r'[.]', self.path)]
            # construct the file name
            self.getFIleName = str(
                self.path[matchedpattern[-1] + 1:  matchedpattern1[-1]])
            self.getFIleName = path.join(path.expanduser(
                '~'), '/media/sami/3E4CCFEF2F1F91FB2', self.getFIleName)

            self.segmentPreview_dir = path.join(
                self.getFIleName, 'segment preview')
            self.rawimg_dir = path.join(self.getFIleName, 'raw files')

            self.getFIleName = self.getFIleName
            if os.path.exists(self.getFIleName) is False:
                os.makedirs(self.getFIleName)
                os.makedirs(self.rawimg_dir)
                os.makedirs(self.segmentPreview_dir)

            # Read input data files
            self.frames, self.timestamp = Filesprocessor.readFile(
                self.path, str(self.rawimg_dir), self.progressdialog)

            # now display a sample of the input data to the panel
            tmp_img = self.frames[0]

            resized = cv2.resize(tmp_img, (600, 350))

            if os.path.exists(path.join(self.getFIleName, 'displayImg.gif')):
                os.remove(path.join(self.getFIleName, 'displayImg.gif'))

            mahotas.imsave(path.join(self.getFIleName,
                                     'displayImg.gif'), resized)
            #image1 = img2.open(path.join(self.getFIleName, 'displayImg.gif'))
            
            image1 = tk.PhotoImage(
                file=str(path.join(self.getFIleName, 'displayImg.gif')))

            #print image1
            self.convax1.image = image1
            _ = self.convax1.create_image(300, 185, image=image1)

    def popup(self):
        self.w = popupWindow(self.mainwindow)
        self.mainwindow.wait_window(self.w.top)

    def entryValue(self):
        return self.w.thr, self.w.frameID, self.w.classLabel

    global prev_image

    def preview_on_click(self):
        """ grab the segmentation settings"""

        self.cellEstimate = self.builder.get_object('Entry_1')
        self.minDistance = self.builder.get_object('Entry_3')
        self.minSize = self.builder.get_object('Entry_4')
        self.cellEstimate = self.cellEstimate.get()
        self.minDistance = self.minDistance.get()
        self.minSize = self.minSize.get()
        self.minAreaSize = self.builder.get_object("minArea")
        self.maxAreaSize = self.builder.get_object("maxArea")
        self.minAreaSize = int(self.minAreaSize.get())
        self.maxAreaSize = int(self.maxAreaSize.get())
        self.segMethod = self.segmentation.get()
        self.preproMethod = self.preprocesing.get()
        self.segTech = None

        if self.frames:
            # get data from the pop window
            self.callBack = tk.Button(
                self.mainwindow, text="Done", command=self.popup())
            self.thre, frameID, classLabel = self.entryValue()

            if frameID >= len(self.frames) or not frameID:
                frameID = 0

            if self.thre is [] or self.thre > 7:
                self.thre = 4
            if classLabel is []:
                classLabel = 1

            frameID = int(frameID)

            if self.segMethod == 1:
                """"perform blob segmentation"""
                tmp_convex, prev_image = None, None
                self.segTech = "blob"
                # clear image content to avoid undesirable output
                self.image = []
                self.image = self.frames[frameID]
                preprocessedImage = common.call_preprocessing(
                    self.image, self.preproMethod)
                _, _, _, prev_image = common.blob_seg(preprocessedImage)
                self.seg_display(prev_image)

            if self.segMethod == 2:
                self.segTech = "watershed"
                if self.color.get() == 1:
                    self.image = []
                    self.image = self.frames[frameID]
                    preprocessedImage1 = common.call_preprocessing(self.image,
                                                                                    self.preproMethod)
                    _, _, _, prev_image, _ = segmentation.black_background(
                        preprocessedImage1, self.image, self.minAreaSize, self.maxAreaSize)
                    self.seg_display(prev_image)

                if self.color.get() == 2:
                    self.image, prev_image = [], []
                    self.image = self.frames[frameID].copy()
                    preprocessedImage2 = common.call_preprocessing(self.image,
                                                                                    self.preproMethod)
                    _, _, _, prev_image, _ = segmentation.white_background(
                        preprocessedImage2, self.image, self.minAreaSize, self.maxAreaSize)
                    self.seg_display(prev_image)

            if self.segMethod == 3:
                """ Perform corner detections"""
                self.segTech = "hariss"
                tmp_convex, self.image = [], []
                self.image = self.frames[frameID].copy()
                preprocessedImage3 = common.call_preprocessing(self.image,
                                                                                self.preproMethod)
                _, _, prev_image = segmentation.harris_corner(preprocessedImage3, int(self.cellEstimate), float(self.fixscale),
                                                 int(self.minDistance))
                self.seg_display(prev_image)

            if self.segMethod == 4:
                self.segTech = "shi"
                tmp_convex, prev_image,  self.image = [], [], []
                self.image = self.frames[frameID].copy()
                preprocessedImage4 = common.call_preprocessing(self.image,
                                                                                int(self.preproMethod))
                _, _, prev_image = segmentation.shi_tomasi(preprocessedImage4, int(self.cellEstimate), float(self.fixscale),
                                              int(self.minDistance))
                self.seg_display(prev_image)

            if self.segMethod == 5:
                self.segTech = "kmeans"
                tmp_convex, prev_image, self.image = [], [], []
                self.image = self.frames[frameID].copy()
                preprocessedImage5 = common.call_preprocessing(self.image,
                                                                                int(self.preproMethod))
                _, _, _, prev_image, _ = segmentation.kmeansSegment(
                    preprocessedImage5, self.frames[frameID], 1, int(self.minAreaSize), int(self.maxAreaSize))
                self.seg_display(prev_image)

            if self.segMethod == 6:
                self.segTech = "graph"
                tmp_convex, prev_image, self.image = [], [], []
                self.image = self.frames[frameID].copy()
                preprocessedImage6 = common.call_preprocessing(self.image,
                                                                                self.preproMethod)
                _, _, _, prev_image, _ = segmentation.graphSegmentation(
                    preprocessedImage6, self.frames[frameID], self.minSize, self.minAreaSize, self.maxAreaSize)
                self.seg_display(prev_image)

            if self.segMethod == 7:
                self.segTech = "meanshift"
                tmp_convex, prev_image, self.image = [], [], []
                self.image = self.frames[frameID].copy()
                preprocessedImage7 = common.call_preprocessing(self.image,
                                                                                self.preproMethod)
                _, _, _, prev_image, _ = segmentation.meanshif(
                    preprocessedImage7, self.frames[frameID], self.minAreaSize, self.maxAreaSize, int(self.fixscale * 100))
                self.seg_display(prev_image)

            if self.segMethod == 8:
                self.segTech = "sheet"
                tmp_convex, prev_image, self.image = [], [], []
                self.image = self.frames[frameID].copy()
                preprocessedImage8 = common.call_preprocessing(self.image,
                                                                                self.preproMethod)
                _, _,  _, prev_image, _ = segmentation.sheetSegment(
                    preprocessedImage8, self.frames[frameID], self.minAreaSize, self.maxAreaSize)
                self.seg_display(prev_image)

            if self.segMethod == 9:
                self.segTech = "contour"
                tmp_convex, prev_image, self.image = [], [], []
                self.image = self.frames[frameID].copy()
                preprocessedImage9 = common.call_preprocessing(self.image,
                                                                                self.preproMethod)
                _, _, _, prev_image, _ = segmentation.findContour(
                    preprocessedImage9, self.frames[frameID], self.minAreaSize, self.maxAreaSize)
                self.seg_display(prev_image)

            if self.segMethod == 10:
                self.segTech = "threshold"
                tmp_convex, prev_image, self.image, preprocessedImage = [], [], [], preprocessedImage
                self.image = self.frames[frameID].copy()
                preprocessedImage = call_back_preprocessing.call_preprocessing(self.image,
                                                                               self.preproMethod)
                _, _, _, prev_image, _ = segmentation.threshold(
                    preprocessedImage, self.frames[frameID], self.minAreaSize, self.maxAreaSize)
                self.seg_display(prev_image)

            if self.segMethod == 11:
                self.segTech = "customized"
                tmp_convex, prev_image, self.image, preprocessedImage = [], [], [], []
                self.image = self.frames[frameID].copy()
                preprocessedImage = common.call_preprocessing(self.image,
                                                                               self.preproMethod)
                _, _, _, prev_image, _ = segmentation.overlapped_seg(
                    preprocessedImage, self.frames[frameID], self.minAreaSize, self.maxAreaSize)
                self.seg_display(prev_image)

            if self.segMethod == 12:
                self.segTech = "local gredient"
                tmp_convex, prev_image, self.image, preprocessedImage = [], [], [], []

                self.image = self.frames[frameID].copy()
                preprocessedImage12 = common.call_preprocessing(self.image,
                                                                                 self.preproMethod)
                _, _, prev_mask, prev_image, _ = segmentation.gredientSeg(preprocessedImage12, self.frames[frameID], self.minAreaSize,
                                                                          self.maxAreaSize, int(self.thre))
                mahotas.imsave(
                    path.join(self.segmentPreview_dir, 'mask.png'), prev_mask)
                self.seg_display(prev_image)
        else:
            tkMessageBox.showinfo('No file', 'no data is found!!!')

    def seg_display(self, prev_image):

        tmp_pre, segmentationPanel = [], []
        if os.path.exists(path.join(self.segmentPreview_dir, 'SegImage.gif')):
            os.remove(path.join(self.segmentPreview_dir, 'SegImage.gif'))
        mahotas.imsave(path.join(self.segmentPreview_dir,
                                 'frame0.png'), prev_image)
        r = 550.0 / prev_image.shape[1]
        dim = (550, int(prev_image.shape[0] * r))

        # perform the actual resizing of the image and show it
        prev_image = cv2.resize(prev_image, dim, interpolation=cv2.INTER_AREA)

        resized = cv2.resize(prev_image, (600, 350))

        mahotas.imsave(
            path.join(self.segmentPreview_dir, 'SegImage.gif'), resized)

        tmp_pre = tk.PhotoImage(
            file=str(path.join(self.segmentPreview_dir, 'SegImage.gif')))
        self.preconvax.image = tmp_pre
        _ = self.preconvax.create_image(283, 182, image=tmp_pre)

    def delete_item(self):
        self.preconvax.delete("all")
        # python = sys.executable
        # os.execv(python, ['Python'] +'segmentation.py')

    # select a tracking method
    def track_on_click(self):
        if self.frames:

            self.cellEstimate = self.builder.get_object('Entry_1')
            self.minDistance = self.builder.get_object('Entry_3')
            self.minSize = self.builder.get_object('Entry_4')
            self.timelapse = self.builder.get_object('Entry_2')
            self.timelapse = self.timelapse.get()
            self.cellEstimate = int(self.cellEstimate.get())
            self.minDistance = int(self.minDistance.get())
            self.minSize = int(self.minSize.get())
            self.minAreaSize = self.builder.get_object("minArea")
            self.maxAreaSize = self.builder.get_object("maxArea")
            self.minAreaSize = int(self.minAreaSize.get())
            self.maxAreaSize = int(self.maxAreaSize.get())
            self.segMethod = self.segmentation.get()
            self.preproMethod = self.preprocesing.get()
            scale2 = self.builder.get_object('Scale_2')
            self.AverageSpeed = int(scale2.get())
            self.fixscale = self.builder.get_object('Scale_1')
            self.fixscale = float(self.fixscale.get())

            # use opticalflow to track cell movements

            if self.track.get() == 8:

                # create directories under the file name, it is easy that way
                track_dir = path.join(self.getFIleName, str(
                    self.segTech), 'opticalflow')
                finalTrack_dir = path.join(track_dir, 'finalTrack')
                overlay_dir = path.join(track_dir, 'overlayedImages')
                self.display_dir = path.join(track_dir, 'displayedImage')
                report_dir = path.join(track_dir, 'report')
                self.movie_dir = path.join(track_dir, 'movie')
                csv_dir = path.join(track_dir, 'csvfiles')

                if path.exists(track_dir) is False:
                    os.makedirs(track_dir)
                    os.makedirs(finalTrack_dir)
                    os.makedirs(report_dir)
                    os.makedirs(self.movie_dir)
                    os.makedirs(csv_dir)
                    os.makedirs(overlay_dir)
                    os.makedirs(self.display_dir)

                self.tmp_path = [str(finalTrack_dir), str(
                    overlay_dir), str(csv_dir), str(self.display_dir)]
                self.exp_para = [self.AverageSpeed, int(
                    self.cellEstimate), self.minAreaSize, self.maxAreaSize, self.fixscale, self.minDistance, int(self.color.get()), int(self.thre)]

                # tkMessageBox.showinfo('..','Segmentation method: {}\n ' %self.segMethod )
                OptflowTracker(self, self.frames, self.frames[0], int(self.preproMethod), int(self.segMethod), self.exp_para,
                               self.trackconvax, self.progressdialog2,  self.timelapse, self.tmp_path)

                """make a movie out of the track images"""

                # extra_modules.make_video(images,outvid=str(path.join(self.movie_dir, 'movie.avi')),fps=5, size=None,
                # is_color=True, format="XVID")

                save_gif = True
                title = ''
                images, imgs = [], []
                for foldername in os.listdir(self.display_dir):
                    images.append(foldername)
                images.sort(key=lambda x: int(x.split('.')[0]))

                for _, file in enumerate(images):
                    im = img2.open(path.join(self.display_dir, file))
                    imgs.append(im)

                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.set_axis_off()

                ims = map(lambda x: (ax.imshow(x), ax.set_title(title)), imgs)

                im_ani = animation.ArtistAnimation(
                    fig, ims, interval=700, repeat_delay=300, blit=False)

                if save_gif:
                    im_ani.save(path.join(self.movie_dir,
                                          'movie.gif'), writer="imagemagick")

                myFormats = [('JPEG / JFIF', '*.jpg'),
                             ('CompuServer GIF', '*.gif'), ]
                filename = asksaveasfilename(filetypes=myFormats)
                if filename:
                    im = img2.open(os.path.join(
                        overlaytrajectoryanidir, 'animation.gif'))
                    original_duration = im.info['duration']
                    frames = [frame.copy()
                              for frame in ImageSequence.Iterator(im)]
                    frames.reverse()

            if self.track.get() == 9:

                # create directories under the file name, it is easy that way
                track_dir = path.join(
                    self.getFIleName, str(self.segTech), 'KNN')
                finalTrack_dir = path.join(track_dir, 'finalTrack')
                overlay_dir = path.join(track_dir, 'overlayedImages')
                self.display_dir = path.join(track_dir, 'displayedImage')
                report_dir = path.join(track_dir, 'report')
                self.movie_dir = path.join(track_dir, 'movie')
                csv_dir = path.join(track_dir, 'csvfiles')

                if path.exists(track_dir) is False:
                    os.makedirs(track_dir)
                    os.makedirs(finalTrack_dir)
                    os.makedirs(report_dir)
                    os.makedirs(self.movie_dir)
                    os.makedirs(csv_dir)
                    os.makedirs(overlay_dir)
                    os.makedirs(self.display_dir)

                self.tmp_path = [str(finalTrack_dir), str(
                    overlay_dir), str(csv_dir), str(self.display_dir)]
                self.exp_para = [self.AverageSpeed, int(
                    self.cellEstimate), self.minAreaSize, self.maxAreaSize, self.fixscale, self.minDistance, int(self.color.get()), self.thre]

                startTime = time.time()
                KNNTracker(self,self.frames, self.frames[0], int(self.preproMethod), int(self.segMethod),
                           self.exp_para,
                           self.trackconvax, self.progressdialog2, self.timelapse, self.tmp_path, self.thre)
                endtime = time.time() - startTime

                print(endtime)
                """make a movie out of the track"""
                save_gif = True
                title = ''
                images, imgs = [], []
                for foldername in os.listdir(self.display_dir):
                    images.append(foldername)
                images.sort(key=lambda x: int(x.split('.')[0]))

                for _, file in enumerate(images):
                    im = img2.open(path.join(self.display_dir, file))

                    imgs.append(im)
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.set_axis_off()

                ims = map(lambda x: (ax.imshow(x), ax.set_title(title)), imgs)

                im_ani = animation.ArtistAnimation(
                    fig, ims, interval=550, repeat_delay=300, blit=False)

                if save_gif:
                    im_ani.save(path.join(self.movie_dir,
                                          'movie.gif'), writer="imagemagick")

            # tracking using KCF
            if self.track.get() == 5:
                # create directories under the file name, it is easy that way
                track_dir = path.join(
                    self.getFIleName, str(self.segTech), 'KCF')
                finalTrack_dir = path.join(track_dir, 'finalTrack')
                overlay_dir = path.join(track_dir, 'overlayedImages')
                self.display_dir = path.join(track_dir, 'displayedImage')
                report_dir = path.join(track_dir, 'report')
                self.movie_dir = path.join(track_dir, 'movie')
                csv_dir = path.join(track_dir, 'csvfiles')

                if path.exists(track_dir) is False:
                    os.makedirs(track_dir)
                    os.makedirs(finalTrack_dir)
                    os.makedirs(report_dir)
                    os.makedirs(self.movie_dir)
                    os.makedirs(csv_dir)
                    os.makedirs(overlay_dir)
                    os.makedirs(self.display_dir)

                self.tmp_path = [str(finalTrack_dir), str(
                    overlay_dir), str(csv_dir), str(self.display_dir)]
                self.exp_para = [self.AverageSpeed, int(
                    self.cellEstimate), self.minAreaSize, self.maxAreaSize, self.fixscale, self.minDistance, int(self.color.get())]

                KCFTrack(self, self.frames, self.frames[0], int(self.preproMethod), int(self.segMethod),
                         self.exp_para,
                         self.trackconvax, self.progressdialog2, self.timelapse, self.tmp_path)

                """make a movie out of the track"""
                save_gif = True
                title = ''
                images, imgs = [], []
                for foldername in os.listdir(self.display_dir):
                    images.append(foldername)
                images.sort(key=lambda x: int(x.split('.')[0]))

                for _, file in enumerate(images):
                    im = img2.open(path.join(self.display_dir, file))

                    imgs.append(im)
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.set_axis_off()

                ims = map(lambda x: (ax.imshow(x), ax.set_title(title)), imgs)

                im_ani = animation.ArtistAnimation(
                    fig, ims, interval=600, repeat_delay=300, blit=False)

                if save_gif:
                    im_ani.save(path.join(self.movie_dir,
                                          'movie.gif'), writer="imagemagick")

        else:
            tkMessageBox.showinfo('Missing data', 'no data to process')

    def extract_morph(self):

        if self.frames:

            self.cellEstimate = self.builder.get_object('Entry_1')
            self.minDistance = self.builder.get_object('Entry_3')
            self.minSize = self.builder.get_object('Entry_4')
            self.timelapse = self.builder.get_object('Entry_2')
            self.timelapse = self.timelapse.get()
            self.cellEstimate = int(self.cellEstimate.get())
            self.minDistance = int(self.minDistance.get())
            self.minSize = int(self.minSize.get())
            self.minAreaSize = self.builder.get_object("minArea")
            self.maxAreaSize = self.builder.get_object("maxArea")
            self.minAreaSize = int(self.minAreaSize.get())
            self.maxAreaSize = int(self.maxAreaSize.get())
            self.segMethod = self.segmentation.get()
            self.preproMethod = self.preprocesing.get()
            scale2 = self.builder.get_object('Scale_2')
            self.AverageSpeed = int(scale2.get())
            self.fixscale = self.builder.get_object('Scale_1')
            self.fixscale = float(self.fixscale.get())
            self.segTech

            # create directories under the file name, it is easy that way
            morph_dir = path.join(self.getFIleName, str(self.segTech), 'Morph')
            finalMorph_dir = path.join(morph_dir, 'finalTrack')
            overlay_dir = path.join(morph_dir, 'overlayedImages')
            self.display_dir = path.join(morph_dir, 'displayedImage')
            report_dir = path.join(morph_dir, 'report')
            self.movie_dir = path.join(morph_dir, 'movie')
            csv_dir = path.join(morph_dir, 'csvfiles')

            if path.exists(morph_dir) is False:
                os.makedirs(morph_dir)
                os.makedirs(finalMorph_dir)
                os.makedirs(report_dir)
                os.makedirs(self.movie_dir)
                os.makedirs(csv_dir)
                os.makedirs(overlay_dir)
                os.makedirs(self.display_dir)

            self.tmp_dir = [str(finalMorph_dir), str(overlay_dir), str(csv_dir), str(self.display_dir)]

            self.exp_para = [self.AverageSpeed, int(self.cellEstimate), self.minAreaSize, self.maxAreaSize,
                             self.fixscale, self.minDistance, int(self.color.get()), int(self.thre)]

            # free the variable and allocate it to a new image
            self.image = []
            print self.tmp_dir
            self.image = self.frames[0].copy
            morph_extraction(self.frames,  int(self.preproMethod), int(self.segMethod),
                            self.exp_para, self.trackconvax,self.tmp_dir)

            """make a movie out of the track"""
            save_gif = True
            title = ''
            images, imgs = [], []
            for foldername in os.listdir(self.display_dir):
                images.append(foldername)
            images.sort(key=lambda x: int(x.split('.')[0]))

            for _, file in enumerate(images):
                im = img2.open(path.join(self.display_dir, file))

                imgs.append(im)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_axis_off()

            ims = map(lambda x: (ax.imshow(x), ax.set_title(title)), imgs)

            im_ani = animation.ArtistAnimation(
                fig, ims, interval=600, repeat_delay=300, blit=False)

            if save_gif:
                im_ani.save(path.join(self.movie_dir, 'movie.gif'),
                            writer="imagemagick")

    # scale
    def on_scale_click(self, event):

        scale = self.builder.get_object('Scale_1')
        self.fixscale = float("%.1f" % round(scale.get(), 1))
        self.label.configure(text=str(self.fixscale))

    def thresholding_click(self, event):

        scale2 = self.builder.get_object('Scale_2')
        self.AverageSpeed = int("%d" % round(scale2.get(), 1))
        self.Speed.configure(text=int(self.AverageSpeed))

    def savefile(self):
        name = asksaveasfilename(initialdir=csvdir)
        f1 = open(os.path.join(name), 'wt')
        writer = csv.writer(f1, lineterminator='\n')
        spamReader = csv.reader(open(os.path.join(csv_dir, 'data.csv')))
        for row in spamReader:
            writer.writerow(row)
        f1.close()

    def save_as_zip(self):
        zf = zipfile.ZipFile("data.zip", "w")
        for dirname, subdirs, files in os.walk(self.getFIleName):
            zf.write(dirname)
            for filename in files:
                zf.write(os.path.join(dirname, filename))
        zf.close()

        myFormats = [('ZIP files', '*.zip'), ]
        filenames = asksaveasfilename(
            initialdir=self.getFIleName, filetypes=myFormats)

        if filenames:
            zf = zipfile.ZipFile(path.join(filenames), 'w')

            for dirname, subdirs, files in os.walk(self.getFIleName):
                zf.write(dirname)
                for filename in files:
                    zf.write(os.path.join(str(dirname), filename))
            zf.close()

    def clear_frame(self):

        from IPython import get_ipython
        # ipython_shell = get_ipython()
        # ipython_shell.magic('%reset -s')

        # os.execv('CellMojo.py')
        python = sys.executable
        os.execl(python, python, 'main.py')

    # move
    def move_start(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def move_move(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    # windows zoom
    def zoomer(self, event):
        if (event.delta > 0):
            self.canvas.scale("all", event.x, event.y, 1.1, 1.1)
        elif (event.delta < 0):
            self.canvas.scale("all", event.x, event.y, 0.9, 0.9)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    # linux zoom
    def zoomerP(self, event):
        self.canvas.scale("all", event.x, event.y, 1.1, 1.1)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def zoomerM(self, event):
        self.canvas.scale("all", event.x, event.y, 0.9, 0.9)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
