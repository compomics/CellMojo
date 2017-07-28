import os
import time

import cv2
try:
    import tkinter as tk  # for python 3
    from tkinter import *
except:
    import Tkinter as tk  # for python 2
    from Tkinter import *

from libtiff import TIFF

os_path = str(os.path)
if 'posix' in os_path:
    import posixpath as path
elif 'nt' in os_path:
    import ntpath as path


def readFile(filepath, tmp_dir, probar):
    print(filepath)
    frames,  timestamp = [],  []
    if filepath.lower().endswith('.tif'):

        tif = TIFF.open(filepath, mode='r')

        try:
            for cc, img in enumerate(tif.iter_images()):
                # r = 500.0 / img.shape[1]
                # dim = (500, int(img.shape[0] * r))

                # img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
                cv2.imwrite(path.join(tmp_dir, "rawImg{}.png".format(cc)), img)
                frames.append(img)
                probar.step(cc)
                probar.update()

        except EOFError or MemoryError:
            try:
                img = cv2.imread(filepath)
                # r = 500.0 / img.shape[1]
                # dim = (500, int(img.shape[0] * r))
                # img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

                cv2.imwrite(path.join(tmp_dir, "rawImg{}.png".format(cc)), img)
                frames.append(img)
                probar.stop()
            except EOFError:
                tkMessageBox.showinfo(
                    'Error...', "File can't be read / File is a not a movie!!!")
                pass

    if '.avi' or '.gif' or '.mp4' or '.mov' in filepath:

        cap = cv2.VideoCapture(filepath)
        cc = 0
        try:
            while cap.isOpened():
                ret, img = cap.read()
                # get the frame in seconds
                t1 = cap.get(0)
                timestamp.append(t1)
                if img is None:
                    break
                # r = 500.0 / img.shape[1]
                # dim = (500, int(img.shape[0] * r))
                # img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

                cv2.imwrite(path.join(tmp_dir, "rawImg{}.png".format(cc)), img)
                frames.append(img)
                probar.step(cc)
                probar.update()
                time.sleep(0.1)
                cc += 1
        except EOFError or MemoryError:
            tkMessageBox.showinfo(
                'Error', "File can't be read / File is a not a movie / File is too big!!!")
            pass

    if filepath.lower().endswith(('.png', '.jpg', '.jpeg')):

        try:
            img = cv2.imread(filepath)

            # r = 500.0 / img.shape[1]
            # dim = (500, int(img.shape[0] * r))
            # img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

            cv2.imwrite(path.join(tmp_dir, "rawImg{}.png".format(cc)), img)
            frames.append(img)

        except EOFError:
            tkMessageBox.showinfo('Error', "File can't be read!!!")
            pass

    probar.stop()

    return frames, timestamp
