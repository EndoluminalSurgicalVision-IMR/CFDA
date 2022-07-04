# -*- coding: utf-8 -*-


from typing import List, Tuple, Dict, Callable, Union, Any
import sys
import os

import SimpleITK as sitk
import numpy as np
import csv


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def save_itk(image, filename, origin=[0.0, 0.0, 0.0], spacing=[1.0, 1.0, 1.0]):
    if type(origin) != tuple:
        if type(origin) == list:
            origin = tuple(reversed(origin))
        else:
            origin = tuple(reversed(origin.tolist()))
    if type(spacing) != tuple:
        if type(spacing) == list:
            spacing = tuple(reversed(spacing))
        else:
            spacing = tuple(reversed(spacing.tolist()))
    itkimage = sitk.GetImageFromArray(image, isVector=False)
    itkimage.SetSpacing(spacing)
    itkimage.SetOrigin(origin)
    sitk.WriteImage(itkimage, filename, True)


def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = list(reversed(itkimage.GetOrigin()))
    numpySpacing = list(reversed(itkimage.GetSpacing()))
    return numpyImage, numpyOrigin, numpySpacing


def getabspath(RelativePathList):
    length = len(RelativePathList)
    path = None
    for i in range(length - 1, 0, -1):
        if i == (length - 1):
            path = os.path.join(RelativePathList[i - 1], RelativePathList[i])
        else:
            path = os.path.join(RelativePathList[i - 1], path)
    return path


def dice_coef_np(y_pred, y_true, smooth=1):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) * 100 / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


###############################################################################
# Helper Classes
###############################################################################
class CSVLogger(object):
    def __init__(self, path):
        self.path = path

    def read(self):
        with open(self.path, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            print(', '.join(row))

    def write(self, buffer):
        with open(self.path, 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(buffer)
