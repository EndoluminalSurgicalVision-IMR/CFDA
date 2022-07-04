# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import torch
from natsort import natsorted
from copy import deepcopy

from monai.transforms import (
    RandCropByLabelClassesd,
    SpatialCropd,
    NormalizeIntensityd,
    Compose,
    LabelFilter,
    AddChanneld,
    RandSpatialCropd,
    RandFlipd,
    Orientationd,
    RandRotated,
    ToTensord
)
from monai.utils import set_determinism

from .base_dataset import BaseDataset
from main_code.util import utils


class CrossDomainAirwayDataset(BaseDataset):
    '''
        CrossDomainAirwayDataset
    '''

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--in_covid19airway', type=str, default='COVID19_Airway', help='filefolder name')
        parser.add_argument('--in_lidc', type=str, default='LIDC', help='filefolder name')
        parser.add_argument('--depth', type=int, default=128, help='depth used in the crop size')
        parser.add_argument('--width', type=int, default=224, help='width used in the crop size')
        parser.add_argument('--height', type=int, default=304, help='height used in the crop size')

        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.isTrain = opt.isTrain

        self.rootdir = os.path.join(opt.dataroot, opt.phase)
        self.covid19airway_rootdir = utils.getabspath([self.rootdir, opt.in_covid19airway])
        self.lidc_rootdir = utils.getabspath([self.rootdir, opt.in_lidc])

        self.covid19airway_filelist = os.listdir(self.covid19airway_rootdir)
        self.covid19airway_filelist = natsorted(self.covid19airway_filelist)
        self.lidc_filelist = os.listdir(self.lidc_rootdir)
        self.lidc_filelist = natsorted(self.lidc_filelist)
        set_determinism(seed=777)

        self.train_transform = Compose(
            [
                AddChanneld(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes='RAS'),
                RandCropByLabelClassesd(keys=["image", "label"],
                                        label_key="label",
                                        spatial_size=[opt.depth, opt.width, opt.height], ratios=[0.2, 0.8],
                                        num_classes=2, num_samples=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=(-2, -1)),
                RandRotated(keys=["image", "label"], prob=0.2, range_x=(-0.174, 0.174), range_y=(-0.174, 0.174),
                            range_z=(-0.174, 0.174), mode=["bilinear", "nearest"]),
                ToTensord(keys=["image"], dtype=torch.float),
                ToTensord(keys=["label"], dtype=torch.uint8)
            ]
        )

        self.test_transform = Compose(
            [
                AddChanneld(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes='RAS'),
                ToTensord(keys=["image"], dtype=torch.float),
                ToTensord(keys=["label"], dtype=torch.uint8)
            ]
        )

    def __len__(self):
        return len(self.lidc_filelist)

    def __getitem__(self, idx):
        idx_covid19 = random.randint(0, len(self.covid19airway_filelist) - 1)
        covid19_airway_casename = self.covid19airway_filelist[idx_covid19]
        covid19_airway_image, covid19_airway_origin, covid19_airway_spacing = utils.load_itk_image(utils.getabspath(
            [self.covid19airway_rootdir, covid19_airway_casename, covid19_airway_casename + "_clean.nii.gz"]))
        covid19_airway_image = self.normalize_ct(covid19_airway_image)

        covid19_airway_label, _, _ = utils.load_itk_image(utils.getabspath(
            [self.covid19airway_rootdir, covid19_airway_casename, covid19_airway_casename + "_label.nii.gz"]))
        covid19_airway_label = covid19_airway_label.astype(np.uint8)

        lidc_airway_casename = self.lidc_filelist[idx]
        lidc_airway_image, lidc_airway_origin, lidc_airway_spacing = utils.load_itk_image(
            utils.getabspath([self.lidc_rootdir, lidc_airway_casename, lidc_airway_casename + "_clean.nii.gz"]))
        lidc_airway_image = self.normalize_ct(lidc_airway_image)

        lidc_airway_label, _, _ = utils.load_itk_image(
            utils.getabspath([self.lidc_rootdir, lidc_airway_casename, lidc_airway_casename + "_label.nii.gz"]))

        origin_info_dict = {'covid19_airway': covid19_airway_origin,
                            'lidc_airway': lidc_airway_origin}

        spacing_info_dict = {'covid19_airway': covid19_airway_spacing,
                             'lidc_airway': lidc_airway_spacing}

        covid19_airway_data_dict = {
            'image': covid19_airway_image,
            'label': covid19_airway_label
        }
        lidc_airway_data_dict = {
            'image': lidc_airway_image,
            'label': lidc_airway_label
        }

        if self.isTrain:
            covid19_airway_data_dict = self.train_transform(covid19_airway_data_dict)
            lidc_airway_data_dict = self.train_transform(lidc_airway_data_dict)
        else:
            covid19_airway_data_dict = self.test_transform(covid19_airway_data_dict)
            lidc_airway_data_dict = self.test_transform(lidc_airway_data_dict)

        metadata_dict = {
            'covid19_airway_image': covid19_airway_data_dict[0]['image'],
            'covid19_airway_label': covid19_airway_data_dict[0]['label'],
            'lidc_airway_image': lidc_airway_data_dict[0]['image'],
            'lidc_airway_label': lidc_airway_data_dict[0]['label']
        }

        if isinstance(metadata_dict, dict):
            return (metadata_dict, origin_info_dict, spacing_info_dict)
        else:
            raise TypeError('meta data must be a list[dict] or dict')

    def normalize_ct(self, image):
        min_value = np.min(image)
        max_value = np.max(image)
        image = (image - min_value) / (max_value - min_value)
        return image
