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


class LungTripletDataset(BaseDataset):
    '''
        LungTripletDataset
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
        parser.add_argument('--in_covid19lesion', type=str, default='COVID19_Infection', help='filefolder name')
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
        self.covid19lesion_rootdir = utils.getabspath([self.rootdir, opt.in_covid19lesion])
        self.lidc_rootdir = utils.getabspath([self.rootdir, opt.in_lidc])

        self.covid19airway_filelist = os.listdir(self.covid19airway_rootdir)
        self.covid19airway_filelist = natsorted(self.covid19airway_filelist)
        self.covid19lesion_filelist = os.listdir(self.covid19lesion_rootdir)
        self.covid19lesion_filelist = natsorted(self.covid19lesion_filelist)
        self.lidc_filelist = os.listdir(self.lidc_rootdir)
        self.lidc_filelist = natsorted(self.lidc_filelist)
        set_determinism(seed=777)

        self.train_transform = Compose(
            [
                AddChanneld(
                    keys=["covid19_airway_image", "covid19_lesion_image", "covid19_lesion_label", "lidc_image"]),
                Orientationd(
                    keys=["covid19_airway_image", "covid19_lesion_image", "covid19_lesion_label", "lidc_image"],
                    axcodes='RAS'),
                RandCropByLabelClassesd(keys=["covid19_lesion_image", "covid19_lesion_label"],
                                        label_key="covid19_lesion_label",
                                        spatial_size=[opt.depth, opt.width, opt.height],
                                        ratios=[0.2, 0.8], num_classes=2, num_samples=1),
                RandSpatialCropd(keys=["covid19_airway_image"],
                                 roi_size=(opt.depth, opt.width, opt.height), random_size=False),
                RandSpatialCropd(keys=["lidc_image"],
                                 roi_size=(opt.depth, opt.width, opt.height), random_size=False),
                RandFlipd(keys=["covid19_airway_image", "covid19_lesion_image", "covid19_lesion_label", "lidc_image"],
                          prob=0.5, spatial_axis=(-2, -1)),
                RandRotated(keys=["covid19_airway_image", "covid19_lesion_image", "covid19_lesion_label", "lidc_image"],
                            prob=1.0, range_x=(-0.174, 0.174), range_y=(-0.174, 0.174), range_z=(-0.174, 0.174),
                            mode=["bilinear", "bilinear", "nearest", "bilinear"]),
                ToTensord(keys=["covid19_airway_image", "covid19_lesion_image", "covid19_lesion_label", "lidc_image"],
                          dtype=torch.float)
            ]
        )
        self.test_transform = Compose(
            [
                AddChanneld(
                    keys=["covid19_airway_image", "covid19_lesion_image", "covid19_lesion_label", "lidc_image"]),
                Orientationd(
                    keys=["covid19_airway_image", "covid19_lesion_image", "covid19_lesion_label", "lidc_image"],
                    axcodes='RAS'),
                ToTensord(keys=["covid19_airway_image", "covid19_lesion_image", "covid19_lesion_label", "lidc_image"],
                          dtype=torch.float)
            ]
        )

    def __len__(self):
        return len(self.covid19lesion_filelist)

    def __getitem__(self, idx):
        idx_lidc = random.randint(0, len(self.lidc_filelist) - 1)
        covid19_airway_casename = self.covid19airway_filelist[idx]
        covid19_airway_image, covid19_airway_origin, covid19_airway_spacing = utils.load_itk_image(utils.getabspath(
            [self.covid19airway_rootdir, covid19_airway_casename, covid19_airway_casename + "_clean.nii.gz"]))
        covid19_airway_image = self.normalize_ct(covid19_airway_image)

        covid19_lesion_casename = self.covid19lesion_filelist[idx]
        covid19_lesion_image, covid19_lesion_origin, covid19_lesion_spacing = utils.load_itk_image(utils.getabspath(
            [self.covid19lesion_rootdir, covid19_lesion_casename, covid19_lesion_casename + '_clean.nii.gz']))
        covid19_lesion_image = self.normalize_ct(covid19_lesion_image)

        covid19_lesion_label, _, _ = utils.load_itk_image(utils.getabspath(
            [self.covid19lesion_rootdir, covid19_lesion_casename, covid19_lesion_casename + '_label.nii.gz']))
        covid19_lesion_label = covid19_lesion_label.astype(np.uint8)

        lidc_casename = self.lidc_filelist[idx_lidc]
        lidc_image, lidc_origin, lidc_spacing = utils.load_itk_image(
            utils.getabspath([self.lidc_rootdir, lidc_casename, lidc_casename + "_clean.nii.gz"]))
        lidc_image = self.normalize_ct(lidc_image)

        origin_info_dict = {'covid19_airway': covid19_airway_origin,
                            'covid19_lesion': covid19_lesion_origin,
                            'lidc': lidc_origin}

        spacing_info_dict = {'covid19_airway': covid19_airway_spacing,
                             'covid19_lesion': covid19_lesion_spacing,
                             'lidc': lidc_spacing}

        metadata_dict = {
            'covid19_airway_image': covid19_airway_image,
            'covid19_lesion_image': covid19_lesion_image,
            'covid19_lesion_label': covid19_lesion_label,
            'lidc_image': lidc_image
        }

        if self.isTrain:
            metadata_dict = self.train_transform(metadata_dict)
        else:
            metadata_dict = self.test_transform(metadata_dict)

        if isinstance(metadata_dict, list) and isinstance(metadata_dict[0], dict):
            return (metadata_dict[0], origin_info_dict, spacing_info_dict)
        elif isinstance(metadata_dict, dict):
            return (metadata_dict, origin_info_dict, spacing_info_dict)
        else:
            raise TypeError('meta data must be a list[dict] or dict')

    def normalize_ct(self, image):
        min_value = np.min(image)
        max_value = np.max(image)
        image = (image - min_value) / (max_value - min_value)
        return image



