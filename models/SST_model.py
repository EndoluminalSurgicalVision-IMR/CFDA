# -*- coding: utf-8 -*-


import time
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks

from main_code.util.losses import Dice_with_Focal


class SSTModel(BaseModel):
    """
     This class implements the SST model, for discriminative covid-19 feature encoder learning.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """
        Add new dataset-specific options, and rewrite default values for existing options.
        """
        parser.add_argument('--lamda_tfl', type=float, default=1.0,
                            help='lamda coefficient for the triplet feature loss')
        parser.add_argument('--lamda_lsl', type=float, default=1.0,
                            help='lamda coefficient for the lesion segmentation loss')
        parser.add_argument('--triplet_margin', type=float, default=3.0,
                            help='margin for the triplet loss function')
        return parser

    ''' ================================================= Core Class Methods ======================================================='''

    def __init__(self, opt):
        """
            Initialize the model.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['triplet_feature', 'lesion_segmentation']
        self.model_names = ['E_N', 'F_D', 'D_L']

        self.netE_N = networks.define_SSTV2_E_N(in_channels=opt.in_channels, out_channels=opt.out_channels,
                                                init_fmaps_degree=opt.init_fmaps_degree,
                                                fmaps_layer_number=opt.fmaps_layer_number, layer_order=opt.layer_order,
                                                final_sigmoid=opt.final_sigmoid, device=self.device)
        self.netF_D = networks.define_SSTV2_F_D(init_fmaps_degree=opt.init_fmaps_degree,
                                                fmaps_layer_number=opt.fmaps_layer_number, device=self.device)
        self.netD_L = networks.define_SSTV2_D_L(in_channels=opt.in_channels, out_channels=opt.out_channels,
                                                init_fmaps_degree=opt.init_fmaps_degree,
                                                fmaps_layer_number=opt.fmaps_layer_number, layer_order=opt.layer_order,
                                                final_sigmoid=opt.final_sigmoid, device=self.device)

        if self.isTrain:
            self.criterionTPL = nn.TripletMarginLoss(margin=opt.triplet_margin, p=2)
            self.criterionLSL = Dice_with_Focal(sigmoid_normalization=1)
            self.lamda_tfl = opt.lamda_tfl
            self.lamda_lsl = opt.lamda_lsl
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            if (opt.optim_mode == 'Adam'):
                self.optimizer_SST = torch.optim.Adam(
                    params=itertools.chain(self.netE_N.parameters(), self.netF_D.parameters(),
                                           self.netD_L.parameters()), lr=opt.init_lr, betas=(opt.adam_beta1, 0.999))
            elif (opt.optim_mode == 'SGD'):
                self.optimizer_SST = torch.optim.SGD(
                    params=itertools.chain(self.netE_N.parameters(), self.netF_D.parameters(),
                                           self.netD_L.parameters()), lr=opt.init_lr, momentum=opt.sgd_momentum,
                    weight_decay=opt.sgd_weight_decay)
            else:
                raise NameError(opt.optim_mode + 'is not specified!')
            self.optimizers.append(self.optimizer_SST)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (tuple): include the data itself and its metadata information.

        """
        assert isinstance(input, tuple), 'input in the SSTModel should be the tuple, and ' \
                                         'contains (meta data, origin_dict, spacing dict)'
        self.meta_data_dict, self.origin_info_dict, self.spacing_info_dict = input
        self.covid19_airway_image = self.meta_data_dict['covid19_airway_image'].to(self.device)
        self.covid19_lesion_image = self.meta_data_dict['covid19_lesion_image'].to(self.device)
        self.covid19_lesion_label = self.meta_data_dict['covid19_lesion_label'].to(self.device)
        self.lidc_image = self.meta_data_dict['lidc_image'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.covid19_lesion_encodefeatures = self.netE_N(self.covid19_lesion_image)
        self.covid19_airway_encodefeature = self.netE_N(self.covid19_airway_image, return_intermediate_features=False)
        self.lidc_encodefeature = self.netE_N(self.lidc_image, return_intermediate_features=False)
        self.covid19_lesion_latentvector, self.covid19_airway_latentvector, self.lidc_latentvector = self.netF_D(
            self.covid19_lesion_encodefeatures[0], self.covid19_airway_encodefeature, self.lidc_encodefeature)
        self.covid19_lesion_prediction = self.netD_L(self.covid19_lesion_encodefeatures)

    def backward(self):
        self.covid19_lesion_latentvector = F.normalize(self.covid19_lesion_latentvector)
        self.covid19_airway_latentvector = F.normalize(self.covid19_airway_latentvector)
        self.lidc_latentvector = F.normalize(self.lidc_latentvector)

        self.loss_triplet_feature = self.criterionTPL(anchor=self.covid19_lesion_latentvector,
                                                      positive=self.covid19_airway_latentvector,
                                                      negative=self.lidc_latentvector)
        self.covid19_lesion_background = 1. - self.covid19_lesion_label
        self.covid19_lesion_groundtruth = torch.cat((self.covid19_lesion_background, self.covid19_lesion_label), dim=1)
        self.loss_lesion_segmentation = self.criterionLSL(self.covid19_lesion_prediction,
                                                          self.covid19_lesion_groundtruth,
                                                          self.covid19_lesion_label.long())
        self.total_loss = self.lamda_tfl * self.loss_triplet_feature + self.lamda_lsl * self.loss_lesion_segmentation
        self.total_loss.backward()

    def fetch_model(self, name):
        return getattr(self, 'net' + name)

    def optimize_parameters(self):
        self.forward()
        self.optimizer_SST.zero_grad()
        self.backward()
        self.optimizer_SST.step()

    ''' ================================================= Core Class Methods ======================================================='''
