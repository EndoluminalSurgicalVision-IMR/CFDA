# -*- coding: utf-8 -*-

import time
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat

from .base_model import BaseModel
from . import networks

from main_code.util.losses import Dice_with_Focal


class CFDAModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """
        Add new dataset-specific options, and rewrite default values for existing options.
        """
        parser.add_argument('--E_N_pretrain_filepath', type=str,
                            default='../checkpoints_models/SST/SSTNet/10_net_E_N.pth',
                            help='E_N_pretrain filepath')
        parser.add_argument('--use_BDE', type=int,
                            default=1,
                            help='0: DO NOT use the fixed Bias-Discriminative Encoder, 1: Use the fixed Bias-Discriminative Encoder')
        parser.add_argument('--lamda_fla', type=float,
                            default=0.1,
                            help='the balance term between the aug and non-aug features in the clean domain reconstruction')
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
        self.loss_names = ['clean_output', 'noisy_output']
        self.model_names = ['E_N', 'E_C', 'D_C', 'D_N']
        self.use_BDE = opt.use_BDE
        self.netE_N = networks.define_SSTV2_E_N(in_channels=opt.in_channels, out_channels=opt.out_channels,
                                                init_fmaps_degree=opt.init_fmaps_degree,
                                                fmaps_layer_number=opt.fmaps_layer_number, layer_order=opt.layer_order,
                                                final_sigmoid=opt.final_sigmoid, device=self.device)

        if (self.use_BDE):
            E_N_pretrained_filepath = opt.E_N_pretrain_filepath
            netE_N_pretrained_state_dict = torch.load(E_N_pretrained_filepath,
                                                      map_location=lambda storage, loc: storage.cuda(0))
            self.netE_N.load_state_dict(netE_N_pretrained_state_dict)
            print('netE_N load the pretrained model successfully!')

        self.netE_C = networks.define_CFDAV1_E_C(in_channels=opt.in_channels, out_channels=opt.out_channels,
                                                 init_fmaps_degree=opt.init_fmaps_degree,
                                                 fmaps_layer_number=opt.fmaps_layer_number, layer_order=opt.layer_order,
                                                 final_sigmoid=opt.final_sigmoid, device=self.device)

        self.netD_C = networks.define_CFDAV1_D_C(in_channels=opt.in_channels, out_channels=opt.out_channels,
                                                 init_fmaps_degree=opt.init_fmaps_degree,
                                                 fmaps_layer_number=opt.fmaps_layer_number, layer_order=opt.layer_order,
                                                 final_sigmoid=opt.final_sigmoid, device=self.device)

        self.netD_N = networks.define_CFDAV1_D_N(in_channels=opt.in_channels, out_channels=opt.out_channels,
                                                 init_fmaps_degree=opt.init_fmaps_degree,
                                                 fmaps_layer_number=opt.fmaps_layer_number, layer_order=opt.layer_order,
                                                 final_sigmoid=opt.final_sigmoid, device=self.device)

        self.use_fla = False
        self.lamda_fla = opt.lamda_fla

        if self.isTrain:
            # D_C : Decoder Clean
            # D_N : Decoder Noisy
            if (opt.loss_mode == 'DwF'):
                self.criterionD_C = Dice_with_Focal(sigmoid_normalization=1)
                self.criterionD_N = Dice_with_Focal(sigmoid_normalization=1)
            else:
                raise NameError(opt.loss_mode + 'is not specified!')
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            if (opt.optim_mode == 'Adam'):
                self.optimizer_CFDA = torch.optim.Adam(
                    params=itertools.chain(self.netE_N.parameters(), self.netE_C.parameters(), self.netD_C.parameters(),
                                           self.netD_N.parameters()), lr=opt.init_lr,
                    betas=(opt.adam_beta1, 0.999))

            else:
                raise NameError(opt.optim_mode + 'is not specified!')
            self.optimizers.append(self.optimizer_CFDA)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (tuple): include the data itself and its metadata information.

        """
        assert isinstance(input, tuple), 'input in the CFDA should be the tuple, and ' \
                                         'contains (meta data, origin_dict, spacing dict)'
        self.meta_data_dict, self.origin_info_dict, self.spacing_info_dict = input
        self.covid19_airway_image = self.meta_data_dict['covid19_airway_image'].to(self.device)
        self.covid19_airway_label = self.meta_data_dict['covid19_airway_label'].to(self.device)
        self.lidc_airway_image = self.meta_data_dict['lidc_airway_image'].to(self.device)
        self.lidc_airway_label = self.meta_data_dict['lidc_airway_label'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.covid19_airway_e_n_features = self.netE_N(self.covid19_airway_image)
        self.covid19_airway_e_c_features = self.netE_C(self.covid19_airway_image)
        self.lidc_airway_e_c_features = self.netE_C(self.lidc_airway_image)

        self.covid19_airway_fusion_features = []
        self.covid19_airway_aug_fusion_features = []
        self.lidc_airway_fusion_features = []
        if self.use_fla:
            for idx in range(0, len(self.covid19_airway_e_n_features)):
                self.covid19_airway_e_n_feature_flip = torch.flip(
                    self.covid19_airway_e_n_features[idx].clone().detach(), dims=[0])
                covid19_airway_fusion_feature = self.covid19_airway_e_n_feature_flip + self.covid19_airway_e_c_features[
                    idx]

                covid19_airway_aug_fusion_feature = self.covid19_airway_e_n_features[idx] + \
                                                    self.covid19_airway_e_c_features[idx]
                self.covid19_airway_fusion_features.append(covid19_airway_fusion_feature)
                self.covid19_airway_aug_fusion_features.append(covid19_airway_aug_fusion_feature)
                lidc_airway_fusion_feature = self.covid19_airway_e_n_features[idx] + self.lidc_airway_e_c_features[idx]
                self.lidc_airway_fusion_features.append(lidc_airway_fusion_feature)
            self.lidc_airway_prediction_aug = self.netD_C(self.lidc_airway_fusion_features)
            self.lidc_airway_prediction = self.netD_C(self.lidc_airway_e_c_features)
            self.covid19_airway_prediction = self.netD_N(self.covid19_airway_fusion_features)

        else:
            for i in range(0, len(self.covid19_airway_e_c_features)):
                covid19_airway_fusion_feature = self.covid19_airway_e_n_features[i] + self.covid19_airway_e_c_features[
                    i]
                self.covid19_airway_fusion_features.append(covid19_airway_fusion_feature)
            self.lidc_airway_prediction = self.netD_C(self.lidc_airway_e_c_features)
            self.covid19_airway_prediction = self.netD_N(self.covid19_airway_fusion_features)

    def backward(self):
        if self.use_fla:
            self.lidc_airway_background = 1. - self.lidc_airway_label
            self.lidc_airway_groundtruth = torch.cat((self.lidc_airway_background, self.lidc_airway_label), dim=1)
            self.loss_clean_output = self.criterionD_C(self.lidc_airway_prediction, self.lidc_airway_groundtruth,
                                                       self.lidc_airway_label.long()) + self.lamda_fla * self.criterionD_C(
                self.lidc_airway_prediction_aug, self.lidc_airway_groundtruth, self.lidc_airway_label.long())
            self.covid19_airway_background = 1. - self.covid19_airway_label
            self.covid19_airway_groundtruth = torch.cat((self.covid19_airway_background, self.covid19_airway_label),
                                                        dim=1)
            self.loss_noisy_output = self.criterionD_N(self.covid19_airway_prediction,
                                                       self.covid19_airway_groundtruth,
                                                       self.covid19_airway_label.long())
            self.total_loss = self.loss_clean_output + self.loss_noisy_output
            self.total_loss.backward()
        else:

            self.lidc_airway_background = 1. - self.lidc_airway_label
            self.lidc_airway_groundtruth = torch.cat((self.lidc_airway_background, self.lidc_airway_label), dim=1)
            self.loss_clean_output = self.criterionD_C(self.lidc_airway_prediction,
                                                       self.lidc_airway_groundtruth,
                                                       self.lidc_airway_label.long())
            self.covid19_airway_background = 1. - self.covid19_airway_label
            self.covid19_airway_groundtruth = torch.cat((self.covid19_airway_background, self.covid19_airway_label),
                                                        dim=1)
            self.loss_noisy_output = self.criterionD_N(self.covid19_airway_prediction,
                                                       self.covid19_airway_groundtruth,
                                                       self.covid19_airway_label.long())
            self.total_loss = self.loss_clean_output + self.loss_noisy_output
            self.total_loss.backward()

    def optimize_parameters(self):
        self.forward()
        if self.use_BDE:
            self.set_requires_grad(nets=self.netE_N, requires_grad=False)
        self.optimizer_CFDA.zero_grad()
        self.backward()
        self.optimizer_CFDA.step()

    ''' ================================================= Core Class Methods ======================================================='''

    ''' =============================================== Private Class Methods ======================================================'''

    def _set_use_fla_status(self, epoch, epoch_thresh):
        if epoch >= epoch_thresh:
            self.use_fla = True

    ''' =============================================== Private Class Methods ======================================================'''
