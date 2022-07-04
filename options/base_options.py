# -*- coding: utf-8 -*-


import argparse
import os
import sys
import torch

from main_code import models
from main_code import dataloader
from main_code.util import utils


class BaseOptions():
    """
    Function
    --------
        This class defines options used during both training and test time.

        It also implements several helper functions such as parsing, printing, and saving the options.

        It also gathers additional options defined in <modify_commandline_options> functions in both
        dataset class and model class. [for future implementation]
    """

    def __init__(self):
        """
        Function
        --------
            Reset the class; indicates the class hasn't been initailized
        """
        self.initialized = False

    def initialize(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--dataroot', default='...',
                            help='paths to the Task, e.g., Task606_LungTriplet, Task607_NCPAirway')
        parser.add_argument('--name', type=str, default='CFDA-V1',
                            help='the name of the experiment')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str,
                            default="../checkpoints_models/xxx",
                            help='models are saved here')
        # model parameters
        parser.add_argument('--model', type=str, default='CFDA',
                            help='chooses which model to use. [SST|CFDA]')
        parser.add_argument('--in_channels', type=int, default=1,
                            help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--out_channels', type=int, default=2,
                            help='# of output image channels: decided by the output classes.')
        parser.add_argument('--final_sigmoid', type=int, default=1,
                            help='Yes, use sigmoid in the test phase, No, use softmax. Equal when is binary segmentation')
        parser.add_argument('--init_fmaps_degree', type=int, default=16,
                            help='#the number of init_fmaps_degree channels')
        parser.add_argument('--layer_order', type=str, default='cip',
                            help='three functions permutation and combination, e.g. cip represents the conv-instancenorm-prelu')
        parser.add_argument('--fmaps_layer_number', type=int, default=4, help='the number of layers of the network')

        # dataset parameters
        parser.add_argument('--dataset_mode', type=str, default='crossdomainairway',
                            help='chooses how datasets are loaded. e.g., lungtriplet | crossdomainairway |')
        parser.add_argument('--num_threads', default=8, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=2, help='input batch size')

        # additional parameters
        parser.add_argument('--epoch', type=str, default='30',
                            help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='xxx', type=str,
                            help='customized suffix: opt.name = opt.name + suffix')

        self.initialized = True
        return parser

    def gather_options(self):
        """
        Function
        --------
            Initialize our parser with basic options(only once). Add additional model-specific and dataset-specific options.

            These options are defined in the <modify_commandline_options> function in model and dataset classes. [for future implementation]
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        '''Latter Supplementation'''
        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = dataloader.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)
        '''Latter Supplementation'''

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        utils.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """
        Function
        --------
            Parse our options, create checkpoints directory suffix, and set up gpu device.
        """
        opt = self.gather_options()
        opt.isTrain = self.isTrain  # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        # set multi steps, be careful with this, lr may be incorrect w/o this. Only use in the train options
        if self.isTrain:
            str_ids = opt.lr_decay_iters.split(',')
            opt.lr_decay_iters = []
            for str_id in str_ids:
                id = int(str_id)
                if id >= 0:
                    opt.lr_decay_iters.append(id)

        self.opt = opt
        return self.opt
