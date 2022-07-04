# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
from torch.optim import lr_scheduler
from torch.utils.checkpoint import checkpoint
from torch.utils.checkpoint import checkpoint_sequential
from einops.layers.torch import Rearrange

import numpy as np
import functools

from .util import swish, Mish

"""************************************* Define Networks *****************************"""


def define_SSTV2_E_N(in_channels, out_channels, init_fmaps_degree, fmaps_layer_number, layer_order, final_sigmoid,
                     device):
    net = SSTV2_E_N(in_channels, out_channels, init_fmaps_degree, fmaps_layer_number, layer_order, final_sigmoid)
    init_weights(net, init_type='kaiming')
    net = net.to(device)
    return net


def define_SSTV2_F_D(init_fmaps_degree, fmaps_layer_number, device):
    net = SSTV2_F_D(init_fmaps_degree, fmaps_layer_number)
    init_weights(net, init_type='kaiming')
    net = net.to(device)
    return net


def define_SSTV2_D_L(in_channels, out_channels, init_fmaps_degree, fmaps_layer_number, layer_order, final_sigmoid,
                     device):
    net = SSTV2_D_L(in_channels, out_channels, init_fmaps_degree, fmaps_layer_number, layer_order, final_sigmoid)
    init_weights(net, init_type='kaiming')
    net = net.to(device)
    return net


def define_CFDAV1_E_C(in_channels, out_channels, init_fmaps_degree, fmaps_layer_number, layer_order, final_sigmoid,
                      device):
    net = CFDAV1_E_C(in_channels, out_channels, init_fmaps_degree, fmaps_layer_number, layer_order, final_sigmoid)
    init_weights(net, init_type='kaiming')
    net = net.to(device)
    return net


def define_CFDAV1_D_C(in_channels, out_channels, init_fmaps_degree, fmaps_layer_number, layer_order, final_sigmoid,
                      device):
    net = CFDAV1_D_C(in_channels, out_channels, init_fmaps_degree, fmaps_layer_number, layer_order, final_sigmoid)
    init_weights(net, init_type='kaiming')
    net = net.to(device)
    return net


def define_CFDAV1_D_N(in_channels, out_channels, init_fmaps_degree, fmaps_layer_number, layer_order, final_sigmoid,
                      device):
    net = CFDAV1_D_N(in_channels, out_channels, init_fmaps_degree, fmaps_layer_number, layer_order, final_sigmoid)
    init_weights(net, init_type='kaiming')
    net = net.to(device)
    return net

"""************************************* Define Networks *****************************"""

###############################################################################
# Common Helpful Functions
###############################################################################

"""************************************* Common Helpful Functions *****************************"""


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=int(opt.lr_decay_iters), gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    elif opt.lr_policy == 'multistep':
        milestones = []
        for (index, step) in enumerate(opt.lr_decay_iters):
            milestones.append(int(step))
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, init_gain=init_gain)
    return net


"""************************************* Common Helpful Functions *****************************"""

###############################################################################
# Common  Conv   Functions
###############################################################################

"""************************************* Common  Conv   Functions *****************************"""


def create_feature_maps(init_channel_number, number_of_fmaps):
    return [init_channel_number * 2 ** k for k in range(number_of_fmaps)]
    # return [32,64,128,256]


def create_conv(in_channels, out_channels, kernel_size, order='cri', GroupNumber=8, padding=1, stride=1, Deconv=False):
    """
    @Compiled by zmh
    create an ordered convlution layer for the UNet

    :param in_channels:
    :param out_channels:
    :param kernel_size:
    :param layer_order: the order of layer common match :
                        'cr'-> conv+relu
                        'crb'->conv+relu+batchnorm
                        'crg'->conv+relu+groupnorm(groupnorm number is designed)
                        ......
    :param GroupNumber:
    :param Padding:
    :return:
    """
    assert 'c' in order, 'Convolution must have a conv operation'
    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(negative_slope=0.1, inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'p':
            modules.append(('pReLU', nn.PReLU(num_parameters=out_channels)))
        elif char == 's':
            modules.append(('swish', swish()))
        elif char == 'm':
            modules.append(('Mish', Mish()))
        elif char == 'c':
            if not Deconv:
                bias = not ('g' in order or 'b' in order)
                modules.append(('conv', nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias, padding=padding,
                                                  stride=stride)))
            else:
                bias = not ('g' in order or 'b' in order)
                modules.append(('convtranspose3d',
                                nn.ConvTranspose3d(in_channels, out_channels, kernel_size, bias=bias, padding=padding,
                                                   stride=stride)))
        elif char == 'g':
            is_before_conv = i < order.index('c')
            assert not is_before_conv, 'GroupNorm MUST go after the Conv3d'
            if out_channels < GroupNumber:
                GroupNumber = out_channels
            modules.append(('groupnorm', nn.GroupNorm(num_groups=GroupNumber, num_channels=out_channels)))
        elif char == 'i':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('instancenorm', nn.InstanceNorm3d(in_channels, affine=True)))
            else:
                modules.append(('instancenorm', nn.InstanceNorm3d(out_channels, affine=True)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('batchnorm', nn.BatchNorm3d(in_channels)))
            else:
                modules.append(('batchnorm', nn.BatchNorm3d(out_channels)))
        else:
            raise ValueError("Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c','i']")

    return modules


# General SingleConv
class SingleConv(nn.Sequential):
    """
    Basic convolutional module consisting of a Conv3d, non-linearity and optional batchnorm/groupnorm.
    The order of operations can be specified via the `order` parameter

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        order (string): determines the order of layers  default crb
        GroupNumber (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, kernel_size, order='crg', GroupNumber=8):
        super(SingleConv, self).__init__()
        for name, module in create_conv(in_channels, out_channels, kernel_size, order, GroupNumber):
            self.add_module(name, module)


# General Doubleconv
class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, encoder, order='crg', GroupNumber=8):
        super(DoubleConv, self).__init__()
        # Encoder
        if encoder:
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if (conv1_out_channels < in_channels):
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        # Decoder
        else:
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # Conv1:
        self.add_module(name='Conv1', module=SingleConv(in_channels=conv1_in_channels,
                                                        out_channels=conv1_out_channels,
                                                        kernel_size=kernel_size,
                                                        order=order,
                                                        GroupNumber=GroupNumber))
        # Conv2:
        self.add_module(name='Conv2', module=SingleConv(in_channels=conv2_in_channels,
                                                        out_channels=conv2_out_channels,
                                                        kernel_size=kernel_size,
                                                        order=order,
                                                        GroupNumber=GroupNumber))


# SingleConvOnce
class SingleConvOnce(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, encoder, order='crg', GroupNumber=8):
        super(SingleConvOnce, self).__init__()
        # Conv1:
        self.add_module(name='Conv1', module=SingleConv(in_channels=in_channels,
                                                        out_channels=out_channels,
                                                        kernel_size=kernel_size,
                                                        order=order,
                                                        GroupNumber=GroupNumber))


"""************************************* Common  Conv   Functions *****************************"""

###############################################################################
# NN Module Classes
###############################################################################
"""*************************************    NN Module Classes    *****************************"""


class _GridAttentionBlock3D(nn.Module):
    "Method from Attention U-Net"

    def __init__(self, in_channels, inter_channels, gate_channels):
        super(_GridAttentionBlock3D, self).__init__()
        self.theta = nn.Conv3d(in_channels=in_channels, out_channels=inter_channels, kernel_size=2, stride=2, padding=0,
                               bias=False)
        # self.Phi^T processes the gate signal: g [C 2D 2H 2W]
        self.phi = nn.Conv3d(in_channels=gate_channels, out_channels=inter_channels, kernel_size=1, stride=1, padding=0,
                             bias=True)
        # self.Psi processes the inter signal: g [interchannel D H W]
        self.psi = nn.Conv3d(in_channels=inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        self.final_conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm3d(in_channels, affine=True))


    def forward(self, x, g):
        input_size = x.size()
        theta_x = self.theta(x)
        phi_g = self.phi(g)
        f = F.relu(theta_x + phi_g, inplace=True)
        sigm_psi_f = torch.sigmoid(self.psi(f))
        sigm_psi_f = F.interpolate(sigm_psi_f, size=input_size[2:], mode='trilinear')
        y = sigm_psi_f.expand_as(x) * x
        y = self.final_conv(y)
        # if not self.training:
        #     self.feature1 = sigm_psi_f.detach()
        return y, sigm_psi_f


class SSTNet_Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pool_kernelsize=(2, 2, 2), pooling_type='max',
                 apply_pooling=True, Basic_Module=DoubleConv, order='cip'):
        super(SSTNet_Encoder, self).__init__()
        assert pooling_type in ['max', 'avg'], 'Pooling_Type must be max or avg'
        if apply_pooling:
            if pooling_type == 'max':
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernelsize)
            else:
                self.pooling = nn.AvgPool3d(kernel_size=pool_kernelsize)
        else:
            self.pooling = None
        if (Basic_Module == DoubleConv) or (Basic_Module == SingleConvOnce):
            self.basic_module = Basic_Module(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=kernel_size,
                                             encoder=True, order=order)

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x


class SSTNet_Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pool_kernelsize=(2, 2, 2), Basic_Module=DoubleConv,
                 order='cip', return_att_map=False):
        super(SSTNet_Decoder, self).__init__()
        if (Basic_Module == DoubleConv or Basic_Module == SingleConvOnce):
            self.upsample = None
        self.return_att_map = return_att_map
        self.basic_module = Basic_Module(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         encoder=False, order=order)
        self.attention_module = _GridAttentionBlock3D(in_channels=out_channels, inter_channels=out_channels // 2,
                                                      gate_channels=out_channels * 2)

    def forward(self, encoder_feature, x):
        if self.upsample is None:
            output_size = encoder_feature.size()[2:]
            encoder_feature, attention_map = self.attention_module(encoder_feature, x)
            x = F.interpolate(input=x, size=output_size, mode='trilinear', align_corners=False)
            x = torch.cat((encoder_feature, x), dim=1)
        x = self.basic_module(x)
        if self.return_att_map:
            return x, attention_map
        else:
            return x


class SSTNet(nn.Module):
    def __init__(self, in_channels, out_channels, init_fmaps_degree, fmaps_layer_number, layer_order, final_sigmoid,
                 **kwargs):
        super(SSTNet, self).__init__()
        assert isinstance(init_fmaps_degree, int), 'fmaps_degree must be an integer!'
        fmaps_list = create_feature_maps(init_fmaps_degree, fmaps_layer_number)
        self.EncoderLayer1 = SSTNet_Encoder(in_channels=in_channels, out_channels=fmaps_list[0], apply_pooling=False,
                                            Basic_Module=SingleConvOnce, order=layer_order)
        self.EncoderLayer2 = SSTNet_Encoder(in_channels=fmaps_list[0], out_channels=fmaps_list[1], apply_pooling=True,
                                            Basic_Module=DoubleConv, order=layer_order)
        self.EncoderLayer3 = SSTNet_Encoder(in_channels=fmaps_list[1], out_channels=fmaps_list[2], apply_pooling=True,
                                            Basic_Module=DoubleConv, order=layer_order)
        self.EncoderLayer4 = SSTNet_Encoder(in_channels=fmaps_list[2], out_channels=fmaps_list[3], apply_pooling=True,
                                            Basic_Module=DoubleConv, order=layer_order)
        decoder_fmaps_list = list(reversed(fmaps_list))
        self.DecoderLayer1 = SSTNet_Decoder(in_channels=decoder_fmaps_list[0] + decoder_fmaps_list[1],
                                            out_channels=decoder_fmaps_list[1], Basic_Module=DoubleConv,
                                            order=layer_order, return_att_map=False)
        self.DecoderLayer2 = SSTNet_Decoder(in_channels=decoder_fmaps_list[1] + decoder_fmaps_list[2],
                                            out_channels=decoder_fmaps_list[2], Basic_Module=DoubleConv,
                                            order=layer_order, return_att_map=False)
        self.DecoderLayer3 = SSTNet_Decoder(in_channels=decoder_fmaps_list[2] + decoder_fmaps_list[3],
                                            out_channels=decoder_fmaps_list[3], Basic_Module=DoubleConv,
                                            order=layer_order, return_att_map=True)
        self.final_conv = nn.Conv3d(in_channels=fmaps_list[0], out_channels=out_channels, kernel_size=1)

        self.latent_spatial_size = 2
        self.latent_feature_size = self.latent_spatial_size ** 3 * fmaps_list[-1]
        self.LatentProjection1 = nn.Sequential(nn.AdaptiveAvgPool3d(output_size=(self.latent_spatial_size)),
                                               # flattening
                                               Rearrange('b c d h w -> b (c d h w)'),
                                               nn.Linear(in_features=self.latent_feature_size, out_features=512),
                                               nn.Linear(in_features=512, out_features=128))
        self.LatentProjection2 = nn.Sequential(nn.AdaptiveAvgPool3d(output_size=(self.latent_spatial_size)),
                                               # flattening
                                               Rearrange('b c d h w -> b (c d h w)'),
                                               nn.Linear(in_features=self.latent_feature_size, out_features=512),
                                               nn.Linear(in_features=512, out_features=128))
        self.LatentProjection3 = nn.Sequential(nn.AdaptiveAvgPool3d(output_size=(self.latent_spatial_size)),
                                               # flattening
                                               Rearrange('b c d h w -> b (c d h w)'),
                                               nn.Linear(in_features=self.latent_feature_size, out_features=512),
                                               nn.Linear(in_features=512, out_features=128))

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3):
        x1_encoder_features = []
        x1_e1 = self.EncoderLayer1(x1)
        x1_e2 = self.EncoderLayer2(x1_e1)
        x1_e3 = self.EncoderLayer3(x1_e2)
        x1_e4 = self.EncoderLayer4(x1_e3)
        x1_e4_proj = self.LatentProjection1(x1_e4)
        x1_encoder_features.extend([x1_e3, x1_e2, x1_e1])

        x2 = self.EncoderLayer1(x2)
        x2 = self.EncoderLayer2(x2)
        x2 = self.EncoderLayer3(x2)
        x2 = self.EncoderLayer4(x2)
        x2 = self.LatentProjection2(x2)

        x3 = self.EncoderLayer1(x3)
        x3 = self.EncoderLayer2(x3)
        x3 = self.EncoderLayer3(x3)
        x3 = self.EncoderLayer4(x3)
        x3 = self.LatentProjection3(x3)

        x1_decoder_feature = self.DecoderLayer1(x1_encoder_features[0], x1_e4)
        x1_decoder_feature = self.DecoderLayer2(x1_encoder_features[1], x1_decoder_feature)
        x1_decoder_feature, att_map_d3 = self.DecoderLayer3(x1_encoder_features[2], x1_decoder_feature)
        x1_decoder_feature = self.final_conv(x1_decoder_feature)
        if not self.training:
            x1_decoder_feature = self.final_activation(x1_decoder_feature)
        return x1_e4_proj, x2, x3, x1_decoder_feature


class SSTV2_E_N(nn.Module):
    '''
    Semi-Supervised-Transfer Network for the
    '''

    def __init__(self, in_channels, out_channels, init_fmaps_degree, fmaps_layer_number, layer_order, final_sigmoid,
                 **kwargs):
        super(SSTV2_E_N, self).__init__()
        assert isinstance(init_fmaps_degree, int), 'fmaps_degree must be an integer!'
        fmaps_list = create_feature_maps(init_fmaps_degree, fmaps_layer_number)
        self.EN1 = SSTNet_Encoder(in_channels=in_channels, out_channels=fmaps_list[0], apply_pooling=False,
                                  Basic_Module=SingleConvOnce, order=layer_order)
        self.EN2 = SSTNet_Encoder(in_channels=fmaps_list[0], out_channels=fmaps_list[1], apply_pooling=True,
                                  Basic_Module=DoubleConv, order=layer_order)
        self.EN3 = SSTNet_Encoder(in_channels=fmaps_list[1], out_channels=fmaps_list[2], apply_pooling=True,
                                  Basic_Module=DoubleConv, order=layer_order)
        self.EN4 = SSTNet_Encoder(in_channels=fmaps_list[2], out_channels=fmaps_list[3], apply_pooling=True,
                                  Basic_Module=DoubleConv, order=layer_order)

    def forward(self, x, return_intermediate_features=True):
        if return_intermediate_features:
            x_encoder_features = []
            x_e1 = self.EN1(x)
            x_e2 = self.EN2(x_e1)
            x_e3 = self.EN3(x_e2)
            x_e4 = self.EN4(x_e3)
            x_encoder_features.extend([x_e4, x_e3, x_e2, x_e1])
            return x_encoder_features
        else:
            x = self.EN1(x)
            x = self.EN2(x)
            x = self.EN3(x)
            x = self.EN4(x)
            return x


class SSTV2_F_D(nn.Module):
    def __init__(self, init_fmaps_degree, fmaps_layer_number):
        super(SSTV2_F_D, self).__init__()
        fmaps_list = create_feature_maps(init_fmaps_degree, fmaps_layer_number)
        self.latent_spatial_size = 2
        self.latent_feature_size = self.latent_spatial_size ** 3 * fmaps_list[-1]
        self.LatentProjection1 = nn.Sequential(nn.AdaptiveAvgPool3d(output_size=(self.latent_spatial_size)),
                                               # flattening
                                               Rearrange('b c d h w -> b (c d h w)'),
                                               nn.Linear(in_features=self.latent_feature_size, out_features=512),
                                               nn.Linear(in_features=512, out_features=128))
        self.LatentProjection2 = nn.Sequential(nn.AdaptiveAvgPool3d(output_size=(self.latent_spatial_size)),
                                               # flattening
                                               Rearrange('b c d h w -> b (c d h w)'),
                                               nn.Linear(in_features=self.latent_feature_size, out_features=512),
                                               nn.Linear(in_features=512, out_features=128))
        self.LatentProjection3 = nn.Sequential(nn.AdaptiveAvgPool3d(output_size=(self.latent_spatial_size)),
                                               # flattening
                                               Rearrange('b c d h w -> b (c d h w)'),
                                               nn.Linear(in_features=self.latent_feature_size, out_features=512),
                                               nn.Linear(in_features=512, out_features=128))

    def forward(self, x1, x2, x3):
        x1 = self.LatentProjection1(x1)
        x2 = self.LatentProjection2(x2)
        x3 = self.LatentProjection3(x3)
        return x1, x2, x3


class SSTV2_D_L(nn.Module):
    '''
    Decode the COVID-19 Lesion Part.
    '''

    def __init__(self, in_channels, out_channels, init_fmaps_degree, fmaps_layer_number, layer_order, final_sigmoid,
                 **kwargs):
        super(SSTV2_D_L, self).__init__()
        fmaps_list = create_feature_maps(init_fmaps_degree, fmaps_layer_number)
        decoder_fmaps_list = list(reversed(fmaps_list))
        self.DL1 = SSTNet_Decoder(in_channels=decoder_fmaps_list[0] + decoder_fmaps_list[1],
                                  out_channels=decoder_fmaps_list[1], Basic_Module=DoubleConv,
                                  order=layer_order, return_att_map=False)
        self.DL2 = SSTNet_Decoder(in_channels=decoder_fmaps_list[1] + decoder_fmaps_list[2],
                                  out_channels=decoder_fmaps_list[2], Basic_Module=DoubleConv,
                                  order=layer_order, return_att_map=False)
        self.DL3 = SSTNet_Decoder(in_channels=decoder_fmaps_list[2] + decoder_fmaps_list[3],
                                  out_channels=decoder_fmaps_list[3], Basic_Module=DoubleConv,
                                  order=layer_order, return_att_map=True)
        self.final_conv = nn.Conv3d(in_channels=fmaps_list[0], out_channels=out_channels, kernel_size=1)
        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x_encodefeatures):
        x = self.DL1(x_encodefeatures[1], x_encodefeatures[0])
        x = self.DL2(x_encodefeatures[2], x)
        x, att_map_d3 = self.DL3(x_encodefeatures[3], x)
        x = self.final_conv(x)
        if not self.training:
            x = self.final_activation(x)
        return x


# Encoder
class Common_Encoder(nn.Module):
    """
    Encode the network
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, pool_kernelsize=(2, 2, 2),
                 pooling_type='max', apply_pooling=True, Basic_Module=DoubleConv, order='crg', GroupNumber=8):
        super(Common_Encoder, self).__init__()
        assert pooling_type in ['max', 'avg'], 'Pooling_Type must be max or avg'
        if apply_pooling:
            if pooling_type == 'max':
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernelsize)
            else:
                self.pooling = nn.AvgPool3d(kernel_size=pool_kernelsize)
        else:
            self.pooling = None
        if (Basic_Module == DoubleConv) or (Basic_Module == SingleConvOnce):
            self.basic_module = Basic_Module(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=kernel_size,
                                             encoder=True, order=order,
                                             GroupNumber=GroupNumber)

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x


# Decoder
class Common_Decoder(nn.Module):
    """
    Decode the network
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, pool_kernelsize=(2, 2, 2),
                 Basic_Module=DoubleConv, order='crb', GroupNumber=8):
        super(Common_Decoder, self).__init__()
        if (Basic_Module == DoubleConv or Basic_Module == SingleConvOnce):
            self.upsample = None
        self.basic_module = Basic_Module(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         encoder=False, order=order)

    def forward(self, encoder_feature, x):
        if self.upsample is None:
            output_size = encoder_feature.size()[2:]
            x = F.interpolate(input=x, size=output_size, mode='trilinear', align_corners=True)
            x = torch.cat((encoder_feature, x), dim=1)
        x = self.basic_module(x)
        return x


class CFDAV1_E_C(nn.Module):
    '''
        Encoder Clean for the Collaborative Feature Disentanglement and Aggregation (CFDA)
    '''

    def __init__(self, in_channels, out_channels, init_fmaps_degree, fmaps_layer_number, layer_order, final_sigmoid,
                 **kwargs):
        super(CFDAV1_E_C, self).__init__()
        assert isinstance(init_fmaps_degree, int), 'fmaps_degree must be an integer!'
        fmaps_list = create_feature_maps(init_fmaps_degree, fmaps_layer_number)
        self.EC1 = Common_Encoder(in_channels=in_channels, out_channels=fmaps_list[0], apply_pooling=False,
                                  Basic_Module=SingleConvOnce, order=layer_order)
        self.EC2 = Common_Encoder(in_channels=fmaps_list[0], out_channels=fmaps_list[1], apply_pooling=True,
                                  Basic_Module=DoubleConv, order=layer_order)
        self.EC3 = Common_Encoder(in_channels=fmaps_list[1], out_channels=fmaps_list[2], apply_pooling=True,
                                  Basic_Module=DoubleConv, order=layer_order)
        self.EC4 = Common_Encoder(in_channels=fmaps_list[2], out_channels=fmaps_list[3], apply_pooling=True,
                                  Basic_Module=DoubleConv, order=layer_order)

    def forward(self, x, return_intermediate_features=True):
        if return_intermediate_features:
            x_encoder_features = []
            x_e1 = self.EC1(x)
            x_e2 = self.EC2(x_e1)
            x_e3 = self.EC3(x_e2)
            x_e4 = self.EC4(x_e3)
            x_encoder_features.extend([x_e4, x_e3, x_e2, x_e1])
            return x_encoder_features
        else:
            # Encoder the x2
            x = self.EC1(x)
            x = self.EC2(x)
            x = self.EC3(x)
            x = self.EC4(x)
            return x


class CFDAV1_D_C(nn.Module):
    '''
        Decoder Clean for the Collaborative Feature Disentanglement and Aggregation (CFDA)
    '''

    def __init__(self, in_channels, out_channels, init_fmaps_degree, fmaps_layer_number, layer_order, final_sigmoid,
                 **kwargs):
        super(CFDAV1_D_C, self).__init__()
        assert isinstance(init_fmaps_degree, int), 'fmaps_degree must be an integer!'
        fmaps_list = create_feature_maps(init_fmaps_degree, fmaps_layer_number)
        decoder_fmaps_list = list(reversed(fmaps_list))
        self.DC1 = Common_Decoder(in_channels=decoder_fmaps_list[0] + decoder_fmaps_list[1],
                                  out_channels=decoder_fmaps_list[1], Basic_Module=DoubleConv,
                                  order=layer_order)
        self.DC2 = Common_Decoder(in_channels=decoder_fmaps_list[1] + decoder_fmaps_list[2],
                                  out_channels=decoder_fmaps_list[2], Basic_Module=DoubleConv,
                                  order=layer_order)
        self.DC3 = Common_Decoder(in_channels=decoder_fmaps_list[2] + decoder_fmaps_list[3],
                                  out_channels=decoder_fmaps_list[3], Basic_Module=SingleConvOnce,
                                  order=layer_order)
        self.final_conv = nn.Conv3d(in_channels=fmaps_list[0], out_channels=out_channels, kernel_size=1)
        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x_encodefeatures):
        x = self.DC1(x_encodefeatures[1], x_encodefeatures[0])
        x = self.DC2(x_encodefeatures[2], x)
        x = self.DC3(x_encodefeatures[3], x)
        x = self.final_conv(x)
        if not self.training:
            x = self.final_activation(x)
        return x


class CFDAV1_D_N(nn.Module):
    '''
        Decoder Noisy for the Collaborative Feature Disentanglement and Aggregation (CFDA)
    '''

    def __init__(self, in_channels, out_channels, init_fmaps_degree, fmaps_layer_number, layer_order, final_sigmoid,
                 **kwargs):
        super(CFDAV1_D_N, self).__init__()
        assert isinstance(init_fmaps_degree, int), 'fmaps_degree must be an integer!'
        fmaps_list = create_feature_maps(init_fmaps_degree, fmaps_layer_number)
        decoder_fmaps_list = list(reversed(fmaps_list))
        self.DN1 = Common_Decoder(in_channels=decoder_fmaps_list[0] + decoder_fmaps_list[1],
                                  out_channels=decoder_fmaps_list[1], Basic_Module=DoubleConv,
                                  order=layer_order)
        self.DN2 = Common_Decoder(in_channels=decoder_fmaps_list[1] + decoder_fmaps_list[2],
                                  out_channels=decoder_fmaps_list[2], Basic_Module=DoubleConv,
                                  order=layer_order)
        self.DN3 = Common_Decoder(in_channels=decoder_fmaps_list[2] + decoder_fmaps_list[3],
                                  out_channels=decoder_fmaps_list[3], Basic_Module=SingleConvOnce,
                                  order=layer_order)
        self.final_conv = nn.Conv3d(in_channels=fmaps_list[0], out_channels=out_channels, kernel_size=1)
        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x_encodefeatures):
        x = self.DN1(x_encodefeatures[1], x_encodefeatures[0])
        x = self.DN2(x_encodefeatures[2], x)
        x = self.DN3(x_encodefeatures[3], x)
        x = self.final_conv(x)
        if not self.training:
            x = self.final_activation(x)
        return x


"""*************************************    NN Module Classes    *****************************"""
