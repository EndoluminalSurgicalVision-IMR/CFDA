# -*- coding: utf-8 -*-


import numpy as np
from torch import einsum
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1e-5, p=2):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        num = torch.sum(torch.mul(predict, target)) * 2 + self.smooth
        den = torch.sum(predict) + torch.sum(target) + self.smooth
        dice = num / den
        loss = 1 - dice
        return loss


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    @zmh Annotation
    target and prediction must Align the dimensions
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """

    def __init__(self, sigmoid_normalization, weight=None, ignore_index=None, use_normalization=True, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.sigmoid_normalization = sigmoid_normalization
        self.weight = weight
        self.ignore_index = ignore_index
        self.use_normalization = use_normalization

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0.0
        if self.use_normalization:
            if not self.sigmoid_normalization:
                predict = torch.softmax(predict, dim=1)
            else:
                predict = torch.sigmoid(predict)
        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                # print('The Index:',i,'The Dice Loss:',dice_loss)
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss
        if self.ignore_index is not None:
            return total_loss / (target.shape[1] - 1)
        else:
            return total_loss / target.shape[1]


class FocalLoss(nn.Module):
    """
    Focal Loss for Dense Object Detection
    """

    def __init__(self, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target, alpha=None):
        """Forward pass
        :param input: shape = NxCxHxW
        :type input: torch.tensor
        :param target: shape = NxHxW
        :type target: torch.tensor
        :return: loss value
        :rtype: torch.tensor
        """
        if isinstance(alpha, (float, int)):
            alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            alpha = torch.Tensor(alpha)
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))
        target = target.view(-1, 1)
        logpt = nn.functional.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = (logpt.data.exp()).clone().detach().requires_grad_(True)
        if alpha is not None:
            if alpha.type() != input.data.type():
                alpha = alpha.type_as(input.data)
            alpha = alpha.view(-1)
            at = alpha.gather(0, target.data.view(-1))
            logpt = logpt * at.clone().detach().requires_grad_(True)
        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class Dice_with_Focal(nn.Module):
    def __init__(self, sigmoid_normalization, gamma=2, ignore_index=None, weight=None, alpha=None, size_average=True):
        super(Dice_with_Focal, self).__init__()
        self.sigmoid_normalization = sigmoid_normalization
        self.gamma = gamma
        self.alpha = alpha
        self.weight = weight
        self.ignore_index = ignore_index
        self.diceloss = DiceLoss(sigmoid_normalization=self.sigmoid_normalization, ignore_index=self.ignore_index)
        self.focalloss = FocalLoss(gamma=self.gamma)

    def forward(self, prediction, target, target_ce, lamda1=1.0, lamda2=1.0, alpha=None):
        loss_1 = self.diceloss(prediction, target)
        loss_2 = self.focalloss(prediction, target_ce, alpha)
        return (lamda1 * loss_1 + lamda2 * loss_2).mean()
