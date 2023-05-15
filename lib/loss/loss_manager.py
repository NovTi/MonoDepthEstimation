##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: DonnyYou, RainbowSecret, JingyiXie, JianyuanGuo
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils.tools.logger import Logger as Log
from lib.utils.distributed import is_distributed


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


class GradientLoss(nn.Module):
    def __init__(self, scales=4):
        super().__init__()

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)
            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=reduction_batch_based)

        return total


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff[torch.where(mask==0)] = 0.0
    # diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x[torch.where(mask_x==0)] = 0.0
    # grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y[torch.where(mask_y==0)] = 0.0
    # grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


# def GradientLossMy(log_prediction_d, mask, log_gt):
#     N = torch.sum(mask)
#     log_d_diff = log_prediction_d - log_gt
#     log_d_diff[torch.where(mask==0)] = 0.0
#     # log_d_diff = torch.mul(log_d_diff, mask)

#     v_gradient = torch.abs(log_d_diff[0:-2, :] - log_d_diff[2:, :])
#     v_mask = torch.mul(mask[0:-2, :], mask[2:, :])
#     v_gradient[torch.where(v_mask==0)] = 0.0
#     # v_gradient = torch.mul(v_gradient, v_mask)

#     h_gradient = torch.abs(log_d_diff[:, 0:-2] - log_d_diff[:, 2:])
#     h_mask = torch.mul(mask[:, 0:-2], mask[:, 2:])
#     h_gradient[torch.where(h_mask==0)] = 0.0
#     # h_gradient = torch.mul(h_gradient, h_mask)

#     gradient_loss = torch.sum(h_gradient) + torch.sum(v_gradient)
#     gradient_loss = gradient_loss / N

#     return gradient_loss


class Silog_loss(nn.Module):
    def __init__(self, variance_focus):
        super(Silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0


class SiGradientLoss(nn.Module):
    def __init__(self, variance_focus):
        super(SiGradientLoss, self).__init__()
        self.variance_focus = variance_focus
        self.grad_loss = GradientLoss()

    def forward(self, depth_est, depth_gt, mask):
        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        sigloss =  torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0
        gradientloss = self.grad_loss(depth_est, depth_gt, mask)

        total_loss = sigloss + 0.5 * gradientloss  # less focus on the gradient loss

        return total_loss



def get_depth_loss(configer):
    Log.info('use loss: {}.'.format(configer.get('loss', 'loss_type')))
    if configer.get('loss', 'loss_type') == 'silog':
        return Silog_loss(configer.get('loss', 'variance_focus'))
    elif configer.get('loss', 'loss_type') == 'si_gradient':
        return SiGradientLoss(configer.get('loss', 'variance_focus'))


