# Copyright (C) 2019 Jin Han Lee
#
# Adapted from https://github.com/cleinc/bts/blob/master/pytorch/bts.py

import pdb
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple

from src.configer import Get


def weights_init_xavier(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class PPMModule(nn.Module):
    """
    Reference: 
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(PPMModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features+len(sizes)*out_features, out_features, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = nn.Sequential(
            nn.BatchNorm2d(out_features),
            nn.ReLU()
        )
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        out = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        out = self.bottleneck(torch.cat(out, 1))
        return out   # bottle


class upconv(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=2):
        super(upconv, self).__init__()
        self.elu = nn.ELU()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=False, kernel_size=3, stride=1, padding=1)
        self.ratio = ratio
        
    def forward(self, x):
        up_x = F.interpolate(x, scale_factor=self.ratio, mode='nearest')
        out = self.conv(up_x)
        out = self.elu(out)
        return out


class reduction_1x1(nn.Sequential):
    def __init__(self, num_in_filters, num_out_filters, max_depth, is_final=False):
        super(reduction_1x1, self).__init__()        
        self.max_depth = max_depth
        self.is_final = is_final
        self.sigmoid = nn.Sigmoid()
        self.reduc = torch.nn.Sequential()
        
        while num_out_filters >= 4:
            if num_out_filters < 8:
                if self.is_final:
                    self.reduc.add_module(
                        'final', 
                        torch.nn.Sequential(
                            nn.Conv2d(num_in_filters, out_channels=1, bias=False,
                            kernel_size=1, stride=1, padding=0), 
                            nn.Sigmoid())
                    )
                else:
                    self.reduc.add_module(
                        'plane_params', 
                        torch.nn.Conv2d(num_in_filters, out_channels=3, bias=False,
                        kernel_size=1, stride=1, padding=0))
                break
            else:
                self.reduc.add_module(
                    'inter_{}_{}'.format(num_in_filters, num_out_filters),
                    torch.nn.Sequential(nn.Conv2d(in_channels=num_in_filters, out_channels=num_out_filters,
                                                bias=False, kernel_size=1, stride=1, padding=0),
                    nn.ELU()))

            num_in_filters = num_out_filters
            num_out_filters = num_out_filters // 2
    
    def forward(self, net):
        net = self.reduc.forward(net)
        if not self.is_final:
            theta = self.sigmoid(net[:, 0, :, :]) * math.pi / 3
            phi = self.sigmoid(net[:, 1, :, :]) * math.pi * 2
            dist = self.sigmoid(net[:, 2, :, :]) * self.max_depth
            n1 = torch.mul(torch.sin(theta), torch.cos(phi)).unsqueeze(1)
            n2 = torch.mul(torch.sin(theta), torch.sin(phi)).unsqueeze(1)
            n3 = torch.cos(theta).unsqueeze(1)
            n4 = dist.unsqueeze(1)
            net = torch.cat([n1, n2, n3, n4], dim=1)
        
        return net


class MyBTS_Decoder(nn.Module):
    def __init__(self, feat_out_channels, num_features=512):
        super(MyBTS_Decoder, self).__init__()
        # dense feature
        self.upconv4    = upconv(feat_out_channels[4], feat_out_channels[3])
        self.bn4        = nn.BatchNorm2d(feat_out_channels[3], momentum=0.01, affine=True, eps=1.1e-5)
        
        self.conv4      = torch.nn.Sequential(nn.Conv2d(feat_out_channels[3] + feat_out_channels[3], feat_out_channels[3], 3, 1, 1, bias=False),
                                              nn.ELU())

        self.ppm = PPMModule(features=feat_out_channels[3], out_features=num_features)

        self.reduc8x8   = reduction_1x1(num_features, num_features // 4, 10.0)
        self.upconv3    = upconv(num_features, num_features)
        self.upconv_depth8 = upconv(4, 1, ratio=8)

        # skip 2
        self.conv2      = torch.nn.Sequential(nn.Conv2d(feat_out_channels[2], num_features, 3, 1, 1, bias=False),
                                              nn.ELU())
        self.concat2_conv = torch.nn.Sequential(nn.Conv2d(num_features*2, num_features, 3, 1, 1, bias=False),
                                              nn.ELU())
        self.bn2        = nn.BatchNorm2d(num_features, momentum=0.01, affine=True, eps=1.1e-5)
        self.reduc4x4   = reduction_1x1(num_features, num_features // 8, 10.0)
        self.upconv2    = upconv(num_features, num_features)
        self.upconv_depth4 = upconv(4, 1, ratio=4)

        # skip 1
        self.conv1      = torch.nn.Sequential(nn.Conv2d(feat_out_channels[1], num_features, 3, 1, 1, bias=False),
                                              nn.ELU())
        self.concat1_conv = torch.nn.Sequential(nn.Conv2d(num_features*2, num_features, 3, 1, 1, bias=False),
                                              nn.ELU())
        self.bn1        = nn.BatchNorm2d(num_features, momentum=0.01, affine=True, eps=1.1e-5)

        self.reduc2x2   = reduction_1x1(num_features, num_features // 16, 10.0)
        self.upconv_depth2 = upconv(4, 1)
        
        # final
        self.upconv    = upconv(num_features, num_features)
        self.upconv_depth_2x2 = upconv(1, 1, ratio=1)
        self.upconv_depth_4x4 = upconv(1, 1, ratio=1)

        self.conv       = torch.nn.Sequential(nn.Conv2d(512+3, num_features // 16, 3, 1, 1, bias=False),
                                              nn.ELU())
        self.get_depth  = torch.nn.Sequential(nn.Conv2d(num_features // 16, 1, 3, 1, 1, bias=False),
                                              nn.Sigmoid())

    def forward(self, features):
        skip0, skip1, skip2, skip3 = features[0], features[1], features[2], features[3]

        # use ppm module to handle the dense deep feature
        dense_features = torch.nn.ReLU()(features[4])
        dense_features = self.upconv4(dense_features) # H/16
        dense_features = self.bn4(dense_features)
        dense_concat = torch.cat([dense_features, skip3], dim=1)
        dense_concat = self.conv4(dense_concat)    # [B, 1024, 26, 34]

        dense_concat = self.ppm(dense_concat)     # [B, 512, 26, 34]

        # lpg for concat3, expand 8 times
        reduc8x8 = self.reduc8x8(dense_concat)      # [B, 4, 26, 34]
        plane_normal_8x8 = reduc8x8[:, :3, :, :]    # [B, 3, 26, 34]
        plane_normal_8x8 = F.normalize(plane_normal_8x8, 2, 1)
        plane_dist_8x8 = reduc8x8[:, 3, :, :]    # [B, 1, 26, 34]
        plane_eq_8x8 = torch.cat([plane_normal_8x8, plane_dist_8x8.unsqueeze(1)], 1)  # [B, 4, 26, 34]
        depth_8x8_scaled = self.upconv_depth8(plane_eq_8x8) / 10.0    # [B, 1, 208, 272]

        concat3 = self.upconv3(dense_concat)   # [B, 512, 52, 68]
        conv2 = self.conv2(skip2)        # [B, 512, 52, 68]
        concat2 = torch.cat([conv2, concat3], dim=1)    # [B, 1024, 52, 68]
        concat2 = self.concat2_conv(concat2)      # [B, 512, 52, 68]
        concat2 = self.bn2(concat2)

        # lpg for concat4, expand 4 times
        reduc4x4 = self.reduc4x4(concat2)   # [B, 4, 52, 68]
        plane_normal_4x4 = reduc4x4[:, :3, :, :]   # [B, 3, 52, 68]
        plane_normal_4x4 = F.normalize(plane_normal_4x4, 2, 1)
        plane_dist_4x4 = reduc4x4[:, 3, :, :]    # [B, 1, 52, 68]
        plane_eq_4x4 = torch.cat([plane_normal_4x4, plane_dist_4x4.unsqueeze(1)], 1)  # [B, 4, 52, 68]
        depth_4x4_scaled = self.upconv_depth4(plane_eq_4x4) / 10.0  # [4, 1, 208, 272]

        concat2 = self.upconv2(concat2)   # [B, 256, 52, 68]
        conv1 = self.conv1(skip1)      # [B, 512, 52, 68]
        concat1 = torch.cat([concat2, conv1], dim=1)   # [B, 1024, 52, 68]
        concat1 = self.concat1_conv(concat1)    # [B, 512,52, 68]
        concat1 = self.bn1(concat1)

        # lpg for concat2, expand 2 times
        reduc2x2 = self.reduc2x2(concat1)    # [B, 4, 104, 136]
        plane_normal_2x2 = reduc2x2[:, :3, :, :]   # [B, 3, 104, 136]
        plane_normal_2x2 = F.normalize(plane_normal_2x2, 2, 1)
        plane_dist_2x2 = reduc2x2[:, 3, :, :]   # [B, 1, 104, 136]
        plane_eq_2x2 = torch.cat([plane_normal_2x2, plane_dist_2x2.unsqueeze(1)], 1)  # [B, 4, 104, 136]
        depth_2x2_scaled = self.upconv_depth2(plane_eq_2x2) / 10.0  # [B, 1, 208, 272]

        # final
        upconv = self.upconv(concat1)  # [B, 512, 208, 272]
        upconv2x2 = self.upconv_depth_2x2(depth_2x2_scaled)  # [B, 1, 208, 272]
        upconv4x4 = self.upconv_depth_4x4(depth_4x4_scaled)  # [B, 1, 208, 272]
        final = torch.cat([upconv, depth_8x8_scaled, upconv4x4, upconv2x2], 1)  # [B, 515, 104, 136]
        final = self.conv(final)
        final_depth = 10.0 * self.get_depth(final)
        final_depth = F.interpolate(final_depth, scale_factor=2.0, mode='bilinear', align_corners=True)
        return final_depth


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        import torchvision.models as models
        if get('network', 'encoder') == 'resnet50_mybts':
            self.base_model = models.resnet50(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        elif get('network', 'encoder') == 'resnet101_mybts':
            self.base_model = models.resnet101(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        else:
            print('Not supported encoder: {}'.format(get('network', 'encoder')))

    def forward(self, x):
        feature = x
        skip_feat = []
        i = 1
        # get the intermediate features
        for k, v in self.base_model._modules.items():
            if 'fc' in k or 'avgpool' in k:
                continue
            feature = v(feature)
            if any(x in k for x in self.feat_names):
                skip_feat.append(feature)
            i = i + 1
        return skip_feat
    

class MyBtsModel(nn.Module):
    def __init__(self, configer):
        super(MyBtsModel, self).__init__()
        global get
        get = Get(configer)
        # set encoder
        self.encoder = Encoder()
        # set decoder
        self.decoder = MyBTS_Decoder(self.encoder.feat_out_channels, get('train', 'bts_size'))

    def forward(self, x):
        # x: [B, 3, 416, 544]
        skip_feat = self.encoder(x)
        return self.decoder(skip_feat)
    