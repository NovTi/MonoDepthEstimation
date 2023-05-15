##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Microsoft Research
## Author: RainbowSecret, LangHuang, JingyiXie, JianyuanGuo
## Copyright (c) 2019
## yuyua@microsoft.com
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.models.mybts_head import MyBtsModel
from lib.models.dinov2.vision_transformer import DinoVisionTransformer

MODEL_DICT = {
    'mybts': MyBtsModel,
    'dinov2': DinoVisionTransformer
}

class ModelManager(object):
    def __init__(self, configer):
        self.configer = configer

    def depth_estimator(self):
        model_name = self.configer.get('network', 'model_name')

        if model_name not in MODEL_DICT:
            Log.error('Model: {} not valid!'.format(model_name))
            exit(1)

        model = MODEL_DICT[model_name](self.configer)

        return model
