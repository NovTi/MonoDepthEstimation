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

##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# So far just BTS, will add more nets or backbone later
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from lib.models.bts import BtsModel
from lib.models.bts2 import Bts2Model

MODEL_DICT = {
    'bts': BtsModel,
    'bts2': Bts2Model,
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
