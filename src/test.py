

from __future__ import division, absolute_import, print_function

import os
import pdb
import time
import random
import argparse
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.backends.cudnn as cudnn

from lib.dataset.dataset import BtsDataLoader
from lib.loss.loss_manager import get_depth_loss
from lib.models.model_manager import ModelManager

from lib.utils.tools.logger import Logger as Log
from lib.utils.distributed import get_world_size, get_rank, is_distributed

from src.configer import Configer, Get
from src.tools.utils import ensure_path, AverageMeter


class Tester(object):
    def __init__(self, configer):
        self.configer = configer

        # keep track of the scores
        self.score_lower_better = torch.zeros(6).cpu() + 1e3
        self.score_higher_better = torch.zeros(3).cpu()
        
        self.model_manager = ModelManager(configer=configer)

        self._init_model()
    
    def _init_model(self):
        """
            Set up model and test loader
        """
        # load net from configs
        self.depth_net = self.model_manager.depth_estimator()

        # move the model to cuda
        if torch.cuda.is_available():
            self.depth_net.cuda()
        else:
            Log.info("Warning: Your model is not on cuda!")

        # load state dict
        model_dict = torch.load(self.configer.get('weight_dir', ), map_location='cpu')['state_dict']
        msg = self.depth_net.load_state_dict(model_dict, strict=True)
        Log.info(f"DepthNet Load Weight: {msg}")
        
        # get test loader
        self.test_loader = BtsDataLoader(configer, 'test')

        # freeze the model parameters
        for name, param in self.depth_net.named_parameters():
            param.requires_grad = False

    def test(self):
        torch.cuda.empty_cache()

        Log.info('***** Start Testing *****')

        start_time = time.time()
        # move the model to evaluation mode
        self.depth_net.eval()

        # get the evulation scores
        # scores include: ['silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3']
        ev_score = self.online_test(self.configer.get('gpu')[0], torch.cuda.device_count())
        
        # log the evaluationn scores
        runtime = time.time() - start_time
        msg = '\nTesting Results: Time: {:.1f} mins | Metrics:\n'.format(runtime / 60)
        msg += 'loss {:.4f} | abs_rel {:.4f} | log10 {:.4f} | rms {:.4f} | sq_rel {:.4f} | log_rms {:.4f} '.format(
            ev_score[0], ev_score[1], ev_score[2], ev_score[3], ev_score[4], ev_score[5]
        )
        msg += '| d1 {:.4f} | d2 {:.4f} | d3 {:.4f}\n'.format(
            ev_score[6], ev_score[7], ev_score[8]
        )
        msg += '======= Done ======='
        Log.info(msg)

    def online_test(self, gpu, ngpus):
        eval_measures = torch.zeros(10).cuda(device=gpu)
        for idx, eval_sample_batched in enumerate(self.test_loader.data):
            # iterate through the test loader
            if idx % (len(self.test_loader.data) // 10) == 0:
                Log.info(f"  Current iter: {idx} | Total {len(self.test_loader.data)} iters")

            with torch.no_grad():
                image = torch.autograd.Variable(eval_sample_batched['image'].cuda(gpu, non_blocking=True))
                gt_depth = eval_sample_batched['depth']
                has_valid_depth = eval_sample_batched['has_valid_depth']

                if not has_valid_depth:
                    continue

                # get the prediction
                pred_depth = self.depth_net(image).squeeze()
                new_pred = torch.zeros(pred_depth.shape)
                new_pred[45:472, 43:608] = pred_depth[45:472, 43:608]
                pred_depth = new_pred

                pred_depth = pred_depth.cpu().numpy()
                gt_depth = gt_depth.cpu().numpy().squeeze()

            # refine the prediction
            pred_depth[pred_depth < self.configer.get('eval', 'min_depth_eval')] = self.configer.get('eval', 'min_depth_eval')
            pred_depth[pred_depth > self.configer.get('eval', 'max_depth_eval')] = self.configer.get('eval', 'max_depth_eval')
            pred_depth[np.isinf(pred_depth)] = self.configer.get('eval', 'max_depth_eval')
            pred_depth[np.isnan(pred_depth)] = self.configer.get('eval', 'min_depth_eval')

            valid_mask = np.logical_and(
                gt_depth > self.configer.get('eval', 'min_depth_eval'), 
                gt_depth < self.configer.get('eval', 'max_depth_eval'))

            # furthur refine the prediction
            if self.configer.get('eval', 'garg_crop') or self.configer.get('eval', 'eigen_crop'):
                gt_height, gt_width = gt_depth.shape
                eval_mask = np.zeros(valid_mask.shape)

                if self.configer.get('eval', 'garg_crop'):
                    eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

                elif self.configer.get('eval', 'eigen_crop'):
                    eval_mask[45:471, 41:601] = 1

                valid_mask = np.logical_and(valid_mask, eval_mask)

            measures = self.compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

            eval_measures[:9] += torch.tensor(measures).cuda(device=gpu)
            eval_measures[9] += 1

        # return the evaluation scores
        eval_measures_cpu = eval_measures.cpu()
        cnt = eval_measures_cpu[9].item()
        eval_measures_cpu /= cnt

        return eval_measures_cpu

    def compute_errors(self, gt, pred):
        thresh = np.maximum((gt / pred), (pred / gt))
        d1 = (thresh < 1.25).mean()
        d2 = (thresh < 1.25 ** 2).mean()
        d3 = (thresh < 1.25 ** 3).mean()

        rms = (gt - pred) ** 2
        rms = np.sqrt(rms.mean())

        log_rms = (np.log(gt) - np.log(pred)) ** 2
        log_rms = np.sqrt(log_rms.mean())

        abs_rel = np.mean(np.abs(gt - pred) / gt)
        sq_rel = np.mean(((gt - pred) ** 2) / gt)

        err = np.log(pred) - np.log(gt)
        silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

        err = np.abs(np.log10(pred) - np.log10(gt))
        log10 = np.mean(err)

        return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str, help='config file path')
    parser.add_argument('--datapath', required=True, type=str, help='test data path')
    parser.add_argument('--exp_name', default='test', help='experiment name')
    parser.add_argument('--exp_id', default='test', help='config modifications')
    return Configer(parser.parse_args())


if __name__ == "__main__":
    # get configer
    configer = parse_config()
    # set data path
    configer.set(['data', 'data_path'], configer.get('datapath'))

    # fix seed
    seed = configer.get('manual_seed')
    if seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # initialize the log output file
    project_dir = os.path.dirname(os.path.realpath(__file__))
    configer.set(['project_dir'], project_dir)
    save_path = f"./results/{datetime.date.today()}:{configer.get('exp_name')}/{configer.get('exp_id')}"
    save_flag = ensure_path(save_path)
    configer.set(['save_path'], save_path)

    Log.init(
        log_file=os.path.join(save_path, 'output.log'),
        logfile_level='info',
        stdout_level='info',
        rewrite=True
    )
    configer.set(['logger'], Log)

    configer.log_config()

    global get
    get = Get(configer)

    # testing
    model = Tester(configer)
    model.test()
