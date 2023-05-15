

from __future__ import division, absolute_import, print_function

import os
import math
import pdb
import time
import random
import argparse
import datetime
import threading
import matplotlib
import matplotlib.cm
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn

from lib.loss.loss_manager import get_depth_loss
from lib.models.mybts_head import weights_init_xavier
from lib.models.model_manager import ModelManager
from lib.dataset.dataset import MyDataLoader

from lib.utils.tools.logger import Logger as Log
from lib.utils.distributed import get_world_size, get_rank, is_distributed

from src.configer import Configer, Get
from src.tools.module_runner import ModuleRunner
from src.tools.optim_scheduler import OptimScheduler
from src.tools.utils import intersectionAndUnionGPU, ensure_path, AverageMeter


class Trainer(object):
    def __init__(self, configer):
        self.configer = configer
        
        # inialize the recorder
        self.train_losses = AverageMeter()
        self.score_lower_better = torch.zeros(6).cpu() + 1e3
        self.score_higher_better = torch.zeros(3).cpu()   # keep track of scores
        
        self.module_runner = ModuleRunner(configer)
        self.model_manager = ModelManager(configer=configer)
        
        self.optim_scheduler = OptimScheduler(configer)

        self.train_loader = None
        self.val_loader = None
        self.optimizer = None

        self._init_model()
    
    def _init_model(self):
        """
            Set up model, dataloader, optimizer, scheduler, loss
        """
        # set model
        self.depth_net = self.model_manager.depth_estimator()
        if torch.cuda.is_available():
            self.depth_net.to('cuda')
        else:
            Log.info("GPU is not available!")

        # initialize the decoder parameter
        self.depth_net.decoder.apply(weights_init_xavier)
        self.module_runner.set_misc(self.depth_net)

        # check number of parameters
        num_params = sum([np.prod(p.size()) for p in self.depth_net.parameters()])
        Log.info("Total number of parameters: {}".format(num_params))
        num_params_update = sum([np.prod(p.shape) for p in self.depth_net.parameters() if p.requires_grad])
        Log.info("Total number of learning parameters: {}".format(num_params_update))

        # set dataloader
        self.train_loader = MyDataLoader(configer, 'train')
        self.val_loader = MyDataLoader(configer, 'online_eval')

        self.configer.set(['train', 'max_iters'], self.configer.get('num_epochs')*len(self.train_loader.data))
        Log.info('Total iterations {}'.format(self.configer.get('train', 'max_iters')))


        # do not train the fully connected layer
        encoder_param = []
        for name, param in self.depth_net.named_parameters():
            
            if name[:7] == 'encoder':
                if name != 'encoder.base_model.fc.weight' and name != 'encoder.base_model.fc.bias':
                    encoder_param.append(param)
                else:
                    param.requires_grad = False
        
        # get parameter groups
        opt = self.configer.get('optim', 'optimizer')
        params_group = [
            {'params': encoder_param, 'weight_decay': self.configer.get('optim', opt)['weight_decay']},
            {'params': self.depth_net.decoder.parameters(), 'weight_decay': 0}
        ]

        # set optimizer
        self.optimizer, _ = self.optim_scheduler.init_optimizer(params_group)

        # set criterion
        self.depth_loss = get_depth_loss(self.configer)

    def adjust_learning_rate(self, epoch):
        """Decay the learning rate with half-cycle cosine after warmup"""
        min_lr = self.configer.get('lr', 'end_lr') * 0.5
        set_lr = self.configer.get('lr', 'base_lr')
        epochs = self.configer.get('num_epochs')
        warmup_epochs = self.configer.get('warmup_epochs', )

        if epoch < warmup_epochs:
            lr = set_lr * epoch / warmup_epochs
        else:
            lr = min_lr + (set_lr - min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
        for param_group in self.optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr

    def __train(self):
        # set the model to tran mode
        self.depth_net.train()
        self.depth_loss.train()
        start_time = time.time()
        
        # set learning rate
        if self.configer.get('lr', 'end_lr') != -1:
            end_lr = self.configer.get('lr', 'end_lr')
        else:
            end_lr =  0.1 * self.configer.get('lr', 'base_lr')

        num_total_steps = self.configer.get('num_epochs') * len(self.train_loader.data)

        while self.configer.get('epoch') < self.configer.get('num_epochs'):
            batch_time = time.time()
            # iterate through the dataloader
            for step, batchsample in enumerate(self.train_loader.data):
                # per iter learning rate adjust
                self.adjust_learning_rate(step / len(self.train_loader.data) + self.configer.get('epoch'))

                before_op_time = time.time()
                image = torch.autograd.Variable(batchsample['image'].cuda())
                depth_gt = torch.autograd.Variable(batchsample['depth'].cuda())

                # add cutmix here
                if self.configer.get('augment', 'cutmix'):
                    if np.random.rand(1) < 0.5: # do cutmix
                        slicing_idx, x_sliced, y_sliced = self.cutmix_data(image, depth_gt)
                        image[:, :, slicing_idx[0]:slicing_idx[2], slicing_idx[1]:slicing_idx[3]] = x_sliced
                        depth_gt[:, :, slicing_idx[0]:slicing_idx[2], slicing_idx[1]:slicing_idx[3]] = y_sliced

                # get the prediction
                depth_est = self.depth_net(image)

                # area in depth map with 0 value does not count for loss
                mask = depth_gt > 0.1

                # back prop
                self.optimizer.zero_grad()
                loss = self.depth_loss.forward(depth_est, depth_gt, mask.to(torch.bool))
                loss.backward()
                self.optimizer.step()

                # update the vars of the train phase.
                self.train_losses.update(loss.item(), self.configer.get("train", "batch_size")) # running avg of loss
                self.configer.plus_one('iters')

                # print the log info & reset the states.
                if self.configer.get('iters') % self.configer.get('log', 'log_freq') == 0:
                    Log.info('Ep {0} Iter {1} | Total {total} iters | loss {loss.val:.4f} (avg {loss.avg:.4f}) | '
                            'lr {3} | time {batch_time:.2f}s/{2}iters'.format(
                        self.configer.get('epoch'), self.configer.get('iters'), self.configer.get('log', 'log_freq'),
                        f"{self.optimizer.param_groups[0]['lr']:.7f}", 
                        batch_time=(time.time()-batch_time), loss=self.train_losses, total=len(self.train_loader.data)))
                    batch_time = time.time()

                    self.train_losses.reset()

                # do the validation
                if self.configer.get('iters') % self.configer.get('eval', 'eval_freq') == 0:
                    Log.info("*** Start evaulating ***")
                    self.evaluate()
        
            self.configer.plus_one('epoch')

    def train(self):
        if self.configer.get('mode') != 'train':
            print('bts_main.py is only for training. Use bts_test.py instead.')
            return -1

        Log.info('***** Start Training *****')

        torch.cuda.empty_cache()

        if self.configer.get('eval', 'do_online_eval'):
            Log.info("You have specified --do_online_eval.")
            Log.info("This will evaluate the model every eval_freq {} steps and save best models for individual eval metrics."
                .format(self.configer.get('eval', 'eval_freq')))

        self.__train()

    def evaluate(self):
        # set the model to eval mode
        start_time = time.time()
        self.depth_net.eval()
        self.depth_loss.eval()
        # get the predefined metric scores
        ev_score = self.online_eval(self.configer.get('gpu')[0], torch.cuda.device_count())
        if ev_score is not None:
            for i in range(9):
                # save the model w.r.t the different metric performance
                measure = ev_score[i]
                is_best = False
                # lower is better
                if i < 6 and measure < self.score_lower_better[i]:
                    self.score_lower_better[i] = measure.item()
                    is_best = True
                # higher is better
                elif i >= 6 and measure > self.score_higher_better[i-6]:
                    self.score_higher_better[i-6] = measure.item()
                    is_best = True
                if is_best:
                    self.module_runner.save_net(self.depth_net, save_mode='performance', pos=i)

        # log the metric evaluation
        runtime = time.time() - start_time
        msg = '\n\nTesting: Ep {} | time {:.1f} mins\n'.format(self.configer.get('epoch'), runtime / 60)
        msg += 'silog {:.4f} | abs_rel {:.4f} | log10 {:.4f} | rms {:.4f} | sq_rel {:.4f} | log_rms {:.4f} '.format(
            ev_score[0], ev_score[1], ev_score[2], ev_score[3], ev_score[4], ev_score[5]
        )
        msg += '| d1 {:.4f} | d2 {:.4f} | d3 {:.4f}\n'.format(
            ev_score[6], ev_score[7], ev_score[8]
        )
        Log.info(msg)

        self.depth_net.train()
        self.depth_loss.train()
        self.module_runner.set_misc(self.depth_net)

    def online_eval(self, gpu, ngpus):
        eval_measures = torch.zeros(10).cuda(device=gpu)
        for _, eval_sample_batched in enumerate(self.val_loader.data):
            # iterate through the validate loader
            with torch.no_grad():
                image = torch.autograd.Variable(eval_sample_batched['image'].cuda(gpu, non_blocking=True))
                gt_depth = eval_sample_batched['depth']
                has_valid_depth = eval_sample_batched['has_valid_depth']

                if not has_valid_depth:
                    continue
                
                # get the prediction and remove the blank boundaries
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

            # some further refinement
            if self.configer.get('eval', 'garg_crop') or self.configer.get('eval', 'eigen_crop'):
                gt_height, gt_width = gt_depth.shape
                eval_mask = np.zeros(valid_mask.shape)

                if self.configer.get('eval', 'garg_crop'):
                    eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

                elif self.configer.get('eval', 'eigen_crop'):
                    eval_mask[45:471, 41:601] = 1

                valid_mask = np.logical_and(valid_mask, eval_mask)

            # get the metric
            # metrics include: ['loss', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3']
            measures = self.compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

            eval_measures[:9] += torch.tensor(measures).cuda(device=gpu)
            eval_measures[9] += 1

        # gather the metircs
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

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = lam
        cut_w = np.int64(W * cut_rat)
        cut_h = np.int64(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def cutmix_data(self, x, y):
        # fix cut ratio to 0.5
        lam = 0.5
        batch_size = x.size()[0]
        
        index = torch.randperm(batch_size).cuda()

        bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)
        x_sliced = x[index, :, bbx1:bbx2, bby1:bby2]
        y_sliced = y[index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        
        return [bbx1, bby1, bbx2, bby2], x_sliced, y_sliced


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str, help='config file path')
    parser.add_argument('--exp_name', required=True, type=str, help='experiment name')
    parser.add_argument('--exp_id', required=True, type=str, help='config modifications')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    return Configer(parser.parse_args())

if __name__ == "__main__":
    # get configer
    configer = parse_config()

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

    # set the save path
    project_dir = os.path.dirname(os.path.realpath(__file__))
    configer.set(['project_dir'], project_dir)
    save_path = f"./results/{datetime.date.today()}:{configer.get('exp_name')}/{configer.get('exp_id')}"
    save_flag = ensure_path(save_path)
    configer.set(['save_path'], save_path)

    # initialize the log output file
    Log.init(
        log_file=os.path.join(save_path, 'output.log'),
        logfile_level='info',
        stdout_level='info',
        rewrite=True
    )
    configer.set(['logger'], Log)
    if save_flag:
        Log.info('remove the existing folder')
    if configer.get('debug'):
        Log.info('***** Debugging Mode *****')

    # log the modified configs
    configer.log_config()
    configer.log_modified_config()

    # set device
    if torch.cuda.is_available():
        configer.set(('gpu', ), [0])
    else:
        configer.set(('gpu', ), None)

    global get
    get = Get(configer)

    # train the model
    model = Trainer(configer)
    model.train()
