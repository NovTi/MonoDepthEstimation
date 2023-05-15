# Copyright (C) 2019 Jin Han Lee
#
# Adapted from https://github.com/cleinc/bts/blob/master/pytorch/bts_dataloader.py


import os
import pdb
import torch
import random
import numpy as np
import torch.utils.data.distributed

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.configer import Get
# from lib.dataset.dist_sampler import *
# from lib.utils.distributed import is_distributed


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])


class BtsDataLoader(object):
    def __init__(self, configer, mode):
        global get
        get = Get(configer)

        if mode == 'train':
            self.training_samples = DataLoadPreprocess(mode, transform=preprocessing_transforms(mode))
            self.train_sampler = None
            self.data = DataLoader(self.training_samples, get('train', 'batch_size'),
                                   shuffle=(self.train_sampler is None),
                                   num_workers=get('distributed', 'num_threads'),
                                   pin_memory=True,
                                   sampler=self.train_sampler)

        elif mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(mode, transform=preprocessing_transforms(mode))
            self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=True,
                                   sampler=self.eval_sampler)
        
        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(mode, transform=preprocessing_transforms(mode))
            self.testing_samples[0]
            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)

        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))
            
            
class DataLoadPreprocess(Dataset):
    def __init__(self, mode, transform=None, is_for_online_eval=False):
        if mode == 'online_eval':
            with open(get('eval', 'file_lst_eval'), 'r') as f:
                self.filenames = f.readlines()
        else:
            self.filenames = []
            files = self.list_all_files(get('data', 'data_path'))
            for file in files:
                lst = file.split('/')
                img_name = lst[-1]
                depth_name = f'sync_depth_{img_name[4:9]}.png'
                self.filenames.append(f'/{lst[-2]}/{img_name} /{lst[-2]}/{depth_name} 518.8579')

        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor
        self.is_for_online_eval = is_for_online_eval
            
    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        focal = float(sample_path.split()[2])

        if self.mode == 'train':
            if get('data', 'dataset') == 'kitti' and get('augment', 'use_right') is True and random.random() > 0.5:
                image_path = os.path.join(get('data', 'data_path'), "./" + sample_path.split()[3])
                depth_path = os.path.join(get('data', 'gt_path'), "./" + sample_path.split()[4])
            else: # nyu v2
                image_path = os.path.join(get('data', 'data_path'), "./" + sample_path.split()[0])
                depth_path = os.path.join(get('data', 'gt_path'), "./" + sample_path.split()[1])
    
            image = Image.open(image_path)
            depth_gt = Image.open(depth_path)
            
            # To avoid blank boundaries due to pixel registration
            depth_gt = depth_gt.crop((43, 45, 608, 472))
            image = image.crop((43, 45, 608, 472))
    
            if get('augment', 'random_rotate') is True:
                random_angle = (random.random() - 0.5) * 2 * get('augment', 'degree')
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)
            
            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)

            # nyuv2 ratio
            depth_gt = depth_gt / 1000.0

            image, depth_gt = self.random_crop(image, depth_gt, get('data', 'input_height'), get('data', 'input_width'))
            image, depth_gt = self.train_preprocess(image, depth_gt)
            sample = {'image': image, 'depth': depth_gt, 'focal': focal}
        
        else:
            if self.mode == 'online_eval':
                data_path = get('eval', 'data_path_eval')
            else:
                data_path = get('data', 'data_path')

            image_path = os.path.join(data_path, "./" + sample_path.split()[0])
            image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0

            # if self.mode == 'online_eval':
            gt_path = get('eval', 'gt_path_eval')
            depth_path = os.path.join(gt_path, "./" + sample_path.split()[1])
            has_valid_depth = False
            
            try:
                depth_gt = Image.open(depth_path)
                has_valid_depth = True
            except IOError:
                depth_gt = False
                # print('Missing gt for {}'.format(image_path))

            if has_valid_depth:
                depth_gt = np.asarray(depth_gt, dtype=np.float32)
                depth_gt = np.expand_dims(depth_gt, axis=2)
                # nyuv2 ratio
                depth_gt = depth_gt / 1000.0

            # if self.mode == 'online_eval':
            sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'has_valid_depth': has_valid_depth}
            # else:
            #     sample = {'image': image, 'focal': focal}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

    def list_all_files(self, rootdir):
        extention = '.jpg'
        _files = []
        lst = os.listdir(rootdir)
        for i in range(0,len(lst)):
            path = os.path.join(rootdir+'/',lst[i])
            if os.path.isdir(path):
                _files.extend(self.list_all_files(path))
            if os.path.isfile(path):
                if '/find' not in path and path[-4:] == extention:
                    _files.append(path)
        return _files

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        return img, depth

    def train_preprocess(self, image, depth_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()
    
        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)
    
        return image, depth_gt
    
    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if get('data', 'dataset') == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug
    
    def __len__(self):
        return len(self.filenames)


class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def __call__(self, sample):
        image, focal = sample['image'], sample['focal']
        image = self.to_tensor(image)
        image = self.normalize(image)

        # if self.mode == 'test':
        #     return {'image': image, 'focal': focal}

        depth = sample['depth']
        if self.mode == 'train':
            depth = self.to_tensor(depth)
            return {'image': image, 'depth': depth, 'focal': focal}
        else:
            has_valid_depth = sample['has_valid_depth']
            return {'image': image, 'depth': depth, 'focal': focal, 'has_valid_depth': has_valid_depth}
    
    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))
        
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img
        
        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img
