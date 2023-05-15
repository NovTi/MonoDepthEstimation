
import os
import pdb
import importlib
from addict import Dict

from src.tools.utils import path2module, log_uniform

class Configer(object):

    def __init__(self, parser):
        self.args = parser.__dict__
        if not os.path.exists(parser.config):
            raise ValueError(f"Config path error: {parser.config}")
        self.config = Dict(importlib.import_module(path2module(parser.config)).CONFIG)
        for k, v in self.args.items():
            self.config[k] = v
        if self.config['exp_name'] != 'test':
            self.modified_config = self.modify_config()
        # self.log_modified_config()

    def __getattr__(self, key):
        return self.config[key]

    def exists(self, *keys):
        item = self.config
        for i in range(len(keys)):
            try:
                item = item[keys[i]]
            except:
                raise KeyError(f"Invalid key: {keys}")
        return not (isinstance(item, Dict) and len(item) == 0)

    def get(self, *keys):
        item = self.config
        for i in range(len(keys)):
            try:
                item = item[keys[i]]
            except:
                raise KeyError(keys)
        if isinstance(item, Dict) and len(item) == 0:
            return None
        return item
    
    def set(self, keys, val):
        item = self.config
        for i in range(len(keys) - 1):
            try:
                item = item[keys[i]]
            except:
                raise KeyError(f"Invalid key: {keys}")
        assert isinstance(item, Dict)
        item[keys[-1]] = val

    def update(self, to_modify):
        for k, v in to_modify.items():
            self.set(k, v)

    def plus_one(self, *keys):
        val = self.get(*keys)
        assert isinstance(val, int)
        self.set(keys, val+1)
    
    def log_config(self):
        msg = ['\n\nConfig:\n']
        for k, v in self.config.items():
            if isinstance(v, Dict):
                msg.append(f'[{k}]:\n')
                for kk, vv in v.items():
                    msg.append(f'\t[{kk}]: {vv}\n')
            else:
                msg.append(f'[{k}]: {v}\n')
        self.logger.info(''.join(msg))
    
    def log_modified_config(self):
        msg = ['\n\nModified Config:\n']
        for k, v in self.modified_config.items():
            msg.append(f'{k}: {v}\n')
        self.logger.info(''.join(msg))

    def modify_config(self):
        to_modify = {}
        exp_name, exp_id = self.args['exp_name'], self.args['exp_id']

        if self.config['lr']['end_lr'] == -1:
            to_modify.update({
                ('lr', 'end_lr'): self.config['lr']['base_lr'] * 0.1
            })

        keys = exp_id.split('_')
        loss_type = 'si_gradient' if keys[2] == 'gradient' else 'silog'
        to_modify.update({
            ('augment', 'cutmix'): keys[1] == 'cutmix',   # cutmix or not
            ('loss', 'loss_type'): loss_type, # loss type
        })

        self.update(to_modify)
        return to_modify


class Get(object):
    def __init__(self, configer):
        self.configer = configer

    def __call__(self, key1, key2=None):
        if key2 is None:
            return self.configer.get(key1)
        else:
            return self.configer.get(key1, key2)

DEBUG_CONFIG = {
    ('train', 'test_interval'): 40,
    ('train', 'display_iter'): 10,
    ('train', 'max_epoch'): 3,
    ('val', 'val_num'): 40
}

if __name__ == '__main__':
    pass