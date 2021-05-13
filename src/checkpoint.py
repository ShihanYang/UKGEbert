"""
================================================================================
@In Project: ukg2vec
@File Name: checkpoint.py
@Author: Shihan Yang
@E-Mail: dr.yangsh@kust.edu.cn
@Create Date: 2020/09/18
@Update Date: 
@Version: 0.0.1
@Functions: 
    1. To save and load check points for too long training process
================================================================================
"""

import os

import h5py
import numpy as np
import yaml
from keras.callbacks import ModelCheckpoint


def checkpoint_base(base):
    if not os.path.exists(base):
        os.mkdir(base)


def load_meta(chk_file):
    # Load meta_information
    meta = {}
    with h5py.File(chk_file, 'r') as f:
        meta_group = f['meta']
        meta['training_args'] = yaml.load(
            meta_group.attrs['training_args'],
            Loader=yaml.FullLoader)
        for k in meta_group.keys():
            meta[k] = list(meta_group[k])
    return meta


def get_last_status(model, chk_file):
    last_epoch = -1
    last_meta = {}
    if os.path.exists(chk_file):
        model.load_weights(chk_file)
        last_meta = load_meta(chk_file)
        last_epoch = last_meta.get('epochs')[-1]
    return last_epoch, last_meta


class ukgeCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1, training_args=None, meta=None):
        super(ukgeCheckpoint, self).__init__(filepath,
                                             monitor=monitor,
                                             verbose=verbose,
                                             save_best_only=save_best_only,
                                             save_weights_only=save_weights_only,
                                             mode=mode,
                                             period=period)
        self.filepath = filepath
        self.new_file_override = True
        self.meta = meta or {'epochs': [], self.monitor: []}
        if training_args:
            self.meta['training_args'] = training_args

    def on_train_begin(self, logs=None):
        if self.save_best_only:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.best = max(self.meta[self.monitor], default=-np.Inf)
            else:
                self.best = min(self.meta[self.monitor], default=np.Inf)
        super(ukgeCheckpoint, self).on_train_begin(logs)

    def on_epoch_end(self, epoch, logs=None):
        if self.save_best_only:
            current = logs.get(self.monitor)
            if self.monitor_op(current, self.best):
                self.new_file_override = True
            else:
                self.new_file_override = False
        super(ukgeCheckpoint, self).on_epoch_end(epoch, logs)

        # Get statistics
        self.meta['epochs'].append(epoch)
        for k, v in logs.items():
            # Get default gets the value or sets (and gets) the default value
            self.meta.setdefault(k, []).append(v)

        # Save to file
        filepath = self.filepath.format(epoch=epoch, **logs)

        if self.new_file_override and self.epochs_since_last_save == 0:
            with h5py.File(filepath, 'r+') as f:
                meta_group = f.create_group('meta')
                meta_group.attrs['training_args'] = yaml.dump(self.meta.get('training_args', '{}'))
                meta_group.create_dataset('epochs', data=np.array(self.meta['epochs']))
                for k in logs:
                    meta_group.create_dataset(k, data=np.array(self.meta[k]))
