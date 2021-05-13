"""
================================================================================
@In Project: ukg2vec
@File Name: pickleloss.py
@Author: Shihan Yang
@E-Mail: dr.yangsh@kust.edu.cn
@Create Date: 2020/09/18
@Update Date: 
@Version: 0.1.0
@Functions: 
    1. To save and load history of training
    2. loss, val_loss, acc, val_acc, mean_absolute_error, val_mean_absolute_error
================================================================================
"""

import pickle


def save(history, loss_file):
    with open(loss_file, 'wb') as f:
        pickle.dump(history, f)


def load(loss_file):
    with open(loss_file, 'rb') as file_pi:
        history = pickle.load(file_pi)
    return history


if __name__ == '__main__':
    import os
    lf = os.path.abspath('..') + '\\data\\ppi5k\\model_e200_100d_sg.h5.loss'
    data = load(lf)
    print(data.keys())
    print('loss', data['loss'])
    print('val_loss', data['val_loss'])