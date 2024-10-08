"""
================================================================================
@In Project: UKGEbert
@File Name: visualizing.py
@Author: Shihan Yang
@E-Mail: dr.yangsh@kust.edu.cn
@Create Date: 2024/10/08
@Update Date: 
@Version: 0.1.0
@Functions: 
    1. To 
    2. Notes:
================================================================================
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Visualization of testing for adding evidence to raise/reduce linearly posterior probability
with open('../data/cn15k/confidenceupdate.pkl', 'rb') as conf:
    X, Y = pickle.load(conf)

coefficients = np.polyfit(X, Y, 1)
polynomial = np.poly1d(coefficients)
Y_ = polynomial(X)  # linear fitting

plt.plot(X, Y_)
plt.scatter(X, Y, color='red', marker='o')
plt.ylabel('Probability ($\\times\\ 10^{-1}$) of an invisible fact')
plt.xlabel('Number of added factual instances (evidences)')
plt.savefig('../log/confidenceupdated.png', bbox_inches='tight', pad_inches=0.5)
plt.show()
