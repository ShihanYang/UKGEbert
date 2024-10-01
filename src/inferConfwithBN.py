"""
================================================================================
@In Project: UKGEbert
@File Name: inferConfwithBN.py
@Author: Shihan Yang
@E-Mail: dr.yangsh@kust.edu.cn
@Create Date: 2024/10/05
@Update Date: 
@Version: 0.2.0
@Functions: 
    1. To answer inference questions based on Bayesian Networks generated from
       uncertain knowledge graphs.
    2. Notes: Read right triples from .csv/.tsv file to build the visible Knowledge
       Graph for inferring some invisible facts, which can be reasonably inferred
       from this visible KG.
================================================================================
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
import bayesianet as net


triple_file = '../data/cn15k/train.tsv'
bayesianNet = net.BN()
facts, kg = bayesianNet.createBNfromKG(triple_file)  # cost too much !
print(facts, len(kg))
bn_file = '../data/cn15k/bayesiannet.pkl'
bayesianNet.save(bn_file)


# load bayesian network from pickel file
bayesianNet = net.BN()
bn_file = '../data/cn15k/bayesiannet.pkl'
bayesianNet = net.BN.load(bn_file)

# read testing file
test_file = '../data/cn15k/test.tsv'
triple_prediction = set()
triple_confidence = set()
with open(test_file, mode='r', encoding='utf-8') as tf:
    lines = tf.readlines()
    for line in lines:
        ll = line.split()
        triple = (ll[0], ll[1], ll[2])
        triple_confidence[triple] = float(ll[3])

print(len(triple_confidence))

# prediction for each testing triple

# testing for adding evidence to raise posterior probability


