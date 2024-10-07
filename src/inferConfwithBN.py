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
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
import src.bayesianet as net
from src.nDCG import NDCG, mean_NDCG, linear_DCG, exponential_DCG

# triple_file = '../data/cn15k/train-.tsv'
# bayesianNet = net.BN()
# facts, kg = bayesianNet.createBNfromKG(triple_file)  # todo: cost too much ! just do it once.
# print(facts, len(kg))  # 1610 / 204984
# bn_file = '../data/cn15k/bayesiannet.pkl'
# bayesianNet.save(bn_file)


# load bayesian network from pickle file
bayesianNet = net.BN()
bn_file = '../data/cn15k/bayesiannet.pkl'
bayesianNet = net.BN.load(bn_file)

# read testing file
test_file = '../data/cn15k/test-.tsv'
triple_prediction = dict()
triple_confidence = dict()
with open(test_file, mode='r', encoding='utf-8') as tf:
    lines = tf.readlines()
    for line in lines:
        ll = line.split()
        triple = (ll[0], ll[1], ll[2])
        triple_confidence[triple] = float(ll[3])

print('testing samples:', len(triple_confidence))  # 1195 / 19166

# todo : choose a fact / some facts to record its confidence changes.
fact_0 = ('195', '2', '14259', 0.8927087856574166)  # (staff,195 | isa,2 | building material,14259)
fact_0 = net.Fact().factFromTriple(fact_0)
fact_0_confidences = list()
# Prediction for each testing triple
for fact in triple_confidence.keys():
    f = net.Fact().factFromTriple(fact)
    conf = bayesianNet.inferFact(f)
    triple_prediction[fact] = conf
    bayesianNet.addFact(f)

    # testing for adding evidence to raise posterior probability
    fact_0.relation.name = fact_0.relation.name[:fact_0.relation.name.find('~')]
    fact_0_conf = bayesianNet.inferFact(fact_0)
    fact_0_confidences.append(fact_0_conf)

# Computing MSE, MAE, linear NDCG, and exponential NDCG
mse = 0
mae = 0
for t in triple_prediction.keys():
    x = triple_confidence[t]
    pre = triple_prediction[t]
    mse += (x - pre) ** 2
    mae += abs(x - pre)
mse = mse / len(triple_confidence)
mae = mae / len(triple_confidence)
print('MSE:', mse)  # MSE: 0.2906198710806215
print('MAE:', mae)  # MAE: 0.4644416332075686
# NDCG
# ranked_predicted_confidence_list = list()
# ranked_real_confidence_list = list()
# v = NDCG(ranked_predicted_confidence_list, ranked_real_confidence_list, 'linear', top_k=10)
# print('linear NDCG =', v)

# Visualization of testing for adding evidence to raise posterior probability
m = 0  # drop some first trivial values
X = np.array([i for i in range(m, len(fact_0_confidences))])
Y = np.array([10*i for i in fact_0_confidences[m:]])
print(Y)
with open('../data/cn15k/confidenceupdate.pkl', 'wb') as conf:
    pickle.dump((X,Y), conf)

Y = np.sort(Y)
coefficients = np.polyfit(X, Y, 1)
polynomial = np.poly1d(coefficients)
Y_ = polynomial(X)  # linear fitting

plt.plot(X, Y_)
plt.scatter(X, Y, color='red', marker='o')
plt.ylabel('$\\times\\ 10^{-1}$')
plt.xlabel('Number of added factual instances')
plt.savefig('../log/confidenceupdated.png')
plt.show()

