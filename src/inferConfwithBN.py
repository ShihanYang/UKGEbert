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


# load bayesian network from pickle file
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

fact_0 = ()
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



# Visualization of testing for adding evidence to raise posterior probability
X = np.array([i for i in range(4, len(fact_0_confidences))])  # drop some trivial values
Y = np.array([100*i for i in fact_0_confidences[4:]])

coefficients = np.polyfit(X, Y, 1)
polynomial = np.poly1d(coefficients)
Y_ = polynomial(X)  # linear fitting

plt.plot(X, Y_)
plt.scatter(X, Y, color='red', marker='o')
plt.ylabel('$\times 10^{-2}$')
plt.xlabel('Scale of factual instances')
plt.show()

