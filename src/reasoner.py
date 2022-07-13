"""
================================================================================
@In Project: ukg_BERT
@File Name: reasoner.py
@Author: Shihan Yang
@E-Mail: dr.yangsh@kust.edu.cn
@Create Date: 2021/04/26
@Update Date: 2022/07/12
@Version: 0.1.1
@Functions: 
    1. To do some kinds of reasoning, or commonsense reasoning on CN15k datasets
    2. Notes: a tail inferring by confidence prediction
              b transitivity inferring
              c composition inferring
              d soft logics reasoning
================================================================================
"""

import os
import time
import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # only ERROR messages
import numpy as np
from keras.models import load_model

dataset = 'CN15k'
base = os.path.abspath('..') + '\\data\\' + dataset + '\\'
model_file = base + 'model_e100_768d.h5'
test_file = base + 'test.tsv'
entity_vec_file = base + 'entity.vec'
relation_vec_file = base + 'relation.vec'

model = load_model(model_file)
model.summary()
print('Model LOADED.')

entity_id_file = base + 'entity_id.csv'
relation_id_file = base + 'relation_id.csv'
word_id = dict()  # like {rush:2047, relatedto:r0, ...}
id_word = dict()
relation_id = dict()
id_relation = dict()
with open(entity_id_file, 'r') as ef, open(relation_id_file, 'r') as rf:
    r_lines = rf.readlines()
    for r in r_lines:
        rl = r.strip().split(',')
        relation_id[rl[0]] = 'r' + rl[1]
        id_relation['r' + rl[1]] = rl[0]
        word_id[rl[0]] = 'r' + rl[1]
        id_word['r' + rl[1]] = rl[0]
    e_lines = ef.readlines()
    for e in e_lines:
        el = e.strip().split(',')
        word_id[el[0]] = el[1]
        id_word[el[1]] = el[0]
print("ID_Dictionary READY.")

embedding = dict()  # like{'pricing':[0.1135, 0.1222, ...], }
dim = 768
with open(entity_vec_file, 'r') as ef, open(relation_vec_file, 'r') as rf:
    lines = rf.readlines()
    for line in lines:
        ll = line.strip().split('\t')
        vec = ll[1:][0]
        f_vec = [float(x) for x in vec[1:-2].strip().split(',')]  # string 2 vector
        embedding[ll[0]] = f_vec
    lines = ef.readlines()
    for line in lines:
        ll = line.strip().split('\t')
        vec = ll[1:][0]
        embedding[ll[0]] = [float(x) for x in vec[1:-2].strip().split(',')]
print('Embedding vectors LOADED.')

original_file = base + 'train.tsv'
original_file_2 = base + 'test.tsv'
triple_with_confidence = dict()  # like {(h_id,r_id,t_id):confidence, }, (int,int,int):float
with open(original_file, 'r') as of, open(original_file_2, 'r') as of2:
    r_lines = of.readlines()
    for r in r_lines:
        rl = r.strip().split()
        triple_with_confidence[(int(rl[0]), int(rl[1]), int(rl[2]))] = float(rl[3])
    r_lines = of2.readlines()
    for r in r_lines:
        rl = r.strip().split()
        triple_with_confidence[(int(rl[0]), int(rl[1]), int(rl[2]))] = float(rl[3])
print('Original confidence LOADED with %d facts.\n' % (len(triple_with_confidence)))


# using prediction model as following:
print('An example of predicting confidence of a fact:')
# head = 'fork'
# head = 'machine'
# head = 'introduction'
head = 'cat'
# relation = 'isa'
# relation = 'createdby'
# relation = 'isa'
relation = 'isa'
# tail = 'hand tool'
# tail = 'people'
# tail = 'textbook'
tail = 'mammal'
triplet = [head, relation, tail]
triplet_vectors = [[embedding[x] for x in triplet[:3]]]
predicted_confidence = model.predict(np.asarray(triplet_vectors), verbose=0)[0][0]
print('  The confidence of (%s, %s, %s) is predicted as %f.' % (head, relation, tail, predicted_confidence))
occurence = 'is not'
confidence = 'N/A'
triplet_id = (int(word_id[triplet[0]]), int(relation_id[triplet[1]][1:]), int(word_id[triplet[2]]))  # 'r' prefix should be dropped
if triplet_id in triple_with_confidence:
    occurence = 'is'
    confidence = str(triple_with_confidence[triplet_id])
print('    where the fact ((%s, %s, %s) = %s) %s occurred in the knowledge base.' %
      (head, relation, tail, confidence, occurence))
# using prediction model as above.


############################################
# INFERRING CASE -- tail prediction
############################################
'''
it is a query (h,r,?t), ranking all candidates of ?t by descending order, 
and the top 1 candidate is the prediction of the tail. 
accuracy rate: (like ndcg computing, check whether top 1 in ranked list of all seen candidates of tails
               is also the top 1 in ranked list of those original tails by confidence, for all test 
               samples 19293 on CN15k.   
'''
print('\n--------------------')
print(' tail predicting... ')
print('--------------------')
# def tail_prediction(head, relation, verbose=0, topk=5):
#     head = head
#     relation = relation
#     topk = topk
#     candidate = {x+1 : ('', 0) for x in range(topk)}
#     for tail in word_id.keys():
#         if tail == head or 'r' in word_id[tail]:
#             continue
#         triplet = [head, relation, tail]
#         tv = [[embedding[x] for x in triplet[:3]]]
#         score = model.predict(np.asarray(tv), verbose=0)[0][0]
#         min_rank = min(candidate, key=lambda x: candidate[x][1])
#         if score > candidate[min_rank][1]:
#             candidate[min_rank] = (tail, score)
#     rank = sorted(candidate, key=lambda x: candidate[x][1], reverse=True)
#     # rank likes [885, 11259, 1163, 7798, 5149, 7946, ....]
#     if verbose == 1:
#         print('Prediction for (HEAD, RELATION, ?tail?) RANKING:')
#         print('Head = \'', head, '\'')
#         print('Relation = \'', relation, '\'')
#         # listing top k candidate tails
#         for i in range(topk):
#             print('  tail candidate %d = %s' % (i + 1, candidate[rank[i]]))
#     return candidate, rank
# tail_prediction('fork', 'isa', verbose = 1)

print('\n--------------------------------')
print(' tail prediction accuracy rate... ')
print('----------------------------------')
# test_triples = list()
# with open(test_file, 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         ll = line.strip().split()
#         temp = ll[:-1]  # [int,int,int] without confidence here
#         test_triples.append(temp)
# total_test_samples = len(test_triples)
# print('Testing samples %d loaded.' % total_test_samples)
#
# correct_prediction = 0
# time.sleep(0.1)
# for triplet in tqdm.tqdm(test_triples, ncols=80):
#     head = id_word[triplet[0]]
#     relation = id_relation['r'+triplet[1]]
#     candidates, ranked = tail_prediction(head, relation, topk=len(id_word))
#     # candidates likes {1:(tail,conf), 2:(tail,conf),...}
#     # ranked likes [i_id,j_id,k_id,...], ranked keys of candidates by conf.
#     # topk=1000 or len(id_word), cost too much runtime here
#     seen = 0
#     for i in range(len(ranked)):
#         candidate_triplet = (int(triplet[0]), int(triplet[1]), ranked[i])
#         if candidate_triplet in triple_with_confidence.keys():
#             seen += 1
#             # find the first ith seen facts with the tail, check whether it is also the triplet[2]
#             if id_word[triplet[2]] == candidates[ranked[i]][0]:
#                 correct_prediction += 1
#             print(head, relation, id_word[triplet[2]], candidates[ranked[i]], seen, correct_prediction)
# print('accuracy rate of predicting is', correct_prediction / total_test_samples)


############################################
# INFERRING CASE -- transitivity reasoning
############################################
print('\n---------------------------')
print(' transitivity inferring... ')
print('---------------------------')
'''
It is also a query (A,?r,C), for (A,R,B) and (B,R,C) implies (A,R,C), to check whether the top 1 candidate
of ?r is the R relation.  
'''
head = 'kitten'  # (fork,atlocation,kitchen) and (kitchen,atloaction,apartment)
tail = 'mammal'  # (central park, partof, manhattan) and (manhattan, partof, new york)
# tail = 'moon'                        # (new york, partof, united states)
topk = 5
candidate = {x + 1: ('', 0) for x in range(topk)}
for relation in word_id.keys():
    if 'r' in word_id[relation]:
        triplet = [head, relation, tail]
        tv = [[embedding[x] for x in triplet[:3]]]
        score = model.predict(np.asarray(tv), verbose=0)[0][0]
        min_rank = min(candidate, key=lambda x: candidate[x][1])
        if score > candidate[min_rank][1]:
            candidate[min_rank] = (relation, score)
print('Prediction for (HEAD, ?relation?, TAIL) RANKING:')
print('Head = \'', head, '\'')
print('Tail = \'', tail, '\'')
rank = sorted(candidate, key=lambda x: candidate[x][1], reverse=True)
# listing top k candidate tails
for i in range(topk):
    print('  relation candidate %d = %s' % (i + 1, candidate[rank[i]]))


############################################
# INFERRING CASE -- combination reasoning
############################################
print('\n---------------------------')
print(' combination inferring... ')
print('---------------------------')
'''
It is also a query (A,?r,C), for (A,R1,B) and (B,R2,C) implies (A,R3,C), to check whether the top 1 candidate
of ?r is the R3 relation, which is some kind of commonsense inferring.  
'''
head = 'car'
tail = 'people'
topk = 5
candidate = {x + 1: ('', 0) for x in range(topk)}
for relation in word_id.keys():
    if 'r' in word_id[relation]:
        triplet = [head, relation, tail]
        tv = [[embedding[x] for x in triplet[:3]]]
        score = model.predict(np.asarray(tv), verbose=0)[0][0]
        min_rank = min(candidate, key=lambda x: candidate[x][1])
        if score > candidate[min_rank][1]:
            candidate[min_rank] = (relation, score)
print('Prediction for (HEAD, ?relation?, TAIL) RANKING:')
print('Head = \'', head, '\'')
print('Tail = \'', tail, '\'')
rank = sorted(candidate, key=lambda x: candidate[x][1], reverse=True)
# listing top k candidate tails
for i in range(topk):
    print('  relation candidate %d = %s' % (i + 1, candidate[rank[i]]))


############################################
# INFERRING CASE -- commonsense reasoning
############################################
print('\n---------------------------')
print(' commonsense inferring... ')
print('---------------------------')
'''
commonsense inferring:  Common sense inference vs. Mathematical inference
    Common sense inference =
        Imprecise definitions
        + Contingent statements
        + Incomplete reasoning
        + Breadth-first exploration
        + Incremental processing
    -(
      Non-monotonic reasoning, Non-monotonic logic and default logics, 
      Circumscription,Situation Calculus, 
      Formalization of Context, Fuzzy logic and probabilistic logics (e.g. Bayesian),
      Multiple-valued logic (yes, no, maybe, dunno), Modal logic (necessary, possible)
     )
    -Common sense inference is neither consistent nor complete.
    -With common sense apps, you might learn stuff while the system is inferring, The user might give you interactive feedback
    -Example-based approaches:
        Go from specific to general rather than general to specific
        Programming by Example
        Case-Based Reasoning
        Reasoning by Analogy
        Abduction
    Mathematical inference =
        Exact definitions
        + Universally true statements
        + Complete reasoning
        + Depth-first exploration
        + Batch processing
'''
