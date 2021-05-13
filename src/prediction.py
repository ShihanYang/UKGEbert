"""
================================================================================
@In Project: ukg_BERT
@File Name: prediction.py
@Author: Shihan Yang
@E-Mail: dr.yangsh@kust.edu.cn
@Create Date: 2021/01/17
@Update Date: 
@Version: 0.1.0
@Functions: 
    1. To perform some kinds of inference based on predicted confidence
    2. Notes: 
================================================================================
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # only ERROR messages
import numpy as np
from keras.models import load_model
from src.nDCG import NDCG

dataset = 'CN15k'
base = os.path.abspath('..') + '\\data\\' + dataset + '\\'
model_file = base + 'model_e100_768d.h5'
entity_vec_file = base + 'entity.vec'
relation_vec_file = base + 'relation.vec'

model = load_model(model_file)
print('model:', model)
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
# print('entity embedding', embedding['pricing'])
# print('relation embedding', embedding['relatedto'])
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
print('Original confidence LOADED with %d facts' % (len(triple_with_confidence)))

##########################################
# INFERRING CASE -- transitivity
##########################################

print('---------------')
print(' Predicting... ')
print('---------------')

# list top k of highest confidence value
head = 'fork'
relation = 'isa'
topk = len(word_id)  # bigger(1000) or all entities if calculating NDCG
ndcg = list()
idcg = list()
candidate = {x + 1: ('', 0) for x in range(topk)}
for tail in word_id.keys():
    if tail == head or 'r' in word_id[tail]:
        continue
    # triplet = [word_id[head], word_id[relation], word_id[tail]]
    triplet = [head, relation, tail]
    tv = [[embedding[x] for x in triplet[:3]]]
    score = model.predict(np.asarray(tv), verbose=0)[0][0]
    min_rank = min(candidate, key=lambda x: candidate[x][1])
    if score > candidate[min_rank][1]:
        candidate[min_rank] = (tail, score)
print('Prediction for (HEAD, RELATION, ?tail?) RANKING:')
print('Head = \'', head, '\'')
print('Relation = \'', relation, '\'')
rank = sorted(candidate, key=lambda x: candidate[x][1], reverse=True)
# listing top 10 candidate tails
for i in range(10):
    print('  candidate %d = %s' % (i + 1, candidate[rank[i]]))

seen = 0
for i in rank:  # search facts in ranked list are also seen in the knowledge base
    if candidate[i][0] == '':
        break
    triple = (word_id[head], word_id[relation][1:], word_id[candidate[i][0]])
    int_triple = (int(triple[0]), int(triple[1]), int(triple[2]))
    if int_triple in triple_with_confidence.keys():
        true_value = triple_with_confidence[int_triple]
    else:
        true_value = 'N/A'
        continue
    seen += 1
    ndcg.append(candidate[i][1])
    idcg.append(true_value)
    print('    seen fact', candidate[i], '& true value =', true_value)
print('    Total seen facts:', seen, '/', topk)

print('True ranking for (', head, ',', relation, ', *tails* ):')
for_ranking = []  # like [(tail, confidence), (-,-), ...]
for i in triple_with_confidence.keys():
    if i[0] == int(word_id[head]) and \
            i[1] == int(word_id[relation][1:]):  # cut of the prefix 'r', changed into int
        for_ranking.append((id_word[str(i[2])], triple_with_confidence[i]))
ranked = sorted(for_ranking, key=lambda x: x[1], reverse=True)
print('  attending rank facts:', len(ranked))
for i in range(len(ranked)):
    print('    true tail %d : %s' % (i + 1, ranked[i]))

# calculate NDCG of (h,r,?)
print('--------------------')
print(' NDCG evaluating... ')
print('--------------------')
print('ndcg list:', len(ndcg), ndcg[:9])
print('idcg list:', len(idcg), idcg[:9])
idcg.sort(reverse=True)
print('  ranked idcg list:', idcg[:9])
linear_ndcg_value = NDCG(ndcg, idcg, 'linear')
exponential_ndcg_value = NDCG(ndcg, idcg)  # exponential
print('linear NDCG for (%s, %s, %s) = %f' % (head, relation, '?', linear_ndcg_value))
print('exponetial NDCG for (%s, %s, %s) = %f' % (head, relation, '?', exponential_ndcg_value))

##########################################
# INFERRING CASE -- commonsense
##########################################

# refer to the file reasoner.py
