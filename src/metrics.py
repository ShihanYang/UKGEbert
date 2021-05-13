"""
================================================================================
@In Project: ukg_BERT
@File Name: metrics.py
@Author: Shihan Yang
@E-Mail: dr.yangsh@kust.edu.cn
@Create Date: 2020/12/31
@Update Date: 
@Version: 0.1.0
@Functions: 
    1. To evaluate link-predicting on hits@1,hits@3,hits@10;
       To evaluate MR and MRR over test dataset；
       based on vectors of entities and relations in files: entity.vec and relation.vec.
    2. Notes:
================================================================================
"""

import os
import time
import tqdm
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # only ERROR messages
basedir = os.path.abspath("..")
print("Base directory : ", basedir)
data_name = 'CN15k\\'
data_path = basedir + '\\data\\'
data_base = data_path + data_name

test_file = data_base + 'test.tsv'  # all testing samples with confidence: 'h_id r_id t_id confidence'
train_file = data_base + 'train.tsv'

entity_vector_file = data_base + 'entity20201230.vec'  # each line likes 'e_string vector'
relation_vector_file = data_base + 'relation20201230.vec'  # each line likes 'r_string vector'

entity_id_file = data_base + 'entity_id.csv'  # each line likes 'e_string,e_id'
relation_id_file = data_base + 'relation_id.csv'  # each line likes 'r_string,r_id'

# Step 1, evaluating link-prediction on hits@X, MR and MRR
embedding = dict()
entities = list()
with open(entity_vector_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        ll = line.strip().split('\t')
        temp = ll[1][1:-1].split(', ')
        embedding[ll[0]] = list(map(float, temp))
        if ll[0] not in entities:
            entities.append(ll[0])
print('enities total:', len(entities))
relations = list()
with open(relation_vector_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        ll = line.strip().split('\t')
        temp = ll[1][1:-1].split(', ')
        embedding[ll[0]] = list(map(float, temp))
        if ll[0] not in relations:
            relations.append(ll[0])
print('relations total:', len(relations))
print("embedding vectors:", len(embedding), "with dim = 768")
print("Embeddings loaded.")

test_triples = list()
with open(test_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        ll = line.strip().split()
        temp = ll[:-1]  # without confidence here
        test_triples.append(temp)
print('Testing samples %d loaded.' % len(test_triples))

# id mapping to object name
id_relation = dict()
id_entity = dict()
with open(relation_id_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        item = line.strip().split(',')
        id_relation[item[1]] = item[0]
print('ID-RelationName %d created.' % len(id_relation))
with open(entity_id_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        item = line.strip().split(',')
        id_entity[item[1]] = item[0]
print('ID-EntityName %d created.' % len(id_entity))

sentences = dict()  # a sentence like '(h,r)':[tails] for measuring hits@X
with open(train_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        ll = line.strip().split()
        if (ll[0], ll[1]) not in sentences.keys():
            sentences[(ll[0], ll[1])] = [ll[2]]
        else:
            sentences[(ll[0], ll[1])].append(ll[2])
print('Sentences ((h,r):[tails]) %d ready.\n' % len(sentences))

print("Metrics on hits@1, hits@3, hits@10, MR, MRR ...")
hits1 = 0
hits3 = 0
hits10 = 0
sum_of_rank = 0
sum_of_reciprocal_rank = 0
time.sleep(0.1)
for triplet in tqdm.tqdm(test_triples, ncols=80):
    fakes = list()
    h_vec = embedding[id_entity[triplet[0]]]
    r_vec = embedding[id_relation[triplet[1]]]
    t_vec = embedding[id_entity[triplet[2]]]
    h_add_r = np.array(h_vec) + np.array(r_vec)
    distance = np.sqrt(np.sum(np.square(h_add_r - np.array(t_vec))))
    for tail in entities:  # all entities should substitute for tail
        if tail == triplet[2]:  # filtered
            continue
        if (triplet[0], triplet[1]) in sentences and \
              tail in sentences[(triplet[0], triplet[1])]:
            continue
        fakes.append(embedding[tail])
    score_list = [np.sqrt(np.sum(np.square(np.array(h_vec) + np.array(r_vec) - np.array(i)))) for i in fakes]
    # Ranking
    rank = 0
    for c in score_list:
        if c <= distance:
            rank += 1
    sum_of_rank += rank
    if rank != 0:
        sum_of_reciprocal_rank += 1/rank
    if rank <= 10:
        hits10 += 1
        if rank <= 3:
            hits3 += 1
            if rank <= 1:
                hits1 += 1

total_test_samples = len(test_triples)
print('For {%d} test samples (relation facts without confidence):' % (total_test_samples))
print("hits@1:", float(hits1) / total_test_samples)
print("hits@3:", float(hits3) / total_test_samples)
print("hits@10:", float(hits10) / total_test_samples)
print('MR:', float(sum_of_rank) / total_test_samples)
print('MRR:', float(sum_of_reciprocal_rank) / total_test_samples)



