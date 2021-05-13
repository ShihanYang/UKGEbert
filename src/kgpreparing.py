"""
================================================================================
@In Project: ukg_BERT
@File Name: kgpreparing.py
@Author: Shihan Yang
@E-Mail: dr.yangsh@kust.edu.cn
@Create Date: 2020/12/17
@Update Date: 
@Version: 0.1.0
@Functions: 
    1. To prepare the knowledge graphs as inputting data of bert
    2. Notes:
================================================================================
"""

import os
basedir = os.path.abspath("..")
print("Base directory : ", basedir)
data_name = 'nl27k\\'
data_path = basedir + '\\data\\'
data_base = data_path + data_name
print('Dataset directory : ', data_base)

def triple_as_input():
    # goal to be : [CLS] head relation tail [SEP]
    #         or : [CLS] head [SEP] relation [SEP] tail [SEP]
    # drop the confidence value
    triples = list()
    with open(data_base + 'triples.tsv', 'r') as tsf:
        lines = tsf.readlines()
        for line in lines:
            triple = line.strip().split('\t')
            triple = triple[:3]
            triples.append(triple)
    return triples


if __name__ == '__main__':
    triples = triple_as_input()
    print('size:', len(triples))
    for triple in triples:
        text = '[CLS] ' + ' '.join(triple) + ' [SEP]'
        print(text)

    # triples = triple_as_input()
    print('size:', len(triples))
    for triple in triples:
        text = '[CLS] ' + triple[0] + ' [SEP] ' + \
                          triple[1] + ' [SEP] ' + \
                          triple[2] + ' [SEP]'
        print(text)
