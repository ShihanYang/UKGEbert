"""
================================================================================
@In Project: ukg_BERT
@File Name: kgbert2.py
@Author: Shihan Yang
@E-Mail: dr.yangsh@kust.edu.cn
@Create Date: 2022/7/12
@Update Date: 
@Version: 0.1.1
@Functions: 
    1. To input triples into bert without confidence as (head, relation, tail)
    2. Notes:
================================================================================
"""

import os
import tqdm
from src.kgpreparing import triple_as_input

basedir = os.path.abspath("..")
print("Base directory : ", basedir)
model_name = 'bert-base-uncased'
model_path = basedir + '\\models\\'
model_base = model_path + model_name
# model_base = 'D:\\pyCharmWorkspace\\ukg_BERT\\models\\' + model_name  # for testing
print("Pre-trained model directory : ", model_base)
data_name = 'cn15k\\'
data_path = basedir + '\\data\\'
data_base = data_path + data_name
print('Dataset directory : ', data_base)

import torch
from transformers import BertModel, BertTokenizer, BertConfig

tokenizer = BertTokenizer.from_pretrained(model_base)  # best to use local downloaded models
config = BertConfig.from_pretrained(model_base, output_hidden_states=True)
model = BertModel.from_pretrained(model_base, config=config)
model.train()


def bertvec(all_triples, batch_bottom, batch_up):
    triples = all_triples[batch_bottom:batch_up]  # for fast running by batching dataset
    triples_vec = dict()
    for triple in tqdm.tqdm(triples):
        text = '[CLS] ' + triple[0] + ' [SEP] ' + \
               triple[1] + ' [SEP] ' + \
               triple[2] + ' [SEP]'
        tokenized_text = tokenizer.tokenize(text)
        position = list()  # [end_pos_head, end_pos_relation, end_pos_tail]
        for i in range(len(tokenized_text)):
            if tokenized_text[i] == '[SEP]':
                position.append(i)
        assert len(position) == 3
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)  # entity for 1, relation for 0
        for i in range(position[0] + 1, position[1] + 1):
            segments_ids[i] = 0
        position_ids = [i for i in range(len(tokenized_text))]
        token_tensor = torch.tensor([indexed_tokens])
        segment_tensor = torch.tensor([segments_ids])
        position_tensor = torch.tensor([position_ids])
        outputs = model(input_ids=token_tensor,
                        token_type_ids=segment_tensor,
                        position_ids=position_tensor)
        # fetch [CLS] embedding as triple vector, first segment as head vector,
        #       second segment as relation vector, third segment as tail vector
        vectors = list()  # [[triple vec], [head vec], [relation vec], [tail vec]]
        triple_vec = outputs[1].detach().numpy().tolist()[0]  # pooled layer
        hidden = outputs[2]  # all hidden state layers

        # strategy one: the last hidden layer as the hrt vectors
        # strategy two: the mean of last four hidden layers as the hrt vectors

        hidden = outputs[0][0]  # strategy one as following

        head_span = [1, position[0]]
        head_vec = [0] * 768
        for i in range(head_span[0], head_span[1]):
            head_vec += hidden[i].detach().numpy()
        head_vec = [i / (head_span[1] - head_span[0]) for i in head_vec]
        # print(head_vec)

        relation_span = [position[0] + 1, position[1]]
        relation_vec = [0] * 768
        for i in range(relation_span[0], relation_span[1]):
            relation_vec += hidden[i].detach().numpy()
        relation_vec = [i / (relation_span[1] - relation_span[0]) for i in relation_vec]
        # print(relation_vec)

        tail_span = [position[1] + 1, position[2]]
        tail_vec = [0] * 768
        for i in range(tail_span[0], tail_span[1]):
            tail_vec += hidden[i].detach().numpy()
        tail_vec = [i / (tail_span[1] - tail_span[0]) for i in tail_vec]

        vectors.append(triple_vec)
        vectors.append(head_vec)
        vectors.append(relation_vec)
        vectors.append(tail_vec)
        assert len(vectors) == 4
        tri_tuple = (triple[0], triple[1], triple[2])
        triples_vec[tri_tuple] = vectors

    print('triples vector:', len(triples_vec))

    # save the embedding vectors of triples
    with open(data_base + 'triples_as_hrt.vec', 'w') as vec_file:
        for triple in triples_vec.keys():
            vec_file.write(str(triple))
            vec_file.write('\t')
            vec_file.write(str(triples_vec[triple]))
            vec_file.write('\n')
    print('Embedding vectors file saved in:\n  ', os.path.abspath(vec_file.name),
          os.path.getsize(vec_file.name) / 1e+6, 'MB')


if __name__ == '__main__':
    triples = triple_as_input()
    print("triples total:", len(triples))
    # for i in range(0, len(triples), 10000):  # avoiding memory error, batch processing in ppi5k dataset
    #     if i != 240000:
    #         step = 10000
    #     else:
    #         step = 9396
    bertvec(triples, 0, len(triples))
