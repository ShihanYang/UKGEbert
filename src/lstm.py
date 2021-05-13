"""
================================================================================
@In Project: ukg_BERT
@File Name: lstm.py
@Author: Shihan Yang
@E-Mail: dr.yangsh@kust.edu.cn
@Create Date: 2020/12/30
@Update Date: 
@Version: 0.1.0
@Functions: 
    1. To training lstm to predict confidence over entity.vec and relation.vec
    2. Notes:
================================================================================
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # only ERROR messages
import sys
import time
import numpy as np
import gc
from matplotlib import pyplot as plt
from keras import metrics, callbacks
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam
from src import pickleloss
from src.checkpoint import checkpoint_base, ukgeCheckpoint, get_last_status


def training(dataset, dimension, batch_size, epochs):
    base = os.path.abspath('..') + '\\data\\' + dataset + '\\'
    entity_vector_file = base + 'entity.vec'
    relation_vector_file = base + 'relation.vec'
    entity_id_file = base + 'entity_id.csv'
    relation_id_file = base + 'relation_id.csv'
    train_file = base + 'train.tsv'  # when small dataset for testing, without valid data outside
    test_file = base + 'test.tsv'
    valid_file = base + 'val.tsv'
    model_file = base + 'model_e'+str(epochs)+'_'+str(dimension)+'d.h5'  # for writing
    checkpoint_dir = base + 'checkpoints\\'
    checkpoint_base(checkpoint_dir)
    checkpoint_file = checkpoint_dir + os.path.basename(model_file) + '-loss.chk'
    loss_file = base + os.path.basename(model_file) + '.loss'

    embedding = dict()
    entities = list()
    dim = 768
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
    print("embedding vectors:", len(embedding), "with dim =", dim)
    print("Embeddings loaded.")

    # id mapping to object name [id: string, ...]
    id_relation = dict()
    id_entity = dict()
    with open(relation_id_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            item = line.strip().split(',')
            id_relation[item[0]] = item[1]
    print('ID-RelationName %d items are created.' % len(id_relation))
    with open(entity_id_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            item = line.strip().split(',')
            id_entity[item[0]] = item[1]
    print('ID-EntityName %d items are created.' % len(id_entity))

    X_train = list()
    y_train = list()
    sentences = dict()  # a sentence like '(h,r)':[tails] for generating negative samples
    with open(train_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            ll = line.strip().split('\t')
            if (ll[0], ll[1]) not in sentences.keys():
                sentences[(ll[0], ll[1])] = [ll[2]]
            else:
                sentences[(ll[0], ll[1])].append(ll[2])
            # h_vec = embedding[id_entity[ll[0]]]
            # r_vec = embedding[id_relation[ll[1]]]
            # t_vec = embedding[id_entity[ll[2]]]
            # for nl27k dataset, different train.tsv form
            h_vec = embedding[ll[0]]
            r_vec = embedding[ll[1]]
            t_vec = embedding[ll[2]]
            vectors = [h_vec, r_vec, t_vec]
            X_train.append(vectors)
            y_train.append(ll[3])
    print("X_train without neg samples:", len(X_train))
    print("y_train without neg samples:", len(y_train))
    print('Entities num:', len(entities))
    print('Relations num:', len(relations))
    print('Sentences num:', len(sentences))
    valid_sum = 0
    for i in sentences:
        valid_sum += len(sentences[i])
    assert valid_sum == len(y_train)
    print('Valid total num:', valid_sum)
    print("Training positive samples loaded.")

    # negative samples by corrupting strategy
    neg_num = 0
    for i in sentences:
        for j in sentences[i]:
            h = i[0]
            r = i[1]
            twice = 0
            while True:
                neg_tail_index = np.random.randint(0, len(entities))
                if entities[neg_tail_index] not in sentences[i]:
                    t = entities[neg_tail_index]
                    # hv = embedding[id_entity[h]]
                    # rv = embedding[id_relation[r]]
                    # for dataset nl27k,different form
                    hv = embedding[h]
                    rv = embedding[r]
                    tv = embedding[t]
                    X_train.append([hv, rv, tv])
                    y_train.append(1e-09)
                    neg_num += 1
                    twice += 1
                    if twice >= 1:  # get one/two/three/... negative samples each time
                        break
    print('Negative samples num:', neg_num)
    print('X_train num:', len(X_train))
    print('y_train num:', len(y_train))
    print('Train samples are ready!')

    X_test = list()
    y_test = list()
    with open(test_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            ll = line.strip().split('\t')
            # h_vec = embedding[id_entity[ll[0]]]
            # r_vec = embedding[id_relation[ll[1]]]
            # t_vec = embedding[id_entity[ll[2]]]
            # for nl27k dataset, different train.tsv form
            h_vec = embedding[ll[0]]
            r_vec = embedding[ll[1]]
            t_vec = embedding[ll[2]]
            vectors = [h_vec, r_vec, t_vec]
            X_test.append(vectors)
            y_test.append(ll[3])
    print('X_test:', len(X_test))
    print('y_test:', len(y_test))
    print('Testing samples loaded.')

    X_valid = list()
    y_valid = list()
    with open(valid_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            ll = line.strip().split('\t')
            # h_vec = embedding[id_entity[ll[0]]]
            # r_vec = embedding[id_relation[ll[1]]]
            # t_vec = embedding[id_entity[ll[2]]]
            # for nl27k dataset, different train.tsv form
            h_vec = embedding[ll[0]]
            r_vec = embedding[ll[1]]
            t_vec = embedding[ll[2]]
            vectors = [h_vec, r_vec, t_vec]
            X_valid.append(vectors)
            y_valid.append(ll[3])
    print('X_valid:', len(X_valid))
    print('y_valid:', len(y_valid))
    print('Validating samples loaded.')

    # STEP: LSTM modelling
    start = time.time()
    model = Sequential(name='bert_fine_tuning_UKGE')
    lstm = LSTM(dim, input_shape=(3, dim),
                activation='softsign',
                return_sequences=True)
    model.add(lstm)
    lstm_2 = LSTM(units=dim)
    model.add(lstm_2)
    model.add(Dense(1, activation='sigmoid'))  # softmax is not good
    model.summary()
    optimizer = Adam(lr=0.000001, beta_1=0.95, beta_2=0.999,
                     amsgrad=True,  # for nl27k
                     epsilon=1e-08, decay=0)  # default lr=0.001
    loss = 'mean_squared_error'
    # loss = 'msle'  # better ?
    model.compile(loss=loss, optimizer=optimizer,
                  metrics=[metrics.mae])  # mae and mse (=loss assigned above line202)
    print('Begin training (confidence learning) ... ...')
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    print('  X_train shape:', X_train.shape)
    print('  y_train shape:', y_train.shape)
    X_valid = np.asarray(X_valid)
    y_valid = np.asarray(y_valid)
    print('  X_valid shape:', X_valid.shape)
    print('  y_valid shape:', y_valid.shape)
    callback = callbacks.EarlyStopping(monitor='val_loss',  # (val_)loss or (val_)mean_absolute_error
                                       mode='min',
                                       min_delta=1e-06, patience=15, verbose=1)
    last_epoch, last_meta = get_last_status(model, checkpoint_file)
    checkpoint = ukgeCheckpoint(checkpoint_file, monitor='val_loss',  # loss or val_loss
                                save_weights_only=True,  # all data or only weights data
                                save_best_only=True,  # best data or latest data
                                verbose=1, meta=last_meta)
    print('  shuffling data ...')
    # shuffle training dataset first, needed for better val_loss decreasing steadily
    # permutation = np.random.permutation(X_train.shape[0])
    # shuffled_X_train = X_train[permutation, :, :]  # NOTE: big memory, how about random.shuffle()?
    # shuffled_y_train = y_train[permutation]

    # VIP: The selection of training data is very important
    # X = shuffled_X_train[:10000]; y = shuffled_y_train[:10000]  # after shuffled including negative
    # X = X_train[:10000]; y = y_train[:10000]  # before shuffling no including negative samples
    # for nl27k dataset : 149100, 298200 including negative samples
    X = X_train[:149100]; y = y_train[:149100]  # without all negative samples, no shuffling
    print('  *_memory_* X_train: %d M, y_train: %d M' %
          (sys.getsizeof(X_train)//1e06, sys.getsizeof(y_train)//1e06))
    del X_train, y_train
    gc.collect()
    time.sleep(1)

    print('  begin fitting ...')
    history = model.fit(X, y,  # training dataset selected very carefully
                        validation_data=(X_valid, y_valid),
                        epochs=epochs, batch_size=batch_size, shuffle=True, verbose=2,
                        callbacks=[callback,  # early stopping
                                   checkpoint],    # check points
                        initial_epoch=last_epoch + 1)
    model.save(model_file)
    print('Train FINISHED.')
    time_consumed = time.time() - start
    print('Time consumed(s):', time_consumed)

    print('Evaluate results:\n', 'MSE, MAE =',
          model.evaluate(np.asarray(X_test), np.asarray(y_test), batch_size=batch_size, verbose=0))

    # visualizing loss and val_loss (MSE, MAE)
    history = history.history
    pickleloss.save(history, loss_file)
    plt.plot(history['loss'])  # mse
    plt.plot(history['val_loss'])  # val_mse
    plt.plot(history['mean_absolute_error'])  # mae
    plt.plot(history['val_mean_absolute_error'])  # val_mae
    plt.title('Losses of LSTM Training')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['MSE', 'val_MSE', 'MAE', 'val_MAE'], loc='upper right')
    plt.show()

    return os.path.split(model_file)[1]


if __name__ == '__main__':
    model_saved_in = training('nl27k', 768, 16, 300)
    print('Trained model was saved in:', model_saved_in)


# NOTE: 2021-01-01 morning training('CN15k', 768, 64, 100), with patience=10, negative sampels with
#  y_train.append(1e-08) MSE, MAE = [0.0729930457055031, 0.20172813534736633]

# NOTE: 2021-01-01 evening training('CN15k', 768, 128, 100), with patience=5, negative sampels with
#  y_train.append(1e-09) GREAT MSE, MAE = [0.06618890672063268, 0.19344262778759003]

# ============================================================================================= #

# NOTE:  2021-01-03 evening training('ppi5k', 768, 128, 100), with patience=5, negative sampels wit
#  y_train.append(1e-09) MSE, MAE = [0.01889842004304268, 0.09566372632980347]

# NOTE:  2021-01-04,2021-01-05,2021-01-06 training('ppi5k', 768, 128, 50), with 2-layers LSTM,
#  patience=5, min_delta=1e-04
#  GREAT MSE, MAE = [0.011211525157615405, 0.06578247249126434] (VIP) lr=0.001

# NOTE:  2021-01-06 training('ppi5k', 768, 128, 120), with 1-layers LSTM, patience = 12
#  (VIP) lr=0.001
#  MSE, MAE = [0.01640378067143001, 0.0871322751045227]
#  continue from the checkpoint (34 epoch), with patience=12,min-delta=1e-05:
#  MSE, MAE = [0.013539432610074679, 0.07769384980201721]

# NOTE:  2021-01-08 training('ppi5k', 768, 128, 200), with 1-layers LSTM, patience = 25，
#  (VIP) lr=0.0001
#  MSE, MAE = [0.015629795897223783, 0.08673037588596344]

# NOTE:  2021-01-10 training('ppi5k', 768, 64, 250), with 2-layers LSTM, patience = 25，
#  (VIP) lr=0.0001, min_delta=1e-05
#  MSE, MAE = 0.01199  [0.013036733530598999, 0.07119816541671753]
#  2021-01-15 continue from the 39 epoch with same parameters and
#  (VIP) lr=0.0001, min_delta=1e-05, patience=10, delay=0.01,
#   with dropout 0.2,0.5 at the hidden layer 1,2, and without negative samples, no shuffling:
#  GREAT improved : MSE, MAE = [0.010249818303205526, 0.061982132494449615] @ 49 epoch
#  2021-01-16 continue from the 70 epoch with patient=15, delay=0:
#  GREAT improved again : 0.00883（until 90epoch 2021-01-17:9:10）
#    NOTE: MSE, MAE = [0.008546466285681856, 0.05389213189482689]
#
# ============================================================================================= #
#
# NOTE: 2021-01-19 training('nl27k', 768, 64, 200), with same parameters as lately above (2-layers)
#   MSE, MAE = [0.03887857010013887, 0.13944561779499054]
#   2021-01-23 morning: continue from 30 epoch with Adam(lr=0.00001, beta_1=0.9, beta_2=0.9999,
#   epsilon=1e-08, decay=0),patient=20, over 100000 positive samples, (vip)lstm.activation = 'softsign'
#   GREAT : MSE, MAE = [0.03785103807740526, 0.14384044706821442]
#    Epoch 00030: val_loss improved from 0.03653 to 0.03652

#   2021-01-26 evening training('nl27k', 768, 64, 300) from the 30th epoch, with two-layer lstm, patient=30
#    Epoch 00035: val_loss improved from 0.03646 to 0.03643
#   MSE, MAE = [0.038943822512229796, 0.14108331501483917]
#    2021-02-14  training('nl27k', 768, 16, 300) from the 36th epoch, with lr=0.000001, patient=15,
#   and adam.asmgrad=True, beta_1=0.95
#    Epoch 00061: val_loss did not improve from 0.03643
#    Epoch 00061: early stopping
#   MSE, MAE =  [0.038169189563515594, 0.14249804615974426]

# NOTE: 2021-02-07 morning training('nl27k', 768, 16, 360) with two-layer lstm,
#   patient=15, lstm.activation='softsign', lr=0.00001, decay=0
#   and (vip)adam.asmgrad=True  # paper: On the Convergence of Adam and Beyond
#   前期用Adam，享受Adam快速收敛的优势；后期切换到SGD，慢慢寻找最优解。
#   refer to the paper : "Improving Generalization Performance by Switching from Adam to SGD"
#   no improved: MSE, MAE = [0.0405034856847942, 0.1482088714838028]
#
# NOTE: 2021-01-21 evening training('nl27k', 768, 64, 150) with one-layer lstm, patient=15
#   MSE, MAE = [0.040978496617909525, 0.16282953321933746] end of the 150 epoch
#
# NOTE: 2021-01-23 evening training('nl27k', 768, 128, 350) with 3-layer lstm, patient = 30,
#   lstm.activation = 'softsign' @ 1,2 layers, adam.lr=0.00001
#   MSE, MAE = [0.04051511701596196, 0.14864884316921234]  Epoch 00146: early stopping
#   Epoch 00116: val_loss improved from 0.03817 to 0.03802

# NOTE: 2021-01-28,29 training('nl27k', 768, 32, 400), with adam.asmgrad=True, patient=15,
#   lr=0.00001, lstm1.activation='softsign'
#   MSE, MAE = [0.03918899538400796, 0.1516125500202179] after 99 epochs
# NOTE: 2021-01-31 training('nl27k', 768, 20, 400) from the 99 epoch, with adam.asmgrad=True, patient=30,
#   lr=0.000001, lstm1.activation='softsign', shuffling 100000 samples
#   MSE, MAE = [0.03925648771647937, 0.15421904623508453]
# NOTE: 2021-02-05 training('nl27k', 768, 16, 400) from the 109 epoch, with adam.asmgrad=True, patient=15,
#   lr=0.0000001, lstm1.activation='softsign', dense.activation='softmax',
#   Train on 140000 samples, validate on 7000 samples, loss='msle'
#   GREAT : MSE, MAE = [0.03194011372202301, 0.09390071034431458, 0.19408050179481506]

