"""
================================================================================
@In Project: ukg_BERT
@File Name: nDCG.py
@Author: Shihan Yang
@E-Mail: dr.yangsh@kust.edu.cn
@Create Date: 2021/04/16
@Update Date: 
@Version: 0.1.0
@Functions: 
    1. To calculate mean normalized DCG (Discounted Cumulative Gain):
       - linear NDCG
       - exponential NDCG
    2. Notes: nDCG is to evaluate ranking accuracy, whose value is between 0 and 1,
              and the higher value, the more accurate the ranking is
    3. formula:
       Gain = r(i)  , the gain of ith item is r(i)
       Cumulative Gain = CG@k = $\Sum^K_i (r(i))$
       Discounted Cumulative Gain = DCG@k = $\Sum^k_i \frac{r(i)}{log_2(i+1)}$  (linear)
       Discounted Cumulative Gain = DCG@k = $\Sum^k_i \frac{2^{r(i)}-1}{log_2(i+1)}$  (exponential)
       Normalized Discounted Cumulative Gain (for each users u):
           NDCG_u@k = \frac{DCG_u@k} {IDCG_u}   (IDCG_u is the Ideal DCG of u)
       so, mean NDCG is as following:
           NDCG@k = \frac{\Sum_u NDCG_u@k} {|u|}
================================================================================
"""
import math

import numpy as np


def linear_DCG(ranked_gain_list):
    # ranked_gain_list: the position of item is the rank, and the value is the gain of it
    sum = 0
    for i in range(len(ranked_gain_list)):
        sum += ranked_gain_list[i] / np.log2(i + 2)  # i begins with 0
    return sum


def exponential_DCG(ranked_gain_list):
    sum = 0
    for i in range(len(ranked_gain_list)):
        sum += (2 ** ranked_gain_list[i] - 1) / np.log2(i + 2)
    return sum


def NDCG(ranked_gain, ranked_real, DCG_type='exponential', top_k=None):
    '''
    :param ranked_gain: a ranked prediction list
    :param ranked_real: the ranked list
    :param DCG_type: linear or exponential NDCG
    :param top_k: only top k items should be considered
    :return NDCG
    '''
    dcg = 0
    ideal_dcg = 1
    ndcg = 0
    if top_k is not None:
        if DCG_type == 'linear':
            dcg = linear_DCG(ranked_gain[:top_k])
            ideal_dcg = linear_DCG(ranked_real[:top_k])
        else:
            dcg = exponential_DCG(ranked_gain[:top_k])
            ideal_dcg = exponential_DCG(ranked_real[:top_k])
    else:
        if DCG_type == 'linear':
            dcg = linear_DCG(ranked_gain)
            ideal_dcg = linear_DCG(ranked_real)
        else:
            dcg = exponential_DCG(ranked_gain)
            ideal_dcg = exponential_DCG(ranked_real)
    if ideal_dcg == 0:
        return 0.0
    ndcg = dcg / ideal_dcg
    if ndcg > 1:  # note: need to normalize it
        ndcg = 1 / ndcg  # or normalizing as 1 / (1 + math.exp(-ndcg))
    # Normalizing, but the value will be reduced.
    # ndcg = 1 / (1 + math.exp(-ndcg))
    return ndcg


def mean_NDCG(list_pred_list, list_real_list, DCG_type='exponential', top_k=None):
    # all lists here is already sorted
    sum_ndcg = 0
    length = len(list_pred_list)
    if top_k is not None and top_k < length:
        length = top_k
    for i in range(length):
        sum_ndcg += NDCG(list_pred_list[i], list_real_list[i], DCG_type, top_k)
    return sum_ndcg / length


if __name__ == '__main__':
    list_pred_list = list()
    list_real_list = list()
    # for (rush, relatedto, ?)
    l1 = [0.709, 0.701, 0.676, 0.669, 0.661]  # sorted prediction confidence
    l2 = [0.968, 0.709, 0.659, 0.105, 0.709]  # sorted real confidence
    l2.sort(reverse=True)
    list_pred_list.append(l1)
    list_real_list.append(l2)
    print(NDCG(l1, l2, 'linear'))
    print(NDCG(l1, l2))

    # for (hotel, usedfor, ?)
    l1 = [0.856, 0.761, 0.694, 0.689, 0.637]
    l2 = [1.0, 0.984, 0.709, 0.709, 0.893]
    l2.sort(reverse=True)
    list_pred_list.append(l1)
    list_real_list.append(l2)
    print(NDCG(l1, l2, 'linear'))
    print(NDCG(l1, l2))

    # for (fork, isa, ?)
    l1 = [0.9912731, 0.9814315, 0.9769801, 0.89755726, 0.8726656, 0.6140684]
    l2 = [0.8927087856574166, 0.8927087856574166, 0.8927087856574166, 0.709293243275961, 0.709293243275961, 0.709293243275961]
    list_pred_list.append(l1)
    list_real_list.append(l2)
    print(NDCG(l1, l2, 'linear'))
    print(NDCG(l1, l2))

    # mean value for all above NDCGs
    mean_ndcg = mean_NDCG(list_pred_list, list_real_list, 'linear')
    print('mean linear NDCG =', mean_ndcg * 100, '%')
    mean_ndcg = mean_NDCG(list_pred_list, list_real_list)
    print('mean exponential NDCG =', mean_ndcg * 100, '%')
