#!/usr/bin/env python37
# -*- coding: utf-8 -*-
"""
Created on 17 Sep, 2019

Reference: https://github.com/CRIPAC-DIG/SR-GNN/blob/master/datasets/preprocess.py

预处理基本流程：
1. 创建两个字典sess_clicks和sess_date来分别保存session的相关信息。两个字典都以sessionId为键，其中session_click以一个Session中用户先后点击的物品id
构成的List为值；session_date以一个Session中最后一次点击的时间作为值，后续用于训练集和测试集的划分；
2. 过滤长度为1的Session和出现次数小于5次的物品；
3. 依据日期划分训练集和测试集。其中Yoochoose数据集以最后一天时长内的Session作为测试集，Diginetica数据集以最后一周时长内的Session作为测试集；
4. 分解每个Session生成最终的数据格式。每个Session中以不包括最后一个物品的其他物品作为特征，以最后一个物品作为标签。同时把物品的id重新编码成从1开始递增的自然数序列
"""

import argparse
import time
import csv
import pickle
import operator
import datetime
import os
from tqdm import tqdm



# add a header for yoochoose dataset
with open('data/ml-100k_daysession.dat', 'r') as f, open('data/ml-100kwithHeader.dat', 'w') as fn:
    fn.write('sessionId,timestamp,itemId'+'\n')
    for line in f:
        fn.write(line)


dataset = 'data/ml-100kwithHeader.dat'


print("-- Starting @ %ss" % datetime.datetime.now())
with open(dataset, "r") as f:
    reader = csv.DictReader(f, delimiter=',')
    sess_clicks = {}
    sess_date = {}
    ctr = 0
    curid = -1
    curdate = None
    for data in reader:
        sessid = data['sessionId']
        if curdate and not curid == sessid:
            date = ''
            date = curdate  #返回以秒计数的时间
            sess_date[curid] = date
        curid = sessid
        item = data['itemId']
        curdate = ''
        curdate = int(data['timestamp'])

        if sessid in sess_clicks:
            sess_clicks[sessid] += [item]
        else:
            sess_clicks[sessid] = [item]
        ctr += 1
    date = ''
    date = curdate
    sess_date[curid] = date
print("-- Reading data @ %ss" % datetime.datetime.now())



# Filter out length 1 sessions
for s in list(sess_clicks):
    if len(sess_clicks[s]) == 1:
        del sess_clicks[s]
        del sess_date[s]

# Count number of times each item appears
iid_counts = {}
for s in sess_clicks:
    seq = sess_clicks[s]
    for iid in seq:
        if iid in iid_counts:
            iid_counts[iid] += 1
        else:
            iid_counts[iid] = 1

sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))
#Fiilter item that appears less 5 times
length = len(sess_clicks)
for s in list(sess_clicks):
    curseq = sess_clicks[s]
    filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq)) #filter(function, sequence)用于过滤序列，返回一个可迭代对象
    if len(filseq) < 2:
        del sess_clicks[s]
        del sess_date[s]
    else:
        sess_clicks[s] = filseq

# Split out test set based on dates
dates = list(sess_date.items())
maxdate = dates[0][1]

for _, date in dates:
    if maxdate < date:
        maxdate = date

# 1 days for test
splitdate = 0

splitdate = maxdate - 86400 * 30 # the number of seconds for a day：86400


print('Splitting date', splitdate)      # Yoochoose: ('Split date', 1411930799.0)
tra_sess = filter(lambda x: x[1] < splitdate, dates)
tes_sess = filter(lambda x: x[1] > splitdate, dates)

# Sort sessions by date
tra_sess = sorted(tra_sess, key=operator.itemgetter(1))     # [(sessionId, timestamp), (), ]
tes_sess = sorted(tes_sess, key=operator.itemgetter(1))     # [(sessionId, timestamp), (), ]
print('训练集session数量:\t{}'.format(len(tra_sess)))    # 186670    # 7966257
print('测试集session数量:\t{}'.format(len(tes_sess)))    # 15979     # 15324
# print(tra_sess[:3])
# print(tes_sess[:3])
print("-- Splitting train set and test set @ %ss" % datetime.datetime.now())

# Choosing item count >=5 gives approximately the same number of items as reported in paper
item_dict = {}
# Convert training sessions to sequences and renumber items to start from 1
def obtian_tra():
    train_ids = []
    train_seqs = []
    train_dates = []
    item_ctr = 1
    for s, date in tra_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
            else:
                outseq += [item_ctr]
                item_dict[i] = item_ctr
                item_ctr += 1
        if len(outseq) < 2:  # Doesn't occur
            continue
        train_ids += [s]
        train_dates += [date]
        train_seqs += [outseq]
    print('item number:\t{}'.format(item_ctr))     # 43098, 37484
    return train_ids, train_dates, train_seqs


# Convert test sessions to sequences, ignoring items that do not appear in training set
def obtian_tes():
    test_ids = []
    test_seqs = []
    test_dates = []
    for s, date in tes_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
        if len(outseq) < 2:
            continue
        test_ids += [s]
        test_dates += [date]
        test_seqs += [outseq]
    return test_ids, test_dates, test_seqs


tra_ids, tra_dates, tra_seqs = obtian_tra()
tes_ids, tes_dates, tes_seqs = obtian_tes()


def process_seqs(iseqs, idates):
    out_seqs = []
    out_dates = []
    labs = []
    ids = []
    for id, seq, date in zip(range(len(iseqs)), iseqs, idates):
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
            out_dates += [date]
            ids += [id]
    return out_seqs, out_dates, labs, ids


tr_seqs, tr_dates, tr_labs, tr_ids = process_seqs(tra_seqs, tra_dates)
te_seqs, te_dates, te_labs, te_ids = process_seqs(tes_seqs, tes_dates)
tra = (tr_seqs, tr_labs)
tes = (te_seqs, te_labs)
print('训练集序列数:\t{}'.format(len(tr_seqs)))
print('测试集序列数:\t{}'.format(len(te_seqs)))
# print(tr_seqs[:3], tr_dates[:3], tr_labs[:3])
# print(te_seqs[:3], te_dates[:3], te_labs[:3])
all = 0

for seq in tra_seqs:
    all += len(seq)
for seq in tes_seqs:
    all += len(seq)
print('序列的avg length: ', all/(len(tra_seqs) + len(tes_seqs) * 1.0))

if not os.path.exists('data/ml-100k'):
    os.makedirs('data/ml-100k')
pickle.dump(tra, open('data/ml-100k/train.txt', 'wb'))
pickle.dump(tes, open('data/ml-100k/test.txt', 'wb'))
pickle.dump(tra_seqs, open('data/ml-100k/all_train_seq.txt', 'wb'))


print('Done.')
