#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import json
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif,chi2
import numpy as np
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import jieba
jieba.setLogLevel(jieba.logging.INFO)
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from lightgbm.sklearn import LGBMClassifier
from catboost import CatBoostClassifier
import joblib
import random
from sklearn.metrics import roc_auc_score
import pickle


input_data_path = '/home/admin/workspace/job/input/train.jsonl'
input_data_path = '/mnt/atec/train.jsonl'
output_model_path = '/home/admin/workspace/job/output/your-model-name'
result_path = '/home/admin/workspace/job/output/result.json'

# 读取训练数据进行训练
info = []
with open(input_data_path, 'r', encoding='utf-8') as fp:
    for line in fp.readlines():
        input_data = json.loads(line.strip())
        info.append(input_data)

print('load data')
df = pd.json_normalize(info)


# d2v
import jieba
import string
punctuation_string = "1234567890QWERTYUIOPASDFGHJKLZXCVBNMqwertyuiopasdfghjklzxcvbnm，。！*"
strs=list(df['memo_polish'].values)
content = []
count = 0
for str in strs:
    try:
        for i in punctuation_string:
            str = str.replace(i, '')
        seg_list = jieba.cut(str, cut_all=False)
    except:
        seg_list = jieba.cut(str, cut_all=False)
    try:
        content.append(list(seg_list))
    except:
        count += 1

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(content)]
# model = Doc2Vec(documents, vector_size=64, window=3, min_count=1, workers=4,seed=2021)
# model.save('d2v_model')
model = Doc2Vec.load('d2v_model')

d2v = []
for idx,row in df.iterrows():
    test = row['memo_polish']
    try:
        for i in punctuation_string:
            test = test.replace(i, '')
        seg_list = jieba.cut(test, cut_all=False)
    except:
        seg_list = jieba.cut(test, cut_all=False)
    try:
        d2v.append(model.infer_vector(list(seg_list)))
    except:
        d2v.append(np.zeros(64,))
d2v = np.array(d2v)
print(d2v.shape)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
content2 = [' '.join(i) for i in content]
tfidf_model = TfidfVectorizer().fit(content2)
pickle.dump(tfidf_model, open("tfidf_model.pickle", "wb"))
tfidf_model = pickle.load(open("tfidf_model.pickle", "rb"))
content3 = []
count = 0 
for idx,row in df.iterrows():
    test = row['memo_polish']
    try:
        for i in punctuation_string:
            test = test.replace(i, '')
        seg_list = jieba.cut(test, cut_all=False)
    except:
        seg_list = jieba.cut(test, cut_all=False)
    try:
        content3.append(','.join(list(seg_list)))
    except:
        content3.append('')
svdT = TruncatedSVD(n_components=64,random_state=2021)
svdT.fit(tfidf_model.transform(content3))
pickle.dump(svdT, open("svdT.pickle", "wb"))
svdT = pickle.load(open("svdT.pickle", "rb"))
# pickle.dump(svdT, "svdT.p")
# with open('svdT.p', 'r') as fp:
#     svdT = pickle.load(fp)
svdTFit = svdT.transform(tfidf_model.transform(content3))
print(svdTFit.shape)

features= [f'x{i}' for i in range(480)]
df_data = df[features].values
d2v = np.hstack([df_data,d2v,svdTFit])
# d2v = df_data.values
print('data shape',d2v.shape)

pos_idx = df[df['label'] == 1].index
neg_idx = df[df['label'] == 0].index
train_idx = list(pos_idx) + list(neg_idx)

train_y = df['label'].iloc[train_idx]
train_y = train_y.values
train_x = d2v[train_idx]

uknown = list(df[df['label'] == -1].index)
test_x = d2v[uknown]


# from sklearn.metrics import roc_auc_score

skf = StratifiedKFold(n_splits=5,
                  shuffle=True,
                  random_state=2021)
fold = 1
acc = 0
for train_index, valid_index in skf.split(train_x, train_y):
    clf = LGBMClassifier(
        learning_rate=0.01,
        n_estimators=10000,
        num_leaves=128,
        reg_alpha=0.6947,
        reg_lambda=0.938,
        subsample=0.6867,
        colsample_bytree=0.7242,
        random_state=2021,
        metric='None',
        class_weight ='balanced'
    )
    clf.fit(
        train_x[train_index], train_y[train_index],
        eval_set=[(train_x[valid_index], train_y[valid_index])],
        eval_metric='auc',
        early_stopping_rounds=50,
        verbose=500
    )
    pred = clf.predict_proba(train_x[valid_index])[:, 1]
    acc += roc_auc_score(train_y[valid_index],pred)
    if fold == 1:
        test_pred = clf.predict_proba(test_x)[:, 1]/5
    else:
        test_pred += clf.predict_proba(test_x)[:, 1]/5
    joblib.dump(clf, f'lgb_model_{fold}.pkl')
    fold+=1
#     break


fake = list(np.where(test_pred > 0.90)[0])+list(np.where(test_pred < 0.05)[0])
fake_x = test_x[fake]
fake_y = test_pred[fake]
fake_y[fake_y>0.5]=1
fake_y[fake_y<0.5]=0
train_x = np.concatenate([train_x,fake_x])
train_y = np.concatenate([train_y,fake_y])

# from sklearn.metrics import roc_auc_score

skf = StratifiedKFold(n_splits=5,
                  shuffle=True,
                  random_state=2021)
fold = 1
acc = 0
for train_index, valid_index in skf.split(train_x, train_y):
    clf = LGBMClassifier(
        learning_rate=0.01,
        n_estimators=10000,
        num_leaves=128,
        reg_alpha=0.6947,
        reg_lambda=0.938,
        subsample=0.6867,
        colsample_bytree=0.7242,
        random_state=2021,
        metric='None',
        class_weight ='balanced'
    )
    clf.fit(
        train_x[train_index], train_y[train_index],
        eval_set=[(train_x[valid_index], train_y[valid_index])],
        eval_metric='auc',
        early_stopping_rounds=50,
        verbose=500
    )
    pred = clf.predict_proba(train_x[valid_index])[:, 1]
    acc += roc_auc_score(train_y[valid_index],pred)
    if fold == 1:
        test_pred = clf.predict_proba(test_x)[:, 1]/5
    else:
        test_pred += clf.predict_proba(test_x)[:, 1]/5
    joblib.dump(clf, f'lgb_model_{fold}.pkl')
    fold+=1
#     break


from bayes_opt import BayesianOptimization
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")
def bayes_parameter_opt_lgb(X, y, init_round=15, opt_round=25, n_folds=5, random_seed=6, n_estimators=10000, learning_rate=0.05, output_process=False):
    # prepare data
    train_data = lgb.Dataset(data=X, label=y, free_raw_data=False)
    # parameters
    def lgb_eval(num_leaves, feature_fraction, bagging_fraction, lambda_l1, lambda_l2):
        skf = StratifiedKFold(n_splits=5,
                  shuffle=True,
                  random_state=2021)
        acc=[]
        for train_index, valid_index in skf.split(train_x, train_y):
            clf = LGBMClassifier(
                learning_rate=0.01,
                n_estimators=5000,
#                 max_depth = int(round(max_depth)),
                num_leaves=int(round(num_leaves)),
                reg_alpha=max(lambda_l1, 0),
                reg_lambda=max(lambda_l2, 0),
                subsample=max(min(feature_fraction, 1), 0),
                colsample_bytree=max(min(bagging_fraction, 1), 0),
#                 min_split_gain = min_split_gain,
#                 min_child_weight = min_child_weight,
                random_state=2021,
                metric='None',
                class_weight ='balanced'
            )
            clf.fit(
                train_x[train_index], train_y[train_index],
                eval_set=[(train_x[valid_index], train_y[valid_index])],
                eval_metric='auc',
                early_stopping_rounds=100,
                verbose=False
            )
            
            pred = clf.predict_proba(train_x[valid_index],num_iteration=clf.best_iteration_)[:, 1]
            acc.append(roc_auc_score(train_y[valid_index],pred))
        print("AUC:",np.mean(acc))
        return np.mean(acc)
    # range 
    lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (32, 128),
                                            'feature_fraction': (0.1, 0.9),
                                            'bagging_fraction': (0.3, 1),
#                                             'max_depth': (5, 15.99),
                                            'lambda_l1': (0, 5),
                                            'lambda_l2': (0, 3)
                                            }, random_state=2021)
    # optimize
    lgbBO.maximize(init_points=init_round, n_iter=opt_round)
    
    # output optimization process
    if output_process==True: lgbBO.points_to_csv("bayes_opt_result.csv")
    
    # return best parameters
    return lgbBO.res['max']['max_params']

opt_params = bayes_parameter_opt_lgb(train_x, train_y, init_round=5, opt_round=100, n_folds=5, random_seed=6, n_estimators=5000, learning_rate=0.01)