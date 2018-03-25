#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 21:07:52 2018

@author: changyueh
"""
import time
import pandas as pd
import xgboost as xgb
from functions import create_time, comb_click_time, comb_click_feature, drop_features

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }

start_time = time.time()
df_train = pd.read_csv('train.csv', skiprows=range(1,122903891), 
                       nrows=62000000, usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed'],
                       dtype=dtypes)
print('It takes {:.2f} seconds for importing two dataset.'.format(time.time()-start_time))

##create time feature
start_time = time.time()
df_train = create_time(df_train)
print('It takes {:.2f} seconds for converting string to datetime.'.format(time.time()-start_time))

#create combination freature
start_time = time.time()
df_train = comb_click_time(df_train, features=['ip'], groups=['hour', 'day'])
print('It takes {:.2f} seconds for creating new features by time.'.format(time.time()-start_time))

start_time = time.time()
df_train = comb_click_feature(df_train, features=['ip'], groups=['app'])
print('It takes {:.2f} seconds for creating new features by dif features.'.format(time.time()-start_time))


##drop features to save spaces
start_time = time.time()
df_train = drop_features(df_train, features=['ip', 'click_time', 'day', 'click'])
print('It takes {:.2f} seconds for dropping features.'.format(time.time()-start_time))

##Training and test in df_train
y = df_train['is_attributed']
df_train = df_train.drop(['is_attributed'], axis=1)
X = df_train
del df_train

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
del X
del y

print('Start to tuning the model...')

##Model's params
params = {'silent': True,                   #It’s generally good to keep it 0 as the messages might help in understanding the model.
          'nthread': 8,                     #Core for using
          'eta': 0.3,                       #Analogous to learning rate in GBM
          'min_child_weight':0,             #Control overfitting.
          'max_depth': 0,                   #0 means no limit, typical values: 3-10
          'max_leaves': 1400,               #Maximum number of nodes to be added. (for lossguide grow policy)
          'subsample': 0.9,
          'colsample_bytree': 0.7,
          'colsample_bylevel': 0.7,
          'alpha': 4,                       #L1 regularization on weights | default=0 | large value == more conservative model
          'scale_pos_weight': 9,            #Bbecause training data is extremely unbalanced: used 1 in the first and second submittion. 
          'objective': 'binary:logistic',   #logistic regression for binary classification, output probability
          'eval_metric': 'auc',
          'tree_method': "hist",            #Fast histogram optimized approximate greedy algorithm. 
          'grow_policy': "lossguide",       #split at nodes with highest loss change
          'random_state': 1}

max_auc = float("Inf")
best_params = None
num_boost_round = 999

dtrain = xgb.DMatrix(X_train, y_train)

for eta in [.3, .2, .1, .05, .01, .005]:    
    print("CV with eta={}".format(eta))

    # We update our parameters
    params['eta'] = eta

    # Run and time CV
    cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=5,
            metrics=['auc'],
            early_stopping_rounds=20
          )

    # Update best score
    mean_auc = cv_results['test-auc-mean'].max()
    boost_rounds = cv_results['test-auc-mean'].argmax()
    print("\tAUC {} for {} rounds\n".format(mean_auc, boost_rounds))
    if mean_auc < max_auc:
        max_auc = mean_auc
        best_params = eta

print("Best params: {}, AUC: {}".format(best_params, max_auc))
