#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 20:15:36 2018

@author: changyueh
"""

import time
import pandas as pd
import lightgbm as lgb
from functions import create_time, comb_click_time, comb_click_feature, drop_features


#Model: xgboost
##import train and test 
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
df_train = pd.read_csv('train.csv', skiprows=range(1,140000000), 
                       usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed'],
                       dtype=dtypes)
df_test = pd.read_csv('test.csv', usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id'],
                      dtype=dtypes)
print('It takes {:.2f} seconds for importing two dataset.'.format(time.time()-start_time))

##create time feature
start_time = time.time()
df_train = create_time(df_train)
df_test = create_time(df_test)
print('It takes {:.2f} seconds for converting string to datetime.'.format(time.time()-start_time))

#create combination freature
start_time = time.time()
df_train = comb_click_time(df_train, features=['ip'], groups=['hour', 'day'])
df_test = comb_click_time(df_test, features=['ip'], groups=['hour', 'day'])
print('It takes {:.2f} seconds for creating new features by time.'.format(time.time()-start_time))

start_time = time.time()
df_train = comb_click_feature(df_train, features=['ip'], groups=['app'])
df_test = comb_click_feature(df_test, features=['ip'], groups=['app'])
print('It takes {:.2f} seconds for creating new features by dif features.'.format(time.time()-start_time))


##drop features to save spaces
start_time = time.time()
df_train = drop_features(df_train, features=['ip', 'click_time', 'day', 'click'])
df_test = drop_features(df_test, features=['ip', 'click_time', 'day', 'click'])
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

##Test
submit = pd.DataFrame()
submit['click_id'] = df_test['click_id'].values
df_test = df_test.drop(['click_id'], axis=1)

print('Start to run the model...')
##Model
params = {'objective': 'binary',        #binary log loss classification application
          'boosting_type': 'gbdt',      #traditional Gradient Boosting Decision Tree
          'learning_rate': 0.05,        
          ##Learning Control Parameters
          'num_leaves': 255,            #number of leaves in one tree
          'max_depth': -1,              #limit the max depth for tree model, <0 equal to no limit
          'min_child_samples': 100,     #minimal number of data in one leaf, can be use to deal with overfitting
          'min_child_weight': 0, 
          'colsample_bytree': 0.7,      #LightGBM will randomly select part of features on each iteration if feature_fraction smaller than 1.0. For example, if set to 0.8, will select 80% features before training each tree
          'subsample': 0.7,             #can be used to speed up training, can be used to deal with over-fitting
          'subsample_freq': 1,          #requency for bagging, 0 means disable bagging. 1 means will perform bagging at every 1 iteration
          'reg_alpha': 0,               #L1 regularization
          'reg_lambda': 0,              #L2 regularization
          ##IO Parameters
          'max_bin': 100,               #max number of bins that feature values will be bucketed in.
          'verbose': 0,                 #=0 Error(Warn)
          'subsample_for_bin': 200000, 
          ##Objective Parameters
          'is_unbalance': True,         #used in binary classification, set this to true if training data are unbalance 
          ##Metric Parameters
          'metric': 'auc', 
          #'scale_pos_weight':99
          }

lgbtrain = lgb.Dataset(X_train, y_train)
lgbtest = lgb.Dataset(X_test, y_test)             
evals_results = {} 
         
start_time = time.time()
bst = lgb.train(params, lgbtrain, valid_sets=[lgbtrain, lgbtest], 
                valid_names=['train','valid'], evals_result=evals_results, 
                num_boost_round=1000, early_stopping_rounds=30, verbose_eval=50, 
                feval=None)
print('[{:.2f} seconds]: Training time for LightGBM model.'.format(time.time() - start_time))
del X_train, X_test, y_train, y_test


###Print features which used in the model
print('Feature names:', bst.feature_name())

###Print features' importances
print('Feature importances:', list(bst.feature_importance()))

##Predict
print('Start the prediction...')
submit['is_attributed'] = bst.predict(df_test, num_iteration=bst.best_iteration)
submit = submit.sort_values(by='click_id')
del df_test
print('finish the prediction...')

print('Save the prediction to csv')
submit.to_csv('submit_lightgbm.csv', index=False)
del submit
print('All tasks are completed...')