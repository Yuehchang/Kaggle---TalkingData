#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 14:25:39 2018

@author: changyueh
"""
import time
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import plot_importance 

path = './project_practice/talkingdata/'

##convert time variables
def create_time(df):
    df['click_time'] = pd.to_datetime(df['click_time'])
    df['day'] = df.click_time.dt.day
    df['hour'] = df.click_time.dt.hour
    return df

##create # of click
def comb_click(df, features=[]):
    for i in features:
        if 'click_{}'.format(i) in df.columns:
            print('You had already created the new feaure click_{}'.format(i))
        
        else:        
            df['click'] = 1
            tmp = df[['{}'.format(i), 'hour', 'day', 'click']].groupby(['{}'.format(i), 'hour', 'day']).sum().sort_values(by='click', ascending=False)
            tmp.reset_index(inplace=True)
            tmp = tmp.rename(columns={'click': 'click_{}'.format(i)})
            df = pd.merge(df, tmp, on=['{}'.format(i), 'hour', 'day'])
            
            print('New features click_{} had been created'.format(i))
            
    return df

##drop features which are unnecessary for model
def drop_features(df, feature=[]):
    return df.drop(feature, axis=1)


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

df_train = pd.read_csv(path+'train.csv', skiprows=range(1,122903891), 
                       nrows=62000000, usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed'],
                       dtype=dtypes)
df_test = pd.read_csv(path+'test.csv', usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id'],
                      dtype=dtypes)

print('It takes {:.2f} second.'.format(time.time()-start_time))

##create time feature
df_train = create_time(df_train)
df_test = create_time(df_test)

#create combination freature
df_train = comb_click(df_train, features=['ip'])
df_test = comb_click(df_test, features=['ip'])


df_train = drop_features(df_train)
df_test = drop_features(df_test)

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

##Model
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
          
          
watchlist = [(xgb.DMatrix(X_train, y_train), 'train'), (xgb.DMatrix(X_test, y_test), 'valid')]

start_time = time.time()
bst = xgb.train(params, xgb.DMatrix(X_train, y_train), 50, watchlist, early_stopping_rounds = 20, verbose_eval=1)
print('[{:.2f} second]: Training time for Histogram Optimized XGBoost model'.format(time.time() - start_time))
del X_train, X_test, y_train, y_test


submit['is_attributed'] = bst.predict(xgb.DMatrix(df_test), ntree_limit=bst.best_ntree_limit)
submit = submit.sort_values(by='click_id')
submit.to_csv(path+'xgboost3.csv', index=False)

##Model evaluation

plot_importance(bst)
plt.gcf().savefig(path+'xgb_fi.png')