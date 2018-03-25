#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 17:03:12 2018

@author: changyueh
"""
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

path = './project_practice/talkingdata/'

#Import train sample for inspecting  
df_train = pd.read_csv(path+'train_sample.csv')

#Detect the feature of missing value
round(df_train.isnull().sum() / df_train.shape[0], 3) #attributed_time contain 0.98 missing value

#Drop attributed_time for two reason: first missing value percentage over 98%, second is_attributed contribute the same info as attributed_time
df_train = df_train.drop('attributed_time', axis=1)

#Analyze feature with plot
df_train.columns

## all features have more than 100 differents divices or sources. 
fcols = ['app', 'device', 'os', 'channel']
for i in fcols:
    print('This feature "{0}" has {1} different types of {0}.'.format(i, len(df_train[i].unique())))

##features' total clicking  
def ffeq(df, x='feature'):
    index = df[x].value_counts().keys().tolist()
    values = df[x].value_counts().tolist()
    tmp = pd.DataFrame({'id': index, 'n_clicking': values})
    ax = tmp.sort_values(by='id').plot(x='id', y='n_clicking')
    ax.legend(['sum of clicking by app'])
    return ax.show()

ffeq(df_train, x='app')
ffeq(df_train, x='device')
ffeq(df_train, x='os')
ffeq(df_train, x='channel')

##Heat map for clicking rate(day, minute and total # of clicking)
def create_time(df):
    df['click_time'] = pd.to_datetime(df['click_time'])
    df['day'] = df.click_time.dt.day
    df['hour'] = df.click_time.dt.hour
    return df

df_train = create_time(df_train)
df_train['click'] = 1

###Covert to pivot for heat map(covert to unstacked/width data)
tmp_heat = df_train[['day', 'hour', 'click']].groupby(['day', 'hour']).sum() #multilevel-index
tmp_heat = tmp_heat.iloc[:, 0] #to Series to discard the columns name 
pivot_heat = tmp_heat.unstack(level='day', fill_value=0) #unstacked
sns.heatmap(pivot_heat, cmap="YlGnBu") #heatmap

##features' clicking time
###1. # of Clicking based on rows
###1-1.Sort by time
tmp_ct = df_train.sort_values(by='click_time')

###1-2.Create new index by click_time
tmp_ct_index = pd.DatetimeIndex(tmp_ct['click_time'])

###1-3.create new df and resample the tmp
tmp_ct = pd.DataFrame(tmp_ct.click.values, index=tmp_ct_index, columns=['click'])

###1-4. Plot the line graph
plot = tmp_ct.resample('2T').sum().plot(xticks=pd.date_range(start=tmp_ct.index[0], end=tmp_ct.index[-1], freq='H'), rot=45)
plot.legend(['# of click'])
plot.show()

##Correlation between ip, is_attributed and # of clicking
###sort by # click
df_train[['ip', 'is_attributed', 'click']].groupby('ip').sum().sort_values(by='click', ascending=False).head(20)
df_train[['ip', 'is_attributed', 'click']].groupby('ip').sum().sort_values(by='is_attributed', ascending=False).head(10)

###Avg of clicking = 2.8
df_train.shape[0] / len(df_train.ip.unique())


###2. # of Clicking based on feature combination
### feature + hour + day
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


df_train = comb_click(df_train, 'ip') #need to add ip because it did not include in fcols.

###Top 10 clicking in each feature
def top_10c(df, feature='ip'):
    return df[['click_{}_hour'.format(feature), feature]].groupby(feature).sum().sort_values('click_{}_hour'.format(feature), ascending=False).head(10)

for i in fcols:
    print (top_10c(df_train, i))

#Deal with imbalanced class

    
#Model: xgboost in talkingdata_model.py
