#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 21:18:19 2018

@author: changyueh
"""
import pandas as pd

##convert time variables
def create_time(df):
    df['click_time'] = pd.to_datetime(df['click_time'])
    df['day'] = df.click_time.dt.day
    df['hour'] = df.click_time.dt.hour
    return df

##create # of click by time
def comb_click_time(df, features=[], groups=[]):
    for i in features:
        if 'clickt_{}'.format(i) in df.columns:
            print('You had already created the new feaure clickt_{}.'.format(i))
        
        else:        
            df['click'] = 1
            tmp_f = []
            tmp_g = []
            for j in groups:
                tmp_g.append(j)
                tmp_f.append(j)
            tmp_g.append(i)
            tmp_f.append(i)    
            tmp_f.append('click')
            tmp = df[tmp_f].groupby(tmp_g).sum()
            tmp.reset_index(inplace=True)
            tmp = tmp.rename(columns={'click': 'clickt_{}'.format(i)})
            df = pd.merge(df, tmp, on=tmp_g)
            print('New features clickt_{} had been created.'.format(i))
            
    return df

##create # of click by features
def comb_click_feature(df, features=[], groups=[]):
    for i in features:
        if 'clickf_{}'.format(i) in df.columns:
            print('You had already created the new feaure clickf_{}.'.format(i))
        
        else:        
            df['click'] = 1
            tmp_f = []
            tmp_g = []
            for j in groups:
                tmp_g.append(j)
                tmp_f.append(j)    
            tmp_g.append(i)
            tmp_f.append(i)
            tmp_f.append('click')
            tmp = df[tmp_f].groupby(tmp_g).sum()
            tmp.reset_index(inplace=True)
            tmp = tmp.rename(columns={'click': 'clickf_{}'.format(i)})
            df = pd.merge(df, tmp, on=tmp_g)
            print('New features clickf_{} had been created.'.format(i))
            
    return df


##drop features which are unnecessary for model
def drop_features(df, features=[]):
    return df.drop(features, axis=1)
