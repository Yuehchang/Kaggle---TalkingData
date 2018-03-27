#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 17:26:59 2018

@author: changyueh
"""

import pandas as pd

##test score xgboost: 0.9655
##test score lightgbm: 0.9657
##blend from gopisaran(one voter in kaggle): 0.9696

print('Start the blending process...')

##read submitted file
xgb = pd.read_csv('submit_xgboost.csv')
lgbm = pd.read_csv('submit_lightgbm.csv')
#gop = pd.read_csv('submit_fromgop.csv')

##prepare some list to use
files = [xgb, 
         lgbm, 
         #gop
         ]

click_id = list(files[0].click_id.values)

##calculate the weight
wights = [0.5,
          0.5,  
          #0.5
          ]

##apply weight to each files
for w, df in zip(wights, files):
    df.is_attributed = df.is_attributed * w

del files, wights
    
##create new submit file
blend = pd.DataFrame({'click_id': click_id,
                      'xgb_score': list(xgb.is_attributed.values),
                      'lgbm_score': list(lgbm.is_attributed.values),
                      #'gop_score': list(gop.is_attributed.values)
                      })

del xgb    
del lgbm
#del gop

blend['is_attributed'] = blend.lgbm_score.values + blend.xgb_score.values # blend.gop_score.values
                        

##drop _score columns
blend = blend.drop(['xgb_score',
                    'lgbm_score', 
                    #'gop_score'
                    ], axis=1)

##save to csv
blend.to_csv('submit_blend.csv', index=False)
print('All tasks are completed...')



    
