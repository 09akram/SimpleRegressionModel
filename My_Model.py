# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 19:23:51 2019

@author: akramilm
"""

import numpy as np
import pandas as pd
import pickle
from word2number import w2n
import matplotlib.pyplot as plt

df = pd.read_csv('hiring.csv')
df['experience'].fillna(0,inplace=True)
df['test_score(out of 10)'].fillna(df['test_score(out of 10)'].mean(),inplace=True)

x = df.iloc[:,0:3] 

def convert_to_int(word):
    word_dic = {'Zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7
                ,'eight':8,'nine':9,'ten':10,'eleven':11,0:0}
    return word_dic[word]
x['experience'] = x['experience'].apply(lambda x: convert_to_int(x))


y = df.iloc[:,-1]

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x,y)

#saving modelto disk
pickle.dump(model,open('myModel.pkl','wb'))
# loading model to compare
my_model = pickle.load(open('myModel','rb'))


