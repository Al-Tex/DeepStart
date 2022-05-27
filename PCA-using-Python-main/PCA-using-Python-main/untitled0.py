# -*- coding: utf-8 -*-
"""
Created on Mon May 23 22:44:02 2022

@author: Alex, Sutac Victor
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import PySimpleGUI as sg


    
companies = pd.read_csv('1000_Companies2.csv')
companie = pd.read_csv('1000_Companies1.csv')
companies.head()
companie.head()
companies[['R&D Spend','Administration','Marketing Spend']].corr()
companie[['R&D Spend','Administration','Marketing Spend']].corr()
companies.drop(['Marketing Spend'],axis=1,inplace=True)
companie.drop(['Marketing Spend'],axis=1,inplace=True)
dummies=pd.get_dummies(companies.State)
dummie = pd.get_dummies(companie.State)
companies=pd.concat([companies,dummies],axis=1)
companie=pd.concat([companie,dummie],axis=1)
companies.drop(['State'],axis=1,inplace=True)
companie.drop(['State'],axis=1,inplace=True)
from sklearn.preprocessing import scale
norm_data=scale(companies.iloc[:,1:])
norm_data

from sklearn.decomposition import PCA
pca=PCA()
pca_values=pca.fit_transform(norm_data)
pca_values.shape
#amount of variance of each PCA
var=pca.explained_variance_ratio_
var
#cumulative variance
cum_var=np.cumsum(np.round(var,decimals=4)*100)
cum_var
#variance plot for PCA components
plt.plot(cum_var,'r')
#plot between PCA1 and PCA2
x=pca_values[:,0]
y=pca_values[:,1]
plt.plot(x,y,'ro');plt.xlabel('time');plt.ylabel('profit')
# no where pca1 and pca2 are correlated
#plt.plot(np,x,"ro")