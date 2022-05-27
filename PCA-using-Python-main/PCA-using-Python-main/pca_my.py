#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

from sklearn.preprocessing import MinMaxScaler
scale=MinMaxScaler()
companies[['R&D Spend','Administration']]=scale.fit_transform(companies[['R&D Spend','Administration']])
companie[['R&D Spend','Administration']]=scale.fit_transform(companie[['R&D Spend','Administration']])
y=companies.iloc[:,2].values 
y_T=companie.iloc[:,2].values
companies.drop(['Profit'],axis=1,inplace=True)
companie.drop(['Profit'],axis=1,inplace=True)
X=companies.iloc[:,:].values
X_T=companies.iloc[:,:].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X,y)

#X_test=X_T
#y_test=y_T
y_pred = model.predict(X_test)
c=0
for i in y_pred:
    c+=i
avg = c /len(y_pred)
print(avg)
#print(y_pred)
# coefficient 
print(model.coef_)
# intercepts
print(model.intercept_)
# calculating the  R squared error
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))

plt.plot(X,model.predict(X),color='k');plt.xlabel('x');plt.ylabel('y')