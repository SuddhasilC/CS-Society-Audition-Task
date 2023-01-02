#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression


# In[2]:


data=pd.read_csv('Life Expectancy Data.csv')


# In[3]:


data['bmi']=data['Weight']*10000/(data['Height']**2)


# In[4]:


data.drop(['Weight','Height'],inplace=True,axis=1)


# In[5]:


cat_cols=['Gender','Lifestyle']
for cat in cat_cols:
    enc = OneHotEncoder(sparse=False)
    color_onehot = enc.fit_transform(data[[cat]])
    x=pd.DataFrame(color_onehot, columns=list(enc.categories_[0]))
    for i in x.columns:
        data[i]=x[i]
    del data[cat]


# In[6]:


x=data.drop(columns=['LE'])
y=data['LE']


# In[7]:


regressor = LinearRegression()
regressor.fit(x.values,y)


# In[8]:


gender=input("Enter your gender (M/F): ")
age=int(input("Enter your age: "))
height=int(input("Enter your height (in cm): "))
weight=int(input("Enter your weight (in kg): "))
income=int(input("Enter your income (in lpa): "))
lifestyle=input("Are you Active/Moderately Active/Inactive?: ")
alcohol=int(input("Do you consume alcohol? (1:Yes 0:No)"))
smoking=int(input("Do you smoke? (1:Yes 0:No)"))


# In[9]:


bmi= (weight * 10000) / (height**2)


# In[10]:


x=[]
x.append(age)
x.append(income)
x.append(alcohol)
x.append(smoking)
x.append(bmi)
if gender == 'F':
    x.append(1)
    x.append(0)
else:
    x.append(0)
    x.append(1)
if lifestyle == 'Active':
    x.append(1)
    x.append(0)
    x.append(0)
elif lifestyle == 'Moderately Active':
    x.append(0)
    x.append(1)
    x.append(0)
else:
    x.append(0)
    x.append(0)
    x.append(1)


# In[14]:


x=np.array(x).reshape(1,-1)


# In[15]:


print("Your expected life expectancy is: "+ str(int(regressor.predict(x))))

