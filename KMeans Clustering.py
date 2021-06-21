#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


df=pd.read_csv('5factors without date.csv')


# In[13]:


df


# In[14]:


df[['TemperatureMAX','TemperatureAVG','TemperatureMIN']]


# In[20]:


plt.scatter(df['TemperatureAVG'],df['COVIDCASES'], color='blue', marker='.')
plt.xlabel('Average Temperature (C)')
plt.ylabel('COVID-19 cases')


# In[21]:


km= KMeans(n_clusters=2)
km


# In[24]:


y_predicted= km.fit_predict(df[['TemperatureAVG']])
y_predicted


# In[35]:


df['cluster']= y_predicted
df.head()


# In[37]:


df1= df[df.cluster==0]
df2=df[df.cluster==1]

plt.scatter(df1.TemperatureAVG, df1['TemperatureAVG'],color='green')
plt.scatter(df2.TemperatureAVG, df2['TemperatureAVG'], color='black')

