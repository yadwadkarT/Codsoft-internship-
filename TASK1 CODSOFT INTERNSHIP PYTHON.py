#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv(r"D:\codsoft\tested.csv")


# In[3]:


df.head()


# In[4]:


df.head(10)


# In[5]:


df.shape()


# In[6]:


df.shape


# In[7]:


df.describe()


# In[8]:


df['Survived'].value_counts()


# In[9]:


sns.countplot(x=df['Survived'], hue=df['Pclass'])


# In[10]:


df["Sex"]


# In[11]:


sns.countplot(x=df['Sex'], hue=df['Survived'])


# In[12]:


df.groupby('Sex')[['Survived']].mean()


# In[13]:


df['Sex'].unique()


# In[14]:


rom sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

df['Sex']= labelencoder.fit_transform(df['Sex'])

df.head()


# In[15]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

df['Sex']= labelencoder.fit_transform(df['Sex'])

df.head()


# In[16]:


sns.countplot(x=df['Sex'], hue=df["Survived"])


# In[17]:


df.isna().sum()


# In[18]:


df=df.drop(['Age'], axis=1)


# In[19]:


df_final = df
df_final.head(10)


# In[20]:


import warnings
warnings.filterwarnings("ignore")

res= log.predict([[2,1]])

if(res==0):
  print("So Sorry! Not Survived")
else:
  print("Survived")


# In[21]:


X= df[['Pclass', 'Sex']]
Y=df['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LogisticRegression

log = LogisticRegression(random_state = 0)
log.fit(X_train, Y_train)


# In[22]:


import warnings
warnings.filterwarnings("ignore")

res= log.predict([[2,1]])

if(res==0):
  print("So Sorry! Not Survived")
else:
  print("Survived")


# In[ ]:




