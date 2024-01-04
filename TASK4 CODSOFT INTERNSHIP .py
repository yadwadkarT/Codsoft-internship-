#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv(r"D:\codsoft\advertising.csv")


# In[3]:


df.head()


# In[4]:


df.head(10)


# In[5]:


df.shape


# In[6]:


df.describe()


# In[7]:


sns.pairplot(df, x_vars=['TV', 'Radio','Newspaper'], y_vars='Sales', kind='scatter')
plt.show()


# In[8]:


df['TV'].plot.hist(bins=10)


# In[9]:


df['Radio'].plot.hist(bins=10, color="green", xlabel="Radio")


# In[10]:


df['Newspaper'].plot.hist(bins=10,color="purple", xlabel="newspaper")


# In[11]:


sns.heatmap(df.corr(),annot = True)
plt.show()


# In[12]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[['TV']], df[['Sales']], test_size = 0.3,random_state=0)
     
print(X_train)
     


# In[13]:


print(y_train)


# In[14]:


print(X_test)


# In[15]:


print(y_test)


# In[16]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
     


# In[17]:


res= model.predict(X_test)
print(res)


# In[ ]:




