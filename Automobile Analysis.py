#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
sns.set(color_codes=True)
import os
print(os.listdir("../workspace"))


# In[2]:


auto = pd.read_csv('../workspace/Automobile_data.csv')


# In[4]:


auto.describe()


# In[14]:


auto.shape


# In[18]:


auto.info()


# In[3]:


auto.head()


# In[30]:


df = auto.replace('?',np.NAN) 
df.isnull().sum()


# In[44]:


df_temp = auto[auto['normalized-losses']!='?']
normalised_mean = df_temp['normalized-losses'].astype(int).mean()
df['normalized-losses'] = auto['normalized-losses'].replace('?',normalised_mean).astype(int)

df_temp = auto[auto['price']!='?']
normalised_mean = df_temp['price'].astype(int).mean()
df['price'] = auto['price'].replace('?',normalised_mean).astype(int)

df_temp = auto[auto['horsepower']!='?']
normalised_mean = df_temp['horsepower'].astype(int).mean()
df['horsepower'] = auto['horsepower'].replace('?',normalised_mean).astype(int)

df_temp = auto[auto['peak-rpm']!='?']
normalised_mean = df_temp['peak-rpm'].astype(int).mean()
df['peak-rpm'] = auto['peak-rpm'].replace('?',normalised_mean).astype(int)

df_temp = auto[auto['bore']!='?']
normalised_mean = df_temp['bore'].astype(float).mean()
df['bore'] = auto['bore'].replace('?',normalised_mean).astype(float)

df_temp = auto[auto['stroke']!='?']
normalised_mean = df_temp['stroke'].astype(float).mean()
df['stroke'] = auto['stroke'].replace('?',normalised_mean).astype(float)


df['num-of-doors'] = auto['num-of-doors'].replace('?','four')
df.info()


# In[9]:


sns.distplot(df['engine-size']);


# In[61]:


df2[['engine-size','curb-weight']].hist(figsize=(8,6),bins=6,color='Y')
plt.tight_layout()
plt.show()


# In[47]:


df[['peak-rpm','horsepower','price']].hist(figsize=(10,8),bins=6,color='y')
plt.tight_layout()
plt.show()


# In[50]:


plt.figure(1)
plt.subplot(221)
df['engine-type'].value_counts(normalize=False).plot(figsize=(10,8),kind='bar',color='red')
plt.title("Engine Type frequency diagram")
plt.ylabel('Number of Engine Type')
plt.xlabel('engine-type');


plt.subplot(222)
df['num-of-doors'].value_counts(normalize=False).plot(figsize=(10,8),kind='bar',color='green')
plt.title("Number of Doors frequency diagram")
plt.ylabel('Number of Doors')
plt.xlabel('num-of-doors');

plt.subplot(223)
df['fuel-type'].value_counts(normalize= False).plot(figsize=(10,8),kind='bar',color='purple')
plt.title("Fuel Type frequency diagram")
plt.ylabel('Number of vehicles')
plt.xlabel('fuel-type');

plt.subplot(224)
df['body-style'].value_counts(normalize=False).plot(figsize=(10,8),kind='bar',color='orange')
plt.title("Body Style frequency diagram")
plt.ylabel('Number of vehicles')
plt.xlabel('body-style');
plt.tight_layout()
plt.show()


# In[10]:


sns.distplot(auto['city-mpg'], kde=False, rug=True);


# In[6]:


sns.jointplot(auto['engine-size'],auto['wheel-base'])


# In[10]:


sns.pairplot(auto[['city-mpg', 'engine-size', 'wheel-base']])


# In[12]:


sns.stripplot(auto['fuel-type'], auto['city-mpg'],jitter=True)


# In[51]:


sns.violinplot(df['city-mpg'])


# In[52]:


sns.violinplot(df['num-of-doors'], df['city-mpg'], hue=df['fuel-type'])


# In[53]:


sns.barplot(df['body-style'], df['city-mpg'], hue=df['engine-location'])


# In[18]:


sns.countplot(auto['body-style'])


# In[54]:


sns.catplot(data=df, x="body-style", y="price", hue="aspiration" ,kind="point")


# In[56]:


plt.rcParams['figure.figsize']=(23,10)
ax = sns.boxplot(x="make", y="price", data=df)


# In[57]:


plt.rcParams['figure.figsize']=(19,7)
ax = sns.boxplot(x="body-style", y="price", data=df)


# In[58]:


plt.rcParams['figure.figsize']=(10,5)
ax = sns.boxplot(x="drive-wheels", y="price", data=df)


# In[59]:


sns.catplot(data=df, y="normalized-losses", x="symboling" , hue="body-style" ,kind="point")


# In[ ]:




