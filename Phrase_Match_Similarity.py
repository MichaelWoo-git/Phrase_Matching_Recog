#!/usr/bin/env python
# coding: utf-8

# ## Todo 04/12/2022
# 
# 1. Need to figure out a way to account for words that have never been seen before
#     * So we need to figure out a way to bin words that are similar to each other
#     * here is what we can do: https://www.analyticsvidhya.com/blog/2020/05/what-is-tokenization-nlp/
#     * https://neptune.ai/blog/wasserstein-distance-and-textual-similarity
#     * Remove stop words (preprocessing)
#     * 

# __Imports__

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
#nltk.download()


# __Read in Data__

# In[2]:


df_1 = pd.read_csv("data/train.csv")
print(df_1.head())


# In[3]:


print(df_1.describe())


# In[4]:


print(df_1.isnull().any())


# __This is the amount of words in each row with respect to the column__
# * Must be applied to anchor and target

# These are the unique amount of words in __anchor__ column

# In[5]:


print(np.unique(np.array(df_1['anchor'].apply(lambda x : len(x.split())))))


# These are the unique amount of words in __target__ column

# In[6]:


print(np.unique(np.array(df_1['target'].apply(lambda x : len(x.split())))))


# We can probably drop __context__ because its a label

# In[7]:


print(np.unique(np.array(df_1['context'].apply(lambda x : len(x.split())))))


# __Tokenization__

# In[8]:


from nltk.tokenize import word_tokenize
df_1['target'] = df_1['target'].apply(lambda x : word_tokenize(x))
df_1['anchor'] = df_1['anchor'].apply(lambda x : word_tokenize(x))


# In[9]:


print(df_1.head())


# __Stopwords Removal__

# In[10]:


from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
df_1['target'] = df_1['target'].apply(lambda x : [w for w in x if not w in stop_words])
df_1['anchor'] = df_1['anchor'].apply(lambda x : [w for w in x if not w in stop_words])


# In[11]:


print(df_1.head())


# __Stemming__

# In[12]:


from nltk.stem.snowball import SnowballStemmer
snowBallStemmer = SnowballStemmer("english")
df_1['target'] = df_1['target'].apply(lambda x : [snowBallStemmer.stem(word) for word in x])
df_1['anchor'] = df_1['anchor'].apply(lambda x: [snowBallStemmer.stem(word) for word in x])


# In[13]:


print(df_1.head())


# __Need to convert array to just strings__

# In[14]:


df_1['anchor'] = df_1['anchor'].apply(lambda x : ','.join(map(str,x)))
df_1['target'] = df_1['target'].apply(lambda x : ','.join(map(str,x)))


# In[15]:


print(df_1.head())


# These the the unique words in the __anchor__ column 10 are shown below

# In[16]:


print(np.unique(np.array(df_1['anchor']))[:10])


# These the the unique words in the __target__ column 10 are shown below

# In[17]:


print(np.unique(np.array(df_1['target']))[:10])


# We have the issue of words going together, we need to seperate these so we can
# we need to seperate the lengths

# In[18]:


anchor_set = set()
target_set = set()


# In[19]:


def separator_anchor(arr):
    for i in arr:
        anchor_set.add(i)


# In[20]:


def separator_target(arr):
    for i in arr:
        target_set.add(i)


# In[21]:


df_1['anchor'] = df_1['anchor'].apply(lambda x: x.split(','))
df_1['target'] = df_1['target'].apply(lambda x: x.split(','))


# In[22]:


df_1['anchor'].apply(lambda x: separator_anchor(x))
df_1['target'].apply(lambda x: separator_target(x))


# __list comprehension__

# In[23]:


anchor_arr = [i for i in anchor_set]
target_arr = [i for i in target_set]


# __Label Encoding__

# In[24]:


labelencoder_anchor = LabelEncoder()
labelencoder_anchor.fit(anchor_arr)
labelencoder_target = LabelEncoder()
labelencoder_target.fit(target_arr)


# In[25]:


def encoding_anchor(arr):
    temp = []
    if len(arr) < 2:
        for i in arr:
            #print(labelencoder_anchor.transform([i]))
            return int(labelencoder_anchor.transform([i]))
    if len(arr) > 1:
        for i in arr:
            #print(labelencoder_anchor.transform([i]))
            temp.extend(labelencoder_anchor.transform([i]))
        return np.array(temp)


# In[26]:


def encoding_target(arr):
    temp = []
    if len(arr) < 2:
        for i in arr:
            #print(labelencoder_anchor.transform([i]))
            return int(labelencoder_target.transform([i]))
    if len(arr) > 1:
        for i in arr:
            #print(labelencoder_anchor.transform([i]))
            temp.extend(labelencoder_target.transform([i]))
        return np.array(temp)


# In[ ]:


print("encoding anchor")


# In[27]:


df_1['anchor'] = df_1['anchor'].apply(lambda x: encoding_anchor(x))


# In[ ]:


print("encoding target")


# In[28]:


df_1['target'] = df_1['target'].apply(lambda x: encoding_target(x))


# In[29]:


print(df_1.head())


# __Spliting (70/30)__
# * Here we test/evaluate our models

# In[37]:


# x = df_1.drop(['id','score'],axis=1).values
x = df_1[['anchor','target']].values[:100]
y = df_1['score'].values[:100]


# In[43]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=141)


# In[44]:


print("Random Forest")


# __Random Forest Regressor__

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(random_state=141)
regr.fit(x_train, y_train)
y_pred = np.round(regr.predict(x_test),decimals=2)
from sklearn.metrics import mean_squared_error, mean_absolute_error
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred,squared=False)
mae = mean_absolute_error(y_test,y_pred)
print("Mean Square Error: {} \nRoot Mean Squared Error: {} \nMean Absolute Error: {} ".format(mse,rmse,mae))


# In[42]:


print("Lasso")


# __Lasso Regression__

# In[ ]:


from sklearn.linear_model import Lasso
lr = Lasso(alpha=0.5)
lr.fit(x_train, y_train)
y_pred = np.round(lr.predict(x_test),decimals=2)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred,squared=False)
mae = mean_absolute_error(y_test,y_pred)
print("Mean Square Error: {} \nRoot Mean Squared Error: {} \nMean Absolute Error: {} ".format(mse,rmse,mae))


# __Prediction Submission__
# * Will be used later for competition
