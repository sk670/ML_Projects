#!/usr/bin/env python
# coding: utf-8

# In[142]:


import numpy as np 
import pandas as pd


# In[143]:


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# In[144]:


train.head(5)


# In[145]:


Y_train = train.iloc[:,-1] 
Y_train


# In[146]:


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
from nltk.stem import WordNetLemmatizer


# In[147]:


wl = WordNetLemmatizer()


# In[148]:


corpus=[]
text=[]
for comment in train.loc[:,'text']:
    comment = re.sub(r'(.)1+', r'1', comment)
    comment = re.sub('((http:\.+)|(www.[^s]+))','',comment) 
    comment=re.sub('[^a-zA-Z]',' ',comment)
    comment=comment.lower()
    comment=comment.split()
    text=text+comment
    comment=[wnl.lemmatize(word) for word in comment if word not in set(stopwords.words('english'))]
    comment=' '.join(comment)
    corpus.append(comment)


# In[149]:


corpus[0]


# In[150]:


from nltk.probability import FreqDist
fd = FreqDist()
for word in text:
    fd[word]+=1 


# In[151]:


fd


# In[152]:


fd_values = list(fd.values())
fd_values.sort(reverse=True)


# In[153]:


len(fd_values)


# In[154]:


s=0
s1=sum(fqd_values)
iteration =1 
l=[]
for i in range(0,20000,200): 
    s=s+sum(fqd_values[i:i+200])
    print(f'for top {i+200} words {(s/s1)*100}% of data')
    l.append((s/s1)*100) 
    iteration+=1 


# In[155]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=4000)
train = tfidf.fit_transform(corpus).toarray()


# In[156]:


train.shape


# In[158]:


tweets1 = []
for st in test.loc[:,'text']:  
    st = re.sub(r'(.)1+', r'1', st)
    st = re.sub('((http:\.+)|(www.[^s]+))','',st) 
    st = re.sub('[^a-zA-Z]',' ',st) 
    st = st.split(' ')
    st = [wl.lemmatize(i) for i in st if i not in set(stopwords.words('english'))] 
    st = ' '.join(st) 
    tweets1.append(st)  
tweets1[0:5] 


# In[159]:


test = tfidf.transform(tweets1) 
test = test.toarray()


# In[160]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=2000)


# In[162]:


from sklearn.model_selection import cross_val_score
cross_val_score(lr,train,Y_train).mean()


# In[168]:


lr.fit(train,Y_train)
Y_prediction = lr.predict(test)  


# In[169]:


result = pd.read_csv("result.csv")


# In[170]:


result['target']=Y_prediction


# In[171]:


result.head(5)


# In[ ]:




