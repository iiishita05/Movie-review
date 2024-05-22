#!/usr/bin/env python
# coding: utf-8

# In[74]:


import numpy as np
import pandas as pd
df=pd.read_csv("../TextFiles/moviereviews.tsv",sep='\t')


df.dropna(inplace=True)
blanks=[]
#(index,label,review text)
for i,lb,rv in df.itertuples():
    if rv.isspace():
        blanks.append(i)
df.drop(blanks,inplace=True)


from sklearn.model_selection import train_test_split
X=df['review']
y=df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[63]:


text_clf=Pipeline([('tfidf',TfidfVectorizer()),('clf',LinearSVC())])
text_clf.fit(X_train,y_train)


# In[65]:


predictions=text_clf.predict(X_test)
print (confusion_matrix(y_test,predictions))


# In[66]:


print(classification_report(y_test,predictions))


# In[67]:


print(accuracy_score(y_test,predictions))


# In[72]:


myreview = "A movie I really wanted to love was terrible. \
I'm sure the producers had the best intentions, but the execution was lacking."
print(text_clf.predict([myreview]))

