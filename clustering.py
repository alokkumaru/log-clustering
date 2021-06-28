#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from cleanmsg import clean_message



# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

vectorizer = CountVectorizer()

def unsupervised_clustering_info(log, number_of_clusters):
    
    log['only_words']=log['message'].apply(lambda x : clean_message(x))
    model = KMeans(n_clusters=number_of_clusters, init='k-means++', max_iter=100, n_init=1) #creating K-Means model
    
    X = vectorizer.fit(log['only_words'])
    v = pd.DataFrame( X.transform(log["only_words"]).toarray(),columns=X.get_feature_names() )
    
    model.fit(v) #fitting model
    log['cluster']=(model.predict(v)) #predict cluster
    
    pickle.dump(model, open("model_info.pkl", "wb")) #save model
    pickle.dump(v.columns, open("cols_info.pkl", "wb")) #save cols
    
    return log


def unsupervised_clustering_err(log, number_of_clusters):

    log['only_words']=log['message'].apply(lambda x : clean_message(x))
    model = KMeans(n_clusters=number_of_clusters, init='k-means++', max_iter=100, n_init=1) #creating K-Means model
    
    X = vectorizer.fit(log['only_words'])
    v = pd.DataFrame( X.transform(log["only_words"]).toarray(),columns=X.get_feature_names() )
    
    model.fit(v) #fitting model
    log['cluster']=(model.predict(v)) #predict cluster
    
    pickle.dump(model, open("model_err.pkl", "wb")) #save model
    pickle.dump(v.columns, open("cols_err.pkl", "wb")) #save cols
    
    return log   


def supervised_clustering_info(log):

    log['only_words']=log['message'].apply(lambda x : clean_message(x))
    Y = vectorizer.fit(log['only_words'])
    v=pd.DataFrame( Y.transform(log["only_words"]).toarray(), columns=Y.get_feature_names())
    
    cols = pickle.load(open("cols_info.pkl", "rb")) #load cols
    model = pickle.load(open("model_info.pkl", "rb")) #load model
        
    v = v.reindex(labels=cols,axis=1)
    v = v.fillna(0)
    
    log['cluster']=(model.predict(v))
    t=(model.transform(v))
    d=[]
    for i in range(len(t)):
        d.append(min(t[i]))
    log['cluster_distance']=d
    return log  


def supervised_clustering_err(log):

    log['only_words']=log['message'].apply(lambda x : clean_message(x))
    Y = vectorizer.fit(log['only_words'])
    v=pd.DataFrame( Y.transform(log["only_words"]).toarray(), columns=Y.get_feature_names())
       
    cols = pickle.load(open("cols_err.pkl", "rb")) #load cols
    model = pickle.load(open("model_err.pkl", "rb")) #load model
        
    v = v.reindex(labels=cols,axis=1)
    v = v.fillna(0)
    
    log['cluster']=(model.predict(v))
    t=(model.transform(v))
    d=[]
    for i in range(len(t)):
        d.append(min(t[i]))
    log['cluster_distance']=d
    return log  

