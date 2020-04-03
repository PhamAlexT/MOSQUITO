# coding: utf-8

''' 
Description : 

    Make Preprocessing of our data (Preprocessed/Raw):
    - fit: fit our data to preprocessing
    - transform: return data preprocessed.
     
'''
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


from fonctions import  extract_features



__author__ = "MOSQUITO"



class PreproPreprocessed(BaseEstimator, TransformerMixin):
    def __init__(self,  n_select=12, n_pca=2):
        '''
        Best preprocessing for classification in the preprocessed challenge
         args:
             n_select = number used for SelectionKbest
             n_pca = number used for pca
        '''
        self.selector = SelectKBest(f_classif, k=n_select)
        self.pca = PCA(n_components=n_pca)


    def fit(self, X, y=None):
        '''
        fit preprocessing with X data.
        Args:
            X: data matrix of dim num_train_samples * num_feat.
            y: label matrix of dim num_train_samples * num_labels.
        Both inputs are numpy arrays.
        '''
        self.selector.fit(X, y)
        self.pca.fit(X)
        return self

    def transform(self, X):
        '''
        transform X data matrix.
        Args:
            X: data matrix of dim num_train_samples * num_feat.
        numpy arrays.
        '''
        X_1 = self.selector.transform(X)
        X_2 = self.pca.transform(X)
        X_new = np.concatenate((X_1, X_2), axis=1)
        return X_new


class PreproRaw(BaseEstimator, TransformerMixin):
    def __init__(self):
        '''
        Best preprocessing for classification in the preprocessed challenge
         args:None
        '''
        self.StSc = StandardScaler()

    def fit(self, X, y=None):
        '''
        fit preprocessing with X data.
        Args:
            X: data matrix of dim num_train_samples * num_feat.
            y: label matrix of dim num_train_samples * num_labels.
        Both inputs are numpy arrays.
        '''

        X_new = extract_features(X, 30)
        self.StSc.fit(X_new, y)

        return self

    def transform(self, X):
        '''
        transform X data matrix.
        Args:
            X: data matrix of dim num_train_samples * num_feat.
        numpy arrays.
        '''
        X_new = extract_features(X, 30)
        self.StSc.transform(X_new)
        
        return X_new
