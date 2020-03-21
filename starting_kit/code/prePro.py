from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import numpy as np



class prepro (BaseEstimator, TransformerMixin):
    
    
   def __init__(self, n_select=12, n_pca=2):    
        
        #selectionne k meilleures features
        self.selector = SelectKBest(f_classif, k=n_select)
        #reduction de dimention à n 
        self.pca = PCA(n_components=n_pca)
        
   def fit(self, X, y=None):

        self.selector.fit(X, y)
        self.pca.fit(X)       
        return self
    
    
   def transform(self, X):
        
        X_1 = self.selector.transform(X)
        X_2 = self.pca.transform(X)
        #On regroupe le tout
        X_new = np.concatenate((X_1, X_2), axis=1)

        return X_new




   
