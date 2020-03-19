from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import numpy as np

'''
def supprOutlier(X,y,retIndex=False):
        clf = IsolationForest(random_state=0).fit(X)
        res = clf.predict(X)

        indexOutlier = []
        for i in range (0,len(res)):
            if res[i]==-1:
                indexOutlier.append(i)

        pourcentOutlier = len(indexOutlier) / len(res) * 100
        print("Pourcentage d'outlier: {0:.3f}%:".format(pourcentOutlier))

        if retIndex:
            return np.delete(X,indexOutlier,axis=0),np.delete(y,indexOutlier,axis=0),indexOutelier

        #delete: Supprime les LIGNES d'indices dans indexOutlier
        return np.delete(X,indexOutlier,axis=0),np.delete(y,indexOutlier,axis=0)
'''    
class prepro (BaseEstimator, TransformerMixin):
    
    
   def __init__(self, n_select=12, n_pca=2):    
    
        self.selector = SelectKBest(f_classif, k=n_select)
        self.pca = PCA(n_components=n_pca)
        
   def fit(self, X, y=None):

        
       # X,y  = supprOutlier(X,y)  
        self.selector.fit(X, y)
        self.pca.fit(X)

        
        return self
    
    
   def transform(self, X):
        
       # X = supprOutlier(X) 
        X_1 = self.selector.transform(X)
        X_2 = self.pca.transform(X)
        X_new = np.concatenate((X_1, X_2), axis=1)

        return X_new
    
   
