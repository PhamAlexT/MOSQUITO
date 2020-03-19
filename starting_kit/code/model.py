'''
Sample predictive model.
You must supply at least 4 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
- save: saves the model.
- load: reloads the model.
'''

import pickle
import numpy as np   # We recommend to use numpy arrays
from os.path import isfile
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA

from prePro import prepro

'''
Sample predictive model.
You must supply at least 4 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
- save: saves the model.
- load: reloads the model.
'''

import pickle
import numpy as np   # We recommend to use numpy arrays
from os.path import isfile
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier

class model (BaseEstimator):
   def __init__(self, classifier = RandomForestClassifier(random_state=42, n_estimators = 120)):
   
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation. 
         args :
            classifier : classifier we will use for making our predictions
        '''
        self.num_train_samples=0
        self.num_feat=1
        self.num_labels=1
        self.is_trained=False
        self.classifier = classifier
        
        #ici
        self.preprocessing = prepro()
        self.X_new = None
        
        
   def fit(self, X, y, sample_weights=None):
        X = self.preprocessing.fit_transform(X,y) 

        self.classifier.fit(X, y)
        return self
    
    
   def predict_proba(self, X):
        X = self.preprocessing.transform(X) 
        y_proba = self.classifier.predict_proba(X)
        return y_proba
    
   def predict(self, X):
        y_proba = self.predict_proba(X)
        y_pred = np.argmax(y_proba, axis=1)
        return y_pred