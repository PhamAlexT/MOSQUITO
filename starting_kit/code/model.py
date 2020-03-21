'''
Sample predictive model.
You must supply at least 4 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
- save: saves the model.
- load: reloads the model.
'''

import pickle
import numpy as np
from os.path import isfile
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
# Preprocessing de la bibliothèque
from prePro import prepro
# Preprocessing de la bib scikit learn
from sklearn.preprocessing import StandardScaler


class model (BaseEstimator):
   def __init__(self, classifier = RandomForestClassifier(random_state=42, n_estimators = 100, max_depth=100)):
   
        '''
        Constructeur de notre classe "model"
        param : 
        classifier = Un modèle de classification (Par défault : RandomForest)
        '''
        # Notre modèle
        self.classifier = classifier
        # Preprocessing de la Team prepro
        self.preprocessing1 = prepro()
        # Preprocessing de la bibliothèque Scikit Learn
        self.preprocessing2 = StandardScaler()
        
        
   def fit(self, X, y, sample_weights=None):
        """
        Preprocess the training set and build a forest of trees from it
        params:
        X : training dataset
        y : Labels of each data on the dataset
        return : 
        Our model 'Trained'
        """
        X = self.preprocessing1.fit_transform(X,y)
        X = self.preprocessing2.fit_transform(X,y)
        self.classifier.fit(X, y)
        return self
    
    
   def predict_proba(self, X):
        """
        Predict class probabilities 
        param :
        X : The input dataset
        return :
        The class probabilities of the input samples
        """    
        X = self.preprocessing1.transform(X) 
        X = self.preprocessing2.transform(X)
        y_proba = self.classifier.predict_proba(X)
        return y_proba
    
   def predict(self, X):
        """
        Predict the class of a given dataset
        param :
        X : The dataset
        return
        The predicted classes
        """
        y_proba = self.predict_proba(X)
        y_pred = np.argmax(y_proba, axis=1)
        return y_pred

   def save(self, path="./"):
      pickle.dump(self, open(path + '_model.pickle', "wb"))

   def load(self, path="./"):
      modelfile = path + '_model.pickle'
      if isfile(modelfile):
         with open(modelfile, 'rb') as f:
            self = pickle.load(f)
         print("Model reloaded from: " + modelfile)
      return self