from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import pickle
import numpy as np
from os.path import isfile

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA


class PreproRaw(BaseEstimator, TransformerMixin):
    def __init__(self, n_compo):
        """
        Best preprocessing for classification in the preprocessed challenge
         args:None
        """
        self.pca = PCA(n_components=n_compo)

    def fit(self, X, y=None):
        """
        fit preprocessing with X data.
        Args:
            X: data matrix of dim num_train_samples * num_feat.
            y: label matrix of dim num_train_samples * num_labels.
        Both inputs are numpy arrays.
        """

        self.pca.fit(X, y)
        return self

    def transform(self, X):
        """
        transform X data matrix.
        Args:
            X: data matrix of dim num_train_samples * num_feat.
        numpy arrays.
        """
        X_new = self.pca.transform(X)
        return X_new


class ModelRaw(BaseEstimator, ClassifierMixin):
    def __init__(self, classifier=RandomForestClassifier(random_state=42, n_estimators=10000, max_depth=40),n=150):
        '''
        Best model for classification in the raw challenge
         args:
             Classifier: classifier used in our model
        '''
        self.prepro = PreproRaw(n_compo=n)
        self.classifier = classifier

    def fit(self, X, y):
        '''
        Training the model.
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
            y: Training label matrix of dim num_train_samples * num_labels.
        Both inputs are numpy arrays.
        '''
        X = self.prepro.fit_transform(X, y)
        self.classifier.fit(X, y)
        return self

    def predict_proba(self, X):
        '''
        Compute probabilities to belong to given classes.
        '''
        X = self.prepro.transform(X)
        y_proba = self.classifier.predict_proba(X)
        return y_proba

    def predict(self, X):
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
