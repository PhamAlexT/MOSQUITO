# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from model import model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from libscores import get_metric
from sklearn.ensemble import AdaBoostClassifier


def f_test_estimator(X_train, Y_train): 
    s = []
    var =[]
    nb_arbre = np.linspace(15,120, num=6).astype(int)     #Nombre d'arbres que l'on teste
    metric_name, scoring_function = get_metric()
    
    for i in range(len(nb_arbre)):
        clf = RandomForestClassifier(random_state = 42, n_estimators = nb_arbre[i])
        M_prime = model(clf)
        scores = cross_val_score(M_prime, X_train, Y_train, cv=5, scoring=make_scorer(scoring_function))
        s.append(scores.mean()) 
        var.append(scores.std())
        

    plt.xlabel("n_estimators")
    plt.ylabel('Score')
    plt.title('Score results of RandomForest with cross-validation')
    plt.errorbar(nb_arbre, s, var, label='Test set')
 

def f_K(X_train, Y_train):     
    metric_name, scoring_function = get_metric()

    #model
    M = model()  
    
    s_prime = []
    K = np.linspace(11,14, num=3).astype(int)       
    
    for i in range(len(K)):
        selector = SelectKBest(f_classif, k=i).fit(X_train, Y_train)
        X_1 = selector.transform(X_train)

        X_train_nouveau = np.concatenate((X_1, X_2), axis=1)
        
        scores = cross_val_score(M, X_train_nouveau, Y_train, cv=5, scoring=make_scorer(scoring_function))
        s_prime.append(scores.mean())    #On prend la moyenne des s core obtenus parmi toutes les partitions pour chaque valeur de nb d'arbre

    plt.plot( K, s_prime, '--b')
    plt.xlabel("max_depth")
    plt.ylabel('Score')
    plt.title('Score results of RandomForest with cross-validation')
    
