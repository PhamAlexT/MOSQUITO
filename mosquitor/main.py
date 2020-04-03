# coding: utf-8

'''
Description : 
    Here we test our model by plotting its performances.
    
    run with
    python main.py
'''
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import numpy as np
import pandas as pd

from sys import path
path.append('../scoring_program/')
import warnings

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate,  cross_val_score
from sklearn.metrics import make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,  ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB


from load import load_train, load_valid, load_test
from libscores import get_metric
from model import ModelPreprocessed, ModelRaw
from visual import *



__author__ = "MOSQUITO"



DIRECTORY = './figs/'
DATA_DIR_PRE = '../malaria_input_data'
DATA_DIR_RAW = '../malaria_input_data_raw'
DATA_NAME = 'malaria'



def main():

    #Choose while executing Preprocessed or RAW
    
    print("Here we goooo ! \n Preprocessed (0, default) or Raw (1) ?")
    try:
        choice = int(input())
    except ValueError:
        print("ERREUR: Saisissez un NOMBRE, Fermeture du programme.")
        exit()
    
    warnings.filterwarnings("ignore")
    np.seterr(divide='ignore', invalid='ignore')
    metric_name, scoring_function = get_metric()

    #Choose appropriate directory and  model
    if (choice == 1):
        directory = DIRECTORY + "Raw_Results/"
        data_dir = DATA_DIR_RAW
        clf = ModelRaw()

    else:
        directory = DIRECTORY + "Preprocessed_Results/"
        data_dir = DATA_DIR_PRE
        clf = ModelPreprocessed()

    #Create Directory
    if not os.path.exists(directory):
        os.makedirs(directory)


    #Load data as panda frame
    d_train = load_train(data_dir, DATA_NAME)

    #Transform to numpy
    X = d_train.drop(columns=['target']).to_numpy()
    y = d_train['target'].to_numpy()


    #split data train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


    #Train classif
    clf.fit(X_train, y_train)


    #Score and ROC curve
    accuracy = clf.score(X_test, y_test)
    print("accuracy =", accuracy)
    y_proba = clf.predict_proba(X_test)
    y_decision = y_proba[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_decision, pos_label=1)
    plot_ROC(fpr, tpr, directory=directory)


    #Confusion Matrix
    y_test_pre = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_test_pre)
    plot_mat_conf(cm, directory=directory)



    if (choice!=1):
        #plot score of 5 different model
        model_name = ["Nearest Neighbors","Decision Tree",
                      "Random Forest",  "AdaBoost",
                      "Naive Bayes"]
        model_list = [
            KNeighborsClassifier(3),
            DecisionTreeClassifier(max_depth=10),
            RandomForestClassifier(max_depth=10, n_estimators=20),
            AdaBoostClassifier(),
            GaussianNB(),
            ]

        s_train = []
        s_test = []
        for i in range(len(model_list)):
            s_prime = cross_validate(ModelPreprocessed(classifier = model_list[i]), X, y, cv=3, scoring=make_scorer(scoring_function), return_train_score=True)
            s_train.append(s_prime['train_score'].mean())
            s_test.append(s_prime['test_score'].mean())

        plot_test_model (s_train, s_test, model_name, directory=directory)


        #plot score with different values of n_estimators
        s = []
        var = []
        n_est = np.linspace(15,120, num=4).astype(int)
        metric_name, scoring_function = get_metric()

        for i in range(len(n_est)):
            clf_prime = RandomForestClassifier(random_state = 42, n_estimators = n_est[i])
            scores = cross_val_score(ModelPreprocessed(classifier = clf_prime), X, y, cv=5, scoring=make_scorer(scoring_function))
            s.append(scores.mean())
            var.append(scores.std())
        plot_test_estimator(n_est, s, var, directory=directory)

        #plot decision surface for Decision tree, RandForest and Adaboost
        plot_decision_surface_tree_classif(X_train, y_train, directory=directory)


if __name__ == '__main__':
    main()
