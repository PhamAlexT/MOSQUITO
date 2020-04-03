# coding: utf-8

'''
Description : 
    Functions to plot most important visualizations.
'''
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import os

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

from sklearn.metrics import auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier



__author__ = "MOSQUITO"

def set_plot_config():
    sns.set()
    sns.set_style("whitegrid")
    sns.set_context("poster")

    mpl.rcParams['figure.figsize'] = [8.0, 6.0]
    mpl.rcParams['figure.dpi'] = 80
    mpl.rcParams['savefig.dpi'] = 100

    mpl.rcParams['font.size'] = 10
    mpl.rcParams['axes.labelsize'] = 10
    mpl.rcParams['axes.titlesize'] = 17
    mpl.rcParams['ytick.labelsize'] = 10
    mpl.rcParams['xtick.labelsize'] = 10
    mpl.rcParams['legend.fontsize'] = 'large'
    mpl.rcParams['figure.titlesize'] = 'medium'


def plot_ROC(fpr, tpr, directory=None, title="ROC curve"):
    '''
    Plot area under the curve ROC
     args:
         fpr(numpy array): label predicted
         tpr(numpy array): true labels
    '''
    try:
        plt.plot(fpr, tpr, label='ROC {}'.format(auc(fpr, tpr)))
        plt.title(title)
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.legend()
        if directory is not None:
            plt.savefig(os.path.join(directory, 'roc.png'))
    except Exception as e:
        print('Plot ROC failed')
        print(e)
    finally:
        plt.clf()



def plot_mat_conf(cm, directory=None, title="Confusion Matrix"):
    '''
    Plot confusion matrix
     args:
         cm(numpy array): confusion matrix
    '''
    try:
        f, ax = plt.subplots(figsize=(8, 6))
        ax = sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap="YlGnBu")


        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels(['Parasitized', 'Uninfected'])
        ax.yaxis.set_ticklabels(['Parasitized', 'Uninfected'])
        plt.title(title)
        plt.legend()
        if directory is not None:
            plt.savefig(os.path.join(directory, 'CM.png'),bbox_inches = 'tight')
    except Exception as e:
        print('Plot Confusion Matrix failed')
        print(e)
    finally:
        plt.clf()



def plot_test_model (s_train, s_test, model_name, directory=None, title="Score different Model"):
    '''
    Plot bar scores of different models
     args:
         s_train(list): train scores of each model
         s_test(list): test scores of each model
    '''
    try:
        f, ax = plt.subplots(figsize=(6, 4))
        d = {'Score_train': s_train,
             'Score_test': s_test}

        #Plot
        sd = pd.DataFrame(d, index=[model_name[i] for i in range(len(model_name))] )
        ax = sd.plot.bar()
        ax.set_ylabel("Score")
        ax.set_xlabel("Model")
        plt.title(title)
        plt.legend()
        if directory is not None:
            plt.savefig(os.path.join(directory, 'ModelComparison.png'),bbox_inches = 'tight')
    except Exception as e:
        print('Plot test model failed')
        print(e)
    finally:
        plt.clf()



def plot_test_estimator(n_est, s, var, directory=None, title='Score results of RandomForest with cross-validation'):
    '''
    Plot function scores depending on n_estimators used in random forest
     args:
         n_est(list): number of estimators used
         s(list):  scores of each number of estimators tested
    '''

    try:
        plt.xlabel("n_estimator")
        plt.ylabel('Score')
        plt.title(title)
        plt.errorbar(n_est, s, var, label='Test set')
        plt.legend()
        if directory is not None:
            plt.savefig(os.path.join(directory, 'function_n_estimator.png'))
    except Exception as e:
        print('Plot test estimator failed')
        print(e)
    finally:
        plt.clf()


        
'''! Code below was taken from a website and modified and adapted for our problem!'''
def plot_decision_surface_tree_classif (X_train,Y_train, directory=None, title="Classifiers on feature subsets of the Medichal dataset"):
    '''
    Plot decision surface for 3 models : adaboost, randomforest and decision tree
     args:
         X_train(numpy array): data matrix
         Y_train(numpy array): label matrix
    '''
    try:
        # Parameters
        n_classes = 2
        cmap = plt.cm.RdYlBu
        plot_step = 0.02
        plot_step_coarser = 0.5
        plot_idx = 1

        model_name = ["Decision Tree", "Random Forest",  "AdaBoost"]
        models = [
            DecisionTreeClassifier(),
            RandomForestClassifier(max_depth=100, n_estimators=100),
            AdaBoostClassifier()
            ]
        i = 0

        #We take 3 features
        for pair in ([7, 5], [7, 8], [7, 4]):
            for m_name in models:

                # we test each pair
                X = X_train[:, pair]
                y = Y_train

                mean = X.mean(axis=0)
                std = X.std(axis=0)
                X = (X - mean) / std

                # Train
                M = m_name
                M.fit(X, y)
                scores = M.score(X, y)

                plt.subplot(3, 3, plot_idx)
                if plot_idx <= len(models):
                    plt.title(model_name[plot_idx-1], fontsize=9)

                #plot the decision boundary
                x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
                y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))

                if isinstance(M, DecisionTreeClassifier):
                    Z = M.predict(np.c_[xx.ravel(), yy.ravel()])
                    Z = Z.reshape(xx.shape)
                    cs = plt.contourf(xx, yy, Z, cmap=cmap)
                else:
                    # Choose alpha depending on number of estimators
                    estimator_alpha = 1.0 / len(M.estimators_)
                    for tree in M.estimators_:
                        Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
                        Z = Z.reshape(xx.shape)
                        cs = plt.contourf(xx, yy, Z, alpha=estimator_alpha, cmap=cmap)

                # build grid
                xx_coarser, yy_coarser = np.meshgrid(np.arange(x_min, x_max, plot_step_coarser), np.arange(y_min, y_max, plot_step_coarser))
                Z_points_coarser = M.predict(np.c_[xx_coarser.ravel(), yy_coarser.ravel()]).reshape(xx_coarser.shape)
                cs_points = plt.scatter(xx_coarser, yy_coarser, s=15, c=Z_points_coarser, cmap=cmap, edgecolors="none")

                # Plot the training points
                plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['r', 'y', 'b']), edgecolor='k', s=20)
                plot_idx += 1


        plt.suptitle(title, fontsize=12)
        plt.axis("tight")
        plt.tight_layout(h_pad=0.2, w_pad=0.2, pad=2.5)
        if directory is not None:
            plt.savefig(os.path.join(directory, 'Decision_surface.png'))
    except Exception as e:
        print('Plot Decision surface failed')
        print(e)
    finally:
        plt.clf()
