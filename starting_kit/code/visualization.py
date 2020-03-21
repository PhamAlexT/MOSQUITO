
"""
Fonctions permettant d'obtenir les plots les plus necessaires 
"""


import numpy as np
import pandas as pd
from model import model

from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from libscores import get_metric
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from matplotlib.colors import ListedColormap
    

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,  ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB


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
    #Plot
    plt.figure(figsize=(6,6))
    plt.xlabel("n_estimator")
    plt.ylabel('Score')
    plt.title('Score results of RandomForest with cross-validation')
    plt.errorbar(nb_arbre, s, var, label='Test set')
 

def f_roc(X_train, Y_train):  
    
    M = model()
    
    size = round(len(X_train)/10)
    probas = M.classifier.fit(X_train[:size], Y_train[:size].ravel()).predict_proba(X_train[size:])

    fpr, tpr, thresholds = roc_curve(Y_train[size:].ravel(), probas[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    
def f_mat_conf(X_train, Y_train):
    
    X_train_pre, X_test_pre, y_train_pre, y_test_pre = train_test_split(X_train,Y_train, test_size=0.33, random_state=42)
    
    M = model()                       
    Y = M.fit(X_train_pre, y_train_pre).predict(X_test_pre)
    confusion_matrix(y_test_pre, Y)
    
    #Plot
    f, ax = plt.subplots(figsize=(8, 6))
    ax = sns.heatmap(confusion_matrix(y_test_pre, Y), annot=True, fmt='d', ax=ax, cmap="YlGnBu"); 


    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['Parasitized', 'Uninfected']);
    ax.yaxis.set_ticklabels(['Parasitized', 'Uninfected']);
    plt.show()

def f_test_models (X_train, Y_train):

    metric_name, scoring_function = get_metric()
    model_name = ["Nearest Neighbors",
         "Decision Tree", "Random Forest",  "AdaBoost",
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
        s_prime = cross_validate(model_list[i], X_train, Y_train,cv=5, scoring=make_scorer(scoring_function), return_train_score=True)
        s_train.append(s_prime['train_score'].mean())
        s_test.append(s_prime['test_score'].mean())
    d = {'Score_train': s_train, 
            'Score_test': s_test} 
    
    #Plot
    sd = pd.DataFrame(d, index=[model_name[i] for i in range(len(model_name))] ) 
    ax = sd.plot.bar()
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    plt.show()

    

#Le code ci dessous est tiré en grande partie d'un exemple de la bib de scikit learn   et a ete adapté a notre probleme ! 

def f_decision_surface (X_train,Y_train):


    # Parameters
    n_classes = 2
    n_estimators = 20
    cmap = plt.cm.RdYlBu
    plot_step = 0.02  # fine step width for decision surface contours
    plot_step_coarser = 0.5  # step widths for coarse classifier guesses
    RANDOM_SEED = 13  # fix the seed on each iteration


    plot_idx = 1


    model_name = [
             "Decision Tree", "Random Forest",  "AdaBoost"]
    models = [
        DecisionTreeClassifier(max_depth=10),
        RandomForestClassifier(max_depth=100, n_estimators=100),
        AdaBoostClassifier()
        ]

    #on prend 3 features (celles qui ont eu les meilleures scores) 
    for pair in ([7, 5], [7, 8], [7, 4]):
        for m_name in models:
            # We only take the two corresponding features
            X = X_train[:, pair]
            y = Y_train

            # Shuffle
            idx = np.arange(X.shape[0])
            np.random.seed(RANDOM_SEED)
            np.random.shuffle(idx)
            X = X[idx]
            y = y[idx]

            # Standardize
            mean = X.mean(axis=0)
            std = X.std(axis=0)
            X = (X - mean) / std

            # Train
            M = m_name
            M.fit(X, y)

            scores = M.score(X, y)
            # Create a title for each column and the console by using str() and
            # slicing away useless parts of the string
            model_title = str(type(M)).split(
                ".")[-1][:-2][:-len("Classifier")]

            model_details = model_title
            if hasattr(M, "estimators_"):
                model_details += " with {} estimators".format(
                    len(M.estimators_))
            print(model_details + " with features", pair,
                  "has a score of", scores)

            plt.subplot(3, 3, plot_idx)
            if plot_idx <= len(models):
                # Add a title at the top of each column
                plt.title(model_title, fontsize=9)

            # Now plot the decision boundary using a fine mesh as input to a
            # filled contour plot
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                                 np.arange(y_min, y_max, plot_step))

            # Plot either a single DecisionTreeClassifier or alpha blend the
            # decision surfaces of the ensemble of classifiers
            if isinstance(M, DecisionTreeClassifier):
                Z = M.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                cs = plt.contourf(xx, yy, Z, cmap=cmap)
            else:
                # Choose alpha blend level with respect to the number
                # of estimators
                # that are in use (noting that AdaBoost can use fewer estimators
                # than its maximum if it achieves a good enough fit early on)
                estimator_alpha = 1.0 / len(M.estimators_)
                for tree in M.estimators_:
                    Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
                    Z = Z.reshape(xx.shape)
                    cs = plt.contourf(xx, yy, Z, alpha=estimator_alpha, cmap=cmap)

            # Build a coarser grid to plot a set of ensemble classifications
            # to show how these are different to what we see in the decision
            # surfaces. These points are regularly space and do not have a
            # black outline
            xx_coarser, yy_coarser = np.meshgrid(
                np.arange(x_min, x_max, plot_step_coarser),
                np.arange(y_min, y_max, plot_step_coarser))
            Z_points_coarser = M.predict(np.c_[xx_coarser.ravel(),
                                             yy_coarser.ravel()]
                                             ).reshape(xx_coarser.shape)
            cs_points = plt.scatter(xx_coarser, yy_coarser, s=15,
                                    c=Z_points_coarser, cmap=cmap,
                                    edgecolors="none")

            # Plot the training points, these are clustered together and have a
            # black outline
            plt.scatter(X[:, 0], X[:, 1], c=y,
                        cmap=ListedColormap(['r', 'y', 'b']),
                        edgecolor='k', s=20)
            plot_idx += 1  # move on to the next plot in sequence

    plt.suptitle("Classifiers on feature subsets of the Medichal dataset", fontsize=12)
    plt.axis("tight")
    plt.tight_layout(h_pad=0.2, w_pad=0.2, pad=2.5)
    plt.show()
