# coding: utf-8

'''
Description :
    Functions to extract features for RAW data.
'''

import numpy as np
import math


__author__ = "MOSQUITO"


def f_min_max_mean_std_gray (X):
    '''
    extract min, max, mean, var of gray of X data
     args:
         X (numpy array): data. 
     return:
         4 numpy arrays
    '''
    min_gray = []
    max_gray = []
    mean_gray = []
    std_gray = []
    for i in range(len(X)):
        X_i = X[i]
        X_prime = X_i[X_i>0.3]
        X_prime = X_prime[X_prime<0.8]
        min_gray.append(X_prime.min())
        max_gray.append(X_prime.max())
        mean_gray.append(X_prime.mean())
        
        X_prime = X_i[X_i>0]
        X_prime = X_prime[X_prime<1]
        std_gray.append(X_prime.std())
    return min_gray, max_gray, mean_gray, std_gray


def histogramme(xmin, xmax, Nb, ech):
    '''
    Histogram of values of ech with nb intervals
     args:
         xmin (int): min of ech
         xmax (int): max of ech
         Nb  (int): number of intervals
         ech (numpy array): data. 
     return:
         numpy array
    '''
    histo = np.zeros(Nb)
    intervalSize = (xmax-xmin)/float(Nb)
    corrInerval = -1
    for i  in range (len (ech)):
        corrInerval = int((ech[i]-xmin)/float(intervalSize))
        histo[corrInerval]+=1

    return histo



def nb_pixels (X, nb):
    '''
    Transform each X[i] data into histogram
     args:
         X (numpy array): data
         nb (int) : number of intervals
     return:
         numpy array
    '''
    Histo = np.zeros(shape=(len(X),nb))
    for i in range(len(X)):
        h = histogramme(0,1,nb,X[i]).astype(int)
        Histo[i] = h
    return Histo


def extract_features(X, n):
    '''
    Concatenate all features extracted from X
     args:
         n (int): number of intervals used for histogram
         X (numpy array): data. 
     return:
         numpy array
    '''
    min_gray, max_gray, mean_gray, std_gray = f_min_max_mean_std_gray (X)
    feat4 = np.vstack((np.array(mean_gray), np.array(min_gray), np.array(max_gray),np.array(std_gray) )).T

    hist = nb_pixels(X, n)
    pixels_i_j = hist.astype(int)
    
    X_new = np.concatenate((feat4, pixels_i_j), axis=1)
    
    return X_new
