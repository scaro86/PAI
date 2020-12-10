#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 15:11:41 2020

@author: dorisfonsecalima
"""

import numpy as np


from sklearn.kernel_approximation import Nystroem
from sklearn import pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern as Mat, RBF as R, WhiteKernel as W, ConstantKernel as C
from sklearn.metrics import make_scorer
from sklearn.cluster import KMeans


import warnings
warnings.filterwarnings('error')



## Constant for Cost function
THRESHOLD = 0.5
W1 = 1
W2 = 20
W3 = 100
W4 = 0.04


def cost_function(true, predicted):
    """
        true: true values in 1D numpy array
        predicted: predicted values in 1D numpy array

        return: float
    """
    cost = (true - predicted)**2

    # true above threshold (case 1)
    mask = true > THRESHOLD
    mask_w1 = np.logical_and(predicted>=true,mask)
    mask_w2 = np.logical_and(np.logical_and(predicted<true,predicted >=THRESHOLD),mask)
    mask_w3 = np.logical_and(predicted<THRESHOLD,mask)

    cost[mask_w1] = cost[mask_w1]*W1
    cost[mask_w2] = cost[mask_w2]*W2
    cost[mask_w3] = cost[mask_w3]*W3

    # true value below threshold (case 2)
    mask = true <= THRESHOLD
    mask_w1 = np.logical_and(predicted>true,mask)
    mask_w2 = np.logical_and(predicted<=true,mask)

    cost[mask_w1] = cost[mask_w1]*W1
    cost[mask_w2] = cost[mask_w2]*W2

    reward = W4*np.logical_and(predicted < THRESHOLD,true<THRESHOLD)
    if reward is None:
        reward = 0
    return np.mean(cost) - np.mean(reward)

"""
Fill in the methods of the Model. Please do not change the given methods for the checker script to work.
You can add new methods, and make changes. The checker script performs:


    M = Model()
    M.fit_model(train_x,train_y)
    prediction = M.predict(test_x)

It uses predictions to compare to the ground truth using the cost_function above.

"""

class Model():

    def __init__(self):
        """
            TODO: enter your code here
        """
        #kernel_gp = 1.0 * Mat(length_scale=[1.1, 1], length_scale_bounds=(1e-4, 100), nu=0.5) 
        kernel_gp = C() + R() + W()
                
        feature_map_nystroem = Nystroem(kernel = kernel_gp, random_state=0, n_components=2)
           
        self.nystroem_approx_gp = pipeline.Pipeline([("feature_map", feature_map_nystroem),
                                        ("gp", GaussianProcessRegressor(alpha=1e-12,  optimizer = make_scorer(cost_function), n_restarts_optimizer=1))])
          
        pass
    
    def minimize_data(self,train_x, train_y):
        kmeans = KMeans(n_clusters=400, random_state=0).fit(np.column_stack((train_x,train_y)))
        data_transformed = kmeans.cluster_centers_
        #dbscan = DBSCAN(min_samples = 200).fit(np.column_stack((train_x,train_y)))
        #data_transformed = dbscan.components_
        self.train_x_minimized = data_transformed[:,0:2]
        self.train_y_minimized = data_transformed[:,2]
        
        pass

    def predict(self, test_x):
        """
            TODO: enter your code here, 
        """
        
        y = self.gp.predict(test_x)
        
        return y

    def fit_model(self, train_x, train_y):
        """
             TODO: enter your code here
        """
        self.minimize_data(train_x,train_y)
        self.gp = self.nystroem_approx_gp.fit(self.train_x_minimized, self.train_y_minimized)


def main():
    train_x_name = "train_x.csv"
    train_y_name = "train_y.csv"

    train_x = np.loadtxt(train_x_name, delimiter=',')
    train_y = np.loadtxt(train_y_name, delimiter=',')

    # load the test dateset
    test_x_name = "test_x.csv"
    test_x = np.loadtxt(test_x_name, delimiter=',')

    M = Model()
    M.fit_model(train_x, train_y)
    prediction = M.predict(test_x)

    print(prediction)
        

if __name__ == "__main__":
    main()
