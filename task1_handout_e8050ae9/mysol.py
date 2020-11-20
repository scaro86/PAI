#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 16:31:51 2020

@author: carolinesauget
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.kernel_approximation import (Nystroem, RBFSampler)


from sklearn.preprocessing import LabelEncoder


train_x_name = "train_x.csv"
train_y_name = "train_y.csv"

train_x = np.loadtxt(train_x_name, delimiter=',')
train_y = np.loadtxt(train_y_name, delimiter=',')

# load the test dateset
test_x_name = "test_x.csv"
test_x = np.loadtxt(test_x_name, delimiter=',')

#preprocessing the data

feature_map_nystroem = Nystroem(kernel='rbf', gamma=1, random_state=1)
feat_map = feature_map_nystroem.fit(train_x)
idx = feat_map.component_indices_
X_features = [train_x[i] for i in idx]


# rbf_feature = RBFSampler(gamma=1, random_state=1)
# X_features = rbf_feature.fit_transform(train_x)

kernel = 1*RBF(1.0)
model = GaussianProcessRegressor(kernel=kernel,random_state=0)
model.fit(X_features, train_y)
model.score(X_features, train_y)



#   #Define the grid 

# results = search.fit(X_features, train_y)
        # #summarize best
        # print('Best mean accuracy : %.3f' %results.best_score_)
        # print('Best Config :%s' %results.best_params_)

# #summarize all
# means = results.cv_results_['mean_test_score']
# params = results.cv_results_['params']
# for mean, param in zi(means, params):
#     print(">%.3f with: %r" % (mean, param))



