import numpy as np


from sklearn.kernel_approximation import Nystroem
from sklearn import pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern as M, RBF as R, WhiteKernel as W, ConstantKernel as C
from sklearn.metrics import make_scorer


import warnings
warnings.filterwarnings('error')

import random

random.seed(1)



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
        kernel_gp = 1.0 * M(length_scale=1.5, length_scale_bounds=(1e-5, 1e5), nu=1.5) #\
                #+W(noise_level=1, noise_level_bounds=(1e-10, 1e+1)) \
                #+C(constant_value=0.3)
        #kernel_gp = R(length_scale=1.0, length_scale_bounds=(1e-5, 1e5))
        feature_map_nystroem = Nystroem(kernel = kernel_gp, random_state=1,n_components=10)
        #feature_map_nystroem = Nystroem(n_components=10)
        self.nystroem_approx_gp = pipeline.Pipeline([("feature_map", feature_map_nystroem),
                                        ("gp", GaussianProcessRegressor(optimizer = make_scorer(cost_function)))])
        #self.nystroem_approx_gp = GaussianProcessRegressor(kernel = kernel_gp, optimizer = make_scorer(cost_function))
          
        pass
    
    

    def predict(self, test_x):
        """
            TODO: enter your code here, 
        """
        ## dummy code below
        #y = np.ones(test_x.shape[0]) * THRESHOLD - 0.00001
        
        y = self.gpr.predict(test_x)
        
        return y

    def fit_model(self, train_x, train_y):
        """
             TODO: enter your code here
        """
        # self.gp = self.search.fit(train_x, train_y)
        self.gp = self.nystroem_approx_gp.fit(train_x, train_y)
        
        train_x_unique, indices_train_unique = np.unique(train_x, axis = 0, return_index = True)
        sorted_indices_unique = np.sort(indices_train_unique)
        # print(train_x[sorted_indices_unique[0]])
        # print(train_x[2*sorted_indices_unique.shape[0]+1])
        # print(train_unique.shape[0])
        # print(train_y.shape)

        train_y_mean = np.zeros((train_x_unique.shape[0], ))
        num_unique = sorted_indices_unique.shape[0]


        for i in range(sorted_indices_unique.shape[0]):
            equal_values = np.array([train_y[sorted_indices_unique[i]], train_y[sorted_indices_unique[i]+num_unique], train_y[sorted_indices_unique[i]+2*num_unique]])
            train_y_mean[i] = np.mean(equal_values)
   

        
        self.gp = self.nystroem_approx_gp.fit(train_x_unique, train_y_mean)
        pass


def main():
    train_x_name = "train_x.csv"
    train_y_name = "train_y.csv"

    train_x = np.loadtxt(train_x_name, delimiter=',')
    train_y = np.loadtxt(train_y_name, delimiter=',')
    
    train_unique, indices_train_unique = np.unique(train_x, axis = 0, return_index = True)
    sorted_indices_unique = np.sort(indices_train_unique)


    train_y_mean = np.zeros((train_unique.shape[0], ))
    num_unique = sorted_indices_unique.shape[0]


    for i in range(sorted_indices_unique.shape[0]):
        equal_values = np.array([train_y[sorted_indices_unique[i]], train_y[sorted_indices_unique[i]+num_unique], train_y[sorted_indices_unique[i]+2*num_unique]])
        train_y_mean[i] = np.mean(equal_values)
   
    
    #plot the dataset
    #nx = 40
    #x1, x2 = np.meshgrid(np.linspace(0,300,nx), np.linspace(0,300,nx))
    #X = np.concatenate([x1.reshape(nx*nx, 1), x2.reshape(nx*nx, 1)], 1)

    #X_obs = train_x
    #y_obs = train_y
    
    #with sns.axes_style("white"):
        #plt.figure(figsize=(10,8))
        #plt.scatter(X_obs[:,0], X_obs[:,1], s=50, c=y_obs, marker='s', cmap=plt.cm.viridis);

    # load the test dateset
    test_x_name = "test_x.csv"
    test_x = np.loadtxt(test_x_name, delimiter=',')

    M = Model()
    M.fit_model(train_unique, train_y_mean)
    prediction = M.predict(test_x)

    print(prediction)
        

if __name__ == "__main__":
    main()
