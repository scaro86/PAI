import numpy as np


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern as Mat, RBF as R, WhiteKernel as W, ConstantKernel as C
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


class Model():

    def __init__(self):
        """
            TODO: enter your code here
        """
        kernel_gp = 1.0 * Mat(length_scale=[1.1, 1], length_scale_bounds=(1e-4, 100), nu=0.5) 
                
        feature_map_nystroem = Nystroem(kernel = kernel_gp, random_state=0, n_components=4)
           
        self.nystroem_approx_gp = pipeline.Pipeline([("feature_map", feature_map_nystroem),
                                        ("gp", GaussianProcessRegressor(alpha=1e-12,  optimizer = make_scorer(cost_function), n_restarts_optimizer=1))])
          
        pass
    
    
<<<<<<< HEAD
    def minimize_data(self,train_x, train_y):
        
        kmeans = KMeans(n_clusters=418, random_state=0).fit(np.column_stack((train_x,train_y)))
        data_clustered = kmeans.cluster_centers_
        self.train_x_minimized = data_clustered[:,0:2]
        self.train_y_minimized = data_clustered[:,2]
        
        pass
    
    
    def optimizer(self):
    
        initial_theta = self.model.kernel_.theta
        bounds = self.model.kernel_.bounds
        theta_opt = scipy.optimize.minimize(self.fun, initial_theta, bounds=bounds).x
    
        return theta_opt
    
    
    def fun(self,place_holder):
=======

    def predict(self, test_x):
        """
            TODO: enter your code here, 
        """
        
        y = self.gpr.predict(test_x)
        
        return y

    def fit_model(self, train_x, train_y):
        """
             TODO: enter your code here
        """

        self.gp = self.nystroem_approx_gp.fit(train_x, train_y)


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
<<<<<<< HEAD
    
=======
        

>>>>>>> doris
if __name__ == "__main__":
    main()
