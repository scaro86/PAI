import numpy as np


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern as M, WhiteKernel as W, ConstantKernel as C
import scipy
from sklearn.cluster import KMeans





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
    
        self.kernel_gp = 1.0 * M(length_scale=[1.6, 1],  length_scale_bounds=(1e-4, 100), nu=2.5)  \
                +W(noise_level=2, noise_level_bounds=(1e-10, 1e+1)) \
                +C(constant_value=1.5) 
       
        self.model = GaussianProcessRegressor(kernel=self.kernel_gp, n_restarts_optimizer=0, random_state=0)
        
        pass
    
    
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
        
        self.model.kernel_.theta = place_holder
        first_prediction = self.model.predict(self.train_x_minimized)
        cost = cost_function(self.train_y_minimized,first_prediction)

        return cost


    def predict(self, test_x):
        
        y = self.model.predict(test_x)
        
        return y

    def fit_model(self, train_x, train_y):
       
        self.minimize_data(train_x,train_y)
        self.model.fit(self.train_x_minimized,self.train_y_minimized)
        self.model.kernel_.theta = self.optimizer()
        
        pass


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
