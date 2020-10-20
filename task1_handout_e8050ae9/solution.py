import numpy as np


from sklearn.kernel_approximation import Nystroem
from sklearn import pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern as M, RBF as R, WhiteKernel as W, ConstantKernel as C
from sklearn.metrics import make_scorer
import scipy
from sklearn.cluster import KMeans, DBSCAN





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
        self.kernel_gp = 1.0 * M(length_scale=1.6, length_scale_bounds=(1e-5, 1e5), nu=1.5)  \
                +C(constant_value=1.5) \
                +W(noise_level=2, noise_level_bounds=(1e-10, 1e+1)) 
        #kernel_gp = M(length_scale=[1.5, 1.5], nu=1.5)
        #kernel_gp = R(length_scale=1.0, length_scale_bounds=(1e-5, 1e5))
       
        self.model = GaussianProcessRegressor(kernel=self.kernel_gp, n_restarts_optimizer=0, random_state=0)
        
        pass
    
    def minimize_data(self,train_x, train_y):
        kmeans = KMeans(n_clusters=100, random_state=0).fit(np.column_stack((train_x,train_y)))
        data_transformed = kmeans.cluster_centers_
        #dbscan = DBSCAN(min_samples = 200).fit(np.column_stack((train_x,train_y)))
        #data_transformed = dbscan.components_
        self.train_x_minimized = data_transformed[:,0:2]
        self.train_y_minimized = data_transformed[:,2]
        
        pass
    
    def optimizer(self):
    # * 'obj_func' is the objective function to be minimized, which
    #   takes the hyperparameters theta as parameter and an
    #   optional flag eval_gradient, which determines if the
    #   gradient is returned additionally to the function value
    # * 'initial_theta': the initial value for theta, which can be
    #   used by local optimizers
    # * 'bounds': the bounds on the values of theta
        first_prediction = self.model.predict(self.train_x_minimized)
        #self.obj_func = cost_function(self.train_y_minimized,first_prediction)
        initial_theta = self.model.kernel_.theta
        bounds = self.model.kernel_.bounds
        theta_opt = scipy.optimize.minimize(self.obj_func, initial_theta, bounds=bounds).x
    # Returned are the best found hyperparameters theta and
    # the corresponding value of the target function.
        return theta_opt
    
    
    def obj_func(self,hyperparams):
        self.model.kernel_.theta = hyperparams
        prediction = self.model.predict(self.train_x_minimized)
        cost = cost_function(self.train_y_minimized,prediction)
        self.model.kernel_.theta = hyperparams

        return cost
    """
	def optimizer(self):
		initial_theta = self.model.kernel_.theta
		optimalResult = scipy.optimize.minimize(self.obj_func, initial_theta, method='BFGS')
		theta_opt = optimalResult.x
		return theta_opt
    """

    def predict(self, test_x):
        """
            TODO: enter your code here
        """
        ## dummy code below
        #y = np.ones(test_x.shape[0]) * THRESHOLD - 0.00001
        
        y = self.model.predict(test_x)
        
        return y

    def fit_model(self, train_x, train_y):
        """
             TODO: enter your code here
        """
        """
        
   
        """
        self.minimize_data(train_x,train_y)
        self.model.fit(self.train_x_minimized,self.train_y_minimized)
        self.model.kernel_.theta = self.optimizer()
        pass


def main():
    train_x_name = "train_x.csv"
    train_y_name = "train_y.csv"

    train_x = np.loadtxt(train_x_name, delimiter=',')
    train_y = np.loadtxt(train_y_name, delimiter=',')
    
    
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
    M.fit_model(train_x, train_y)
    prediction = M.predict(test_x)

    print(prediction)
    #print(cost_function(train_y, prediction))

if __name__ == "__main__":
    main()
