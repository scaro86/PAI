import numpy as np
import matplotlib.pylab as plt
import pymc3 as pm
import theano
#import seaborn as sns
#theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"



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
        with pm.Model() as self.spatial_model:
    
            l = pm.HalfCauchy("l", beta=3, shape=(2,))
            sf2 = pm.HalfCauchy("sf2", beta=3)
            self.sn2 = pm.HalfCauchy("sn2", beta=3)

            K = pm.gp.cov.ExpQuad(2, l) * sf2**2
    
            self.gp_spatial = pm.gp.MarginalSparse(cov_func=K, approx="FITC")
        pass

    def predict(self, test_x):
        """
            TODO: enter your code here
        """
        ## dummy code below
        #y = np.ones(test_x.shape[0]) * THRESHOLD - 0.00001
        
        with self.spatial_model:

            f_pred = self.gp_spatial.conditional('f_pred', test_x)
            y = f_pred
    
        
        return y

    def fit_model(self, train_x, train_y):
        """
             TODO: enter your code here
        """
        nd = 15
        xu1, xu2 = np.meshgrid(np.linspace(-1, 1, nd), np.linspace(-1, 1, nd))
        Xu = np.concatenate([xu1.reshape(nd*nd, 1), xu2.reshape(nd*nd, 1)], 1)
        
        with self.spatial_model:
            
            obs = self.gp_spatial.marginal_likelihood("obs", X=train_x, Xu=Xu, y=train_y, noise=self.sn2)

            mp = pm.find_MAP()

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

    X_obs = train_x
    y_obs = train_y
    
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


if __name__ == "__main__":
    main()