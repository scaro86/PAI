import numpy as np


from sklearn.kernel_approximation import Nystroem
from sklearn import pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern as Mat, RBF as R, WhiteKernel as W, ConstantKernel as C, RationalQuadratic as RQ, DotProduct
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV






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
        # kernel_gp = 1.0 * Mat(length_scale=0.5, length_scale_bounds=(1e-3, 2), nu=0.5) \
        #         +W(noise_level=1, noise_level_bounds=(1e-10, 1e+1)) \
        #         +C(constant_value=0.3)
        #kernel_gp = 1.0*R(1.0)+W(noise_level=1, noise_level_bounds=(1e-10, 1e+1))+C(constant_value=0.3)

        my_scorer = make_scorer(cost_function)

        grid = dict()
        grid['alpha'] = [1e0, 1e-1, 1e-2, 1e-3]
        grid['kernel']= [1*R(), 1*DotProduct(), 1*Mat(), 1*RQ(), 1*W()]
        # grid['optimizer'] = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
        
        #Define the model 
        
        feature_map_nystroem = Nystroem(kernel = kernel_gp, random_state=1,n_components=10)
        nystroem_approx_gp = pipeline.Pipeline([("feature_map", Nystroem()),
                                        ("gp", GaussianProcessRegressor())])
        
        #Define search
        # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        cv = 3
        self.search = GridSearchCV(GaussianProcessRegressor(), param_grid=grid, scoring = my_scorer, cv =cv)
        #perform the search
                        




            
        pass

    def predict(self, test_x):
        """
            TODO: enter your code here, 
        """
        ## dummy code below
        #y = np.ones(test_x.shape[0]) * THRESHOLD - 0.00001
        
        y = self.gp.predict(test_x)
        
        return y

    def fit_model(self, train_x, train_y):
        """
             TODO: enter your code here
        """
        self.gp = self.search.fit(train_x, train_y)
        # self.gp = self.nystroem_approx_gp.fit(train_x, train_y)
        
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
        

if __name__ == "__main__":
    main()
