import numpy as np
import scipy as sp
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern as M
from sklearn.gaussian_process.kernels import ConstantKernel

domain = np.array([[0, 5]])

np.random.seed(1)


""" Solution """


class BO_algo():
    def __init__(self):
               
        """Initializes the algorithm with a parameter configuration. """

        self.sigma_f = 0.15
        self.sigma_v = 1e-4
        self.v_min = 1.2
        
        #initialize the first datapoint to sample from
        self.xpoints = np.array([[2]])
        self.fpoints = np.array([[0]])
        self.vpoints = np.array([[2]])
        
        #define GP prior kernel for funciton f
        var_f = 0.5
        kernel_f = var_f * M(length_scale=0.5, nu=2.5) #cf @294 pour *var_f
        self.gpf = GaussianProcessRegressor(kernel=kernel_f,alpha=0.15, random_state=1)
        
        #define GP prior kernel for funciton v
        mean_v = 1.5
        var_v = np.sqrt(2)
        kernel_v = var_v * M(length_scale= np.sqrt(2), nu=2.5) + ConstantKernel(constant_value=mean_v, constant_value_bounds="fixed")
        self.gpv = GaussianProcessRegressor(kernel=kernel_v, alpha = 0.0001, random_state=1)
        
        pass


    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """
        

        # TODO: enter your code here
        # In implementing this function, you may use optimize_acquisition_function() defined below.
        #updates model and then optimizes the acquisition function to find next  point to 
        self.gpf.fit(self.xpoints.reshape(-1,1), self.fpoints)
        self.gpv.fit(self.xpoints.reshape(-1,1), self.vpoints)
        
        
        recom_x = self.optimize_acquisition_function()
        return recom_x


    def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.

        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximize the acquisition function.
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * \
                 np.random.rand(domain.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=domain,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *domain[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        return np.atleast_2d(x_values[ind])

    def acquisition_function(self, x):
        """
        Compute the acquisition function.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f

        Returns
        ------
        af_value: float
            Value of the acquisition function at x
        """
        
        #values for f
        #x = x.reshape(-1, 1)
        xi = 0.02
            
        mu_f, sigma_f = self.gpf.predict(x.reshape(-1,1), return_std=True)
        #print(type(x[0].item()))
        #y_f = np.random.normal(0, self.sigma_f, y_f.shape[0])
        #print(type(sigma_f))
        
        
        f_max = np.max(self.fpoints)
        
        Z = (mu_f - f_max - xi) / sigma_f 
        
        #Aquisition function corresponding to expected improvement
        
        if sigma_f.any() == 0:
            af_value_f = 0
        else:
            af_value_f = (mu_f - f_max) * sp.stats.norm.cdf(Z) + sigma_f * sp.stats.norm.pdf(Z)
        
        #values for v
        #v_out = v(x)
        #v_out = np.array([v_out])
        #output_v = self.gpv.fit(x, v_out)
        vcurrent = self.vpoints[[0]][-1]
        constraint_func = -np.log(self.v_min) + np.log(vcurrent)
        #print(type(constraint_func.item()))
        mu_v, sigma_v = self.gpv.predict(constraint_func.reshape(1,-1), return_std=True)
        
              
        #final af_value including constraint v
        af_value = af_value_f*sp.stats.norm.cdf(mu_v/sigma_v)
        #print(type(af_value))
        
        return af_value[0].item()


    def add_data_point(self, x, f, v):
        """
        Add data points to the model.

        Parameters
        ----------
        x: np.ndarray
            Hyperparameters
        f: np.ndarray
            Model accuracy
        v: np.ndarray
            Model training speed
        """

        # TODO: enter your code here
        self.xpoints = np.append(self.xpoints,x)
        self.fpoints = np.append(self.fpoints,f)
        self.vpoints = np.append(self.vpoints,v)


    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """
            
        # TODO: enter your code here
        #check here if v(x)<1.2 
        #print(self.fpoints)
        x_pos = np.argmax(self.fpoints)
        x_opt = self.xpoints[x_pos]
        
        
        
        if self.vpoints[x_pos] >= 1.2:
            print("perfect")
            
            
        else:
            print("v violated")
            counter = 0
            while self.vpoints[x_pos] < 1.2 and counter < self.xpoints.shape[0]:
                #print("fpoints")
                #print(self.fpoints)
                #print("vpoints")
                #print(self.vpoints)
                self.fpoints[x_pos] = -1e5
                x_pos = np.argmax(self.fpoints)
                x_opt = self.xpoints[x_pos]
                counter = counter + 1
            
            
        return x_opt
           # raise NotImplementedError


""" Toy problem to check code works as expected """

def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= domain[None, :, 0]) and np.all(x <= domain[None, :, 1])


def f(x): #don't have to access it see @254
    """Dummy objective"""
    mid_point = domain[:, 0] + 0.5 * (domain[:, 1] - domain[:, 0])
    return - np.linalg.norm(x - mid_point, 2)  # -(x - 2.5)^2


def v(x):
    """Dummy speed"""
    return 2.0


def main():
    # Init problem
    agent = BO_algo()

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, domain.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {domain.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x)
        cost_val = v(x)
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain.shape[0]), \
        f"The function get solution must return a numpy array of shape (" \
        f"1, {domain.shape[0]})"
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the ' \
        f'domain, {solution} returned instead'

    # Compute regret
    if v(solution) < 1.2:
        regret = 1
    else:
        regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret{regret}')


if __name__ == "__main__":
    main()