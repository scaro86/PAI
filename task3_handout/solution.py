import numpy as np
import scipy as sp
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern as M

domain = np.array([[0, 5]])

np.random.seed(1)


""" Solution """


class BO_algo():
    def __init__(self):
               
        """Initializes the algorithm with a parameter configuration. """

        self.v_min = 1.2
        
        #define GP prior kernel for funciton f
        kernel_f = M(length_scale=0.5, nu=2.5)  
        var_f = 0.5
        self.gpf = GaussianProcessRegressor(kernel=kernel_f, alpha=var_f, random_state=1)
        
        #define GP prior kernel for funciton v
        mean_v = 1.5
        kernel_v = M(length_scale= np.sqrt(2), nu=2.5) + mean_v 
        var_v = np.sqrt(2)
        self.gpv = GaussianProcessRegressor(kernel=kernel_v, alpha=var_v, random_state=1)
        
        #initialize the first datapoint to sample from
        self.xpoints = np.array([[2]])
        self.predpoints = self.gpf.predict(self.xpoints).reshape(-1,1)
        self.constpoints = self.gpv.predict(self.xpoints).reshape(-1,1)

        pass


    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """
        
        #updates model and then optimizes the aquisition funciton to find next point to sample
        self.gpf.fit(self.xpoints, self.predpoints)
        
        #test to check model is being updated
        print(self.xpoints)
        print(self.predpoints)
        

        x_opt = self.optimize_acquisition_function()
        x_opt = np.array([[x_opt]])
        

        # TODO: enter your code here
        # In implementing this function, you may use optimize_acquisition_function() defined below.
        return x_opt


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
        
        print(np.atleast_2d(x_values[ind]).shape)
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
        
        xi = 0.01
        
        #values for f
        mu_f, var_f = self.gpf.predict(x, return_std=True)
        #y_f = pred_f + np.random.normal(0, self.sigma_f, pred_f.shape[0])
        
        
        ymax_f = np.max(self.predpoints)
        
        vmax = np.max(self.costpoints)
        
        Z = (mu_f - ymax_f - xi) / var_f 
        
        xi = 0.01
        
        #Aquisition function corresponding to expected improvement
        
        if var_f == 0:
            af_value_f = 0
        else:
            af_value_f = (mu_f - ymax_f - xi) * sp.norm.cdf(Z) + var_f * sp.norm.pdf(Z)
        
        #values for v
        output_v = self.gpv.fit(x)
        constraint_func = -np.log(self.v_min) + np.log(vmax)
        
              
        #final af_value including constraint v
        af_value = af_value_f*(1 - sp.norm.cdf(constraint_func))
        
        return af_value


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
        
        x_new_point = self.next_recommendation()
        f_new_point = self.gpf.predict(x_new_point)
        v_new_point = self.gpv.predict(x_new_point)
        
        self.xpoints = np.vstack((self.xpoints, x_new_point))
        self.predpoints = np.vstack((self.predpoints, f_new_point))
        self.constpoints = np.vstack((self.constpoints, v_new_point))


    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """

        # TODO: enter your code here
        raise NotImplementedError


""" Toy problem to check code works as expected """

def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= domain[None, :, 0]) and np.all(x <= domain[None, :, 1])


def f(x):
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