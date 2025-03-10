from os import setegid
import numpy as np

class Optimizer:
    """
    This class contains some continuous optimization algorithms
    """
    def __init__(self, func, constrain=None):
        """
        Parameters:
        ----------- 
        func : 
            is the objective function, i.e. function that minimal is to be found.
        X : tuple of np.ndarray
            The data points, variable of func (not to be confused with the parameter)
        method : 
            the optimization method to optimize the objective function
        constrain :
            the type of constrain.
            This value should be set at None for an unconstrained optimization.
            Others possible value are : "projection", "penalty", or barrier.
            The default is None
        """
        #super().__init__()
        self.func = func
        self.constrain = constrain

    def forward_diff(self, *args, h) -> np.ndarray:
        """
        evaluate derivative by forward difference
        --------------
        Parameters
        func: function
            the objective function
        args:
            arguments of func
        h: float of np.ndarray
            value of small steps for differentiation one for each argument
            for e.g. if args of len 3 h must be np.ndarray of len 3 or a list. 
        """
        # convert args to a modifiable format
        args = np.asarray(args, dtype="float")
        # initialise the gradient matrix
        df = np.zeros_like(args)
        # computing the gradient with repect to parameters
        for i in range(len(df)):
            args_step = args.copy()
            args_step[i] += h[i]
            df[i] = (self.func(*args_step) - self.func(*args)) / h[i]
        return df

    def centered_diff(self, *args, h)-> np.ndarray:
        """
        evaluate derivative by centered difference
        --------------
        Parameters
        func: function
            the objective function
        args:
            arguments of func
        h: np.ndarray of float
            value of small steps for differentiation one for each argument
            for e.g. if args of len 3 h must be np.ndarray of len 3 or a list. 
        """
        # convert args to a modifiable format
        args = np.asarray(args)
        # initialise the gradient matrix
        df = np.zeros_like(args)
        # computing the gradient with repect to parameters
        for i in range(len(df)):
            args_step_1 = args.copy()
            args_step_2 = args.copy()
            args_step_1[i] += h[i]
            args_step_2 -= h[i]
            df[i] = (self.func(*args_step_1) - self.func(*args_step_2)) / (2*h[i])
        return df

    def steepest_direction(self, grad:np.ndarray, norm=2, H:np.ndarray=None)->np.ndarray:
        """
        return dual norm of vec given norm
        """
        if norm == 2:
            return - grad
        elif norm == 1:
            new_grad = np.zeros_like(grad)
            index_max = np.argmax(abs(grad)) 
            new_grad[index_max] = grad[index_max]
            return new_grad
        elif norm == np.inf:
            return np.linalg.norm(grad, ord=1) * np.sign(grad)
        elif norm == "quad" and H != None:
            return - np.inv(H) @ grad

    def backtracking(self, *args, grad, deltax, step_size, beta:float, alpha:float=0.1)->float:
        """
        Line search algorithm (See chapter 3 of the book 
        "Convex Optimization" by Stephen Boyd & Lieven Vandenberghe)
        ----------
        Parameter
        args:
            arguments of function to minimize
        grad: np.ndarray
            the gradient vector
        deltax: np.ndarray
            a steepest direction
        step_size: float or np.ndarray
            the steepest step size
        beta: float or np.ndarray
            factor scaling step_size
            is often chosen to be between 0.1 (which corresponds
            to a very crude search) and 0.8 (which corresponds
            to a less crude search).
            See page 466 of "Convex Optimization" by Stephen Boyd & Lieven Vandenberghe
        alpha: float or np.ndarray
            a parameter of the backtracking
            Values of alpha typically range between 0.01 and 0.3
            The default is 0.1
        """
        # convert args to a modifiable format
        args = np.asarray(args)
        while self.func((args + step_size*deltax)) < self.func(*args) + alpha * step_size * grad @ deltax:
            step_size = beta * step_size
        return step_size


    def steepest_descent(self, *args, jac="forward", norm=2, H=None, step_size=0.5, backtrack=False, eta=1e-6,
                        max_iter=100, h=5.e-4, history=False, beta=0.5, alpha=0.1):
        """
        Search for local minima using following the steepest descent.
        ------
        Parameters
        args:
            are the arguments of the objective function including both the independent variables
            and eventual parameter to fit
        jac: str or function
            the jacobian matrix of the objective function
            or alternatively, indicate the differentiation method to use to estimate it
            either "forward" of "centered".
            The default is "forward"
        step_size: float or np.ndarray of float
            the learning rate (also called step_size or aggressivity)
            The default is 0.5.
        eta: float
            the tolerance criteria on the convergence of the gradient.
            The default is 1e-5
        max_iter: int
            maximum step of iteration
            The default is 100.
        h: np.ndarray of float
            value of small steps for differentiation one for each argument
            for e.g. if args of len 3 h must be np.ndarray of len 3 or a list.
        """
        traj = []
        args = np.asarray(args)
        # h must be an iterable
        if isinstance(h, (float, int)):
            h = np.full_like(args, h, dtype="float")
        # ensure step_size type or size match expectation
        if isinstance(step_size, (list, tuple)):
            step_size = np.asarray(step_size)
            if len(step_size) != len(args):
                raise ValueError(f"Size of step_size must match number of parameters to optimize")

        #selection of the method for differentiation
        if jac.lower() == "forward":
            jac = lambda *args : self.forward_diff(*args, h=h)
        elif jac.lower() == "centered":
            jac = lambda *args : self.centered_diff(*args, h=h)
        # optimisation procedure
        for i in range(max_iter):
            #record history
            if history:
                traj.append(*args)

            # direction search
            grad = jac(*args)
            step = self.steepest_direction(grad, norm=norm, H=H)
            # backtracking
            if backtrack:
                step_size = self.backtracking(*args, grad=grad, deltax=step,
                                            deltax=step_size, beta=beta, alpha=alpha
                                            )
            #criteria on the gradient to not waste resources when stuck in a flat region
            if np.linalg.norm(grad) >= eta:
                    args += step_size * self.steepest_direction
            else:
                print(f"Converged at step {i}\n")
                break

        return args, self.func(*args), np.asarray(traj, dtype="float")
