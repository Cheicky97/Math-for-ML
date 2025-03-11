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
        Compute steepest descent direction regarding the norm chosen
        ------------
        Parameters:
        grad: np.ndarray
            Gradient
        norm : int or str
            fix the norm.
            Possible values:
            norm = 0 for quadratic norm, in this case a matrix H has to be provided.
            norm = 1 for l1-norm
            norm = np.inf for linf-norm
            norm = 2 for the euclidean norm 
        H: np.ndarray
            matrix used in the definition of the quadratic norm.
        -----------
        return
            (np.ndarray) steepest descent direction
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
        elif norm == 0 and H != None:
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
        while self.func((args + step_size*deltax)) > self.func(*args) + alpha * step_size * grad @ deltax:
            step_size *= beta
        return step_size

    def steepest_descent(self, *args, jac="forward", norm:int=2, P:np.ndarray=None, step_size:float=1.0, eta:float=1e-6,
                          backtrack=False, beta_bt:float=0.5, alpha_bt:float=0.1,
                         moment_gd:bool=False, zeta_momentum:float=0.8, max_iter:int=100, h:float=5.e-4,
                         history=False):
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
        norm : int or str
            fix the norm.
            Possible values:
            norm = 0 for quadratic norm, in this case a matrix P has to be provided.
            norm = 1 for l1-norm
            norm = np.inf for linf-norm
            norm = 2 for the euclidean norm 
            The default is 2.
        P: np.ndarray
            matrix used in the definition of the quadratic norm.
            The default is None.
        step_size: float or np.ndarray of float
            the learning rate (also called step_size or aggressivity)
            The default is 0.5.
        backtrack: bool
            if true, bracktracking method will be used for line search.
            Note that is recommended to set step_size to 1.0
        eta: float
            the tolerance criteria on the convergence of the gradient.
            The default is 1e-5
        max_iter: int
            maximum step of iteration
            The default is 100.
        h: np.ndarray of float
            value of small steps for differentiation one for each argument
            for e.g. if args of len 3 h must be np.ndarray of len 3 or a list.
        history: bool
            If True will store and return the history of the optimization process
            The default is False.
        beta: float or np.ndarray
            factor scaling step_size
            is often chosen to be between 0.1 (which corresponds
            to a very crude search) and 0.8 (which corresponds
            to a less crude search).
            See page 466 of "Convex Optimization" by Stephen Boyd & Lieven 
            The default is 0.5
        alpha: float or np.ndarray
            a parameter of the backtracking
            Values of alpha typically range between 0.01 and 0.3
            The default is 0.1
        """
        args = np.asarray(args)
        traj = []
        delta_x = np.zeros_like(args) # do not need to be modified (use for momentum GD)

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

        # function for calculating the direction
        direction = lambda grad : self.steepest_direction(grad, norm=norm, H=P)

        # optimisation procedure
        for i in range(max_iter):
            #record history
            if history:
                traj.append(*args)
            # direction search
            grad = jac(*args)
            step = direction(grad)
            # line seach using backtracking method
            if backtrack:
                step_size = self.backtracking(*args, grad=grad, deltax=step, step_size=step_size,
                                                beta=beta_bt, alpha=alpha_bt)
    
            # criteria on the gradient to not waste resources when stuck in a flat region
            if np.linalg.norm(grad) >= eta:
                    if moment_gd:
                        x_0 = args
                        # heavy ball method Polyak's approach
                        args += step_size * step + zeta_momentum * delta_x
                        delta_x = args - x_0
                    else:
                        args += step_size * step
            else:
                print(f"Converged at step {i}\n")
                break

        return args, self.func(*args), np.asarray(traj, dtype="float")
