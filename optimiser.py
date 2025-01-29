import numpy as np
from scipy.optimize import minimize

'''This module contains the OptimiseWeights class which handles optimising portfolio weights given a specific objective 
function and (additional) constraints.'''


class OptimiseWeights:
    '''OptimiseWeights class initiates optimisation of current portfolio weights to be used in the next time step,
    subject to an objective function and constraints.

    --- Paramaters --
    w_0 - initial weights from the previous time step.
    mu - current estimate of mean asset returns i.e. mean of posterior distribution of mean returns
    COV - current estimate of asset covariance structure i.e. mean of posterior inv-Wishart distribution
    longonly - controls whether only long positions or shorting is allowed. Defaults to True in which all portfolio
    weights must be positive
    '''

    def __init__(self, w0:np.ndarray, mu:np.ndarray, COV:np.ndarray, longonly:bool=True):
        self.w0 = w0
        self.mu = mu
        self.COV = COV
        self.N = mu.shape[0]
        self.longonly = longonly

    def weight_constraint(self, w:np.ndarray):
        '''Weight constraint function. Weights must sum to 1.
        returns constraint for minimize'''
        return np.sum(w) - 1

    def WeightBound(self):
        '''This function selects appropriate bounds on weights depending on longonly.
        returns list of bounds for minimize.'''
        if self.longonly == True:
            return [(0, 1) for _ in range(self.N)]
        else:
            return [(-1, 1) for _ in range(self.N)]

    def optimise(self, objective:str, **objective_params)->np.ndarray:
        '''This method handles optimising weights with respect to the objective function.

        --- Paramaters --
        objective - string specifying which objective to optimise weights for. Currently implemented are:
            sharpe - maximises portfolio sharpe ratio (assuming zero risk-free rate)
            min_variance - minimises portfolio variance
            max_return - maximises portfolio expected returns
            risk_bounded_return - maximises portfolio expected return subject to portfolio variance being less than or
            equal to a specified threshold passed objective paramater
        objective_params - dict containing objective parameters should for relevant objective functions.
        Currently, accepts:
            risk_bounded_return - objective_params = {'max_risk':threshold}

        returns optimised weights

        '''
        OBJECTIVE_FUNCTIONS = {
            "sharpe": self.Sharpe,
            "min_variance": self.MinVariance,
            "max_return": self.ExpectedReturn,
            "risk_bounded_return": self.RiskBoundedReturn,
        }
        # Check to see ifg passed objective function parameter is valid. If not display options
        if objective not in OBJECTIVE_FUNCTIONS:
            raise ValueError(
                f"Invalid objective '{objective}'. Valid options are: {list(OBJECTIVE_FUNCTIONS.keys())}")

        # Retrieve objective function and additional constraints
        if objective == "risk_bounded_return":
            # risk_bounded_return requires additional constraint for portfolio risk.
            objective_func, objective_constraints = OBJECTIVE_FUNCTIONS[objective](**objective_params)

        else:
            # Other objective functions require no additional constraints
            objective_func = OBJECTIVE_FUNCTIONS[objective]
            objective_constraints = None

        # Retrieve weight bound
        weight_bound = self.WeightBound()
        # Select additional objective constraints
        objective_constraints = objective_constraints or []
        # Add additional constraints to default weight constraint
        constraints = [{'type': 'eq', 'fun': self.weight_constraint}] + objective_constraints

        # Run optimisation of weights with scipy.optimise.minimize
        opt_weights = minimize(objective_func, self.w0, bounds=weight_bound, constraints=constraints).x

        return opt_weights

    def Sharpe(self, w:np.ndarray)->float:
        '''Sharpe ratio objective function. Given portfolio weights up to observed time w, calculates Sharpe ratio.
        returns -sharpe'''

        expected_return = w.T @ self.mu
        risk = np.sqrt(w.T @ self.COV @ w)
        sharpe = expected_return / risk
        return -sharpe # Negative is needed to maximise

    def MinVariance(self, w:np.ndarray):
        '''Min variance objective function. Given portfolio weights up to observed time w, calculates portfolio variance.
        returns variance'''

        return w.T @ self.COV @ w

    def ExpectedReturn(self, w:np.ndarray)->float:
        '''Max return objective function. Given portfolio weights up to observed time w, calculates portfolio expected
         return. returns -expected return'''
        return -w.T @ self.mu  # Negative is needed to maximise

    def RiskBoundedReturn(self, max_risk:float):
        '''This function gets the max return objective function and additional risk constraint function for
        risk_bounded_return objective. Takes the risk threshold as parameter.
        returns expected return objective function and risk inequality constraint '''
        def expected_return_obj_func(w):
            return -w.T @ self.mu  # Negative for maximization

        def risk_constraint(w):
            return max_risk - np.sqrt(w.T @ self.COV @ w) # Portfolio risk must be less than or equal to threshold

        constraints = [{'type': 'ineq', 'fun': risk_constraint}] # Package constraint for minimize
        return expected_return_obj_func, constraints
