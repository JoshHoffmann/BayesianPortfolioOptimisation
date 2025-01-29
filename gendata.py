import numpy as np

'''This module handles generating synthetic returns data for use in each simulation. Asset returns are currently drawn 
from a multivariate normal distribution with a randomly generated covariance structure.'''
def ReturnsData(N:int,T:int)->np.ndarray:
    '''Function handling the generating of correlated asset returns.
    --- Paramaters ---
    N - Number of assets
    T - Number of time steps
    '''
    # Randomly generate asset return means and volatilities
    mu = np.random.uniform(-1, 1, N)
    sigma = np.random.uniform(0, 1, N)


    # Generate covariance structure
    rho = 2 * np.random.beta(2.5, 5, (N, N)) - 1 # rho initially drawn from a beta distribution
    rho = (rho + rho.T) / 2 # symmetrise
    np.fill_diagonal(rho, 1) # Ensure unit diagonal

    # Ensure that correlation matrix is positive semi-definite
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.maximum(eigvals, 0)
    rho = eigvecs @ np.diag(eigvals) @ eigvecs.T

    COV = np.diag(sigma) @ rho @ np.diag(sigma)
    r = np.random.multivariate_normal(mu, COV, T)

    return r