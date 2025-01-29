import numpy as np

'''This module handles Bayesian inference updating of posteriors for the mean asset returns and covariance distributions.
Mean asset returns are assumed to be normally distributed. 
Covariance structure is assumed to follow an inverse-Wishart distribution.
'''


def UpdatePosteriors(r:np.ndarray,mu:np.ndarray,sigma:np.ndarray,psi:np.ndarray,nu:int)->dict:
    '''This function handles updating of posterior ditributions given data observed up to time t.
     --- Parameters ---
     r -  returns up to current time
     mu - mean prior for mean returns normal distribution
     sigma - covariance matrix prior for mean returns normal distribution
     psi - scale matrix prior for covariance inv-Wishart distribution
     nu - degree of freedom prior for covariance inv-Wishart distribution

     returns dictionary of parameters for updated posterior distributions
     '''
    t = r.shape[0] # Current time step/No. of observations
    N = r.shape[1] # Number of assets
    # Observed mean returns and mean returns covariance matrix
    mu_r = np.mean(r, axis=0)
    sigma_r = np.cov(r, rowvar=False)

    def UpdateCovarianceDistribution(psi:np.ndarray,nu:int,sigma_r:np.ndarray):
        '''This function specifically handles updating the covariance distribution given the priors psi and nu and
        observed covariance sigma_r

        --- returns ---
        Cov_post_mean - mean of posterior inverse-Wishart distribution
        psi - scale matrix of posterior distribution
        nu - degrees of freedom of posterior distribution
        '''

        psi = psi + (t - 1) * sigma_r
        nu = nu + t - 1
        Cov_post_mean = psi / (nu - N - 1)

        return Cov_post_mean, psi, nu

    def UpdateMeanDistribution(mu:np.ndarray,sigma:np.ndarray,Cov:np.ndarray):
        '''this function specifically handles updating the distribution of asset mean returns, given priors mu, sigma
        and estimated covariance.

        --- returns ---
        mu_post - mean of posterior distribution of mean asset returns
        sigma_post - covariance matrix of posterior distribution of mean asset returns.
        '''

        sigma_post = np.linalg.inv(np.linalg.inv(sigma) + t * np.linalg.inv(Cov))
        mu_post = sigma_post @ (np.linalg.inv(sigma) @ mu + t * np.linalg.inv(Cov) @ mu_r)
        return mu_post, sigma_post

    # Perform updating of parameters
    Cov_post, psi_post,nu_post = UpdateCovarianceDistribution(psi,nu,sigma_r) # Update returns covariance structure
    mu_post, sigma_post = UpdateMeanDistribution(mu,sigma,Cov_post) # Update mean returns

    # New params are stored in a dict to be returned by function
    params = {'mu':mu_post,'sigma':sigma_post,'Cov':Cov_post,'psi':psi_post,'nu':nu_post}

    return params

