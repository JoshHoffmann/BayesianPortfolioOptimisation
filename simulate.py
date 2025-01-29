import gendata
import numpy as np
import pandas as pd
import updating
import optimiser
import metrics
import plotting
import matplotlib.pyplot as plt

''' This module contains the BayesianOptimiser class, used to initialise and run a given number of simulations of the 
optimisation procedure for a number of assets, time steps and objective.'''

class BayesianOptimiser:
    '''BayesianOptimiser class. Initialises simulation for specified number of assets, time steps and objective function
    --- Parameters---
    N - Number of assets in the generated data
    T - Number of time steps in asset returns time series
    objective - Objective function to optimise. Defaults to 'sharpe' in which portfolio weights are optimised to
    maximise portfolio sharpe ratio
    objective_params - additional parameters needed for certain objective functions. Defaults to None
    longonly - Controls whether shorting of assets is allowed. i.e. if negative weights are allowed. Defaults to True
    in which only long positions (positive weights) are allowed

    Does not return
    '''
    def __init__(self,N:int,T:int,objective:str = 'sharpe',objective_params=None,longonly:bool=True):
        self.N = N
        self.T = T
        self.objective = objective
        self.objective_params = objective_params or {}
        self.longonly = longonly
        self.objective_results = np.array([]) # Stores collected portfolio objective results from each sim
        self.benchmark_objectve_results = np.array([]) # Stores collected benchmark objective results from each sim
        self.weight_steps = []


    def RunSimulation(self,N_sim:int=1):
        '''This method is called to run the specified simulation. It takes a single parameter N_sim specifying the
         number of simulations to run.
         Does not return
        '''
        print('Running Simulation...')
        for _ in range(N_sim): # Loop through each simulation
            # Get objective metrics for optimised portfolio and benchmark hold portfolio
            objective, benchmark_objective = self.GetResults()

            # Store portfolio and benchmark results
            self.objective_results = np.append(self.objective_results,[objective])
            self.benchmark_objectve_results = np.append(self.benchmark_objectve_results,benchmark_objective)

            print('{}% complete'.format(np.round(100* _/N_sim,4))) # % completion of all sims

        # Call plotting functions to compare portfolio and benchmark objectives
        plotting.PlotObjectiveDiffDistribution(self.objective_results,self.benchmark_objectve_results,self.objective,
                                               self.objective_params)
        plotting.PlotObjectiveComparison(self.objective_results,self.benchmark_objectve_results,self.objective, self.objective_params)
        plotting.PlotWeightSteps(self.weight_steps)
        plt.show()


    def GetResults(self):
        '''This method handles running each individual simulation. Optimised weights are obtained at each time step
        and the objective metrics are obtained once the simulation is complete.

        returns portfolio and benchmark objective metrics
        '''

        r = gendata.ReturnsData(self.N, self.T) # Get simulated returns time series

        w = np.round(np.ones(self.N) / self.N, 2) # Define initial equally weighted portfolio

        # Parameters for initial returns prior. Returns are assumed to be normally distributed.
        mu_prior = np.zeros(self.N)
        sigma_prior = np.eye(self.N) * 0.5

        # Parameters for inital covariance structure prior. Covariance is assumed to be inverse-Wishart distributed.
        nu = self.N + 2
        psi = np.eye(self.N)

        # Define returns and weights time series data frames for later use
        returns_series = pd.DataFrame(r, columns=['asset {}'.format(i) for i in range(1, self.N + 1)])
        weights_series = pd.DataFrame().reindex_like(returns_series)
        weights_series.iloc[0] = w
        w_hold = weights_series.ffill() # Weights of benchmark portfolio. Equally weighted and constant.

        # run simulation loop, iterating over each time step.
        for t in range(1, self.T):
            rt = r[:t]  # Observed returns up to time t

            # Get parameters from new posterior distributions
            posterior_params = updating.UpdatePosteriors(rt, mu_prior, sigma_prior, psi, nu)

            # Replace current prior params with new posterior params for next time step
            mu_prior = posterior_params['mu']
            sigma_prior = posterior_params['sigma']
            Cov_post_mean = posterior_params['Cov']
            psi = posterior_params['psi']
            nu = posterior_params['nu']

            # Optimise portfolio weights subject to objective
            optimiser_inst = optimiser.OptimiseWeights(w, mu_prior, Cov_post_mean,longonly=self.longonly)
            # Set weights for next time step as optimised weights
            w = optimiser_inst.optimise(objective=self.objective, **self.objective_params)
            weights_series.iloc[t] = w # current weights to time series

        # Get metrics from observed returns and optimised portfolio weights and benchmark portfolio weights
        results = self.GetMetrics(returns_series,weights_series,w_hold)

        # Obtain metrics depending on objective function
        if self.objective == 'sharpe':
            return results['Portfolio Sharpe'], results['Hold Sharpe']
        if self.objective == 'min_variance':
            return results['Portfolio Variance'], results['Hold Portfolio Variance']
        if self.objective == 'max_return':
            return results['Portfolio Expected Return'], results['Hold Portfolio Expected Return']
        if self.objective == 'risk_bounded_return':
            return results['Portfolio Expected Return'], results['Hold Portfolio Expected Return']

    def GetMetrics(self,r:pd.DataFrame,w:pd.DataFrame,w_hold:pd.DataFrame)->dict:
        '''This method handles calculating metrics given observed returns data and portfolio weights.
        --- Parameters ---
        r - observed returns time series
        w - optimised weights time series
        w_hold - benchmark portfolio weights

        returns dictionary of metrics
        '''
        # Get Returns
        portfolio_returns = metrics.PortflioReturns(r, w)
        hold_portfolio_returns = metrics.PortflioReturns(r, w_hold)

        # Get Expected Returns
        portfolio_expected_return = portfolio_returns.mean()
        hold_portfolio_expected_return = hold_portfolio_returns.mean()

        # Get Variance/Risk
        portfolio_variance = portfolio_returns.std()**2
        hold_portfolio_variance = hold_portfolio_returns.std()**2

        # Get Cumulative Returns
        portfolio_cumulative_returns = metrics.CumulativeReturns(portfolio_returns)
        hold_portfolio_cumulative_returns = metrics.CumulativeReturns(hold_portfolio_returns)

        # Get Sharpe
        sharpe = metrics.Sharpe(portfolio_returns)
        hold_sharpe = metrics.Sharpe(hold_portfolio_returns)
        #print('Sharpe = ', sharpe)
        #print('Hold Sharpe = ', hold_sharpe)
        avg_weight_movement = metrics.AvgWeightMovement(w)
        #print('Average Weight Movement = ', avg_weight_movement)
        weight_displacement = metrics.WeightDisplacement(w)
        #print('Weight Displacement = ', weight_displacement)
        step_weight_displacement = metrics.WeightStepDisplacement(w)
        self.weight_steps.append(step_weight_displacement)

        result_metrics = {'Portfolio Returns':portfolio_returns,'Hold Portfolio Returns':hold_portfolio_returns,
                          'Portfolio Expected Return': portfolio_expected_return,
                          'Hold Portfolio Expected Return':hold_portfolio_expected_return,
                          'Portfolio Variance':portfolio_variance,
                          'Hold Portfolio Variance':hold_portfolio_variance,
                          'Portfolio Cumulative Returns':portfolio_cumulative_returns,
                          'Hold Portfolio Cumulative Returns':hold_portfolio_cumulative_returns,
                          'Portfolio Sharpe':sharpe,'Hold Sharpe':hold_sharpe,
                          'Avg Weight Movement': avg_weight_movement, 'Weight Displacement': weight_displacement}

        return result_metrics
