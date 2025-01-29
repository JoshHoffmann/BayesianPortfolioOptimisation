'''
Bayesian Portfolio Optimisation
##############################

This module provides simulation of Bayesian Portfolio optimisation.
Usage of the main script is described below

--- Module Import ---
simulate: Contains the BayesianOptimiser class used to run simulations and optimise portfolio weights subject to
objective

--- Parameters ---
    N - Number of assets to include in the portfolio.
    T - Number of time periods for which returns are simulated.

--- Simulation Overview ---
The BayesianOptimiser class supports portfolio optimisation with the following objectives:
    'sharpe' -  Maximise the Sharpe ratio.
    'max_return' -  Maximise portfolio returns.
    'min_variance' - Minimise portfolio variance.
    'risk_bounded_return' - Maximise returns subject to a risk constraint, portfolio_variance <= threshold
        - For this objective, additional threshold parameter 'max_risk` must be specified via the
        'objective_params' argument.



--- Simulation Examples ---

    Maximizing Sharpe Ratio

    sim1 = simulate.BayesianOptimiser(N, T, objective='sharpe').RunSimulation(100)

        - This runs 100 simulations and optimises weights maximise the Sharpe ratio in each simulation

     Maximizing Portfolio Returns

    sim2 = simulate.BayesianOptimiser(N, T, objective='max_return').RunSimulation(100)

        - This runs 100 simulations and optimises weights maximise the portfolio expected return in each simulation

    Minimizing Portfolio Variance

    sim3 = simulate.BayesianOptimiser(N, T, objective='min_variance').RunSimulation(100)

        - This runs 100 simulations and optimises weights minimise the portfolio variance in each simulation.

    Risk-Bounded Return Optimisation

    sim4 = simulate.BayesianOptimiser(N, T, objective='risk_bounded_return',
                                  objective_params={'max_risk': 0.03}).RunSimulation(100)

        - This runs 100 simulations and optimises weights maximise the portfolio expected return in each simulation,
         with a constraint that portfolio risk does not exceed 0.03.


- The RunSimulation method executes the optimisation process for the specified number of iterations.
- Additional parameters for objectives can be passed through the objective_params dictionary.

--- Outputs ---
Plots comparing objective metrics for simulated portfolios and a benchmark equally weighted constant portfolio will
    be outputted.

'''


import simulate


N = 5  # Number of assets
T = 50


sim1 = simulate.BayesianOptimiser(N,T,objective='sharpe').RunSimulation(300)
sim2 = simulate.BayesianOptimiser(N,T,objective='max_return').RunSimulation(300)
sim3 = simulate.BayesianOptimiser(N,T,objective='min_variance').RunSimulation(300)
sim4 = simulate.BayesianOptimiser(N,T,objective='risk_bounded_return', objective_params={'max_risk':0.03}).RunSimulation(100)


