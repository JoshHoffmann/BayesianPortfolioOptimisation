import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns



def PlotObjectiveDiffDistribution(portfolio_obj,benchmark_obj,obj, obj_params):
    print(obj_params)

    obj_diff  = portfolio_obj - benchmark_obj
    plt.figure(figsize=(8, 6))
    sns.histplot(obj_diff, kde=True, bins=50, color='skyblue', edgecolor='black')

    if obj == 'sharpe':
        plt.title('Distribution of Sharpe Difference')
        plt.xlabel(r'$S_{\text{portfolio}} - S_{\text{benchmark}}$')
    elif obj == 'min_variance':
        plt.title('Distribution of Variance Difference')
        plt.xlabel(r'$\sigma^2_{\text{portfolio}} - \sigma^2_{\text{benchmark}}$')
    elif obj == 'max_return':
        plt.title('Distribution of Expected Return Difference')
        plt.xlabel(r'$E[R_{\text{portfolio}}] - E[R_{\text{benchmark}}]$')
    elif obj == 'risk_bounded_return':
        plt.title(r'Distribution of Expected Return Difference With Risk Constraint $\sigma^2_{portfolio} \leq $' + '{}'.format(obj_params['max_risk']))
        plt.xlabel(r'$E[R_{\text{portfolio}}] - E[R_{\text{benchmark}}]$')


    plt.ylabel('Frequency')
    plt.grid(True)


def PlotObjectiveComparison(portfolio_obj,benchmark_obj,obj,obj_params):
    plt.figure()


    x = benchmark_obj
    y = portfolio_obj
    sns.scatterplot(x = x, y=y)
    if obj == 'sharpe':
        plt.title('Portfolio vs Benchmark Sharpe')
        plt.xlabel(r'$S_{\text{benchmark}}$')
        plt.ylabel(r'$S_{\text{portfolio}}$')
        label = r'$S_{\text{benchmark}} = S_{\text{portfolio}} $'
    elif obj == 'min_variance':
        plt.title('Portfolio vs Benchmark Risk')
        plt.xlabel(r'$\sigma^2_{\text{benchmark}}$')
        plt.ylabel(r'$\sigma^2_{\text{portfolio}}$')
        label = r'$\sigma^2_{\text{benchmark}} = \sigma^2_{\text{portfolio}} $'
    elif obj == 'max_return':
        plt.title('Portfolio vs Benchmark Expected Return')
        plt.xlabel(r'$E[R_{\text{benchmark}}]$')
        plt.ylabel(r'$E[R_{\text{portfolio}}]$')
        label = r'$E[R_{\text{benchmark}}] = E[R_{\text{portfolio}}] $'
    elif obj == 'risk_bounded_return':
        plt.title(r'Portfolio vs Benchmark Expected Return With Risk Constraint $\sigma^2_{\text{portfolio}} \leq $' +
                  '{}'.format(obj_params['max_risk']))
        plt.xlabel(r'$E[R_{\text{benchmark}}]$')
        plt.ylabel(r'$E[R_{\text{portfolio}}]$')
        label = r'$E[R_{\text{benchmark}}] = E[R_{\text{portfolio}}] $'
    plt.plot(x, x, color='red', linestyle='--', label = label)

    plt.legend()


def PlotWeightSteps(w_master: list[pd.Series], stabilization_threshold: float = 0.01):

    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap("viridis")
    num_sims = len(w_master)
    colors = [cmap(i / num_sims) for i in range(num_sims)]

    stabilization_times = []

    # Plot weight disp for each sim
    for sim, (w, color) in enumerate(zip(w_master, colors)):
        plt.plot(w.index, w.values, label=f'Sim {sim + 1}' if num_sims <= 10 else None,
                 color=color, alpha=0.5 if num_sims > 10 else 1.0)

        # Compute stabilization times
        stable_indices = np.where(w.values < stabilization_threshold)[0]
        if len(stable_indices) > 0:
            stabilization_times.append(w.index[stable_indices[0]])

    # Compute and plot the average stabilization time
    if stabilization_times:
        avg_stabilization_time = np.mean(stabilization_times)
        plt.axvline(avg_stabilization_time, color='red', linestyle='--',
                    label=f'Avg Stabilization Time: {avg_stabilization_time:.2f}')


        plt.annotate(f'Avg Stabilization Time\n(t = {avg_stabilization_time:.2f})',
                     xy=(avg_stabilization_time, plt.ylim()[1] * 0.8),
                     xytext=(avg_stabilization_time + 5, plt.ylim()[1] * 0.9),
                     arrowprops=dict(arrowstyle='->', color='red'))


    plt.title(r'Weight Space Displacement $|\mathbf{w}_{t+1} - \mathbf{w}_t|$', fontsize=14)
    plt.xlabel('Time Step $t$', fontsize=12)
    plt.ylabel(r'$|\mathbf{w_{t+1}} - \mathbf{w_t}|$', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Handle legend placement
    if num_sims <= 10:
        plt.legend()
    elif num_sims <= 50:
        plt.legend([f'Sim {i + 1}' for i in range(min(num_sims, 10))] + ["..."], loc='upper right', fontsize=10)
    else:
        plt.legend(["Multiple Simulations"], loc='upper right', fontsize=10)

    plt.show()

