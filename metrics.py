import pandas as pd
import numpy as np


def PortflioReturns(r: pd.DataFrame, w: pd.DataFrame) -> pd.DataFrame:
    portfolio = r.multiply(w.shift(1), axis=1).sum(axis=1).apply(lambda x: np.round(x, 4))
    return portfolio


def CumulativeReturns(r: pd.DataFrame) -> pd.DataFrame:
    cumulative_returns = (1 + r).cumprod() - 1

    return cumulative_returns


def Sharpe(r: pd.DataFrame) -> float:
    sharpe_ratio = np.round(r.mean() / r.std(), 2)

    return sharpe_ratio


def AvgWeightMovement(w: pd.DataFrame) -> float:
    weight_diff = w.diff().dropna()
    avg_movement = np.round((weight_diff ** 2).sum(axis=1).mean(), 4)

    return avg_movement


def WeightDisplacement(w: pd.DataFrame) -> float:
    w0 = w.iloc[0].values
    wf = w.iloc[-1].values

    displacement = np.round(np.sqrt(np.sum((wf - w0) ** 2)), 4)

    return displacement


def WeightStepDisplacement(w: pd.DataFrame) -> pd.DataFrame:
    weight_diff = w.diff().dropna()
    disp = np.round(np.sqrt((weight_diff ** 2).sum(axis=1)), 2)
    return disp
