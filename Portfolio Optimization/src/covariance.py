"""
Covariance estimators for the algorithm.

ledoit_wolf_cc: Ledoit & Wolf (2004), "Honey, I Shrunk the Sample Covariance
Matrix", Journal of Portfolio Management 30(4). Shrinks the sample covariance toward a constant-correlation target
(each stock keeps its own variance; every pair gets the average correlation),
with the shrinkage intensity estimated from the data.
"""
from typing import Tuple

import numpy as np
import pandas as pd


def sample_covariance(returns: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    return returns.cov(), 0.0


def ledoit_wolf_constant_correlation(returns: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """
    Args:
        returns: T x N periodic returns (rows = dates, columns = tickers), no NaN.

    Returns:
        (shrunk covariance in the same units as the input, shrinkage intensity in [0, 1])
    """
    x = returns.to_numpy(dtype=float)
    t, n = x.shape
    if n < 2:
        return returns.cov(), 0.0

    y = x - x.mean(axis=0)
    sample = y.T @ y / t
    var = np.diag(sample).copy()
    sd = np.sqrt(var)

    corr = sample / np.outer(sd, sd)
    r_bar = (corr.sum() - n) / (n * (n - 1))
    target = r_bar * np.outer(sd, sd)
    np.fill_diagonal(target, var)

    # pi: sum of asymptotic variances of the sample covariance entries
    y2 = y ** 2
    pi_mat = y2.T @ y2 / t - sample ** 2
    pi_hat = pi_mat.sum()

    # rho: sum of asymptotic covariances between target and sample entries
    theta = (y ** 3).T @ y / t - var[:, None] * sample
    np.fill_diagonal(theta, 0.0)
    rho_hat = np.trace(pi_mat) + r_bar * (np.outer(1 / sd, sd) * theta).sum()

    # gamma: misspecification of the target
    gamma_hat = ((target - sample) ** 2).sum()

    kappa = (pi_hat - rho_hat) / gamma_hat if gamma_hat > 0 else 0.0
    delta = float(np.clip(kappa / t, 0.0, 1.0))

    shrunk = delta * target + (1 - delta) * sample
    return pd.DataFrame(shrunk, index=returns.columns, columns=returns.columns), delta


ESTIMATORS = {
    "sample": sample_covariance,
    "ledoit_wolf_cc": ledoit_wolf_constant_correlation,
}


def estimate_covariance(returns: pd.DataFrame, method: str) -> Tuple[pd.DataFrame, float]:
    if method not in ESTIMATORS:
        raise ValueError(f"Unknown covariance estimator {method!r}; choose from {sorted(ESTIMATORS)}")
    return ESTIMATORS[method](returns)
