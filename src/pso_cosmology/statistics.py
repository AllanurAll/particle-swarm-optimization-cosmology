from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModelStatistics:
    chi2: float
    dof: int
    reduced_chi2: float
    aic: float
    bic: float


def chi_square(theory, observed, sigma) -> float:
    theory = np.asarray(theory, dtype=float)
    observed = np.asarray(observed, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if np.any(~np.isfinite(theory)):
        return np.inf
    residual = (theory - observed) / sigma
    return float(np.sum(residual**2))


def model_statistics(chi2_min: float, n_data: int, n_parameters: int) -> ModelStatistics:
    dof = n_data - n_parameters
    return ModelStatistics(
        chi2=float(chi2_min),
        dof=dof,
        reduced_chi2=float(chi2_min / dof),
        aic=float(chi2_min + 2 * n_parameters),
        bic=float(chi2_min + n_parameters * np.log(n_data)),
    )
