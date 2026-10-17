import numpy as np


def _safe_hubble(inside: np.ndarray, H0: float) -> np.ndarray:
    inside = np.asarray(inside, dtype=float)
    if np.any(inside <= 0.0):
        return np.full_like(inside, np.nan, dtype=float)
    return H0 * np.sqrt(inside)


def h_flat_lcdm(z, H0: float, omega_m: float):
    z = np.asarray(z, dtype=float)
    omega_lambda = 1.0 - omega_m
    return _safe_hubble(omega_m * (1.0 + z) ** 3 + omega_lambda, H0)


def h_nonflat_lcdm(z, H0: float, omega_m: float, omega_k: float):
    z = np.asarray(z, dtype=float)
    omega_lambda = 1.0 - omega_m - omega_k
    inside = omega_m * (1.0 + z) ** 3 + omega_k * (1.0 + z) ** 2 + omega_lambda
    return _safe_hubble(inside, H0)


def h_flat_xcdm(z, H0: float, omega_m: float, w: float):
    z = np.asarray(z, dtype=float)
    omega_x = 1.0 - omega_m
    inside = omega_m * (1.0 + z) ** 3 + omega_x * (1.0 + z) ** (3.0 * (1.0 + w))
    return _safe_hubble(inside, H0)


def h_nonflat_xcdm(z, H0: float, omega_m: float, omega_k: float, w: float):
    z = np.asarray(z, dtype=float)
    omega_x = 1.0 - omega_m - omega_k
    inside = (
        omega_m * (1.0 + z) ** 3
        + omega_k * (1.0 + z) ** 2
        + omega_x * (1.0 + z) ** (3.0 * (1.0 + w))
    )
    return _safe_hubble(inside, H0)
