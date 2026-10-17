from pathlib import Path

import numpy as np


def load_cosmic_chronometers(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load redshift, H(z), and 1-sigma uncertainty columns."""
    data = np.loadtxt(path, dtype=float)
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError("Cosmic-chronometer data must have three columns: z, H(z), sigma_H.")
    return data[:, 0], data[:, 1], data[:, 2]
