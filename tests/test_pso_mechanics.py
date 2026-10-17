import numpy as np

from pso_cosmology.data import load_cosmic_chronometers
from pso_cosmology.models import h_flat_lcdm
from pso_cosmology.pso import run_pso
from pso_cosmology.statistics import chi_square


def _objective_factory():
    z, observed, sigma = load_cosmic_chronometers("data/cosmic_chronometers_31.txt")

    def objective(theta):
        H0, omega_m = theta
        if not (50.0 <= H0 <= 100.0 and 0.1 <= omega_m <= 0.9):
            return np.inf
        return chi_square(h_flat_lcdm(z, H0, omega_m), observed, sigma)

    return objective


def test_five_particle_presentation_scores():
    objective = _objective_factory()
    initial = np.array([
        [60.0, 0.20],
        [68.0, 0.30],
        [75.0, 0.50],
        [85.0, 0.20],
        [55.0, 0.70],
    ])
    updated = np.array([
        [70.976, 0.3187],
        [68.144, 0.3144],
        [68.022, 0.2328],
        [60.763, 0.2966],
        [69.279, 0.3876],
    ])

    initial_scores = np.array([objective(p) for p in initial])
    updated_scores = np.array([objective(p) for p in updated])

    np.testing.assert_allclose(initial_scores, [88.92, 15.05, 116.90, 61.34, 32.56], atol=0.01)
    np.testing.assert_allclose(updated_scores, [17.67, 14.53, 24.05, 42.84, 22.73], atol=0.01)


def test_global_best_history_is_nonincreasing():
    objective = _objective_factory()
    result = run_pso(
        objective,
        [(50.0, 100.0), (0.1, 0.9)],
        n_particles=30,
        n_iter=40,
        inertia=0.72,
        cognitive=1.49,
        social=1.49,
        seed=42,
    )
    assert np.all(np.diff(result.best_score_history) <= 1e-12)
