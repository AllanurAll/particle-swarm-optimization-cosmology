from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class PSOResult:
    best_position: np.ndarray
    best_score: float
    best_score_history: np.ndarray
    mean_score_history: np.ndarray
    global_best_position_history: np.ndarray
    initial_positions: np.ndarray
    final_positions: np.ndarray
    final_scores: np.ndarray
    positions_history: np.ndarray
    personal_best_history: np.ndarray


def run_pso(
    objective: Callable[[np.ndarray], float],
    bounds,
    *,
    n_particles: int,
    n_iter: int,
    inertia: float = 0.72,
    cognitive: float = 1.49,
    social: float = 1.49,
    seed: int = 42,
) -> PSOResult:
    """Minimize an objective with the canonical particle-swarm update."""
    rng = np.random.default_rng(seed)
    bounds = np.asarray(bounds, dtype=float)
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    ndim = len(bounds)

    positions = rng.uniform(lower, upper, size=(n_particles, ndim))
    velocity_scale = 0.15 * (upper - lower)
    velocities = rng.uniform(-velocity_scale, velocity_scale, size=(n_particles, ndim))

    initial_positions = positions.copy()
    scores = np.array([objective(p) for p in positions], dtype=float)
    personal_best_positions = positions.copy()
    personal_best_scores = scores.copy()

    best_index = int(np.argmin(personal_best_scores))
    global_best_position = personal_best_positions[best_index].copy()
    global_best_score = float(personal_best_scores[best_index])

    best_score_history = []
    mean_score_history = []
    global_best_position_history = []
    positions_history = [positions.copy()]
    personal_best_history = [personal_best_positions.copy()]

    for _ in range(n_iter):
        r1 = rng.random(size=(n_particles, ndim))
        r2 = rng.random(size=(n_particles, ndim))

        velocities = (
            inertia * velocities
            + cognitive * r1 * (personal_best_positions - positions)
            + social * r2 * (global_best_position - positions)
        )
        positions = np.clip(positions + velocities, lower, upper)
        scores = np.array([objective(p) for p in positions], dtype=float)

        improved = scores < personal_best_scores
        personal_best_positions[improved] = positions[improved]
        personal_best_scores[improved] = scores[improved]

        best_index = int(np.argmin(personal_best_scores))
        if personal_best_scores[best_index] < global_best_score:
            global_best_score = float(personal_best_scores[best_index])
            global_best_position = personal_best_positions[best_index].copy()

        best_score_history.append(global_best_score)
        finite = scores[np.isfinite(scores)]
        mean_score_history.append(float(np.mean(finite)))
        global_best_position_history.append(global_best_position.copy())
        positions_history.append(positions.copy())
        personal_best_history.append(personal_best_positions.copy())

    return PSOResult(
        best_position=global_best_position,
        best_score=global_best_score,
        best_score_history=np.asarray(best_score_history),
        mean_score_history=np.asarray(mean_score_history),
        global_best_position_history=np.asarray(global_best_position_history),
        initial_positions=initial_positions,
        final_positions=positions.copy(),
        final_scores=scores.copy(),
        positions_history=np.asarray(positions_history),
        personal_best_history=np.asarray(personal_best_history),
    )
