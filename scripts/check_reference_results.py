#!/usr/bin/env python3
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pso_cosmology import (
    chi_square,
    h_flat_lcdm,
    h_nonflat_lcdm,
    h_flat_xcdm,
    h_nonflat_xcdm,
    load_cosmic_chronometers,
    run_pso,
)

z, H, s = load_cosmic_chronometers(ROOT / "data" / "cosmic_chronometers_31.txt")


def flat_obj(x):
    H0, om = x
    if not (50 <= H0 <= 100 and 0.1 <= om <= 0.9): return np.inf
    return chi_square(h_flat_lcdm(z,H0,om),H,s)


def nonflat_obj(x):
    H0, om, ok = x
    ol=1-om-ok
    if not (50 <= H0 <=100 and 0.01 <=om<=1 and -0.5<=ok<=0.5 and 0<=ol<=2): return np.inf
    return chi_square(h_nonflat_lcdm(z,H0,om,ok),H,s)


def flat_x_obj(x):
    H0,om,w=x
    ox=1-om
    if not (50<=H0<=100 and 0.01<=om<=0.99 and -2.5<=w<=0 and 0<=ox<=2): return np.inf
    return chi_square(h_flat_xcdm(z,H0,om,w),H,s)


def nonflat_x_obj(x):
    H0,om,ok,w=x
    ox=1-om-ok
    if not (50<=H0<=100 and 0.01<=om<=1 and -0.5<=ok<=0.5 and -2.5<=w<=0 and 0<=ox<=2): return np.inf
    return chi_square(h_nonflat_xcdm(z,H0,om,ok,w),H,s)

checks = [
    ("flat LCDM", run_pso(flat_obj,[(50,100),(0.1,0.9)],n_particles=100,n_iter=300,seed=42), np.array([68.149916,0.319555]),14.499986),
    ("non-flat LCDM", run_pso(nonflat_obj,[(50,100),(0.01,1),(-0.5,0.5)],n_particles=200,n_iter=500,seed=42), np.array([68.998731,0.359749,-0.125924]),14.461770),
    ("flat XCDM", run_pso(flat_x_obj,[(50,100),(0.01,0.99),(-2.5,0)],n_particles=250,n_iter=700,seed=43), np.array([70.820175,0.322310,-1.249724]),14.360644),
    ("non-flat XCDM", run_pso(nonflat_x_obj,[(50,100),(0.01,1),(-0.5,0.5),(-2.5,0)],n_particles=300,n_iter=900,seed=44), np.array([75.362696,0.169184,0.324478,-2.307880]),14.102573),
]

failed=False
for name,res,position,score in checks:
    pos_ok=np.allclose(res.best_position,position,rtol=0,atol=8e-6)
    score_ok=np.isclose(res.best_score,score,rtol=0,atol=8e-6)
    print(f"{name:16s} position={res.best_position} chi2={res.best_score:.6f} {'OK' if pos_ok and score_ok else 'MISMATCH'}")
    failed |= not(pos_ok and score_ok)

if failed:
    raise SystemExit(1)
print("Reference PSO results reproduced.")
