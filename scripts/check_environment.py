#!/usr/bin/env python3
from pathlib import Path
import importlib
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

packages = ["numpy", "matplotlib", "nbformat", "nbclient", "ipykernel", "pytest"]
print("Python", sys.version.split()[0])
print("Executable", sys.executable)
failed = False
for name in packages:
    try:
        mod = importlib.import_module(name)
        print(f"{name:12s} {getattr(mod, '__version__', 'installed')}")
    except Exception as exc:
        failed = True
        print(f"{name:12s} MISSING ({exc})")

try:
    import pso_cosmology
    source = Path(pso_cosmology.__file__).resolve()
    expected = (ROOT / "src" / "pso_cosmology").resolve()
    print(f"pso_cosmology source: {source}")
    if expected not in source.parents:
        failed = True
        print(f"Expected local source under: {expected}")
except Exception as exc:
    failed = True
    print(f"pso_cosmology MISSING ({exc})")

rows = [line for line in (ROOT / "data" / "cosmic_chronometers_31.txt").read_text().splitlines() if line.strip()]
print(f"cosmic chronometers: {len(rows)} rows")
if len(rows) != 31:
    failed = True
if failed:
    raise SystemExit(1)
print("Environment check passed.")
