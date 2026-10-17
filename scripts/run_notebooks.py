#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute the PSO cosmology notebooks.")
    parser.add_argument("--quick", action="store_true", help="Use reduced swarm sizes and iteration counts.")
    parser.add_argument("--only", nargs="*", default=None, help="Notebook filename prefixes to execute.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    notebooks = sorted((root / "notebooks").glob("*.ipynb"))
    if args.only:
        notebooks = [p for p in notebooks if any(p.name.startswith(prefix) for prefix in args.only)]

    base = root / ("smoke_outputs" if args.quick else "results")
    figure_dir = base / "figures"
    notebook_dir = base / "executed_notebooks"
    figure_dir.mkdir(parents=True, exist_ok=True)
    notebook_dir.mkdir(parents=True, exist_ok=True)

    old_quick = os.environ.get("PSO_QUICK")
    old_output = os.environ.get("PSO_OUTPUT_DIR")
    os.environ["PSO_QUICK"] = "1" if args.quick else "0"
    os.environ["PSO_OUTPUT_DIR"] = str(figure_dir)

    try:
        for path in notebooks:
            print(f"Executing {path.name} ...", flush=True)
            nb = nbformat.read(path, as_version=4)
            client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(root)}})
            executed = client.execute()
            out = notebook_dir / path.name
            nbformat.write(executed, out)
            print(f"  wrote {out.relative_to(root)}", flush=True)
    finally:
        if old_quick is None:
            os.environ.pop("PSO_QUICK", None)
        else:
            os.environ["PSO_QUICK"] = old_quick
        if old_output is None:
            os.environ.pop("PSO_OUTPUT_DIR", None)
        else:
            os.environ["PSO_OUTPUT_DIR"] = old_output


if __name__ == "__main__":
    main()
