"""Reproduce the experiments and statistics of the paper.

    python reproduce.py test          # short end-to-end run on one layout
    python reproduce.py main          # original layouts: survival and internal measures
    python reproduce.py coverage      # original layouts: learned model and state coverage
    python reproduce.py calibration   # 72 further layouts: baseline-only difficulty groups
    python reproduce.py evaluation    # 72 further layouts: four conditions, analysis

Each command ends by computing its statistics from its own runs. Runs and
statistics are written to runs/.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from analysis import (CALIBRATION_SEEDS, CELLS, EPISODES, EVALUATION_SEEDS, MAIN_SEEDS, ROOT, STEPS,
                        assign_difficulty, confirmatory, coverage, load_layouts, main_statistics)

RUNS = ROOT / "runs"

ENV = {
    "JAX_PLATFORMS": "cpu",
    "MALLOC_ARENA_MAX": "2",
}
UNSET = ("XLA_FLAGS", "JAX_ENABLE_X64", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def call(script: str, *args: object, threads: str = "8") -> None:
    """Run one script from the repository root and stop on failure."""
    env = {k: v for k, v in os.environ.items() if k not in UNSET}
    env.update(ENV)
    env["OMP_NUM_THREADS"] = threads
    env["PYTHONPATH"] = str(ROOT)
    cmd = [sys.executable, str(ROOT / script), *map(str, args)]
    print("+", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, env=env, check=True)


def layout_ids(bank: str) -> list[str]:
    return [str(layout["layout_id"]) for layout in load_layouts(bank)]


def run_units(bank: str, directory: Path, *, layouts: list[str], cells: tuple[str, ...],
              seeds: tuple[int, ...], episodes: int, steps: int, threads: str = "8") -> list[Path]:
    """Run each layout, condition and seed as its own job.

    Each job uses its own seed index, so results do not depend on how units are
    grouped. Existing outputs are kept, so an interrupted command can be restarted.
    """
    directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    for layout in layouts:
        for cell in cells:
            for seed in seeds:
                out = directory / f"{layout}_{cell}_s{seed}.json"
                outputs.append(out)
                if out.exists():
                    continue
                call("run_experiment.py", "--bank", bank, "--layouts", layout,
                     "--cells", cell, "--seeds", 1, "--seed-start", seed,
                     "--episodes", episodes, "--steps", steps, "--out-json", out, threads=threads)
    return outputs


def merge_units(paths: list[Path], out: Path) -> Path:
    """Combine single-unit outputs into one run file with units in layout, seed order."""
    runs = [json.loads(path.read_text()) for path in paths]
    rows: dict[str, list[tuple[tuple[str, int], dict]]] = {}
    for run in runs:
        for row in run["cells"]:
            rows.setdefault(row["key"], []).append(
                ((row["unit_layouts"][0], int(row["unit_seeds"][0])), row))
    cells = []
    for key in [cell for cell in CELLS if cell in rows]:
        ordered = [row for _, row in sorted(rows[key], key=lambda item: item[0])]
        per_unit = [k for k in ordered[0]
                    if k.endswith("_units") or k in ("unit_layouts", "unit_seeds")]
        combined = {k: v for k, v in ordered[0].items() if k not in per_unit}
        for field in per_unit:
            combined[field] = [value for row in ordered for value in row[field]]
        cells.append(combined)
    out.write_text(json.dumps({"steps": runs[0]["steps"], "cells": cells}, indent=2) + "\n")
    return out


def main_experiment(layouts: list[str] | None) -> None:
    directory = RUNS / "main"
    units = run_units("original", directory, layouts=layouts or layout_ids("original"),
                      cells=CELLS, seeds=MAIN_SEEDS, episodes=EPISODES, steps=STEPS, threads="2")
    if layouts:
        print("Ran a subset of layouts; run again without --layout once all are complete.")
        return
    merged = merge_units(units, RUNS / "main.json")
    main_statistics(merged, RUNS / "main_statistics.json")


def coverage_experiment(layouts: list[str] | None) -> None:
    directory = RUNS / "coverage"
    units = run_units("original", directory, layouts=layouts or layout_ids("original"),
                      cells=CELLS, seeds=MAIN_SEEDS, episodes=30, steps=150, threads="2")
    if layouts:
        print("Ran a subset of layouts; run again without --layout once all are complete.")
        return
    coverage(units, RUNS / "coverage.json")


def confirmatory_phase(phase: str, layouts: list[str] | None, analyse_only: bool) -> None:
    """Run one confirmatory phase, then analyse it once every layout is complete."""
    directory = RUNS / phase
    cells = ("baseuniform",) if phase == "calibration" else CELLS
    seeds = CALIBRATION_SEEDS if phase == "calibration" else EVALUATION_SEEDS
    if not analyse_only:
        run_units("confirmatory", directory, layouts=layouts or layout_ids("confirmatory"),
                  cells=cells, seeds=seeds, episodes=EPISODES, steps=STEPS)
        if layouts:
            print("Ran a subset of layouts; run with --analyse-only once all are complete.")
            return
    assignment = RUNS / "calibration_assignment.json"
    if phase == "calibration":
        assign_difficulty(directory, assignment)
        return
    if not assignment.exists():
        raise SystemExit(f"{assignment.relative_to(ROOT)} not found: run "
                         "'python reproduce.py calibration' before the evaluation analysis")
    confirmatory(directory, assignment, RUNS / "confirmatory_analysis.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["test", "main", "coverage", "calibration", "evaluation"])
    parser.add_argument("--layout", action="append", dest="layouts", metavar="ID",
                        help="run only this layout (repeatable), e.g. to split work across jobs")
    parser.add_argument("--analyse-only", action="store_true",
                        help="skip runs and analyse existing outputs (confirmatory phases)")
    args = parser.parse_args()

    if args.command == "test":
        units = run_units("original", RUNS / "test", layouts=["layout0001"], cells=CELLS,
                          seeds=(0,), episodes=4, steps=60)
        coverage(units, RUNS / "test" / "coverage.json")
    elif args.command == "main":
        main_experiment(args.layouts)
    elif args.command == "coverage":
        coverage_experiment(args.layouts)
    else:
        confirmatory_phase(args.command, args.layouts, args.analyse_only)


if __name__ == "__main__":
    main()
