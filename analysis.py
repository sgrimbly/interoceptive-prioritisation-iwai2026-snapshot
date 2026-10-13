"""Statistics of the experiments, and the design constants and run-output loading they share.

Survival and internal-measure statistics of the original layouts, learned-model
coverage, the difficulty groups of the further layouts from the calibration
runs, and the prespecified confirmatory analysis.
"""
from __future__ import annotations

import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
LAYOUTS = ROOT / "layouts.json"
BANKS = ("original", "confirmatory")
# Side of the square grid set by width and height in run_experiment.SETTINGS.
GRID_SIZE = 6

# Condition keys, in the order of run_experiment.CONDITIONS.
CELLS = ("baseuniform", "kappa", "baseuniform_dynC", "kappa_dynC")
MAIN_SEEDS = tuple(range(8))
CALIBRATION_SEEDS = tuple(range(200, 204))
EVALUATION_SEEDS = tuple(range(300, 308))
EPISODES = 20
STEPS = 120


def load_layouts(bank: str) -> list[dict]:
    """The layouts of one bank, in file order."""
    return json.loads(LAYOUTS.read_text())[bank]


def read_units(directory: Path) -> dict[tuple[str, str, int], dict]:
    """Index every unit in the run JSONs under directory by (layout, cell, seed).

    A run JSON holds one or more units per cell, identified by unit_layouts and
    unit_seeds; each per-unit field (name ending in _units) is reduced to that
    unit's entry.
    """
    units: dict[tuple[str, str, int], dict] = {}
    for path in sorted(directory.glob("*.json")):
        for row in json.loads(path.read_text())["cells"]:
            for index, (layout, seed) in enumerate(zip(row["unit_layouts"], row["unit_seeds"])):
                key = (str(layout), str(row["key"]), int(seed))
                if key in units:
                    raise ValueError(f"unit {key} appears more than once in {directory}")
                units[key] = {
                    field: values[index]
                    for field, values in row.items()
                    if field.endswith("_units")
                }
    return units


def require_complete(
    units: dict[tuple[str, str, int], dict],
    *,
    layouts: list[str],
    cells: tuple[str, ...],
    seeds: tuple[int, ...],
) -> None:
    """Raise unless units holds exactly the layout x cell x seed design."""
    expected = {(layout, cell, seed) for layout in layouts for cell in cells for seed in seeds}
    missing = sorted(expected - set(units))
    unexpected = sorted(set(units) - expected)
    if missing or unexpected:
        raise ValueError(
            f"expected {len(layouts)} layouts x {len(cells)} cells x {len(seeds)} seeds "
            f"= {len(expected)} units; {len(missing)} missing (first: {missing[:3]}), "
            f"{len(unexpected)} unexpected (first: {unexpected[:3]})"
        )


BASE, *CONTRASTS = CELLS
MAIN_BOOTSTRAP_RESAMPLES = 10_000
MAIN_RANDOM_SEED = 0
EQUIVALENCE_MARGIN = 0.06


def _exact_signature(layout: dict) -> str:
    """Signature of the layout's start, food, water and sleep positions."""

    def positions(field: str) -> str:
        return ",".join(str(value) for value in sorted(int(value) for value in layout.get(field, [])))

    return (f"s={int(layout['start_pos'])}|f={positions('food_positions')}"
            f"|w={positions('water_positions')}|z={positions('sleep_positions')}")


def _canonical_signature(layout: dict, *, size: int) -> str:
    """Smallest signature over the eight rotations and reflections of the square grid.

    The grid has clamped boundaries, so these are its symmetries and
    translations are not. Food is never exchanged with water.
    """

    def d4(x: int, y: int) -> tuple[tuple[int, int], ...]:
        n = size - 1
        return ((x, y), (n - y, x), (n - x, n - y), (y, n - x),
                (n - x, y), (x, n - y), (y, x), (n - y, n - x))

    def xy(position: int) -> tuple[int, int]:
        return int(position) % size, int(position) // size

    start = xy(layout["start_pos"])
    groups = [[xy(value) for value in layout.get(field, [])]
              for field in ("food_positions", "water_positions", "sleep_positions")]
    candidates = []
    for transform in range(8):
        sx, sy = d4(*start)[transform]
        candidates.append((sy * size + sx, *(
            tuple(sorted(ty * size + tx for tx, ty in (d4(x, y)[transform] for x, y in group)))
            for group in groups)))
    start_pos, food, water, sleep = min(candidates)

    def encode(values: tuple[int, ...]) -> str:
        return ",".join(str(value) for value in values)

    return f"s={start_pos}|f={encode(food)}|w={encode(water)}|z={encode(sleep)}"


def _percentile_interval(values: np.ndarray, lo: float, hi: float) -> list[float]:
    return [float(x) for x in np.percentile(values, [lo, hi])]


def _cluster_bootstrap_mean(
    values: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
    percentiles: tuple[float, float] = (2.5, 97.5),
) -> tuple[float, list[float]]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_resamples, values.size))
    distribution = values[idx].mean(axis=1)
    return float(values.mean()), _percentile_interval(distribution, *percentiles)


def _exact_sign_flip_p(cluster_differences: np.ndarray) -> float:
    """Exact two-sided randomisation p-value over all cluster-level sign flips."""
    differences = np.asarray(cluster_differences, dtype=float)
    observed = abs(float(differences.mean()))
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=differences.size)))
    null = np.abs((signs * differences[None, :]).mean(axis=1))
    return float(np.mean(null >= observed - 1e-15))


def _holm(raw: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values."""
    ordered = sorted(raw, key=raw.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, key in enumerate(ordered):
        running = max(running, (len(ordered) - rank) * raw[key])
        adjusted[key] = min(1.0, running)
    return {key: adjusted[key] for key in raw}


def _load_run_json(
    path: Path,
) -> tuple[list[str], list[str], list[str], np.ndarray, dict[str, np.ndarray]]:
    """Load the per-unit arrays written by run_experiment.py."""
    raw = json.loads(path.read_text())
    rows = {row["key"]: row for row in raw["cells"]}
    cells = [BASE, *CONTRASTS]
    missing = [cell for cell in cells if cell not in rows]
    if missing:
        raise ValueError(f"missing cells in {path}: {', '.join(missing)}")

    keys_by_cell: dict[str, list[tuple[str, str]]] = {}
    for cell in cells:
        row = rows[cell]
        layouts = [str(value) for value in row["unit_layouts"]]
        seeds = [str(value) for value in row["unit_seeds"]]
        keys_by_cell[cell] = list(zip(layouts, seeds))
    reference_keys = keys_by_cell[BASE]
    if len(set(reference_keys)) != len(reference_keys):
        raise ValueError(f"duplicate layout-seed unit in {path}")
    for cell in cells[1:]:
        if keys_by_cell[cell] != reference_keys:
            raise ValueError(f"units are not paired between {BASE} and {cell}")

    layouts = sorted({layout for layout, _ in reference_keys})
    seeds = sorted({seed for _, seed in reference_keys}, key=int)
    expected = {(layout, seed) for layout in layouts for seed in seeds}
    if set(reference_keys) != expected:
        raise ValueError("run does not contain a complete layout-by-seed rectangle")
    position = {key: index for index, key in enumerate(reference_keys)}
    steps = float(raw["steps"])

    def scalar(field: str, *, scale: float = 1.0) -> np.ndarray:
        return np.asarray(
            [
                [
                    [float(rows[cell][field][position[(layout, seed)]]) / scale for seed in seeds]
                    for layout in layouts
                ]
                for cell in cells
            ],
            dtype=float,
        )

    def channel_mean(field: str) -> np.ndarray:
        return np.asarray(
            [
                [
                    [
                        float(np.mean(rows[cell][field][position[(layout, seed)]]))
                        for seed in seeds
                    ]
                    for layout in layouts
                ]
                for cell in cells
            ],
            dtype=float,
        )

    survival = scalar("survival_units", scale=steps)
    internal = {
        "posterior_entropy": channel_mean("posterior_entropy_units"),
        "visited_state_entropy": channel_mean("visited_state_entropy_units"),
        "min_buffer": scalar("buffer_units"),
    }
    if not np.all(np.isfinite(survival)):
        raise ValueError(f"non-finite survival value in {path}")
    return cells, layouts, seeds, survival, internal


def _geometry_clusters(
    bank: list[dict],
    layouts: list[str],
) -> tuple[list[str], dict[str, list[str]], dict[str, int]]:
    """Return canonical geometry signatures, their member layouts and summary counts."""
    rows = {str(row["layout_id"]): row for row in bank}
    missing = sorted(set(layouts) - set(rows))
    if missing:
        raise ValueError(
            f"the original layouts are missing result IDs: {', '.join(missing)}"
        )

    exact = {layout: _exact_signature(rows[layout]) for layout in layouts}
    canonical = {layout: _canonical_signature(rows[layout], size=GRID_SIZE) for layout in layouts}
    members: dict[str, list[str]] = {}
    for layout in layouts:
        members.setdefault(canonical[layout], []).append(layout)
    members = {
        signature: sorted(group)
        for signature, group in sorted(members.items())
    }
    summary = {
        "n_layout_ids": len(layouts),
        "n_exact_geometries": len(set(exact.values())),
        "n_canonical_geometries": len(set(canonical.values())),
        "n_inferential_clusters": len(members),
    }
    return list(members), members, summary


def _collapse_to_clusters(
    values: np.ndarray,
    *,
    layouts: list[str],
    clusters: list[str],
    members: dict[str, list[str]],
) -> np.ndarray:
    """Average layout IDs within each geometry, keeping the cell and seed axes."""
    position = {layout: index for index, layout in enumerate(layouts)}
    return np.stack(
        [
            values[:, [position[layout] for layout in members[cluster]], :].mean(axis=1)
            for cluster in clusters
        ],
        axis=1,
    )


def _correlations_by_row(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x0 = x - x.mean(axis=-1, keepdims=True)
    y0 = y - y.mean(axis=-1, keepdims=True)
    denominator = np.sqrt(np.sum(x0 * x0, axis=-1) * np.sum(y0 * y0, axis=-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sum(x0 * y0, axis=-1) / denominator


def _cross_fitted_difficulty(
    baseline: np.ndarray,
    cell: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
) -> tuple[float, list[float], int]:
    """Cross-fitted difficulty correlation and its geometry bootstrap interval."""
    n_layouts, n_seeds = baseline.shape
    if n_seeds % 2:
        raise ValueError("balanced disjoint seed splits require an even seed count")
    split_size = n_seeds // 2
    split_indices = list(itertools.combinations(range(n_seeds), split_size))
    selector = np.zeros((len(split_indices), n_seeds), dtype=bool)
    for row, split in enumerate(split_indices):
        selector[row, list(split)] = True

    # Shape (split, layout); each effect half is disjoint from its difficulty half.
    difficulty = np.stack([baseline[:, mask].mean(axis=1) for mask in selector])
    paired_effect = cell - baseline
    effects = np.stack([paired_effect[:, ~mask].mean(axis=1) for mask in selector])
    split_correlations = _correlations_by_row(difficulty, effects)
    finite_split_correlations = split_correlations[np.isfinite(split_correlations)]
    if finite_split_correlations.size == 0:
        raise ValueError("difficulty correlation is undefined for every balanced seed split")
    point = float(finite_split_correlations.mean())

    rng = np.random.default_rng(seed)
    bootstrap = np.empty(n_resamples, dtype=float)
    batch_size = 500
    for start in range(0, n_resamples, batch_size):
        stop = min(start + batch_size, n_resamples)
        idx = rng.integers(0, n_layouts, size=(stop - start, n_layouts))
        # Advanced indexing gives (split, bootstrap, layout).
        x = difficulty[:, idx]
        y = effects[:, idx]
        correlations = _correlations_by_row(x, y)
        finite = np.isfinite(correlations)
        counts = finite.sum(axis=0)
        sums = np.where(finite, correlations, 0.0).sum(axis=0)
        bootstrap[start:stop] = np.divide(
            sums,
            counts,
            out=np.full(stop - start, np.nan),
            where=counts > 0,
        )

    bootstrap = bootstrap[np.isfinite(bootstrap)]
    if bootstrap.size == 0:
        raise ValueError("difficulty bootstrap produced no finite estimates")
    return point, _percentile_interval(bootstrap, 2.5, 97.5), len(split_indices)


def _contrast_record(
    cluster_differences: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, float | bool | list[float]]:
    mean, ci95 = _cluster_bootstrap_mean(
        cluster_differences,
        n_resamples=n_resamples,
        seed=seed,
    )
    standard_error = float(
        np.std(cluster_differences, ddof=1) / np.sqrt(cluster_differences.size)
    )
    return {
        "mean": mean,
        "ci95": ci95,
        "cluster_standard_error": standard_error,
        "signal_to_error": float(abs(mean) / standard_error) if standard_error > 0 else None,
        "ci95_excludes_zero": bool(ci95[0] > 0 or ci95[1] < 0),
    }


def main_statistics(run_json: Path, out: Path) -> None:
    """Survival and internal-measure statistics of the 12 original layouts (Tables 1 and 3).

    Layout IDs that share a geometry up to rotation or reflection of the clamped
    square grid are averaged within that geometry, and seeds are averaged within
    geometry, before bootstrap resampling or sign flipping. The difficulty analysis
    uses all balanced 4/4 seed splits: one half estimates baseline difficulty and
    the disjoint half estimates the paired cell-minus-baseline effect. Its point
    estimate is the mean correlation over the 70 splits; its interval bootstraps
    geometries and recomputes the full cross-fitted estimator in every resample.
    """
    cells, layouts, seeds, survival, internal = _load_run_json(run_json)

    clusters, cluster_members, cluster_summary = _geometry_clusters(load_layouts("original"), layouts)
    survival = _collapse_to_clusters(
        survival,
        layouts=layouts,
        clusters=clusters,
        members=cluster_members,
    )
    internal = {
        metric: _collapse_to_clusters(
            values,
            layouts=layouts,
            clusters=clusters,
            members=cluster_members,
        )
        for metric, values in internal.items()
    }

    cluster_survival = survival.mean(axis=2)
    survival_cells: dict[str, dict[str, float | list[float]]] = {}
    for cell_index, cell in enumerate(cells):
        mean, ci95 = _cluster_bootstrap_mean(
            cluster_survival[cell_index],
            n_resamples=MAIN_BOOTSTRAP_RESAMPLES,
            seed=MAIN_RANDOM_SEED,
        )
        survival_cells[cell] = {"mean": mean, "ci95": ci95}

    raw_p: dict[str, float] = {}
    survival_contrasts: dict[str, dict[str, float | bool | list[float]]] = {}
    for cell_index, cell in enumerate(cells[1:], start=1):
        cluster_differences = cluster_survival[cell_index] - cluster_survival[0]
        record = _contrast_record(
            cluster_differences,
            n_resamples=MAIN_BOOTSTRAP_RESAMPLES,
            seed=MAIN_RANDOM_SEED,
        )
        raw_p[cell] = _exact_sign_flip_p(cluster_differences)
        _, ci90 = _cluster_bootstrap_mean(
            cluster_differences,
            n_resamples=MAIN_BOOTSTRAP_RESAMPLES,
            seed=MAIN_RANDOM_SEED,
            percentiles=(5.0, 95.0),
        )
        record["ci90"] = ci90
        record["equivalence_margin"] = EQUIVALENCE_MARGIN
        record["equivalent"] = bool(
            ci90[0] > -EQUIVALENCE_MARGIN and ci90[1] < EQUIVALENCE_MARGIN
        )
        survival_contrasts[cell] = record

    adjusted_p = _holm(raw_p)
    for cell in CONTRASTS:
        survival_contrasts[cell]["exact_sign_flip_p"] = raw_p[cell]
        survival_contrasts[cell]["holm_p"] = adjusted_p[cell]

    difficulty: dict[str, dict[str, float | int | list[float]]] = {}
    for cell_index, cell in enumerate(cells[1:], start=1):
        point, ci95, n_splits = _cross_fitted_difficulty(
            survival[0],
            survival[cell_index],
            n_resamples=MAIN_BOOTSTRAP_RESAMPLES,
            seed=MAIN_RANDOM_SEED,
        )
        difficulty[cell] = {
            "correlation": point,
            "ci95": ci95,
            "balanced_seed_splits": n_splits,
        }

    internal_contrasts: dict[str, dict[str, dict[str, float | bool | list[float]]]] = {}
    for metric, values in internal.items():
        cluster_means = values.mean(axis=2)
        internal_contrasts[metric] = {}
        for cell_index, cell in enumerate(cells[1:], start=1):
            internal_contrasts[metric][cell] = _contrast_record(
                cluster_means[cell_index] - cluster_means[0],
                n_resamples=MAIN_BOOTSTRAP_RESAMPLES,
                seed=MAIN_RANDOM_SEED,
            )

    output = {
        "method": {
            "inferential_cluster": "canonical",
            **cluster_summary,
            "n_seeds_per_cluster": len(seeds),
            "n_cluster_seed_units": len(clusters) * len(seeds),
            "bootstrap_resamples": MAIN_BOOTSTRAP_RESAMPLES,
            "random_seed": MAIN_RANDOM_SEED,
            "bootstrap_interval": "percentile",
            "p_value": "exact two-sided sign flip of cluster-level paired differences",
            "multiplicity": "Holm adjustment across the three survival contrasts",
            "difficulty_estimator": (
                "mean Pearson correlation over all balanced 4/4 seed splits, with "
                "baseline difficulty and paired effect estimated on disjoint halves"
            ),
            "difficulty_interval": (
                "cluster percentile bootstrap that recomputes the complete "
                "cross-fitted estimator"
            ),
        },
        "layout_ids": layouts,
        "clusters": clusters,
        "cluster_members": cluster_members,
        "seeds": [int(seed) for seed in seeds],
        "survival_cells": survival_cells,
        "survival_contrasts_vs_baseuniform": survival_contrasts,
        "difficulty_crossfit": difficulty,
        "internal_contrasts_vs_baseuniform": internal_contrasts,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {out}")


MIN_VISITS = 1
COVERAGE_METRICS = [
    ("tp_full", "model quality, all"),
    ("tp_vis", "model quality, visited"),
    ("coverage", "levels visited"),
]
# Summary written to the output: (output key, analysis key) for metrics and (output key, cell) for cells.
SUMMARY_METRICS = (("full", "tp_full"), ("visited", "tp_vis"), ("coverage", "coverage"))
SUMMARY_CELLS = (("base", BASE), ("dynC", "baseuniform_dynC"))


def _load_units(paths: list[Path]):
    """Return per-cell lists of dict(layout, seed, visited, tp), one per unit."""
    by_cell = defaultdict(list)
    for path in paths:
        payload = json.loads(path.read_text())
        for row in payload["cells"]:
            cell = str(row["key"])
            if cell not in CELLS:
                continue
            vt = np.asarray(row["visited_levels_units"], dtype=float)              # (S, M, L)
            tp = np.asarray(row["true_prob_per_state_units"], dtype=float)  # (S, M, L)
            for si in range(vt.shape[0]):
                by_cell[cell].append(dict(
                    layout=str(row["unit_layouts"][si]), seed=int(row["unit_seeds"][si]),
                    visited=vt[si], tp=tp[si]))
    return by_cell


def _valid_mask(units):
    """Valid-column mask (M, L) over all units.

    Padded columns are exactly 0 in the learned-model measure while real columns are clipped
    to at least 1e-12, so a column that is real in any unit is real.
    """
    M, L = units[0]["tp"].shape
    valid = np.zeros((M, L), dtype=bool)
    for u in units:
        valid |= (u["tp"] > 0.0)
    return valid


def _unit_reductions(u, valid):
    """All-column and visited-column means of the learned-model measure, and coverage, for one unit."""
    tp, vis = u["tp"], u["visited"]
    # All valid columns are pooled across channels with equal weight.
    visited_mask = valid & (vis >= MIN_VISITS)
    return {
        "tp_full": float(tp[valid].mean()) if valid.any() else np.nan,
        "tp_vis": float(tp[visited_mask].mean()) if visited_mask.any() else np.nan,
        "coverage": float(visited_mask.sum() / max(1, valid.sum())),
    }


def _cluster_boot_contrast(base_layout_means, cell_layout_means, layouts, n=10000, seed=0):
    """Layout-clustered paired bootstrap of the cell-minus-baseline per-layout means."""
    rng = np.random.default_rng(seed)
    L = [l for l in layouts if l in base_layout_means and l in cell_layout_means
         and np.isfinite(base_layout_means[l]) and np.isfinite(cell_layout_means[l])]
    if not L:
        return float("nan"), float("nan"), float("nan")
    diffs = np.array([cell_layout_means[l] - base_layout_means[l] for l in L])
    idx = rng.integers(0, len(L), size=(n, len(L)))
    boot = diffs[idx].mean(axis=1)
    return float(diffs.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _cluster_boot_mean(layout_means, layouts, n=10000, seed=0):
    """Layout-clustered bootstrap interval for one cell's mean over layouts."""
    rng = np.random.default_rng(seed)
    values = np.array([layout_means[l] for l in layouts
                       if l in layout_means and np.isfinite(layout_means[l])])
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, values.size, size=(n, values.size))
    boot = values[idx].mean(axis=1)
    return float(values.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _per_layout_means(units, key, valid):
    by_layout = defaultdict(list)
    for u in units:
        by_layout[u["layout"]].append(_unit_reductions(u, valid)[key])
    return {lay: float(np.nanmean(v)) for lay, v in by_layout.items()}


def coverage(run_jsons: list[Path], out: Path) -> None:
    """State coverage and the learned-model measure of the learned interoceptive likelihoods (Table S2).

    The measure is learned-model quality: in each column, the learned probability of
    the observation the true likelihood ranks highest. Each unit is reduced to its mean over
    all valid columns (``tp_full``), its mean over the columns visited at least
    MIN_VISITS times (``tp_vis``) and the fraction of valid columns visited
    (``coverage``).
    Intervals are layout-clustered percentile bootstraps over per-layout means.
    """
    by_cell = _load_units(run_jsons)
    present = {c: len(by_cell.get(c, [])) for c in CELLS}
    print("units per cell:", present)
    if not by_cell[BASE]:
        raise SystemExit("no baseline units found")

    # The valid-column mask is structural, so the baseline mask applies to every cell.
    valid = _valid_mask(by_cell[BASE])
    layouts = sorted({u["layout"] for u in by_cell[BASE]})
    print(f"valid columns per channel: {valid.sum(axis=1).tolist()}  | layouts: {len(layouts)}")

    print("\nper-cell means (mean of per-layout means)")
    hdr = f"{'metric':>20} | " + " | ".join(f"{c:>16}" for c in CELLS)
    print(hdr)
    print("-" * len(hdr))
    cell_layout_means = {}
    for key, lab in COVERAGE_METRICS:
        row = []
        for c in CELLS:
            plm = _per_layout_means(by_cell[c], key, valid)
            cell_layout_means[(key, c)] = plm
            row.append(float(np.nanmean(list(plm.values()))) if plm else float("nan"))
        print(f"{lab:>20} | " + " | ".join(f"{v:>16.4f}" for v in row))

    cell_estimates = {}
    for key, _ in COVERAGE_METRICS:
        cell_estimates[key] = {}
        for c in CELLS:
            mean, lo, hi = _cluster_boot_mean(cell_layout_means[(key, c)], layouts)
            cell_estimates[key][c] = dict(mean=mean, lo=lo, hi=hi)

    print("\ncontrasts against baseline (layout-clustered 95% CI)")
    results = {}
    for key, lab in COVERAGE_METRICS:
        if key == "coverage":
            continue
        print(f"\n{lab}:")
        for c in CELLS:
            if c == BASE:
                continue
            mean, lo, hi = _cluster_boot_contrast(
                cell_layout_means[(key, BASE)], cell_layout_means[(key, c)], layouts)
            sig = "" if (lo <= 0 <= hi) else "  *"
            print(f"   {c:>18}: {mean:+.4f}  [{lo:+.4f}, {hi:+.4f}]{sig}")
            results[f"{key}::{c}"] = dict(mean=mean, lo=lo, hi=hi)

    metrics = {
        summary_key: {
            summary_cell: [cell_estimates[key][cell][part] for part in ("mean", "lo", "hi")]
            for summary_cell, cell in SUMMARY_CELLS
        }
        for summary_key, key in SUMMARY_METRICS
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        n=present[BASE], n_layouts=len(layouts), metrics=metrics,
        units_per_cell=present, valid_columns=valid.sum(axis=1).tolist(),
        minvisit=MIN_VISITS, cell_estimates=cell_estimates,
        contrasts=results), indent=2) + "\n")
    print(f"\nwrote {out}")


def assign_difficulty(calibration_dir: Path, out: Path) -> None:
    """Rank the 72 confirmatory layouts by baseline-only calibration survival and assign tertiles."""
    bank = load_layouts("confirmatory")
    layout_ids = [str(layout["layout_id"]) for layout in bank]
    units = read_units(calibration_dir)
    require_complete(
        units, layouts=layout_ids, cells=("baseuniform",), seeds=CALIBRATION_SEEDS
    )
    rows = []
    for layout in bank:
        layout_id = str(layout["layout_id"])
        outcomes = [
            float(units[(layout_id, "baseuniform", seed)]["survival_units"]) / STEPS
            for seed in CALIBRATION_SEEDS
        ]
        rows.append(
            {
                "layout_id": layout_id,
                "canonical_signature": layout["canonical_signature"],
                "calibration_survival_by_seed": {
                    str(seed): value for seed, value in zip(CALIBRATION_SEEDS, outcomes)
                },
                "calibration_survival_mean": sum(outcomes) / len(outcomes),
                "travel_burden": int(layout["travel_burden"]),
                "max_start_to_resource": int(layout["max_start_to_resource"]),
            }
        )

    ranked = sorted(
        rows,
        key=lambda row: (
            float(row["calibration_survival_mean"]),
            -int(row["travel_burden"]),
            -int(row["max_start_to_resource"]),
            str(row["canonical_signature"]),
        ),
    )
    if len(ranked) != 72:
        raise ValueError(f"the confirmatory bank must hold 72 layouts, found {len(ranked)}")
    labels = ("hard", "middle", "easy")
    groups: dict[str, list[str]] = {label: [] for label in labels}
    for rank, row in enumerate(ranked, start=1):
        tier = labels[(rank - 1) // 24]
        row["rank"] = rank
        row["tier"] = tier
        groups[tier].append(str(row["layout_id"]))

    output = {
        "method": {
            "cell": "baseuniform",
            "calibration_seeds": list(CALIBRATION_SEEDS),
            "evaluation_seeds": list(EVALUATION_SEEDS),
            "episodes": EPISODES,
            "steps": STEPS,
            "rank_order": (
                "ascending mean baseline survival, then descending travel burden, "
                "descending max start-resource distance, ascending canonical signature"
            ),
            "tertiles": "ranks 1--24 hard, 25--48 middle, 49--72 easy",
        },
        "groups": groups,
        "ranked_geometries": ranked,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote 24/24/24 difficulty assignment -> {out}")


SECONDARY_FACTORIAL_KEYS = (
    "marginal_kappa",
    "marginal_dynamic_C",
    "kappa_by_dynamic_C",
)
CONFIRMATORY_METRICS = {
    "survival": "survival_units",
    "conflict_exposure": "conflict_units",
    "min_buffer": "buffer_units",
    "posterior_entropy": "posterior_entropy_units",
    "visited_state_entropy": "visited_state_entropy_units",
}
CONFIRMATORY_BOOTSTRAP_RESAMPLES = 20_000
PERMUTATIONS = 100_000
CONFIRMATORY_RANDOM_SEED = 20_260_810


def _interval(values: np.ndarray, *, n: int, seed: int) -> list[float]:
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("bootstrap values must be finite")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n, values.size))
    boot = values[idx].mean(axis=1)
    return [float(value) for value in np.percentile(boot, [2.5, 97.5])]


def _mean_record(values: np.ndarray, *, n: int, seed: int) -> dict[str, object]:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "ci95": _interval(values, n=n, seed=seed),
        "n_geometry_clusters": int(values.size),
    }


def _exact_sign_flip_integer(
    values: np.ndarray,
    *,
    scale: int = 19_200,
) -> tuple[float, dict[str, object]]:
    """Exact two-sided sign-flip p-value by integer dynamic programming.

    Survival effects are multiples of 1/19,200 (120 steps, ten second-half
    episodes, eight seeds and the one-half factorial weight), so scaling to
    integers loses nothing and avoids enumerating 2**72 sign vectors.
    """
    values = np.asarray(values, dtype=float)
    signed = np.rint(values * scale).astype(np.int64)
    error = float(np.max(np.abs(values - signed / scale))) if values.size else 0.0
    if error > 1e-10:
        raise ValueError(f"survival contrast is not a multiple of 1/{scale}: {error}")
    weights = np.abs(signed).tolist()
    distribution: Counter[int] = Counter({0: 1})
    for weight in weights:
        if weight == 0:
            distribution = Counter({total: 2 * count for total, count in distribution.items()})
        else:
            updated: Counter[int] = Counter()
            for total, count in distribution.items():
                updated[total + weight] += count
                updated[total - weight] += count
            distribution = updated
    observed = abs(int(signed.sum()))
    extreme = sum(count for total, count in distribution.items() if abs(total) >= observed)
    denominator = 2 ** len(weights)
    return extreme / denominator, {
        "method": "exact integer dynamic-programming sign flip",
        "integer_scale": scale,
        "observed_absolute_integer_sum": observed,
        "distinct_null_sums": len(distribution),
        "extreme_assignments": extreme,
        "total_assignments": denominator,
        "max_integerisation_error": error,
    }


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    x0 = x - x.mean()
    denominator = float(np.dot(x0, x0))
    if denominator == 0:
        raise ValueError("calibration survival has zero variance")
    return float(np.dot(x0, y - y.mean()) / denominator)


def _moderation(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_bootstrap: int,
    n_permutations: int,
    seed: int,
) -> dict[str, object]:
    """Least-squares slope of the effect on calibration survival, with bootstrap CI and permutation p."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("moderation inputs must be finite")

    try:
        point = _slope(x, y)
    except ValueError as error:
        return {
            "status": "undefined",
            "reason": str(error),
            "n_geometry_clusters": int(x.size),
            "slope": None,
            "ci95": None,
            "ci_status": "undefined_full_sample",
            "permutation_p": None,
            "bootstrap_resamples": n_bootstrap,
            "finite_bootstrap_slopes": 0,
            "degenerate_bootstrap_slopes": None,
            "permutations": n_permutations,
            "permutations_completed": 0,
            "permutation_seed": seed,
            "expected_direction": "negative",
        }

    rng = np.random.default_rng(seed)
    slopes = np.empty(n_bootstrap)
    for index in range(n_bootstrap):
        sample = rng.integers(0, len(x), size=len(x))
        try:
            slopes[index] = _slope(x[sample], y[sample])
        except ValueError:
            slopes[index] = np.nan
    finite = slopes[np.isfinite(slopes)]
    # The interval is defined over all draws; dropping undefined slopes would
    # condition it on an outcome-dependent subset.
    bootstrap_complete = finite.size == n_bootstrap
    ci = (
        [float(value) for value in np.percentile(finite, [2.5, 97.5])]
        if bootstrap_complete
        else None
    )

    extreme = 0
    observed = abs(point)
    for _ in range(n_permutations):
        if abs(_slope(rng.permutation(x), y)) >= observed - 1e-15:
            extreme += 1
    p_value = (extreme + 1) / (n_permutations + 1)
    return {
        "status": (
            "estimated"
            if bootstrap_complete
            else "estimated_without_bootstrap_interval"
        ),
        "reason": (
            None
            if bootstrap_complete
            else "one or more bootstrap resamples had zero calibration variance"
        ),
        "n_geometry_clusters": int(x.size),
        "slope": point,
        "ci95": ci,
        "ci_status": (
            "available" if bootstrap_complete else "unavailable_degenerate_resamples"
        ),
        "permutation_p": p_value,
        "bootstrap_resamples": n_bootstrap,
        "finite_bootstrap_slopes": int(finite.size),
        "degenerate_bootstrap_slopes": int(n_bootstrap - finite.size),
        "permutations": n_permutations,
        "permutations_completed": n_permutations,
        "permutation_seed": seed,
        "expected_direction": "negative",
    }


def _adjust_secondary_family(
    moderation: dict[str, object],
    factorial_p: dict[str, float],
) -> tuple[dict[str, float], float]:
    """Holm-adjust moderation and the three factorial tests as one family of four."""
    if moderation["status"] == "undefined":
        # An undefined moderation stays in the family with p = 1 rather than
        # shrinking the family after seeing the data.
        moderation_p = 1.0
    else:
        moderation_p = float(moderation["permutation_p"])
    raw = {"difficulty_moderation": moderation_p, **factorial_p}
    return _holm(raw), moderation_p


def _moderation_conclusion(moderation: dict[str, object]) -> str:
    """Classify the moderation slope by its sign and its 95% bootstrap interval."""
    status = moderation["status"]
    if status == "undefined":
        return "undefined"
    if status == "estimated_without_bootstrap_interval":
        return "inconclusive"
    slope = float(moderation["slope"])
    low, high = (float(value) for value in moderation["ci95"])
    if slope < 0 and high < 0:
        return "expected_supported"
    if slope > 0 and low > 0:
        return "opposite_supported"
    return "inconclusive"


def _factorial(cell_means: np.ndarray) -> dict[str, np.ndarray]:
    base, kappa, dyn_c, combined = (cell_means[:, index] for index in range(4))
    return {
        "marginal_kappa": 0.5 * ((kappa - base) + (combined - dyn_c)),
        "marginal_dynamic_C": 0.5 * ((dyn_c - base) + (combined - kappa)),
        "kappa_by_dynamic_C": (combined - kappa) - (dyn_c - base),
    }


def confirmatory(evaluation_dir: Path, assignment: Path, out: Path) -> None:
    """Prespecified analysis of the 72 further layouts (Table 2, Fig. S1).

    The primary test is the marginal dynamic-C survival effect on the 24 hard
    geometries assigned by assign_difficulty from baseline-only calibration
    runs. Secondary tests are the difficulty moderation slope and the three
    factorial survival contrasts on all 72 geometries, Holm-adjusted as one family.
    """
    calibration = json.loads(assignment.read_text())
    layout_ids = [str(row["layout_id"]) for row in load_layouts("confirmatory")]
    tier_by_layout = {
        str(row["layout_id"]): str(row["tier"])
        for row in calibration["ranked_geometries"]
    }
    baseline_calibration = {
        str(row["layout_id"]): float(row["calibration_survival_mean"])
        for row in calibration["ranked_geometries"]
    }
    if set(tier_by_layout) != set(layout_ids):
        raise ValueError("assignment and bank layout sets differ")

    seeds = EVALUATION_SEEDS
    units = read_units(evaluation_dir)
    require_complete(units, layouts=layout_ids, cells=CELLS, seeds=seeds)
    arrays = {
        metric: np.empty((len(layout_ids), len(CELLS), len(seeds)), dtype=float)
        for metric in CONFIRMATORY_METRICS
    }
    for layout_index, layout in enumerate(layout_ids):
        for cell_index, cell in enumerate(CELLS):
            for seed_index, seed in enumerate(seeds):
                unit = units[(layout, cell, seed)]
                for metric, field in CONFIRMATORY_METRICS.items():
                    value = unit[field]
                    if metric in {"posterior_entropy", "visited_state_entropy"}:
                        value = float(np.mean(value))
                    arrays[metric][layout_index, cell_index, seed_index] = float(value)
    arrays["survival"] /= STEPS

    cell_means = {metric: values.mean(axis=2) for metric, values in arrays.items()}
    effects = {metric: _factorial(values) for metric, values in cell_means.items()}
    hard_mask = np.asarray([tier_by_layout[layout] == "hard" for layout in layout_ids])
    d_c = effects["survival"]["marginal_dynamic_C"]

    primary = _mean_record(
        d_c[hard_mask], n=CONFIRMATORY_BOOTSTRAP_RESAMPLES, seed=CONFIRMATORY_RANDOM_SEED
    )
    primary_p, primary_exact = _exact_sign_flip_integer(d_c[hard_mask])
    primary["exact_sign_flip_p"] = primary_p
    primary["exact_test"] = primary_exact
    primary["supported"] = bool(
        primary["mean"] > 0
        and primary["ci95"][0] > 0
        and primary_p < 0.05
    )

    calibration_x = np.asarray(
        [baseline_calibration[layout] for layout in layout_ids], dtype=float
    )
    moderation = _moderation(
        calibration_x,
        d_c,
        n_bootstrap=CONFIRMATORY_BOOTSTRAP_RESAMPLES,
        n_permutations=PERMUTATIONS,
        seed=CONFIRMATORY_RANDOM_SEED + 1,
    )
    secondary: dict[str, dict[str, object]] = {}
    raw_factorial_p: dict[str, float] = {}
    for index, name in enumerate(SECONDARY_FACTORIAL_KEYS, start=2):
        record = _mean_record(
            effects["survival"][name],
            n=CONFIRMATORY_BOOTSTRAP_RESAMPLES,
            seed=CONFIRMATORY_RANDOM_SEED + index,
        )
        p_value, exact = _exact_sign_flip_integer(effects["survival"][name])
        record["exact_sign_flip_p"] = p_value
        record["exact_test"] = exact
        secondary[name] = record
        raw_factorial_p[name] = p_value
    adjusted, moderation_family_p = _adjust_secondary_family(
        moderation, raw_factorial_p
    )
    moderation["holm_family_p_used"] = moderation_family_p
    moderation["holm_p"] = (
        adjusted["difficulty_moderation"]
        if moderation["status"] != "undefined"
        else None
    )
    moderation["conclusion"] = _moderation_conclusion(moderation)
    for name in secondary:
        secondary[name]["holm_p"] = adjusted[name]

    strata = {}
    for offset, tier in enumerate(("hard", "middle", "easy"), start=10):
        mask = np.asarray([tier_by_layout[layout] == tier for layout in layout_ids])
        record = _mean_record(
            d_c[mask], n=CONFIRMATORY_BOOTSTRAP_RESAMPLES, seed=CONFIRMATORY_RANDOM_SEED + offset
        )
        p_value, exact = _exact_sign_flip_integer(d_c[mask])
        record["exact_sign_flip_p_unadjusted"] = p_value
        record["exact_test"] = exact
        strata[tier] = record

    mechanism = {}
    for metric in CONFIRMATORY_METRICS:
        if metric == "survival":
            continue
        mechanism[metric] = {
            name: _mean_record(
                values,
                n=CONFIRMATORY_BOOTSTRAP_RESAMPLES,
                seed=CONFIRMATORY_RANDOM_SEED + 20 + effect_index,
            )
            for effect_index, (name, values) in enumerate(effects[metric].items())
        }

    median_cells = np.median(arrays["survival"], axis=2)
    median_dc = _factorial(median_cells)["marginal_dynamic_C"][hard_mask]
    median_primary = _mean_record(
        median_dc, n=CONFIRMATORY_BOOTSTRAP_RESAMPLES, seed=CONFIRMATORY_RANDOM_SEED + 40
    )
    median_primary["exact_sign_flip_p"], median_primary["exact_test"] = (
        _exact_sign_flip_integer(median_dc, scale=4_800)
    )
    hard_values = d_c[hard_mask]
    leave_one_out = np.asarray(
        [np.delete(hard_values, index).mean() for index in range(len(hard_values))]
    )

    raw_rows = []
    for index, layout in enumerate(layout_ids):
        raw_rows.append(
            {
                "layout_id": layout,
                "tier": tier_by_layout[layout],
                "calibration_survival": baseline_calibration[layout],
                "evaluation_cell_survival": {
                    cell: float(cell_means["survival"][index, cell_index])
                    for cell_index, cell in enumerate(CELLS)
                },
                "marginal_dynamic_C": float(d_c[index]),
                "marginal_kappa": float(effects["survival"]["marginal_kappa"][index]),
                "kappa_by_dynamic_C": float(
                    effects["survival"]["kappa_by_dynamic_C"][index]
                ),
            }
        )

    output = {
        "method": {
            "inferential_cluster": "D4-unique clamped-square geometry",
            "n_geometries": len(layout_ids),
            "n_seeds_per_cell_geometry": len(seeds),
            "evaluation_seeds": list(seeds),
            "bootstrap_resamples": CONFIRMATORY_BOOTSTRAP_RESAMPLES,
            "permutations": PERMUTATIONS,
            "random_seed": CONFIRMATORY_RANDOM_SEED,
            "secondary_multiplicity": "Holm across moderation and three factorial contrasts",
            "undefined_moderation_holm_policy": (
                "retain the fixed four-test family with p=1 for adjustment, while "
                "reporting the moderation p-values as undefined"
            ),
            "moderation_narrative_rule": (
                "expected-direction support requires a negative slope and a complete "
                "95% bootstrap interval below zero; Holm p is reported but is not an "
                "additional criterion"
            ),
        },
        "primary_hard_marginal_dynamic_C": primary,
        "difficulty_moderation": moderation,
        "secondary_factorial_survival": secondary,
        "dynamic_C_by_tertile": strata,
        "mechanism_factorial_effects": mechanism,
        "sensitivity": {
            "median_across_seeds_primary": median_primary,
            "leave_one_hard_geometry_out_mean_range": [
                float(leave_one_out.min()),
                float(leave_one_out.max()),
            ],
        },
        "raw_geometry_effects": raw_rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote confirmatory analysis -> {out}")
