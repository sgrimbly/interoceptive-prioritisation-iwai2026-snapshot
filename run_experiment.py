#!/usr/bin/env python3
"""Run the four conditions (uniform or routed kappa, fixed or dynamic C) on chosen layouts.

Each condition learns its interoceptive likelihood over a chain of episodes per
seed. The measures are pooled over the second half of the episodes for every
(layout, seed) unit and written per unit to JSON.

Example (from the repository root):
  python run_experiment.py --bank original --layouts layout0001 --cells baseuniform \\
      --seeds 1 --out-json runs/layout0001_baseuniform_s0.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from jax import random as jr

import environment
import simulation
from agent import Agent
from environment import BodyConfig, Gridworld, make_channels
from analysis import BANKS, CELLS, load_layouts

# Settings shared by the four conditions (Table S1 of the supplement).
SETTINGS = {
    "planning_horizon": 3,
    "inference_iterations": 4,
    "dirichlet_init_concentration": 0.1,
    "width": 6,
    "height": 6,
    "hunger_max": 10,
    "thirst_max": 10,
    "oxygen_max": 6,
    "hunger_decay": 1,
    "thirst_decay": 1,
    "oxygen_loss_in_water": 2,
    "kappa_attended_cap": 0.9,
    "kappa_floor": 0.05,
    "kappa_budget": 1.95,
}
# The two switches that distinguish the conditions.
CONDITIONS = {
    "baseuniform": {"kappa_force_uniform": True, "dynamic_resource_preferences": False},
    "kappa": {"kappa_force_uniform": False, "dynamic_resource_preferences": False},
    "baseuniform_dynC": {"kappa_force_uniform": True, "dynamic_resource_preferences": True},
    "kappa_dynC": {"kappa_force_uniform": False, "dynamic_resource_preferences": True},
}


def build_runner(cell: str, layout: dict, *, steps: int, episodes: int):
    """Unjitted learning runner for condition ``cell`` on one layout.

    The runner maps ``(env_keys, body_keys)``, each of shape (n_seeds, 2), to an
    ``EpisodeSummary`` with leaves of shape (n_seeds, episodes, ...).
    """
    cfg = {**SETTINGS, **CONDITIONS[cell]}
    grid = Gridworld(
        int(cfg["width"]), int(cfg["height"]),
        start_pos=int(layout["start_pos"]),
        food_positions=layout["food_positions"],
        water_positions=layout["water_positions"],
    )
    body_cfg = BodyConfig(
        hunger_max=int(cfg["hunger_max"]),
        thirst_max=int(cfg["thirst_max"]),
        oxygen_max=int(cfg["oxygen_max"]),
        hunger_decay=int(cfg["hunger_decay"]),
        thirst_decay=int(cfg["thirst_decay"]),
        oxygen_loss_in_water=int(cfg["oxygen_loss_in_water"]),
        kappa_attended_cap=float(cfg["kappa_attended_cap"]),
        kappa_floor=float(cfg["kappa_floor"]),
        kappa_budget=float(cfg["kappa_budget"]),
    )
    agent = Agent(
        grid, make_channels(body_cfg),
        planning_horizon=int(cfg["planning_horizon"]),
        num_iter=int(cfg["inference_iterations"]),
        init_concentration=float(cfg["dirichlet_init_concentration"]),
    )
    config = simulation.RunnerConfig(
        max_steps=int(steps),
        start_pos=grid.start_pos,
        kappa_force_uniform=bool(cfg["kappa_force_uniform"]),
        dynamic_resource_preferences=bool(cfg["dynamic_resource_preferences"]),
    )
    return simulation.make_learning_runner(
        n_episodes=int(episodes),
        env_params=environment.make_params(grid),
        body_params=environment.make_body(body_cfg),
        agent=agent,
        config=config,
    )


def _seed_keys(seeds: int, start: int = 0):
    # Keys depend on the global seed index, so a run with --seed-start k gives seed k
    # the same keys it gets in a single run over all seeds.
    a = jnp.arange(start, start + seeds)
    return (jax.vmap(lambda s: jr.fold_in(jr.PRNGKey(0), s))(a),
            jax.vmap(lambda s: jr.fold_in(jr.PRNGKey(1), s))(a))


def harvest_cell(cell, layout, *, seeds, seed_start, episodes, steps):
    """Run one condition on one layout and return per-seed measures pooled over the second half.

    Each seed runs through the compiled runner as a batch of one, since every
    step evaluates all 25^3 = 15,625 policies. Conflict exposure, minimum
    headroom and the entropies are sums over the second-half episodes divided
    by the survived steps; the learned-model measure is taken at the final
    episode.
    """
    runner_jit = jax.jit(build_runner(cell, layout, steps=steps, episodes=episodes))
    ek, bk = _seed_keys(seeds, seed_start)
    summ = jax.tree.map(
        lambda *xs: jnp.concatenate(xs, axis=0),
        *[runner_jit(ek[s:s + 1], bk[s:s + 1]) for s in range(seeds)])

    post = episodes // 2
    surv = np.asarray(summ.survival_time, dtype=np.float64)[:, post:]         # (seeds, P)
    conflict = np.asarray(summ.conflict_steps, dtype=np.float64)[:, post:]    # (seeds, P)
    minbuf = np.asarray(summ.min_buffer_sum, dtype=np.float64)[:, post:]      # (seeds, P)
    qs_ent = np.asarray(summ.qs_entropy_sum, dtype=np.float64)[:, post:, :]   # (seeds, P, M)
    vhist = np.asarray(summ.visited_state_histogram, dtype=np.float64)[:, post:, :, :]  # (seeds, P, M, L)
    vhist_levels = np.asarray(summ.visited_levels_histogram, dtype=np.float64)[:, post:, :, :]  # (seeds, P, M, Lmax)
    true_prob_all = np.asarray(summ.true_prob_per_state, dtype=np.float64)  # (seeds, episodes, M, Lmax)

    surv_pool = surv.sum(axis=1)                                            # (seeds,)
    denom = np.maximum(surv_pool, 1.0)
    vh_pool = vhist.sum(axis=1)                                             # (seeds, M, L)
    vh_p = vh_pool / np.maximum(vh_pool.sum(axis=-1, keepdims=True), 1.0)   # (seeds, M, L)
    # Empty bins take log(1) = 0 so they add nothing to the entropy.
    vh_logp = np.log(np.where(vh_p > 0.0, vh_p, 1.0))
    # A seed with no survived second-half step has no defined regulation measure.
    no_steps = surv_pool <= 0.0
    return {
        # Survival is not masked: a seed that dies early is data.
        "survival_units": surv.mean(axis=1),
        "conflict_units": np.where(no_steps, np.nan, conflict.sum(axis=1) / denom),
        # Mean over steps of the per-step minimum headroom across channels.
        "buffer_units": np.where(no_steps, np.nan, minbuf.sum(axis=1) / denom),
        "posterior_entropy_units": np.where(no_steps[:, None], np.nan, qs_ent.sum(axis=1) / denom[:, None]),
        "visited_state_entropy_units": np.where(no_steps[:, None], np.nan, -np.sum(vh_p * vh_logp, axis=-1)),
        "visited_levels_units": vhist_levels.sum(axis=1),
        "true_prob_per_state_units": true_prob_all[:, -1, :, :],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", choices=BANKS, default="original")
    ap.add_argument("--layouts", nargs="+", default=["layout0001"],
                    help="layout_id(s) from the chosen bank in layouts.json")
    ap.add_argument("--cells", nargs="+", choices=CELLS, default=list(CELLS),
                    help="conditions to run (default all four)")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed-start", type=int, default=0,
                    help="index of the first seed, so seeds can run as separate jobs")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args(argv)

    by_id = {str(l["layout_id"]): l for l in load_layouts(args.bank)}
    missing_layouts = [lid for lid in args.layouts if lid not in by_id]
    if missing_layouts:
        print(f"Requested layouts absent from the {args.bank} layouts: {missing_layouts}")
        return 1

    rows_out = []
    for key in [cell for cell in CELLS if cell in args.cells]:
        row = {"key": key, "unit_layouts": [], "unit_seeds": []}
        for layout_id in args.layouts:
            units = harvest_cell(key, by_id[layout_id], seeds=args.seeds, seed_start=args.seed_start,
                                 episodes=args.episodes, steps=args.steps)
            for s in range(args.seeds):
                row["unit_layouts"].append(layout_id)
                row["unit_seeds"].append(int(args.seed_start + s))
                for field, values in units.items():
                    value = values[s]
                    row.setdefault(field, []).append(
                        float(value) if np.ndim(value) == 0 else np.asarray(value, dtype=float).tolist())
                print(f"{layout_id} {key} seed {args.seed_start + s}: "
                      f"survival {row['survival_units'][-1]:.1f} of {args.steps} steps")
        rows_out.append(row)

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out_json.with_suffix(args.out_json.suffix + ".tmp")
        temporary.write_text(json.dumps({"steps": args.steps, "cells": rows_out}, indent=2) + "\n")
        temporary.replace(args.out_json)
        print(f"Wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
