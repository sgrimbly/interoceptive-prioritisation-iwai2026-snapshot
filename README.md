# Two Routes to Interoceptive Prioritisation

Code for

> St John Grimbly, Nicolas Kuske, Evert A. Boonstra, Charel van Hoof, Rowan Hodson,
> Benjamin Rosman, Ryan Smith, Mark Solms and Jonathan P. Shock.
> *Two Routes to Interoceptive Prioritisation: Dissociating Likelihood Precision and
> Preference Reweighting in Active Inference.* International Workshop on Active
> Inference (IWAI), 2026.

Paper page: <https://stjohngrimbly.com/iwai2026>

An active-inference agent forages in a gridworld with three lethal needs: hunger,
thirst and oxygen. It responds to the need it believes most urgent in one of two
ways. Under κ-routing it raises the precision of that need's interoceptive
channel, both in the observations the body emits and in the agent's likelihood.
Under dynamic-C it raises its preference for the resource that
relieves the need. The experiments compare four conditions, using the same agent
in every one: a uniform baseline, κ-routing, dynamic-C, and both together. The
supplementary material is `supplement.pdf`.

## Contents

```
reproduce.py        every experiment command, each ending with its statistics
run_experiment.py   one run: the four conditions and their settings, command line, per-unit output
environment.py      grid world, tiles, actions and body
agent.py            generative model and planner
simulation.py       JAX episode and learning loop
analysis.py         survival statistics, coverage, difficulty groups, confirmatory analysis
layouts.json        the 12 original layouts and the 72 further layouts
```

## Installation

Python 3.12 is required. With [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip sync requirements.txt
```

With pip, create a Python 3.12 virtual environment and run
`pip install -r requirements.txt`. The file pins every package, including the
`pymdp` commit the experiments used.

## Reproducing the paper

```bash
python reproduce.py test          # short end-to-end check on one layout
python reproduce.py main          # original layouts: survival and internal measures
python reproduce.py coverage      # original layouts: learned model and state coverage
python reproduce.py calibration   # 72 further layouts: baseline-only difficulty groups
python reproduce.py evaluation    # 72 further layouts: four conditions and analysis
```

Each command runs its units and then computes its statistics from those runs.
Everything is written to `runs/`:

| Paper | Command | Output in `runs/` |
|---|---|---|
| Table 1 and the difficulty correlation on the original layouts | `main` | `main_statistics.json` |
| Table 3 | `main` | `main_statistics.json` |
| Table 2 | `evaluation` | `confirmatory_analysis.json` |
| Table S1 | none, constants | `SETTINGS` and `CONDITIONS` in `run_experiment.py` |
| Table S2 | `coverage` | `coverage.json` |
| Difficulty groups (supplement Section B) | `calibration` | `calibration_assignment.json` |
| Fig. S1 values | `evaluation` | `confirmatory_analysis.json` |

Table S1 also uses the dynamic-C constants at the top of `simulation.py`, the
number of need levels in `environment.py`, and the seeds, run lengths and
resamples at the top of `analysis.py`.

`evaluation` uses the difficulty groups that `calibration` wrote to
`runs/calibration_assignment.json` and stops with a message if that file is
missing. A command skips any run whose output already exists, so an interrupted
command can be restarted. Every experiment command accepts `--layout ID`
(repeatable) to split the layouts across jobs; run it again without `--layout`
once every layout is complete. The confirmatory phases also accept
`--analyse-only`, which analyses existing outputs without running.

`run_experiment.py` runs single units, for example
`python run_experiment.py --bank confirmatory --layouts confirmatory_s010000 --cells kappa --out-json runs/unit.json`.

## Compute

Each unit is one layout, condition and seed, run for 20 episodes of 120 steps. A
unit takes about 85 s on one NVIDIA RTX 4090 GPU and about 35 to 45 min on one
CPU core, after compilation. The experiments contain:

| Command | Units |
|---|---|
| `main` | 12 layouts × 4 conditions × 8 seeds = 384 |
| `coverage` | 384, each 30 episodes of 150 steps |
| `calibration` | 72 layouts × 1 condition × 4 seeds = 288 |
| `evaluation` | 72 layouts × 4 conditions × 8 seeds = 2,304 |

Every unit runs as its own job with its own seed index, so the units can be
spread across machines with `--layout` and the result does not depend on how
they are grouped.

## How the reported values were computed

The survival runs on the original layouts and both confirmatory phases ran on
x86 CPUs, one unit per job, as `reproduce.py` does.

Every value in Table S2, the learned-model rows and the coverage row, comes from
an earlier learning runner, run on an NVIDIA GPU, which scored the learned
likelihoods against the true ones. That run differs from `reproduce.py coverage`
in three ways: hunger and thirst had levels 0 to 15 rather than 0 to 10, only the
baseline and dynamic-C conditions were run, and its rule for counting a column as
visited is not part of this release. `reproduce.py coverage` computes the same
measures with the released model. It is not expected to reproduce the recorded
values, and a full run has not yet been compared with them. `reproduce.py` always
uses the CPU.

## Notes

- Condition keys in the code and outputs are `baseuniform`, `kappa`,
  `baseuniform_dynC` and `kappa_dynC`.
- A run file from `run_experiment.py` lists, for each condition, the units'
  `unit_layouts` and `unit_seeds` and one value per unit of `survival_units`
  (mean survived steps), `conflict_units`, `buffer_units` (minimum headroom),
  `posterior_entropy_units` and `visited_state_entropy_units` (per channel),
  `visited_levels_units` (steps at each level) and
  `true_prob_per_state_units`. Each is pooled over the second half of the
  episodes, except the last, which is read from the likelihood learned by the
  final episode.
- Layout IDs `layout0001` and `layout0068` share coordinates. The survival
  statistics average them before inference, so the original experiment has
  eleven distinct layouts. The coverage analysis in `analysis.py` clusters on
  the twelve layout IDs and reports per-condition estimates and the learned-model
  contrasts; it does not compute the interval for the coverage contrast in
  Table S2.
- Dynamic-C computes the hunger and thirst deficits against a maximum of 15,
  while the body's maximum for both is 10, so each deficit lies between 5 and 15.
  The critical threshold is 14.
- The learned-model measure, `true_prob_per_state_units`, is learned-model
  quality as the paper defines it: at each level of each channel, the learned
  probability of the observation the true likelihood ranks highest. The true
  likelihood peaks at the need level the body reports, whatever the precision, so
  the measure does not depend on κ. An earlier release scored against the agent's
  initial, uniform likelihood instead. The recorded Table S2 values were computed
  against the true likelihood.

## Scope

The release covers the experiments in the table above. It does not contain the
code for the checks in Section 4.4 of the paper (true body levels, the penalty
sweep, removing the threshold, held-out seeds and the belief-only κ variant),
the five-and-six layout split, or the conflict rates on the original layouts.

## Licence

MIT. See `LICENSE`.
