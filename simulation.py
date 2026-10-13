"""The learning runner: episodes of the agent in the gridworld, simulated in JAX.

One episode is a ``lax.scan`` over steps. Episodes are chained with a second
``lax.scan`` whose carry is the Dirichlet counts of the interoceptive
likelihoods, so learning carries over between episodes, and the chain is
mapped over seeds with ``vmap``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.tree_util as jtu
from jax import lax
from jax import random as jr

from agent import Agent
import environment
from environment import NO_CHANNEL, Resource, kappa_allocation

# Dynamic C constants. The believed deficits are measured from a maximum of 15
# for hunger and thirst, not from the body's maxima of 10, so a deficit is at
# least 5 and exceeds the critical value 14 only when the believed level is
# below 1.
DYN_C_MAX_LEVEL = 15.0
DYN_C_CRITICAL = 14.0
DYN_C_CRITICAL_PENALTY = -500.0
DYN_C_EMPTY_BIAS = -1.0
DYN_C_POISON_PENALTY = -4.0

# Every episode starts in context 0, which is held fixed.
START_CONTEXT = 0

# A channel is critical when its headroom (level / max_level) is at or below this.
CRIT_FRAC = 0.34


@dataclass(frozen=True)
class RunnerConfig:
    """Settings of one condition and run.

    ``kappa_force_uniform`` and ``dynamic_resource_preferences`` are the two
    factors of the 2x2 design.
    """

    max_steps: int
    start_pos: int
    kappa_force_uniform: bool
    dynamic_resource_preferences: bool


class EpisodeSummary(eqx.Module):
    """Summary of one episode. M is the number of channels and Lmax the largest number of levels.

    Headroom of a channel is its level as a fraction of its maximum, measured at
    the start of each step. ``conflict_steps`` counts steps with at least two
    channels at headroom <= ``CRIT_FRAC``. The learned-model field scores the
    final learned interoceptive likelihoods against the true ones.
    """

    survival_time: jnp.ndarray  # int32 (), steps survived
    conflict_steps: jnp.ndarray  # int32 ()
    min_buffer_sum: jnp.ndarray  # float32 (), minimum headroom across channels summed over steps
    qs_entropy_sum: jnp.ndarray  # float32 (M,), entropy of the posterior over each level, summed over steps
    visited_state_histogram: jnp.ndarray  # int32 (M, NEED_MAX + 1), start-of-step levels clipped to NEED_MAX
    # Per level s of channel m (column s of A^(m)), right-padded to Lmax:
    visited_levels_histogram: jnp.ndarray  # int32 (M, Lmax), steps spent at level s
    # Learned probability of the observation the true likelihood ranks highest in
    # column s, the need level the body reports for level s.
    true_prob_per_state: jnp.ndarray  # float32 (M, Lmax)


class _Carry(eqx.Module):
    """``lax.scan`` carry of one episode."""

    env_state: environment.EnvState
    env_key: jnp.ndarray
    pos_obs: jnp.ndarray
    res_obs: jnp.ndarray
    context_cue: jnp.ndarray
    levels: jnp.ndarray
    body_key: jnp.ndarray
    need_obs: jnp.ndarray
    qs: list[jnp.ndarray]
    prev_action: jnp.ndarray
    pA_counts: list[Optional[jnp.ndarray]]
    alive: jnp.ndarray
    steps: jnp.ndarray
    current_kappa: jnp.ndarray
    conflict_steps: jnp.ndarray
    min_buffer_sum: jnp.ndarray
    qs_entropy_sum: jnp.ndarray
    visited_state_histogram: jnp.ndarray
    visited_levels_histogram: jnp.ndarray


def _decode_actions(pymdp_agent, action: jnp.ndarray) -> jnp.ndarray:
    """Decode flat sampled actions into (Move, Interact), shape (batch, 2)."""
    out = jnp.zeros((int(pymdp_agent.batch_size), len(pymdp_agent.num_controls_multi)), dtype=action.dtype)
    for f, action_map in enumerate(pymdp_agent.action_maps):
        deps = list(action_map["multi_dependency"])
        if not deps:
            continue
        x_rev = []
        rest = action[..., f].astype(jnp.int32)
        for base in reversed(action_map["multi_dims"]):
            b = jnp.asarray(int(base), dtype=jnp.int32)
            x_rev.append(jnp.mod(rest, b))
            rest = jnp.floor_divide(rest, b)
        decoded = jnp.stack(list(reversed(x_rev)), axis=-1).astype(action.dtype)
        out = out.at[..., deps].set(decoded)
    return out


def _normalise_counts(counts: jnp.ndarray) -> jnp.ndarray:
    """Normalise Dirichlet counts over the observation axis."""
    return counts / jnp.maximum(jnp.sum(counts, axis=0, keepdims=True), 1e-12)


def _latest_marginal(q: jnp.ndarray) -> jnp.ndarray:
    """Latest normalised marginal of one factor; ``q`` is (batch, time, states)."""
    v = jnp.asarray(q, dtype=jnp.float32)[0, -1, :]
    return v / jnp.maximum(v.sum(), 1e-8)


def _expected_deficit(q: jnp.ndarray) -> jnp.ndarray:
    """Expected deficit fraction E_q[(L - s) / L] under a belief q over levels 0..L."""
    probs = jnp.maximum(jnp.asarray(q, dtype=jnp.float32).reshape((-1,)), 1e-8)
    probs = probs / jnp.maximum(jnp.sum(probs), 1e-8)
    frac = jnp.arange(probs.shape[0], dtype=jnp.float32) / jnp.asarray(
        float(max(1, int(probs.shape[0]) - 1)), dtype=jnp.float32)
    return jnp.sum(probs * (1.0 - frac))


def make_episode_runner(*, env_params: environment.EnvParams, body_params: environment.BodyParams,
                        agent: Agent, config: RunnerConfig) -> Callable:
    """Single-episode runner ``(env_key, body_key, pA) -> (EpisodeSummary, pA_final)``."""
    pymdp = agent.pymdp
    max_steps = int(config.max_steps)
    specs = tuple(body_params.specs)
    n_channels = len(specs)
    n_extero_states = len(pymdp.num_states) - n_channels
    n_extero_obs = len(pymdp.num_obs) - n_channels
    n_pos, n_res = int(pymdp.num_obs[0]), int(pymdp.num_obs[1])
    n_need = int(pymdp.num_obs[n_extero_obs])
    level_sizes = [int(pymdp.num_states[n_extero_states + c]) for c in range(n_channels)]
    lmax = max(level_sizes)
    intero_modalities = range(n_extero_obs, n_extero_obs + n_channels)

    headroom_scale = jnp.asarray([1.0 / max(int(s.max_level), 1) for s in specs], dtype=jnp.float32)

    # One-hot observations of shape (n_levels, 1, n_levels), the layout pymdp expects.
    oh_pos = jnp.eye(n_pos, dtype=jnp.float32)[:, None, :]
    oh_res = jnp.eye(n_res, dtype=jnp.float32)[:, None, :]
    oh_need = jnp.eye(n_need, dtype=jnp.float32)[:, None, :]

    # The agent's initial likelihoods. The exteroceptive ones stay fixed; the
    # interoceptive ones start uniform and are replaced by the learned counts at
    # run time.
    A_init = [jnp.asarray(a[0], dtype=jnp.float32) for a in pymdp.A]
    C_base = [jnp.asarray(c[0], dtype=jnp.float32) for c in pymdp.C]

    def _learned_A(pA: list[Optional[jnp.ndarray]]) -> list[jnp.ndarray]:
        """Likelihoods with each learned modality replaced by its normalised counts."""
        return [A_m if alpha is None else _normalise_counts(alpha) for A_m, alpha in zip(A_init, pA)]

    def _kappa_focused_A(A: list[jnp.ndarray], attended: jnp.ndarray) -> list[jnp.ndarray]:
        """Interoceptive likelihoods sharpened by the kappa allocation for channel ``attended``.

        Column s of A^(m) puts kappa_m on the observation that the current A^(m)
        ranks highest for s and (1 - kappa_m) / (n - 1) on every other one.
        """
        kappa = kappa_allocation(
            attended=jnp.asarray(attended, dtype=jnp.int32),
            n_channels=n_channels,
            att_cap=body_params.kappa_attended_cap,
            floor=body_params.kappa_floor,
            budget=body_params.kappa_budget,
        )
        out = list(A)
        for c, m in enumerate(intero_modalities):
            n_obs = int(A[m].shape[0])
            k = jnp.clip(kappa[c], 0.0, 1.0)
            target = jnn.one_hot(jnp.argmax(A[m], axis=0).astype(jnp.int32), n_obs, dtype=jnp.float32)
            off = jnp.asarray(1.0 / float(n_obs - 1), dtype=jnp.float32) * (1.0 - k)
            out[m] = off + (k - off) * target.T
        return out

    def _dirichlet_update(pA, observation, qs):
        """a^(m) += o^(m) (outer) q(s^(m)) for each learned modality, floored at 1e-8."""
        out = list(pA)
        for m, alpha in enumerate(pA):
            if alpha is None:
                continue
            (factor,) = pymdp.A_dependencies[m]
            q = jnp.maximum(jnp.asarray(qs[factor], dtype=jnp.float32)[0, -1], 1e-8)
            q = q / jnp.maximum(jnp.sum(q), 1e-8)
            o = jnp.asarray(observation[m], dtype=jnp.float32).reshape((int(alpha.shape[0]),))
            out[m] = jnp.maximum(alpha + o[:, None] * q[None, :], 1e-8)
        return out

    dyn_c = bool(config.dynamic_resource_preferences)
    dyn_max = jnp.asarray(DYN_C_MAX_LEVEL, dtype=jnp.float32)
    dyn_critical = jnp.asarray(DYN_C_CRITICAL, dtype=jnp.float32)
    dyn_penalty = jnp.asarray(DYN_C_CRITICAL_PENALTY, dtype=jnp.float32)
    dyn_empty = jnp.asarray(DYN_C_EMPTY_BIAS, dtype=jnp.float32)
    dyn_poison = jnp.asarray(DYN_C_POISON_PENALTY, dtype=jnp.float32)
    # Channels relieved at a resource tile; oxygen recovers on any dry tile and takes no part.
    channel_index = {spec.name: c for c, spec in enumerate(specs)}
    dyn_participants = [(channel_index["hunger"], int(Resource.FOOD)),
                        (channel_index["thirst"], int(Resource.WATER))]

    def _dynamic_C(qs: list[jnp.ndarray]) -> list[jnp.ndarray]:
        """Dynamic C: the resource preferences C^res set from the believed deficits.

        The deficit of hunger (thirst) is t = max(0, 15 - E_q[level]) and sets
        C^res[food] (C^res[water]). AIR and SLEEP get -1 and POISON -4. When a
        deficit exceeds 14, the other resource and AIR and SLEEP get -500.
        """
        deficits = []
        for c, _ in dyn_participants:
            q_c = _latest_marginal(qs[n_extero_states + c])
            expected = jnp.sum(q_c * jnp.arange(q_c.shape[0], dtype=jnp.float32), axis=-1)
            deficits.append(jnp.maximum(0.0, dyn_max - expected))
        pref = {res: t for (_, res), t in zip(dyn_participants, deficits)}
        empty = dyn_empty
        for (_, res), t in zip(dyn_participants, deficits):
            urgent = t > dyn_critical
            empty = jnp.where(urgent, dyn_penalty, empty)
            for other in pref:
                if other != res:
                    pref[other] = jnp.where(urgent, dyn_penalty, pref[other])

        c_res = jnp.zeros_like(C_base[1], dtype=jnp.float32)
        for slot in (Resource.AIR, Resource.FOOD, Resource.WATER, Resource.SLEEP):
            c_res = c_res.at[int(slot)].set(empty)
        for res, value in pref.items():
            c_res = c_res.at[res].set(value)
        c_res = c_res.at[int(Resource.POISON)].set(dyn_poison)
        C = list(C_base)
        C[1] = c_res
        return C

    def simulate_episode(env_key, body_key, pA_start):
        """Run one episode from the layout's start position; return its summary and final counts."""
        env_key_reset, env_key = jr.split(env_key)
        env_state0, env_obs0 = environment.reset(env_params, pos=int(config.start_pos), ctx=START_CONTEXT,
                                             key=env_key_reset)

        # The first interoceptive observation has no attended channel.
        levels0 = environment.initial_levels(body_params)
        body_key_intero0, body_key = jr.split(body_key)
        need_obs0, _ = environment.sample_interoception(
            body_params, levels0, attended=jnp.asarray(-1, dtype=jnp.int32), key=body_key_intero0)

        carry0 = _Carry(
            env_state=env_state0,
            env_key=env_key,
            pos_obs=env_obs0.pos_obs,
            res_obs=env_obs0.res_obs,
            context_cue=env_obs0.context_cue,
            levels=levels0,
            body_key=body_key,
            need_obs=need_obs0,
            # Placeholders; at t == 0 the prior is D.
            qs=jtu.tree_map(lambda d: d[:, None, :], pymdp.D),
            prev_action=jnp.zeros((int(pymdp.batch_size), int(pymdp.num_factors)), dtype=jnp.int32),
            pA_counts=pA_start,
            alive=jnp.asarray(True),
            steps=jnp.asarray(0, dtype=jnp.int32),
            current_kappa=jnp.asarray(NO_CHANNEL, dtype=jnp.int32),
            conflict_steps=jnp.asarray(0, dtype=jnp.int32),
            min_buffer_sum=jnp.asarray(0.0, dtype=jnp.float32),
            qs_entropy_sum=jnp.zeros((n_channels,), dtype=jnp.float32),
            visited_state_histogram=jnp.zeros((n_channels, n_need), dtype=jnp.int32),
            visited_levels_histogram=jnp.zeros((n_channels, lmax), dtype=jnp.int32),
        )

        def step_alive(carry: _Carry, t: jnp.ndarray) -> _Carry:
            """Infer states, select kappa, plan and act, then step the environment and the body."""
            need_obs = carry.need_obs.astype(jnp.int32)
            observation = [
                oh_pos[carry.pos_obs.astype(jnp.int32)],
                oh_res[carry.res_obs.astype(jnp.int32)],
                carry.context_cue.astype(jnp.float32)[None, :],
            ] + [oh_need[need_obs[c]] for c in range(n_channels)]

            prior = lax.cond(
                t == 0,
                lambda: pymdp.D,
                lambda: pymdp.update_empirical_prior(carry.prev_action, carry.qs),
            )
            # State inference uses the learned A focused on the channel attended at the previous step.
            attended_prev = jnp.where(carry.current_kappa == NO_CHANNEL, -1, carry.current_kappa).astype(jnp.int32)
            qs = agent.infer_states(observation, prior, _kappa_focused_A(_learned_A(carry.pA_counts), attended_prev))
            pA_next = _dirichlet_update(carry.pA_counts, observation, qs)

            # Kappa selection: attend to the channel with the largest expected deficit.
            beliefs = [_latest_marginal(qs[n_extero_states + c]) for c in range(n_channels)]
            selected = jnp.argmax(jnp.stack([_expected_deficit(q) for q in beliefs], axis=0)).astype(jnp.int32)
            if config.kappa_force_uniform:
                selected = jnp.asarray(NO_CHANNEL, dtype=jnp.int32)
            attended = jnp.where(selected == NO_CHANNEL, -1, selected).astype(jnp.int32)

            # Planning uses the likelihood learned up to and including this step.
            q_pi = agent.infer_policies(
                qs,
                A_runtime=_kappa_focused_A(_learned_A(pA_next), attended),
                C_runtime=_dynamic_C(qs) if dyn_c else None,
            )
            action = pymdp.sample_action(q_pi)
            action_multi = _decode_actions(pymdp, action)
            move = action_multi[0, 0].astype(jnp.int32)
            interact = action_multi[0, 1].astype(jnp.int32)

            env_key_step, env_key_next = jr.split(carry.env_key)
            env_state1, env_obs1, tile_res = environment.step(env_params, carry.env_state, move=move, key=env_key_step)

            _, body_key_intero, body_key_next = jr.split(carry.body_key, 3)
            levels1, dead = environment.tick(body_params, carry.levels, interact=interact, tile_res=tile_res)
            need_obs1, _ = environment.sample_interoception(body_params, levels1, attended=attended,
                                                         key=body_key_intero)

            # Headroom and conflict from the true levels at the start of the step.
            start_levels = carry.levels.astype(jnp.float32)
            headroom = jnp.clip(headroom_scale * start_levels, 0.0, 1.0)
            n_critical = jnp.sum((headroom <= CRIT_FRAC).astype(jnp.int32))
            qs_entropy = jnp.stack(
                [-jnp.sum(q * jnp.log(jnp.clip(q, 1e-16, 1.0))) for q in beliefs], axis=0)
            level_idx = carry.levels.astype(jnp.int32)
            channels = jnp.arange(n_channels)

            return _Carry(
                env_state=env_state1,
                env_key=env_key_next,
                pos_obs=env_obs1.pos_obs,
                res_obs=env_obs1.res_obs,
                context_cue=env_obs1.context_cue,
                levels=levels1,
                body_key=body_key_next,
                need_obs=need_obs1,
                qs=qs,
                prev_action=action,
                pA_counts=pA_next,
                alive=~dead,
                steps=carry.steps + jnp.asarray(1, dtype=jnp.int32),
                current_kappa=selected,
                conflict_steps=carry.conflict_steps + (n_critical >= 2).astype(jnp.int32),
                min_buffer_sum=carry.min_buffer_sum + jnp.min(headroom),
                qs_entropy_sum=carry.qs_entropy_sum + qs_entropy,
                visited_state_histogram=carry.visited_state_histogram.at[
                    channels, jnp.clip(level_idx, 0, n_need - 1)].add(1),
                visited_levels_histogram=carry.visited_levels_histogram.at[
                    channels, jnp.clip(level_idx, 0, lmax - 1)].add(1),
            )

        def step_fn(carry: _Carry, t: jnp.ndarray):
            """Run a step while alive, else keep the carry."""
            return lax.cond(carry.alive, lambda c: step_alive(c, t), lambda c: c, carry), ()

        final, _ = lax.scan(step_fn, carry0, jnp.arange(max_steps, dtype=jnp.int32))

        # The true likelihood of each channel peaks, in every column, at the need level
        # the body reports for that level (environment.need_levels_jax). That holds for
        # any kappa above 1 / (NEED_MAX + 1), so the reference does not depend on kappa.
        true_prob_rows = []
        for c, m in enumerate(intero_modalities):
            learned = _normalise_counts(final.pA_counts[m])
            n_states = int(learned.shape[1])
            max_level = float(specs[c].max_level)
            states = jnp.arange(n_states, dtype=jnp.float32)
            true_top = jnp.rint(environment.NEED_MAX * jnp.clip((max_level - states) / max_level, 0.0, 1.0)).astype(jnp.int32)
            pad = jnp.zeros((lmax - n_states,), dtype=jnp.float32)
            true_prob = jnp.clip(learned[true_top, jnp.arange(n_states, dtype=jnp.int32)], 1e-12, 1.0)
            true_prob_rows.append(jnp.concatenate([true_prob, pad], axis=0) if lmax > n_states else true_prob)

        summary = EpisodeSummary(
            survival_time=final.steps,
            conflict_steps=final.conflict_steps,
            min_buffer_sum=final.min_buffer_sum,
            qs_entropy_sum=final.qs_entropy_sum,
            visited_state_histogram=final.visited_state_histogram,
            visited_levels_histogram=final.visited_levels_histogram,
            true_prob_per_state=jnp.stack(true_prob_rows, axis=0),
        )
        return summary, final.pA_counts

    return simulate_episode


def make_learning_runner(*, n_episodes: int, env_params: environment.EnvParams,
                         body_params: environment.BodyParams, agent: Agent, config: RunnerConfig) -> Callable:
    """Learning runner ``(env_keys, body_keys) -> EpisodeSummary``.

    Keys have shape (n_seeds, 2) and summary leaves (n_seeds, n_episodes, ...).
    Episode e of a seed uses ``jr.fold_in`` of the seed's keys with e. Action
    selection is deterministic, so planning draws no random numbers.
    """
    episode = make_episode_runner(env_params=env_params, body_params=body_params, agent=agent, config=config)
    pA0 = [None if alpha is None else jnp.asarray(alpha, dtype=jnp.float32) for alpha in agent.pA]

    def run_seed(env_key, body_key):
        def run_episode(pA, ep_idx):
            summary, pA_final = episode(jr.fold_in(env_key, ep_idx), jr.fold_in(body_key, ep_idx), pA)
            return pA_final, summary

        _, summaries = lax.scan(run_episode, pA0, jnp.arange(int(n_episodes)))
        return summaries

    return jax.vmap(run_seed)
