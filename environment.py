"""The gridworld and the body, as numpy model inputs and as JAX dynamics.

Positions are flat row-major indices on the grid. The enums fix the actions,
the tile resources and the attended-channel code. Every member is kept,
including ones no channel uses, because the enum sizes set array shapes and
their integer values index those arrays.

Each interoceptive channel has a latent level in ``0..max_level`` that is
lethal at 0. The need level the agent observes is the deficit fraction
``(max_level - level) / max_level`` rounded onto ``0..NEED_MAX``. The same
per-step dynamics drive the simulated body (JAX) and the agent's transition
model B (numpy). In JAX the body state is the (M,) int32 vector of channel
levels; the channel specs are Python constants, so the loops over channels
unroll under tracing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple, Sequence

import jax.numpy as jnp
import numpy as np
from jax import lax
from jax import random as jr

# States of the agent's context factor. The context is held fixed in every
# experiment, but the factor keeps its four states in the generative model.
N_CONTEXTS = 4


class Resource(IntEnum):
    """Tile resources, as observed by the agent."""

    AIR = 0
    WATER = 1
    FOOD = 2
    SLEEP = 3
    POISON = 4


class Move(IntEnum):
    """Moves on the grid; moves off the edge leave the position unchanged."""

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    REST = 4


class Interact(IntEnum):
    """Interactions with the destination tile."""

    NONE = 0
    BREATHE = 1
    EAT = 2
    DRINK = 3
    SLEEP = 4


# Attended-channel code meaning that no channel is attended.
NO_CHANNEL = 4


class Gridworld:
    """A ``width x height`` grid with one layout of food and water tiles.

    Positions are flat row-major indices. ``trans[move, pos]`` is the position
    reached by ``move``, clamped at the edges. ``res_map[pos, ctx]`` is the
    resource at ``pos`` in context ``ctx``; the context is held fixed, so all
    columns are equal. The centre cell ``hill_idx`` shows the context cue.
    """

    def __init__(self, width: int, height: int, *, start_pos: int,
                 food_positions: Sequence[int], water_positions: Sequence[int]):
        self.w, self.h = int(width), int(height)
        self.n_pos = self.w * self.h
        self.hill_idx = (self.h // 2) * self.w + self.w // 2
        self.start_pos = int(start_pos)

        self.trans = np.empty((len(Move), self.n_pos), np.int64)
        for idx in range(self.n_pos):
            x, y = idx % self.w, idx // self.w
            self.trans[Move.UP, idx] = max(0, y - 1) * self.w + x
            self.trans[Move.DOWN, idx] = min(self.h - 1, y + 1) * self.w + x
            self.trans[Move.LEFT, idx] = y * self.w + max(0, x - 1)
            self.trans[Move.RIGHT, idx] = y * self.w + min(self.w - 1, x + 1)
            self.trans[Move.REST, idx] = idx

        food = tuple(int(p) for p in food_positions)
        water = tuple(int(p) for p in water_positions)
        resources = food + water
        if any(not 0 <= p < self.n_pos for p in resources):
            raise ValueError(f"resource position off the grid: {resources}")
        if len(set(resources)) != len(resources):
            raise ValueError(f"resource positions must be distinct: {resources}")
        if {self.start_pos, self.hill_idx} & set(resources):
            raise ValueError("resource positions must not overlap the start or the hill")

        self.res_map = np.full((self.n_pos, N_CONTEXTS), int(Resource.AIR), dtype=np.int8)
        for p in food:
            self.res_map[p, :] = Resource.FOOD
        for p in water:
            self.res_map[p, :] = Resource.WATER


class EnvParams(NamedTuple):
    """Hill cell, move table (len(Move), n_pos) and resource map (n_pos, N_CONTEXTS)."""

    hill_idx: int
    trans: jnp.ndarray
    res_map: jnp.ndarray


class EnvState(NamedTuple):
    """Position and context (both scalar int32)."""

    pos: jnp.ndarray
    ctx: jnp.ndarray


class EnvObs(NamedTuple):
    """Observed position, observed resource and the one-hot context cue (index 0 for no cue)."""

    pos_obs: jnp.ndarray
    res_obs: jnp.ndarray
    context_cue: jnp.ndarray


def make_params(grid: Gridworld) -> EnvParams:
    """JAX parameters of a gridworld with its layout applied."""
    return EnvParams(
        hill_idx=int(grid.hill_idx),
        trans=jnp.asarray(grid.trans, dtype=jnp.int32),
        res_map=jnp.asarray(grid.res_map, dtype=jnp.int8),
    )


def _observe(params: EnvParams, state: EnvState, *, key: jnp.ndarray) -> tuple[EnvObs, jnp.ndarray]:
    """Sample the exteroceptive observation of ``state``."""
    key_u, key_choice, key_next = jr.split(key, 3)

    pos = state.pos
    ctx = state.ctx
    res_truth = params.res_map[pos, ctx].astype(jnp.int32)

    # The resource is observed correctly with probability 0.95, else as a uniformly drawn wrong label.
    u = jr.uniform(key_u, (), dtype=jnp.float32)

    def _wrong_res() -> jnp.ndarray:
        r = jr.randint(key_choice, (), 0, len(Resource) - 1, dtype=jnp.int32)
        return jnp.where(r >= res_truth, r + 1, r)

    res_obs = lax.cond(u < 0.95, lambda: res_truth, _wrong_res)

    # The context is cued only on the hill cell; elsewhere the cue reads index 0.
    cue = jnp.zeros((N_CONTEXTS + 1,), dtype=jnp.int8)
    cue = lax.cond(
        pos == params.hill_idx,
        lambda: cue.at[(ctx + 1).astype(jnp.int32)].set(jnp.int8(1)),
        lambda: cue.at[0].set(jnp.int8(1)),
    )
    return EnvObs(pos_obs=pos, res_obs=res_obs, context_cue=cue), key_next


def reset(params: EnvParams, *, pos: int, ctx: int, key: jnp.ndarray) -> tuple[EnvState, EnvObs]:
    """Start an episode at ``pos`` in context ``ctx``."""
    _, _, key_obs = jr.split(key, 3)
    state = EnvState(pos=jnp.asarray(int(pos), dtype=jnp.int32), ctx=jnp.asarray(int(ctx), dtype=jnp.int32))
    obs, _ = _observe(params, state, key=key_obs)
    return state, obs


def step(params: EnvParams, state: EnvState, *, move: jnp.ndarray,
         key: jnp.ndarray) -> tuple[EnvState, EnvObs, jnp.ndarray]:
    """Apply ``move``; return the new state, its observation and the destination tile's resource."""
    _, _, _, key_obs = jr.split(key, 4)
    pos = params.trans[move.astype(jnp.int32), state.pos]
    tile_res = params.res_map[pos, state.ctx].astype(jnp.int32)
    new_state = EnvState(pos=pos.astype(jnp.int32), ctx=state.ctx)
    obs, _ = _observe(params, new_state, key=key_obs)
    return new_state, obs, tile_res


NEED_MAX = 5
EAT_GAIN = 3
POISON_LOSS = 4
OXYGEN_RECOVERY = 1


@dataclass(frozen=True)
class BodyConfig:
    """Channel ranges and rates, and the interoceptive precision budget.

    The precision allocation gives the attended channel ``kappa_attended_cap``
    and shares the rest of the budget ``kappa_budget`` (K = M * mean kappa)
    over the other channels, each at least ``kappa_floor``.
    """

    hunger_max: int
    thirst_max: int
    oxygen_max: int
    hunger_decay: int
    thirst_decay: int
    oxygen_loss_in_water: int
    kappa_attended_cap: float
    kappa_floor: float
    kappa_budget: float


@dataclass(frozen=True)
class ChannelSpec:
    """Per-step dynamics of one channel.

    Each step the level changes by ``drift``, then by ``tile_delta[resource]``
    of the destination tile, then by ``refill_amount`` if the interaction is
    ``refill_interact`` on a ``refill_tile`` tile, and is clipped to
    ``0..max_level``.
    """

    name: str
    max_level: int
    drift: int
    tile_delta: tuple[int, ...]
    refill_interact: int = -1
    refill_tile: int = -1
    refill_amount: int = 0


def make_channels(cfg: BodyConfig) -> tuple[ChannelSpec, ...]:
    """Hunger, thirst and oxygen, in that order.

    Hunger falls by ``hunger_decay`` per step and by ``POISON_LOSS`` on poison,
    and each EAT on food adds ``EAT_GAIN``. Thirst falls by ``thirst_decay``
    per step and each DRINK on water adds a third of its range. Oxygen falls by
    ``oxygen_loss_in_water`` on water and recovers by ``OXYGEN_RECOVERY`` on
    any other tile, so drinking trades against air.
    """
    no_tile_effect = (0,) * len(Resource)
    hunger_tiles = list(no_tile_effect)
    hunger_tiles[Resource.POISON] = -POISON_LOSS
    oxygen_tiles = [OXYGEN_RECOVERY] * len(Resource)
    oxygen_tiles[Resource.WATER] = -int(cfg.oxygen_loss_in_water)
    hunger = ChannelSpec(
        "hunger", int(cfg.hunger_max), -int(cfg.hunger_decay), tuple(hunger_tiles),
        refill_interact=int(Interact.EAT), refill_tile=int(Resource.FOOD), refill_amount=EAT_GAIN,
    )
    thirst = ChannelSpec(
        "thirst", int(cfg.thirst_max), -int(cfg.thirst_decay), no_tile_effect,
        refill_interact=int(Interact.DRINK), refill_tile=int(Resource.WATER),
        refill_amount=max(1, int(cfg.thirst_max) // 3),
    )
    oxygen = ChannelSpec("oxygen", int(cfg.oxygen_max), 0, tuple(oxygen_tiles))
    return (hunger, thirst, oxygen)


def next_level_np(spec: ChannelSpec, level: int, *, tile_res: int, interact: int) -> int:
    """Level of one channel after one step (numpy)."""
    x = int(level) + spec.drift + spec.tile_delta[int(tile_res)]
    if int(interact) == spec.refill_interact and int(tile_res) == spec.refill_tile:
        x += spec.refill_amount
    return int(np.clip(x, 0, spec.max_level))


def next_levels_jax(specs, levels: jnp.ndarray, *, tile_res, interact) -> jnp.ndarray:
    """Levels of all channels after one step (JAX)."""
    new = []
    for idx, spec in enumerate(specs):
        hit = (interact == spec.refill_interact) & (tile_res == spec.refill_tile)
        x = levels[idx] + spec.drift + jnp.asarray(spec.tile_delta, dtype=jnp.int32)[tile_res]
        x = jnp.where(hit, x + spec.refill_amount, x)
        new.append(jnp.clip(x, 0, spec.max_level).astype(jnp.int32))
    return jnp.stack(new, axis=0)


def need_levels_jax(specs, levels: jnp.ndarray) -> jnp.ndarray:
    """Need levels ``rint(NEED_MAX * (max_level - level) / max_level)`` of all channels (JAX)."""
    cols = []
    for idx, spec in enumerate(specs):
        max_level = float(spec.max_level)
        frac = jnp.clip((max_level - levels[idx].astype(jnp.float32)) / max_level, 0.0, 1.0)
        cols.append(jnp.rint(frac * float(NEED_MAX)).astype(jnp.int32))
    return jnp.stack(cols, axis=0)


def kappa_allocation(
    *,
    attended: jnp.ndarray,
    n_channels: int,
    att_cap: jnp.ndarray,
    floor: jnp.ndarray,
    budget: jnp.ndarray,
) -> jnp.ndarray:
    """Precision kappa_m of each channel when channel ``attended`` is attended (-1 for none).

    The attended channel gets ``min(att_cap, K - (M - 1) * floor)`` and the
    other M - 1 channels share the rest of the budget K, each at least
    ``floor``. With no attended channel every channel gets K / M. The same
    allocation sets the body's observation noise and the agent's likelihoods.
    """
    n = int(n_channels)
    att = jnp.clip(jnp.asarray(att_cap, dtype=jnp.float32), 0.0, 1.0)
    floor = jnp.clip(jnp.asarray(floor, dtype=jnp.float32), 0.0, 1.0)
    total = jnp.clip(jnp.asarray(budget, dtype=jnp.float32), 0.0, float(n))
    focus = attended.astype(jnp.int32)
    has_focus = (focus >= 0) & (focus < n)

    p_uniform = jnp.full((n,), jnp.clip(total / float(n), 0.0, 1.0), dtype=jnp.float32)

    max_bg = jnp.clip(total / float(n - 1), 0.0, 1.0)
    floor = jnp.minimum(floor, max_bg)

    fg_max = jnp.clip(total - float(n - 1) * floor, 0.0, 1.0)
    fg = jnp.minimum(att, fg_max)
    rem = jnp.maximum(0.0, total - fg)
    bg = rem / float(n - 1)
    bg = jnp.where(bg < floor, floor, bg)
    fg = jnp.where(
        bg <= floor + 1e-12,
        jnp.clip(total - float(n - 1) * bg, 0.0, 1.0),
        fg,
    )
    p = jnp.full((n,), bg, dtype=jnp.float32)
    p = p.at[jnp.clip(focus, 0, n - 1)].set(fg)
    return jnp.where(has_focus, p, p_uniform)


class BodyParams(NamedTuple):
    """Channel specs and the precision budget as float32 scalars."""

    specs: tuple[ChannelSpec, ...]
    kappa_attended_cap: jnp.ndarray
    kappa_floor: jnp.ndarray
    kappa_budget: jnp.ndarray


def make_body(cfg: BodyConfig) -> BodyParams:
    """Body parameters for ``cfg``."""
    return BodyParams(
        specs=make_channels(cfg),
        kappa_attended_cap=jnp.asarray(float(cfg.kappa_attended_cap), dtype=jnp.float32),
        kappa_floor=jnp.asarray(float(cfg.kappa_floor), dtype=jnp.float32),
        kappa_budget=jnp.asarray(float(cfg.kappa_budget), dtype=jnp.float32),
    )


def initial_levels(params: BodyParams) -> jnp.ndarray:
    """Every channel starts full."""
    return jnp.asarray([int(s.max_level) for s in params.specs], dtype=jnp.int32)


def tick(params: BodyParams, levels: jnp.ndarray, *, interact: jnp.ndarray,
         tile_res: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Advance the levels by one step on tile ``tile_res``; return the new levels and whether the body died."""
    levels = next_levels_jax(params.specs, levels, tile_res=tile_res, interact=interact)
    return levels, jnp.any(levels == 0)


def sample_interoception(params: BodyParams, levels: jnp.ndarray, *, attended: jnp.ndarray,
                         key: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Sample the (M,) need observations when channel ``attended`` is attended (-1 for none).

    Channel m reports its true need level with probability kappa_m from
    ``kappa_allocation`` and otherwise a uniformly drawn wrong level.
    """
    specs = params.specs
    m = len(specs)
    true_levels = need_levels_jax(specs, levels)
    keys = jr.split(key, 2 * m + 1)

    kappa = kappa_allocation(
        attended=attended.astype(jnp.int32),
        n_channels=m,
        att_cap=params.kappa_attended_cap,
        floor=params.kappa_floor,
        budget=params.kappa_budget,
    )

    def sample_one(i: int) -> jnp.ndarray:
        true_i = true_levels[i]
        u = jr.uniform(keys[2 * i], (), dtype=jnp.float32)

        def _wrong():
            r = jr.randint(keys[2 * i + 1], (), 0, NEED_MAX, dtype=jnp.int32)
            return jnp.where(r >= true_i, r + 1, r)

        return lax.cond(u < kappa[i], lambda: true_i, _wrong)

    obs = jnp.stack([sample_one(i) for i in range(m)], axis=0).astype(jnp.int32)
    return obs, keys[-1]
