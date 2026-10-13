"""The agent's generative model, its pymdp agent, and the state-inference and planning callables.

Hidden-state factors: 0 position, 1 context, then 2 + m for the level of
channel m. Observation modalities: 0 position, 1 resource, 2 context cue,
then 3 + m for the need level of channel m.

The policy posterior comes from the expected free energy of every policy in
the library. The agent's fixed arrays are closure constants of the planner;
only the posterior and the runtime A and C pass through the traced call.
"""

from __future__ import annotations

import itertools
from functools import partial
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np
from jax import nn, vmap
from pymdp import inference
from pymdp.agent import Agent as PymdpAgent
from pymdp.control import compute_neg_efe_policy_inductive

from environment import N_CONTEXTS, NEED_MAX, ChannelSpec, Gridworld, Interact, Move, Resource, next_level_np

RESOURCE_CUE_ACCURACY = 0.95
CONTEXT_STAY_PROB = 0.95
PREFERENCE_SCALE = 4.0
D_EPSILON = 1e-3

N_EXTERO_FACTORS = 2
N_EXTERO_MODALITIES = 3


def _log_softmax_scores(scores: np.ndarray) -> np.ndarray:
    """log softmax(scores), with 1e-16 added before the log."""
    probs = np.exp(scores - scores.max())
    probs /= probs.sum()
    return np.log(probs + 1e-16)


def build_A_extero(grid: Gridworld) -> list[np.ndarray]:
    """Exteroceptive likelihoods A.

    Position is observed exactly. The resource of the current tile is observed
    correctly with probability 0.95, and otherwise as one of the other
    resources. The context cue reports the context on the hill cell and reads
    0 (no cue) elsewhere.
    """
    n_pos, n_res = grid.n_pos, len(Resource)
    A_pos = np.eye(n_pos, dtype=np.float32)

    A_res = np.zeros((n_res, n_pos, N_CONTEXTS), dtype=np.float32)
    prob_false = (1.0 - RESOURCE_CUE_ACCURACY) / (n_res - 1)
    for p in range(n_pos):
        for s in range(N_CONTEXTS):
            A_res[:, p, s] = prob_false
            A_res[int(grid.res_map[p, s]), p, s] = RESOURCE_CUE_ACCURACY

    A_cue = np.zeros((N_CONTEXTS + 1, n_pos, N_CONTEXTS), dtype=np.float32)
    for p in range(n_pos):
        for s in range(N_CONTEXTS):
            if p == grid.hill_idx:
                A_cue[s + 1, p, s] = 1.0
            else:
                A_cue[0, p, s] = 1.0
    return [A_pos, A_res, A_cue]


def build_B(grid: Gridworld, specs: tuple[ChannelSpec, ...]) -> list[np.ndarray]:
    """Transitions B for every hidden-state factor.

    Position moves deterministically under Move. The context stays with
    probability 0.95 and otherwise advances by one. For channel m, B^(m) gives
    the next level from the current level, position and context under (Move,
    Interact): the level follows the channel dynamics on the destination tile,
    whose resource is marginalised over the next context.
    """
    n_pos = grid.n_pos
    B_pos = np.zeros((n_pos, n_pos, len(Move)), dtype=np.float32)
    for move_idx, move in enumerate(Move):
        for pos in range(n_pos):
            B_pos[int(grid.trans[move, pos]), pos, move_idx] = 1.0

    B_ctx = np.zeros((N_CONTEXTS, N_CONTEXTS), dtype=np.float32)
    for s in range(N_CONTEXTS):
        B_ctx[s, s] = CONTEXT_STAY_PROB
        B_ctx[(s + 1) % N_CONTEXTS, s] = 1.0 - CONTEXT_STAY_PROB

    n_levels = [int(spec.max_level) + 1 for spec in specs]
    B_channels = [
        np.zeros((n, n, n_pos, N_CONTEXTS, len(Move), len(Interact)), dtype=np.float32)
        for n in n_levels
    ]
    next_level = [
        np.array([[[next_level_np(spec, prev, tile_res=tile, interact=act) for prev in range(n)]
                   for act in range(len(Interact))]
                  for tile in range(len(Resource))], dtype=np.int32)
        for spec, n in zip(specs, n_levels)
    ]

    # This loop order, with the next contexts in ascending order and those of
    # zero probability skipped, fixes the float32 accumulation of the marginal.
    for pos, ctx_prev in itertools.product(range(n_pos), range(N_CONTEXTS)):
        for move_idx, move in enumerate(Move):
            dest = int(grid.trans[move, pos])
            res_pairs = []
            for ctx_next in range(N_CONTEXTS):
                p_s = float(B_ctx[ctx_next, ctx_prev])
                if p_s <= 0.0:
                    continue
                res_pairs.append((int(grid.res_map[dest, ctx_next]), p_s))
            for interact_idx in range(len(Interact)):
                for tile_res, p_s in res_pairs:
                    for c, B_c in enumerate(B_channels):
                        for prev in range(n_levels[c]):
                            nxt = int(next_level[c][tile_res, interact_idx, prev])
                            B_c[(nxt, prev, pos, ctx_prev, move_idx, interact_idx)] += p_s

    return [B_pos, B_ctx, *B_channels]


def build_C(grid: Gridworld, n_channels: int) -> list[np.ndarray]:
    """Log preferences C.

    Position and context cue are flat. Poison has score -4 against 0 for
    every other resource. For each channel, C^(m) falls linearly from 0 at need
    level 0 to -4 at the maximum need level. All scores pass through a log
    softmax.
    """
    scores_res = np.zeros(len(Resource), dtype=np.float32)
    scores_res[Resource.POISON] = -PREFERENCE_SCALE
    C_need = _log_softmax_scores(np.linspace(0.0, -PREFERENCE_SCALE, NEED_MAX + 1, dtype=np.float32))
    return [
        np.zeros(grid.n_pos, dtype=np.float32),
        _log_softmax_scores(scores_res),
        np.zeros(N_CONTEXTS + 1, dtype=np.float32),
        *[C_need.copy() for _ in range(n_channels)],
    ]


def build_D(grid: Gridworld, specs: tuple[ChannelSpec, ...]) -> list[np.ndarray]:
    """Initial-state priors D.

    Position and context are uniform. Each channel prior puts 1 - 1e-3 on the
    full level, where the body starts, and spreads 1e-3 over the other levels.
    """
    D = [
        np.ones(grid.n_pos, dtype=np.float32) / grid.n_pos,
        np.ones(N_CONTEXTS, dtype=np.float32) / N_CONTEXTS,
    ]
    for spec in specs:
        n = int(spec.max_level) + 1
        d = np.full(n, D_EPSILON / (n - 1), dtype=np.float32)
        d[int(spec.max_level)] = 1.0 - D_EPSILON
        D.append(d)
    return D


def _flatten_action_dims(B: list[np.ndarray], b_action_dependencies: list[list[int]],
                         num_controls: list[int]) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Merge each factor's control axes into one flat action axis, as pymdp expects."""
    action_maps: list[dict[str, Any]] = []
    B_flat: list[np.ndarray] = []
    for factor_idx, (B_f, deps) in enumerate(zip(B, b_action_dependencies)):
        if not deps:
            B_flat.append(np.expand_dims(B_f, axis=-1))
            action_maps.append({"multi_dependency": [], "multi_dims": [],
                                "flat_dependency": [factor_idx], "flat_dims": [1]})
            continue
        dims = [int(num_controls[d]) for d in deps]
        flat_dim = int(np.prod(np.asarray(dims, dtype=np.int64)))
        B_flat.append(B_f.reshape(list(B_f.shape[: -len(deps)]) + [flat_dim]))
        action_maps.append({"multi_dependency": list(deps), "multi_dims": dims,
                            "flat_dependency": [factor_idx], "flat_dims": [flat_dim]})
    return B_flat, action_maps


def _policy_library(horizon: int, action_maps: list[dict[str, Any]]) -> np.ndarray:
    """Every sequence of (Move, Interact) pairs over ``horizon`` steps, as flat actions per factor.

    Returns shape (n_policies, horizon, n_factors); the last step varies fastest.
    """
    pairs = np.stack(
        [g.reshape(-1) for g in np.meshgrid(np.arange(len(Move)), np.arange(len(Interact)), indexing="ij")],
        axis=-1,
    ).astype(np.int32)
    n_total = len(pairs) ** int(horizon)
    multi = np.empty((n_total, int(horizon), 2), dtype=np.int32)
    repeat = 1
    for t in reversed(range(int(horizon))):
        block = np.repeat(pairs, repeats=repeat, axis=0)
        multi[:, t, :] = np.tile(block, (n_total // (len(pairs) * repeat), 1))
        repeat *= len(pairs)

    flat = []
    for action_map in action_maps:
        deps = action_map["multi_dependency"]
        if not deps:
            flat.append(np.zeros_like(multi[..., 0], dtype=np.int32))
            continue
        dims = action_map["multi_dims"]
        combos = multi[..., deps]
        idx = np.zeros(multi.shape[:-1], dtype=np.int32)
        product = 1
        for dep_idx in reversed(range(len(dims))):
            idx += combos[..., dep_idx] * product
            product *= dims[dep_idx]
        flat.append(idx)
    return np.stack(flat, axis=-1).astype(np.int32, copy=False)


class Agent:
    """Generative model and pymdp agent shared by all four conditions.

    The exteroceptive likelihoods are fixed. Each interoceptive likelihood
    A^(m) is learned from Dirichlet counts a^(m) that start at
    ``init_concentration`` everywhere, so A^(m) starts uniform. Kappa routing
    and dynamic C act inside the simulation, so the conditions share this
    object. Exposes ``pA`` (initial counts, None for the fixed modalities),
    ``pymdp`` (the pymdp agent), ``infer_states(observation, prior, A)`` and
    ``infer_policies(qs, A_runtime, C_runtime)``.
    """

    def __init__(self, grid: Gridworld, specs: tuple[ChannelSpec, ...], *,
                 planning_horizon: int, num_iter: int, init_concentration: float):
        n_channels = len(specs)
        A = [np.array(a, dtype=np.float32, copy=True) for a in build_A_extero(grid)]
        self.pA: list[np.ndarray | None] = [None] * N_EXTERO_MODALITIES
        for spec in specs:
            counts = np.full((NEED_MAX + 1, int(spec.max_level) + 1), float(init_concentration), dtype=np.float32)
            self.pA.append(counts)
            den = np.maximum(counts.sum(axis=0, keepdims=True), 1e-12)
            A.append((counts / den).astype(np.float32, copy=False))

        B = build_B(grid, specs)
        C = build_C(grid, n_channels)
        D = build_D(grid, specs)

        # Each channel's level transition depends on its own level and on
        # position and context, which set the destination tile's resource.
        channel_factors = range(N_EXTERO_FACTORS, N_EXTERO_FACTORS + n_channels)
        A_dependencies = [[0], [0, 1], [0, 1]] + [[f] for f in channel_factors]
        B_dependencies = [[0], [1]] + [[f, 0, 1] for f in channel_factors]
        B_action_dependencies = [[0], []] + [[0, 1] for _ in channel_factors]
        num_controls = [len(Move), len(Interact)]

        B_flat, action_maps = _flatten_action_dims(B, B_action_dependencies, num_controls)
        self.pymdp = PymdpAgent(
            A=A,
            B=B_flat,
            C=C,
            D=D,
            policies=_policy_library(planning_horizon, action_maps),
            A_dependencies=A_dependencies,
            B_dependencies=B_dependencies,
            B_action_dependencies=B_action_dependencies,
            num_controls=num_controls,
            policy_len=int(planning_horizon),
            categorical_obs=True,
            action_selection="deterministic",
            sampling_mode="full",
            inference_algo="fpi",
            use_states_info_gain=True,
            num_iter=int(num_iter),
            batch_size=1,
        )
        # The simulation decodes flat actions back to (Move, Interact) with these maps.
        object.__setattr__(self.pymdp, "action_maps", action_maps)
        self.infer_policies = make_efe_infer_policies(self.pymdp)

    def infer_states(self, observation, prior, A_runtime):
        """Posterior q(s) by fixed-point iteration under the likelihoods ``A_runtime``."""
        agent = self.pymdp
        A_batched = []
        for A_m, A_base in zip(A_runtime, agent.A):
            A_m = jnp.asarray(A_m, dtype=jnp.float32)
            if A_m.shape != A_base.shape[1:]:
                raise ValueError(f"runtime A shape {A_m.shape} does not match {A_base.shape[1:]}")
            A_batched.append(A_m[None, ...])
        infer = partial(
            inference.update_posterior_states,
            A_dependencies=agent.A_dependencies,
            B_dependencies=agent.B_dependencies,
            num_iter=int(agent.num_iter),
            method=str(agent.inference_algo),
            distr_obs=True,
            inference_horizon=agent.inference_horizon,
            return_info=False,
        )
        return vmap(infer)(A_batched, agent.B, observation, None, prior=prior, qs_hist=None, valid_steps=None)


def _batched(runtime: list[Any], shapes: tuple[tuple[int, ...], ...], batch_size: int,
             label: str) -> tuple[jnp.ndarray, ...]:
    """Add the leading batch axis to each modality of a runtime A or C."""
    if len(runtime) != len(shapes):
        raise ValueError(f"{label}: got {len(runtime)} modalities, expected {len(shapes)}")
    out = []
    for arr, shape in zip(runtime, shapes):
        a = jnp.asarray(arr, dtype=jnp.float32)
        if a.shape != shape:
            raise ValueError(f"{label}: shape {tuple(a.shape)} does not match {shape}")
        out.append(jnp.broadcast_to(a[None, ...], (int(batch_size),) + shape))
    return tuple(out)


def make_efe_infer_policies(pymdp_agent: Any) -> Callable[..., jnp.ndarray]:
    """Return ``infer_policies(qs, A_runtime, C_runtime=None) -> q_pi`` for ``pymdp_agent``.

    -G(pi) is the expected utility under C plus the state information gain
    under A (no parameter information gain), and q(pi) = softmax(gamma * -G(pi)
    + ln E). ``C_runtime`` None means the agent's own C.
    """
    batch_size = int(pymdp_agent.batch_size)

    # Closure constants keep the leading batch axis; the outer vmap slices it.
    B_default = tuple(jnp.asarray(b) for b in pymdp_agent.B)
    C_default = tuple(jnp.asarray(c) for c in pymdp_agent.C)
    E_default = jnp.asarray(pymdp_agent.E)
    I_const = tuple(jnp.asarray(x) for x in pymdp_agent.I)

    policy_matrix = jnp.asarray(pymdp_agent.policies.policy_arr)
    A_deps = tuple(tuple(d) for d in pymdp_agent.A_dependencies)
    B_deps = tuple(tuple(d) for d in pymdp_agent.B_dependencies)
    gamma_vec = jnp.asarray(pymdp_agent.gamma)
    inductive_eps_vec = jnp.asarray(pymdp_agent.inductive_epsilon)

    use_utility = bool(pymdp_agent.use_utility)
    use_states_info_gain = bool(pymdp_agent.use_states_info_gain)
    use_param_info_gain = bool(pymdp_agent.use_param_info_gain)
    use_inductive = bool(pymdp_agent.use_inductive)

    A_shapes = tuple(tuple(jnp.asarray(a).shape[1:]) for a in pymdp_agent.A)
    C_shapes = tuple(tuple(jnp.asarray(c).shape[1:]) for c in pymdp_agent.C)

    def _compute_one(qs_latest_b, A_list_b, B_list_b, C_list_b, I_list_b, E_vec_b, gamma_b, ind_eps_b):
        def score(policy_i):
            return compute_neg_efe_policy_inductive(
                list(qs_latest_b),
                list(A_list_b),
                list(B_list_b),
                list(C_list_b),
                None,
                None,
                list(A_deps),
                list(B_deps),
                list(I_list_b),
                policy_i,
                inductive_epsilon=ind_eps_b,
                use_utility=use_utility,
                use_states_info_gain=use_states_info_gain,
                use_param_info_gain=use_param_info_gain,
                use_inductive=use_inductive,
            )

        neg_efe = vmap(score)(policy_matrix)
        q_pi = nn.softmax(gamma_b * neg_efe + jnp.log(E_vec_b + 1e-16), axis=-1)
        # The softmax output is renormalised once more; removing this changes the last bits of q_pi.
        return q_pi / jnp.clip(q_pi.sum(), 1e-8, None)

    def infer_policies(qs, A_runtime, C_runtime=None):
        """Posterior over policies, shape (batch, n_policies), scored from the latest step of ``qs``."""
        A_list = _batched(A_runtime, A_shapes, batch_size, "runtime A")
        C_list = C_default if C_runtime is None else _batched(C_runtime, C_shapes, batch_size, "runtime C")
        qs_latest = tuple(q[:, -1] for q in qs)
        return vmap(_compute_one)(
            qs_latest, A_list, B_default, C_list, I_const, E_default, gamma_vec, inductive_eps_vec,
        )

    return infer_policies
