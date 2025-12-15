from __future__ import annotations

from typing import NamedTuple

import jax

Array = jax.Array


class DevelopmentState(NamedTuple):
    age_step: Array  # int32 scalar
    phi: Array  # float32 scalar in [0, 1]


class InternalState(NamedTuple):
    energy: Array  # float32 scalar in [0, 1]
    integrity: Array  # float32 scalar in [0, 1]


class ChemotaxisEnvSpec(NamedTuple):
    width: int
    height: int
    max_steps: int
    energy_init: float
    energy_decay: float
    energy_gain: float
    terminate_on_reach: bool
    obs_noise: float = 0.0
    grad_dropout_p: float = 0.0

    # Developmental “nursing” hooks (sensory gating / resolution).
    #
    # Gradient is multiplied by a phase-scheduled gain:
    #   gain(phi) = lerp(grad_gain_min, grad_gain_max) over [grad_gain_start_phi, grad_gain_end_phi]
    #
    # If grad_bins_{min,max} > 0, the gradient is also quantized to `bins(phi)` steps.
    grad_gain_min: float = 1.0
    grad_gain_max: float = 1.0
    grad_gain_start_phi: float = 0.0
    grad_gain_end_phi: float = 0.0

    grad_bins_min: float = 0.0
    grad_bins_max: float = 0.0
    grad_bins_start_phi: float = 0.0
    grad_bins_end_phi: float = 0.0

    # L0.2: multiple identical sources. The observation gradient targets the nearest source.
    # NOTE: must be >= 1.
    num_sources: int = 1

    # L0.2 variant: some sources are harmful (integrity loss).
    # The agent cannot distinguish source types from the gradient signal alone.
    num_bad_sources: int = 0
    bad_source_integrity_loss: float = 0.0

    # L0.2 cue ablation: when enabled, the gradient points only to good (non-harmful) sources.
    # This is a control condition that removes consequence-driven valence discrimination pressure.
    good_only_gradient: bool = False

    # L1.0: depleting/respawning sources (temporal structure).
    #
    # If enabled, a source becomes inactive when arrived, then respawns at a new random location
    # after `source_respawn_delay` steps. When all sources are inactive, the gradient is zero.
    source_deplete: bool = False
    source_respawn_delay: int = 0


class ChemotaxisEnvState(NamedTuple):
    t: Array  # int32 scalar
    pos: Array  # int32[2]
    source_pos: Array  # int32[K, 2]
    source_is_bad: Array  # bool[K]
    source_active: Array  # bool[K]
    source_respawn_t: Array  # int32[K] countdown (0 => ready)
    energy: Array  # float32 scalar in [0, 1]
    integrity: Array  # float32 scalar in [0, 1]


# Treat environment specs as static configuration in JAX transforms.
#
# This keeps `jax.jit(simulate_lifetime)(..., env_spec, ...)` usable even when
# `env_spec` controls shapes (e.g., `num_sources`).
def _chemotaxis_env_spec_flatten(spec: ChemotaxisEnvSpec):  # type: ignore[name-defined]
    return (), spec


def _chemotaxis_env_spec_unflatten(aux_data, children):  # type: ignore[no-untyped-def]
    del children
    return aux_data


jax.tree_util.register_pytree_node(  # type: ignore[attr-defined]
    ChemotaxisEnvSpec,
    _chemotaxis_env_spec_flatten,
    _chemotaxis_env_spec_unflatten,
)


class EnvLog(NamedTuple):
    reached_source: Array  # bool scalar
    energy_gained: Array  # float32 scalar
    bad_arrivals: Array  # float32 scalar (count)
    integrity_lost: Array  # float32 scalar


class DevConfig(NamedTuple):
    n_neurons: int
    obs_dim: int
    num_actions: int
    edge_index: Array  # int32[E, 2], columns: (src, dst)

    theta: float = 1.0
    tau_m: float = 10.0
    plast_enabled: bool = False
    plast_eta: float = 0.0
    plast_lambda: float = 0.9
    # 0: spike-derived modulator (legacy).
    # 1: drive/reward-derived modulator (drive_delta; consequence-aligned).
    # 2: event-derived modulator (energy_gained - integrity_lost; consequence-aligned).
    modulator_kind: int = 0
    mod_drive_scale: float = 1.0


class AgentParams(NamedTuple):
    obs_w: Array  # float32[N, obs_dim]
    edge_index: Array  # int32[E, 2]
    w0: Array  # float32[E]
    motor_w: Array  # float32[N, A]
    motor_b: Array  # float32[A]
    mod_w: Array  # float32[N]
    v_decay: Array  # float32[N]
    theta: Array  # float32[N]
    plast_enabled: Array  # bool scalar
    plast_eta: Array  # float32 scalar
    plast_lambda: Array  # float32 scalar
    modulator_kind: Array  # int32 scalar
    mod_drive_scale: Array  # float32 scalar


class AgentState(NamedTuple):
    v: Array  # float32[N]
    spike: Array  # float32[N] (0/1)
    w: Array  # float32[E]
    elig: Array  # float32[E]


class AgentLog(NamedTuple):
    spike_rate: Array  # float32 scalar
    modulator: Array  # float32 scalar
    mean_abs_dw: Array  # float32 scalar


class SimConfig(NamedTuple):
    fitness_alpha: float = 1.0
    fitness_beta: float = 10.0
    success_bonus: float = 50.0

    setpoint_energy: float = 1.0
    setpoint_integrity: float = 1.0
    drive_w_energy: float = 1.0
    drive_w_integrity: float = 1.0


class FitnessSummary(NamedTuple):
    fitness_scalar: Array
    t_alive: Array
    energy_gained_total: Array
    integrity_min: Array
    bad_arrivals_total: Array
    integrity_lost_total: Array
    success: Array
    action_entropy: Array
    action_mode_frac: Array
    mean_abs_dw_mean: Array


class MVTConfig(NamedTuple):
    enabled: bool = False
    episodes: int = 2
    steps: int = 64
    min_alive_steps: int = 32
    min_action_entropy: float = 0.0
    min_energy_gained: float = 0.0
    fail_fitness: float = -1e9


class EvalConfig(NamedTuple):
    episodes: int = 4


class ESConfig(NamedTuple):
    pop_size: int = 64
    generations: int = 50
    sigma: float = 0.1
    lr: float = 0.05
