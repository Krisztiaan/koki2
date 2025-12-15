from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from koki2.agent.heuristic import greedy_gradient_action
from koki2.agent.snn import (
    agent_apply_plasticity,
    agent_apply_plasticity_nolog,
    agent_forward,
    agent_forward_nolog,
    agent_init,
)
from koki2.envs.chemotaxis import env_init, env_step
from koki2.genome.direct import DirectGenome, develop
from koki2.types import (
    AgentLog,
    Array,
    ChemotaxisEnvSpec,
    DevConfig,
    DevelopmentState,
    EnvLog,
    FitnessSummary,
    InternalState,
    SimConfig,
)


def drive(internal: InternalState, sim_cfg: SimConfig) -> Array:
    e = internal.energy - jnp.array(sim_cfg.setpoint_energy, dtype=jnp.float32)
    i = internal.integrity - jnp.array(sim_cfg.setpoint_integrity, dtype=jnp.float32)
    return (
        jnp.array(sim_cfg.drive_w_energy, dtype=jnp.float32) * (e * e)
        + jnp.array(sim_cfg.drive_w_integrity, dtype=jnp.float32) * (i * i)
    )


def _phi(age_step: Array, steps: int) -> Array:
    denom = float(max(steps - 1, 1))
    return age_step.astype(jnp.float32) / jnp.array(denom, dtype=jnp.float32)


def simulate_lifetime(
    genome: DirectGenome,
    env_spec: ChemotaxisEnvSpec,
    dev_cfg: DevConfig,
    sim_cfg: SimConfig,
    rng: Array,
    t_idx: Array,
) -> FitnessSummary:
    steps = t_idx.shape[0]
    rng_develop, rng_env, rng_agent, rng_rollout = jax.random.split(rng, 4)
    params = develop(genome, dev_cfg, rng_develop)

    env_state, obs, internal = env_init(env_spec, rng_env)
    agent_state = agent_init(params, rng_agent)

    drive_prev = drive(internal, sim_cfg)
    alive = jnp.array(True, dtype=jnp.bool_)
    energy_gained_total = jnp.array(0.0, dtype=jnp.float32)
    bad_arrivals_total = jnp.array(0.0, dtype=jnp.float32)
    integrity_lost_total = jnp.array(0.0, dtype=jnp.float32)
    integrity_min = internal.integrity
    t_alive = jnp.array(0, dtype=jnp.int32)
    success = jnp.array(False, dtype=jnp.bool_)
    num_actions = params.motor_b.shape[0]
    action_counts = jnp.zeros((num_actions,), dtype=jnp.float32)
    dw_sum = jnp.array(0.0, dtype=jnp.float32)

    init = (
        env_state,
        obs,
        internal,
        agent_state,
        drive_prev,
        alive,
        energy_gained_total,
        bad_arrivals_total,
        integrity_lost_total,
        integrity_min,
        t_alive,
        success,
        action_counts,
        dw_sum,
    )

    def step(carry, t):
        (
            env_state,
            obs,
            internal,
            agent_state,
            drive_prev,
            alive,
            energy_gained_total,
            bad_arrivals_total,
            integrity_lost_total,
            integrity_min,
            t_alive,
            success,
            action_counts,
            dw_sum,
        ) = carry

        dev_state = DevelopmentState(age_step=t, phi=_phi(t, steps))
        step_key = jax.random.fold_in(rng_rollout, t)
        rng_agent, rng_env = jax.random.split(step_key, 2)

        def do(_):
            fwd = agent_forward(params, agent_state, obs, internal, dev_state, rng_agent)
            env_state2, obs2, internal2, env_log, done = env_step(
                env_spec, env_state, fwd.action, dev_state, rng_env
            )
            drive2 = drive(internal2, sim_cfg)
            _reward = drive_prev - drive2
            event_delta = env_log.energy_gained - env_log.integrity_lost
            mod_signal_raw = jnp.where(
                params.modulator_kind == jnp.array(1, dtype=jnp.int32),
                _reward,
                jnp.where(
                    params.modulator_kind == jnp.array(2, dtype=jnp.int32),
                    event_delta,
                    jnp.array(0.0, dtype=jnp.float32),
                ),
            )
            agent_state2, agent_log = agent_apply_plasticity(params, agent_state, fwd, mod_signal_raw)

            energy_gained_total2 = energy_gained_total + env_log.energy_gained
            bad_arrivals_total2 = bad_arrivals_total + env_log.bad_arrivals
            integrity_lost_total2 = integrity_lost_total + env_log.integrity_lost
            integrity_min2 = jnp.minimum(integrity_min, internal2.integrity)
            t_alive2 = t_alive + jnp.array(1, dtype=jnp.int32)
            success2 = jnp.logical_or(success, env_log.reached_source)
            alive2 = jnp.logical_and(alive, jnp.logical_not(done))
            action_counts2 = action_counts.at[fwd.action].add(jnp.array(1.0, dtype=jnp.float32))
            dw_sum2 = dw_sum + agent_log.mean_abs_dw

            return (
                (
                    env_state2,
                    obs2,
                    internal2,
                    agent_state2,
                    drive2,
                    alive2,
                    energy_gained_total2,
                    bad_arrivals_total2,
                    integrity_lost_total2,
                    integrity_min2,
                    t_alive2,
                    success2,
                    action_counts2,
                    dw_sum2,
                ),
                (agent_log, env_log, _reward),
            )

        def skip(_):
            zero_agent_log = AgentLog(
                spike_rate=jnp.array(0.0, dtype=jnp.float32),
                modulator=jnp.array(0.0, dtype=jnp.float32),
                mean_abs_dw=jnp.array(0.0, dtype=jnp.float32),
            )
            zero_env_log = EnvLog(
                reached_source=jnp.array(False, dtype=jnp.bool_),
                energy_gained=jnp.array(0.0, dtype=jnp.float32),
                bad_arrivals=jnp.array(0.0, dtype=jnp.float32),
                integrity_lost=jnp.array(0.0, dtype=jnp.float32),
            )
            return (
                (
                    env_state,
                    obs,
                    internal,
                    agent_state,
                    drive_prev,
                    alive,
                    energy_gained_total,
                    bad_arrivals_total,
                    integrity_lost_total,
                    integrity_min,
                    t_alive,
                    success,
                    action_counts,
                    dw_sum,
                ),
                (zero_agent_log, zero_env_log, jnp.array(0.0, dtype=jnp.float32)),
            )

        return jax.lax.cond(alive, do, skip, operand=None)

    # NOTE: xs carries the integer time index, used only for deterministic fold_in + dev phase.
    final, _logs = jax.lax.scan(step, init, xs=t_idx)
    (
        _env_state,
        _obs,
        _internal,
        _agent_state,
        _drive_prev,
        _alive,
        energy_gained_total,
        bad_arrivals_total,
        integrity_lost_total,
        integrity_min,
        t_alive,
        success,
        action_counts,
        dw_sum,
    ) = final

    safe_steps = jnp.maximum(t_alive.astype(jnp.float32), jnp.array(1.0, dtype=jnp.float32))
    p = action_counts / safe_steps
    action_entropy = -jnp.sum(p * jnp.log(p + jnp.array(1e-8, dtype=jnp.float32)))
    action_mode_frac = jnp.max(p)
    mean_abs_dw_mean = dw_sum / safe_steps

    fitness = (
        jnp.array(sim_cfg.fitness_alpha, dtype=jnp.float32) * t_alive.astype(jnp.float32)
        + jnp.array(sim_cfg.fitness_beta, dtype=jnp.float32) * energy_gained_total
        + jnp.array(sim_cfg.success_bonus, dtype=jnp.float32) * success.astype(jnp.float32)
    )
    return FitnessSummary(
        fitness_scalar=fitness,
        t_alive=t_alive,
        energy_gained_total=energy_gained_total,
        integrity_min=integrity_min,
        bad_arrivals_total=bad_arrivals_total,
        integrity_lost_total=integrity_lost_total,
        success=success,
        action_entropy=action_entropy,
        action_mode_frac=action_mode_frac,
        mean_abs_dw_mean=mean_abs_dw_mean,
    )


class MVTSummary(NamedTuple):
    fitness_scalar: Array
    t_alive: Array
    energy_gained_total: Array
    action_entropy: Array


def simulate_lifetime_fitness(
    genome: DirectGenome,
    env_spec: ChemotaxisEnvSpec,
    dev_cfg: DevConfig,
    sim_cfg: SimConfig,
    rng: Array,
    t_idx: Array,
) -> Array:
    steps = t_idx.shape[0]
    rng_develop, rng_env, rng_agent, rng_rollout = jax.random.split(rng, 4)
    params = develop(genome, dev_cfg, rng_develop)

    env_state, obs, internal = env_init(env_spec, rng_env)
    agent_state = agent_init(params, rng_agent)

    drive_prev = drive(internal, sim_cfg)
    alive = jnp.array(True, dtype=jnp.bool_)
    energy_gained_total = jnp.array(0.0, dtype=jnp.float32)
    t_alive = jnp.array(0, dtype=jnp.int32)
    success = jnp.array(False, dtype=jnp.bool_)

    init = (
        env_state,
        obs,
        internal,
        agent_state,
        drive_prev,
        alive,
        energy_gained_total,
        t_alive,
        success,
    )

    def step(carry, t):
        (
            env_state,
            obs,
            internal,
            agent_state,
            drive_prev,
            alive,
            energy_gained_total,
            t_alive,
            success,
        ) = carry

        dev_state = DevelopmentState(age_step=t, phi=_phi(t, steps))
        step_key = jax.random.fold_in(rng_rollout, t)
        rng_agent, rng_env = jax.random.split(step_key, 2)

        def do(_):
            fwd = agent_forward_nolog(params, agent_state, obs, internal, dev_state, rng_agent)
            env_state2, obs2, internal2, env_log, done = env_step(
                env_spec, env_state, fwd.action, dev_state, rng_env
            )
            drive2 = drive(internal2, sim_cfg)
            _reward = drive_prev - drive2
            event_delta = env_log.energy_gained - env_log.integrity_lost
            mod_signal_raw = jnp.where(
                params.modulator_kind == jnp.array(1, dtype=jnp.int32),
                _reward,
                jnp.where(
                    params.modulator_kind == jnp.array(2, dtype=jnp.int32),
                    event_delta,
                    jnp.array(0.0, dtype=jnp.float32),
                ),
            )
            agent_state2 = agent_apply_plasticity_nolog(params, agent_state, fwd, mod_signal_raw)

            energy_gained_total2 = energy_gained_total + env_log.energy_gained
            t_alive2 = t_alive + jnp.array(1, dtype=jnp.int32)
            success2 = jnp.logical_or(success, env_log.reached_source)
            alive2 = jnp.logical_and(alive, jnp.logical_not(done))
            return (
                env_state2,
                obs2,
                internal2,
                agent_state2,
                drive2,
                alive2,
                energy_gained_total2,
                t_alive2,
                success2,
            )

        def skip(_):
            return (
                env_state,
                obs,
                internal,
                agent_state,
                drive_prev,
                alive,
                energy_gained_total,
                t_alive,
                success,
            )

        carry2 = jax.lax.cond(alive, do, skip, operand=None)
        return carry2, ()

    final, _ = jax.lax.scan(step, init, xs=t_idx)
    (
        _env_state,
        _obs,
        _internal,
        _agent_state,
        _drive_prev,
        _alive,
        energy_gained_total,
        t_alive,
        success,
    ) = final

    fitness = (
        jnp.array(sim_cfg.fitness_alpha, dtype=jnp.float32) * t_alive.astype(jnp.float32)
        + jnp.array(sim_cfg.fitness_beta, dtype=jnp.float32) * energy_gained_total
        + jnp.array(sim_cfg.success_bonus, dtype=jnp.float32) * success.astype(jnp.float32)
    )
    return fitness


def simulate_lifetime_mvt(
    genome: DirectGenome,
    env_spec: ChemotaxisEnvSpec,
    dev_cfg: DevConfig,
    sim_cfg: SimConfig,
    rng: Array,
    t_idx: Array,
) -> MVTSummary:
    steps = t_idx.shape[0]
    rng_develop, rng_env, rng_agent, rng_rollout = jax.random.split(rng, 4)
    params = develop(genome, dev_cfg, rng_develop)

    env_state, obs, internal = env_init(env_spec, rng_env)
    agent_state = agent_init(params, rng_agent)

    drive_prev = drive(internal, sim_cfg)
    alive = jnp.array(True, dtype=jnp.bool_)
    energy_gained_total = jnp.array(0.0, dtype=jnp.float32)
    t_alive = jnp.array(0, dtype=jnp.int32)
    success = jnp.array(False, dtype=jnp.bool_)
    num_actions = params.motor_b.shape[0]
    action_counts = jnp.zeros((num_actions,), dtype=jnp.float32)

    init = (
        env_state,
        obs,
        internal,
        agent_state,
        drive_prev,
        alive,
        energy_gained_total,
        t_alive,
        success,
        action_counts,
    )

    def step(carry, t):
        (
            env_state,
            obs,
            internal,
            agent_state,
            drive_prev,
            alive,
            energy_gained_total,
            t_alive,
            success,
            action_counts,
        ) = carry

        dev_state = DevelopmentState(age_step=t, phi=_phi(t, steps))
        step_key = jax.random.fold_in(rng_rollout, t)
        rng_agent, rng_env = jax.random.split(step_key, 2)

        def do(_):
            fwd = agent_forward_nolog(params, agent_state, obs, internal, dev_state, rng_agent)
            env_state2, obs2, internal2, env_log, done = env_step(
                env_spec, env_state, fwd.action, dev_state, rng_env
            )
            drive2 = drive(internal2, sim_cfg)
            _reward = drive_prev - drive2
            event_delta = env_log.energy_gained - env_log.integrity_lost
            mod_signal_raw = jnp.where(
                params.modulator_kind == jnp.array(1, dtype=jnp.int32),
                _reward,
                jnp.where(
                    params.modulator_kind == jnp.array(2, dtype=jnp.int32),
                    event_delta,
                    jnp.array(0.0, dtype=jnp.float32),
                ),
            )
            agent_state2 = agent_apply_plasticity_nolog(params, agent_state, fwd, mod_signal_raw)

            energy_gained_total2 = energy_gained_total + env_log.energy_gained
            t_alive2 = t_alive + jnp.array(1, dtype=jnp.int32)
            success2 = jnp.logical_or(success, env_log.reached_source)
            alive2 = jnp.logical_and(alive, jnp.logical_not(done))
            action_counts2 = action_counts.at[fwd.action].add(jnp.array(1.0, dtype=jnp.float32))
            return (
                env_state2,
                obs2,
                internal2,
                agent_state2,
                drive2,
                alive2,
                energy_gained_total2,
                t_alive2,
                success2,
                action_counts2,
            )

        def skip(_):
            return (
                env_state,
                obs,
                internal,
                agent_state,
                drive_prev,
                alive,
                energy_gained_total,
                t_alive,
                success,
                action_counts,
            )

        carry2 = jax.lax.cond(alive, do, skip, operand=None)
        return carry2, ()

    final, _ = jax.lax.scan(step, init, xs=t_idx)
    (
        _env_state,
        _obs,
        _internal,
        _agent_state,
        _drive_prev,
        _alive,
        energy_gained_total,
        t_alive,
        success,
        action_counts,
    ) = final

    safe_steps = jnp.maximum(t_alive.astype(jnp.float32), jnp.array(1.0, dtype=jnp.float32))
    p = action_counts / safe_steps
    action_entropy = -jnp.sum(p * jnp.log(p + jnp.array(1e-8, dtype=jnp.float32)))
    fitness = (
        jnp.array(sim_cfg.fitness_alpha, dtype=jnp.float32) * t_alive.astype(jnp.float32)
        + jnp.array(sim_cfg.fitness_beta, dtype=jnp.float32) * energy_gained_total
        + jnp.array(sim_cfg.success_bonus, dtype=jnp.float32) * success.astype(jnp.float32)
    )
    return MVTSummary(
        fitness_scalar=fitness,
        t_alive=t_alive,
        energy_gained_total=energy_gained_total,
        action_entropy=action_entropy,
    )


def simulate_lifetime_baseline_greedy(
    env_spec: ChemotaxisEnvSpec,
    sim_cfg: SimConfig,
    rng: Array,
    t_idx: Array,
) -> FitnessSummary:
    steps = t_idx.shape[0]
    rng_env, rng_rollout = jax.random.split(rng, 2)

    env_state, obs, internal = env_init(env_spec, rng_env)
    drive_prev = drive(internal, sim_cfg)
    alive = jnp.array(True, dtype=jnp.bool_)
    energy_gained_total = jnp.array(0.0, dtype=jnp.float32)
    bad_arrivals_total = jnp.array(0.0, dtype=jnp.float32)
    integrity_lost_total = jnp.array(0.0, dtype=jnp.float32)
    integrity_min = internal.integrity
    t_alive = jnp.array(0, dtype=jnp.int32)
    success = jnp.array(False, dtype=jnp.bool_)
    num_actions = 5
    action_counts = jnp.zeros((num_actions,), dtype=jnp.float32)

    init = (
        env_state,
        obs,
        internal,
        drive_prev,
        alive,
        energy_gained_total,
        bad_arrivals_total,
        integrity_lost_total,
        integrity_min,
        t_alive,
        success,
        action_counts,
    )

    def step(carry, t):
        (
            env_state,
            obs,
            internal,
            drive_prev,
            alive,
            energy_gained_total,
            bad_arrivals_total,
            integrity_lost_total,
            integrity_min,
            t_alive,
            success,
            action_counts,
        ) = carry

        dev_state = DevelopmentState(age_step=t, phi=_phi(t, steps))
        step_key = jax.random.fold_in(rng_rollout, t)

        def do(_):
            action = greedy_gradient_action(obs)
            env_state2, obs2, internal2, env_log, done = env_step(
                env_spec, env_state, action, dev_state, step_key
            )
            drive2 = drive(internal2, sim_cfg)
            energy_gained_total2 = energy_gained_total + env_log.energy_gained
            bad_arrivals_total2 = bad_arrivals_total + env_log.bad_arrivals
            integrity_lost_total2 = integrity_lost_total + env_log.integrity_lost
            integrity_min2 = jnp.minimum(integrity_min, internal2.integrity)
            t_alive2 = t_alive + jnp.array(1, dtype=jnp.int32)
            success2 = jnp.logical_or(success, env_log.reached_source)
            alive2 = jnp.logical_and(alive, jnp.logical_not(done))
            action_counts2 = action_counts.at[action].add(jnp.array(1.0, dtype=jnp.float32))
            return (
                env_state2,
                obs2,
                internal2,
                drive2,
                alive2,
                energy_gained_total2,
                bad_arrivals_total2,
                integrity_lost_total2,
                integrity_min2,
                t_alive2,
                success2,
                action_counts2,
            )

        def skip(_):
            return carry

        carry2 = jax.lax.cond(alive, do, skip, operand=None)
        return carry2, None

    final, _ = jax.lax.scan(step, init, xs=t_idx)
    (
        _env_state,
        _obs,
        _internal,
        _drive_prev,
        _alive,
        energy_gained_total,
        bad_arrivals_total,
        integrity_lost_total,
        integrity_min,
        t_alive,
        success,
        action_counts,
    ) = final

    safe_steps = jnp.maximum(t_alive.astype(jnp.float32), jnp.array(1.0, dtype=jnp.float32))
    p = action_counts / safe_steps
    action_entropy = -jnp.sum(p * jnp.log(p + jnp.array(1e-8, dtype=jnp.float32)))
    action_mode_frac = jnp.max(p)
    mean_abs_dw_mean = jnp.array(0.0, dtype=jnp.float32)
    fitness = (
        jnp.array(sim_cfg.fitness_alpha, dtype=jnp.float32) * t_alive.astype(jnp.float32)
        + jnp.array(sim_cfg.fitness_beta, dtype=jnp.float32) * energy_gained_total
        + jnp.array(sim_cfg.success_bonus, dtype=jnp.float32) * success.astype(jnp.float32)
    )
    return FitnessSummary(
        fitness_scalar=fitness,
        t_alive=t_alive,
        energy_gained_total=energy_gained_total,
        integrity_min=integrity_min,
        bad_arrivals_total=bad_arrivals_total,
        integrity_lost_total=integrity_lost_total,
        success=success,
        action_entropy=action_entropy,
        action_mode_frac=action_mode_frac,
        mean_abs_dw_mean=mean_abs_dw_mean,
    )


def simulate_lifetime_baseline_random(
    env_spec: ChemotaxisEnvSpec,
    sim_cfg: SimConfig,
    rng: Array,
    t_idx: Array,
) -> FitnessSummary:
    steps = t_idx.shape[0]
    rng_env, rng_rollout = jax.random.split(rng, 2)

    env_state, obs, internal = env_init(env_spec, rng_env)
    drive_prev = drive(internal, sim_cfg)
    alive = jnp.array(True, dtype=jnp.bool_)
    energy_gained_total = jnp.array(0.0, dtype=jnp.float32)
    bad_arrivals_total = jnp.array(0.0, dtype=jnp.float32)
    integrity_lost_total = jnp.array(0.0, dtype=jnp.float32)
    integrity_min = internal.integrity
    t_alive = jnp.array(0, dtype=jnp.int32)
    success = jnp.array(False, dtype=jnp.bool_)
    num_actions = 5
    action_counts = jnp.zeros((num_actions,), dtype=jnp.float32)

    init = (
        env_state,
        obs,
        internal,
        drive_prev,
        alive,
        energy_gained_total,
        bad_arrivals_total,
        integrity_lost_total,
        integrity_min,
        t_alive,
        success,
        action_counts,
    )

    def step(carry, t):
        (
            env_state,
            obs,
            internal,
            drive_prev,
            alive,
            energy_gained_total,
            bad_arrivals_total,
            integrity_lost_total,
            integrity_min,
            t_alive,
            success,
            action_counts,
        ) = carry

        dev_state = DevelopmentState(age_step=t, phi=_phi(t, steps))
        step_key = jax.random.fold_in(rng_rollout, t)
        rng_act, rng_env = jax.random.split(step_key, 2)

        def do(_):
            action = jax.random.randint(rng_act, shape=(), minval=0, maxval=num_actions, dtype=jnp.int32)
            env_state2, obs2, internal2, env_log, done = env_step(env_spec, env_state, action, dev_state, rng_env)
            drive2 = drive(internal2, sim_cfg)
            energy_gained_total2 = energy_gained_total + env_log.energy_gained
            bad_arrivals_total2 = bad_arrivals_total + env_log.bad_arrivals
            integrity_lost_total2 = integrity_lost_total + env_log.integrity_lost
            integrity_min2 = jnp.minimum(integrity_min, internal2.integrity)
            t_alive2 = t_alive + jnp.array(1, dtype=jnp.int32)
            success2 = jnp.logical_or(success, env_log.reached_source)
            alive2 = jnp.logical_and(alive, jnp.logical_not(done))
            action_counts2 = action_counts.at[action].add(jnp.array(1.0, dtype=jnp.float32))
            return (
                env_state2,
                obs2,
                internal2,
                drive2,
                alive2,
                energy_gained_total2,
                bad_arrivals_total2,
                integrity_lost_total2,
                integrity_min2,
                t_alive2,
                success2,
                action_counts2,
            )

        def skip(_):
            return carry

        carry2 = jax.lax.cond(alive, do, skip, operand=None)
        return carry2, None

    final, _ = jax.lax.scan(step, init, xs=t_idx)
    (
        _env_state,
        _obs,
        _internal,
        _drive_prev,
        _alive,
        energy_gained_total,
        bad_arrivals_total,
        integrity_lost_total,
        integrity_min,
        t_alive,
        success,
        action_counts,
    ) = final

    safe_steps = jnp.maximum(t_alive.astype(jnp.float32), jnp.array(1.0, dtype=jnp.float32))
    p = action_counts / safe_steps
    action_entropy = -jnp.sum(p * jnp.log(p + jnp.array(1e-8, dtype=jnp.float32)))
    action_mode_frac = jnp.max(p)
    mean_abs_dw_mean = jnp.array(0.0, dtype=jnp.float32)
    fitness = (
        jnp.array(sim_cfg.fitness_alpha, dtype=jnp.float32) * t_alive.astype(jnp.float32)
        + jnp.array(sim_cfg.fitness_beta, dtype=jnp.float32) * energy_gained_total
        + jnp.array(sim_cfg.success_bonus, dtype=jnp.float32) * success.astype(jnp.float32)
    )
    return FitnessSummary(
        fitness_scalar=fitness,
        t_alive=t_alive,
        energy_gained_total=energy_gained_total,
        integrity_min=integrity_min,
        bad_arrivals_total=bad_arrivals_total,
        integrity_lost_total=integrity_lost_total,
        success=success,
        action_entropy=action_entropy,
        action_mode_frac=action_mode_frac,
        mean_abs_dw_mean=mean_abs_dw_mean,
    )


def simulate_lifetime_baseline_stay(
    env_spec: ChemotaxisEnvSpec,
    sim_cfg: SimConfig,
    rng: Array,
    t_idx: Array,
) -> FitnessSummary:
    steps = t_idx.shape[0]
    rng_env, rng_rollout = jax.random.split(rng, 2)

    env_state, obs, internal = env_init(env_spec, rng_env)
    drive_prev = drive(internal, sim_cfg)
    alive = jnp.array(True, dtype=jnp.bool_)
    energy_gained_total = jnp.array(0.0, dtype=jnp.float32)
    bad_arrivals_total = jnp.array(0.0, dtype=jnp.float32)
    integrity_lost_total = jnp.array(0.0, dtype=jnp.float32)
    integrity_min = internal.integrity
    t_alive = jnp.array(0, dtype=jnp.int32)
    success = jnp.array(False, dtype=jnp.bool_)
    num_actions = 5
    action_counts = jnp.zeros((num_actions,), dtype=jnp.float32)
    action = jnp.array(0, dtype=jnp.int32)

    init = (
        env_state,
        obs,
        internal,
        drive_prev,
        alive,
        energy_gained_total,
        bad_arrivals_total,
        integrity_lost_total,
        integrity_min,
        t_alive,
        success,
        action_counts,
    )

    def step(carry, t):
        (
            env_state,
            obs,
            internal,
            drive_prev,
            alive,
            energy_gained_total,
            bad_arrivals_total,
            integrity_lost_total,
            integrity_min,
            t_alive,
            success,
            action_counts,
        ) = carry

        dev_state = DevelopmentState(age_step=t, phi=_phi(t, steps))
        step_key = jax.random.fold_in(rng_rollout, t)

        def do(_):
            env_state2, obs2, internal2, env_log, done = env_step(env_spec, env_state, action, dev_state, step_key)
            drive2 = drive(internal2, sim_cfg)
            energy_gained_total2 = energy_gained_total + env_log.energy_gained
            bad_arrivals_total2 = bad_arrivals_total + env_log.bad_arrivals
            integrity_lost_total2 = integrity_lost_total + env_log.integrity_lost
            integrity_min2 = jnp.minimum(integrity_min, internal2.integrity)
            t_alive2 = t_alive + jnp.array(1, dtype=jnp.int32)
            success2 = jnp.logical_or(success, env_log.reached_source)
            alive2 = jnp.logical_and(alive, jnp.logical_not(done))
            action_counts2 = action_counts.at[action].add(jnp.array(1.0, dtype=jnp.float32))
            return (
                env_state2,
                obs2,
                internal2,
                drive2,
                alive2,
                energy_gained_total2,
                bad_arrivals_total2,
                integrity_lost_total2,
                integrity_min2,
                t_alive2,
                success2,
                action_counts2,
            )

        def skip(_):
            return carry

        carry2 = jax.lax.cond(alive, do, skip, operand=None)
        return carry2, None

    final, _ = jax.lax.scan(step, init, xs=t_idx)
    (
        _env_state,
        _obs,
        _internal,
        _drive_prev,
        _alive,
        energy_gained_total,
        bad_arrivals_total,
        integrity_lost_total,
        integrity_min,
        t_alive,
        success,
        action_counts,
    ) = final

    safe_steps = jnp.maximum(t_alive.astype(jnp.float32), jnp.array(1.0, dtype=jnp.float32))
    p = action_counts / safe_steps
    action_entropy = -jnp.sum(p * jnp.log(p + jnp.array(1e-8, dtype=jnp.float32)))
    action_mode_frac = jnp.max(p)
    mean_abs_dw_mean = jnp.array(0.0, dtype=jnp.float32)
    fitness = (
        jnp.array(sim_cfg.fitness_alpha, dtype=jnp.float32) * t_alive.astype(jnp.float32)
        + jnp.array(sim_cfg.fitness_beta, dtype=jnp.float32) * energy_gained_total
        + jnp.array(sim_cfg.success_bonus, dtype=jnp.float32) * success.astype(jnp.float32)
    )
    return FitnessSummary(
        fitness_scalar=fitness,
        t_alive=t_alive,
        energy_gained_total=energy_gained_total,
        integrity_min=integrity_min,
        bad_arrivals_total=bad_arrivals_total,
        integrity_lost_total=integrity_lost_total,
        success=success,
        action_entropy=action_entropy,
        action_mode_frac=action_mode_frac,
        mean_abs_dw_mean=mean_abs_dw_mean,
    )
