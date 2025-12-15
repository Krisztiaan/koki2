from __future__ import annotations

import jax
import jax.numpy as jnp

from koki2.nursing.schedules import quantize, schedule
from koki2.types import (
    Array,
    ChemotaxisEnvSpec,
    ChemotaxisEnvState,
    DevelopmentState,
    EnvLog,
    InternalState,
)


def _observe(
    state: ChemotaxisEnvState, internal: InternalState, spec: ChemotaxisEnvSpec, phi: Array, rng: Array
) -> Array:
    sources = state.source_pos.astype(jnp.float32)  # (K, 2)
    active = state.source_active.astype(jnp.bool_)  # (K,)
    good_only = jnp.asarray(spec.good_only_gradient, dtype=jnp.bool_)
    good_mask = jnp.logical_and(active, jnp.logical_not(state.source_is_bad.astype(jnp.bool_)))
    mask = jnp.where(good_only, good_mask, active)
    pos = state.pos.astype(jnp.float32)  # (2,)
    diffs = sources - pos[None, :]  # (K, 2)
    dist2 = jnp.sum(diffs * diffs, axis=1)  # (K,)
    dist2 = jnp.where(mask, dist2, jnp.array(jnp.inf, dtype=jnp.float32))
    nearest = jnp.argmin(dist2)
    diff = diffs[nearest]
    diff = jnp.where(jnp.any(mask), diff, jnp.zeros_like(diff))
    width = jnp.asarray(spec.width, dtype=jnp.float32)
    height = jnp.asarray(spec.height, dtype=jnp.float32)
    scale = jnp.stack([jnp.maximum(width - 1.0, 1.0), jnp.maximum(height - 1.0, 1.0)], axis=0)
    grad = diff / scale

    grad_bins = schedule(
        phi, spec.grad_bins_min, spec.grad_bins_max, spec.grad_bins_start_phi, spec.grad_bins_end_phi
    )
    grad = quantize(grad, grad_bins)

    grad_gain = schedule(
        phi, spec.grad_gain_min, spec.grad_gain_max, spec.grad_gain_start_phi, spec.grad_gain_end_phi
    )
    grad = grad * grad_gain

    rng_mask, rng_noise = jax.random.split(rng, 2)
    drop_p = jnp.asarray(spec.grad_dropout_p, dtype=jnp.float32)
    drop_p = jnp.clip(drop_p, 0.0, 1.0)
    keep = jax.random.bernoulli(rng_mask, p=1.0 - drop_p)
    grad = jnp.where(keep, grad, jnp.zeros_like(grad))

    obs = jnp.concatenate(
        [
            grad,
            jnp.array([internal.energy, internal.integrity], dtype=jnp.float32),
        ],
        axis=0,
    )
    obs = obs + jnp.asarray(spec.obs_noise, dtype=jnp.float32) * jax.random.normal(
        rng_noise, obs.shape, dtype=jnp.float32
    )
    return obs


def env_init(spec: ChemotaxisEnvSpec, rng: Array) -> tuple[ChemotaxisEnvState, Array, InternalState]:
    rng_src, rng_kind, rng_obs = jax.random.split(rng, 3)

    width = jnp.asarray(spec.width, dtype=jnp.int32)
    height = jnp.asarray(spec.height, dtype=jnp.int32)
    pos = jnp.stack([width // 2, height // 2], axis=0).astype(jnp.int32)
    hi = jnp.stack([width, height], axis=0).astype(jnp.int32)
    num_sources = int(spec.num_sources)
    if num_sources < 1:
        raise ValueError(f"ChemotaxisEnvSpec.num_sources must be >= 1, got {num_sources}")
    num_bad = int(spec.num_bad_sources)
    if num_bad < 0:
        raise ValueError(f"ChemotaxisEnvSpec.num_bad_sources must be >= 0, got {num_bad}")
    if num_bad > num_sources:
        raise ValueError(
            f"ChemotaxisEnvSpec.num_bad_sources must be <= num_sources, got num_bad_sources={num_bad} num_sources={num_sources}"
        )
    source_pos = jax.random.randint(
        rng_src,
        shape=(num_sources, 2),
        minval=0,
        maxval=hi,
        dtype=jnp.int32,
    )
    perm = jax.random.permutation(rng_kind, num_sources)
    source_is_bad = jnp.zeros((num_sources,), dtype=jnp.bool_).at[perm[:num_bad]].set(jnp.array(True, dtype=jnp.bool_))
    source_active = jnp.ones((num_sources,), dtype=jnp.bool_)
    source_respawn_t = jnp.zeros((num_sources,), dtype=jnp.int32)

    internal = InternalState(
        energy=jnp.asarray(spec.energy_init, dtype=jnp.float32),
        integrity=jnp.array(1.0, dtype=jnp.float32),
    )
    state = ChemotaxisEnvState(
        t=jnp.array(0, dtype=jnp.int32),
        pos=pos,
        source_pos=source_pos,
        source_is_bad=source_is_bad,
        source_active=source_active,
        source_respawn_t=source_respawn_t,
        energy=internal.energy,
        integrity=internal.integrity,
    )
    obs = _observe(state, internal, spec, jnp.array(0.0, dtype=jnp.float32), rng_obs)
    return state, obs, internal


def env_step(
    spec: ChemotaxisEnvSpec,
    state: ChemotaxisEnvState,
    action: Array,
    dev: DevelopmentState,
    rng: Array,
) -> tuple[ChemotaxisEnvState, Array, InternalState, EnvLog, Array]:
    pos_prev = state.pos
    active_prev = state.source_active
    hit_prev = jnp.all(pos_prev[None, :] == state.source_pos, axis=1) & active_prev

    move = jnp.array(
        [
            [0, 0],  # 0 stay
            [0, -1],  # 1 up
            [0, 1],  # 2 down
            [-1, 0],  # 3 left
            [1, 0],  # 4 right
        ],
        dtype=jnp.int32,
    )
    action = jnp.clip(action.astype(jnp.int32), 0, move.shape[0] - 1)
    delta = move[action]

    new_pos = state.pos + delta
    width = jnp.asarray(spec.width, dtype=jnp.int32)
    height = jnp.asarray(spec.height, dtype=jnp.int32)
    max_pos = jnp.stack([width - 1, height - 1], axis=0).astype(jnp.int32)
    new_pos = jnp.clip(new_pos, min=jnp.array([0, 0], dtype=jnp.int32), max=max_pos)
    t = state.t + jnp.array(1, dtype=jnp.int32)
    state = ChemotaxisEnvState(
        t=t,
        pos=new_pos,
        source_pos=state.source_pos,
        source_is_bad=state.source_is_bad,
        source_active=state.source_active,
        source_respawn_t=state.source_respawn_t,
        energy=state.energy,
        integrity=state.integrity,
    )

    hit_now = jnp.all(state.pos[None, :] == state.source_pos, axis=1) & state.source_active
    reached = jnp.any(hit_now & jnp.logical_not(state.source_is_bad))
    arrived = hit_now & jnp.logical_not(hit_prev)
    arrived_good = arrived & jnp.logical_not(state.source_is_bad)
    arrived_bad = arrived & state.source_is_bad
    num_good = jnp.sum(arrived_good.astype(jnp.float32))
    num_bad = jnp.sum(arrived_bad.astype(jnp.float32))
    energy_gained = jnp.asarray(spec.energy_gain, dtype=jnp.float32) * num_good

    if spec.source_deplete:
        delay_good = jnp.asarray(spec.source_respawn_delay, dtype=jnp.int32)
        bad_delay_int = int(spec.bad_source_respawn_delay)
        if bad_delay_int < 0:
            bad_delay_int = int(spec.source_respawn_delay)
        delay_bad = jnp.asarray(bad_delay_int, dtype=jnp.int32)

        if int(spec.num_bad_sources) > 0 and float(spec.bad_source_deplete_p) < 1.0:
            rng_deplete, rng = jax.random.split(rng, 2)
            bad_p = jnp.asarray(spec.bad_source_deplete_p, dtype=jnp.float32)
            bad_p = jnp.clip(bad_p, 0.0, 1.0)
            deplete_bad = jax.random.bernoulli(rng_deplete, p=bad_p, shape=state.source_active.shape)
            deplete_now = arrived & (jnp.logical_not(state.source_is_bad) | deplete_bad)
        else:
            deplete_now = arrived

        source_active = jnp.where(deplete_now, jnp.array(False, dtype=jnp.bool_), state.source_active)
        delay_per_source = jnp.where(state.source_is_bad, delay_bad, delay_good)
        source_respawn_t = jnp.where(deplete_now, delay_per_source, state.source_respawn_t)

        # Countdown and respawn.
        dec = jnp.array(1, dtype=jnp.int32)
        dec_mask = jnp.logical_and(jnp.logical_not(source_active), jnp.logical_not(deplete_now))
        dec_t = jnp.maximum(source_respawn_t - dec, jnp.array(0, dtype=jnp.int32))
        source_respawn_t = jnp.where(dec_mask, dec_t, source_respawn_t)
        source_respawn_t = jnp.where(source_active, jnp.array(0, dtype=jnp.int32), source_respawn_t)
        respawn = jnp.logical_and(jnp.logical_not(source_active), source_respawn_t == 0)
        rng_respawn, rng_rest = jax.random.split(rng, 2)
        width = jnp.asarray(spec.width, dtype=jnp.int32)
        height = jnp.asarray(spec.height, dtype=jnp.int32)
        hi = jnp.stack([width, height], axis=0).astype(jnp.int32)
        new_pos_all = jax.random.randint(
            rng_respawn,
            shape=state.source_pos.shape,
            minval=0,
            maxval=hi,
            dtype=jnp.int32,
        )
        source_pos = jnp.where(respawn[:, None], new_pos_all, state.source_pos)
        source_active = jnp.where(respawn, jnp.array(True, dtype=jnp.bool_), source_active)
        source_respawn_t = jnp.where(respawn, jnp.array(0, dtype=jnp.int32), source_respawn_t)
        rng = rng_rest
        state = state._replace(
            source_pos=source_pos,
            source_active=source_active,
            source_respawn_t=source_respawn_t,
        )

    energy = jnp.clip(
        state.energy - jnp.asarray(spec.energy_decay, dtype=jnp.float32) + energy_gained, 0.0, 1.0
    )
    bad_loss = jnp.asarray(spec.bad_source_integrity_loss, dtype=jnp.float32)
    integrity_lost = bad_loss * num_bad
    integrity = jnp.clip(
        state.integrity - integrity_lost, 0.0, 1.0
    )
    internal = InternalState(energy=energy, integrity=integrity)
    state = state._replace(energy=energy, integrity=integrity)

    env_log = EnvLog(
        reached_source=reached,
        energy_gained=energy_gained,
        bad_arrivals=num_bad,
        integrity_lost=integrity_lost,
    )
    done = jnp.logical_or(state.t >= spec.max_steps, jnp.logical_or(energy <= 0.0, integrity <= 0.0))
    terminate = jnp.asarray(spec.terminate_on_reach, dtype=jnp.bool_)
    done = jnp.logical_or(done, jnp.logical_and(terminate, reached))
    obs = _observe(state, internal, spec, dev.phi, rng)
    return state, obs, internal, env_log, done
