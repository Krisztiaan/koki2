from __future__ import annotations

import jax.numpy as jnp

from typing import NamedTuple

from koki2.types import (
    AgentLog,
    AgentParams,
    AgentState,
    Array,
    DevelopmentState,
    InternalState,
)


def agent_init(params: AgentParams, rng: Array) -> AgentState:
    del rng
    n = params.obs_w.shape[0]
    e = params.w0.shape[0]
    v = jnp.zeros((n,), dtype=jnp.float32)
    spike = jnp.zeros((n,), dtype=jnp.float32)
    w = params.w0
    elig = jnp.zeros((e,), dtype=jnp.float32)
    return AgentState(v=v, spike=spike, w=w, elig=elig)


def _segment_sum(data: Array, segment_ids: Array, num_segments: int) -> Array:
    out = jnp.zeros((num_segments,), dtype=data.dtype)
    return out.at[segment_ids].add(data)


class AgentForwardOut(NamedTuple):
    v: Array
    spike: Array
    elig_next: Array
    action: Array
    spike_rate: Array
    mod_spike: Array


def agent_forward(
    params: AgentParams,
    state: AgentState,
    obs: Array,
    internal: InternalState,
    dev: DevelopmentState,
    rng: Array,
) -> AgentForwardOut:
    del internal, dev, rng

    n = params.obs_w.shape[0]
    src = params.edge_index[:, 0]
    dst = params.edge_index[:, 1]

    obs_in = params.obs_w @ obs.astype(jnp.float32)
    rec_in = _segment_sum(state.w * state.spike[src], dst, num_segments=n)
    i_t = obs_in + rec_in

    v = state.v * params.v_decay + i_t
    spike = (v > params.theta).astype(jnp.float32)
    v = v - spike * params.theta

    mod_spike = jnp.tanh(jnp.dot(params.mod_w, spike))

    pre = state.spike[src]
    post = spike[dst]
    elig_next = params.plast_lambda * state.elig + (pre * post)

    logits = jnp.dot(spike, params.motor_w) + params.motor_b
    action = jnp.argmax(logits).astype(jnp.int32)

    spike_rate = jnp.mean(spike)
    return AgentForwardOut(
        v=v,
        spike=spike,
        elig_next=elig_next,
        action=action,
        spike_rate=spike_rate,
        mod_spike=mod_spike,
    )


def agent_apply_plasticity(
    params: AgentParams,
    state: AgentState,
    fwd: AgentForwardOut,
    mod_signal_raw: Array,
) -> tuple[AgentState, AgentLog]:
    mod_drive = jnp.tanh(params.mod_drive_scale * mod_signal_raw.astype(jnp.float32))
    use_signal = params.modulator_kind != jnp.array(0, dtype=jnp.int32)
    modulator = jnp.where(use_signal, mod_drive, fwd.mod_spike)

    dw = params.plast_eta * modulator * fwd.elig_next
    w_next = jnp.clip(state.w + dw, min=-3.0, max=3.0)

    # If plasticity is disabled, keep weights and traces unchanged.
    elig = jnp.where(params.plast_enabled, fwd.elig_next, state.elig)
    w = jnp.where(params.plast_enabled, w_next, state.w)
    dw_applied = jnp.where(params.plast_enabled, dw, jnp.zeros_like(dw))

    log = AgentLog(
        spike_rate=fwd.spike_rate,
        modulator=modulator,
        mean_abs_dw=jnp.mean(jnp.abs(dw_applied)),
    )
    return AgentState(v=fwd.v, spike=fwd.spike, w=w, elig=elig), log
