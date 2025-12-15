from __future__ import annotations

import jax.numpy as jnp

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


def agent_step(
    params: AgentParams,
    state: AgentState,
    obs: Array,
    internal: InternalState,
    dev: DevelopmentState,
    rng: Array,
) -> tuple[AgentState, Array, AgentLog]:
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

    modulator = jnp.tanh(jnp.dot(params.mod_w, spike))

    pre = state.spike[src]
    post = spike[dst]
    elig_next = params.plast_lambda * state.elig + (pre * post)
    dw = params.plast_eta * modulator * elig_next
    w_next = jnp.clip(state.w + dw, min=-3.0, max=3.0)

    # If plasticity is disabled, keep weights and traces unchanged.
    elig = jnp.where(params.plast_enabled, elig_next, state.elig)
    w = jnp.where(params.plast_enabled, w_next, state.w)

    logits = jnp.dot(spike, params.motor_w) + params.motor_b
    action = jnp.argmax(logits).astype(jnp.int32)

    log = AgentLog(
        spike_rate=jnp.mean(spike),
        modulator=modulator,
        mean_abs_dw=jnp.mean(jnp.abs(dw)),
    )
    return AgentState(v=v, spike=spike, w=w, elig=elig), action, log
