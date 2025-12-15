import math

import jax
import jax.numpy as jnp

from koki2.agent.snn import agent_step
from koki2.types import AgentParams, AgentState, DevelopmentState, InternalState


def _make_params(*, plast_enabled: bool, plast_eta: float) -> AgentParams:
    n = 2
    obs_dim = 1
    e = 1
    a = 5
    return AgentParams(
        obs_w=jnp.array([[0.0], [10.0]], dtype=jnp.float32),
        edge_index=jnp.array([[0, 1]], dtype=jnp.int32),
        w0=jnp.zeros((e,), dtype=jnp.float32),
        motor_w=jnp.zeros((n, a), dtype=jnp.float32),
        motor_b=jnp.zeros((a,), dtype=jnp.float32),
        mod_w=jnp.array([0.0, 1.0], dtype=jnp.float32),
        v_decay=jnp.ones((n,), dtype=jnp.float32),
        theta=jnp.ones((n,), dtype=jnp.float32),
        plast_enabled=jnp.array(plast_enabled, dtype=jnp.bool_),
        plast_eta=jnp.array(plast_eta, dtype=jnp.float32),
        plast_lambda=jnp.array(0.9, dtype=jnp.float32),
    )


def test_plasticity_disabled_keeps_state_and_logs_zero_dw() -> None:
    params = _make_params(plast_enabled=False, plast_eta=0.5)
    state = AgentState(
        v=jnp.zeros((2,), dtype=jnp.float32),
        spike=jnp.array([1.0, 0.0], dtype=jnp.float32),
        w=jnp.zeros((1,), dtype=jnp.float32),
        elig=jnp.zeros((1,), dtype=jnp.float32),
    )
    obs = jnp.array([1.0], dtype=jnp.float32)
    internal = InternalState(energy=jnp.array(1.0, dtype=jnp.float32), integrity=jnp.array(1.0, dtype=jnp.float32))
    dev = DevelopmentState(age_step=jnp.array(0, dtype=jnp.int32), phi=jnp.array(0.0, dtype=jnp.float32))

    state2, _action, log = agent_step(params, state, obs, internal, dev, jax.random.PRNGKey(0))

    assert float(jax.device_get(state2.w[0])) == float(jax.device_get(state.w[0]))
    assert float(jax.device_get(state2.elig[0])) == float(jax.device_get(state.elig[0]))
    assert float(jax.device_get(log.mean_abs_dw)) == 0.0


def test_plasticity_enabled_updates_weights_and_logs_dw() -> None:
    params = _make_params(plast_enabled=True, plast_eta=0.5)
    state = AgentState(
        v=jnp.zeros((2,), dtype=jnp.float32),
        spike=jnp.array([1.0, 0.0], dtype=jnp.float32),
        w=jnp.zeros((1,), dtype=jnp.float32),
        elig=jnp.zeros((1,), dtype=jnp.float32),
    )
    obs = jnp.array([1.0], dtype=jnp.float32)
    internal = InternalState(energy=jnp.array(1.0, dtype=jnp.float32), integrity=jnp.array(1.0, dtype=jnp.float32))
    dev = DevelopmentState(age_step=jnp.array(0, dtype=jnp.int32), phi=jnp.array(0.0, dtype=jnp.float32))

    state2, _action, log = agent_step(params, state, obs, internal, dev, jax.random.PRNGKey(0))

    expected_dw = 0.5 * math.tanh(1.0)  # elig_next==1 in this constructed case
    assert abs(float(jax.device_get(state2.w[0])) - expected_dw) < 1e-6
    assert float(jax.device_get(state2.elig[0])) == 1.0
    assert abs(float(jax.device_get(log.mean_abs_dw)) - expected_dw) < 1e-6

