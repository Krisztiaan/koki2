import jax
import jax.numpy as jnp

from koki2.envs.chemotaxis import env_step
from koki2.types import ChemotaxisEnvSpec, ChemotaxisEnvState, DevelopmentState


def _state_with_source(*, pos_x: int, src_x: int) -> ChemotaxisEnvState:
    return ChemotaxisEnvState(
        t=jnp.array(0, dtype=jnp.int32),
        pos=jnp.array([pos_x, 0], dtype=jnp.int32),
        source_pos=jnp.array([[src_x, 0]], dtype=jnp.int32),
        source_is_bad=jnp.array([False], dtype=jnp.bool_),
        source_active=jnp.array([True], dtype=jnp.bool_),
        source_respawn_t=jnp.array([0], dtype=jnp.int32),
        energy=jnp.array(1.0, dtype=jnp.float32),
        integrity=jnp.array(1.0, dtype=jnp.float32),
    )


def test_grad_dropout_p_one_zeros_gradient() -> None:
    spec = ChemotaxisEnvSpec(
        width=10,
        height=1,
        max_steps=4,
        energy_init=1.0,
        energy_decay=0.0,
        energy_gain=0.0,
        terminate_on_reach=False,
        obs_noise=0.0,
        grad_dropout_p=1.0,
    )
    state0 = _state_with_source(pos_x=0, src_x=5)
    dev = DevelopmentState(age_step=jnp.array(0, dtype=jnp.int32), phi=jnp.array(0.0, dtype=jnp.float32))

    _s1, obs, _internal, _log, _done = env_step(spec, state0, jnp.array(0, dtype=jnp.int32), dev, jax.random.PRNGKey(0))

    assert float(jax.device_get(obs[0])) == 0.0
    assert float(jax.device_get(obs[1])) == 0.0


def test_grad_dropout_p_zero_preserves_gradient() -> None:
    spec = ChemotaxisEnvSpec(
        width=10,
        height=1,
        max_steps=4,
        energy_init=1.0,
        energy_decay=0.0,
        energy_gain=0.0,
        terminate_on_reach=False,
        obs_noise=0.0,
        grad_dropout_p=0.0,
    )
    state0 = _state_with_source(pos_x=0, src_x=5)
    dev = DevelopmentState(age_step=jnp.array(0, dtype=jnp.int32), phi=jnp.array(0.0, dtype=jnp.float32))

    _s1, obs, _internal, _log, _done = env_step(spec, state0, jnp.array(0, dtype=jnp.int32), dev, jax.random.PRNGKey(0))

    assert abs(float(jax.device_get(obs[0])) - (5.0 / 9.0)) < 1e-6


def test_grad_dropout_matches_rng_mask_draw() -> None:
    spec = ChemotaxisEnvSpec(
        width=10,
        height=1,
        max_steps=4,
        energy_init=1.0,
        energy_decay=0.0,
        energy_gain=0.0,
        terminate_on_reach=False,
        obs_noise=0.0,
        grad_dropout_p=0.5,
    )
    state0 = _state_with_source(pos_x=0, src_x=5)
    dev = DevelopmentState(age_step=jnp.array(0, dtype=jnp.int32), phi=jnp.array(0.0, dtype=jnp.float32))
    key = jax.random.PRNGKey(0)

    _s1, obs, _internal, _log, _done = env_step(spec, state0, jnp.array(0, dtype=jnp.int32), dev, key)

    rng_mask, _rng_noise = jax.random.split(key, 2)
    keep = jax.random.bernoulli(rng_mask, p=0.5)
    expected = jnp.where(keep, jnp.array(5.0 / 9.0, dtype=jnp.float32), jnp.array(0.0, dtype=jnp.float32))

    assert float(jax.device_get(obs[0])) == float(jax.device_get(expected))

