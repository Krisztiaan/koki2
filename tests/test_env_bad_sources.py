import jax
import jax.numpy as jnp

from koki2.envs.chemotaxis import env_step
from koki2.types import ChemotaxisEnvSpec, ChemotaxisEnvState, DevelopmentState


def test_bad_source_reduces_integrity_and_is_not_success() -> None:
    spec = ChemotaxisEnvSpec(
        width=8,
        height=1,
        max_steps=8,
        energy_init=1.0,
        energy_decay=0.0,
        energy_gain=0.25,
        terminate_on_reach=False,
        obs_noise=0.0,
        num_sources=1,
        num_bad_sources=1,
        bad_source_integrity_loss=0.25,
    )
    state0 = ChemotaxisEnvState(
        t=jnp.array(0, dtype=jnp.int32),
        pos=jnp.array([0, 0], dtype=jnp.int32),
        source_pos=jnp.array([[1, 0]], dtype=jnp.int32),
        source_is_bad=jnp.array([True], dtype=jnp.bool_),
        source_active=jnp.array([True], dtype=jnp.bool_),
        source_respawn_t=jnp.array([0], dtype=jnp.int32),
        energy=jnp.array(1.0, dtype=jnp.float32),
        integrity=jnp.array(1.0, dtype=jnp.float32),
    )
    dev = DevelopmentState(age_step=jnp.array(0, dtype=jnp.int32), phi=jnp.array(0.0, dtype=jnp.float32))

    state1, _obs1, internal1, log1, done1 = env_step(
        spec, state0, jnp.array(4, dtype=jnp.int32), dev, jax.random.PRNGKey(0)
    )

    assert float(jax.device_get(log1.energy_gained)) == 0.0
    assert bool(jax.device_get(log1.reached_source)) is False
    assert float(jax.device_get(internal1.integrity)) == 0.75
    assert float(jax.device_get(state1.integrity)) == 0.75
    assert bool(jax.device_get(done1)) is False


def test_integrity_depletion_ends_episode() -> None:
    spec = ChemotaxisEnvSpec(
        width=8,
        height=1,
        max_steps=8,
        energy_init=1.0,
        energy_decay=0.0,
        energy_gain=0.25,
        terminate_on_reach=False,
        obs_noise=0.0,
        num_sources=1,
        num_bad_sources=1,
        bad_source_integrity_loss=1.0,
    )
    state0 = ChemotaxisEnvState(
        t=jnp.array(0, dtype=jnp.int32),
        pos=jnp.array([0, 0], dtype=jnp.int32),
        source_pos=jnp.array([[1, 0]], dtype=jnp.int32),
        source_is_bad=jnp.array([True], dtype=jnp.bool_),
        source_active=jnp.array([True], dtype=jnp.bool_),
        source_respawn_t=jnp.array([0], dtype=jnp.int32),
        energy=jnp.array(1.0, dtype=jnp.float32),
        integrity=jnp.array(0.5, dtype=jnp.float32),
    )
    dev = DevelopmentState(age_step=jnp.array(0, dtype=jnp.int32), phi=jnp.array(0.0, dtype=jnp.float32))

    _state1, _obs1, internal1, log1, done1 = env_step(
        spec, state0, jnp.array(4, dtype=jnp.int32), dev, jax.random.PRNGKey(0)
    )

    assert float(jax.device_get(log1.energy_gained)) == 0.0
    assert bool(jax.device_get(log1.reached_source)) is False
    assert float(jax.device_get(internal1.integrity)) == 0.0
    assert bool(jax.device_get(done1)) is True


def test_good_only_gradient_targets_good_source() -> None:
    base = dict(
        width=10,
        height=1,
        max_steps=4,
        energy_init=1.0,
        energy_decay=0.0,
        energy_gain=0.0,
        terminate_on_reach=False,
        obs_noise=0.0,
        num_sources=2,
        num_bad_sources=1,
        bad_source_integrity_loss=0.0,
    )
    spec_any = ChemotaxisEnvSpec(**base, good_only_gradient=False)
    spec_good = ChemotaxisEnvSpec(**base, good_only_gradient=True)
    state0 = ChemotaxisEnvState(
        t=jnp.array(0, dtype=jnp.int32),
        pos=jnp.array([0, 0], dtype=jnp.int32),
        source_pos=jnp.array([[2, 0], [9, 0]], dtype=jnp.int32),
        source_is_bad=jnp.array([True, False], dtype=jnp.bool_),
        source_active=jnp.array([True, True], dtype=jnp.bool_),
        source_respawn_t=jnp.array([0, 0], dtype=jnp.int32),
        energy=jnp.array(1.0, dtype=jnp.float32),
        integrity=jnp.array(1.0, dtype=jnp.float32),
    )
    dev = DevelopmentState(age_step=jnp.array(0, dtype=jnp.int32), phi=jnp.array(0.0, dtype=jnp.float32))

    _s_any, obs_any, _internal_any, _log_any, _done_any = env_step(
        spec_any, state0, jnp.array(0, dtype=jnp.int32), dev, jax.random.PRNGKey(0)
    )
    _s_good, obs_good, _internal_good, _log_good, _done_good = env_step(
        spec_good, state0, jnp.array(0, dtype=jnp.int32), dev, jax.random.PRNGKey(0)
    )

    # obs[0] is x-gradient normalized by (width-1) = 9.
    assert abs(float(jax.device_get(obs_any[0])) - (2.0 / 9.0)) < 1e-6
    assert abs(float(jax.device_get(obs_good[0])) - (9.0 / 9.0)) < 1e-6
