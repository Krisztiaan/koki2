import jax
import jax.numpy as jnp

from koki2.envs.chemotaxis import env_step
from koki2.types import ChemotaxisEnvSpec, ChemotaxisEnvState, DevelopmentState


def test_observation_targets_nearest_source() -> None:
    spec = ChemotaxisEnvSpec(
        width=10,
        height=1,
        max_steps=4,
        energy_init=1.0,
        energy_decay=0.0,
        energy_gain=0.0,
        terminate_on_reach=False,
        obs_noise=0.0,
        num_sources=2,
    )
    state = ChemotaxisEnvState(
        t=jnp.array(0, dtype=jnp.int32),
        pos=jnp.array([0, 0], dtype=jnp.int32),
        source_pos=jnp.array([[9, 0], [3, 0]], dtype=jnp.int32),
        source_is_bad=jnp.array([False, False], dtype=jnp.bool_),
        source_active=jnp.array([True, True], dtype=jnp.bool_),
        source_respawn_t=jnp.array([0, 0], dtype=jnp.int32),
        energy=jnp.array(1.0, dtype=jnp.float32),
        integrity=jnp.array(1.0, dtype=jnp.float32),
    )
    dev = DevelopmentState(age_step=jnp.array(0, dtype=jnp.int32), phi=jnp.array(0.0, dtype=jnp.float32))

    _, obs, _, _, _ = env_step(spec, state, jnp.array(0, dtype=jnp.int32), dev, jax.random.PRNGKey(0))
    gx = float(jax.device_get(obs[0]))
    assert abs(gx - (3.0 / 9.0)) < 1e-6


def test_energy_gain_only_on_arrival_to_any_source() -> None:
    spec = ChemotaxisEnvSpec(
        width=8,
        height=1,
        max_steps=8,
        energy_init=0.5,
        energy_decay=0.0,
        energy_gain=0.25,
        terminate_on_reach=False,
        obs_noise=0.0,
        num_sources=2,
    )
    sources = jnp.array([[1, 0], [6, 0]], dtype=jnp.int32)
    state0 = ChemotaxisEnvState(
        t=jnp.array(0, dtype=jnp.int32),
        pos=jnp.array([0, 0], dtype=jnp.int32),
        source_pos=sources,
        source_is_bad=jnp.array([False, False], dtype=jnp.bool_),
        source_active=jnp.array([True, True], dtype=jnp.bool_),
        source_respawn_t=jnp.array([0, 0], dtype=jnp.int32),
        energy=jnp.array(0.5, dtype=jnp.float32),
        integrity=jnp.array(1.0, dtype=jnp.float32),
    )
    dev = DevelopmentState(age_step=jnp.array(0, dtype=jnp.int32), phi=jnp.array(0.0, dtype=jnp.float32))

    # Move onto a source: gain applies.
    state1, _obs1, _internal1, log1, _done1 = env_step(
        spec, state0, jnp.array(4, dtype=jnp.int32), dev, jax.random.PRNGKey(0)
    )
    assert float(jax.device_get(log1.energy_gained)) == 0.25

    # Stay on the source: no additional gain.
    state2, _obs2, _internal2, log2, _done2 = env_step(
        spec, state1, jnp.array(0, dtype=jnp.int32), dev, jax.random.PRNGKey(1)
    )
    assert float(jax.device_get(log2.energy_gained)) == 0.0
