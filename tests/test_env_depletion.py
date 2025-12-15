import jax
import jax.numpy as jnp

from koki2.envs.chemotaxis import env_step
from koki2.types import ChemotaxisEnvSpec, ChemotaxisEnvState, DevelopmentState


def test_deplete_and_respawn_counts_down_and_resamples_position() -> None:
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
        source_deplete=True,
        source_respawn_delay=2,
    )
    state0 = ChemotaxisEnvState(
        t=jnp.array(0, dtype=jnp.int32),
        pos=jnp.array([0, 0], dtype=jnp.int32),
        source_pos=jnp.array([[1, 0]], dtype=jnp.int32),
        source_is_bad=jnp.array([False], dtype=jnp.bool_),
        source_active=jnp.array([True], dtype=jnp.bool_),
        source_respawn_t=jnp.array([0], dtype=jnp.int32),
        energy=jnp.array(1.0, dtype=jnp.float32),
        integrity=jnp.array(1.0, dtype=jnp.float32),
    )
    dev = DevelopmentState(age_step=jnp.array(0, dtype=jnp.int32), phi=jnp.array(0.0, dtype=jnp.float32))

    # Step onto source: gain, then source depletes and starts countdown.
    key0 = jax.random.PRNGKey(0)
    state1, obs1, _internal1, log1, _done1 = env_step(spec, state0, jnp.array(4, dtype=jnp.int32), dev, key0)
    assert float(jax.device_get(log1.energy_gained)) == 0.25
    assert bool(jax.device_get(log1.reached_source))
    assert bool(jax.device_get(state1.source_active[0])) is False
    assert int(jax.device_get(state1.source_respawn_t[0])) == 2
    assert float(jax.device_get(obs1[0])) == 0.0  # no active sources => zero gradient

    # One more step: countdown decrements.
    key1 = jax.random.PRNGKey(1)
    state2, obs2, _internal2, log2, _done2 = env_step(spec, state1, jnp.array(0, dtype=jnp.int32), dev, key1)
    assert float(jax.device_get(log2.energy_gained)) == 0.0
    assert bool(jax.device_get(state2.source_active[0])) is False
    assert int(jax.device_get(state2.source_respawn_t[0])) == 1
    assert float(jax.device_get(obs2[0])) == 0.0

    # Next step: countdown hits zero and the source respawns (position resampled deterministically from key2).
    key2 = jax.random.PRNGKey(2)
    state3, obs3, _internal3, log3, _done3 = env_step(spec, state2, jnp.array(0, dtype=jnp.int32), dev, key2)

    rng_respawn, _ = jax.random.split(key2, 2)
    hi = jnp.array([spec.width, spec.height], dtype=jnp.int32)
    expected_pos = jax.random.randint(rng_respawn, shape=(1, 2), minval=0, maxval=hi, dtype=jnp.int32)

    assert bool(jax.device_get(state3.source_active[0])) is True
    assert int(jax.device_get(state3.source_respawn_t[0])) == 0
    assert bool(jax.device_get(jnp.all(state3.source_pos == expected_pos)))

    # Gradient now points to the (possibly new) active source.
    # obs[0] is x-gradient normalized by (width-1).
    dx = float(jax.device_get(state3.source_pos[0, 0] - state3.pos[0]))
    expected_gx = dx / float(max(spec.width - 1, 1))
    assert abs(float(jax.device_get(obs3[0])) - expected_gx) < 1e-6
    assert float(jax.device_get(log3.energy_gained)) == 0.0
