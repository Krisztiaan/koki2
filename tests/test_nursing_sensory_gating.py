import jax
import jax.numpy as jnp

from koki2.envs.chemotaxis import env_step
from koki2.types import ChemotaxisEnvSpec, ChemotaxisEnvState, DevelopmentState


def test_grad_gain_gates_gradient_channels() -> None:
    spec = ChemotaxisEnvSpec(
        width=16,
        height=1,
        max_steps=8,
        energy_init=1.0,
        energy_decay=0.0,
        energy_gain=0.0,
        terminate_on_reach=False,
        obs_noise=0.0,
        grad_gain_min=0.0,
        grad_gain_max=1.0,
        grad_gain_start_phi=0.5,
        grad_gain_end_phi=1.0,
    )
    state = ChemotaxisEnvState(
        t=jnp.array(0, dtype=jnp.int32),
        pos=jnp.array([0, 0], dtype=jnp.int32),
        source_pos=jnp.array([[15, 0]], dtype=jnp.int32),
        source_is_bad=jnp.array([False], dtype=jnp.bool_),
        source_active=jnp.array([True], dtype=jnp.bool_),
        source_respawn_t=jnp.array([0], dtype=jnp.int32),
        energy=jnp.array(1.0, dtype=jnp.float32),
        integrity=jnp.array(1.0, dtype=jnp.float32),
    )
    action = jnp.array(0, dtype=jnp.int32)

    dev0 = DevelopmentState(age_step=jnp.array(0, dtype=jnp.int32), phi=jnp.array(0.0, dtype=jnp.float32))
    _, obs0, _, _, _ = env_step(spec, state, action, dev0, jax.random.PRNGKey(0))

    dev1 = DevelopmentState(age_step=jnp.array(0, dtype=jnp.int32), phi=jnp.array(1.0, dtype=jnp.float32))
    _, obs1, _, _, _ = env_step(spec, state, action, dev1, jax.random.PRNGKey(0))

    assert float(jax.device_get(jnp.abs(obs0[0]) + jnp.abs(obs0[1]))) == 0.0
    assert float(jax.device_get(jnp.abs(obs1[0]) + jnp.abs(obs1[1]))) > 0.0


def test_grad_quantization_coarsens_signal() -> None:
    base = dict(
        width=16,
        height=1,
        max_steps=8,
        energy_init=1.0,
        energy_decay=0.0,
        energy_gain=0.0,
        terminate_on_reach=False,
        obs_noise=0.0,
    )
    state = ChemotaxisEnvState(
        t=jnp.array(0, dtype=jnp.int32),
        pos=jnp.array([0, 0], dtype=jnp.int32),
        source_pos=jnp.array([[5, 0]], dtype=jnp.int32),  # grad_x ~ 5/15 ~= 0.333...
        source_is_bad=jnp.array([False], dtype=jnp.bool_),
        source_active=jnp.array([True], dtype=jnp.bool_),
        source_respawn_t=jnp.array([0], dtype=jnp.int32),
        energy=jnp.array(1.0, dtype=jnp.float32),
        integrity=jnp.array(1.0, dtype=jnp.float32),
    )
    action = jnp.array(0, dtype=jnp.int32)
    dev = DevelopmentState(age_step=jnp.array(0, dtype=jnp.int32), phi=jnp.array(0.0, dtype=jnp.float32))

    spec_coarse = ChemotaxisEnvSpec(**base, grad_bins_min=1.0, grad_bins_max=1.0)
    _, obs_c, _, _, _ = env_step(spec_coarse, state, action, dev, jax.random.PRNGKey(0))

    spec_fine = ChemotaxisEnvSpec(**base, grad_bins_min=10.0, grad_bins_max=10.0)
    _, obs_f, _, _, _ = env_step(spec_fine, state, action, dev, jax.random.PRNGKey(0))

    gx_c = float(jax.device_get(obs_c[0]))
    gx_f = float(jax.device_get(obs_f[0]))
    assert gx_c == 0.0
    assert abs(gx_f - 0.3) < 1e-6
