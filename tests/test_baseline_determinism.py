import jax
import jax.numpy as jnp

from koki2.sim.orchestrator import (
    simulate_lifetime_baseline_greedy,
    simulate_lifetime_baseline_random,
    simulate_lifetime_baseline_stay,
)
from koki2.types import ChemotaxisEnvSpec, SimConfig


def test_baseline_rollouts_are_deterministic() -> None:
    env_spec = ChemotaxisEnvSpec(
        width=16,
        height=1,
        max_steps=64,
        energy_init=1.0,
        energy_decay=1.0 / 64,
        energy_gain=0.05,
        terminate_on_reach=False,
        obs_noise=0.0,
        num_sources=2,
    )
    sim_cfg = SimConfig()
    rng = jax.random.PRNGKey(0)
    t_idx = jnp.arange(env_spec.max_steps, dtype=jnp.int32)

    for fn in (simulate_lifetime_baseline_greedy, simulate_lifetime_baseline_random, simulate_lifetime_baseline_stay):
        out1 = fn(env_spec, sim_cfg, rng, t_idx)
        out2 = fn(env_spec, sim_cfg, rng, t_idx)
        assert float(jax.device_get(out1.fitness_scalar)) == float(jax.device_get(out2.fitness_scalar))
        assert int(jax.device_get(out1.t_alive)) == int(jax.device_get(out2.t_alive))
        assert float(jax.device_get(out1.energy_gained_total)) == float(jax.device_get(out2.energy_gained_total))
        assert float(jax.device_get(out1.bad_arrivals_total)) == float(jax.device_get(out2.bad_arrivals_total))
        assert float(jax.device_get(out1.integrity_lost_total)) == float(jax.device_get(out2.integrity_lost_total))
        assert float(jax.device_get(out1.integrity_min)) == float(jax.device_get(out2.integrity_min))
        assert bool(jax.device_get(out1.success)) == bool(jax.device_get(out2.success))


def test_baseline_jit_matches_eager() -> None:
    env_spec = ChemotaxisEnvSpec(
        width=16,
        height=1,
        max_steps=32,
        energy_init=1.0,
        energy_decay=1.0 / 32,
        energy_gain=0.05,
        terminate_on_reach=False,
        obs_noise=0.0,
        num_sources=3,
    )
    sim_cfg = SimConfig()
    rng = jax.random.PRNGKey(1)
    t_idx = jnp.arange(env_spec.max_steps, dtype=jnp.int32)

    eager = simulate_lifetime_baseline_greedy(env_spec, sim_cfg, rng, t_idx)
    jitted = jax.jit(simulate_lifetime_baseline_greedy)(env_spec, sim_cfg, rng, t_idx)
    assert float(jax.device_get(eager.fitness_scalar)) == float(jax.device_get(jitted.fitness_scalar))
    assert float(jax.device_get(eager.bad_arrivals_total)) == float(jax.device_get(jitted.bad_arrivals_total))
    assert float(jax.device_get(eager.integrity_lost_total)) == float(jax.device_get(jitted.integrity_lost_total))
    assert float(jax.device_get(eager.integrity_min)) == float(jax.device_get(jitted.integrity_min))
