import jax
import jax.numpy as jnp

from koki2.genome.direct import genome_init, make_dev_config
from koki2.sim.orchestrator import simulate_lifetime, simulate_lifetime_fitness, simulate_lifetime_mvt
from koki2.types import ChemotaxisEnvSpec, SimConfig


def _assert_close(a, b) -> None:
    assert bool(jax.device_get(jnp.allclose(a, b)))


def test_fastpaths_match_full_non_plastic() -> None:
    env_spec = ChemotaxisEnvSpec(
        width=16,
        height=1,
        max_steps=32,
        energy_init=1.0,
        energy_decay=0.0,
        energy_gain=0.1,
        terminate_on_reach=False,
        obs_noise=0.0,
    )
    dev_cfg = make_dev_config(
        n_neurons=8,
        obs_dim=4,
        num_actions=5,
        k_edges_per_neuron=2,
        topology_seed=0,
        plast_enabled=False,
        plast_eta=0.0,
        modulator_kind=0,
    )
    sim_cfg = SimConfig()
    t_idx = jnp.arange(env_spec.max_steps, dtype=jnp.int32)

    genome = genome_init(jax.random.PRNGKey(0), dev_cfg, scale=0.1)
    rng = jax.random.PRNGKey(42)

    full = simulate_lifetime(genome, env_spec, dev_cfg, sim_cfg, rng, t_idx)
    fast_fit = simulate_lifetime_fitness(genome, env_spec, dev_cfg, sim_cfg, rng, t_idx)
    fast_mvt = simulate_lifetime_mvt(genome, env_spec, dev_cfg, sim_cfg, rng, t_idx)

    _assert_close(full.fitness_scalar, fast_fit)
    _assert_close(full.fitness_scalar, fast_mvt.fitness_scalar)
    _assert_close(full.t_alive, fast_mvt.t_alive)
    _assert_close(full.energy_gained_total, fast_mvt.energy_gained_total)
    _assert_close(full.action_entropy, fast_mvt.action_entropy)


def test_fastpaths_match_full_with_plasticity() -> None:
    env_spec = ChemotaxisEnvSpec(
        width=16,
        height=1,
        max_steps=32,
        energy_init=1.0,
        energy_decay=0.0,
        energy_gain=0.1,
        terminate_on_reach=False,
        obs_noise=0.0,
    )
    dev_cfg = make_dev_config(
        n_neurons=8,
        obs_dim=4,
        num_actions=5,
        k_edges_per_neuron=2,
        topology_seed=0,
        theta=0.5,
        plast_enabled=True,
        plast_eta=0.05,
        plast_lambda=0.9,
        modulator_kind=0,
    )
    sim_cfg = SimConfig()
    t_idx = jnp.arange(env_spec.max_steps, dtype=jnp.int32)

    genome = genome_init(jax.random.PRNGKey(0), dev_cfg, scale=0.1)
    rng = jax.random.PRNGKey(42)

    full = simulate_lifetime(genome, env_spec, dev_cfg, sim_cfg, rng, t_idx)
    fast_fit = simulate_lifetime_fitness(genome, env_spec, dev_cfg, sim_cfg, rng, t_idx)
    fast_mvt = simulate_lifetime_mvt(genome, env_spec, dev_cfg, sim_cfg, rng, t_idx)

    _assert_close(full.fitness_scalar, fast_fit)
    _assert_close(full.fitness_scalar, fast_mvt.fitness_scalar)
    _assert_close(full.t_alive, fast_mvt.t_alive)
    _assert_close(full.energy_gained_total, fast_mvt.energy_gained_total)
    _assert_close(full.action_entropy, fast_mvt.action_entropy)

