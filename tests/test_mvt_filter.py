import jax
import jax.numpy as jnp

from koki2.evo.openai_es import evaluate_population
from koki2.genome.direct import DirectGenome, make_dev_config
from koki2.types import ChemotaxisEnvSpec, EvalConfig, MVTConfig, SimConfig


def _replicate(genome: DirectGenome, pop_size: int) -> DirectGenome:
    return jax.tree_util.tree_map(lambda x: jnp.stack([x] * pop_size, axis=0), genome)


def test_mvt_disabled_does_not_change_fitness() -> None:
    env_spec = ChemotaxisEnvSpec(
        width=1,
        height=1,
        max_steps=8,
        energy_init=1.0,
        energy_decay=0.0,
        energy_gain=0.0,
        terminate_on_reach=False,
        obs_noise=0.0,
    )
    dev_cfg = make_dev_config(
        n_neurons=4,
        obs_dim=4,
        num_actions=5,
        k_edges_per_neuron=1,
        topology_seed=0,
    )
    sim_cfg = SimConfig()
    eval_cfg = EvalConfig(episodes=1)

    # A deterministic, constant-action genome (bias selects action 0).
    n = dev_cfg.n_neurons
    e = dev_cfg.edge_index.shape[0]
    a = dev_cfg.num_actions
    genome = DirectGenome(
        obs_w=jnp.zeros((n, dev_cfg.obs_dim), dtype=jnp.float32),
        rec_w=jnp.zeros((e,), dtype=jnp.float32),
        motor_w=jnp.zeros((n, a), dtype=jnp.float32),
        motor_b=jnp.array([10.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        mod_w=jnp.zeros((n,), dtype=jnp.float32),
    )
    pop = _replicate(genome, pop_size=4)

    rng = jax.random.PRNGKey(0)
    fit0, mask0 = evaluate_population(
        pop, env_spec=env_spec, dev_cfg=dev_cfg, sim_cfg=sim_cfg, eval_cfg=eval_cfg, rng=rng
    )
    fit1, mask1 = evaluate_population(
        pop,
        env_spec=env_spec,
        dev_cfg=dev_cfg,
        sim_cfg=sim_cfg,
        eval_cfg=eval_cfg,
        mvt_cfg=MVTConfig(enabled=False),
        rng=rng,
    )

    assert mask0 is None
    assert mask1 is None
    assert jnp.all(fit0 == fit1)


def test_mvt_strict_thresholds_can_filter_population() -> None:
    env_spec = ChemotaxisEnvSpec(
        width=1,
        height=1,
        max_steps=8,
        energy_init=1.0,
        energy_decay=0.0,
        energy_gain=0.0,
        terminate_on_reach=False,
        obs_noise=0.0,
    )
    dev_cfg = make_dev_config(
        n_neurons=4,
        obs_dim=4,
        num_actions=5,
        k_edges_per_neuron=1,
        topology_seed=0,
    )
    sim_cfg = SimConfig()
    eval_cfg = EvalConfig(episodes=1)

    n = dev_cfg.n_neurons
    e = dev_cfg.edge_index.shape[0]
    a = dev_cfg.num_actions
    genome = DirectGenome(
        obs_w=jnp.zeros((n, dev_cfg.obs_dim), dtype=jnp.float32),
        rec_w=jnp.zeros((e,), dtype=jnp.float32),
        motor_w=jnp.zeros((n, a), dtype=jnp.float32),
        motor_b=jnp.array([10.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        mod_w=jnp.zeros((n,), dtype=jnp.float32),
    )
    pop = _replicate(genome, pop_size=4)

    mvt_cfg = MVTConfig(
        enabled=True,
        episodes=1,
        steps=8,
        min_alive_steps=1,
        min_action_entropy=0.1,  # constant action -> entropy 0 -> fail
        min_energy_gained=0.0,
        fail_fitness=-123.0,
    )
    fit, mvt_pass = evaluate_population(
        pop,
        env_spec=env_spec,
        dev_cfg=dev_cfg,
        sim_cfg=sim_cfg,
        eval_cfg=eval_cfg,
        mvt_cfg=mvt_cfg,
        rng=jax.random.PRNGKey(0),
    )

    assert mvt_pass is not None
    assert bool(jax.device_get(jnp.all(mvt_pass == jnp.array(False))))
    assert bool(jax.device_get(jnp.all(fit == jnp.array(mvt_cfg.fail_fitness, dtype=jnp.float32))))

