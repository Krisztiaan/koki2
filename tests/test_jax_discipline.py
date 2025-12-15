from collections.abc import Iterable

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from koki2.evo.openai_es import evaluate_population, make_openai_es_jit_runner
from koki2.genome.direct import genome_init, make_dev_config
from koki2.sim.orchestrator import simulate_lifetime_fitness
from koki2.types import ChemotaxisEnvSpec, ESConfig, EvalConfig, MVTConfig, SimConfig


def _iter_nested_jaxprs(obj) -> Iterable[object]:
    if obj is None:
        return
    tname = type(obj).__name__
    if tname == "ClosedJaxpr":
        yield obj.jaxpr
        return
    if tname == "Jaxpr":
        yield obj
        return
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_nested_jaxprs(v)
        return
    if isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_nested_jaxprs(v)
        return


def _count_primitives(jaxpr) -> int:
    n = 0

    def walk(jp) -> None:
        nonlocal n
        for eqn in jp.eqns:
            n += 1
            for nested in _iter_nested_jaxprs(eqn.params):
                walk(nested)

    walk(jaxpr)
    return n


def test_no_recompiles_es_runner_same_shape() -> None:
    env_spec = ChemotaxisEnvSpec(
        width=16,
        height=1,
        max_steps=32,
        energy_init=1.0,
        energy_decay=1.0 / 32,
        energy_gain=0.05,
        terminate_on_reach=False,
        obs_noise=0.0,
    )
    sim_cfg = SimConfig()
    eval_cfg = EvalConfig(episodes=2)
    es_cfg = ESConfig(pop_size=8, generations=3, sigma=0.1, lr=0.05)
    mvt_cfg = MVTConfig(enabled=False)

    dev_cfg0 = make_dev_config(
        n_neurons=8,
        obs_dim=4,
        num_actions=5,
        k_edges_per_neuron=4,
        topology_seed=0,
    )
    mean0 = genome_init(jax.random.PRNGKey(0), dev_cfg0, scale=0.1)
    mean_vec0, unravel = ravel_pytree(mean0)
    dim = int(mean_vec0.shape[0])

    es_run = make_openai_es_jit_runner(
        env_spec=env_spec,
        sim_cfg=sim_cfg,
        eval_cfg=eval_cfg,
        es_cfg=es_cfg,
        mvt_cfg=mvt_cfg,
        unravel=unravel,
        dim=dim,
        log_every=1,
    )

    assert es_run._cache_size() == 0
    best_fit0, _, _ = es_run(jax.random.PRNGKey(0), mean_vec0, dev_cfg0)
    best_fit0.block_until_ready()
    assert es_run._cache_size() == 1

    dev_cfg1 = make_dev_config(
        n_neurons=8,
        obs_dim=4,
        num_actions=5,
        k_edges_per_neuron=4,
        topology_seed=123,
    )
    best_fit1, _, _ = es_run(jax.random.PRNGKey(1), mean_vec0, dev_cfg1)
    best_fit1.block_until_ready()
    assert es_run._cache_size() == 1


def test_check_tracer_leaks_es_runner() -> None:
    env_spec = ChemotaxisEnvSpec(
        width=16,
        height=1,
        max_steps=16,
        energy_init=1.0,
        energy_decay=1.0 / 16,
        energy_gain=0.05,
        terminate_on_reach=False,
        obs_noise=0.0,
    )
    sim_cfg = SimConfig()
    eval_cfg = EvalConfig(episodes=2)
    es_cfg = ESConfig(pop_size=4, generations=2, sigma=0.1, lr=0.05)
    mvt_cfg = MVTConfig(enabled=False)

    with jax.check_tracer_leaks():
        dev_cfg = make_dev_config(
            n_neurons=8,
            obs_dim=4,
            num_actions=5,
            k_edges_per_neuron=4,
            topology_seed=0,
        )
        mean = genome_init(jax.random.PRNGKey(0), dev_cfg, scale=0.1)
        mean_vec, unravel = ravel_pytree(mean)
        dim = int(mean_vec.shape[0])

        es_run = make_openai_es_jit_runner(
            env_spec=env_spec,
            sim_cfg=sim_cfg,
            eval_cfg=eval_cfg,
            es_cfg=es_cfg,
            mvt_cfg=mvt_cfg,
            unravel=unravel,
            dim=dim,
            log_every=1,
        )
        best_fit, _, _ = es_run(jax.random.PRNGKey(0), mean_vec, dev_cfg)
        best_fit.block_until_ready()


def test_jaxpr_size_budget() -> None:
    env_spec = ChemotaxisEnvSpec(
        width=16,
        height=1,
        max_steps=32,
        energy_init=1.0,
        energy_decay=1.0 / 32,
        energy_gain=0.05,
        terminate_on_reach=False,
        obs_noise=0.0,
    )
    dev_cfg = make_dev_config(
        n_neurons=8,
        obs_dim=4,
        num_actions=5,
        k_edges_per_neuron=4,
        topology_seed=0,
    )
    sim_cfg = SimConfig()
    t_idx = jnp.arange(env_spec.max_steps, dtype=jnp.int32)
    genome = genome_init(jax.random.PRNGKey(0), dev_cfg, scale=0.1)

    jp_fit = jax.make_jaxpr(simulate_lifetime_fitness)(genome, env_spec, dev_cfg, sim_cfg, jax.random.PRNGKey(1), t_idx)
    assert _count_primitives(jp_fit.jaxpr) < 5_000

    pop = jax.tree_util.tree_map(lambda x: jnp.stack([x] * 4, axis=0), genome)
    eval_cfg = EvalConfig(episodes=2)
    mvt_cfg = MVTConfig(
        enabled=True,
        episodes=2,
        steps=16,
        min_alive_steps=8,
        min_action_entropy=0.0,
        min_energy_gained=0.0,
        fail_fitness=-1e9,
    )

    def eval_fixed(pop, rng):
        return evaluate_population(
            pop,
            env_spec=env_spec,
            dev_cfg=dev_cfg,
            sim_cfg=sim_cfg,
            eval_cfg=eval_cfg,
            mvt_cfg=mvt_cfg,
            rng=rng,
        )

    jp_eval = jax.make_jaxpr(eval_fixed)(pop, jax.random.PRNGKey(0))
    assert _count_primitives(jp_eval.jaxpr) < 8_000
