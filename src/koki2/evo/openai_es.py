from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from koki2.genome.direct import DirectGenome, genome_init
from koki2.ops.run_io import append_jsonl, ensure_dir, write_json
from koki2.sim.orchestrator import simulate_lifetime
from koki2.types import ChemotaxisEnvSpec, DevConfig, ESConfig, EvalConfig, MVTConfig, SimConfig


def _centered_ranks(x: jax.Array) -> jax.Array:
    x = x.astype(jnp.float32)
    ranks = jnp.argsort(jnp.argsort(x))
    ranks = ranks.astype(jnp.float32)
    denom = jnp.maximum(x.size - 1, 1)
    return (ranks / denom) - 0.5


def evaluate_population(
    pop: DirectGenome,
    *,
    env_spec: ChemotaxisEnvSpec,
    dev_cfg: DevConfig,
    sim_cfg: SimConfig,
    eval_cfg: EvalConfig,
    mvt_cfg: MVTConfig | None = None,
    rng: jax.Array,
) -> tuple[jax.Array, jax.Array | None]:
    pop_size = pop.obs_w.shape[0]
    keys = jax.random.split(rng, pop_size)
    t_idx = jnp.arange(env_spec.max_steps, dtype=jnp.int32)

    def eval_one(genome: DirectGenome, key: jax.Array) -> jax.Array:
        ep_keys = jax.random.split(key, eval_cfg.episodes)
        fitness = jax.vmap(
            lambda k: simulate_lifetime(genome, env_spec, dev_cfg, sim_cfg, k, t_idx).fitness_scalar
        )(
            ep_keys
        )
        return jnp.mean(fitness)

    fitness = jax.vmap(eval_one)(pop, keys)

    if mvt_cfg is None or not mvt_cfg.enabled:
        return fitness, None

    mvt_env = env_spec._replace(max_steps=mvt_cfg.steps)
    t_mvt = jnp.arange(mvt_cfg.steps, dtype=jnp.int32)

    def mvt_one(genome: DirectGenome, key: jax.Array) -> jax.Array:
        ep_keys = jax.random.split(key, mvt_cfg.episodes)
        out = jax.vmap(lambda k: simulate_lifetime(genome, mvt_env, dev_cfg, sim_cfg, k, t_mvt))(ep_keys)
        alive_ok = out.t_alive >= jnp.array(mvt_cfg.min_alive_steps, dtype=jnp.int32)
        entropy_ok = out.action_entropy >= jnp.array(mvt_cfg.min_action_entropy, dtype=jnp.float32)
        energy_ok = out.energy_gained_total >= jnp.array(mvt_cfg.min_energy_gained, dtype=jnp.float32)
        finite_ok = jnp.isfinite(out.fitness_scalar)
        ok = alive_ok & entropy_ok & energy_ok & finite_ok
        return jnp.any(ok)

    mvt_pass = jax.vmap(mvt_one)(pop, keys)
    fitness = jnp.where(
        mvt_pass,
        fitness,
        jnp.array(mvt_cfg.fail_fitness, dtype=jnp.float32),
    )
    return fitness, mvt_pass


@dataclass(frozen=True)
class ESResult:
    best_genome: DirectGenome
    best_fitness: float


def run_openai_es(
    *,
    seed: int,
    out_dir: str | Path | None,
    env_spec: ChemotaxisEnvSpec,
    dev_cfg: DevConfig,
    sim_cfg: SimConfig,
    eval_cfg: EvalConfig,
    es_cfg: ESConfig,
    mvt_cfg: MVTConfig | None = None,
) -> ESResult:
    master = jax.random.PRNGKey(seed)
    init_key = jax.random.fold_in(master, 0)
    mean = genome_init(init_key, dev_cfg, scale=0.1)

    mean_vec, unravel = ravel_pytree(mean)
    dim = mean_vec.shape[0]

    out_path = Path(out_dir) if out_dir is not None else None
    if out_path is not None:
        ensure_dir(out_path)
        write_json(
            out_path / "config.json",
            {
                "seed": seed,
                "env_spec": env_spec._asdict(),
                "dev_cfg": {**dev_cfg._asdict(), "edge_index": None, "edge_index_shape": tuple(dev_cfg.edge_index.shape)},
                "sim_cfg": sim_cfg._asdict(),
                "eval_cfg": eval_cfg._asdict(),
                "es_cfg": es_cfg._asdict(),
                "mvt_cfg": None if mvt_cfg is None else mvt_cfg._asdict(),
            },
        )

    best_fitness = -float("inf")
    best_genome = mean

    for gen in range(es_cfg.generations):
        gen_key = jax.random.fold_in(master, gen + 1)
        key_eps, key_eval = jax.random.split(gen_key, 2)

        eps = jax.random.normal(key_eps, shape=(es_cfg.pop_size, dim), dtype=jnp.float32)
        pop_vec = mean_vec[None, :] + jnp.array(es_cfg.sigma, dtype=jnp.float32) * eps
        pop = jax.vmap(unravel)(pop_vec)

        fitness, mvt_pass = evaluate_population(
            pop,
            env_spec=env_spec,
            dev_cfg=dev_cfg,
            sim_cfg=sim_cfg,
            eval_cfg=eval_cfg,
            mvt_cfg=mvt_cfg,
            rng=key_eval,
        )

        weights = _centered_ranks(fitness)
        grad = jnp.dot(weights, eps) / (jnp.array(es_cfg.pop_size, dtype=jnp.float32) * jnp.array(es_cfg.sigma, dtype=jnp.float32))
        mean_vec = mean_vec + jnp.array(es_cfg.lr, dtype=jnp.float32) * grad
        mean = unravel(mean_vec)

        gen_best_idx = int(jax.device_get(jnp.argmax(fitness)))
        gen_best_fit = float(jax.device_get(fitness[gen_best_idx]))
        if gen_best_fit > best_fitness:
            best_fitness = gen_best_fit
            best_genome = jax.tree_util.tree_map(lambda x, idx=gen_best_idx: x[idx], pop)

        if out_path is not None:
            mvt_pass_rate = None if mvt_pass is None else float(jax.device_get(jnp.mean(mvt_pass.astype(jnp.float32))))
            append_jsonl(
                out_path / "generations.jsonl",
                {
                    "generation": gen,
                    "best_fitness": gen_best_fit,
                    "mean_fitness": float(jax.device_get(jnp.mean(fitness))),
                    "median_fitness": float(jax.device_get(jnp.median(fitness))),
                    "mvt_pass_rate": mvt_pass_rate,
                },
            )

    if out_path is not None:
        # Save best genome as a numpy archive for easy reload.
        import numpy as np

        best_np: dict[str, Any] = jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), best_genome)._asdict()
        np.savez(out_path / "best_genome.npz", **best_np)

    return ESResult(best_genome=best_genome, best_fitness=best_fitness)
