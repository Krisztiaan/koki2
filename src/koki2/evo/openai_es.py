from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from koki2.genome.direct import DirectGenome, genome_init
from koki2.ops.run_io import append_jsonl, append_jsonl_many, ensure_dir, write_json
from koki2.sim.orchestrator import simulate_lifetime_fitness, simulate_lifetime_mvt
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

    # Pre-split episode keys outside the per-genome eval to reduce overhead, while preserving
    # the exact RNG schedule (each genome key is split into `eval_cfg.episodes` keys).
    ep_keys = jax.vmap(lambda k: jax.random.split(k, eval_cfg.episodes))(keys)

    def eval_one(genome: DirectGenome, genome_ep_keys: jax.Array) -> jax.Array:
        fitness = jax.vmap(lambda k: simulate_lifetime_fitness(genome, env_spec, dev_cfg, sim_cfg, k, t_idx))(
            genome_ep_keys
        )
        return jnp.mean(fitness)

    fitness = jax.vmap(eval_one)(pop, ep_keys)

    if mvt_cfg is None or not mvt_cfg.enabled:
        return fitness, None

    mvt_env = env_spec._replace(max_steps=mvt_cfg.steps)
    t_mvt = jnp.arange(mvt_cfg.steps, dtype=jnp.int32)
    mvt_ep_keys = jax.vmap(lambda k: jax.random.split(k, mvt_cfg.episodes))(keys)

    def mvt_one(genome: DirectGenome, genome_ep_keys: jax.Array) -> jax.Array:
        out = jax.vmap(lambda k: simulate_lifetime_mvt(genome, mvt_env, dev_cfg, sim_cfg, k, t_mvt))(genome_ep_keys)
        alive_ok = out.t_alive >= jnp.array(mvt_cfg.min_alive_steps, dtype=jnp.int32)
        entropy_ok = out.action_entropy >= jnp.array(mvt_cfg.min_action_entropy, dtype=jnp.float32)
        energy_ok = out.energy_gained_total >= jnp.array(mvt_cfg.min_energy_gained, dtype=jnp.float32)
        finite_ok = jnp.isfinite(out.fitness_scalar)
        ok = alive_ok & entropy_ok & energy_ok & finite_ok
        return jnp.any(ok)

    mvt_pass = jax.vmap(mvt_one)(pop, mvt_ep_keys)
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


class ESGenerationLogs(NamedTuple):
    # float32[generations] with NaNs for generations that were not logged.
    best_fitness: jax.Array
    mean_fitness: jax.Array
    median_fitness: jax.Array
    # float32[generations] with NaNs for generations that were not logged, or if MVT is disabled.
    mvt_pass_rate: jax.Array


def make_openai_es_jit_runner(
    *,
    env_spec: ChemotaxisEnvSpec,
    sim_cfg: SimConfig,
    eval_cfg: EvalConfig,
    es_cfg: ESConfig,
    mvt_cfg: MVTConfig | None,
    unravel: Callable[[jax.Array], DirectGenome],
    dim: int,
    log_every: int,
) -> Callable[[jax.Array, jax.Array, DevConfig], tuple[jax.Array, jax.Array, ESGenerationLogs]]:
    """Create a reusable jit+scan ES runner.

    This keeps the ES loop on-device and returns per-generation logs as arrays.
    Crucially, the RNG master key is an argument so the compiled code can be reused across seeds.
    """
    if log_every < 1:
        raise ValueError("log_every must be >= 1")

    pop_size = int(es_cfg.pop_size)
    sigma = jnp.array(es_cfg.sigma, dtype=jnp.float32)
    lr = jnp.array(es_cfg.lr, dtype=jnp.float32)
    pop_size_f = jnp.array(pop_size, dtype=jnp.float32)
    log_every_i = jnp.array(log_every, dtype=jnp.int32)
    nan = jnp.array(jnp.nan, dtype=jnp.float32)

    if mvt_cfg is None or not mvt_cfg.enabled:

        def eval_pop(pop: DirectGenome, dev_cfg: DevConfig, key: jax.Array) -> tuple[jax.Array, jax.Array]:
            fit, _ = evaluate_population(
                pop,
                env_spec=env_spec,
                dev_cfg=dev_cfg,
                sim_cfg=sim_cfg,
                eval_cfg=eval_cfg,
                mvt_cfg=None,
                rng=key,
            )
            return fit, nan

    else:

        def eval_pop(pop: DirectGenome, dev_cfg: DevConfig, key: jax.Array) -> tuple[jax.Array, jax.Array]:
            fit, mvt_pass = evaluate_population(
                pop,
                env_spec=env_spec,
                dev_cfg=dev_cfg,
                sim_cfg=sim_cfg,
                eval_cfg=eval_cfg,
                mvt_cfg=mvt_cfg,
                rng=key,
            )
            pass_rate = jnp.mean(mvt_pass.astype(jnp.float32))
            return fit, pass_rate

    eval_pop_jit = jax.jit(eval_pop)

    def es_run(master_key: jax.Array, mean_vec: jax.Array, dev_cfg: DevConfig) -> tuple[jax.Array, jax.Array, ESGenerationLogs]:
        best_fit0 = jnp.array(-jnp.inf, dtype=jnp.float32)
        best_vec0 = mean_vec

        def step(carry: tuple[jax.Array, jax.Array, jax.Array], gen: jax.Array):
            mean_vec, best_fit, best_vec = carry

            gen_key = jax.random.fold_in(master_key, gen + jnp.array(1, dtype=jnp.int32))
            key_eps, key_eval = jax.random.split(gen_key, 2)

            eps = jax.random.normal(key_eps, shape=(pop_size, dim), dtype=jnp.float32)
            pop_vec = mean_vec[None, :] + sigma * eps
            pop = jax.vmap(unravel)(pop_vec)

            fitness, mvt_pass_rate = eval_pop_jit(pop, dev_cfg, key_eval)
            weights = _centered_ranks(fitness)
            grad = jnp.dot(weights, eps) / (pop_size_f * sigma)
            mean_vec_next = mean_vec + lr * grad

            gen_best_idx = jnp.argmax(fitness)
            gen_best_fit = fitness[gen_best_idx]
            gen_best_vec = pop_vec[gen_best_idx]
            better = gen_best_fit > best_fit
            best_fit_next = jnp.where(better, gen_best_fit, best_fit)
            best_vec_next = jnp.where(better, gen_best_vec, best_vec)

            do_log = (gen % log_every_i) == jnp.array(0, dtype=jnp.int32)

            def _log_stats(_):
                return (
                    gen_best_fit,
                    jnp.mean(fitness),
                    jnp.median(fitness),
                    mvt_pass_rate,
                )

            def _skip_stats(_):
                return (nan, nan, nan, nan)

            best_fit_log, mean_fit_log, median_fit_log, mvt_pass_rate_log = jax.lax.cond(
                do_log, _log_stats, _skip_stats, operand=None
            )

            logs = ESGenerationLogs(
                best_fitness=best_fit_log,
                mean_fitness=mean_fit_log,
                median_fitness=median_fit_log,
                mvt_pass_rate=mvt_pass_rate_log,
            )
            return (mean_vec_next, best_fit_next, best_vec_next), logs

        gens = jnp.arange(es_cfg.generations, dtype=jnp.int32)
        (_, best_fit_final, best_vec_final), logs = jax.lax.scan(step, (mean_vec, best_fit0, best_vec0), gens)
        return best_fit_final, best_vec_final, logs

    return jax.jit(es_run)


def run_openai_es(
    *,
    seed: int,
    out_dir: str | Path | None,
    env_spec: ChemotaxisEnvSpec,
    dev_cfg: DevConfig,
    topology_seed: int | None = None,
    sim_cfg: SimConfig,
    eval_cfg: EvalConfig,
    es_cfg: ESConfig,
    mvt_cfg: MVTConfig | None = None,
    jit_es: bool = False,
    log_every: int = 1,
) -> ESResult:
    if log_every < 1:
        raise ValueError("log_every must be >= 1")

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
                "dev_cfg": {
                    **dev_cfg._asdict(),
                    "edge_index": None,
                    "edge_index_shape": tuple(dev_cfg.edge_index.shape),
                    "topology_seed": topology_seed,
                },
                "sim_cfg": sim_cfg._asdict(),
                "eval_cfg": eval_cfg._asdict(),
                "es_cfg": es_cfg._asdict(),
                "mvt_cfg": None if mvt_cfg is None else mvt_cfg._asdict(),
            },
        )

    best_fitness = -float("inf")
    best_genome = mean

    if jit_es:
        es_run = make_openai_es_jit_runner(
            env_spec=env_spec,
            sim_cfg=sim_cfg,
            eval_cfg=eval_cfg,
            es_cfg=es_cfg,
            mvt_cfg=mvt_cfg,
            unravel=unravel,
            dim=dim,
            log_every=log_every,
        )
        best_fit_final, best_vec_final, logs = es_run(master, mean_vec, dev_cfg)
        best_genome = unravel(best_vec_final)
        best_fitness = float(jax.device_get(best_fit_final))

        if out_path is not None:
            host_logs = jax.device_get(logs)
            host_best = np.asarray(host_logs.best_fitness)
            host_mean = np.asarray(host_logs.mean_fitness)
            host_median = np.asarray(host_logs.median_fitness)
            host_mvt = np.asarray(host_logs.mvt_pass_rate)

            records: list[dict[str, Any]] = []
            for gen in range(int(es_cfg.generations)):
                if np.isnan(host_best[gen]):
                    continue
                mvt_val = None if np.isnan(host_mvt[gen]) else float(host_mvt[gen])
                records.append(
                    {
                        "generation": int(gen),
                        "best_fitness": float(host_best[gen]),
                        "mean_fitness": float(host_mean[gen]),
                        "median_fitness": float(host_median[gen]),
                        "mvt_pass_rate": mvt_val,
                    }
                )
            append_jsonl_many(out_path / "generations.jsonl", records)

            best_np: dict[str, Any] = jax.tree_util.tree_map(
                lambda x: np.asarray(jax.device_get(x)), best_genome
            )._asdict()
            np.savez(out_path / "best_genome.npz", **best_np)

        return ESResult(best_genome=best_genome, best_fitness=best_fitness)

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
            if gen % log_every == 0:
                mvt_pass_rate = None if mvt_pass is None else float(
                    jax.device_get(jnp.mean(mvt_pass.astype(jnp.float32)))
                )
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
        best_np: dict[str, Any] = jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), best_genome)._asdict()
        np.savez(out_path / "best_genome.npz", **best_np)

    return ESResult(best_genome=best_genome, best_fitness=best_fitness)
