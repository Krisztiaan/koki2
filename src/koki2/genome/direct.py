from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from koki2.types import AgentParams, Array, DevConfig


class DirectGenome(NamedTuple):
    obs_w: Array  # float32[N, obs_dim]
    rec_w: Array  # float32[E]
    motor_w: Array  # float32[N, A]
    motor_b: Array  # float32[A]
    mod_w: Array  # float32[N]


def make_dev_config(
    *,
    n_neurons: int,
    obs_dim: int,
    num_actions: int,
    k_edges_per_neuron: int,
    topology_seed: int,
    theta: float = 1.0,
    tau_m: float = 10.0,
    plast_enabled: bool = False,
    plast_eta: float = 0.0,
    plast_lambda: float = 0.9,
) -> DevConfig:
    if n_neurons < 1:
        raise ValueError("n_neurons must be >= 1")
    if obs_dim < 1:
        raise ValueError("obs_dim must be >= 1")
    if num_actions < 1:
        raise ValueError("num_actions must be >= 1")

    max_k = max(1, n_neurons - 1)
    k = int(min(k_edges_per_neuron, max_k))

    rng = np.random.default_rng(topology_seed)
    src = np.repeat(np.arange(n_neurons, dtype=np.int32), k)
    dst = np.empty((n_neurons * k,), dtype=np.int32)
    all_nodes = np.arange(n_neurons, dtype=np.int32)
    for i in range(n_neurons):
        candidates = all_nodes[all_nodes != i]
        dst[i * k : (i + 1) * k] = rng.choice(candidates, size=(k,), replace=False)

    edge_index = jnp.asarray(np.stack([src, dst], axis=1), dtype=jnp.int32)
    return DevConfig(
        n_neurons=n_neurons,
        obs_dim=obs_dim,
        num_actions=num_actions,
        edge_index=edge_index,
        theta=theta,
        tau_m=tau_m,
        plast_enabled=plast_enabled,
        plast_eta=plast_eta,
        plast_lambda=plast_lambda,
    )


def genome_init(rng: Array, dev_cfg: DevConfig, *, scale: float = 0.1) -> DirectGenome:
    rng_obs, rng_rec, rng_motor_w, rng_motor_b, rng_mod = jax.random.split(rng, 5)
    n = dev_cfg.n_neurons
    e = dev_cfg.edge_index.shape[0]
    a = dev_cfg.num_actions

    obs_w = scale * jax.random.normal(rng_obs, shape=(n, dev_cfg.obs_dim), dtype=jnp.float32)
    rec_w = scale * jax.random.normal(rng_rec, shape=(e,), dtype=jnp.float32)
    motor_w = scale * jax.random.normal(rng_motor_w, shape=(n, a), dtype=jnp.float32)
    motor_b = scale * jax.random.normal(rng_motor_b, shape=(a,), dtype=jnp.float32)
    mod_w = scale * jax.random.normal(rng_mod, shape=(n,), dtype=jnp.float32)
    return DirectGenome(obs_w=obs_w, rec_w=rec_w, motor_w=motor_w, motor_b=motor_b, mod_w=mod_w)


def develop(genome: DirectGenome, dev_cfg: DevConfig, rng: Array) -> AgentParams:
    del rng
    n = genome.obs_w.shape[0]
    v_decay = jnp.full((n,), jnp.exp(jnp.array(-1.0 / dev_cfg.tau_m, dtype=jnp.float32)), dtype=jnp.float32)
    theta = jnp.full((n,), jnp.array(dev_cfg.theta, dtype=jnp.float32), dtype=jnp.float32)
    return AgentParams(
        obs_w=genome.obs_w,
        edge_index=dev_cfg.edge_index,
        w0=genome.rec_w,
        motor_w=genome.motor_w,
        motor_b=genome.motor_b,
        mod_w=genome.mod_w,
        v_decay=v_decay,
        theta=theta,
        plast_enabled=jnp.array(dev_cfg.plast_enabled, dtype=jnp.bool_),
        plast_eta=jnp.array(dev_cfg.plast_eta, dtype=jnp.float32),
        plast_lambda=jnp.array(dev_cfg.plast_lambda, dtype=jnp.float32),
    )
