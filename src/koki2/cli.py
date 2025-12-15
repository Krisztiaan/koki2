from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp

from koki2.evo.openai_es import run_openai_es
from koki2.genome.direct import make_dev_config
from koki2.ops.manifest import collect_manifest, write_manifest
from koki2.ops.run_io import ensure_dir, utc_now_iso
from koki2.sim.orchestrator import (
    simulate_lifetime_baseline_greedy,
    simulate_lifetime_baseline_random,
    simulate_lifetime_baseline_stay,
)
from koki2.types import ChemotaxisEnvSpec, ESConfig, EvalConfig, MVTConfig, SimConfig


def _default_out_dir(tag: str, seed: int) -> Path:
    stamp = utc_now_iso().replace(":", "").replace("+", "").replace(".", "")
    return Path("runs") / f"{stamp}_{tag}_seed{seed}"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="koki2")
    sub = p.add_subparsers(dest="cmd", required=True)

    evo = sub.add_parser("evo-l0", help="Run a tiny OpenAI-ES loop on L0 chemotaxis.")
    evo.add_argument("--seed", type=int, default=0)
    evo.add_argument("--out-dir", type=str, default=None)

    evo.add_argument("--width", type=int, default=64)
    evo.add_argument("--height", type=int, default=1)
    evo.add_argument("--num-sources", type=int, default=1, help="Number of identical energy sources (L0.2).")
    evo.add_argument("--num-bad-sources", type=int, default=0, help="Number of harmful sources (L0.2 variant).")
    evo.add_argument(
        "--bad-source-integrity-loss",
        type=float,
        default=0.0,
        help="Integrity loss applied when arriving on a harmful source (L0.2 variant).",
    )
    evo.add_argument(
        "--deplete-sources",
        action="store_true",
        default=False,
        help="Deplete sources on arrival and respawn later (L1.0).",
    )
    evo.add_argument("--respawn-delay", type=int, default=0, help="Respawn delay in steps (only if --deplete-sources).")
    evo.add_argument("--steps", type=int, default=128)
    evo.add_argument("--energy-init", type=float, default=1.0)
    evo.add_argument("--energy-decay", type=float, default=None)
    evo.add_argument("--energy-gain", type=float, default=0.05)
    evo.add_argument("--terminate-on-reach", action="store_true", default=False)
    evo.add_argument("--obs-noise", type=float, default=0.0)
    evo.add_argument("--grad-gain-min", type=float, default=1.0)
    evo.add_argument("--grad-gain-max", type=float, default=1.0)
    evo.add_argument("--grad-gain-start-phi", type=float, default=0.0)
    evo.add_argument("--grad-gain-end-phi", type=float, default=0.0)
    evo.add_argument("--grad-bins-min", type=float, default=0.0)
    evo.add_argument("--grad-bins-max", type=float, default=0.0)
    evo.add_argument("--grad-bins-start-phi", type=float, default=0.0)
    evo.add_argument("--grad-bins-end-phi", type=float, default=0.0)

    evo.add_argument("--n-neurons", type=int, default=32)
    evo.add_argument("--k-edges", type=int, default=8)
    evo.add_argument("--topology-seed", type=int, default=None)
    evo.add_argument("--theta", type=float, default=1.0)
    evo.add_argument("--tau-m", type=float, default=10.0)

    evo.add_argument("--episodes", type=int, default=4)
    evo.add_argument("--pop-size", type=int, default=64)
    evo.add_argument("--generations", type=int, default=20)
    evo.add_argument("--sigma", type=float, default=0.1)
    evo.add_argument("--lr", type=float, default=0.05)

    evo.add_argument("--success-bonus", type=float, default=50.0)
    evo.add_argument("--fitness-alpha", type=float, default=1.0)
    evo.add_argument("--fitness-beta", type=float, default=10.0)

    evo.add_argument("--mvt", action="store_true", default=False, help="Enable minimal viability test filtering.")
    evo.add_argument("--mvt-episodes", type=int, default=2)
    evo.add_argument("--mvt-steps", type=int, default=64)
    evo.add_argument("--mvt-min-alive-steps", type=int, default=32)
    evo.add_argument("--mvt-min-action-entropy", type=float, default=0.0)
    evo.add_argument("--mvt-min-energy-gained", type=float, default=0.0)
    evo.add_argument("--mvt-fail-fitness", type=float, default=-1e9)

    base = sub.add_parser("baseline-l0", help="Evaluate simple baselines on L0 chemotaxis.")
    base.add_argument("--seed", type=int, default=0)
    base.add_argument("--policy", type=str, default="greedy", choices=["greedy", "random", "stay"])
    base.add_argument("--episodes", type=int, default=16)

    base.add_argument("--width", type=int, default=64)
    base.add_argument("--height", type=int, default=1)
    base.add_argument("--num-sources", type=int, default=1)
    base.add_argument("--num-bad-sources", type=int, default=0)
    base.add_argument("--bad-source-integrity-loss", type=float, default=0.0)
    base.add_argument("--deplete-sources", action="store_true", default=False)
    base.add_argument("--respawn-delay", type=int, default=0)
    base.add_argument("--steps", type=int, default=128)
    base.add_argument("--energy-init", type=float, default=1.0)
    base.add_argument("--energy-decay", type=float, default=None)
    base.add_argument("--energy-gain", type=float, default=0.05)
    base.add_argument("--terminate-on-reach", action="store_true", default=False)
    base.add_argument("--obs-noise", type=float, default=0.0)
    base.add_argument("--grad-gain-min", type=float, default=1.0)
    base.add_argument("--grad-gain-max", type=float, default=1.0)
    base.add_argument("--grad-gain-start-phi", type=float, default=0.0)
    base.add_argument("--grad-gain-end-phi", type=float, default=0.0)
    base.add_argument("--grad-bins-min", type=float, default=0.0)
    base.add_argument("--grad-bins-max", type=float, default=0.0)
    base.add_argument("--grad-bins-start-phi", type=float, default=0.0)
    base.add_argument("--grad-bins-end-phi", type=float, default=0.0)

    base.add_argument("--success-bonus", type=float, default=50.0)
    base.add_argument("--fitness-alpha", type=float, default=1.0)
    base.add_argument("--fitness-beta", type=float, default=10.0)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.cmd == "evo-l0":
        topology_seed = args.topology_seed if args.topology_seed is not None else args.seed + 12345
        energy_decay = args.energy_decay if args.energy_decay is not None else (args.energy_init / max(args.steps, 1))
        num_sources = max(int(args.num_sources), 1)
        num_bad_sources = max(min(int(args.num_bad_sources), num_sources), 0)

        env_spec = ChemotaxisEnvSpec(
            width=args.width,
            height=args.height,
            max_steps=args.steps,
            energy_init=args.energy_init,
            energy_decay=energy_decay,
            energy_gain=args.energy_gain,
            terminate_on_reach=args.terminate_on_reach,
            obs_noise=args.obs_noise,
            grad_gain_min=args.grad_gain_min,
            grad_gain_max=args.grad_gain_max,
            grad_gain_start_phi=args.grad_gain_start_phi,
            grad_gain_end_phi=args.grad_gain_end_phi,
            grad_bins_min=args.grad_bins_min,
            grad_bins_max=args.grad_bins_max,
            grad_bins_start_phi=args.grad_bins_start_phi,
            grad_bins_end_phi=args.grad_bins_end_phi,
            num_sources=num_sources,
            num_bad_sources=num_bad_sources,
            bad_source_integrity_loss=max(float(args.bad_source_integrity_loss), 0.0),
            source_deplete=bool(args.deplete_sources),
            source_respawn_delay=max(int(args.respawn_delay), 0),
        )
        dev_cfg = make_dev_config(
            n_neurons=args.n_neurons,
            obs_dim=4,
            num_actions=5,
            k_edges_per_neuron=args.k_edges,
            topology_seed=topology_seed,
            theta=args.theta,
            tau_m=args.tau_m,
            plast_enabled=False,
        )
        sim_cfg = SimConfig(
            fitness_alpha=args.fitness_alpha,
            fitness_beta=args.fitness_beta,
            success_bonus=args.success_bonus,
        )
        eval_cfg = EvalConfig(episodes=args.episodes)
        es_cfg = ESConfig(
            pop_size=args.pop_size,
            generations=args.generations,
            sigma=args.sigma,
            lr=args.lr,
        )
        mvt_cfg = MVTConfig(
            enabled=args.mvt,
            episodes=args.mvt_episodes,
            steps=args.mvt_steps,
            min_alive_steps=args.mvt_min_alive_steps,
            min_action_entropy=args.mvt_min_action_entropy,
            min_energy_gained=args.mvt_min_energy_gained,
            fail_fitness=args.mvt_fail_fitness,
        )

        out_dir = Path(args.out_dir) if args.out_dir is not None else _default_out_dir("evo-l0", args.seed)
        ensure_dir(out_dir)

        config = {
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
            "mvt_cfg": mvt_cfg._asdict(),
        }
        manifest = collect_manifest(seed=args.seed, config=config, cwd=Path.cwd())
        write_manifest(out_dir, manifest)

        res = run_openai_es(
            seed=args.seed,
            out_dir=out_dir,
            env_spec=env_spec,
            dev_cfg=dev_cfg,
            sim_cfg=sim_cfg,
            eval_cfg=eval_cfg,
            es_cfg=es_cfg,
            mvt_cfg=mvt_cfg,
        )
        print(f"best_fitness={res.best_fitness:.4f} out_dir={out_dir}")
        return

    if args.cmd == "baseline-l0":
        energy_decay = args.energy_decay if args.energy_decay is not None else (args.energy_init / max(args.steps, 1))
        num_sources = max(int(args.num_sources), 1)
        num_bad_sources = max(min(int(args.num_bad_sources), num_sources), 0)
        env_spec = ChemotaxisEnvSpec(
            width=args.width,
            height=args.height,
            max_steps=args.steps,
            energy_init=args.energy_init,
            energy_decay=energy_decay,
            energy_gain=args.energy_gain,
            terminate_on_reach=args.terminate_on_reach,
            obs_noise=args.obs_noise,
            grad_gain_min=args.grad_gain_min,
            grad_gain_max=args.grad_gain_max,
            grad_gain_start_phi=args.grad_gain_start_phi,
            grad_gain_end_phi=args.grad_gain_end_phi,
            grad_bins_min=args.grad_bins_min,
            grad_bins_max=args.grad_bins_max,
            grad_bins_start_phi=args.grad_bins_start_phi,
            grad_bins_end_phi=args.grad_bins_end_phi,
            num_sources=num_sources,
            num_bad_sources=num_bad_sources,
            bad_source_integrity_loss=max(float(args.bad_source_integrity_loss), 0.0),
            source_deplete=bool(args.deplete_sources),
            source_respawn_delay=max(int(args.respawn_delay), 0),
        )
        sim_cfg = SimConfig(
            fitness_alpha=args.fitness_alpha,
            fitness_beta=args.fitness_beta,
            success_bonus=args.success_bonus,
        )

        sim_fn = {
            "greedy": simulate_lifetime_baseline_greedy,
            "random": simulate_lifetime_baseline_random,
            "stay": simulate_lifetime_baseline_stay,
        }[args.policy]

        master = jax.random.PRNGKey(args.seed)
        keys = jax.random.split(master, args.episodes)
        t_idx = jnp.arange(env_spec.max_steps, dtype=jnp.int32)
        outs = jax.vmap(lambda k: sim_fn(env_spec, sim_cfg, k, t_idx))(keys)

        mean_fit = float(jax.device_get(jnp.mean(outs.fitness_scalar)))
        succ_rate = float(jax.device_get(jnp.mean(outs.success.astype(jnp.float32))))
        mean_alive = float(jax.device_get(jnp.mean(outs.t_alive.astype(jnp.float32))))
        mean_gain = float(jax.device_get(jnp.mean(outs.energy_gained_total)))
        mean_ent = float(jax.device_get(jnp.mean(outs.action_entropy)))
        mean_mode = float(jax.device_get(jnp.mean(outs.action_mode_frac)))

        print(
            "baseline_l0"
            f" policy={args.policy}"
            f" episodes={args.episodes}"
            f" mean_fitness={mean_fit:.4f}"
            f" success_rate={succ_rate:.3f}"
            f" mean_t_alive={mean_alive:.1f}"
            f" mean_energy_gained={mean_gain:.4f}"
            f" mean_action_entropy={mean_ent:.4f}"
            f" mean_action_mode_frac={mean_mode:.3f}"
        )
        return

    raise SystemExit(f"unknown cmd: {args.cmd}")
