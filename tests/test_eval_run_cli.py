import json

import numpy as np

from koki2.cli import main


def test_eval_run_smoke(tmp_path, capsys) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    n = 4
    k_edges = 1
    e = n * k_edges

    manifest = {
        "config": {
            "env_spec": {
                "width": 1,
                "height": 1,
                "max_steps": 4,
                "energy_init": 1.0,
                "energy_decay": 0.0,
                "energy_gain": 0.0,
                "terminate_on_reach": False,
            },
            "dev_cfg": {
                "n_neurons": n,
                "obs_dim": 4,
                "num_actions": 5,
                "edge_index_shape": [e, 2],
                "topology_seed": 123,
                "theta": 1.0,
                "tau_m": 10.0,
                "plast_enabled": False,
                "plast_eta": 0.0,
                "plast_lambda": 0.9,
            },
            "sim_cfg": {
                "fitness_alpha": 1.0,
                "fitness_beta": 10.0,
                "success_bonus": 50.0,
                "setpoint_energy": 1.0,
                "setpoint_integrity": 1.0,
                "drive_w_energy": 1.0,
                "drive_w_integrity": 1.0,
            },
        }
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    np.savez(
        run_dir / "best_genome.npz",
        obs_w=np.zeros((n, 4), dtype=np.float32),
        rec_w=np.zeros((e,), dtype=np.float32),
        motor_w=np.zeros((n, 5), dtype=np.float32),
        motor_b=np.zeros((5,), dtype=np.float32),
        mod_w=np.zeros((n,), dtype=np.float32),
    )

    main(["eval-run", "--run-dir", str(run_dir), "--episodes", "2", "--seed", "0", "--baseline-policy", "greedy"])

    out = capsys.readouterr().out
    assert "eval_run" in out
    assert "best_genome" in out
    assert "mean_fitness=54.0000" in out
    assert "baseline_l0 policy=greedy" in out

