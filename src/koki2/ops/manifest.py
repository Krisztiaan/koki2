from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from koki2.ops.run_io import utc_now_iso, write_json


def _run(cmd: list[str], *, cwd: Path) -> str | None:
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), check=True, capture_output=True, text=True)
    except Exception:
        return None
    out = (proc.stdout or "").strip()
    return out or None


def try_get_git_commit(cwd: Path) -> str | None:
    return _run(["git", "rev-parse", "HEAD"], cwd=cwd)


def collect_manifest(*, seed: int, config: dict[str, Any], cwd: Path) -> dict[str, Any]:
    import jax

    return {
        "created_at_utc": utc_now_iso(),
        "seed": seed,
        "config": config,
        "git_commit": try_get_git_commit(cwd),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_implementation": platform.python_implementation(),
        },
        "jax": {
            "version": jax.__version__,
            "devices": [str(d) for d in jax.devices()],
            "backend": jax.default_backend(),
        },
    }


def write_manifest(out_dir: str | Path, manifest: dict[str, Any]) -> None:
    write_json(Path(out_dir) / "manifest.json", manifest)
