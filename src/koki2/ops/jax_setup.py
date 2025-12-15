from __future__ import annotations

import os
import platform
from pathlib import Path


def default_jax_cache_dir(*, app_name: str = "koki2") -> Path:
    system = platform.system()
    if system == "Darwin":
        base = Path.home() / "Library" / "Caches"
    elif system == "Windows":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / app_name / "jax"


def configure_jax_compilation_cache(*, cache_dir: str | Path | None, disable: bool) -> Path | None:
    """Configure the persistent compilation cache for JAX/XLA.

    This is semantics-preserving and primarily reduces compilation overhead for repeated runs.
    Should be called before importing `jax` for maximum effect.
    """
    if disable:
        os.environ.pop("JAX_COMPILATION_CACHE_DIR", None)
        return None

    if cache_dir is None:
        existing = os.environ.get("JAX_COMPILATION_CACHE_DIR")
        resolved = Path(existing) if existing else default_jax_cache_dir()
    else:
        resolved = Path(cache_dir)

    resolved.mkdir(parents=True, exist_ok=True)
    os.environ["JAX_COMPILATION_CACHE_DIR"] = str(resolved)
    return resolved


def activate_jax_compilation_cache() -> None:
    """Best-effort activation for older/newer JAX versions.

    Safe to call after importing `jax`.
    """
    cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    if not cache_dir:
        return

    try:
        from jax.experimental import compilation_cache as cc

        cc.compilation_cache.set_cache_dir(cache_dir)
    except Exception:
        return

