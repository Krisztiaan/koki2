from __future__ import annotations

import jax.numpy as jnp

from koki2.types import Array


def ramp(phi: Array, start_phi: float, end_phi: float) -> Array:
    phi = jnp.asarray(phi, dtype=jnp.float32)
    start = jnp.asarray(start_phi, dtype=jnp.float32)
    end = jnp.asarray(end_phi, dtype=jnp.float32)

    degenerate = end <= start
    denom = jnp.maximum(end - start, jnp.array(1e-6, dtype=jnp.float32))
    x = (phi - start) / denom
    x = jnp.clip(x, 0.0, 1.0)
    smooth = x * x * (3.0 - 2.0 * x)  # smoothstep
    return jnp.where(degenerate, jnp.array(1.0, dtype=jnp.float32), smooth)


def schedule(phi: Array, min_value: float, max_value: float, start_phi: float, end_phi: float) -> Array:
    t = ramp(phi, start_phi, end_phi)
    a = jnp.asarray(min_value, dtype=jnp.float32)
    b = jnp.asarray(max_value, dtype=jnp.float32)
    return a + t * (b - a)


def quantize(x: Array, bins: Array) -> Array:
    bins = jnp.asarray(bins, dtype=jnp.float32)
    use = bins > 0.0
    safe_bins = jnp.where(use, bins, 1.0)
    q = jnp.round(x * safe_bins) / safe_bins
    return jnp.where(use, q, x)

