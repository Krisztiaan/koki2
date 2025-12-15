from __future__ import annotations

import jax.numpy as jnp

from koki2.types import Array


def greedy_gradient_action(obs: Array, *, deadband: float = 1e-3) -> Array:
    dx, dy = obs[0], obs[1]
    adx, ady = jnp.abs(dx), jnp.abs(dy)
    move_x = adx >= ady
    stay = jnp.logical_and(adx < deadband, ady < deadband)

    # Action mapping (see env):
    # 0 stay, 1 up, 2 down, 3 left, 4 right
    act_x = jnp.where(dx >= 0, jnp.array(4, dtype=jnp.int32), jnp.array(3, dtype=jnp.int32))
    act_y = jnp.where(dy >= 0, jnp.array(2, dtype=jnp.int32), jnp.array(1, dtype=jnp.int32))
    act = jnp.where(move_x, act_x, act_y)
    return jnp.where(stay, jnp.array(0, dtype=jnp.int32), act)

