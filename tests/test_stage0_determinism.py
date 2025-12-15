import jax
import jax.numpy as jnp

from koki2.agent.heuristic import greedy_gradient_action
from koki2.envs.chemotaxis import env_init, env_step
from koki2.genome.direct import genome_init, make_dev_config
from koki2.sim.orchestrator import simulate_lifetime
from koki2.types import ChemotaxisEnvSpec, DevelopmentState, SimConfig


def test_chemotaxis_heuristic_reaches_source() -> None:
    spec = ChemotaxisEnvSpec(
        width=16,
        height=1,
        max_steps=32,
        energy_init=1.0,
        energy_decay=1.0 / 32,
        energy_gain=0.0,
        terminate_on_reach=True,
        obs_noise=0.0,
    )
    key = jax.random.PRNGKey(0)
    state, obs, internal = env_init(spec, key)

    def step(carry, t):
        state, obs, internal, alive, reached = carry
        dev = DevelopmentState(age_step=t, phi=jnp.array(0.0, dtype=jnp.float32))
        rng = jax.random.fold_in(key, t)

        def do(_):
            action = greedy_gradient_action(obs)
            state2, obs2, internal2, log, done = env_step(spec, state, action, dev, rng)
            reached2 = jnp.logical_or(reached, log.reached_source)
            alive2 = jnp.logical_and(alive, jnp.logical_not(done))
            return (state2, obs2, internal2, alive2, reached2)

        def skip(_):
            return (state, obs, internal, alive, reached)

        return jax.lax.cond(alive, do, skip, operand=None), None

    init = (state, obs, internal, jnp.array(True), jnp.array(False))
    final, _ = jax.lax.scan(step, init, xs=jnp.arange(spec.max_steps, dtype=jnp.int32))
    reached = final[-1]
    assert bool(jax.device_get(reached))


def test_simulate_lifetime_is_deterministic() -> None:
    env_spec = ChemotaxisEnvSpec(
        width=16,
        height=1,
        max_steps=64,
        energy_init=1.0,
        energy_decay=1.0 / 64,
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

    genome = genome_init(jax.random.PRNGKey(123), dev_cfg, scale=0.1)
    rng = jax.random.PRNGKey(999)

    t_idx = jnp.arange(env_spec.max_steps, dtype=jnp.int32)
    out1 = simulate_lifetime(genome, env_spec, dev_cfg, sim_cfg, rng, t_idx)
    out2 = simulate_lifetime(genome, env_spec, dev_cfg, sim_cfg, rng, t_idx)

    assert float(jax.device_get(out1.fitness_scalar)) == float(jax.device_get(out2.fitness_scalar))
    assert int(jax.device_get(out1.t_alive)) == int(jax.device_get(out2.t_alive))
    assert float(jax.device_get(out1.energy_gained_total)) == float(jax.device_get(out2.energy_gained_total))
    assert float(jax.device_get(out1.bad_arrivals_total)) == float(jax.device_get(out2.bad_arrivals_total))
    assert float(jax.device_get(out1.integrity_lost_total)) == float(jax.device_get(out2.integrity_lost_total))
    assert float(jax.device_get(out1.integrity_min)) == float(jax.device_get(out2.integrity_min))
    assert bool(jax.device_get(out1.success)) == bool(jax.device_get(out2.success))
    assert float(jax.device_get(out1.action_entropy)) == float(jax.device_get(out2.action_entropy))
    assert float(jax.device_get(out1.action_mode_frac)) == float(jax.device_get(out2.action_mode_frac))


def test_simulate_lifetime_jit_matches_eager() -> None:
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
    genome = genome_init(jax.random.PRNGKey(0), dev_cfg, scale=0.1)
    rng = jax.random.PRNGKey(1)

    t_idx = jnp.arange(env_spec.max_steps, dtype=jnp.int32)
    eager = simulate_lifetime(genome, env_spec, dev_cfg, sim_cfg, rng, t_idx)
    jitted = jax.jit(simulate_lifetime)(genome, env_spec, dev_cfg, sim_cfg, rng, t_idx)
    assert float(jax.device_get(eager.fitness_scalar)) == float(jax.device_get(jitted.fitness_scalar))
    assert float(jax.device_get(eager.bad_arrivals_total)) == float(jax.device_get(jitted.bad_arrivals_total))
    assert float(jax.device_get(eager.integrity_lost_total)) == float(jax.device_get(jitted.integrity_lost_total))
    assert float(jax.device_get(eager.integrity_min)) == float(jax.device_get(jitted.integrity_min))
