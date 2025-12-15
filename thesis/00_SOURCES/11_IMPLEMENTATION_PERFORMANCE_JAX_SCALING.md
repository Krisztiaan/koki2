# 11 — Performance plan: maximizing throughput with JAX (population search at scale)

**Purpose:** Provide an efficiency-first execution strategy for running many rollouts and simulations under JAX/XLA.

---

## 1. Throughput target and constraints

Neuroevolution is dominated by:
- number of environment steps simulated,
- number of candidates evaluated,
- model forward cost per step,
- and compilation/dispatch overhead.

The performance plan is therefore structured around:

1. keep the inner loop (env step + agent act) entirely within XLA,
2. minimize host-device synchronization,
3. avoid dynamic shapes,
4. maximize vectorization across population and across environments.

---

## 2. Core execution pattern: `vmap` × `scan`

### 2.1 Canonical structure

- `scan` over time steps inside an episode
- `vmap` over population members (and possibly over parallel env instances)
- `jit` around the whole evaluation function

Conceptually:

- Outer: evaluate many candidates in parallel
- Inner: simulate each candidate in a vectorized environment batch

### 2.2 Static shapes as a design constraint

To get good XLA compilation and reuse:

- observation dimension is fixed per env level
- action dimension fixed
- network parameter shapes fixed per strain/genome family
- if variable-sized graphs are needed: pad to max size and carry a mask

---

## 3. Managing compilation cost

Compilation cost can dominate if:
- many different shapes are compiled,
- jitted functions are re-traced by Python changes.

Mitigations:

- isolate “shape-changing” configuration (env level, network size) so compilation occurs once per experiment condition
- run warm-up compilation explicitly at start
- keep evaluation functions pure and stable

---

## 4. Memory scaling considerations

### 4.1 Parameter memory

If parameters are float32:
- 1M params ≈ 4 MB
- 100M params ≈ 400 MB

Population evaluation can replicate parameters; avoid broadcasting large parameter arrays unnecessarily.

### 4.2 Activations and state

For recurrent/spiking networks, per-step state can dominate:
- membrane potentials
- synaptic traces
- adaptation variables

Use `scan` so state is carried forward rather than stored for all timesteps (unless required for learning).

---

## 5. Sparse vs dense trade-offs

Dense matmuls are extremely fast on GPUs/TPUs but scale as $O(n^2)$. Sparse operations reduce compute but can be slower if sparsity patterns are not hardware-friendly.

Guidance:

- early thesis stages: small-to-medium dense networks (simplicity and speed)
- later scaling: consider structured sparsity (block-sparse) before fully irregular sparsity

---

## 6. Population parallelism options

### 6.1 Single device: `vmap`

Use `vmap` over population and env instances.

### 6.2 Multi-device: sharding / pmap

When multiple accelerators are available:
- shard population across devices
- keep per-device batch sizes large enough to amortize overhead

Determinism considerations:
- derive per-device keys via fold_in(device_id)
- ensure reductions are deterministic if needed (some collectives can be nondeterministic)

---

## 7. Logging without killing performance

Logging is the classic throughput killer.

Rules:

- never log per-step scalars from the hot loop unless absolutely necessary
- compute episode-level summaries inside JIT and return them as arrays
- log trajectories only for a small sampled subset, optionally via a separate evaluation pass

---

## 8. Profiling and performance regression tests

### 8.1 Profiling

Profile at three points:
- compilation time
- steady-state steps/sec
- memory footprint

### 8.2 Regression tests

Maintain a small “performance benchmark suite”:
- fixed env level (L1)
- fixed network size
- fixed population size
- fixed number of steps

Track steps/sec across commits to prevent accidental slowdowns.

---

## 9. Backend strategy (Spyx vs BrainPy)

Default plan:
- use Spyx for throughput-critical experiments
- use BrainPy for selected mechanistic experiments requiring richer neuron models

Maintain backend parity tests:
- same genotype evaluated under both backends on a toy environment should produce qualitatively similar behavior (within tolerances).

---

## 10. Practical scaling roadmap

1. Get L0–L2 running with small networks and dense ops.
2. Increase population size and env parallelism before increasing network size.
3. Only then increase network size; introduce sparsity if needed.
4. Introduce multi-device sharding once single-device pipeline is stable and reproducible.

