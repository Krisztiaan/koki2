# JAX scaling guide (efficiency-first implementation)

## 1. Objective

This thesis requires running many simulations:

- populations (64–1024 genomes)
- long lifetimes (10^3–10^5 steps)
- multiple seeds and environment variants
- multiple strains and ablations

Therefore, efficiency is a core design requirement, not an afterthought.

---

## 2. Core performance principles in JAX

### 2.1 Prefer `lax.scan` over Python loops

- express time evolution with `lax.scan`
- keep state sizes static

### 2.2 Vectorize across population with `vmap`

- evaluate many genomes in parallel
- avoid host-side loops over genomes

### 2.3 Avoid host↔device transfers

- keep rollout, fitness computation, and summary aggregation on device
- log only small summaries per episode to host

### 2.4 Static shapes

- avoid variable-length edge lists inside jitted code
- represent sparsity with fixed `E_max` and masks

---

## 3. Sparse connectivity implementation notes

### 3.1 Edge list + segment reductions

Represent edges as:

- `src`, `dst` arrays of length `E_max`
- `w` and mask arrays

Compute input currents with segment sums:

\[
I_{\text{dst}} = \sum_{e: dst(e)=i} w_e \, s_{src(e)}
\]

Implementation uses:
- `jax.ops.segment_sum` or equivalent.

### 3.2 Avoid dynamic indexing

Dynamic indexing patterns can cause performance cliffs. Prefer:

- precomputed index arrays
- masked operations

---

## 4. Multi-fidelity evaluation performance

Multi-fidelity adds control flow (promotion decisions). For JAX efficiency:

- compute rung-1 proxy fitness for all viable genomes as a vectorized batch
- perform promotion selection on host **only on small arrays** (scores)
- then run rung-2 full evaluation on the promoted subset

Alternatively, keep selection on-device if population size is large and selection is frequent.

---

## 5. Compilation strategy

### 5.1 Minimize recompilations

Use one of:

- pad/truncate to fixed shapes
- compile separate variants per environment level (acceptable)
- avoid conditional branches that change shapes

### 5.2 Profiling

Use:

- `jax.profiler` traces
- XLA HLO inspection for hotspots
- measure:
  - compile time
  - step throughput
  - memory allocation patterns

---

## 6. Device parallelism and sharding

If running on multi-GPU/TPU:

- shard population across devices (data parallel)
- aggregate fitness across devices at generation boundaries

Ensure:
- PRNG streams are device-consistent
- determinism is preserved across sharded evaluation (log device topology)

---

## 7. Memory budgeting

Key contributors:

- neural state: \(O(N)\)
- synapse state (eligibility traces): \(O(E)\)
- population factor: multiplied by population size \(P\)

Therefore, eligibility traces can dominate memory.

Mitigations:

- compress traces (float16) if stable
- store only a small number of traces
- consider sparse update schedules (update only active edges)

---

## 8. Throughput milestones (engineering gates)

Define explicit engineering milestones:

1. L0 chemotaxis rollout at \(P=256\), \(T=1000\) steps within target walltime.
2. L2 multi-resource rollout at \(P=256\), \(T=5000\).
3. L4 long-horizon rollout at \(P=128\), \(T=20000\).

These milestones should be achieved before running full ablation matrices.

---

## 9. Deliverables

- a benchmark harness that:
  - runs fixed rollouts
  - reports steps/sec and memory
  - saves profiler traces
- documented performance “dos and don’ts” for contributors
