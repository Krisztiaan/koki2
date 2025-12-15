# Evolution engine specification (search, budgeting, and selection)

## 1. Requirements

The evolution engine must:

- support population-based search in genome space
- integrate nursing and pruning without contaminating the core fitness definition
- run efficiently in JAX:
  - vectorized population evaluation (`vmap`)
  - JIT compiled rollout loops (`scan`)
- support reproducible checkpoints and resumption

---

## 2. Search algorithm choices

### Option A — Evolution strategies (ES)

Examples:
- (separable) CMA-ES
- OpenAI-ES style gradient estimates
- NES variants

Pros:
- works with continuous genome parameters
- compatible with ask-eval-tell loops
- stable scaling with JAX vectorization

Implementation: use `evosax` (JAX-based ES toolkit) [@lange2022evosax] as the default backend.

### Option B — Genetic algorithms / mutation-selection

Pros:
- natural for structured mutation and NEAT-like topological changes
- easier to incorporate discrete genome structure

Cons:
- may be less GPU-friendly unless carefully vectorized

### Option C — Hybrid

- ES for continuous parameters of fixed-topology genomes
- GA operators for occasional structural mutations

**Recommendation:** start with ES for initial milestones; introduce hybrid GA when structural evolution becomes necessary.

---

## 3. Evaluation pipeline

Evaluation is a pipeline with explicit rungs:

1. MVT (minimal viability)
2. short-horizon early-life evaluation
3. full-horizon multi-level evaluation

All of these call into the Simulation Orchestrator.

### 3.1 FitnessSummary schema

Fitness should include:

- `fitness_scalar`
- subcomponents:
  - survival time
  - resources acquired
  - learning gain (optional)
  - constraint penalties/bonuses (strain-specific)
- evaluation metadata:
  - rung level used
  - environment seed list
  - nursing schedule used

This enables later auditing of pruning bias.

---

## 4. Nursing and pruning integration

### 4.1 Nursing schedule in evaluation

Each simulated life uses:

- a fixed lifespan length \(T_{\text{life}}\),
- developmental phase \(\phi(t)\),
- and environment/agent schedules parameterized by \(\phi\).

Evaluation includes both early and late phases; fitness weights late phases more.

### 4.2 Pruning / multi-fidelity

Pruning and multi-fidelity are applied only to allocate budget:

- MVT eliminates clearly non-viable genomes.
- rung-1 eliminates obviously poor but viable genomes.
- rung-2 evaluates promoted genomes fully.

Crucially:
- all pruning parameters are logged and ablated experimentally.

---

## 5. Novelty and diversity support

The evolution engine optionally maintains:

- behavior descriptor archive
- novelty scores per genome

Novelty can be used for:

- promotion quotas in multi-fidelity,
- a secondary objective in selection.

Start with novelty quotas only (minimal intrusion), then expand to QD if required.

---

## 6. Pseudocode sketch

```text
for generation in 1..G:
  genomes = ask(evo_state)

  # Rung 0: viability
  viable_mask = vmap(MVT)(genomes)
  viable_genomes = genomes[viable_mask]

  # Rung 1: short eval
  proxy = vmap(short_eval)(viable_genomes)

  # compute novelty (optional)
  novelty = compute_novelty(viable_genomes, proxy.trajectories)

  # promote
  promoted = select_top_by_proxy_plus_novelty(viable_genomes, proxy, novelty)

  # Rung 2: full eval
  fitness = vmap(full_eval)(promoted)

  evo_state = tell(evo_state, genomes, assemble_fitness(genomes, viable_mask, promoted, fitness))
```

---

## 7. Checkpointing and resumption

Checkpoint contents:

- evolution engine state
- population genomes
- novelty archive state (if used)
- RNG keys for each stream
- experiment config hashes

To guarantee determinism, we record:

- list of environment seeds per evaluation batch
- derived PRNG keys used for development and rollout

---

## 8. Deliverables

- ES backend integration (evosax) and configuration.
- Multi-fidelity evaluation policy implementation.
- Novelty quota mechanism (optional in early stages).
- Checkpoint/resume support.
- Benchmark results:
  - throughput (genomes/sec) on target hardware,
  - scaling with population size and episode length.
