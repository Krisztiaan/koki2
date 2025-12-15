# 05 — Implementation architecture: abstraction layers and interface contracts

**Purpose:** Define a modular architecture with clear interfaces so environments, agent substrate, genome/development, evolution/budgeting, and observability can be developed in parallel and integrated incrementally.

**Non-negotiable constraint:** end-to-end evaluation must be executable as a **pure JAX program** (or a thin wrapper around it), with deterministic randomness and static shapes where possible.

---

## 1. Layered architecture overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  L7  Experiment Ops (configs, manifests, checkpoints, reproducibility)│
└─────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────┐
│  L6  Observability (metrics schema, trajectory logs, analysis hooks)  │
└─────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────┐
│  L5  Evolution + Budgeting (ask-eval-tell, multi-fidelity, pruning)   │
└─────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────┐
│  L4  Evaluation Orchestrator (vectorized rollouts; seeding; batching) │
└─────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────┐
│  L3  Genome→Phenotype Compiler (direct/indirect encodings)            │
└─────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────┐
│  L2  Agent Substrate (SNN/RNN core; plasticity hooks; constraints)    │
└─────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────┐
│  L1  Environment Ladder (JAX envs; nursing schedule; task variants)   │
└─────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────┐
│  L0  Runtime Backends (Spyx/BrainPy; accelerator; sharding)           │
└─────────────────────────────────────────────────────────────────────┘
```

### Parallelism principle

Each layer exposes a **narrow, testable contract**.  
Teams can work on:
- environments (L1),
- agent substrate/backends (L2/L0),
- genome compiler (L3),
- evolution/budgeting (L5),
- and ops/observability (L6/L7),

without blocking on the others.

---

## 2. Data model conventions (JAX-friendly)

### 2.1 Core types are PyTrees with static structure

- `EnvState`, `AgentState`, `Genome`, `Phenotype`, `Metrics` must be PyTrees.
- Avoid Python objects with dynamic fields.
- Prefer fixed-size arrays; if variable-size graphs are needed, use padded arrays + masks.

### 2.2 Randomness conventions

- Every function that uses randomness accepts a `jax.random.PRNGKey`.
- Derive subkeys via `split` and/or `fold_in` with deterministic counters:
  - `key_env = fold_in(key, env_id)`
  - `key_agent = fold_in(key, agent_id)`
  - `key_step = fold_in(key, t)`

This makes evaluation replayable and supports multi-device determinism.

### 2.3 Deterministic evaluation contract

A candidate evaluation must be deterministic given:
- genotype
- environment params and seed
- evaluation config

Stochasticity is allowed only if it is fully keyed and recorded.

---

## 3. Environment interface (L1)

### 3.1 Minimal contract

```text
EnvSpec:
  reset(key, params) -> (state, obs)
  step(state, action, params, key) -> (state', obs', reward, done, info)

Requirements:
  - pure functions (no side effects)
  - compatible with lax.scan for T steps
  - shapes static across episode (obs/action dims fixed per env level)
```

### 3.2 Environment ladder packaging

Provide a `TaskSuite` wrapper:

```text
TaskSuite:
  sample_task(key, level, variant_id) -> EnvParams
  reset(key, EnvParams) -> (EnvState, Obs)
  step(EnvState, Action, EnvParams, key) -> ...
```

So evolution can evaluate on multiple variants without changing code.

### 3.3 Nursing integration

Environment params include a `NursingParams` block and (optionally) expose an `age` variable in `EnvState` (not in observations unless explicitly desired).

---

## 4. Agent substrate interface (L2)

### 4.1 Minimal contract (supports both SNN and RNN)

```text
AgentSpec:
  init(key, phenotype, agent_params) -> AgentState
  act(agent_state, obs, phenotype, agent_params, key) -> (agent_state', action, aux)

Where:
  phenotype = parameters + structural masks (from genome compiler)
  aux = optional diagnostics (spike rate, energy use, etc.)
```

### 4.2 Optional lifetime learning hook

We separate “acting” from “learning” to keep evaluation pure:

```text
learn_update(agent_state, transition, phenotype, agent_params, key)
  -> (agent_state', phenotype' or learning_state')
```

This supports:
- fixed-weights agents (no-op learn_update),
- plasticity-based updates,
- inner-loop gradient updates (if used later).

---

## 5. Genome and compiler interface (L3)

### 5.1 Genome object

`Genome` is an evolvable representation. It must support:
- mutation (stochastic but keyed)
- recombination (optional)
- compilation to a phenotype for execution

### 5.2 Compiler contract

```text
compile(genome, compiler_params, key) -> (phenotype, compile_info)

compile_info includes:
  - validity flags
  - compression ratio (if applicable)
  - structural statistics (sparsity, E/I balance, etc.)
```

This enables P0 viability checks before running expensive rollouts.

---

## 6. Evolution + budgeting interface (L5)

### 6.1 Ask-eval-tell loop

We standardize on an ask/eval/tell interface, compatible with evosax [@lange2022evosax]:

```text
ask(evo_state, key) -> (candidate_genomes, evo_state')
evaluate(candidate_genomes, suite, evaluator_config, key)
  -> (fitnesses, metrics, eval_artifacts)
tell(evo_state, candidate_genomes, fitnesses, key) -> evo_state'
```

### 6.2 Budgeting and pruning are first-class

Evaluation returns both:
- fitness (scalar used by ES)
- structured metrics for pruning and analysis

Budgeting is handled by a `BudgetAllocator`:

```text
BudgetAllocator:
  allocate(population, previous_metrics) -> evaluation_plan
```

Where `evaluation_plan` includes fidelity levels and early termination thresholds.

---

## 7. Observability interface (L6)

### 7.1 Metric schema

Metrics are produced at three scopes:
- per-step (optional, expensive)
- per-episode (preferred)
- per-generation (preferred)

Define a fixed schema per environment level.

### 7.2 Trajectory logging

Trajectory logging must not destroy throughput.
Rule: log full trajectories only for a small sampled subset of candidates per generation.

---

## 8. Experiment ops interface (L7)

### 8.1 Run manifest

Every run writes:
- full config (including git commit hash if available)
- all random seeds and key derivation method
- environment suite version identifiers
- strain definition and annealing schedule
- evaluation/budgeting configuration

### 8.2 Replay contract

A replay tool must be able to:
- reconstruct environment + agent state
- rerun the exact rollout from stored seeds + genotype

This is essential for thesis-grade claims.

---

## 9. What each team can implement independently

- **Environment team:** implements `TaskSuite` and nursing knobs; unit tests for determinism and acceptance criteria.
- **Agent team:** implements `AgentSpec` with Spyx backend; optional BrainPy backend; exposes diagnostics.
- **Genome team:** implements `Genome` representation and `compile`; provides compile_info stats.
- **Evolution team:** implements ask/eval/tell + budget allocator + pruning; uses environment/agent via interfaces.
- **Ops/analysis team:** implements manifests, storage, dashboards, replay.

---

## 10. References

See `references.bib`.
