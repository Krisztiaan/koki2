# Thesis outline master

This document is the **canonical thesis plan** (structure + intended claims + experiments), merged from:

- *First-Principles Framework* source,
- *Meta-Learning Framework* source,
- *Bio-Strain Comparative Framework* source,
- and the reworked implementation plan documents in this collection.

It is written to keep the thesis **incremental**, **verifiable**, and **scope-controlled**.

---

## Part 0 — Framing and methodology

### 0.1 Research goal

Establish a computational framework in which biologically constrained learning agents evolve from survival pressures, with cognition treated as emergent.

### 0.2 Inside-out stance and operationalization

- Explain inside-out theory and why it implies:
  - action-first environments,
  - analysis of spontaneous dynamics,
  - caution in “encoding” interpretations [@buzsaki2019brain].

### 0.3 What counts as an “emergent mechanism”

Define operational criteria for:
- emergent neuromodulation (activity correlated with predicted drive change),
- memory-like dynamics (state persistence, attractor behavior),
- mode switching (distinct policy regimes with hysteresis),
- learning-to-learn (within-life improvement across episodes).

---

## Part I — Minimal substrate and genotype-to-phenotype encoding

### Chapter 1: Minimal substrate specification

- Neuron/synapse state definitions
- Local plasticity rule family (eligibility traces, modulatory gating)
- Heterogeneity: time constants, thresholds, adaptation
- What is deliberately *not* pre-specified (Strain A baseline)

Deliverable: `04_THEORY_MINIMAL_SUBSTRATE_AND_PLASTICITY.md`
### Chapter 1B: Meta-learning and memory emergence

- operational definitions of memory types
- how memory pressures appear across the environment ladder
- ablation and measurement plan

Deliverable: `04B_THEORY_META_LEARNING_AND_MEMORY_EMERGENCE.md`


### Chapter 2: Genomic bottleneck and development

- Genome structure (CPPN / identity + rule networks)
- Development (“compiler”) from genome to phenotype weights + plasticity parameters
- Compression/regularization argument; scaling intuition [@shuvaev2024genomic]
- Alternatives and ablations (direct encoding vs bottleneck; CPPN vs NCA growth)

Deliverable: `05_THEORY_GENOMIC_BOTTLENECK_AND_DEVELOPMENT.md`

---

## Part II — Environments and survival pressures

### Chapter 3: Homeostasis, drives, and reward

- Internal state variables and viability bounds
- Drive function and reward as drive reduction
- Action-conditioned sensing (active perception)
- Definition of “death” and episode termination

Deliverable: `06_THEORY_ENVIRONMENTS_HOMEOSTASIS_AND_DRIVES.md`

### Chapter 4: Environment ladder

- Granular environment progression (L0–L6), each introducing one new challenge:
  - chemotaxis → depletion → partial observability → multi-resource → threats/conditioning → long-horizon changes
- For each step: acceptance tests, expected emergent pressures, ablations

Deliverable: `12_IMPLEMENTATION_ENVIRONMENT_LADDER_SPEC.md`

---

## Part III — Search, evaluators, and controlled biological constraints

### Chapter 5: Evolutionary search and biasing evaluators

- Base fitness: survival + resources acquired
- Why base fitness alone may be insufficient (deception, sparse reward)
- Biasing evaluators that guide *without specifying mechanisms*:
  - novelty / diversity,
  - learning capability measures,
  - energy efficiency,
  - modularity/regularity as soft signals
- Open-ended options (POET, QD, MCC) and how to incorporate them safely

Deliverable: `07_THEORY_BIASING_EVALUATORS_AND_OPEN_ENDEDNESS.md`

### Chapter 6: Bio-strain comparative framework

- Strain A (unconstrained baseline)
- Strain B (hard constraints)
- Strain C (soft constraints)
- Strain D (adaptive reinforcement)
- Ablation matrix and cross-strain metrics

Deliverable: `10_FRAMEWORK_BIO_STRAINS_AND_ABLATIONS.md`

---

## Part IV — Addenda for realism and feasibility

### Chapter X1: Nursing (developmental niche)

- Developmental phase variable
- Energy buffer + nursery richness + hazard gating + motor maturation
- Plasticity schedules (critical-period analogue)
- Experiments to identify “Goldilocks” scaffolding regime

Deliverable: `08_ADDENDUM_DEVELOPMENTAL_NICHE_NURSING.md`

### Chapter X2: Pruning (viability filters and multi-fidelity evaluation)

- Minimal viability tests (MVT)
- Multi-fidelity budgets and runged evaluation
- Novelty/coverage safeguards
- Measurement of pruning bias and false negatives

Deliverable: `09_ADDENDUM_VIABILITY_PRUNING_AND_MULTI_FIDELITY.md`

---

## Part V — Implementation plan and verification strategy

### Chapter 7: Architecture layers and interfaces

- Layering that enables parallel work:
  - Environment Engine, Agent Core, Genome Compiler, Simulation Orchestrator, Evolution Engine, Observability, Experiment Ops
- Interface contracts and schemas

Deliverable: `11_IMPLEMENTATION_ARCHITECTURE_LAYERS_AND_INTERFACES.md`

### Chapter 8: Agent Core implementation plan (JAX-first)

- Recommended frameworks and tradeoffs (SNNAX vs Spyx vs BrainPy)
- State representation and update kernels
- Plasticity engine integration
- Strain hooks

Deliverable: `13_IMPLEMENTATION_AGENT_CORE_SPEC.md`

### Chapter 9: Genome Compiler implementation plan

- Deterministic development pipeline
- Sparse connectivity representations
- Mutation operators and parameterizations

Deliverable: `14_IMPLEMENTATION_GENOME_COMPILER_SPEC.md`

### Chapter 10: Evolution engine and budgeting

- evosax strategies and ask-eval-tell loop
- Multi-fidelity evaluation policy integrated with nursing/pruning
- Deterministic PRNG streams, checkpointing

Deliverable: `15_IMPLEMENTATION_EVOLUTION_ENGINE_SPEC.md`

### Chapter 11: Observability, reproducibility, and analysis

- Logging schema and event sampling
- Reproducibility: manifests, seeds, deterministic replay
- Emergence detection tests and dashboards

Deliverable: `16_IMPLEMENTATION_OBSERVABILITY_REPRODUCIBILITY.md`

### Chapter 12: Performance engineering

- JAX scaling patterns and constraints
- Static shapes, scan/vmap, sparse ops, device parallelism
- Profiling plan and scaling milestones

Deliverable: `17_IMPLEMENTATION_JAX_SCALING_GUIDE.md`

---

## Part VI — Experimental program and milestones

### Chapter 13: Experiments and milestones

- Stage-by-stage plan:
  - Infrastructure validation → L0/L1 baseline → bottleneck → plasticity → nursing/pruning → bio-strains → long-horizon
- “Go/no-go” acceptance tests per stage
- Compute budgeting and replication policy

Deliverable: `18_EXPERIMENTS_AND_MILESTONES.md`

---

## Part VII — Future work

### Chapter 14: Future extensions and validation pathways

- Deferred multi-agent social extension
- Cross-simulator validation and hardware export (NIR)
- More biological neuron models (dendrites) as post-thesis explorations

Deliverable: `19_FUTURE_WORK_AND_VALIDATION.md`
