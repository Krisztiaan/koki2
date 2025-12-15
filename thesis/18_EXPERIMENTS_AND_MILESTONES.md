# Experiments and milestones (incremental, verifiable progression)

## 1. Philosophy

The thesis should progress via **incremental gates**:

- each gate adds one major capability,
- each gate has explicit acceptance tests,
- later stages build on validated earlier stages.

This is necessary to avoid building a large, untestable system all at once.

---

## 2. Stage plan

### Stage 0 — Infrastructure and determinism

Deliverables:
- end-to-end rollout stub (dummy agent + L0 env)
- deterministic replay working
- logging + manifest system working

Acceptance tests:
- identical fitness under replay across multiple runs
- stable JIT compilation (no shape errors)

### Stage 1 — L0 baseline competence

Deliverables:
- L0 environment suite (chemotaxis variants)
- minimal agent substrate without plasticity (fixed weights)
- evolution loop (ES) that improves performance

Acceptance tests:
- evolution improves over random baselines across seeds
- runtime throughput meets minimal benchmark

### Stage 2 — Plasticity sanity and benefits

Deliverables:
- eligibility trace + modulator implementation
- plasticity-enabled agents
- compare plastic vs no-plastic in L1 tasks (depletion/noise)

Acceptance tests:
- statistically significant improvement of plastic agents in noisy/partial observability
- plasticity does not destabilize dynamics

### Stage 3 — Genomic bottleneck scaling

Deliverables:
- CPPN-rule genome compiler
- bottleneck encoding replaces direct weight encoding
- scaling experiments with increasing N/E

Acceptance tests:
- indirect encoding maintains or improves performance at larger N relative to direct encoding baseline
- mutation robustness improves (smaller catastrophic failure rate)

### Stage 4 — Nursing integration

Deliverables:
- DevelopmentState propagation
- environment schedules (resources/hazards) and motor gating
- plasticity schedule hooks

Acceptance tests:
- higher MVT pass rate under nursing
- evidence of “Goldilocks” nursing regime (not too weak/strong)

### Stage 5 — Pruning and multi-fidelity

Deliverables:
- MVT viability test
- runged evaluation pipeline and logging
- novelty promotion safeguard

Acceptance tests:
- compute savings measurable
- final performance not degraded under fixed compute budget
- diversity preserved (no collapse)

### Stage 6 — Bio-strain A/B/C/D comparisons

Deliverables:
- constraint hooks implemented
- ablation matrix defined and executed for L2/L3 levels

Acceptance tests:
- fair comparison reports across strains
- at least one clear empirical insight about constraint impact

### Stage 7 — Long-horizon emergence probes

Deliverables:
- L4/L5 environments
- emergence detection analysis suite

Acceptance tests:
- at least one robust emergent phenomenon observed (e.g., mode switching, memory-like dynamics)
- negative controls rule out simple artifacts (e.g., clamping modulators)

---

## 3. Replication and statistical policy

- Every key claim must be replicated across multiple seeds.
- Report means and confidence intervals.
- Use fixed evaluation protocols across strains to avoid “moving the goalposts”.

---

## 4. Risk register and mitigations

### Risk: No emergence beyond reactive behavior

Mitigations:
- adjust environment ladder to better isolate pressures that require memory
- increase heterogeneity ranges
- introduce Tier-2 learning evaluators cautiously

### Risk: Plasticity destabilizes networks

Mitigations:
- restrict learning rate ranges
- add weight clipping and decay
- tighten eligibility trace time constants

### Risk: Compute cost explodes

Mitigations:
- enforce pruning and multi-fidelity early
- reduce traces and store less state
- increase sparsity

### Risk: Constraints dominate outcomes and confound emergence

Mitigations:
- keep Strain A as baseline
- introduce constraints only after baseline is stable
- include hard/soft/adaptive strains for contrast

---

## 5. Deliverables

- a staged execution plan with acceptance tests (this document)
- a runbook per stage describing:
  - configs to run
  - expected plots and dashboards
  - criteria for proceeding
