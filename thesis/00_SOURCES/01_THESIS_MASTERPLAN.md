# 01 — Thesis masterplan (reworked): biologically‑constrained neuroevolution in JAX

**Document status:** working spec (v1.0)  
**Scope:** thesis-aligned rework of the implementation plan; staged execution plan; module boundaries; verification.  
**Primary goals:** (i) incremental, verifiable progress; (ii) efficiency-first simulation; (iii) biologically-motivated constraints applied *comparatively* (strain framework), not as a monolith.

---

## 1. Thesis objective and non-goals

### 1.1 Objective

Build and study a **biologically-constrained learning system** where:

1. **Evolution** builds the substrate (architecture, parameters, developmental program).
2. **Lifetime learning** (plasticity and/or inner-loop optimization) adapts within an episode or lifespan.
3. The system scales to large numbers of rollouts and ablations through **JAX acceleration**.
4. Biological constraints are treated as **controlled experimental factors** (strain framework) rather than assumptions.

The end result is not “one best agent”; it is a **thesis-grade comparative framework** that can answer:
- Which constraints help/hurt *at which developmental stages*?
- Which mechanisms appear necessary for progression to harder environments?
- What are the compute/efficiency tradeoffs of each mechanism?

### 1.2 Explicit non-goals (for this thesis)

- Full “social cognition” (defer to Future Work; see 12_FUTURE_WORK_AND_DEPLOYMENT.md).
- Realistic biophysical neuron simulations at scale (only where they are needed to test a claim).
- A single monolithic simulator stack; instead provide **interfaces** so backends can be swapped.

---

## 2. Critical review: what must change vs the earlier implementation plan

The earlier plan is strong on literature coverage and tool enumeration, but it risks three thesis failures:

1. **Over-instantiation risk:** too many mechanisms (constraints, plasticity variants, export formats, hardware paths) proposed simultaneously. This obstructs attribution and slows iteration.
2. **Weak developmental framing:** the plan mentions “stages” (chemotaxis→memory→social), but does not enforce a **developmental niche** that makes early competence plausible and testable.
3. **Compute risk:** without an explicit multi-fidelity budget allocator and viability pruning, large-scale neuroevolution becomes intractable.

This rework enforces:
- **A strict environment ladder** with acceptance tests and minimal criterion gates.
- Two thesis addenda implemented as first-class modules:
  - **Nursing** (developmental niche scaffolding; 03_THEORY_DEVELOPMENTAL_NICHE_NURSING.md)
  - **Pruning** (viability screening + multi-fidelity evaluation; 04_THEORY_VIABILITY_PRUNING_AND_SELECTION.md)
- **Abstraction layers** so work streams can proceed in parallel and be integrated incrementally (05_IMPLEMENTATION_ARCHITECTURE_LAYERS.md).

---

## 3. The biological stance: “inside-out” + developmental niche + genomic bottleneck

This thesis is explicitly aligned to:
- **Inside-out**: action-first, internally generated dynamics as the primary driver; perception is shaped by action and internal models [@buzsaki2019brain].  
- **Developmental niche**: early-life scaffolding and resource provision (the inherited environment) are part of what evolves [@westking1987ontogenetic; @stotz2017dnc; @vanschaik2023provisioning].  
- **Genomic bottleneck**: indirect encoding as an efficiency and generalization mechanism; compact genomes generate large circuits [@shuvaev2024genomic; @zador2019critique].

Implication for engineering: the environment cannot immediately demand optimal behavior. Instead, we explicitly model **ontogeny** through resource provisioning and/or reward richness early, then tighten constraints.

---

## 4. Thesis contributions (what is novel, even if components are known)

The novelty is in **integration + instrumentation + controlled comparisons**, not in claiming “first-ever SNN evolution”.

### 4.1 Novel module A: Developmental Niche Scheduler (Nursing)

A mechanism that parameterizes **early-life advantages** and phases them out according to either:
- time/age; or
- competence milestones (performance triggers).

This turns qualitative biology into an implementable schedule (see 03_…).

### 4.2 Novel module B: Viability Envelope + Multi-fidelity Budgeting (Pruning)

A compute governance system that:
- quickly rejects degenerate genomes (NaNs, saturation, no sensorimotor coupling),
- allocates more compute only to candidates with evidence of viability,
- reduces selection distortion via carefully designed thresholds and minimum evaluation.

This is not optional: without it, the thesis becomes a compute project.

### 4.3 Novel module C: Strain Manager for constraints as experimental factors

Implement a **comparative “bio-strain” framework**: strains are coherent bundles of constraints (E/I, Dale, sparsity, topology, plasticity scope). Each strain runs through the same environment ladder with the same budget allocator.

The strain manager produces:
- a manifest of biological assumptions per run,
- a reproducible annealing schedule (soft→hard where relevant),
- and an ablation grid.

### 4.4 Novel module D: Verified environment ladder with “minimal criterion” gates

Use minimal-criterion concepts to ensure progression is not dominated by reward hacking or brittle overfitting [@brant2017mcc; @lehman2011novelty]. The thesis contribution is a *practical* ladder with measurable checks.

---

## 5. Architecture choices for efficiency (JAX-first)

### 5.1 Why JAX is non-negotiable here

Population-based search + many rollouts means you must:
- vectorize evaluation (`vmap`/sharding),
- keep execution inside XLA,
- avoid Python-in-the-loop per timestep.

The planning assumption is a JAX environment and JAX agent step function.

### 5.2 SNN stack options and selection criteria

**Option 1: Spyx** (JAX/Haiku; optimized SNN training & neuroevolution) [@heckel2024spyx]  
Pros: end-to-end JIT, good data-packing, already targets large-scale training loops.  
Cons: research flexibility constrained by library abstractions.

**Option 2: BrainPy** (differentiable brain simulator in JAX; multiple scales) [@wang2024brainpy]  
Pros: broader neuroscience modeling surface; event-driven sparse operators; autodiff.  
Cons: may require more careful engineering to reach maximum throughput on target workloads.

**Plan:** Keep an interface so the backend is swappable; default to **Spyx for throughput**, keep BrainPy as a validation backend for selected experiments, not for the entire thesis.

### 5.3 Evolution engine options

**evosax** (JAX-based evolution strategies) is the baseline implementation target because it is designed for accelerator execution and an ask-eval-tell loop [@lange2022evosax].  
Alternative: custom ES implementations if we need niche features (e.g., staged budget allocation tightly coupled to evaluation). The initial plan uses evosax, extending only if required.

---

## 6. Staged environment ladder (granular) and thesis milestones

The environment ladder is designed to satisfy:
- **incremental difficulty**
- clear acceptance tests
- minimal dependence on complex sensors or physics early
- support for “nursing” schedules.

Detailed specs are in 06_IMPLEMENTATION_ENVIRONMENT_LADDER.md.

### Level L0 — Sensorimotor viability (“alive and coupled”)

Goal: prove the agent can generate actions that change sensory input and internal energy.  
Success criteria: positive correlation between actions and improved internal energy; non-trivial sensor→motor mutual information.

### Level L1 — Chemotaxis in smooth fields

Goal: gradient following with minimal memory.  
Success criteria: reliably reach source above threshold under noise.

### Level L2 — Chemotaxis with inertia + obstacles

Goal: introduce control and planning constraints without memory requirements.  
Success criteria: avoid obstacles and still reach resource.

### Level L3 — Foraging with depletion (short-term state)

Goal: repeated decisions; exploration/exploitation; satiety.  
Success criteria: maintain energy above survival threshold for lifespan.

### Level L4 — Two-cue environments (context gating)

Goal: disambiguate cues; start requiring internal state (context).  
Success criteria: choose correct cue-conditioned policy.

### Level L5 — Delayed reward + hazard fields (credit assignment)

Goal: stress eligibility traces / neuromodulation or inner-loop learning.  
Success criteria: delayed outcomes handled without collapse.

### Level L6 — Sparse reward navigation with partial observability (working memory)

Goal: require memory to solve; demonstrate emergence of memory-like internal dynamics.

> Social/cultural environments are deferred.

---

## 7. Verification philosophy: “incremental and falsifiable”

This thesis is structured so each stage has:
- an explicit **hypothesis**,
- a minimal set of mechanisms needed to test it,
- and acceptance tests that can be passed/fail without interpretation.

### 7.1 Acceptance tests at three layers

1. **Functional:** does the agent solve the environment?  
2. **Mechanistic:** do we observe the intended mechanism (e.g., modulation improves learning vs fixed weights)?  
3. **Comparative:** across strains, does the constraint change performance/robustness/cost?

### 7.2 Reproducibility rules

Every run must produce:
- a full config + seed manifest,
- exact code version identifiers,
- environment and evaluation determinism checks,
- and enough recorded state to replay key trajectories.

See 10_IMPLEMENTATION_OBSERVABILITY_REPRODUCIBILITY.md.

---

## 8. Bio-strain comparative framework (high-level)

We formalize constraints as **strains** so claims are comparative.

Example strain families (illustrative; finalized in 07_IMPLEMENTATION_AGENT_SUBSTRATE.md):

- **Strain A (Minimal):** minimal constraints; baseline for learnability and throughput.
- **Strain B (E/I + Dale):** enforce sign constraints and population split.
- **Strain C (Plasticity):** enable local plasticity + neuromodulation schedules.
- **Strain D (Developmental encoding):** genomic bottleneck hypernetwork; optional growth programs.

Key: strains are not “better”; they are experimental conditions.

---

## 9. Development plan: staged implementation and parallel workstreams

### Workstream 1 — Environment engine + ladder (L0–L6)

Deliverables:
- deterministic JAX env API
- environment tasks with golden tests
- “nursing knobs” implemented as environment parameters

### Workstream 2 — Agent substrate + backends

Deliverables:
- SNN/RNN agent interface
- baseline simple controller (sanity)
- Spyx backend + optional BrainPy backend

### Workstream 3 — Genome + developmental compiler

Deliverables:
- direct encoding baseline
- genomic bottleneck encoding [@shuvaev2024genomic]
- mutation operators + genome validity checks

### Workstream 4 — Evolution engine + budgeting

Deliverables:
- ask-eval-tell outer loop (evosax baseline) [@lange2022evosax]
- multi-fidelity evaluation (successive halving/hyperband style) [@jamieson2016sh; @li2018hyperband]
- viability pruning + safety against selection artefacts

### Workstream 5 — Observability + reproducibility

Deliverables:
- run manifests, replayable trajectories
- metric schema, dashboards
- experiment registry for strain comparisons

---

## 10. Minimal set of experiments required for a defensible thesis

The thesis should be defensible even if only these are completed:

1. L0–L3 environments complete with acceptance tests.
2. Two strains (A minimal, C plasticity) run through ladder with:
   - nursing enabled vs disabled
   - pruning enabled vs disabled
3. Genomic bottleneck encoding integrated at L3, compared to direct encoding.
4. Reported compute cost and throughput metrics for every condition.

Everything else is extension, not dependency.

---

## 11. Risk register (and mitigation)

### Risk: compute explosion
Mitigation: pruning + multi-fidelity budgeting is mandatory (04_…, 09_…).

### Risk: attribution failure (“too many knobs”)
Mitigation: strains + staged ladder; only add one new factor per milestone.

### Risk: backend instability / research drift
Mitigation: keep the core agent interface tiny; treat simulators as replaceable modules.

### Risk: non-reproducible results
Mitigation: deterministic evaluation contracts; strict manifests; replay tests.

---

## 12. References

See `references.bib`.
