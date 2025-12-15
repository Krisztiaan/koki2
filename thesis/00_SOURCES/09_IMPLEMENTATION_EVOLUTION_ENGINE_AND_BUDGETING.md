# 09 — Evolution engine & compute budgeting: ES + multi-fidelity + pruning/nursing integration

**Purpose:** Specify the outer-loop evolutionary process, including how evaluation budgets are allocated, how pruning is performed safely, and how nursing schedules interact with selection.

---

## 1. Outer-loop objective: what evolution optimizes

At environment level $\ell$, evaluation produces:
- fitness $F$ (scalar for selection),
- metrics $M$ (structured, for analysis and pruning),
- artifacts $A$ (sampled trajectories, replay seeds).

### 1.1 Fitness as “survival + competence”

For early levels, fitness should strongly reflect survival/homeostasis:
\[
F = \sum_{t=0}^{T-1} \gamma^t \, r_t \;+\; \lambda \cdot \mathbb{1}[\text{survived to }T]
\]
Where reward $r_t$ is derived from energy gain and penalties (hazards). Avoid overly shaped distance rewards except as debugging variants.

### 1.2 Multi-task evaluation

For robustness, fitness is averaged across:
- multiple random seeds
- multiple environment variants (source locations, noise)

This reduces reward hacking.

---

## 2. Evolution strategy selection: baseline and extensions

### 2.1 Baseline: evosax ask-eval-tell

The default implementation target is **evosax**, which provides JAX-based evolution strategies [@lange2022evosax]. Advantages:
- accelerator-friendly vectorized evaluation
- standardized loop

### 2.2 When to extend beyond evosax

Implement custom logic only if required by thesis claims:
- tight coupling with multi-fidelity evaluation (rungs)
- novelty/quality-diversity archives
- genotype-phenotype caching across generations

---

## 3. Budget allocation: successive halving / Hyperband adapted for neuroevolution

### 3.1 Why multi-fidelity budgeting

Neuroevolution evaluation is expensive and noisy. Multi-fidelity methods allocate more compute to promising candidates.

Successive halving and Hyperband provide principled schemes for allocating resources across configurations [@jamieson2016sh; @li2018hyperband]. We adapt the scheme to candidates/genotypes.

### 3.2 Rung structure

Define rungs $k=0..K$ with increasing budgets $b_k$:
- $b_0$: micro-rollout (P1)
- $b_1$: short episode (f1)
- $b_2$: full episode (f2)
- $b_3$: multi-seed battery (f3)

At each rung:
- evaluate all candidates assigned to that rung,
- promote top fraction $1/\eta$,
- optionally promote an exploration quota $\rho$ chosen at random.

### 3.3 Promotion score vs selection fitness

**Important separation:**
- **Promotion score** decides who gets more compute.
- **Selection fitness** decides who reproduces.

Promotion can be based on conservative viability/competence measures; selection uses full-fidelity fitness when available.

This prevents early noisy scores from dominating evolution.

---

## 4. Pruning integration: viability envelopes and conservative rejection

Pruning rules are defined in 04_THEORY_VIABILITY_PRUNING_AND_SELECTION.md.

In implementation, pruning occurs at:
- compile time (P0)
- micro-rollout (P1)
- within-episode envelope checks (early termination)

**Invariant:** any stochastic pruning decision must be replayable (seed recorded).

---

## 5. Nursing integration: avoiding “selection on nursing” artefacts

Nursing makes early evaluation easier. That is intended for development, but it can distort selection if mishandled.

### 5.1 Rule: nursing is not part of the agent observation (by default)

If the agent observes nursing parameters directly, it may overfit to “being nursed.”

### 5.2 Dual-evaluation protocol (recommended)

For levels where nursing is active, evaluate each candidate on:
- **nursed evaluation:** high resource support (developmental phase)
- **weaned evaluation:** reduced support (adult competence)

Define selection fitness as a weighted combination:
\[
F = \alpha F_\mathrm{nursed} + (1-\alpha) F_\mathrm{weaned}
\]
with $\alpha$ decreasing over generations/age.

This enforces that learned competence survives weaning.

---

## 6. Minimal criterion and novelty: preventing deceptive optimization

### 6.1 Minimal criterion gates

Use minimal criterion thresholds to decide eligibility for progressing to harder levels [@brant2017mcc].

This prevents:
- spending compute optimizing a level past what’s needed
- reward-hacking local optima

### 6.2 Novelty search as an optional auxiliary

Novelty search encourages exploration by rewarding behavioral novelty rather than objective performance [@lehman2011novelty]. It can be used as:
- an exploration quota selection criterion, or
- a second objective in a multi-objective ES.

**Thesis guidance:** novelty is a tool for avoiding stagnation; it should be introduced only if baseline evolution collapses into repetitive strategies.

---

## 7. Determinism and evaluation noise management

### 7.1 Seed handling

For each candidate $i$, generation $g$, and rung $k$:
- derive a deterministic key:
  \[
  \mathrm{key}_{i,g,k} = \mathrm{fold\_in}(\mathrm{fold\_in}(\mathrm{base\_key}, g), i \oplus 1000k)
  \]
This ensures reproducibility and stable comparisons across strains.

### 7.2 Re-evaluation and “fitness inflation”

To avoid overfitting to a small set of seeds:
- rotate evaluation seeds on a schedule
- keep a fixed validation seed set for reporting

---

## 8. Outputs required for thesis reporting

For each run (strain × nursing × pruning):
- compute budget used (total simulated steps)
- wall-clock throughput
- learning curves per environment level
- viability/pruning statistics (dead-end rate)
- mechanistic metrics (activity, plasticity magnitude, etc.)
- replayable trajectories for qualitative analysis

---

## 9. References

See `references.bib`.
