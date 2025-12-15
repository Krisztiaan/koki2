# Executive summary and contributions

## Problem statement

We seek a **first-principles computational framework** in which:

- evolution shapes an agent’s *innate developmental program* (compressed genome),
- the agent learns *within its lifetime* via **local plasticity** and internal modulatory signals,
- and increasingly complex behavior emerges **from survival pressure and interaction**, not from task-specific architectural priors.

The scientific objective is not “train an SNN benchmark model”, but to test a stronger claim:

> A minimal, biologically motivated substrate + homeostatic environments + indirect genomic encoding is sufficient to produce emergent mechanisms we associate with cognition (learning-to-learn, memory, mode switching), and to characterize how additional biological constraints change this emergence.

This collection rewrites the thesis plan to be *cohesive, staged, and falsifiable*.

---

## Core thesis commitments

### Commitment A — Inside-out and action-first framing

We treat behavior and cognition as products of **internally generated dynamics calibrated by action–consequence loops** (inside-out view) rather than stimulus→representation→response pipelines [@buzsaki2019brain].

Operationally, this means:

- environments are **homeostatic survival worlds**, not label prediction tasks,
- evaluation emphasizes **spontaneous dynamics**, **action-conditioned sensory consequences**, and **self-generated internal structure**.

### Commitment B — Minimal substrate, maximal emergence

Baseline (Strain A) avoids hard-coding:

- excitatory vs inhibitory neuron classes,
- specific neuromodulators,
- specialized memory modules,
- fixed small-world or columnar topologies.

Instead, the baseline provides:

- a recurrent substrate with heterogeneous time constants,
- local plasticity with eligibility traces and modulatory gating,
- and indirect genomic encoding.

Any “biological realism” constraints are introduced only as controlled strains (Strain B/C/D).

### Commitment C — Genomic bottleneck as the scaling mechanism

To avoid scaling limits of direct weight evolution, we rely on **indirect encoding** (genomic bottleneck / hypernetwork style): small genomes generate large phenotypes [@shuvaev2024genomic].

---

## Addenda that make the program feasible and biologically credible

### Addendum 1 — Nursing: developmental niche and gradual difficulty

Real organisms are not evaluated from birth in the full adult niche. We formalize **developmental scaffolding** (energy buffers, rich nursery environments, motor/action gating, plasticity schedules) as an explicit module (Chapter X addendum) [@westking1987ontogenetic; @stotz2017dnc].

### Addendum 2 — Pruning: early termination of genetic dead ends

Real evolution eliminates many genotypes early (embryonic/perinatal lethality). We formalize **viability filters** and multi-fidelity evaluation policies that reduce compute waste while protecting diversity [@li2018hyperband; @lehman2011novelty].

---

## Main research questions

1. **Emergence in Strain A:** Under minimal assumptions, do we see evidence of:
   - within-lifetime learning (plasticity that improves outcomes),
   - emergent modulatory signals correlated with drive reduction and prediction errors,
   - memory-like dynamics (persistent state, attractor structure),
   - behavioral mode switching?

2. **Effect of biological constraints:** Do constraints like Dale’s principle, E/I ratio, wiring cost, and small-worldness:
   - accelerate or hinder emergence,
   - improve robustness/generalization,
   - change the *form* of the learned internal mechanisms?

3. **Role of nursing and pruning:** Do developmental scaffolding and viability filters:
   - reduce compute without harming end performance,
   - change which solutions are reachable,
   - improve stability of open-ended search?

---

## Thesis contributions (deliverables)

### Scientific contributions

- A **minimal substrate** specification (state variables, plasticity form, modulatory signals) and a sequence of environments that create graded survival pressures.
- A **measurement and analysis suite** for detecting emergent phenomena (neuromodulation, memory, behavioral modes) consistent with the inside-out stance.
- A controlled comparative methodology (bio-strains A–D + ablation matrix) for evaluating which biological constraints are helpful at which complexity levels.

### Engineering contributions

- A modular JAX-first architecture enabling parallel development of:
  - environments,
  - agent substrate,
  - genome compiler,
  - evolution engine,
  - observability and reproducibility tooling.
- A compute-aware evaluation policy (nursing + multi-fidelity pruning) that makes large-scale experiments feasible.

---

## Scope management

- The main program is **single-agent** with a granular environment ladder.
- Multi-agent “social cognition” is treated as **future work** (explicitly scoped out for thesis delivery), while the architecture keeps multi-agent extension paths open.

---

## How to use this collection

- Start with the master outline: `02_THESIS_OUTLINE_MASTER.md`.
- Use addenda (nursing and pruning) to finalize the evaluation/evolution budget plan early.
- Use the implementation specs to create issue trackers for parallel work.
