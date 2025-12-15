# Bio-strain comparative framework and ablations

## 1. Purpose

The core thesis claim depends on a **first-principles baseline** (Strain A) where mechanisms emerge rather than being imposed. However, the thesis also needs to test:

- which biological constraints help/hurt,
- when they help (task complexity dependence),
- and whether they change *how* learning emerges.

We therefore define a controlled **bio-strain comparative framework**.

---

## 2. Constraint catalog (what we will vary)

The following constraints are treated as experimental factors, not baseline assumptions.

### 2.1 Dale’s principle (sign constraint)

Constraint: each neuron has either all-positive or all-negative outgoing synapses.

Implementation options:
- hard projection of signs at development and after plasticity updates,
- soft penalty on sign violations (Strain C).

### 2.2 Excitatory/inhibitory (E/I) ratio

Constraint: fraction of inhibitory neurons fixed (e.g., 20%).

Implementation options:
- hard architectural split (Strain B),
- soft selection pressure based on emergent sign clustering (Strain C).

### 2.3 Sparse activity / sparse coding

Constraint: firing rates remain low; activity is selective.

Implementation options:
- soft penalty on average firing rate,
- homeostatic activity regulation (optional module).

### 2.4 Wiring cost / distance-dependent connectivity

Constraint: long connections are penalized or less likely.

Implementation options:
- incorporate distance in connection rule logits,
- add a wiring-length penalty term to fitness (soft).

### 2.5 Small-worldness and modularity

Constraint: encourage high clustering with short path lengths.

Implementation options:
- soft metrics computed on developed graph,
- rewiring operators (hard) as a future experiment, not baseline.

### 2.6 Temporal dynamics constraints

Constraint: distributions over time constants and synaptic delays match plausible ranges.

Implementation options:
- hard parameter bounds,
- soft penalties for extreme values.

---

## 3. Strain definitions

### Strain A — First-principles baseline

- No hard-coded E/I, Dale’s, topology, sparsity.
- Only minimal substrate + bottleneck + local plasticity + homeostatic environments.
- Biological structure may emerge; we measure it.

### Strain B — Bio-hard

- Hard constraints at development time:
  - E/I split enforced,
  - Dale’s sign constraints enforced,
  - optional wiring cost implemented structurally (distance-based connection probability).

### Strain C — Bio-soft

- No hard constraints.
- Fitness includes soft penalties/bonuses:
  - sign consistency (Dale-like),
  - E/I ratio deviation,
  - wiring cost,
  - sparsity metrics.

### Strain D — Bio-adaptive

- Start like Strain A.
- Gradually introduce biases if certain motifs repeatedly emerge among high performers.
- Goal: “lock in” beneficial motifs without blocking early exploration.

---

## 4. Ablation matrix

We define an ablation matrix across three axes:

1. Encoding:
   - direct encoding
   - CPPN-rule bottleneck
2. Learning:
   - fixed weights (no plasticity)
   - evolved plasticity with modulatory gating
3. Constraints:
   - Strains A/B/C/D

This yields a structured comparison such as:

| Encoding | Plasticity | Strain | Expected outcome |
|---------:|-----------:|:------:|------------------|
| direct   | none       | A      | baseline reflexes; limited scaling |
| bottleneck | none     | A      | innate structured reflexes; limited adaptation |
| direct   | yes        | A      | adaptation possible but mutation fragile at scale |
| bottleneck | yes      | A      | primary thesis target |
| bottleneck | yes      | B/C/D  | test constraint impacts |

---

## 5. Cross-strain metrics

We compare strains across:

### 5.1 Performance metrics

- survival time distribution
- resource acquisition
- robustness to environment perturbations
- generalization to new seeds / layouts

### 5.2 Learning metrics

- within-life improvement curves
- adaptation speed after environment shifts
- stability of learning (no catastrophic drift)

### 5.3 Structural metrics

- weight sign consistency (Dale-like clustering)
- emergent E/I-like partition (sign-separated connectivity)
- wiring length distributions
- small-worldness / modularity measures (if computationally feasible)

### 5.4 Efficiency metrics

- compute cost per achieved performance
- evaluation steps wasted on dead ends (pruning effectiveness)

---

## 6. Predictions (what we expect to learn)

Hypotheses to test:

- Hard constraints (Strain B) may **help** in complex tasks by shrinking the search space, but **hurt** early exploration.
- Soft constraints (Strain C) may provide a better exploration/exploitation tradeoff.
- Adaptive constraints (Strain D) may yield the best of both worlds if motifs are genuinely beneficial.

Importantly, these are empirical hypotheses; the framework is designed to measure them cleanly.

---

## 7. Implementation hooks

Constraints are implemented as hooks at two points:

1. Development time (hard constraints): modify `develop(genome, dev_cfg)` to enforce:
   - sign projections,
   - neuron-type partitioning,
   - distance-dependent wiring.

2. Evaluation time (soft constraints): add evaluator terms to fitness aggregation.

This maintains modularity and prevents entangling constraints with the baseline agent implementation.

---

## 8. Deliverables

- A strain configuration schema.
- A run plan specifying which strains are run on which environment levels.
- A dashboard/report template for cross-strain comparisons.
- A reproducible ablation protocol ensuring fair comparison.
