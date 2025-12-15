# Biasing evaluators and open-endedness

## 1. Why we need biasing evaluators

Even with homeostatic reward, pure survival fitness can be:

- sparse (many genomes die early),
- deceptive (short-term hacks beat long-term learning),
- brittle (overfit to one environment instance),
- slow to explore (local optima and dead ends).

In biology, evolution is not guided by a single scalar objective either; selection pressures are multifactorial and environments vary.

We therefore introduce **biasing evaluators**: additional metrics that shape selection pressure **without specifying the internal mechanism** that must be used to succeed.

This approach preserves the thesis’ core claim: mechanisms emerge because they are useful, not because they are forced.

---

## 2. Base fitness (always active)

We keep a simple base objective:

- survival time,
- net resource acquisition.

This objective must remain present in all experiments to preserve comparability.

---

## 3. Evaluator taxonomy

### Tier 1 — Viability (always active)

- survive at least a minimal time in nursery conditions,
- maintain internal variables above lethal thresholds.

(Operationally implemented via nursing + pruning modules.)

### Tier 2 — Learning capability (soft, delayed activation)

We want to select for *learning-to-learn* rather than fixed reflexes. Possible evaluators:

- within-life improvement: \( \Delta F_{\text{episode}} = F_{\text{late}} - F_{\text{early}} \)
- adaptation to environmental shifts mid-episode
- performance recovery after perturbations

Tier 2 should be introduced only after basic viability emerges.

### Tier 3 — Behavioral diversity / novelty

Novelty search rewards behavioral novelty rather than objective score [@lehman2011novelty]. This is valuable when objective-driven search is deceptive—an argument related to broader open-endedness perspectives emphasizing stepping stones over fixed objectives [@stanley2015greatness]. This is valuable when:

- objective landscapes are deceptive,
- novel stepping stones are needed.

Define a behavior descriptor \(b(\tau)\) from trajectory \(\tau\) (e.g., visitation histogram, resource consumption pattern), and novelty as:

\[
\text{novelty}(\tau) = \frac{1}{k}\sum_{i=1}^k \|b(\tau)-b(\tau_i)\|
\]

where \(\tau_i\) are nearest neighbors in an archive.

Tier 3 can be used either as:
- a multi-objective component, or
- a promotion safeguard (for pruning/multi-fidelity).

### Tier 4 — Quality-Diversity (QD)

QD methods aim to fill a map of diverse high-performing behaviors, e.g. MAP-Elites [@mouret2015mapelites]. QD is relevant for this thesis because:

- it preserves diversity under strong selection,
- it yields a structured “atlas” of strategies,
- it supports interpretability (what behaviors exist?).

We do not require full QD early; we can introduce a minimal archive later once evaluation cost is manageable.

---

## 4. Environment co-evolution and open-ended search (optional)

### 4.1 POET-style co-evolution

POET evolves environments and agents together, generating a curriculum automatically and maintaining novelty in both spaces [@wang2019poet].

This is potentially powerful but high risk for thesis scope because:

- it adds another search space (environments),
- analysis becomes harder.

We treat POET as a late-phase optional extension (post-baseline success).

### 4.2 Minimal Criterion Coevolution (MCC)

MCC proposes maintaining open-ended progress by requiring only that solutions pass a minimal criterion, rather than optimizing a fixed objective [@brant2017mcc].

MCC aligns well with:
- our viability filters (MVT),
- and our reluctance to over-specify fitness.

We can adopt MCC-style “minimal criteria” even without full coevolution by:

- requiring survival thresholds,
- and prioritizing novelty among those that pass.

---

## 5. Staged bias introduction (critical to avoid “baked in” claims)

Biasing evaluators are introduced in stages:

1. Stage A: only viability + base fitness in very easy environments.
2. Stage B: add learning capability evaluators once stable survival emerges.
3. Stage C: add novelty/QD once the search begins to stagnate.
4. Stage D: consider environment co-evolution only after we can demonstrate stable mechanistic emergence.

This staging prevents:
- the evaluator itself becoming the “solution”,
- or early biases drowning out the emergence of learning.

---

## 6. Soft vs hard constraints (interface with bio-strains)

Biasing evaluators can also implement biological constraints as **soft objectives** (Strain C):

- encourage E/I balance,
- encourage sparse activity,
- encourage small-worldness or modularity,
- penalize wiring cost.

Hard constraints (Strain B) operate at development time instead.

---

## 7. Deliverables

- A formal specification of evaluator tiers and when they activate.
- A novelty descriptor and archive definition appropriate to each environment level.
- A QD “on-ramp” plan: minimal archive early, full MAP-Elites later if needed.
- Clear ablation experiments demonstrating that emergence is not an artifact of a particular evaluator.
