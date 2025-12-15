# Addendum: viability pruning and multi-fidelity evaluation

## 1. Motivation

Large-scale neuroevolution wastes compute on genomes that are:

- developmentally degenerate (invalid connectivity, NaNs),
- behaviorally inert (no actions),
- immediately lethal (cannot survive even in nursery conditions).

Biological evolution removes analogous genotypes early through viability selection (embryonic/perinatal lethality), which is common in large-scale knockout and viability studies [@dickinson2016impc]. Computationally, we implement an explicit **viability filter**.

However, naive early stopping can discard “slow starters” whose competence appears only after sufficient ontogeny—particularly relevant under nursing schedules. The evolutionary computation literature also emphasizes careful handling of lethal/non-viable regions to avoid biasing search [@allmendinger2011lethal].

This addendum formalizes pruning in a way that is:

- biologically motivated,
- compute-efficient,
- and explicitly safeguarded against premature convergence.

---

## 2. Definitions

### 2.1 Episode-level lethal termination

If internal state violates viability bounds, terminate the episode immediately. This is handled inside the environment loop and is always active.

### 2.2 Genome-level viability

Define a genome \(G\) as **minimally viable** if, under a fixed nursery environment and nursing schedule, the developed agent satisfies:

- survival time \(T_{\text{alive}} \ge T_{\min}\),
- non-degenerate behavior (action distribution not collapsed; e.g., entropy above \(\epsilon_H\)),
- minimal task engagement (e.g., net energy gained above \(\epsilon_E\) to rule out “alive but inert” edge cases),
- numerically stable neural dynamics.

Formally, define a viability score \(V(G)\in\{0,1\}\).

---

## 3. Minimal Viability Test (MVT)

### 3.1 Procedure

For each genome:

1. Develop phenotype via `develop(G)`.
2. Run \(K\) short episodes of length \(T_{\text{mvt}}\) in nursery conditions.
3. Compute:
   - survival fraction \(S = \frac{1}{K}\sum_k \mathbb{1}[T_{\text{alive}}^{(k)} \ge T_{\min}]\)
   - action entropy \(H(\pi)\) from the discrete action histogram (low values indicate collapse)
   - total energy gained \(\Delta E_{\text{gain}}\) (detects “alive but non-interacting” policies)
   - stability flags (NaNs, overflow)

Declare non-viable if:

\[
S=0 \;\;\text{or}\;\; H < \epsilon_H \;\;\text{or}\;\; \Delta E_{\text{gain}} < \epsilon_E \;\;\text{or}\;\; \text{unstable}
\]

### 3.2 Output and semantics

Non-viable genomes receive:

- fixed minimal fitness,
- no full evaluation allocation,
- and do not reproduce (except as mutation sources if configured).

This is not “performance ranking”; it is a minimal criterion filter in the spirit of minimal-criterion approaches in open-ended evolution [@brant2017mcc].

---

## 4. Multi-fidelity evaluation policy (Hyperband-inspired)

After MVT, we still need to allocate evaluation budget efficiently among viable genomes.

We use a runged evaluation policy inspired by multi-fidelity / successive halving [@jamieson2016sh; @li2018hyperband]:

- Rung 0: MVT viability
- Rung 1: short-horizon evaluation under early-life phases
- Rung 2: full-horizon evaluation across full nursing-to-adult schedule and multiple environment variants

### 4.1 Promotion rule

A conservative promotion rule:

- Promote the top \(p\%\) by Rung-1 proxy fitness **plus** a novelty quota (see below).

### 4.2 Why not pure successive halving?

We explicitly avoid aggressive halving because:

- learning may be delayed,
- some genomes may require longer to express competence.

Therefore, rung-1 is used to eliminate only *obviously poor* viable genomes, not to perfectly rank candidates.

---

## 5. Novelty/coverage safeguards

Objective-only pruning risks premature convergence and loss of stepping stones. We therefore include novelty safeguards:

- compute a behavior descriptor \(b(\tau)\) (e.g., visitation histogram, resource usage, internal state trajectory summary),
- compute novelty relative to an archive [@lehman2011novelty].

Promotion includes:

- always promote the top \(q\%\) most novel genomes passing MVT, regardless of early proxy fitness.

This ensures exploration continues even under strict compute budgets.

---

## 6. Interaction with nursing

Nursing and pruning are complementary:

- nursing reduces the number of false “dead ends” by giving marginal genomes a safe niche to express viability,
- pruning prevents nursing from causing runaway compute cost by filtering clearly non-viable genomes early.

We treat nursery conditions for MVT as fixed and explicitly report them per experiment.

---

## 7. Metrics and diagnostics

We log:

- MVT pass rate per generation,
- reasons for failure (death early, no action, instability),
- correlation between rung-1 proxy and final fitness,
- diversity metrics under different pruning settings.

This supports a key thesis requirement: pruning should not silently bias results.

---

## 8. Ablations

We compare:

1. No pruning (full evaluation for all genomes)
2. MVT only
3. MVT + multi-fidelity (with and without novelty quota)

Outcomes:

- compute spent per generation,
- best and median final fitness for fixed compute budget,
- diversity retention.

---

## 9. Deliverables

- A formal MVT definition and fixed nursery evaluation protocol.
- A multi-fidelity evaluation policy integrated into the evolution engine.
- Novelty/coverage safeguards and logging to quantify pruning bias.
