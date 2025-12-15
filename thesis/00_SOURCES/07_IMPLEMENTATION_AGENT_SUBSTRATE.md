# 07 — Agent substrate: SNN/RNN core, plasticity hooks, and bio‑strain knobs

**Purpose:** Specify the agent “execution core” and the controlled biological constraint bundles (“strains”) used for comparative experiments.

---

## 1. Design principles

1. **Start minimal, then add biological mechanisms only when the environment demands them.**  
   This is essential for attribution and compute management.

2. **Inside‑out alignment:** agents must be stateful and capable of internally generated dynamics [@buzsaki2019brain].

3. **Comparative by design:** biological constraints are treated as controlled factors (strains), not assumptions.

4. **Backend independence:** define an agent interface that can be implemented by Spyx [@heckel2024spyx], BrainPy [@wang2024brainpy], or a custom minimal SNN kernel.

---

## 2. Agent interface (recap)

The substrate must implement:

- `init(key, phenotype, agent_params) -> AgentState`
- `act(agent_state, obs, phenotype, agent_params, key) -> (agent_state', action, aux)`
- optional: `learn_update(...)`

See 05_IMPLEMENTATION_ARCHITECTURE_LAYERS.md for contracts.

---

## 3. Baseline substrates (staged)

### 3.1 Phase 0 baseline: small recurrent rate network (debugging and fast iteration)

Use a minimal RNN/GRU-like core early for:
- environment ladder debugging
- verifying orchestrator and evolution loop
- establishing throughput baselines

This phase is not the thesis end state; it is a reliability scaffold.

### 3.2 Phase 1 substrate: recurrent spiking network (primary thesis substrate)

Target: a recurrent SNN capable of:
- low-dimensional sensorimotor control (L0–L2)
- strategies and internal state (L3–L4)
- credit assignment with plasticity (L5)
- memory emergence (L6)

Spyx is the default execution backend for throughput [@heckel2024spyx]. BrainPy is reserved for selected validation experiments [@wang2024brainpy].

---

## 4. Plasticity as a controlled factor

### 4.1 Why plasticity must be optional

If plasticity is always on, you cannot distinguish:
- what evolution encoded vs what lifetime learning acquired.

Thus, plasticity is introduced as a **strain dimension**.

### 4.2 Supported plasticity paradigms (ranked by thesis relevance)

1. **Local synaptic plasticity with modulators (preferred for bio-claims)**  
   Implement as eligibility traces + modulatory scalar.

2. **E-prop style online learning (bio-plausible gradient surrogate)**  
   E-prop provides an online learning approximation for recurrent spiking networks [@bellec2020eprop]. Use as an optional learning rule in plastic strains, especially for L5–L6.

3. **Differentiable plasticity (useful as a “best effort” lifetime learner)**  
   Differentiable plasticity optimizes Hebbian update coefficients with gradient descent [@miconi2018differentiable]. This is less biologically strict but valuable as an upper bound.

### 4.3 Surrogate gradient training (if used)

Surrogate gradients can be used for inner-loop updates or for training components. Recent theory clarifies their relationship to stochastic spiking models [@gygax2024surrogate]. For stabilizing direct SNN training, consider threshold-robust surrogate gradients [@kook2025trsg] and sparse/“masked” gradient updates [@li2024msg] as optional tools.

**Thesis stance:** surrogate gradients are permitted as implementation tools, but the thesis questions focus on evolution + lifetime learning rather than purely supervised accuracy.

---

## 5. Bio-strain design: constraints as experimental bundles

### 5.1 Why strains (instead of individual knobs)

Many biological constraints interact. Strains are coherent bundles with clear hypotheses. Each strain must be:

- precisely defined,
- reproducible (manifested),
- comparable across environments and budgets.

### 5.2 Proposed strain set (minimum viable for thesis)

#### Strain A — Minimal baseline
- recurrent network (rate or spiking)
- no sign constraints
- fixed weights (no plasticity)
- direct encoding of parameters

**Use:** learnability baseline; throughput benchmark; controls.

#### Strain B — E/I separation + Dale-style sign constraints
- explicit excitatory and inhibitory populations (fixed ratio)
- outgoing weights constrained by neuron type
- constraints applied softly early, annealed toward hard projection later

**Hypothesis:** improves stability/robustness without killing learnability in L3+.

#### Strain C — Plasticity + neuromodulation
- synaptic traces / eligibility
- modulatory scalar (reward prediction error proxy or global reward)
- critical-period schedule: high plasticity early, reduced later

**Hypothesis:** enables faster adaptation and solves L5+ where fixed weights fail.

#### Strain D — Developmental encoding (genomic bottleneck)
- genotype encodes a compact generator that produces a larger network (“g-network” style) [@shuvaev2024genomic]
- optional plasticity (D can be paired with A/B/C as a subfactor)

**Hypothesis:** improved generalization/transfer and reduced search dimensionality.

> Additional strains (small-world topology, dendritic heterogeneity, etc.) are explicitly secondary unless demanded by results.

---

## 6. Constraint implementation patterns (engineering guidance)

### 6.1 Soft→hard annealing

For constraints that can block exploration (e.g., sign constraints), start as penalties and anneal to hard constraints when populations show competence. This provides exploration early and plausibility later.

### 6.2 Separation of “structure” vs “parameters”

Prefer to keep:
- architecture structure (masks, connectivity patterns) in the phenotype’s structural fields,
- trainable/evolved weights in parameter arrays.

This enables pruning and compile-time checks.

### 6.3 Heterogeneity knobs

Heterogeneous neuron time constants and thresholds can be introduced gradually:
- start with shared parameters for speed
- later allow per-neuron heterogeneity (in plastic or developmental strains)
Optional: dendritic heterogeneity models show multi-timescale benefits [@zheng2024dendritic], but integrate only after baseline ladder is stable.

---

## 7. Diagnostics required from the agent substrate

The `aux` output of `act()` should minimally include:

- spike rate statistics (if SNN)
- action saturation fraction
- energy usage proxy (if internal metabolic model is used)
- plasticity magnitude (if plastic strain)

These are used for:
- pruning (detect degeneracy),
- mechanistic analysis (thesis claims),
- and debugging.

---

## 8. References

See `references.bib`.
