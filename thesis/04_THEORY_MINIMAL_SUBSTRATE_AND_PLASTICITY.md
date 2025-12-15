# Minimal substrate and plasticity

## 1. Design goal

The minimal substrate must satisfy two competing requirements:

1. **Expressivity**: capable of rich temporal dynamics, memory-like state, and adaptive behavior under survival pressures.
2. **Non-prescriptiveness**: does not smuggle in high-level cognitive modules (dopamine circuits, hippocampus, etc.).

The substrate is therefore defined as a set of *state variables* and *update rules* that are:

- local (synapse- and neuron-local),
- evolvable (parameters produced by the genome),
- and compatible with large-scale JAX simulation.

---

## 2. State definitions

### 2.1 Neuron state

For neuron \(i\) at time \(t\):

- membrane potential \(v_i(t)\)
- spike \(s_i(t)\in\{0,1\}\)
- adaptive threshold or adaptation current \(a_i(t)\) (optional for ALIF)
- neuron parameters (possibly evolvable):
  - membrane time constant \(\tau_{m,i}\)
  - refractory time \(\tau_{r,i}\)
  - base threshold \(\theta_i\)
  - adaptation time constant \(\tau_{a,i}\) and coupling \(\beta_i\) (ALIF)

A minimal discrete-time update (Euler) might be:

\[
v_i(t+1) = \alpha_i v_i(t) + (1-\alpha_i)\left(\sum_j w_{ij}(t)\,x_j(t) + b_i\right) - s_i(t)\,v_{\text{reset}}
\]
\[
s_i(t) = \mathbb{1}[v_i(t) > \theta_i + \beta_i a_i(t)]
\]
\[
a_i(t+1) = \gamma_i a_i(t) + s_i(t)
\]

where \(\alpha_i = e^{-\Delta t/\tau_{m,i}}\) and \(\gamma_i = e^{-\Delta t/\tau_{a,i}}\).

**Key point:** The neuron model is intentionally simple; complexity is expected to emerge from recurrent connectivity, heterogeneity, and plasticity.

### 2.2 Synapse state

For synapse \(j\rightarrow i\):

- synaptic weight \(w_{ij}(t)\)
- eligibility trace(s) \(e_{ij}(t)\) (plasticity state)
- optional additional traces \(p_{ij}(t)\) for pre/post history

We assume current-based synapses for computational simplicity and compatibility with neuromorphic export. Conductance-based models are optional future work.

---

## 3. Local plasticity rule family

### 3.1 Eligibility traces + modulatory factor (three-factor learning)

We treat **eligibility traces** as the core mechanism for temporal credit assignment:

\[
e_{ij}(t+1) = \lambda e_{ij}(t) + f(\text{pre}_j(t), \text{post}_i(t))
\]

where \(f\) is a local function of pre and post activity/spikes.

Weights update via a modulatory signal \(m(t)\) (global or semi-global):

\[
\Delta w_{ij}(t) = \eta_{ij}\, m(t)\, e_{ij}(t)
\]

This is a canonical form consistent with neuromodulated synaptic plasticity and the e-prop family of approaches [@bellec2020eprop].

### 3.2 ABCD(E) plasticity family (evolvable rules)

To make plasticity broadly expressive while still local, we define a parametric family:

Let \(o_i(t)\) and \(o_j(t)\) be filtered activities (e.g., pre/post traces). Then:

\[
\Delta w_{ij}(t) = \eta \Big(
A_{ij} o_i(t)o_j(t) + B_{ij} o_i(t) + C_{ij} o_j(t) + D_{ij} + E_{ij} m(t)
\Big)
\]

The coefficients \(A_{ij},B_{ij},C_{ij},D_{ij},E_{ij}\) are **genome-produced parameters** (via the plasticity-rule network). This turns “learning rules” into the object of evolution (Baldwin effect style).

### 3.3 Modulatory signals: discovered, not imposed

In Strain A, \(m(t)\) must be computed by the network, not injected as a dopamine-like oracle. Concretely:

- a subset of neurons contributes to a modulatory readout,
- the genome defines how modulatory neurons are selected (via identity-to-role mappings),
- the agent can also compute multiple modulators \(m_k(t)\) (e.g., reward-like, threat-like) but their meaning is emergent.

---

## 4. Heterogeneity as a built-in degrees-of-freedom

Heterogeneity is treated as a low-level mechanism that can yield high-level computational benefits:

- distributions over \(\tau_m\), \(\tau_a\), thresholds, and synaptic time constants
- potentially neuron-wise adaptation strengths

This is motivated both computationally and biologically: diverse time scales allow recurrent networks to represent and integrate information over multiple horizons.

Implementation implication: heterogeneity parameters are not trained by gradient descent in the baseline; they are produced by the genome and tuned by evolution.

---

## 5. Meta-learning perspective (Baldwin effect)

Evolution is responsible for:

- the innate developmental program (network structure and initial parameters),
- the plasticity rule parameters (how the agent changes during life),
- and the “learning regime” (how plasticity is gated).

Within lifetime, learning is local and online.

This implements the Baldwin effect: genomes that encode better learning rules achieve higher fitness over time, which indirectly selects for learning-to-learn.

For context, differentiable plasticity provides a gradient-based analogue of this idea in conventional networks [@miconi2018differentiable].

---

## 6. What training mechanisms are allowed (and where)

To maintain the first-principles claim:

- Within-life learning must remain local (no BPTT over full episodes).
- However, **meta-optimization of genomes** may be done by evolution (primary) or gradients (optional, secondary), as long as the within-life learning rule remains local.

Surrogate gradients may be used for:
- debugging small models,
- or meta-learning the genome parameters,
but not as the primary within-life learning rule in Strain A.

---

## 7. Parameterization and constraints (baseline ranges)

Suggested evolvable ranges (initially broad; narrowed by evidence):

- \(\tau_m \in [5, 100]\) ms (discrete-time equivalent)
- \(\tau_a \in [50, 500]\) ms
- \(\beta \in [0, 2]\)
- \(\theta \in [0.5, 2.0]\) (normalized units)
- plasticity learning rates \(\eta \in [10^{-5}, 10^{-1}]\) (log scale)
- eligibility decay \(\lambda \in [0.8, 0.999]\) per step

These are not biological measurements; they are search bounds to allow evolution to discover effective regimes.

---

## 8. Strain hooks (how biological constraints enter later)

This minimal substrate supports later constraints as separate “strains”:

- Dale’s principle can be imposed by mapping neuron identities to sign constraints on outgoing weights.
- E/I ratios can be imposed by forcing a fraction of neurons to be inhibitory.
- Wiring cost can be imposed by distance-dependent connection penalties in development.

These are excluded from Strain A, but included in Strains B/C/D.

---

## 9. Verification tests (what to validate early)

Before moving to complex environments, validate:

1. **Numerical stability** (no NaNs, bounded state variables).
2. **Signal propagation** (activity does not trivially die or explode).
3. **Plasticity effect** (weights can change and influence behavior).
4. **Modulator effectiveness** (modulatory output can meaningfully gate plasticity updates).

These tests must be runnable in the simplest chemotaxis environments.

---

## 10. Deliverables

- A formal substrate spec (this document) with:
  - precise state variables,
  - local plasticity family,
  - modulatory signal discovery mechanism.
- A set of unit tests for stability and sanity.
- A strain interface for later experimental manipulations.
