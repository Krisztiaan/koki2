# Agent Core specification (neural substrate, plasticity, modulators)

## 1. Requirements

Agent Core must:

- be expressively sufficient (recurrent temporal dynamics + plasticity),
- run efficiently in JAX (`jit`, `scan`, `vmap`),
- support multiple strains (constraints toggles),
- expose logs for analysis (but keep overhead low).

---

## 2. Framework choice: technology options (JAX)

### Option A — SNNAX (Equinox-based)

Pros:
- flexible recurrent dynamics and explicit step-by-step control,
- good fit for custom learning rules and eligibility traces.

Cons:
- may require more careful performance tuning;
- framework maturity relative to larger ecosystems should be validated.

Reference: SNNAX preprint (arXiv:2409.02842).

### Option B — Spyx (Haiku-based)

Pros:
- designed for JAX spiking networks;
- supports JIT-compiled training loops and neuromorphic workloads.

Cons:
- some design choices may optimize feedforward / specific workloads; recurrent and custom plasticity workflows need validation.

Reference: Spyx preprint [@heckel2024spyx].

### Option C — BrainPy (JAX brain simulator)

Pros:
- broad neuroscience simulator features;
- event-driven sparse operators;
- suitable for heterogeneous neuron and synapse models.

Cons:
- may be heavier than needed for minimal thesis substrate;
- integration into population-based evolution loops must be tested.

Reference: BrainPy (ICLR 2024) [@wang2024brainpy].

**Recommendation for thesis implementation:** start with SNNAX-style explicit stepping for maximal control of plasticity and modulators; evaluate Spyx/BrainPy as performance alternatives once correctness is established.

---

## 3. Data structures

### 3.1 AgentParams (static)

- neuron parameters:
  - \(\tau_m, \tau_a, \theta, \beta\) etc
- sparse connectivity:
  - `edge_index` (E×2)
  - `w` (E,)
  - plasticity coefficients per edge (E×P)
- role mappings:
  - which neurons contribute to modulatory readouts
  - motor readout mapping

### 3.2 AgentState (dynamic)

- neuron state arrays:
  - `v` (N,), `a` (N,), `spike` (N,)
- synapse plasticity state:
  - eligibility traces `e` (E,)
  - optional additional traces

We strongly prefer struct-of-arrays for JAX efficiency.

---

## 4. Update kernel design

### 4.1 Step function

At each time step:

1. encode observation + internal state into input currents
2. propagate through sparse recurrent connections
3. update neuron dynamics
4. compute modulatory signals from designated neurons
5. update eligibility traces and weights (plasticity)
6. decode motor action from motor readout mapping

All of this must be expressed as a pure JAX function.

### 4.2 Sparse operations

We implement recurrent input as:

\[
I_i(t) = \sum_{(j\rightarrow i)\in E} w_{ij}(t)\, s_j(t)
\]

Using `segment_sum` style operations over `edge_index`.

---

## 5. Modulatory signals

In Strain A, modulators are emergent:

- the genome selects which units contribute to modulator readouts,
- modulator is a scalar (or small vector) computed as:
  \[
  m_k(t) = \tanh\left(\sum_{i\in \mathcal{M}_k} u_{k,i}\,s_i(t)\right)
  \]

where \(u_{k,i}\) are genome-produced weights.

Modulators gate plasticity and can also gate action selection if desired (but keep minimal initially).

---

## 6. Plasticity integration

We implement an eligibility trace per edge:

\[
e_{ij}(t+1)=\lambda_{ij} e_{ij}(t) + f(s_j(t), s_i(t))
\]

Weight update:

\[
w_{ij}(t+1) = w_{ij}(t) + \eta_{ij}\,m(t)\,e_{ij}(t)
\]

We must implement:
- weight clipping / stabilization policies (soft, minimal)
- optional sign projection for Dale’s principle in Strain B.

---

## 7. Strain hooks

### Strain B (hard constraints)

- Dale’s: project outgoing weights to fixed sign per neuron
- E/I ratio: enforce neuron classes and sign mapping
- wiring cost: enforced at development (not agent core)

### Strain C (soft)

- compute constraint metrics from weights/activity and pass to fitness
- agent core only logs required stats (e.g., mean firing rate)

### Strain D (adaptive)

- agent core unchanged
- evolution engine changes evaluator weights over time

---

## 8. Validation tests

- deterministic replay given seeds
- numerical stability under random inputs
- plasticity sanity:
  - weights change when modulator is active
  - weights do not drift uncontrollably under zero modulator
- performance baseline:
  - small direct-encoded networks solve L0 tasks

---

## 9. Deliverables

- a reference Agent Core implementation meeting the interface contract
- a test suite for stability and determinism
- minimal observability logs (spike rate, modulator stats, weight update magnitudes)
