# Meta-learning and memory emergence (theory and test plan)

## 1. Why memory is central (and why we avoid hard-coding it)

The thesis aims to see whether memory-like mechanisms emerge from:

- recurrent dynamics,
- heterogeneous time constants,
- and local plasticity with modulatory gating,

under survival pressures.

We explicitly avoid installing a “memory module” (e.g., LSTM gates, DNC controller) in the baseline, because that would trivialize the claim. Instead, we treat memory as an emergent computational property:

- persistent state, attractors, and slow variables in recurrent dynamics,
- synaptic traces acting as fast weights,
- modulatory gating enabling context-dependent retention and forgetting.

---

## 2. Memory types to test (operational definitions)

We define memory categories in terms of measurable behavioral and dynamical signatures.

### 2.1 Working memory (short horizon)

Operational signature:
- task requires retaining a cue over delay \(d\)
- performance degrades with \(d\) without internal persistence

Dynamical signature:
- neural state remains informative about cue after input removal.

### 2.2 Episodic-like memory (event history)

Operational signature:
- agent benefits from remembering where resources were found in this episode (L4 sparse world)
- or from remembering which cue predicted hazard (L3 conditioning)

Dynamical signature:
- structured replay-like sequences are possible but not required.

### 2.3 Procedural habit / policy memory

Operational signature:
- stable action routines that are reused across episodes and robust to noise

Dynamical signature:
- recurrent attractors corresponding to behavioral modes.

---

## 3. Mechanisms expected to support memory (without hard-coding)

### 3.1 Recurrent state + heterogeneity

- longer \(\tau_m\), \(\tau_a\) create slow variables
- distributed time scales support multi-horizon integration

### 3.2 Eligibility traces and fast synaptic state

Eligibility traces can act as a short-term memory of recent co-activity, enabling delayed credit assignment and facilitating cue–outcome learning.

### 3.3 Modulatory gating for consolidation

Modulators allow learning to be:
- sparse and event-driven,
- sensitive to internal drive changes,
- and context-dependent (e.g., learn during surprise/threat).

---

## 4. Meta-learning framing

The genome evolves:

- *how* synapses change (plasticity coefficients),
- *when* they change (modulator gating),
- and *which parts* of the network are plastic (role mappings).

Within-life, these rules implement the agent’s “learning algorithm”.

This is meta-learning by evolution: selection acts on learning capability, not only on fixed policies.

A useful conceptual baseline is differentiable plasticity, which shows how plasticity parameters can be optimized to support rapid adaptation [@miconi2018differentiable].

---

## 5. Environment ladder links to memory requirements

Memory is made necessary by design, progressively:

- L1 (depletion + noise): short temporal integration improves chemotaxis
- L2 (multi-resource): state-dependent switching benefits from short-term memory
- L3 (cues + delay): explicit delay conditioning requires memory traces
- L4 (sparse resources + large world): longer horizon memory improves navigation and search

Each level provides a clearer signal of whether memory-like mechanisms are present.

---

## 6. Measurement and ablation plan

### 6.1 Behavioral tests

- delay-cue tasks: vary delay length and measure survival/avoidance
- hidden regime shifts (seasons): measure adaptation time
- partial observability: degrade sensors and measure robustness

### 6.2 Dynamical analyses

- probe internal state encoding:
  - train linear probes (offline) to predict past cue from neural state
- attractor analysis:
  - cluster neural states and compare to behavioral modes
- “lesion” tests:
  - freeze plasticity → measure performance drop
  - clamp modulators → measure performance drop

### 6.3 Control conditions

- fixed-weight agents (no plasticity)
- plasticity enabled but modulators forced to zero
- direct encoding vs bottleneck

These controls help establish that memory emerges from the proposed mechanisms.

---

## 7. Deliverables

- A standard suite of memory-relevant tasks (integrated into L1–L4).
- A measurement toolkit for memory-like dynamics.
- A set of ablations that isolate which substrate components are necessary.
