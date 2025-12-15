# Future work and validation pathways

## 1. Deferred scope: multi-agent social cognition

The original concept includes a progression to social cognition. This thesis defers that extension because:

- multi-agent introduces confounds (credit assignment, non-stationarity),
- analysis complexity increases substantially,
- engineering cost increases (multi-agent simulation, communication channels).

However, the architecture is designed to allow a future “social ladder” by:

- generalizing environment to multiple agents,
- allowing agent–agent observations and interaction actions,
- extending fitness to inclusive fitness or cooperative outcomes.

A future social ladder would start with:
- resource cooperation tasks,
- competition for resources,
- communication signals with action consequences.

---

## 2. Cross-simulator validation

A major threat to credibility is simulator-specific artifacts. Therefore, a post-baseline validation plan is:

- export developed networks (weights and dynamics) into a simulator-independent format,
- reproduce key behaviors in a second simulator backend.

Candidate pathways:
- NeuroML-based model export for conventional simulators (NEURON/NEST/Brian)
- NIR export for neuromorphic stacks [@pedersen2024nir]

This is an extension once the core results are established in the JAX simulator.

---

## 3. NIR and neuromorphic deployment (optional)

NIR provides a common intermediate representation for spiking models and supports multiple simulators/hardware [@pedersen2024nir].

A deployment pathway:

1. train/evolve in JAX environment
2. export to NIR
3. import into a neuromorphic toolchain (Loihi, SpiNNaker, etc.)
4. confirm behavioral equivalence on a subset of tasks

This is not required for thesis completion but strengthens impact.

---

## 4. Richer neuron models (dendrites, compartments)

If the minimal substrate fails to express needed dynamics, the next extension is adding multi-compartment neurons:

- dendritic branches with separate time constants,
- soma–dendrite coupling.

This must be treated as a new strain (bio-strain extension), not folded into baseline.

---

## 5. Environment co-evolution (POET) and open-endedness

If fixed ladders saturate, adopt POET-style coevolution [@wang2019poet] as a post-thesis extension:

- evolve environment instances and transfer agents between them
- maintain novelty in environment space

This is high-impact but also high-complexity; defer until core emergence is demonstrated.

---

## 6. Deliverables (post-thesis roadmap)

- cross-simulator export prototypes (NeuroML or NIR)
- a minimal multi-agent environment and evaluation protocol
- additional bio-strains incorporating dendrites or richer synapses
