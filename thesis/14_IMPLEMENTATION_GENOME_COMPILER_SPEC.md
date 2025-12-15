# Genome compiler specification (genotype → phenotype)

## 1. Purpose

The genome compiler is responsible for translating a compact genome into a concrete `AgentParams`:

- connectivity (sparse edges),
- initial weights,
- plasticity rule coefficients,
- neuron parameters,
- role assignments (modulators, motor readouts).

It is the key scaling mechanism (genomic bottleneck) and must be deterministic and efficient.

---

## 2. Genome representation

A genome is a PyTree containing parameter sets for:

1. `cppn`: neuron identity generator
2. `conn_rule`: connection logits generator
3. `weight_rule`: initial weight generator
4. `plast_rule`: plasticity coefficient generator
5. `role_rule`: assigns neuron roles / readout weights

Optional:
- NEAT topology metadata [@stanley2002neat] if we evolve CPPN topologies rather than fixed MLPs.

---

## 3. Development inputs (DevConfig)

The compiler takes a `DevConfig` describing:

- number of neurons \(N\)
- neuron “positions” or coordinates (fixed lattice, random, learned)
- candidate edge set strategy:
  - fully connected (small N only)
  - k-nearest neighbors in coordinate space
  - sampled candidate pairs
- desired sparsity target (edge count \(E\))

To ensure static shapes for JAX, we typically fix:

- maximum edge count \(E_{\max}\)
- and represent absent edges with masks.

---

## 4. Development outputs (AgentParams)

Outputs must have deterministic, static shapes:

- `edge_index`: (E_max, 2)
- `edge_mask`: (E_max,)
- `w0`: (E_max,)
- `plast_params`: (E_max, P)
- neuron params arrays: (N, ...)
- role/readout arrays: (N, ...)

---

## 5. Candidate edge generation and masking

To avoid \(O(N^2)\) scaling:

- generate a candidate edge set using coordinates:
  - for each neuron, connect to k nearest candidates (directed)
- compute logits on candidates only
- sample/threshold to obtain a mask

This yields \(E_{\max} = N\cdot k\).

---

## 6. Determinism and PRNG discipline

All stochasticity must be explicit:

- edge sampling uses a specific PRNG stream
- any stochastic neuron parameter initialization uses streams

We define a reproducible PRNG splitting scheme:

- `rng_develop = fold_in(master_rng, genome_id)`
- then split into `rng_conn`, `rng_weights`, `rng_roles`, etc.

---

## 7. Mutation operators

We use mutation at the genome parameter level:

- Gaussian perturbation per parameter group
- group-specific sigma values
- occasional structured mutations:
  - toggle neuron role weights
  - adjust sparsity targets via thresholds

If NEAT-style topology evolution is used for CPPN/rule networks, include:
- add node, add connection, disable connection operations [@stanley2002neat].

---

## 8. Caching and compilation performance

Development is called for every genome evaluation. To reduce overhead:

- ensure development is JIT-friendly (static shapes)
- keep CPPN and rule nets small
- consider caching developed phenotypes for elites when genomes are reused (optional)

---

## 9. Validation tests

- determinism: same inputs produce identical outputs
- sanity: no NaNs, bounded weights/parameters
- sparsity: edge count matches target distribution
- sensitivity: small genome perturbations cause small phenotype perturbations (robustness metric)

---

## 10. Deliverables

- genome schema and config objects
- deterministic, tested development function producing `AgentParams`
- mutation operators compatible with selected evolution strategies
- benchmarking results for development cost vs rollout cost
