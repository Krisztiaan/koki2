# 08 — Genome & developmental compiler: genotype→phenotype in a scalable JAX pipeline

**Purpose:** Specify how genotypes are represented, mutated, validated, and compiled into executable phenotypes, including direct encodings and genomic bottleneck encodings.

---

## 1. Thesis motivation: why indirect encoding matters

Directly evolving large neural networks is expensive because search dimensionality scales with parameter count. Indirect encodings propose that a compact genome can generate a large circuit with regularities and structure.

The genomic bottleneck work of Shuvaev et al. provides a concrete computational demonstration: a small generator network can produce large weight matrices with large compression ratios while retaining near-full performance [@shuvaev2024genomic]. This supports the thesis assumption that *evolutionary priors* and compact encodings can be a practical route to scalable neuroevolution, consistent with critiques of “pure learning” [@zador2019critique].

---

## 2. Genome families (staged integration)

### 2.1 G0: Direct parameter encoding (baseline)

**Representation:** a flat parameter vector (or pytree) representing weights/biases, optionally with structural masks.

Pros:
- simplest; fastest to implement
- ideal for early ladder stabilization (L0–L2)

Cons:
- scales poorly with network size
- mutations often destructive at scale

### 2.2 G1: Structured direct encoding (mask + weights)

Add structural genes:
- connectivity mask (sparsity pattern)
- neuron parameter genes (time constants, thresholds)

Pros:
- introduces structure without full indirect encoding complexity
- supports strain constraints (E/I partitioning) cleanly

### 2.3 G2: Genomic bottleneck encoding (primary indirect encoding)

Implement a “generator network” (hypernetwork) that maps neuron labels to weights:

- label each neuron with a binary (or categorical) identifier vector $\ell_i$
- generator $g_\theta$ produces connection weight:
  \[
  w_{ij} = g_\theta(\ell_i, \ell_j)
  \]
Compression ratio is controlled by generator size [@shuvaev2024genomic].

Pros:
- compact genotype; scalable phenotype
- regularization and potential transfer benefits [@shuvaev2024genomic]

Cons:
- requires careful implementation to keep compilation efficient
- introduces a second “network” (generator) to evolve

---

## 3. Compiler architecture (what “compile” must do)

### 3.1 Compiler contract (recap)

```text
compile(genome, compiler_params, key) -> (phenotype, compile_info)
```

Where `phenotype` contains:
- executable parameters for the agent backend
- structural masks and metadata

And `compile_info` includes:
- validity flags for pruning
- statistics for analysis (sparsity, E/I ratios, compression ratio)

### 3.2 Determinism requirement

Compilation must be deterministic given:
- genome
- compiler_params
- key (if compilation uses randomness)

**Rule:** If compilation is stochastic (e.g., sampling sparse connections), the key must be stored and replayable.

### 3.3 Caching

Compilation can be expensive (e.g., generating full weight matrices). Provide a cache keyed by:
- genome hash
- compiler_params hash
- backend version id

Cache is optional for correctness but important for throughput in iterative evaluation.

---

## 4. Mutation and recombination operators (per genome family)

### 4.1 Direct encoding mutation (G0)

- additive Gaussian noise on parameters
- per-parameter mutation rates
- optional structured noise per layer

### 4.2 Mask mutation (G1)

- bit flips in connectivity mask under sparsity budget
- rewiring operator: remove low-contribution connections, add random new ones
- constraints respected (e.g., no inhibitory→inhibitory if disallowed)

### 4.3 Genomic bottleneck mutation (G2)

Mutate generator network parameters:
- parameter-space Gaussian noise
- structured noise per generator layer
- optional mutation of neuron label assignment scheme (advanced; risky)

**Important:** because generator networks can be small, mutation steps must be tuned to avoid collapse into near-constant outputs.

---

## 5. Genomic bottleneck implementation details (G2)

### 5.1 Label scheme

Options:
1. **Fixed random binary labels** per neuron (seeded, deterministic).  
   Pros: simple; stable.  
   Cons: labels not optimized for modular structure.

2. **Structured labels** encoding neuron coordinates or type.  
   Pros: encourages regular connectivity; supports modularity.  
   Cons: may bias results; must be treated as an experimental assumption.

**Plan:** start with fixed random binary labels (deterministic by neuron index and global seed). Introduce structured labels only as a clearly documented extension.

### 5.2 Weight generation strategy

Generating all $w_{ij}$ for dense recurrent networks is $O(n^2)$; expensive at large $n$.

Options:
- **Dense generation** for small-to-medium networks (initial thesis scope).
- **Blockwise generation** with masks for sparse networks.
- **Low-rank factorization** produced by generator (advanced; optional).

**Thesis plan:** begin with networks sized so dense generation is feasible under JAX/XLA; introduce sparsity only if required by scaling.

### 5.3 Compression reporting

Compute:
\[
\text{compression ratio} = \frac{\text{phenotype parameter count}}{\text{genotype parameter count}}
\]
and record in `compile_info` for every candidate [@shuvaev2024genomic].

---

## 6. Developmental programs beyond genomic bottleneck (explicitly optional)

The earlier plan listed CPPNs and neural cellular automata. These may be valuable, but the reworked thesis treats them as **post-baseline extensions** because they introduce additional complexity.

A defensible thesis only requires:
- direct encoding baseline (G0/G1)
- one indirect encoding (G2 genomic bottleneck)

---

## 7. Verification and unit tests for the compiler

Required tests:

1. **Determinism:** same inputs produce identical phenotype.
2. **Validity flags:** invalid genomes are detected without running an environment.
3. **Constraint compliance (strain-specific):** if the phenotype is claimed to satisfy sign constraints, verify the constraint mechanically.
4. **Performance sanity:** on a toy environment, compilation does not systematically produce degenerate dynamics (all-zero outputs).

---

## 8. References

See `references.bib`.
