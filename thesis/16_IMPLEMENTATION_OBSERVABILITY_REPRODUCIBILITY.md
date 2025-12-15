# Observability, reproducibility, and analysis plan

## 1. Goals

The thesis requires evidence that:

- observed mechanisms are not artifacts,
- results are reproducible across seeds and hardware,
- comparative strain studies are fair.

Therefore observability and reproducibility are not “engineering nice-to-haves” but core scientific infrastructure.

---

## 2. Reproducibility design

### 2.1 Deterministic PRNG discipline

All randomness must be explicit and derived from:

- a master seed per run
- fold-in keys per:
  - generation
  - genome id
  - episode id
  - time step (only if necessary)

Avoid non-deterministic host operations during jitted execution.

### 2.2 Configuration manifests

Each run writes a manifest capturing:

- all configs (env, dev, agent, evo, nursing, pruning)
- git/file hashes
- device/hardware and software versions
- references to checkpoint files

### 2.3 Replay

A run is replayable if we can:

- regenerate the population genomes from checkpoint
- replay development and rollouts using logged seeds
- obtain identical fitness/metrics

Replay is a hard requirement for publication-grade results.

---

## 3. Logging schema (minimal and scalable)

### 3.1 Episode summaries (always logged)

Per episode:

- survival time
- resources acquired by type
- mean and variance of internal variables
- mean action magnitude, action entropy (if discrete)
- mean spike rate (if spiking)
- modulator summary stats (mean, variance, correlation with drive change)
- weight update magnitude summaries (mean abs Δw)

### 3.2 Generation summaries (always logged)

Per generation:

- best/median fitness
- MVT pass rate and failure reasons
- diversity metrics (genetic distance, behavior descriptor coverage)
- compute budget usage

### 3.3 Sampled rich traces (subsampled)

For a small sample of genomes/episodes:

- full time series: obs, action, internal state, modulators
- partial neural traces (subset of neurons) or summary latent projections
- plasticity trace statistics

Rich traces should be stored sparsely and compressed to control disk usage.

---

## 4. Emergence detection analyses

We define concrete tests for “emergent mechanisms”.

### 4.1 Learning and adaptation

- improvement from early to late life episodes
- adaptation after environment regime shift (L4 seasons)

### 4.2 Memory-like dynamics

- performance on tasks requiring integration over time
- internal state-space analysis:
  - persistence of neural state after input removal
  - presence of attractors or metastable states

### 4.3 Modulatory signal semantics

- correlation between modulators and drive reduction/prediction errors
- causal probing:
  - clamp modulator to zero and measure performance drop
  - randomize modulator and measure stability

### 4.4 Mode switching

- clustering of trajectory regimes (foraging vs avoidance)
- hysteresis tests:
  - same observation produces different actions depending on internal state/history

---

## 5. Cross-strain reporting

For strain comparisons we produce standardized reports:

- performance curves across environment levels
- learning metrics
- structural metrics (E/I partition, Dale consistency, wiring cost)
- compute efficiency metrics

Reports must include confidence intervals across seeds.

---

## 6. Deliverables

- A structured logging format (JSONL/Parquet recommended).
- A deterministic manifest and checkpoint system.
- An analysis toolkit producing:
  - dashboards
  - cross-strain comparison plots
  - emergence detection tests.
