# 10 — Observability & reproducibility: experiment operations for thesis‑grade evidence

**Purpose:** Define what must be logged, how determinism is ensured, and how results can be audited and replayed.

---

## 1. Reproducibility requirements (hard constraints)

A result is admissible in the thesis only if:

1. The run can be reproduced from a stored manifest and code snapshot.
2. Key trajectories can be replayed exactly (or within a defined numerical tolerance).
3. All reported metrics can be traced to raw logs with a consistent schema.

---

## 2. Run manifest (required fields)

Every run produces a `run_manifest.json` containing:

### 2.1 Identity and provenance

- run_id (UUID)
- timestamp (UTC)
- hostname / device summary (GPU/TPU type)
- code version identifiers:
  - git commit hash (if applicable)
  - dirty flag (uncommitted changes)
  - dependency lock snapshot (pip/conda export)

### 2.2 Experiment configuration snapshot

- environment ladder config (levels enabled, variants)
- nursing schedule parameters
- pruning/budgeting configuration
- strain definition and annealing schedule
- genome family and compiler params
- evolution strategy hyperparameters
- evaluation seed protocol

### 2.3 Determinism contract

- base PRNG seed(s)
- key derivation method (fold_in scheme)
- numeric precision settings (float32/float64)
- backend settings (XLA flags if relevant)

---

## 3. Logging schema (performance-aware)

### 3.1 Metric scopes

- **Generation-level summary** (always logged):
  - best/median fitness
  - rung promotion counts
  - dead-end/pruning counts
  - compute used (sim steps)
  - throughput (steps/sec)

- **Episode-level logs** (for evaluated candidates; stored sparsely):
  - survival time, energy stats
  - environment level/variant id
  - agent diagnostics summary (spike rates etc.)

- **Trajectory logs** (sampled subset only):
  - full state/obs/action sequences for N candidates per generation
  - stored with exact replay seeds and genotype identifiers

### 3.2 Storage format

- generation summaries: JSONL or Parquet
- episode logs: Parquet (preferred for analysis)
- trajectories: compressed NumPy arrays or chunked storage (e.g., zarr/tensorstore) depending on scale

Rule: trajectory logging must not run inside the JIT hot loop. Instead:
- gather trajectories for selected candidates only
- run a separate evaluation pass for those candidates if needed

---

## 4. Replay protocol

A replay bundle for a candidate includes:

- genotype snapshot (or reference to genotype in checkpoint)
- compiler params
- environment params and variant id
- full seed chain (base seed + derivation indices)
- number of steps and termination reason
- optional: full compiled phenotype snapshot (for debugging caching issues)

Replay tool responsibilities:
- reconstruct phenotype deterministically
- rerun env+agent step loop
- verify match against stored trajectory hashes

---

## 5. Checkpointing strategy

Checkpoint at generation boundaries:

- evolution state
- population genotypes
- (optional) compiled phenotype cache metadata
- environment ladder progress state (current level gates)

Use atomic writes and include checksum hashes.

---

## 6. Analysis pipeline conventions

To keep analysis reproducible:

- analysis scripts read only from logged artifacts + manifests
- every figure/table is generated from a versioned analysis notebook/script
- derived datasets are cached with content hashes and linked to source run_ids

---

## 7. Minimal “audit package” for thesis submission

For each key experiment condition:
- run manifest
- final checkpoint
- summary metrics file
- 5–10 replayable trajectories demonstrating typical and edge-case behaviors

This allows external reviewers (and future you) to verify claims.

