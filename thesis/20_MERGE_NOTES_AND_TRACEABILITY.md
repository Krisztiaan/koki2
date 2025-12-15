# Merge notes and traceability

This document explains how the source plans were merged into the cohesive thesis collection.

---

## 1. Source documents

- `00_SOURCES/first_principles_evolution_framework.md`
- `00_SOURCES/meta_learning_framework.md`
- `00_SOURCES/bio_strain_comparative_framework.md`
- earlier rework drafts in `00_SOURCES/` (architecture, ladder, scaling)

---

## 2. Major consolidations

### 2.1 Inside-out alignment

- The inside-out framing from Buzsáki is treated as a *methodological constraint*:
  - action-first environments,
  - explicit spontaneous dynamics probes,
  - careful interpretation guidance.
- Canonical text: `03_THEORY_FOUNDATIONS_INSIDE_OUT_FIRST_PRINCIPLES.md`.

### 2.2 Minimal substrate (Strain A) as baseline

- The previous “hard realism” impulses are moved into **bio-strains** rather than baseline.
- Canonical text: `04_THEORY_MINIMAL_SUBSTRATE_AND_PLASTICITY.md` + `10_FRAMEWORK_BIO_STRAINS_AND_ABLATIONS.md`.

### 2.3 Genome + development as a compiler

- Genome specification from source documents is consolidated into:
  - CPPN/identity embeddings,
  - connection/weight/plasticity rule networks,
  - deterministic sparse compilation.
- Canonical text: `05_THEORY_GENOMIC_BOTTLENECK_AND_DEVELOPMENT.md` + `14_IMPLEMENTATION_GENOME_COMPILER_SPEC.md`.

### 2.4 Environment progression becomes a granular ladder

- The environment plan is restructured to avoid “jumping”:
  - each level introduces one major pressure,
  - each level has acceptance tests and hypotheses.
- Canonical text: `12_IMPLEMENTATION_ENVIRONMENT_LADDER_SPEC.md`.

### 2.5 Nursing and pruning become explicit addenda

- “Reward richer early life / larger reserve energy” concepts are formalized as nursing schedules.
- “Trim genetic dead ends” is formalized as MVT + multi-fidelity with novelty safeguards.
- Canonical text:
  - `08_ADDENDUM_DEVELOPMENTAL_NICHE_NURSING.md`
  - `09_ADDENDUM_VIABILITY_PRUNING_AND_MULTI_FIDELITY.md`.

### 2.6 Modularity for parallel work

- Architecture layering and interface contracts are elevated to a canonical design doc:
  - enables parallel development,
  - enforces reproducibility requirements.
- Canonical text: `11_IMPLEMENTATION_ARCHITECTURE_LAYERS_AND_INTERFACES.md`.

---

## 3. What was explicitly deferred

- Multi-agent social environments are deferred to future work.
- Canonical text: `19_FUTURE_WORK_AND_VALIDATION.md`.

---

## 4. Bibliography policy

- Only sources actually cited in canonical docs are required in `references.bib`.
- If additional citations are needed, add them by:
  1. adding citekeys in the relevant `.md`,
  2. adding BibTeX entries in `references.bib`,
  3. re-running rendering with Pandoc citeproc.

---

## 5. Next step

Use these canonical docs to drive an issue tracker:

- environments (ladder implementation)
- agent core
- genome compiler
- evolution engine + budgets
- observability/analysis
- performance benchmarks
