# Thesis / research collection — cohesive plan (generated 2025-12-14)

This folder contains a **cohesive, thesis-aligned** document set for *Biologically-Constrained Neuroevolution / Evolvable Plastic Agents*.

It merges and extends:

- `00_SOURCES/first_principles_evolution_framework.md`
- `00_SOURCES/meta_learning_framework.md`
- `00_SOURCES/bio_strain_comparative_framework.md`
- earlier rework drafts in `00_SOURCES/` (architecture / environment ladder / scaling)

into a single consistent set of long-form `.md` documents and a shared `references.bib`.

---

## Recommended reading order

1. `01_EXECUTIVE_SUMMARY_AND_CONTRIBUTIONS.md`
2. `02_THESIS_OUTLINE_MASTER.md`
3. Theory core (Part I–III):
   - `03_THEORY_FOUNDATIONS_INSIDE_OUT_FIRST_PRINCIPLES.md`
   - `04_THEORY_MINIMAL_SUBSTRATE_AND_PLASTICITY.md`
   - `05_THEORY_GENOMIC_BOTTLENECK_AND_DEVELOPMENT.md`
   - `06_THEORY_ENVIRONMENTS_HOMEOSTASIS_AND_DRIVES.md`
   - `07_THEORY_BIASING_EVALUATORS_AND_OPEN_ENDEDNESS.md`
4. Addenda (development and evaluation efficiency):
   - `08_ADDENDUM_DEVELOPMENTAL_NICHE_NURSING.md`
   - `09_ADDENDUM_VIABILITY_PRUNING_AND_MULTI_FIDELITY.md`
5. Comparative framework:
   - `10_FRAMEWORK_BIO_STRAINS_AND_ABLATIONS.md`
6. Implementation planning:
   - `11_IMPLEMENTATION_ARCHITECTURE_LAYERS_AND_INTERFACES.md`
   - `12_IMPLEMENTATION_ENVIRONMENT_LADDER_SPEC.md`
   - `13_IMPLEMENTATION_AGENT_CORE_SPEC.md`
   - `14_IMPLEMENTATION_GENOME_COMPILER_SPEC.md`
   - `15_IMPLEMENTATION_EVOLUTION_ENGINE_SPEC.md`
   - `16_IMPLEMENTATION_OBSERVABILITY_REPRODUCIBILITY.md`
   - `17_IMPLEMENTATION_JAX_SCALING_GUIDE.md`
7. Execution plan and future work:
   - `18_EXPERIMENTS_AND_MILESTONES.md`
   - `19_FUTURE_WORK_AND_VALIDATION.md`

---

## Citation format and bibliography

These documents use Pandoc / citeproc style citation keys: `[@buzsaki2019brain]`.

The master bibliography is in:

- `references.bib`

---

## Rendering to PDF (optional)

Example:

```bash
cd thesis_cohesive
pandoc 02_THESIS_OUTLINE_MASTER.md \
  --toc --number-sections \
  --citeproc --bibliography=references.bib \
  -o thesis_outline.pdf
```

You can render any other chapter similarly.

---

## Conventions and assumptions

- **JAX-first**: the implementation plan assumes end-to-end simulation in JAX with `jit` + `vmap` + `lax.scan`.
- **Strain A is the baseline**: *no pre-specified E/I split, no Dale’s constraint, no fixed topology biases*. Biological constraints enter as explicit experimental *strains*.
- **Social/multi-agent is deferred**: treated as future work to reduce scope risk; the design keeps the door open.

---

## Source provenance

The original source plans are preserved (verbatim) in:

- `00_SOURCES/`

The integrated documents are the canonical ones for the thesis plan moving forward.
