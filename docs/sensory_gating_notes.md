# Developmental sensory gating & resolution — notes + experiment ideas

This repo’s thesis plan already includes **nursing** (developmental niche schedules) and **pruning** (MVT + multi-fidelity) as compute-enabling mechanisms.

This note adds a specific nursing sub-mechanism: **developmental sensory gating and resolution schedules**.

## What we mean by “sensory gating”

Given developmental phase \(\phi\in[0,1]\), define sensor transforms that are applied as a function of age:

- **availability gating:** hide or attenuate some observation channels early; enable them later.
- **resolution gating:** quantize / coarsen (and later refine) some channels, e.g. a coarse gradient early, fine gradient later.
- **noise scheduling:** change observation SNR over age (either “easy early → hard later”, or “robustness early → clean later”).

Key constraint for JAX: keep tensor *shapes* constant and express gating as pure array ops.

## Why this might help (hypotheses)

This is a *curriculum* over the sensor interface that can be ablated independently of task difficulty:

1. **Faster convergence / fewer dead ends:** early behavior can be learned using a low-precision, low-bandwidth interface.
2. **Stabilization:** gradual increases in information content can reduce early policy churn and brittle attractors.
3. **Earlier pruning:** MVT in a “nursery” + low-resolution sensing can separate “hopeless” genomes from “late bloomers” more safely.
4. **Compute savings:** by improving pass rates in early rungs and reducing wasted full evaluations.

## Pointers from the literature (reading list)

For full citations, see `thesis/references.bib`.

Machine learning:
- Curriculum learning (Bengio et al., 2009).
- Progressive training schedules / coarse-to-fine strategies (Tan & Le, 2021; Karras et al., 2017).

Developmental robotics:
- Intrinsically motivated goal exploration / staged learning (Baranes & Oudeyer, 2013; Oudeyer, 2018).
- Simulating body + sensory development in an infant model (L{\'o}pez et al., 2025).

Developmental psychobiology:
- Developmental limitations of sensory input as an organizing constraint (Turkewitz, 1982; Turkewitz & Kenny, 1985).
- Atypical perinatal sensory stimulation and early perceptual development (Lickliter, 2000).

## How it maps onto this codebase (current implementation)

L0 chemotaxis (`src/koki2/envs/chemotaxis.py`) now supports:

- **gradient gain schedule:** `grad_gain(phi)` multiplies the gradient channels (availability gating).
- **gradient quantization schedule:** `grad_bins(phi)` controls quantization bins (resolution gating).

Config lives in `ChemotaxisEnvSpec` (`src/koki2/types.py`), with CLI flags in `src/koki2/cli.py`.

MVT (Minimal Viability Test) filtering can be enabled via `koki2 evo-l0 --mvt ...` and is logged per-generation as `mvt_pass_rate` in `runs/.../generations.jsonl`.

### Minimal demo runs

Hide gradient until late-life:

```bash
uv run koki2 evo-l0 --steps 128 \
  --grad-gain-min 0 --grad-gain-max 1 \
  --grad-gain-start-phi 0.5 --grad-gain-end-phi 1.0
```

Use a coarse gradient early:

```bash
uv run koki2 evo-l0 --steps 128 \
  --grad-bins-min 1 --grad-bins-max 20 \
  --grad-bins-start-phi 0.0 --grad-bins-end-phi 1.0
```

Enable MVT filtering (example: require some action diversity):

```bash
uv run koki2 evo-l0 --steps 128 --mvt \
  --mvt-steps 64 --mvt-min-alive-steps 32 --mvt-min-action-entropy 0.1
```

## Next experiments (thesis-aligned)

### A. L0/L1 ablations (nursing-only)

Compare:

1. no gating (baseline),
2. availability gating only,
3. resolution gating only,
4. both,

at fixed compute budgets.

Measure:

- success rate to reach source (L0),
- time-to-source,
- MVT pass rate under nursery settings,
- variance across seeds.

### B. Interaction with pruning (once MVT exists)

Test whether sensory gating reduces false negatives:

- Use “nursery” MVT (high buffers, low hazards),
- Evaluate promotions to adult settings,
- Track how often “MVT-fail” would have succeeded without pruning (bias diagnostic).

### C. Make it evolvable (future extension)

Two ways:

1. **Nursing factor (environment-side):** fixed schedules shared across population (current approach).
2. **Evolved sensor maturation (agent-side):** each genome outputs gating schedule parameters, and the *sensor transform* becomes part of phenotype.

The latter aligns with “developmental evolution in-agent” but needs careful controls (to avoid agents simply hiding hard inputs during evaluation).
