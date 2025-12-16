# PLAN (living) — koki2 thesis research implementation

This plan starts after checkpoint `stage1_l10_grid`.
- Archived snapshot: `PLAN_stage1_l10_grid.md`, `WORK_stage1_l10_grid.md`
- Repo: `https://github.com/Krisztiaan/koki2`

Guiding principles:
- Determinism-first; keep shapes static under `jax.jit`.
- Record every substantive change in `WORK.md` (with commands + log paths).
- Do not invent numbers; cite concrete files (logs/JSONL) for any claims.
- Keep Colab notebooks output-free; use **File → Save a copy in Drive** when running in Colab.

---

## Current stage gate

Stage 2 — Plasticity sanity and benefits (`thesis/18_EXPERIMENTS_AND_MILESTONES.md`).

Goal:
- Show plastic agents outperform fixed-weight agents on L1 tasks (noise/depletion/partial observability) while staying stable (no NaNs), with fair held-out evaluation and reporting “plasticity usage”.

---

## Stage 2 protocol (baseline)

Environment (L1.0 + L0.2 hazards + L1.1 intermittent gradient):
- `--num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25`
- `--deplete-sources --respawn-delay 4`
- Hazard persistence knob (amplify avoidance pressure): `--bad-source-respawn-delay 0`
  - Keep `--bad-source-deplete-p 1.0` to avoid the known “camp on non-depleting hazard” confound.
- Partial observability: `--grad-dropout-p 0.5`
- Horizon/effect size: `--steps 256`
- Shaping axis:
  - Default: `--success-bonus 50`
  - Ablation: `--success-bonus 0` (treated as an explicit experimental axis, not a post-hoc tweak)

Compute:
- Local sanity runs: ES30 `--generations 30 --pop-size 64 --episodes 4 --jit-es`
- Scale-up runs (Colab): ES100+ and/or more seeds once the protocol is stable

Evaluation (required for interpretation):
- Always evaluate saved `best_genome.npz` via `koki2 eval-run` on held-out episodes (>=512).
- Use at least two eval seeds (e.g., 424242 and 0) and report per-seed summaries.
- Always report:
  - `mean_fitness`, `success_rate`, `mean_t_alive`
  - `mean_bad_arrivals`, `mean_integrity_min`
  - plasticity usage diagnostics:
    - `mean_abs_dw_mean` (averaged across all alive steps)
    - `mean_abs_dw_on_event` + `event_step_frac` (to detect sparse, event-gated updates)
    - `mean_abs_modulator_mean` (to sanity-check modulator magnitude)
- Compare against baselines (`koki2 baseline-l0 --policy greedy/random`) on the same eval seeds/episodes.

---

## Work items (next)

Status (since checkpoint):
- Completed local pilot + scale-up comparisons (see `WORK.md` and `colab/stage2_*` notebooks).
- Added event-gated plasticity usage metrics (`mean_abs_dw_on_event`, `event_step_frac`) to disambiguate “near-zero mean” from “sparse learning”.
- Ran an ES100 replication sweep with `plast_eta=0` controls and an `override_plast_eta=0.0` eval probe; in the current stronger-hazard L1.0+L1.1 setup this did **not** produce a strong causal “learning is necessary” signal.

Next (Stage 2):
1. Create a hazard-persistent variant where within-episode consequence learning should matter more:
   - keep the current stronger-hazard setup, but set `--bad-source-deplete-p < 1.0` (bad sources persist after contact).
2. Run a small pilot matrix (few seeds) on this variant:
   - A0 no-plastic vs drive/event plastic at fixed compute, and evaluate each best genome with `koki2 eval-run --override-plast-eta 0.0` to test causality.
3. If plasticity still looks too weak/sparse (low `mean_abs_modulator_mean` and tiny `mean_abs_dw_on_event`), preregister a small `--mod-drive-scale` sweep (e.g. 1, 5, 10) and re-run the pilot.
4. Once a protocol shows a clear causal delta under the eta override, scale budgets (more seeds and/or ES generations) and keep Colab notebooks in sync.

---

## Verification checklist (before saying “done”)

- `uv run pytest`
- If new env dynamics / rollout logic are added: eager vs `jax.jit` parity is covered (add a focused test if not).
- For notebooks: outputs/execution counts stripped (pytest enforces this).
