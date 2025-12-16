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
  - `mean_abs_dw_mean` (to verify the run is meaningfully plastic)
- Compare against baselines (`koki2 baseline-l0 --policy greedy/random`) on the same eval seeds/episodes.

---

## Work items (next)

1. Pre-register the Stage 2 run matrix in `WORK.md` (before running anything):
   - no-plastic vs plastic
   - plastic variants: `--modulator-kind {spike,drive,event}` and a small `--plast-eta` grid (hold `--plast-lambda` fixed)
   - keep env + compute fixed except where explicitly ablated (e.g., success-bonus)
2. Run a small local sweep (few seeds) to sanity-check:
   - plasticity usage isn’t effectively zero (`mean_abs_dw_mean` not ~0)
   - rollouts remain stable (no NaNs; no tracer leaks)
3. Scale the sweep budgets on Colab (more generations and/or more seeds), keeping Colab notebooks in sync:
   - add a dedicated notebook under `colab/` for this Stage 2 protocol
4. Analyze results and decide next move:
   - if plastic improves fitness but worsens hazard contact, treat modulator kind + eta as the primary knobs to test first

---

## Verification checklist (before saying “done”)

- `uv run pytest`
- If new env dynamics / rollout logic are added: eager vs `jax.jit` parity is covered (add a focused test if not).
- For notebooks: outputs/execution counts stripped (pytest enforces this).
