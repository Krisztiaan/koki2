# WORK (living) — incremental lab notebook

This log starts after checkpoint `stage1_l10_grid`.
- Archived snapshot: `WORK_stage1_l10_grid.md` (and `PLAN_stage1_l10_grid.md`)
- Repo: `https://github.com/Krisztiaan/koki2`

Policy:
- Log every meaningful change with: **Goal**, **Changes**, **Verification**, **Results** (only if actually run).
- Do not invent numbers; cite concrete run logs/JSONL paths for any quantitative statements.

---

## 2025-12-16 — Checkpoint: stage1_l10_grid

Checkpoint created:
- Archived the previous plan + lab notebook to `PLAN_stage1_l10_grid.md` and `WORK_stage1_l10_grid.md`.
- This checkpoint includes the Stage 1 L1.0 “effect-size grid” protocol + Colab notebooks and the latest local results (see `WORK_stage1_l10_grid.md` for commands and log paths).

Next:
- Proceed to Stage 2 plasticity comparisons on L1.0 + L1.1, using the protocol defined in `PLAN.md`.
- First action: pre-register the Stage 2 run matrix and budgets here before running.

---

## 2025-12-16 — Stage 2 prereg: consequence-aligned plasticity on L1.0 + L1.1 (stronger effect size)

Goal:
- Build on the Stage 2 checkpoint evidence in `WORK_stage1_l10_grid.md` that spike-modulated plasticity can improve held-out fitness in L1.0+L1.1 but with a hazard-contact regression.
- Test whether **consequence-aligned modulators** (drive/event) and/or milder plasticity strength can preserve performance gains **without** increasing hazard contact.

Primary hypothesis (operational):
- Relative to no-plastic, at fixed compute, plastic agents with consequence-aligned modulators will:
  - improve held-out `mean_fitness` and/or `success_rate`, **and**
  - not worsen (or improve) hazard-contact metrics (`mean_bad_arrivals`, `mean_integrity_min`),
  - while showing non-trivial plasticity usage (`mean_abs_dw_mean` not ~0).

Benchmark environment (fixed across conditions unless explicitly ablated):
- L1.0 deplete/respawn + L0.2 harmful sources + L1.1 intermittent gradient:
  - `--num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25`
  - `--deplete-sources --respawn-delay 4`
  - hazard persistence pressure: `--bad-source-respawn-delay 0`
  - keep `--bad-source-deplete-p 1.0` (avoid “camp on non-depleting hazard” confound)
  - `--grad-dropout-p 0.5`
  - longer horizon / effect size: `--steps 256`
- Shaping axis (tracked, not tuned post-hoc): default `--success-bonus 50` (ablation `0` is deferred to a later dedicated sweep).

Compute budget:
- Pilot sanity (local): ES30 `--generations 30 --pop-size 64 --episodes 4 --jit-es` with 3 seeds (0..2).
- Scale-up (Colab): same protocol with 5+ seeds (0..4) and/or ES100 once the pilot shows stable, non-trivial plasticity usage.

Conditions (pre-registered):
- A0) no-plastic (baseline): (no `--plast-enabled`)
- A1) plastic spike modulator: `--plast-enabled --modulator-kind spike`
- A2) plastic drive modulator: `--plast-enabled --modulator-kind drive --mod-drive-scale 1.0`
- A3) plastic event modulator: `--plast-enabled --modulator-kind event`
- Plasticity strength sweep (small, fixed grid; `--plast-lambda 0.9` held constant):
  - `--plast-eta ∈ {0.01, 0.05}`

Planned evaluation (held-out; required for interpretation):
- For each run dir: `koki2 eval-run --episodes 512 --seed {424242,0} --baseline-policy none`
- Baselines on the same eval seeds/episodes:
  - `koki2 baseline-l0 --policy greedy`
  - `koki2 baseline-l0 --policy random`
- Metrics to report per eval seed (mean ± stdev across ES seeds within each condition):
  - `mean_fitness`, `success_rate`, `mean_t_alive`
  - `mean_bad_arrivals`, `mean_integrity_min`
  - `mean_abs_dw_mean`

Decision rule after pilot:
- If plasticity usage is effectively zero for a condition (near-zero `mean_abs_dw_mean` across seeds), treat that condition as “effectively non-plastic” and adjust `plast-eta` upward *only in a separately logged follow-up*, not in-place.
- If consequence-aligned modulators reduce hazard regression relative to spike modulator, prioritize scaling those on Colab before exploring more axes.

---

## 2025-12-16 — Stage 2 pilot run: modulator kind × eta (seeds 0..2; ES30; steps=256; bad_respawn=0)

Goal:
- Execute the pre-registered pilot (above) to (a) sanity-check runtime on this machine and (b) see whether consequence-aligned modulators produce non-trivial plasticity usage and/or reduce the hazard-contact regression we saw previously under spike-modulated plasticity.

Training runs (logs):
- `runs/stage2_scans/2025-12-16_stage2_pilot_train_steps256_badresp0_succ50_g30_p64_ep4_seeds0-2.txt`

Held-out evaluation (512 episodes; eval seeds 424242 and 0; logs + structured JSONL):
- `runs/stage2_scans/2025-12-16_stage2_pilot_eval_steps256_badresp0_succ50_ep512.txt`
- `runs/stage2_scans/2025-12-16_stage2_pilot_eval_steps256_badresp0_succ50_ep512.jsonl`

Baseline (same env knobs; from eval log):
- eval seed 424242:
  - greedy: `mean_fitness=203.9268`, `success_rate=0.900`, `mean_t_alive=157.2`, `mean_bad_arrivals=3.4434`, `mean_integrity_min=0.1416`
  - random: `mean_fitness=277.9775`, `success_rate=0.445`, `mean_t_alive=255.4`, `mean_bad_arrivals=0.5918`, `mean_integrity_min=0.8521`
- eval seed 0:
  - greedy: `mean_fitness=203.1641`, `success_rate=0.877`, `mean_t_alive=157.7`, `mean_bad_arrivals=3.4141`, `mean_integrity_min=0.1494`
  - random: `mean_fitness=277.3730`, `success_rate=0.453`, `mean_t_alive=254.4`, `mean_bad_arrivals=0.5996`, `mean_integrity_min=0.8501`

Pilot aggregates (mean ± stdev across ES seeds 0..2; eval seed 424242; from JSONL):
- A0 no-plastic:
  - `mean_fitness=283.6289±5.0873`, `success_rate=0.7637±0.0422`
  - `mean_bad_arrivals=1.6302±0.4252`, `mean_integrity_min=0.5931±0.1057`
- A1 spike modulator:
  - `eta=0.01`: `mean_fitness=278.9310±17.8541`, `mean_bad_arrivals=1.9095±0.9380`, `mean_integrity_min=0.5233±0.2337`, `mean_abs_dw_mean=0.0032±0.0055`
  - `eta=0.05`: `mean_fitness=283.0765±10.5485`, `mean_bad_arrivals=1.9792±0.6420`, `mean_integrity_min=0.5052±0.1605`, `mean_abs_dw_mean=0.0153±0.0235`
- A2 drive modulator (`--mod-drive-scale 1.0`):
  - `eta=0.01`: `mean_fitness=289.2878±1.3511`, `mean_bad_arrivals=1.2214±0.0115`, `mean_integrity_min=0.6947±0.0029`, `mean_abs_dw_mean≈0` (very small in raw logs)
  - `eta=0.05`: `mean_fitness=288.8834±0.4894`, `mean_bad_arrivals=1.2839±0.0638`, `mean_integrity_min=0.6790±0.0159`, `mean_abs_dw_mean≈0` (very small in raw logs)
- A3 event modulator:
  - `eta=0.01`: `mean_fitness=287.0439±2.6463`, `mean_bad_arrivals=1.1680±0.3347`, `mean_integrity_min=0.7082±0.0835`, `mean_abs_dw_mean≈0` (very small in raw logs)
  - `eta=0.05`: `mean_fitness=288.9368±1.0776`, `mean_bad_arrivals=1.2428±0.0383`, `mean_integrity_min=0.6893±0.0096`, `mean_abs_dw_mean≈0` (very small in raw logs)

Quick interpretation (provisional; pilot n=3 seeds):
- Spike-modulated plasticity shows **non-trivial plasticity usage** (`mean_abs_dw_mean` noticeably > 0), but in this stronger-hazard L1.0+L1.1 setup it does **not** obviously improve held-out fitness over no-plastic and it appears to **worsen hazard contact** (higher bad arrivals, lower integrity minima).
- Drive/event modulators at `mod_drive_scale=1.0` appear to yield **negligible plasticity usage** (raw `mean_abs_dw_mean` on the order of ~1e-6 to 1e-4 in the eval log), so these runs may be effectively non-plastic; the apparent hazard improvements could therefore be search variance. This needs a larger-seed rerun before we treat it as evidence.

Next step (pre-registered decision rule application):
- Scale A0 vs A2/A3 to ≥5 seeds (and/or ES100) **before** changing axes.
- If we still see near-zero `mean_abs_dw_mean` for drive/event, treat “eliciting non-trivial plasticity under consequence modulation” as the next design problem (likely via `mod_drive_scale` sweeps and/or changing the eligibility signal), and log that as a new prereg entry.
