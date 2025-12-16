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

Planned scale-up run (next; execute as written, no mid-run tuning):
- Same env as the pilot above (steps=256, bad_respawn=0, succ_bonus=50).
- ES30 with seeds 0..4.
- Conditions:
  - A0 no-plastic
  - A1 spike `eta=0.05, lambda=0.9`
  - A2 drive `eta=0.05, lambda=0.9, mod_drive_scale=1.0`
  - A3 event `eta=0.05, lambda=0.9`
- Held-out eval: 512 episodes, eval seeds {424242, 0}, baselines greedy/random on the same episode keys.

---

## 2025-12-16 — Stage 2 scale-up: A0 vs spike/drive/event (eta=0.05) on stronger-hazard L1.0+L1.1 (seeds 0..4)

Goal:
- Execute the planned 5-seed scale-up to determine whether the pilot patterns persist:
  - spike-modulated plasticity: does it still trade off success for hazard contact/survival?
  - drive/event modulators: do they still show near-zero plasticity usage, and are any apparent gains robust?

Training (logs):
- `runs/stage2_scans/2025-12-16_stage2_scale_train_steps256_badresp0_succ50_g30_p64_ep4_seeds0-4.txt`

Held-out evaluation (512 episodes; eval seeds 424242 and 0; logs + structured JSONL):
- `runs/stage2_scans/2025-12-16_stage2_scale_eval_steps256_badresp0_succ50_ep512.txt`
- `runs/stage2_scans/2025-12-16_stage2_scale_eval_steps256_badresp0_succ50_ep512.jsonl`

Aggregate results across seeds 0..4 (mean ± stdev; from JSONL):
- eval seed 424242:
  - A0 no-plastic: `mean_fitness=283.4750±4.6161`, `success_rate=0.7870±0.0495`, `mean_t_alive=243.28±6.53`, `mean_bad_arrivals=1.7551±0.4448`, `mean_integrity_min=0.5616±0.1111`
  - A1 spike (eta=0.05): `mean_fitness=274.5316±21.8355`, `success_rate=0.8586±0.0632`, `mean_t_alive=230.50±23.51`, `mean_bad_arrivals=2.3024±0.7355`, `mean_integrity_min=0.4248±0.1834`, `mean_abs_dw_mean=0.014302±0.018244`
  - A2 drive (eta=0.05, scale=1.0): `mean_fitness=286.8070±2.9211`, `success_rate=0.7476±0.0265`, `mean_t_alive=248.76±2.26`, `mean_bad_arrivals=1.3442±0.1374`, `mean_integrity_min=0.6640±0.0343`, `mean_abs_dw_mean=0.000005±0.000005`
  - A3 event (eta=0.05): `mean_fitness=289.0738±2.8994`, `success_rate=0.7714±0.0522`, `mean_t_alive=249.82±5.38`, `mean_bad_arrivals=1.4180±0.4252`, `mean_integrity_min=0.6457±0.1059`, `mean_abs_dw_mean=0.000002±0.000001`
- eval seed 0:
  - A0 no-plastic: `mean_fitness=282.0189±5.8863`, `success_rate=0.7790±0.0554`, `mean_t_alive=242.22±8.14`, `mean_bad_arrivals=1.7539±0.4251`, `mean_integrity_min=0.5621±0.1057`
  - A1 spike (eta=0.05): `mean_fitness=273.0981±21.3372`, `success_rate=0.8466±0.0733`, `mean_t_alive=229.68±23.43`, `mean_bad_arrivals=2.2961±0.6991`, `mean_integrity_min=0.4263±0.1744`, `mean_abs_dw_mean=0.014436±0.018349`
  - A2 drive (eta=0.05, scale=1.0): `mean_fitness=285.2109±2.5313`, `success_rate=0.7356±0.0179`, `mean_t_alive=247.76±2.59`, `mean_bad_arrivals=1.3852±0.1305`, `mean_integrity_min=0.6537±0.0326`, `mean_abs_dw_mean=0.000005±0.000005`
  - A3 event (eta=0.05): `mean_fitness=286.7557±2.5159`, `success_rate=0.7442±0.0673`, `mean_t_alive=248.86±5.62`, `mean_bad_arrivals=1.4730±0.4112`, `mean_integrity_min=0.6320±0.1024`, `mean_abs_dw_mean=0.000003±0.000001`

Interpretation (provisional; n=5 seeds, fixed ES30 budget):
- Spike-modulated plasticity shows the clearest “plasticity usage” signal (`mean_abs_dw_mean ~ 1e-2`) and increases success rate, but it also increases hazard contact and reduces survival time, resulting in lower mean fitness than the no-plastic baseline under this stronger-hazard L1.0+L1.1 setup.
- Drive/event modulators produce **extremely small** average weight updates (`mean_abs_dw_mean ~ 1e-6`) yet show higher mean fitness and improved hazard metrics vs A0 across both held-out eval seeds; this may indicate (a) rare but effective within-life updates that are diluted in the average, or (b) that these settings are effectively “almost fixed-weight” and the apparent gains reflect the changed genotype→phenotype mapping (still needs follow-up).

Next step (new prereg needed before changing axes):
- Decide whether Stage 2’s next objective is:
  A) “demonstrate plasticity helps” (then prioritize *eliciting non-trivial* drive/event plasticity via a `--mod-drive-scale` sweep and/or eligibility redesign), or
  B) “compare strains under fixed compute” (then scale A0 vs A2/A3 budgets/seeds and keep reporting `mean_abs_dw_mean` as a guardrail).
