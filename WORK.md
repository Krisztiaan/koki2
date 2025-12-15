# WORK (living) — incremental lab notebook

This file records what we changed, why, and what we verified as we iterate.

---

## 2025-12-14 — L0.2 foundation: multi-source chemotaxis

Decision: implement the smallest L0.2 extension that later stages can build on (L1 depletion, hazards, homeostasis), while keeping shapes static and tests simple.

Changes:
- Added **multi-source support** to L0 chemotaxis via `ChemotaxisEnvSpec.num_sources` (default `1`).
  - `src/koki2/types.py`: `ChemotaxisEnvSpec.num_sources`, `ChemotaxisEnvState.source_pos` now `int32[K,2]`.
  - `src/koki2/envs/chemotaxis.py`: observation gradient targets the **nearest** source; reach/arrival checks are “any source”.
- Registered `ChemotaxisEnvSpec` as a **static JAX pytree node** so `jax.jit(simulate_lifetime)(..., env_spec, ...)` remains valid even when `env_spec` controls shapes (e.g., `num_sources`).
- Exposed `--num-sources` on the CLI for quick experiments (`src/koki2/cli.py`).
- Updated sensory-gating unit tests to the new state shape (`tests/test_nursing_sensory_gating.py`).
- Added new unit tests for multi-source behavior (`tests/test_env_multisource.py`).

Semantics (current):
- `EnvLog.reached_source`: true if the agent is on **any** source.
- `EnvLog.energy_gained`: applies only when transitioning from “not on any source” → “on a source” (no gain while staying on a source).
- Observation stays 4D; gradient is to the nearest source (keeps the agent API stable).

Verification:
- Ran `uv run pytest` (9 tests) — all passing.

Next candidates:
- Implement L0.2 “positive + negative sources” (integrity loss) as a separate, ablatable variant.
- Add baseline evaluations (random vs greedy-gradient vs evolved) across multiple seeds for Stage 1 acceptance.

---

## 2025-12-14 — Baseline evaluation harness (Stage 1 acceptance)

Goal: enable an immediate, repeatable comparison of evolved policies vs simple baselines (greedy/random/stay) without needing notebooks.

Changes:
- Added baseline rollout functions in `src/koki2/sim/orchestrator.py`:
  - `simulate_lifetime_baseline_greedy`
  - `simulate_lifetime_baseline_random`
  - `simulate_lifetime_baseline_stay`
- Added CLI subcommand `koki2 baseline-l0` (`src/koki2/cli.py`) with `--policy {greedy,random,stay}` and standard env knobs (including `--num-sources`).
- Added determinism + JIT parity tests for baselines (`tests/test_baseline_determinism.py`).

Example usage:
```bash
uv run koki2 baseline-l0 --policy greedy --episodes 32 --steps 128 --num-sources 2
uv run koki2 baseline-l0 --policy random --episodes 32 --steps 128 --num-sources 2
```

Verification:
- Ran `uv run pytest` — all passing.

---

## 2025-12-14 — Quick empirical check (baselines + ES)

Goal: sanity-check Stage 1 progress with the new baseline harness and confirm what the objective is rewarding.

Baselines (128 episodes, `--steps 128`, default env; exact command/output in shell history):
- `num_sources=1`:
  - greedy: `mean_fitness=178.4883`, `success_rate=1.000`, `mean_energy_gained=0.0488`
  - random: `mean_fitness=136.4805`, `success_rate=0.164`, `mean_energy_gained=0.0277`
  - stay: `mean_fitness=129.1719`, `success_rate=0.023`, `mean_energy_gained=0.0000`
- `num_sources=3`:
  - greedy: `mean_fitness=178.4805`, `success_rate=1.000`, `mean_energy_gained=0.0480`
  - random: `mean_fitness=146.3867`, `success_rate=0.352`, `mean_energy_gained=0.0809`
  - stay: `mean_fitness=129.9531`, `success_rate=0.039`, `mean_energy_gained=0.0000`

ES (tiny run, 10 generations, pop 64, 4 episodes, seed 0):
- `runs/2025-12-14T2319390792920000_evo-l0_seed0`: best fitness reached `182.625` (gen 7).

Interpretation (provisional):
- L0 is already “solved” by the greedy baseline (100% success). ES can exceed greedy’s *mean* fitness by increasing repeated **arrival events** (since `energy_gained_total` is rewarded).
- This suggests the next environment step should reduce “energy farming by oscillation” and introduce temporal structure: **L1 depleting/respawning sources** is a natural follow-up.

---

## 2025-12-14 — Implement L1.0 depleting/respawning sources

Goal: prevent “arrival farming” on a single static source and introduce temporal structure for the environment ladder.

Changes:
- Added deplete/respawn controls:
  - `ChemotaxisEnvSpec.source_deplete`, `ChemotaxisEnvSpec.source_respawn_delay` (`src/koki2/types.py`).
  - `ChemotaxisEnvState.source_active`, `ChemotaxisEnvState.source_respawn_t` (`src/koki2/types.py`).
- Implemented deplete/respawn dynamics and “nearest active source” observation in `src/koki2/envs/chemotaxis.py`.
- Exposed CLI flags on both ES and baseline commands:
  - `--deplete-sources`, `--respawn-delay` (`src/koki2/cli.py`).
- Added unit test for countdown + deterministic respawn sampling (`tests/test_env_depletion.py`).

Verification:
- Ran `uv run pytest` (12 tests) — all passing.

Quick check (128 baseline episodes, `--steps 128`, `--deplete-sources --respawn-delay 4`, `num_sources=1`):
- greedy: `mean_fitness=180.3516`, `success_rate=1.000`, `mean_energy_gained=0.2352`
- random: `mean_fitness=136.2969`, `success_rate=0.164`, `mean_energy_gained=0.0094`

Tiny ES check (10 generations, pop 64, 4 episodes, seed 0, same env):
- `runs/2025-12-14T2328012132230000_evo-l0_seed0`: best fitness reached `179.0` (gen 8).

Interpretation (provisional):
- Depletion reduces reward-hacking via repeated arrivals at a single static source, and makes the greedy baseline collect multiple sources over time.
- ES at this small budget does not yet match the greedy baseline on the deplete+respawn variant (expected, since greedy has direct access to the gradient signal).

---

## 2025-12-15 — Repository conventions for future agents

Goal: encode our thesis-driven workflow (incremental stage gates, planning, and work logging) so future agents can continue consistently.

Changes:
- Added `AGENTS.md` documenting:
  - canonical sources of truth (`thesis/`, `PLAN.md`, `WORK.md`),
  - how to translate thesis concepts into incremental implementation steps,
  - required planning + logging conventions,
  - verification expectations and `uv` workflow.

Verification:
- No code changes; documentation-only.

---

## 2025-12-15 — L0.2 variant: positive + negative sources (integrity loss)

Goal: implement the L0.2 “positive + negative sources” rung (integrity loss) as an ablatable variant, while keeping observation tensor shapes fixed.

Changes:
- `src/koki2/types.py`:
  - `ChemotaxisEnvSpec.num_bad_sources`, `ChemotaxisEnvSpec.bad_source_integrity_loss`
  - `ChemotaxisEnvState.source_is_bad`
- `src/koki2/envs/chemotaxis.py`:
  - sample bad sources at init; keep source type fixed across respawns
  - only “good” sources count for `reached_source` and `energy_gained`
  - arriving at a bad source reduces integrity; terminate if `integrity <= 0`
- `src/koki2/cli.py`: expose `--num-bad-sources` and `--bad-source-integrity-loss` on both `evo-l0` and `baseline-l0`.
- Tests:
  - updated manual `ChemotaxisEnvState(...)` constructors to include `source_is_bad`
  - added `tests/test_env_bad_sources.py`
- Docs:
  - updated `PLAN.md` and `README.md` to mention the new L0.2/L1.0 toggles
  - tightened the sensory gating note’s reading list to point at `thesis/references.bib` (`docs/sensory_gating_notes.md`)

Verification:
- Ran `uv run pytest` (14 tests) — all passing.

Quick check (L0.2 positive/negative variant; all commands use default `--seed 0`):

```bash
uv run koki2 baseline-l0 --policy greedy --episodes 64 --steps 128 --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25
uv run koki2 baseline-l0 --policy random --episodes 64 --steps 128 --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25
uv run koki2 baseline-l0 --policy stay   --episodes 64 --steps 128 --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25
```

Outputs:
- greedy: `mean_fitness=154.8125`, `success_rate=0.531`, `mean_t_alive=128.0`, `mean_energy_gained=0.0250`
- random: `mean_fitness=133.5469`, `success_rate=0.281`, `mean_t_alive=118.9`, `mean_energy_gained=0.0563`
- stay: `mean_fitness=130.3438`, `success_rate=0.047`, `mean_t_alive=128.0`, `mean_energy_gained=0.0000`

Control (same setup, but no bad sources):

```bash
uv run koki2 baseline-l0 --policy greedy --episodes 64 --steps 128 --num-sources 4 --num-bad-sources 0
```

Output:
- greedy: `mean_fitness=178.4844`, `success_rate=1.000`, `mean_t_alive=128.0`, `mean_energy_gained=0.0484`

Tiny ES check (same env, seed 0):

```bash
uv run koki2 evo-l0 --seed 0 --generations 10 --pop-size 64 --episodes 4 --steps 128 --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25
```

Output:
- `runs/2025-12-15T0106069770530000_evo-l0_seed0`: `best_fitness=181.1250` (printed at end)

---

## 2025-12-15 — Theory note: cue informationality for L0.2 (good/bad sources)

Goal: clarify the research meaning of the L0.2 “positive + negative sources” rung by making the observation semantics explicit (what does the gradient reveal?), so later experiments can attribute results to memory/plasticity vs sensor informativeness.

Changes:
- `thesis/12_IMPLEMENTATION_ENVIRONMENT_LADDER_SPEC.md`: specified the default (valence-ambiguous) cue design and the key ablations (good-only cue; hazard cue channel as a nursing-scheduled knob).
- `thesis/06_THEORY_ENVIRONMENTS_HOMEOSTASIS_AND_DRIVES.md`: added a short “cue informationality (valence ambiguity)” section under observations to connect this choice back to inside-out commitments.

Verification:
- Documentation-only; no code paths changed.

---

## 2025-12-15 — L0.2 cue ablation: good-only gradient control

Goal: make the L0.2 positive/negative source setup explicitly ablatable by adding an informative-cue control where the gradient points only to good sources.

Changes:
- `src/koki2/types.py`: added `ChemotaxisEnvSpec.good_only_gradient`.
- `src/koki2/envs/chemotaxis.py`: `good_only_gradient=True` masks bad sources from the gradient target selection.
- `src/koki2/cli.py`: added `--good-only-gradient` for both `evo-l0` and `baseline-l0`.
- `tests/test_env_bad_sources.py`: added a unit test showing the cue difference between default vs control.
- Docs: updated `PLAN.md` and `README.md` to include the new control flag in the runbook/examples.

Verification:
- Ran `uv run pytest` (15 tests) — all passing.

---

## 2025-12-15 — Cue ablation sweep (L0.2 positive/negative sources)

Goal: measure how much the L0.2 positive/negative variant depends on cue informationality by comparing the default (valence-ambiguous) gradient vs the informative-cue control (`--good-only-gradient`) across multiple seeds.

Env setup (both conditions):
- `--num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25`
- `--steps 128`
- Baselines evaluated with `--episodes 128`
- ES runs: `--generations 10 --pop-size 64 --episodes 4`

Baselines (default cue; seeds 0/1/2):

```bash
for seed in 0 1 2; do
  for policy in greedy random stay; do
    uv run koki2 baseline-l0 --seed $seed --policy $policy --episodes 128 --steps 128 \
      --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25
  done
  echo "----"
done
```

Outputs:
- seed 0:
  - greedy: `mean_fitness=153.2344`, `success_rate=0.500`, `mean_t_alive=128.0`, `mean_energy_gained=0.0234`
  - random: `mean_fitness=135.7227`, `success_rate=0.297`, `mean_t_alive=120.3`, `mean_energy_gained=0.0613`
  - stay: `mean_fitness=129.9531`, `success_rate=0.039`, `mean_t_alive=128.0`, `mean_energy_gained=0.0000`
- seed 1:
  - greedy: `mean_fitness=150.4766`, `success_rate=0.445`, `mean_t_alive=128.0`, `mean_energy_gained=0.0211`
  - random: `mean_fitness=129.2148`, `success_rate=0.242`, `mean_t_alive=116.5`, `mean_energy_gained=0.0590`
  - stay: `mean_fitness=129.1719`, `success_rate=0.023`, `mean_t_alive=128.0`, `mean_energy_gained=0.0000`
- seed 2:
  - greedy: `mean_fitness=153.2383`, `success_rate=0.500`, `mean_t_alive=128.0`, `mean_energy_gained=0.0238`
  - random: `mean_fitness=129.4883`, `success_rate=0.289`, `mean_t_alive=114.3`, `mean_energy_gained=0.0707`
  - stay: `mean_fitness=129.5625`, `success_rate=0.031`, `mean_t_alive=128.0`, `mean_energy_gained=0.0000`

Baselines (informative cue control; `--good-only-gradient`; seeds 0/1/2):

```bash
for seed in 0 1 2; do
  for policy in greedy random stay; do
    uv run koki2 baseline-l0 --seed $seed --policy $policy --episodes 128 --steps 128 \
      --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25 --good-only-gradient
  done
  echo "----"
done
```

Outputs:
- seed 0:
  - greedy: `mean_fitness=178.4922`, `success_rate=1.000`, `mean_t_alive=128.0`, `mean_energy_gained=0.0492`
  - random: `mean_fitness=135.7227`, `success_rate=0.297`, `mean_t_alive=120.3`, `mean_energy_gained=0.0613`
  - stay: `mean_fitness=129.9531`, `success_rate=0.039`, `mean_t_alive=128.0`, `mean_energy_gained=0.0000`
- seed 1:
  - greedy: `mean_fitness=178.4922`, `success_rate=1.000`, `mean_t_alive=128.0`, `mean_energy_gained=0.0492`
  - random: `mean_fitness=129.2148`, `success_rate=0.242`, `mean_t_alive=116.5`, `mean_energy_gained=0.0590`
  - stay: `mean_fitness=129.1719`, `success_rate=0.023`, `mean_t_alive=128.0`, `mean_energy_gained=0.0000`
- seed 2:
  - greedy: `mean_fitness=178.4922`, `success_rate=1.000`, `mean_t_alive=128.0`, `mean_energy_gained=0.0492`
  - random: `mean_fitness=129.4883`, `success_rate=0.289`, `mean_t_alive=114.3`, `mean_energy_gained=0.0707`
  - stay: `mean_fitness=129.5625`, `success_rate=0.031`, `mean_t_alive=128.0`, `mean_energy_gained=0.0000`

Tiny ES (default cue; seeds 0/1/2):

```bash
for seed in 0 1 2; do
  uv run koki2 evo-l0 --seed $seed --generations 10 --pop-size 64 --episodes 4 --steps 128 \
    --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25
  echo "----"
done
```

Outputs:
- seed 0: `best_fitness=181.1250` (`runs/2025-12-15T0130367649710000_evo-l0_seed0`)
- seed 1: `best_fitness=181.8750` (`runs/2025-12-15T0130483509970000_evo-l0_seed1`)
- seed 2: `best_fitness=182.0000` (`runs/2025-12-15T0130597260120000_evo-l0_seed2`)

Tiny ES (informative cue control; `--good-only-gradient`; seeds 0/1/2):

```bash
for seed in 0 1 2; do
  uv run koki2 evo-l0 --seed $seed --generations 10 --pop-size 64 --episodes 4 --steps 128 \
    --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25 --good-only-gradient
  echo "----"
done
```

Outputs:
- seed 0: `best_fitness=181.0000` (`runs/2025-12-15T0131248851320000_evo-l0_seed0`)
- seed 1: `best_fitness=182.3750` (`runs/2025-12-15T0131362698960000_evo-l0_seed1`)
- seed 2: `best_fitness=181.1250` (`runs/2025-12-15T0131476363170000_evo-l0_seed2`)

Interpretation (provisional):
- Greedy baseline is highly sensitive to cue informationality: with valence ambiguity it often fails, while `--good-only-gradient` makes it reliably reach a good source in this setup.
- Tiny ES reaches best fitness above the greedy baseline in both cue modes, suggesting it can exploit the fitness components beyond “reach once and stop” (e.g., repeated arrivals / energy collection).

---

## 2025-12-15 — Cue ablation sweep under L1.0 deplete/respawn (temporal structure)

Goal: check whether L1.0 deplete/respawn reduces repeated-arrival reward-hacking and how it interacts with cue informationality in the positive/negative setup.

Env setup (both conditions):
- `--num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25`
- `--deplete-sources --respawn-delay 4`
- `--steps 128`

Baselines (default cue; seeds 0/1/2):

```bash
for seed in 0 1 2; do
  for policy in greedy random stay; do
    uv run koki2 baseline-l0 --seed $seed --policy $policy --episodes 128 --steps 128 \
      --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25 \
      --deplete-sources --respawn-delay 4
  done
  echo "----"
done
```

Outputs:
- seed 0:
  - greedy: `mean_fitness=128.2812`, `success_rate=0.898`, `mean_t_alive=81.7`, `mean_energy_gained=0.1656`
  - random: `mean_fitness=143.0312`, `success_rate=0.297`, `mean_t_alive=128.0`, `mean_energy_gained=0.0188`
  - stay: `mean_fitness=129.9531`, `success_rate=0.039`, `mean_t_alive=128.0`, `mean_energy_gained=0.0000`
- seed 1:
  - greedy: `mean_fitness=126.7266`, `success_rate=0.938`, `mean_t_alive=78.0`, `mean_energy_gained=0.1844`
  - random: `mean_fitness=140.2656`, `success_rate=0.242`, `mean_t_alive=128.0`, `mean_energy_gained=0.0156`
  - stay: `mean_fitness=129.1719`, `success_rate=0.023`, `mean_t_alive=128.0`, `mean_energy_gained=0.0000`
- seed 2:
  - greedy: `mean_fitness=129.7305`, `success_rate=0.898`, `mean_t_alive=83.0`, `mean_energy_gained=0.1770`
  - random: `mean_fitness=144.2266`, `success_rate=0.320`, `mean_t_alive=128.0`, `mean_energy_gained=0.0211`
  - stay: `mean_fitness=129.5625`, `success_rate=0.031`, `mean_t_alive=128.0`, `mean_energy_gained=0.0000`

Baselines (informative cue control; `--good-only-gradient`; seeds 0/1/2):

```bash
for seed in 0 1 2; do
  for policy in greedy random stay; do
    uv run koki2 baseline-l0 --seed $seed --policy $policy --episodes 128 --steps 128 \
      --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25 --good-only-gradient \
      --deplete-sources --respawn-delay 4
  done
  echo "----"
done
```

Outputs:
- seed 0:
  - greedy: `mean_fitness=167.0234`, `success_rate=1.000`, `mean_t_alive=113.6`, `mean_energy_gained=0.3383`
  - random: `mean_fitness=143.0312`, `success_rate=0.297`, `mean_t_alive=128.0`, `mean_energy_gained=0.0188`
  - stay: `mean_fitness=129.9531`, `success_rate=0.039`, `mean_t_alive=128.0`, `mean_energy_gained=0.0000`
- seed 1:
  - greedy: `mean_fitness=163.6914`, `success_rate=1.000`, `mean_t_alive=110.5`, `mean_energy_gained=0.3184`
  - random: `mean_fitness=140.2656`, `success_rate=0.242`, `mean_t_alive=128.0`, `mean_energy_gained=0.0156`
  - stay: `mean_fitness=129.1719`, `success_rate=0.023`, `mean_t_alive=128.0`, `mean_energy_gained=0.0000`
- seed 2:
  - greedy: `mean_fitness=166.8242`, `success_rate=1.000`, `mean_t_alive=113.6`, `mean_energy_gained=0.3246`
  - random: `mean_fitness=144.2266`, `success_rate=0.320`, `mean_t_alive=128.0`, `mean_energy_gained=0.0211`
  - stay: `mean_fitness=129.5625`, `success_rate=0.031`, `mean_t_alive=128.0`, `mean_energy_gained=0.0000`

Tiny ES (default cue; seeds 0/1/2):

```bash
for seed in 0 1 2; do
  uv run koki2 evo-l0 --seed $seed --generations 10 --pop-size 64 --episodes 4 --steps 128 \
    --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25 \
    --deplete-sources --respawn-delay 4
  echo "----"
done
```

Outputs:
- seed 0: `best_fitness=179.5000` (`runs/2025-12-15T0134300608910000_evo-l0_seed0`)
- seed 1: `best_fitness=179.5000` (`runs/2025-12-15T0134443221870000_evo-l0_seed1`)
- seed 2: `best_fitness=179.3750` (`runs/2025-12-15T0134590861400000_evo-l0_seed2`)

Tiny ES (informative cue control; `--good-only-gradient`; seeds 0/1/2):

```bash
for seed in 0 1 2; do
  uv run koki2 evo-l0 --seed $seed --generations 10 --pop-size 64 --episodes 4 --steps 128 \
    --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25 --good-only-gradient \
    --deplete-sources --respawn-delay 4
  echo "----"
done
```

Outputs:
- seed 0: `best_fitness=179.5000` (`runs/2025-12-15T0135217304720000_evo-l0_seed0`)
- seed 1: `best_fitness=179.5000` (`runs/2025-12-15T0135357936710000_evo-l0_seed1`)
- seed 2: `best_fitness=179.5000` (`runs/2025-12-15T0135499917790000_evo-l0_seed2`)

Interpretation (provisional):
- Deplete/respawn reduces the top-end fitness reached in the tiny ES runs vs the non-deplete variant, consistent with it limiting repeated-arrival “energy farming”.
- The informative-cue control helps the greedy baseline substantially even under deplete/respawn, but it does not fully prevent integrity loss (the agent can still step on bad sources while moving toward good ones).

---

## 2025-12-15 — L1.1 partial observability: intermittent gradient (`grad_dropout_p`)

Goal: implement the L1.1 “intermittent observations” pressure in a shape-stable way by randomly dropping the gradient channels with probability `grad_dropout_p`.

Changes:
- `src/koki2/types.py`: added `ChemotaxisEnvSpec.grad_dropout_p`.
- `src/koki2/envs/chemotaxis.py`: implemented dropout in `_observe()` by masking gradient channels with a Bernoulli draw (RNG split so dropout and noise are independent).
- `src/koki2/cli.py`: added `--grad-dropout-p` to both `evo-l0` and `baseline-l0`.
- `tests/test_env_grad_dropout.py`: added deterministic unit tests for `grad_dropout_p` (0, 0.5, 1.0).
- `PLAN.md`: updated status + runbook example for `--grad-dropout-p`.

Quick check (L1.0 deplete/respawn + positive/negative; seed 0; `--grad-dropout-p 0.5`):

```bash
uv run koki2 baseline-l0 --seed 0 --policy greedy --episodes 128 --steps 128 \
  --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25 \
  --deplete-sources --respawn-delay 4 \
  --grad-dropout-p 0.5

uv run koki2 baseline-l0 --seed 0 --policy greedy --episodes 128 --steps 128 \
  --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25 --good-only-gradient \
  --deplete-sources --respawn-delay 4 \
  --grad-dropout-p 0.5

uv run koki2 baseline-l0 --seed 0 --policy random --episodes 128 --steps 128 \
  --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25 \
  --deplete-sources --respawn-delay 4 \
  --grad-dropout-p 0.5
```

Outputs:
- greedy (default cue): `mean_fitness=160.7227`, `success_rate=0.859`, `mean_t_alive=116.5`, `mean_energy_gained=0.1246`, `mean_bad_arrivals=2.5703`, `mean_integrity_min=0.3574`
- greedy (`--good-only-gradient`): `mean_fitness=178.2188`, `success_rate=1.000`, `mean_t_alive=126.3`, `mean_energy_gained=0.1922`, `mean_bad_arrivals=1.5156`, `mean_integrity_min=0.6211`
- random: `mean_fitness=143.0312`, `success_rate=0.297`, `mean_t_alive=128.0`, `mean_energy_gained=0.0188`, `mean_bad_arrivals=0.3594`, `mean_integrity_min=0.9102`

Tiny ES (same env; seed 0; `--generations 10 --pop-size 64 --episodes 4`):

```bash
uv run koki2 evo-l0 --seed 0 --generations 10 --pop-size 64 --episodes 4 --steps 128 \
  --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25 \
  --deplete-sources --respawn-delay 4 \
  --grad-dropout-p 0.5

uv run koki2 evo-l0 --seed 0 --generations 10 --pop-size 64 --episodes 4 --steps 128 \
  --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25 --good-only-gradient \
  --deplete-sources --respawn-delay 4 \
  --grad-dropout-p 0.5
```

Outputs:
- default cue: `best_fitness=179.5000` (`runs/2025-12-15T0141434148910000_evo-l0_seed0`)
- `--good-only-gradient`: `best_fitness=179.5000` (`runs/2025-12-15T0142137021180000_evo-l0_seed0`)

Interpretation (provisional):
- In this specific setup, adding intermittency improved the greedy baseline (likely by reducing hazardous movement), which is consistent with the “sensory gating can stabilize search/behavior” hypothesis.

---

## 2025-12-15 — Reproducibility + interpretation audit (so far)

Goal: reduce the chance that our current conclusions are flukes or misinterpretations by (a) re-running previously reported commands with explicit seeds and (b) probing what ES “best fitness” corresponds to on larger evaluation sets than the 4-episode ES objective.

Re-run checks (baseline; explicit seed 0; 128 episodes; `--steps 128`):

```bash
uv run koki2 baseline-l0 --seed 0 --policy greedy --episodes 128 --steps 128 --num-sources 1
uv run koki2 baseline-l0 --seed 0 --policy random --episodes 128 --steps 128 --num-sources 1
uv run koki2 baseline-l0 --seed 0 --policy stay   --episodes 128 --steps 128 --num-sources 1

uv run koki2 baseline-l0 --seed 0 --policy greedy --episodes 128 --steps 128 --num-sources 3
uv run koki2 baseline-l0 --seed 0 --policy random --episodes 128 --steps 128 --num-sources 3
uv run koki2 baseline-l0 --seed 0 --policy stay   --episodes 128 --steps 128 --num-sources 3
```

Outputs:
- `num_sources=1` (matches earlier values exactly):
  - greedy: `mean_fitness=178.4883`, `success_rate=1.000`, `mean_energy_gained=0.0488`
  - random: `mean_fitness=136.4805`, `success_rate=0.164`, `mean_energy_gained=0.0277`
  - stay: `mean_fitness=129.1719`, `success_rate=0.023`, `mean_energy_gained=0.0000`
- `num_sources=3` (canonical for `seed=0`; earlier values were logged without an explicit command line in that entry):
  - greedy: `mean_fitness=178.4961`, `success_rate=1.000`, `mean_energy_gained=0.0496`
  - random: `mean_fitness=146.4258`, `success_rate=0.352`, `mean_energy_gained=0.0848`
  - stay: `mean_fitness=129.9531`, `success_rate=0.039`, `mean_energy_gained=0.0000`

Re-run check (L1.0 deplete/respawn; explicit seed 0; matches earlier values exactly):

```bash
uv run koki2 baseline-l0 --seed 0 --policy greedy --episodes 128 --steps 128 --deplete-sources --respawn-delay 4 --num-sources 1
uv run koki2 baseline-l0 --seed 0 --policy random --episodes 128 --steps 128 --deplete-sources --respawn-delay 4 --num-sources 1
uv run koki2 evo-l0 --seed 0 --generations 10 --pop-size 64 --episodes 4 --steps 128 --deplete-sources --respawn-delay 4 --num-sources 1
```

Best-genome probes (interpretation support; evaluate ES “best_genome.npz” on 64 fresh episodes and compare to greedy on the same episode keys):

```bash
uv run python - <<'PY'
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from koki2.genome.direct import DirectGenome, make_dev_config
from koki2.sim.orchestrator import simulate_lifetime, simulate_lifetime_baseline_greedy
from koki2.types import ChemotaxisEnvSpec, SimConfig

def eval_run(run_dir: str, episodes: int = 64):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / 'manifest.json').read_text())
    cfg = manifest['config']

    env_spec = ChemotaxisEnvSpec(**cfg['env_spec'])
    sim_cfg = SimConfig(**cfg['sim_cfg'])
    dev = cfg['dev_cfg']
    n = int(dev['n_neurons'])
    e = int(dev['edge_index_shape'][0])
    k = e // n
    dev_cfg = make_dev_config(
        n_neurons=n,
        obs_dim=int(dev['obs_dim']),
        num_actions=int(dev['num_actions']),
        k_edges_per_neuron=int(k),
        topology_seed=int(dev['topology_seed']),
        theta=float(dev['theta']),
        tau_m=float(dev['tau_m']),
        plast_enabled=bool(dev['plast_enabled']),
        plast_eta=float(dev['plast_eta']),
        plast_lambda=float(dev['plast_lambda']),
    )

    data = np.load(run_dir / 'best_genome.npz')
    genome = DirectGenome(
        obs_w=jnp.asarray(data['obs_w']),
        rec_w=jnp.asarray(data['rec_w']),
        motor_w=jnp.asarray(data['motor_w']),
        motor_b=jnp.asarray(data['motor_b']),
        mod_w=jnp.asarray(data['mod_w']),
    )

    keys = jax.random.split(jax.random.PRNGKey(0), episodes)
    t_idx = jnp.arange(env_spec.max_steps, dtype=jnp.int32)
    best = jax.vmap(lambda k: simulate_lifetime(genome, env_spec, dev_cfg, sim_cfg, k, t_idx))(keys)
    greedy = jax.vmap(lambda k: simulate_lifetime_baseline_greedy(env_spec, sim_cfg, k, t_idx))(keys)

    def summ(tag, out):
        mean_fit = float(jax.device_get(jnp.mean(out.fitness_scalar)))
        succ = float(jax.device_get(jnp.mean(out.success.astype(jnp.float32))))
        mean_alive = float(jax.device_get(jnp.mean(out.t_alive.astype(jnp.float32))))
        mean_gain = float(jax.device_get(jnp.mean(out.energy_gained_total)))
        mean_ent = float(jax.device_get(jnp.mean(out.action_entropy)))
        mean_mode = float(jax.device_get(jnp.mean(out.action_mode_frac)))
        print(f"{tag} episodes={episodes} mean_fitness={mean_fit:.4f} success_rate={succ:.3f} "
              f"mean_t_alive={mean_alive:.1f} mean_energy_gained={mean_gain:.4f} "
              f"mean_action_entropy={mean_ent:.4f} mean_action_mode_frac={mean_mode:.3f}")

    print(f"run_dir={run_dir}")
    summ("best_genome", best)
    summ("greedy", greedy)
    print()

eval_run("runs/2025-12-15T0130367649710000_evo-l0_seed0")
eval_run("runs/2025-12-15T0134300608910000_evo-l0_seed0")
PY
```

Observed outputs (from the above command):
- `runs/2025-12-15T0130367649710000_evo-l0_seed0` (non-deplete, positive/negative):
  - best_genome: `mean_fitness=161.0703`, `success_rate=0.703`, `mean_t_alive=124.7`, `mean_energy_gained=0.1180`
  - greedy: `mean_fitness=154.8125`, `success_rate=0.531`, `mean_t_alive=128.0`, `mean_energy_gained=0.0250`
- `runs/2025-12-15T0134300608910000_evo-l0_seed0` (deplete+respawn, positive/negative):
  - best_genome: `mean_fitness=163.7031`, `success_rate=0.734`, `mean_t_alive=126.4`, `mean_energy_gained=0.0562`
  - greedy: `mean_fitness=132.1094`, `success_rate=0.953`, `mean_t_alive=82.8`, `mean_energy_gained=0.1703`

Interpretation (provisional but better-grounded):
- In the non-deplete positive/negative setup, ES best_genome improves mean fitness vs greedy and does so with much higher `mean_energy_gained`, supporting the earlier suspicion that repeated arrivals/collection behavior is a major contributor to high fitness.
- Under deplete/respawn + hazards, greedy can collect a lot of energy but tends to die earlier (lower `mean_t_alive`), while the ES best_genome shifts toward longer survival at the cost of lower energy gained.

Verification:
- Ran `uv run pytest` (18 tests) — all passing.

---

## 2025-12-15 — Add “hazard contact” metrics (bad arrivals + integrity minima)

Goal: reduce narrative drift (e.g., “avoidance emerged”) by logging explicit outcome-relevant metrics: how often agents step on bad sources and how low integrity gets during an episode.

Changes:
- `src/koki2/types.py`:
  - `EnvLog.bad_arrivals`, `EnvLog.integrity_lost`
  - `FitnessSummary.integrity_min`, `FitnessSummary.bad_arrivals_total`, `FitnessSummary.integrity_lost_total`
- `src/koki2/envs/chemotaxis.py`: compute per-step `bad_arrivals` and `integrity_lost` from `arrived_bad`.
- `src/koki2/sim/orchestrator.py`: accumulate new totals and track `integrity_min` for agents and baselines.
- `src/koki2/cli.py`: `baseline-l0` now prints `mean_bad_arrivals`, `mean_integrity_lost`, and `mean_integrity_min`.
- Tests:
  - updated determinism/JIT parity to include the new metrics (`tests/test_stage0_determinism.py`, `tests/test_baseline_determinism.py`).

Verification:
- Ran `uv run pytest` (18 tests) — all passing.

Quick sweep (baselines; seeds 0/1/2; `--episodes 128 --steps 128`; positive/negative variant):

Commands:

```bash
for seed in 0 1 2; do
  for policy in greedy random; do
    uv run koki2 baseline-l0 --seed $seed --policy $policy --episodes 128 --steps 128 \
      --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25
    uv run koki2 baseline-l0 --seed $seed --policy $policy --episodes 128 --steps 128 \
      --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25 --good-only-gradient
  done
  echo "----"
done
```

Selected outputs (greedy only):
- seed 0:
  - default cue: `success_rate=0.500`, `mean_bad_arrivals=0.4844`, `mean_integrity_min=0.8789`
  - `--good-only-gradient`: `success_rate=1.000`, `mean_bad_arrivals=0.2969`, `mean_integrity_min=0.9258`
- seed 1:
  - default cue: `success_rate=0.445`, `mean_bad_arrivals=0.5625`, `mean_integrity_min=0.8594`
  - `--good-only-gradient`: `success_rate=1.000`, `mean_bad_arrivals=0.3594`, `mean_integrity_min=0.9102`
- seed 2:
  - default cue: `success_rate=0.500`, `mean_bad_arrivals=0.5078`, `mean_integrity_min=0.8730`
  - `--good-only-gradient`: `success_rate=1.000`, `mean_bad_arrivals=0.4141`, `mean_integrity_min=0.8965`

Quick sweep (baselines; seeds 0/1/2; `--episodes 128 --steps 128`; +L1.0 deplete/respawn):

Commands:

```bash
for seed in 0 1 2; do
  for policy in greedy random; do
    uv run koki2 baseline-l0 --seed $seed --policy $policy --episodes 128 --steps 128 \
      --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25 \
      --deplete-sources --respawn-delay 4
    uv run koki2 baseline-l0 --seed $seed --policy $policy --episodes 128 --steps 128 \
      --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25 --good-only-gradient \
      --deplete-sources --respawn-delay 4
  done
  echo "----"
done
```

Selected outputs (greedy only):
- seed 0:
  - default cue: `success_rate=0.898`, `mean_bad_arrivals=3.5312`, `mean_integrity_min=0.1211`
  - `--good-only-gradient`: `success_rate=1.000`, `mean_bad_arrivals=2.3750`, `mean_integrity_min=0.4082`
- seed 1:
  - default cue: `success_rate=0.938`, `mean_bad_arrivals=3.6328`, `mean_integrity_min=0.0957`
  - `--good-only-gradient`: `success_rate=1.000`, `mean_bad_arrivals=2.8125`, `mean_integrity_min=0.2969`
- seed 2:
  - default cue: `success_rate=0.898`, `mean_bad_arrivals=3.4766`, `mean_integrity_min=0.1328`
  - `--good-only-gradient`: `success_rate=1.000`, `mean_bad_arrivals=2.5547`, `mean_integrity_min=0.3633`

Interpretation (provisional):
- The informative-cue control (`--good-only-gradient`) reduces bad-source arrivals and increases integrity minima for the greedy baseline, as expected.
- Under deplete/respawn, the greedy baseline makes frequent bad arrivals (and reaches very low integrity) even when the gradient points only to good sources; this suggests “cue informativeness” alone does not guarantee safe trajectories once sources move.
- Random policy results are unchanged between cue modes (as expected, since it ignores the gradient), which is a good sanity check on the instrumentation.

ES “best genome” hazard metrics (64-episode probe; compares best_genome vs greedy on the same episode keys):

```bash
uv run python - <<'PY'
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from koki2.genome.direct import DirectGenome, make_dev_config
from koki2.sim.orchestrator import simulate_lifetime, simulate_lifetime_baseline_greedy
from koki2.types import ChemotaxisEnvSpec, SimConfig

def eval_run(run_dir: str, episodes: int = 64):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    cfg = manifest["config"]

    env_spec = ChemotaxisEnvSpec(**cfg["env_spec"])
    sim_cfg = SimConfig(**cfg["sim_cfg"])
    dev = cfg["dev_cfg"]
    n = int(dev["n_neurons"])
    e = int(dev["edge_index_shape"][0])
    k = e // n
    dev_cfg = make_dev_config(
        n_neurons=n,
        obs_dim=int(dev["obs_dim"]),
        num_actions=int(dev["num_actions"]),
        k_edges_per_neuron=int(k),
        topology_seed=int(dev["topology_seed"]),
        theta=float(dev["theta"]),
        tau_m=float(dev["tau_m"]),
        plast_enabled=bool(dev["plast_enabled"]),
        plast_eta=float(dev["plast_eta"]),
        plast_lambda=float(dev["plast_lambda"]),
    )

    data = np.load(run_dir / "best_genome.npz")
    genome = DirectGenome(
        obs_w=jnp.asarray(data["obs_w"]),
        rec_w=jnp.asarray(data["rec_w"]),
        motor_w=jnp.asarray(data["motor_w"]),
        motor_b=jnp.asarray(data["motor_b"]),
        mod_w=jnp.asarray(data["mod_w"]),
    )

    keys = jax.random.split(jax.random.PRNGKey(0), episodes)
    t_idx = jnp.arange(env_spec.max_steps, dtype=jnp.int32)
    best = jax.vmap(lambda k: simulate_lifetime(genome, env_spec, dev_cfg, sim_cfg, k, t_idx))(keys)
    greedy = jax.vmap(lambda k: simulate_lifetime_baseline_greedy(env_spec, sim_cfg, k, t_idx))(keys)

    def summ(tag, out):
        mean_fit = float(jax.device_get(jnp.mean(out.fitness_scalar)))
        succ = float(jax.device_get(jnp.mean(out.success.astype(jnp.float32))))
        mean_alive = float(jax.device_get(jnp.mean(out.t_alive.astype(jnp.float32))))
        mean_gain = float(jax.device_get(jnp.mean(out.energy_gained_total)))
        mean_bad = float(jax.device_get(jnp.mean(out.bad_arrivals_total)))
        mean_ilost = float(jax.device_get(jnp.mean(out.integrity_lost_total)))
        mean_imin = float(jax.device_get(jnp.mean(out.integrity_min)))
        mean_ent = float(jax.device_get(jnp.mean(out.action_entropy)))
        mean_mode = float(jax.device_get(jnp.mean(out.action_mode_frac)))
        print(
            f"{tag} episodes={episodes} mean_fitness={mean_fit:.4f} success_rate={succ:.3f} mean_t_alive={mean_alive:.1f} "
            f"mean_energy_gained={mean_gain:.4f} mean_bad_arrivals={mean_bad:.4f} mean_integrity_lost={mean_ilost:.4f} "
            f"mean_integrity_min={mean_imin:.4f} mean_action_entropy={mean_ent:.4f} mean_action_mode_frac={mean_mode:.3f}"
        )

    print(f"run_dir={run_dir}")
    summ("best_genome", best)
    summ("greedy", greedy)
    print()

eval_run("runs/2025-12-15T0130367649710000_evo-l0_seed0")
eval_run("runs/2025-12-15T0134300608910000_evo-l0_seed0")
PY
```

Observed outputs:
- `runs/2025-12-15T0130367649710000_evo-l0_seed0` (non-deplete, positive/negative):
  - best_genome: `mean_bad_arrivals=1.2969`, `mean_integrity_min=0.6758`, `mean_energy_gained=0.1180`
  - greedy: `mean_bad_arrivals=0.4688`, `mean_integrity_min=0.8828`, `mean_energy_gained=0.0250`
- `runs/2025-12-15T0134300608910000_evo-l0_seed0` (deplete+respawn, positive/negative):
  - best_genome: `mean_bad_arrivals=1.1562`, `mean_integrity_min=0.7109`, `mean_energy_gained=0.0562`
  - greedy: `mean_bad_arrivals=3.5625`, `mean_integrity_min=0.1172`, `mean_energy_gained=0.1703`

Interpretation (provisional):
- In the non-deplete setup, ES is not “avoiding” bad sources on average; it takes more bad arrivals and reaches lower integrity minima than greedy while gaining more energy. This strongly suggests the improved fitness there is not evidence of integrity-driven avoidance.
- Under deplete/respawn, ES reduces bad-source contact compared to greedy and maintains much higher integrity minima, consistent with a more survival-weighted strategy.

---

## 2025-12-15 — Checkpoint: what we can claim so far (and what we can’t)

Goal: mark a “stop and summarize” checkpoint before moving toward Stage 2, tightening our hypotheses to avoid narrative drift and identifying the minimum replication + tooling needed for reliable progress.

What we can claim (data-backed, within the limits of current sweeps/probes):
- **Cue informativeness is a dominant confound / axis** in L0.2 (+bad sources). For greedy across seeds 0/1/2 with `--episodes 128 --steps 128 --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25`, default cue success is ~0.445–0.500, while `--good-only-gradient` makes greedy `success_rate=1.000` in all three seeds. Random is unchanged between cue modes (sanity check that this flag only affects gradient-using behavior).
- **“Reach” is not “safe” once temporal structure exists.** Under L1.0 deplete/respawn (+bad sources), greedy often reaches good sources (`success_rate≈0.898–0.938`) but dies early (`mean_t_alive≈78–83` in the logged 128-step, 128-episode sweeps).
- **Even an informative “good-only” cue does not guarantee safe trajectories** under deplete/respawn: greedy still makes frequent bad-source contacts (high `mean_bad_arrivals`) and reaches low integrity minima (very low `mean_integrity_min`), so cue informativeness alone doesn’t solve hazard contact once sources move.
- **ES “best_fitness” can mislead about the behavioral story** because ES optimizes mean fitness over 4 episodes; evaluating `best_genome.npz` on 64 episodes is a better interpretation probe:
  - non-deplete (+bad sources, seed 0 run): `best_genome` shows **more bad-source contact** and **lower integrity minima** than greedy while gaining more energy (`mean_bad_arrivals=1.2969`, `mean_integrity_min=0.6758` vs greedy `0.4688`, `0.8828`) → not evidence of avoidance.
  - deplete/respawn (+bad sources, seed 0 run): `best_genome` shows **less bad-source contact** and **higher integrity minima** than greedy (`mean_bad_arrivals=1.1562`, `mean_integrity_min=0.7109` vs greedy `3.5625`, `0.1172`) → consistent with a survival-weighted shift under temporal structure (still only probed on one seed so far).

What we cannot claim yet (needs replication / better protocols):
- “Avoidance emerged” as a general phenomenon, or that intermittency (`--grad-dropout-p`) reliably improves outcomes: the intermittency observation is currently a single-seed quick check, and the ES hazard probe is seed 0 only.
- Any statistically stable effect sizes across seeds/held-out episodes: we have multiple-seed baselines for the cue ablation and deplete/respawn sweeps, but not yet the same standard for intermittency and best-genome probes.

Hypothesis adjustments (to align with evidence so far):
- Treat “avoidance” as a **measured outcome** (via `bad_arrivals_total`, `integrity_min`) rather than an inferred narrative from fitness alone.
- Expect temporal structure (L1.0+) to be a key pressure for survival-weighted strategies; in L0.2 non-deplete, fitness gains can plausibly come from energy harvesting patterns even with increased hazard contact.

Next steps to solidify before Stage 2 claims:
- Add a standardized “evaluate a saved run directory on N episodes” tool (CLI-level), so interpretation probes are not ad-hoc scripts.
- Replicate (multi-seed) the intermittency effect and the best-genome hazard-metric probe on additional seeds / run dirs.

---

## 2025-12-15 — Replication: intermittency (multi-seed) + best-genome probes (multi-seed)

Goal: reduce fluke risk by replicating the two weakest-current-evidence items:
- intermittency (`--grad-dropout-p`) effect beyond one seed, and
- ES best-genome interpretation probes beyond seed 0.

### A) Intermittency replication (greedy baseline; seeds 1/2)

Env: L1.0 deplete/respawn + L0.2 positive/negative.
- `--episodes 128 --steps 128`
- `--num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25`
- `--deplete-sources --respawn-delay 4`
- `--grad-dropout-p 0.5`

Commands:
```bash
for seed in 1 2; do
  uv run koki2 baseline-l0 --seed $seed --policy greedy --episodes 128 --steps 128 \
    --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25 \
    --deplete-sources --respawn-delay 4 \
    --grad-dropout-p 0.5

  uv run koki2 baseline-l0 --seed $seed --policy greedy --episodes 128 --steps 128 \
    --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25 --good-only-gradient \
    --deplete-sources --respawn-delay 4 \
    --grad-dropout-p 0.5
done
```

Observed outputs:
- seed 1:
  - default cue: `mean_fitness=161.3477`, `success_rate=0.922`, `mean_t_alive=113.8`, `mean_bad_arrivals=2.7422`, `mean_integrity_min=0.3145`
  - `--good-only-gradient`: `mean_fitness=176.6641`, `success_rate=1.000`, `mean_t_alive=124.6`, `mean_bad_arrivals=1.6094`, `mean_integrity_min=0.5977`
- seed 2:
  - default cue: `mean_fitness=156.0625`, `success_rate=0.891`, `mean_t_alive=110.2`, `mean_bad_arrivals=2.7031`, `mean_integrity_min=0.3262`
  - `--good-only-gradient`: `mean_fitness=176.7227`, `success_rate=1.000`, `mean_t_alive=124.9`, `mean_bad_arrivals=1.5938`, `mean_integrity_min=0.6016`

Interpretation (still provisional, but stronger than single-seed):
- Compared to the no-dropout deplete/respawn sweeps earlier in this file (same env, seeds 0/1/2), `--grad-dropout-p 0.5` materially increases `mean_t_alive` and improves the hazard metrics for greedy in all tested seeds so far (0/1/2).

### B) Best-genome probes via `koki2 eval-run` (seeds 0/1/2)

Protocol:
- Evaluate `best_genome.npz` on 64 episodes (fixed eval PRNG seed 0).
- Evaluate greedy baseline on the same episode keys for comparison.

Commands (non-deplete, positive/negative; seeds 0/1/2):
```bash
for run_dir in \
  runs/2025-12-15T0130367649710000_evo-l0_seed0 \
  runs/2025-12-15T0130483509970000_evo-l0_seed1 \
  runs/2025-12-15T0130597260120000_evo-l0_seed2; do
  uv run koki2 eval-run --run-dir "$run_dir" --episodes 64 --seed 0 --baseline-policy greedy
done
```

Observed outputs (selected):
- non-deplete (+bad sources):
  - seed 0: best genome `mean_fitness=161.0703`, greedy `154.8125` (but best has higher `mean_bad_arrivals=1.2969` vs `0.4688` and lower `mean_integrity_min=0.6758` vs `0.8828`)
  - seed 1: best genome `mean_fitness=149.0781`, greedy `154.8125` (best has higher `mean_bad_arrivals=1.9844` vs `0.4688` and lower `mean_integrity_min=0.5039` vs `0.8828`)
  - seed 2: best genome `mean_fitness=149.4844`, greedy `154.8125` (best has higher `mean_bad_arrivals=0.9531` vs `0.4688` and lower `mean_integrity_min=0.7617` vs `0.8828`)

Commands (deplete/respawn, positive/negative; seeds 0/1/2):
```bash
for run_dir in \
  runs/2025-12-15T0134300608910000_evo-l0_seed0 \
  runs/2025-12-15T0134443221870000_evo-l0_seed1 \
  runs/2025-12-15T0134590861400000_evo-l0_seed2; do
  uv run koki2 eval-run --run-dir "$run_dir" --episodes 64 --seed 0 --baseline-policy greedy
done
```

Observed outputs (selected):
- deplete/respawn (+bad sources):
  - seed 0: best genome `mean_fitness=163.7031`, greedy `132.1094` (best has lower `mean_bad_arrivals=1.1562` vs `3.5625` and higher `mean_integrity_min=0.7109` vs `0.1172`)
  - seed 1: best genome `mean_fitness=164.6094`, greedy `132.1094` (best has lower `mean_bad_arrivals=1.3125` vs `3.5625` and higher `mean_integrity_min=0.6719` vs `0.1172`)
  - seed 2: best genome `mean_fitness=164.7891`, greedy `132.1094` (best has lower `mean_bad_arrivals=1.2500` vs `3.5625` and higher `mean_integrity_min=0.6875` vs `0.1172`)

Interpretation (provisional but now multi-seed):
- In the non-deplete (+bad sources) variant, ES best-genome behavior does not consistently outperform greedy on mean fitness at this tiny budget (10 gens × pop 64 × 4 episodes), and it consistently shows worse hazard contact metrics than greedy under this evaluation protocol.
- Under deplete/respawn (+bad sources), ES best-genome consistently outperforms greedy on mean fitness and shows substantially improved hazard contact metrics (lower bad arrivals, higher integrity minima), now replicated across seeds 0/1/2.

---

## 2025-12-15 — Tooling: `koki2 eval-run` (standard best-genome evaluation)

Goal: make “best genome probes” a first-class, repeatable protocol so Stage 1/2 interpretation doesn’t depend on ad-hoc scripts and ES `best_fitness` (4-episode objective).

Changes:
- `src/koki2/cli.py`: added `koki2 eval-run --run-dir ...` to evaluate `best_genome.npz` on `--episodes N` and optionally compare to a baseline policy on the same episode keys.
- `tests/test_eval_run_cli.py`: added a small smoke test for `eval-run` using a temporary run directory.
- Docs: updated `PLAN.md` runbook and `README.md` quickstart with the recommended eval command.

Verification:
- Ran `uv run pytest` (19 tests) — all passing.

---

## 2025-12-15 — Stage 2 prep: plasticity knobs + minimal unit tests

Goal: take the smallest next idiomatic step toward Stage 2 by (a) making plasticity actually runnable from the CLI/ES loop and (b) adding a minimal correctness guardrail for plastic weight updates.

Changes:
- `src/koki2/cli.py`: added `--plast-enabled`, `--plast-eta`, `--plast-lambda` to `koki2 evo-l0` so we can run plastic vs non-plastic ES under identical env configs.
- `src/koki2/agent/snn.py`: `AgentLog.mean_abs_dw` now reports **applied** plastic changes (0 when plasticity is disabled) to avoid confusion in non-plastic runs.
- `tests/test_agent_plasticity.py`: added focused tests that plasticity gating behaves as intended (disabled ⇒ no weight/trace updates; enabled ⇒ bounded weight update in a constructed case).
- `PLAN.md`: updated Stage 2 dev tasks to reflect “expose knobs + compare” rather than re-implementing the mechanism.

Verification:
- Ran `uv run pytest` (21 tests) — all passing.

---

## 2025-12-15 — Stage 2 (first pass): plastic vs non-plastic on L1.0+L1.1 benchmark

Goal: run the smallest replicated comparison of plastic vs non-plastic ES on a task where temporal integration should matter (L1.0 deplete/respawn + L1.1 intermittent sensing), and evaluate saved genomes on held-out episodes using `koki2 eval-run`.

Benchmark env:
- `--num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25`
- `--deplete-sources --respawn-delay 4`
- `--grad-dropout-p 0.5`
- `--steps 128`

### A) Baselines (seed 0 context)

Commands:
```bash
for cue in default good_only; do
  for policy in greedy random stay; do
    if [ "$cue" = "good_only" ]; then
      uv run koki2 baseline-l0 --seed 0 --policy $policy --episodes 128 --steps 128 \
        --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25 --good-only-gradient \
        --deplete-sources --respawn-delay 4 \
        --grad-dropout-p 0.5
    else
      uv run koki2 baseline-l0 --seed 0 --policy $policy --episodes 128 --steps 128 \
        --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25 \
        --deplete-sources --respawn-delay 4 \
        --grad-dropout-p 0.5
    fi
  done
done
```

Observed outputs:
- default cue:
  - greedy: `mean_fitness=160.7227`, `success_rate=0.859`, `mean_t_alive=116.5`, `mean_energy_gained=0.1246`, `mean_bad_arrivals=2.5703`, `mean_integrity_min=0.3574`
  - random: `mean_fitness=143.0312`, `success_rate=0.297`, `mean_t_alive=128.0`, `mean_energy_gained=0.0188`, `mean_bad_arrivals=0.3594`, `mean_integrity_min=0.9102`
  - stay: `mean_fitness=129.9531`, `success_rate=0.039`, `mean_t_alive=128.0`, `mean_energy_gained=0.0000`, `mean_bad_arrivals=0.0000`, `mean_integrity_min=1.0000`
- `--good-only-gradient`:
  - greedy: `mean_fitness=178.2188`, `success_rate=1.000`, `mean_t_alive=126.3`, `mean_energy_gained=0.1922`, `mean_bad_arrivals=1.5156`, `mean_integrity_min=0.6211`
  - random: unchanged vs default cue (as expected)
  - stay: unchanged vs default cue (as expected)

### B) ES runs (10 generations; seeds 0/1/2)

Compute budget:
- `--generations 10 --pop-size 64 --episodes 4 --steps 128`

Commands (non-plastic):
```bash
for seed in 0 1 2; do
  uv run koki2 evo-l0 --seed $seed --out-dir runs/2025-12-15_stage2_es_noplast_seed${seed} \
    --generations 10 --pop-size 64 --episodes 4 --steps 128 \
    --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25 \
    --deplete-sources --respawn-delay 4 \
    --grad-dropout-p 0.5
done
```

Observed `best_fitness`:
- seed 0: `best_fitness=179.5000` (`runs/2025-12-15_stage2_es_noplast_seed0`)
- seed 1: `best_fitness=179.5000` (`runs/2025-12-15_stage2_es_noplast_seed1`)
- seed 2: `best_fitness=179.5000` (`runs/2025-12-15_stage2_es_noplast_seed2`)

Commands (plastic; `--plast-enabled --plast-eta 0.05 --plast-lambda 0.9`):
```bash
for seed in 0 1 2; do
  uv run koki2 evo-l0 --seed $seed --out-dir runs/2025-12-15_stage2_es_plast_eta0.05_seed${seed} \
    --generations 10 --pop-size 64 --episodes 4 --steps 128 \
    --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25 \
    --deplete-sources --respawn-delay 4 \
    --grad-dropout-p 0.5 \
    --plast-enabled --plast-eta 0.05 --plast-lambda 0.9
done
```

Observed `best_fitness`:
- seed 0: `best_fitness=179.6250` (`runs/2025-12-15_stage2_es_plast_eta0.05_seed0`)
- seed 1: `best_fitness=179.5000` (`runs/2025-12-15_stage2_es_plast_eta0.05_seed1`)
- seed 2: `best_fitness=180.3750` (`runs/2025-12-15_stage2_es_plast_eta0.05_seed2`)

### C) Held-out evaluation via `koki2 eval-run` (128 episodes; two eval seeds)

Protocol:
- Evaluate `best_genome.npz` from each run dir on 128 episodes.
- Compare to greedy baseline on the same episode keys (`--baseline-policy greedy`).
- Repeat evaluation with `--seed 0` and `--seed 1` to check sensitivity to the evaluation episode set.

Commands (eval seed 0):
```bash
for run_dir in \
  runs/2025-12-15_stage2_es_noplast_seed0 \
  runs/2025-12-15_stage2_es_noplast_seed1 \
  runs/2025-12-15_stage2_es_noplast_seed2 \
  runs/2025-12-15_stage2_es_plast_eta0.05_seed0 \
  runs/2025-12-15_stage2_es_plast_eta0.05_seed1 \
  runs/2025-12-15_stage2_es_plast_eta0.05_seed2; do
  uv run koki2 eval-run --run-dir "$run_dir" --episodes 128 --seed 0 --baseline-policy greedy
done
```

Observed outputs (selected; best-genome only):
- eval seed 0, non-plastic:
  - seed 0: `mean_fitness=162.5234`, `success_rate=0.703`, `mean_t_alive=126.7`, `mean_bad_arrivals=1.1875`, `mean_integrity_min=0.7031`
  - seed 1: `mean_fitness=162.5195`, `success_rate=0.742`, `mean_t_alive=124.8`, `mean_bad_arrivals=1.2500`, `mean_integrity_min=0.6875`
  - seed 2: `mean_fitness=163.9922`, `success_rate=0.742`, `mean_t_alive=126.2`, `mean_bad_arrivals=1.4062`, `mean_integrity_min=0.6484`
- eval seed 0, plastic (`eta=0.05`):
  - seed 0: `mean_fitness=170.7461`, `success_rate=0.852`, `mean_t_alive=127.0`, `mean_bad_arrivals=2.0234`, `mean_integrity_min=0.4941`
  - seed 1: `mean_fitness=162.5195`, `success_rate=0.742`, `mean_t_alive=124.8`, `mean_bad_arrivals=1.2500`, `mean_integrity_min=0.6875`
  - seed 2: `mean_fitness=168.8633`, `success_rate=0.898`, `mean_t_alive=122.9`, `mean_bad_arrivals=1.8672`, `mean_integrity_min=0.5352`

Commands (eval seed 1):
```bash
for run_dir in \
  runs/2025-12-15_stage2_es_noplast_seed0 \
  runs/2025-12-15_stage2_es_noplast_seed1 \
  runs/2025-12-15_stage2_es_noplast_seed2 \
  runs/2025-12-15_stage2_es_plast_eta0.05_seed0 \
  runs/2025-12-15_stage2_es_plast_eta0.05_seed1 \
  runs/2025-12-15_stage2_es_plast_eta0.05_seed2; do
  uv run koki2 eval-run --run-dir "$run_dir" --episodes 128 --seed 1 --baseline-policy greedy
done
```

Observed outputs (selected; best-genome only):
- eval seed 1, non-plastic:
  - seed 0: `mean_fitness=162.1094`, `success_rate=0.695`, `mean_t_alive=126.7`, `mean_bad_arrivals=1.3125`, `mean_integrity_min=0.6719`
  - seed 1: `mean_fitness=164.2227`, `success_rate=0.758`, `mean_t_alive=125.6`, `mean_bad_arrivals=1.2656`, `mean_integrity_min=0.6836`
  - seed 2: `mean_fitness=167.3711`, `success_rate=0.789`, `mean_t_alive=127.2`, `mean_bad_arrivals=1.3203`, `mean_integrity_min=0.6699`
- eval seed 1, plastic (`eta=0.05`):
  - seed 0: `mean_fitness=164.5508`, `success_rate=0.766`, `mean_t_alive=125.3`, `mean_bad_arrivals=1.9219`, `mean_integrity_min=0.5195`
  - seed 1: `mean_fitness=164.2227`, `success_rate=0.758`, `mean_t_alive=125.6`, `mean_bad_arrivals=1.2656`, `mean_integrity_min=0.6836`
  - seed 2: `mean_fitness=170.3477`, `success_rate=0.875`, `mean_t_alive=125.7`, `mean_bad_arrivals=1.9062`, `mean_integrity_min=0.5234`

Interpretation (provisional; early Stage 2 evidence):
- With `eta=0.05` at this tiny ES budget, plasticity **tends to increase mean fitness** mainly via higher success rates, but it also **tends to worsen hazard-contact metrics** (higher bad-arrivals and lower integrity minima) in the same benchmark.
- One of the three seeds (seed 1) produced best-genome behavior that is effectively indistinguishable between plastic and non-plastic under both evaluation seeds; this suggests plasticity may sometimes be “unused” (e.g., near-zero modulator / near-zero applied updates) or the evolved policy is insensitive to within-life adaptation under this budget.
- This is compatible with the hypothesis that our current plasticity implementation is not yet strongly “consequence-driven” (the agent’s modulator does not currently read internal state), so enabling plasticity can improve task success without necessarily improving integrity-preservation outcomes.

---

## 2025-12-15 — Stage 2 instrumentation: measure plasticity usage (`mean_abs_dw_mean`)

Goal: make it observable whether “plastic runs” are actually using within-life learning (and how strongly), to interpret mixed results (e.g., a seed where plastic/non-plastic look identical).

Changes:
- `src/koki2/types.py`: added `FitnessSummary.mean_abs_dw_mean` (mean absolute applied weight update per alive step).
- `src/koki2/sim/orchestrator.py`: accumulate `AgentLog.mean_abs_dw` into `FitnessSummary.mean_abs_dw_mean`; baselines report 0.
- `src/koki2/cli.py`: print `mean_abs_dw_mean` for both `baseline-l0` and `eval-run`.
- Tests: extended determinism/JIT parity checks to include `mean_abs_dw_mean` (`tests/test_stage0_determinism.py`, `tests/test_baseline_determinism.py`).

Verification:
- Ran `uv run pytest` (21 tests) — all passing.

Probe (re-evaluate Stage 2 run dirs; `--episodes 128 --seed 0`; baseline omitted):
```bash
for run_dir in \
  runs/2025-12-15_stage2_es_noplast_seed0 \
  runs/2025-12-15_stage2_es_noplast_seed1 \
  runs/2025-12-15_stage2_es_noplast_seed2 \
  runs/2025-12-15_stage2_es_plast_eta0.05_seed0 \
  runs/2025-12-15_stage2_es_plast_eta0.05_seed1 \
  runs/2025-12-15_stage2_es_plast_eta0.05_seed2; do
  uv run koki2 eval-run --run-dir "$run_dir" --episodes 128 --seed 0 --baseline-policy none
done
```

Observed `mean_abs_dw_mean` (best-genome only):
- non-plastic: all seeds report `mean_abs_dw_mean=0.000000` (expected).
- plastic (`eta=0.05`):
  - seed 0: `mean_abs_dw_mean=0.002359`
  - seed 1: `mean_abs_dw_mean=0.000037` (essentially “unused” under this probe)
  - seed 2: `mean_abs_dw_mean=0.036644` (substantial within-life adaptation)

Interpretation:
- The “no effect” seed (seed 1) is consistent with negligible applied plastic updates rather than a deep failure of the evaluation protocol; future comparisons should report both performance and `mean_abs_dw_mean`.

---

## 2025-12-15 — Stage 2 follow-up: plasticity strength (`plast_eta`) sweep (0.01 vs 0.1)

Goal: check whether the Stage 2 effect depends on plasticity update magnitude, and whether any setting improves performance without degrading hazard metrics.

Env and compute budget: same as the Stage 2 benchmark above:
- env: `--num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25 --deplete-sources --respawn-delay 4 --grad-dropout-p 0.5 --steps 128`
- ES: `--generations 10 --pop-size 64 --episodes 4`

### A) ES runs

Commands:
```bash
for eta in 0.01 0.1; do
  for seed in 0 1 2; do
    uv run koki2 evo-l0 --seed $seed --out-dir runs/2025-12-15_stage2_es_plast_eta${eta}_seed${seed} \
      --generations 10 --pop-size 64 --episodes 4 --steps 128 \
      --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25 \
      --deplete-sources --respawn-delay 4 \
      --grad-dropout-p 0.5 \
      --plast-enabled --plast-eta $eta --plast-lambda 0.9
  done
done
```

Observed `best_fitness` (printed at end of each run):
- `eta=0.01`: seed 0 `179.3750`, seed 1 `179.5000`, seed 2 `179.8750`
- `eta=0.1`: seed 0 `179.7500`, seed 1 `179.5000`, seed 2 `180.2500`

### B) Held-out eval (`koki2 eval-run`; 128 episodes; eval seeds 0 and 1)

Commands (eval seed 0):
```bash
for eta in 0.01 0.1; do
  for seed in 0 1 2; do
    uv run koki2 eval-run --run-dir runs/2025-12-15_stage2_es_plast_eta${eta}_seed${seed} --episodes 128 --seed 0 --baseline-policy none
  done
done
```

Observed outputs (selected; best-genome only; eval seed 0):
- `eta=0.01`:
  - seed 0: `mean_fitness=164.0742`, `success_rate=0.734`, `mean_bad_arrivals=1.1484`, `mean_integrity_min=0.7129`, `mean_abs_dw_mean=0.000082`
  - seed 1: `mean_fitness=164.5273`, `success_rate=0.734`, `mean_bad_arrivals=1.0703`, `mean_integrity_min=0.7324`, `mean_abs_dw_mean=0.000007`
  - seed 2: `mean_fitness=163.7031`, `success_rate=0.711`, `mean_bad_arrivals=1.1484`, `mean_integrity_min=0.7129`, `mean_abs_dw_mean=0.000068`
- `eta=0.1`:
  - seed 0: `mean_fitness=169.8711`, `success_rate=0.898`, `mean_bad_arrivals=2.5234`, `mean_integrity_min=0.3711`, `mean_abs_dw_mean=0.023372`
  - seed 1: `mean_fitness=162.5195`, `success_rate=0.742`, `mean_bad_arrivals=1.2500`, `mean_integrity_min=0.6875`, `mean_abs_dw_mean=0.000063`
  - seed 2: `mean_fitness=164.5312`, `success_rate=0.945`, `mean_bad_arrivals=2.2344`, `mean_integrity_min=0.4434`, `mean_abs_dw_mean=0.071393`

Commands (eval seed 1):
```bash
for eta in 0.01 0.1; do
  for seed in 0 1 2; do
    uv run koki2 eval-run --run-dir runs/2025-12-15_stage2_es_plast_eta${eta}_seed${seed} --episodes 128 --seed 1 --baseline-policy none
  done
done
```

Observed outputs (selected; best-genome only; eval seed 1):
- `eta=0.01`:
  - seed 0: `mean_fitness=159.3086`, `success_rate=0.656`, `mean_bad_arrivals=1.2578`, `mean_integrity_min=0.6855`, `mean_abs_dw_mean=0.000082`
  - seed 1: `mean_fitness=159.0664`, `success_rate=0.641`, `mean_bad_arrivals=1.0781`, `mean_integrity_min=0.7305`, `mean_abs_dw_mean=0.000007`
  - seed 2: `mean_fitness=162.5391`, `success_rate=0.711`, `mean_bad_arrivals=1.2109`, `mean_integrity_min=0.6973`, `mean_abs_dw_mean=0.000068`
- `eta=0.1`:
  - seed 0: `mean_fitness=166.9180`, `success_rate=0.859`, `mean_bad_arrivals=2.5312`, `mean_integrity_min=0.3672`, `mean_abs_dw_mean=0.023081`
  - seed 1: `mean_fitness=164.2227`, `success_rate=0.758`, `mean_bad_arrivals=1.2656`, `mean_integrity_min=0.6836`, `mean_abs_dw_mean=0.000065`
  - seed 2: `mean_fitness=161.9219`, `success_rate=0.922`, `mean_bad_arrivals=2.4062`, `mean_integrity_min=0.3984`, `mean_abs_dw_mean=0.072393`

Interpretation (provisional):
- Larger `plast_eta` correlates with substantially larger applied updates (`mean_abs_dw_mean`) in some seeds and yields very high success rates, but it also tends to increase bad-source contact and reduce integrity minima (riskier behavior).
- Very small `plast_eta` yields near-zero updates and is effectively “almost non-plastic” under this benchmark; its performance appears sensitive to the evaluation episode set (eval seed).

---

## 2025-12-15 — Stage 2 robustness check: increase ES budget (30 generations; `eta=0.05` vs non-plastic)

Goal: verify that the earlier plastic-vs-non-plastic pattern is not an artifact of the tiny 10-generation budget by rerunning at 30 generations on the same benchmark, then evaluating on held-out episodes and reporting `mean_abs_dw_mean`.

Env: same Stage 2 benchmark (see above).

Compute budget:
- `--generations 30 --pop-size 64 --episodes 4 --steps 128`

### A) ES runs

Commands (non-plastic; seeds 0/1/2):
```bash
for seed in 0 1 2; do
  uv run koki2 evo-l0 --seed $seed --out-dir runs/2025-12-15_stage2_es30_noplast_seed${seed} \
    --generations 30 --pop-size 64 --episodes 4 --steps 128 \
    --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25 \
    --deplete-sources --respawn-delay 4 \
    --grad-dropout-p 0.5
done
```

Observed `best_fitness`:
- seed 0: `179.5000` (`runs/2025-12-15_stage2_es30_noplast_seed0`)
- seed 1: `179.7500` (`runs/2025-12-15_stage2_es30_noplast_seed1`)
- seed 2: `179.6250` (`runs/2025-12-15_stage2_es30_noplast_seed2`)

Commands (plastic; seeds 0/1/2; `eta=0.05`):
```bash
for seed in 0 1 2; do
  uv run koki2 evo-l0 --seed $seed --out-dir runs/2025-12-15_stage2_es30_plast_eta0.05_seed${seed} \
    --generations 30 --pop-size 64 --episodes 4 --steps 128 \
    --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25 \
    --deplete-sources --respawn-delay 4 \
    --grad-dropout-p 0.5 \
    --plast-enabled --plast-eta 0.05 --plast-lambda 0.9
done
```

Observed `best_fitness` (note: the seed 2 run to `runs/2025-12-15_stage2_es30_plast_eta0.05_seed2` was interrupted; reran to a fresh directory):
- seed 0: `180.2500` (`runs/2025-12-15_stage2_es30_plast_eta0.05_seed0`)
- seed 1: `179.5000` (`runs/2025-12-15_stage2_es30_plast_eta0.05_seed1`)
- seed 2: `180.3750` (`runs/2025-12-15_stage2_es30_plast_eta0.05_seed2_retry1`)

### B) Held-out eval (`koki2 eval-run`; 128 episodes; eval seeds 0 and 1)

Commands (eval seed 0):
```bash
for run_dir in \
  runs/2025-12-15_stage2_es30_noplast_seed0 \
  runs/2025-12-15_stage2_es30_noplast_seed1 \
  runs/2025-12-15_stage2_es30_noplast_seed2 \
  runs/2025-12-15_stage2_es30_plast_eta0.05_seed0 \
  runs/2025-12-15_stage2_es30_plast_eta0.05_seed1 \
  runs/2025-12-15_stage2_es30_plast_eta0.05_seed2_retry1; do
  uv run koki2 eval-run --run-dir "$run_dir" --episodes 128 --seed 0 --baseline-policy none
done
```

Observed outputs (selected; best-genome only; eval seed 0):
- non-plastic:
  - seed 0: `mean_fitness=162.5234`, `success_rate=0.703`, `mean_bad_arrivals=1.1875`, `mean_integrity_min=0.7031`
  - seed 1: `mean_fitness=164.2188`, `success_rate=0.742`, `mean_bad_arrivals=1.2969`, `mean_integrity_min=0.6758`
  - seed 2: `mean_fitness=164.2109`, `success_rate=0.742`, `mean_bad_arrivals=1.2891`, `mean_integrity_min=0.6777`
- plastic (`eta=0.05`):
  - seed 0: `mean_fitness=171.4141`, `success_rate=0.875`, `mean_bad_arrivals=1.8594`, `mean_integrity_min=0.5371`, `mean_abs_dw_mean=0.007055`
  - seed 1: `mean_fitness=162.5195`, `success_rate=0.742`, `mean_bad_arrivals=1.2500`, `mean_integrity_min=0.6875`, `mean_abs_dw_mean=0.000037`
  - seed 2: `mean_fitness=168.8633`, `success_rate=0.898`, `mean_bad_arrivals=1.8672`, `mean_integrity_min=0.5352`, `mean_abs_dw_mean=0.036644`

Commands (eval seed 1):
```bash
for run_dir in \
  runs/2025-12-15_stage2_es30_noplast_seed0 \
  runs/2025-12-15_stage2_es30_noplast_seed1 \
  runs/2025-12-15_stage2_es30_noplast_seed2 \
  runs/2025-12-15_stage2_es30_plast_eta0.05_seed0 \
  runs/2025-12-15_stage2_es30_plast_eta0.05_seed1 \
  runs/2025-12-15_stage2_es30_plast_eta0.05_seed2_retry1; do
  uv run koki2 eval-run --run-dir "$run_dir" --episodes 128 --seed 1 --baseline-policy none
done
```

Observed outputs (selected; best-genome only; eval seed 1):
- non-plastic:
  - seed 0: `mean_fitness=162.1094`, `success_rate=0.695`, `mean_bad_arrivals=1.3125`, `mean_integrity_min=0.6719`
  - seed 1: `mean_fitness=166.2148`, `success_rate=0.766`, `mean_bad_arrivals=1.2266`, `mean_integrity_min=0.6934`
  - seed 2: `mean_fitness=164.7461`, `success_rate=0.766`, `mean_bad_arrivals=1.2422`, `mean_integrity_min=0.6895`
- plastic (`eta=0.05`):
  - seed 0: `mean_fitness=170.3203`, `success_rate=0.852`, `mean_bad_arrivals=1.7656`, `mean_integrity_min=0.5586`, `mean_abs_dw_mean=0.007266`
  - seed 1: `mean_fitness=164.2227`, `success_rate=0.758`, `mean_bad_arrivals=1.2656`, `mean_integrity_min=0.6836`, `mean_abs_dw_mean=0.000038`
  - seed 2: `mean_fitness=170.3477`, `success_rate=0.875`, `mean_bad_arrivals=1.9062`, `mean_integrity_min=0.5234`, `mean_abs_dw_mean=0.039221`

Interpretation (provisional, but stronger than 10 generations):
- Across eval seeds 0 and 1, `eta=0.05` remains **higher mean fitness** in 2/3 seeds and increases `mean_abs_dw_mean` in those same seeds, suggesting the effect is not purely a 10-gen fluke.
- The same trade-off persists: improved success/fitness often comes with worse hazard metrics (higher bad arrivals / lower integrity minima).

---

## 2025-12-15 — Stage 2 hypothesis refinement: consequence-aligned neuromodulation (drive-based modulator)

Goal: align plasticity with the thesis by making the neuromodulatory signal explicitly consequence-driven (drive reduction), rather than only spike-derived. This is intended to reduce the observed “success improves but hazard metrics worsen” pattern by tying learning updates to viability-relevant outcomes.

Changes:
- `src/koki2/types.py`: added `DevConfig.modulator_kind` (0=spike, 1=drive) and `DevConfig.mod_drive_scale`; added corresponding `AgentParams` fields.
- `src/koki2/cli.py`: added `--modulator-kind {spike,drive}` and `--mod-drive-scale` to `koki2 evo-l0` so we can run Stage 2 comparisons without changing tensor shapes.
- `src/koki2/sim/orchestrator.py`: threads a per-step modulator signal through the rollout (`mod_signal`), set to the reward/drive reduction `drive_prev - drive2` each step and fed into the agent on the next step.
- `src/koki2/agent/snn.py`: if `modulator_kind=drive`, uses a bounded drive signal as modulator; learning updates use the eligibility trace `elig_next` (so the drive signal is strong enough to produce measurable within-life updates under the current dynamics).

Verification:
- Ran `uv run pytest` (22 tests) — all passing.

### Stage 2 quick experiment: drive-modulated plasticity vs spike-modulated plasticity

Benchmark env: same as earlier Stage 2 runs:
- `--num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25`
- `--deplete-sources --respawn-delay 4`
- `--grad-dropout-p 0.5`
- `--steps 128`

ES budget:
- `--generations 10 --pop-size 64 --episodes 4`

Runs (drive modulator; `eta=0.05`, `mod_drive_scale=2000`; seeds 0/1/2):
```bash
for seed in 0 1 2; do
  uv run koki2 evo-l0 --seed $seed --out-dir runs/2025-12-15_stage2_drive_mod2_es10_eta0.05_scale2000_seed${seed} \
    --generations 10 --pop-size 64 --episodes 4 --steps 128 \
    --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25 \
    --deplete-sources --respawn-delay 4 \
    --grad-dropout-p 0.5 \
    --plast-enabled --plast-eta 0.05 --plast-lambda 0.9 \
    --modulator-kind drive --mod-drive-scale 2000
done
```

Held-out eval (128 episodes; eval seeds 0 and 1):
```bash
for eval_seed in 0 1; do
  for seed in 0 1 2; do
    uv run koki2 eval-run --run-dir runs/2025-12-15_stage2_drive_mod2_es10_eta0.05_scale2000_seed${seed} \
      --episodes 128 --seed $eval_seed --baseline-policy none
  done
done
```

Observed outputs (selected; best-genome only):
- eval seed 0:
  - seed 0: `mean_fitness=164.7852`, `mean_bad_arrivals=1.0625`, `mean_integrity_min=0.7344`, `mean_abs_dw_mean=0.000232`
  - seed 1: `mean_fitness=162.5195`, `mean_bad_arrivals=1.2500`, `mean_integrity_min=0.6875`, `mean_abs_dw_mean=0.000131`
  - seed 2: `mean_fitness=163.6211`, `mean_bad_arrivals=1.3750`, `mean_integrity_min=0.6562`, `mean_abs_dw_mean=0.000281`
- eval seed 1:
  - seed 0: `mean_fitness=158.6523`, `mean_bad_arrivals=1.1953`, `mean_integrity_min=0.7012`, `mean_abs_dw_mean=0.000231`
  - seed 1: `mean_fitness=164.2227`, `mean_bad_arrivals=1.2656`, `mean_integrity_min=0.6836`, `mean_abs_dw_mean=0.000127`
  - seed 2: `mean_fitness=164.3086`, `mean_bad_arrivals=1.3125`, `mean_integrity_min=0.6719`, `mean_abs_dw_mean=0.000286`

Interpretation (provisional):
- Drive-modulated plasticity produces **measurable but small** within-life updates (`mean_abs_dw_mean≈2e-4` at this setting), and it does not show the large “risky success” shifts observed with spike-modulated plasticity at the same `eta` (which had much larger `mean_abs_dw_mean`).
- On this benchmark and tiny ES budget, drive-modulated plasticity appears closer to the non-plastic regime (safer hazard metrics, modest mean fitness changes), suggesting we may need either:
  - a better-shaped consequence signal (e.g., centering/baselines), or
  - a different credit-assignment timing (update weights after observing the step’s consequence), or
  - a more direct consequence proxy (e.g., event-based energy/integrity deltas),
  to see a clear Stage 2 advantage without hazard regression.

---

## 2025-12-15 — Stage 2 extension: event-based consequence modulator (`event_delta`)

Goal: add a second consequence-derived option that is closer to “direct internal consequences” than drive delta: modulate plasticity by `event_delta = energy_gained - integrity_lost`.

Changes:
- `src/koki2/types.py`: extended `modulator_kind` semantics: 0=spike, 1=drive_delta, 2=event_delta.
- `src/koki2/cli.py`: added `--modulator-kind event` (reuses `--mod-drive-scale` to scale the signal).
- `src/koki2/sim/orchestrator.py`: sets the per-step `mod_signal` based on `modulator_kind`:
  - drive: `drive_prev - drive2`
  - event: `env_log.energy_gained - env_log.integrity_lost`
- `tests/test_agent_plasticity.py`: added a unit test that drive modulation can update weights given a nonzero modulator signal.

Verification:
- Ran `uv run pytest` (22 tests) — all passing.

Pilot run (event modulator; seed 0 only; benchmark env as above):
```bash
uv run koki2 evo-l0 --seed 0 --out-dir runs/2025-12-15_stage2_event_mod_es10_eta0.05_scale4_seed0 \
  --generations 10 --pop-size 64 --episodes 4 --steps 128 \
  --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25 \
  --deplete-sources --respawn-delay 4 \
  --grad-dropout-p 0.5 \
  --plast-enabled --plast-eta 0.05 --plast-lambda 0.9 \
  --modulator-kind event --mod-drive-scale 4

uv run koki2 eval-run --run-dir runs/2025-12-15_stage2_event_mod_es10_eta0.05_scale4_seed0 \
  --episodes 128 --seed 0 --baseline-policy none
```

Observed output (best-genome):
- `mean_fitness=164.2617`, `mean_bad_arrivals=1.0781`, `mean_integrity_min=0.7305`, `mean_abs_dw_mean=0.000012`

Replication (event modulator; seeds 0/1/2; same env and ES budget; eval seeds 0 and 1):
```bash
for seed in 0 1 2; do
  uv run koki2 evo-l0 --seed $seed --out-dir runs/2025-12-15_stage2_event_mod_es10_eta0.05_scale4_seed${seed} \
    --generations 10 --pop-size 64 --episodes 4 --steps 128 \
    --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25 \
    --deplete-sources --respawn-delay 4 \
    --grad-dropout-p 0.5 \
    --plast-enabled --plast-eta 0.05 --plast-lambda 0.9 \
    --modulator-kind event --mod-drive-scale 4
done

for eval_seed in 0 1; do
  for seed in 0 1 2; do
    uv run koki2 eval-run --run-dir runs/2025-12-15_stage2_event_mod_es10_eta0.05_scale4_seed${seed} \
      --episodes 128 --seed $eval_seed --baseline-policy none
  done
done
```

Observed outputs (selected; best-genome only):
- eval seed 0:
  - seed 0: `mean_fitness=164.2617`, `mean_bad_arrivals=1.0781`, `mean_integrity_min=0.7305`, `mean_abs_dw_mean=0.000012`
  - seed 1: `mean_fitness=162.5273`, `mean_bad_arrivals=1.2500`, `mean_integrity_min=0.6875`, `mean_abs_dw_mean=0.000004`
  - seed 2: `mean_fitness=164.9336`, `mean_bad_arrivals=1.2500`, `mean_integrity_min=0.6875`, `mean_abs_dw_mean=0.000013`
- eval seed 1:
  - seed 0: `mean_fitness=158.8164`, `mean_bad_arrivals=1.1641`, `mean_integrity_min=0.7090`, `mean_abs_dw_mean=0.000013`
  - seed 1: `mean_fitness=164.2227`, `mean_bad_arrivals=1.2656`, `mean_integrity_min=0.6836`, `mean_abs_dw_mean=0.000004`
  - seed 2: `mean_fitness=165.3828`, `mean_bad_arrivals=1.1562`, `mean_integrity_min=0.7109`, `mean_abs_dw_mean=0.000012`

Interpretation (provisional):
- With this particular shaping (event delta; `scale=4`), within-life updates remain extremely small on average (`mean_abs_dw_mean≈1e-5`) and behavior is close to the non-plastic regime (good hazard metrics, modest fitness changes).

---

## 2025-12-15 — Stage 2 refinement: same-step consequence modulation (credit timing)

Goal: make consequence-derived modulators (drive/event) arrive in the **same step** as the consequence (after `env_step`), so eligibility traces can bridge action→outcome credit without an extra 1-step delay.

Changes:
- `src/koki2/agent/snn.py`: split into `agent_forward` (spikes/action/eligibility) and `agent_apply_plasticity` (apply modulated update once the modulator signal is known).
- `src/koki2/sim/orchestrator.py`: restructure loop to `agent_forward → env_step → mod_signal_raw → agent_apply_plasticity` (no carry of `mod_signal`).
- `src/koki2/agent/__init__.py`: export the updated agent functions.
- `tests/test_agent_plasticity.py`: updated to the new API and strengthened the drive-mod test (mod signal `0.5` to disambiguate from spike-mod).

Verification:
- Ran `uv run pytest` (22 tests) — all passing.
- Spot-check: re-evaluated an existing spike-modulated plastic run dir and reproduced the previously logged metrics exactly:
```bash
uv run koki2 eval-run --run-dir runs/2025-12-15_stage2_es_plast_eta0.05_seed0 --episodes 128 --seed 0 --baseline-policy none
```
Observed output (best-genome):
- `mean_fitness=170.7461`, `mean_bad_arrivals=2.0234`, `mean_integrity_min=0.4941`, `mean_abs_dw_mean=0.002359`

### Stage 2 comparison rerun (timing fix): drive modulator

Benchmark env: same as earlier Stage 2 runs:
- `--num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25`
- `--deplete-sources --respawn-delay 4`
- `--grad-dropout-p 0.5`
- `--steps 128`

ES budget:
- `--generations 10 --pop-size 64 --episodes 4`

Runs (drive modulator; `eta=0.05`, `mod_drive_scale=2000`; seeds 0/1/2):
```bash
for seed in 0 1 2; do
  uv run koki2 evo-l0 --seed $seed --out-dir runs/2025-12-15_stage2_drive_mod_timingfix_es10_eta0.05_scale2000_seed${seed} \
    --generations 10 --pop-size 64 --episodes 4 --steps 128 \
    --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25 \
    --deplete-sources --respawn-delay 4 \
    --grad-dropout-p 0.5 \
    --plast-enabled --plast-eta 0.05 --plast-lambda 0.9 \
    --modulator-kind drive --mod-drive-scale 2000
done
```

Observed `best_fitness`:
- seed 0: `179.2500`
- seed 1: `179.5000`
- seed 2: `179.5000`

Held-out eval (128 episodes; eval seeds 0 and 1):
```bash
for eval_seed in 0 1; do
  for seed in 0 1 2; do
    uv run koki2 eval-run --run-dir runs/2025-12-15_stage2_drive_mod_timingfix_es10_eta0.05_scale2000_seed${seed} \
      --episodes 128 --seed $eval_seed --baseline-policy none
  done
done
```

Observed outputs (selected; best-genome only):
- eval seed 0:
  - seed 0: `mean_fitness=162.9023`, `mean_bad_arrivals=1.2969`, `mean_integrity_min=0.6758`, `mean_abs_dw_mean=0.000232`
  - seed 1: `mean_fitness=163.8359`, `mean_bad_arrivals=1.3281`, `mean_integrity_min=0.6680`, `mean_abs_dw_mean=0.000079`
  - seed 2: `mean_fitness=164.4180`, `mean_bad_arrivals=1.3672`, `mean_integrity_min=0.6582`, `mean_abs_dw_mean=0.000392`
- eval seed 1:
  - seed 0: `mean_fitness=162.5820`, `mean_bad_arrivals=1.3828`, `mean_integrity_min=0.6543`, `mean_abs_dw_mean=0.000234`
  - seed 1: `mean_fitness=166.1484`, `mean_bad_arrivals=1.2344`, `mean_integrity_min=0.6914`, `mean_abs_dw_mean=0.000079`
  - seed 2: `mean_fitness=163.6289`, `mean_bad_arrivals=1.3672`, `mean_integrity_min=0.6582`, `mean_abs_dw_mean=0.000405`

Interpretation (provisional):
- Under this setting, same-step drive modulation does not materially change the earlier finding that consequence-derived plasticity stays close to the non-plastic regime: `mean_abs_dw_mean` remains small (`~8e-5` to `~4e-4`) and held-out hazard metrics are in the same range as the non-plastic baselines.

### Stage 2 comparison rerun (timing fix): event modulator

Runs (event modulator; `eta=0.05`, `scale=4`; seeds 0/1/2):
```bash
for seed in 0 1 2; do
  uv run koki2 evo-l0 --seed $seed --out-dir runs/2025-12-15_stage2_event_mod_timingfix_es10_eta0.05_scale4_seed${seed} \
    --generations 10 --pop-size 64 --episodes 4 --steps 128 \
    --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25 \
    --deplete-sources --respawn-delay 4 \
    --grad-dropout-p 0.5 \
    --plast-enabled --plast-eta 0.05 --plast-lambda 0.9 \
    --modulator-kind event --mod-drive-scale 4
done
```

Observed `best_fitness`:
- seed 0: `179.2500`
- seed 1: `179.5000`
- seed 2: `179.3750`

Held-out eval (128 episodes; eval seeds 0 and 1):
```bash
for eval_seed in 0 1; do
  for seed in 0 1 2; do
    uv run koki2 eval-run --run-dir runs/2025-12-15_stage2_event_mod_timingfix_es10_eta0.05_scale4_seed${seed} \
      --episodes 128 --seed $eval_seed --baseline-policy none
  done
done
```

Observed outputs (selected; best-genome only):
- eval seed 0:
  - seed 0: `mean_fitness=156.5938`, `mean_bad_arrivals=1.1406`, `mean_integrity_min=0.7148`, `mean_abs_dw_mean=0.000012`
  - seed 1: `mean_fitness=164.1250`, `mean_bad_arrivals=1.3125`, `mean_integrity_min=0.6719`, `mean_abs_dw_mean=0.000004`
  - seed 2: `mean_fitness=163.8125`, `mean_bad_arrivals=1.3359`, `mean_integrity_min=0.6660`, `mean_abs_dw_mean=0.000028`
- eval seed 1:
  - seed 0: `mean_fitness=156.3750`, `mean_bad_arrivals=1.1953`, `mean_integrity_min=0.7012`, `mean_abs_dw_mean=0.000012`
  - seed 1: `mean_fitness=164.7852`, `mean_bad_arrivals=1.2656`, `mean_integrity_min=0.6836`, `mean_abs_dw_mean=0.000004`
  - seed 2: `mean_fitness=165.0703`, `mean_bad_arrivals=1.2188`, `mean_integrity_min=0.6953`, `mean_abs_dw_mean=0.000026`

Interpretation (provisional):
- Event-based modulation still produces extremely small average applied updates (`mean_abs_dw_mean≈1e-5` to `3e-5` at `scale=4`), and most runs remain close to the non-plastic regime.
- One seed shows a lower-fitness but “safer”-looking pattern (higher `mean_integrity_min`, lower `mean_bad_arrivals`) that could be real or could be an ES-budget artifact; it needs replication and a more systematic scale sweep before treating it as evidence.

---

## 2025-12-15 — Infrastructure check: RunPod GPU bursts for `koki2 evo-l0`

Goal: test whether a “burst GPU” workflow (spin up a cheap GPU pod, run a handful of experiments, tear down) yields meaningful wall-time speedups for our current `evo-l0` runs compared to local CPU, without changing the research task.

Local baseline (Mac, force CPU; identical env knobs to our Stage 2 benchmark):
```bash
/usr/bin/time -p env JAX_PLATFORM_NAME=cpu uv run koki2 evo-l0 --seed 0 --out-dir runs/2025-12-15_bench_local_cpu_es10_seed0 \
  --generations 10 --pop-size 64 --episodes 4 --steps 128 \
  --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25 \
  --deplete-sources --respawn-delay 4 \
  --grad-dropout-p 0.5

/usr/bin/time -p env JAX_PLATFORM_NAME=cpu uv run koki2 evo-l0 --seed 0 --out-dir runs/2025-12-15_bench_local_cpu_es50_seed0 \
  --generations 50 --pop-size 64 --episodes 4 --steps 128 \
  --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25 \
  --deplete-sources --respawn-delay 4 \
  --grad-dropout-p 0.5

/usr/bin/time -p env JAX_PLATFORM_NAME=cpu uv run koki2 evo-l0 --seed 0 --out-dir runs/2025-12-15_bench_local_cpu_es10_pop512_seed0 \
  --generations 10 --pop-size 512 --episodes 4 --steps 128 \
  --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25 \
  --deplete-sources --respawn-delay 4 \
  --grad-dropout-p 0.5
```

Observed times (wall clock):
- `es10 pop64`: `real 15.52s`
- `es50 pop64`: `real 59.11s`
- `es10 pop512`: `real 20.28s`

RunPod setup notes (CLI + SSH):
- Used `runpodctl` + public IP pod and installed CUDA JAX via `uv pip install -U "jax[cuda12]"`.
- SSH auth required injecting the public key into the container start command (the `$PUBLIC_KEY` placeholder approach did not work as expected in our first attempt).
- Verified JAX sees GPU with `devices [CudaDevice(id=0)]`.

### RunPod test A: RTX 3070 (community cloud)

Command (remote; same args, only out-dir differs):
```bash
time -p koki2 evo-l0 --seed 0 --out-dir runs/2025-12-15_bench_runpod_rtx3070_gpu_es10_seed0 \
  --generations 10 --pop-size 64 --episodes 4 --steps 128 \
  --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25 \
  --deplete-sources --respawn-delay 4 \
  --grad-dropout-p 0.5

time -p koki2 evo-l0 --seed 0 --out-dir runs/2025-12-15_bench_runpod_rtx3070_gpu_es50_seed0 \
  --generations 50 --pop-size 64 --episodes 4 --steps 128 \
  --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25 \
  --deplete-sources --respawn-delay 4 \
  --grad-dropout-p 0.5

time -p koki2 evo-l0 --seed 0 --out-dir runs/2025-12-15_bench_runpod_rtx3070_gpu_es10_pop512_seed0 \
  --generations 10 --pop-size 512 --episodes 4 --steps 128 \
  --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25 \
  --deplete-sources --respawn-delay 4 \
  --grad-dropout-p 0.5
```

Observed times:
- `es10 pop64`: `real 26.28s`
- `es50 pop64`: `real 88.08s`
- `es10 pop512`: `real 27.38s`

### RunPod test B: RTX 3090 Ti (community cloud)

Observed times:
- `es10 pop64`: `real 34.79s`
- `es50 pop64`: `real 130.18s`
- `es10 pop512`: `real 33.53s`

Interpretation (provisional):
- For our current execution pattern (“one CLI invocation per ES run”), GPU pods are slower than local CPU on wall time at these sizes. The likely cause is that our ES loop does frequent host/device synchronization per generation and does not amortize compile/startup costs (especially punishing in short burst runs).
- Conclusion: to make burst GPUs useful for our thesis workflow, we likely need a harness change that amortizes overhead without changing the research task itself: e.g., batch many runs in a single long-lived process (multi-seed/sweep runner) and/or a JIT-friendly ES loop that avoids per-generation `device_get`/I/O.

---

## 2025-12-15 — Tooling: scripted RunPod “burst benchmark” (auto tear-down)

Goal: encode a safe “temporary pod” workflow (create → run → fetch → destroy) so we don’t leave pods running when we stop working or abandon a run.

Changes:
- `tools/runpod_burst_bench.sh`: creates a short-lived pod via `runpodctl`, uploads the repo as a tarball, runs a remote command via SSH, downloads selected artifacts, and always removes the pod on exit (including Ctrl-C).

Verification:
- `bash -n tools/runpod_burst_bench.sh`
- `tools/runpod_burst_bench.sh --help`

Notes:
- This script currently uses `runpodctl` for lifecycle + SSH connect details, and uses `ssh`/`scp` for execution and file transfer (our current local `runpodctl exec` subcommand appears limited to `exec python`).

---

## 2025-12-15 — Harness improvement: JIT-friendly ES loop (`--jit-es`, `--log-every`)

Goal: improve throughput of large experiment batches (and make GPUs viable) without changing the scientific task, by eliminating per-generation host/device synchronization and Python-loop overhead in the ES harness.

Changes:
- `src/koki2/evo/openai_es.py`: added an optional JIT/scan path controlled by `jit_es`, which keeps ES state on-device and writes `generations.jsonl` only after the run (still deterministically keyed by `seed` + generation index).
- `src/koki2/cli.py`: added `koki2 evo-l0 --jit-es` and `--log-every N` (log thinning; also applied to the non-JIT path).

Quick local verification (CPU):
```bash
env JAX_PLATFORM_NAME=cpu uv run koki2 evo-l0 --seed 0 --out-dir runs/2025-12-15_bench_local_cpu_es3_jit_seed0 \
  --jit-es --log-every 1 \
  --generations 3 --pop-size 64 --episodes 4 --steps 128 \
  --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25 \
  --deplete-sources --respawn-delay 4 \
  --grad-dropout-p 0.5
```

Observed output:
- `best_fitness=179.2500`
- `generations.jsonl` has 3 entries (generations 0/1/2).

Interpretation (provisional):
- This makes it possible to re-test RunPod GPU bursts fairly: `--jit-es` should reduce the per-generation sync that likely dominated the earlier GPU wall time.

---

## 2025-12-15 — Harness improvement: rollout fast paths + batch runner ergonomics

Goal: reduce per-rollout overhead and amortize repeated setup work (compilation + process startup) without changing the scientific task.

Changes:
- `src/koki2/agent/snn.py`: added `agent_forward_nolog` and `agent_apply_plasticity_nolog` to avoid computing logging-only statistics in throughput-critical paths.
- `src/koki2/sim/orchestrator.py`: added `simulate_lifetime_fitness` + `simulate_lifetime_mvt` (minimal summaries), used for ES evaluation and MVT checks.
- `src/koki2/evo/openai_es.py`: `evaluate_population` now uses the new fast paths; `run_openai_es` now records `topology_seed` in `config.json` when provided.
- `src/koki2/ops/run_io.py`: added `append_jsonl_many` to reduce per-record file open/close overhead.
- `src/koki2/cli.py`: `evo-l0` passes `topology_seed` through to `run_openai_es`; `batch-evo-l0` uses single-pass JSONL writes and reduced host/device transfers for logs.

Verification:
```bash
uv run pytest
uv run koki2 evo-l0 --seed 0 --jit-es --log-every 1 --generations 2 --pop-size 16 --episodes 2 --steps 32
uv run koki2 batch-evo-l0 --seed-start 0 --seed-count 2 --log-every 1 --generations 2 --pop-size 16 --episodes 2 --steps 32
uv run koki2 eval-run --run-dir runs/2025-12-15_smoke_evo_jit2 --episodes 4 --seed 0 --baseline-policy greedy
```

Notes (provisional):
- These changes should improve throughput on both CPU and GPU by removing logging-only computations from the inner rollout loop (e.g., spike-rate means, per-step plasticity magnitude summaries) and by reducing per-run/per-generation file I/O overhead.
- A dedicated throughput benchmark is still needed to quantify gains and to check for backend-specific regressions.

---

## 2025-12-15 — JAX “sharp bits” + JAXPR audit (harness-focused)

Goal: sanity-check that our “fast path” harness is JAX-friendly (no accidental host callbacks / dynamic-shape traps) and identify obvious hotspots to address next.

Observations (JAXPR; qualitative):
- `simulate_lifetime_fitness` / `simulate_lifetime_mvt` compile to a single `scan` each; no host callbacks showed up.
- `es_run` (JIT+scan ES loop) includes `sort` ops when `log_every` triggers median computation (expected); for large populations, this is a non-trivial logging cost if `--log-every 1`.
- JAX v0.7+ requires primitive parameters (notably shapes) to be hashable; tracing a config object that supplies `episodes`/`steps` as a tracer can fail.

Changes:
- `src/koki2/types.py`: registered `SimConfig`, `MVTConfig`, and `EvalConfig` as static pytrees (like `ChemotaxisEnvSpec`) so shape-like config fields remain static under `jit` / `make_jaxpr`.
- `src/koki2/cli.py`: added `--jax-transfer-guard {log,disallow,...}` to help catch accidental implicit host<->device transfers while optimizing throughput.
- `src/koki2/cli.py`: added debug flags `--jax-log-compiles`, `--jax-explain-cache-misses`, `--jax-debug-nans`, `--jax-disable-jit`, `--jax-check-tracer-leaks`.
- `tests/test_jax_discipline.py`: added CI-style checks for (a) no unexpected recompiles (via jit cache size), (b) tracer-leak detection, and (c) a coarse JAXPR-size budget to catch accidental Python loops/dynamic control flow.

Verification:
```bash
uv run pytest
```

---

## 2025-12-15 — Google Colab bootstrap (GPU/TPU)

Goal: make the repo easy to run in Google Colab (TPU/GPU runtimes) with a notebook-first bootstrap and without requiring a Python 3.12-only kernel.

Changes:
- `pyproject.toml`: broadened `requires-python` to `>=3.10,<3.14` (Colab compatibility) and set Ruff `target-version` to `py310`.
- `src/koki2/ops/run_io.py`: replaced the Python 3.11+ `datetime.UTC` import with `timezone.utc` to keep `utc_now_iso()` working on Python 3.10.
- `colab/koki2_colab.ipynb`: added a Colab notebook that installs the appropriate JAX build for TPU/GPU/CPU and runs a tiny `evo-l0` smoke test.
- `colab/README.md`, `README.md`: added a pointer to the Colab notebook.

Published:
- https://github.com/Krisztiaan/koki2

Verification (local):
```bash
uv run python -c "import json; json.load(open('colab/koki2_colab.ipynb','r',encoding='utf-8'))"
uv run --python 3.10 --extra dev pytest
uv run --python 3.12 --extra dev pytest
```

---

## 2025-12-15 — Stage 1 multi-seed check: L0.2 harmful sources (no plasticity)

Goal: start a Stage 1 “across seeds” check on a non-trivial L0.2 variant (positive + negative sources via integrity loss), and standardize evaluation via `koki2 eval-run` on held-out episode keys.

Environment:
- `--num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25`
- `--steps 128`

Baseline scan (seed 0, 256 episodes; log: `runs/stage1_scans/2025-12-15_baseline_scan.txt`):
```bash
uv run koki2 baseline-l0 --seed 0 --policy greedy --episodes 256 --steps 128 \
  --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25
uv run koki2 baseline-l0 --seed 0 --policy random --episodes 256 --steps 128 \
  --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25
uv run koki2 baseline-l0 --seed 0 --policy greedy --episodes 256 --steps 128 \
  --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25 --good-only-gradient
```

Observed output:
- greedy: `mean_fitness=153.8184`, `success_rate=0.512`, `mean_energy_gained=0.0232`, `mean_bad_arrivals=0.4766`, `mean_integrity_min=0.8809`
- random: `mean_fitness=135.5723`, `success_rate=0.320`, `mean_t_alive=118.9`, `mean_energy_gained=0.0693`, `mean_bad_arrivals=0.9922`, `mean_integrity_min=0.7520`
- greedy + `--good-only-gradient` (informative cue control): `mean_fitness=178.4805`, `success_rate=1.000`, `mean_bad_arrivals=0.3125`, `mean_integrity_min=0.9219`

ES (no plasticity; 30 generations, pop 64, 4 episodes; log: `runs/stage1_scans/2025-12-15_es_badsrc_seed0-2.txt`):
```bash
uv run koki2 batch-evo-l0 \
  --seed-start 0 --seed-count 3 \
  --out-root runs/stage1_es --tag stage1_badsrc \
  --generations 30 --pop-size 64 --episodes 4 --steps 128 \
  --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25 \
  --log-every 1
```

Observed output:
- seed 0: `best_fitness=181.1250`, `out_dir=runs/stage1_es/2025-12-15T1938347374200000_stage1_badsrc_seed0`
- seed 1: `best_fitness=183.1250`, `out_dir=runs/stage1_es/2025-12-15T1938347374200000_stage1_badsrc_seed1`
- seed 2: `best_fitness=182.0000`, `out_dir=runs/stage1_es/2025-12-15T1938347374200000_stage1_badsrc_seed2`

Held-out evaluation (256 episodes, fixed eval seed; log: `runs/stage1_scans/2025-12-15_eval_badsrc_seed0-2.txt`):
```bash
uv run koki2 eval-run --run-dir runs/stage1_es/2025-12-15T1938347374200000_stage1_badsrc_seed0 --episodes 256 --seed 424242 --baseline-policy greedy
uv run koki2 eval-run --run-dir runs/stage1_es/2025-12-15T1938347374200000_stage1_badsrc_seed0 --episodes 256 --seed 424242 --baseline-policy random

uv run koki2 eval-run --run-dir runs/stage1_es/2025-12-15T1938347374200000_stage1_badsrc_seed1 --episodes 256 --seed 424242 --baseline-policy greedy
uv run koki2 eval-run --run-dir runs/stage1_es/2025-12-15T1938347374200000_stage1_badsrc_seed1 --episodes 256 --seed 424242 --baseline-policy random

uv run koki2 eval-run --run-dir runs/stage1_es/2025-12-15T1938347374200000_stage1_badsrc_seed2 --episodes 256 --seed 424242 --baseline-policy greedy
uv run koki2 eval-run --run-dir runs/stage1_es/2025-12-15T1938347374200000_stage1_badsrc_seed2 --episodes 256 --seed 424242 --baseline-policy random
```

Observed (best genome; mean_fitness):
- seed 0: `153.4395` (baseline greedy `154.4160`, baseline random `131.7891`)
- seed 1: `141.8105` (baseline greedy `154.4160`, baseline random `131.7891`)
- seed 2: `150.3379` (baseline greedy `154.4160`, baseline random `131.7891`)

Aggregate (computed from the three values above):
- mean best_genome `mean_fitness=148.5293` (vs baseline random `131.7891`, vs baseline greedy `154.4160`)

Interpretation (provisional):
- At this small budget (30×64×4 episodes), ES clears the random baseline on this hazard variant across seeds, but does not yet beat the greedy-gradient baseline on held-out evaluation.
- The evolved policies appear to trade off **higher energy gained** with **lower survival time** (integrity loss from bad arrivals), suggesting we should consider (a) increasing ES budget, and/or (b) Stage 2 plasticity comparisons where within-life adaptation can respond to “bad arrival” events.

---

## 2025-12-15 — Stage 1 bigger budget: L0.2 harmful sources (no plasticity)

Goal: re-run the same L0.2 harmful-sources variant with a larger ES budget and more seeds, and check whether fixed-weight evolution can surpass the gradient-greedy baseline on held-out episodes.

Thesis grounding:
- The L0.2 spec explicitly warns that in **non-depleting** L0.2, fitness gains can plausibly come even with increased hazard contact; avoidance should be measured (bad arrivals / integrity minima), not inferred from fitness alone (`thesis/12_IMPLEMENTATION_ENVIRONMENT_LADDER_SPEC.md`).

Environment:
- `--num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25`
- `--steps 128`

ES (200 generations, pop 128, 8 episodes; seeds 0..4; log: `runs/stage1_scans/2025-12-15_es_badsrc_big_seed0-4.txt`):
```bash
uv run koki2 batch-evo-l0 \
  --seed-start 0 --seed-count 5 \
  --out-root runs/stage1_es_big --tag stage1_badsrc_g200_p128_ep8 \
  --generations 200 --pop-size 128 --episodes 8 --steps 128 \
  --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25 \
  --log-every 10
```

Observed output (best_fitness + out_dir):
- seed 0: `best_fitness=181.9375`, `out_dir=runs/stage1_es_big/2025-12-15T1959242949320000_stage1_badsrc_g200_p128_ep8_seed0`
- seed 1: `best_fitness=183.1875`, `out_dir=runs/stage1_es_big/2025-12-15T1959242949320000_stage1_badsrc_g200_p128_ep8_seed1`
- seed 2: `best_fitness=181.3125`, `out_dir=runs/stage1_es_big/2025-12-15T1959242949320000_stage1_badsrc_g200_p128_ep8_seed2`
- seed 3: `best_fitness=180.8125`, `out_dir=runs/stage1_es_big/2025-12-15T1959242949320000_stage1_badsrc_g200_p128_ep8_seed3`
- seed 4: `best_fitness=180.6250`, `out_dir=runs/stage1_es_big/2025-12-15T1959242949320000_stage1_badsrc_g200_p128_ep8_seed4`

Held-out evaluation (512 episodes, fixed eval seed; log: `runs/stage1_scans/2025-12-15_eval_badsrc_big_seed0-4.txt`):
```bash
uv run koki2 eval-run --run-dir runs/stage1_es_big/2025-12-15T1959242949320000_stage1_badsrc_g200_p128_ep8_seed0 --episodes 512 --seed 424242 --baseline-policy greedy
uv run koki2 eval-run --run-dir runs/stage1_es_big/2025-12-15T1959242949320000_stage1_badsrc_g200_p128_ep8_seed0 --episodes 512 --seed 424242 --baseline-policy random
```

Observed (best genome; held-out `mean_fitness`, seeds 0..4):
- seed 0: `162.3066`
- seed 1: `163.0488`
- seed 2: `166.0693`
- seed 3: `163.6104`
- seed 4: `166.3066`

Aggregate across seeds 0..4 (computed from the five values above; same eval seed + episode keys):
- best_genome: mean `mean_fitness=164.2683` (sample stdev `1.8143`)
- baseline greedy: `mean_fitness=154.9092`
- baseline random: `mean_fitness=133.9463`

Other held-out metrics (mean across seeds 0..4; computed from `runs/stage1_scans/2025-12-15_eval_badsrc_big_seed0-4.txt`):
- best_genome success_rate: mean `0.7292` (baseline greedy `0.533`, baseline random `0.309`)
- best_genome mean_bad_arrivals: mean `1.3472` (baseline greedy `0.4492`, baseline random `0.9844`)
- best_genome mean_integrity_min: mean `0.6632` (baseline greedy `0.8877`, baseline random `0.7539`)

Robustness check (second held-out eval seed):
- Re-ran the same `best_genome.npz` evaluations with `--seed 0` (512 episodes; log: `runs/stage1_scans/2025-12-15_eval_badsrc_big_seed0-4_evalseed0.txt`).
- Aggregate across seeds 0..4: best_genome mean `mean_fitness=164.1002` (baseline greedy `153.7236`, baseline random `133.8994`).

Interpretation (provisional):
- At this larger budget, fixed-weight ES **beats the greedy-gradient baseline on held-out fitness** across 5 seeds in L0.2 harmful sources.
- Consistent with the L0.2 spec warning, this improvement does **not** imply better avoidance: the evolved policies show **more bad arrivals** and **lower integrity minima** than the greedy baseline on the same held-out episode keys.

---

## 2025-12-15 — Stage 1 (A): L1.0 deplete/respawn + L0.2 harmful sources (no plasticity)

Goal: test the thesis expectation that adding **temporal structure** (L1.0 depleting resources) changes the tradeoffs in L0.2 harmful sources and makes “survive safely” strategies more competitive (relative to naive gradient chasing).

Thesis grounding:
- L1.0 (depleting resources) is intended to encourage exploration and non-trivial policy beyond “always go uphill” (`thesis/12_IMPLEMENTATION_ENVIRONMENT_LADDER_SPEC.md`).
- L0.2 warns that “avoidance” should be measured (bad arrivals / integrity minima), not inferred from fitness alone (`thesis/12_IMPLEMENTATION_ENVIRONMENT_LADDER_SPEC.md`).

Environment:
- `--deplete-sources --respawn-delay 4`
- `--num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25`
- `--steps 128`

Baseline scan (seed 0, 512 episodes; log: `runs/stage1_scans/2025-12-15_baseline_scan_l10_deplete_badsrc.txt`):
```bash
uv run koki2 baseline-l0 --seed 0 --policy greedy --episodes 512 --steps 128 \
  --deplete-sources --respawn-delay 4 \
  --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25
uv run koki2 baseline-l0 --seed 0 --policy random --episodes 512 --steps 128 \
  --deplete-sources --respawn-delay 4 \
  --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25
```

Observed (baseline):
- greedy: `mean_fitness=129.7090`, `success_rate=0.898`, `mean_t_alive=83.1`, `mean_bad_arrivals=3.3945`, `mean_integrity_min=0.1572`
- random: `mean_fitness=145.5098`, `success_rate=0.346`, `mean_t_alive=128.0`, `mean_bad_arrivals=0.4102`, `mean_integrity_min=0.8975`

ES (200 generations, pop 128, 8 episodes; seeds 0..4; log: `runs/stage1_scans/2025-12-15_es_l10_deplete_badsrc_big_seed0-4.txt`):
```bash
uv run koki2 batch-evo-l0 \
  --seed-start 0 --seed-count 5 \
  --out-root runs/stage1_es_big_l10 --tag stage1_l10_deplete_badsrc_g200_p128_ep8 \
  --generations 200 --pop-size 128 --episodes 8 --steps 128 \
  --deplete-sources --respawn-delay 4 \
  --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25 \
  --log-every 10
```

Held-out evaluation (512 episodes, fixed eval seeds; logs):
- `runs/stage1_scans/2025-12-15_eval_l10_deplete_badsrc_big_seed0-4_evalseed424242.txt`
- `runs/stage1_scans/2025-12-15_eval_l10_deplete_badsrc_big_seed0-4_evalseed0.txt`

Aggregate (computed from the 5 held-out runs; eval seed 424242):
- best_genome mean `mean_fitness=164.1209` (baseline random `144.1484`, baseline greedy `130.7910`)
- best_genome mean `mean_t_alive=126.7000` (baseline greedy `82.5`)
- best_genome mean `mean_bad_arrivals=1.1879` and `mean_integrity_min=0.7030` (baseline greedy `3.4375` and `0.1416`)

Robustness (eval seed 0; 5 held-out runs):
- best_genome mean `mean_fitness=162.6670` (baseline random `145.5098`, baseline greedy `129.7090`)

Interpretation (provisional):
- In this L1.0+L0.2 setting, naive greedy gradient following becomes **high-risk** (many bad arrivals → integrity collapse → early death), and random “low-contact” behavior can outscore greedy on mean fitness by surviving.
- Fixed-weight ES adapts by substantially reducing hazard contact relative to greedy (higher integrity minima, near-full survival time), consistent with the thesis expectation that temporal structure amplifies survival-weighted strategies.
- Relative to the random baseline, the evolved policies still accept more hazard contact (lower integrity minima), so “avoidance” remains a separate measured outcome rather than something we can infer from fitness alone.

---

## 2025-12-15 — Stage 2 (B): ES30 × seeds 0..4, held-out 512 episodes (L1.0 deplete + L1.1 intermittent gradient)

Goal: extend the earlier Stage 2 benchmark to 5 seeds and a larger held-out evaluation, to compare **spike-modulated plasticity** vs **no-plastic** under L1.0+L1.1.

Benchmark env (same as earlier Stage 2 runs in this file):
- `--num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25`
- `--deplete-sources --respawn-delay 4`
- `--grad-dropout-p 0.5`
- `--steps 128`

Compute budget (per run):
- `--generations 30 --pop-size 64 --episodes 4`

Runs:
- Reused existing run dirs for seeds 0/1/2:
  - `runs/2025-12-15_stage2_es30_noplast_seed{0,1,2}`
  - `runs/2025-12-15_stage2_es30_plast_eta0.05_seed{0,1}` and `runs/2025-12-15_stage2_es30_plast_eta0.05_seed2_retry1`
- Added seeds 3/4 (no-plastic):
```bash
for seed in 3 4; do
  uv run koki2 evo-l0 --seed $seed --out-dir runs/2025-12-15_stage2_es30_noplast_seed${seed} \
    --generations 30 --pop-size 64 --episodes 4 --steps 128 \
    --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25 \
    --deplete-sources --respawn-delay 4 \
    --grad-dropout-p 0.5
done
```
- Observed:
  - seed 3: `best_fitness=179.5000` (`runs/2025-12-15_stage2_es30_noplast_seed3`)
  - seed 4: `best_fitness=179.5000` (`runs/2025-12-15_stage2_es30_noplast_seed4`)
- Added seeds 3/4 (plastic; spike modulator default; `eta=0.05`, `lambda=0.9`):
```bash
for seed in 3 4; do
  uv run koki2 evo-l0 --seed $seed --out-dir runs/2025-12-15_stage2_es30_plast_eta0.05_seed${seed} \
    --generations 30 --pop-size 64 --episodes 4 --steps 128 \
    --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25 \
    --deplete-sources --respawn-delay 4 \
    --grad-dropout-p 0.5 \
    --plast-enabled --plast-eta 0.05 --plast-lambda 0.9
done
```
- Observed:
  - seed 3: `best_fitness=179.6250` (`runs/2025-12-15_stage2_es30_plast_eta0.05_seed3`)
  - seed 4: `best_fitness=180.1250` (`runs/2025-12-15_stage2_es30_plast_eta0.05_seed4`)

Held-out eval (`koki2 eval-run`; 512 episodes; baseline omitted; logs):
- eval seed 424242: `runs/stage2_scans/2025-12-15_eval_stage2_es30_l10l11_noplast_vs_plast_eta0.05_evalseed424242_ep512.txt`
- eval seed 0: `runs/stage2_scans/2025-12-15_eval_stage2_es30_l10l11_noplast_vs_plast_eta0.05_evalseed0_ep512.txt`

Observed `mean_fitness` per seed (held-out; 512 episodes):
- eval seed 424242:
  - no-plastic: seed 0 `164.2197`, seed 1 `165.4854`, seed 2 `165.1064`, seed 3 `163.0225`, seed 4 `161.7217`
  - plastic: seed 0 `170.8857`, seed 1 `165.4131`, seed 2 `168.1543`, seed 3 `161.1787`, seed 4 `170.5039`
- eval seed 0:
  - no-plastic: seed 0 `163.5801`, seed 1 `162.6572`, seed 2 `163.0166`, seed 3 `163.2158`, seed 4 `162.7012`
  - plastic: seed 0 `170.2354`, seed 1 `162.4463`, seed 2 `168.9834`, seed 3 `162.5176`, seed 4 `170.4082`

Aggregate across seeds 0..4 (computed from the held-out outputs above; sample stdev):
- eval seed 424242:
  - no-plastic: mean `mean_fitness=163.9111` (stdev `1.5480`), `success_rate=0.7360` (stdev `0.0178`), `mean_t_alive=126.5` (stdev `0.8`), `mean_bad_arrivals=1.2199` (stdev `0.0781`), `mean_integrity_min=0.6950` (stdev `0.0195`)
  - plastic: mean `mean_fitness=167.2271` (stdev `4.0267`), `success_rate=0.8344` (stdev `0.0959`), `mean_t_alive=124.7` (stdev `1.8`), `mean_bad_arrivals=1.7242` (stdev `0.5322`), `mean_integrity_min=0.5691` (stdev `0.1331`), `mean_abs_dw_mean=0.014106` (stdev `0.016841`)
- eval seed 0:
  - no-plastic: mean `mean_fitness=163.0342` (stdev `0.3822`), `success_rate=0.7264` (stdev `0.0079`), `mean_t_alive=126.1` (stdev `0.3`), `mean_bad_arrivals=1.2293` (stdev `0.0551`), `mean_integrity_min=0.6927` (stdev `0.0138`)
  - plastic: mean `mean_fitness=166.9182` (stdev `4.0869`), `success_rate=0.8260` (stdev `0.0933`), `mean_t_alive=124.7` (stdev `1.1`), `mean_bad_arrivals=1.7250` (stdev `0.5075`), `mean_integrity_min=0.5691` (stdev `0.1265`), `mean_abs_dw_mean=0.014151` (stdev `0.016862`)

Interpretation (provisional):
- At this ES30 budget on L1.0+L1.1, spike-modulated plasticity (`eta=0.05`) increases held-out mean fitness and success rate **but** also increases hazard contact (more bad arrivals, lower integrity minima) and slightly reduces mean survival time, consistently across both held-out episode sets.
- `mean_abs_dw_mean` is highly seed-dependent (some runs are near “effectively non-plastic”), so future comparisons should keep reporting “plasticity usage” alongside performance/hazard metrics.

---

## 2025-12-15 — Evaluation note: strengthening L1.0 “survival-weighted” evidence

Motivation:
- We can already see L1.0 deplete/respawn pushing evolution away from naive gradient chasing and toward longer survival / reduced hazard contact relative to greedy (`WORK.md` entries above), which matches the L1.0 intent in `thesis/12_IMPLEMENTATION_ENVIRONMENT_LADDER_SPEC.md`.
- To make **stronger, cleaner statements** (and reduce “could this be a loophole?” ambiguity), we want larger effect sizes and fewer confounds in the L1.0+hazards setup.

Potential confound to address:
- In the current L1.0 implementation, **bad sources also deplete immediately on arrival**, meaning an agent can sometimes “pay damage to remove a hazard” (self-damage as a clearing mechanism). This can blur the interpretation of hazard contact vs avoidance.

Decisions / next steps:
- Add an env knob so **bad sources deplete more slowly than good sources** (e.g., a per-arrival deplete probability for bad sources). This keeps tensor shapes unchanged while increasing the pressure for sustained avoidance (hazards persist more).
- Run a small success-bonus ablation (`--success-bonus 0` vs default) under L1.0+hazards to check how much reach-shaping influences risk-taking vs survival-weighted strategies.
- Continue treating “avoidance” as a measured outcome (`mean_bad_arrivals`, `mean_integrity_min`), not inferred from fitness alone.

---

## 2025-12-15 — L1.0 hazard persistence knobs + effect-size sweep (stronger evidence)

Goal:
- Increase the decisiveness of the “L1.0 amplifies survival-weighted strategies” evidence by (a) reducing the “clear hazard by stepping on it” loophole and (b) increasing effect size via a longer horizon.

Implementation (hazard persistence knobs):
- Added:
  - `ChemotaxisEnvSpec.bad_source_deplete_p`: probability that a bad source depletes on arrival (default `1.0` preserves existing behavior).
  - `ChemotaxisEnvSpec.bad_source_respawn_delay`: override respawn delay for bad sources (`-1` uses `source_respawn_delay`).
- CLI flags:
  - `--bad-source-deplete-p`
  - `--bad-source-respawn-delay` / `--bad-respawn-delay`
- Tests:
  - `tests/test_env_depletion.py`: bad-source deplete probability + bad-source respawn delay override.

Quick note (important for interpretation):
- In this environment, if a bad source does **not** deplete on arrival, the gradient at the source location is exactly zero, so the greedy baseline can “camp” and avoid further arrivals. This can make greedy **safer** (but less successful), which is not the intended “amplify avoidance pressure” direction for this sweep.
- For the experiments below, we therefore kept `--bad-source-deplete-p 1.0` (deplete on arrival) and used `--bad-source-respawn-delay 0` to make hazards hard to “clear” (they respawn immediately elsewhere).

Verification:
```bash
uv run pytest
```

### Effect-size sweep: longer horizon + fast bad respawn (ES30 × seeds 0..4)

Benchmark env:
- L1.0 deplete/respawn + L0.2 harmful sources
- longer horizon: `--steps 256`
- hazards persist (hard to clear): `--bad-source-respawn-delay 0`
- hazards still deplete on arrival: `--bad-source-deplete-p 1.0`

Compute budget (per run):
- `--generations 30 --pop-size 64 --episodes 4` (JIT ES; `--jit-es`)

#### A) Default shaping (`--success-bonus 50`)

ES runs (log: `runs/stage1_scans/2025-12-15_es30_l10_badsrc_steps256_badresp0_succ50_seed0-4.txt`):
```bash
uv run koki2 batch-evo-l0 \
  --seed-start 0 --seed-count 5 \
  --out-root runs/stage1_es_l10_stronger \
  --tag stage1_l10_badsrc_steps256_badresp0_g30_p64_ep4_succ50 \
  --generations 30 --pop-size 64 --episodes 4 --steps 256 \
  --deplete-sources --respawn-delay 4 --bad-source-respawn-delay 0 \
  --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25 \
  --bad-source-deplete-p 1.0 \
  --success-bonus 50.0 \
  --jit-es --log-every 10
```

Baselines (512 episodes; eval seeds 424242 and 0; log: `runs/stage1_scans/2025-12-15_baselines_l10_badsrc_steps256_badresp0_succ50_ep512.txt`):
- eval seed 424242:
  - greedy: `mean_fitness=147.9619`, `mean_t_alive=101.3`, `mean_bad_arrivals=3.4805`, `mean_integrity_min=0.1318`
  - random: `mean_fitness=277.9775`, `mean_t_alive=255.4`, `mean_bad_arrivals=0.5918`, `mean_integrity_min=0.8521`

Held-out best-genome eval (512 episodes; logs):
- eval seed 424242: `runs/stage1_scans/2025-12-15_eval_es30_l10_badsrc_steps256_badresp0_succ50_evalseed424242_ep512.txt`
- eval seed 0: `runs/stage1_scans/2025-12-15_eval_es30_l10_badsrc_steps256_badresp0_succ50_evalseed0_ep512.txt`

Aggregate across seeds 0..4 (computed from the held-out logs above; sample stdev):
- eval seed 424242:
  - best_genome: mean `mean_fitness=286.9103` (stdev `2.0329`), `success_rate=0.7322` (stdev `0.0202`), `mean_t_alive=249.7` (stdev `1.2`), `mean_bad_arrivals=1.2730` (stdev `0.0556`), `mean_integrity_min=0.6818` (stdev `0.0138`)

Interpretation (provisional):
- This longer-horizon L1.0 setup makes the survival tradeoff much more decisive: greedy dies very early on average, while ES best-genomes survive nearly the full horizon and substantially reduce hazard contact relative to greedy (fewer bad arrivals; much higher integrity minima).
- Best-genomes also surpass the random baseline on held-out mean fitness in this particular configuration, primarily via much higher success rates (while still being riskier than random in hazard metrics).

#### B) Success-bonus ablation (`--success-bonus 0`)

ES runs (log: `runs/stage1_scans/2025-12-15_es30_l10_badsrc_steps256_badresp0_succ0_seed0-4.txt`):
```bash
uv run koki2 batch-evo-l0 \
  --seed-start 0 --seed-count 5 \
  --out-root runs/stage1_es_l10_stronger \
  --tag stage1_l10_badsrc_steps256_badresp0_g30_p64_ep4_succ0 \
  --generations 30 --pop-size 64 --episodes 4 --steps 256 \
  --deplete-sources --respawn-delay 4 --bad-source-respawn-delay 0 \
  --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25 \
  --bad-source-deplete-p 1.0 \
  --success-bonus 0.0 \
  --jit-es --log-every 10
```

Baselines (512 episodes; eval seed 424242; log: `runs/stage1_scans/2025-12-15_baselines_l10_badsrc_steps256_badresp0_succ0_ep512.txt`):
- random: `mean_fitness=255.7119`, `mean_t_alive=255.4`

Held-out best-genome eval (512 episodes; logs):
- eval seed 424242: `runs/stage1_scans/2025-12-15_eval_es30_l10_badsrc_steps256_badresp0_succ0_evalseed424242_ep512.txt`
- eval seed 0: `runs/stage1_scans/2025-12-15_eval_es30_l10_badsrc_steps256_badresp0_succ0_evalseed0_ep512.txt`

Aggregate across seeds 0..4 (computed from the held-out logs above; sample stdev):
- eval seed 424242:
  - best_genome: mean `mean_fitness=247.5529` (stdev `7.4015`), `mean_t_alive=246.8` (stdev `7.7`), `mean_bad_arrivals=1.4922` (stdev `0.5538`), `mean_integrity_min=0.6270` (stdev `0.1384`)

Interpretation (provisional):
- With `success_bonus=0`, the random baseline (near-full survival) is harder to beat on fitness in this long-horizon setup; the ES best-genomes are lower-fitness on average (and show substantially higher variance across seeds).
- This suggests the reach/success shaping term is not just a cosmetic metric tweak: it meaningfully changes which strategies are selected under fixed compute, and should be treated as an explicit experimental axis (report both `success_bonus` setting and hazard metrics when comparing “survival-weighted” claims).

---

## 2025-12-15 — Pre-registered mini-grid: steps × bad-respawn × success-bonus (L1.0 + harmful sources)

Goal:
- Run a small, pre-registered grid to test whether “L1.0 amplifies survival-weighted strategies” is robust across (a) horizon length, (b) hazard persistence, and (c) success shaping — and to get stronger, more general statements than a single hand-picked configuration.

Protocol:
- Env base (all conditions):
  - `--deplete-sources --respawn-delay 4`
  - `--num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25`
  - `--bad-source-deplete-p 1.0` (avoid the “camp on a non-depleting hazard” confound noted above)
- Grid:
  - `--steps ∈ {128, 256}`
  - `--bad-source-respawn-delay ∈ {0, 1, 4}`
  - `--success-bonus ∈ {0, 50}`
- ES budget (per condition): `--generations 30 --pop-size 64 --episodes 4` (JIT ES; `--jit-es`, `--log-every 10`)
- Evaluation:
  - best genomes: `koki2 eval-run` with `--episodes 512` at eval seeds `424242` and `0`
  - baselines: `koki2 baseline-l0` greedy/random on the same eval seeds/episodes

Runtime (local CPU):
- End-to-end wall time was `12.51 min` for all 12 conditions (including training, baselines, and held-out eval).

Artifacts:
- Full stdout log: `runs/stage1_scans/2025-12-15_grid_l10_effectsize_es30_p64_ep4_eval512_2025-12-15_231730.txt`
- Structured results (one JSON per baseline/best-genome eval): `runs/stage1_scans/2025-12-15_grid_l10_effectsize_es30_p64_ep4_eval512_2025-12-15_231730.jsonl`

Key observed pattern (eval seed 424242; aggregated across seeds 0..4 per condition):
- Across all 12 conditions:
  - best-genome mean survival time `mean_t_alive` ranges `125.28–251.26`, while greedy ranges `77.8–103.9` (random stays near the horizon: `127.8–255.4`).
  - best-genome `mean_bad_arrivals` ranges `1.1914–1.4922`, while greedy ranges `3.4082–3.5469` (random ranges `0.4062–0.6230`).
  - best-genome `mean_integrity_min` ranges `0.6270–0.7021`, while greedy ranges `0.1196–0.1499` (random ranges `0.8442–0.8984`).
- Put differently: for every grid point, ES best-genomes survive much longer and contact hazards much less than greedy; relative to greedy, the per-condition differences are large:
  - `mean_bad_arrivals` improvement (greedy − best) ranges `1.9883–2.3102`
  - `mean_integrity_min` improvement (best − greedy) ranges `0.4952–0.5712`
  - `mean_t_alive` improvement (best − greedy) ranges `43.88–153.46` steps

Success-bonus ablation (eval seed 424242; holds for all 6 grid points per horizon):
- With `--success-bonus 50`, ES best-genomes beat the random baseline on mean fitness in **all** tested grid points.
- With `--success-bonus 0`, ES best-genomes are below the random baseline on mean fitness in **all** tested grid points (random’s near-full survival dominates fitness when reaching sources carries no explicit bonus).

Interpretation (provisional):
- This grid supports a stronger statement than our earlier single-run comparisons: in L1.0 deplete/respawn + harmful sources, evolution reliably shifts toward substantially higher survival and lower hazard contact than naive greedy gradient chasing, and this holds across multiple hazard-persistence settings and horizons at a fixed ES budget.
- The `success_bonus` term is a major selection-axis: it changes whether “survive + reach” strategies can beat “survive only” strategies under fixed compute; we should therefore keep reporting it explicitly in any claim about “survival-weighted strategy amplification”.
