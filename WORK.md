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
