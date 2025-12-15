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
