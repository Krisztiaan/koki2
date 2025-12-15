# PLAN (living) — koki2 thesis research implementation

This is the working execution plan for building the thesis-aligned research project in incremental, verifiable stages.

Guiding principles:
- **Incremental gates:** each stage adds one major capability and has explicit acceptance tests (`thesis/18_EXPERIMENTS_AND_MILESTONES.md`).
- **Determinism-first:** every new mechanism must preserve reproducible rollouts and stable `jit`.
- **Local-first:** keep workloads small enough to iterate quickly on the current MacBook; scale out only after correctness is locked in.

---

## 0. Environment + tooling

This repo pins Python 3.12 via `.python-version` (the system Python may differ). We use `uv` + a local venv:

```bash
uv python install 3.12
uv venv --python 3.12
UV_LINK_MODE=copy uv pip install -e '.[dev]'
uv run pytest
```

Notes:
- `uv python install 3.12`: ensures a 3.12 interpreter exists for this project (independent of system Python).
- `uv venv --python 3.12`: creates `.venv/` and binds it to that interpreter (reproducible across machines).
- `uv pip install -e '.[dev]'`: installs the project in editable mode plus dev deps (tests, lint tooling).
- `UV_LINK_MODE=copy`: avoids hardlinks/symlinks edge cases on some filesystems and makes the venv more “portable”.

---

## 1. Current status (snapshot)

	Implemented (verified by unit tests):
	- Stage 0: deterministic rollouts + JIT sanity tests (`tests/test_stage0_determinism.py`)
		- Stage 1: L0 chemotaxis env suite + baseline harness + minimal OpenAI-ES loop (`koki2 evo-l0`, `koki2 baseline-l0`)
		  - L0.2: multi-source support (`--num-sources`)
		  - L0.2 variant: positive+negative sources via integrity loss (`--num-bad-sources`, `--bad-source-integrity-loss`)
		  - L1.0: optional deplete/respawn temporal structure (`--deplete-sources`, `--respawn-delay`)
		  - L1.1 (partial): intermittent gradient sensing via `--grad-dropout-p` (no shape changes)
		  - Evaluation: `koki2 eval-run` (evaluate `best_genome.npz` on held-out episodes + compare to baselines)
	- Stage 4 (partial): nursing sub-mechanism for **developmental sensory gating + resolution** in L0 (`src/koki2/envs/chemotaxis.py`, `src/koki2/nursing/schedules.py`)
	- Stage 5 (partial): **MVT viability filtering** for ES (action entropy + alive steps + energy gained) (`src/koki2/evo/openai_es.py`)

	Not implemented yet (planned):
	- Plasticity-enabled agents (eligibility traces, modulators) and comparisons (Stage 2)
	- CPPN/rule genome compiler + bottleneck scaling experiments (Stage 3)
	- L1+ environments beyond L1.1 (obstacles, richer partial observability) and the full L2/L3 ladder (homeostasis, threats) (Stages 2,6,7 depend on these)
	- Multi-fidelity rungs beyond MVT + novelty safeguards (Stage 5)

---

## 2. Runbook (small local workloads)

### 2.0 Baseline checks (fast)

Compare simple baselines before/after changes:

```bash
uv run koki2 baseline-l0 --policy greedy --episodes 32
uv run koki2 baseline-l0 --policy random --episodes 32
uv run koki2 baseline-l0 --policy stay --episodes 32
```

### 2.1 Tiny ES smoke test (fast)

```bash
uv run koki2 evo-l0 --generations 5 --pop-size 64 --steps 128
```

	Artifacts:
	- `runs/<timestamp>_evo-l0_seed<seed>/config.json`
	- `runs/<timestamp>_evo-l0_seed<seed>/generations.jsonl`
	- `runs/<timestamp>_evo-l0_seed<seed>/best_genome.npz`
	- `runs/<timestamp>_evo-l0_seed<seed>/manifest.json`

### 2.1.1 Evaluate a saved run directory (recommended)

ES reports `best_fitness` on its small training objective (`--episodes`, often 4). For interpretation and comparisons, evaluate the saved `best_genome.npz` on more episodes and report hazard metrics too:

```bash
uv run koki2 eval-run --run-dir runs/<timestamp>_evo-l0_seed<seed> --episodes 64 --seed 0 --baseline-policy greedy
```

### 2.2 Sensory gating experiments (L0)

Example: start with coarse/low-gain gradient sensing, mature to full precision:

```bash
uv run koki2 evo-l0 \
  --grad-gain-min 0.0 --grad-gain-max 1.0 --grad-gain-start-phi 0.0 --grad-gain-end-phi 0.3 \
  --grad-bins-min 2   --grad-bins-max 16  --grad-bins-start-phi 0.0 --grad-bins-end-phi 0.6
```

### 2.3 MVT (viability filtering) experiments

```bash
uv run koki2 evo-l0 --mvt --mvt-steps 64 --mvt-episodes 2 --mvt-min-alive-steps 32
```

Diagnostics to watch:
- `mvt_pass_rate` in `generations.jsonl`
- whether best fitness improves with/without MVT at fixed compute

### 2.4 L0.2 and L1.0 variants (quick toggles)

Harmful sources (integrity loss on arrival):

```bash
uv run koki2 baseline-l0 --policy greedy --episodes 64 --steps 128 \
  --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25
```

Control: make the gradient point only to good sources (informative cue; removes consequence-driven discrimination pressure):

```bash
uv run koki2 baseline-l0 --policy greedy --episodes 64 --steps 128 \
  --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25 --good-only-gradient
```

	Deplete/respawn (temporal structure):

	```bash
	uv run koki2 baseline-l0 --policy greedy --episodes 64 --steps 128 \
	  --deplete-sources --respawn-delay 4
	```

	Intermittent gradient (partial observability; L1.1):

	```bash
	uv run koki2 baseline-l0 --policy greedy --episodes 64 --steps 128 --grad-dropout-p 0.5
	```

### 2.5 Burst benchmarking on a temporary RunPod pod (tear-down by default)

This repo is local-first, but for occasional throughput checks you can spin up a short-lived GPU pod, run a batch command, pull back artifacts, and tear the pod down automatically:

```bash
tools/runpod_burst_bench.sh --gpu-type 'NVIDIA GeForce RTX 3090' \
  --image 'runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel-ubuntu22.04' \
  --fetch runs \
  -- uv run koki2 batch-evo-l0 --seed-count 3 --generations 50 --jit-es
```

Artifacts are downloaded under `runs/runpod_burst/` along with `manifest.txt` and the remote stdout/stderr log.

---

## 3. Incremental stage plan (dev + tests + verification)

The stage names align with `thesis/18_EXPERIMENTS_AND_MILESTONES.md`.

### Stage 0 — Infrastructure and determinism (done)

Acceptance checks:
- `uv run pytest -k stage0`
- `jit` and eager match for `simulate_lifetime`

### Stage 1 — L0 baseline competence (in progress)

Goal: reliable improvement over random baselines across seeds.

Next dev tasks (incremental):
- Keep a repeatable multi-seed acceptance workflow (ES vs baselines across ≥3 seeds) and record results in `WORK.md`.
- Standardize evaluation to reduce misinterpretation: evaluate saved `best_genome.npz` on held-out episodes (e.g., 64/128/256) and report hazard metrics (bad arrivals, integrity minima), not just `best_fitness`.
- Add a throughput micro-benchmark (steps/sec) to detect regressions (local CPU).

Latest check (2025-12-15; see `WORK.md` for full commands/output):
- L0.2 harmful sources variant: `--num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25`
- ES budget: `--generations 200 --pop-size 128 --episodes 8` across seeds 0..4.
- Held-out eval: `koki2 eval-run --episodes 512 --seed 424242`.
- Observed: mean best-genome held-out `mean_fitness=164.2683` (baseline random `133.9463`, baseline greedy `154.9092`).
- Robustness: with `koki2 eval-run --seed 0` (512 episodes), mean best-genome held-out `mean_fitness=164.1002` (baseline greedy `153.7236`).

Acceptance checks:
- Across ≥3 seeds, best fitness improves over initial/random baseline.
- No JAX shape instability when toggling nursing/MVT options.

### Stage 2 — Plasticity sanity and benefits (planned)

Goal: demonstrate that within-life plasticity helps in L1 (noise/depletion/partial observability).

Dev tasks:
- Expose plasticity knobs on the CLI (`--plast-enabled`, `--plast-eta`, `--plast-lambda`) and ensure they are recorded in manifests for replay.
- Add consequence-aligned neuromodulation as a tunable primitive (`--modulator-kind {spike,drive,event}` with drive/event-derived signals).
- Add focused unit tests for plasticity gating (disabled ⇒ no weight/trace updates; enabled ⇒ bounded updates) and rollout stability (`isfinite`).
- Add reporting for plasticity usage (e.g., `mean_abs_dw_mean`) so “plastic runs” can be checked for actually applying within-life updates.
- Pre-register a small comparison protocol: fixed compute budget, multi-seed ES runs with/without plasticity on L1.0/L1.1 variants; evaluate saved `best_genome.npz` via `koki2 eval-run` on held-out episodes.

Acceptance checks:
- Plastic agents outperform fixed-weight agents on L1 across seeds (pre-registered metrics).
- Plasticity does not destabilize rollouts (no NaNs; bounded internal states).

### Stage 3 — Genomic bottleneck scaling (planned)

Goal: swap direct weight encoding for CPPN/rule encoding and evaluate scaling.

Dev tasks:
- Add genome compiler module (CPPN → sparse edges + rule parameters).
- Add deterministic “develop(genome)” tests (same genome/seed → same phenotype).
- Run scaling sweep over N/E for direct vs bottleneck.

Acceptance checks:
- Comparable (or better) performance at larger N vs direct baseline.
- Mutation robustness improves (lower catastrophic failure rate).

### Stage 4 — Nursing integration (partial, planned to extend)

Goal: integrate DevelopmentState schedules beyond sensory gating (motor gating, resource/hazard schedules, plasticity schedules).

Dev tasks:
- Standardize schedule utilities and ensure all schedules are shape-static.
- Add ablations toggling each nursing factor independently.

Acceptance checks:
- MVT pass rate improves under nursing (but adult performance does not collapse).
- Evidence of “Goldilocks” regime (too weak/too strong both worse).

### Stage 5 — Pruning and multi-fidelity (partial, planned to extend)

Goal: add rungs beyond MVT and novelty safeguards without biasing results.

Dev tasks:
- Add rung-1 proxy evaluation and promotion policy.
- Add novelty descriptor + archive (MAP-Elites-ish or novelty quota).

Acceptance checks:
- Compute savings measurable at fixed budget.
- Final performance not degraded; diversity preserved.

---

## 4. Verification workflow (what we run before scaling up)

Per PR-sized change:
1. `uv run pytest`
2. A tiny ES smoke run (few generations) to catch tracing/shape regressions.
   - Prefer `--jit-es` when testing GPU/backends to avoid per-generation host sync overhead.
   - For multi-seed sanity checks, prefer `koki2 batch-evo-l0` to amortize compilation.
3. If touching determinism or RNG: rerun `tests/test_stage0_determinism.py`.
4. If touching JAX/jit structure: check `tests/test_jax_discipline.py` for tracer leaks/recompile guards.

Before moving to distributed/GPU:
- Freeze a small set of “golden” configs + seeds and verify exact-match metrics on the laptop.
- Add a minimal benchmark script and record baseline throughput on CPU.

---

## 5. Experiment matrix (near-term, thesis-relevant)

Sensory gating × MVT (L0, fixed compute budget):
- Baseline: no gating, no MVT
- Gating only: vary (gain schedule, bins schedule)
- MVT only: vary thresholds to measure false negatives
- Gating + MVT: test whether gating reduces false negatives and improves search throughput

Primary metrics:
- best/median fitness per generation
- MVT pass rate
- action entropy + mode fraction
- compute proxy: rollouts evaluated per wall-clock second (local)

---

## 6. Open questions to resolve with you (before next implementation sprint)

1. For L0.2 positive+negative: should we add an explicit “hazard smell” channel (nursing-scheduled cue strength), or keep valence ambiguity until L1.1?
2. Do we want sensory gating to remain an **environment-side nursing** factor only, or also become an **agent-evolvable developmental program** (genome controls schedule parameters)?
3. Next L1 pressure to prioritize: observation noise/intermittency, or simple obstacles?
