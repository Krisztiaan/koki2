# koki2

Thesis-aligned research code for *Biologically-Constrained Neuroevolution / Evolvable Plastic Agents*.

Primary design goals:
- JAX-first simulation (`jit`/`vmap`/`scan`)
- deterministic replay (seed discipline)
- incremental “environment ladder” experiments starting from L0 chemotaxis

## Quickstart (uv)

This repo assumes a supported Python (recommended: 3.12). On this machine the system Python is 3.14, so create a 3.12 venv via `uv`:

```bash
uv python install 3.12
uv venv --python 3.12
UV_LINK_MODE=copy uv pip install -e '.[dev]'
uv run pytest
```

Run a tiny L0 evolution smoke test:

```bash
uv run koki2 evo-l0 --generations 5 --pop-size 64 --steps 128
```

## Performance notes

For longer runs and GPU backends, use the JIT-friendly ES loop to avoid per-generation host/device synchronization:

```bash
uv run koki2 evo-l0 --jit-es --log-every 5 --generations 200 --pop-size 256 --steps 128
```

For multi-seed sweeps, amortize compilation and process startup by running many seeds in a single process:

```bash
uv run koki2 batch-evo-l0 --seed-start 0 --seed-count 8 --log-every 5 --generations 200 --pop-size 256 --steps 128
```

The CLI enables a persistent JAX compilation cache by default (via `JAX_COMPILATION_CACHE_DIR`). Override with `--jax-cache-dir <path>` or disable with `--no-jax-cache`.

To detect accidental host/device boundary crossings while iterating on the harness, set `--jax-transfer-guard log` (or `disallow` once things are stable).

To hunt accidental recompiles, enable `--jax-log-compiles --jax-explain-cache-misses`. For debugging NaNs and tracer leaks, use `--jax-debug-nans` and `--jax-check-tracer-leaks` (both are slow and intended for debugging).

Evaluate the saved `best_genome.npz` on more episodes (recommended for interpretation; compares to a baseline on the same episode keys):

```bash
uv run koki2 eval-run --run-dir runs/<timestamp>_evo-l0_seed0 --episodes 64 --seed 0 --baseline-policy greedy
```

Run baseline policies for comparison:

```bash
uv run koki2 baseline-l0 --policy greedy --episodes 32 --steps 128
uv run koki2 baseline-l0 --policy random --episodes 32 --steps 128
```

Try L0.2 / L1.0 variants:

```bash
# Multi-source (L0.2)
uv run koki2 baseline-l0 --policy greedy --episodes 32 --num-sources 4

# Positive+negative sources via integrity loss (L0.2 variant)
uv run koki2 baseline-l0 --policy greedy --episodes 32 --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25

# Control: make the gradient point only to good sources (informative cue)
uv run koki2 baseline-l0 --policy greedy --episodes 32 --num-sources 4 --num-bad-sources 2 --bad-source-integrity-loss 0.25 --good-only-gradient

# Deplete/respawn temporal structure (L1.0)
uv run koki2 baseline-l0 --policy greedy --episodes 32 --deplete-sources --respawn-delay 4
```

Outputs land in `runs/` (JSONL logs + a manifest for replay).
