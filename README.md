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

# Deplete/respawn temporal structure (L1.0)
uv run koki2 baseline-l0 --policy greedy --episodes 32 --deplete-sources --respawn-delay 4
```

Outputs land in `runs/` (JSONL logs + a manifest for replay).
