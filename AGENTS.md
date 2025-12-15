# AGENTS.md — koki2 working conventions (canonical)

This repository implements a thesis-aligned research codebase for **biologically-constrained neuroevolution / evolvable plastic agents**.

Future agents should use this file as the “how to work here” guide: how to interpret the thesis, how to plan, and how to log work.

---

## 1) Canonical sources of truth

- **This folder is canonical.** Do not work in alternate/mounted copies.
- **Thesis (design + intent):** `thesis/`
  - Stage gates and acceptance tests: `thesis/18_EXPERIMENTS_AND_MILESTONES.md`
  - Environment ladder: `thesis/12_IMPLEMENTATION_ENVIRONMENT_LADDER_SPEC.md`
  - Nursing / developmental niche: `thesis/08_ADDENDUM_DEVELOPMENTAL_NICHE_NURSING.md`
  - Pruning / multi-fidelity: `thesis/09_ADDENDUM_VIABILITY_PRUNING_AND_MULTI_FIDELITY.md`
- **Living execution plan:** `PLAN.md`
- **Living lab notebook:** `WORK.md`

When unclear, prefer the thesis documents over ad-hoc intuition, and prefer `PLAN.md`/`WORK.md` over memory.

---

## 2) How to use the thesis concepts in code (practical translation)

The thesis is implemented as an **incremental environment + mechanism ladder**:

- **Environment ladder** (L0→L5): add *one major pressure at a time*.
- **Mechanism ladder** (Stage 0→7): add *one major capability at a time*.
- **Nursing**: schedule difficulty/capabilities over developmental phase `phi` without changing tensor shapes.
- **Pruning / MVT**: filter obviously non-viable genomes early, but safeguard against false negatives (esp. under nursing).

Implementation rule of thumb:
- If a new feature changes shapes dynamically or breaks `jax.jit`, it’s not “stage-ready”.
- Prefer *ablatable knobs* (config fields + CLI flags) over hard-coded behavior.

---

## 3) Planning workflow (PLAN.md)

`PLAN.md` is the living contract for “what we do next”.

When you start a non-trivial chunk of work:
1. Identify the **next stage gate** from `thesis/18_EXPERIMENTS_AND_MILESTONES.md`.
2. Add/adjust a small set of **incremental tasks** in `PLAN.md` that:
   - have explicit acceptance checks,
   - minimize surface area (touch fewer modules),
   - preserve determinism + `jit`.
3. Prefer the “idiomatic next step”: something later steps can build on with minimal rewrite (e.g., add an env mechanic before adding learning).

When a plan item is completed:
- update `PLAN.md` “Current status” and the runbook if new CLI flags / workflows exist.

---

## 4) Work logging workflow (WORK.md)

`WORK.md` is the lab notebook. Every meaningful change should get a short entry.

Each entry should include:
- **Date + title**
- **Goal** (why this exists in the thesis progression)
- **Changes** (file-level bullet list)
- **Verification** (what tests/commands were run)
- **Results** (only if you actually ran them)
- **Interpretation** (clearly marked as provisional if not replicated)

Important: do not invent numbers. If you report metrics, include:
- command used (or enough detail to reproduce),
- which run directory (if applicable),
- and the exact values observed.

---

## 5) Verification standards (determinism-first)

Before saying something is “done”:
- Run `uv run pytest`.
- If you touched env dynamics, RNG, or rollout logic: ensure eager and `jax.jit` match (existing tests cover this; add more if needed).
- Prefer small, fast smoke runs (few generations/episodes) over long runs during iteration.

Do not rely on “no errors” alone—check semantic requirements (e.g., does the new env mechanic actually remove the exploit it was meant to remove?).

---

## 6) Tooling (uv + venv)

Use `uv` with a project-local venv:

```bash
uv python install 3.12
uv venv --python 3.12
UV_LINK_MODE=copy uv pip install -e '.[dev]'
uv run pytest
```

---

## 7) Command avoidance

- Avoid running development servers unless explicitly requested.
- Prefer unit tests + small CLI runs.

