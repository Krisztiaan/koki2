# Theory foundations: inside-out and first-principles commitments

## 1. Why inside-out matters for this thesis

Buzsáki’s “inside-out” perspective argues that brains are not primarily stimulus-driven encoding devices. Rather, brains are **self-organized dynamical systems** that generate patterns of activity and actions; the world constrains and calibrates these patterns via the consequences of action [@buzsaki2019brain].

For this thesis, adopting inside-out is not a philosophical ornament—it changes *what we build* and *how we interpret results*:

- We prioritize **action and survival** (homeostasis) over supervised label tasks.
- We treat “representations” as **action hypotheses** and policy states rather than as static encodings.
- We explicitly analyze **spontaneous dynamics** and how experience calibrates them.

---

## 2. Operational commitments derived from inside-out

### 2.1 Action-first environment design

We design tasks where the only reliable way to acquire “meaning” is through **acting**:

- objects may be perceptually ambiguous but have different action consequences,
- sensor inputs change as a consequence of self-motion (active sensing),
- “reward” is computed from internal drives rather than externally labeled success.

This reduces the likelihood that agents solve tasks by learning brittle stimulus–response shortcuts.

### 2.2 Spontaneous dynamics is a first-class object

Inside-out predicts that internal dynamics exist before they are “about” anything. Therefore we include explicit experimental probes:

- **spontaneous runs** with muted or simplified inputs,
- **calibration runs** where the same agent is placed into structured environments.

We then test:

- does the agent exhibit stable internal sequences/assemblies even without strong sensory drive?
- does experience selectively stabilize/reuse certain sequences?

These probes are essential to avoid retrofitting outside-in “coding” interpretations onto the results.

### 2.3 “Meaning” is defined by consequences for viability

We adopt the thesis’ core grounding:

- internal variables define viability,
- actions are meaningful insofar as they reduce drive (restore homeostasis).

This is consistent with the claim that cognition is embedded in organismal needs rather than in abstract tasks.

---

## 3. First-principles baseline: what we refuse to pre-specify

A major risk when trying to “build brains” is embedding the very mechanisms we later claim have emerged. This concern is aligned with critiques of “pure learning” approaches that ignore strong biological priors and developmental programs [@zador2019critique]. Strain A is therefore defined by explicit omissions:

- no hard-coded excitatory/inhibitory labels (weight signs can self-organize),
- no explicit dopamine-like reward prediction error circuits,
- no explicit hippocampus-like memory module,
- no enforced small-worldness or column structure.

Instead, Strain A provides only:

1. a minimal recurrent substrate capable of rich dynamics,
2. a local plasticity family with eligibility traces and modulatory gating,
3. a genome that encodes **rules for development** rather than weight tables,
4. environments that impose survival pressures.

All additional “bio realism” enters as controlled strains (B/C/D), not as baseline.

---

## 4. What “alignment with inside-out” will look like in results

We define several *interpretability checkpoints* that, if achieved, provide evidence the system is inside-out consistent:

### 4.1 Intrinsic sequence structure

- stable trajectories in neural state space (e.g., repeating motifs),
- phase-dependent activity patterns not trivially explained by current observation alone.

### 4.2 Action-conditioned prediction structure

- modulatory signals correlate with unexpected changes in internal drive (surprise),
- plasticity events align with action-outcome contingencies rather than raw stimuli.

### 4.3 Contextual mode switching

- the same sensory inputs produce different action policies depending on internal state and history,
- hysteresis effects indicating discrete “modes” (foraging vs defensive).

---

## 5. Practical guidance: how not to slide back into outside-in

The system architecture may still look like classic RL loops (observation→action), because that is computationally convenient. To keep the science aligned:

- Use analysis that conditions on **internal state and action history**, not only current sensory inputs.
- Prefer tasks that **cannot** be solved by direct mapping from input to action without memory.
- Include **spontaneous dynamics probes** as standard evaluation, not post-hoc.

---

## 6. Relation to other theoretical framings

Inside-out is compatible with several other frameworks often cited in biologically grounded AI:

- **Active inference / free energy principle** as a normative account of action-perception loops [@friston2010freeenergy].
- **Predictive coding** as a local learning mechanism for reducing prediction errors in hierarchical generative models [@rao1999predictivecoding].
- **Enactive / embodied cognition** as the view that cognition is constituted by organism–environment coupling.

This thesis does not need to endorse any one formalism. Instead, it uses inside-out as an empirical stance:

> build a substrate where internal dynamics and action consequences are primary; then measure what emerges.

---

## 7. Scope note

Inside-out arguments can tempt overreach into claims about consciousness or human-level cognition. This thesis remains focused on:

- emergence of learning and memory mechanisms under survival pressures,
- controlled comparative effects of constraints and scaffolding.

Multi-agent social cognition is explicitly deferred (see `19_FUTURE_WORK_AND_VALIDATION.md`).
