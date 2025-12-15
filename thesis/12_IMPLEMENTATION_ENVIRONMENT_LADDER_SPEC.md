# Environment ladder specification (granular progression)

## 1. Purpose

This ladder is the backbone of incremental verification. Each level:

- introduces **one major new pressure**,
- defines clear acceptance tests,
- and motivates specific hypotheses about what should emerge in agents.

The ladder is designed to avoid “jumping” directly to tasks that require many mechanisms simultaneously.

Nursing (developmental scaffolding) applies within each level by scheduling environment parameters over developmental phase \(\phi\).

---

## 2. Shared environment substrate

All levels share the following common substrate.

### 2.1 World state

- 2D continuous or grid world (choose one early; grid is simpler and faster)
- agent pose \(x_t\) (and heading if needed)
- resource fields and hazard fields as composable components

### 2.2 Internal state

At minimum:

- energy \(E\in[0,1]\)
- integrity \(I\in[0,1]\)

Additional variables introduced later:

- hydration \(W\)
- temperature \(T\) (bounded band)
- fatigue \(F\) (optional)

### 2.3 Observations

Observation vector is composed of:

- exteroceptive local sensors (gradients, raycasts)
- interoceptive internal state readings (possibly noisy)

### 2.4 Actions

Minimal action set:

- movement: \((\Delta x, \Delta y)\) or discrete moves
- consume / interact (optional)
- rest (optional)

Motor maturation gating is handled by nursing schedules, not by separate env levels.
Sensory availability / resolution gating can be handled the same way (as an ablatable nursing factor), without changing observation tensor shapes.

---

## 3. Level L0 — Chemotaxis (reactive control)

### L0.0: 1D gradient

- single energy source producing a smooth gradient
- no obstacles, no hazards
- objective: reach the source before energy decays to 0

Acceptance criteria:
- a fixed-weight baseline controller can succeed (sanity)
- evolution can produce reliable source finding under multiple seeds

Key measurements:
- time-to-source
- path efficiency

### L0.1: 2D gradient

- same but 2D radial gradient

### L0.2: Multiple sources (positive and negative)

- some sources increase energy, others decrease integrity
- forces basic discrimination through action consequences

Hypothesis:
- agents begin to show avoidance behavior driven by integrity preservation.

---

## 4. Level L1 — Temporal structure (memory is useful)

### L1.0: Depleting resources

- energy sources deplete after consumption
- respawn after a delay in new locations

Pressure:
- encourages exploration and non-trivial policy beyond “always go uphill”.

### L1.1: Noisy gradients / partial observability

- gradient sensors are noisy or intermittent
- encourages temporal integration

### L1.2: Simple obstacles

- corridors or walls force detours

Hypotheses:
- recurrent dynamics and/or plasticity improves performance
- internal state begins to correlate with exploration/exploitation shifts

Acceptance criteria:
- plastic agents outperform fixed-weight agents on average across seeds.

---

## 5. Level L2 — Multi-resource homeostasis

Introduce multiple internal variables and trade-offs.

### L2.0: Energy + hydration

- two resource types in different locations
- both decay over time; death if either hits 0

Pressure:
- requires prioritization and switching between objectives.

### L2.1: Trade-offs

- some actions restore one variable while harming another (e.g., salty water)
- introduces non-trivial decision policies

### L2.2: Resource distribution shifts

- resource layout changes between episodes
- tests generalization and within-life adaptation

Hypotheses:
- modulatory signals correlate with multi-dimensional drive changes
- mode switching (foraging for energy vs hydration) emerges.

---

## 6. Level L3 — Threats, cues, and conditioning

### L3.0: Safe vs unsafe zones

- hazard zones cause integrity loss

### L3.1: Predictive cues (conditioning)

- cues predict hazard onset after a delay
- encourages temporal credit assignment and anticipatory avoidance

### L3.2: Stochastic threats and misleading cues

- adds noise and uncertainty

Hypotheses:
- emergent “threat prediction” signals appear
- plasticity aligns with cue–outcome contingencies.

Acceptance criteria:
- agents learn to avoid hazards using cues, not only direct contact.

---

## 7. Level L4 — Long-horizon partial observability

### L4.0: Large world, sparse resources

- requires navigation memory and efficient search

### L4.1: Seasons (mid-episode regime shift)

- resource availability changes mid-episode
- forces rapid adaptation

### L4.2: Task-within-task (hidden goal changes)

- weights in the drive function change occasionally (e.g., integrity becomes more important)
- tests flexible policy control driven by internal state.

Hypotheses:
- internal dynamics show distinct regimes (modes)
- learning-to-learn becomes strongly beneficial.

---

## 8. Level L5 — Meta-learning without social

This level is optional but useful to test generality.

### L5.0: Family of related worlds

- each episode samples from a distribution of worlds (layouts, resource rules)
- within-life adaptation is required to do well across the family

Evaluation focuses on:
- speed of adaptation in early episode steps
- retention vs forgetting across episodes.

---

## 9. Nursing integration (“weaning schedules”)

Within each level, environment difficulty is scheduled by developmental phase \(\phi\):

- early life: high resources, low hazards, high buffers
- later life: adult difficulty

This allows the same level to host:

- “infant training” dynamics (safe exploration),
- “adult performance” evaluation.

---

## 10. Level-by-level deliverables and tests

For each level, we implement:

1. deterministic world generation given seed
2. baseline heuristic agents for sanity checks
3. a suite of metrics and visualizations
4. an “acceptance notebook” (optional) demonstrating expected behaviors

---

## 11. Scope note: social is future work

Multi-agent environments and social cognition are deferred to post-thesis work. The ladder is intentionally designed to build strong evidence for emergence in the single-agent setting first.

See `19_FUTURE_WORK_AND_VALIDATION.md`.
