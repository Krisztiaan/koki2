# Environments, homeostasis, and drives

## 1. Why homeostasis is the right “objective”

We want environments that:

- produce meaningful pressures without hand-crafted reward shaping,
- are biologically interpretable,
- and fit inside the inside-out, organism-centered stance.

The key design is **internal state**: agents must keep survival variables within viability bounds. Reward is derived from drive reduction (deviation from setpoints), consistent with homeostatic reinforcement learning ideas [@keramati2014homeostatic].

---

## 2. Internal state model

Let internal state be a vector:

\[
H(t) = (h_1(t), h_2(t), \dots, h_K(t))
\]

with:

- viability bounds \(h_k \in [h_k^{\min}, h_k^{\max}]\),
- setpoints \(h_k^\*\) representing “healthy” values.

Baseline includes two variables:

- energy \(E(t)\in[0,1]\), death if 0
- integrity \(I(t)\in[0,1]\), death if 0

Additional variables are introduced gradually:

- hydration
- temperature (bounded interval)
- fatigue / sleep pressure (optional)

### 2.1 Dynamics

A generic discrete update:

\[
H(t+1) = \mathrm{clip}\left(H(t) + \Delta H_{\text{metabolic}}(t) + \Delta H_{\text{interaction}}(t) + \epsilon(t)\right)
\]

where:

- metabolic terms decay energy, increase fatigue, etc.
- interaction terms depend on environment (consuming resources, taking damage).
- noise \(\epsilon(t)\) increases realism and prevents brittle policies.

### 2.2 Death

Episode terminates immediately if any critical variable violates viability:

\[
\exists k \in \mathcal{K}_{\text{critical}}: h_k(t)\le h_k^{\min}
\]

Death is a *natural* early termination mechanism and is required for pruning logic later.

---

## 3. Drive and reward

### 3.1 Drive function

Define drive as a weighted distance from setpoints:

\[
D(H(t)) = \sum_{k=1}^K w_k \, \rho\left(|h_k(t) - h_k^\*|\right)
\]

where:

- \(w_k\) weights importance of each variable,
- \(\rho(\cdot)\) is typically convex (e.g., quadratic) or piecewise linear.

### 3.2 Reward as drive reduction

Reward is the negative change in drive:

\[
r(t) = D(H(t-1)) - D(H(t)) = -\Delta D
\]

This is important:

- the environment does not provide “goal reward”,
- reward emerges from maintaining viability.

Fitness aggregates reward implicitly via survival and resource acquisition.

---

## 4. Observation and action design (active sensing)

Inside-out alignment suggests we should avoid environments solvable by passive observation. Therefore:

- observations should be **local** and **action-contingent**.

### 4.1 Observation vector

Observations are decomposed into:

1. **Exteroception** (world sensing):
   - local gradients (energy smell, hazard smell),
   - short-range raycasts (obstacle proximity),
   - sparse event cues (shock cue present).

2. **Interoception** (internal sensing):
   - internal state values \(H(t)\) (possibly noisy),
   - derivatives / change estimates (optional).

3. **Efference copy** (optional):
   - previous action or motor command.

#### 4.1.1 Cue informationality (valence ambiguity)

A recurring design axis in this thesis is **how much the exteroceptive cue reveals about outcome valence** (e.g., “food vs poison”):

- **Ambiguous cues (preferred early for inside-out tests):** objects share the same outward cue, and “meaning” is learned via action consequences for viability. This operationalizes the inside-out stance: consequences calibrate internal dynamics.
- **Informative cues (useful controls):** hazards have explicit cues (separate hazard gradients/cues). These variants are valuable as sanity checks and ablations, but they reduce the demand for consequence-driven discrimination.

We treat cue informationality as a controlled knob (often nursing-scheduled) rather than a hidden confound, so we can attribute performance changes to memory/plasticity vs sensor informativeness.

### 4.2 Action space

Keep early levels simple:

- 2D continuous movement \((\Delta x, \Delta y)\) or discrete moves,
- optional “consume” action when on a resource cell,
- optional “rest” action that trades time for internal recovery.

Motor/action gating for juveniles is handled by the nursing module.

---

## 5. Environment components

We structure environments as compositions of components:

- resource fields (energy sources, hydration sources)
- hazards (damage zones, predators)
- geometry (walls, corridors)
- stochastic events (storms, shocks)
- global cycles (day/night, seasons)

This compositional approach supports a granular ladder without re-implementing everything.

---

## 6. Fitness function (base)

The base fitness is intentionally simple:

\[
F_{\text{base}} = \alpha \, T_{\text{alive}} + \beta \sum_t \max(0, \Delta E_{\text{consumed}}(t))
\]

where \(T_{\text{alive}}\) is survival time and energy consumed is a proxy for successful foraging.

**Important:** We avoid fitting fitness directly to internal drive reduction, because drive-based reward is already inside the life loop. Fitness remains coarse, letting evolution select for strategies without specifying mechanisms.

Biasing evaluators (novelty, learning capability) are added later and controlled.

---

## 7. Developmental curriculum hook

Every environment parameter can be scheduled by developmental phase \(\phi\) (see nursing addendum). This is crucial for biological realism and for making evolution computationally feasible.

---

## 8. Deliverables

- A formal definition of internal state variables and dynamics.
- A drive function and reward computation used consistently across all environment levels.
- A modular environment component library enabling a granular ladder.
- A validation suite:
  - death termination correctness,
  - reward invariance checks,
  - observation/action sanity.
