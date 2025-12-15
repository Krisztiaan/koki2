# 06 — Environment ladder: granular progression with acceptance tests

**Purpose:** Specify a sequence of environments that (i) increases difficulty gradually, (ii) aligns with inside‑out action primacy, (iii) supports nursing schedules, and (iv) yields verifiable milestones.

**Design rule:** Each level must be solvable by a strictly smaller mechanism set than the next. If a level requires memory, the previous level must not.

---

## 1. Shared environment substrate: “homeostatic foraging world”

All environments share a common substrate:

- Agent has position $x_t \in \mathbb{R}^d$ (initially $d=1$ or $2$), optional velocity $v_t$.
- World contains one or more resource fields $R(x)$ and optional hazard fields $H(x)$.
- Agent has internal energy $E_t$ with dynamics:
  \[
  E_{t+1} = \mathrm{clip}\big(E_t + g(R(x_t)) - c(u_t) - c_\mathrm{base}, 0, E_\mathrm{max}\big)
  \]
- Episode ends when:
  - energy drops below threshold for $k$ steps (death), or
  - time horizon reached.

This makes all tasks variations of **action-conditioned survival**, consistent with inside‑out framing [@buzsaki2019brain].

---

## 2. Observation and action spaces (kept minimal early)

### 2.1 Observations (early levels)

Early levels use low-dimensional sensory inputs designed for learnability and speed:

- **Chemoreception:** local concentration samples, e.g. $R(x_t)$ and optionally a finite difference estimate:
  - $R(x_t+\delta)$, $R(x_t-\delta)$ in 1D
  - left/right/front samples in 2D
- **Proprioception:** optional velocity, last action, energy $E_t$ (either observed or latent)

### 2.2 Actions

- In 1D: scalar movement $\Delta x$ or force $u_t \in [-1,1]$.
- In 2D: 2D force or heading+speed.

**Rule:** actions must causally affect future observations (L0 viability check).

---

## 3. Nursing integration (global knobs)

Each environment supports a `NursingParams(age)` schedule (see 03_…):

- initial energy reserve $E_0$
- resource density / reward scaling
- action cost scaling
- hazard scaling
- noise scaling

Nursing is treated as an environment factor, not an agent factor, and is always ablated on/off across strains.

---

## 4. Level-by-level specifications

### L0 — Sensorimotor viability (“alive and coupled”)

**Goal:** eliminate trivial failures and confirm closed-loop coupling.

- World: single smooth resource gradient in 1D; no obstacles.
- Observation: $(R(x+\delta), R(x-\delta), E_t)$.
- Action: scalar force $u_t$.
- Reward: not required; fitness can be survival time or cumulative energy.

**Acceptance tests:**
1. **Coupling:** action influences future observation:
   - empirical $I(a_t; o_{t+1}\mid o_t)$ proxy > threshold.
2. **Non-degenerate policy:** variance of actions > small epsilon for successful candidates.
3. **Energy regulation:** median survival time exceeds baseline random policy.

**Mechanisms required:** none beyond a stateful controller; no memory needed.

---

### L1 — Chemotaxis (smooth field) with noise

**Goal:** robust gradient following.

- World: 1D or 2D gradient + observation noise.
- Observation: local samples + noise.
- Reward: energy gain; optional shaping for distance-to-source (only for debugging).

**Variants:**
- multiple random source locations
- varying gradient steepness

**Acceptance tests:**
- success rate of reaching source within time > threshold under multiple seeds.

**Mechanisms stressed:** robustness; simple internal dynamics.

---

### L2 — Chemotaxis with inertia + obstacles

**Goal:** introduce control constraints.

- World: 2D, inertia (velocity state), obstacles (hard walls).
- Observation: chemoreception + short-range obstacle sensor (e.g., distance to wall in a few rays).
- Action: force.

**Acceptance tests:**
- reach resource above threshold while avoiding walls (collision penalties optional but avoid being the only signal).

**Mechanisms stressed:** control with dynamics; still minimal memory.

---

### L3 — Foraging with depletion (short-term state and strategy)

**Goal:** repeated decisions; avoid single-shot “go to source and stop”.

- World: multiple patches; patches deplete when consumed; respawn with delay.
- Observation: local resource + optionally a short-range “patch indicator”.
- Action: movement.

**Fitness:** total survival time, or cumulative energy over lifespan.

**Acceptance tests:**
- maintain $E_t$ above survival threshold for a target fraction of episode time on average.

**Mechanisms stressed:** persistence; behavioral strategy; begins to benefit from short-term memory/plasticity but not strictly required.

---

### L4 — Two-cue context gating (latent state)

**Goal:** require internal state to disambiguate cues.

- World: two resource fields $R_1(x)$, $R_2(x)$ with a hidden context variable $c \in \{1,2\}$.
- Only one resource is “nutritive” depending on context.
- Context changes per episode; may switch mid-episode in harder variants.

**Observation:** both cue concentrations; context is not observed.

**Acceptance tests:**
- choose correct cue-conditioned behavior reliably across contexts.

**Mechanisms stressed:** internal state / memory; modulatory learning helps.

---

### L5 — Delayed reward + hazards (temporal credit assignment)

**Goal:** stress credit assignment.

- Add hazard field $H(x)$ with delayed penalty (or delayed reward after reaching a “processing zone”).
- Introduce delay between resource contact and energy gain.

**Acceptance tests:**
- maintain survival and avoid hazard under delay; outperform non-plastic baselines.

**Mechanisms stressed:** eligibility traces, neuromodulation, or inner-loop learning.

---

### L6 — Sparse reward navigation with partial observability (working memory)

**Goal:** require working memory-like capability.

- World: maze-like obstacles; resources sparse and sometimes invisible until near.
- Observation: local sensors only; no global position.
- Requires integrating history to infer location.

**Acceptance tests:**
- solve with success rate above threshold; show generalization to held-out maze seeds.

**Mechanisms stressed:** memory; internal dynamics; benefit from genomic bottleneck generalization [@shuvaev2024genomic].

---

## 5. Minimal-criterion gates and progression rules

Progression is controlled by gates (not automatic difficulty jumps):

- Each level $\ell$ defines a criterion $C_\ell$ (e.g., survive ≥ T, reach source ≥ k times).
- Evolution only proceeds to level $\ell+1$ after meeting $C_\ell$ reliably on held-out variants.

This prevents “skipping” and makes thesis claims falsifiable.

---

## 6. Measurement suite (required metrics)

Per episode:
- survival time
- cumulative energy
- energy variance (homeostatic stability)
- distance-to-resource statistics
- action magnitude and smoothness
- collision/hazard exposure

Mechanistic probes (optional but recommended):
- action→sensory coupling proxy
- internal-state predictability probes (see 02_…)
- spike rate statistics (if SNN)

---

## 7. Implementation notes for efficiency

- Environments must support `lax.scan` stepping.
- Prefer analytic resource fields (Gaussians, exponentials) to avoid grids early.
- Use fixed-size arrays for obstacles (padded lists).
- Keep observation dimension constant per level.

---

## 8. References

See `references.bib`.
