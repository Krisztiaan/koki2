# Addendum: developmental niche (“nursing”) and gradual difficulty

## 1. Motivation

The baseline thesis framework evaluates agents in survival environments. Without additional structure, two failures are likely:

1. **Biological implausibility**: newborn organisms are not expected to forage and survive unaided in the full adult niche.
2. **Optimization pathology**: early random policies cause immediate death, starving evolution of useful gradients.

Biology solves both via evolved **developmental niches**: reliable patterns of provisioning, protection, and staged exposure to risk. West & King introduced the “ontogenetic niche” concept to emphasize that development depends on inherited niche structure, not genes alone [@westking1987ontogenetic]. Developmental niche construction further generalizes this to extended inheritance [@stotz2017dnc].

Comparative work suggests that extended parental provisioning can enable larger brains by subsidizing costly development [@vanschaik2023provisioning]. This is directly relevant to our goal: evolving learning systems is metabolically and behaviorally expensive.

We therefore formalize **nursing** as a first-class module: an ontogenetic curriculum that shapes both environment difficulty and plasticity regime over age. This is related to, but not equivalent to, ML curriculum learning; the primary justification here is biological developmental niche structure rather than optimization convenience [@bengio2009curriculum].

---

## 2. Developmental phase variable

Define:

- lifespan length \(T_{\text{life}}\) (steps)
- developmental phase \(\phi(t)=t/T_{\text{life}}\in[0,1]\)

All nursing schedules are expressed as functions of \(\phi\).

This preserves generality: we can map “infant”, “juvenile”, “adult” to ranges of \(\phi\) without hard-coded discrete stages.

---

## 3. Nursing mechanisms (what changes with age)

We implement nursing via five mechanism families. Each mechanism is independently toggleable and ablatable.

### 3.1 Energetic buffer (initial reserve)

New agents start with higher energy and slower metabolic decay:

\[
E(0) = E_{\text{adult}} + \Delta E_{\text{infant}}
\]
\[
c_{\text{metabolic}}(\phi) = c_{\text{adult}}\cdot (1 - \delta_E(\phi))
\]

where \(\delta_E(\phi)\) decays from a high value near \(\phi=0\) to 0 at adulthood.

**Interpretation:** parental provisioning and protected feeding.

### 3.2 Nursery richness (resource density / gain)

Early environments provide higher resource density and/or larger gains:

\[
\rho(\phi)=\rho_{\text{adult}}\cdot (1 + \delta_\rho(\phi))
\]
\[
g(\phi)=g_{\text{adult}}\cdot (1 + \delta_g(\phi))
\]

This increases the probability that exploratory behavior produces informative consequences.

### 3.3 Risk gating (hazards delayed or softened)

Hazards are suppressed early:

\[
h(\phi)=h_{\text{adult}}\cdot s_h(\phi)
\]

with \(s_h(0)\approx 0\), \(s_h(1)=1\).

Hazard gating can control:

- hazard spawn rates,
- damage magnitude,
- distance of hazards from “home” safe zones.

### 3.4 Motor/action gating (maturation)

We restrict early action magnitude and optionally restrict action types:

\[
a_{\max}(\phi)=a_{\text{adult}}\cdot s_a(\phi)
\]

where \(s_a(\phi)\) increases from small to 1.

This prevents early death by reckless high-velocity behavior and better matches infant motor maturation.

### 3.5 Plasticity schedules (critical period analogue)

Plasticity is age-dependent:

\[
\eta_{\text{global}}(\phi)=\eta_0\cdot s_\eta(\phi)
\]

Possible schedules:

- monotone decay (high early, lower later),
- biphasic (high early, consolidation, adolescent reopening).

This mirrors the idea of sensitive / critical periods, without committing to a specific biological mechanism [@hensch2005critical].

Plasticity is still gated by emergent modulatory signals; nursing modulates the baseline sensitivity.

### 3.6 Sensory gating and resolution (perceptual maturation)

We can also schedule **what the agent can sense** and **at what precision** as a function of age:

- availability gating: certain observation channels are partially or fully suppressed early and introduced later,
- resolution gating: sensors start *coarse* (quantized / low SNR) and become *fine* as \(\phi\to 1\),
- noise scaling: observation noise can be higher early (encouraging robust control) or lower early (easier coupling), then annealed.

This is analogous to a developmental progression where early behavior can be learned with low-dimensional, low-precision cues, while later life requires higher-fidelity sensing. In developmental psychobiology, early sensory limitations have been proposed to play an organizing role rather than being purely deficits [@turkewitz1982limitations; @turkewitz1985role; @lickliter2000sensory].

In ML, this is related to curriculum learning and staged complexity schedules [@bengio2009curriculum]. Progressive training schemes that explicitly increase task difficulty and/or input fidelity over time are common and empirically useful (e.g., progressive learning in EfficientNetV2, progressive growing of GANs) [@tan2021efficientnetv2; @karras2017progressivegan]. In developmental robotics, “maturational constraints” (staging capabilities over developmental time) have been argued to aid long-horizon developmental learning [@law2014longitudinal], and intrinsically motivated goal exploration can be viewed as complementary scaffolding for efficient skill acquisition [@baranes2013active; @oudeyer2018curiosity]. Recent simulation work that explicitly models body and sensory development (MIMo v2) provides a concrete platform for studying these ideas in multimodal settings [@lopez2025mimogrows].

Hypothesis: sensory gating improves early survivability and reduces premature “dead ends”, enabling more reliable pruning and faster evolution under fixed compute budgets.

---

## 4. Implementation integration

### 4.1 API additions

Introduce a shared `DevelopmentState` passed into:

- `env_step(state, action, dev_state)`
- `agent_step(params, state, obs, internal, dev_state)`

### 4.2 Separation of responsibilities

- environment implements \(\rho(\phi), h(\phi)\), safe zones, and resource availability
- agent core implements \(\eta_{\text{global}}(\phi)\), action gating, and modulator thresholds (optional)
- evolution engine implements delayed reproduction / phase-weighted fitness

This separation makes nursing compatible with parallel development and clean ablations.

---

## 5. Fitness aggregation under nursing

We define phase-weighted life fitness:

\[
F = \sum_{e\in \text{episodes}} w(\phi_e)\,F_e
\]

with higher weights for near-adult phases, while still requiring survival through early phases.

Optionally enforce:

- reproduction eligibility only after \(\phi\ge \phi_{\text{repro}}\)

This models juvenile dependence and avoids selecting for infant-only hacks.

---

## 6. Experiments: identifying a “Goldilocks” nursing regime

We systematically vary nursing strength:

- energy buffer magnitude,
- nursery richness,
- hazard gating schedule steepness,
- plasticity schedule amplitude.

We measure:

- fraction of genomes that are viable (pass MVT),
- time to reach adult competence under a fixed compute budget,
- stability of learned behaviors under adult conditions.

Hypothesis:

- too little nursing → extinction and no progress,
- too much nursing → “spoiled” agents that fail when weaned,
- intermediate nursing → maximal evolvability.

---

## 7. Interaction with pruning and strains

- Nursing reduces false negatives in pruning by giving marginal genomes a fair viability chance.
- Nursing interacts with bio-strains:
  - hard E/I constraints may reduce early exploration, increasing need for nursing,
  - soft constraints may be easier to introduce as the environment hardens.

---

## 8. Deliverables

- A formal nursing schedule specification.
- A nursery environment configuration used for MVT and early episodes.
- A suite of nursing ablations demonstrating effects on compute and emergence.
