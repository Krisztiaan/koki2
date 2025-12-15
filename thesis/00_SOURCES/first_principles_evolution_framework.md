# First-Principles Framework for Evolving Biologically-Constrained Learning Agents

## Core Philosophy: From Survival Pressures to Emergent Complexity

**Central Thesis**: Rather than pre-specifying systems like "dopamine-like reward" or "fear circuits," we define only the most primitive survival requirements and let evolution discover why such systems are useful—allowing us to observe *functional necessity* rather than imposing architectural choices.

This approach inverts the typical neuroevolution paradigm: instead of asking "how do we implement reward prediction error?" we ask "what environmental pressures would cause evolution to invent a mechanism that computes reward prediction error?"

---

## Part I: The Minimal Substrate — Building Blocks That Can Express Anything

### 1.1 Primitive Computational Units

The neural substrate must be minimal yet sufficient. We define only:

**Basic Unit Properties:**
```
Each unit has:
├── Activation state: a ∈ ℝ (continuous, bounded)
├── Threshold/bias: θ ∈ ℝ (evolvable)
├── Time constant: τ ∈ [τ_min, τ_max] (evolvable)
├── Connection weights: w_ij ∈ ℝ (plastic)
└── Plasticity state: p_ij ∈ ℝ (eligibility trace)
```

**Why this minimal set?**
- **Activation**: Required for information processing
- **Threshold**: Required for nonlinearity (basis of computation)
- **Time constant**: Required for temporal processing (memory emerges from diversity)
- **Weights**: Required for learning
- **Plasticity state**: Required for temporal credit assignment

We do NOT pre-specify:
- Excitatory vs inhibitory (let it emerge from weight signs)
- Neuromodulation (let it emerge as a *function* of activity)
- Specialized neuron types (let functional specialization emerge)
- Memory systems (let attractor dynamics emerge from recurrence)

### 1.2 Plasticity as a Meta-Parameter Space

Following the ABCD Hebbian model (Najarro & Risi, 2020), each synapse has evolvable plasticity coefficients:

**Generalized Local Learning Rule:**
```
Δw_ij = η(A·o_i·o_j + B·o_i + C·o_j + D + E·m)
```

Where:
- A, B, C, D are evolvable per-synapse coefficients
- o_i, o_j are pre/post-synaptic activations
- m is a *computed* modulation signal (not pre-specified as "dopamine")
- η is an evolvable learning rate

**Critical insight**: The modulation signal m is computed BY THE NETWORK ITSELF from its own activity. Evolution must discover:
1. Which neurons should compute m
2. What inputs those neurons should receive
3. How m should affect plasticity

This is how "dopamine-like" signaling emerges—not because we specified it, but because evolution discovers that broadcasting a scalar value computed from reward prediction improves survival.

### 1.3 The Genomic Bottleneck: Developmental Encoding

The genome does NOT specify weights directly. Instead, it specifies:

**Layer 1: Cell Identity Genes**
```
Each position (x, y) in neural space receives a "cell identity" vector:
identity(x, y) = f_CPPN(x, y, d_from_input, d_from_output, ...)
```

**Layer 2: Wiring Rules (Genetic Connectome Model)**
```
Connection probability: P(connect) = σ(identity_i · W_rule · identity_j)
Weight initialization: w_ij = g(identity_i, identity_j, distance)
Plasticity coefficients: ABCD_ij = h(identity_i, identity_j)
```

**Compression Targets:**
| Environment Complexity | Genome Size | Network Size | Compression |
|----------------------|-------------|--------------|-------------|
| Chemotaxis | ~100 params | ~50 units | 0.5x |
| Navigation | ~500 params | ~200 units | 2.5x |
| Foraging | ~2000 params | ~1000 units | 10x |
| Social | ~5000 params | ~10000 units | 100x |

The genomic bottleneck forces discovery of *developmental rules* that generalize, not memorized solutions.

---

## Part II: The Environment — Survival Pressures That Shape Everything

### 2.1 The Internal State: Homeostatic Variables

This is the crucial innovation: agents have *internal states* that must be maintained within viability bounds.

**Core Internal Variables:**
```python
class InternalState:
    # Primary survival variables
    energy: float      # [0, 1] — death if 0
    integrity: float   # [0, 1] — death if 0 (damage/pain)
    
    # Secondary homeostatic variables (added with environment complexity)
    hydration: float   # [0, 1] — death if 0
    temperature: float # [0.3, 0.7] — death outside bounds
    
    # Dynamics
    energy_decay: float        # Constant metabolic cost
    energy_cost_per_action: float  # Movement/computation cost
    damage_decay: float        # Natural healing rate
```

**Why this matters**: The agent's "reward" is not externally specified. Instead, reward is computed as:

```
reward = -Δdrive = -Δ(distance from homeostatic setpoint)
```

This is Homeostatic Reinforcement Learning (Keramati & Gutkin, 2014). The agent doesn't maximize arbitrary reward—it minimizes deviation from survival-compatible states.

### 2.2 The External Environment: Sources of Challenge

**Level 0: Chemotaxis**
```
Environment:
├── Energy gradient field (chemical concentration)
├── Damage gradient field (toxic zones)
├── Simple kinematics (velocity control)
└── Sensors: [gradient_magnitude, gradient_direction, energy_level, integrity]
```

Evolutionary pressure: Survive by following energy gradients while avoiding damage gradients.

**What should emerge**: Biased random walk toward energy, avoidance of damage zones. The connection between gradient detection and motor output is the minimal "SEEKING" behavior.

**Level 1: Temporal Navigation**
```
Environment:
├── Energy sources (discrete, depletable, respawning)
├── Damage sources (predators with simple dynamics)
├── Spatial memory requirement (remember depleted sources)
└── Sensors: [local_gradient, current_position, energy, integrity, time_since_last_food]
```

Evolutionary pressure: Survive in an environment where resources deplete and predators move.

**What should emerge**: 
- Temporal integration (eligibility traces become useful)
- Spatial memory (attractor dynamics for locations)
- Distinct responses to energy vs damage gradients

**Level 2: Foraging with Energy Management**
```
Environment:
├── Multiple resource types (vary in energy/acquisition cost)
├── Day/night cycles (predator activity varies)
├── Internal state complexity (energy, integrity, fatigue)
└── Actions with different costs (move, rest, forage, hide)
```

Evolutionary pressure: Manage multiple internal variables over extended time.

**What should emerge**:
- Risk assessment (cost-benefit of foraging vs safety)
- Anticipatory behavior (allostasis—act before need arises)
- Distinct behavioral modes (active foraging vs hiding)

**Level 3: Unpredictable Threats**
```
Environment:
├── Stochastic predator appearance
├── Safe zones with lower energy availability
├── "Warning" signals preceding some threats
└── Conditioning opportunities (neutral stimulus → threat association)
```

Evolutionary pressure: Survive unpredictable threats while maintaining energy.

**What should emerge**:
- Fear-like responses (threat detection → freezing/fleeing)
- Associative learning (conditioned stimulus → threat response)
- Distinct "anxious" vs "explorative" behavioral modes

**Level 4: Social/Multi-Agent**
```
Environment:
├── Other agents (conspecifics)
├── Resources requiring cooperation (high energy, but needs 2+ agents)
├── Communication channel (optional actions that produce signals)
└── Kinship structure (shared genomic heritage)
```

Evolutionary pressure: Survive in a world where cooperation yields higher fitness.

**What should emerge**:
- Social signaling (communication for coordination)
- Theory of mind precursors (predicting other agents)
- Kin-biased behavior (inclusive fitness)

---

## Part III: Biasing Evaluators — Guiding Without Specifying

### 3.1 The Core Fitness Function: Survival + Learning Capability

The base fitness is simple:
```
fitness_base = survival_time + Σ(lifetime_energy_acquired)
```

But this alone may not guide evolution efficiently. We add **biasing evaluators** that shape the fitness landscape without specifying HOW to achieve good outcomes.

### 3.2 Hierarchy of Biasing Evaluators

**Tier 1: Survival Bias (Always Active)**
```
F_survival = t_alive / t_max + bonus_if_reproduced
```
Simple: live longer, pass on genes.

**Tier 2: Learning Capability Bias**
```
F_learning = (performance_end - performance_start) / performance_start
```
This rewards agents that IMPROVE during their lifetime. Critical for discovering plasticity.

**Tier 3: Behavioral Novelty Bias (Optional, for diversity)**
```
F_novelty = distance(behavior_vector, archive_of_behaviors)
```
Prevents convergence to single solution; maintains population diversity.

**Tier 4: Homeostatic Efficiency Bias**
```
F_homeostatic = -mean(|internal_state - setpoint|) over lifetime
```
Rewards tight regulation of internal states.

**Tier 5: Anticipatory Bias (Higher environments)**
```
F_anticipatory = correlation(action_t, threat_t+Δ)
```
Rewards acting BEFORE threats materialize (allostatic control).

### 3.3 Staged Bias Introduction

Different biases are introduced at different complexity levels:

| Environment Level | Active Biases |
|-------------------|---------------|
| Chemotaxis | F_survival |
| Navigation | F_survival + F_learning |
| Foraging | F_survival + F_learning + F_homeostatic |
| Threats | All above + F_anticipatory |
| Social | All above + F_novelty |

This prevents premature optimization for metrics that aren't yet relevant.

### 3.4 Soft vs Hard Constraints

**Hard constraints (death conditions):**
- energy ≤ 0
- integrity ≤ 0
- internal variables outside viability bounds

**Soft constraints (fitness penalties):**
- Prolonged deviation from setpoints
- High variance in internal states
- Excessive metabolic cost of neural activity

The soft constraints bias evolution toward efficient, stable solutions without mandating specific implementations.

---

## Part IV: What Should Emerge — Predictions and Checkpoints

### 4.1 Predicted Emergent Mechanisms

Based on the environmental pressures, we predict evolution will discover functional analogues of:

**Chemotaxis Level:**
- Gradient-following behavior (trivially necessary)
- Biased random walk when gradient weak
- Energy-state-dependent exploration (explore more when hungry)

**Navigation Level:**
- Eligibility traces (for temporal credit assignment)
- Attractor dynamics (for spatial memory)
- Distinct "approach" vs "avoid" pathways

**Foraging Level:**
- Neuromodulatory broadcasting (a signal that gates plasticity globally)
- State-dependent action selection (behavioral switching)
- Prediction of future internal states (anticipatory regulation)

**Threat Level:**
- Fast threat-detection pathway (short-latency, high-sensitivity)
- Conditioned fear (neutral → threat association)
- Freezing/fleeing behavioral modes
- Anxiety-like sustained avoidance after threat exposure

**Social Level:**
- Communication emergence (signals that benefit sender when receiver responds)
- Partner choice (discrimination of cooperative vs defecting agents)
- Social learning (copying successful agents)

### 4.2 Measurement Checkpoints

At each level, we measure whether predicted mechanisms emerged:

**Checkpoint 1: Neuromodulation Emergence**
- Does the network contain neurons whose activity correlates with:
  - Reward prediction error? (proto-dopamine)
  - Threat detection? (proto-fear signal)
  - Energy deficit? (proto-hunger signal)
- Do these neurons' outputs gate plasticity elsewhere?

**Checkpoint 2: Memory System Emergence**
- Does the network contain:
  - Attractor states that persist after stimulus removal? (working memory)
  - Pattern completion from partial cues? (associative memory)
  - Distinct fast-learning and slow-learning populations? (complementary learning systems)

**Checkpoint 3: Behavioral Mode Emergence**
- Does the agent exhibit:
  - Distinct exploration vs exploitation modes?
  - State-dependent behavioral switching?
  - Hysteresis (mode persists after trigger removed)?

**Checkpoint 4: Social Cognition Emergence**
- Does the agent exhibit:
  - Behavior modification based on other agents' presence?
  - Predictive modeling of other agents?
  - Communication that transfers information?

---

## Part V: The Evolutionary Loop — Implementation Details

### 5.1 Genome Structure

```python
genome = {
    # CPPN for spatial identity
    'cppn_weights': [...],      # ~100-500 parameters
    'cppn_topology': [...],     # Encoded via historical markings
    
    # Wiring rule matrices
    'connection_rule': [...],   # How identities map to connectivity
    'weight_rule': [...],       # How identities map to initial weights
    'plasticity_rule': [...],   # How identities map to ABCD coefficients
    
    # Global parameters
    'time_constants': [...],    # Distribution of τ values
    'learning_rate': float,     # Global η
    'modulation_target': [...], # Which neurons receive modulation
}
```

### 5.2 Developmental Process

```python
def develop(genome, network_size):
    # 1. Generate identity vectors for each position
    identities = []
    for pos in grid(network_size):
        identity = cppn_forward(genome['cppn_weights'], pos)
        identities.append(identity)
    
    # 2. Generate connectivity from identities
    connections = []
    for i, j in all_pairs(network_size):
        prob = sigmoid(identities[i] @ genome['connection_rule'] @ identities[j])
        if random() < prob:
            weight = compute_weight(identities[i], identities[j], genome['weight_rule'])
            abcd = compute_plasticity(identities[i], identities[j], genome['plasticity_rule'])
            connections.append((i, j, weight, abcd))
    
    # 3. Assign time constants
    for i in range(network_size):
        tau[i] = sample_from_distribution(genome['time_constants'], identities[i])
    
    return Network(identities, connections, tau)
```

### 5.3 Lifetime Simulation

```python
def simulate_lifetime(agent, environment, max_steps):
    internal_state = InternalState()
    network = develop(agent.genome, network_size)
    
    for t in range(max_steps):
        # 1. Sense environment
        sensors = environment.get_sensors(agent.position, internal_state)
        
        # 2. Neural forward pass
        motor_output, modulation = network.forward(sensors)
        
        # 3. Execute action
        action = decode_action(motor_output)
        environment.step(agent, action)
        
        # 4. Update internal state
        internal_state.update(action, environment.get_consequences())
        
        # 5. Apply plasticity (gated by modulation)
        network.apply_plasticity(modulation)
        
        # 6. Check death conditions
        if internal_state.is_dead():
            return t, accumulated_fitness
    
    return max_steps, accumulated_fitness
```

### 5.4 Evolutionary Operators

**Selection**: Tournament selection with speciation (NEAT-style)

**Crossover**: Alignment by historical markings for CPPN; parameter interpolation for rules

**Mutation**:
- CPPN mutations: add node, add connection, perturb weight
- Rule mutations: perturb matrices, change distribution parameters
- Global mutations: perturb learning rate, time constant distribution

**Speciation**: Genomic distance based on CPPN topology + rule matrix distance

### 5.5 Environment Co-Evolution (POET-style)

```python
def coevolve(population, environments, generations):
    for gen in range(generations):
        # 1. Evaluate all agents on their paired environments
        fitnesses = evaluate(population, environments)
        
        # 2. Attempt transfers (agent → different environment)
        for agent, env in pairs:
            for other_env in environments:
                if agent.fitness_in(other_env) > other_env.champion.fitness:
                    other_env.champion = agent  # Transfer!
        
        # 3. Mutate environments (if agents are too successful)
        for env in environments:
            if env.champion.fitness > env.difficulty_threshold:
                env.increase_difficulty()
        
        # 4. Create new environments from successful ones
        if len(environments) < max_environments:
            parent_env = select_by_novelty(environments)
            child_env = mutate_environment(parent_env)
            if child_env.is_solvable_by(population):
                environments.append(child_env)
        
        # 5. Evolve agents within their environments
        population = evolve_step(population, fitnesses)
```

---

## Part VI: Research Questions and Experimental Design

### 6.1 Primary Research Questions

**Q1**: Do neuromodulatory mechanisms emerge from first principles?
- Hypothesis: Evolution will discover neurons that compute reward prediction error and broadcast it to gate plasticity.
- Measurement: Correlation between neuron activity and RPE; effect of ablating these neurons.

**Q2**: Does memory emerge from plasticity and recurrence alone?
- Hypothesis: Without pre-specified memory modules, attractor dynamics and eligibility traces will emerge.
- Measurement: Persistence of activity after stimulus; pattern completion performance.

**Q3**: Does the complexity of evolved networks match the complexity of environments?
- Hypothesis: Genomic bottleneck forces efficient encoding; network complexity scales sub-linearly with task complexity.
- Measurement: Network size/parameter count vs environment complexity.

**Q4**: Do distinct "emotional" systems emerge for distinct survival pressures?
- Hypothesis: Separate neural pathways will emerge for approach (energy-seeking) vs avoidance (threat-fleeing).
- Measurement: Network analysis of connectivity between sensors and motor outputs.

**Q5**: Does the Baldwin effect occur?
- Hypothesis: Initially learned behaviors become genetically encoded over evolutionary time.
- Measurement: Performance of naive agents (before learning) across generations.

### 6.2 Control Experiments

**Control 1**: Direct encoding baseline
- Same environments, but genome directly specifies weights
- Expected result: Worse scaling, less transfer, more brittle solutions

**Control 2**: Fixed plasticity rule
- ABCD coefficients are not evolvable; fixed Hebbian rule
- Expected result: Worse learning capability, simpler behaviors

**Control 3**: External reward signal
- Reward provided externally rather than computed from internal state
- Expected result: Different (possibly degenerate) solutions; less robust to perturbation

**Control 4**: No genomic bottleneck
- Compression ratio = 1 (genome size = network size)
- Expected result: Overfitting to specific environments; poor transfer

### 6.3 Analysis Methods

**Network Analysis:**
- Modularity detection (Louvain algorithm)
- Connectivity motif analysis (feedforward, feedback, lateral)
- Time constant distribution analysis
- Neuromodulatory hub identification

**Behavioral Analysis:**
- State-space trajectory analysis
- Behavioral mode detection (HMM or clustering)
- Response latency to threats
- Learning curve analysis

**Evolutionary Analysis:**
- Fitness trajectory over generations
- Genomic diversity over time
- Speciation dynamics
- Transfer success rates

**Biological Comparison:**
- Comparison of emerged circuits to known biological circuits
- Functional analogy assessment (does it work like the biological version?)
- Structural similarity analysis (does it look like the biological version?)

---

## Part VII: Theoretical Foundations

### 7.1 Connection to Free Energy Principle

The homeostatic framework maps directly onto active inference:
- Internal state setpoints = Prior preferences
- Deviation from setpoint = Prediction error (interoceptive)
- Actions that reduce deviation = Active inference policies

Evolution discovers policies that minimize long-term free energy (deviation from survival-compatible states).

### 7.2 Connection to Panksepp's Affective Neuroscience

Panksepp identified seven primary emotional systems in mammals:
- SEEKING (expectancy/desire) ↔ Approach to energy gradients
- FEAR (anxiety) ↔ Avoidance of damage gradients
- RAGE (anger) ↔ Response to blocked approach
- LUST (sexual desire) ↔ Not modeled initially (add with reproduction)
- CARE (nurturing) ↔ Not modeled initially (add with offspring)
- PANIC (separation distress) ↔ Social environment with attachment
- PLAY (social joy) ↔ Social environment with positive-sum interactions

Our framework predicts SEEKING and FEAR will emerge first (survival-critical), with others emerging as environment complexity increases.

### 7.3 Connection to Complementary Learning Systems

The dual systems of hippocampus (fast, episodic) and neocortex (slow, semantic) evolved to solve catastrophic interference. Our framework predicts:
- Heterogeneous time constants will emerge (fast and slow populations)
- Distinct connectivity patterns for fast vs slow learners
- "Consolidation-like" dynamics during periods of low activity

### 7.4 Connection to Allostasis

Allostasis extends homeostasis by emphasizing:
- Predictive regulation (act before deviation occurs)
- Context-dependent setpoints (adjust targets based on expected demands)
- Energy efficiency (minimize long-term regulatory cost)

Our biasing evaluators (F_anticipatory) specifically select for allostatic capacity.

---

## Part VIII: Implementation Roadmap

### Phase 1: Infrastructure (Months 1-3)
- Implement minimal neural substrate with evolvable plasticity
- Implement developmental encoding (CPPN → network)
- Implement homeostatic internal state dynamics
- Implement Level 0-1 environments (chemotaxis, navigation)
- Validate that simple solutions evolve

### Phase 2: Emergence Testing (Months 4-8)
- Run evolution on Level 0-2 environments
- Analyze emerged networks for predicted mechanisms
- Implement measurement checkpoints
- Compare to control conditions
- Document unexpected emergent phenomena

### Phase 3: Complexity Scaling (Months 9-14)
- Implement Level 3-4 environments (threats, social)
- Implement POET-style environment co-evolution
- Analyze scaling properties (genome size vs network size vs task complexity)
- Test transfer across environment levels
- Identify failure modes and address

### Phase 4: Biological Interpretation (Months 15-20)
- Systematic comparison of emerged circuits to biological counterparts
- Ablation studies (remove suspected "dopamine" neurons, observe effects)
- Perturbation studies (add noise to suspected "fear" circuits)
- Collaboration with neuroscientists for validation

### Phase 5: Extensions (Months 21-24)
- Add more internal state variables (sleep, social bonding hormones)
- Add developmental periods (juvenile vs adult phases)
- Add body evolution (morphology co-evolves with control)
- Write up findings for publication

---

## Appendix A: Mathematical Details

### A.1 Homeostatic Drive Function

The drive function D maps internal state H to motivational intensity:

```
D(H) = Σ_i |H_i - H_i^*|^n / σ_i^n
```

Where:
- H_i is current level of internal variable i
- H_i^* is setpoint for variable i
- σ_i is tolerance for variable i
- n controls sharpness (n=2 is quadratic)

### A.2 Reward as Drive Reduction

```
r_t = D(H_t) - D(H_{t+1})
```

Positive reward = drive decreased = moved toward setpoint
Negative reward = drive increased = moved away from setpoint

### A.3 Plasticity Gating

```
Δw_ij = (A·o_i·o_j + B·o_i + C·o_j + D) · g(m_t) · e_ij^t
```

Where:
- g(m) is a gating function of modulation signal m
- e_ij^t is eligibility trace (product of recent pre/post activity)
- m is computed by designated "modulator" neurons from network activity

### A.4 Compression Ratio

```
Compression = |phenotype| / |genotype| = N_synapses / N_genome_params
```

Higher compression = more developmental encoding = better generalization (hypothesis)

---

## Appendix B: Biological Reference Points

### B.1 Dopamine System Evolution

- Dopamine signaling exists in C. elegans (nematode)
- Conserved across 500+ million years of evolution
- Original function: motor control + simple reward learning
- Expanded function: reward prediction error, motivation, movement

**Prediction**: A neuromodulatory signal with RPE-like properties will emerge in navigation-level environments.

### B.2 Amygdala/Fear Circuit Evolution

- Fear-like responses in insects (Drosophila)
- Amygdala-like structures in all vertebrates
- Fast (subcortical) and slow (cortical) pathways
- Conditioning emerges early in evolutionary history

**Prediction**: A fast threat-detection pathway will emerge when stochastic predators are introduced.

### B.3 Hippocampal Memory System Evolution

- Hippocampus-like structures in all vertebrates
- Spatial mapping + episodic memory
- Fast learning, sparse coding
- Complementary to slow neocortical learning

**Prediction**: A fast-learning, pattern-separated subsystem will emerge when environments require memory.

---

## References (Selected)

**Homeostatic RL:**
- Keramati & Gutkin (2014). Homeostatic reinforcement learning. eLife.
- Juechems & Bhui (2024). Linking homeostasis to reinforcement learning. Current Opinion in Behavioral Sciences.

**Neuroevolution:**
- Najarro & Risi (2020). Meta-learning through Hebbian plasticity in random networks. NeurIPS.
- Stanley & Miikkulainen (2002). NEAT. Evolutionary Computation.
- Shuvaev et al. (2024). Encoding innate ability through a genomic bottleneck. PNAS.

**Affective Neuroscience:**
- Panksepp (1998). Affective Neuroscience. Oxford University Press.
- Davis & Montag (2019). Selected principles of Pankseppian affective neuroscience. Frontiers in Neuroscience.

**Active Inference:**
- Friston (2010). The free-energy principle. Nature Reviews Neuroscience.
- Pezzulo et al. (2015). Active inference and interoceptive processing. Biological Psychology.

**Environment Co-evolution:**
- Wang et al. (2019). POET: Open-ended coevolution of environments and their optimized solutions. GECCO.
- Brant & Stanley (2017). Minimal criterion coevolution. GECCO.

**Fear Circuits:**
- LeDoux (2012). Evolution of human emotion: A view through fear. Progress in Brain Research.
- Silva et al. (2016). The neural circuits of innate fear. Learning & Memory.

**Interoception:**
- Khalsa et al. (2021). Computational models of interoception and body regulation. Trends in Cognitive Sciences.
- Stephan et al. (2016). Allostatic self-efficacy. Frontiers in Psychology.
