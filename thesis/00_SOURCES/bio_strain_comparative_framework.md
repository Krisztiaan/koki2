# Bio-Strain Comparative Neuroevolution Framework

## Extension to First-Principles Evolution Framework

**Core Rationale**: While the first-principles approach discovers *what* evolution invents under survival pressure, the Bio-Strain framework tests *why* specific biological patterns exist by running parallel experiments with different architectural biases. This creates a powerful comparative methodology that can:

1. **Test causality**: Does Dale's principle actually *help*, or is it just a biological accident?
2. **Accelerate discovery**: If bio-patterns are computationally useful, nudging toward them saves evolutionary time
3. **Flag deviations**: When emergence diverges from biology, is it a failure mode OR a discovery worth investigating?
4. **Provide controls**: Bio-constrained runs serve as baselines for first-principles emergence

---

## Part I: Catalog of Bio-Inspired Constraints

### 1.1 Dale's Principle (Neurotransmitter Sign Constraint)

**Biological Observation**: Neurons release the same neurotransmitter(s) from all their synaptic terminals—each neuron is either excitatory OR inhibitory, not both.

**Computational Evidence**:
- Haber & Schneidman (2022 NeurIPS): Daleian networks can approximate non-Daleian computation to high accuracy
- Daleian networks are MORE robust to synaptic noise
- Can learn efficiently by tuning single-neuron features (simpler than full weight learning)
- Spectral properties matter more than sign constraints per se (Cornford et al., 2021)

**Implementation**:
```python
class DaleConstraint:
    """Enforce Dale's principle during evolution"""
    
    def __init__(self, excitatory_fraction=0.8):
        self.exc_fraction = excitatory_fraction
    
    def apply_to_genome(self, genome):
        # Assign neuron types at birth (developmental)
        for neuron in genome.neurons:
            neuron.is_excitatory = random() < self.exc_fraction
    
    def constrain_weights(self, network):
        for neuron in network.neurons:
            if neuron.is_excitatory:
                neuron.output_weights = abs(neuron.output_weights)
            else:
                neuron.output_weights = -abs(neuron.output_weights)
    
    def fitness_penalty(self, network):
        # Soft constraint: penalize sign violations during development
        violations = sum(1 for n in network.neurons 
                        for w in n.output_weights
                        if (n.is_excitatory and w < 0) or 
                           (not n.is_excitatory and w > 0))
        return -0.01 * violations
```

**Experimental Conditions**:
- **Hard Dale**: Signs fixed at initialization, cannot change
- **Soft Dale**: Penalty for sign violations, can be overcome
- **Developmental Dale**: Sign determined by cell lineage (CPPN output)
- **No Dale** (control): Weights can have any sign

---

### 1.2 Excitatory/Inhibitory Ratio (80:20 Balance)

**Biological Observation**: Cortex maintains remarkably consistent ~4:1 ratio of excitatory to inhibitory neurons (20-30% inhibitory across mammals).

**Computational Evidence**:
- 80:20 E:I stabilizes information-rich dynamics (recent 2025 studies)
- Networks self-organize toward this ratio via homeostatic mechanisms
- Optimal for maintaining balanced dynamics in finite-size networks
- Higher E:I ratios (with Dale's law) maximize accuracy in noisy environments

**Implementation**:
```python
class EIRatioConstraint:
    """Enforce or bias toward biological E:I ratios"""
    
    def __init__(self, target_ratio=0.8, tolerance=0.1, mode='hard'):
        self.target_exc = target_ratio
        self.tolerance = tolerance
        self.mode = mode  # 'hard', 'soft', 'developmental'
    
    def constrain_network(self, network):
        if self.mode == 'hard':
            # Force exact ratio
            n_exc = int(len(network.neurons) * self.target_exc)
            assign_types_fixed(network, n_exc)
        
        elif self.mode == 'soft':
            # Bias developmental process
            current_ratio = count_excitatory(network) / len(network.neurons)
            if abs(current_ratio - self.target_exc) > self.tolerance:
                return fitness_penalty(current_ratio, self.target_exc)
    
    def developmental_bias(self, cppn_output, position):
        # Cell fate influenced by position (like cortical layers)
        base_prob = 0.8  # baseline excitatory probability
        layer_modifier = get_layer_bias(position)  # different E:I by layer
        return sigmoid(cppn_output + layer_modifier - base_prob)
```

**Layer-Specific E:I Ratios** (from biology):
| Layer | Approximate E:I Ratio |
|-------|----------------------|
| Layer 2/3 | 5.3:1 |
| Layer 4 | 7.35:1 |
| Layer 5 | 4:1 |
| Layer 6 | 5:1 |

---

### 1.3 Small-World Network Topology

**Biological Observation**: Neural networks exhibit high clustering coefficient (local connectivity) combined with short average path lengths (efficient global communication).

**Computational Evidence**:
- Supports enhanced computational power via swift information flow
- Enables simultaneous functional segregation AND integration
- Short path length + high clustering = optimal for synchronization
- Hub neurons dominate encoding of key task variables (2025 primate study)

**Implementation**:
```python
class SmallWorldBias:
    """Bias connectivity toward small-world topology"""
    
    def __init__(self, target_clustering=0.5, target_path_length=2.5):
        self.target_C = target_clustering
        self.target_L = target_path_length
    
    def compute_small_world_metrics(self, network):
        C = clustering_coefficient(network)
        L = average_path_length(network)
        
        # Small-world coefficient (Watts-Strogatz)
        C_rand, L_rand = random_network_baseline(network)
        sigma = (C / C_rand) / (L / L_rand)
        return C, L, sigma
    
    def connectivity_bias(self, source_pos, target_pos, distance):
        """Bias connection probability for small-world structure"""
        # High local connectivity (clustering)
        local_prob = exp(-distance / self.local_scale)
        
        # Sparse long-range "shortcut" connections
        longrange_prob = self.shortcut_rate if distance > self.local_cutoff else 0
        
        return local_prob + longrange_prob
    
    def fitness_component(self, network):
        C, L, sigma = self.compute_small_world_metrics(network)
        
        # Reward networks with sigma > 1 (small-world)
        if sigma > 1:
            return 0.1 * log(sigma)
        return -0.1 * (1 - sigma)
```

**Measurement Checkpoints**:
- Clustering coefficient C vs random baseline
- Path length L vs random baseline  
- Small-world propensity (SWP) - density-independent measure
- Hub neuron identification and role analysis

---

### 1.4 Neural Colonies / Modular Grouping (Cortical Columns)

**Biological Observation**: Cortex organized into vertical minicolumns (~80-120 neurons) that share inputs/outputs and function as computational units. Minicolumns aggregate into larger columns (~100 minicolumns per column).

**Computational Evidence**:
- Columnar organization reduces wiring cost while enabling specialization
- Canonical microcircuit: Layer 4 receives input → L2/3 processes → L5/6 outputs
- Neurons in same column more likely to be co-active
- Provides modular structure supporting both local computation and global integration

**Implementation**:
```python
class ColumnarOrganization:
    """Bias toward columnar/modular neural organization"""
    
    def __init__(self, column_size=100, n_layers=6):
        self.column_size = column_size
        self.n_layers = n_layers
        
    def assign_column_identity(self, neuron, position):
        """Developmental assignment of column membership"""
        column_id = spatial_hash(position.x, position.y) % self.n_columns
        layer = int(position.z * self.n_layers)
        return column_id, layer
    
    def intra_column_bias(self, source, target):
        """Strong bias for within-column connections"""
        if source.column_id == target.column_id:
            # Layer-specific connection patterns
            return self.canonical_circuit_prob(source.layer, target.layer)
        else:
            return self.inter_column_prob
    
    def canonical_circuit_prob(self, src_layer, tgt_layer):
        """Bias toward canonical cortical circuit flow"""
        # L4 → L2/3 → L5 → L6 (and feedback)
        canonical_flows = {
            (4, 2): 0.8, (4, 3): 0.8,  # L4 → L2/3
            (2, 5): 0.6, (3, 5): 0.6,  # L2/3 → L5
            (5, 6): 0.5,                # L5 → L6
            (6, 4): 0.3,                # L6 → L4 (feedback)
        }
        return canonical_flows.get((src_layer, tgt_layer), 0.1)
    
    def modularity_fitness(self, network):
        """Reward modular organization"""
        Q = compute_modularity(network, self.column_assignments)
        return 0.1 * Q if Q > 0.3 else -0.05 * (0.3 - Q)
```

**Columnar Metrics**:
- Modularity Q (Newman)
- Within-module degree z-score
- Participation coefficient (cross-module connectivity)
- Canonical circuit adherence score

---

### 1.5 Sparse Coding / Activity Sparseness

**Biological Observation**: Cortical neurons fire rarely (~1 Hz average), with sparse population codes where only a small fraction of neurons respond to any given stimulus.

**Computational Evidence**:
- Energy efficiency (spikes are metabolically expensive)
- Maximizes information per spike (bits/spike)
- Enables high-capacity associative memory
- Reduces interference between stored patterns
- Improves adversarial robustness (curved iso-response surfaces)

**Implementation**:
```python
class SparseCodingBias:
    """Bias toward sparse neural activity"""
    
    def __init__(self, target_sparsity=0.05, mode='lifetime'):
        self.target_sparsity = target_sparsity
        self.mode = mode  # 'lifetime', 'population', 'both'
    
    def compute_sparsity(self, activations, mode):
        if mode == 'lifetime':
            # Fraction of time each neuron is active
            return mean([fraction_active(a) for a in activations])
        elif mode == 'population':
            # Fraction of neurons active at each timestep
            return mean([fraction_neurons_active(t) for t in activations.T])
    
    def intrinsic_bias(self, network):
        """Bias network parameters toward sparsity"""
        # Higher thresholds promote sparsity
        for neuron in network.neurons:
            neuron.threshold *= 1.1  # slight bias toward higher thresholds
        
        # Lateral inhibition promotes sparsity
        return self.add_lateral_inhibition_bias(network)
    
    def activity_penalty(self, activations):
        """Penalize non-sparse activity during evaluation"""
        current_sparsity = self.compute_sparsity(activations, self.mode)
        deviation = abs(current_sparsity - self.target_sparsity)
        return -0.1 * deviation if deviation > 0.02 else 0
    
    def kurtosis_bonus(self, activations):
        """Reward kurtotic (heavy-tailed) activity distributions"""
        k = kurtosis(activations.flatten())
        return 0.01 * max(0, k - 3)  # excess kurtosis > 0 is good
```

**Sparsity Metrics**:
- Lifetime sparseness (per-neuron)
- Population sparseness (per-timestep)
- Activity kurtosis (heavy-tailed = sparse)
- Information per spike

---

### 1.6 Distance-Dependent Connectivity / Wiring Cost

**Biological Observation**: Connection probability decays exponentially with distance. Brains minimize total wiring length while maintaining functional connectivity.

**Computational Evidence**:
- Wiring cost minimization explains ~90% of neuron placement in C. elegans
- Optimal wiring cost correlates with high modularity and clustering
- Distance-dependent connectivity naturally produces small-world topology
- Brain networks are near-optimal for wiring cost given their topology

**Implementation**:
```python
class WiringCostConstraint:
    """Incorporate wiring cost into evolution"""
    
    def __init__(self, decay_constant=50, cost_exponent=2):
        self.lambda_ = decay_constant  # distance scale (μm equivalent)
        self.zeta = cost_exponent  # typically 2 (supralinear)
    
    def connection_probability(self, distance):
        """Exponential decay with distance"""
        return exp(-distance / self.lambda_)
    
    def compute_wiring_cost(self, network):
        """Total wiring cost as sum over all connections"""
        cost = 0
        for conn in network.connections:
            d = euclidean_distance(conn.source.position, conn.target.position)
            w = abs(conn.weight)
            cost += w * (d ** self.zeta)
        return cost
    
    def fitness_component(self, network, alpha=0.001):
        """Penalize high wiring cost"""
        cost = self.compute_wiring_cost(network)
        normalized_cost = cost / (len(network.connections) + 1)
        return -alpha * normalized_cost
    
    def developmental_constraint(self, cppn, source_pos, target_pos):
        """Bias CPPN connection decisions by distance"""
        d = euclidean_distance(source_pos, target_pos)
        base_prob = self.connection_probability(d)
        cppn_prob = cppn.query_connection(source_pos, target_pos)
        return base_prob * cppn_prob  # combined probability
```

**Wiring Metrics**:
- Total wiring length
- Wiring cost (weighted by connection strength)
- Connection distance distribution
- Cost-efficiency ratio (performance / cost)

---

### 1.7 Temporal Dynamics Constraints

**Biological Observation**: Neural dynamics operate across multiple timescales—fast synaptic transmission (ms), slower adaptation (100ms), very slow plasticity (minutes-hours).

**Implementation**:
```python
class MultiTimescaleBias:
    """Bias toward biologically realistic temporal dynamics"""
    
    def __init__(self):
        self.tau_fast = (1, 10)    # ms - fast synaptic
        self.tau_medium = (50, 200)  # ms - adaptation
        self.tau_slow = (500, 5000)  # ms - slow modulation
    
    def constrain_time_constants(self, network):
        """Ensure diversity of time constants"""
        taus = [n.tau for n in network.neurons]
        
        # Check for multi-modal distribution
        n_fast = sum(1 for t in taus if self.tau_fast[0] <= t <= self.tau_fast[1])
        n_medium = sum(1 for t in taus if self.tau_medium[0] <= t <= self.tau_medium[1])
        n_slow = sum(1 for t in taus if self.tau_slow[0] <= t <= self.tau_slow[1])
        
        # Reward diversity across timescales
        diversity = min(n_fast, n_medium, n_slow) / len(network.neurons)
        return 0.05 * diversity
    
    def spike_frequency_adaptation(self, neuron):
        """Bias toward realistic adaptation dynamics"""
        # Firing rate should decrease with sustained input
        pass
```

---

## Part II: Experimental Strains

### 2.1 Strain Definitions

We define four primary experimental strains that run in parallel:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    EXPERIMENTAL STRAINS                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  STRAIN A: UNCONSTRAINED (First-Principles)                         │
│  ───────────────────────────────────────────                        │
│  • No architectural biases                                          │
│  • Only survival pressure shapes evolution                          │
│  • Maximum discovery potential                                       │
│  • Baseline for emergence detection                                 │
│                                                                      │
│  STRAIN B: BIO-HARD (Strong Constraints)                            │
│  ───────────────────────────────────────────                        │
│  • All bio-constraints enforced as hard limits                      │
│  • Dale's law: fixed at initialization                              │
│  • E:I ratio: exactly 80:20                                         │
│  • Wiring cost: in fitness function                                 │
│  • Tests if bio-constraints help or hurt                            │
│                                                                      │
│  STRAIN C: BIO-SOFT (Soft Biases)                                   │
│  ───────────────────────────────────────────                        │
│  • Bio-patterns as soft penalties/rewards                           │
│  • Can violate constraints if beneficial                            │
│  • Tests which constraints are truly necessary                      │
│  • May discover superior non-bio solutions                          │
│                                                                      │
│  STRAIN D: BIO-ADAPTIVE (Dynamic Reinforcement)                     │
│  ───────────────────────────────────────────                        │
│  • Detects emergent bio-like patterns                               │
│  • Reinforces patterns that emerge naturally                        │
│  • Flags and investigates serious deviations                        │
│  • Accelerates discovery via adaptive bias                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Strain Configuration

```python
class ExperimentalStrain:
    """Configuration for a single experimental strain"""
    
    STRAIN_CONFIGS = {
        'A_UNCONSTRAINED': {
            'dale': None,
            'ei_ratio': None,
            'small_world': None,
            'columns': None,
            'sparsity': None,
            'wiring_cost': None,
        },
        
        'B_BIO_HARD': {
            'dale': {'mode': 'hard', 'exc_fraction': 0.8},
            'ei_ratio': {'mode': 'hard', 'target': 0.8},
            'small_world': {'mode': 'structural', 'sigma_target': 2.0},
            'columns': {'mode': 'fixed', 'size': 100},
            'sparsity': {'mode': 'penalty', 'target': 0.05},
            'wiring_cost': {'mode': 'fitness', 'alpha': 0.01},
        },
        
        'C_BIO_SOFT': {
            'dale': {'mode': 'soft', 'penalty': 0.001},
            'ei_ratio': {'mode': 'soft', 'penalty': 0.001},
            'small_world': {'mode': 'reward', 'weight': 0.01},
            'columns': {'mode': 'bias', 'strength': 0.5},
            'sparsity': {'mode': 'soft_penalty', 'weight': 0.005},
            'wiring_cost': {'mode': 'soft', 'alpha': 0.001},
        },
        
        'D_BIO_ADAPTIVE': {
            'dale': {'mode': 'adaptive'},
            'ei_ratio': {'mode': 'adaptive'},
            'small_world': {'mode': 'adaptive'},
            'columns': {'mode': 'adaptive'},
            'sparsity': {'mode': 'adaptive'},
            'wiring_cost': {'mode': 'adaptive'},
            'deviation_threshold': 0.3,
            'reinforcement_rate': 0.1,
        },
    }
```

### 2.3 Adaptive Reinforcement System (Strain D)

The Bio-Adaptive strain dynamically detects and responds to emergent patterns:

```python
class AdaptiveReinforcement:
    """Detect emergent bio-patterns and adapt constraints dynamically"""
    
    def __init__(self, detection_window=50, reinforcement_rate=0.1):
        self.window = detection_window
        self.rate = reinforcement_rate
        self.pattern_history = defaultdict(list)
        self.constraint_strengths = defaultdict(lambda: 0.0)
        
    def analyze_population(self, population, generation):
        """Detect emergent bio-like patterns in current population"""
        
        patterns = {}
        
        # Check for Dale's principle emergence
        patterns['dale_like'] = self.detect_sign_consistency(population)
        
        # Check for E:I ratio convergence
        patterns['ei_ratio'] = self.detect_ei_ratio(population)
        
        # Check for small-world emergence
        patterns['small_world'] = self.detect_small_world(population)
        
        # Check for columnar organization
        patterns['modularity'] = self.detect_modularity(population)
        
        # Check for sparse coding
        patterns['sparsity'] = self.detect_sparsity(population)
        
        return patterns
    
    def detect_sign_consistency(self, population):
        """Detect if neurons are becoming sign-consistent (Dale-like)"""
        consistencies = []
        for agent in population:
            for neuron in agent.network.neurons:
                weights = neuron.output_weights
                if len(weights) > 0:
                    # Fraction of weights with same sign as majority
                    pos = sum(1 for w in weights if w > 0)
                    neg = len(weights) - pos
                    consistency = max(pos, neg) / len(weights)
                    consistencies.append(consistency)
        return mean(consistencies) if consistencies else 0
    
    def detect_ei_ratio(self, population):
        """Detect E:I ratio in population"""
        ratios = []
        for agent in population:
            exc = sum(1 for n in agent.network.neurons 
                     if mean(n.output_weights) > 0)
            ratio = exc / len(agent.network.neurons)
            ratios.append(ratio)
        return mean(ratios)
    
    def detect_small_world(self, population):
        """Detect small-world coefficient emergence"""
        sigmas = []
        for agent in population:
            C = clustering_coefficient(agent.network)
            L = average_path_length(agent.network)
            C_rand, L_rand = random_baseline(agent.network)
            sigma = (C / C_rand) / (L / L_rand) if L_rand > 0 else 0
            sigmas.append(sigma)
        return mean(sigmas)
    
    def update_constraints(self, patterns, generation):
        """Adapt constraint strengths based on detected patterns"""
        
        for pattern_name, value in patterns.items():
            self.pattern_history[pattern_name].append(value)
            
            # Only analyze after sufficient history
            if len(self.pattern_history[pattern_name]) < self.window:
                continue
            
            recent = self.pattern_history[pattern_name][-self.window:]
            trend = self.compute_trend(recent)
            
            # If pattern is naturally emerging, reinforce it
            if self.is_converging_to_bio(pattern_name, recent):
                self.constraint_strengths[pattern_name] += self.rate
                self.log_reinforcement(pattern_name, generation, 'converging')
            
            # If pattern is diverging from bio, flag for investigation
            elif self.is_diverging_from_bio(pattern_name, recent):
                self.flag_deviation(pattern_name, generation, recent)
    
    def is_converging_to_bio(self, pattern_name, recent_values):
        """Check if pattern is trending toward biological value"""
        bio_targets = {
            'dale_like': 0.9,      # High sign consistency
            'ei_ratio': 0.8,       # 80% excitatory
            'small_world': 2.0,    # sigma > 1
            'modularity': 0.4,     # Q > 0.3
            'sparsity': 0.05,      # 5% active
        }
        
        target = bio_targets.get(pattern_name, None)
        if target is None:
            return False
        
        current = recent_values[-1]
        initial = recent_values[0]
        
        # Converging if getting closer to target
        return abs(current - target) < abs(initial - target)
    
    def flag_deviation(self, pattern_name, generation, values):
        """Flag significant deviation from biological pattern"""
        deviation_report = {
            'pattern': pattern_name,
            'generation': generation,
            'current_value': values[-1],
            'trend': self.compute_trend(values),
            'severity': self.compute_deviation_severity(pattern_name, values[-1]),
            'recommendation': self.generate_recommendation(pattern_name, values),
        }
        
        self.deviation_log.append(deviation_report)
        
        if deviation_report['severity'] > 0.5:
            print(f"⚠️ SIGNIFICANT DEVIATION: {pattern_name}")
            print(f"   Generation {generation}: value = {values[-1]:.3f}")
            print(f"   This may indicate:")
            print(f"   - A superior non-biological solution (investigate!)")
            print(f"   - A failure mode (check fitness)")
            print(f"   - Environment not requiring this constraint")
```

---

## Part III: Ablation Study Design

### 3.1 Systematic Ablation Matrix

Each constraint can be independently toggled, creating a 2^6 = 64 condition matrix:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ABLATION MATRIX                                   │
├────────┬────────┬────────┬────────┬────────┬────────┬───────────────┤
│ Dale   │ E:I    │ S.W.   │ Column │ Sparse │ Wiring │ Condition ID  │
├────────┼────────┼────────┼────────┼────────┼────────┼───────────────┤
│   ✗    │   ✗    │   ✗    │   ✗    │   ✗    │   ✗    │ 0 (BASELINE)  │
│   ✓    │   ✗    │   ✗    │   ✗    │   ✗    │   ✗    │ 1 (Dale only) │
│   ✗    │   ✓    │   ✗    │   ✗    │   ✗    │   ✗    │ 2 (E:I only)  │
│   ✓    │   ✓    │   ✗    │   ✗    │   ✗    │   ✗    │ 3 (Dale+E:I)  │
│   ...  │   ...  │   ...  │   ...  │   ...  │   ...  │ ...           │
│   ✓    │   ✓    │   ✓    │   ✓    │   ✓    │   ✓    │ 63 (ALL BIO)  │
└────────┴────────┴────────┴────────┴────────┴────────┴───────────────┘
```

### 3.2 Ablation Analysis Protocol

```python
class AblationAnalysis:
    """Systematic analysis of constraint contributions"""
    
    def __init__(self, constraints):
        self.constraints = constraints
        self.n_constraints = len(constraints)
        self.results = {}
        
    def run_full_ablation(self, n_seeds=10):
        """Run all 2^n conditions with multiple seeds"""
        
        for condition_id in range(2 ** self.n_constraints):
            active_constraints = self.decode_condition(condition_id)
            
            for seed in range(n_seeds):
                result = self.run_condition(active_constraints, seed)
                self.results[(condition_id, seed)] = result
    
    def compute_main_effects(self):
        """Compute main effect of each constraint"""
        main_effects = {}
        
        for i, constraint in enumerate(self.constraints):
            # Average performance WITH constraint
            with_constraint = [r for (cid, _), r in self.results.items()
                             if (cid >> i) & 1]
            
            # Average performance WITHOUT constraint
            without_constraint = [r for (cid, _), r in self.results.items()
                                 if not ((cid >> i) & 1)]
            
            effect = mean([r['fitness'] for r in with_constraint]) - \
                     mean([r['fitness'] for r in without_constraint])
            
            main_effects[constraint] = {
                'effect_size': effect,
                'p_value': ttest_ind(
                    [r['fitness'] for r in with_constraint],
                    [r['fitness'] for r in without_constraint]
                ).pvalue,
            }
        
        return main_effects
    
    def compute_interactions(self):
        """Compute pairwise interaction effects"""
        interactions = {}
        
        for i, c1 in enumerate(self.constraints):
            for j, c2 in enumerate(self.constraints):
                if i >= j:
                    continue
                
                # 2x2 factorial analysis
                both = [r for (cid, _), r in self.results.items()
                       if ((cid >> i) & 1) and ((cid >> j) & 1)]
                c1_only = [r for (cid, _), r in self.results.items()
                          if ((cid >> i) & 1) and not ((cid >> j) & 1)]
                c2_only = [r for (cid, _), r in self.results.items()
                          if not ((cid >> i) & 1) and ((cid >> j) & 1)]
                neither = [r for (cid, _), r in self.results.items()
                          if not ((cid >> i) & 1) and not ((cid >> j) & 1)]
                
                # Interaction = (both - c1_only) - (c2_only - neither)
                interaction = (mean_fitness(both) - mean_fitness(c1_only)) - \
                             (mean_fitness(c2_only) - mean_fitness(neither))
                
                interactions[(c1, c2)] = interaction
        
        return interactions
    
    def identify_minimal_sufficient_set(self):
        """Find smallest set of constraints achieving near-optimal performance"""
        
        full_bio_fitness = mean([r['fitness'] for (cid, _), r in self.results.items()
                                if cid == 2**self.n_constraints - 1])
        
        # Search for minimal sets within 95% of full bio
        threshold = 0.95 * full_bio_fitness
        
        minimal_sets = []
        for n_active in range(1, self.n_constraints + 1):
            for combo in combinations(range(self.n_constraints), n_active):
                condition_id = sum(1 << i for i in combo)
                results = [r for (cid, _), r in self.results.items() 
                          if cid == condition_id]
                
                if mean([r['fitness'] for r in results]) >= threshold:
                    minimal_sets.append(combo)
        
        return min(minimal_sets, key=len) if minimal_sets else None
```

---

## Part IV: Deviation Detection and Investigation

### 4.1 Deviation Classification

When emergent patterns diverge from biological norms, we classify them:

```
┌─────────────────────────────────────────────────────────────────────┐
│                 DEVIATION CLASSIFICATION                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  TYPE 1: SUPERIOR NON-BIO SOLUTION                                  │
│  ─────────────────────────────────                                  │
│  • Deviation achieves higher fitness than bio-constrained           │
│  • Consistent across seeds and environments                         │
│  • Action: INVESTIGATE - potential discovery!                       │
│  • Questions:                                                        │
│    - Why does evolution prefer this?                                │
│    - Is biology suboptimal, or is our environment wrong?            │
│    - What tradeoff is biology making that we're not modeling?       │
│                                                                      │
│  TYPE 2: ENVIRONMENT-SPECIFIC SOLUTION                              │
│  ─────────────────────────────────────                              │
│  • Deviation only in specific environment conditions                │
│  • Bio-patterns emerge in other conditions                          │
│  • Action: INVESTIGATE environment design                           │
│  • Questions:                                                        │
│    - What pressure is missing that would select for bio-pattern?    │
│    - Is our environment too simple/different from biology?          │
│                                                                      │
│  TYPE 3: LOCAL OPTIMUM / FAILURE MODE                               │
│  ─────────────────────────────────────                              │
│  • Lower fitness than bio-constrained                               │
│  • Evolution stuck in suboptimal region                             │
│  • Action: Adjust evolutionary parameters                           │
│  • Solutions:                                                        │
│    - Increase population diversity                                   │
│    - Adjust mutation rates                                           │
│    - Add explicit diversity maintenance                              │
│                                                                      │
│  TYPE 4: SCALE-DEPENDENT DIFFERENCE                                 │
│  ─────────────────────────────────────                              │
│  • Pattern differs at current network scale                         │
│  • May converge to bio at larger scales                             │
│  • Action: Test at multiple scales                                   │
│  • Note: Some bio-patterns only emerge at sufficient scale          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Deviation Investigation Protocol

```python
class DeviationInvestigator:
    """Systematic investigation of bio-pattern deviations"""
    
    def investigate_deviation(self, deviation_report, strain_results):
        """Full investigation protocol for a detected deviation"""
        
        pattern = deviation_report['pattern']
        
        investigation = {
            'pattern': pattern,
            'deviation_type': None,
            'findings': [],
            'recommendations': [],
        }
        
        # Step 1: Compare fitness
        fitness_comparison = self.compare_fitness_with_bio(
            strain_results, pattern)
        
        if fitness_comparison['deviant_better']:
            investigation['deviation_type'] = 'TYPE_1_SUPERIOR'
            investigation['findings'].append(
                f"Non-bio solution achieves {fitness_comparison['improvement']:.1%} "
                f"higher fitness")
        
        # Step 2: Check environment dependency
        env_analysis = self.analyze_environment_dependency(
            strain_results, pattern)
        
        if env_analysis['environment_specific']:
            investigation['deviation_type'] = 'TYPE_2_ENVIRONMENT'
            investigation['findings'].append(
                f"Deviation specific to environments: {env_analysis['specific_envs']}")
        
        # Step 3: Check for local optima
        diversity_analysis = self.check_population_diversity(
            strain_results, pattern)
        
        if diversity_analysis['low_diversity']:
            investigation['deviation_type'] = 'TYPE_3_LOCAL_OPTIMUM'
            investigation['recommendations'].append(
                "Increase population diversity or mutation rates")
        
        # Step 4: Scale analysis
        scale_analysis = self.analyze_scale_dependency(
            strain_results, pattern)
        
        if scale_analysis['scale_dependent']:
            investigation['deviation_type'] = 'TYPE_4_SCALE'
            investigation['findings'].append(
                f"Pattern shows scale dependency: converges to bio at N > "
                f"{scale_analysis['convergence_threshold']}")
        
        # Generate detailed report
        self.generate_investigation_report(investigation)
        
        return investigation
    
    def analyze_why_superior(self, deviation_report, networks):
        """Deep analysis of why non-bio solution is superior"""
        
        analysis = {
            'computational_advantage': None,
            'missing_pressure': None,
            'tradeoff_avoided': None,
        }
        
        # Analyze computational properties
        for network in networks['deviant']:
            props = self.extract_computational_properties(network)
            # What can this network do that bio-constrained can't?
            
        for network in networks['bio']:
            props = self.extract_computational_properties(network)
            # What limitations does bio-constraint impose?
        
        # Identify missing selective pressures
        analysis['missing_pressure'] = self.identify_missing_pressures(
            deviation_report['pattern'])
        
        return analysis
```

---

## Part V: Cross-Strain Comparison Metrics

### 5.1 Comparison Dashboard

```python
class CrossStrainComparison:
    """Compare results across all experimental strains"""
    
    def __init__(self, strain_results):
        self.results = strain_results
        
    def generate_comparison_table(self):
        """Generate comprehensive comparison across strains"""
        
        metrics = [
            'final_fitness',
            'generations_to_threshold',
            'fitness_variance',
            'transfer_performance',
            'emergent_complexity',
        ]
        
        bio_metrics = [
            'dale_consistency',
            'ei_ratio',
            'small_world_sigma',
            'modularity',
            'activity_sparsity',
            'wiring_cost',
        ]
        
        table = pd.DataFrame()
        
        for strain in ['A_UNCONSTRAINED', 'B_BIO_HARD', 'C_BIO_SOFT', 'D_BIO_ADAPTIVE']:
            strain_data = self.results[strain]
            
            row = {
                'strain': strain,
                **{m: self.compute_metric(strain_data, m) for m in metrics},
                **{f'bio_{m}': self.compute_bio_metric(strain_data, m) 
                   for m in bio_metrics},
            }
            
            table = table.append(row, ignore_index=True)
        
        return table
    
    def compute_emergence_comparison(self):
        """Compare what emerged in unconstrained vs bio-constrained"""
        
        unconstrained = self.results['A_UNCONSTRAINED']
        bio_hard = self.results['B_BIO_HARD']
        
        comparison = {
            'patterns_emerged_both': [],
            'patterns_emerged_unconstrained_only': [],
            'patterns_emerged_bio_only': [],
            'functional_equivalents': [],
        }
        
        # Check each bio-pattern
        for pattern in ['dale', 'ei_ratio', 'small_world', 'columns', 'sparsity']:
            unc_value = self.measure_pattern(unconstrained, pattern)
            bio_value = self.measure_pattern(bio_hard, pattern)
            
            if unc_value > 0.7 and bio_value > 0.7:  # threshold for "emerged"
                comparison['patterns_emerged_both'].append(pattern)
            elif unc_value > 0.7:
                comparison['patterns_emerged_unconstrained_only'].append(pattern)
            elif bio_value > 0.7:
                comparison['patterns_emerged_bio_only'].append(pattern)
        
        return comparison
    
    def identify_functional_equivalents(self):
        """Find non-bio solutions that achieve same function"""
        
        equivalents = []
        
        # For each bio-pattern, check if unconstrained found alternative
        for pattern, function in BIO_PATTERN_FUNCTIONS.items():
            bio_achieves = self.test_function(
                self.results['B_BIO_HARD'], function)
            unc_achieves = self.test_function(
                self.results['A_UNCONSTRAINED'], function)
            
            if bio_achieves and unc_achieves:
                bio_mechanism = self.extract_mechanism(
                    self.results['B_BIO_HARD'], function)
                unc_mechanism = self.extract_mechanism(
                    self.results['A_UNCONSTRAINED'], function)
                
                if bio_mechanism != unc_mechanism:
                    equivalents.append({
                        'function': function,
                        'bio_mechanism': bio_mechanism,
                        'evolved_mechanism': unc_mechanism,
                    })
        
        return equivalents
```

### 5.2 Key Research Questions by Comparison

| Comparison | Research Question | Expected Insight |
|------------|------------------|------------------|
| A vs B (fitness) | Does bio-constraint help or hurt? | Whether biology is optimized |
| A vs B (patterns) | Do bio-patterns emerge naturally? | Whether patterns are necessary |
| C vs B | Which constraints are necessary vs sufficient? | Minimal bio-constraint set |
| D vs A | Does adaptive bias accelerate discovery? | Optimal training strategy |
| A (deviations) | What superior non-bio solutions exist? | Potential discoveries |
| All strains (transfer) | Which produces most generalizable agents? | Best training paradigm |

---

## Part VI: Integration with First-Principles Framework

### 6.1 Combined Experimental Protocol

```python
class CombinedEvolutionExperiment:
    """Run first-principles and bio-strain experiments together"""
    
    def __init__(self, environment_suite, strains=['A', 'B', 'C', 'D']):
        self.environments = environment_suite
        self.strains = strains
        self.results = {}
        
    def run_parallel_evolution(self, n_generations, n_seeds):
        """Run all strains in parallel for each environment"""
        
        for env_name, env in self.environments.items():
            print(f"\n{'='*60}")
            print(f"Environment: {env_name}")
            print(f"{'='*60}")
            
            for strain in self.strains:
                print(f"\n  Strain {strain}...")
                
                config = ExperimentalStrain.STRAIN_CONFIGS[f'{strain}_*']
                
                for seed in range(n_seeds):
                    result = run_evolution(
                        env=env,
                        constraints=config,
                        n_generations=n_generations,
                        seed=seed,
                    )
                    
                    self.results[(env_name, strain, seed)] = result
                    
                    # Real-time monitoring for Strain D
                    if strain == 'D':
                        self.monitor_adaptive_strain(result, env_name, seed)
        
        return self.results
    
    def analyze_across_environments(self):
        """Analyze which constraints matter in which environments"""
        
        env_constraint_importance = {}
        
        for env_name in self.environments:
            strain_fitness = {}
            for strain in self.strains:
                fitness = mean([self.results[(env_name, strain, s)]['fitness']
                               for s in range(self.n_seeds)])
                strain_fitness[strain] = fitness
            
            # Which constraints helped in this environment?
            env_constraint_importance[env_name] = {
                'bio_helps': strain_fitness['B'] > strain_fitness['A'],
                'bio_advantage': strain_fitness['B'] - strain_fitness['A'],
                'soft_vs_hard': strain_fitness['C'] - strain_fitness['B'],
                'adaptive_advantage': strain_fitness['D'] - strain_fitness['A'],
            }
        
        return env_constraint_importance
```

### 6.2 Environment-Constraint Interaction Predictions

Based on the research, we predict:

| Environment Level | Expected Constraint Value |
|------------------|--------------------------|
| **Chemotaxis** | Minimal benefit from bio-constraints; too simple |
| **Navigation** | Sparse coding + wiring cost should help |
| **Foraging** | E:I balance important for stable dynamics |
| **Threats** | Small-world enables fast threat response |
| **Social** | Columnar organization for multi-agent modeling |

---

## Part VII: Implementation Roadmap Extension

### Phase 1 (Months 1-3): Add to Base Framework
- Implement all 6 bio-constraint modules
- Create ablation testing infrastructure
- Build cross-strain comparison metrics

### Phase 2 (Months 4-6): Run Initial Comparisons  
- Run all 4 strains on Level 0-2 environments
- Initial ablation studies (32 conditions)
- Calibrate adaptive reinforcement system

### Phase 3 (Months 7-10): Full Analysis
- Complete 64-condition ablation
- Deep investigation of any deviations
- Identify minimal sufficient constraint sets

### Phase 4 (Months 11-14): Publication & Extensions
- Prepare comparative analysis paper
- Document functional equivalents discovered
- Extend to Level 3-4 environments

---

## Appendix A: Bio-Constraint Quick Reference

| Constraint | Biological Value | Implementation Mode | Key Metric |
|------------|-----------------|---------------------|------------|
| Dale's Principle | 100% sign-consistent | Hard/Soft/Developmental | Sign consistency |
| E:I Ratio | 80:20 (varies by layer) | Hard/Soft | Exc/Total ratio |
| Small-World | σ > 1 | Structural/Reward | Small-world coefficient |
| Columnar Org. | ~100 neurons/column | Fixed/Bias | Modularity Q |
| Sparse Coding | ~5% active | Penalty/Soft | Population sparsity |
| Wiring Cost | Minimized | Fitness/Soft | Total wire length |

## Appendix B: Key References

**Dale's Principle:**
- Haber & Schneidman (2022) NeurIPS: "Computational and learning benefits of Daleian networks"
- Cornford et al. (2021) ICLR: "Learning to live with Dale's principle"

**E:I Balance:**
- 80/20 cortical balance paper (2025): "stabilizes information-rich dynamics"
- PNAS (2021): "Networks adapt via connection number changes"

**Small-World:**
- Sporns (2016): "Small-World Brain Networks Revisited"
- Research Square (2025): "Single-neuron network topology governs computation"

**Cortical Columns:**
- Mountcastle (1997): Original columnar hypothesis
- Horton & Adams (2005): "The cortical column: a structure without a function"

**Sparse Coding:**
- Olshausen & Field (1996): Original sparse coding in V1
- Willshaw et al. (2019): "Neural correlates of sparse coding"

**Wiring Cost:**
- PNAS Nexus (2024): "Brain-inspired wiring economics for ANNs"
- Chen et al. (2006): "Wiring optimization can relate structure and function"
