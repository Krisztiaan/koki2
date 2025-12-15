# Evolving Biologically-Constrained Learning Agents: A Meta-Learning Framework

**Evolutionary optimization can bootstrap human-like intelligence from chemotaxis by combining genomic bottlenecking, evolvable plasticity rules, and environment co-evolution.** This framework synthesizes current SOTA research in neuroevolution, biological learning rules, and open-ended evolution to enable memory systems and learning capabilities to emerge from low-level neural building blocks. The key insight is that evolution should search the space of *learning rules*, not weights, while genomic bottlenecking forces discovery of generalizable developmental programs. This plan specifies concrete methods, architectures, and progression pathways grounded in neurobiological theory.

---

## The core architecture: evolvable plastic networks

The framework requires a three-level optimization hierarchy operating across different timescales. The **evolutionary outer loop** (thousands of generations) searches over developmental encodings and plasticity rule parameters. The **lifetime learning loop** (episode duration) adapts weights through evolved local plasticity rules. The **behavioral loop** (milliseconds) generates actions through spiking or continuous neural dynamics.

### Neural substrate selection

**Long short-term memory spiking neural networks (LSNNs)** offer the best balance of biological plausibility and trainability for this framework. Bellec et al. (2020) demonstrated that LSNNs combining leaky integrate-and-fire neurons with spike-frequency adaptation match LSTM computational capabilities while enabling biologically plausible online learning through **eligibility propagation (E-prop)**.

The E-prop learning rule computes weight updates as: **ΔW_ji = Σ_t L_j^t · e_ji^t**, where e_ji^t is a locally computable eligibility trace and L_j^t is a top-down learning signal. This three-factor rule maps naturally to neuromodulated plasticity, where dopamine or acetylcholine provides the third factor gating Hebbian associations into lasting weight changes.

For trainability during early framework development, **surrogate gradient methods** like SuperSpike (Zenke & Ganguli, 2018) or differentiable spike (DSpike) enable backpropagation through spiking dynamics by replacing the non-differentiable spike function with smooth approximations. The Zenke Lab (2024) provided theoretical grounding showing surrogate gradients connect to smoothed probabilistic models.

### Recommended hybrid architecture

The neural substrate should combine:
- **Heterogeneous LIF + ALIF populations** with diverse time constants (10-500ms)
- **Random feedback alignment** for biologically plausible credit assignment
- **Neuromodulated eligibility traces** (dopamine-gated STDP)
- **Predictive coding layers** for unsupervised representation learning

This architecture enables evolution to tune time constants, feedback distributions, neuromodulatory gain functions, and surrogate gradient shapes as meta-parameters while local plasticity handles lifetime adaptation.

---

## Genomic bottlenecking constrains the search space

### The fundamental compression problem

The human genome encodes approximately **1 GB of information** yet specifies brains with **10^15 synapses**. This massive compression ratio forces the genome to encode *rules for generating connectivity* rather than connections themselves. Shuvaev et al. (PNAS 2024) formalized this as lossy compression, showing that forcing neural architectures through a "genomic bottleneck" acts as a powerful regularizer selecting for generalizable circuits.

### Implementing indirect encodings

**Compositional Pattern-Producing Networks (CPPNs)** provide the most mature approach for indirect genetic encoding. In HyperNEAT, a small CPPN computes connection weights as: **w = CPPN(x₁, y₁, x₂, y₂)** where coordinates represent neuron positions. CPPNs use diverse activation functions (Gaussian, sine, sigmoid) that naturally encode biological regularities like symmetry, locality, and repetition with variation.

**ES-HyperNEAT** (Risi & Stanley, 2012) extends this by automatically deducing hidden node positions from variance patterns in the CPPN function—areas of uniform weight encode little information, so node density follows information content.

For this framework, the recommended progression is:

| Genome Component | What It Encodes | Compression Ratio |
|------------------|-----------------|-------------------|
| CPPN weights | Connectivity patterns | 100-300x |
| Plasticity parameters | Learning rule coefficients (A, B, C, D) | Per-synapse, evolvable |
| Time constants | Neuron dynamics | Per-population |
| Neuromodulatory mappings | When/where plasticity occurs | Network-wide |

### The genomic bottleneck algorithm

Shuvaev et al. (PNAS 2024) demonstrated that a small "G-network" generating weights for a large "P-network" achieves **322-fold compression on MNIST** with 76% initial accuracy (vs 98% trained). Critically, bottlenecked networks show **enhanced transfer learning**—the regularization effect selects circuits adaptable to novel tasks. Barabási et al. (Nature Communications 2023) showed similar results with "Genetic Connectome Models" that update wiring rules rather than weights directly.

**Recommended compression targets:**
- Simple tasks (chemotaxis): 10-50x compression
- Medium tasks (navigation): 50-100x compression  
- Complex tasks (multi-agent): 100-500x compression

Higher compression forces extraction of essential motifs while reducing overfitting to specific environments.

---

## Meta-learning through evolved plasticity rules

### Evolution of learning rules, not weights

The most promising approach for this framework is **evolving the learning rule rather than the weights**, following Najarro & Risi (NeurIPS 2020). Their generalized Hebbian ABCD model computes weight updates as:

**Δw_ij = η(A·o_i·o_j + B·o_i + C·o_j + D)**

where A, B, C, D are learnable coefficients per synapse, and o_i, o_j are pre/post-synaptic activations. Networks with random initial weights self-organize through these evolved rules, achieving:
- CarRacing-v0 convergence from random weights within ~100 timesteps
- Adaptation to morphological damage never seen during training
- Over 450,000 trainable plasticity parameters

### Neuromodulated plasticity extends capabilities

**Backpropamine** (Miconi et al., ICLR 2019) adds differentiable neuromodulation where the network itself controls when and where plasticity occurs:

**w_ij(t) = w_ij + α_ij · η · Hebb_ij(t)**

where η = tanh(h2mod(hidden_activity)) is a neuromodulatory signal computed by a subnetwork. This enables true self-modifying capabilities—the agent learns to control its own learning.

### The Baldwin effect provides evolutionary scaffolding

The **Baldwin effect** (Fernando et al., DeepMind 2018) shows how learned behaviors become genetically encoded over evolutionary time. This three-phase process:
1. Learning smooths fitness landscapes, enabling survival of individuals who *can* learn beneficial behaviors
2. Genetic assimilation gradually encodes initially learned behaviors
3. Full genetic accommodation hardwires optimal behaviors

For this framework, the Baldwin effect means evolution will discover good initializations AND learning capabilities simultaneously without requiring Lamarckian inheritance of learned weights.

### Recommended plasticity architecture

```
Meta-Parameters (Evolved):
├── Plasticity coefficients α_ij (per-synapse)
├── Hebbian rule parameters A, B, C, D
├── Neuromodulation network weights
├── Learning rate schedules
└── Time constants and decay rates

Dynamic State (Learned in Lifetime):
├── Current synaptic weights
├── Hebbian traces
├── Eligibility traces
└── Neuromodulatory state
```

---

## Memory systems emerge from low-level mechanisms

### Complementary Learning Systems as an evolutionary target

The brain's **Complementary Learning Systems (CLS)** evolved to solve catastrophic interference—rapid learning in a single network disrupts stored information. The solution pairs a fast-learning hippocampal system (sparse, pattern-separated) with a slow-learning neocortical system (distributed, overlapping).

For artificial systems, this suggests evolution should discover:
- **Dual learning rates**: Fast weights for episodic memory, slow weights for semantic integration
- **Pattern separation mechanisms**: Sparse connectivity or winner-take-all competition
- **Consolidation dynamics**: Offline replay periods that transfer memories between systems

### Building blocks for memory emergence

The framework should provide low-level mechanisms that can self-organize into memory systems:

**Attractor dynamics through recurrence**: Hopfield networks store memories as local minima in an energy landscape. Modern Hopfield networks (Ramsauer et al., 2020) achieve exponential storage capacity and connect directly to transformer attention mechanisms—the softmax attention update rule equals the Hopfield energy minimization.

**Eligibility traces for temporal credit assignment**: Three-factor learning rules (Gerstner et al., 2018) combine pre-post activity into an eligibility trace that awaits a neuromodulatory signal (dopamine for reward) to convert into lasting weight change. This solves temporal credit assignment biologically.

**Synaptic tagging and capture**: Frey & Morris (1997) showed that weak stimulation creates transient "tags" (~hours) that capture plasticity-related proteins triggered by nearby strong stimulation. Luboeinski & Tetzlaff (2021) implemented this in recurrent spiking networks, showing memory recall improves 8 hours post-learning.

**Sleep-like consolidation**: Bazhenov et al. (Nature Communications 2022) demonstrated that sleep-like unsupervised replay reduces catastrophic forgetting in ANNs. The mechanism: spontaneous reactivation during "sleep" with Hebbian plasticity strengthens important synapses while pruning irrelevant ones.

### Evolutionary pressures to drive memory emergence

Apply fitness functions that require:
- **Temporal credit assignment**: Rewards delayed from actions
- **Stability-plasticity trade-off**: Retain old skills while learning new ones
- **Generalization**: Test on novel instances, not just training patterns
- **Recall from partial cues**: Content-addressable retrieval

These pressures should drive emergence of attractor dynamics, eligibility traces, and dual-system architectures without explicit specification.

---

## Environment co-evolution and curriculum emergence

### POET demonstrates the power of co-evolution

**Paired Open-Ended Trailblazer (POET)** (Wang et al., 2019) maintains a population of environment-agent pairs where agents evolve to solve their paired environments while new environments arise from mutations of existing ones. Critical mechanisms include:

- **Minimal criterion filter**: Environments must be neither too easy nor too hard
- **Transfer mechanisms**: Agents periodically test on all environments; if a transferred agent outperforms the current paired agent, replacement occurs
- **Novelty ranking**: Candidate environments ranked by behavioral difference from archive

The key result: the same Evolution Strategies algorithm that fails alone can solve complex environments within POET's open-ended process. Hand-designed linear curricula also fail where POET succeeds.

### Minimal Criterion Coevolution simplifies further

**Minimal Criterion Coevolution (MCC)** (Brant & Stanley, 2017) uses only a simple reproductive constraint: agents must solve at least one environment to reproduce, and environments must be solvable by at least one agent. This mimics nature's fundamental constraint—survive long enough to reproduce—and produces both complexity and diversity without explicit novelty archives or fitness rankings.

### Quality-Diversity algorithms maintain innovation

**MAP-Elites** (Mouret & Clune, 2015) organizes a behavior space into grid cells, maintaining the highest-performing solution in each cell. This forces diversity while preserving quality. **PGA-MAP-Elites** (Nilsson & Cully, 2021) combines genetic operators with policy gradients for deep neuroevolution, using critic networks for efficient gradient-based variation.

### Environment progression: chemotaxis to social cognition

| Level | Environment | Cognitive Requirements | Fitness Components |
|-------|-------------|------------------------|-------------------|
| **1** | **Chemotaxis** | Gradient sensing, approach/avoid | Distance to source |
| **2** | **Navigation** | Multiple gradients, obstacles, spatial memory | Path efficiency, survival |
| **3** | **Foraging** | Resource management, energy systems, temporal planning | Sustained survival, acquisition rate |
| **4** | **Physics-rich** | Gravity, friction, tool use, body-environment interaction | Task completion, energy efficiency |
| **5** | **Multi-agent** | Other agents, communication, theory of mind, social learning | Group success, cooperation metrics |

**Progression principle**: Use MCC-style co-evolution rather than hand-designed curricula. Let complexity emerge through agent-environment coupling with transfer mechanisms enabling solutions to propagate across difficulty levels.

---

## Theoretical grounding in biological principles

### Active inference provides unified objective

The **Free Energy Principle** (Friston) posits that biological organisms minimize variational free energy—a tractable proxy for surprise. This unifies perception (inferring causes of sensations), learning (updating generative models), and action (changing the world to match predictions).

Isomura, Shimazaki & Friston (Communications Biology, 2021) showed that canonical neural networks implicitly perform active inference. For this framework, free energy minimization provides a principled objective that:
- Drives exploration through epistemic foraging (information gain terms)
- Maintains homeostasis through preferred state specifications
- Unifies reward and curiosity through expected free energy decomposition

### Predictive coding enables local learning

**Predictive coding** views the brain as hierarchical Bayesian inference—top-down connections generate predictions, bottom-up connections signal prediction errors. Millidge et al. (2022) demonstrated theoretical equivalence to backpropagation while using only local computations.

For evolved networks, predictive coding is valuable because:
- No weight transport problem (each layer learns locally)
- Same network functions as classifier, generator, and associative memory
- Arbitrary graph topologies supported (important for evolved architectures)
- More biologically plausible learning dynamics

### Information-theoretic intrinsic motivation

**Empowerment** (Mohamed & Rezende, 2015) measures an agent's control over its environment as channel capacity between actions and future sensory states. Maximizing empowerment naturally drives agents toward states with high optionality—positions of influence.

**Curiosity/Information Gain** rewards exploring states the agent cannot predict well. The combination of empowerment (agent→environment information flow) and curiosity (environment→agent information flow) provides intrinsic motivation for open-ended exploration without dense extrinsic rewards.

### Enactivism grounds embodied cognition

**Enactivism** holds that cognition emerges through embodied interaction—"cognition = life." For this framework:
- Agents should maintain their own organizational identity (autopoietic closure)
- Survival/metabolic pressures provide natural fitness functions
- Sense-making replaces representationalism with relational cognition
- Affordances (action possibilities) emerge from agent-environment coupling

---

## Implementation roadmap and scalability

### Computational infrastructure

**EvoX** (IEEE TEVC, 2024) provides GPU-accelerated evolution evaluating 1M candidate solutions per millisecond on RTX 3090 with near-linear distributed speedups. **TensorNEAT** (GECCO 2024 Best Paper) achieves 500x speedups for topology evolution via JAX tensorization.

Recommended stack:
- **Framework**: EvoX or EvoJAX for GPU-accelerated evolution
- **Topology evolution**: TensorNEAT for efficient NEAT-style complexification
- **Physics simulation**: Brax (JAX-based, vectorized)
- **Environment generation**: Procedural with CPPN-based terrain

### Genome inheritance operators

From NEAT, three critical mechanisms preserve beneficial structure:

1. **Historical markings**: Each structural mutation gets a unique global number enabling meaningful crossover between different topologies
2. **Speciation**: Groups similar networks into species; individuals compete primarily within species, protecting innovations from premature extinction
3. **Complexification from minimal structure**: Start with minimal networks, add complexity only when beneficial

**Safe Mutation (SM-G)** computes gradients of outputs w.r.t. weights for exploratory steps that don't disrupt function, enabling evolution of 100+ layer networks.

### Fitness function design

```
Total_Fitness = α₁·Survival + α₂·Task_Performance + α₃·Novelty + α₄·Learning_Capability

Where:
- Survival: Binary minimal criterion (threshold)
- Task_Performance: Environment-specific objective
- Novelty: Behavioral distance to archive
- Learning_Capability: Post-training vs pre-training performance gap
```

The **Learning_Capability** term is critical—it explicitly rewards architectures that can adapt during lifetime, not just those pre-configured for specific tasks.

---

## Synthesis: a phased research plan

### Phase 1: Foundation (Months 1-6)

**Goal**: Establish core infrastructure and validate basic components.

- Implement LSNN substrate with E-prop learning in JAX
- Build CPPN-based indirect encoding with ES-HyperNEAT-style substrate placement
- Create Level 1-2 environments (chemotaxis, simple navigation)
- Validate that evolved plasticity rules (ABCD model) enable lifetime learning
- Benchmark against direct encoding baselines

**Key milestone**: Agents evolved in chemotaxis transfer to navigation without retraining.

### Phase 2: Memory emergence (Months 6-12)

**Goal**: Demonstrate emergence of memory-like capabilities from low-level mechanisms.

- Add calcium-based plasticity dynamics enabling STC-like consolidation
- Implement neuromodulatory signals (dopamine for reward, novelty for consolidation)
- Create Level 3 environments (foraging) requiring temporal credit assignment
- Apply fitness pressures for stability-plasticity trade-off
- Analyze evolved networks for attractor dynamics, eligibility traces, dual-system structure

**Key milestone**: Evidence of working memory (persistent activity) and episodic-like recall (pattern completion).

### Phase 3: Open-ended evolution (Months 12-18)

**Goal**: Achieve unbounded complexity growth through environment co-evolution.

- Implement POET-style environment-agent co-evolution
- Add MAP-Elites for quality-diversity maintenance
- Create Level 4 environments (physics-rich) with procedural generation
- Enable transfer of evolved architectures across complexity levels
- Integrate predictive coding layers for unsupervised representation learning

**Key milestone**: Agents solving environments significantly more complex than initially possible, with spontaneous emergence of novel behaviors.

### Phase 4: Social and interpretability (Months 18-24)

**Goal**: Multi-agent dynamics and biological cross-reference.

- Create Level 5 environments (multi-agent, cooperative/competitive)
- Implement communication channels as evolvable parameters
- Develop analysis tools for comparing evolved circuits to biological systems
- Apply symbolic plasticity rule evolution (eLife 2021) for interpretable mechanisms
- Document canonical solutions and their biological analogues

**Key milestone**: Evidence of social learning, communication emergence, and identifiable biological parallels in evolved architectures.

---

## Key papers and specific citations

### Neural architectures
- Bellec et al. (2020) "A solution to the learning dilemma for recurrent networks of spiking neurons" *Nature Communications*
- Zenke & Ganguli (2018) "SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks" *Neural Computation*
- Neftci et al. (2019) "Surrogate Gradient Learning in Spiking Neural Networks" *IEEE Signal Processing Magazine*

### Genomic bottlenecking
- Shuvaev et al. (2024) "Encoding innate ability through a genomic bottleneck" *PNAS*
- Stanley (2007) "Compositional Pattern Producing Networks" *Genetic Programming and Evolvable Machines*
- Risi & Stanley (2012) "An Enhanced Hypercube-Based Encoding" *Artificial Life*

### Meta-learning and plasticity
- Najarro & Risi (2020) "Meta-Learning through Hebbian Plasticity in Random Networks" *NeurIPS*
- Miconi et al. (2019) "Backpropamine" *ICLR*
- Fernando et al. (2018) "Meta-Learning by the Baldwin Effect" *GECCO*

### Memory systems
- McClelland, McNaughton, O'Reilly (1995) "Complementary Learning Systems" *Psychological Review*
- Bazhenov et al. (2022) "Sleep replay in ANNs" *Nature Communications*
- Gerstner et al. (2018) "Three-factor learning rules" *Frontiers in Neural Circuits*

### Environment co-evolution
- Wang et al. (2019) "POET: Open-Ended Co-Evolution" *GECCO* (Best Paper)
- Brant & Stanley (2017) "Minimal Criterion Coevolution" *GECCO*
- Mouret & Clune (2015) "MAP-Elites" *arXiv*

### Theoretical frameworks
- Friston et al. (2023) "Experimental validation of the free-energy principle" *Nature Communications*
- Millidge et al. (2022) "Predictive Coding: Beyond Backpropagation?" *arXiv*
- Mohamed & Rezende (2015) "Variational Information Maximisation" *NeurIPS*

---

## Conclusion: narrowing the solution space biologically

This framework narrows the enormous search space of possible intelligent systems by imposing biologically-motivated constraints at every level:

**Architectural constraints**: LSNNs with E-prop enforce local, online learning compatible with neuromorphic hardware. Heterogeneous neuron populations with diverse time constants match cortical diversity.

**Encoding constraints**: CPPN-based genomic bottlenecking forces discovery of developmental programs that generalize, rejecting brittle solutions. Compression ratios of 100-500x select for essential circuit motifs.

**Learning constraints**: Evolving plasticity rules rather than weights searches a smaller, more general space. Neuromodulated three-factor rules match biological dopaminergic and cholinergic systems.

**Optimization constraints**: MCC-style co-evolution with speciation protects innovation while minimal criteria prevent degenerate solutions. Quality-diversity maintains behavioral coverage.

**Theoretical constraints**: Active inference provides unified objective (free energy minimization). Predictive coding ensures local learning. Information-theoretic intrinsic motivation enables exploration without dense rewards.

The result is a framework that channels evolutionary search toward biologically-plausible solutions while maintaining sufficient expressivity for complex cognition to emerge. By starting from chemotaxis and allowing complexity to grow through agent-environment co-evolution, this approach may discover the developmental and learning programs that bridge simple reflexes and human-like intelligence.