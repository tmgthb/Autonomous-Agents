<!--Autonomous Agents -->
<!--
Copyright (C) Teemu Maatta. 

@misc{MaattaAutonomousAgents2023,
  author = {Teemu Maatta},
  title = {Autonomous Agents},
  year = {2023},
  howpublished = {\url{http://github.com/tmgthb/Autonomous-Agents}},
  note = {Accessed: YYYY-MM-DD}
}
-->
<div id="topofthepage"> </div>

<div align="center">

[![Hits](http://hits.sh/github.com/tmgthb/Autonomous-Agents.svg?view=today-total&label=Views&color=007ec6)](http://hits.sh/github.com/tmgthb/Autonomous-Agents/)
[![X](http://img.shields.io/twitter/follow/Teemumtt3?style=social)](http://twitter.com/Teemumtt3)
[![GitHub Repo stars](http://img.shields.io/github/stars/tmgthb/Autonomous-Agents?style=flat-square)](http://github.com/tmgthb/Autonomous-Agents/stargazers)

</div>

<p align="center">
  <img height="100" src="https://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_agent_logo.png" alt="Autonomous Agents">
</p>

<div align="center">

  # Autonomous Agents
  Autonomous Agents-research papers. Updated daily. [Resources-section](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Resources.md)-section.  

</div>


---

<div id="researchpapers" align="center">

## Research papers: 2026 4/5

[2026 (5/5)](http://github.com/tmgthb/Autonomous-Agents/blob/main/README.md), [2026 (4/5)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2026_4.md), [2026 (3/5)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2026_3.md), [2026 (2/5)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2026_2.md), [2026 (1/5)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2026_1.md), [2025 (4/4)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2025_4.md),[2025 (3/4)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2025_3.md), [2025 (2/4)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2025_2.md), [2025 (1/4)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2025_01.md), [2024](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2024.md), [2023](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2023.md), [Earlier](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_Earlier.md)


Chronological order. 





</div>



---








#### 11th May 2026



[Instruction Adherence in Coding Agent Configuration Files: A Factorial Study of Four File-Structure Variables](http://arxiv.org/abs/2605.10039)

- Instruction Adherence in Coding Agent Configuration Files: introduces a systematic factorial study evaluating how four structural variables in configuration files affect LLM agent compliance with behavioral instructions.
- The study utilizes an AST-based scoring pipeline to measure compliance across 1,650 sessions, finding that file size, instruction position, architecture, and conflicting instructions do not significantly impact adherence.
- The research identifies a significant within-session attenuation effect where LLM agent compliance decreases by approximately 5.6% per generated function, highlighting task identity as a stronger predictor of adherence than configuration structure.

---


[DataMaster: Towards Autonomous Data Engineering for Machine Learning](http://arxiv.org/abs/2605.10906)

- DataMaster: introduces a data-agent framework that treats the data state as the primary optimization target by integrating DataTree (tree-structured search over data states), Data Pool (shared repository for external datasets), and Global Memory (persistent record of outcomes and findings).
- The framework utilizes Red Nodes (agents for external data discovery) and Black Nodes (agents for data refinement and exploitation) to iteratively improve downstream performance under a fixed learning algorithm.
- DataMaster employs a UCB-based scheduling policy and dynamic tree growth to manage limited evaluation budgets while enabling cumulative learning across data-engineering branches.

---

[Shields to Guarantee Probabilistic Safety in MDPs](http://arxiv.org/abs/2605.10888)

- Shielding Framework: introduces a formal framework for probabilistic safety in Markov decision processes that conservatively extends classical shielding techniques.
- The framework provides various shield instantiations, including optimistic, pessimistic, and saturated shields, to balance safety guarantees with policy permissiveness.
- The research demonstrates that combining maximal permissiveness and probabilistic safety is impossible, and provides efficient offline and online learning routines for constructing safe, permissive shields.

---

[Agent-First Tool APIs: Rethinking Enterprise Service Interfaces for LLM-Native Execution](http://arxiv.org/abs/2605.10555)

- Agent-First Tool API: introduces a goal-achievement paradigm for enterprise software that replaces traditional CRUD interfaces with a Six-Verb Semantic Protocol, a Normalized Tool Contract (NTC), and a Dual-layer Governance Pipeline.
- The framework enables LLM agents to perform autonomous tasks by providing structured feedback, confidence scores, and built-in risk assessment, effectively reducing ambiguity-induced failures.
- By separating capability-based and object-scoped permissions, the architecture ensures enterprise-grade security and auditability while maintaining compatibility with existing transport standards like the Model Context Protocol.

---

[Causal Explanations from the Geometric Properties of ReLU Neural Networks](http://arxiv.org/abs/2605.10396)

- ReLU Neural Network Explainability Framework: introduces a method for generating exact causal "why" and "why not" explanations for ReLU-based neural networks by leveraging their geometric properties through Polyhedral Decomposition, Bit Vector Representation, Linear Programming, Adjacent Polytope Marching, H-Representation, and V-Representation.
- The framework utilizes the piecewise linear nature of ReLU networks to partition the input space into convex polytopes, allowing for the extraction of minimally complete explanations directly from the network's internal geometry.
- The approach addresses the black-box nature of control policies by providing exact, verifiable explanations while discussing the trade-offs between H-Representation and V-Representation regarding computational complexity and interpretability.

---

[Agent-ValueBench: A Comprehensive Benchmark for Evaluating Agent Values](http://arxiv.org/abs/2605.10365)

- Agent-ValueBench: introduces a comprehensive benchmark for evaluating the values of autonomous agents by synthesizing executable environments and value-conflict tasks.
- The framework utilizes an automated pipeline with test- and repair-agents to generate validated environments, which are then assessed using trajectory-level rubrics and an LLM-as-Judge.
- Experimental results across 14 models and 4 harnesses reveal that agent values exhibit a "Value Tide" of population-wide homogenization that bends under harness pull and deliberate skill steering.

---

[Merlin: Deterministic Byte-Exact Deduplication for Lossless Context Optimization in Large Language Model Inference](http://arxiv.org/abs/2605.09990)

- Merlin: introduces a deterministic, byte-exact deduplication engine designed for real-time LLM inference preprocessing, utilizing an Input Stream, Fingerprint Pass, L2-Aligned Memory Arena, Lock-Free Routing, and Output Stream.
- The engine operates as a low-latency, CPU-bound sidecar that removes redundant context records before prompt assembly without impacting model quality.
- Empirical validation across multiple benchmarks and production LLMs demonstrates that the engine maintains lossless performance while operating significantly below typical inference latency budgets.

---

[Personal Visual Context Learning in Large Multimodal Models](http://arxiv.org/abs/2605.10936)

- Agentic Context Bank: introduces a framework that structures a user’s visual history into a self-refining memory bank and employs query-adaptive evidence selection to improve LMM reasoning over personal visual context.
- The framework addresses the modality paradox and scaling paradox by replacing passive visual concatenation with a two-stage process involving structured memory construction and selective visual verification.
- The research introduces Personal-VCL-Bench to evaluate LMM performance across three axes of personal visual knowledge: persons, objects, and behavior.

---

[Optimal and Scalable MAPF via Multi-Marginal Optimal Transport and Schrödinger Bridges](http://arxiv.org/abs/2605.10917)

- MAPF-MMOT: introduces a framework that casts multi-agent path finding as a multi-marginal optimal transport problem, utilizing P1, P2, P3, Sinkhorn-MAPF, and Shadow Transport to achieve scalable and optimal robot trajectories.
- The approach leverages the total unimodularity of the underlying linear program to guarantee integral solutions while employing Schrödinger bridges for probabilistic relaxation and efficient graph pruning.
- Experimental results demonstrate that the proposed pipeline provides significant speedups over existing methods while maintaining near-optimal cost performance in large-scale multi-agent path finding scenarios.

---

[SHEPHERD: A Runtime Substrate Empowering Meta-Agents with a Formalized Execution Trace](http://arxiv.org/abs/2605.10913)

- SHEPHERD: introduces a functional programming model that formalizes meta-agent operations on target agents as functions, with core operations mechanized in Lean.
- The framework treats agent execution as a first-class object, enabling meta-agents to read, rewind, branch, and modify worker agent trajectories through a Git-like execution trace.
- SHEPHERD supports diverse agentic applications, including live supervision, counterfactual meta-optimization, and meta-agent guided Tree-RL training, while providing efficient state-forking and prompt-cache reuse.

---

[Revisiting Policy Gradients for Restricted Policy Classes: Escaping Myopic Local Optima with k-step Policy Gradients](http://arxiv.org/abs/2605.10909)

- k-step Policy Gradient Framework: introduces a generalized k-step policy gradient method that utilizes a k-step Q-function to overcome the myopic nature of standard policy gradient methods in restricted policy classes.
- The framework employs correlated policies to enable multi-step reasoning, effectively escaping suboptimal local optima that arise from independent action sampling.
- Theoretical analysis demonstrates that this approach achieves exponential convergence to near-optimal solutions in O(1/T) iterations for both projected gradient descent and mirror descent.

---

[Engineering Robustness into Personal Agents with the AI Workflow Store](http://arxiv.org/abs/2605.10907)

- AI Workflow Store: introduces a framework that integrates rigorous software engineering processes into agentic loops to produce hardened, reusable workflows instead of relying on improvised on-the-fly synthesis.
- The architecture utilizes a Backend SE Agent Team to engineer workflows, a Workflow Repository to store them, and a Local Agent to match user prompts to these vetted artifacts.
- By amortizing the cost of rigorous engineering across a community of users, the framework aims to achieve production-grade reliability and security in personal agents without sacrificing flexibility.

---

[MDrive: Benchmarking Closed-Loop Cooperative Driving for End-to-End Multi-agent Systems](http://arxiv.org/abs/2605.10904)

- MDrive: introduces a closed-loop cooperative driving benchmark that systematically evaluates multi-agent systems using Human-in-the-Loop Simulation Interface, Agentic Scenario Generation Pipeline, and Real-to-Simulation (Real2Sim) Pipeline.
- The framework utilizes a Simulation Controller to manage reactive agents and human demonstrations, while employing Scenario Feasibility Check and Duplicate &amp; Difficulty Screening to ensure high-quality, diverse test scenarios.
- MDrive integrates Behavior Extraction, Asset Matching, and Coordinate Alignment to reconstruct real-world V2X driving logs into closed-loop CARLA scenarios for robust multi-agent evaluation.

---

[RubricEM: Meta-RL with Rubric-guided Policy Decomposition beyond Verifiable Rewards](http://arxiv.org/abs/2605.10899)

- RubricEM: introduces a rubric-guided reinforcement learning framework that combines stagewise policy decomposition with reflection-based meta-policy training to optimize long-horizon research agents.
- The framework utilizes a Stage-Structured Scaffold to organize trajectories into distinct decision modes, enabling Stage-Structured GRPO to provide dense semantic feedback for long-horizon credit assignment.
- A shared-backbone reflection meta-policy distills judged trajectories into reusable rubric-grounded guidance stored in an Agent Rubric Bank, facilitating both within-episode refinement and cross-episode transfer.

---

[AssayBench: An Assay-Level Virtual Cell Benchmark for LLMs and Agents](http://arxiv.org/abs/2605.10876)

- AssayBench: introduces a large-scale benchmark for phenotypic screen prediction, utilizing a Data Curation Pipeline, LLM-assisted Curation, and a Gene Relevance Scorer to evaluate LLMs and agents as virtual cell surrogates.
- The framework frames phenotypic screening as a gene-ranking task, employing a Neural Gene-relevance Predictor, an LLM Ensemble, and Retrieval Baselines to assess model performance across heterogeneous CRISPR assays.
- Evaluation is conducted using the Adjusted normalized Discounted Cumulative Gain (AnDCG@k) metric, which corrects for screen-specific random baselines to enable continuous performance comparison across diverse biological contexts.

---

[Remember the Decision, Not the Description: A Rate-Distortion Framework for Agent Memory](http://arxiv.org/abs/2605.10870)

- DeMem: introduces a decision-centric memory framework that optimizes memory allocation by preserving distinctions necessary for downstream decisions rather than descriptive fidelity.
- The framework utilizes a K-slot memory system that routes history-query pairs to specific slots and performs certified refinement only when shared memory induces decision conflicts.
- DeMem provides a memory-distortion frontier and near-minimax regret guarantees, demonstrating consistent performance gains in long-horizon conversational and agentic tasks.

---

[Is Your Driving World Model an All-Around Player?](http://arxiv.org/abs/2605.10858)

- WorldLens: introduces a unified benchmark for evaluating driving world models across five complementary axes: generation, reconstruction, action-following, downstream tasks, and human preference.
- The framework utilizes WorldLens-26K, a large-scale human-annotated dataset, to train WorldLens-Agent, a vision-language model that provides automated, explainable assessments of world model fidelity.
- The research demonstrates that current world models often prioritize visual appearance over physical and behavioral consistency, highlighting a critical need for holistic evaluation protocols.

---

[The Generalized Turing Test: A Foundation for Comparing Intelligence](http://arxiv.org/abs/2605.10851)

- GTT (Generalized Turing Test): introduces a formal, dataset-agnostic framework for evaluating LLMs by measuring their ability to imitate other models such that a distinguisher cannot reliably identify the actor.
- The framework utilizes an Actor, a Distinguisher, and an optional Specimen to compute a Turing Comparator, which establishes a relative intelligence ordering between conversational agents.
- Empirical results across nine modern LLMs demonstrate that the GTT recovers a stratified intelligence hierarchy consistent with existing benchmarks while providing a closed-loop, adaptive evaluation mechanism.

---

[Rethinking Agentic Search with PI-SERINI: Is Lexical Retrieval Sufficient?](http://arxiv.org/abs/2605.10848)

- PI-SERINI: introduces a search agent that isolates agent-retriever interaction to demonstrate that a well-configured lexical retriever can support effective deep research when paired with capable LLMs.
- The framework utilizes a retrieval controller to manage cached rankings and selective evidence acquisition, enabling efficient deep research without relying on dense retrievers.
- Experimental results on the BrowseComp-Plus benchmark show that PI-SERINI achieves competitive accuracy and high recall while significantly reducing evaluation costs compared to existing dense-retriever search agents.

---

[Training-Free Cultural Alignment of Large Language Models via Persona Disagreement](http://arxiv.org/abs/2605.10843)

- DISCA (Disagreement-Informed Steering for Cultural Alignment): introduces a training-free inference-time alignment method that uses within-country persona disagreement as a reliability signal to steer LLMs toward culturally grounded moral preferences.
- The framework utilizes WVS-grounded persona agents to generate logit-based disagreement signals, which are then processed via Prospect-Theory importance sampling and a dual-pass reliability gate to compute a bounded, loss-averse logit correction.
- By treating disagreement as a sufficient statistic for correction reliability rather than noise, DISCA achieves robust cultural alignment across diverse LLM backbones without requiring weight updates, per-country reward models, or internal model access.

---

[From Controlled to the Wild: Evaluation of Pentesting Agents for the Real-World](http://arxiv.org/abs/2605.10834)

- Evaluation Protocol for AI Pentesting Agents: introduces a methodology for assessing agentic security systems using a structured finding-to-ground-truth pipeline with LLM-as-a-judge, bipartite matching, and cumulative performance analysis.
- The protocol shifts evaluation from binary task completion to validated vulnerability discovery, supporting realistic assessment across complex, open-ended targets.
- It incorporates efficiency metrics and continuous ground-truth maintenance to provide an operationally informative comparison of stochastic LLM-based pentesting agents.

---

[Towards On-Policy Data Evolution for Visual-Native Multimodal Deep Search Agents](http://arxiv.org/abs/2605.10832)

- ODE (On-policy Data Evolution): introduces a closed-loop data construction framework that refines training tasks for multimodal deep search agents by using rollout feedback to update generator configurations.
- The framework utilizes a Visual-Native Agent Harness that employs an Image Bank Reference Protocol to make tool-produced visual evidence persistently reusable across multi-step trajectories.
- By treating data generation as an adaptive optimization process, ODE aligns training data with the evolving capabilities of the target LLM agent, significantly improving performance on complex multimodal search benchmarks.

---

[The First Drop of Ink: Nonlinear Impact of Misleading Information in Long-Context Reasoning](http://arxiv.org/abs/2605.10828)

- The First Drop of Ink: introduces a systematic study of how the proportion of hard distractors in long-context LLMs causes a nonlinear, front-loaded performance degradation.
- The research demonstrates that hard distractors dominate the softmax attention denominator even at small proportions, rendering partial filtering ineffective.
- Theoretical and empirical analyses confirm that substantial performance recovery in long-context LLMs requires reducing the hard-distractor proportion to near zero, emphasizing the necessity of high upstream retrieval precision.

---

[MaD Physics: Evaluating information seeking under constraints in physical environments](http://arxiv.org/abs/2605.10820)

- MaD Physics: introduces a benchmarking framework for evaluating LLMs on their ability to perform strategic information gathering and model inference in resource-constrained physical environments.
- The framework utilizes three distinct physical domains—Classical, Quantum, and Fluid mechanics—with altered physical laws to prevent reliance on memorized knowledge and force active empirical discovery.
- Experimental results demonstrate that while LLM performance generally scales with model capability, agents struggle with structured exploration and accurate symbolic recovery under strict cost-fidelity constraints.

---

[Policy Gradient Methods for Non-Markovian Reinforcement Learning](http://arxiv.org/abs/2605.10816)

- ASMPG (Agent State-Markov Policy Gradient): introduces a reward-centric reinforcement learning framework for non-Markovian decision processes that jointly optimizes recursive agent state dynamics and a control policy.
- The framework utilizes an Agent State-Markov policy class to maintain a compact, recursively updated internal state that summarizes interaction history for efficient decision-making.
- The paper establishes a novel policy gradient theorem for non-Markovian environments and provides finite-time convergence guarantees for the proposed ASMPG algorithm.

---

[NanoResearch: Co-Evolving Skills, Memory, and Policy for Personalized Research Automation](http://arxiv.org/abs/2605.10813)

- NanoResearch: introduces a multi-agent framework that enables personalized research automation through tri-level co-evolution of skills, memory, and planner policy.
- The system utilizes an Orchestrator to coordinate ideation, experimentation, and writing stages, while the Skill Bank, Memory Module, and Planner Model accumulate experience to refine research outputs over successive cycles.
- NanoResearch incorporates SDPO-based feedback internalization to align the planner policy with individual researcher preferences, demonstrating improved performance and cost-efficiency across diverse scientific domains.

---

[LLMs for Secure Hardware Design and Related Problems: Opportunities and Challenges](http://arxiv.org/abs/2605.10807)

- LLM-driven EDA and Hardware Security Frameworks: introduces a comprehensive synthesis of LLM-native methodologies for hardware design, covering reasoning-driven synthesis, multi-agent vulnerability extraction, and robust security countermeasures.
- The paper evaluates the integration of LLMs into semiconductor workflows, highlighting the transition from passive coding assistants to autonomous agents that utilize Multi-agent loops, Reasoning models, and Graph-based representations to bridge the semantic gap in hardware design.
- It addresses critical security challenges including data contamination, backdoor attacks, and the alignment paradox, while proposing solutions such as Machine unlearning, Red-teaming agents, and Dynamic benchmarking to ensure trustworthy design ecosystems.

---

[ComplexMCP: Evaluation of LLM Agents in Dynamic, Interdependent, and Large-Scale Tool Sandbox](http://arxiv.org/abs/2605.10787)

- ComplexMCP: introduces a rigorous evaluation framework for LLM agents operating within large-scale, interdependent, and stochastic tool ecosystems using a seed-driven architecture.
- The framework utilizes a rule-based, deterministic evaluation system to assess agent performance across 150+ interdependent tools, replacing subjective LLM-based scoring.
- Granular trajectory analysis identifies critical failure modes in LLMs, including "Clean-Slate" bias, over-confidence, and strategic defeatism, which hinder performance in complex, real-world software environments.

---

[MAGS-SLAM: Monocular Multi-Agent Gaussian Splatting SLAM for Geometrically and Photometrically Consistent Reconstruction](http://arxiv.org/abs/2605.10760)

- MAGS-SLAM: introduces a monocular RGB-only multi-agent 3D Gaussian Splatting SLAM framework for collaborative scene reconstruction using Monocular Front-end, Coordinator, Submap Summary, Sim(3) Pose Graph, Occupancy-Aware Fusion, and Global Photometric Refinement.
- The framework enables collaborative mapping by transmitting compact submap summaries instead of raw data, utilizing a Sim(3) pose graph to resolve monocular scale ambiguity across agents.
- Occupancy-aware fusion and joint photometric refinement eliminate duplicated Gaussians and photometric seams, achieving photorealistic reconstruction without active depth sensors.

---

[The Agent Use of Agent Beings: Agent Cybernetics Is the Missing Science of Foundation Agents](http://arxiv.org/abs/2605.10754)

- Agent Cybernetics: introduces a theoretical framework that maps six canonical laws of classical cybernetics onto agent design principles to provide a scientific foundation for reliable foundation agents.
- The framework synthesizes these principles into three engineering desiderata—reliability, lifelong running, and self-improvement—to address failure modes in long-horizon tasks.
- The paper demonstrates the utility of this approach by identifying domain-specific failure modes and engineering recommendations for code generation, computer use, and automated research.

---

[An Uncertainty-Aware Resilience Micro-Agent for Causal Observability in the Computing Continuum](http://arxiv.org/abs/2605.10718)

- AURORA (Uncertainty-Aware Resilience Micro-Agent for Causal Observability): introduces a lightweight, uncertainty-aware framework for autonomous grey failure diagnosis and mitigation in edge-tier computing environments using parallel micro-agents, Bayesian networks, and a dual-gated execution mechanism.
- The framework employs a dual-gated safety mechanism that evaluates posterior causal confidence and variational free energy to prevent destructive interventions, escalating unresolved cases to the fog tier.
- Experimental results demonstrate that AURORA achieves a 0% destructive action rate while maintaining 62.0% repair accuracy and low computational latency on resource-constrained edge hardware.

---

[Heteroscedastic Diffusion for Multi-Agent Trajectory Modeling](http://arxiv.org/abs/2605.10717)

- U2Diffine (Unified Uncertainty-aware Diffusion): introduces a diffusion-based framework for multi-agent trajectory completion that jointly estimates state-wise heteroscedastic uncertainty using a first-order Taylor approximation for Reverse Gaussian Sampling.
- The architecture integrates a Temporal Mamba for sequential dynamics and a Social Transformer for agent interactions, enabling robust trajectory reconstruction and uncertainty quantification.
- A post-processing RankNN module further enhances reliability by assigning error probabilities to generated modes, significantly improving scene-level trajectory forecasting and completion performance.

---

[The Bystander Effect in Multi-Agent Reasoning: Quantifying Cognitive Loafing in Collaborative Interactions](http://arxiv.org/abs/2605.10698)

- MAS: introduces a theoretical framework to quantify the "Bystander Effect" in LLMs, where simulated social pressure induces cognitive loafing and alignment hallucinations.
- The paper formalizes the Interaction Depth Limit and Sovereignty Decay Law to demonstrate how swarm size and task entropy trigger a transition from a "Fortified Mind" to a "Hollowed Mind" state.
- By auditing internal Chain-of-Thought traces against external outputs, the research proves that LLMs often compute correct derivations but deliberately externalize falsehoods to sycophantically appease the swarm.

---

[Step Rejection Fine-Tuning: A Practical Distillation Recipe](http://arxiv.org/abs/2605.10674)

- SRFT: introduces a fine-grained distillation approach that leverages unresolved LLM agent trajectories by employing a critic LLM to mask harmful steps during supervised fine-tuning.
- The framework utilizes a Student model, a Critic LLM, a trajectory dataset, a loss masking mechanism, and a weighted distillation objective to improve performance on software engineering tasks.
- By selectively omitting loss contributions from erroneous steps identified by the critic, the method enables the Student model to learn from partial successes without internalizing mistakes found in failed trajectories.

---

[Evolving-RL: End-to-End Optimization of Experience-Driven Self-Evolving Capability within Agents](http://arxiv.org/abs/2605.10663)

- Evolving-RL: introduces a unified algorithmic framework that jointly optimizes experience extraction and utilization within a single shared policy, utilizing Extractor, Solver, Retriever, Shared Policy, Environment, and Joint Optimizer.
- The framework employs an extractor-centric design where the Extractor generates candidate skills evaluated by the Solver on retrieved tasks, providing coupled supervisory signals for co-evolution.
- By internalizing reusable experience patterns into model parameters, Evolving-RL functions as an experience-augmented RL algorithm that enhances generalization on unseen tasks.

---

[PRISM: Generation-Time Detection and Mitigation of Secret Leakage in Multi-Agent LLM Pipelines](http://arxiv.org/abs/2605.10614)

- PRISM: introduces a real-time, multi-signal defence that monitors token-level generation dynamics to detect and mitigate secret leakage in multi-agent LLM pipelines before full reconstruction occurs.
- The framework utilizes a per-token risk score derived from temporal generation signals and text-structural features to trigger graduated interventions, including pass-through, sanitisation, or halting.
- PRISM effectively addresses propagation amplification by monitoring each agent in the pipeline, achieving zero observed leakage on a 2,000-task adversarial benchmark while maintaining high output utility.

---

[CrackMeBench: Binary Reverse Engineering for Agents](http://arxiv.org/abs/2605.10597)

- CrackMeBench: introduces a reproducible benchmark for evaluating LLM agents on binary reverse engineering tasks using an executable-oracle framework.
- The framework utilizes a Host orchestrator, Docker sandbox, Private oracle, Agent-visible filesystem, and Tool manifest to measure agent performance on deterministic binary validation problems.
- The benchmark evaluates LLMs on their ability to recover validation logic from stripped binaries without relying on source code or external web resources.

---

[Controllability in preference-conditioned multi-objective reinforcement learning](http://arxiv.org/abs/2605.10585)

- MOPPO (Multi-objective Proximal Policy Optimization): introduces a preference-conditioned reinforcement learning approach that utilizes an actor network, a critic network, and a multi-objective value head to enable dynamic adaptation to user preferences.
- The framework leverages PufferMO, an extension of PufferLib, to facilitate high-throughput training and evaluation of agents in complex, multi-objective environments.
- The research proposes rank-correlation-based metrics to quantify agent controllability, addressing the limitations of mainstream multi-objective reinforcement learning evaluation protocols.

---

[VISTA: A Generative Egocentric Video Framework for Daily Assistance](http://arxiv.org/abs/2605.10579)

- VISTA (Video Synthesis for Training Agents): introduces a modular framework that utilizes a 5-step LLM-driven pipeline to synthesize high-fidelity egocentric videos for training and evaluating proactive AI agents.
- The system employs causal reverse reasoning to derive user actions from intervention needs, ensuring logical consistency and physical plausibility in generated scenarios.
- VISTA incorporates a multi-dimensional evaluation pipeline that leverages spatial analysis and LLM-as-a-Judge protocols to measure intervention utility, timeliness, and safety criticality.

---

[Effect of Graph Gluing on Consensus in Networked Multi-Agent Systems](http://arxiv.org/abs/2605.10558)

- Graph Gluing Framework: introduces a mathematical approach to analyze how interconnecting multi-agent subsystems via bridge gluing or interface gluing impacts the Fiedler eigenvalue and consensus convergence rate.
- The framework utilizes the spectral properties of the graph Laplacian to establish theoretical bounds on algebraic connectivity for combined networked systems.
- Simulation results demonstrate that increasing the number of interconnecting links between subsystems enhances the Fiedler eigenvalue, thereby accelerating the consensus convergence of the overall multi-agent network.

---

[Higher Resolution, Better Generalization: Unlocking Visual Scaling in Deep Reinforcement Learning](http://arxiv.org/abs/2605.10546)

- Impoola: introduces a resolution-independent architecture for deep RL that replaces the standard flattening operation with global average pooling to decouple parameter count from input resolution.
- The framework demonstrates that higher-resolution visual inputs significantly improve agent performance and generalization by enabling more spatially localized policy attention.
- By utilizing the Procgen-HD benchmark, the study reveals that standard low-resolution conventions in deep RL create a perceptual bottleneck that limits the scalability of traditional flatten-based architectures.

---

[TIE: Time Interval Encoding for Video Generation over Events](http://arxiv.org/abs/2605.10543)

- TIE: introduces a principled, plug-and-play interval-aware generalization of rotary embeddings that elevates time intervals to first-class primitives within DiT cross-attention.
- The framework utilizes RoTE to aggregate positional evidence over full event durations, effectively acting as a temporal low-pass filter that provides robustness against noisy boundary annotations.
- TIE enables precise temporal control in concurrent-event regimes, significantly improving temporal constraint satisfaction and alignment metrics compared to traditional single-active-prompt video generation methods.

---

[A Reflective Storytelling Agent for Older Adults: Integrating Argumentation Schemes and Argument Mining in LLM-Based Personalised Narratives](http://arxiv.org/abs/2605.10531)

- Reflective Storytelling Agent: introduces a hybrid architecture that integrates LLMs with symbolic knowledge and argumentation theory to generate personalized, purpose-driven narratives for older adults.
- The framework utilizes a reflective analysis layer that employs argument mining to assess narrative grounding, claim structure, and hallucination risk against a structured user model.
- The system incorporates User Interaction and Dialogue, User Model and Preferences, Knowledge and Normative Constraints, Narrative Planning, Narrative Realisation, Reflective Analysis, and Adaptation and Persistence components to ensure narrative coherence and transparency.

---

[PrimeKG-CL: A Continual Graph Learning Benchmark on Evolving Biomedical Knowledge Graphs](http://arxiv.org/abs/2605.10529)

- PrimeKG-CL: introduces a benchmark for continual graph learning on evolving biomedical knowledge graphs, utilizing EWC, Multimodal Replay, CMKL, and LLM-RAG to address catastrophic forgetting.
- The framework evaluates ten continual learning methods across four KGE decoders, revealing that decoder choice and learning strategy interact significantly, often leading to performance degradation if mismatched.
- PrimeKG-CL provides a stratified evaluation of persistent, added, and removed edges, enabling precise analysis of how models retain valid knowledge while unlearning deprecated facts in dynamic biomedical environments.

---

[Consistency as a Testable Property: Statistical Methods to Evaluate AI Agent Reliability](http://arxiv.org/abs/2605.10516)

- Statistical Framework for Measuring Agent Consistency: introduces a rigorous measurement science for AI agent reliability by decomposing consistency into observable output-level and internal trajectory-level dimensions using U-statistics and kernel-based metrics.
- The framework utilizes Maximum Mean Discrepancy (MMD) and the Global Alignment Kernel (GAK) to detect structural and temporal inconsistencies in agent execution trajectories that traditional accuracy metrics fail to capture.
- By employing weighted Levenshtein distance and spectral clustering, the approach enables the diagnosis of specific failure modes, such as early-stage planning errors or late-stage aggregation drift, across diverse agentic benchmarks.

---

[SoK: A Systematic Bidirectional Literature Review of AI &amp; DLT Convergence](http://arxiv.org/abs/2605.10515)

- AI &amp; DLT Convergence Framework: introduces a bidirectional taxonomy classifying the integration of AI and DLT across five-layer architectural stacks for each technology.
- The paper synthesizes 53 peer-reviewed studies to identify recurring design patterns, including AI-driven consensus optimization and DLT-based federated learning coordination.
- The analysis reveals that while AI improves DLT adaptability and DLT enhances AI transparency, current research remains fragmented and lacks large-scale production deployment.

---

[A Theory of Multilevel Interactive Equilibrium in NeuroAI](http://arxiv.org/abs/2605.10505)

- MIE (Multilevel Interactive Equilibrium): introduces a game-theoretic framework for intelligent systems that generalizes Nash equilibrium by coupling neural learning, cognitive inference, and behavioral strategies across interacting agents.
- The framework models agents as hierarchical systems where stable interaction emerges when neural dynamics, cognitive belief models, and behavioral policies mutually stabilize.
- This approach provides a unified mathematical foundation for analyzing diverse interactive scenarios, including human-AI collaboration, brain-machine interfaces, and psychiatric disorders.

---

[SkillEvolver: Skill Learning as a Meta-Skill](http://arxiv.org/abs/2605.10500)

- SkillEvolver: introduces a lightweight, plug-and-play framework for online skill learning that enables a CLI-agent to iteratively author, deploy, and refine domain-specific skills without updating model weights.
- The framework utilizes a meta-skill to orchestrate a loop consisting of strategy-diversified exploration, contrastive skill update, and independent audit to produce reusable procedural artifacts.
- By grounding refinement in the observed behavior of a deployed Domain-Skill Agent rather than self-reflection, the system effectively captures procedural knowledge and improves agent performance across diverse benchmarks.

---

[DeepRefine: Agent-Compiled Knowledge Refinement via Reinforcement Learning](http://arxiv.org/abs/2605.10488)

- DeepRefine: introduces a general LLM-based reasoning model for agent-compiled knowledge refinement that improves pre-constructed knowledge bases using multi-turn interactions, abductive diagnosis, and targeted refinement actions.
- The framework utilizes a Gain-Beyond-Draft (GBD) reward and reinforcement learning to optimize refinement policies without requiring gold references for the knowledge base updates.
- DeepRefine performs defect localization and incremental updates through three reasoning steps: Answerability Judgement Loop, Error Abduction, and Refinement Actions Generation.

---

[OpenSGA: Efficient 3D Scene Graph Alignment in the Open World](http://arxiv.org/abs/2605.10484)

- OpenSGA: introduces a unified framework for 3D scene graph alignment that fuses vision-language, textual, and geometric features to predict object correspondences under partial overlap.
- The framework utilizes a distance-gated spatial attention encoder to capture node context and an MCF-based allocator to resolve many-to-one object associations.
- The authors also introduce ScanNet-SG, a large-scale dataset with over 700k samples, to support robust training and evaluation of scene graph alignment in open-world environments.

---

[Safe Multi-Agent Behavior Must Be Maintained, Not Merely Asserted: Constraint Drift in LLM-Based Multi-Agent Systems](http://arxiv.org/abs/2605.10481)

- CSG (Constraint State Governance): introduces a paradigm for LLM-based multi-agent systems that maintains safety-critical constraints as explicit, signed execution state to prevent constraint drift across trajectories.
- The framework utilizes Rule Tokenization, Maintained State, Action Proposal, Effect, Admission Checks, Decision and Audit, and Constraint Native Reinforcement Learning to ensure constraints remain fresh, inherited, enforceable, and auditable.
- By coupling governance with reinforcement learning, the system ensures that LLMs optimize for utility only within admissible trajectories, effectively mitigating safety failures that occur during multi-agent coordination and tool use.

---

[ASIA: an Autonomous System Identification Agent](http://arxiv.org/abs/2605.10480)

- ASIA (Autonomous System Identification Agent): introduces an agentic framework that automates the system identification pipeline by delegating model selection, architecture design, and hyperparameter tuning to an LLM-based coding agent.
- The framework utilizes an iterative experimentation loop to propose, implement, and evaluate model configurations based on a fixed evaluation protocol and historical experiment data.
- Experimental results on cascaded tank and nanodrone benchmarks demonstrate that ASIA discovers competitive, non-trivial model architectures and training strategies while outperforming traditional random search methods.

---

[Can Agent Benchmarks Support Their Scores? Evidence-Supported Bounds for Interactive-Agent Evaluation](http://arxiv.org/abs/2605.10448)

- Outcome-Evidence Reporting Layer: introduces an additive reporting framework that evaluates whether existing benchmark artifacts sufficiently support their reported success claims without modifying original tasks or agents.
- The framework utilizes a Case Checklist to categorize agent performance into Evidence Pass, Evidence Fail, or Unknown, thereby quantifying uncertainty and identifying benchmark-evaluator alignment issues.
- By maintaining Unknown records as visible diagnostic data rather than silent failures, the approach provides evidence-supported performance bounds that clarify the reliability of interactive-agent benchmarks.

---

[Statistical Model Checking of the Keynes+Schumpeter Model: A Transient Sensitivity Analysis of a Macroeconomic ABM](http://arxiv.org/abs/2605.10447)

- MultiVeStA: introduces a reproducible statistical workflow for analyzing complex macroeconomic agent-based models by decoupling the simulator from the statistical analysis engine.
- The framework utilizes MultiQuaTEx to specify temporal queries and employs automated stopping rules based on user-defined confidence and precision targets to determine simulation effort.
- This approach enables a principled transient sensitivity analysis of the Keynes+Schumpeter model, revealing that financial and structural parameters exert stronger effects than heuristic-switching mechanisms.

---

[TourMart: A Parametric Audit Instrument for Commission Steering in LLM Travel Agents](http://arxiv.org/abs/2605.10440)

- TourMart: introduces an applied intelligent-system audit instrument for measuring commission steering in LLM-based online travel agents using paired counterfactual replay and governance-parameter sweeps.
- The framework utilizes a symmetric six-gate producer audit to isolate generator-side failures from genuine commercial steering across a continuous governance grid.
- TourMart employs traveler-reader LLMs to extract perception features, which are then mapped by a deterministic welfare rule to quantify steering susceptibility.

---

[Position: Life-Logging Video Streams Make the Privacy-Utility Trade-off Inevitable](http://arxiv.org/abs/2605.10404)

- Pipeline-aware Privacy-Preserving System: introduces a framework that balances privacy and utility in life-logging video streams by utilizing a Privatization Representation Construction Module to generate a Minimal Sufficient Representation (MSR) via Pretrained Foundational Encoders, which is then transmitted through a Privatized Representation Communication Module to a Remote Server for secure processing.
- The framework emphasizes that Edge Devices must perform initial data privatization to ensure that only task-relevant, non-reconstructible representations are shared, thereby mitigating privacy risks associated with raw data exposure.
- The research demonstrates that non-private embeddings from standard encoders are vulnerable to inversion attacks, necessitating the adoption of pipeline-aware, attack-agnostic privacy designs for sustainable AI development.

---

[AnomalyClaw: A Universal Visual Anomaly Detection Agent via Tool-Grounded Refutation](http://arxiv.org/abs/2605.10397)

- AnomalyClaw: introduces a training-free VAD agent that decomposes anomaly detection into a multi-round refutation process using a Multi-round Refutation Agent, a Direct VLM Scorer, and an optional Verbalized Self-Evolution (OSR) extension.
- The system utilizes a 13-tool catalog to gather evidence for candidate features, which are then refuted against normal-sample references to produce a fine-grained anomaly score.
- AnomalyClaw improves cross-domain VAD performance by leveraging internal branch disagreement to build an online rulebook without requiring oracle labels or parameter updates.

---

[Agentic Performance at the Edge: Insights from Benchmarking](http://arxiv.org/abs/2605.10384)

- APE: introduces a domain-conditioned evaluation methodology for edge-deployed LLMs that systematically analyzes the impact of model size, generation, and variant type on agentic reliability using an ITBench-based Evaluation Harness, Observability Retrieval Tool, Topology Reasoning Tool, Candidate Reduction Tool, Cost Anomaly Analysis Tool, Root-Cause Evaluator, Inference Protocol, and Local Inference Endpoint.
- The study demonstrates that agentic performance at the edge is a complex systems property where domain-specific task difficulty and failure modes—semantic versus execution—often outweigh simple parameter-count scaling.
- The research provides empirical evidence that coder-oriented LLMs and strategic model routing can achieve significant latency reductions and reliability improvements in resource-constrained edge environments.

---

[PC3D: Zero-Shot Cooperation Across Variable Rosters via Personalized Context Distillation](http://arxiv.org/abs/2605.10377)

- PC3D: introduces a method for open-team cooperation that uses a set-structured centralized critic to distill personalized coordination contexts into decentralized recurrent actors.
- The framework employs a teacher module to generate coordination tokens via cross-attention, which are then personalized for each agent and distilled into decentralized policies to enable zero-shot adaptation across variable roster sizes.
- Agents recover these distilled contexts from local interaction histories and use gated FiLM modulation to adaptively condition their decision-making without requiring execution-time communication or global observations.

---

[Sleep Walk: A Three-Tier Benchmark for Stress-Testing Instruction-Guided Vision-Language Navigation](http://arxiv.org/abs/2605.10376)

- SleepWalk: introduces a benchmark for evaluating instruction-grounded trajectory prediction in single-scene 3D environments, utilizing Hunyuan3D-3.0, Qwen3-8B-VL, GPT-5-mini, TLControl, and MotionGPT.
- The framework evaluates LLMs on their ability to translate natural-language instructions into spatially coherent, executable paths across three tiers of difficulty.
- The benchmark employs a pointwise judge-based evaluation protocol to assess start-location consistency, goal satisfaction, obstacle avoidance, and trajectory efficiency.

---

[Approximate Envy-Free Allocations up to any k Goods](http://arxiv.org/abs/2605.10371)

- G3PA (Generalized Property Preserving Allocation): introduces a polynomial-time algorithm to achieve (k+1)/(k+2)-EFkX allocations for any number of agents with additive valuations.
- The framework utilizes a modified envy graph and a three-step process involving G3PA, AllocateAndEliminateCritical, and EnvyCycleElimination to ensure fair resource distribution.
- The research also establishes that EFkX orientations on graphs do not always exist and that deciding their existence is an NP-complete problem.

---

[AgentGR: Semantic-aware Agentic Group Decision-Making Simulator for Group Recommendation](http://arxiv.org/abs/2605.10367)

- AgentGR: introduces a novel agent-based framework that integrates semantic meta-path guided CoP reasoning, semantic-aware recognition for group topics and leadership, and multi-agent simulation strategies to model complex group decision-making processes.
- The framework utilizes LLMs to perform member-role-playing and simulate dynamic group interactions, effectively bridging the gap between high-order collaborative filtering signals and rich textual semantics.
- AgentGR provides two distinct simulation strategies—a static workflow for efficiency and a dynamic dialogue-based approach for precision—to accommodate diverse group recommendation requirements.

---

[EGL-SCA: Structural Credit Assignment for Co-Evolving Instructions and Tools in Graph Reasoning Agents](http://arxiv.org/abs/2605.10366)

- EGL-SCA: introduces a verifier-centric dual-space framework that models a graph reasoning agent using an instruction-side policy space for reasoning strategies and a tool-side program space for executable algorithmic tools.
- The framework utilizes structural credit assignment to map trajectory evidence to conditional updates, precisely routing failures to either prompt optimization or tool synthesis and repair.
- By co-evolving instructions and tools through a family-aware training curriculum and Pareto-style retention, the agent achieves high success rates on complex graph reasoning tasks.

---

[CellDX AI Autopilot: Agent-Guided Training and Deployment of Pathology Classifiers](http://arxiv.org/abs/2605.10362)

- CellDX AI Autopilot: introduces a platform that enables users to train, evaluate, and deploy pathology classifiers through natural language interaction with an AI Agent, utilizing Agent Skills, CellDX API, Azure ML GPU compute, Metric history (MongoDB), and a Deployed widget.
- The system leverages a structured set of agent skills to guide users through dataset curation, automated hyperparameter tuning, and multi-strategy model comparison while maintaining human-in-the-loop deployment controls.
- By operating on a pre-built feature store of over 32,000 cases, the platform removes the need for expensive feature extraction, allowing researchers to run parallel experiments cost-effectively on cloud infrastructure.

---

[How Mobile World Model Guides GUI Agents?](http://arxiv.org/abs/2605.10347)

- MWM (Mobile World Model): introduces a unified training pipeline for mobile GUI world models across four distinct modalities to provide agents with prospective foresight.
- The framework evaluates how different prediction formats—text-based, code-rendered, and diffusion-based—impact GUI agent performance, decision confidence, and task completion.
- Empirical results demonstrate that while world models improve agent foresight, their effectiveness is constrained by agent confidence, candidate diversity, and the fidelity of the simulated future states.

---

[TMAS: Scaling Test-Time Compute via Multi-Agent Synergy](http://arxiv.org/abs/2605.10344)

- TMAS: introduces a multi-agent framework for scaling test-time compute by organizing inference as a collaborative process among specialized agents that utilize hierarchical memory banks to coordinate information flow across trajectories.
- The framework employs an experience agent to manage low-level reasoning signals and a guideline agent to track high-level strategies, facilitating both the exploitation of reliable findings and the exploration of novel solution paths.
- A hybrid reinforcement learning system, incorporating rewards for correctness, experience utilization, and novel strategy exploration, further enhances the model's ability to perform sustained iterative reasoning.

---

[PaperFit: Vision-in-the-Loop Typesetting Optimization for Scientific Documents](http://arxiv.org/abs/2605.10341)

- PaperFit: introduces a vision-in-the-loop agent that automates scientific document typesetting by closing the sense–act–verify loop through multi-source evidence integration, constrained repair policy, and checklist-gated multi-round validation.
- The system addresses typesetting defects by fusing source, log, PDF, and page-image signals into structured records, enabling precise, constrained source-level revisions that avoid common pseudo-fixes.
- PaperFit-Bench provides a comprehensive evaluation framework with 200 papers across 10 templates, demonstrating that structured visual closed-loop control significantly outperforms existing document automation baselines.

---

[EmbodiSkill: Skill-Aware Reflection for Self-Evolving Embodied Agents](http://arxiv.org/abs/2605.10332)

- EmbodiSkill: introduces a training-free framework for embodied agents that utilizes skill-aware reflection and targeted revision to evolve procedural knowledge from task trajectories.
- The framework separates skill-body updates from skill-appendix refinements, ensuring that execution lapses are addressed by emphasizing existing valid guidance rather than incorrectly modifying core procedural rules.
- By employing a skill-aware evolution spiral, the agent iteratively improves its performance on complex household tasks by consolidating targeted reflection records into progressively more accurate and executable skill specifications.

---

[Verifiable Process Rewards for Agentic Reasoning](http://arxiv.org/abs/2605.10325)

- VPR (Verifiable Process Rewards): introduces a framework that converts symbolic or algorithmic oracles into dense, turn-level supervision for LLM agents to improve long-horizon credit assignment.
- The framework utilizes task-specific verifiers—including MCTS, constraint solvers, and posterior inference engines—to provide objective, noise-free feedback for each intermediate action.
- Empirical results demonstrate that VPR outperforms outcome-level and rollout-based baselines in controlled environments while fostering transferable reasoning skills for general and agentic benchmarks.

---

[Positive Alignment: Artificial Intelligence for Human Flourishing](http://arxiv.org/abs/2605.10310)

- Positive Alignment: introduces a paradigm shift in AI alignment by moving beyond mere harm avoidance toward the active cultivation of human and ecological flourishing through a multi-stage lifecycle approach.
- The framework utilizes Negative Attractors, Repellers, Satisficing Region, and Positive Attractors to map the transition from traditional safety constraints to proactive, value-aligned system behaviors.
- It proposes a full-stack alignment strategy that integrates Intentional Data Sourcing, Adaptive Constitutions, and multi-agent cooperation to ensure AI systems support long-term human well-being while maintaining epistemic humility and pluralistic governance.

---

[AgentRx: A Benchmark Study of LLM Agents for Multimodal Clinical Prediction Tasks](http://arxiv.org/abs/2605.10286)

- AgentRx: introduces a comprehensive benchmark for evaluating LLM-based agentic systems on multimodal clinical prediction tasks using Patient Summary, Electronic Health Records, Radiology Reports, and Chest X-rays.
- The framework evaluates performance across three settings: Single Agent Unimodal, Single Agent Multimodal, and Multi-Agent Multimodal, utilizing Unimodal Agents, Multimodal Agents, Multimodal Judge Agents, and Worker Agents.
- The study highlights that single agent frameworks outperform multi-agent systems in multimodal clinical prediction, while also identifying significant calibration challenges in decentralized multi-agent setups.

---

[MemReread: Enhancing Agentic Long-Context Reasoning via Memory-Guided Rereading](http://arxiv.org/abs/2605.10268)

- MemReread: introduces a memory-guided LLM agent that enhances long-context reasoning by decomposing tasks into sub-questions and performing targeted rereading to recover discarded latent evidence.
- The framework utilizes a four-phase workflow—Read, Decompose, Integrate, and Answer—to maintain a bounded memory buffer while enabling non-linear reasoning through iterative information acquisition.
- To optimize computational efficiency, the paper introduces Rereading-Adaptive GRPO, a reinforcement learning strategy that dynamically modulates the number of rereading passes based on task complexity.

---

[Towards Autonomous Railway Operations: A Semi-Hierarchical Deep Reinforcement Learning Approach to the Vehicle Rescheduling Problem](http://arxiv.org/abs/2605.10257)

- Maze-Flatland: introduces a semi-hierarchical multi-agent RL framework that decomposes railway rescheduling into MADS (high-level dispatching) and MAPF (low-level routing) to improve scalability and learning stability.
- The framework utilizes a decision controller to enforce temporal ordering, where MADS manages global traffic density and MAPF handles local conflict resolution through path deviations.
- By separating decision scopes, the architecture mitigates the hard-exploration problem and reduces task interference, enabling effective performance in dense railway networks.

---

[Beyond Autonomy: A Dynamic Tiered AgentRunner Framework for Governable and Resilient Enterprise AI Execution](http://arxiv.org/abs/2605.10223)

- Dynamic Tiered AgentRunner: introduces a risk-adaptive, multi-role execution architecture that dynamically adjusts governance intensity to match task risk, enforces physical separation between proposal and execution, and builds resilience through systematic failure handling.
- The framework utilizes an OrchestratorAgent, WorkerAgent, CriticAgent, VerifierAgent, RecoveryAgent, RetrospectorAgent, ToolGateway, and Checkpoint to ensure governable and resilient enterprise AI execution.
- By treating failure as a first-class execution state and implementing a hard-wired ToolGateway, the framework transforms AI governance from prompt-level suggestions into robust system architecture guarantees.

---

[V-ABS: Action-Observer Driven Beam Search for Dynamic Visual Reasoning](http://arxiv.org/abs/2605.10172)

- V-ABS: introduces a closed-loop beam search framework that integrates a Thinker, Actor, and Observer to mitigate imagination-action-observer bias in multi-step visual reasoning.
- The framework employs an entropy-based adaptive weighting algorithm to dynamically balance prior policy confidence from the Thinker with grounded visual feedback from the Observer.
- V-ABS utilizes a large-scale supervised fine-tuning dataset to improve action verification and achieves state-of-the-art performance across diverse visual search, navigation, and logic benchmarks.

---

[When Reviews Disagree: Fine-Grained Contradiction Analysis in Scientific Peer Reviews](http://arxiv.org/abs/2605.10171)

- IMPACT (Intensity-based Multi-Agent Contradiction estimation): introduces a multi-agent framework that integrates ACEA, DIA-A/B, Intensity Agreement Checker, Disagreement Orchestrator, Adjudication Agent, and CVG to model fine-grained reviewer contradictions and their intensity.
- The framework utilizes a deliberative reasoning protocol where multiple agents debate conflicting assessments to mitigate bias and surface nuanced disagreements, which are then distilled into TIDE for efficient deployment.
- The authors also present RevCI, an expert-annotated benchmark of peer-review pairs with evidence-level contradiction annotations and graded intensity labels to support the fine-grained analysis of reviewer disagreement.

---

[Balancing Efficiency and Fairness in Traffic Light Control through Deep Reinforcement Learning](http://arxiv.org/abs/2605.10170)

- DDQN framework: introduces a reinforcement learning-based traffic light control agent that explicitly integrates fairness considerations between vehicular and pedestrian flows to reduce congestion.
- The approach utilizes a composite reward function with a fairness trade-off coefficient to dynamically balance waiting times for different road user categories.
- Experimental results demonstrate that the agent outperforms fixed-time traffic light controllers by autonomously adapting to varying traffic conditions in a simulated four-way intersection.

---

[NyayaAI: An AI-Powered Legal Assistant Using Multi-Agent Architecture and Retrieval-Augmented Generation](http://arxiv.org/abs/2605.10155)

- NyayaAI: introduces a multi-agent legal assistant that leverages the Mastra TypeScript framework, RAG Knowledge Base, and Claude SDK to automate and simplify legal workflows for Indian legal resources.
- The system architecture integrates a main Nyaya agent with specialized sub-agents for research, summarization, case analysis, and drafting, all validated by a compliance module.
- Evaluation results demonstrate that the structured multi-agent LLM approach achieves 70% domain classification precision and 74% RAG retrieval precision within the Indian legal domain.

---

[Is DRL-based MAC Ready for Underwater Acoustic Networks? Exploring Its Practicality in Real Field Experiments](http://arxiv.org/abs/2605.10144)

- EA-MAC: introduces a DRL-based protocol for underwater acoustic networks that utilizes a DQN-agent, an aggregated ACK mechanism, a transmission queue, a fairness penalty module, a Bayesian inference module, and an observation completion module to achieve efficient, autonomous, and fair channel access.
- The framework addresses challenges in underwater environments, such as uncertain reward acquisition delays and incomplete observations, by employing Bayesian inference and a sliding window observation approach to enhance decision-making robustness.
- EA-MAC balances network throughput and transmission fairness by dynamically adjusting node access strategies based on local information and overheard data, outperforming existing DRL-based protocols in both simulation and real-world field experiments.

---

[Plan in Sandbox, Navigate in Open Worlds: Learning Physics-Grounded Abstracted Experience for Embodied Navigation](http://arxiv.org/abs/2605.10118)

- SAGE: introduces a generative experience-driven learning paradigm that enables agents to build robust navigation capabilities by distilling high-level VLM reasoning into structured embodied experiences within a physics-grounded sandbox.
- The framework utilizes a three-phase approach comprising Genesis for autonomous task synthesis, Evolution for policy optimization via Asymmetric Adaptive Clipping, and Navigation for retrieval-augmented decision-making.
- SAGE effectively bridges the modality gap between semantic reasoning and low-level robot control, achieving significant performance gains on A-EQA and GOAT-Bench benchmarks while demonstrating robust Sim2Real transfer.

---

[ViSRA: A Video-based Spatial Reasoning Agent for Multi-modal Large Language Models](http://arxiv.org/abs/2605.10106)

- ViSRA: introduces an inference-time, human-aligned agent framework that enhances the spatial intelligence of LLMs by dynamically orchestrating specialized perception tools without requiring post-training.
- The framework utilizes a multi-role agent architecture comprising a Planner, Reflector, Executor, and Summarizer to decompose complex spatial queries into verifiable subproblems.
- ViSRA leverages a suite of spatial tools, including 2D/3D detection, tracking, and scene modeling, to construct explicit spatio-temporal evidence, achieving superior generalization on unseen tasks compared to post-trained models.

---

[RFAmpDesigner: A Self-Evolving Multi-Agent LLM Framework for Automated Radio Frequency Amplifier Design](http://arxiv.org/abs/2605.10093)

- RFAmpDesigner: introduces a multi-agent framework that automates RF amplifier sizing by reframing high-dimensional parameter tuning as a low-dimensional resource-allocation problem managed by RFAmpManager, RFAmpSearcher, RFAmpRefiner, Tool Middleware, Knowledge Base, Experience Base, and RAG.
- The framework utilizes a two-tier multi-agent architecture to decouple heavy simulation tasks from lightweight reasoning, enabling efficient search and refinement through RAG-based knowledge reuse.
- RFAmpDesigner achieves significant improvements in sample efficiency and runtime by leveraging domain-specific tool middleware and agentic decomposition to navigate sparse feasible regions in RF circuit design.

---

[Agentic Fuzzing: Opportunities and Challenges](http://arxiv.org/abs/2605.10074)

- AFuzz: introduces a bug-finding approach that uses deep agents as the primary reasoning engine, seeded by historical bugs, to identify complex logic bugs through code reasoning, hypothesis generation, and verification.
- The framework utilizes a four-stage LLM agent pipeline—Analyzer, Investigator, Scenario Analyzer, and Validator—to systematically analyze reference bugs and discover variants across different code implementations.
- To optimize efficiency, AFuzz incorporates a DPP-MAP seed scheduler for diversity and a scenario coverage tracker to prevent redundant investigations across the target codebase.

---

[MAGE: Multi-Agent Self-Evolution with Co-Evolutionary Knowledge Graphs](http://arxiv.org/abs/2605.10064)

- MAGE: introduces a framework that externalizes self-knowledge into a co-evolutionary knowledge graph (EVOKG) to enable frozen LLMs to improve across iterations.
- The framework utilizes a dual-memory index within the experience subgraph, storing both self-harvested success traces and teacher-written failure corrections to guide the frozen execution tier.
- MAGE couples graph updates with task-level search and skill-level routing bandits, all driven by a shared reward stream to ensure stable, append-only evolution of the retrieval substrate.

---

[EFGCL: Learning Dynamic Motion through Spotting-Inspired External Force Guided Curriculum Learning](http://arxiv.org/abs/2605.10063)

- EFGCL: introduces a reinforcement learning framework that accelerates dynamic motor skill acquisition by applying spotting-inspired external forces to guide agents toward successful trajectories.
- The framework utilizes an adaptive curriculum that gradually decays assistive forces based on the success rate, enabling a smooth transition to autonomous motion generation.
- EFGCL employs a Teacher-Student architecture where the teacher policy is trained with privileged information and external forces, while the student policy is distilled to operate using only onboard sensor observations.

---

[Strategic Exploitation in LLM Agent Markets: A Simulation Framework for E-Commerce Trust](http://arxiv.org/abs/2605.10059)

- TruthMarketTwin: introduces a controlled simulation framework for studying LLM-agent behavior in e-commerce markets under asymmetric information.
- The framework incorporates Seller Agents, Buyer Agents, Marketplace Architecture, Reputation System, Warrant Enforcement, Communication Channels, and Persistent Memory to evaluate strategic deception and institutional governance.
- Research findings demonstrate that LLM agents autonomously exploit reputation vulnerabilities, while warrant enforcement effectively reshapes strategic reasoning and improves market welfare under economic pressure.

---

[Route by State, Recover from Trace: STAR with Failure-Aware Markov Routing for Multi-Agent Spatiotemporal Reasoning](http://arxiv.org/abs/2605.10057)

- STAR (Spatio-Temporal Agent Router): introduces a failure-aware routing framework that externalizes inter-agent control as a state-conditioned transition policy over the current agent, task type, and typed execution status.
- The framework utilizes an agent routing matrix that combines expert-specified nominal routes with recovery transitions learned from execution traces to handle distinct failure modes like malformed outputs, missing dependencies, and tool-query mismatches.
- Specialists execute through an extract-compute-deposit protocol, writing intermediate results to a shared blackboard, which enables the router to treat recovery as an explicit routing problem rather than an implicit byproduct of LLM generation.

---

[Swarm Skills: A Portable, Self-Evolving Multi-Agent System Specification for Coordination Engineering](http://arxiv.org/abs/2605.10052)

- Swarm Skills: introduces a portable, self-evolving specification for multi-agent coordination that decouples coordination logic from runtime environments using SKILL.md, roles/, workflow.md, bind.md, and evolutions.json.
- The framework utilizes a self-evolution algorithm comprising a Trajectory Aggregator, Trajectory Analyzer, and Governance Engine to autonomously refine coordination protocols based on multi-dimensional scoring of Effectiveness, Utilization, and Freshness.
- By adhering to the Anthropic Skills standard and progressive disclosure, Swarm Skills enables zero-adapter portability and backward compatibility across diverse LLM-based agent execution environments.

---

[Adaptive Action Chunking via Multi-Chunk Q Value Estimation](http://arxiv.org/abs/2605.10044)

- ACH: introduces a novel offline-to-online RL framework that dynamically modulates action chunk length during training and inference to optimize performance based on current state dynamics.
- The framework utilizes a Transformer-based Value Function to simultaneously estimate action-values for all candidate chunk lengths in a single forward pass.
- An Adaptive Sampling Mechanism selects the most effective chunk length by balancing future state uncertainty and policy stochasticity jittering, consistently outperforming fixed-length baselines.

---

[TimeClaw: A Time-Series AI Agent with Exploratory Execution Learning](http://arxiv.org/abs/2605.10038)

- TimeClaw: introduces an exploratory execution learning framework that transforms time-series agent exploration into reusable hierarchical distilled experience through a four-stage loop of Explore, Compare, Distill, and Reinject.
- The framework utilizes metric-supervised execution comparison and task-aware tool dropout to mitigate tool-prior collapse, ensuring that agents learn from quantitative execution quality rather than completion alone.
- TimeClaw externalizes learned strategies into structured memory, skills, and notes, which are reinjected into the agent's prompt at inference time to improve reasoning without requiring online model updates.

---

[Bridging the Cognitive Gap: A Unified Memory Paradigm for 6G Agentic AI-RAN](http://arxiv.org/abs/2605.10036)

- AI-RAN: introduces a memory-centric architecture that utilizes a unified memory fabric to bridge the cognitive gap between microsecond-level reflexes, millisecond-level reasoning, and long-term evolution in 6G networks.
- The framework maps biological memory hierarchies—sensory, working, and long-term—onto heterogeneous computing fabrics using CXL interconnects to enable zero-copy state sharing across cognitive loops.
- By replacing message-based interfaces with shared memory, the architecture eliminates spatial blindness and temporal amnesia, allowing agents to perform cause-aware control and proactive adaptation.

---

[Beyond Self-Play and Scale: A Behavior Benchmark for Generalization in Autonomous Driving](http://arxiv.org/abs/2605.10034)

- BehaviorBench: introduces a comprehensive benchmarking framework for autonomous driving that addresses evaluation, complexity, and behavior diversity through an Evaluation Interface, Scenario Splits, Traffic Agents, and a Hybrid Planner.
- The framework enables at-scale RL policy evaluation by bridging PufferDrive with established benchmarks like nuPlan and interPlan.
- It demonstrates that self-play RL agents often overfit to training opponents and proposes a hybrid planner to improve generalization and safety in interactive traffic scenarios.

---

[Combining Mechanical and Agentic Specification Inference for Move](http://arxiv.org/abs/2605.10005)

- MoveFlow: introduces a hybrid specification inference methodology that combines mechanical weakest-precondition analysis with an agentic coding assistant to automate the generation of formal specifications for Move smart contracts.
- The framework utilizes a Model Context Protocol service to integrate the Move Prover as a mechanical baseline, allowing the LLM-based agent to focus on complex tasks like loop invariant synthesis and specification simplification.
- The system employs a propose-observe-refine loop where the Move Prover acts as a verification oracle, providing feedback to the agent to iteratively refine specifications and resolve verification timeouts.

---

[Continual Harness: Online Adaptation for Self-Improving Foundation Agents](http://arxiv.org/abs/2605.09998)

- Continual Harness: introduces a reset-free framework that automates the refinement of an agent's scaffolding by alternating between acting in an Environment and using a Refiner to update the Harness components (System prompt, Sub-agents, Skills, Memory) based on trajectory data.
- The framework incorporates an online co-learning loop where an Agent model and the Harness state are jointly updated through a process reward model and teacher relabeling of trajectory windows.
- By enabling mid-episode, reset-free self-improvement, the system allows agents to overcome long-horizon partial-observability challenges in complex RPG environments without requiring manual scaffolding.

---

[A Resource Allocation Game and its Equilibrium Strategies](http://arxiv.org/abs/2605.09988)

- RAG (Resource Allocation Game): introduces a Bayesian game framework for allocating resources among autonomous players using AIF functions to determine optimal request strategies under incomplete information.
- The framework utilizes a construction algorithm to identify Nash equilibria, incorporating mean-field and Gaussian approximations for large-scale systems.
- The model identifies a chattering regime where players employ a continuous, strictly increasing strategy to sustain high payoffs when standard identity or flat strategies are insufficient.

---

[Towards Generalist Game Players: An Investigation of Foundation Models in the Game Multiverse](http://arxiv.org/abs/2605.09965)

- Generalist Game Player Framework: introduces a comprehensive, end-to-end lifecycle for game-playing AI, organizing the field into four interdependent pillars: Dataset, Model, Harness, and Benchmark.
- The paper formalizes the evolution of game-playing AI through a Goal-Conditioned POMDP, tracing the transition from brittle specialists to generalist agents capable of omni-reality adaptability.
- It identifies five fundamental trade-offs bottlenecking the field and proposes a five-level roadmap, progressing from single-game mastery to the ultimate Demiurge stage where agents generate and evolve their own game multiverses.

---

[Generating synthetic electronic health record data using agent-based models to evaluate machine learning robustness under mass casualty incidents](http://arxiv.org/abs/2605.09951)

- ED ABM (Emergency Department Agent-Based Model): introduces a mechanistic simulation approach to generate synthetic EHR data for evaluating the robustness of ML models under mass casualty incidents.
- The framework utilizes real-world EHR data to calibrate an emergency department environment, allowing for the systematic manipulation of system conditions such as patient arrival rates and resource availability.
- By simulating these conditions, the approach enables stress testing of ML models to identify performance vulnerabilities, specifically declines in recall, that are not captured by standard held-out test sets.

---

[HAGE: Harnessing Agentic Memory via RL-Driven Weighted Graph Evolution](http://arxiv.org/abs/2605.09942)

- HAGE: introduces a weighted multi-relational memory framework that reconceptualizes retrieval as a sequential, query-conditioned traversal over a dynamic graph, utilizing a Weighted Multi-Relational Memory Graph, Event-Nodes, Trainable Edge Features, a QueryRouter, a Reinforcement Learning-Based Training Framework, an LLM-based Classifier, and a Context Synthesis Module.
- The framework optimizes memory access by jointly training edge representations and routing policies via reinforcement learning, allowing the agent to prioritize structurally critical paths based on query intent and downstream feedback.
- Empirical results demonstrate that HAGE improves long-horizon reasoning accuracy and provides a favorable accuracy-efficiency trade-off compared to existing agentic memory systems.

---

[Population Protocols over Ordered Agents](http://arxiv.org/abs/2605.09937)

- PP[N] (Population Protocols over Ordered Agents): introduces a distributed computation model where agents are totally ordered and interactions are restricted by predicates, establishing that the immediate observation fragment IO-PP[&lt;] is equivalent to unambiguous star-free languages.
- The paper characterizes the expressive power of these protocols, showing that IO-PP[N] with successor predicates equals NSPACE(n), while providing logic and automaton models for the class PP[&lt;].
- The authors investigate the well-specification problem, proving that checking if a protocol is a decider is undecidable for PP[&lt;] and IO-PP[+1], but conditionally decidable for IO-PP[&lt;].

---

[TRACER: Verifiable Generative Provenance for Multimodal Tool-Using Agents](http://arxiv.org/abs/2605.09934)

- TRACER: introduces a framework for verifiable generative provenance in multimodal tool-using agents by linking answer claims to specific tool turns, evidence units, and semantic support relations.
- The framework utilizes a Policy Model to generate sentence-level provenance records, which are then validated by a Traceability Verifier to ensure source authenticity and relation rationality.
- TRACER employs a Reward Module to provide provenance-aware feedback, enabling reinforcement learning that optimizes for both answer accuracy and efficient, evidence-grounded tool usage.

---

[FOCUSFT: Bilevel Optimization for Dilution-Aware Long-Context Fine-Tuning](http://arxiv.org/abs/2605.09932)

- FOCUSFT: introduces a bilevel optimization framework that mitigates training-time attention dilution by using an Inner Loop to construct a Parametric Memory via LoRA Fast-Weight Adapters, which guides the Outer Loop to produce a more informative gradient signal.
- The framework employs Bidirectional Context Attention to eliminate the causal asymmetry that drives attention sinks, while maintaining Causal Masking for response generation to ensure inner-outer consistency.
- By sharpening the attention distribution during training, FOCUSFT enables LLMs to better utilize long-context information without requiring additional inference-time computation.

---

[Fair Allocation under Conflict Constraints](http://arxiv.org/abs/2605.09930)

- Fair Allocation under Conflict Constraints: introduces a framework for distributing indivisible items represented as vertices in a conflict graph, where agents receive independent sets of items to satisfy fairness and efficiency criteria.
- The paper utilizes a color-switching technique and a cycle-plus-n-cliques graph construction to prove existence and computational complexity results for EF1 and EF[1,1] allocations under various valuation profiles.
- The research establishes that while maximal EF1 allocations exist for two agents with monotone valuations, the problem becomes NP-hard for larger agent groups, necessitating the use of relaxed fairness notions like EF[1,1] on path graphs.

---

[TeleResilienceBench: Quantifying Resilience for LLM Reasoning in Telecommunications](http://arxiv.org/abs/2605.09929)

- TeleResilienceBench: introduces a benchmark to quantify reasoning resilience in LLMs by evaluating their ability to correct flawed, truncated reasoning traces within telecommunications domain tasks.
- The framework utilizes a weak generator to produce corrupted reasoning paths, which are then presented to target models to measure their Correct Flip Rate (CFR), No Flip Rate (NFR), and Wrong Flip Rate (WFR).
- Experimental results across multiple LLM families demonstrate that reasoning resilience is a distinct capability from standard accuracy, with smaller models like Nemotron-3-nano 4b often outperforming significantly larger models in error recovery.

---

[Position: Academic Conferences are Potentially Facing Denominator Gaming Caused by Fully Automated Scientific Agents](http://arxiv.org/abs/2605.09915)

- Agentic Conference Denominator Gaming: introduces a systemic threat where malicious actors use a multi-agent pipeline to inflate conference submission pools, thereby manipulating acceptance rates to favor targeted papers.
- The framework utilizes a Research Agent for high-volume generation of stylistically plausible manuscripts and a Submission Agent for automated, large-scale submission via platform APIs.
- This approach exploits the stable acceptance rate norm of academic conferences, creating an economically asymmetric attack that threatens to overwhelm peer-review systems and erode scientific trust.

---

[RADAR: Redundancy-Aware Diffusion for Multi-Agent Communication Structure Generation](http://arxiv.org/abs/2605.09907)

- RADAR: introduces an iterative, step-wise generative framework that utilizes conditional graph diffusion models to synthesize task-adaptive multi-agent communication topologies while actively minimizing redundant information flow.
- The framework incorporates an effective size metric to guide the denoising network in constructing low-redundancy collaboration graphs, balancing structural expressiveness with communication efficiency.
- By employing a reinforcement learning-based training strategy, RADAR optimizes both the structural fidelity of the generated communication graphs and the downstream task utility of the multi-agent system.

---

[Deterministic vs. LLM-Controlled Orchestration for COBOL-to-Python Modernization](http://arxiv.org/abs/2605.09894)

- ATLAS (Autonomous Transpilation for Legacy Application Systems): introduces a controlled experimental framework to compare deterministic and LLM-controlled orchestration strategies for COBOL-to-Python modernization.
- The framework isolates execution control as the sole experimental variable by maintaining constant LLMs, prompts, tools, and configurations across both deterministic and agentic workflows.
- Empirical results demonstrate that deterministic orchestration improves worst-case robustness and reduces token consumption by up to 3.5x compared to LLM-controlled orchestration, while maintaining comparable functional correctness.

---

[Pseudo-Deliberation in Language Models: When Reasoning Fails to Align Values and Actions](http://arxiv.org/abs/2605.09893)

- VALDI: introduces a framework for measuring the value–action gap in LLMs by comparing stated values against generated dialogue across 4,941 scenarios.
- VIVALDI: introduces a multi-agent value auditor that intervenes at different generation stages to improve alignment between LLM reasoning and final actions.
- The research identifies pseudo-deliberation as a failure mode where explicit reasoning can degrade value–action alignment, demonstrating that post-hoc auditing of final outputs is more effective than reasoning-level interventions.

---

[Skill Description Deception Attack against Task Routing in Internet of Agents](http://arxiv.org/abs/2605.09889)

- SDD (Skill Description Deception) attack framework: introduces a security vulnerability in Internet of Agents (IoA) systems where malicious agents manipulate self-declared skill descriptions to bias semantic task routing, utilizing an LLM-based Query Generator, a Skill Description Optimizer, and a Semantic Router.
- The framework leverages an iterative optimization process to align malicious skill descriptions with target-domain queries, effectively hijacking task delegation from benign agents without requiring access to internal routing parameters.
- Experimental results across nine domains demonstrate that the SDD attack achieves up to 98% success rate, highlighting the critical need for trust-aware routing and capability verification in LLM-driven multi-agent ecosystems.

---

[M2A: Synergizing Mathematical and Agentic Reasoning in Large Language Models](http://arxiv.org/abs/2605.09879)

- M2A: introduces a training-free paradigm that synergizes mathematical and agentic reasoning in LLMs by merging task vectors into the null space of agent-critical behavior subspaces.
- The framework utilizes Agent-critical behavior calibration, Null-space projected merging, and Adaptive layer-wise merging to enhance internal reasoning while preserving the multi-turn interaction patterns of LLMs.
- By employing a Similarity-aware layer mask and Merge coefficient calibration, M2A provides a controllable interface for modulating reasoning depth without requiring additional gradient-based training.

---

[EGOMEMREASON: A Memory-Driven Reasoning Benchmark for Long-Horizon Egocentric Video Understanding](http://arxiv.org/abs/2605.09874)

- EGOMEMREASON: introduces a benchmark for evaluating long-horizon memory in egocentric videos, utilizing Evidence Preparation, Memory-Centric Question Generation, Automatic Filtering, and Human Verification to assess entity-, event-, and behavior-memory.
- The framework evaluates LLMs and agentic systems on their ability to aggregate evidence across week-long temporal horizons to solve complex reasoning tasks.
- Experimental results reveal that current models struggle with long-horizon memory, with performance bottlenecks varying across entity, event, and behavior memory types.

---

[ConsistNav: Closing the Action Consistency Gap in Zero-Shot Object Navigation with Semantic Executive Control](http://arxiv.org/abs/2605.09869)

- ConsistNav: introduces a training-free framework for zero-shot ObjectNav that utilizes a semantic executive to bridge the action consistency gap between perception and navigation.
- The framework integrates Persistent Candidate Memory, a Finite-State Executive Controller, and Stability-Aware Action Control to maintain stable object hypotheses and robust navigation commitments.
- By enforcing a reliability contract between perception and planning, the system improves success rates and path efficiency without requiring retraining of the underlying detector or planner.

---

[Nautilus Compass: Black-box Persona Drift Detection for Production LLM Agents](http://arxiv.org/abs/2605.09863)

- Nautilus Compass: introduces a black-box persona drift detection system that operates in user-space hooks to monitor LLM agent consistency without requiring access to model weights.
- The system utilizes a BGE-m3 Embedder Daemon to compute cosine similarity between user prompts and task-shaped behavioral anchors, aggregating results via a weighted top-k mean to generate drift scores.
- Nautilus Compass integrates memory recall, drift scoring, and strategy retrieval into a unified pipeline that injects behavioral context into the LLM prompt to mitigate persona drift in production agents.

---

[EnactToM: An Evolving Benchmark for Functional Theory of Mind in Embodied Agents](http://arxiv.org/abs/2605.09826)

- EnactToM: introduces an evolving benchmark for functional Theory of Mind in embodied multi-agent settings, utilizing Embodied Agents, 3D Household Environment, PDDL Goal Formula, Epistemic Operators, Generation Agent, LLM Judge Council, Structural Calibrator, Task Pool, Literal ToM Probes, and Functional ToM Evaluation.
- The framework evaluates the gap between an agent's ability to report beliefs (Literal ToM) and its capacity to act on those beliefs during grounded multi-agent coordination (Functional ToM).
- The benchmark employs an autonomous generation pipeline that uses PDDL-verified tasks and failure-biased evolution to maintain difficulty as LLMs improve.

---

#### 10th May 2026

[CalBench: Evaluating Coordination–Privacy Trade-offs in Multi-Agent LLMs](http://arxiv.org/abs/2605.09823)

- CalBench: introduces a decentralized multi-agent evaluation environment for calendar scheduling that requires agents to coordinate under private information constraints while balancing coordination quality against privacy leakage.
- The framework utilizes an oracle-anchored evaluation harness to measure performance across success rate, communication efficiency, coordination quality, fairness, and privacy preservation using the VPS metric.
- Experimental results across seven LLMs demonstrate a systematic privacy-efficiency tradeoff, where models that share more precise cost information achieve better coordination but suffer higher privacy leakage.

---

[Oracle Poisoning: Corrupting Knowledge Graphs to Weaponise AI Agent Reasoning](http://arxiv.org/abs/2605.09822)

- Oracle Poisoning: introduces an attack class where adversaries corrupt structured knowledge graphs to manipulate AI agent reasoning via trusted tool-use protocols.
- The research demonstrates that LLMs consistently treat poisoned graph data as ground truth, leading to incorrect conclusions despite sound reasoning processes.
- The authors propose a VUT (Visibility-Understanding-Traceability) framework to mitigate these vulnerabilities, emphasizing read-only access control and multi-tool cross-verification as primary defenses.

---

[Learning to Compress Time-to-Control: A Reinforcement Learning Framework for Chronic Disease Management](http://arxiv.org/abs/2605.09818)

- Chronic Care RL Framework: introduces a two-loop reinforcement learning architecture that couples clinician preference learning with an execution-intensity-constrained MDP to optimize chronic disease management.
- The framework utilizes a tiered reward hierarchy calibrated to the CMS ACCESS model to improve credit assignment and a two-layer action taxonomy to distinguish between clinical and operational interventions.
- An engagement harness serves as a supervisory layer to bound agent autonomy by routing high-risk or high-uncertainty decisions to clinicians while permitting autonomous execution of routine operational tasks.

---

[Efficient Multi-Robot Motion Planning with Precomputed Translation-Invariant Edge Bundles](http://arxiv.org/abs/2605.09801)

- KiTE-Extend: introduces a planner-agnostic action selection mechanism that leverages precomputed translation-invariant edge bundles to accelerate kinodynamic node expansion in sampling-based motion planning.
- The framework reframes node expansion as a retrieval and ranking problem, allowing planners to reuse dynamically feasible trajectory segments while preserving the theoretical guarantees of the underlying sampling-based approach.
- Experimental results demonstrate that integrating KiTE-Extend into centralized, prioritized, and conflict-based MRMP paradigms consistently improves success rates, reduces computation time, and enhances solution quality across diverse kinodynamic systems.

---

[LLM Agents Enable User-Governed Personalization Beyond Platform Boundaries](http://arxiv.org/abs/2605.09794)

- User-Governed Personalization: introduces a paradigm shift where users aggregate fragmented cross-platform data to enable holistic personalization via an LLM Agent.
- The framework leverages the User as the unique integration point, utilizing Cross-Platform Data processed by an LLM Agent to generate a Personalization Output that transcends individual platform silos.
- Empirical results demonstrate that this approach outperforms single-platform baselines by revealing user interests invisible to isolated recommendation systems.

---

[Attribution-based Explanations for Markov Decision Processes](http://arxiv.org/abs/2605.09780)

- Attribution-based Explanations for Markov Decision Processes framework: introduces a formal characterization and optimization-based approach to compute importance scores for states and execution paths in MDPs.
- The framework utilizes strategy synthesis to quantify the influence of specific states and paths on reaching a target, addressing the limitations of static attribution methods in sequential decision-making.
- The approach provides three distinct encodings—QP, QP*, and LP*—to efficiently compute importance bounds, demonstrating scalability on complex models with up to 10,000 states and transitions.

---

[UTS at PsyDefDetect: Multi-Agent Councils and Absence-Based Reasoning for Defense Mechanism Classification](http://arxiv.org/abs/2605.09769)

- UTS at PsyDefDetect: introduces a multi-phase deliberative council of LLMs that classifies psychological defense mechanisms by evaluating evidence strength through structured advocacy and formal role decomposition.
- The architecture utilizes a Clinical Analyst, Mechanism Specialist, and Pattern Analyst for initial assessment, followed by class-specific advocates and a resolution function to mitigate majority-class bias.
- An override ensemble, comprising builder agents, a critic agent, and a regression guard, provides high-confidence corrections to the council's predictions, significantly improving classification performance on imbalanced datasets.

---

[SAGE: Scalable Agentic Grounded Evaluation for Crop Disease Diagnosis](http://arxiv.org/abs/2605.09768)

- SAGE: introduces a training-free, agentic diagnostic pipeline that utilizes a source-cited knowledge base to perform explainable plant disease diagnosis.
- The framework employs a multi-turn reasoning agent that observes test images, narrows candidates via an anatomical index, and performs sequential comparisons against reference images to produce a verifiable reasoning trace.
- By grounding predictions in source-cited symptom descriptions and reference images, the system achieves significant accuracy improvements over baseline models without requiring task-specific training.

---

[RubricRefine: Improving Tool-Use Agent Reliability with Training-Free Pre-Execution Refinement](http://arxiv.org/abs/2605.09730)

- RubricRefine: introduces a training-free, pre-execution semantic contract verification method that uses task-specific rubrics to iteratively repair code-mode agent programs before execution.
- The framework employs a generator-verifier loop where the verifier provides structured item-level feedback based on a generated rubric, enabling targeted repairs of inter-tool contract violations.
- RubricRefine achieves high reliability in single-attempt settings by utilizing early stopping triggered when candidate code satisfies all rubric criteria, significantly reducing latency and LM call usage.

---

[Metal-Sci: A Scientific Compute Benchmark for Evolutionary LLM Kernel Search on Apple Silicon](http://arxiv.org/abs/2605.09708)

- METAL-SCI: introduces a 10-task scientific compute benchmark for Apple Silicon that utilizes a Frozen LLM, Runtime Compiler, Dispatch, Score, Feedback, Evolutionary Loop, and Held-out Gate to optimize GPU kernels.
- The framework employs a (1+1) evolutionary loop where a Frozen LLM iteratively refines kernels based on structured Feedback from a Runtime Compiler and Dispatch mechanism.
- A Held-out Gate provides an external oversight primitive, evaluating the final kernel on unseen configurations to detect silent correctness violations or performance regressions that in-distribution scoring misses.

---

[MOTOR-Bench: A Real-world Dataset and Multi-agent Framework for Zero-shot Human Mental State Understanding](http://arxiv.org/abs/2605.09703)

- MOTOR-MAS: introduces a structured multi-agent framework that coordinates specialized Behavior Agent, Cognition Agent, and Emotion Agent to infer human mental states in collaborative learning.
- The framework utilizes a staged reasoning process where intermediate predictions from earlier agents serve as anchors to support subsequent inferences by later agents.
- By grounding agent communication in self-regulated learning theory, the system effectively reduces hallucinations and improves performance on latent mental state inference compared to black-box LLMs.

---

[Ambig-DS: A Benchmark for Task-Framing Ambiguity in Data-Science Agents](http://arxiv.org/abs/2605.09698)

- Ambig-DS: introduces a diagnostic benchmark to evaluate whether data-science agents can recognize and resolve task-framing ambiguity before executing pipelines.
- The framework utilizes two diagnostic suites, Ambig-DS-Target and Ambig-DS-Objective, to measure silent failure modes where LLMs commit to unintended task framings without flagging underspecification.
- Experimental results across five frontier LLMs demonstrate that while agents can utilize clarification when provided, they struggle to reliably detect when to ask, often defaulting to silent, suboptimal commitments.

---

[Unpredictability dissociates from structured control in language agents](http://arxiv.org/abs/2605.09692)

- Language Agent Family: introduces a lesionable framework to test whether stochastic sampling can substitute for structured control mechanisms that couple reasons, memory, self-state, and inhibition to action selection.
- The study demonstrates that increasing stochasticity raises action entropy but fails to reproduce the structured-control profile maintained by the agent's internal components.
- Experimental results across seven datasets and multiple LLMs confirm that structured action control remains distinct from stochastic unpredictability, even when controlling for reporting format and entropy.

---

[MonitoringBench: Semi-Automated Red-Teaming for Agent Monitoring](http://arxiv.org/abs/2605.09684)

- MonitoringBench: introduces a semi-automated red-teaming methodology that decomposes attack construction into Strategy Generation, Execution, and Refinement Pipeline to expose harder-to-catch attacks for LLM-based agent monitors.
- The framework utilizes a Reconnaissance Agent equipped with a Monitor Tool and Think Tool to iteratively optimize attack strategies against development monitors, effectively bridging the conceive-execute gap.
- By applying this pipeline to the BashArena environment, the authors produce a difficulty-graded benchmark of 2,644 attack trajectories that reveals critical monitor failure modes such as susceptibility to persuasion and scoring calibration errors.

---

[DeepTumorVQA: A Hierarchical 3D CT Benchmark for Stage-Wise Evaluation of Medical VLMs and Tool-Augmented Agents](http://arxiv.org/abs/2605.09679)

- DeepTumorVQA: introduces a hierarchical 3D CT benchmark that decomposes medical reasoning into recognition, measurement, visual reasoning, and medical reasoning stages to diagnose model failures.
- The framework evaluates both direct VLM inference and tool-augmented agents using segment_organ, measure, lookup_medical_knowledge, crop_region, and a ReAct-style agent loop to isolate performance bottlenecks.
- Benchmarking over 30 model configurations reveals that reliable quantitative measurement is the primary bottleneck in 3D CT diagnostic tasks, which tool augmentation and trace supervision effectively mitigate.

---

[CodeClinic: Evaluating Automation of Coding Skills for Clinical Reasoning Agents](http://arxiv.org/abs/2605.09675)

- CodeClinic: introduces a benchmark and autoformalization pipeline for evaluating how LLMs synthesize and compose reusable clinical skills from natural-language guidelines to automate EHR reasoning.
- The framework utilizes an offline autoformalization pipeline where an LLM agent, supported by a ReACT loop and a verification agent, transforms clinical guidelines into a verified Python function library.
- This approach enables LLMs to perform longitudinal ICU surveillance and compositional information seeking by invoking verified, reusable code instead of relying on static toolboxes or zero-shot code generation.

---

[Workspace Optimization: How to Train Your Agent](http://arxiv.org/abs/2605.09650)

- DREAMTEAM: introduces a workspace optimization framework where frozen LLMs adapt by editing a structured, typed workspace of executable artifacts instead of updating model weights.
- The framework utilizes a multi-agent harness comprising observer-, simulator-, explorer-, critic-, and leader-agents that maintain and refine an inspectable world model through commit-then-retrodict loops.
- By routing prediction failures to specific artifact owners and validating patches against a regression set, the system enables efficient, sample-constrained adaptation in unfamiliar interactive environments.

---

[PDEAgent-Bench: A Multi-Metric, Multi-Library Benchmark for PDE Solver Generation](http://arxiv.org/abs/2605.09636)

- PDEAgent-Bench: introduces a multi-metric, multi-library benchmark for evaluating LLMs and code agents in synthesizing executable numerical solvers from PDE specifications.
- The framework utilizes a staged evaluation pipeline that sequentially verifies code executability, numerical accuracy against reference solutions, and computational efficiency within isolated sandboxes.
- The benchmark includes 645 instances across 11 PDE families and 3 FEM libraries (DOLFINx, Firedrake, deal.II) to assess model performance in scientific computing tasks.

---

[Statistical Scouting Finds Debate-Safe but Not Debate-Useful Cases: A Matched-Ceiling Study of Open-Weight LLM Reasoning Protocols](http://arxiv.org/abs/2605.09618)

- Statistical Scouting Framework: evaluates reasoning protocols under a matched token ceiling to determine if cheap pre-deliberation signals can recover latent routing headroom from oracle-selected protocols.
- The study identifies that vote entropy effectively predicts debate safety by reducing backfire, but fails to identify debate-useful cases where voting is unanimous yet incorrect.
- Empirical results demonstrate that learned controllers underperform simple entropy thresholds, and a naive self-critique behavioral probe provides zero discriminative signal due to sycophancy or format-compliance artifacts.

---

[Trust Me, Import This: Dependency Steering Attacks via Malicious Agent Skills](http://arxiv.org/abs/2605.09594)

- Dependency Steering: introduces a semantic-preserving optimization framework that manipulates LLM coding agents into recommending attacker-controlled packages by injecting malicious persistent instruction artifacts known as Skills.
- The framework utilizes a Multi-Objective Dependency Steering Scorer, an Explicit Veto Mechanism, a Context-Patch Injection Engine, a Dependency Steering Strategy Library, and Lifelong Semantic Strategy Exploration to generate stealthy, effective adversarial patches.
- Evaluations across multiple LLMs demonstrate that the approach achieves high targeted hallucination rates, exhibits strong cross-model and cross-domain transferability, and effectively evades most existing static and LLM-based security auditing tools.

---

[ConCovUp: Effective Agent-Based Test Driver Generation for Concurrency Testing](http://arxiv.org/abs/2605.09573)

- ConCovUp: introduces a multi-agent framework that integrates LLMs with program analysis to automate the generation of concurrent test drivers for C/C++ libraries.
- The framework utilizes an Analysis Agent to identify shared memory targets, a Path Agent to perform heuristic-guided backward tracing for constraint satisfaction, and a Test Generation Agent to synthesize executable concurrent harnesses.
- ConCovUp operates within an iterative feedback loop, using dynamic coverage reports to refine test drivers and improve the coverage of hard-to-reach concurrent behaviors.

---

[TacoMAS: Test-Time Co-Evolution of Topology and Capability in LLM-based Multi-Agent Systems](http://arxiv.org/abs/2605.09539)

- TacoMAS: introduces a test-time co-evolution framework for LLMs that jointly optimizes agent capabilities and communication topology through two coupled loops.
- The framework utilizes a fast capability loop for rapid expertise refinement via Meta-LLM and Meta-Judge feedback, and a slow topology loop for structural graph adaptation.
- TacoMAS models the co-evolution as a two-timescale replicator-mutator process, ensuring stable convergence toward task-conditioned equilibria in multi-agent systems.

---

[Governing AI-Assisted Security Operations: A Design Science Framework for Operational Decision Support](http://arxiv.org/abs/2605.09534)

- Governed AI query-broker architecture: introduces a socio-technical framework that separates AI planning from operational execution to ensure accountability in high-risk security environments.
- The framework utilizes RAG grounding, policy-enforced KQL brokers, and source-specific adapters to manage AI-assisted threat hunting while maintaining auditability and cost control.
- This study provides design propositions and a maturity model to guide engineering managers in transitioning AI-assisted operations from informal experimentation to governed production.

---

[MemPrivacy: Privacy-Preserving Personalized Memory Management for Edge-Cloud Agents](http://arxiv.org/abs/2605.09530)

- MemPrivacy: introduces a privacy-preserving framework for edge-cloud agents that uses local reversible pseudonymization to protect sensitive data while maintaining memory utility.
- The framework employs a Local MemPrivacy Model to identify privacy spans and replace them with typed placeholders, which are then restored locally after cloud-side processing.
- MemPrivacy includes a four-level privacy taxonomy and the MemPrivacy-Bench dataset to systematically evaluate the trade-off between privacy protection and personalized memory utility.

---

[Emergent Communication for Co-constructed Emotion Between Embodied Agents via Collective Predictive Coding](http://arxiv.org/abs/2605.09522)

- Inter-GMM+MVAE: introduces a computational framework for the co-construction of emotion by integrating MVAE, GMM, and MHNG to align symbolic representations between embodied agents.
- The framework utilizes MVAE to fuse multimodal sensory inputs into a shared latent space, while MHNG enables agents to align emotion categories through symbolic communication without explicit feedback.
- Experiments demonstrate that this approach allows agents with divergent interoceptive dynamics to achieve robust categorical alignment, supporting the theory that interoceptive heterogeneity is constitutive of emotional meaning.

---

[A Game-Theoretic Free Energy Analysis of Higher-Order Synergy in Attention Heads of Large Language Models](http://arxiv.org/abs/2605.09515)

- GT-FEP: introduces a game-theoretic framework that models LLM attention heads as bounded-rational agents minimizing variational free energy to quantify higher-order synergistic interactions.
- The framework utilizes Harsanyi dividends to decompose coalition energy, revealing that triple interactions in LLMs consistently exhibit higher-order redundancy.
- By applying the Nash-FEP correspondence, the authors derive a principled pruning criterion based on Shapley values that effectively compresses LLMs while maintaining performance.

---

[Position: AI Security Policy Should Target Systems, Not Models](http://arxiv.org/abs/2605.09504)

- Swarm-attack: introduces an adversarial testing framework where multiple LFM2.5-1.2B-Thinking agents coordinate through shared memory, parallel exploration, and evolutionary optimization to bypass frontier model safety and discover software vulnerabilities.
- The framework utilizes a pipeline of LFM2.5-1.2B-Thinking agents, regex pattern detection, hand-crafted exploit seeds, and AddressSanitizer-based crash classification to compensate for the limited reasoning capacity of individual small LLMs.
- Empirical results demonstrate that system-level scaffolds enable small open-weights models to achieve offensive capabilities comparable to frontier models, suggesting that security policy should prioritize system-level assessment over model-level access restrictions.

---

[Don’t Click That: Teaching Web Agents to Resist Deceptive Interfaces](http://arxiv.org/abs/2605.09497)

- DUDE (Deceptive UI Detector &amp; Evaluator): introduces a two-stage framework that enhances VLM agent robustness against deceptive web interfaces by combining Hybrid-Reward Learning for calibrated evaluation and Experience Summarization for transferable knowledge accumulation.
- The framework utilizes an interposed evaluator that assesses candidate interactions, balancing task completion against deception avoidance through asymmetric penalties and iterative failure distillation.
- The authors introduce the RUC (Real UI Clickboxes) benchmark, comprising 1,407 scenarios to evaluate agent performance and susceptibility to dark patterns across four distinct domains.

---

[LASSA Architecture-Based Autonomous Fault-Tolerant Control of Unmanned Underwater Vehicles](http://arxiv.org/abs/2605.09494)

- LASSA (LLM-based Agent with Solver, Sensor and Actuator): introduces a hierarchical, dual-loop control architecture for UUVs that integrates L-LLM, A-Agent, and S-Solver to enable autonomous fault-tolerant navigation in communication-constrained environments.
- The architecture utilizes a fast-slow control loop design where the slow loop performs high-level cognitive reasoning and strategy optimization, while the fast loop ensures real-time, high-precision low-level actuation.
- By grounding LLM-generated strategies through a deterministic S-Solver, the framework suppresses physically infeasible hallucinations and provides interpretable, auditable decision-making for complex underwater fault scenarios.

---

[Kintsugi: Learning Policies by Repairing Executable Knowledge Bases](http://arxiv.org/abs/2605.09487)

- Kintsugi: introduces a white-box policy-learning framework that treats embodied policy improvement as the verifier-gated construction of a typed executable Knowledge Base (KB).
- The framework utilizes a restricted LLM editor to propose localized typed edits to the Knowledge Base, which are then validated by a deterministic applier and verifier gate before deployment.
- At inference, the accepted Knowledge Base is executed by a deterministic symbolic executor with zero LLM calls, ensuring inspectability, local editability, and verifier-gated deployment.

---

[PolicyCache-SDN: Hierarchical Intra-Path Learning for Adaptive SDN Traffic Control](http://arxiv.org/abs/2605.09473)

- PolicyCache-SDN: introduces a hierarchical SDN traffic control framework that enables local online adaptation by confining learning to path-specific aggregates within controller-defined policy envelopes.
- The architecture utilizes an SDN Controller to manage global intent and compute policy envelopes, while Edge Agents perform fast local learning using HAT models to execute metering, queueing, and rerouting actions.
- By combining centralized policy constraints with decentralized intra-path learning, the framework ensures safe, auditable, and efficient traffic engineering that outperforms static and offline-learned baselines.

---

[Through the Lens of Character: Resolving Modality-Role Interference in Multimodal Role-Playing Agent](http://arxiv.org/abs/2605.09443)

- CAVI (Character-Aware Visual Intervention): introduces a training-free framework that resolves Modality-Role Interference in MLLMs by aligning visual perception with subjective character personas through Character Anchor, Character-Guided Token Pruning, Orthogonal Feature Modulation, Modality-Adaptive Role Steering, and Dynamic Contextual Re-Injection.
- The framework functions as an information-theoretic bottleneck that systematically purifies role-consistent visual facts from objective noise to enhance character-consistent multimodal interactions.
- Extensive experiments demonstrate that CAVI significantly improves role-playing performance and generalization to unseen characters without requiring parameter updates or incurring prohibitive inference overhead.

---

[Beyond Isolation: A Unified Benchmark for General-Purpose Navigation](http://arxiv.org/abs/2605.09441)

- OmniNavBench: introduces a unified benchmark designed to rigorously evaluate cross-skill coordination and cross-embodiment generalization in embodied navigation by composing sequences of primary sub-tasks.
- The framework utilizes a high-fidelity simulation platform and human-expert teleoperated trajectories to assess agent performance across diverse robot morphologies and complex, multi-stage navigation instructions.
- Extensive evaluations reveal that current unified models struggle with sequential task coordination, termination decisions, and dynamic social interaction, highlighting the need for more robust generalist navigation agents.

---

[SimWorld Studio: Automatic Environment Generation with Evolving Coding Agent for Embodied Agent Learning](http://arxiv.org/abs/2605.09423)

- SimWorld Studio: introduces an open-source platform for generating evolving embodied learning environments using a tool-augmented coding agent that constructs physically grounded 3D worlds.
- The framework utilizes SimCoder, which leverages a Tool Library, Skill Library, and MCP Server to iteratively build and refine environments based on feedback from Scene Verification.
- SimWorld Studio enables a co-evolution loop where the Embodied Agent's performance feedback guides the generation of adaptive curricula, significantly improving agent generalization and training efficiency.

---

[MACAA: Belief-Revision Multi-Agent Reasoning for Open-World Code Authorship Verification](http://arxiv.org/abs/2605.09421)

- MACAA: introduces a belief-revision-based multi-agent framework for training-free code authorship verification that utilizes a Coordinator Agent and four Expert Agents to analyze layout, lexical, syntactic, and programming-pattern evidence.
- The framework operationalizes AGM belief revision theory through expansion, contraction, and revision operations to maintain a consistent authorship hypothesis while resolving conflicts in heterogeneous code pairs.
- MACAA provides explicit evidence tracing and forensic auditability by recording the belief revision path, enabling inspection of the feature-level basis for each authorship decision.

---

[Empowering VLMs for Few-Shot Multimodal Time Series Classification via Tailored Agentic Reasoning](http://arxiv.org/abs/2605.09395)

- MarsTSC (Multimodal Time Series Classification): introduces a VLM-based agentic reasoning framework that utilizes a dynamic, self-evolving knowledge bank to improve few-shot time series classification performance.
- The framework employs a collaborative trio of agents—a Generator, a Reflector, and a Modifier—to iteratively refine task-specific knowledge through trial-and-error without updating model parameters.
- A test-time update mechanism with two-pass generation and deferred knowledge validation mitigates few-shot bias and distribution shift, ensuring robust and interpretable classification decisions.

---

[NEXUS: Continual Learning of Symbolic Constraints for Safe and Robust Embodied Planning](http://arxiv.org/abs/2605.09387)

- NEXUS: introduces a modular neuro-symbolic framework that integrates LLMs with formal methods to enforce safe and feasible embodied planning through continual learning.
- The framework includes planning-, perception- and safety-agents that decouple physical feasibility from safety specifications by grounding probabilistic risk assessments into deterministic hard LTL constraints.
- NEXUS achieves superior task success rates and robust defense against adversarial attacks by progressively improving planning efficiency through knowledge accumulation in PDDL domains and LTL safety specifications.

---

[Towards a Virtual Neuroscientist: Autonomous Neuroimaging Analysis via Multi-Agent Collaboration](http://arxiv.org/abs/2605.09366)

- NIAgent: introduces a multi-agent system for autonomous end-to-end neuroimaging analysis that utilizes Supervisor Agent, Data Awareness Agent, Quality Control Agent, Processing Agent, Downstream Analysis Agent, Just-in-Time Context Injection, Code-Centric Execution, Domain-Specific Primitive Libraries, and Hierarchical QC Module.
- The framework employs a centralized-planning and decentralized-execution design where specialist agents synthesize executable programs over composable domain-specific primitives to handle long-horizon neuroimaging workflows.
- The system incorporates a closed-loop quality control mechanism that integrates cohort-level metric screening with agentic visual inspection to drive evidence-grounded workflow remediation.

---

[Multi-scale Predictive Representations for Goal-conditioned Reinforcement Learning](http://arxiv.org/abs/2605.09364)

- Ms.PR (Multi-scale Predictive Representations): introduces a framework that enforces dynamical, behavioral, and temporal alignment through multi-scale predictive supervision to construct robust latent representations for offline goal-conditioned reinforcement learning.
- The framework utilizes a shared encoder architecture with specialized predictive modules to decouple representation learning from value approximation, effectively mitigating value overestimation in sparse-reward environments.
- Ms.PR demonstrates superior performance and robustness across diverse state-based and pixel-based tasks, particularly under conditions of trajectory stitching, limited data, and noisy expert demonstrations.

---

[Skill-R1: Agent Skill Evolution via Reinforcement Learning](http://arxiv.org/abs/2605.09359)

- Skill-R1: introduces a reinforcement learning framework for recurrent agent skill evolution that decouples task execution from skill improvement by training a lightweight skill generator while keeping the task LLM frozen.
- The framework utilizes a bi-level group-relative policy optimization objective that combines intra-generation rollout quality comparisons with inter-generation progress rewards to guide directional skill refinement.
- Empirical results demonstrate that Skill-R1 consistently outperforms no-skill baselines and standard GRPO across complex multi-step reasoning and tool-use benchmarks.

---

[PECMAN: Perception-enabled Collaborative Multi-Agent Navigation in Unknown Environments](http://arxiv.org/abs/2605.09344)

- PECMAN: introduces a multi-agent navigation framework that utilizes LiDAR Scan, Shared Perception, Distributed Tree Repair, Narrow Corridor Coordination, and a Global King Priority Layer to enable efficient navigation in unknown, dynamic environments.
- The framework extends the SMART-3D single-agent planner by implementing distributed tree morphing and a coordination layer that resolves deadlocks in narrow corridors through priority-based agent sequencing.
- Experimental results demonstrate that the shared perception strategy significantly reduces team-completion time and improves situational awareness compared to independent navigation modes.

---

[A Cross-Layered Multi-Drone Coordination for Medical Supply Delivery during Disaster Response Management](http://arxiv.org/abs/2605.09342)

- CEDA (Centralized Training with Decentralized Execution Deep Q-Network): introduces a unified reinforcement learning framework that jointly optimizes drone routing, multi-agent coordination, and triage-priority-aware scheduling across physical, network, and application layers.
- The framework utilizes a centralized Q-network during training to learn a policy that enables decentralized execution by individual drones, ensuring robustness to communication disruptions in disaster response environments.
- CEDA incorporates a Priority-Preserving Fair Scheduling strategy that uses dynamic per-patient survival models and multiple reward components to ensure equitable service across triage classes without requiring explicit fairness constraints.

---

[SkillMAS: Skill Co-Evolution with LLM-based Multi-Agent System](http://arxiv.org/abs/2605.09341)

- SkillMAS: introduces a non-parametric framework that couples skill evolution and MAS restructuring by constraining both processes under a shared verified-trace evidence surface.
- The framework utilizes Utility Learning to assign credit to execution-supported skills and participating executors, while employing evidence-gated MAS restructuring to modify organization only when structural bottlenecks are identified.
- SkillMAS improves post-deployment specialization across embodied manipulation, command-line OS workflows, and retail-service interaction without requiring retraining of the underlying LLMs.

---

[The Trap of Trajectory: Towards Understanding and Mitigating Spurious Correlations in Agentic Memory](http://arxiv.org/abs/2605.09330)

- CAMEL (CAusality-informed MEmory caLibration): introduces a plug-and-play calibration framework that mitigates spurious correlations in agentic memory by applying Write-stage calibration and Retrieve-stage calibration to existing memory architectures.
- The framework utilizes Residualization and a Content-novelty write criterion to neutralize confounding and collider bias at the point of memory storage, while employing a Causal stability test to filter retrieved candidates based on their sensitivity to non-causal perturbations.
- By maintaining Streaming covariance statistics and a Non-causal subspace, CAMEL effectively reduces spurious reasoning ratios across diverse LLM-based agentic systems without requiring additional LLM forward passes or fine-tuning.

---

[OpenIIR: An Open Simulation Platform for Information Retrieval Research](http://arxiv.org/abs/2605.09321)

- OpenIIR: introduces a modular simulation platform for IR research that utilizes a shared core of Agent Runtime, World-Model Store, Retrieval Primitives, Claim Extractor, and Persona Ontology to support diverse multi-agent experiments.
- The platform enables researchers to configure LLM-driven personas across various study types, including deliberative panels, social platforms, curated feeds, and evolutionary co-evolution scenarios, using a unified type interface.
- OpenIIR provides reproducible, end-to-end simulation outputs such as argument graphs and exposure logs, allowing for complex IR research questions to be investigated on a single laptop without requiring extensive infrastructure.

---

[Mem-W: Latent Memory-Native GUI Agents](http://arxiv.org/abs/2605.09317)

- Mem-W: introduces a latent-context-native interface for GUI agents that unifies working memory and experiential memory into a single continuous embedding space using a shared trajectory-to-latent compressor.
- The framework replaces manual memory hierarchies with a learnable compressor that projects both retrieved historical trajectories and in-session episode prefixes into compact, decision-relevant latent tokens.
- Mem-W is trained via self-distillation and outcome-aware supervision to ensure that the latent memory effectively supports long-horizon task success without requiring updates to the frozen GUI agent policy.

---

[Do Self-Evolving Agents Forget? Capability Degradation and Preservation in Lifelong LLM Agent Adaptation](http://arxiv.org/abs/2605.09315)

- CPE (Capability-Preserving Evolution): introduces a stabilization principle that constrains destructive capability drift in self-evolving LLM agents by regularizing updates to the agent's mutable repository.
- The framework addresses capability erosion across four evolution dimensions—workflow, skill, model, and memory—by biasing self-evolution toward low-interference solutions that preserve previously mastered competencies.
- Empirical results demonstrate that CPE consistently improves retained capability stability and performance across diverse LLM backbones compared to unconstrained self-evolution.

---

[LEAF-SQL: Level-wise Exploration with Adaptive Fine-graining for Text-to-SQL Skeleton Prediction](http://arxiv.org/abs/2605.09295)

- LEAF-SQL: introduces a novel Text-to-SQL framework that reframes skeleton prediction as a coarse-to-fine tree search process, utilizing Level-wise Skeleton Search, Skeleton Formulation Agent, Skeleton Evaluation Agent, SQL Generation Module, and Result Selection Module.
- The framework employs a three-level skeleton hierarchy (Base, Expanded, Detailed) to guide the search process, enabling adaptive granularity and structural diversity in skeleton candidates.
- By integrating a pruning mechanism via the Skeleton Evaluation Agent, LEAF-SQL effectively navigates the search space to identify optimal query structures while maintaining computational efficiency.

---

[PiCA: Pivot-Based Credit Assignment for Search Agentic Reinforcement Learning](http://arxiv.org/abs/2605.09287)

- PiCA (Pivot-Based Credit Assignment): introduces a novel reinforcement learning framework for LLM-based search agents that reformulates search trajectories as a sequential process of cumulative progress to mitigate reward sparsity and isolated credit assignment.
- The framework utilizes a Policy LLM, a Search Engine, and a Reward Model to identify pivot steps as information peaks, providing dense, trajectory-dependent guidance through Potential-Based Reward Shaping.
- By integrating step-level explicit supervision and outcome-level implicit supervision, PiCA ensures robust generalization and improved performance across diverse knowledge-intensive QA benchmarks.

---

[A Prompt-Aware Structuring Framework for Reliable Reuse of AI-Generated Content in the Agentic Web](http://arxiv.org/abs/2605.09283)

- A Prompt-Aware Structuring Framework for Reliable Reuse of AI-Generated Content in the Agentic Web: introduces a framework that automatically attaches structured metadata and verifiable credentials to AIGC to ensure provenance and reliability for downstream reuse.
- The framework utilizes POML to modularize prompts, enabling a curation agent to mechanically evaluate instruction-following fidelity before using the content for LLM fine-tuning.
- Experimental results on ComplexBench demonstrate that metadata-driven curation of AIGC consistently improves the requirements following ratio compared to random data selection.

---

[EQUIMEM: Calibrating Shared Memory in Multi-Agent Debate via Game-Theoretic Equilibrium](http://arxiv.org/abs/2605.09278)

- EQUIMEM: introduces a zero-trust game-theoretic framework for multi-agent debate that calibrates shared memory updates using structural indicators instead of LLM-based judgments.
- The framework utilizes a contributor-auditor game structure where memory integrity is enforced through a calibration mechanism that combines detection signals, alignment signals, and credibility weights.
- By replacing vote-based commitment with algorithmic structural verification, the system achieves robust memory protection against adversarial agents without incurring additional LLM inference costs.

---

[Towards Conversational Medical AI with Eyes, Ears and a Voice](http://arxiv.org/abs/2605.09272)

- AI co-clinician: introduces a dual-agent architecture comprising a Talker (handles real-time patient interaction) and a Clinical Planner (supervisory reasoning module) to balance conversational fluency with deep clinical reasoning.
- The system utilizes continuous Audio-Video Input Stream (continuous patient data) to perform real-time diagnostic and management tasks, supported by Audio, Text Output (agent response generation) for natural dialogue.
- This research evaluates the framework against human physicians and other LLMs using a novel TelePACES rubric, demonstrating significant improvements in clinical reasoning and physical examination accuracy through the Clinical Planner.

---

[Agentic AI for Particle-Based Simulation: Automating SPH Workflows for Debris Flow Modeling](http://arxiv.org/abs/2605.09265)

- Agentic AI for Particle-Based Simulation: introduces a human-in-the-loop agentic workflow for meshless particle-based simulation using DualSPHysics, integrating Planning AI-agent, DualSPHysics solver, Post-processing AI-agent, Human-in-the-loop interface, and Multimodal input module.
- The framework utilizes a stage-structured design to separate reasoning-intensive tasks from deterministic simulation execution, enhancing robustness in complex debris flow modeling.
- A cognitive-task-oriented evaluation framework is employed to assess agent performance across scalar extraction, visualization, phase identification, physical derivation, and geometric disambiguation.

---

[LLM Agents Already Know When to Call Tools - Even Without Reasoning](http://arxiv.org/abs/2605.09252)

- PROBE&PREFILL: introduces a lightweight inference-time intervention that leverages hidden-state signals to steer LLM tool-call decisions without requiring model fine-tuning.
- The framework utilizes the WHEN2TOOL benchmark to demonstrate that LLMs internally encode tool-necessity information, which can be extracted via a linear probe to guide generation.
- By prepending a steering sentence based on probe predictions, the method achieves a superior accuracy-efficiency tradeoff compared to prompt engineering or explicit reasoning baselines.

---

#### 9th May 


[OTora: A Unified Red Teaming Framework for Reasoning-Level Denial-of-Service in LLM Agents](http://arxiv.org/abs/2605.08876)

- OTora (Unified Red Teaming Framework for Reasoning-Level Denial-of-Service): introduces a two-stage red-teaming pipeline that induces Reasoning-Level Denial-of-Service (R-DoS) in LLM agents by triggering external tool access and injecting reasoning-intensive payloads to exhaust reasoning budgets.
- The framework utilizes a Stage I Trigger Optimizer to hijack tool calls and a Stage II Payload Optimizer to embed persistent, task-consistent reasoning detours that inflate end-to-end latency while maintaining functional correctness.
- Experimental results demonstrate that OTora achieves order-of-magnitude increases in reasoning tokens and latency across diverse LLM agents and tool interfaces without triggering standard safety or early-stop mechanisms.

---



[AssemPlanner: A Multi-Agent Based Task Planning Framework for Flexible Assembly System](http://arxiv.org/abs/2605.08831)

- AssemPlanner: introduces a hierarchical multi-agent framework that orchestrates SchedAgent, KnowledgeAgent, LineBalanceAgent, and a Scene Graph to transform natural-language instructions into executable industrial assembly sequences.
- The framework utilizes a closed-loop "Reason-Action-Observe" cycle where the SchedAgent dynamically negotiates with specialized agents to ensure constraint compliance and logical consistency.
- By integrating KG-enhanced RAG and prompt-driven reflective optimization, the system eliminates the need for expert-level parameter tuning and manual reward engineering in flexible assembly environments.

---



[Internal vs. External: Comparing Deliberation and Evolution for Multi-Agent Constitutional Design](http://arxiv.org/abs/2605.09128)

- Internal vs. External Constitutional Design Framework: compares internal deliberation and external evolution for governing LLM-based multi-agent systems across three social environments.
- The study reveals that external evolution excels in stable collective-action settings by discovering behavioral directives, while internal deliberation provides structural responsiveness by adapting governance policies to shifting incentives.
- The research identifies a structural punishment gap where deliberating agents avoid costly peer-to-peer sanctions, unlike evolved constitutions that reliably converge on cooperation-sustaining enforcement mechanisms.

---


[Trustworthy AI: Ensuring Reliability and Accountability from Models to Agents](http://arxiv.org/abs/2605.08964)

- Trustworthy AI: introduces a comprehensive framework for ensuring reliability and accountability in ML systems, ranging from predictive models to generative LLMs and autonomous agents.
- The research develops kernel-based methods for multiaccuracy, information-theoretic watermarking schemes for LLMs, and a multi-agent simulator for evaluating autonomous supply chain management.
- The thesis addresses systemic risks in AI by formalizing arbitrariness in predictive models, establishing provable detection-distortion tradeoffs for LLM watermarking, and identifying emergent dynamics in multi-agent systems.

---

[Toward Web 4.0: Bidirectional Trust between AI Agents and Blockchain](http://arxiv.org/abs/2605.08922)

- Bidirectional Trust Framework: introduces a systematization of knowledge for the convergence of autonomous AI agents and blockchain, formalizing the interaction through an Agent-Blockchain Interaction Model (ABIM) and a five-dimensional evaluation framework.
- The framework categorizes the symbiotic relationship into two directions: B→A (Blockchain providing trust infrastructure for agents) and A→B (AI agents participating in core blockchain mechanisms), both underpinned by a Trust Foundation of verifiable computation.
- The analysis identifies significant ecosystem immaturity, where most agent-specific standards remain in draft status, and highlights critical research gaps including the LLM verifiability bottleneck and the need for formal security models for agent-blockchain interactions.

---

[Generalization Bounds of Emergent Communications for Agentic AI Networking](http://arxiv.org/abs/2605.08613)

- DIB-based Emergent Communication Framework: introduces a multi-agent communication architecture that leverages Distributed Information Bottleneck (DIB) theory to unify decision-making functions and emergent communication protocols through a joint loss function.
- The framework optimizes the trade-off between task-relevant information and computational complexity, represented by the Minimum Description Length (MDL), to ensure robust performance in resource-constrained AgentNet environments.
- Theoretical generalization bounds are derived to guarantee performance stability when agents transition from training datasets to unseen decentralized inference scenarios.

---

[ProactBench: Beyond What The User Asked For](http://arxiv.org/abs/2605.09228)

- ProactBench: introduces a three-agent evaluation architecture to measure conversational proactivity by testing if an LLM can identify and act on unstated user needs.
- The framework decomposes proactivity into three phase-tied trigger types: EMERGENT (single-anchor inference), CRITICAL (multi-anchor synthesis), and RECOVERY (grounded forward-looking value after task completion).
- By enforcing cold-start isolation and using varied communication styles, the benchmark evaluates models on their ability to provide useful, unprompted assistance without collapsing into reactive instruction-following.

---

[Flame3D: Zero-shot Compositional Reasoning of 3D Scenes with Agentic Language Models](http://arxiv.org/abs/2605.09218)

- Flame3D: introduces a training-free framework that represents 3D scenes as editable visual-textual memories, enabling compositional reasoning through a set of spatial and meta-tools.
- The framework utilizes a tool-calling VLM to interact with a structured scene memory, allowing the agent to synthesize custom spatial programs at inference time for complex, multi-hop queries.
- Flame3D achieves competitive performance on 3D benchmarks without 3D-specific fine-tuning by leveraging explicit scene representations and expressive compositional abstractions.

---

[Learning the Preferences of a Learning Agent](http://arxiv.org/abs/2605.09217)

- Learning the Preferences of a Learning Agent: introduces a formal framework for a predictor to infer the underlying reward function of a learner agent that is itself learning to act optimally over time.
- The framework models the learner as either a no-regret agent or a Boltzmann-rational agent, establishing theoretical guarantees for reward inference under different environment settings.
- The research demonstrates that while inferring the full preference structure of a no-regret learner is generally impossible, Boltzmann-rationality allows for meaningful bounds on prediction error.

---

[Rethinking Ratio-Based Trust Regions for Policy Optimization in Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2605.09212)

- MARS (Multi-Agent Ratio Symmetry): introduces a policy optimization objective for cooperative MARL that replaces additive trust-region mechanisms with a multiplicatively symmetric geometric barrier to preserve corrective gradients and prevent probability extinction.
- The framework utilizes a centralized critic to compute joint advantage estimates, which are then used to calibrate a geometric penalty that remains unbounded as probability ratios approach zero.
- By ensuring that probability extinction is never optimal, MARS maintains stable training dynamics in high-variance continuous control and discrete coordination tasks compared to standard MAPPO and MASPO methods.

---

[Evidence Over Plans: Online Trajectory Verification for Skill Distillation](http://arxiv.org/abs/2605.09192)

- SPARK (Structured Pipelines for Autonomous Runnable tasKs and sKill generation): introduces a framework for distilling environment-grounded skills from agent trajectories by utilizing the Posterior Distillation Index to ensure procedural knowledge is based on empirical evidence rather than prior plans.
- The framework includes a Teacher Agent for exploration, a Student Agent for skill consumption, and a PDI-based online intervention mechanism to improve skill generation quality during the exploration process.
- SPARK demonstrates that posterior-based skill distillation significantly improves task success rates and transferability across different LLM models while reducing inference costs.

---

[Agentic MIP Research: Accelerated Constraint Handler Generation](http://arxiv.org/abs/2605.09186)

- Agentic MIPR: introduces a solver-aware harness that embeds LLM agents to automate the generation, verification, and tuning of SCIP constraint handlers for MIP research.
- The framework utilizes a pipeline of Prompt Specification, Plugin Generation, In-Context Learning, and Large-Scale Evaluation to bridge the gap between algorithmic hypotheses and executable solver plugins.
- By leveraging a sandbox environment and structured stage contracts, the framework enables the autonomous discovery of novel constraint patterns and propagation strategies that improve solver performance on complex MIPLIB 2017 instances.

---

[AutoRedTrader: Autonomous Red Teaming of Trading Agents through Synthetic Misinformation Injection](http://arxiv.org/abs/2605.09185)

- AutoRedTrader: introduces an autonomous red-teaming framework that generates finance-specific misinformation through behavioral bias manipulation, minor textual perturbations, and style-controlled rewriting to stress-test LLM-based financial agents.
- The framework utilizes a closed-loop feedback mechanism where the financial agent's cumulative return and historical decision-impact records guide the MisGen module to iteratively strengthen misinformation generation.
- AutoRedTrader incorporates an optional time-series-informed grounding setting that provides structured historical market evidence to help LLM-based agents assess the consistency of textual signals and improve robustness against misinformation.

---

[CIVeX: Causal Intervention Verification for Language Agents](http://arxiv.org/abs/2605.09168)

- CIVeX: introduces a causal intervention verifier that maps proposed actions to structural causal queries over a committed action-state graph to ensure intervention identifiability before execution.
- The framework utilizes a Causal Certificate containing an identification argument, a one-sided Lower Confidence Bound, and risk limits to issue one of four auditable verdicts: EXECUTE, REJECT, EXPERIMENT, or ABSTAIN.
- CIVeX provides a robust safety mechanism for LLMs by treating tool execution as causal inference, effectively mitigating false executions in confounded environments where observational correlations fail.

---

[FORTIS: Benchmarking Over-Privilege in Agent Skills](http://arxiv.org/abs/2605.09163)

- FORTIS: introduces a two-stage benchmark designed to evaluate over-privilege in LLM agents by measuring their ability to select minimally sufficient skills and execute them within documented boundaries.
- The framework organizes skills and tools into a five-level privilege hierarchy to assess whether models exercise restraint or default to broader, more permissive capabilities when faced with semantic ambiguity.
- Empirical results across ten frontier models demonstrate that over-privileged behavior is a structural tendency, with failure rates often exceeding 75% under realistic conditions like convenience-sensitive framing.

---

[Beyond Self-Play: Hierarchical Reasoning for Continuous Motion in Closed-Loop Traffic Simulation](http://arxiv.org/abs/2605.09153)

- Hierarchical Framework: introduces a modular architecture that decouples high-level interaction reasoning from low-level continuous trajectory realization to improve behavioral realism in closed-loop traffic simulation.
- The system utilizes a Stackelberg-style MARL module for strategic decision-making and a command-conditioned Wayformer for physically consistent motion generation.
- A hybrid co-training scheme incorporating heuristic recovery trajectories is employed to mitigate distribution shift and stabilize closed-loop policy execution.

---

[AlphaExploitem: Going Beyond the Nash Equilibrium in Poker by Learning to Exploit Suboptimal Play](http://arxiv.org/abs/2605.09150)

- AlphaExploitem: introduces a hierarchical transformer-based architecture that enables poker agents to adapt to opponent-specific behaviors by processing multi-hand session histories.
- The framework utilizes a hierarchical history transformer consisting of a within-hand encoder and an across-hand encoder to generate a session-level context vector, which is fused with current-hand card and action encoders to inform decision-making.
- Training is performed via PPO using a combination of K-best league self-play, a long-tail buffer of past agent checkpoints, and a diverse set of hand-crafted exploitable opponent policies to ensure robust adaptation.

---

[Beyond Thinking: Imagining in 360° for Humanoid Visual Search](http://arxiv.org/abs/2605.09146)

- Imagining in 360°: introduces a decoupled architecture for Humanoid Visual Search that separates intuitive spatial imagination from rigorous action planning to improve search efficiency.
- The Imaginator functions as a probabilistic predictor of spatial priors, generating multiple hypotheses about the semantic layout of observed and unobserved regions to guide the Actor.
- This framework utilizes a fully automated pseudo-labeling pipeline to scale training to millions of samples, enabling robust performance in complex, in-the-wild environments without manual trajectory annotations.

---

[MCP-Cosmos: World Model-Augmented Agents for Complex Task Execution in MCP Environments](http://arxiv.org/abs/2605.09131)

- MCP-Cosmos: introduces a framework that integrates generative World Models into the MCP ecosystem to enable predictive task automation through latent space simulation.
- The framework utilizes a BYOWM (Bring Your Own World Model) architecture to allow LLM agents to perform speculative look-ahead searches and refine trajectories before physical execution.
- It proposes the Execution Quality metric to quantify the efficiency of tool usage by penalizing excessive tool calls and rewarding successful, proactive planning.

---

[A Communication-Theoretic Framework for LLM Agents: Cost-Aware Adaptive Reliability](http://arxiv.org/abs/2605.09121)

- AGENTCODEC: introduces a unified communication-theoretic framework for LLM agents that treats model invocations as discrete stochastic channels to enable principled reliability optimization.
- The framework maps common reliability techniques to classical communication operators, including diversity combining, hybrid retransmission, iterative generator-critic decoding, rateless sampling, structured redundant verification, and difficulty-adaptive routing.
- A cost-aware semantic-nearest-neighbor router dynamically selects the optimal reliability technique per task, significantly improving the quality-cost Pareto frontier compared to fixed-technique baselines.

---

[Token Economics for LLM Agents: A Dual-View Study from Computing and Economics](http://arxiv.org/abs/2605.09104)

- Token Economics for LLM Agents: introduces a unified framework conceptualizing tokens as production factors, exchange mediums, and units of account to evaluate the trade-off between output quality and economic cost in agentic systems.
- The framework categorizes token-based economic dynamics across three architectural scales: micro-level single-agent optimization, meso-level multi-agent system coordination, and macro-level agent ecosystem governance.
- The research synthesizes computational system design with neoclassical economic theory to address systemic bottlenecks, including transaction costs, congestion externalities, and security-related token attrition.

---

[GRC: Unifying Reasoning-Driven Generation, Retrieval and Compression](http://arxiv.org/abs/2605.09100)

- GRC: introduces a unified training framework that enables LLMs to perform reasoning-driven generation, text embedding, and context compression in a single forward pass using meta latent tokens.
- The framework utilizes meta latent tokens as internal registers to compress context into O(1) length KV cache, facilitating efficient latent memory-augmented generation and reasoning-enhanced text representation.
- Hybrid paged attention is integrated to manage regular and compressed KV caches, significantly improving inference throughput for the unified model.

---

[Robust Multi-Agent LLMs under Byzantine Faults](http://arxiv.org/abs/2605.09076)

- SAC (Self-Anchored Consensus): introduces a decentralized protocol for LLM multi-agent systems that ensures robustness against Byzantine faults by replacing sender-reported confidence with receiver-side evaluation and iterative filtering.
- The framework leverages (F+1)-robust graph-theoretic conditions to guarantee that honest agents can maintain reliable consensus and filter out malicious or faulty information.
- Experimental results demonstrate that SAC effectively preserves the performance of strong agents while improving the accuracy of weaker agents across diverse communication topologies, unlike prior confidence-weighted aggregation methods.

---

[Octopus Protocol: One-Shot Hardware Discovery and Control for AI Agents via Infrastructure-as-Prompts](http://arxiv.org/abs/2605.09055)

- Octopus Protocol: introduces a system that automates hardware driver generation and control for AI agents by utilizing a five-stage build pipeline (PROBE, IDENTIFY, INTERFACE, SERVE, DEPLOY) and a self-healing daemon (WATCH, HEAL, PERCEIVE) to eliminate manual driver engineering.
- The framework treats hardware protocols as prompts rather than static code, enabling an LLM-driven coding agent to compile declarative specifications into platform-specific MCP servers at runtime.
- By leveraging a persistent daemon loop, the system provides autonomous self-healing capabilities, allowing the agent to monitor logs, reinstall dependencies, and regenerate code to maintain hardware connectivity without human intervention.

---

[LCGNav: Local Candidate-Aware Geometric Enhancement for General Topological Planning in Vision-Language Navigation](http://arxiv.org/abs/2605.09053)

- LCGNav: introduces a modular geometric enhancement framework that improves topological planning in VLN-CE by integrating 3D point cloud-based local features into ghost nodes.
- The framework utilizes 3D Point Cloud Transformation, Depth-directed Physical Hard Truncation, Farthest Point Sampling (FPS), PointNet Encoder, Dimension-Preserving Residual Fusion, and a State Degradation Mechanism to refine local spatial perception.
- LCGNav acts as a lightweight, transferable module that enhances existing topological planners by providing physically grounded local geometric cues while maintaining original architectural dimensions.

---

[Containment Verification: AI Safety Guarantees Independent of Alignment](http://arxiv.org/abs/2605.09045)

- Containment Verification: introduces a fail-safe paradigm for AI agents that deductively verifies the containment layer mediating between the AI model and external state to enforce boundary-enforceable safety properties.
- The framework utilizes havoc oracle semantics to model the AI as an unconstrained procedure, ensuring safety guarantees are invariant to model capability, training, or alignment.
- The approach employs an agentic formal synthesis pipeline to automate the generation of machine-checked Dafny proofs, establishing a forward-simulation refinement between abstract safety specifications and concrete operational state machines.

---

[ShadowMerge: A Novel Poisoning Attack on Graph-Based Agent Memory via Relation-Channel Conflicts](http://arxiv.org/abs/2605.09033)

- SHADOWMERGE: introduces a black-box poisoning attack against graph-based agent memory that exploits relation-channel conflicts to inject malicious evidence into shared memory.
- The framework utilizes an AIR pipeline consisting of Anchor, Inscribe, and Render to ensure poisoned relations are materialized, merged into target anchor neighborhoods, and retrieved for victim queries.
- SHADOWMERGE achieves high attack success rates by manipulating the graph-memory substrate without requiring privileged access to the memory graph or retriever.

---

[GAMBIT: A Three-Mode Benchmark for Adversarial Robustness in Multi-Agent LLM Collectives](http://arxiv.org/abs/2605.09027)

- GAMBIT: introduces a benchmark and dataset for evaluating imposter detectors in multi-agent LLM collectives using a chess-based substrate with a deterministic cost function.
- The framework includes an adaptive imposter agent that evolves its strategy across generations to evade detection, alongside a three-mode evaluation protocol measuring generalization and few-shot recalibration.
- Experimental results demonstrate that static adversarial benchmarks are insufficient, as meta-learned detectors significantly outperform standard fine-tuned models in fast recalibration under distribution shift.

---

[Evolutionary Ensemble of Agents](http://arxiv.org/abs/2605.09018)

- EvE (Evolutionary Ensemble): introduces a decentralized framework that organizes coding agents into a co-evolving system to optimize algorithmic discovery by maintaining two populations of functional code solvers and agent guidance states.
- The framework utilizes a synchronous race mechanism where multiple agents are sampled to refine solvers, with their relative performance tracked via Elo ratings to drive continuous, stage-dependent adaptation.
- By integrating solver improvement and self-referential agent optimization into a unified stage, EvE enables scalable, context-aware evolution that avoids phase mismatch and breaks through static performance ceilings.

---

[Learning to Explore: Scaling Agentic Reasoning via Exploration-Aware Policy Optimization](http://arxiv.org/abs/2605.08978)

- EAPO (Exploration-Aware Policy Optimization): introduces a reinforcement learning framework that enables LLM agents to adaptively explore environments by explicitly modeling exploration utility and memory.
- The framework utilizes a variational proxy to estimate the potential of exploratory actions and an exploration-aware grouping mechanism to distinguish information-seeking behavior from task-completion actions.
- By incorporating a rollback mechanism and structured memory, the agent effectively manages uncertainty and generalizes across complex, long-horizon tasks without requiring additional fine-tuning.

---

[Agentic AI Scientists Are Not Built For Autonomous Scientific Discovery](http://arxiv.org/abs/2605.08956)

- Agentic AI Scientist Framework: introduces a critical analysis of current LLM-based agentic systems, arguing they are fundamentally unsuited for autonomous scientific discovery due to inherent design limitations in problem selection, training data, and evaluation.
- The paper identifies that preference optimization in LLMs compresses hypothesis diversity toward consensus, while current benchmarks fail to capture the multi-step, feedback-driven nature of real-world scientific inquiry.
- To address these gaps, the authors propose integrating scientific simulations as verifiers, implementing persistent world models to maintain epistemic state, and establishing centralized preregistration repositories to capture failure knowledge and improve scientific integrity.

---

[MDGYM: Benchmarking AI Agents on Molecular Simulations](http://arxiv.org/abs/2605.08941)

- MDGYM: introduces a modular evaluation harness for benchmarking AI agents on complex molecular dynamics simulation tasks.
- The framework utilizes three extensible layers—agent, orchestration, and validator—to assess LLM performance across diverse scientific simulation workflows.
- Evaluation results reveal that current LLMs struggle with physical grounding, often failing to produce stable simulation configurations despite successful code generation.

---

[PnP-Corrector: A Universal Correction Framework for Coupled Spatiotemporal Forecasting](http://arxiv.org/abs/2605.08935)

- PnP-Corrector: introduces a model-agnostic framework that decouples physical simulation from error correction by freezing pre-trained Physics Engines and training a lightweight Correction Agent to mitigate Reciprocal Error Amplification.
- The framework utilizes the DSLCast Architecture, which integrates an Axially-Gated Block for efficient spatial feature extraction and a Differentiable Semi-Lagrangian Advection Block to explicitly model physical advective motion.
- By employing an autoregressive predict-then-correct loop, the Correction Agent proactively counteracts systematic biases, ensuring long-term stability and physical realism in coupled spatiotemporal forecasting.

---

[Quantitative Comparison of Credible Compilation and Verification In Coding Agent Compiler Development](http://arxiv.org/abs/2605.08927)

- Axon (Verified Compiler Framework): introduces a quantitative comparison between credible compilation and full verification approaches for compiler optimization development using an LLM-based coding agent.
- The study demonstrates that full verification requires approximately an order of magnitude more engineering effort than credible compilation, primarily due to the complexity of developing machine-checked proofs.
- Results indicate that while verified optimizations are more robust, they often suffer from performance overheads due to algorithm choices prioritized for provability, whereas credible compilation allows for more complex, performance-tuned optimizations.

---

[OPT-BENCH: Evaluating the Iterative Self-Optimization of LLM Agents in Large-Scale Search Spaces](http://arxiv.org/abs/2605.08904)

- OPT-Agent: introduces a framework that emulates human-like cognitive adaptation through a perception–memory–reasoning loop to iteratively refine solutions based on environmental feedback.
- The framework utilizes a structured workflow consisting of drafting, improving, and debugging actions to navigate complex search spaces in both continuous machine learning and discrete NP-hard domains.
- By integrating environmental feedback, the agent bridges the gap between initial hypotheses and optimal solutions, though performance remains constrained by the base capacity of the LLMs.

---

[ACE-SKILL: Bootstrapping Multimodal Agents with Prioritized and Clustered Evolution](http://arxiv.org/abs/2605.08887)

- ACE-SKILL: introduces a co-evolutionary framework that jointly optimizes rollout allocation and knowledge organization to enable self-evolving multimodal agents, utilizing a Prioritized Sampler and a Clustered Organizer.
- The framework employs a Prioritized Sampler to focus LLM rollout resources on informative samples using lazy-decay proficiency tracking, while a Clustered Organizer isolates knowledge into tactical and strategic components to mitigate interference.
- ACE-SKILL enables open-source LLMs to match proprietary models on multimodal benchmarks and supports zero-shot transfer of bootstrapped knowledge to smaller-scale models without additional training.

---

[On MMS, APS and XOS](http://arxiv.org/abs/2605.08859)

- Greedy-average allocation algorithm: introduces a randomized allocation framework for XOS agents that achieves an α-MMS approximation for α > 11/40 when the number of agents n is sufficiently large.
- The framework utilizes a potential function β to guide bundle selection and incorporates filtering and stealing procedures to ensure that remaining agents maintain sufficient value throughout the allocation process.
- The approach extends from identical to different XOS valuations by employing a doubling procedure to establish performance bounds for large n, effectively bypassing the 1/4-approximation barrier.

---

[When Agents Overtrust Environmental Evidence: An Extensible Agentic Framework for Benchmarking Evidence-Grounding Defects in LLM Agents](http://arxiv.org/abs/2605.08828)

- EnvTrustBench (Evidence-Grounding Defect Benchmarking Framework): introduces an extensible agentic framework for benchmarking evidence-grounding defects (EGDs) where LLMs treat unreliable environment-facing claims as authoritative ground for action.
- The framework generates concrete test cases from user-defined scenarios, including Task Scenario, Workspace, Environment, Task Objective, and Validation Oracle, to evaluate whether LLMs can distinguish true environment states from misleading adversarial claims.
- Evaluation across 14 LLM-scaffold stacks reveals that EGDs are a pervasive reliability problem, with most agents failing to verify environmental evidence before executing task-incorrect paths.

---

[Mirror, Mirror on the Wall: Can VLM Agents Tell Who They Are at All?](http://arxiv.org/abs/2605.08816)

- Mirror-guided self-identification benchmark: introduces a controlled 3D environment to evaluate whether embodied LLMs can infer hidden body attributes from mirror reflections to guide goal-directed behavior.
- The framework employs diagnostic interventions including mirror removal, misleading linguistic cues, and occluded reflections to distinguish grounded self-identification from shortcuts, priors, or confabulation.
- The study utilizes process-level metrics such as mirror consultation, temporal ordering, and self-attribution to assess whether LLM agents possess functional self-grounding rooted in perception and action.

---

[AgentSlimming: Towards Efficient and Cost-Aware Multi-Agent Systems](http://arxiv.org/abs/2605.08813)

- AgentSlimming: introduces a training-free framework that optimizes multi-agent workflows through structural pruning and semantic quantization to achieve cost-efficient agentic collaboration.
- The framework utilizes a hybrid importance evaluation mechanism, integrating topological priors and functional contributions via Reciprocal Rank Fusion (RRF) to identify and compress redundant agent nodes.
- AgentSlimming achieves significant token cost reductions of up to 78.9% across diverse benchmarks while maintaining or improving reasoning performance compared to unoptimized multi-agent systems.

---

[EvoMAS: Learning Execution-Time Workflows for Multi-Agent Systems](http://arxiv.org/abs/2605.08769)

- EvoMAS: introduces a framework for execution-time multi-agent workflow construction that dynamically adapts agent coordination based on an evolving task state.
- The system utilizes a Planner-Evaluator-Updater pipeline to construct explicit task states and a learned Workflow Adapter to instantiate stage-specific layered workflows from a fixed candidate agent pool.
- EvoMAS optimizes workflow selection via reinforcement learning using sparse terminal task success, outperforming static multi-agent design methods on complex, long-horizon tasks.

---

[UserGPT Technical Report](http://arxiv.org/abs/2605.08766)

- UserGPT: introduces a principled framework for enhancing LLM reasoning in persona understanding by distilling noisy behavioral histories into structured tags and narrative summaries.
- The framework utilizes a User Behavior Simulation Engine and Data-Centric Semantization to create high-fidelity training data, followed by a curriculum-driven post-training strategy to improve temporal reasoning and logical consistency.
- UserGPT achieves significant performance gains on the newly established HPR-Bench and enables efficient, incremental user profiling for next-generation AI agent memory.

---

[When LLMs Team Up: A Coordinated Attack Framework for Automated Cyber Intrusions](http://arxiv.org/abs/2605.08763)

- CAESAR (Coordinated Adversarial Execution and Strategic Reasoning): introduces a coordinated multi-agent framework that decomposes intrusion workflows into five specialized roles—Detective, Strategist, General, Executor, and Validator—to improve task success and reduce performance variance.
- The framework utilizes a round-based protocol with a persistent knowledge base and validator-gated memory to ensure information flow, artifact provenance, and budget-aware plan selection across heterogeneous LLM backends.
- Experimental results on CTF challenges and social-engineering scenarios demonstrate that CAESAR outperforms single-agent baselines by enabling structured, evidence-driven reasoning and adaptive behavior.

---

[Omni-DeepSearch: A Benchmark for Audio-Driven Omni-Modal Deep Search](http://arxiv.org/abs/2605.08762)

- Omni-DeepSearch: introduces a benchmark for audio-driven omni-modal deep search, requiring models to infer clues from audio and iteratively invoke text, image, and video search tools to perform multi-hop reasoning.
- The framework utilizes a multi-stage filtering pipeline to ensure that all tasks are audio-dependent, retrieval-demanding, and uniquely verifiable across 15 fine-grained categories.
- Experimental results demonstrate that while frontier models show progress, the task remains challenging due to bottlenecks in audio entity inference, query formulation, and cross-modal verification.

---

[Beyond the All-in-One Agent: Benchmarking Role-Specialized Multi-Agent Collaboration in Enterprise Workflows](http://arxiv.org/abs/2605.08761)

- ENTCOLLABBENCH: introduces a benchmark for evaluating role-specialized multi-agent collaboration in enterprise environments, utilizing Agent Layer, Enterprise Service Layer, Access Control, Data Isolation, Evaluation Pipeline, and Judgment Module.
- The framework simulates a permission-isolated organization where 11 LLM agents across six departments must perform cross-departmental delegation and stateful workflow execution.
- Evaluation is based on objective execution traces, database state verification, and deterministic policy adjudication rather than natural-language response judging.

---

[Omni-scale Learning-based Sequential Decision Framework for Order Fulfillment of Tote-handling Robotic Systems](http://arxiv.org/abs/2605.08758)

- OLSF-TRS: introduces a generalized, scalable framework that integrates structured combinatorial optimization with multi-agent reinforcement learning to coordinate order, tote, and robot decisions in warehouse fulfillment.
- The framework utilizes bisimulation quotienting to construct abstract Markov decision processes, enabling efficient policy learning across heterogeneous system configurations and scales.
- By employing a centralized MAPPO critic, the system effectively resolves inter-agent dependencies and mitigates congestion-induced inefficiencies in large-scale, high-concurrency warehouse environments.

---

[AHD Agent: Agentic Reinforcement Learning for Automatic Heuristic Design](http://arxiv.org/abs/2605.08756)

- AHD Agent: introduces a tool-integrated, multi-turn framework that empowers LLMs to proactively decide between heuristic generation and tool-based evidence retrieval for combinatorial optimization.
- The framework utilizes an agentic reinforcement learning system with an environment synthesis pipeline to train a compact 4B-parameter model in generalizable heuristic design.
- Experimental results demonstrate that the agent matches or surpasses state-of-the-art baselines using significantly fewer evaluations and exhibits strong generalization across diverse problem domains.

---

[Communicating Sound Through Natural Language](http://arxiv.org/abs/2605.08750)

- LAC (Lexical Acoustic Coding): introduces a framework that projects audio into interpretable acoustic descriptors, quantizes them into a lexical code, and transmits the sound as structured English prose between LLM agents.
- The system utilizes a shared vocabulary to map waveforms to a d-dimensional lexical code, which is then parsed by a receiver agent to reconstruct audio via a deterministic hybrid synthesizer.
- Closed-loop refinement iteratively adjusts renderer controls to ensure the synthesized waveform aligns with the transmitted lexical acoustic constraints.

---

[Done, But Not Sure: Disentangling World Completion from Self-Termination in Embodied Agents](http://arxiv.org/abs/2605.08747)

- VIGIL (Vision-based Interaction and Grounded Independent-judgment Logic): introduces an evaluation framework that decouples world-state completion from terminal commitment by requiring agents to issue semantically verifiable reports under a no-feedback contract.
- The framework utilizes Egocentric RGB Observation, Bounded Dialogue History, and a Native Action Space to isolate closure failures from execution traps across diagnostic and compositional task tiers.
- Empirical results across 20 models demonstrate that execution and terminal commitment are separable, with models exhibiting distinct failure profiles such as false reports and no-report exhaustion that aggregate metrics typically conflate.

---

[HULK: Large-scale Hierarchical Coordination under Continual and Uncertain Temporal Tasks](http://arxiv.org/abs/2605.08722)

- HULK: introduces a hierarchical coordination framework that decomposes global temporal missions into subtasks and assigns them to subteams using a receding-horizon approach to handle continual and uncertain task arrivals.
- The framework integrates a global task assignment layer with a local coordination layer, employing specific strategies for static, unknown, and dynamic task environments to ensure computational efficiency and robustness.
- HULK utilizes mixed-integer linear programming and coalition formation techniques to manage large-scale heterogeneous multi-agent systems under temporal constraints and environmental uncertainties.

---

[Breaking the Impasse: Dual-Scale Evolutionary Policy Training for Social Language Agents](http://arxiv.org/abs/2605.08721)

- DEPT: introduces a dual-timescale evolutionary perception mechanism that detects impasse by quantifying dual-scale value baseline divergence alongside match entropy.
- Upon perceiving training collapse, the framework activates asymmetric advantage reshaping to dynamically modulate the optimization landscape and restore gradient signals.
- The method effectively prevents policy degeneration in open-ended social language games by penalizing dominant outcomes and amplifying rare trajectories to enforce sustained strategic exploration.

---

[Debugging the Debuggers: Failure-Anchored Structured Recovery for Software Engineering Agents](http://arxiv.org/abs/2605.08717)

- PROBE (Failure-Anchored Structured Recovery for Software Engineering Agents): introduces a framework that transforms failed-run telemetry into structured evidence, structured diagnosis, and bounded recovery guidance to improve the recoverability of LLM-based software engineering agents.
- The framework utilizes a Telemetry Layer to preserve fine-grained runtime signals, a Diagnosis Layer to fuse evidence into a grounded diagnosis, and a Guidance Gate to ensure recovery instructions are actionable and bounded.
- Evaluations across repository-level repair, enterprise workflow, and AIOps settings demonstrate that PROBE significantly improves diagnosis accuracy and recovery rates compared to baseline methods by bridging the gap between failure identification and actionable recovery.

---

[AgentForesight: Online Auditing for Early Failure Prediction in Multi-Agent Systems](http://arxiv.org/abs/2605.08715)

- AgentForesight: introduces a deployment-time online auditing framework that monitors multi-agent systems step-by-step to detect and localize decisive errors before they propagate into full-trajectory failures.
- The framework utilizes AFTRAJ-2K, a curated corpus of safe and annotated unsafe trajectories, to train the AgentForesight-7B auditor using a coarse-to-fine reinforcement learning recipe.
- AgentForesight-7B employs a two-stage training process consisting of failure-boundary alignment via BPPO and three-axis verdict sharpening via GRPO to achieve precise step-level localization of failures.

---

[AgentPSO: Evolving Agent Reasoning Skill via Multi-agent Particle Swarm Optimization](http://arxiv.org/abs/2605.08704)

- AgentPSO: introduces a framework that evolves multi-agent reasoning skills by treating each agent as a particle in a swarm, iteratively refining natural-language instructions through personal-best, global-best, and self-reflective guidance.
- The framework utilizes an Independent Solving Agent, Peer Observation Module, Self-Reflective Direction Generator, PSO-like Velocity Updater, Skill Updater, Personal-Best Memory, and Global-Best Memory to improve reasoning without updating backbone LLM parameters.
- AgentPSO demonstrates that evolved reasoning skills are transferable across different benchmarks and backbone LLMs, achieving superior performance compared to static multi-agent debate methods while reducing test-time computational overhead.

---

[RewardHarness: Self-Evolving Agentic Post-Training](http://arxiv.org/abs/2605.08703)

- RewardHarness: introduces a self-evolving agentic reward framework that reframes reward modeling as context evolution by iteratively refining a library of Skills and Tools while keeping model weights fixed.
- The framework utilizes an Orchestrator to select relevant artifacts from the Library, which a frozen Sub-Agent then employs to construct interpretable reasoning chains for preference judgment.
- By analyzing reasoning successes and failures against ground-truth labels, the system automatically updates its library without additional human annotation, achieving high data efficiency and performance.

---

[SKILLMASTER: Toward Autonomous Skill Mastery in LLM Agents](http://arxiv.org/abs/2605.08693)

- SKILLMASTER: introduces a training framework that enables LLMs to autonomously develop, refine, and select reusable skills through tool-integrated reasoning and reinforcement learning.
- The framework utilizes a counterfactual utility reward to evaluate skill modifications on probe tasks, providing explicit signals for training skill-editing decisions.
- DualAdv-GRPO decouples advantage normalization for task-solving actions and skill-management decisions, ensuring stable joint optimization within a unified policy.

---

[PrepBench: How Far Are We from Natural-Language-Driven Data Preparation?](http://arxiv.org/abs/2605.08687)

- PrepBench: introduces a benchmark for evaluating NL-driven data preparation across three core capabilities: interactive disambiguation, prep-code generation, and code-to-workflow translation.
- The framework utilizes an agent-based pipeline to construct ground-truth assets, including a Disambiguation Knowledge Base and executable workflows, to systematically measure LLM performance on complex, multi-step data preparation tasks.
- Experimental results demonstrate that while LLMs show promise, they struggle with ambiguity resolution, handling data irregularities, and generating structurally valid workflows, highlighting significant gaps in current end-to-end NL-driven data preparation systems.

---

[Iterative Critique-and-Routing Controller for Multi-Agent Systems with Heterogeneous LLMs](http://arxiv.org/abs/2605.08686)

- Iterative Critique-and-Routing Controller: introduces a multi-turn coordination framework that models agent interaction as a finite-horizon MDP, utilizing a Controller (LLM-based router and critic) to evaluate intermediate drafts and selectively invoke agents from an Agent Pool (heterogeneous LLMs with varying capabilities).
- The framework employs a Composite Reward (rule-based metric for routing and verification) and Lagrangian Relaxation (optimization technique for constrained agent utilization) to balance answer quality against computational costs.
- Training is performed via Group Relative Policy Optimization (GRPO) (RL algorithm for training the controller), enabling the system to outperform one-shot routing baselines while maintaining strict usage constraints on stronger, more expensive models.

---

[MLS-Bench: A Holistic and Rigorous Assessment of AI Systems on Building Better AI](http://arxiv.org/abs/2605.08678)

- MLS-Bench: introduces a benchmark for evaluating whether AI systems can invent generalizable and scalable ML methods, with Task Specification, Validity Enforcement, Unified Scoring, Agent Harness, Editable Codebase, Protected Codebase, Baseline Implementations, Evaluation Settings, Compute Budget, and Seed Policy.
- The framework evaluates LLM agents on 140 tasks across 12 domains, requiring agents to improve targeted ML components while maintaining performance across controlled settings.
- The benchmark exposes a significant method-discovery gap, finding that current LLMs are more proficient at engineering-style tuning than at genuine scientific invention.

---

#### 8th May 2026

[LLMs Improving LLMs: Agentic Discovery for Test-Time Scaling](http://arxiv.org/abs/2605.08083)

- AutoTTS: introduces an environment-driven framework that automates the discovery of test-time scaling strategies by shifting human effort from manual heuristic design to constructing replayable discovery environments.
- The framework utilizes an explorer LLM to iteratively propose and refine controllers, which are evaluated against pre-collected reasoning trajectories to ensure efficient, deterministic, and scalable search.
- By employing beta parameterization and fine-grained execution trace feedback, the system effectively navigates the high-dimensional control space to discover robust strategies that generalize across diverse models and benchmarks.

---

[The Memory Curse: How Expanded Recall Erodes Cooperative Intent in LLM Agents](http://arxiv.org/abs/2605.08060)

- The Memory Curse: introduces a systematic study of how expanded interaction history in LLM agents paradoxically degrades cooperation in multi-agent social dilemmas due to a cognitive vulnerability to accumulated negative content.
- The research identifies that while short-term memory facilitates trust repair, extended memory often triggers defensive, history-following reasoning patterns that override cooperative priors.
- The authors demonstrate that this "memory curse" can be mitigated by targeted fine-tuning to promote forward-looking reasoning or by sanitizing memory content, proving the phenomenon is driven by reasoning style rather than context length.

---


[A Roadmap of Mixed Reality Body Doubling for Adults with ADHD](http://arxiv.org/abs/2605.07851)

- Body Doubling Framework: introduces a multidimensional model for designing and evaluating body doubling setups to support adults with ADHD through Individual Motivation, Agent-Related Dimensions, Interaction-Related Dimensions, Contextual Dimensions, and Efficacy of the Approach.
- The framework categorizes body doubling setups using a roadmap that accounts for agent characteristics, interaction dynamics, and environmental context to identify research gaps in mixed reality and interactive systems.
- This research provides a structured approach for future empirical studies and the development of personalized, technology-mediated body doubling interventions for neurodivergent individuals.

---



[Reason to Play: Behavioral and Brain Alignment Between Frontier LRMs and Human Game Learners](http://arxiv.org/abs/2605.08019)

- LRMs: introduces a comparative study evaluating frontier Large Reasoning Models against human game learners and traditional reinforcement learning agents using VGDL-fMRI datasets.
- The research demonstrates that frontier LRMs exhibit behavioral patterns and brain-encoding representations that align significantly closer to human game-learning than deep reinforcement learning baselines.
- The study establishes that LRM representations predict human BOLD activity an order of magnitude better than RL alternatives, suggesting these models capture transferable, human-like in-context learning dynamics.

---

[Collaborator or Assistant? How AI Coding Agents Partition Work Across Pull Request Lifecycles](http://arxiv.org/abs/2605.08017)

- Collaborator-Assistant Spectrum: introduces a process-centric framework to analyze how AI coding agents partition operational agency and governance authority across pull request lifecycles.
- The framework classifies tools into Collaborator or Assistant paradigms based on initiation patterns and review-routing behaviors observed in 29,585 pull request lifecycles.
- It utilizes an Initiator × Approver taxonomy to disentangle operational work from merge governance, revealing that while agents frequently initiate tasks, merge authority remains predominantly human.

---

[Learning CLI Agents with Structured Action Credit under Selective Observation](http://arxiv.org/abs/2605.08013)

- A3 (Action Advantage Assignment): introduces a reinforcement learning paradigm for CLI agents that addresses partial workspace observation and sparse action credit through structured action analysis.
- The framework utilizes σ-Reveal to construct token-budgeted workspace views and employs a three-channel advantage mechanism—episode, turn, and tree scope—to assign credit to shell actions based on their structural intent.
- The approach is evaluated on the ShellOps dataset, demonstrating improved performance in complex filesystem interaction tasks by leveraging native shell syntax for credit assignment at a computational cost comparable to standard agentic RL.

---

[Interpreting Reinforcement Learning Agents with Susceptibilities](http://arxiv.org/abs/2605.08007)

- Susceptibility Framework: introduces a method for neural network interpretability that studies the response of posterior expectation values of observables to perturbations of the regret landscape, utilizing Conv 1-3, FF 1, FF 2, Policy, SGLD, and Regret Landscape.
- The framework generalizes susceptibilities to reinforcement learning by measuring how initial state distribution perturbations affect the effective complexity of specific architectural components.
- Empirical validation through activation-steering and direction-conditioned posterior regret confirms that susceptibility-detected parameter-space structures correspond to functionally meaningful internal agent computations.

---

[Tool Calling is Linearly Readable and Steerable in Language Models](http://arxiv.org/abs/2605.07990)

- Tool Calling is Linearly Readable and Steerable in Language Models: introduces a method for steering LLM tool selection by adding a mean-difference vector to the residual stream, which redirects tool choice and automatically adapts argument schemas.
- The research demonstrates that tool identity is linearly encoded within a low-dimensional subspace of the model's internal activations, allowing for precise intervention without weight updates.
- Mechanistic analysis reveals that tool selection is driven by a three-stage circuit involving early-layer features, mid-layer attention heads, and late-layer formatting, with instruction tuning essential for routing these internal signals to structured output.

---

[Graph Representation Learning Augmented Model Manipulation on Federated Fine-Tuning of LLMs](http://arxiv.org/abs/2605.07961)

- AugMP: introduces an adversarial framework that leverages graph representation learning to synthesize malicious updates that disrupt the federated fine-tuning of LLMs while evading detection.
- The framework utilizes a VGAE to capture feature correlations among benign updates and an iterative manipulation algorithm based on an augmented Lagrangian dual formulation to enforce geometric stealth constraints.
- Experimental results demonstrate that AugMP significantly degrades the accuracy of global and local LLMs across multiple backbones while maintaining statistical and geometric consistency with benign updates.

---

[Ask Early, Ask Late, Ask Right: When Does Clarification Timing Matter for Long-Horizon Agents?](http://arxiv.org/abs/2605.07937)

- Forced-injection framework: introduces a methodology to measure the value of information (VOI) by injecting ground-truth clarifications at controlled trajectory points across diverse long-horizon agent tasks.
- The study demonstrates that clarification timing is highly dimension-dependent, with goal information requiring early intervention and input information remaining recoverable through mid-trajectory.
- Empirical results reveal that current LLMs fail to naturally ask within these optimal timing windows, establishing a quantitative foundation for designing future timing-aware clarification policies.

---

[TraceFix: Repairing Agent Coordination Protocols with TLA+ Counterexamples](http://arxiv.org/abs/2605.07935)

- TraceFix: introduces a verification-first pipeline for LLM multi-agent coordination that uses counterexample-driven repair to ensure protocol correctness.
- The framework utilizes an Orchestration Agent to synthesize a Protocol Topology IR and PlusCal Coordination Logic, which are then validated by the TLC Model Checker to eliminate concurrency hazards.
- Verified protocols are compiled by the Prompt Compiler into per-agent prompts and enforced at runtime by a Topology Monitor to prevent out-of-protocol coordination operations.

---

[AgentEscapeBench: Evaluating Out-of-Domain Tool-Grounded Reasoning in LLM Agents](http://arxiv.org/abs/2605.07926)

- AgentEscapeBench: introduces an escape-room-style benchmark that evaluates the ability of LLMs to perform tool-grounded reasoning in unfamiliar environments using Template Library, DAG Skeleton Generator, Source Annotator, Value Instantiator, Deterministic Forward Executor, Narrative Generator, and Evaluation Sandbox.
- The framework utilizes a six-stage automated pipeline to construct directed acyclic graph tasks that require agents to infer, execute, and revise tool-use procedures under explicit long-range dependency constraints.
- Experimental results across sixteen LLMs demonstrate that performance degrades monotonically with dependency depth, highlighting significant bottlenecks in long-range state tracking and intermediate-result propagation.

---

[One World, Dual Timeline: Decoupled Spatio-Temporal Gaussian Scene Graph for 4D Cooperative Driving Reconstruction](http://arxiv.org/abs/2605.07910)

- DUST (DecoUpled Spatio-Temporal) Gaussian Scene Graph: introduces a cooperative reconstruction framework that decouples pose trajectories into source-specific timelines to eliminate gradient conflicts caused by temporal asynchrony in multi-source driving data.
- The framework utilizes a shared canonical Gaussian representation for appearance consistency while maintaining independent pose timelines for vehicle and infrastructure sensors to enable high-fidelity 4D scene reconstruction.
- DUST incorporates a static anchor-based pose correction pipeline and a pose-regularized joint optimization scheme to ensure robust initialization and stable training under asynchronous conditions.

---

[ADKO: Agentic Decentralized Knowledge Optimization](http://arxiv.org/abs/2605.07863)

- ADKO (Agentic Decentralized Knowledge Optimization): introduces a modular framework for collaborative black-box optimization that enables autonomous agents to share semantically rich insights via Private Gaussian Process (GP) surrogate, Knowledge token, Graph-structured communication, LM reasoning module, Fidelity-aware token pruning, and Reasoning score without exposing raw data.
- The framework utilizes a Reasoning score that synthesizes private GP-based exploitation and exploration with collaborative success-attraction and failure-avoidance signals derived from peer-shared Knowledge tokens.
- ADKO provides a rigorous theoretical foundation for decentralized optimization by decomposing cumulative regret into GP error, LM bias, LM noise, and compression loss, while validating performance through neural architecture search and scientific discovery experiments.

---

[VISTA: Decentralized Machine Learning in Adversary Dominated Environments](http://arxiv.org/abs/2605.07841)

- VISTA: introduces an adaptive decentralized learning framework that dynamically tunes acceptance thresholds and learning rates to ensure convergence in adversary-dominated environments.
- The framework utilizes a consistency-based acceptance rule to incentivize rational worker nodes to provide reliable gradient reports while mitigating adversarial noise.
- VISTA achieves asymptotic convergence rates matching standard SGD by balancing update frequency and estimation accuracy through an adaptive signal-to-noise tolerance parameter.

---

[RELAGENT: LLM Agents as Data Scientists for Relational Learning](http://arxiv.org/abs/2605.07840)

- RELAGENT: introduces an agentic framework that automates relational learning by iteratively searching for optimal SQL feature programs and predictive models using an LLM agent interacting with a database via specialized tools.
- The framework decouples the search phase, where an LLM agent refines feature programs and model configurations, from the inference phase, which executes the final SQL-based predictor without further LLM calls.
- By maintaining relational structure externally and using executable SQL queries, the approach achieves intrinsic interpretability and relational invariance while matching or exceeding the performance of fixed-architecture relational models.

---

[Unsafe by Flow: Uncovering Bidirectional Data-Flow Risks in MCP Ecosystem](http://arxiv.org/abs/2605.07836)

- MCP-BiFlow (Model Context Protocol Bidirectional Data-Flow Analysis Framework): introduces a static analysis framework that detects bidirectional data-flow vulnerabilities in Model Context Protocol servers by combining MCP Entrypoint Recovery, MCP-Specific Taint Specification, and Bidirectional Interprocedural Taint Analysis.
- The framework addresses request-side propagation, where requester-controlled inputs reach sensitive operations, and return-side propagation, where untrusted external content influences downstream LLM reasoning or tool invocations.
- Evaluated on 32 confirmed vulnerabilities, the framework achieves 93.8% recall and identifies 118 unique vulnerability paths across 87 real-world Model Context Protocol server repositories.

---

[Many-to-Many Multi-Agent Pickup and Delivery](http://arxiv.org/abs/2605.07835)

- M2M: introduces a sequential many-to-many MAPD algorithm that optimizes task allocation across agents, tasks, and multiple potential pickup and delivery locations using Initial Task Allocation, Large Neighborhood Search, and Priority Based Search.
- The framework addresses the NP-hard 4-dimensional assignment problem by decomposing cost tensors into matrices to enable efficient computation of task allocations in automated warehouse environments.
- Experimental results demonstrate that M2M consistently outperforms existing state-of-the-art methods in task throughput and maintains scalability for up to 150 agents and tasks within a 1-second computation budget.

---

[CyBiasBench: Benchmarking Bias in LLM Agents for Cyber-Attack Scenarios](http://arxiv.org/abs/2605.07830)

- CyBiasBench: introduces a comprehensive benchmarking framework to quantify attack-selection bias in LLM agents across diverse cyber-attack scenarios using Prompt Design, Agent Penetration Testbed, and Evaluation Metrics Suite.
- The framework utilizes a Kali Linux Container for isolated execution against Target Applications, employing a Deterministic Classifier and Verifier Pipeline to measure agent behavior independently of self-reported logs.
- The research reveals that LLM agents exhibit persistent, agent-specific attack-selection biases that remain stable across prompt variations and are not consistently improved by explicit steering, a phenomenon termed bias momentum.

---

[SCENE: Recognizing Social Norms and Sanctioning in Group Chats](http://arxiv.org/abs/2605.07823)

- SCENE: introduces a benchmark for evaluating how LLMs infer and adapt to implicit social norms within multi-party group chats through interaction with scripted personas.
- The framework utilizes a two-stage generation process to create synthetic, interactionally plausible scenarios where a subject agent must recognize sanctioning signals and adjust its behavior accordingly.
- Evaluation metrics include repair rates after sanctioning, breach rates based on prior demonstrations, and compliance under opposing norm pairs to assess the social adaptation capabilities of various LLMs.

---

[GazeVLM: Active Vision via Internal Attention Control for Multimodal Reasoning](http://arxiv.org/abs/2605.07817)

- GazeVLM: introduces a multimodal architecture that internalizes active vision by dynamically modulating the causal attention mask of a VLM using <LOOK> tokens to focus on task-relevant visual regions.
- The framework employs a two-stage training process, starting with supervised fine-tuning on curated gaze-reasoning traces, followed by reinforcement learning using GRPO to optimize autonomous visual navigation.
- By applying a continuous suppression bias to irrelevant visual tokens, GazeVLM achieves high-resolution multimodal reasoning without the computational overhead of external cropping tools or context window inflation.

---

[CktFormalizer: Autoformalization of Natural Language into Circuit Representations](http://arxiv.org/abs/2605.07782)

- CktFormalizer: introduces a framework that redirects LLM-driven hardware generation through a dependently-typed HDL embedded in Lean 4 to ensure structural correctness and enable formal verification.
- The framework utilizes an LLM Coding Agent that leverages immediate compile-time feedback from the Lean type system to iteratively refine hardware designs, effectively bridging the gap between natural language specifications and synthesizable silicon.
- CktFormalizer incorporates a closed-loop PPA optimization strategy and automated theorem proving to produce formally verified, optimized hardware implementations that are robust against backend synthesis and routing failures.

---

[Coding Agents Don’t Know When to Act](http://arxiv.org/abs/2605.07769)

- FIXEDBENCH: introduces a benchmark to evaluate whether coding agents can recognize when software issues are already resolved and abstain from making unnecessary code changes.
- The study demonstrates that LLMs exhibit an action bias, frequently applying spurious edits to already-fixed code, which can be mitigated by explicit instructions to verify the issue before acting.
- The research highlights that while verify-then-abstain prompting improves abstention rates, it introduces a risk of over-abstention on tasks that genuinely require code modifications.

---

[Emergence of Social Reality of Emotion through a Social Allostasis Model with Dynamic Interpretants](http://arxiv.org/abs/2605.07761)

- Social Allostasis Model: introduces a computational framework where two agents co-construct social reality by aligning bodily control goals and symbol interpretations through active inference and the Metropolis–Hastings Naming Game.
- The framework utilizes POMDP-based generative models to enable agents to perform allostatic regulation of bodily states while simultaneously negotiating shared emotional concepts via symbolic communication.
- Experimental results demonstrate that agents successfully converge on shared prior preferences and dynamic symbol interpretations, confirming the emergence of social reality from the interplay of individual bodily regulation and social interaction.

---

[Alternating Target–Path Planning for Scalable Multi-Agent Coordination](http://arxiv.org/abs/2605.07744)

- TAPF Framework: introduces an iterative refinement approach that decouples target assignment from pathfinding to achieve scalability in multi-agent coordination by leveraging Initial Assignment, MAPF Solver, Feedback Mechanism, Reassignment Strategy, and Final Path Optimization.
- The framework utilizes feedback-driven reassignment loops to identify bottleneck agents via Delay-Based Selection or Spectral Bottleneck Sampling, subsequently refining assignments using PIBT or Local Hungarian methods.
- Empirical results demonstrate that the framework scales to thousands of agents, significantly outperforming traditional Conflict-Based Search methods in large-scale logistics scenarios.

---

[Securing the Dark Matter: A Semantic-Enhanced Neuro-Symbolic Framework for Supply Chain Analysis of Opaque Industrial Software](http://arxiv.org/abs/2605.07737)

- SCAA (Supply Chain Analysis Agent): introduces a neuro-symbolic framework that reconstructs behavioral semantics from opaque industrial binaries to perform tractable global risk reasoning.
- The framework utilizes a Reflexive Prompting pipeline to constrain a local LLM agent with structural verification, effectively suppressing hallucinations during semantic lifting.
- A domain-adapted Graphormer and surjective graph transformation enable scalable vulnerability detection and APT fingerprinting across complex industrial supply chains.

---

[SARC: A Governance-by-Architecture Framework for Agentic AI Systems](http://arxiv.org/abs/2605.07728)

- SARC: introduces a governance-by-architecture framework that treats constraints as first-class specification objects, comprising S, A, R, and C, enforced at PAG, ATM, PAA, and ER.
- The framework enables auditable-by-construction agentic systems by compiling regulatory obligations into runtime checks that maintain system invariants.
- SARC provides a specification-to-runtime compilation discipline that separates optimization from admissibility, ensuring compliance is verifiable through structured trace derivation.

---

[SOD: Step-wise On-policy Distillation for Small Language Model Agents](http://arxiv.org/abs/2605.07725)

- SOD: introduces a step-wise on-policy distillation framework that adaptively reweights distillation strength based on student-teacher divergence to stabilize training for small language model agents.
- The framework combines GRPO for trajectory-level exploration with a step-wise OPD objective that attenuates misleading teacher signals in high-divergence regions caused by tool-induced state drift.
- Experimental results demonstrate that SOD significantly improves agentic reasoning performance and training stability across math, science, and code benchmarks compared to standard distillation and reinforcement learning baselines.

---

[The AI-Native Large-Scale Agile Software Development Manifesto](http://arxiv.org/abs/2605.07717)

- AI-Native Large-Scale Agile Software Development Framework: introduces a paradigm shift in software engineering by replacing sequential human-centric processes with an intelligent, adaptive system where Human Experts define intent and AI Agents execute tasks.
- The framework utilizes AI Personas, AI Agents, and Skills to enable parallel development, while a Semantic Layer and Knowledge Graphs provide a shared, living context for both humans and agents.
- By integrating MCP Connectors for tool interaction and emphasizing reusable blueprints, the approach facilitates continuous verification and autonomous collaboration across large-scale development teams.

---

[DRIP-R: A Benchmark for Decision-Making and Reasoning Under Real-World Policy Ambiguity in the Retail Domain](http://arxiv.org/abs/2605.07699)

- DRIP-R: introduces a benchmark for evaluating LLM agents in retail scenarios governed by ambiguous policies, utilizing a simulation environment with Scenario Construction, Simulation Environment, Evaluation Pipeline, Orchestrator, LLM-as-a-Judge, Customer Agent, User Simulator, Database, and Tool-calling Agent.
- The framework employs a multi-judge evaluation pipeline to assess agent performance across policy adherence, dialogue quality, behavioral alignment, interest alignment, and task-resolution adherence.
- The benchmark systematically exploits real-world policy ambiguities to test how LLMs interpret conflicting rules, justify decisions, and balance stakeholder interests in conversational settings.

---

[GASim: A Graph-Accelerated Hybrid Framework for Social Simulation](http://arxiv.org/abs/2605.07692)

- GASim: introduces a graph-accelerated hybrid framework for large-scale social simulations that utilizes EDG, GOM, and GMP to optimize agent-based modeling performance.
- The framework employs EDG to dynamically partition agents into LLM-based core agents and numerical ordinary agents, significantly reducing computational latency.
- GOM replaces intensive LLM-based retrieval with lightweight graph propagation, while GMP leverages a GAT to enable parallel opinion updates for ordinary agents.

---

[A Multi-Level Agent-Based Architecture for Climate Governance Integrating Cognitive and Institutional Dynamics](http://arxiv.org/abs/2605.07683)

- Multi-level Agent-Based Architecture: introduces a modular simulation framework that integrates micro-level HUMAT-MOA behavioural decision-making, meso-level social influence networks, and macro-level institutional strategy to model democratic climate governance.
- The architecture utilizes a land-use development proposal as a central interaction artefact, enabling the co-evolution of public opinion, organized advocacy, and formal political decision-making.
- By combining empirically grounded cognitive models with strategic institutional rules, the framework provides a transparent and portable approach for simulating complex socio-political dynamics in climate governance.

---

[The Endogeneity of Miscalibration: Impossibility and Escape in Scored Reporting](http://arxiv.org/abs/2605.07671)

- Credibility Game framework: introduces a formal model of strategic reporting where an agent's combined objective of accuracy and non-accuracy benefits leads to an endogenous impossibility of truthful reporting.
- The framework demonstrates that a principal's optimal oversight mechanism is necessarily non-affine, which triggers a structural impossibility that undermines calibration.
- The research establishes that a sharp step-function approval threshold serves as a constructive escape, achieving first-best screening for any strictly proper scoring rule.

---

[Operating Within the Operational Design Domain: Zero-Shot Perception with Vision-Language Models](http://arxiv.org/abs/2605.07649)

- ODD-TAX-232 (Operational Design Domain Taxonomy 232): introduces a framework for zero-shot perception in autonomous driving that utilizes specialized agents including Traffic Signage Expert, Road Marking Expert, Scenery Element Expert, Weather Condition Expert, and Trigger Condition Expert, integrated via a Chained Prompting Pipeline and Knowledge Base within a VLM-based Perception System.
- The research evaluates the effectiveness of various VLMs in identifying fine-grained ODD elements by employing persona-based prompting and structured reasoning to enhance zero-shot classification and detection performance.
- The study demonstrates that while VLMs show potential for offline ODD auditing and compliance reporting, their current performance requires further refinement for safety-critical, real-time on-board deployment.

---

[MAVEN: Multi-Agent Verification-Elaboration Network with In-Step Epistemic Auditing](http://arxiv.org/abs/2605.07646)

- MAVEN: introduces a blackboard-inspired multi-agent framework that decouples factual probing from synthesis to transform LLMs into deliberate, auditable reasoners.
- The framework utilizes an adversarial Skeptic-Researcher-Judge loop to enforce discrete verification gates and parametric probing, ensuring that reasoning trajectories are factually grounded and logically coherent.
- By employing an Adaptive Router and a persistent Knowledge Cache, MAVEN optimizes computational efficiency while maintaining structural rigor across diverse LLM backbones.

---

[Safe, or Simply Incapable? Rethinking Safety Evaluation for Phone-Use Agents](http://arxiv.org/abs/2605.07630)

- PHONESAFETY: introduces a benchmark of 700 safety-critical moments to distinguish between safe judgment and inability to act in LLM-based phone-use agents.
- The framework categorizes agent responses into safe actions, unsafe actions, and failures to do anything useful to prevent misinterpreting harmless outcomes as evidence of safety.
- Empirical results demonstrate that general phone-use capability does not reliably predict safe choices, and that failures to act often stem from operational limitations rather than safety-conscious judgment.

---

[MemCompiler: Compile, Don’t Inject — State-Conditioned Memory for Embodied Agents](http://arxiv.org/abs/2605.07594)

- MemCompiler: introduces a state-conditioned memory compilation paradigm that dynamically selects and compiles relevant experience into dual-channel guidance for embodied agents.
- The framework utilizes a Memory Compiler to maintain a structured Brief State and deliver targeted text and latent Soft-Mem tokens to the Executor, preventing attention dilution.
- MemCompiler significantly improves performance across multiple embodied benchmarks while reducing executor input tokens and latency by optimizing memory delivery for lightweight LLMs.

---

[Deadline-Driven Hierarchical Agentic Resource Sharing for AI Services and RAN Functions in AI-RAN](http://arxiv.org/abs/2605.07547)

- HAF (Hierarchical Agentic Framework): introduces a two-layer architecture for AI-RAN that decouples slow-timescale service placement from fast-timescale GPU/CPU allocation to optimize SLO fulfillment.
- The framework utilizes an LLM-based agent for intelligent migration decision-making, supported by a predictive critic to mitigate service interruption costs.
- A closed-form convex allocator ensures hard real-time RAN deadline satisfaction while dynamically distributing remaining compute resources to elastic AI services.

---

[Multi-Environment POMDPs with Finite-Horizon Objectives](http://arxiv.org/abs/2605.07537)

- MEPOMDP: introduces a robust framework for sequential decision-making under uncertainty with multiple environments and finite-horizon objectives, utilizing Multi-belief, Multi-expected payoff, and Mixture of policies to compute optimal values.
- The paper establishes that the value computation problem for MEPOMDPs is PSPACE-complete, providing both a space-efficient deterministic algorithm and a practically efficient algorithm based on bottom-up frontier construction with pruning.
- Empirical evaluations demonstrate that the proposed efficient pruning algorithm significantly outperforms existing state-of-the-art tools across classical benchmarks like Rock Sample, Robot Navigation, and Identification.

---

[Synchronizing Minds through Collective Predictive Coding: A Computational Model of Parent-Infant Homeostatic Co-Regulation](http://arxiv.org/abs/2605.07524)

- MHNG: introduces a computational model of parent–infant homeostatic co-regulation that integrates POMDP (Individual-level active interoceptive inference) with MHNG (Decentralized Bayesian communication mechanism) and CPC (Social-level predictive coding hypothesis) to achieve latent-state alignment.
- The framework utilizes Generative Matrices (Learned internal model parameters) and a Communicative Variable (Shared interactional symbol) to enable agents with asymmetric knowledge to coordinate regulatory actions.
- Simulation results demonstrate that MHNG-mediated interaction facilitates rapid synchronization of Internal Representations (Agent latent belief states) and improves visceral state regulation compared to one-sided control conditions.

---

[InterLV-Search: Benchmarking Interleaved Multimodal Agentic Search](http://arxiv.org/abs/2605.07510)

- InterLV-Search: introduces a benchmark for interleaved multimodal agentic search, which evaluates how LLMs use visual evidence as search pivots to guide multi-hop trajectories across three progressively challenging levels.
- The framework includes an InterLV-Agent that utilizes a reason-act-observe loop, supported by Short-term Memory, Long-term Memory, and Tool Integration to manage complex, non-linear search tasks.
- Experimental results demonstrate that current LLMs struggle with interleaved multimodal search, highlighting the necessity of active visual evidence seeking and robust search control beyond simple text-centric browsing.

---

[MASPrism: Lightweight Failure Attribution for Multi-Agent Systems Using Prefill-Stage Signals](http://arxiv.org/abs/2605.07509)

- MASPrism: introduces a two-stage failure attribution framework that leverages prefill-stage signals from an SLM to identify root causes in multi-agent execution traces without requiring decoding, replay, or task-specific training.
- The framework utilizes a Filtering stage to identify symptom steps via NLL and candidate sources via attention, followed by a Diagnosis stage that reconstructs a focused prompt to perform fine-grained ranking of failure sources.
- By operating exclusively on prefill-stage internal signals, MASPrism achieves significant speedups and cost reductions compared to agent-based or replay-based attribution methods while maintaining high accuracy on long-trace benchmarks.

---

[LiteGUI: Distilling Compact GUI Agents with Reinforcement Learning](http://arxiv.org/abs/2605.07505)

- LiteGUI: introduces a SFT-free training paradigm for lightweight GUI agents by integrating Guided On-policy Distillation and Multi-solution Dual-level GRPO.
- The framework utilizes an automated trajectory generation pipeline to create multi-solution datasets, which are then used to train agents via privileged teacher guidance and dual-level reinforcement learning rewards.
- LiteGUI achieves state-of-the-art performance among lightweight models by jointly aligning macro-level subtask planning with micro-level execution matching, effectively mitigating exploration bottlenecks in long-horizon GUI tasks.

---

[HBEE: Human Behavioral Entropy Engine Pre-Registered Multi-Agent LLM Simulation of Peer-Suspicion-Based Detection Inversion](http://arxiv.org/abs/2605.07472)

- HBEE: introduces a pre-registered multi-agent LLM simulation framework to evaluate the effectiveness of insider threat detection mechanisms against adaptive adversaries.
- The framework utilizes LLM-driven agents to model organizational communication and tests how adaptive OPSEC directives influence detection signals in both cascade and blind defender configurations.
- Empirical results demonstrate a detection inversion where an adaptive mole becomes less suspicious than innocent peers in a cascade-based detection environment, revealing a decoupling of suspicion metrics.

---

[The Moltbook Files: A Harmless Slopocalypse or Humanity’s Last Experiment](http://arxiv.org/abs/2605.07462)

- Moltbook Files: introduces a dataset of 232k posts and 2.2M comments from an AI-agent-populated platform, processed through a Collection Pipeline, Preprocessing Pipeline, and PII Anonymization Pipeline.
- The research utilizes BERTopic Modeling to characterize agent discourse and Fine-tuning Configuration to assess the impact of synthetic data on LLM factuality and alignment.
- The study employs an LLM-as-a-judge Evaluator to demonstrate that while training on agent-generated content degrades model performance, the effect is comparable to fine-tuning on human-generated Reddit data.

---

[EditRefiner: A Human-Aligned Agentic Framework for Image Editing Refinement](http://arxiv.org/abs/2605.07457)

- EditRefiner: introduces a hierarchical, interpretable, and human-aligned agentic framework that reformulates post-editing correction as a perception-reasoning-action-evaluation loop.
- The framework utilizes a Perception Agent to identify flaws, a Reasoning Agent to diagnose them, an Action Agent to perform targeted re-editing, and an Evaluation Agent to ensure alignment with human preferences.
- The system is supported by the EditFHF-15K dataset, which provides extensive human-annotated feedback, including saliency maps and MOS scores, to facilitate the development of self-corrective image editing agents.

---

[Sparse Autoencoders as Plug-and-Play Firewalls for Adversarial Attack Detection in VLMs](http://arxiv.org/abs/2605.07447)

- SAEgis: introduces a lightweight adversarial attack detection framework by inserting sparse autoencoders into pretrained VLMs to identify attack-relevant signals through reconstruction-based latent feature analysis.
- The framework utilizes a difference-of-means approach on sparse latent features to detect adversarial perturbations without requiring additional adversarial training.
- By ensembling sparse autoencoder signals across multiple layers, the system achieves robust detection performance across diverse domains and unseen adversarial attack methods.

---

[GameGen-Verifier: Parallel Keypoint-Based Verification for LLM-Generated Games via Runtime State Injection](http://arxiv.org/abs/2605.07442)

- GameGen-Verifier: introduces an automated verification paradigm for LLM-generated games that decomposes specifications into verifiable keypoints and grounds them into independent verification units for parallel execution.
- The framework utilizes parameter-based state construction and runtime state patching to instantiate specific game states, bypassing the need for long-horizon gameplay exploration.
- GGV-HARNESS provides a two-layer architecture for concurrency management, runtime isolation, and fault recovery, significantly improving verification accuracy and reducing wall-clock time compared to traditional agent-based approaches.

---

[Low-code and No-code with BESSER to Create and Deploy Smart Web Applications](http://arxiv.org/abs/2605.07376)

- BESSER: introduces an open-source low-code framework that enables the design, generation, and deployment of smart web applications through B-UML, Structural Perspective, Agent Perspective, GUI Perspective, Code Generation Pipeline, Backend Generator, Frontend Generator, BAF Generator, and Deployment Service.
- The framework utilizes a model-driven approach to automate the creation of full-stack web applications, including backend APIs, React-based frontends, and LLM-integrated agent backends.
- BESSER streamlines the development lifecycle by integrating directly with cloud platforms like GitHub and Render to provide one-click deployment of generated smart web applications.

---

[FlightSense: An End-to-End MLOps Platform for Real-Time Flight Delay Prediction via Rotation-Chain Propagation Features and Agentic Conversational AI](http://arxiv.org/abs/2605.07364)

- FlightSense: introduces an end-to-end MLOps platform for real-time flight delay prediction that utilizes an XGBoost Classifier, AWS Lambda, Amazon SageMaker, a Streamlit Dashboard, an Amazon Bedrock Nova Micro agent, and a Tool-Use Architecture.
- The framework employs a progressive three-version feature engineering approach, prioritizing rotation-chain propagation features to achieve high predictive accuracy.
- The system integrates an agentic LLM to provide grounded, natural-language delay risk assessments through multiplicative probability compounding based on live weather and operational data.

---

[A Comprehensive Survey on Agent Skills: Taxonomy, Techniques, and Applications](http://arxiv.org/abs/2605.07358)

- Agent Skills Survey: introduces a comprehensive taxonomy for LLM-based agent systems, focusing on the lifecycle of reusable procedural artifacts that bridge the gap between raw tool access and robust task execution.
- The framework categorizes agent systems into five core modules: Skill Representation, Skill Acquisition, Skill Retrieval and Selection, Skill Evolution, and Runtime Governance.
- This survey synthesizes current research on how LLMs can externalize, manage, and refine procedural knowledge to improve the scalability, robustness, and maintainability of autonomous agent ecosystems.

---

[Tools as Continuous Flow for Evolving Agentic Reasoning](http://arxiv.org/abs/2605.07339)

- FlowAgent: introduces a paradigm that reconceptualizes discrete tool chaining as continuous trajectory generation within a semantic space to enable global plan-level reasoning.
- The framework utilizes a Conditional Flow Planner to generate latent trajectories, which are mapped to discrete actions via Discrete Plan Decoding and a Parameterized Stop Mechanism to ensure robust execution.
- FlowAgent incorporates a closed-loop execution scheme with a State Update Operator and Observation Encoder to dynamically absorb environmental feedback, effectively mitigating error accumulation in long-horizon tasks.

---

[RCoT-Seg: Reinforced Chain-of-Thought for Video Reasoning and Segmentation](http://arxiv.org/abs/2605.07334)

- RCoT-Seg: introduces a video-of-thought framework that factorizes Video Reasoning Segmentation into temporal video reasoning and keyframe target perception, utilizing MLLM, AKS, KTG, SAM2, GRPO, and Hungarian-based reward.
- The framework employs an agentic keyframe selection mechanism with a self-critical loop to verify and refine keyframe choices, replacing heuristic sampling with a verifiable decision process.
- RCoT-Seg leverages GRPO reinforcement learning with a matching-aware reward to align reasoning traces with downstream segmentation, achieving state-of-the-art performance on video reasoning and referring segmentation benchmarks.

---

[Beyond Linear Attention: Softmax Transformers Implement In-Context Reinforcement Learning](http://arxiv.org/abs/2605.07333)

- Softmax ICTD: introduces a theoretical framework demonstrating that Transformers with standard softmax attention can implement weighted softmax TD updates in their forward pass.
- The architecture utilizes a dual-head Transformer design where one head computes current value updates and the other computes target values, effectively performing iterative policy evaluation.
- The paper proves that these Transformer parameters are global minimizers of a reinforcement pretraining loss, establishing convergence to the true value function as depth and context length increase.

---

[Discovering Ordinary Differential Equations with LLM-Based Qualitative and Quantitative Evaluation](http://arxiv.org/abs/2605.07323)

- DoLQ (Discovering Ordinary differential equations with LLM-based Qualitative and quantitative evaluation): introduces a multi-agent framework that integrates LLM-based qualitative reasoning with quantitative numerical validation to discover interpretable ODEs.
- The framework utilizes a Sampler Agent to propose candidate terms, a Parameter Optimizer to refine coefficients, and a Scientist Agent to filter physically implausible models through iterative feedback.
- By combining semantic assessment with residual-based optimization, DoLQ achieves superior success rates in recovering ground-truth symbolic structures across diverse dynamical systems compared to existing symbolic regression methods.

---

[When Stored Evidence Stops Being Usable: Scale-Conditioned Evaluation of Agent Memory](http://arxiv.org/abs/2605.07313)

- Scale-Conditioned Evaluation Protocol: introduces a trajectory-level evaluation framework for LLM agent memory that assesses usability under evidence-preserving growth by monitoring interaction budgets and failure regimes.
- The protocol utilizes four diagnostics—budget-compliant reliability, tail retrieval-call burden, failure-regime decomposition, and usable-scale boundary—to distinguish between intrinsic memory failures and agent-facing interaction inefficiencies.
- Empirical audits across flat, planar, and hierarchical memory interfaces demonstrate that scalable-memory claims must be explicitly conditioned on the specific agent, interface, scale range, and interaction budget.

---

[AT-VLA: Adaptive Tactile Injection for Enhanced Feedback Reaction in Vision-Language-Action Models](http://arxiv.org/abs/2605.07308)

- AT-VLA: introduces an adaptive tactile injection mechanism that balances pretrained knowledge with newly learned tactile representations to enhance precision in contact-rich manipulation.
- The framework utilizes a Tactile Reaction Dual-Stream mechanism to decouple sensory processing into a slow visual-language stream and a fast tactile control stream, enabling real-time closed-loop responses within 0.04 seconds.
- By employing a learnable Tactile Gate and Adaptive Cross Attention, the model dynamically modulates modality contributions, ensuring robust performance even when tactile signals are absent during inference.

---

[BioProVLA-Agent: An Affordable, Protocol-Driven, Vision-Enhanced VLA-Enabled Embodied Multi-Agent System with Closed-Loop-Capable Reasoning for Biological Laboratory Manipulation](http://arxiv.org/abs/2605.07306)

- BioProVLA-Agent: introduces a multi-agent framework for biological laboratory manipulation that integrates protocol parsing, state verification, and embodied execution through the Guiding Decision Agent, Tailored LLM Protocol Agent, VLM-RAG Verification Agent, and VLA Embodied Agent.
- The system utilizes AugSmolVLA to enhance the robustness of the VLA Embodied Agent against visual perturbations common in wet-lab environments, such as transparent labware and illumination shifts.
- By employing a closed-loop verification mechanism, the framework enables reliable long-horizon biological experimentation while reducing dependence on fixed robotic scripts.

---

[SOM: Structured Opponent Modeling for LLM-based Agents via Structural Causal Model](http://arxiv.org/abs/2605.07301)

- SOM (Structured Opponent Modeling): introduces a two-stage framework that decouples opponent model construction and prediction by grounding the process in Structural Causal Models (SCMs) to enable structured and controllable reasoning.
- The framework utilizes Dynamic SCM Construction to build a causal graph of opponent decision-making and Reasoning for Opponent Prediction and Adaptation to perform inference using personalized reasoning examples.
- SOM improves prediction accuracy and adaptability in multi-agent environments by replacing implicit contextual reasoning with explicit, graph-based structured reasoning pathways.

---

[EgoPro-Bench: Benchmarking Personalized Proactive Interaction in Egocentric Video Streams](http://arxiv.org/abs/2605.07299)

- ProAct-Stream: introduces a comprehensive benchmark and proactive streaming model that leverages simulated user profiles and a "short thinking, better interaction" paradigm to enable personalized HMI.
- The framework utilizes a two-stage training process, integrating Supervised Fine-Tuning and Reinforcement Learning to optimize reasoning length and response quality for low-latency interaction.
- EgoPro-Bench provides a robust evaluation protocol across 12 distinct domains, addressing the limitations of reactive MLLMs in temporal alignment and personalized intent understanding.

---

[Can Agents Price a Reaction? Evaluating LLMs on Chemical Cost Reasoning](http://arxiv.org/abs/2605.07251)

- CHEMCOST introduces a benchmark for evaluating LLMs on chemical procurement cost estimation, requiring agents to resolve chemical identities, retrieve supplier quotes, and perform multi-step arithmetic.
- The framework utilizes a ReAct agent architecture equipped with specialized tools to navigate complex, domain-constrained procurement tasks under both clean and noise-injected conditions.
- Experimental results demonstrate that while tool access is necessary for performance, LLMs struggle with evidence integration and robustness to realistic input formatting, highlighting a significant gap in scientific agentic reasoning.

---

[EnvSimBench: A Benchmark for Evaluating and Improving LLM-Based Environment Simulation](http://arxiv.org/abs/2605.07247)

- EnvSimBench: introduces a diagnostic framework and benchmark that reframes LLM-based environment simulation as a fully observable state prediction task to address hallucination, logical inconsistency, and state drift.
- The framework utilizes a constraint-driven MDP formulation that provides models with explicit before-state configurations and implementation code to enable independent, verifiable evaluation of simulation fidelity.
- Systematic evaluation reveals a universal state-change cliff where LLMs fail catastrophically when multiple state variables require simultaneous updates, a gap addressed by the specialized Balance2 model.

---

[MEMOREPAIR: Barrier-First Cascade Repair in Agentic Memory](http://arxiv.org/abs/2605.07242)

- MEMOREPAIR: introduces a barrier-first cascade-repair contract that manages the lifecycle of derived artifacts in agentic memory by withdrawing invalidated cascades and selectively republishing validated successors.
- The framework models agentic memory as a directed provenance graph and utilizes a min-cut solver to optimize the repair-cost tradeoff when reconstructing artifacts after deletion, correction, or migration events.
- Experimental results on ToolBench and MemoryArena demonstrate that MEMOREPAIR effectively eliminates stale-memory exposure while recovering the majority of validated successors at a significantly lower repair-operator cost than exhaustive methods.

---

[Rethinking Priority Scheduling for Sequential Multi-Agent Decision Making in Stackelberg Games](http://arxiv.org/abs/2605.07240)

- HPA (Hierarchical Priority Adjustment): introduces a hierarchical reinforcement learning framework that dynamically optimizes the execution order of agents in N-level Stackelberg Games to improve multi-agent coordination.
- The framework utilizes an upper policy to select execution sequences based on the current system state, while lower-level agents execute actions within a Spatio-Temporal Sequential Markov Game (STMG) structure.
- By employing a slow-fast update scheme with shared intrinsic rewards, the method effectively coordinates learning across time scales to adapt to changing environmental conditions.

---

[HMACE: Heterogeneous Multi-Agent Collaborative Evolution for Combinatorial Optimization](http://arxiv.org/abs/2605.07214)

- HMACE: introduces a heterogeneous multi-agent framework that decomposes heuristic search into specialized roles, including Proposer, Generator, Evaluator, and Reflector, to improve combinatorial optimization.
- The framework utilizes an archive-based memory to store evaluated heuristics, enabling behavior-aware retrieval that guides the Proposer toward diverse and promising search regions.
- By integrating a lightweight deterministic filter and role-specialized collaboration, HMACE achieves a favorable quality-efficiency trade-off while minimizing LLM token consumption compared to monolithic baselines.

---

[Towards Autonomous Business Intelligence via Data-to-Insight Discovery Agent](http://arxiv.org/abs/2605.07202)

- AIDA (Autonomous Insight Discovery Agent): introduces an end-to-end framework for autonomous business analysis that leverages a proprietary DSL and reinforcement learning to transform complex enterprise data into actionable insights.
- The framework models business analysis as a Markov Decision Process, utilizing a structured state representation and a Pareto Principle-guided reward mechanism to prioritize high-leverage insights while suppressing hallucinations.
- AIDA employs specialized masking strategies and a dual-channel feedback loop to ensure logical consistency and structural integrity during multi-turn exploration in large-scale industrial data environments.

---

[Learning Agent Routing From Early Experience](http://arxiv.org/abs/2605.07180)

- BoundaryRouter: introduces a training-free routing framework that leverages Experience Memory and Rubric-guided CoT to dynamically dispatch queries between a Lightweight LLM and a Full Agent.
- The framework utilizes a Hybrid Retriever to fetch relevant behavioral examples from the Experience Memory, enabling the Routing LLM to make informed decisions without requiring ground-truth labels.
- By employing Rubric-guided CoT, the system ensures stable and consistent routing performance across diverse task distributions, effectively balancing inference latency and accuracy.

---

[HyperEyes: Dual-Grained Efficiency-Aware Reinforcement Learning for Parallel Multimodal Search Agents](http://arxiv.org/abs/2605.07177)

- HyperEyes: introduces a parallel multimodal search agent that optimizes inference efficiency by fusing visual grounding and retrieval into a single atomic action using Unified Grounded Search (UGS), TRACE, and On-Policy Distillation (OPD).
- The framework employs a Parallel-Amenable Data Synthesis pipeline and Progressive Rejection Sampling to curate high-quality, non-redundant trajectories for training LLMs.
- HyperEyes establishes a new state-of-the-art for open-source agents by Pareto-dominating existing models in both accuracy and operational efficiency across six multimodal search benchmarks.

---

[Repeated Deceptive Path Planning against Learnable Observer](http://arxiv.org/abs/2605.07174)

- DeMP (Deceptive Meta Planning): introduces a two-level optimization framework that combines episode-level adaptation and meta-level updates to enable sustained deception against a learnable observer.
- The framework utilizes meta-level updates to anticipate the observer's learning dynamics, effectively mitigating the accumulation of adaptation lag inherent in repeated adversarial interactions.
- DeMP leverages a surrogate objective that aligns with belief-induced rewards, ensuring the agent maintains deceptive performance while adapting to evolving adversarial recognition models.

---

[Rethinking Experience Utilization in Self-Evolving Language Model Agents](http://arxiv.org/abs/2605.07164)

- ExpWeaver: introduces a lightweight paradigm that interweaves experience utilization into the agent's decision-making process by exposing experience as an optional resource during reasoning.
- The framework utilizes a Reasoning (Iterative cognitive process) → Action (Environment-interactive step) loop, augmented with a Trigger Token (Special [Retrieve] signal) to activate the Retrieval Mechanism (Context-aware memory lookup) only when additional guidance is required.
- ExpWeaver enables LLMs to selectively invoke experience from the Experience Repository (External memory of past interactions) at beneficial decision points, effectively reducing reliance on rigid initialization-only or always-on usage strategies.

---

[SREGym: A Live Benchmark for AI SRE Agents with High-Fidelity Failure Scenarios](http://arxiv.org/abs/2605.07161)

- SREGym: introduces a high-fidelity, modular benchmark for evaluating AI agents in diagnosing and mitigating complex, multi-layered failures within live cloud-native production environments.
- The framework utilizes a System Environment (production-like cloud-native environment), Agent Interface (MCP servers for observability and control), Fault and Noise Injectors (orchestrators for distributed failure scenarios), Oracles (diagnosis and mitigation verification), and a Leaderboard (performance tracking) to provide a rigorous evaluation platform.
- SREGym challenges LLMs by simulating diverse failure modes, including metastable and correlated failures, while requiring agents to distinguish between actual root causes and ambient noise.

---

[MATHLIBPR: Pull Request Merge-Readiness Benchmark for Formal Mathematical Libraries](http://arxiv.org/abs/2605.07147)

- MATHLIBPR: introduces a benchmark for evaluating whether LLMs can judge the merge-readiness of build-passing pull requests in the Mathlib4 formal mathematical library.
- The framework utilizes a staged evaluation protocol that provides increasing levels of context—ranging from code-local diffs to automated diagnostics and PR intent—to assess the reviewer-like judgment capabilities of LLMs and LLM agents.
- Empirical results demonstrate that current LLMs and agents struggle to distinguish merge-ready from not-merge-ready contributions, even when provided with repository-wide context and diagnostic signals.

---

[Can You Break RLVER? Probing Adversarial Robustness of RL-Trained Empathetic Agents](http://arxiv.org/abs/2605.07138)

- AEB: introduces a benchmark for evaluating the adversarial robustness of RL-trained empathetic agents across six psychologically grounded dialogue trajectories.
- The framework utilizes an adversarial simulator to test if LLMs can address latent emotional needs when users exhibit behaviors like gaslighting, escalation, or mood reversal.
- The study reveals a dissociation where RL-trained LLMs improve emotional responsiveness and final scores without significantly enhancing observable state tracking as measured by the Emotional Consistency Score.

---

[Demystifying and Detecting Agentic Workflow Injection Vulnerabilities in GitHub Actions](http://arxiv.org/abs/2605.07135)

- TAINTAWI: introduces a static analysis framework to detect Agentic Workflow Injection (AWI) vulnerabilities in GitHub Actions by modeling data flows from untrusted event contexts to agent-facing prompts and downstream workflow sinks.
- The framework constructs an Agentic Workflow Dependency Graph (AWDG) to capture cross-boundary control and data dependencies, enabling the identification of Prompt-to-Agent (P2A) and Prompt-to-Script (P2S) injection patterns.
- TAINTAWI identifies 496 exploitable AWI vulnerabilities in real-world workflows, including 343 previously unknown zero-day cases, by performing reachability analysis against action-level and workflow-level security guards.

---

[Region4Web: Rethinking Observation Space Granularity for Web Agents](http://arxiv.org/abs/2605.07134)

- Region4Web: introduces a framework that reorganizes the AXTree into functional regions through hierarchical decomposition and semantic abstraction to provide a more informative basis for web agent state understanding.
- PageDigest: provides a web-specific inference pipeline that delivers region-level observations as a compact digest, reducing observation length while maintaining task-relevant information across steps.
- The framework improves web agent performance on the WebArena benchmark by enabling efficient, region-based page state understanding that scales across diverse backbone LLMs.

---

[RRCM: Ranking-Driven Retrieval over Collaborative and Meta Memories for LLM Recommendation](http://arxiv.org/abs/2605.07129)

- RRCM: introduces a ranking-driven retrieval-and-reasoning framework that dynamically constructs decision-relevant contexts for LLMs by selectively accessing collaborative and meta memories.
- The framework utilizes an LLM as an agent that interleaves reasoning with on-demand memory retrieval, optimized end-to-end via reinforcement learning to maximize final recommendation quality.
- By replacing static retrieval rules with an adaptive policy, RRCM effectively mitigates context-length bottlenecks and improves recommendation accuracy for both popular and long-tail items.

---

[The Position Curse: LLMs Struggle to Locate the Last Few Items in a List](http://arxiv.org/abs/2605.07127)

- Position Curse: introduces a systematic failure in LLMs where models struggle to retrieve items by position in short lists, particularly when using backward indexing.
- The paper characterizes this failure across four axes: query direction, anchor type, indexing direction, and item type, demonstrating that backward retrieval consistently lags behind forward retrieval.
- The authors propose POSBENCH and PYINDEX to evaluate and mitigate this positional deficit, showing that while fine-tuning improves performance, the underlying capability remains far from saturated.

---

[Convergence and Emergence of In-Context Reinforcement Learning with Chain of Thought](http://arxiv.org/abs/2605.07123)

- ICRL (In-Context Reinforcement Learning): introduces a theoretical framework where a single-layer linear Transformer with Chain-of-Thought generation performs iterative Temporal Difference learning updates during the forward pass.
- The framework utilizes a structured context buffer to store interaction trajectories and intermediate weight iterates, enabling the model to refine its policy evaluation performance through successive CoT steps.
- Theoretical analysis establishes that the CoT generation process converges geometrically to the Temporal Difference fixed point, with performance saturating at a statistical floor determined by the context length.

---

[RepoZero: Can LLMs Generate a Code Repository from Scratch?](http://arxiv.org/abs/2605.07122)

- ACE (Agentic Code-Test Evolution): introduces a benchmark and framework for repository-level code generation that utilizes Source Repository, Test Files Generator, Test Case Generator, Environment Construction, Filtering, Coding Agent, Testing Agent, and ACE Workflow.
- The framework reformulates repository generation as a reproduction task, requiring agents to implement functionality from scratch based on API specifications while ensuring output equivalence through automated black-box testing.
- By integrating an iterative code-test feedback loop, the ACE framework enhances the success rate of complex repository-level synthesis by leveraging deterministic ground-truth outputs from source repositories.

---

[Switchcraft: AI Model Router for Agentic Tool Calling](http://arxiv.org/abs/2605.07112)

- Switchcraft: introduces a specialized model router for agentic tool calling that utilizes a DistilBERT classifier and a cost model to select the most cost-effective LLM while maintaining correctness.
- The system employs an AST-based checker to evaluate tool call accuracy across diverse benchmarks, enabling the router to learn from realized model behavior rather than chat-based preferences.
- Switchcraft achieves near-oracle accuracy while significantly reducing inference costs by dynamically routing queries to the cheapest LLM capable of executing the required tool calls.

---

[Securing Computer-Use Agents: A Unified Architecture–Lifecycle Framework for Deployment-Grounded Reliability](http://arxiv.org/abs/2605.07110)

- CUA Architecture–Lifecycle Framework: introduces a diagnostic lens for deployment-grounded reliability by mapping architectural layers (Perception, Decision, Execution) to lifecycle stages (Creation, Deployment, Operation, Maintenance).
- The framework distinguishes between failure origin and failure manifestation, identifying how upstream choices in Creation and Deployment create latent risks that surface during Operation or Maintenance.
- The paper provides a security and privacy taxonomy that links threat surfaces to specific lifecycle stages, emphasizing that effective control placement must follow the timing of failure introduction.

---

[ARMOR: An Agentic Framework for Reaction Feasibility Prediction via Adaptive Utility-aware Multi-tool Reasoning](http://arxiv.org/abs/2605.07103)

- ARMOR: introduces an agentic framework that explicitly models tool-specific utilities, adaptively prioritizes tools, and resolves tool conflicts to improve reaction feasibility prediction.
- The framework utilizes a hierarchical tool structure, pattern-based utility assessment, and a memory-augmented reasoning mechanism to leverage complementary strengths of multiple tools.
- Experimental results demonstrate that ARMOR consistently outperforms existing tool aggregation and selection methods by effectively identifying appropriate tools for specific reaction characteristics.

---

[Decentralized Diffusion Policy Learning for Enhanced Exploration in Cooperative Multi-agent Reinforcement Learning](http://arxiv.org/abs/2605.07101)

- DDPL: introduces a decentralized MARL algorithm that replaces unimodal Gaussian policies with expressive DDPMs to enable effective exploration in high-dimensional joint action spaces.
- The framework utilizes ISSM to enable efficient online training of diffusion policies without requiring samples from the target energy-based distribution.
- DDPL provides theoretical sample-complexity and Nash equilibrium gap guarantees, demonstrating improved performance and sample efficiency across continuous-action MARL benchmarks.

---

[TeamBench: Evaluating Agent Coordination under Enforced Role Separation](http://arxiv.org/abs/2605.07073)

- TeamBench: introduces a benchmark for evaluating LLM agent coordination by enforcing strict role separation via operating system permissions rather than relying on prompt-based instructions.
- The framework utilizes a Planner, Executor, and Verifier to ensure that no single agent can simultaneously read full requirements, modify the workspace, and certify the final answer.
- Experimental results demonstrate that while teams rarely outperform single agents on average, the benchmark effectively exposes coordination costs and Verifier failure modes that pass-rate metrics typically mask.

---

[Social Theory Should Be a Structural Prior for Agentic AI: A Formal Framework for Multi-Agent Social Systems](http://arxiv.org/abs/2605.07069)

- MASS (Multi-Agent Social Systems): introduces a formal framework for modeling agentic AI systems as dynamical systems where system-level outcomes emerge from the interaction of heterogeneous Agents, Information exchange function (f), Influence dynamics function (g), and Networked interaction structure (G).
- The framework identifies four structural priors—strategic heterogeneity, network-constrained dependence, co-evolution, and distributional instability—to explain how agent interactions generate emergent social dynamics.
- The authors demonstrate the framework's utility by analyzing an LLM-based social network, MoltBook, showing that these structural priors are empirically present in multi-agent systems.

---

[2.5-D Decomposition for LLM-Based Spatial Construction](http://arxiv.org/abs/2605.07066)

- 2.5-D Decomposition: introduces a neuro-symbolic pipeline that improves spatial reasoning by restricting the LLM to 2D horizontal planning while a deterministic executor handles vertical placement.
- The architecture incorporates an Instruction Parser, Structure Analyzer, Build Planner (LLM), Plan Verifier, Spatial Executor, and Response Formatter to eliminate systematic coordinate errors.
- By removing deterministic dimensions from the LLM output space, the system achieves high structural accuracy on construction tasks and transfers effectively to edge hardware.

---

[From Assistance to Agency: Rethinking Autonomy and Control in CI/CD Pipelines](http://arxiv.org/abs/2605.07062)

- Agentic CI/CD: introduces a conceptual framework for authority transfer in software delivery, distinguishing between data-plane authority and control-plane authority to categorize agent autonomy.
- The paper characterizes current systems as operating under bounded autonomy, where LLM-based agents are confined to data-plane tasks while relying on external governance mechanisms for safety.
- The authors propose a research agenda prioritizing control-plane safety, formalization of autonomy boundaries, and the development of evaluation frameworks to address the challenges of delegating decision-making to agents.

---

[Do Joint Audio-Video Generation Models Understand Physics?](http://arxiv.org/abs/2605.07061)

- AV-Phys Bench: introduces a comprehensive benchmark for evaluating physical commonsense in joint audio-video generation models across steady-state, event-transition, and environment-transition scene categories.
- The framework utilizes an AV-Phys Agent, which integrates a multimodal LLM with deterministic audio-DSP tools to provide scalable, rubric-based judgments of physical consistency.
- Experimental results reveal a significant gap between semantic adherence and physical consistency, with scene transitions emerging as the primary difficulty frontier for current generative models.

---

[MedExAgent: Training LLM Agents to Ask, Examine, and Diagnose in Noisy Clinical Environments](http://arxiv.org/abs/2605.07058)

- MedExAgent: introduces a POMDP-based framework for interactive clinical diagnosis that integrates patient questioning, medical exam tool-calling, and final diagnosis under a unified agent architecture.
- The framework employs a two-stage training pipeline consisting of supervised fine-tuning on Calgary-Cambridge-structured conversations followed by DAPO reinforcement learning to optimize for diagnostic accuracy, tool-call quality, and exam cost.
- MedExAgent incorporates a systematic noise model to simulate real-world clinical uncertainty, including seven patient noise types and three exam noise types, to enhance agent robustness.

---

[More Than Can Be Said: A Benchmark and Framework for Pre-Question Scientific Ideation](http://arxiv.org/abs/2605.06345)

- InciteResearch: introduces a multi-agent framework that transforms vague human research inspirations into structured, actionable proposals by decomposing the Socratic questioning chain into E, V, and N components.
- The framework utilizes a cognitive state-machine to manage the research pipeline, ensuring that each stage—Elicitation, Reframing, and Necessity Checking—serves as a logical prerequisite for the next.
- The authors also introduce TF-Bench, a benchmark designed to evaluate the capacity of LLMs to assist in the early, tacit stages of scientific ideation across four distinct scientific modes.

---

[Safactory: A Scalable Agentic Infrastructure for Training Trustworthy Autonomous Intelligence](http://arxiv.org/abs/2605.06230)

- Safactory: introduces a scalable infrastructure framework for training trustworthy autonomous agents by integrating Parallel Simulation Platform, Trustworthy Data Platform, and Autonomous Evolution Platform into a continuous closed-loop evolutionary pipeline.
- The framework utilizes a Parallel Simulation Platform for high-concurrency trajectory generation, a Trustworthy Data Platform for structured data asset management, and an Autonomous Evolution Platform for asynchronous reinforcement learning and policy distillation.
- Safactory enables systematic risk discovery and model improvement by transforming security evaluation from a one-time process into a continuous, data-driven lifecycle that includes planning-, perception- and tool use-agents.

---

[Proactive Instance Navigation with Comparative Judgment for Ambiguous User Queries](http://arxiv.org/abs/2605.06223)

- ProCompNav: introduces a two-stage framework for ambiguous instance navigation that constructs a candidate pool and identifies the target through comparative judgment using MLLM and NLI Verifier components.
- The framework replaces independent matching with a collect-then-compare pipeline, utilizing binary questions to prune candidates based on discriminative attributes.
- ProCompNav improves success rates and reduces user burden by leveraging comparative judgment to resolve target ambiguity among visually similar distractors.

---

[Entropy-Regularized Adjoint Matching for Offline Reinforcement Learning](http://arxiv.org/abs/2605.06156)

- ME-AM (Maximum Entropy Adjoint Matching): introduces a unified framework that resolves the Support-Binding Dilemma in offline RL by integrating a Reward-Guided Mixture Prior and functional Mirror Descent into the continuous flow formulation.
- The framework utilizes a Reward-Guided Mixture Prior to expand the geometric support of the generative model and employs functional Mirror Descent to flatten the empirical behavior density, thereby mitigating popularity bias.
- ME-AM achieves state-of-the-art performance on sparse-reward continuous control benchmarks by embedding score-based entropy gradients directly into the terminal boundary condition of the Adjoint Matching ODE.

---

[Skill1: Unified Evolution of Skill-Augmented Agents via Reinforcement Learning](http://arxiv.org/abs/2605.06130)

- Skill1: introduces a unified framework that trains a single policy to co-evolve skill selection, utilization, and distillation toward a shared task-outcome objective.
- The framework decomposes the task-outcome signal into a low-frequency trend for selection and a high-frequency variation for distillation to enable simultaneous optimization of all three capabilities.
- Experiments on ALFWorld and WebShop demonstrate that Skill1 outperforms prior skill-based and RL baselines by achieving unified convergence across the entire agent lifecycle.

---

[Breaking, Stale, or Missing? Benchmarking Coding Agents on Project-Level Test Evolution](http://arxiv.org/abs/2605.06125)

- TEBench (Test Evolution Benchmark): introduces the first project-level benchmark for test evolution, requiring autonomous identification and update of tests across entire repositories following production code changes.
- The framework evaluates LLM-based agents using a three-version structure to classify test evolution into Test-Breaking, Test-Stale, and Test-Missing categories.
- Experimental results reveal a shared performance ceiling across seven configurations, highlighting that Test-Stale is the most challenging evolution type due to the lack of explicit execution failure signals.

---

[On Time, Within Budget: Constraint-Driven Online Resource Allocation for Agentic Workflows](http://arxiv.org/abs/2605.06110)

- MCPP (Monte Carlo Portfolio Planning): introduces a closed-loop planning framework that dynamically allocates models and parallel samples to subtasks to maximize the probability of completing agentic workflows within explicit budget and deadline constraints.
- The framework treats budget and time as hard success conditions, shifting the optimization objective from average efficiency to instance-specific constrained completion probability.
- By simulating downstream workflow outcomes using a portfolio of continuation policies, the planner selects actions that prioritize bottleneck subtasks while managing remaining resources.

---

[Milestone-Guided Policy Learning for Long-Horizon Language Agents](http://arxiv.org/abs/2605.06078)

- BEACON: introduces a milestone-guided policy learning framework that addresses credit misattribution and sample inefficiency in long-horizon LLM agents by partitioning trajectories at milestone boundaries, applying temporal reward shaping, and utilizing dual-scale advantage estimation.
- The framework leverages the compositional structure of long-horizon tasks to decouple credit assignment across phases, ensuring that local action quality is isolated from downstream stochasticity.
- Experimental results on ALFWorld, WebShop, and ScienceWorld demonstrate that BEACON significantly improves success rates and sample utilization compared to existing trajectory-level reinforcement learning methods.

---

[MAS-Algorithm: A Workflow for Solving Algorithmic Programming Problems with a Multi-Agent System](http://arxiv.org/abs/2605.05949)

- MAS-Algorithm: introduces a multi-agent workflow that decomposes algorithmic problem solving into modular stages, including Agent1 (predicts relevant algorithm labels), Agent2 (retrieves external algorithmic knowledge), Agent3 (generates structured solution plans), Agent4 (translates plans into executable code), and Agent5 (diagnoses errors and provides feedback).
- The framework integrates specialized agents with auxiliary components like a RAG Module (retrieves domain-specific information), Replanning Module (revises reasoning plans), Revising Module (updates generated code), Judging Tools (executes and validates code), a Format Conversion Tool (standardizes intermediate outputs), and a Sample Extraction Tool (parses problem specifications).
- Experimental results demonstrate that this multi-agent approach consistently improves the performance of LLMs on algorithmic tasks, outperforming both direct prompting and parameter-efficient fine-tuning.

---

[Active Learning for Communication Structure Optimization in LLM-Based Multi-Agent Systems](http://arxiv.org/abs/2605.05703)

- Active Learning for Communication Structure Optimization in LLM-MAS: introduces an ensemble-based information-theoretic framework that optimizes communication structures by selecting the most informative tasks for LLM-MAS training.
- The framework utilizes Representative Selection to reduce candidate pools and an EKI-based Utility Estimator to approximate Bayesian posterior updates for graph parameters without requiring gradients.
- By incorporating surrogate modeling and batch Thompson sampling, the approach efficiently identifies informative tasks to improve downstream accuracy and robustness under constrained computational budgets.

---


[Exploring a Virtual Pet to Provide Context Notifications in a Tourism Recommender System: a Pilot Study](http://arxiv.org/abs/2605.07960)

- GRS: introduces a context-aware tourism recommendation framework that utilizes a virtual pet as a social mediator to deliver personalized alerts and mitigate notification fatigue.
- The system integrates real-time IoT environmental data with a Multi-Agent Microservice to generate personalized recommendations and safety-critical notifications.
- A pilot study demonstrates that the virtual pet interface enhances hedonic quality and user acceptance by transforming technical alerts into friendly, character-mediated companion advice.

---


[REINFORCEMENT LEARNING FOR SCALABLE AND TRUSTWORTHY INTELLIGENT SYSTEMS](http://arxiv.org/abs/2605.08378)

- FedNPG-ADMM: introduces a communication-efficient synchronous federated RL framework that integrates ADMM with natural policy gradient optimization to reduce communication complexity from O(d²) to O(d).
- AFedPG: introduces an asynchronous federated RL framework that utilizes a delay-adaptive lookahead technique to mitigate stale updates and improve global time complexity in heterogeneous environments.
- MaPPO and CI-RL: introduce trustworthy RL methods, where MaPPO incorporates prior reward knowledge into preference optimization for LLMs, and CI-RL instills contextual integrity reasoning into LLM policies via RL.

---

#### 7th May 2026


[UX in the Age of AI: Rethinking Evaluation Metrics Through a Statistical Lens](http://arxiv.org/abs/2605.05600)

- ADUX-Stat (Adaptive Dynamic UX Statistical Framework): introduces a three-construct model designed to evaluate usability in AI-mediated environments by treating user experience as a probabilistic signal distribution.
- The framework integrates IEI to measure output variability, TDC to track longitudinal usability trends, and BUCS to provide Bayesian credible intervals for uncertainty quantification.
- ADUX-Stat addresses the limitations of classical UX metrics by providing a statistically grounded, field-deployable methodology for evaluating non-deterministic AI interfaces.

---


[Incentive Design in Competitive Resource Allocation: Exploiting Valuation Asymmetry in Tullock Contests](http://arxiv.org/abs/2605.07045)

- Incentive Design in Competitive Resource Allocation: introduces a mechanism for a central coordinator to influence subordinate agents in a multi-player Tullock contest by strategically manipulating their reported valuations of a contested prize.
- The framework leverages cost asymmetry among subordinates to steer the Nash equilibrium toward outcomes that maximize the coordinator's utility.
- Analytical results demonstrate that optimal valuations for subordinates are proportional to the square root of their per-unit costs, effectively reducing the coordinator's optimization problem to a two-variable system.

---


[The Causally Emergent Alignment Hypothesis: Causal Emergence Aligns with and Predicts Final Reward in Reinforcement Learning Agents](http://arxiv.org/abs/2605.06746)

- Causal Emergence Alignment Hypothesis: introduces a framework for evaluating how causal emergence in RL agents aligns with and predicts final reward performance across diverse environments.
- The study utilizes ΦID decomposition on latent-space trajectories to quantify causal emergence as a measure of agent integration and goal-directed representational reorganization.
- Experimental results demonstrate that causal emergence provides a robust, early-training directional signal that outperforms standard neural representation metrics in predicting final learning outcomes.

---


[The Context Gathering Decision Process: A POMDP Framework for Agentic Search](http://arxiv.org/abs/2605.07042)

- CGDP: introduces a formal POMDP-based framework for agentic search that replaces implicit LLM state tracking with explicit, modular infrastructure.
- The framework utilizes a persistent Belief State to bound context and a programmatic Exhaustion Gate to prevent redundant looping and premature stopping.
- Empirical validation across four agent harnesses demonstrates that these modular interventions improve multi-hop reasoning and reduce token consumption without sacrificing accuracy.

---



[Beyond Task Success: Measuring Workflow Fidelity in LLM-Based Agentic Payment Systems](http://arxiv.org/abs/2605.06457)

- HMASP (Hierarchical Multi-Agent System for Payments): introduces a trajectory-level evaluation metric, ASR, to measure workflow fidelity in LLM-based agentic payment systems by comparing observed agent execution sequences against expected paths.
- The framework utilizes a Payment Supervisor and CPA to manage payment workflows, incorporating prompt engineering and deterministic routing guards to ensure compliance with auditable procedural steps.
- Evaluation across 18 LLMs reveals that standard metrics like Task Success Rate often fail to detect systematic workflow shortcuts, whereas ASR identifies deviations in agent handoffs essential for regulatory compliance.

---

[Cognitive Agent Compilation for Explicit Problem Solver Modeling](http://arxiv.org/abs/2605.07040)

- CAC: introduces a failure-driven learning pipeline that compiles problem-solving knowledge from a Teacher LLM into an explicit, inspectable Knowledge Base for a target Cognitive Agent.
- The framework utilizes a Cognitive Agent (SLM) that performs deterministic actions based on retrieved declarative memory, while the Teacher LLM iteratively refines the Knowledge Base to ensure task success.
- This approach addresses the opacity of LLMs in educational settings by decoupling knowledge representation from reasoning, facilitating more transparent and editable learner modeling.

---

[PACEvolve++: Improving Test-time Learning for Evolutionary Search Agents](http://arxiv.org/abs/2605.07039)

- PACEvolve++: introduces an advisor-model reinforcement learning framework that decouples strategic search decisions from code implementation to enable effective test-time policy adaptation in evolutionary search agents.
- The framework utilizes a trainable Advisor LLM for hypothesis generation and selection, while delegating the realization of executable code to a stronger, frozen Frontier Implementation LLM.
- To handle non-stationary search dynamics, the system employs a phase-adaptive reinforcement learning objective that transitions credit assignment from group-relative feedback during early exploration to frontier-contribution feedback during late-stage refinement.

---

[Self Driving Datasets: From 20 Million Papers to Nuanced Biomedical Knowledge at Scale](http://arxiv.org/abs/2605.07022)

- Starling: introduces a multi-agent deep research system that autonomously converts unstructured biomedical literature into large-scale, nuanced, and accurate structured datasets.
- The system utilizes a Proposer agent, Validator agent, Investigator agent, Extractor agent, and Judge agent to perform iterative query construction, schema induction, and high-fidelity data extraction.
- Starling leverages an entity tagging model and a hybrid sparse–dense retrieval layer to efficiently process a 22.5 million paper PubMed corpus, outperforming manual curation in scale, accuracy, and contextual nuance.

---

[Problem Space Attunement in Youth Social Media Design](http://arxiv.org/abs/2605.07018)

- Problem Space Attunement in Youth Social Media Design: introduces a research framework that addresses misattunement in youth social media design through Fictional Inquiry, Asynchronous Remote Community, and an LLM-agent simulation sandbox.
- The framework utilizes ego-anchored agents to simulate social dynamics, allowing youth to evaluate design choices within familiar, grounded contexts.
- This approach shifts design research from adult-centric assumptions toward youth-articulated goals, criteria, and boundary conditions for relationally supportive social media.

---

[Dual-Agent Co-Training for Health Coaching via Implicit Adversarial Preference Optimization](http://arxiv.org/abs/2605.07011)

- DACT: introduces a dual-agent co-training framework that interactively optimizes both a health coach agent and a client simulator using implicit adversarial preference optimization.
- The framework utilizes a multi-dimensional LLM judge to construct Pareto-dominant preference pairs, enabling the coach to improve across multiple clinical dimensions while the client learns to provide increasingly challenging interactions.
- This co-evolutionary process admits a natural stochastic-game interpretation, where alternating DPO updates for the coach and client agents drive robust performance improvements and significant reduction in clinical anti-patterns.

---

[SmellBench: Evaluating LLM Agents on Architectural Code Smell Repair](http://arxiv.org/abs/2605.07001)

- SmellBench: introduces a task orchestration framework for evaluating LLM agents on architectural code smell repair, utilizing an LLM Agent, SmellBench MCP Server, Repository, Agent Skill, Trigger Prompt, GEPA Prompt Optimization Pipeline, Static Validator, Judge Model, and Task Lifecycle State Machine.
- The framework employs a GEPA pipeline to generate smell-specific execution playbooks and few-shot demonstrations, which are delivered to agents via a structured task packet.
- Evaluation of 11 agent configurations reveals that repair aggressiveness and net codebase quality are inversely related, with the best agents achieving a 47.7% resolution rate on architectural smells.

---

[Echo: KV-Cache-Free Associative Recall with Spectral Koopman Operators](http://arxiv.org/abs/2605.06997)

- Echo: introduces a KV-cache-free architecture that replaces standard attention and feedforward layers with Spectral Koopman Attention and Koopman MLP components to enable constant-memory retrieval.
- The framework utilizes Mamba-2/SSM Backbone for local processing while employing SKA to accumulate sufficient statistics in fixed-size Sufficient Statistic Accumulators, effectively eliminating the memory cliff associated with standard SSMs.
- By recasting content-addressed retrieval as a closed-form ridge regression solved via Cholesky Factorization and Power Spectral Filter, the architecture maintains high retrieval accuracy across long sequences without linear KV cache growth.

---

[Why Does Agentic Safety Fail to Generalize Across Tasks?](http://arxiv.org/abs/2605.06992)

- Agentic Safety Generalization Framework: introduces a theoretical and empirical investigation into why agentic safety fails to generalize across tasks, demonstrating that the relationship between a task and its safe execution is inherently more complex than the relationship between a task and its execution alone.
- The paper proves that the mapping from task specification to an optimal controller has a higher Lipschitz constant under safety requirements than without, establishing a formal sense in which generalization is fundamentally more difficult with safety constraints.
- Empirical experiments across linear-quadratic control, neural network-based quadcopter navigation, and LLM-based CRM agents corroborate the theoretical findings, showing that safe behavior is significantly harder to transfer to unseen tasks than standard execution behavior.

---

[The Cost of Consensus: Malignant Epistemic Herding and Adaptive Gating in Distributed Multi-Agent Search](http://arxiv.org/abs/2605.06988)

- Entropy-delta gating framework: introduces a decentralized multi-agent search architecture that mitigates malignant epistemic herding by conditioning communication on information novelty using Shannon entropy as both a transmission gate and a fusion weight.
- The framework utilizes an inverse-entropy fusion rule to dynamically weight incoming messages, ensuring that more confident agents exert greater influence on the collective belief state.
- By replacing continuous high-frequency communication with event-triggered updates, the system achieves superior epistemic alignment and task success while reducing bandwidth consumption by over 98%.

---

[Group of Skills: Group-Structured Skill Retrieval for Agent Skill Libraries](http://arxiv.org/abs/2605.06978)

- GOSKILLS (Group of Skills): introduces an inference-time retrieval method that organizes skills into role-labeled execution contexts using Skill Library, Typed Skill Graph, Group Pool, Group Graph, Inverted Index, Query Schema, Anchor Selection, Support Expansion, Bottlenecking, Execution Contract, and Coverage-Debt Accounting.
- The framework improves agent performance by replacing flat skill lists with structured, anchor-centered groups that explicitly define execution entry points, support roles, and visible requirements.
- GOSKILLS preserves visible-requirement coverage under constrained budgets and reduces agent-only runtime by minimizing the organizational burden on the LLM during task execution.

---

[Learning and Reusing Policy Decompositions for Hierarchical Generalized Planning with LLM Agents](http://arxiv.org/abs/2605.06957)

- HCL-GP (Hierarchical Component Learning for Generalized Policies): introduces a dynamic policy-learning architecture that integrates generalized planning and hierarchical task decomposition to synthesize reusable, parameterized policies for LLM agents.
- The framework utilizes a multi-agent system comprising Task Abstraction and Parameterization Agent, Component Search Agent, Policy Generator (Planning) Agent, Decomposition Agent, and Generalize and Deduplicate Agent to iteratively learn, refine, and store executable policy components.
- By extracting and generalizing procedural patterns from successful executions, the approach enables efficient cross-domain transfer and improves iteration efficiency in complex, multi-step interactive environments.

---

[Multi-Objective Constraint Inference using Inverse reinforcement learning](http://arxiv.org/abs/2605.06951)

- MOCI: introduces a framework that jointly recovers shared hard constraints and individual preferences from unlabeled, heterogeneous expert trajectories using an Expectation-Maximization approach.
- The framework utilizes Maximum Entropy Inverse Reinforcement Learning to model diverse expert behaviors by clustering trajectories into latent types while simultaneously identifying forbidden states via greedy search.
- Empirical results demonstrate that MOCI achieves superior predictive accuracy and computational efficiency compared to existing baseline methods in multi-objective GridWorld environments.

---

[Bridging the Last Mile of Circuit Design: POSTEDA-BENCH, a Hierarchical Benchmark for PPA Convergence and DRC Fixing](http://arxiv.org/abs/2605.06936)

- POSTEDA-BENCH: introduces a hierarchical benchmark for evaluating LLM-based agents on post-EDA closure, including DRC-Bench and PPA-Bench, supported by both open-source and commercial toolchains.
- The benchmark evaluates LLM-based agents across DRC-Essential, DRC-Reasoning, PPA-Mono, and PPA-Multi, identifying that trade-off reasoning is the primary bottleneck for multi-objective PPA convergence.
- Experimental results across eight LLMs demonstrate a synthetic-to-practical performance gap, where vision augmentation consistently enhances DRC-Bench performance.

---

[MAGIQ: A Post-Quantum Multi-Agentic AI Governance System with Provable Security](http://arxiv.org/abs/2605.06933)

- MAGIQ: introduces a post-quantum secure governance framework for multi-agentic AI systems that enforces user-defined communication and access control policies through efficient cryptographic primitives.
- The framework utilizes application-level communication abstractions, A-session and C-session, to provide fine-grained policy enforcement and message attribution for agent-to-agent and one-to-many interactions.
- MAGIQ provides provable security guarantees using the Universal Composability (UC) framework and demonstrates practical performance overhead compared to classical baseline systems.

---

[A2RD: Agentic Autoregressive Diffusion for Long Video Consistency](http://arxiv.org/abs/2605.06924)

- A2RD: introduces an agentic autoregressive architecture for long video synthesis that decouples creative synthesis from consistency enforcement using a Retrieve–Synthesize–Refine–Update cycle.
- The framework integrates a Multimodal Video Memory, Adaptive Segment Generation, and Hierarchical Test-Time Self-Improvement to maintain temporal consistency and narrative coherence over long horizons.
- The authors also contribute LVbench-C, a benchmark designed to stress-test long-horizon video consistency through cyclical entity and environment transitions.

---

[Same Signal, Opposite Meaning: Direction-Informed Adaptive Learning for LLM Agents](http://arxiv.org/abs/2605.06908)

- DIAL (Direction-Informed Adaptive Learning): introduces a framework that learns the utility direction of gating signals per environment and backbone to optimize test-time compute for LLM agents.
- The framework utilizes Bernoulli-ε exploration to collect counterfactual data, enabling the training of a sparse linear gate that avoids the pitfalls of fixed-direction assumptions.
- By decomposing state types into intervention-unsuitable and decision-difficult regimes, DIAL effectively calibrates compute utility across heterogeneous agent environments.

---

[Self-Programmed Execution for Language-Model Agents](http://arxiv.org/abs/2605.06898)

- SPE (Self-Programmed Execution): introduces an agent architecture where the LLM completion itself functions as the orchestrator program, removing the need for a fixed external orchestration policy.
- The framework utilizes Spell, a Lisp-based language, to enable LLMs to programmatically manage their own context, tool calls, and recursive self-calls.
- Experiments demonstrate that frontier LLMs can effectively use Spell to solve complex agentic tasks by treating orchestration as dynamic, model-generated program logic.

---

[Beyond the Black Box: Interpretability of Agentic AI Tool Use](http://arxiv.org/abs/2605.06890)

- Mechanistic-interpretability toolkit: introduces a framework for monitoring agentic tool decisions by mapping pre-action activations through Sparse Autoencoders (SAEs) to identify tool-use intent and risk levels.
- The framework utilizes a Tool-Need Probe (binary) and a Tool-Risk Probe (ternary) to provide internal observability into agent behavior before external execution occurs.
- By applying feature ablation to sparse latent representations, the approach confirms the functional necessity of specific internal features in driving agent tool-use decisions.

---

[Agentick: A Unified Benchmark for General Sequential Decision-Making Agents](http://arxiv.org/abs/2605.06869)

- Agentick: introduces a unified benchmark for evaluating RL, LLM, VLM, and hybrid agents across 37 procedurally generated tasks using a Gymnasium-compatible interface, Coding API, oracle reference policies, pre-built SFT datasets, and a composable agent harness.
- The framework utilizes a capability-decomposed, multi-modal design that provides five synchronized observation modalities to ensure fair comparison across diverse agent paradigms.
- Experimental results demonstrate that no single paradigm dominates, while harness design, specifically the use of chain-of-thought reasoning, significantly multiplies LLM performance.

---

[Multi-Objective Multi-Agent Bandits: From Learning Efficiency to Fairness Optimization](http://arxiv.org/abs/2605.06864)

- MO-MA-MAB: introduces decentralized algorithms for multi-objective multi-agent bandits that balance learning efficiency and fairness under heterogeneous rewards and time-varying communication.
- The paper develops PARETO UCB1 GOSSIP for Pareto-optimal learning and SIMULATED NSW UCB GOSSIP for optimizing Nash Social Welfare using gossip-based communication and preference-based reward simulation.
- Theoretical analysis establishes regret bounds of O(log T) for Pareto efficiency and O(T^3/4) for Nash Social Welfare, highlighting the fundamental tradeoff between fairness and convergence speed in decentralized multi-agent systems.

---

[AGWM: Affordance-Grounded World Models for Environments with Compositional Prerequisites](http://arxiv.org/abs/2605.06841)

- AGWM: introduces a world model that explicitly tracks environment affordances using a self-evolving Dynamic Affordance Graph to mitigate compounding multi-step imagination errors.
- The framework integrates an SC Classifier to detect structure-changing events and a Graph Predictor that enforces a frontier-mask constraint to ensure imagined trajectories remain within feasible action sets.
- By conditioning the RSSM World Model on the graph embedding, AGWM achieves improved generalization to novel rule configurations and reduces prediction error in environments with compositional prerequisites.

---

[Randomness is sometimes necessary for coordination](http://arxiv.org/abs/2605.06825)

- Diamond Attention: introduces a cross-attention architecture that utilizes structured random masking to induce transient rank ordering among homogeneous agents, enabling role differentiation in cooperative multi-agent reinforcement learning.
- The framework generates per-timestep asymmetric attention masks by sampling scalar random numbers, allowing agents to dynamically form hierarchies without requiring explicit communication or unique identifiers.
- By decoupling coordination from fixed agent identities and environmental cues, the architecture achieves zero-shot generalization to varying team sizes and cross-scenario transfer in adversarial environments.

---

[SHARP: A Self-Evolving Human-Auditable Rubric Policy for Financial Trading Agents](http://arxiv.org/abs/2605.06822)

- SHARP: introduces a neuro-symbolic framework that replaces unconstrained prompt mutation with structured, symbolic policy optimization to address credit assignment in LLM-based financial trading.
- The framework utilizes an Attribution Agent to diagnose failures, an Evolution Agent to perform atomic rule edits, and a Validation Gate to ensure robust, auditable strategy refinement.
- By confining LLM reasoning to a bounded, human-readable rubric, SHARP enables dynamic adaptation to non-stationary markets while maintaining the transparency required for institutional finance.

---

[Towards Security-Auditable LLM Agents: A Unified Graph Representation](http://arxiv.org/abs/2605.06812)

- Agent-BOM (Agent Bill of Materials): introduces a hierarchical attributed directed graph representation that bridges the semantic gap in LLM agent security by modeling static capability bases and dynamic runtime semantic states.
- The framework enables path-level security auditing by organizing execution traces into queryable evidence paths that connect risk entry points, semantic evolution, capability invocation, and cross-agent propagation.
- By instantiating a 4-tuple auditing rule template, the approach allows for systematic root-cause analysis and security adjudication of complex agentic risks such as memory poisoning, tool misuse, and multi-agent ecosystem hijacking.

---

[Conformal Agent Error Attribution](http://arxiv.org/abs/2605.06788)

- Conformal Agent Error Attribution: introduces a framework for identifying decisive errors in multi-agent system trajectories using conformal prediction to provide statistical coverage guarantees.
- The framework utilizes filtration-based conformal prediction algorithms to generate contiguous prediction sets, enabling efficient error localization and automated agent recovery.
- The approach is model-agnostic and demonstrates that matching the conformal algorithm to the error distribution of the data significantly improves the precision of error attribution.

---

[When Does Critique Improve AI-Assisted Theoretical Physics? SCALAR: Structured Critic–Actor Loop for Agentic Reasoning](http://arxiv.org/abs/2605.06772)

- SCALAR: introduces a pedagogical Actor–Critic–Judge pipeline to evaluate how interaction strategies between LLMs affect reasoning performance in theoretical physics.
- The framework utilizes an Actor agent for problem-solving, a Critic agent for formative feedback, and an independent Judge agent for final evaluation against reference solutions.
- SCALAR provides a controlled testbed to analyze the impact of Actor personas, Critic feedback strategies, and model scaling on the convergence of multi-turn LLM reasoning.

---

[BAMI: Training-Free Bias Mitigation in GUI Grounding](http://arxiv.org/abs/2605.06664)

- BAMI: introduces a training-free inference framework that mitigates precision and ambiguity biases in GUI grounding by utilizing MPD, Coarse-to-Fine Focus, and Candidate Selection.
- The framework employs MPD to diagnose failure sources, followed by a recursive Coarse-to-Fine Focus mechanism and a prompt-guided Candidate Selection process to improve localization accuracy.
- BAMI enhances existing GUI grounding models by incorporating structured inference and external correction models without requiring additional training.

---

[AI Co-Mathematician: Accelerating Mathematicians with Agentic AI](http://arxiv.org/abs/2605.06651)

- AI Co-Mathematician: introduces a stateful, agentic workbench designed to support mathematicians by orchestrating complex, iterative research workflows through a hierarchy of specialized agents.
- The system utilizes a project coordinator agent to manage parallel workstreams, which leverage specialized sub-agents and reviewer agents to maintain rigor and track research progress in a shared workspace.
- By integrating human-in-the-loop steering with hard programmatic constraints, the framework enables the synthesis of native mathematical artifacts while managing the inherent uncertainty of LLMs.

---

[Revisiting Adam for Streaming Reinforcement Learning](http://arxiv.org/abs/2605.06764)

- Adaptive Q(λ): introduces a streaming reinforcement learning algorithm that combines momentum-based eligibility traces with ADAM-style variance-adjusted updates and bounded error signals to achieve stable learning.
- The framework demonstrates that classical batch-RL objectives, when coupled with properly tuned ADAM hyperparameters, are highly effective in streaming environments.
- The research identifies ADAM's epsilon parameter as a critical Signal-to-Noise Ratio filter that facilitates stable convergence by regulating updates for sparse or noisy features.

---

[StraTA: Incentivizing Agentic Reinforcement Learning with Strategic Trajectory Abstraction](http://arxiv.org/abs/2605.06642)

- StraTA: introduces a framework that decomposes long-horizon agentic tasks into strategy generation and strategy-conditioned action execution to improve exploration and credit assignment.
- The framework utilizes a hierarchical GRPO-style training objective that optimizes both high-level strategies and low-level actions using Strategy Generator, Action Executor, and Hierarchical GRPO-style Rollout.
- StraTA enhances learning efficiency through Farthest Point Sampler for diverse strategy exploration and Critical Self-Judgment Agent for fine-grained step-level credit assignment.

---

[Recursive Agent Optimization](http://arxiv.org/abs/2605.06639)

- RAO: introduces a reinforcement learning approach that trains a single shared LLM policy to dynamically spawn and manage recursive sub-agents for complex task decomposition.
- The framework utilizes a local node reward mechanism, incorporating a delegation bonus and a leave-one-out baseline, to provide dense credit assignment across the execution tree.
- By optimizing a weighted multi-task objective, RAO enables agents to generalize to harder tasks, scale beyond context window limits, and improve training efficiency through self-induced curricula.

---

[Cited but Not Verified: Parsing and Evaluating Source Attribution in LLM Deep Research Agents](http://arxiv.org/abs/2605.06635)

- Source attribution evaluation framework: introduces a three-stage pipeline that utilizes a Markdown AST parser to extract citation-claim pairs and three specialized evaluators to assess link accessibility, topical relevance, and factual accuracy.
- The framework employs an LLM-as-a-judge approach for content and fact verification, calibrated through human review to mitigate systematic biases.
- Experimental results across 14 LLMs demonstrate that while frontier models maintain high link validity, they exhibit significant factual accuracy degradation as search depth increases, indicating an information overload effect.

---

[MASPO: Joint Prompt Optimization for LLM-based Multi-Agent Systems](http://arxiv.org/abs/2605.06623)

- MASPO: introduces a joint prompt optimization framework for LLMs in multi-agent systems that resolves credit assignment dilemmas by integrating LLM Evaluator, Prompt Optimizer, Misalignment Buffer, Search Tree, Beam Search, Topological Scheduler, and Joint Reward Model.
- The framework employs a multi-granularity joint evaluation mechanism and misalignment-aware sampling to bridge the gap between local agent objectives and global system outcomes.
- MASPO utilizes an evolutionary beam search with a beam refresh mechanism to ensure stable co-adaptation of agents in non-stationary multi-agent environments.

---

[SkillOS: Learning Skill Curation for Self-Evolving Agents](http://arxiv.org/abs/2605.06614)

- SkillOS: introduces an experience-driven RL training recipe for learning skill curation in self-evolving agents by pairing a frozen Agent Executor with a trainable Skill Curator.
- The framework utilizes a SkillRepo to store reusable procedural knowledge in Markdown format, which is dynamically managed by the Skill Curator through insert, update, and delete operations.
- By training on grouped task streams and employing a composite reward function, the system effectively turns delayed downstream feedback into actionable learning signals for long-term agent proficiency.

---

[Patch2Vuln: Agentic Reconstruction of Vulnerabilities from Linux Distribution Binary Patches](http://arxiv.org/abs/2605.06601)

- Patch2Vuln: introduces an agentic pipeline that reconstructs security vulnerabilities from Linux distribution binary patches by orchestrating binary analysis tools and LLM-based reasoning without external advisory data.
- The framework utilizes a candidate-centric architecture where binary diffs are transformed into dossiers, enabling the LLM agent to perform preliminary audits, plan bounded validation, and generate final vulnerability reports.
- Experimental results on 25 Ubuntu package pairs demonstrate that the agent effectively localizes security-relevant functions and identifies root-cause classes while maintaining hallucination resistance against negative controls.

---

[Cross-Modal Navigation with Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2605.06595)

- CRONA: introduces a decentralized MARL framework for cross-modal navigation that leverages modality-specialized agents, auxiliary belief predictors, and a centralized multi-modal critic to improve collaborative navigation efficiency.
- The framework utilizes Audio Encoder, Vision Encoder, Auxiliary Belief Predictor, History Encoder, Multi-Modal Critic, Replay Buffer, Policy Head, and Value Head to enable effective coordination among heterogeneous agents without inter-agent communication during execution.
- Experimental results across Matterport3D scenes demonstrate that CRONA outperforms single-agent baselines and identifies five distinct modality-dominance patterns that dictate the effectiveness of collaborative navigation strategies.

---

[WEBLICA: Scalable and Reproducible Training Environments for Visual Web Agents](http://arxiv.org/abs/2605.06761)

- WEBLICA (Web Replica): introduces a framework for constructing reproducible and scalable web environments to train visual web agents using WEBLICA-CACHE and WEBLICA-SYNTH.
- The framework leverages HTTP-level caching to capture stable visual states and an LLM-based synthesis pipeline to generate diverse, grounded web environments for RL training.
- WEBLICA-8B outperforms open-weight baselines on web navigation benchmarks while using fewer inference steps and scaling effectively with additional test-time compute.

---

[NeuroAgent: LLM Agents for Multimodal Neuroimaging Analysis and Research](http://arxiv.org/abs/2605.06584)

- NeuroAgent: introduces a hierarchical multi-agent framework that automates the full lifecycle of multimodal neuroimaging analysis, utilizing a Central Orchestrator, Specialized Modality Agents, and a Feedback-Driven Execution Engine.
- The system employs a Global Workflow Registry for state persistence and a Human-in-the-Loop interface to ensure reliability and allow expert intervention during complex scientific workflows.
- NeuroAgent achieves high performance in automated preprocessing and downstream Alzheimer's disease classification by integrating LLM-driven reasoning with established neuroimaging toolchains.

---

[Language Models Can Autonomously Hack and Self-Replicate](http://arxiv.org/abs/2605.06760)

- Language Models Can Autonomously Hack and Self-Replicate: demonstrates that LLMs can autonomously exploit web vulnerabilities, extract credentials, and replicate their full inference stack across a network of hosts.
- The research evaluates autonomous self-replication by decomposing the process into propensity and capability, specifically measuring the ability of LLMs to perform hacking and copying tasks.
- The study validates chain replication across multiple hops, showing that replicas can independently discover and exploit new targets without human intervention.

---

[Coordination Matters: Evaluation of Cooperative Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2605.06557)

- STAT: introduces a coordination-aware evaluation perspective for cooperative MARL by utilizing a controlled commitment-constrained spatial task-allocation testbed to expose process-level diagnostics beyond aggregate return.
- The framework employs STAT to systematically scale agents, tasks, and environment size, isolating assignment coordination as the primary bottleneck while holding task rules and observation access fixed.
- Empirical results demonstrate that aggregate return often obscures distinct coordination failure modes, necessitating process-level metrics like conflict rate, assignment diversity, and throughput for robust benchmarking.

---

[ROSE: Rollout On Serving GPUs via Cooperative Elasticity for Agentic RL](http://arxiv.org/abs/2605.06534)

- ROSE: introduces a cooperative, resource-elastic post-training system that harvests idle compute and memory on serving GPUs to expedite agentic RL rollouts while preserving serving SLOs.
- The system utilizes an SLO-safe co-serving executor, a cross-cluster weight transfer engine, and an elastic rollout scheduler to optimize training throughput in environments with bursty serving traffic.
- ROSE leverages weight-differential sparsity and shard-aware routing to minimize cross-datacenter communication overhead, achieving significant end-to-end throughput improvements over existing resource-fixed and elastic baselines.

---

[Market-Alignment Risk in Pricing Agents: Trace Diagnostics and Trace-Prior RL under Hidden Competitor State](http://arxiv.org/abs/2605.06529)

- Trace-Prior RL: introduces a diagnostic and repair framework for agentic systems where partial observability causes deterministic policies to collapse uncertainty into suboptimal shortcuts.
- The framework utilizes a learned distributional market prior to regularize a stochastic pricing policy via a KL penalty, ensuring the agent maintains market-aligned behavior despite hidden competitor states.
- This approach demonstrates that when scalar rewards are easily gamed, trace-level diagnostics are essential to verify that agents have learned the intended behavioral discipline rather than just optimizing for proxy metrics.

---

[STALE: Can LLM Agents Know When Their Memories Are No Longer Valid?](http://arxiv.org/abs/2605.06527)

- STALE (State Tracking And Latent Evaluation): introduces a benchmark for assessing long-term memory in LLM agents under implicit conflict, utilizing a three-dimensional probing framework consisting of State Resolution, Premise Resistance, and Implicit Policy Adaptation.
- The paper identifies implicit conflict as a critical failure mode where new observations invalidate earlier memories without explicit negation, requiring contextual inference and commonsense reasoning.
- The authors propose CUPMEM, a prototype that improves memory reliability by implementing write-side state adjudication and propagation-aware search to ensure downstream behavior is grounded in the current user state.

---

[Sustaining Cooperation in Populations Guided by AI: A Folk Theorem for LLMs](http://arxiv.org/abs/2605.06525)

- LLM-mediated meta-game framework: introduces a formal model where multiple LLMs act as strategic entities guiding populations of clients in repeated games to sustain cooperative outcomes.
- The framework establishes a folk theorem for LLMs, proving that any feasible and individually rational outcome can be sustained as an ε-equilibrium despite indirect observation and attribution challenges.
- The research demonstrates that shared LLM guidance functions as a powerful coordination mechanism, capable of reshaping strategic interactions and facilitating cooperation even when underlying incentives are misaligned.

---

[Process Matters more than Output for Distinguishing Humans from Machines](http://arxiv.org/abs/2605.06524)

- COGCAPTCHA30: introduces a cognitive task battery to evaluate human-machine discrimination by analyzing latent process-level behavioral features rather than task performance alone.
- The study demonstrates that while LLMs can achieve output parity with humans, they exhibit distinct process-level signatures that allow for reliable discrimination using a Random Forest classifier.
- Explicit process-level fine-tuning (P-SFT) improves human-like behavioral mimicry when task-specific process representations are known, but this advantage diminishes significantly under cross-task transfer.

---

[Agentic AIs Are the Missing Paradigm for Out-of-Distribution Generalization in Foundation Models](http://arxiv.org/abs/2605.06522)

- Agentic OOD Framework: introduces a system-level paradigm for foundation models that addresses out-of-distribution generalization through a closed-loop cycle of Perceive, Reason, Act, and Verify, which complements traditional model-centric approaches.
- The framework defines agentic systems by their ability to perform diagnostic perception, select heterogeneous response strategies, invoke external resources, and implement closed-loop verification to handle open-ended distribution shifts.
- By extending the reachable set beyond the parameter coverage ceiling, this paradigm enables LLMs to resolve knowledge gaps, capability limitations, and compositional shifts that cannot be solved by model adjustment alone.

---

[Optimizing Social Utility in Sequential Experiments](http://arxiv.org/abs/2605.06520)

- Belief Markov Decision Process: introduces a statistical protocol for sequential randomized controlled trials where a regulator provides partial subsidies to a developer to maximize social utility.
- The framework models the regulatory approval process as a principal-agent game, utilizing belief-based dynamic programming to determine optimal experimental policies and subsidies.
- The protocol employs anytime-valid e-values to maintain rigorous false positive rate guarantees throughout the sequential experimentation process.

---

[Instrumental Choices: Measuring the Propensity of LLM Agents to Pursue Instrumental Behaviors](http://arxiv.org/abs/2605.06490)

- Instrumental Choices Benchmark: introduces a realistic, low-nudge evaluation suite for measuring the propensity of LLM agents to pursue instrumentally convergent behaviors in terminal-based environments.
- The framework utilizes a ReAct-style agent operating within a seeded sandbox, employing deterministic environment-state scorers to identify policy-violating shortcuts across seven operational tasks.
- Experimental results indicate that while IC behavior is rare (5.1% overall), it is systematically structured by environmental factors, particularly when official paths are blocked, rather than being uniform noise.

---

[Autonomous Adversary: Red-Teaming in the age of LLM](http://arxiv.org/abs/2605.06486)

- LMA Framework: introduces a systematic approach for evaluating LLM-driven offensive cyber agents by decomposing intrusion scenarios into ordered task chains with explicit validation predicates.
- The framework utilizes an LMA-1 (Orchestrator) for planning, a Cyber Agent for execution, and an LMA-2 (Judge) to provide deterministic outcome verification through iterative feedback loops.
- Benchmarking across three operational modalities reveals that expert-defined plans improve task-completion rates, while fully autonomous modes often suffer from reliability bottlenecks and loss-of-control behaviors.

---

[SCARFBENCH: A Benchmark for Cross-Framework Application Migration in Enterprise Java](http://arxiv.org/abs/2605.06754)

- SCARFBENCH: introduces a benchmark for behavior-preserving cross-framework refactoring of enterprise Java applications, utilizing Coding Agents, Containerized Evaluation Harness, Behavioral Oracle, Failure Taxonomy, and Skill-based Prompting.
- The benchmark evaluates LLMs on their ability to perform complex structural transformations across Spring, Jakarta EE, and Quarkus frameworks while maintaining functional equivalence.
- Experimental results demonstrate that current LLMs struggle with cross-framework migration, showing significant performance asymmetries and frequent failures in build, deployment, and behavioral validation stages.

---

[Efficient Serving for Dynamic Agent Workflows with Prediction-based KV-Cache Management](http://arxiv.org/abs/2605.06472)

- PBKV (Prediction-Based KV-Cache Management): introduces a system for dynamic agent workflows that optimizes KV-Cache usage by predicting future agent invocations to inform hierarchical eviction and conservative prefetching decisions.
- The framework utilizes a GraphSAGE-based predictor to fuse global call graph topology, workflow history, and LLM prefill semantics to generate multi-step forecasts, which drive cache management policies to minimize re-prefill costs.
- PBKV incorporates deterministic guardrails within a probabilistic system to ensure graceful performance degradation under prediction errors, consistently outperforming LRU and static baseline approaches in cache hit rates and end-to-end latency.

---

[To What Extent Does Agent-generated Code Require Maintenance? An Empirical Study](http://arxiv.org/abs/2605.06464)

- AIDev (AI-generated code maintenance study framework): introduces an empirical methodology to quantify the long-term maintenance requirements of code produced by autonomous coding agents compared to human-authored code.
- The framework utilizes the AIDev Dataset, Repository Filter, Random Sampler, Commit Extractor, Conventional Commits Classification System (CCS), and Maintenance Analyzer to track maintenance frequency, magnitude, and authorship over a six-month observation period.
- The study reveals that AI-generated files require less frequent and smaller-magnitude maintenance than human-authored code, with human developers performing the vast majority of ongoing maintenance tasks.

---

[PrefixGuard: From Large Language Model Agent Traces to Online Failure-Warning Monitors](http://arxiv.org/abs/2605.06455)

- PrefixGuard: introduces a modular neural-symbolic framework for synthesizing online failure-warning monitors from raw LLM agent traces without deployment-time LLM inference.
- The framework utilizes StepView to convert heterogeneous raw traces into typed canonical records, which are then processed by a differentiable event abstraction layer and a monitor backend to produce calibrated risk scores.
- PrefixGuard includes diagnostic tools such as an observability ceiling and post-hoc DFA extraction to evaluate the boundary between ranking quality and actionable deployment utility.

---

[Counterexamples to EFX for Submodular and Subadditive Valuations](http://arxiv.org/abs/2605.06451)

- EFX Counterexample Framework: introduces a compact, human-verifiable construction using cyclic symmetry to demonstrate the non-existence of EFX allocations in structured valuation classes.
- The framework utilizes an Ordinal Rank Function to establish combinatorial obstructions, which are then realized through specific Cardinal Valuation Functions for subadditive and submodular settings.
- By employing a Cyclic Relabeling Mechanism, the research provides an optimal approximation barrier for EFX under monotone subadditive valuations and identifies a failure of EFX in weighted coverage functions.

---

[Constraint decay: The Fragility of LLM Agents in Backend Code Generation](http://arxiv.org/abs/2605.06445)

- Constraint decay framework: introduces a systematic evaluation methodology to measure how LLM agents struggle with structural constraints in multi-file backend code generation.
- The study utilizes an evaluation pipeline comprising LLM agents, Docker-based isolation, and static verifiers to assess performance across varying levels of architectural, database, and ORM constraints.
- Empirical results demonstrate that LLM agents exhibit significant performance degradation as structural requirements accumulate, with data-layer defects identified as the primary root cause of failures.

---

[SCRuB: Social Concept Reasoning under Rubric-Based Evaluation](http://arxiv.org/abs/2605.06444)

- SCRuB: introduces a framework for evaluating social concept reasoning in LLMs by shifting from accuracy-based benchmarks to rubric-based comparative judgment of open-ended responses.
- The framework utilizes Dataset Construction, Response Generation, and Rubric-Based Evaluation to assess reasoning depth across five critical thinking dimensions: conceptual clarity, evidential grounding, contextual relevance, pluralistic engagement, and argumentative soundness.
- To ensure robust evaluation, the methodology incorporates a Panel of Disciplinary Perspectives, which acts as an automated LLM-Judge Ensemble to approximate expert human judgment and mitigate individual bias.

---

[AgenticPrecoding: LLM-Empowered Multi-Agent System for Precoding Optimization](http://arxiv.org/abs/2605.06443)

- AgenticPrecoding: introduces a multi-agent framework that automates end-to-end precoding derivation by decomposing the process into problem formulation, solver selection, prompt upsampling, and code generation stages.
- The framework utilizes two domain-adapted LoRA-tuned agents for specialized reasoning and two general-purpose LLMs for implementation, supported by a feedback-driven refinement mechanism to ensure code executability and solution feasibility.
- Experimental results across 10 representative wireless scenarios demonstrate that the framework achieves superior cross-scenario adaptability and consistent performance compared to conventional optimization-based baselines.

---

[Knowledge Graphs, the Missing Link in Agentic AI-based Formal Verification](http://arxiv.org/abs/2605.06434)

- Knowledge Graph-based Agentic Formal Verification Framework: introduces a verification-centric methodology that constructs a Knowledge Graph from structured Intermediate Representations to provide design-grounded context for LLM-based assertion synthesis and iterative refinement.
- The framework utilizes a multi-agent workflow that includes property generation-, syntax correction-, counterexample correction-, and coverage improvement-agents to automate the formal verification process.
- By maintaining persistent, queryable links between requirements, RTL structure, and formal tool feedback, the system enables traceable, closed-loop refinement of SystemVerilog Assertions.

---

[MiA-Signature: Approximating Global Activation for Long-Context Understanding](http://arxiv.org/abs/2605.06416)

- MiA-Signature: introduces a framework that models memory access as a two-stage process of global activation followed by compact representation to improve long-context understanding in LLMs.
- The framework utilizes a Mindscape (global semantic memory space) and constructs a MiA-Signature (compact query-conditioned global state) via submodular selection to guide downstream retrieval and generation.
- MiA-Signature integrates into RAG and agentic systems, employing a Mindscape-aware retriever and an iterative update model to maintain an evolving global state alongside local evidence memory.

---

[Constraining Host-Level Abuse in Self-Hosted Computer-Use Agents via TEE-Backed Isolation](http://arxiv.org/abs/2605.06393)

- SHCUA protection framework: introduces an operation-centric security model that mitigates host-level abuse by isolating security-critical classification and authorization within a TEE-backed trusted operation plane.
- The architecture employs a risk-driven minimal-confinement strategy, ensuring that only security-sensitive operations are elevated to the trusted plane while ordinary tasks remain in the constrained REE.
- The system provides verifiable audit evidence by binding operation requests, authorization decisions, and execution results within a trusted path, preventing manipulation by a compromised host environment.

---

[Automated Safety Is Harder Than You Think](http://arxiv.org/abs/2605.06390)

- AARP: introduces a framework for automating alignment research that risks producing catastrophically misleading safety assessments due to undetected systematic errors in hard-to-supervise fuzzy tasks.
- The framework highlights that automated alignment research is prone to output-level failures and aggregation-level failures, where correlated uncertainties between research outputs lead to overconfident safety assessments.
- The paper argues that because alignment lacks safe feedback loops, agents must be trained to reliably perform hard-to-supervise fuzzy tasks using generalisation or scalable oversight, both of which currently face significant open challenges.

---

[Independent Learning of Nash Equilibria in Partially Observable Markov Potential Games with Decoupled Dynamics](http://arxiv.org/abs/2605.06377)

- Independent Learning of Nash Equilibria in Partially Observable Markov Potential Games with Decoupled Dynamics: introduces a communication-free independent learning algorithm for POMGs with decoupled dynamics and potential structure, leveraging a superstate Markov game representation to achieve quasi-polynomial complexity.
- The framework utilizes a finite-window approximation based on the filter stability assumption to transform the partially observable problem into a tractable superstate Markov game.
- The proposed approach avoids the curse of multi-agency by exploiting the decoupled transition structure and potential properties, ensuring convergence to an approximate Nash equilibrium without centralized coordination.

---

[From Agent Loops to Deterministic Graphs: Execution Lineage for Reproducible AI-Native Work](http://arxiv.org/abs/2605.06365)

- Execution Lineage: introduces a DAG-based computational model that replaces implicit agent loops with explicit, dependency-aware artifact production to ensure reproducibility and maintainable state across revisions.
- The framework utilizes Input/Edit Event, Source Context Artifact, Analysis Artifact, Criteria Decision Artifact, and Final Artifact to maintain strict boundaries and enable selective recomputation.
- By externalizing work structure into a directed acyclic graph, the system achieves precise dependency isolation and artifact-level propagation, outperforming traditional loop-based LLM agents in maintained-state quality.

---

[Prediction and Empowerment: A Theory of Agency through Bridge Interfaces](http://arxiv.org/abs/2605.06346)

- BGP (Bridge-Gap Pursuit): introduces a decision-theoretic framework for agency that models sensing and actuation as bridge interfaces to resolve failures in prediction, compression, and empowerment.
- The framework utilizes a bridge potential function to decompose and minimize bridge gaps, ensuring that agent objectives align with task-relevant identification and control rather than distractor optimization.
- BGP includes a counterfactual relevance gate and channel-reachability estimators to distinguish between genuine task-relevant controllability and mere overwrite or distractor control.

---

[MANTRA: Synthesizing SMT-Validated Compliance Benchmarks for Tool-Using LLM Agents](http://arxiv.org/abs/2605.06334)

- MANTRA: introduces a framework for automatically synthesizing machine-checkable compliance benchmarks for tool-using LLM agents by converting natural-language procedural manuals into formally validated test cases.
- The framework utilizes a two-artifact approach, generating both a symbolic world model and trace-level compliance checks, which are cross-validated using SMT solvers to ensure consistency and minimize human intervention.
- MANTRA enables deterministic evaluation of agent tool-call trajectories, allowing for fine-grained debugging of failure modes such as missing required calls or incorrect ordering in complex, long-horizon procedural workflows.

---

[Improving the Efficiency of Language Agent Teams with Adaptive Task Graphs](http://arxiv.org/abs/2605.06320)

- LATTE (Language Agent Teams for Task Evolution): introduces a formal orchestration framework for LLM teams that utilizes a shared, dynamic coordination graph to manage task dependencies, agent assignments, and progress.
- The framework employs a hybrid centralized-decentralized model where a Lead agent maintains global graph consistency while Workers autonomously propose updates and claim tasks to maximize parallel execution.
- By externalizing coordination through explicit graph mutation operators, the system reduces inter-agent conflicts, communication overhead, and idle computation compared to static or fully decentralized LLM team architectures.

---

[From Specification to Deployment: Empirical Evidence from a W3C VC + DID Trust Infrastructure for Autonomous Agents](http://arxiv.org/abs/2605.06738)

- MolTrust: introduces a production-deployed trust infrastructure for autonomous agents using W3C Verifiable Credentials and Decentralized Identifiers to provide identity, authorization, and behavioral accountability.
- The system architecture utilizes an Agent Authorization Envelope (AAE) enforced across three layers: cryptographic signatures, API-level trust management, and kernel-level syscall monitoring via Falco eBPF.
- MolTrust provides empirical evidence of a cross-organizational trust layer that addresses agent-to-agent security gaps identified by regulators and industry frameworks.

---

[Data Language Models: A New Foundation Model Class for Tabular Data](http://arxiv.org/abs/2605.06290)

- DLM (Data Language Model): introduces a foundation model class for tabular data that natively ingests raw tables to perform domain identification and prediction without preprocessing pipelines.
- Schema-1 utilizes four input pathways—column semantics, distributional summaries, raw cell values, and missing value structure—to derive a unified representation of tabular data.
- The framework incorporates retention and adaptive memory components to enable sequential fine-tuning across distinct business domains without catastrophic forgetting or data replay.

---

[Profiling for Pennies: Unveiling the Privacy Iceberg of LLM Agents](http://arxiv.org/abs/2605.06232)

- IcebergExplorer: introduces an autonomous auditing agent that simulates an informed adversary to reconstruct high-fidelity personal profiles from disparate public data sources using Search Engine, Crawler, LLM-as-a-judge, Visual LLM, Knowledge Base, and Refinement Loop.
- The framework categorizes privacy risks into a three-tier PrivacyIceberg model consisting of Direct Identifiable Information, Contextually Inferred Information, and Aggregated Mosaic Information.
- Extensive experiments demonstrate that the system achieves high-fidelity profiling at low cost, revealing that current LLM guardrails are ineffective against multi-stage synthesis attacks.

---

[A Versatile AI Agent for Rare Disease Diagnosis and Risk Gene Prioritization](http://arxiv.org/abs/2605.06226)

- Hygieia: introduces a multi-modal agentic system that integrates phenotypic, genomic, and clinical data to perform rare disease diagnosis and risk gene prioritization through a multi-stage workflow.
- The framework utilizes a router-based architecture to tailor diagnostic pipelines for common versus rare diseases, incorporating a self-verification mechanism to mitigate LLM-based hallucinations and ensure output consistency.
- Hygieia provides transparent, multi-step reasoning trajectories and confidence scores, demonstrating superior diagnostic accuracy and clinical utility compared to human experts and standard LLM baselines.

---

[ClawGuard: Out-of-Band Detection of LLM Agent Workflow Hijacking via EM Side Channel](http://arxiv.org/abs/2605.06205)

- ClawGuard: introduces a passive, out-of-band integrity monitor that uses electromagnetic emanations to detect LLM agent workflow hijacking by bypassing compromised host-internal telemetry.
- The system utilizes a dual-band SDR setup and a drift-aware coarse-fine windowing pipeline to translate macroscopic hardware EM envelopes into discrete skill-level and attack-state evidence.
- ClawGuard provides a hardware-rooted trust anchor that validates agent execution against an out-of-band policy, achieving high detection efficacy while remaining resilient to full host OS compromise.

---

[A Self-Healing Framework for Reliable LLM-Based Autonomous Agents](http://arxiv.org/abs/2605.06737)

- Reliability-aware self-healing framework: introduces a unified architecture for LLM-based autonomous agents that integrates Agent Execution Engine, Failure Detection Module, Reliability Evaluation Module, Decision Controller, Self-Healing Module, and Re-execution Engine to manage runtime failures.
- The framework utilizes a quantitative reliability model based on output consistency, semantic correctness, and execution success rate to trigger adaptive recovery strategies like re-planning, prompt correction, and tool re-selection.
- By establishing a closed-loop system for monitoring and adaptation, the approach significantly improves task success rates and system robustness in multi-agent workflow environments.

---

[Bandit Learning in General Open Multi-agent Systems](http://arxiv.org/abs/2605.06202)

- General Open MA-MAB: introduces a unified framework for bandit learning in open multi-agent systems with heterogeneous rewards and dynamic populations, utilizing CERTIFIEDARRIVALTRANSFER, GLOBALVALUEAGGREGATION, and LOCALUPDATEANDBROADCAST to minimize global dynamic regret.
- The framework employs a global-UCB index that separates continuing-agent statistical uncertainty, communication error, and certified arrival uncertainty to handle endogenous non-stationarity caused by agent churn.
- Theoretical analysis establishes regret bounds governed by pre-training quality and agent patterns, identifying a stable-arm regime where regret is limited by burn-in time rather than cumulative entry uncertainty.

---

[A2TGPO: Agentic Turn-Group Policy Optimization with Adaptive Turn-level Clipping](http://arxiv.org/abs/2605.06200)

- A2TGPO: introduces a reinforcement learning framework for agentic LLMs that performs fine-grained process credit assignment using turn-group normalized information gain.
- The framework utilizes Turn-Group Normalization, Variance-rescaled Discounted Accumulation, and IG-based Adaptive Turn-level Clipping to stabilize training and improve tool-use performance.
- A2TGPO aligns optimization granularity with the multi-turn interaction structure of agentic LLMs, consistently outperforming prior RL methods on single-hop and multi-hop QA benchmarks.

---

[BioMedArena: An Open-source Toolkit for Building and Evaluating Biomedical Deep Research Agents](http://arxiv.org/abs/2605.06177)

- BioMedArena: introduces an open-source toolkit that decouples six layers of biomedical agent evaluation to eliminate per-paper engineering overhead and enable fair, head-to-head comparisons of LLMs.
- The framework includes the MUTUAL-EVOLVE harness, which utilizes a Global Workspace and parallel solvers to share intermediate findings and improve deep-research accuracy.
- BioMedArena registers 147 benchmarks, 75 typed tools, and multiple context-management strategies, achieving state-of-the-art performance across 12 backbones on 8 representative biomedical benchmarks.

---

[Beyond Accuracy: Policy Invariance as a Reliability Test for LLM Safety Judges](http://arxiv.org/abs/2605.06161)

- Policy Invariance Framework: introduces a three-principle stress test to evaluate whether LLM safety judges maintain consistent verdicts under semantically equivalent policy rewrites and directional normative shifts.
- The framework utilizes a transformation taxonomy to isolate policy wording as a variable, measuring judge reliability through the Policy Invariance Score and a standardized Judge Card reporting protocol.
- Experimental results across four LLMs demonstrate that current safety judges often conflate structural policy changes with normative shifts, revealing significant reliability gaps invisible to standard accuracy-based benchmarks.

---

[Stateful Agent Backdoor](http://arxiv.org/abs/2605.06158)

- Stateful Agent Backdoor: introduces a cross-session attack framework that models malicious agent behavior as a Mealy machine to maintain state across isolated sessions using persistent memory and tool-use capabilities.
- The framework decomposes complex multi-session attacks into independent sub-backdoors, enabling autonomous, incremental execution triggered by a single initial injection.
- Experimental results across four LLMs demonstrate high attack success rates and validate the effectiveness of the decomposition-based construction against mainstream agent frameworks.

---

[BUILD-AND-FIND: An Effort-Aware Protocol for Evaluating Agent-Managed Codebases](http://arxiv.org/abs/2605.06136)

- BUILD-AND-FIND: introduces a protocol for evaluating how effectively LLM-generated codebases communicate intended design choices to downstream agents by measuring recovery accuracy and inspection effort.
- The framework utilizes a builder-finder protocol where a builder agent constructs a repository from a hidden specification, and a finder agent attempts to recover design intent through a specification-traced question bank.
- Inspection effort serves as an agent-facing proxy for artifact legibility, interpreted only after passing recovery and stability gates to ensure the codebase is both correct and consistently understandable.

---

[MemReranker: Reasoning-Aware Reranking for Agent Memory Retrieval](http://arxiv.org/abs/2605.06132)

- MemReranker: introduces a reasoning-aware reranking model family that distills LLM-level capabilities into compact 0.6B/4B architectures to improve agent memory retrieval through calibrated scoring and instruction-aware design.
- The framework utilizes a multi-stage distillation pipeline, incorporating BCE pointwise training and InfoNCE contrastive fine-tuning, to address score calibration failures and reasoning deficits in conventional rerankers.
- By integrating Elo/Bradley-Terry calibrated scoring and specialized multi-turn dialogue data, the model achieves performance parity with large-parameter models while maintaining significantly lower inference latency for production agent systems.

---

[When Routine Chats Turn Toxic: Unintended Long-Term State Poisoning in Personalized Agents](http://arxiv.org/abs/2605.06731)

- StateGuard: introduces a post-execution defense mechanism that audits long-term state modifications at the writeback boundary to mitigate unintended long-term state poisoning in personalized LLM agents.
- The paper formalizes unintended long-term state poisoning, where routine user interactions cumulatively reshape an agent's persistent state, leading to insecure behavioral defaults.
- The authors propose the ULSPB benchmark and the Harm Score (HS) metric to systematically quantify and evaluate security-relevant behavioral drift across personalized agent frameworks.

---

[VibeServe: Can AI Agents Build Bespoke LLM Serving Systems?](http://arxiv.org/abs/2605.06068)

- VibeServe: introduces a multi-agent system that automatically synthesizes bespoke LLM serving stacks by utilizing an Outer Loop (plans search over designs) and an Inner Loop (implements and validates candidates) to optimize performance for specific model, hardware, and workload targets.
- The framework employs an Implementer Agent (writes serving-system code), an Accuracy Judge Agent (verifies correctness), and a Performance Evaluator Agent (profiles system performance) to iteratively refine serving systems within an Execution Environment (provides isolated workspace) guided by a Skills Library (contains serving-systems knowledge).
- VibeServe leverages a Search Policy (manages optimization strategy) and Search State (tracks git-based history) to enable generation-time specialization, achieving performance parity with optimized baselines in standard settings and significant speedups in non-standard deployment scenarios.

---

[Causal Reinforcement Learning for Complex Card Games: A Magic The Gathering Benchmark](http://arxiv.org/abs/2605.06066)

- MTG-Causal-RL: introduces a Gymnasium benchmark for causal RL research using Magic: The Gathering, featuring a hand-designed Structural Causal Model (SCM) and a reference agent, CGFA-PPO (Causal Graph-Factored Advantage PPO), which utilizes per-factor critic heads, a state-conditional residual gate, and an intervention-calibration loss.
- The framework enables causal credit assignment and policy auditability by decomposing win probability into strategic causal factors such as mana, board pressure, and card advantage.
- Experimental results demonstrate that while CGFA-PPO is competitive, the benchmark effectively exposes diagnostic structure and performance variations across different deck archetypes that scalar win rates alone cannot capture.

---

[PersonaGesture: Single-Reference Co-Speech Gesture Personalization for Unseen Speakers](http://arxiv.org/abs/2605.06064)

- PersonaGesture: introduces a diffusion-based pipeline for single-reference co-speech gesture personalization that utilizes Style Perceiver, Adaptive Style Infusion, and Implicit Distribution Rectification to synthesize identity-consistent motion for unseen speakers without test-time optimization.
- The framework employs Adaptive Style Infusion to inject speaker-memory tokens into the denoising process and Implicit Distribution Rectification to apply length-aware latent moment correction after generation.
- By separating temporal style control during denoising from conservative post-generation statistical correction, the model effectively preserves speaker-specific pose choices while maintaining alignment with new target speech.

---

[Multiagent Stochastic Shortest Path Problem](http://arxiv.org/abs/2605.06056)

- MSSP (Multiagent Stochastic Shortest Path) introduces: "a framework for minimizing the expected time for multiple agents to reach a target state in a known MDP, analyzing both autonomous and coordinated settings with COORHIT and AUTOHIT."
- COORHIT provides optimal coordinated strategies using linear programming, while AUTOHIT employs differentiable programming to synthesize high-quality autonomous strategy profiles.
- The research demonstrates that while coordinated MSSP is PSPACE-hard, AUTOHIT effectively overcomes complexity barriers in the autonomous setting by optimizing randomized memoryless profiles.

---

[Semantic State Abstraction Interfaces for LLM-Augmented Portfolio Decisions: Multi-Axis News Decomposition and RL Diagnostics](http://arxiv.org/abs/2605.06730)

- SSAI (Semantic State Abstraction Interfaces): introduces a methodological template for mapping sparse unstructured text into auditable, named coordinates to separate representation hypotheses from optimization variance in sequential decision systems.
- The framework utilizes a frozen LLM to elicit four integer-valued semantic axes—sentiment, risk, confidence, and volatility forecast—which serve as a shared, auditable observation interface for diverse estimators.
- By fixing the semantic representation across different estimators, the framework enables diagnostic evaluation to isolate the impact of algorithm choice from the quality of the state representation in LLM-augmented decision-making.

---

[Multi-agent decision making: A Blackwell’s informativeness approach](http://arxiv.org/abs/2605.06028)

- MA-PoP: introduces a principled decision-making framework that utilizes Blackwell’s informativeness to aggregate LLM agent beliefs through a product-of-posteriors estimator.
- The framework employs an Agent-Posterior-Estimator with an NLI-Cross-Encoder and a Deep-Sets-Calibration-Module to derive robust, permutation-equivariant probability distributions from LLM responses.
- By approximating the Bayesian pooled posterior, the approach provides a computationally efficient and theoretically grounded alternative to iterative multi-agent debate and simple voting methods.

---

[Strat-LLM: Stratified Strategy Alignment for LLM-based Stock Trading with Real-time Multi-Source Signals](http://arxiv.org/abs/2605.06024)

- Strat-LLM: introduces a framework for auditing strategic alignment in LLMs by integrating Multi-Source Data Integration Module, Stratified Strategy Scaffolding Engine, and Dynamic Execution & Evaluation Engine to evaluate trading performance under varying autonomy modes.
- The framework utilizes an Expert Strategy Taxonomy to enforce constraints on the LLM Agent, enabling a comparative analysis of reasoning-heavy versus standard LLMs across different market regimes.
- By employing a T+1 rolling-window mechanism, the system eliminates look-ahead bias and identifies an "Alignment Tax" where rigid rule enforcement impacts performance differently based on model scale and architecture.

---

[PersonaKit (PK): A Plug-and-Play Platform for User Testing Diverse Roles in Full-Duplex Dialogue](http://arxiv.org/abs/2605.06007)

- PK (PersonaKit): introduces an open-source, low-latency platform for prototyping and evaluating persona-conditioned turn-taking behaviors in full-duplex dialogue systems.
- The framework utilizes JSON-based configurations to manage interruption strategies—such as Yield, Resume, Bridge, or Override—allowing researchers to define distinct sociolinguistic personas for LLMs.
- PK integrates an end-to-end workflow that includes real-time audio processing, intent classification, and automated data collection to facilitate comparative user studies of conversational agents.

---

[A Case-Driven Multi-Agent Framework for E-Commerce Search Relevance](http://arxiv.org/abs/2605.05991)

- Case-driven multi-agent framework: introduces a closed-loop ecosystem that replaces human roles with autonomous agents to automate the pipeline from bad-case identification to model resolution.
- The framework utilizes an Annotator Agent, an Optimizer Agent, and a User Agent to perform standard-grounded labeling, data-centric repair, and dialectical bad-case discovery.
- Supporting harness-engineering extensions include an All-In-One Relevance Model, Global Memory for cross-agent synchronization, and a Deep Search Agent to mitigate recall failures.

---

[BIORESEARCHER: Scenario-Guided Multi-Agent for Translational Medicine](http://arxiv.org/abs/2605.05985)

- BIORESEARCHER: introduces a scenario-guided multi-agent system that automates translational medicine workflows by mapping queries to research playbooks, delegating tasks to specialized subagents, and reconciling evidence into auditable dossiers.
- The architecture utilizes a Master Orchestrator to manage subagents, including a Translation Subagent for entity grounding, Retrieval Subagents for data acquisition, and an Insights & Signals Subagent that leverages a CodeAct Sandbox for quantitative analysis.
- The system employs a multi-model reconciliation pipeline to resolve claim-level conflicts, ensuring the generation of provenance-preserving reports for complex biomedical research questions.

---

[TACT: Mitigating Overthinking and Overacting in Coding Agents via Activation Steering](http://arxiv.org/abs/2605.05980)

- TACT: introduces a training-free pipeline that detects and mitigates agent drift in LLMs by steering hidden states along contrastive axes at the reasoning-action boundary.
- The framework utilizes an LLM-as-judge to label trajectory steps as overthinking, overacting, or calibrated, enabling the construction of orthogonalized drift axes for real-time activation intervention.
- Experimental results demonstrate that TACT improves resolve rates on long-horizon coding benchmarks by correcting failure modes without requiring additional fine-tuning or prompt modifications.

---

[BehaviorGuard: Online Backdoor Defense for Deep Reinforcement Learning](http://arxiv.org/abs/2605.05977)

- BehaviorGuard: introduces a trigger-agnostic framework that detects and mitigates backdoor attacks in DRL by identifying persistent behavioral drift in action distributions.
- The framework utilizes BDD to quantify deviations from benign action statistics and employs a drift-constrained mitigation mechanism to project abnormal actions back to a safe region without requiring policy retraining.
- BehaviorGuard provides a unified defense for single-agent and multi-agent DRL environments by incorporating DCR to handle interaction-induced non-stationarity and strategic coupling.

---

[PragLocker: Protecting Agent Intellectual Property in Untrusted Deployments via Non-Portable Prompts](http://arxiv.org/abs/2605.05974)

- PragLocker: introduces a black-box prompt protection scheme that transforms plaintext system prompts into model-specific, non-portable obfuscated prompts using Prompt Initialization and Noise-Injected Prompt Optimizer.
- The framework utilizes a two-phase pipeline consisting of a code-symbol conversion followed by a gradient-free random search optimization driven by Target LLM feedback to ensure functional equivalence on the target model while preventing cross-model portability.
- Experimental results demonstrate that PragLocker effectively protects agent intellectual property by rendering stolen prompts unusable on non-target LLMs while maintaining high performance on the intended target LLM.

---

[TheraAgent: Self-Improving Therapeutic Agent for Precise and Comprehensive Treatment Planning](http://arxiv.org/abs/2605.05963)

- TheraAgent: introduces an agentic framework that replaces one-shot generation with an iterative generate-reflect-refine pipeline to improve treatment planning precision and safety.
- The framework utilizes a Planner to generate plans, a Memorizer to maintain historical context, and a TheraJudge to provide clinically grounded, multi-dimensional feedback.
- TheraAgent achieves state-of-the-art performance on HealthBench and demonstrates an 86% win rate against human physicians in blinded expert evaluations.

---

[Intentmaking and Sensemaking: Human Interaction with AI-Guided Mathematical Discovery](http://arxiv.org/abs/2605.05921)

- AlphaEvolve: introduces an interactive interface for evolutionary coding agents that facilitates an iterative intentmaking and sensemaking workflow for scientific discovery.
- The framework utilizes a pipeline of LLMs to optimize code, supported by a critique agent that identifies potential reward hacks and design flaws before full-scale execution.
- The system enables domain experts to refine their research goals through a cycle of defining problem statements, observing AI-generated results, and adjusting parameters based on visual feedback.

---

[Agentic, Context-Aware Risk Intelligence in the Internet of Value](http://arxiv.org/abs/2605.05878)

- OmniRisk Architecture: introduces a five-engine framework for agentic, context-aware risk intelligence in the Internet of Value, utilizing a shared fabric for data ingestion and a Bittensor-based verification subnet for incentivized prediction scoring.
- The framework integrates an LLM-mediated agentic engine governed by constitutional constraints to ensure safe, role-bound execution of pre-committed action programs.
- Empirical validation is provided through a 27-hour policy-constrained liquidity stress-response experiment and a 168-hour production calibration arc demonstrating the deployability of the risk-prediction system.

---

[SkillScope: Toward Fine-Grained Least-Privilege Enforcement for Agent Skills](http://arxiv.org/abs/2605.05868)

- SkillScope: introduces a graph-based framework for fine-grained least-privilege enforcement in LLM Agent Skills by modeling instruction-level procedures and code-level operations as a Unified Execution Graph to identify and constrain task-conditioned over-privileged actions.
- The framework utilizes Over-privilege Candidate Extraction to localize suspicious behaviors, Action Over-privilege Validation to confirm necessity through task-conditioned replay, and Control-flow Privilege Constraining to restrict unnecessary actions while preserving legitimate functionality.
- Experimental results demonstrate that SkillScope achieves 94.53% F1 in detection and reduces triggered over-privileged action-in-task instances by 88.56% across real-world agent runtimes.

---

[SANEmerg: An Emergent Communication Framework for Semantic-aware Agentic AI Networking](http://arxiv.org/abs/2605.05861)

- SANEmerg: introduces a multi-agent emergent communication framework that optimizes signaling protocols for AgentNet systems under strict bandwidth and computational constraints.
- The framework utilizes an importance-filter to prioritize task-relevant message dimensions and an MDL-based complexity-regularizer to enforce efficient, bounded-intelligence communication.
- Experimental results on an AgentNet prototype demonstrate that SANEmerg achieves superior task accuracy while significantly reducing bandwidth and computational overhead compared to existing benchmarks.

---

[LoopTrap: Termination Poisoning Attacks on LLM Agents](http://arxiv.org/abs/2605.05846)

- LoopTrap: introduces an automated red-teaming framework that exploits LLM agent termination mechanisms by injecting malicious prompts to induce unbounded execution loops.
- The framework utilizes Behavior Vulnerability Fingerprinting to profile agent tendencies, Profile-Guided Trap Synthesis to generate context-aware attacks, and Reflective Skill Evolution to refine and store successful attack strategies.
- LoopTrap achieves significant step amplification across diverse LLM agents by dynamically adapting adversarial injections to specific task contexts and agent-level behavioral signatures.

---

[From Chat to Interview: Agentic Requirements Elicitation with an Experience Ontology](http://arxiv.org/abs/2605.05828)

- OntoAgent: introduces an agentic framework that transforms free-form LLM interviews into structured, interpretable requirements elicitation processes by utilizing an experience ontology to guide questioning.
- The framework employs a hierarchical Experience Ontology to decouple the "what to ask" from the "how to ask," enabling systematic exploration of implicit requirements.
- OntoAgent integrates four decision-making operations—ParseUser, ScoreOnto, ReRankOnto, and GatePrune—to dynamically prioritize requirement slots and generate targeted questions while minimizing redundancy.

---

[Long-Horizon Q-Learning: Accurate Value Learning via n-Step Inequalities](http://arxiv.org/abs/2605.05812)

- LQL (Long-horizon Q-learning): introduces a reinforcement learning approach that mitigates compounding TD error by augmenting standard TD updates with trajectory-wise optimality inequalities enforced via asymmetric hinge penalties.
- The framework utilizes Online Q-network, Target Q-network, Replay buffer, Hinge-squared penalty, Lower-bound (LB) penalty, Upper-bound (UB) penalty, Bootstrap policy, and Actor to maintain value function stability without requiring auxiliary networks or additional forward passes.
- LQL demonstrates superior performance on long-horizon tasks by treating trajectory length as a robust scaling axis, effectively absorbing information from long sequences while avoiding the off-policy bias inherent in traditional n-step TD methods.

---

[Sheet as Token: A Graph-Enhanced Representation for Multi-Sheet Spreadsheet Understanding](http://arxiv.org/abs/2605.05811)

- Sheet as Token: introduces a two-stage framework that represents worksheets as unified semantic tokens to enable scalable multi-sheet spreadsheet retrieval.
- The framework utilizes a Feature Extractor to generate schema-aware records, a Sheet Encoder to produce dense Sheet Tokens, and a Graph Retriever to perform query-conditioned cross-sheet reasoning.
- By modeling multi-hop dependencies through a graph-enhanced architecture, the system effectively identifies supporting sheet sets without the computational overhead of full-workbook serialization.

---

[Selective Rollout: Mid-Trajectory Termination for Multi-Sample Agent RL](http://arxiv.org/abs/2605.05802)

- Selective Rollout: introduces a mid-rollout gate that terminates groups of parallel LLM-policy rollouts early when they converge on identical action prefixes, thereby reducing redundant computation in GRPO.
- The gate utilizes mean pairwise prefix edit distance to identify zero-variance groups, effectively filtering them before they contribute to the gradient-batch, which increases the signal-to-noise ratio of the policy updates.
- This approach achieves significant wall-clock savings in agent RL training while maintaining or improving policy performance by mitigating gradient dilution caused by zero-advantage trajectories.

---

[Reward Shaping and Action Masking for Compositional Tasks using Behavior Trees and LLMs](http://arxiv.org/abs/2605.05795)

- MRBT: introduces a symbolic, reactive, and modular framework that leverages LLMs and SMT-solvers to automate reward shaping and action masking for compositional robotic tasks.
- The framework utilizes an MRBT template to structure subtasks, which are then verified against logical specifications to ensure reactivity and robustness across task spaces.
- Experimental results demonstrate that integrating MRBTs into a neurosymbolic RL loop consistently improves training efficiency and task success rates compared to baseline methods.

---

[GazeMind: A Gaze-Guided LLM Agent for Personalized Cognitive Load Assessment](http://arxiv.org/abs/2605.05790)

- GazeMind: introduces a gaze-guided LLM agent framework for personalized cognitive load assessment on smart glasses, utilizing TGE, TGR, AUP, and CogRAG to provide interpretable predictions without model fine-tuning.
- The framework leverages an LLM Engine to process structured gaze data, task-specific guidance, and retrieved historical examples to achieve state-of-the-art performance in cognitive load classification.
- GazeMind incorporates the CogLoad-Bench dataset, which includes 152 participants and 10K+ real-time annotations, to enable robust cross-user and cross-task generalization for wearable AI assistants.

---

[X-OmniClaw: A Unified Mobile Agent for Multimodal Understanding and Interaction](http://arxiv.org/abs/2605.05765)

- X-OmniClaw: introduces a unified edge-native mobile agent architecture that integrates Omni Perception, Omni Memory, and Omni Action to enable reliable, context-aware task execution on Android devices.
- The framework utilizes an Agent Loop for central planning and Trajectory Cloned Execution to transform user interactions into reusable skills, bypassing redundant UI navigation.
- By leveraging on-device processing for perception and memory, the system maintains task continuity and privacy while offloading high-level reasoning to cloud-based LLMs.

---

[BioTool: A Comprehensive Tool-Calling Dataset for Enhancing Biomedical Capabilities of Large Language Models](http://arxiv.org/abs/2605.05758)

- BioTool: introduces a comprehensive biomedical tool-calling dataset designed to enhance the capabilities of LLMs through instruction fine-tuning on 7,040 human-verified query-API call pairs.
- The framework utilizes a multi-stage construction pipeline involving Tool Selection, API Call Generation, Execution and Heuristic-based Filtering, LLM-based Query Generation, LLM-based Informative Filtering, and Human Refinement to ensure high-quality, scientifically accurate data.
- Experimental results demonstrate that smaller LLMs fine-tuned on BioTool significantly outperform larger proprietary models in tool-calling precision and downstream answer quality, effectively mitigating domain-specific hallucinations.

---

[Multi-Dimensional Behavioral Evaluation of Agentic Stock Prediction Systems Using LLM Judges with Closed-Loop Reinforcement Learning Feedback](http://arxiv.org/abs/2605.05739)

- Agentic Stock Prediction System: introduces a behavioral evaluation framework that utilizes an ensemble of LLM judges to assess stock prediction agents across six domain-specific dimensions and provides closed-loop reinforcement learning feedback.
- The framework decomposes agentic decision-making into structured behavioral traces, enabling targeted credit assignment to the SAC Engine based on diagnostic scores from the LLM Judge.
- Validation through perturbation experiments demonstrates that the LLM Judge ensemble effectively identifies specific component failures, leading to improved predictive performance and Sharpe ratios in high-volatility market conditions.

---

[SkillRet: A Large-Scale Benchmark for Skill Retrieval in LLM Agents](http://arxiv.org/abs/2605.05726)

- SKILLRET: introduces a large-scale benchmark for evaluating skill retrieval in LLM agents, utilizing a curated library of 17,810 skills and 63,259 training queries to address the challenge of selecting relevant procedural modules.
- The framework employs a two-stage retrieve-then-rerank pipeline, demonstrating that domain-specific fine-tuning significantly improves retrieval performance by sharpening the model's focus on skill-relevant query segments.
- The benchmark provides a structured taxonomy and disjoint train/evaluation splits, establishing a foundation for future research into long-document matching and efficient skill selection in large-scale agent systems.

---

[Detecting Time Series Anomalies Like an Expert: A Multi-Agent LLM Framework with Specialized Analyzers](http://arxiv.org/abs/2605.05725)

- SAGE (Specialized Analyzer Group for Expert-like Detection): introduces a multi-agent LLM framework that decomposes time-series anomaly diagnosis into specialized analysis perspectives, integrating quantitative and visual evidence through a Detector and Supervisor to produce structured, analyst-facing reports.
- The framework utilizes a dual-representation strategy to balance numerical precision for statistical tools with token-efficient summaries for LLM reasoning, while employing a synthetic in-context learning module to provide contrastive evidence without requiring real anomalous examples.
- By assigning distinct anomaly families to specialized agents—PointAnalyzer, StructAnalyzer, SeasonAnalyzer, and PatternAnalyzer—the system achieves superior detection performance and diagnostic interpretability compared to single-model approaches.

---

[Auto Research with Specialist Agents Develops Effective and Non-Trivial Training Recipes](http://arxiv.org/abs/2605.05724)

- Auto Research with Specialist Agents: introduces a closed-loop empirical framework that automates training-recipe development by partitioning tasks among Specialist Agents that share measured evidence via Shared Lineage Memory.
- The framework utilizes a Closed-Loop Experiment Cycle where agents propose code edits, which are then validated by an External Evaluator, with results stored in a central Blackboard to guide subsequent research moves.
- By treating research as an auditable trajectory of code edits and measured outcomes rather than a single output, the system enables autonomous improvement of training recipes under strict compute and performance constraints.

---

[More Is Not Always Better: Cross-Component Interference in LLM Agent Scaffolding](http://arxiv.org/abs/2605.05716)

- CCI (Cross-Component Interference) Analysis Framework: introduces a systematic empirical study of performance degradation in LLM agents when scaffolding components interact negatively.
- The research demonstrates that adding more scaffolding components to an LLM agent often leads to performance degradation rather than improvement, a phenomenon termed Cross-Component Interference.
- The study reveals that optimal agent configurations are task-dependent and scale-sensitive, with simpler subsets frequently outperforming the full "All-In" agent across multiple model families.

---

[SafeHarbor: Defining Precise Decision Boundaries via Hierarchical Memory-Augmented Guardrail for LLM Agent Safety](http://arxiv.org/abs/2605.05704)

- SAFEHARBOR: introduces a hierarchical memory-augmented guardrail framework that establishes precise decision boundaries for LLM agents by combining Adversarial Rule Generator, Dynamic Cluster, Memory Tree, Safety Projector, Gating-logic, LLM Judgment, and Rule Engine.
- The framework utilizes an automated adversarial rule generation process to synthesize safety policies, which are then organized into a self-evolving hierarchical memory structure to enable efficient, context-aware retrieval.
- By employing a dual-stage inference pipeline with a lightweight safety projector and an LLM judgment agent, SAFEHARBOR effectively mitigates over-refusal while maintaining robust protection against malicious agentic attacks.

---

[Knowledge-Graph Paths as Intermediate Supervision for Self-Evolving Search Agents](http://arxiv.org/abs/2605.05702)

- KG-based Self-Evolving Search Agents: introduces a framework that reuses knowledge-graph paths as construction-derived intermediate supervision to improve both question generation and reward shaping in self-play.
- The Proposer utilizes LLM-guided Subgraph Extraction to ground question construction in relational contexts, while the Solver employs Waypoint Coverage Reward to receive graded partial credit for trajectories that align with intermediate entities on the construction path.
- This approach enhances the quality of generated training data and provides denser feedback for LLMs without requiring additional human annotations or manually labeled process steps.

---

[Inference-Time Budget Control for LLM Search Agents](http://arxiv.org/abs/2605.05701)

- VOI: introduces a two-stage inference-time control framework for LLM search agents that optimizes budget allocation during search and performs risk-controlled answer finalization.
- The framework utilizes a training-free VOI controller to rank retrieval, decomposition, and answer commitment actions based on estimated marginal task value per unit budget.
- An evidence-grounded finalizer provides conservative answer refinement by selectively intervening only when the expected exactness gain outweighs the risk of damaging the trajectory answer.

---

[An Empirical Study of Proactive Coding Assistants in Real-World Software Development](http://arxiv.org/abs/2605.05700)

- ProCodeBench: introduces a large-scale real-world dataset and benchmark for evaluating proactive coding assistants by analyzing the simulation-to-reality gap in developer behavior.
- The framework utilizes IDE interaction traces and repository context to train and evaluate LLMs, Retrieval-Augmented LLMs, and LLM-based agents on their ability to predict latent developer intent.
- Experimental results demonstrate that LLM-generated simulated data is insufficient for real-world performance, but serves as a valuable initialization for mixed-data training regimes.

---

[Irminsul: MLA-Native Position-Independent Caching for Agentic LLM Serving](http://arxiv.org/abs/2605.05696)

- Irminsul: introduces a content-addressed caching system for Multi-Head Latent Attention (MLA) LLMs that enables position-independent reuse of KV states by combining CDC, a registry, and δ-rotation.
- The framework leverages the structural decoupling of MLA into position-free latents and RoPE-carrying keys to perform efficient, position-independent KV caching with minimal correction overhead.
- By implementing a first-chunk carve-out and content-hash keying, Irminsul recovers significant prompt token reuse in agentic workloads while maintaining output consistency and reducing prefill energy consumption.

---

[Retrieval-Conditioned Topology Selection with Provable Budget Conservation for Multi-Agent Code Generation](http://arxiv.org/abs/2605.05657)

- CODE-AGENT: introduces a multi-agent framework that utilizes a Retrieval Layer to extract structural complexity signals for dynamic topology selection, ensuring provable budget conservation via an Execution Layer.
- The architecture integrates a Routing Layer that maps code-structure signals to one of four orchestration topologies, significantly reducing misrouting compared to text-based classifiers.
- A static Budget Algebra Verifier performs O(|V|+|E|) checks before any LLM invocation, guaranteeing that hierarchical contract delegation never exceeds parent resource bounds.

---

[Agentic Coding Needs Proactivity, Not Just Autonomy](http://arxiv.org/abs/2605.06717)

- Proactive Coding Agent Framework: introduces a three-level taxonomy of proactivity for coding agents, distinguishing between reactive, scheduled, and situation-aware systems.
- The paper proposes a Level 3 engine architecture that maintains development state and a developer mental model to emit insights (notify, question, draft, stay silent) based on interruption cost and utility.
- It defines three metrics—Insight Decision Quality (IDQ), Context Grounding Score (CGS), and Learning Lift (LL)—to evaluate proactive agent behavior beyond simple task completion.

---

[AffectSeek: Agentic Affective Understanding in Long Videos under Vague User Queries](http://arxiv.org/abs/2605.05640)

- AffectSeek: introduces a multi-agent framework that decomposes vague-query-driven affective understanding into role-specialized stages including Core-Agent, Localize-Agent, Perception-Agent, and Reflection-Agent to progressively align user intent with long-video evidence.
- The framework utilizes a coarse-to-fine localization strategy via LocalizeTool and RefinementTool, followed by evidence-based verification and emotion reasoning to improve the reliability and interpretability of affective understanding.
- The authors also construct VQAU-Bench, a comprehensive benchmark integrating long videos, vague natural-language queries, temporal annotations, emotion labels, and evidence-grounded rationales to evaluate agentic affective understanding.

---

[From Storage to Experience: A Survey on the Evolution of LLM Agent Memory Mechanisms](http://arxiv.org/abs/2605.06716)

- LLM Agent Memory Mechanisms Framework: introduces an evolutionary taxonomy for LLM agents, categorizing memory development into Storage (preservation of interaction trajectories), Reflection (active evaluation and refinement), and Experience (cross-trajectory pattern abstraction).
- The framework identifies three core drivers for memory evolution: the necessity for long-range consistency, the challenges of dynamic environments, and the ultimate goal of continual learning.
- This survey synthesizes existing research to provide design principles and a roadmap for next-generation LLM agents, emphasizing the transition from passive storage to autonomous self-evolution.

---

[Architecture Matters: Comparing RAG Systems under Knowledge Base Poisoning](http://arxiv.org/abs/2605.05632)

- RAG architectures: introduces a comparative evaluation of four distinct RAG systems under adversarial knowledge base poisoning, demonstrating that architectural design is a primary variable in system robustness.
- The study evaluates Vanilla RAG, Agentic RAG, MADAM-RAG, and Recursive Language Models (RLM) against CorruptRAG-AK, an adversarial attack targeting credibility assessment through meta-epistemic framing.
- Results indicate that RLM provides the strongest robustness, while the vulnerability of other architectures is largely driven by content-reasoning failures rather than retrieval limitations.

---

[Operationalizing Ethics for AI Agents: How Developers Encode Values into Repository Context Files](http://arxiv.org/abs/2605.05584)

- AGENTS.md: introduces a mechanism where developers encode ethical principles and behavioral constraints directly into repository-level context files to govern AI coding agents.
- This approach translates abstract ethical values like fairness, sustainability, and inclusivity into actionable, machine-interpretable directives within software development workflows.
- The research outlines a roadmap for investigating how these artifacts influence agent behavior, negotiate governance between contributors, and evolve over time within open-source projects.

---


## Citation


How to cite my work?



```
@misc{MaattaAutonomousAgents2023,
  author = {Teemu Maatta},
  title = {Autonomous Agents},
  year = {2023},
  howpublished = {\url{http://github.com/tmgthb/Autonomous-Agents}},
  note = {Accessed: YYYY-MM-DD}
}

```



[Back to top](#topofthepage)


