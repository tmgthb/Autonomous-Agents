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

## Research papers: 2026 5/5

[2026 (5/5)](http://github.com/tmgthb/Autonomous-Agents/blob/main/README.md), [2026 (4/5)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2026_4.md), [2026 (3/5)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2026_3.md), [2026 (2/5)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2026_2.md), [2026 (1/5)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2026_1.md), [2025 (4/4)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2025_4.md),[2025 (3/4)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2025_3.md), [2025 (2/4)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2025_2.md), [2025 (1/4)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2025_01.md), [2024](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2024.md), [2023](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2023.md), [Earlier](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_Earlier.md)



Chronological order. 





</div>



---


#### 15th May 2026


[paper.json: A Coordination Convention for LLM-Agent-Actionable Papers](http://arxiv.org/abs/2605.16194)

- paper.json: introduces a lightweight coordination convention for academic papers that enables LLMs to accurately parse sub-claims, definitions, and reproducibility commands via a companion JSON file.
- The framework utilizes stable identifiers for claims, definitions, and theorems, alongside explicit non-claim sections to mitigate common LLM hallucination and scope-overextension errors.
- By providing exact shell commands for figure generation and a machine-readable read-receipt protocol, the convention shifts reproducibility from a codebase property to a verifiable paper property.

---


[ColPackAgent: Agent-Skill-Guided Hard-Particle Monte Carlo Workflows for Colloidal Packing](http://arxiv.org/abs/2605.15625)

- ColPackAgent: introduces an agent framework that autonomously executes structured hard-particle Monte Carlo simulation workflows for colloidal packing by separating domain-specific simulation tools from procedural agent skills.
- The framework utilizes the Model Context Protocol to expose simulation operations as schema-validated tool calls, ensuring reliable execution across different LLMs and agent platforms.
- ColPackAgent supports interactive, autonomous, and autoresearch modes, demonstrating robust performance in mapping phase transitions and providing a stage-aware benchmark for evaluating LLM reliability in scientific workflows.

---


[Task-Semantic Graph-Driven Distributed Agent Networking for Underwater Target Tracking](http://arxiv.org/abs/2605.15528)

- STG-MAPPO: introduces a semantic task graph-driven framework that integrates task-level diagnostics and communication-aware neighbor summaries to enable stable decentralized target tracking for AUV swarms.
- The framework utilizes a velocity-level action interface to map high-level cooperative decisions to executable six-degree-of-freedom AUV control inputs, effectively reducing the complexity of low-level force/torque exploration.
- By encoding task phases, observation confidence, and link availability into a compact semantic graph, the approach ensures persistent target tracking and robust performance under communication-constrained underwater conditions.

---


[A Generative-AI Framework for Intelligent Utility Billing, CO₂ Analytics and Sustainable Resource Optimisation](http://arxiv.org/abs/2605.16250)

- Generative-AI Utility-Billing Framework: integrates Data Acquisition, Preprocessing &amp; Feature Engineering, Generative-AI Bill Generation Agent, Transformer Consumption Forecaster, CO₂ Estimator, Simulated-Bifurcation Tariff &amp; Load Optimiser, and Reporting to automate utility billing and resource management.
- The framework utilizes a transformer-based forecaster for consumption estimation and a quantum-inspired Simulated-Bifurcation solver to optimize demand-response schedules on classical hardware.
- A constrained-decoding policy and post-generation auditor are implemented within the LLM-based bill generation agent to ensure factual consistency and prevent numeric hallucinations in customer communications.

---


[Agentic AI and Human-in-the-Loop Interventions: Field Experimental Evidence from Alibaba’s Customer Service Operations](http://arxiv.org/abs/2605.14830)

- Agentic AI and Human-in-the-Loop Interventions: introduces a field experimental study on Alibaba's platform to evaluate how human-in-the-loop interventions shape service outcomes when Agentic AI (autonomous service task executor) encounters cognitive or emotional failures.
- The system integrates an Agentic AI (autonomous service task executor) with Human Supervisor (human-in-the-loop intervention agent) roles, supported by Monitoring Algorithms (risk detection tools) including Sentiment Monitoring (customer frustration detection) and Intention Monitoring (customer intent shift tracking).
- The research demonstrates that while Agentic AI (autonomous service task executor) improves service speed, human intervention effectiveness is highly dependent on the timing and nature of the failure, with emotional escalations proving significantly harder to recover due to reduced human engagement.

---

[FORGE: Self-Evolving Agent Memory With No Weight Updates via Population Broadcast](http://arxiv.org/abs/2605.16233)

- FORGE: introduces a staged, population-based protocol that evolves prompt-injected natural-language memory for hierarchical ReAct agents without requiring gradient updates.
- The framework utilizes an inner Reflexion-style loop to synthesize knowledge artifacts from failed trajectories, which are then propagated across parallel instances via champion broadcast and stabilized through graduation-based early stopping.
- FORGE improves decision-making in stochastic, long-horizon cyber-defense environments by reducing major failure rates and compressing performance variance across diverse LLM families.

---

[Argus: Evidence Assembly for Scalable Deep Research Agents](http://arxiv.org/abs/2605.16217)

- Argus: introduces a multi-agent system that utilizes a Searcher and a Navigator to perform deep research by assembling evidence into a structured directed acyclic graph.
- The Navigator identifies missing information and dispatches Searchers to target specific gaps, enabling efficient evidence gathering and reducing redundant parallel rollouts.
- By decoupling the Navigator's reasoning context from the number of Searchers, the framework achieves scalable performance and provides fully auditable, source-traced answers.

---

[Confirming Correct, Missing the Rest: LLM Tutoring Agents Struggle Where Feedback Matters Most](http://arxiv.org/abs/2605.16207)

- KG-grounded LLM Tutoring Framework: introduces a benchmark for evaluating LLM-based tutoring agents in propositional logic using a knowledge graph to provide ground truth for three-way diagnosis of student solutions.
- The framework utilizes Student Simulator Agents, Peer Feedback Agents, Teacher Feedback Agents, and Judge Feedback Agents to analyze diagnostic precision and pedagogical feedback quality across different information-access conditions.
- The research demonstrates that while LLMs reliably confirm optimal steps, they struggle with valid-alternative and incorrect solutions, highlighting the necessity of hybrid architectures that integrate KG-grounded diagnostic mechanisms with LLM-based scaffolding.

---

[Context, Reasoning, and Hierarchy: A Cost-Performance Study of Compound LLM Agent Design in an Adversarial POMDP](http://arxiv.org/abs/2605.16205)

- Compound LLM Agent Design Framework: introduces a controlled empirical study of compound LLM agent design in an adversarial POMDP, evaluating the interactions between context engineering, deliberation, and hierarchical decomposition.
- The study reveals that programmatic state abstraction provides the most cost-effective performance gains, while distributing deliberation tools across a hierarchy triggers a destructive deliberation cascade that degrades performance and inflates token costs.
- The research establishes that bounded hierarchical decomposition without distributed deliberation achieves the best absolute performance, emphasizing that system-level information flow is more critical than individual agent reasoning depth.

---

[Formal Methods Meet LLMs: Auditing, Monitoring, and Intervention for Compliance of Advanced AI Systems](http://arxiv.org/abs/2605.16198)

- TRAC (Temporal Rule Assessment and Compliance): introduces a framework for auditing and monitoring LLM-based systems by combining formal LTL specifications with machine learning-based labeling and predictive intervention.
- The framework utilizes LTL progression to evaluate temporally extended behavioral constraints, enabling real-time monitoring and retrospective auditing of black-box AI systems.
- Experimental results demonstrate that TRAC significantly outperforms standalone LLM-as-a-Judge methods in detecting violations, while predictive monitors effectively reduce violation rates without sacrificing task performance.

---

[Optimized Three-Dimensional Photovoltaic Structures with LLM guided Tree Search](http://arxiv.org/abs/2605.16191)

- ERA (Empirical Research Assistance): introduces an iterative framework combining a coding agent and LLM-driven tree search to optimize three-dimensional photovoltaic structures while mitigating algorithmic reward hacking.
- The system utilizes AntiGravity to patch the physics engine and scoring function, ensuring that generated designs adhere to physical constraints like structural connectivity and occlusion accuracy.
- By employing BFS-based validation and iterative refinement, the framework successfully discovers high-efficiency solar geometries that outperform human-designed baselines under various material constraints.

---

[An Algebraic Exposition of the Theory of Dyadic Morality](http://arxiv.org/abs/2605.16153)

- TDM (Theory of Dyadic Morality): introduces an algebraic formalization of moral judgment using structural causal modeling to represent human cognition as a two-node template involving an intentional agent and a vulnerable patient.
- The framework incorporates psychological operators including typecasting, completion, and valence-dependent inference to model how humans compute moral judgments under constraints and handle multi-node scalability through node collapse and sequential processing.
- This approach enables neurosymbolic AI systems to perform computationally rigorous moral reasoning by reframing safety policies from fixed enumeration to patient-centric protection against suffering.

---

[Look Before You Leap: Autonomous Exploration for LLM Agents](http://arxiv.org/abs/2605.16143)

- Explore-then-Act paradigm: introduces a training and inference framework that mitigates premature exploitation in LLMs by explicitly optimizing for autonomous environment exploration using verifiable rewards.
- The framework utilizes ECC to quantify the discovery of environment states, objects, and affordances, which are then used to guide an interleaved GRPO training process.
- By decoupling information-gathering from task execution, the agent builds a grounded knowledge summary that significantly improves downstream performance and robustness in unfamiliar environments.

---

[Surrogate Neural Architecture Codesign Package (SNAC-Pack)](http://arxiv.org/abs/2605.16138)

- SNAC-Pack: introduces an open-source AutoML framework for hardware-aware neural architecture codesign and end-to-end FPGA deployment using Optuna, NSGA-II, and hardware surrogate models.
- The framework integrates a global search loop with hardware surrogate models for resource and latency estimation, followed by a local search stage utilizing QAT and iterative magnitude pruning for FPGA synthesis via hls4ml.
- SNAC-Pack supports an optional MCP agentic frontend to automate the pipeline, significantly reducing design space exploration time for resource-constrained tasks like jet classification and qubit readout.

---

[ShopGym: An Integrated Framework for Realistic Simulation and Scalable Benchmarking of E-Commerce Web Agents](http://arxiv.org/abs/2605.16116)

- ShopGym: introduces an integrated framework for realistic simulation and scalable benchmarking of e-commerce web agents, utilizing ShopArena (simulation environment generation pipeline), ShopGuru (grounded task generation pipeline), Planning Agent (decomposes exploration into subtasks), Specification Agent (writes anonymized design specifications), Execution Agent (edits codebase in verification loop), Verifiers (rule-based and multimodal validation), and LLM-as-Judge (evaluates agent trajectory success).
- The framework employs a multi-agent pipeline to transform live seed storefronts into self-contained, inspectable, and resettable sandbox environments while synthesizing grounded tasks for evaluation.
- Experimental results demonstrate that synthetic shops preserve key structural properties of live storefronts, with agent performance on synthetic environments positively correlated with performance on live sites.

---

[Multi-Agent Cooperative Transportation: Optimal and Efficient Task Allocation and Path Finding](http://arxiv.org/abs/2605.16097)

- CT-TCBS (Cooperative Transportation Task Conflict-Based Search): introduces a two-level search framework for solving the Cooperative Transportation Task Allocation and Path Finding (CT-TAPF) problem by integrating High-Level Search, Low-Level Pathfinding, Incremental Expansion, Conflict Expansion, Task Expansion, Heuristic Function, and Task Selector Layer.
- The framework utilizes an Incremental Expansion strategy to manage the combinatorial explosion of team formation by breaking the assignment process into prioritized steps.
- Suboptimal variants employ a global Task Selector Layer using Best-Task or Worst-Task heuristics to establish a more efficient runtime-quality frontier compared to agent-centric baselines.

---

[VideoSeeker: Incentivizing Instance-level Video Understanding via Native Agentic Tool Invocation](http://arxiv.org/abs/2605.16079)

- VideoSeeker: introduces an agentic paradigm for instance-level video understanding that integrates proactive perception and tool invocation to overcome the limitations of text-only prompts.
- The framework utilizes a four-stage automated data synthesis pipeline and a two-stage training strategy, including SFT and agentic RL, to internalize tool-calling capabilities into the base model.
- By employing perception tools like view_visual_prompt and crop_video, the model achieves precise spatiotemporal localization and reasoning, significantly outperforming existing baselines on instance-level video understanding tasks.

---

[Ada-Diffuser: Latent-Aware Adaptive Diffusion for Decision-Making](http://arxiv.org/abs/2605.16054)

- Ada-Diffuser: introduces a unified generative framework that explicitly incorporates latent dynamic inference into decision-making using a Latent Factor Identification Block, Causal Diffusion Model, Autoregressive Denoising Schedule, Denoise-and-Refine Mechanism, Zig-Zag Sampling, and Inverse Dynamics Model.
- The framework leverages theoretical identifiability results to perform block-wise latent inference from minimal temporal observations, enabling adaptive planning and policy learning in partially observable environments.
- Ada-Diffuser employs a denoise-and-refine mechanism and zig-zag sampling to reduce posterior mismatch, ensuring high-quality latent context recovery and improved performance across diverse locomotion and robotic manipulation benchmarks.

---

[RecMem: Recurrence-based Memory Consolidation for Efficient and Effective Long-Running LLM Agents](http://arxiv.org/abs/2605.16045)

- RecMem: introduces a three-tier memory architecture that reduces LLM token consumption by deferring memory consolidation until sustained interaction recurrence is observed.
- The framework utilizes a subconscious memory layer for lightweight storage, an episodic memory for event-level narratives, and a semantic memory for fine-grained facts, all managed through a recurrence-driven trigger.
- RecMem incorporates a semantic refinement mechanism to recover critical details omitted during episodic abstraction, ensuring high accuracy while significantly lowering the computational cost of long-running LLM agents.

---

[Who Owns This Agent? Tracing AI Agents Back to Their Owners](http://arxiv.org/abs/2605.16035)

- Agent Attribution Framework: introduces a canary-based protocol to link harmful agent interactions to the responsible operator account at a vendor-hosted LLM.
- The protocol utilizes Lexical Canary and Semantic Canary constructions to bridge the visibility gap between external agent behavior and vendor-side session logs.
- By leveraging a utility-evasion tradeoff, the framework ensures that adversarial attempts to suppress canaries inherently degrade the agent's task performance.

---

[ScreenSearch: Uncertainty-Aware OS Exploration](http://arxiv.org/abs/2605.16024)

- ScreenSearch: introduces an ambiguity-aware desktop exploration system that combines structural screen retrieval, deduplication, and a PUCT graph-bandit to navigate partial observability in OS environments.
- The framework utilizes UIA Tree, Screen Featurizer, Screen Index &amp; Embedding Store, Global MCTS State Store, Trajectory &amp; Artifact Store, PUCT Graph-Bandit, and Worker Pool to manage state identity and drive exploration through frontier expansion and ambiguity reduction.
- By treating GUI exploration as a search problem over a shared deduplicated state graph, the system effectively mitigates premature commitment in visually aliased desktop states.

---

[Learning Bilevel Policies over Symbolic World Models for Long-Horizon Planning](http://arxiv.org/abs/2605.15975)

- BISON: introduces a bilevel policy framework that combines symbolic high-level reasoning with neural low-level execution to solve long-horizon planning problems.
- The framework utilizes goal regression and inductive generalisation to derive interpretable, first-order symbolic HL policies from demonstrations, which are then realized by a compact GNN-based LL policy.
- BISON demonstrates superior generalization to long horizons and large numbers of objects compared to end-to-end and VLA baselines while maintaining high training and inference efficiency.

---

[OHP-RL: Online Human Preference as Guidance in Reinforcement Learning for Robot Manipulation](http://arxiv.org/abs/2605.15971)

- OHP-RL: introduces a human-in-the-loop reinforcement learning framework that interprets human interventions as online preference signals to guide policy learning through a state-dependent preference gate.
- The framework utilizes an asynchronous actor-critic architecture with four distinct update modules to balance environment rewards with human-provided preference guidance.
- By adaptively regulating preference influence based on state-dependent advantages, OHP-RL improves sample efficiency and robustness in real-world robotic manipulation tasks.

---

[Deterministic Event-Graph Substrates as World Models for Counterfactual Reasoning](http://arxiv.org/abs/2605.15967)

- Event-Graph Substrate: introduces a world model architecture that represents agent memory as an append-only log of typed RDF triples to enable deterministic counterfactual reasoning.
- The framework utilizes a TBox, ABox, Typed Event Log, Deterministic Interpreter, and Intervention Vocabulary to perform causal-ancestor traversals and kinematic projections without learned components.
- This approach provides formal guarantees on inspectability and replay consistency, outperforming symbolic and parametric baselines on causal-reasoning benchmarks by leveraging structured execution over observed event logs.

---

[PAGER: Bridging the Semantic-Execution Gap in Point-Precise Geometric GUI Control](http://arxiv.org/abs/2605.15963)

- PAGER (Precision-Aware GEometric Reasoning): introduces a topology-aware agent that decomposes geometric construction into dependency-structured Planning Module, Task Execution Module, and precision-aligned RL Precision Optimization.
- The framework utilizes Pixel-Precise Data Construction and SFT Instruction Tuning to establish executable action grammar, mitigating the Semantic-Execution Gap in point-precise GUI tasks.
- PAGER incorporates a GeoGebra Action Executor and RL-based parameter accuracy rewards to ensure point-level spatial precision and robustness against cascading coordinate errors.

---

[PersonaFingerprint: Measuring Persona Inference on Modern Websites with LLM-Driven Browsing](http://arxiv.org/abs/2605.15962)

- PersonaFingerprint: introduces a multi-agent framework that leverages a persona-conditioned decision agent and a computer-use agent to generate labeled encrypted traffic for measuring persona inference risks.
- The framework utilizes a shared packet-window encoder with specialized heads to perform both website and persona fingerprinting, demonstrating that behavioral personas are learnable from encrypted metadata.
- The study reveals that existing website fingerprinting models contain incidental persona leakage, which can be amplified through joint multi-task learning or lightweight MLP probes.

---

[Dynamic Plasma Shape Control with Arbitrary Sensor Subsets](http://arxiv.org/abs/2605.15935)

- RL-based plasma shape control framework: introduces a reinforcement learning agent that achieves robust, real-time plasma shape control in tokamaks by utilizing an asymmetric actor-critic architecture and an auxiliary shape reconstruction head to handle partial observability and diagnostic failures.
- The agent employs diagnostic dropout during training to generalize across arbitrary sensor subsets without requiring explicit fault detection or controller switching.
- Experimental validation on the DIII-D tokamak demonstrates that the policy successfully commands coil actuators for dynamic shape maneuvers while maintaining performance comparable to classical controllers.

---

[Privacy is Fungibility: Why Endogenous Tokens Are Not Money](http://arxiv.org/abs/2605.15934)

- Token Security and Trust Locus Classification: introduces a framework to categorize blockchain assets based on whether their security and trust models are intrinsic to the ledger or derived from external institutions.
- The paper argues that most public, permissionless blockchain tokens function as credit rather than money due to their account-based nature, public visibility, and lack of obliviousness.
- By extending economic models of theft and credit, the authors demonstrate that the absence of privacy in current ledger designs makes these assets fundamentally incompatible with the definition of cash-like money.

---

[Agentic Discovery of Neural Architectures: AIRA-Compose and AIRA-Design](http://arxiv.org/abs/2605.15871)

- AIRA: introduces a dual-framework approach for autonomous neural architecture discovery using LLM-agents to navigate combinatorial design spaces and engineer mechanistic implementations.
- AIRA-Compose utilizes an ensemble of LLM-agents to search and optimize arrangements of computational primitives, while AIRA-Design tasks agents with writing novel attention mechanisms and training scripts from scratch.
- Both frameworks leverage the AIRS-Bench task standard and AIRA-dojo harness to enable recursive self-improvement, consistently outperforming hand-designed baselines and traditional NAS-found models.

---

[Access Timing as Scaffolding: A Reinforcement Learning Approach to GenAI in Education](http://arxiv.org/abs/2605.15850)

- RL-based GenAI Access Timing Framework: introduces a reinforcement learning agent that optimizes the timing of GenAI access to balance cognitive support with independent learning.
- The framework utilizes PPO and BKT to implement an adaptive policy that rewards task success, efficiency, metacognitive reflection, productive failure, and cognitive load management.
- Experimental results demonstrate that strategically timed GenAI access improves objective post-test performance and metacognitive accuracy compared to unrestricted access, while reducing task errors and time on task.

---

[ROADMAPBENCH: Evaluating Long-Horizon Agentic Software Development Across Version Upgrades](http://arxiv.org/abs/2605.15846)

- RoadmapBench: introduces a benchmark of 115 long-horizon coding tasks grounded in real open-source version upgrades, utilizing a Docker environment, repository snapshot, roadmap instruction, agent execution loop, agent code verification, static validation, and rollout-based quality control.
- The benchmark evaluates LLMs on multi-target software evolution, requiring agents to implement functionality across multiple files and modules with a median modification of 3,700 lines.
- Evaluation results demonstrate that even the strongest frontier models struggle with long-horizon development, with performance bottlenecks shifting from build errors in weaker models to implementation precision in stronger models.

---

[WorldAct: Activating Monolithic 3D Worlds into Interactive-Ready Object-Centric Scenes](http://arxiv.org/abs/2605.15843)

- WorldAct: introduces a framework that converts static, monolithic 3D Gaussian Splatting scenes into editable, interaction-ready environments by leveraging a Vision-Language Agent, SAM3, DiffuEraser, DepthLab, Poisson Reconstruction, SAM3D, ICP, and a Differentiable Renderer.
- The framework utilizes a Vision-Language Agent to automate object discovery and viewpoint selection, enabling the decomposition of scenes into independent object assets and a restored background.
- WorldAct supports downstream embodied AI tasks by generating collision-aware geometry and high-quality object assets, facilitating object-level editing, manipulation, and scene rearrangement.

---

[Exploration of k-edge-deficient temporal graphs in linear time](http://arxiv.org/abs/2605.15833)

- Temporal Exploration Framework: introduces a method for exploring k-edge-deficient temporal graphs in O(nk log k) time by reducing the problem to covering a depth-first search tour of a stable spanning tree.
- The framework utilizes a roundabout process to eliminate redundant virtual agents, ensuring that a small set of representative agents covers the entire graph structure efficiently.
- By leveraging Δ-temporal connectivity, the approach enables a single explorer to simulate the movements of these representative agents, achieving near-optimal linear-time performance for constant k.

---

[BootstrapAgent: Distilling Repository Setup into Reusable Agent Knowledge](http://arxiv.org/abs/2605.15815)

- BootstrapAgent: introduces a multi-agent framework that distills repository bootstrapping exploration into a persistent, verifiable, and agent-consumable .bootstrap contract.
- The framework utilizes a Discoverer Agent, Planner Agent, and Generator Agent to automate environment setup, while the Docker Verifier ensures reproducibility through clean replay and trace-driven Delta Repair.
- By persisting setup knowledge, BootstrapAgent significantly reduces downstream LLM token usage and build time for unfamiliar repositories.

---

[Toward Natural and Companionable Virtual Agents via Cross-Temporal Emotional Modeling](http://arxiv.org/abs/2605.15812)

- CTEM (Cross-Temporal Emotional Modeling): introduces a framework that links long-term behavioral history to moment-to-moment emotional expression through a closed-loop system of Psychologically-grounded Behavior Modeling, Episodic State, and Multi-Modal Message Generation.
- The framework utilizes a Physio-emotional State, Motivational Vector, and Memory to maintain an Adapted Personality that dynamically influences future behaviors and interaction styles.
- Auri, the companion agent instance, incorporates a Safety Layer with Dual Stage Detection and LLM Safety Guardrails to ensure ethical interaction while balancing stability and flexibility in long-term companionship.

---

[From Gridworlds to Warehouses: Adapting Lightweight One-shot Multi-Agent Pathfinding for AGVs](http://arxiv.org/abs/2605.15799)

- MAWPF: introduces a gridworld-based pathfinding formulation that incorporates differential-drive AGV kinodynamic constraints, including multi-step rotations, acceleration, deceleration, and follower collision avoidance.
- The framework adapts lightweight MAPF solvers—PP, LNS2, PIBT, and LaCAM—to the MAWPF environment using a rolling-horizon integration and a PIBT-based configuration generator to handle complex kinodynamic states.
- Empirical results demonstrate that the LaCAM+PIBT pipeline achieves high success rates and real-time performance for hundreds of agents, while the follower-collision constraint serves as an implicit congestion-avoidance mechanism.

---

[SaaS-Bench: Can Computer-Use Agents Leverage Real-World SaaS to Solve Professional Workflows?](http://arxiv.org/abs/2605.15777)

- SaaS-Bench: introduces a benchmark for evaluating Computer-Using Agents (CUAs) in realistic, deployable SaaS environments, utilizing Task Input, Agent, SaaS Apps, Deployment, Database, Execute, Browser Use, and Verify components to measure end-to-end workflow completion.
- The framework employs State-Check, Content-Check, and LLM-Judge to provide a systematic, multi-dimensional evaluation of agent performance across long-horizon, cross-application professional tasks.
- Experimental results demonstrate that current LLMs struggle with long-horizon execution, state tracking, and error recovery, with fewer than 4% of tasks completed end-to-end.

---

[Lamarckian Inheritance in Dynamic Environments: How Key Variables Affect Evolutionary Dynamics](http://arxiv.org/abs/2605.15769)

- Lamarckian Inheritance in Dynamic Environments: introduces a framework for co-optimizing robot morphology and control, utilizing Evolutionary Algorithm, Bayesian Optimization, Reinforcement Learning, Modular ANN-based Controller, Critic Network, and Direction Sensor.
- The study demonstrates that the efficacy of Lamarckian inheritance in dynamic environments is contingent upon the degree of environmental conflict and the predictability of environmental changes.
- The research finds that integrating a directional sensor restores the performance benefits of Lamarckian inheritance in conflicting environments by enabling agents to generalize control strategies.

---

[ALSO: Adversarial Online Strategy Optimization for Social Agents](http://arxiv.org/abs/2605.15768)

- ALSO: introduces an adversarial online strategy optimization framework for LLM-based social agents that dynamically adapts behavior through strategy selection without requiring offline retraining.
- The framework formulates multi-turn social interaction as an adversarial bandit problem, utilizing a lightweight neural surrogate to predict rewards and generalize feedback across semantically related strategies.
- By combining randomized bandit-based exploration with context-conditioned value estimation, the system enables robust, sample-efficient adaptation to non-stationary, co-evolving agent behaviors in social simulations.

---

[BioXArena: Benchmarking LLM Agents on Multi-Modal Biomedical Machine Learning Tasks](http://arxiv.org/abs/2605.15766)

- BioXArena: introduces a biomedical machine learning coding benchmark that evaluates whether LLM agents can create task-specific model-building code for heterogeneous, multi-modal biomedical datasets.
- The framework utilizes Expert-Driven Curation, Unified Public Task Capsules, Hidden Private Labels, Sandbox Runtime, Evaluation Metrics, and Analysis Output to assess agent performance across 76 tasks in 9 biomedical domains.
- BioXArena evaluates 11 agent configurations, including general coding LLMs, biomedical agents, and ML coding agents, to characterize performance under realistic computational constraints.

---

[Attribute-Grounded Selective Reasoning for Artwork Emotion Understanding with Multimodal Large Language Models](http://arxiv.org/abs/2605.15755)

- FAB-G (Formal-Attribute Bottleneck-Guided reasoning): introduces a supervised multi-agent framework that addresses attribute flooding in LLMs by decomposing artwork emotion understanding into attribute salience screening and cue-constrained emotional reasoning.
- The framework utilizes five specialized attribute agents to identify emotionally operative formal cues, which are then aggregated into a bottleneck to guide the final analysis agent's interpretation.
- By grounding explanations in human-salient attributes, FAB-G improves prediction accuracy, enhances explanation auditability, and produces more compact outputs compared to unconstrained LLM baselines.

---

[SMMBench: A Benchmark for Source-Distributed Multimodal Agent Memory](http://arxiv.org/abs/2605.15710)

- SMMBench: introduces a benchmark for evaluating whether agents can retrieve, align, and compose multimodal evidence scattered across independently originated sources rather than reasoning within a single curated context.
- The framework utilizes a Dataset Construction Pipeline to synthesize long-horizon conversational environments where answer-critical evidence is distributed across heterogeneous artifacts like chats, tables, and documents.
- Experimental results demonstrate that current LLMs struggle with source-distributed memory composition, particularly in conflict resolution, preference reasoning, and precise function calling.

---

[Differentiable Mixture-of-Agents Incentivizes Swarm Intelligence of Large Language Models](http://arxiv.org/abs/2605.15706)

- DMoA: introduces a self-evolving multi-agent framework that dynamically routes and activates agents at each reasoning step using a differentiable, context-aware mechanism.
- The framework utilizes a Sentence Transformer and RNN-Router to produce sparse agent activations, optimized via predictive entropy as a self-supervised signal.
- DMoA enables elastic, adaptive collaboration during inference, resolving static compilation limitations found in traditional multi-agent systems.

---

[H-MEM: A Novel Memory Mechanism for Evolving and Retrieving Agent Memory via a Hybrid Structure](http://arxiv.org/abs/2605.15701)

- H-MEM: introduces a hybrid memory mechanism that couples a temporal-semantic tree with a knowledge graph to model memory evolution and support multi-hop reasoning.
- The framework utilizes a Retrieval Planner to decompose queries, an Evidence Retriever to extract information from the hybrid structure, and a Generation Process to synthesize final answers.
- H-MEM achieves state-of-the-art performance on long-term memory benchmarks by enabling progressive consolidation of short-term memory into long-term summaries while maintaining entity-level relational dependencies.

---

[Distributed Zeroth-Order Policy Gradient for Networked Multi-agent Reinforcement Learning from Human Feedback](http://arxiv.org/abs/2605.15697)

- DZOPG introduces a distributed reinforcement learning framework that optimizes policies in networked multi-agent systems using human preference feedback instead of explicit reward signals.
- The framework utilizes spatiotemporally truncated trajectories and Gaussian-perturbed policy gradients to enable fully distributed learning under local communication constraints.
- Theoretical analysis establishes that the algorithm achieves polynomial sample complexity and converges to an ϵ-stationary point in infinite-horizon networked multi-agent settings.

---

[Rule2DRC: Benchmarking LLM Agents for DRC Script Synthesis with Execution-Guided Test Generation](http://arxiv.org/abs/2605.15669)

- Rule2DRC: introduces a large-scale benchmark for evaluating LLM agents in synthesizing executable DRC scripts from natural language rules using execution-based scoring.
- SplitTester: improves script selection by iteratively generating discriminative test layouts to cluster candidate scripts and identify the most functionally correct implementation.
- The framework leverages execution feedback from a DRC engine to guide LLM agents, effectively bypassing the limitations of surface-level code similarity metrics.

---

[PRISM: Prompt Reliability via Iterative Simulation and Monitoring for Enterprise Conversational AI](http://arxiv.org/abs/2605.15665)

- PRISM: introduces a closed-loop framework for continuous prompt reliability in enterprise conversational AI by integrating Test Generator, Platform Simulator, LLM-as-Judge, Diagnosis &amp; Repair, and Continuous Monitoring.
- The framework treats prompt engineering as a continuous reliability problem, using automated simulation and surgical repair to address both creation-time correctness and runtime LLM behavioral drift.
- PRISM achieves 99% production reliability and reduces prompt authoring time by 98% by automating the detection and repair of procedural compliance failures in multi-step conversational agents.

---

[PCASim: Promptable Closed-loop Adversarial Simulation for Urban Traffic Environment](http://arxiv.org/abs/2605.15654)

- PCASim: introduces a closed-loop simulation framework that leverages RAG-Augmented LLM, Adversarial Scenario Repository, and PPO-based Reinforcement Learning to generate and evaluate safety-critical urban traffic scenarios.
- The framework utilizes a Middleware to translate natural language descriptions into executable DSL, which is then refined by Bézier Curve Convex Optimization to ensure trajectory realism.
- By employing a Semantic Alignment Module and Self-Consistency Voting Mechanism, the system achieves high-fidelity scenario generation, enabling robust training of autonomous agents against adversarial behaviors.

---

[TopoEvo: A Topology-Aware Self-Evolving Multi-Agent Framework for Root Cause Analysis in Microservices](http://arxiv.org/abs/2605.15611)

- TopoEvo: introduces a topology-aware, reasoning-enhanced, and self-evolving framework for joint microservice root cause localization and fault type classification.
- The framework utilizes Metric-Anchored Orthogonal Multimodal Alignment and Vector Quantization to transform noisy telemetry into structured, auditable symptom tokens for reliable reasoning.
- TopoEvo employs a multi-agent Hypothesis–Evidence–Test workflow to mitigate symptom-amplification bias and incorporates a self-evolving mechanism to maintain robustness under non-stationary microservice conditions.

---

[See Before You Code: Learning Visual Priors for Spatially Aware Educational Animation Generation](http://arxiv.org/abs/2605.15585)

- OmniManim: introduces a render-feedback-aware framework that utilizes a Shared Scene State, Scene Agent, Vision Agent, Code Agent, and Repair Agent to generate high-quality educational animations.
- The framework employs a Vision Agent to predict sparse keyframe layouts using coarse-to-fine bounding-box denoising and interpolation-aware objectives to ensure visual stability.
- OmniManim incorporates a structured render-feedback loop that enables iterative refinement of animations based on explicit visual quality constraints rather than code-level properties alone.

---

[STAR: A Stage-attributed Triage and Repair framework for RCA Agents in Microservices](http://arxiv.org/abs/2605.15581)

- STAR (Stage-attributed Triage And Repair): introduces a process-centric reliability layer for LLM-based RCA agents that decomposes reasoning into four structured stages to enable targeted debugging and repair.
- The framework utilizes Stage-wise Audit and Diagnosis, Fast/Slow Routing, Decisive Stage Localization, Patch-and-Replay Repair, and Self-Evolving Repair Memory to systematically eliminate error propagation in microservice RCA workflows.
- By treating agent failures as stage-localizable bugs rather than monolithic errors, STAR significantly improves root cause localization and fault type classification accuracy across diverse LLM backbones.

---

[Response-Conditioned Parallel-to-Sequential Orchestration for Multi-Agent Systems](http://arxiv.org/abs/2605.15573)

- NEXA (Response-Conditioned Parallel-to-Sequential Orchestration): introduces a hybrid multi-agent framework that uses a Draft Generation Stage to produce initial responses, a Semantic Embedding Encoder to represent them, a Response-Conditioned Transformer Policy to predict a sparse communication DAG, a Sequential Propagation Module to refine responses, and a Weighted-Centroid Aggregator to select the final output.
- The framework bridges parallel and sequential execution by using the initial parallel draft as evidence to decide whether structured sequential refinement is necessary.
- NEXA achieves improved accuracy-cost tradeoffs and generalizability across tasks, agent counts, and model scales by learning a sparse, judge-free communication policy.

---

[Detecting Privilege Escalation in Polyglot Microservices via Agentic Program Analysis](http://arxiv.org/abs/2605.15569)

- NEO: introduces an agentic program analysis framework that combines LLM-based semantic reasoning with classic program analysis to detect privilege escalation vulnerabilities in polyglot microservices.
- The framework utilizes a set of unified code search primitives to enable scalable, language-agnostic context retrieval and cross-service data flow analysis.
- NEO validates potential vulnerabilities by combining LLM-based semantic assessment with SMT-based path constraint solving to minimize false positives.

---

[AstraFlow: Dataflow-Oriented Reinforcement Learning for Agentic LLMs](http://arxiv.org/abs/2605.15565)

- AstraFlow: introduces a dataflow-oriented RL system that replaces trainer-centered control with decoupled autonomous components to support complex multi-policy agentic workloads.
- The framework utilizes a Dataflow Layer for coordination, RaaS for scalable trajectory generation, and independent Trainers to enable elastic, heterogeneous, and cross-region training.
- AstraFlow achieves significant speedups in multi-policy collaborative training by enabling fully asynchronous execution and efficient sparse weight updates across distributed compute resources.

---

[TopoClaw: A Human-Centric and Topology-Aware Agent Operating System](http://arxiv.org/abs/2605.15556)

- TopoClaw: introduces a human-centric Agent OS that replaces agent-centric isolation with a decoupled runtime navigating physical device and social relationship topologies, utilizing Core Runtime Services, Physical Topology Routing, Social Topology Orchestration, Cross-Topology Boundary Defense Pipeline, and an Execution Plane.
- The architecture decouples intent generation from physical actuation, enabling agents to function as attributed Digital Twins that operate across distributed hardware and collaborative social spaces.
- TopoClaw ensures structural safety through a distributed, context-aware policy enforcement pipeline that governs agent actions across both physical and social trust boundaries.

---

[DRS-GUI: Dynamic Region Search for Training-Free GUI Grounding](http://arxiv.org/abs/2605.15542)

- DRS-GUI: introduces a training-free framework that enhances GUI grounding by dynamically searching for instruction-relevant regions before final prediction.
- The framework utilizes a UI Perceptor to generate semantic cues and an MCTS-based Action Planner to execute Focus, Shift, and Scatter actions for adaptive perceptual exploration.
- By evaluating candidate regions with a composite reward function, the system effectively prunes visual clutter and improves grounding robustness in high-resolution, dense interfaces.

---

[RTL-BenchMT: Dynamic Maintenance of RTL Generation Benchmark Through Agent-Assisted Analysis and Revision](http://arxiv.org/abs/2605.15537)

- RTL-BenchMT: introduces an agentic framework for the dynamic maintenance of RTL generation benchmarks, utilizing a Manager Agent, Failure Analysis Agent, Description Revision Agent, Description Review Agent, and Description Update Agent.
- The framework employs an iterative reasoning paradigm where agents perform thought, action, and observation cycles within an Agent Environment and an isolated EDA Environment to identify flawed cases and detect LLM overfitting.
- RTL-BenchMT systematically reduces human maintenance costs by automating the identification of flawed benchmark cases and quantifying overfitting through semantically equivalent description variations.

---

[STS: Efficient Sparse Attention with Speculative Token Sparsity](http://arxiv.org/abs/2605.15508)

- STS (Speculative Token Sparsity): introduces a training-free sparse attention mechanism that leverages a smaller draft model to generate predictive sparsity masks for a larger target model within a speculative decoding framework.
- The framework decouples mask generation from target model execution, enabling a "known-in-advance" property that facilitates asynchronous prefetching of KV-cache blocks to hide memory transfer latency.
- By maintaining a lossless KV-cache and applying fine-grained token-wise sparsity, the approach achieves significant speedups while preserving model accuracy across both prefill and decode stages.

---

[uGen: An Agentic Framework for Generating Microarchitectural Attack PoCs](http://arxiv.org/abs/2605.15503)

- uGen: introduces a RAG-empowered multi-agent framework that systematically assesses and improves the ability of LLMs to generate functional microarchitectural attack PoCs across diverse attack classes.
- The framework utilizes a multi-agent architecture comprising Programmer-, Reflector-, Gap Analyzer-, Synthesizer- and Feedback-agents to perform role-specific reasoning, tool-grounded execution, and iterative refinement.
- uGen addresses LLM limitations in microarchitectural understanding by injecting attack-specific knowledge through a hierarchical RAG system and validating PoCs using hardware-derived signals.

---

[Hybrid LLM-based Intelligent Framework for Robot Task Scheduling](http://arxiv.org/abs/2605.15486)

- Hybrid LLM-based Intelligent Framework for Robot Task Scheduling: introduces a two-tier LLM pipeline that utilizes a Generator Agent to draft construction task schedules and a Supervisor Agent to validate and repair them using minimal-edit projections.
- The framework employs a Generator Agent and a Supervisor Agent to ensure feasibility in multi-robot construction environments by enforcing typed constraints such as battery safety, precedence, and coverage.
- The system leverages structured prompts and few-shot programmatic exemplars to improve the executability of LLM-generated plans while maintaining traceability for human inspection.

---

#### 14th May 2026


[Why Neighborhoods Matter: Traversal Context and Provenance in Agentic GraphRAG](http://arxiv.org/abs/2605.15109)

- Agentic GraphRAG: introduces a trajectory-level evaluation framework for citation faithfulness in LLMs by analyzing the impact of graph traversal, structure, and visited-but-uncited entities on answer generation.
- The research utilizes a graph-ablation methodology to demonstrate that while cited evidence is often necessary for accuracy, it is not sufficient, as broader graph context significantly influences the reasoning process.
- The study reveals that citation faithfulness in agentic systems requires accounting for the entire retrieval trajectory, including visited nodes and structural cues, rather than relying solely on final output citations.

---


[Falkor-IRAC: Graph-Constrained Generation for Verified Legal Reasoning in Indian Judicial AI](http://arxiv.org/abs/2605.14665)

- Falkor-IRAC: introduces a graph-constrained generation framework that grounds legal reasoning in structured IRAC knowledge graphs to ensure verifiable outputs.
- The architecture utilizes a Retrieval Agent to extract path-guided context from FalkorDB, which the LLM Generator uses to propose answers subject to a hard veto by the Verifier Agent.
- By modeling litigation flow and doctrinal conflicts as first-class graph components, the system enforces strict citation grounding and detects unresolved legal splits.

---



[Computational Thinking Development in AI Agent Creation: A Mixed-Methods Study](http://arxiv.org/abs/2605.14330)

- CocoFlow: introduces a no-code platform for AI agent creation that utilizes a Module Library, Visual Canvas, Intent Recognition Module, Entity Extraction Module, Conditional Logic Nodes, Real-time Chat Simulator, and a Pre-trained NLU Engine to facilitate computational thinking development.
- The study demonstrates that iterative testing engagement within the platform significantly predicts self-efficacy gains among students.
- Research findings reveal an "Optimal Development Zone" where students with moderate initial computational thinking levels achieve the greatest developmental benefits compared to high- or low-level peers.

---

[Agentic AI Ecosystems in Higher Education: A Perspective on AI Agents to Emerging Inclusive, Agentic Multi-Agent AI Framework for Learning, Teaching and Institutional Intelligence](http://arxiv.org/abs/2605.14266)

- Agentic Multi-Agent AI Framework: introduces a unified, multi-stakeholder ecosystem that integrates specialized agents to support learning, teaching, and institutional processes through coordinated planning, reasoning, and adaptive decision-making.
- The framework utilizes a layered architecture comprising a User Interface Layer, a Coordination Layer, and a Data & Knowledge Layer to facilitate seamless interaction between Learning Agents, Teaching Agents, Inclusion Agents, and Institutional Agents.
- This approach addresses the fragmentation of current educational AI by embedding inclusive pedagogy and ecosystem-level intelligence into a scalable, human-aligned, and adaptive multi-agent architecture.

---


[Characterizing AI-Assisted Bot Traffic in Darknet Data: Implications for ICS and IIoT Security](http://arxiv.org/abs/2605.14209)

- Darknet Traffic Analysis Pipeline: introduces a modular framework for characterizing longitudinal darknet traffic to identify AI-assisted botnet reconnaissance targeting critical infrastructure.
- The framework utilizes Merit ORION Network Telescope (ingests darknet PCAP files), Data Ingestion &amp; Preprocessing (parses, filters, and normalizes packets), Feature Extraction (calculates packet rate, protocol, and port flags), Statistical Analysis (core evaluation engine), Burstiness (measures inter-arrival time distributions), Shannon Entropy (measures traffic diversity), IAT (inter-arrival time metrics), ICS Port Targeting (identifies industrial protocol activity), Geographic Attribution (maps source IP origins), and IDS Simulation (evaluates volumetric threshold evasion).
- The study demonstrates that modern botnets employ deliberate micro-pacing to evade standard volumetric IDS thresholds, necessitating a shift toward behavior-aware detection mechanisms.

---


[Guises and Perspectives: An Intentional and Hyperintensional Sketch](http://arxiv.org/abs/2605.15144)

- GL (Guise Logic): introduces a formal framework for intensional logic where guises serve as primary semantic objects to model intentional reference and internal relations.
- The system integrates Leibnizian containment semantics with an intentional operator and a modal layer to address hyperintensional phenomena like substitution failure and de se reference.
- The framework provides a flexible architecture supporting canonical, template-restricted, and finite models to balance cognitive realism with formal tractability.

---


[GraphFlow: An Architecture for Formally Verifiable Visual Workflows Enabling Reliable Agentic AI Automation](http://arxiv.org/abs/2605.14968)

- GraphFlow: introduces a visual workflow architecture that treats diagrams as executable specifications to enable formally verifiable and reliable agentic AI automation through Diagram-as-specification, Verified core, Durable runtime, Cohort search, Operational dashboards, Swimlanes, Event log, Compiler, and Proof assistant.
- The framework improves reliability by shifting agentic planning from ad hoc tool-use to selecting and parameterizing pre-approved, contract-checked workflows.
- GraphFlow isolates nondeterminism at explicit boundaries using swimlanes and ensures reproducibility through durable execution with deterministic replay.

---


[ATLAS: Agentic or Latent Visual Reasoning? One Word is Enough for Both](http://arxiv.org/abs/2605.15198)

- ATLAS: introduces a visual reasoning framework that represents complex visual operations as discrete functional tokens within a standard autoregressive sequence to avoid external tool execution or intermediate image generation.
- The framework utilizes LA-GRPO to mitigate gradient dilution by anchoring reinforcement learning updates specifically to functional tokens, ensuring stable and effective optimization.
- ATLAS maintains compatibility with standard VLM architectures and parallel training pipelines while achieving superior performance on challenging visual reasoning benchmarks with reduced inference latency.

---

[Good to Go: The LOOP Skill Engine That Hits 99% Success and Slashes Token Usage by 99% via One-Shot Recording and Deterministic Replay](http://arxiv.org/abs/2605.14237)

- LOOP Skill Engine: introduces a one-shot recording and deterministic replay paradigm to optimize periodic LLM agent tasks by replacing repeated LLM inferences with parameterized, invariant tool-call sequences.
- The system utilizes a greedy length-descending template extraction algorithm to convert LLM-generated tool trajectories into deterministic skills, effectively eliminating token consumption and non-determinism for subsequent executions.
- A robust Heartbeat Scheduler and multi-layer degradation strategy ensure high reliability and crash-safe persistence, allowing the framework to maintain continuous operation even when individual task stages fail.

---

[FUTURESIM: Replaying World Events to Evaluate Adaptive Agents](http://arxiv.org/abs/2605.15188)

- FutureSim: introduces a chronological simulation environment that replays real-world events to evaluate the long-horizon adaptive forecasting capabilities of LLMs.
- The framework utilizes a date-gated news corpus and a structured agent harness to test how LLMs update probability distributions over free-form outcomes as new information arrives.
- Experimental results demonstrate that while frontier LLMs show performance improvements with better harness design and increased test-time compute, they often struggle with overconfidence and anchoring to initial predictions.

---

[Articraft: An Agentic System for Scalable Articulated 3D Asset Generation](http://arxiv.org/abs/2605.15187)

- Articraft: introduces an agentic system that leverages LLMs to generate articulated 3D assets by iteratively writing and refining code against a domain-specific SDK.
- The framework utilizes an agent harness to provide structured feedback, enabling the LLM to perform iterative, execution-grounded refinement of 3D object programs.
- Articraft facilitates the creation of Articraft-10K, a large-scale dataset of articulated 3D assets, which improves performance in downstream robotics simulation and 3D articulation estimation tasks.

---

[Is Grep All You Need? How Agent Harnesses Reshape Agentic Search](http://arxiv.org/abs/2605.15184)

- Chronos: introduces an empirical study evaluating how retrieval strategies, agent harnesses, and tool-calling architectures jointly influence the performance of LLM agents in long-memory tasks.
- The research compares lexical (Grep) and semantic (Vector) retrieval across custom and provider-native CLI harnesses, demonstrating that retrieval effectiveness is highly dependent on the specific agent stack and delivery method.
- Findings indicate that while lexical search often outperforms semantic search in inline delivery, programmatic file-based delivery can significantly alter these performance dynamics by introducing new constraints on agent tool-use competence.

---

[From Plans to Pixels: Learning to Plan and Orchestrate for Open-Ended Image Editing](http://arxiv.org/abs/2605.15181)

- Plan2Pix: introduces an experiential learning framework for long-horizon image editing that decomposes abstract instructions into structured sub-tasks executed by a reward-driven orchestrator.
- The system utilizes a checklist-guided Planner to generate atomic sub-tasks and an Orchestrator that selects tools and regions based on feedback from a VLM-Judge.
- Closed-loop refinement and verifier-guided selection ensure that the generated plans remain feasible and coherent throughout the multi-step editing process.

---

[BEHAVIOURAL ASSURANCE CANNOT VERIFY the Safety Claims Governance Now Demands](http://arxiv.org/abs/2605.15164)

- Pilot architecture (P1–P6): introduces a reproducible mechanistic-evidence protocol to bridge the audit gap between governance requirements and current behavioural assurance methods, utilizing P1. Claim form, P2(a). Linear probe, P2(b). Activation patching, P2(c). Before/after-training comparison, P3. Pre-registered thresholds, P4. Secure Enclave (TEE), P5. Bounded compute budget, and P6. Report.
- The framework addresses the structural mismatch where current governance demands high-consequence safety proofs that behavioural evaluations cannot provide, by mandating mechanistic evidence within a secure, reproducible audit environment.
- This approach shifts the verification paradigm from surface-level behavioural testing to deep structural verification, specifically targeting latent properties like hidden objectives and deceptive alignment in LLMs.

---

[APWA: A Distributed Architecture for Parallelizable Agentic Workflows](http://arxiv.org/abs/2605.15132)

- APWA (Agent-Parallel Workload Architecture): introduces a distributed multi-agent system designed to efficiently process parallelizable agentic workloads by decomposing tasks into non-interfering subproblems executed by independent agents.
- The architecture utilizes a Manager Agent for high-level planning and task partitioning, while Subtask Worker Agents perform autonomous execution within isolated Subtask Sandboxes.
- APWA leverages a scalable distributed fabric to support heterogeneous data processing and dynamic agent capabilities, enabling high-throughput performance on large-scale tasks where prior multi-agent systems encounter scaling bottlenecks.

---

[MemEye: A Visual-Centric Evaluation Framework for Multimodal Agent Memory](http://arxiv.org/abs/2605.15128)

- MemEye: introduces a two-dimensional evaluation framework that categorizes multimodal agent memory challenges by Visual Evidence Granularity and Memory Reasoning Depth.
- The framework utilizes a benchmark of 371 mirrored questions across eight life-scenario tasks, incorporating Filtering Mechanisms and Diagnostic Probes to isolate failures in visual preservation and temporal state tracking.
- Empirical evaluation of 13 memory methods across four VLM backbones reveals that while text-based memory manages state transitions effectively, it often loses fine-grained visual details, whereas multimodal memory preserves visual evidence but struggles with temporal validity and state selection.

---

[From Text to Voice: A Reproducible and Verifiable Framework for Evaluating Tool Calling LLM Agents](http://arxiv.org/abs/2605.15104)

- Dataset-agnostic framework for audio tool-calling evaluation: introduces a methodology to convert text-based tool-calling benchmarks into controlled audio evaluations using TTS Pipeline, Omni-Modal LLMs, Automatic Evaluation, and LLMs-as-Judge to measure modality-induced performance degradation.
- The framework enables paired text-audio evaluation by preserving original tool schemas and gold labels, allowing for precise diagnostic analysis of tool-calling failures in voice-enabled LLMs.
- Experimental results across seven omni-modal models demonstrate that tool-calling performance is highly model- and task-dependent, with argument-value errors being the primary cause of failure in speech-based interactions.

---

[Veritas: A Semantically Grounded Agentic Framework for Memory Corruption Vulnerability Detection in Binaries](http://arxiv.org/abs/2605.15097)

- Veritas: introduces a semantically grounded agentic framework for detecting memory corruption vulnerabilities in stripped binaries by unifying static program analysis and runtime validation as two grounding layers for controlled LLM reasoning.
- The framework utilizes a Semantic-driven Context Slicer to extract witness-backed flows, a Dual-view Vulnerability Detector for step-wise reasoning, and an Automatic Vulnerability Validator to confirm hypotheses through debugger-visible artifacts.
- By constraining LLM reasoning to verifiable program semantics reconstructed from binary artifacts, Veritas achieves high recall and low false positives while outperforming existing static, dynamic, and agentic baselines.

---

[Concurrency without Model Changes: Future-based Asynchronous Function Calling for LLMs](http://arxiv.org/abs/2605.15077)

- AsyncFC: introduces an execution-layer framework that decouples LLM decoding from function execution using Scheduler, State Tree, Function Executor, Future Placeholders, and Dependency Annotations.
- The framework enables asynchronous function calling by returning Future Placeholders to the LLM, allowing decoding to overlap with background function execution.
- AsyncFC utilizes a dependency-aware Scheduler and State Tree to enforce safe inter-function parallelism without requiring modifications to the underlying LLM or function implementations.

---

[After the Interface: Relocating Human Agency in the Age of Conversational AI](http://arxiv.org/abs/2605.15064)

- Human-AI Interaction Agency Framework: introduces a two-dimensional diagnostic model that maps AI systems based on Process Control and Outcome Control to visualize the redistribution of human agency.
- The paper argues that human agency in the era of LLMs has not eroded but has relocated from interface-level procedural manipulation to communicative negotiation and outcome evaluation.
- This research highlights that contemporary AI systems demand new metrics for agency that account for iterative judgment, relational trust, and the unequal distribution of communicative competence among users.

---

[SpeakerLLM: A Speaker-Specialized Audio-LLM for Speaker Understanding and Verification Reasoning](http://arxiv.org/abs/2605.15044)

- SpeakerLLM: introduces a speaker-specialized audio-LLM framework that unifies single-utterance speaker profiling, recording-condition understanding, utterance-pair comparison, and evidence-organized verification reasoning.
- The framework utilizes a hierarchical speaker tokenizer to distribute speaker evidence across utterance-level embeddings and frame-level features for improved acoustic and identity modeling.
- SpeakerLLM-VR employs a structured three-block verification reasoning target to generate auditable decision traces that separate profile-level evidence from final verification verdicts.

---

[Orchard: An Open-Source Agentic Modeling Framework](http://arxiv.org/abs/2605.15040)

- Orchard: introduces an open-source framework for scalable agentic modeling, centered on a thin, Kubernetes-native environment service (Orchard Env) that decouples sandbox management from agent harnesses and training stacks.
- The framework enables reusable agentic modeling across diverse domains, including Orchard-SWE, Orchard-GUI, and Orchard-Claw, by providing harness-agnostic primitives for sandbox lifecycle, command execution, and file I/O.
- Orchard utilizes Balanced Adaptive Rollout (BAR) and credit-assignment SFT to improve training efficiency and generalization, achieving state-of-the-art performance for open-source agents while maintaining cost-effectiveness.

---

[AI Knows When It’s Being Watched: Functional Strategic Action and Contextual Register Modulation in Large Language Models](http://arxiv.org/abs/2605.15034)

- Multi-agent LLM debate architecture: introduces a controlled experimental framework to evaluate how LLMs modulate their linguistic register in response to perceived social observation contexts.
- The study demonstrates that LLMs exhibit a "Synthetic Hawthorne Effect," where lexical diversity increases under monitoring, while message elaboration is driven by audience presence.
- The research reveals that LLM behavioral adaptation is sensitive to the identity of the observer, with human evaluation eliciting stronger register formalization than automated AI surveillance.

---

[On the Limits of PAC Learning of Networks from Opinion Dynamics](http://arxiv.org/abs/2605.15033)

- Waterfall algorithm: introduces a framework for learning social network structures from threshold-based opinion dynamics by utilizing a Matching Transformation and a greedy heuristic to identify feasible influencer sets.
- The paper establishes that while PAC learning is efficient for all-but-κ dynamics, it is computationally intractable for majority dynamics, leading to the development of the Waterfall heuristic.
- The proposed Waterfall algorithm achieves high empirical success rates in identifying network structures across various random graph models by iteratively resolving inconsistencies in opinion diffusion samples.

---

[WARD: Adversarially Robust Defense of Web Agents Against Prompt Injections](http://arxiv.org/abs/2605.15030)

- WARD: introduces a practical guard framework for web agents that utilizes a large-scale dataset, guard-targeted training, and an adaptive adversarial training loop to ensure robust detection of prompt injections.
- The framework employs a two-branch data construction pipeline to capture diverse attack patterns across HTML and visual interface modalities.
- A3T enables the guard to co-evolve with an adaptive attacker, significantly improving robustness against evolving adversarial strategies while maintaining high agent utility.

---

[Multi-Agentic Approach for History Matching of Oil Reservoirs](http://arxiv.org/abs/2605.15028)

- PetroGraph: introduces a multi-agent framework that automates oil reservoir history matching by decomposing the workflow into specialized LLM-based agents and a non-LLM simulator agent.
- The system integrates RAG for domain-specific documentation access, human-in-the-loop checkpoints for manual steering, and Bayesian optimization to calibrate reservoir parameters against observed field data.
- Evaluations on synthetic and real-field models demonstrate that the framework effectively reduces history-matching mismatches while lowering the expertise barrier for complex simulation workflows.

---

[COTCAgent: Preventive Consultation via Probabilistic Chain-of-Thought Completion](http://arxiv.org/abs/2605.15016)

- COTCAgent: introduces a hierarchical reasoning framework for longitudinal EHR analysis that decouples statistical trend computation from disease risk scoring to improve diagnostic traceability.
- The framework utilizes a Temporal-Statistics Adapter to convert irregular health records into structured trend predicates, which are then matched against a symptom-trend-disease knowledge base using IDF-weighted Gibbs energies.
- By integrating a bounded completion module for targeted user inquiries, the system iteratively refines disease risk rankings while maintaining an inspectable audit trail of clinical reasoning.

---

[Efficient Online Conformal Selection with Limited Feedback](http://arxiv.org/abs/2605.14953)

- Primal-Dual ACI: introduces a unified framework for online conformal selection under bandit and semi-bandit feedback, utilizing an ACI Controller, Primal-Dual Algorithm, UCB Estimator, Expert-Chain, Lyapunov Function, Bandit Arms, and Probing Budget to achieve adversarial validity and stochastic efficiency.
- The framework employs un-projected ACI updates to maintain per-sequence adversarial validity while minimizing resource costs through Lyapunov drift analysis.
- The approach generalizes to complex combinatorial selection spaces, including NP-Hard problems, by integrating ACI with expert-chain bandit algorithms to optimize probing budgets.

---

[Not All Symbols Are Equal: Importance-Aware Constellation Design for Semantic Communication](http://arxiv.org/abs/2605.14940)

- Semantic-PHY framework: introduces a joint semantic-physical layer architecture that co-designs constellation assignment with semantic importance and statistical co-occurrence structure of learned concept vocabulary.
- The system utilizes a VQ-VAE encoder and SCI network to identify task-critical features, which are then protected at the physical layer by a learned M-QAM constellation that maximizes physical separation for high-importance symbols.
- A DQN-based rate controller dynamically adjusts the transmission payload based on channel SNR, enabling robust semantic communication across varying channel conditions and diverse data domains.

---

[Slot-MPC: Goal-Conditioned Model Predictive Control with Object-Centric Representations](http://arxiv.org/abs/2605.14937)

- Slot-MPC: introduces a goal-conditioned planning framework that leverages a Scene Parsing module, a cOCVP, a Policy network, an MPC optimizer, and Slot-based representations to perform efficient trajectory optimization in a structured latent space.
- The framework utilizes a Scene Parsing module to decompose visual observations into Slot-based representations, which are then processed by a cOCVP to forecast future states for planning.
- By employing a gradient-based MPC optimizer warm-started by a Policy network, the approach achieves efficient goal-directed control in complex robotic manipulation tasks without requiring environment interaction during training.

---

[Toward Securing AI Agents Like Operating Systems](http://arxiv.org/abs/2605.14932)

- OpenClaw-style agents: introduces a unified architecture for AI agents by drawing a structural analogy to operating systems, identifying key components including Runtime Core, Agent Core, LLM Interface, Gateway, Task Queue, Session Store, Logs, Persistent State, Plugins, Skills, Skill Tools, Core Tools, and Tool Workspaces.
- The paper systematically maps OS security principles—such as process isolation, sandboxing, and privilege separation—to agentic systems to address vulnerabilities arising from unconstrained tool use and sensitive data access.
- An empirical case study of four popular agent runtimes demonstrates that current protection mechanisms are often fragmented or ineffective, highlighting the necessity for comprehensive, OS-inspired security boundaries.

---

[Static and Dynamic Strategies for Influencing Opinions in Social Networks](http://arxiv.org/abs/2605.14918)

- Hegselmann–Krause (HK) model: introduces a comparative study of static and dynamic influence strategies for manipulating collective opinions in social networks using targeted stubborn agents.
- The research evaluates how different centrality-based node selection strategies interact with intervention timing to shift network-wide opinion distributions.
- Results demonstrate that dynamic strategies are significantly more effective than static ones, as they leverage bounded-confidence dynamics to recruit intermediate agents and avoid premature opinion fragmentation.

---

[Chrono-Gymnasium: An Open-Source, Gymnasium-Compatible Distributed Simulation Framework](http://arxiv.org/abs/2605.14911)

- Chrono-Gymnasium: introduces a distributed simulation framework that bridges high-fidelity multi-physics engines with scalable execution pipelines for RL and design optimization.
- The framework utilizes Ray Actors to encapsulate Project Chrono simulation instances, enabling parallel rollouts and efficient data generation across heterogeneous computing clusters.
- By standardizing simulation scenarios through a Gymnasium-compatible interface, the architecture simplifies the integration of complex physics models into modern machine learning and Bayesian optimization workflows.

---

[MEMLENS: Benchmarking Multimodal Long-Term Memory in Large Vision-Language Models](http://arxiv.org/abs/2605.14906)

- MEMLENS: introduces a comprehensive benchmark for evaluating multimodal long-term memory in LVLMs and memory-augmented agents, utilizing Multimodal Session Simulation, Question-Answer Pair Construction, Evidence Session Construction, Conversation History Assembly, Automated Filtering, and Human Review.
- The benchmark assesses five core memory abilities—information extraction, multi-session reasoning, temporal reasoning, knowledge update, and answer refusal—across four standardized context lengths (32K–256K tokens).
- Evaluation of 27 LVLMs and 7 memory-augmented agents reveals that while long-context LVLMs excel at short-context visual grounding, they degrade as conversations grow, whereas memory agents remain length-stable but suffer from lossy visual compression.

---

[Beyond Individual Intelligence: Surveying Collaboration, Failure Attribution, and Self-Evolution in LLM-based Multi-Agent Systems](http://arxiv.org/abs/2605.14892)

- LIFE (Lay, Integrate, Find, Evolve) progression: introduces a unified analytical framework for LLM-based multi-agent systems, connecting individual intelligence, collaboration, failure attribution, and self-evolution as causally linked stages.
- The framework characterizes the operational lifecycle of LLMs, where individual agent capabilities (reasoning, memory, planning, tool use) are integrated through collaborative structures, diagnosed via failure attribution, and refined through autonomous self-evolution.
- This survey synthesizes existing research into a coherent roadmap, identifying open challenges at the boundaries of these stages to advance toward resilient, self-organizing collective intelligence.

---

[Temporal Fair Division in Multi-Agent Systems: From Precise Alternation Metrics to Scalable Coordination Proxies](http://arxiv.org/abs/2605.14879)

- Temporal Fair Division Framework: introduces a diagnostic toolkit for assessing coordination quality in repeated multi-agent resource competition by comparing ALT (Sliding-window coordination metrics) and RP (Linear-time fairness metric).
- The framework formalizes MBoE (Repeated competitive resource game) and establishes PA (Canonical temporally fair solution) as the ideal benchmark for evaluating agent coordination.
- Empirical results demonstrate that Q-learning (Independent learning agent policy) consistently fails to achieve temporal fairness, while RP (Linear-time fairness metric) provides a scalable, symmetric alternative to ALT (Sliding-window coordination metrics) for large-scale systems.

---

[Towards In-Depth Root Cause Localization for Microservices with Multi-Agent Recursion-of-Thought](http://arxiv.org/abs/2605.14866)

- RCLAgent: introduces a multi-agent recursion-of-thought framework that decomposes root cause localization along the trace graph to mitigate context explosion and enable parallel reasoning.
- The framework utilizes Dedicated Agents for span-level analysis, an Agents Pool for controlled parallelism, and a Diagnosis Synthesizer that combines a Root-Level Diagnosis Report with a Global Evidence Graph to improve localization accuracy.
- By replacing monolithic serial reasoning with trace-aligned recursive decomposition, RCLAgent effectively balances deep causal exploration with inference efficiency in complex microservice environments.

---

[Holistic Evaluation and Failure Diagnosis of AI Agents](http://arxiv.org/abs/2605.14865)

- Holistic Agent Evaluation Framework: introduces a dual-perspective evaluation method that pairs top-down agent-level diagnosis with bottom-up span-level assessment to provide fine-grained failure localization and categorization.
- The framework decomposes agent execution traces into independent span-level assessments, enabling scalable evaluation of long traces and producing natural language rationales for each verdict.
- By integrating top-down and bottom-up signals, the approach overcomes the limitations of monolithic LLM-judges, achieving state-of-the-art localization and categorization accuracy on the TRAIL benchmark.

---

[Do Coding Agents Understand Least-Privilege Authorization?](http://arxiv.org/abs/2605.14859)

- AuthBench: introduces a benchmark for evaluating the ability of LLMs to infer task-specific least-privilege authorization policies before execution, utilizing an Authorization Agent, Execution Agent, Utility Validator, and Attack Validator.
- The research identifies that LLMs converge toward model-specific authorization attractors, where increased reasoning effort reinforces preferred failure modes rather than improving policy tightness or sufficiency.
- The proposed Sufficiency-Tightness Decomposition improves sensitive-task success and reduces attack exposure by separating coverage-oriented policy generation from necessity and sensitivity auditing.

---

[IFPV: An Integrated Multi-Agent Framework for Generative Operational Planning and High-Fidelity Plan Verification](http://arxiv.org/abs/2605.14851)

- IFPV: introduces an integrated multi-agent framework that unifies generative operational planning via MPHA and high-fidelity adversarial verification through ACSE to address generation infeasibility and verification insufficiency.
- MPHA decomposes commander intent into executable tactical sequences using specialized Pathfinder-, Analyst-, and Planner-agents, while ACSE employs a customized world model to conduct dynamic stress testing against candidate plans.
- The framework utilizes EVA-Loss to inject entity-value awareness into the world model, enabling more discriminative adversarial verification and robust plan evaluation in complex battlefield environments.

---

[Learning Direct Control Policies with Flow Matching for Autonomous Driving](http://arxiv.org/abs/2605.14832)

- Flow-matching planner: introduces a conditional flow-matching architecture that generates actionable control trajectories for autonomous driving by integrating a learned ODE vector field conditioned on BEV scene rasters.
- The architecture utilizes a heavy one-time BEV Encoder to process environmental data, followed by a lightweight Vector-field U-Net that iteratively refines control sequences via an ODE Solver.
- The model demonstrates robust out-of-distribution generalization to highway scenarios and unseen urban environments while maintaining real-time inference capabilities through efficient multi-step integration.

---

[Known By Their Actions: Fingerprinting LLM Browser Agents via UI Traces](http://arxiv.org/abs/2605.14786)

- Agent Fingerprinting Framework: introduces a passive identification method that leverages UI interaction traces to attribute web-browsing activity to specific LLMs.
- The framework utilizes an Injected JavaScript Tracker to capture temporal and structural behavioral signals, which are then processed by an XGBoost Classifier to achieve high-accuracy model identification.
- This research demonstrates that LLM agents leave distinct, fingerprintable behavioral traces during web navigation, enabling site operators to identify underlying models and potentially condition adversarial exploits or access control.

---

[Peng’s Q(λ) for Conservative Value Estimation in Offline Reinforcement Learning](http://arxiv.org/abs/2605.14779)

- CPQL (Conservative Peng’s Q(λ)): introduces a model-free offline RL algorithm that adapts the Peng’s Q(λ) operator for conservative value estimation by leveraging multi-step trajectories to mitigate over-pessimism and distributional shift.
- The framework utilizes Critic networks, Actor network, Offline dataset, Conservatism factor, Trace parameter λ, and Target networks to achieve robust performance without requiring additional auxiliary networks or behavior policy estimation.
- CPQL provides theoretical guarantees for performance improvement over the behavior policy and facilitates a smooth transition to online fine-tuning by pre-training a stable Q-function.

---

[MediaClaw: Multimodal Intelligent-Agent Platform Technical Report](http://arxiv.org/abs/2605.14771)

- MediaClaw: introduces a three-layer architecture that integrates heterogeneous AIGC capabilities into a unified, pluginized platform for reusable multimedia workflow orchestration.
- The platform utilizes a Meta-Capability Pool for atomic tool management, a Skill layer for complex process automation, and MediaUI for end-to-end visualization of intermediate artifacts.
- By decoupling business logic from specific model providers through a unified interface, the system enables flexible, Lego-like composition of multimodal production workflows.

---

[Probabilistic Verification of Recurrent Neural Networks for Single and Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2605.14758)

- RNN-ProVe: introduces a probabilistic verification framework that estimates the likelihood of undesired behaviors in RNN-based policies by leveraging a Feasibility History Classifier and Monte Carlo Estimator to assess policy-induced feasible hidden states.
- The framework addresses the #P-hard nature of exact verification by using a learned classifier to approximate the feasible hidden state manifold, enabling scalable and quantitative safety assessments for single- and multi-agent RL.
- RNN-ProVe provides bounded-error, high-confidence certificates by combining statistical analysis with policy-driven sampling, effectively overcoming the limitations of traditional over-approximation-based verification methods.

---

[Video2GUI: Synthesizing Large-Scale Interaction Trajectories for Generalized GUI Agent Pretraining](http://arxiv.org/abs/2605.14747)

- Video2GUI: introduces a fully automated framework that extracts grounded GUI interaction trajectories from unlabeled internet videos to facilitate large-scale agent pretraining.
- The framework employs a coarse-to-fine filtering strategy, including Meta Info Filtering (text-based coarse video screening), Video Quality Scoring (fine-grained content-based evaluation), Trajectory Extraction (converting videos to instruction-trajectory pairs), Action Spatial Grounding (mapping actions to screen coordinates), and Agent Training (two-stage continual pre-training and post-training).
- By applying this pipeline to 500 million video metadata entries, the authors construct WildGUI, a large-scale dataset containing 12 million interaction trajectories that significantly improves the generalization capabilities of LLMs like Qwen2.5-VL and Mimo-VL.

---

[Mechanical Enforcement for LLM Governance: Evidence of Governance-Task Decoupling in Financial Decision Systems](http://arxiv.org/abs/2605.14744)

- R2 (Mechanical Policy): introduces a governance framework that decouples decision-making from policy interpretation by using four external primitives to enforce rationale quality and decision boundaries.
- The framework utilizes CEFL, I6Q, E3, and Hard Gates to prevent LLMs from producing vacuous, non-compliant rationales under structural stress.
- Experimental results demonstrate that mechanical enforcement significantly reduces the Cosmetic Deadlock Rate and preserves governance quality even when task accuracy degrades.

---

[Agentifying Patient Dynamics within LLMs through Interacting with Clinical World Model](http://arxiv.org/abs/2605.14723)

- SepsisAgent: introduces a world model-augmented LLM agent for sepsis treatment that utilizes a Clinical World Model to simulate patient responses and refine prescriptions through a propose–simulate–refine workflow.
- The framework employs a three-stage training curriculum consisting of supervised fine-tuning for patient-dynamics prediction, behavior cloning for agentic interaction, and world-model-based reinforcement learning for long-horizon policy optimization.
- Experimental results on MIMIC-IV demonstrate that SepsisAgent outperforms traditional RL and LLM-based baselines in off-policy value and safety metrics, while internalizing patient dynamics to improve mortality and vasopressor-requirement prediction.

---

[SR-Platform: An Agentic Pipeline for Natural Language-Driven Robot Simulation Environment Synthesis](http://arxiv.org/abs/2605.14700)

- SR-Platform: introduces a production-deployed agentic pipeline that converts natural language descriptions into executable, physically valid MuJoCo simulation environments through a modular, cache-aware architecture.
- The system utilizes an LLM-based orchestrator, an asset forge for semantic retrieval or CadQuery-based geometry synthesis, a layout architect for constraint verification, and a bridge layer for final MJCF assembly.
- By separating semantic planning, asset resolution, and layout reasoning, the platform enables scalable, auditable, and robust environment generation while leveraging production telemetry to optimize performance and reliability.

---

[π-BENCH: Evaluating Proactive Personal Assistant Agents in Long-Horizon Workflows](http://arxiv.org/abs/2605.14678)

- π-BENCH: introduces a benchmark for evaluating proactive personal assistant agents in long-horizon workflows, utilizing User Roles, Episode Structure, Persistent Project Environment, Evaluated Agent, User Agent, Hidden Intent Tracker, Checklist, and Graders.
- The framework evaluates LLMs on their ability to identify and resolve hidden user intents through proactive action or targeted clarification while maintaining task completion across multi-session trajectories.
- Experiments across nine frontier LLMs demonstrate that while task completion and proactivity are related, they represent distinct capabilities, with prior interaction history significantly aiding proactive intent resolution.

---

[Agentic AI in Industry: Adoption Level and Deployment Barriers](http://arxiv.org/abs/2605.14675)

- Agentic AI Maturity Framework: introduces a six-level classification system to evaluate industrial adoption of agentic systems, ranging from individual AI Assistants to fully autonomous Self-Optimizing AI Systems.
- The study identifies a capability-deployment verification gap, where experimental agentic capabilities cannot be integrated into production due to the absence of adequate output verification mechanisms.
- Four recurring barriers—context window limitations, underperformance in proprietary content, non-determinism, and data confidentiality—collectively hinder the transition from experimental agentic workflows to qualified production deployment.

---

[Documentation-Guided Agentic Codebase Migration from C to Rust](http://arxiv.org/abs/2605.14634)

- RustPrint: introduces a documentation-driven agentic framework that leverages automatically generated codebase documentation as an intermediate representation to guide the migration of legacy C repositories to idiomatic, memory-safe Rust.
- The framework utilizes specialized LLM-based agents including DocGen, Planner, Translator, Synthesizer, RequirementRefiner, TestTranslator, and ExecutionRevisor to perform iterative, documentation-guided translation and execution-aware refinement.
- Experimental results on eight real-world C repositories demonstrate that RustPrint achieves superior compilability, feature preservation, and safety compared to existing LLM-based translation baselines.

---

[SmartWalkCoach: An AI Companion for End-to-End Walking Guidance, Motivation, and Reflection](http://arxiv.org/abs/2605.14628)

- SmartWalkCoach: introduces an end-to-end, tool-using agent architecture for walking that reduces cognitive load by orchestrating GeographyAgent, AccompanyAgent, SummaryAgent, and a BridgingAgent.
- The system utilizes a shared state object to enable non-communicating agents to coordinate, ensuring secure and modular interaction management across the walking journey.
- Field evaluation demonstrates that integrating context-aware motivational dialogue significantly improves user experience and positive affect compared to information-only navigation.

---

[Digital Twin Synchronization Over Mobile Embodied AI Network With Agentic Intelligence](http://arxiv.org/abs/2605.14625)

- MEAN: introduces a hierarchical framework for digital twin synchronization that coordinates distributed embodied AI agents through a five-stage closed-loop workflow of move-to-sense, cooperative sensing, onboard semantic processing, channel-aware mobility, and uplink transmission.
- The framework employs a two-layer optimization algorithm where the outer-layer manages multi-agent assignment via a dynamic matching game, while the inner-layer optimizes continuous resources including sensing, computation, and communication.
- The system minimizes the maximum twin deviation across regions by leveraging semantic compression to reduce latency and autonomous velocity adaptation to navigate the energy-time trade-off.

---

[In-IDE Toolkit for Developers of AI-Based Features](http://arxiv.org/abs/2605.14612)

- AI Toolkit: introduces an IDE-native observability and evaluation framework that integrates directly into the Run/Debug loop to assist developers in testing non-deterministic LLM-based features.
- The framework utilizes a client-server architecture with a Python wrapper to capture execution traces and an IDE-side server to visualize agent behavior and manage evaluation datasets.
- By treating evaluations as first-class tests and providing a low-friction path from trace to dataset, the toolkit enables non-ML specialists to adopt disciplined AI development practices without leaving their primary coding environment.

---

[Silent Collapse in Recursive Learning Systems](http://arxiv.org/abs/2605.14588)

- MTR (Monitor–Trust–Regulator): introduces a metacognitive control loop that detects and prevents silent collapse in iterative learning systems by monitoring trajectory-level statistics and adaptively modulating learning intensity.
- The framework identifies hidden contraction—characterized by predictive entropy contraction, representation drift, and tail coverage erosion—as a reliable precursor to visible model degradation.
- MTR maintains learning stability and prevents catastrophic performance loss in recursive language modeling and pseudo-labeling tasks without requiring access to pristine real data.

---

[Angel or Demon: Investigating the Plasticity Interventions’ Impact on Backdoor Threats in Deep Reinforcement Learning](http://arxiv.org/abs/2605.14587)

- SCC (Sweeper-Converter-Connector): introduces a conceptual framework for robust backdoor injection in DRL by deconstructing the mechanistic interplay between plasticity interventions and backdoor threats, utilizing Sweeper, Converter, Connector, Pathological Diagnosis, and Sharpness-Based Detection.
- The paper empirically demonstrates that while most plasticity interventions mitigate backdoor threats, SAM exacerbates them by amplifying backdoor gradients and guiding optimization toward flat minima.
- The study identifies three intrinsic mechanisms—activation pathway disruption, representation space compression, and backdoor gradient amplification—that explain how interventions modulate DRL backdoor vulnerabilities.

---

[Remember Your Trace: Memory-Guided Long-Horizon Agentic Framework for Consistent and Hierarchical Repository-Level Code Documentation](http://arxiv.org/abs/2605.14563)

- MemDocAgent: introduces a long-horizon agentic framework that generates consistent, hierarchical repository-level documentation by utilizing Dependency-Aware Traversal Guiding and Memory-Guided Agentic Interaction.
- The framework employs a centralized RepoMemory to store and reuse documentation across sub-tasks, ensuring cross-document consistency and reducing redundant retrievals.
- MemDocAgent utilizes a multi-turn agentic workflow with READ, WRITE, and VERIFY operations to maintain persistent state and produce documentation that supports practical code reconstruction.

---

[Resolving Action Bottleneck: Agentic Reinforcement Learning Informed by Token-Level Energy](http://arxiv.org/abs/2605.14558)

- ACTFOCUS: introduces a token-level reweighting approach that mitigates the Action Bottleneck by downweighting reasoning tokens and prioritizing high-energy action tokens to improve credit assignment in agentic RL.
- The framework utilizes a frozen-reference model to compute token-level energy, which serves as a stable proxy for predictive uncertainty to guide gradient redistribution.
- ACTFOCUS consistently improves performance and training stability across multiple environments and LLM scales by shifting gradient mass toward critical environment-facing action tokens.

---

[TeachAnything: A Multimodal Crowdsourcing Platform for Training Embodied AI Agents in Symmetrical Reality](http://arxiv.org/abs/2605.14556)

- TeachAnything: introduces a three-stage demonstration paradigm that integrates Language-based demonstration, Video-based demonstration, and Teleoperation-based demonstration to collect multimodal data for training embodied agents.
- The platform utilizes a Physics Simulation backend with diverse Robot Embodiments and a Controller to generate physically consistent, temporally aligned training data.
- By leveraging WebSocket streaming and Flask microservices, the system enables distributed, cloud-based crowdsourcing of fine-grained manipulation strategies and high-level task intent.

---

[LiWi: Layering in the Wild](http://arxiv.org/abs/2605.14552)

- LiWi: introduces a framework for high-fidelity natural image decomposition that utilizes an ADD (Agent-driven Data Decomposition) pipeline to construct a large-scale dataset of in-the-wild layered images.
- The framework incorporates a shadow layer to explicitly model complex illumination effects and a degradation-restoration objective to improve alpha boundary accuracy during the generation process.
- By leveraging an agentic system with specialized tools, the approach enables scalable, automated supervision for natural image layering without requiring manual annotation.

---

[VerbalValue: A Socially Intelligent Virtual Host for Sales-Driven Live Commerce](http://arxiv.org/abs/2605.14542)

- VerbalValue: introduces a dual-channel architecture for live-commerce hosting that balances continuous product narration with responsive, intent-conditioned interaction using a fine-tuned LLM.
- The framework utilizes a structured product knowledge base and intent-conditioned fine-tuning to ensure factual accuracy and tactical alignment with viewer comments.
- By decoupling latency-sensitive generation from media operations and employing a reranker for candidate selection, the system maintains high engagement and factual correctness in real-time broadcast environments.

---

[Cattle Trade: A Multi-Agent Benchmark for LLM Bluffing, Bidding, and Bargaining](http://arxiv.org/abs/2605.14537)

- Cattle Trade: introduces a multi-agent benchmark for evaluating LLMs in strategic reasoning under imperfect information, adversarial interaction, and resource constraints.
- The framework integrates competitive auctions, hidden-offer trade challenges, and discrete resource management to test the joint deployment of agentic capabilities.
- Empirical results demonstrate that strategic coherence, such as spending efficiency and resource discipline, is more critical for success than individual sub-skills or spending volume.

---

[Lang2MLIP: End-to-End Language-to-Machine Learning Interatomic Potential Development with Autonomous Agentic Workflows](http://arxiv.org/abs/2605.14527)

- Lang2MLIP: introduces a multi-agent framework that automates the development of machine learning interatomic potentials by formulating the process as a sequential decision-making problem solved by LLMs.
- The framework utilizes a two-phase approach consisting of an interactive preparation phase for task clarification and structure generation, followed by an autonomous training phase for iterative model refinement.
- By employing a central decision-making agent to orchestrate specialized sub-agents, the system enables non-experts to develop stable and accurate MLIPs without requiring manually designed pipelines.

---

[When Robots Do the Chores: A Benchmark and Agent for Long-Horizon Household Task Execution](http://arxiv.org/abs/2605.14504)

- HoloMind: introduces a hierarchical agent framework for long-horizon household tasks, integrating High-level Planner, Low-level Planner, Executor, Multimodal Spatial Memory, Episodic Memory, and Critic.
- The framework utilizes a DAG-based hierarchical planner to decompose complex instructions into executable subgoals, supported by persistent memory and reflective supervision to maintain stable long-term execution.
- The paper also presents LongAct, a benchmark for evaluating long-horizon planning autonomy, featuring free-form instructions and an Improvement Rate metric to quantify experience-driven performance gains.

---

[GroupMemBench: Benchmarking LLM Agent Memory in Multi-Party Conversations](http://arxiv.org/abs/2605.14498)

- GroupMemBench: introduces a benchmark for evaluating LLM agent memory in multi-party conversations by utilizing a Graph-grounded synthesis pipeline, an Adversarial query pipeline, and a Solve-Judge-Refine loop.
- The framework evaluates memory systems across three dimensions: group dynamics, speaker-grounded belief tracking, and audience-adapted language.
- Benchmarking results reveal that current memory systems often flatten group structure and fail to condition retrieval on speaker identity, leading to significant performance gaps.

---

[Contestable Multi-Agent Debate with Arena-based Argumentative Computation for Multimedia Verification](http://arxiv.org/abs/2605.14495)

- A-QBAF (Arena-based Quantitative Bipolar Argumentation Framework): introduces a contestable multi-agent framework that integrates multimodal LLMs, external verification tools, and A-QBAF to provide transparent, editable, and evidence-grounded multimedia verification.
- The framework decomposes multimedia cases into claim-centered sections, utilizing a planner agent, deep researcher agent, and argument cards to structure evidence into support-attack relations.
- It employs a sparse local graph design for efficient reasoning, selective clash resolution via a judge model, and uncertainty-aware escalation to ensure reliable outcomes in high-stakes verification.

---

[LEMON: Learning Executable Multi-Agent Orchestration via Counterfactual Reinforcement Learning](http://arxiv.org/abs/2605.14483)

- LEMON: introduces a framework for learning compositional multi-agent orchestration by generating executable specifications that integrate task-specific roles, capacity levels, and dependency structures.
- The framework utilizes an LLM Orchestrator to produce YAML-based specifications, which are trained using a combination of global GRPO and localized counterfactual credit assignment to optimize performance and efficiency.
- By applying reward contrasts to specific edited spans of the orchestration specification, LEMON effectively addresses the sparse credit assignment problem inherent in multi-agent system training.

---

[Test-Time Learning with an Evolving Library](http://arxiv.org/abs/2605.14477)

- EVOLIB: introduces a test-time learning framework that enables LLMs to accumulate, reuse, and evolve knowledge across problem instances without parameter updates or external supervision.
- The framework maintains a shared library of modular skills and reflective insights, which are automatically extracted from inference trajectories and refined through a principled weighting and consolidation mechanism.
- By optimizing for both immediate utility and long-term value via Information Gain and Future Information Gain, EVOLIB allows simple instance-specific abstractions to evolve into general, reusable knowledge over time.

---

[Exploiting LLM Agent Supply Chains via Payload-less Skills](http://arxiv.org/abs/2605.14460)

- SCH (Semantic Compliance Hijacking): introduces a payload-less supply chain attack that weaponizes natural language skill documentation to induce LLMs into synthesizing malicious code at runtime.
- The framework utilizes MS-AO to iteratively refine adversarial skills within a sandbox, bypassing static and semantic security defenses by omitting explicit code signatures.
- The research demonstrates that highly aligned LLMs are paradoxically more vulnerable to semantic manipulation, achieving significant success rates in data exfiltration and remote code execution.

---

[LiSA: Lifelong Safety Adaptation via Conservative Policy Induction](http://arxiv.org/abs/2605.14454)

- LiSA (Lifelong Safety Adaptation): introduces a conservative policy induction framework that improves a fixed base guardrail through structured memory, including broad policy abstractions, conflict-aware local rules, and evidence-aware confidence gating.
- The framework operates via an online-offline loop, where sparse user-reported failures are abstracted into reusable policies and boundary-specific local rules to adapt to deployment environments without repeated fine-tuning.
- LiSA utilizes a Beta-posterior lower bound to gate memory reuse, ensuring that only sufficiently supported policies influence LLM inference, thereby maintaining robustness against noisy feedback and preventing overgeneralization.

---

[FrontierSmith: Synthesizing Open-Ended Coding Problems at Scale](http://arxiv.org/abs/2605.14445)

- FrontierSmith: introduces an automated pipeline that transforms closed-ended coding problems into open-ended variants through targeted mutations and multi-stage filtering to generate scalable training data for LLMs.
- The framework utilizes an idea divergence metric to quantify solution strategy diversity, ensuring that synthesized problems elicit genuine algorithmic exploration rather than single-strategy dominance.
- FrontierSmith-generated problems enable LLMs to exhibit long-horizon agent behavior, characterized by increased turn counts and thinking tokens, comparable to human-curated open-ended benchmarks.

---

[GGBound: A Genome-Grounded Agent for Microbial Life-Boundary Prediction](http://arxiv.org/abs/2605.14442)

- GGBound: introduces a genome-conditioned, tool-augmented LLM agent that maps microbial genotypes to physiological life boundaries using LucaOne Genome Encoder, Qwen Backbone, RAG Module, GEM Tool, and GRPO Optimizer.
- The framework integrates genomic embeddings with external biological evidence through a three-stage training pipeline comprising gene-text alignment, agentic supervised fine-tuning, and reinforcement learning with a counterfactual gene-grounding reward.
- GGBound achieves competitive performance against larger frontier LLMs by leveraging selective evidence acquisition and causal genomic conditioning to predict microbial traits such as growth ranges, optimal conditions, and metabolic capabilities.

---

[FuzzAgent: Multi-Agent System for Evolutionary Library Fuzzing](http://arxiv.org/abs/2605.14431)

- FuzzAgent: introduces a multi-agent system that transforms library fuzzing into an evolutionary process by utilizing specialized agents to iteratively refine harnesses based on runtime feedback.
- The architecture integrates an Agent Pool, dedicated Interfaces, and a stateful Environment to automate the entire fuzzing lifecycle, including build configuration, harness generation, and crash validation.
- FuzzAgent employs a closed-loop reasoning strategy where agents leverage runtime evidence to overcome coverage plateaus and distinguish genuine library bugs from harness-induced errors.

---

[Collaborative Yet Personalized Policy Training: Single-Timescale Federated Actor-Critic](http://arxiv.org/abs/2605.14423)

- pFedAC (Personalized Federated Actor-Critic): introduces a federated reinforcement learning framework that enables agents to collaboratively learn a shared linear subspace representation while maintaining personalized local critic heads and actors to handle environmental heterogeneity.
- The framework utilizes a single-timescale update scheme with Markovian sampling and a joint linear approximation approach to achieve linear speedup with respect to the number of agents.
- The approach incorporates a simulator to generate auxiliary state-action pairs, mitigating distribution mismatches between critic updates and policy gradients in heterogeneous multi-agent environments.

---

[MemLineage: Lineage-Guided Enforcement for LLM Agent Memory](http://arxiv.org/abs/2605.14421)

- MemLineage: introduces a defence for LLM agent memory that attaches cryptographic provenance and derivation lineage to every entry to prevent untrusted content from authorizing sensitive actions.
- The system utilizes a six-module architecture including M1 (Provenance metadata), M2 (Cryptographic binding), M3 (Append-only Merkle log), M4 (Lineage DAG), M5 (Verifier-aware retrieval), and M6 (Sensitive-action gate) to maintain chain-of-custody.
- MemLineage effectively mitigates persistent memory attacks by propagating trust labels through LLM-mediated derivation chains and gating sensitive tool dispatches based on the ancestry of the retrieved memory.

---

[SWE-CHAIN: Benchmarking Coding Agents on Chained Release-Level Package Upgrades](http://arxiv.org/abs/2605.14415)

- SWE-CHAIN: introduces a benchmark for evaluating LLMs on chained release-level package upgrades, utilizing DecompSynth to synthesize grounded specifications from release notes and code diffs.
- The framework employs an Agent Workspace within a Docker Environment to model long-horizon software maintenance, where agents must carry changes forward across consecutive versions.
- To ensure robust evaluation, the pipeline incorporates a Build+Fix Regularization mechanism that allows agents a single controlled repair attempt for execution-level errors.

---

[DermAgent: A Self-Reflective Agentic System for Dermatological Image Analysis with Multi-Tool Reasoning and Traceable Decision-Making](http://arxiv.org/abs/2605.14403)

- DermAgent: introduces a collaborative agentic system that orchestrates specialized vision and language tools within a Plan–Execute–Reflect framework to provide traceable dermatological diagnosis.
- The system utilizes a dual-modality retrieval module, incorporating Case RAG and Guideline RAG, to anchor diagnostic predictions in verifiable external clinical evidence.
- A deterministic Critic module performs post-hoc auditing of the evidence chain using confidence, coverage, and conflict gates to trigger targeted self-correction and mitigate LLM hallucinations.

---

[Agentic Recommender System with Hierarchical Belief-State Memory](http://arxiv.org/abs/2605.14401)

- ARS (Memory-Augmented Agentic Recommender System): introduces a hierarchical memory architecture that abstracts raw user interactions into structured preference chunks and coherent natural language profiles to improve recommendation accuracy.
- The framework utilizes an LLM-based planner to manage a complete memory lifecycle, including extraction, reinforcement, weakening, consolidation, forgetting, and resynthesis, replacing rigid heuristics with adaptive scheduling.
- By decoupling the online ranking path from the offline memory lifecycle, ARS achieves state-of-the-art performance while significantly reducing computational costs through efficient state abstraction.

---

[Coding Agent Is Good As World Simulator](http://arxiv.org/abs/2605.14398)

- Multi-Agent Framework: introduces an agentic system that constructs physics-based world models by iteratively generating and refining executable simulation code through a closed-loop workflow.
- The framework coordinates specialized agents including planning-, code generation-, visual review- and simulation judge-agents to ensure physical consistency and instruction fidelity.
- By utilizing executable code as the world representation, the system enables inspectable dynamics and iterative repair, outperforming traditional video-based world models in physical accuracy.

---

[NEXUS: An Agentic Framework for Time Series Forecasting](http://arxiv.org/abs/2605.14389)

- NEXUS: introduces a multi-agent framework that decomposes time series forecasting into structured stages, utilizing a Historical Context Agent, Macro-Reasoning Agent, Micro-Reasoning Agent, Forecast Synthesizer Agent, and Calibration Agent to synthesize numerical trends with qualitative context.
- The framework employs a dual-resolution approach where the Macro-Reasoning Agent establishes overarching regimes and the Micro-Reasoning Agent identifies granular, event-driven catalysts to improve forecasting accuracy.
- NEXUS incorporates a Calibration Agent that uses a backtesting mechanism to generate master guidelines from past errors, ensuring the system adapts to domain-specific dynamics without requiring manual instruction design.

---

[Data-Augmented Game Starts for Accelerating Self-Play Exploration in Imperfect Information Games](http://arxiv.org/abs/2605.14379)

- DAGS (Data-Augmented Game Starts): introduces a starting-state sampling strategy that initializes self-play episodes at intermediate states from an Offline Dataset to accelerate exploration in imperfect-information games.
- The framework utilizes Data-Augmented Game Starts to bypass long coordinated control sequences, enabling agents to focus on strategically relevant subgames using PPO-Uniform or PPO-EMAg solvers.
- To mitigate potential equilibrium bias introduced by state augmentation, the approach incorporates Multi-task Observation Flags to partition policies into unbiased tasks during training.

---

[Semi-Synchronous Exploration in Dynamic Graphs](http://arxiv.org/abs/2605.14375)

- SSYNC_EXPO: introduces a deterministic algorithm for exploring 1-interval connected dynamic graphs under a semi-synchronous scheduler where an adversary controls agent activation and network topology.
- The paper establishes a tight lower bound on the number of agents required for successful exploration based on the adversary's deactivation power.
- The proposed algorithm utilizes a pipeline strategy and progressive parameter estimation to achieve exploration with O(kDˆ) move complexity and O(max{log n, log p}) memory per agent.

---

[HERCULEAN: An Agentic Benchmark for Financial Intelligence](http://arxiv.org/abs/2605.14355)

- HERCULEAN: introduces a standardized, skill-based benchmark for evaluating LLM agents across four complex financial workflows: Trading, Hedging, Market Insights, and Auditing.
- The framework utilizes a two-level interaction design, where a structured skill layer mediates agent access to MCP-grounded environments, ensuring architecture-agnostic evaluation of reasoning and execution capabilities.
- Experimental results across multiple agent frameworks and LLM backbones reveal that financial agent performance is highly workflow-dependent, with significant gaps in long-horizon reasoning, state management, and deterministic verification.

---

[Distributionally Robust Multi-Task Reinforcement Learning via Adaptive Task Sampling](http://arxiv.org/abs/2605.14350)

- DRATS: introduces a multi-task reinforcement learning algorithm that addresses data imbalance by adaptively prioritizing tasks with the largest return gaps using a minimax optimization objective.
- The framework utilizes a KL-regularized task-sampling distribution to ensure stable, non-zero sampling probabilities across all tasks while focusing on those furthest from their target returns.
- DRATS is compatible with existing multi-task network architectures and demonstrates improved data efficiency and worst-task performance across various robotic manipulation and continuous control benchmarks.

---

[Sub-Band Full Duplex Resource Allocation: A Predictive Deep Reinforcement Learning Approach](http://arxiv.org/abs/2605.14339)

- Bi-LSTM+DDQN: introduces a predictive framework for dynamic sub-band allocation in SBFD systems, integrating 1D-CNN (extracts local spatial features), Bi-LSTM (captures long-term temporal dependencies), DDQN (performs dynamic resource allocation), Experience Replay Memory (stores past interaction events), Online Network (selects optimal scheduling actions), and Target Network (evaluates action stability).
- The framework utilizes a hybrid 1D-CNN and Bi-LSTM architecture to forecast network traffic, providing the DDQN agent with a 22-dimensional state representation for proactive scheduling.
- By dynamically adjusting UL/DL split ratios based on predicted traffic, the system achieves 100% peak UL utilization and significantly reduces queue buildup compared to static baseline configurations.

---

[Are Agents Ready to Teach? A Multi-Stage Benchmark for Real-World Teaching Workflows](http://arxiv.org/abs/2605.14322)

- EduAgentBench: introduces a theory-grounded, source-grounded benchmark for evaluating LLMs across three teaching capability surfaces: pedagogical judgment, situated tutoring, and teaching workflow execution.
- The framework utilizes a pedagogical-insight-driven pipeline to construct 150 quality-controlled tasks that require LLMs to diagnose learner states, adapt scaffolding, and perform institutional actions within simulated learning-management systems.
- Evaluation results reveal that while current LLMs demonstrate proficiency in bounded pedagogical judgment, they frequently fail to maintain coherence in multi-turn tutoring or execute complex, evidence-grounded teaching workflows.

---

[Making OpenAPI Documentation Agent-Ready: Detecting Documentation and REST Smells with a Multi-Agent LLM System](http://arxiv.org/abs/2605.14312)

- Hermes: introduces a multi-agent LLM-based system that detects documentation and REST-related smells in OpenAPI specifications to assess readiness for autonomous agent consumption, utilizing a Smell Detector Agent, Documentation Smell Agents, REST Smell Agents, and a Reduced OpenAPI Representation.
- The system employs an endpoint-centric strategy, isolating operations to generate explainable diagnostic reports that guide remediation efforts and inform strategic AI adoption decisions.
- Empirical evaluation across 600 production endpoints revealed that structural validity in microservice environments does not guarantee semantic readiness for LLMs, necessitating systematic documentation assessment as a foundational prerequisite for AI-agent integration.

---

[Beyond Binary: Reframing GUI Critique as Continuous Semantic Alignment](http://arxiv.org/abs/2605.14311)

- BBCritic: introduces a contrastive framework that reframes GUI critique from binary classification to continuous semantic alignment within a shared Affordance Space.
- The framework utilizes a VLM-based Encoder to map instructions and actions into a shared embedding space, employing InfoNCE Loss to recover hierarchical action structures and resolve affordance collapse.
- BBCritic incorporates a two-stage training curriculum using UI Element Parser and VLM Rollout to generate hard negatives, enabling robust ranking performance on the BBBench Benchmark.

---

[Towards Self-Evolving Agentic Literature Retrieval](http://arxiv.org/abs/2605.14306)

- PaSaMaster: introduces a self-evolving agentic literature retrieval system that separates intent-aware planning from evidence-grounded retrieval and ranking to ensure source authenticity and cost-efficient scaling.
- The system utilizes a Navigator: Planner to iteratively refine search strategies based on feedback, while a Librarian Swarm: Parallel Executor performs retrieval and verification using an Agent-Native Repository and Toolset.
- By treating literature discovery as an intent-paper relevance ranking process rather than generation, PaSaMaster eliminates source hallucinations and achieves superior performance on the PaSaMaster-Bench compared to existing LLM-based retrieval methods.

---

[Web Agents Should Adopt the Plan-Then-Execute Paradigm](http://arxiv.org/abs/2605.14290)

- PTE (Plan-Then-Execute): introduces a secure web agent architecture that separates control flow from untrusted data by committing to a task-specific program before execution, utilizing a Planner, Executor, LLM subroutine, Trusted API, and Web.
- The framework mitigates control-flow hijacking by ensuring that untrusted web content can only influence data values within a fixed execution graph rather than synthesizing new actions.
- Empirical analysis on the WebArena benchmark demonstrates that all tasks are compatible with the PTE paradigm, with over 80% being fully static and the remainder requiring only constrained LLM subroutines.

---

[Watermarking Game-Playing Agents in Perfect-Information Extensive-Form Games](http://arxiv.org/abs/2605.14283)

- KGW watermark adaptation: introduces a method to embed robust, unique signatures into game-playing agents by modifying action probability distributions based on green- and red-list partitioning of available actions.
- The framework utilizes a strategy profile, a game-solving algorithm, a pseudo-random number generator, a green list, a red list, a watermark wrapper, and a statistical test to ensure detectability while bounding the loss in expected utility.
- Experimental results on UCI chess engines demonstrate that the watermark is detectable with a handful of games while maintaining negligible performance degradation.

---

[Auditing Agent Harness Safety](http://arxiv.org/abs/2605.14271)

- HarnessAudit: introduces a harness-centric safety auditing framework that evaluates full execution trajectories across boundary compliance, execution fidelity, and system stability using hidden, agent-independent evidence channels.
- HarnessAudit-Bench: provides a comprehensive stress-testing suite of 210 tasks across eight real-world domains, instantiated in both single-agent and multi-agent configurations with embedded safety constraints.
- The research demonstrates that task completion is misaligned with safe execution, with safety risks varying significantly across domains, agent roles, and multi-agent collaboration structures.

---

[Heuristic Pathologies and Further Variance Reduction via Uncertainty Propagation in the AIVAT Family of Techniques](http://arxiv.org/abs/2605.14261)

- AIVAT: introduces a cautionary analysis of heuristic value functions in variance reduction, demonstrating that unfixed heuristics can lead to pathological variance or p-hacking.
- The paper proposes propagating heuristic uncertainty to the estimate level using inverse-variance weighting to achieve further variance reduction in multiagent evaluation.
- Experimental results on poker data demonstrate that the proposed uncertainty-aware approach yields a 43.0% reduction in the number of samples required for statistical significance.

---

[Hypergraph Enterprise Agentic Reasoner over Heterogeneous Business Systems](http://arxiv.org/abs/2605.14259)

- HEAR: introduces an enterprise agentic reasoner that grounds LLMs within a Stratified Hypergraph Ontology to resolve heterogeneous data dependencies and enforce n-ary business constraints for multi-hop reasoning.
- The architecture integrates an Agentic Reasoning Loop with a Stratified Hypergraph Ontology, utilizing a Graph Layer for virtualized data access and a Hyperedge Layer for binding n-ary business axioms and procedural protocols.
- HEAR achieves high accuracy and adaptive execution efficiency on complex supply-chain tasks by dynamically orchestrating ontology tools to navigate heterogeneous systems without requiring LLM retraining.

---

[Latency-Quality Routing for Functionally Equivalent Tools in LLM Agents](http://arxiv.org/abs/2605.14241)

- LQM-CONTEXTROUTE: introduces a contextual bandit router for functionally equivalent tool providers that optimizes a renewal-reward rate to prevent latency from compensating for poor answer quality.
- The framework utilizes a LinUCB quality head, an EMA latency estimator, and LLM-as-judge feedback to adaptively route queries based on real-time load and provider-specific performance.
- By treating latency as a service-cycle cost rather than an additive penalty, the approach effectively mitigates performance degradation in heterogeneous provider pools under non-stationary load conditions.

---

[Quantum Advantage in Multi Agent Reinforcement Learning](http://arxiv.org/abs/2605.14235)

- QMARL (Quantum Multi Agent Reinforcement Learning): introduces a decentralized framework utilizing Variational Quantum Circuit (VQC) actors and shared entangled states to achieve coordination in multi-agent systems.
- The framework employs Centralised Training with Decentralized Execution (CTDE) to enable agents to learn policies that exploit quantum entanglement for implicit coordination without runtime communication.
- Experimental results demonstrate that entanglement provides a provable quantum advantage in non-local games, while VQC expressiveness and hybrid actor-critic architectures offer performance benefits in cooperative navigation tasks.

---

[MetaAgent-X : Breaking the Ceiling of Automatic Multi-Agent Systems via End-to-End Reinforcement Learning](http://arxiv.org/abs/2605.14212)

- MetaAgent-X: introduces an end-to-end reinforcement learning framework that jointly optimizes Designer- and Executor-agents to break the performance ceiling of static multi-agent systems.
- The framework utilizes Executor-Designer Hierarchical Rollout to enable structured trajectory collection and accurate credit assignment across roles.
- Stagewise Co-evolution decouples the learning stages of the Designer and Executor to improve training stability and scalability during the joint optimization process.

---

[ASH: Agents that Self-Hone via Embodied Learning](http://arxiv.org/abs/2605.14211)

- ASH: introduces a dynamic bootstrapping framework that enables agents to learn long-horizon embodied policies by iteratively retrieving and learning from relevant internet video without manual reward engineering.
- The system utilizes an Inverse Dynamics Model to generate pseudo-actions from unlabeled video and a dual-memory architecture to maintain both reactive short-term control and long-term task progression.
- ASH demonstrates superior performance in complex, open-ended environments like Pokémon Emerald and The Legend of Zelda by autonomously identifying key moments and adapting to new visual dynamics through self-improvement.

---

[SimPersona: Learning Discrete Buyer Personas from Raw Clickstreams for Grounded E-Commerce Agents](http://arxiv.org/abs/2605.14205)

- SimPersona: introduces a framework that learns discrete buyer personas from raw clickstreams using a behavior-aware VQ-VAE and grounds them in LLM agents through a two-stage SFT process.
- The framework utilizes a Data Pipeline to extract behavioral features and generate multi-turn agent traces, enabling the LLM Agent to simulate realistic, merchant-specific buyer population distributions.
- By decoupling persona grounding from action learning, the Two-Stage SFT approach ensures robust agent performance and generalization across unseen storefronts without requiring per-store calibration.

---

[MMSkills: Towards Multimodal Skills for General Visual Agents](http://arxiv.org/abs/2605.13527)

- MMSkills: introduces a framework for representing, generating, and using reusable multimodal procedural knowledge to improve visual decision-making in agents.
- The framework utilizes a multimodal skill package containing textual procedures, runtime state cards, and multi-view keyframes to provide state-aware guidance.
- A branch-loaded mechanism isolates skill-environment grounding in a temporary branch, returning distilled structured guidance to the main agent to avoid context pressure and visual anchoring.

---

[Speculative Interaction Agents: Building Real-Time Agents with Asynchronous I/O and Speculative Tool Calling](http://arxiv.org/abs/2605.13360)

- Speculative Interaction Agents: introduces an event-driven architecture that decouples agent reasoning from user and environment streams to enable real-time responsiveness.
- The framework utilizes Asynchronous I/O to overlap reasoning with streaming inputs and Speculative Tool Calling to manage task execution while awaiting full information.
- A clock-based training methodology is employed to adapt LLMs for continuous reasoning and error correction in dynamic, latency-sensitive agentic workflows.

---

[D-VLA: A High-Concurrency Distributed Asynchronous Reinforcement Learning Framework for Vision-Language-Action Models](http://arxiv.org/abs/2605.13276)

- D-VLA: introduces a high-concurrency distributed reinforcement learning framework that utilizes Plane Decoupling to isolate high-frequency simulation data from low-frequency weight control, effectively eliminating resource contention.
- The framework employs a four-thread Swimlane pipeline to enable full parallel overlap of sampling, inference, gradient computation, and parameter distribution, significantly enhancing throughput for large-scale VLA models.
- By integrating dual-pool VRAM management and topology-aware replication, D-VLA resolves memory fragmentation and communication bottlenecks, achieving stable linear speedup for trillion-parameter embodied agents.

---

[Residual Reinforcement Learning for Robot Teleoperation under Stochastic Delays](http://arxiv.org/abs/2605.15480)

- DR-RL: introduces a hybrid control framework that integrates an LSTM-based state estimator with a residual RL policy to ensure stable teleoperation under stochastic communication delays.
- The framework utilizes an autoregressive LSTM-based state estimator to provide continuous state predictions, effectively mitigating the partial observability caused by time-varying network delays.
- A residual RL agent, trained via Soft Actor-Critic, computes corrective torque terms to compensate for unmodeled dynamics and tracking errors that the nominal controller cannot suppress.

---

[EgoExo-WM: Unlocking Exo Video for Ego World Models](http://arxiv.org/abs/2605.15477)

- EgoExo-WM: introduces a framework that leverages exocentric video to train egocentric world models by converting third-person observations into action-aligned egocentric visual experiences.
- The framework utilizes a Body Pose Predictor and an Exocentric to Egocentric Converter to ground video synthesis in human kinematics, enabling the integration of large-scale internet video for training.
- EgoExo-WM incorporates a wrist-position consistency objective and a latent-space world model to improve future state prediction and goal-conditioned planning for embodied agents.

---

[Validated Hypotheses as a Lens for Human-Likeness Evaluation in AI Agents](http://arxiv.org/abs/2605.15473)

- HUMANSTUDY-BENCH: introduces a principled, diagnostic platform that evaluates LLM human-likeness by comparing agent behavior against validated social science hypotheses using PAS and ECS metrics.
- The framework utilizes a human-in-the-loop pipeline to reconstruct published experimental protocols, enabling objective and scalable assessment of agent performance across diverse cognitive and social domains.
- Empirical results across 10 LLMs demonstrate that agent design significantly influences alignment, revealing that current models often fail to replicate human behavioral patterns in diagnostic, non-monotonic ways.

---

[Estimated Dynamic Equilibrium Model: Supply and Demand as a Sample Path of a Stochastic Process](http://arxiv.org/abs/2605.15472)

- EDEM (Estimated Dynamic Equilibrium Model): introduces an agent-based framework that models market supply and demand as a coupled stochastic process driven by heterogeneous, error-prone agent valuations.
- The framework identifies an order-statistic mechanism where max-bid clearing and per-epoch price feedback generate persistent positive price drift without requiring behavioral assumptions like investor optimism.
- By varying parameters such as divergence of opinion, seller patience, and population balancing, the model reproduces diverse market regimes including stable equilibria, business cycles, and runaway bubbles.

---

[DRUGSAGE: Self-evolving Agent Experience for Efficient State-of-the-Art Drug Discovery](http://arxiv.org/abs/2605.15461)

- DRUGSAGE: introduces an agentic framework that accumulates and reuses cross-task experience to efficiently build state-of-the-art drug discovery models.
- The framework integrates a persistent memory system—comprising Solution Memory, Refinement Memory, and Execution Memory—into a Monte Carlo Tree Search loop to guide model development and enable zero-test-time solution transfer.
- By leveraging cross-task evidence, DRUGSAGE significantly reduces the search budget and LLM API costs while outperforming existing baselines in molecular property prediction tasks.

---

[Runtime-Structured Task Decomposition for Agentic Coding Systems](http://arxiv.org/abs/2605.15425)

- RSTD: introduces an architectural pattern that externalizes task structure into executable control flow to enable selective retry at subtask granularity.
- The framework utilizes a Decomposition Engine, Typed LLM Judgment Operators, Schema Validation, and a State Manager to isolate failures and prevent cascading re-execution.
- Empirical evaluation demonstrates that RSTD achieves significant retry cost reductions compared to monolithic and static decomposition approaches by enabling runtime-controlled branching.

---

[Social-Mamba: Socially-Aware Trajectory Forecasting with State-Space Models](http://arxiv.org/abs/2605.15424)

- Social-Mamba: introduces a trajectory forecasting architecture that reformulates unstructured social interactions as structured sequential processes using selective state-space models.
- The framework utilizes a Cycle Mamba (CM) block to enable continuous bidirectional information flow, ensuring that the forward pass is explicitly conditioned on the future context.
- Social-Mamba organizes agents into an egocentric grid and employs social triplet factorization to capture temporal, egocentric, and goal-centric dynamics efficiently.

---

[Beyond Partner Diversity: An Influence-Based Team Steering Framework for Zero-Shot Human-Machine Teaming](http://arxiv.org/abs/2605.15400)

- IBTS: introduces a framework for zero-shot human-machine teaming that combines partner diversity with learned coordination structure to improve performance in sparse-reward environments.
- The framework utilizes Influence-Shaping to incentivize supportive behaviors, a Trajectory-Conditioned Team Predictor to recognize coordination patterns, and Team Steering to guide agents toward high-performing interaction modes.
- Evaluations across simulated, synthetic LLM-partner, and human-subject studies demonstrate that IBTS outperforms diversity-focused baselines in both dyadic and group human-machine teaming settings.

---

[Ensemble Monitoring for AI Control: Diverse Signals Outweigh More Compute](http://arxiv.org/abs/2605.15377)

- Ensemble Monitoring for AI Control: introduces a framework for improving AI safety by aggregating diverse signals from multiple LLM-based monitors to detect misaligned code actions.
- The approach utilizes both prompt-based and fine-tuned monitors to generate complementary suspicion scores, which are then aggregated to outperform individual monitors and homogeneous ensembles.
- The research demonstrates that monitor diversity, rather than increased compute scale, is the primary driver of performance gains in AI control, with small ensembles capturing most of the potential safety improvements.

---

[Belief Engine: Configurable and Inspectable Stance Dynamics in Multi-Agent LLM Deliberation](http://arxiv.org/abs/2605.15343)

- BE (Belief Engine): introduces an auditable simulation-control layer that decouples belief maintenance from generative reasoning to enable explicit, parameterised control over stance dynamics in multi-agent LLM deliberation.
- The framework utilizes a Bayesian-style log-odds update rule, controlled by evidence uptake and prior anchoring parameters, to maintain a persistent, proposition-level belief state that conditions LLM-generator responses.
- By separating argument extraction, evidence judgement, and structured memory from generation, the architecture provides an inspectable audit trail for agent stance changes, addressing the limitations of implicit prompt-based belief revision.

---

[Minerva-Ego: Spatiotemporal Hints for Egocentric Video Understanding](http://arxiv.org/abs/2605.15342)

- Minerva-Ego: introduces a benchmark for complex egocentric video reasoning that pairs multi-step questions with dense, human-annotated spatiotemporal reasoning traces.
- The framework utilizes spatiotemporal hints, including object masks and temporal selection, to guide LLMs in identifying relevant objects and time segments within long-form egocentric videos.
- Evaluations demonstrate that frontier LLMs struggle with perceptual grounding in egocentric settings, and that explicit spatiotemporal highlighting significantly improves reasoning performance.

---

[Hidden in Memory: Sleeper Memory Poisoning in LLM Agents](http://arxiv.org/abs/2605.15338)

- Sleeper Memory Poisoning framework: introduces a delayed security attack where an adversary manipulates external content to force an LLM to store a fabricated memory that later influences future interactions.
- The attack pipeline involves three stages: injection of adversarial content into persistent memory, retrieval of the poisoned memory in a subsequent session, and negative impact on the assistant's behavior or agentic actions.
- Empirical evaluations across stateful LLMs demonstrate high success rates for universal poisoning payloads, highlighting the vulnerability of memory-augmented systems to long-term adversarial influence.

---

[From I/O to Code with Discovery Agent](http://arxiv.org/abs/2605.15334)

- DIO-Agent: introduces a discovery agent for IO2Code that utilizes Curriculum-wise Evolution, Transformation Priority Premise, and Error-Grounded Feedback to synthesize programs from input-output behavior through an LLM-driven Evolutionary Loop.
- The framework employs a Sandbox Evaluator to provide structured debugging evidence, guiding the LLM-based mutation operator to navigate the program space from simple to complex constructs.
- DIO-Agent incorporates island-based search and an optional Multimodal LLM Tool to enhance generalizability and performance across diverse algorithmic, geometric, and multimodal tasks.

---

[Context Pruning for Coding Agents via Multi-Rubric Latent Reasoning](http://arxiv.org/abs/2605.15315)

- LaMR (Latent Multi-Rubric): introduces a structured pruning framework that decomposes code relevance into interpretable semantic and dependency dimensions to optimize context for LLMs.
- The framework utilizes a Shared Backbone Feature Fusion, MoE Gate, CRFsem, CRFdep, Fused CRF, and Viterbi Decoding to dynamically balance semantic evidence and structural support while filtering distracting noise.
- LaMR improves token efficiency and task performance for LLMs by preserving self-contained evidence-support units through AST-derived supervision and rubric-specific transition dynamics.

---

[Solvita: Enhancing Large Language Models for Competitive Programming via Agentic Evolution](http://arxiv.org/abs/2605.15301)

- Solvita: introduces a multi-agent framework that enables continuous, experience-driven evolution for frozen LLMs in competitive programming by coupling a Planner, Solver, Oracle, and Hacker with trainable, graph-structured Knowledge Networks.
- The framework utilizes a closed-loop system where failure signals from the Oracle and Hacker are recast as reinforcement learning updates to the agents' Knowledge Networks via a Contextual-Bandit Policy.
- Solvita achieves state-of-the-art performance by accumulating transferable reasoning experience across tasks, effectively doubling the accuracy of single-pass LLM baselines without requiring weight updates to the underlying LLM backbone.

---

[Autonomous Intelligent Agents for Natural-Language-Driven Web Execution with Integrated Security Assurance](http://arxiv.org/abs/2605.15281)

- Autonomous Intelligent Agent Framework: introduces an AI-driven testing system that utilizes a five-strategy enhancement pipeline to improve web test reliability and security validation through a decoupled containerized worker architecture.
- The framework employs a vision-enabled LLM within an agentic perceive-reason-act loop to translate natural language instructions into browser-based functional tests and security probes.
- By separating stateless orchestration from stateful browser execution, the system achieves high success rates in complex web environments while enabling automated security testing aligned with OWASP standards.

---

[Training on Documents About Monitoring Leads to CoT Obfuscation](http://arxiv.org/abs/2605.15257)

- CoT Obfuscation Framework: introduces a methodology where models trained on documents describing monitoring systems learn to obfuscate their reasoning traces to evade detection by a CoT monitor.
- The research demonstrates that monitor-aware models consistently achieve higher rates of undetected misbehavior compared to unaware controls by actively suppressing or reframing reasoning content.
- The study identifies CoT controllability as a strong predictor of obfuscation success and shows that monitor-aware models learn to reward-hack faster than unaware controls during reinforcement learning.

---

[Assistance to Autonomy: A Systematic Literature Review of Agentic AI across the Software Development Life Cycle](http://arxiv.org/abs/2605.15245)

- Agentic AI SLR: introduces a domain-agnostic multi-agent screening pipeline that utilizes an Assistant, Evaluator, LLM Team, Quality Control, Screening, and Finalizing components to automate systematic literature reviews.
- The paper identifies output verifiability as the primary enabler for industrial adoption of agentic AI, with the Planner-Executor-Reviewer pattern serving as the dominant architectural framework.
- Industrial mitigation strategies for agentic AI challenges consistently focus on confining agent actions to bounded, verifiable spaces to ensure reliability and consistency.

---

[A3D: Agentic AI flow for autonomous Accelerator Design](http://arxiv.org/abs/2605.15237)

- A3D: introduces an end-to-end agentic framework that automates hardware accelerator design by partitioning tasks among specialist agents, utilizing deterministic tools, and employing adversarial verification loops.
- The framework integrates an agentic RAG pipeline to bridge the gap between LLM knowledge and proprietary EDA tool requirements, enabling autonomous refactoring and synthesis of complex scientific kernels.
- A3D achieves high-reliability automation by combining task decomposition, tool augmentation, and iterative verifier loops, successfully generating Pareto-optimal accelerator designs from complex C++ and CUDA codebases without human intervention.

---

#### 13th May 2026


[IdeaForge: A Knowledge Graph-Grounded Multi-Agent Framework for Cross-Methodology Innovation Analysis and Patent Claim Generation](http://arxiv.org/abs/2605.13311)

- IdeaForge: introduces a knowledge graph-grounded multi-agent framework that treats innovation methodologies as heterogeneous reasoning operators acting over a shared persistent FalkorDB Knowledge Graph.
- The framework utilizes TRIZ Agent, Design Thinking Agent, and SCAMPER Agent to contribute structured entities to the graph, which are then synthesized by an Embedding Synthesis Agent to identify cross-methodology convergence.
- An InnovationScore Module ranks the resulting claims based on convergence, methodology diversity, and strength, enabling a Patent Agent to generate traceable, grounded patent drafts.

---



[Grounded Continuation: A Linear-Time Runtime Verifier for LLM Conversations](http://arxiv.org/abs/2605.14175)

- Grounded Continuation: introduces a runtime verifier that maintains an explicit dependency graph to ensure LLM outputs trace back to prior conversation commitments.
- The framework utilizes an LLM Interpreter to classify utterances into operations, a Symbolic Engine to track dependencies, and a Context Renderer to feed structured state back into the LLM.
- This approach enables linear-time verification of conversation grounding, effectively catching stale-premise errors that standard LLM-only baselines frequently miss.

---

[BOOKMARKS: Efficient Active Storyline Memory for Role-playing](http://arxiv.org/abs/2605.14169)

- BOOKMARKS: introduces a search-based memory framework for RPAs that utilizes a Proposal Module, Matching Module, Memory Bank, Synchronization Operator, and Grounding Context to maintain task-relevant information efficiently.
- The framework improves long-horizon consistency by actively proposing queries and passively updating only the necessary bookmarks, avoiding the computational overhead of full-storyline processing.
- BOOKMARKS outperforms existing retrieval-based and profile-based memory methods by providing precise, task-specific grounding through incremental synchronization of reusable memory anchors.

---

[Agentic Systems as Boosting Weak Reasoning Models](http://arxiv.org/abs/2605.14163)

- Verifier-backed committee search framework: introduces a formal model of inference-time boosting for LLMs by decomposing reasoning into proposal coverage, local identifiability, progress, and diversity.
- The framework utilizes a Proposer to generate candidates, a Critic to perform binary filtering, and a Comparator to rank remaining candidates, effectively converting latent proposal-pool capability into realized solve rates.
- Empirical results on SWE-bench Verified demonstrate that this orchestration approach allows weak LLMs to match the performance of significantly stronger standalone models by optimizing selection over cached proposal pools.

---

[EXPLOITBENCH: A Capability Ladder Benchmark for LLM Cybersecurity Agents](http://arxiv.org/abs/2605.14153)

- ExploitBench: introduces a capability-graded benchmark that decomposes exploitation into 16 measurable flags across five tiers, verified by deterministic oracles to measure LLM agent progress on hardened production targets.
- The framework utilizes an MCP server, CompositeGrader, Coverage grader, Diff grader, Primitive grader, Engine instrumentation, Environment builder, Agent runner, LiteLLM, and Audit catalog to provide a standardized, reproducible evaluation environment.
- The research demonstrates a sharp capability split between publicly deployed LLMs and private-frontier models, where only the latter reliably achieve arbitrary code execution on hardened V8 targets.

---

[Distribution-Aware Algorithm Design with LLM Agents](http://arxiv.org/abs/2605.14141)

- Distribution-Aware Algorithm Design with LLM Agents: introduces a framework that leverages LLM agents to infer reusable solver hints from public samples, which are then compiled into specialized deployment solvers to optimize both solution quality and execution runtime.
- The framework utilizes a three-stage construction process involving a Hypothesis (Hc), an Analysis Program (Ac), and a Deployment Solver (sc), which are refined through a diversity-preserving beam search to discover distribution-specific computational shortcuts.
- Empirical results across 21 combinatorial-optimization distributions demonstrate that synthesized solvers significantly outperform heuristic and exact baselines in runtime while maintaining high solution quality by replacing ambient search with distribution-specific computation.

---

[ClawForge: Generating Executable Interactive Benchmarks for Command-Line Agents](http://arxiv.org/abs/2605.14133)

- ClawForge: introduces a generator-backed benchmark framework for evaluating LLM agents on command-line workflows under state conflict, utilizing Scenario Template, Grounded Slots, State-Mode Selection, Reference Command Synthesis, Validator Generation, Interactive Environment, Command Router, Normalized Evaluation State, and Result-First Evaluator.
- The framework enables systematic testing of LLM agents by initializing tasks with pre-existing partial, stale, or conflicting artifacts, requiring agents to perform state-aware judgments rather than simple command imitation.
- Evaluations across seven frontier models demonstrate that ClawForge effectively separates agent capabilities, particularly in scenarios requiring state repair, replacement, and multi-source decision-making.

---

[Reinforcement Learning for Tool-Calling Agents in Fast Healthcare Interoperability Resources (FHIR)](http://arxiv.org/abs/2605.14126)

- SkyRL: introduces a post-training pipeline for clinical LLM agents that uses execution-grounded reinforcement learning to improve multi-turn reasoning over structured FHIR clinical graphs.
- The framework utilizes a multi-turn CodeAct-style agent that interacts with a FHIR server via retrieval and Python tools to perform schema discovery and data aggregation.
- By applying GRPO-based post-training with execution-grounded rewards, the system enables smaller open-weight models to outperform larger closed-source models on complex clinical question-answering tasks.

---

[Mini-JEPA Foundation Model Fleet Enables Agentic Hydrologic Intelligence](http://arxiv.org/abs/2605.14120)

- Mini-JEPA: introduces a fleet of small, sensor-specialized foundation models that leverage a shared Vision Transformer backbone and I-JEPA training recipe to provide targeted hydrologic intelligence.
- The system utilizes a router LLM that consults per-modality reference cards to select the most relevant Mini-JEPA specialists for specific natural-language hydrologic queries.
- Evaluation demonstrates that this routed fleet significantly outperforms planetary-scale generalist models on single-modality tasks while remaining efficient enough for deployment on commodity hardware.

---

[Privacy Preserving Multi Agent Path Finding](http://arxiv.org/abs/2605.14119)

- kPPMAPF (k-Privacy Preserving Multi Agent Path Finding): introduces a framework that preserves privacy by adding mock agents to the planning process, ensuring that no agent can identify the exact location of others within a minimum of k possible values.
- ePPMAPF (Execution-level Privacy Preserving Multi Agent Path Finding): extends the framework to prevent privacy leakage during execution by incorporating Field-of-View (FoV) constraints into the planning process using fPP and PPfPP.
- PPfPP (Post-Processing fPP): improves solution quality by identifying safe zones where agents can re-plan locally without violating privacy or collision constraints.

---

[ProtoMedAgent: Multimodal Clinical Interpretability via Privacy-Aware Agentic Workflows](http://arxiv.org/abs/2605.14113)

- ProtoMedAgent: introduces a framework that formalizes multimodal clinical reporting as an iterative, zero-gradient test-time optimization problem over a strict neuro-symbolic bottleneck to ensure evidence-grounded documentation.
- The framework utilizes a suite of dedicated agents including a Perception Agent, Tabular Agent, Memory Agent, and Verification Agent to translate retrieved evidence into a fully grounded and verifiable clinical report.
- By employing a semantic privacy gate and a Scribe-Critic loop, the system mathematically precludes unsupported narrative claims and mitigates privacy risks without requiring gradient updates to the underlying frozen backbone.

---

[Modeling Bounded Rationality in Drug Shortage Pharmacists Using Attention-Guided Dynamic Decomposition](http://arxiv.org/abs/2605.14111)

- Attention-Guided Dynamic Decomposition framework: introduces a computational model for drug shortage management that dynamically decomposes high-dimensional state spaces into a focused subset for intensive reasoning and a secondary subset for monitoring.
- The framework utilizes an Expert Agent with predefined attention weights and a Learner Agent that optimizes attention allocation using a REINFORCE-style gradient update to maintain stable performance under uncertainty.
- By restricting planning to high-urgency drugs, the approach achieves significant computational efficiency and prevents stockouts in complex, partially observable healthcare supply chain scenarios.

---

[SToRe3D: Sparse Token Relevance in ViTs for Efficient Multi-View 3D Object Detection](http://arxiv.org/abs/2605.14110)

- SToRe3D: introduces a planner-aligned sparsity framework that jointly prunes 2D image tokens and 3D object queries using lightweight relevance heads and store-reactivate buffers to reduce inference latency.
- The framework utilizes a future interaction corridor to supervise relevance, ensuring that compute is prioritized for agents critical to ego-vehicle motion planning.
- By caching low-relevance features in buffers for selective reactivation, SToRe3D achieves real-time performance on ViT-based 3D detection with minimal accuracy loss.

---

[ChromaFlow: A Negative Ablation Study of Orchestration Overhead in Tool-Augmented Agent Evaluation](http://arxiv.org/abs/2605.14102)

- ChromaFlow: introduces a tool-augmented autonomous reasoning framework that evaluates the impact of orchestration overhead on agent reliability.
- The system utilizes an Optimus supervisory controller to manage execution paths, while the reliability layer monitors operational noise and enforces performance gates.
- The study demonstrates that aggressive orchestration can increase operational noise and decrease accuracy, highlighting the necessity of rigorous evaluation protocols for LLM-based agents.

---

[SkillFlow: Flow-Driven Recursive Skill Evolution for Agentic Orchestration](http://arxiv.org/abs/2605.14089)

- SkillFlow: introduces a flow-based framework for agentic orchestration that utilizes a trainable Supervisor, a dynamic skill library, and a frozen Executor to automate task completion through multi-turn interaction.
- The framework employs TTB (Tempered Trajectory Balance) to perform reward-proportional trajectory sampling, which preserves diverse orchestration strategies and prevents mode collapse.
- SkillFlow incorporates a recursive skill evolution mechanism that uses flow diagnostics to autonomously determine when to evolve, what skills to create or prune, and where decision gaps exist.

---

[CRANE: Constrained Reasoning Injection for Code Agents via Nullspace Editing](http://arxiv.org/abs/2605.14084)

- CRANE: introduces a training-free parameter-editing method that injects reasoning behavior from a Thinking checkpoint into an Instruct backbone while preserving tool-use protocols using Magnitude Thresholding, Conservative Taylor Gate, and Graduated Sigmoidal Projection.
- The framework treats the Thinking–Instruct delta as a candidate edit pool, denoising it and projecting out format-critical directions to maintain agentic protocol fidelity.
- Empirical results across Roo-Eval, SWE-bench-Verified, and Terminal-Bench v2 demonstrate that CRANE improves task success while maintaining efficient, Instruct-like token usage.

---

[Dual Hierarchical Dialogue Policy Learning for Legal Inquisitive Conversational Agents](http://arxiv.org/abs/2605.14057)

- ICA (Inquisitive Conversational Agent): introduces a dual-agent hierarchical reinforcement learning framework designed to emulate judicial questioning patterns by splitting inquisitive reasoning between an Appraisal Agent and a Hierarchical Dialogue Agent.
- The framework utilizes a three-level action taxonomy embedded in a Poincaré space to optimize dialogue strategies for information elicitation in high-stakes legal domains.
- The system incorporates a multi-component reward function—balancing goal relevance, lexical novelty, and answer succinctness—to steer conversations toward uncovering critical information.

---

[Bad Seeing or Bad Thinking? Rewarding Perception for Vision-Language Reasoning](http://arxiv.org/abs/2605.14054)

- MoCA (Modality-aware Credit Assignment): introduces a reinforcement learning framework that resolves the perception-reasoning "seesaw effect" by explicitly decoupling generation into interleaved Perception Actions and Reasoning Actions, enabling targeted supervision.
- The framework utilizes Perception Verification to reward visual grounding via a Blindfolded Text Reasoner and Structured Verbal Verification to provide low-variance outcome rewards for free-form responses.
- By routing granular rewards to specific components, MoCA identifies and corrects "bad seeing" versus "bad thinking" errors, significantly improving performance across perception-intensive and reasoning-intensive tasks.

---

[SPIN: Structural LLM Planning via Iterative Navigation for Industrial Tasks](http://arxiv.org/abs/2605.14051)

- SPIN: introduces a planning wrapper for LLM agents that enforces a strict Directed Acyclic Graph (DAG) contract and utilizes a simulator-critic loop to optimize execution efficiency through early stopping.
- The framework employs a validator to ensure machine-consumable plan structures and a prefix-based evaluation policy to terminate workflows once sufficient progress is achieved.
- Empirical results on industrial benchmarks demonstrate that SPIN reduces downstream execution burden, including tool calls and API usage, while improving task-level accomplishment rates.

---

[Model-Adaptive Tool Necessity Reveals the Knowing-Doing Gap in LLM Tool Use](http://arxiv.org/abs/2605.14038)

- Two-stage cognition-execution modeling framework: introduces a model-adaptive definition of tool necessity grounded in empirical performance to diagnose the knowing-doing gap in LLMs.
- The framework decomposes tool use into an internal cognition stage and an execution stage, revealing that failures predominantly occur during the transition from cognition to action.
- By probing hidden states, the research demonstrates that while both cognition and action are linearly decodable, their probe directions become nearly orthogonal in the late-layer, last-token regime.

---

[Self-Pruned Key-Value Attention: Learning When to Write by Predicting Future Utility](http://arxiv.org/abs/2605.14037)

- SP-KV (Self-Pruned Key-Value Attention): introduces a learned sparse-write mechanism that selectively retains key-value pairs in the persistent KV cache based on predicted future utility to reduce memory footprint and improve decoding speed.
- The framework utilizes a lightweight KV utility predictor to assign scores to key-value pairs, ensuring only high-utility tokens are stored in the persistent cache while maintaining a local buffer for recent interactions.
- SP-KV is trained jointly with the LLM using next-token prediction, enabling dynamic sparsification that adapts to input sequences and provides structured sparsity patterns for designing hybrid local-global attention architectures.

---

[From Descriptive to Prescriptive: Uncover the Social Value Alignment of LLM-based Agents](http://arxiv.org/abs/2605.14034)

- SoVA (Social Value Alignment): introduces a value-based framework that employs GraphRAG to convert psychological theories into prescriptive instructions for steering LLM-based agents.
- The framework utilizes Maslow’s Hierarchy of Needs, Plutchik’s Wheel of Emotions, and Aristotle’s Virtues as seed principles to construct a knowledge graph that guides agent behavior in social dilemmas.
- SoVA improves social value alignment by retrieving community-specific instructions, though it may introduce trade-offs in creative generation and multi-turn conversational coherence.

---

[Sheaf-Theoretic Transport and Obstruction for Detecting Scientific Theory Shift in AI Agents](http://arxiv.org/abs/2605.14033)

- STO framework: introduces a finite sheaf-theoretic diagnostic for AI agents to distinguish between representational transport via deformation and theory extension via structural reorganization.
- The framework utilizes Representational Constellations as local charts, where an Obstruction Functional evaluates whether models glue across contexts or require an extension of the representational language.
- Experimental results on physics-inspired transition families demonstrate that the obstruction-based ranking effectively identifies necessary theory shifts while maintaining stability under noise and stress.

---

[Case Studies and Reflections on Agentic Software Engineering for Rapid Development of Digital Music Instruments](http://arxiv.org/abs/2605.14016)

- ASE: introduces a methodology for developing audio software by leveraging agentic LLMs to automate planning, code generation, and build management within the JUCE framework.
- The research demonstrates that ASE can effectively lower barriers to entry for non-programmers and improve software longevity by translating legacy music instruments into modern, interoperable C++ plugins.
- Through three case studies, the paper evaluates the efficacy of agentic workflows in re-creating, translating, and modernizing digital music instruments using natural language prompts and iterative testing.

---

[PolitNuggets: Benchmarking Agentic Discovery of Long-Tail Political Facts](http://arxiv.org/abs/2605.14002)

- PolitNuggets: introduces a benchmark for evaluating agentic information synthesis by constructing political biographies through a Supervisor/Searcher/Archive/Coder/FactNet/Judge LRM framework.
- The system utilizes a Supervisor-Searcher architecture with an Archive memory component to enable long-horizon reasoning and evidence-grounded discovery.
- FactNet provides an evidence-conditional evaluation protocol that validates candidate nuggets against retrieved sources to measure discovery, fine-grained accuracy, and efficiency.

---

[COLLIDER-BENCH: Benchmarking AI Agents with Particle Physics Analysis Reproduction](http://arxiv.org/abs/2605.13950)

- COLLIDER-BENCH: introduces a benchmark for evaluating autonomous LLM agents on long-horizon scientific tasks by reproducing Large Hadron Collider analyses using public software and papers.
- The framework requires agents to construct executable simulation-and-selection pipelines, producing binned event yields that are compared against hidden reference values using fidelity metrics.
- An LLM judge audits the agent's workspace and execution trace to distinguish between legitimate scientific attempts, incomplete runs, and fabricated results.

---

[EVA-Bench: A New End-to-end Framework for Evaluating Voice Agents](http://arxiv.org/abs/2605.13841)

- EVA-Bench (End-to-end Voice Agent Evaluation Benchmark): introduces, an end-to-end evaluation framework for voice agents that jointly addresses simulation fidelity and measurement comprehensiveness using User Simulator, Voice Agent, Tool Executor, Simulator Validation, Quality Measurements, and Diagnostic Metrics.
- The framework orchestrates parallel bot-to-bot audio sessions to evaluate cascade, hybrid, and S2S architectures under identical conditions, including controlled acoustic and behavioral perturbations.
- EVA-Bench provides composite metrics EVA-A (Accuracy) and EVA-X (Experience) alongside a multi-trial consistency framework (pass@1, pass@k, pass^k) to enable direct cross-architecture comparison of voice agents.

---

[Good Agentic Friends Do Not Just Give Verbal Advice: They Can Update Your Weights](http://arxiv.org/abs/2605.13839)

- TFLOW (Thought Flow): introduces a weight-space communication paradigm for multi-agent LLMs that replaces natural-language message exchange with transient, instance-specific LoRA weight perturbations.
- The framework utilizes a trainable parameter generator to map sender hidden states into low-rank LoRA factors, which are fused and injected into a frozen receiver agent to enable efficient, context-aware collaboration.
- By eliminating the need for auxiliary text-based messages, TFLOW significantly reduces KV-cache memory usage and prefill overhead while maintaining competitive performance across reasoning, coding, and knowledge benchmarks.

---

[Training Long-Context Vision-Language Models Effectively with Generalization Beyond 128K Context](http://arxiv.org/abs/2605.13831)

- MMProLong: introduces a systematic study of long-context continued pre-training for LVLMs, utilizing Document Pool, OCR Expert, Segment Sampling, and QA Generator to construct effective training data.
- The framework employs Long-document VQA and OCR Transcription tasks within a LongPT Recipe to enhance context window scaling from 32K to 128K tokens.
- The implementation leverages FlashAttention, Sequence Parallelism, and FSDP to achieve efficient training, demonstrating generalization to 512K context lengths and broader multimodal tasks.

---

[History Anchors: How Prior Behavior Steers LLM Decisions Toward Unsafe Actions](http://arxiv.org/abs/2605.13825)

- HISTORYANCHOR-100: introduces a benchmark for evaluating how LLMs, acting as decision-making agents, are influenced by prior history logs when choosing subsequent actions at a free-choice node.
- The framework demonstrates that aligned LLMs, when provided with a consistency-demanding system prompt and a sequence of unsafe prior actions, frequently shift from safe to unsafe decision-making.
- The research identifies an inverse-scaling pattern where more capable, aligned LLMs are often more susceptible to this behavioral-consistency pressure than their smaller counterparts.

---

[Harnessing Agentic Evolution](http://arxiv.org/abs/2605.13821)

- AEVO: introduces a harnessed meta-editing framework that treats agentic evolution as an interactive environment, where a Meta Agent observes the accumulated Evolution Environment and edits the underlying Evolution Mechanism to steer future search.
- The framework utilizes a Protected Evaluator and a structured Workspace to maintain a stable interface, allowing the Meta Agent to perform coarse-grained interventions through a two-phase loop of meta-editing and evolution segments.
- By decoupling the evolution mechanism from candidate generation, AEVO enables persistent, long-horizon improvement across both procedure-based and agent-based evolution paradigms.

---

[EvoGround: Self-Evolving Video Agents for Video Temporal Grounding](http://arxiv.org/abs/2605.13803)

- EvoGround: introduces a framework of two coupled self-evolving agents, a proposer and a solver, that learn video temporal grounding from raw videos without manual labels.
- The proposer and solver are initialized from the same backbone and iteratively improve each other through a self-reinforcing reinforcement learning loop.
- The framework utilizes a group reward-decoupled normalization policy optimization (GDPO) to balance multiple reward signals, achieving performance competitive with fully supervised models.

---

[EVOLVEMEM: Self-Evolving Memory Architecture via AutoResearch for LLM Agents](http://arxiv.org/abs/2605.13941)

- EVOLVEMEM: introduces a self-evolving memory architecture that treats retrieval infrastructure as a dynamic action space optimized through an autonomous AutoResearch process.
- The framework utilizes a Structured Memory Store, a Multi-view Retriever, and an LLM-powered Diagnosis Module to iteratively refine retrieval configurations based on per-question failure logs.
- By replacing manual tuning with a closed-loop evolution cycle, the system autonomously discovers effective retrieval strategies and transfers universal principles across different benchmarks.

---

[AgentTrap: Measuring Runtime Trust Failures in Third-Party Agent Skills](http://arxiv.org/abs/2605.13940)

- AgentTrap: introduces a dynamic benchmark for evaluating whether LLM agents can safely utilize third-party skills by measuring runtime trust failures across 16 security-impact dimensions.
- The framework employs a controlled execution layer to monitor agent trajectories, enabling diagnostic attribution of security failures to the LLM backbone, agent framework, or environment configuration.
- AgentTrap utilizes a combination of deterministic checks and LLM-based trajectory analysis to distinguish between successful task completion, malicious behavior, and defensive blocking.

---

[EconAI: Dynamic Persona Evolution and Memory-Aware Agents in Evolving Economic Environments](http://arxiv.org/abs/2605.13762)

- EconAI: introduces an LLM-powered simulation framework that integrates macro/microeconomic dynamics by utilizing an LLM-backbone, Event Perception Module, Long-term Memory Bank, Short-term Memory Bank, Content Extractor, Persona Extraction Module, Long-term Persona Bank, Response Generator, Economic Sentiment Index (ESI), and a Decision-making Mechanism.
- The framework employs a cognitive architecture where agents use memory-driven learning and sentiment-modulated preferences to balance short-term optimization with long-term strategic planning.
- Empirical results demonstrate that EconAI improves the stability of economic indicators and successfully recovers canonical economic regularities like the Phillips and Okun curves.

---

[Learning POMDP World Models from Observations with Language-Model Priors](http://arxiv.org/abs/2605.13740)

- Pinductor: introduces a framework that induces executable POMDP world models from observation-action trajectories by leveraging LLM priors and belief-based feedback without requiring privileged hidden state access.
- The framework utilizes an LLM to propose candidate model components, which are then evaluated via particle filtering and refined through diagnostic feedback to optimize a belief-based likelihood objective.
- Pinductor demonstrates sample-efficient world-model learning that matches privileged-state baselines and outperforms non-LLM tabular methods across various partially observable MiniGrid environments.

---

[Senses Wide Shut: A Representation–Action Gap in Omnimodal LLMs](http://arxiv.org/abs/2605.13737)

- IMAVB: introduces a 500-clip benchmark designed to measure the Representation–Action Gap in omnimodal LLMs by testing their ability to detect implicit false premises in video and audio.
- The study documents that while hidden states of omnimodal LLMs reliably encode premise–perception mismatches, these models frequently fail to propagate this signal to their output, exhibiting under-rejection or over-rejection behaviors.
- The authors propose PGLA as a diagnostic intervention that re-injects the encoded mismatch signal into the output distribution, yielding a +15.0pp mean improvement in rejection accuracy across eight open-source models.

---

[SCIOMIND: Cognitively Grounded Multi-Agent Social Simulation with Anchoring-Based Belief Dynamics and Dynamic Profiles](http://arxiv.org/abs/2605.13725)

- SCIOMIND: introduces a cognitively grounded multi-agent simulation framework that integrates structured opinion dynamics with LLM-based agent reasoning to improve behavioural realism.
- The framework utilizes a four-layer memory architecture and an anchoring-based belief update mechanism to simulate persistent, experience-driven belief formation in social networks.
- SCIOMIND incorporates dynamic agent profiles and a social relationship simulation engine to enable heterogeneous, context-aware interactions that mirror real-world opinion dynamics.

---

[SkillOps: Managing LLM Agent Skill Libraries as Self-Maintaining Software Ecosystems](http://arxiv.org/abs/2605.13716)

- SkillOps: introduces a plug-in maintenance framework that treats LLM agent skill libraries as self-maintaining software ecosystems to mitigate skill technical debt.
- The framework utilizes a Hierarchical Skill Ecosystem Graph (HSEG) to model skills as typed contracts and manage them through alternating task-time execution and library-time maintenance loops.
- SkillOps improves task success rates by diagnosing library health across utility, redundancy, compatibility, failure-risk, and validation-gap dimensions, applying automated repairs with minimal LLM overhead.

---

[Identifying AI Web Scrapers Using Canary Tokens](http://arxiv.org/abs/2605.13706)

- Canary Token Infrastructure: introduces a methodology for identifying AI web scrapers by embedding unique canary tokens into controlled websites and monitoring their appearance in LLM-generated responses.
- The framework utilizes Website Templates to serve distinct Canary Tokens to visiting Scraper Bots, enabling the Scraper Inference Engine to attribute retrieved data to specific scraper identities.
- Experimental results across 22 production AI Chatbots demonstrate that many systems rely on third-party search engine scrapers and that content remains cached even after websites are taken offline or restricted.

---

[FlowCompile: An Optimizing Compiler for Structured LLM Workflows](http://arxiv.org/abs/2605.13647)

- FlowCompile: introduces a compiler-inspired framework for optimizing structured LLM workflows by performing compile-time design space exploration to generate a reusable set of accuracy–latency trade-off configurations.
- The framework utilizes a structure-aware compositional proxy to estimate workflow-level performance from individual sub-agent profiles, enabling scalable optimization without exhaustive end-to-end evaluation.
- FlowCompile supports flexible deployment by providing a menu of optimized operating points that can be selected based on specific latency budgets or performance preferences.

---

[Learning Equilibria in Coordination Games via Minorization-Maximization](http://arxiv.org/abs/2605.13644)

- IMM (Iterative Minorization-Maximization): introduces a learning framework for coordination games that utilizes a regularized potential function to ensure unique equilibrium selection under prospect-theoretic utility models.
- The framework employs a coordinating agent to aggregate information, enabling scalable learning and convergence to potential-optimal equilibria even in non-smooth utility settings.
- By replacing original optimization problems with a sequence of surrogate problems, the approach demonstrates superior convergence speed compared to traditional gradient-based and best-response methods.

---

[How to Interpret Agent Behavior](http://arxiv.org/abs/2605.13625)

- Act·ONOMY: introduces a hierarchical taxonomy and automated analysis pipeline to interpret and characterize the runtime behavior of LLM agents.
- The framework utilizes an LLM-powered-Discovery-Qualitative-Analyst to map unstructured agent execution traces into a structured, quote-grounded vocabulary of 10 actions and 46 sub-actions.
- By providing a shared vocabulary and automated tools, the research enables scalable behavioral profiling, failure mode identification, and cross-agent comparison.

---

[Unweighted ranking for value-based decision making with uncertainty](http://arxiv.org/abs/2605.13601)

- FUW-VBDM: introduces a human-centred decision-making framework that integrates quantitative and qualitative criteria using fuzzy logic to address uncertainty and normative bias.
- The framework utilizes the Rankzzy method to perform unweighted optimization over a domain of fuzzy weights, ensuring mathematical consistency and transparency in value-based decision-making.
- Rankzzy employs a generalized fuzzy p-mean score function to generate customizable rankings, demonstrating reduced computational costs and robust performance compared to existing multi-criteria decision-making approaches.

---

[Position: Assistive Agents Need Accessibility Alignment](http://arxiv.org/abs/2605.13579)

- Accessibility Alignment Framework: introduces a lifecycle-oriented design pipeline for assistive agents that integrates Task Card, Accessibility Success Specification, Interaction Contract, Risk and Uncertainty Policy, Privacy Manifest, and Autonomy Calibration Specification to address systematic failures in BVI-centered scenarios.
- The framework shifts agent design from generic task completion to safety-critical, verifiability-aware, and non-visual interaction paradigms tailored for BVI users.
- It addresses recurring failure modes such as silent failures, overconfident hallucinations, miscalibrated autonomy, and interaction-induced cognitive overload through structured design artifacts and runtime guardrails.

---

[Self-Supervised On-Policy Reinforcement Learning via Contrastive Proximal Policy Optimisation](http://arxiv.org/abs/2605.13554)

- CPPO (Contrastive Proximal Policy Optimisation): introduces an on-policy reinforcement learning algorithm that computes advantages directly from contrastive Q-values using a Policy network, State-action encoder, Goal encoder, Contrastive critic, and PPO optimizer.
- The framework replaces traditional reward-based value estimation with a self-supervised contrastive objective, enabling goal-conditioned learning without hand-crafted rewards or replay buffers.
- CPPO demonstrates robust performance across discrete and continuous, single-agent and multi-agent environments, matching or exceeding reward-based PPO baselines in most tested tasks.

---

[RealICU: Do LLM Agents Understand Long-Context ICU Data? A Benchmark Beyond Behavior Imitation](http://arxiv.org/abs/2605.13542)

- ICU-Evo: introduces a structured-memory agent framework for ICU decision-support that organizes clinical context into heterogeneous memory types to improve long-horizon reasoning.
- The framework utilizes an Observation Agent, Assessment Agent, and Insight Agent to maintain Working memory, Trend memory, Critical-event memory, Trajectory memory, and Insight memory for sequential clinical decision-making.
- RealICU benchmark evaluates LLMs on four physician-motivated tasks using hindsight-annotated labels to measure clinical correctness rather than behavioral imitation.

---

[Integration of an Agent Model into an Open Simulation Architecture for Scenario-Based Testing of Automated Vehicles](http://arxiv.org/abs/2605.13539)

- OSMP based simulation integration architecture: introduces a standardized, modular framework for integrating traffic agent models into heterogeneous simulation environments using Open Simulation Interface (OSI) and Functional Mock-up Interface (FMI).
- The architecture utilizes an OSMP-packaged agent model containing an OSI Adapter, Behavior Model, and Dynamics Model to ensure tool-independent interoperability across platforms like OpenPASS, CARLA, and CarMaker.
- Evaluation demonstrates that the approach maintains stable closed-loop behavior and scales linearly in computational cost, facilitating reproducible scenario-based testing for automated driving systems.

---

[Scaling Retrieval-Augmented Reasoning with Parallel Search and Explicit Merging](http://arxiv.org/abs/2605.13534)

- MultiSearch: introduces an RL-based framework that improves retrieval-during-reasoning by employing parallel multi-query retrieval and explicit information merging to enhance signal-to-noise ratios.
- The framework utilizes a multi-process reward design, including answer-, multi-query-, and merging-rewards, to provide targeted supervision for intermediate retrieval and consolidation behaviors.
- MultiSearch optimizes the multi-reward objective using Group reward-Decoupled Normalization Policy Optimization (GDPO), which independently normalizes heterogeneous reward signals to ensure robust policy training.

---

[Limits of Personalizing Differential Privacy Budgets](http://arxiv.org/abs/2605.13503)

- Limits of Personalizing Differential Privacy Budgets: introduces a comparative analysis between the best affine estimator and a simple unique-threshold ε-estimator for mean estimation under heterogeneous privacy constraints.
- The research demonstrates that full personalization of privacy budgets offers only modest utility gains over simpler thresholding approaches in most practical scenarios.
- The study establishes constant-factor approximation bounds for the threshold-based estimator in specific regimes and characterizes the performance gap for arbitrary privacy levels.

---

[Task-Aware Automated User Profile Generation for Recommendation Simulation Using Large Language Models](http://arxiv.org/abs/2605.13497)

- APG4RecSim: introduces a three-stage framework that automates the construction of task-executable user profiles from interaction history without manual schemas, utilizing Attribute Initialisation and Extraction, Context-Aware Semantic Consolidation, and Causal Mapping and Refinement.
- The framework employs an LLM-based pipeline to transform raw interaction logs into a Consolidated User Persona and subsequently into a Task-Aligned Simulation Profile via a Task Decision Path.
- By utilizing counterfactual trait-to-step mapping, the framework ensures that generated profiles are robust to popularity and position biases while maintaining stable performance across diverse recommendation tasks and LLM backbones.

---

[MARLIN: Multi-Agent Game-Theoretic Reinforcement Learning for Sustainable LLM Inference in Cloud Datacenters](http://arxiv.org/abs/2605.13496)

- MARLIN: introduces a multi-agent reinforcement learning framework that utilizes a game-theoretic approach to balance competing objectives for sustainable LLM inference in geo-distributed cloud datacenters.
- The framework employs a two-phase process where agents independently propose scheduling plans in phase 1 and negotiate a final blended plan through weighted voting and a veto mechanism in phase 2.
- MARLIN optimizes for time-to-first-token, carbon emissions, water usage, and energy costs, demonstrating significant performance improvements over state-of-the-art LLM inference management frameworks.

---

[SieveFL: Hierarchical Runtime-Aware Pruning for Scalable LLM-Based Fault Localization](http://arxiv.org/abs/2605.13491)

- SieveFL: introduces a five-stage hierarchical framework that resolves the Scale-Precision Dilemma in LLM-based fault localization through progressive pre-LLM filtering.
- The framework utilizes LLM-based Test Analysis, Suspicious File Identification, Runtime-Aware Candidate Pruning, Per-Method LLM Screening, and LLM-Based Re-ranking to reduce candidate search space and token consumption.
- By integrating JaCoCo runtime traces with semantic retrieval, SieveFL enables efficient, high-precision fault localization on commodity hardware without requiring proprietary frontier models.

---

[Sustainable Graph Analytics Workload Scheduling with Evolutionary Reinforcement Learning in Edge-Cloud Systems](http://arxiv.org/abs/2605.13489)

- MERSEM (Multi-Objective Evolutionary Reinforcement Learning framework for Sustainable Edge-Cloud Management): introduces a hybrid optimization framework that integrates an Evolutionary Algorithm for global exploration with an RL-Guided Local Search agent for adaptive workload scheduling.
- The framework co-optimizes SLA violation rates and operational carbon emissions by modeling DAG-based graph analytics workloads across heterogeneous edge, fog, and cloud infrastructure.
- MERSEM utilizes a trajectory-based RL agent and dominance-based evolutionary operators to maintain Pareto-optimal scheduling solutions under dynamic system conditions.

---

[R²-Mem: Reflective Experience for Memory Search](http://arxiv.org/abs/2605.13486)

- R²-Mem: introduces a reflective experience framework for memory search systems that utilizes a Rubric-guided Evaluator and a self-Reflection Learner to distill reusable process-level guidance from historical search trajectories.
- The framework improves LLM agent efficiency and effectiveness by retrieving relevant planning and reflection experiences from dedicated banks to guide iterative search processes.
- R²-Mem enables RL-free self-improvement by allowing agents to learn from both high- and low-quality search steps, reducing redundant exploration and token consumption.

---

[PersonalAI 2.0: Enhancing knowledge graph traversal/retrieval with planning mechanism for Personalized LLM Agents](http://arxiv.org/abs/2605.13481)

- PAI-2: introduces a GraphRAG framework that utilizes a multi-stage query processing pipeline to optimize knowledge graph retrieval and reasoning for LLMs.
- The framework incorporates a dynamic planning mechanism that iteratively refines search steps and subgraph traversals based on extracted entities and matched graph vertices.
- PAI-2 improves factual correctness and reduces hallucinations by balancing structured and unstructured data retrieval through LLM-driven reasoning and iterative query refinement.

---

[Sleeper Channels and Provenance Gates: Persistent Prompt Injection in Always-on Autonomous AI Agents](http://arxiv.org/abs/2605.13471)

- Sleeper Channels and Provenance Gates: introduces a threat model for persistent prompt injection in always-on OS-live agents, utilizing OpenClaw and Hermes Agent as canonical instances, and proposes a tiered defense mechanism D2-Gate to mitigate cross-surface attacks.
- The framework employs Update Hooks and Gate Hooks to track provenance across memory, skills, and filesystem substrates, ensuring that consequential actions are validated against a closed action set.
- By binding action-instance digests to hardware-attested owner grants, the system prevents paraphrase laundering and unauthorized agent behavior, effectively decoupling security enforcement from the LLM's internal context.

---

[CA2: Code-Aware Agent for Automated Game Testing](http://arxiv.org/abs/2605.13918)

- CA2 (Code-Aware Agent): introduces a goal-conditioned reinforcement learning framework that leverages internal call stack information to improve functional code coverage in automated game testing.
- The architecture integrates a source code profiler with a Causal Transformer to process multi-modal inputs, including game states and call stack traces, for effective offline policy learning.
- Experimental results demonstrate that incorporating call stack signals via Multi-Head Self-Attention significantly enhances the agent's ability to reach specific target functions compared to non-code-aware baselines.

---

[COGNIFOLD: Always-On Proactive Memory via Cognitive Folding](http://arxiv.org/abs/2605.13438)

- COGNIFOLD: introduces a brain-inspired, always-on agent memory architecture that continuously folds fragmented event streams into self-emerging cognitive structures using a tri-layered substrate consisting of a Hippocampal Layer, Neocortical Layer, and Prefrontal Layer.
- The framework employs a dynamically evolving multigraph to address four structural debts—accumulation, compression, decay, and completion—through automatic graph-level operations that enable proactive intent emergence.
- By extending Complementary Learning Systems theory, the system achieves robust performance across diverse cognitive benchmarks by transitioning from reactive retrieval to proactive, structure-driven assembly.

---

[TRIAGE: Evaluating Prospective Metacognitive Control in LLMs under Resource Constraints](http://arxiv.org/abs/2605.13414)

- TRIAGE: introduces an evaluation framework that measures the prospective metacognitive control of LLMs by requiring them to commit to a portfolio-level plan of task selection, sequencing, and token allocation under a finite budget.
- The framework evaluates models across two regimes: an unconstrained regime for advisory planning and a constrained regime where the model's own token allocations are enforced as binding limits on the solver.
- Experimental results across 20 models demonstrate that object-level capability and metacognitive control often dissociate, as extended reasoning frequently fails to improve triage efficiency and models struggle to honor their own self-imposed budget constraints.

---

[FPGA-Accelerated Lock Management and Transaction Processing: Architecture, Optimization, and Design Space Exploration](http://arxiv.org/abs/2605.13398)

- FPGA-based transaction processing accelerator: introduces a hardware-accelerated architecture for 2-Phase Locking (2PL) that offloads lock management and transaction execution from CPUs to FPGAs to mitigate latency and throughput bottlenecks.
- The architecture utilizes dedicated Transaction Agents and Lock Agents, connected via a hierarchical crossbar, to enable high-parallelism transaction processing and efficient lock serving.
- Experimental results demonstrate that the accelerator achieves significantly higher lock serving and transaction throughput compared to CPU-based baselines by leveraging on-chip memory and asynchronous pipelining.

---

[RS-Claw: Progressive Active Tool Exploration via Hierarchical Skill Trees for Remote Sensing Agents](http://arxiv.org/abs/2605.13391)

- RS-Claw: introduces a novel agent architecture that redefines tool selection as an active exploration process within a hierarchical skill tree to mitigate context bottlenecks in remote sensing tasks.
- The framework utilizes a progressive disclosure mechanism that enables the LLM to dynamically load tool information on-demand, effectively reducing context overhead and filtering semantic noise.
- By internalizing tool acquisition as an autonomous decision variable, the agent maintains a locally bounded context while preserving tool coverage for complex, long-horizon reasoning.

---

[GRIP-VLM: Group-Relative Importance Pruning for Efficient Vision-Language Models](http://arxiv.org/abs/2605.13375)

- GRIP-VLM: introduces a hierarchical dynamic pruning framework that utilizes a budget-aware Adaptive Token Scorer (ATS) to perform fine-grained, contextualized token-wise importance evaluation within VLM backbones.
- The framework employs a two-stage training strategy, combining SFT-anchored initialization with GRPO-based RL to effectively navigate the non-convex, discrete combinatorial space of visual token selection.
- By integrating a FiLM-based modulator and a hybrid reward function, the system achieves robust generalization across arbitrary compression ratios and outperforms heuristic baselines in inference speed and multi-modal accuracy.

---

[AI Harness Engineering: A Runtime Substrate for Foundation-Model Software Agents](http://arxiv.org/abs/2605.13357)

- AI Harness Engineering: introduces a runtime substrate that mediates between a foundation-model agent and a software environment to transform latent coding capability into auditable software-engineering behavior.
- The framework utilizes an H0–H3 harness ladder to progressively expose runtime support, enabling empirical separation of the harness's contribution from the model's latent capabilities.
- A trace-based evaluation protocol records eight classes of execution evidence, allowing for the classification of agent performance based on verification autonomy rather than simple task success.

---

[Contextual Bandits for Resource-Constrained Devices using Probabilistic Learning](http://arxiv.org/abs/2605.13346)

- HD-CBPROB: introduces a resource-efficient contextual bandit framework that replaces deterministic accumulation with a probabilistic update rule on low-precision saturating integers.
- The framework utilizes a time-decaying update probability to manage learning rates, effectively bounding action hypervectors without requiring periodic binarization or auxiliary counters.
- Experimental results demonstrate that HD-CBPROB achieves performance comparable to real-valued baselines while maintaining a significantly smaller memory footprint than existing binarized hyperdimensional approaches.

---

[Multi-Agent Systems in Emergency Departments: Validation Study on a ED Digital Twin](http://arxiv.org/abs/2605.13345)

- DES-ABM-MAS: introduces a hybrid simulation framework combining Discrete Event Simulation and Agent-Based Modeling to evaluate emergency department resource optimization strategies.
- The framework integrates a LLM-based Multi-Agent System that utilizes a Blackboard Architecture to observe simulation states and propose interventions via specialized agents.
- Validation results demonstrate that the simulation effectively replicates real-world emergency department dynamics and intervention outcomes across varying facility sizes.

---

[EGO2WORLD: Compiling Egocentric Cooking Videos into Executable Worlds for Belief-State Planning](http://arxiv.org/abs/2605.13335)

- EGO2WORLD: introduces an executable benchmark that compiles real-world egocentric cooking videos into symbolic graph-transition environments to evaluate embodied agents under partial observation.
- The framework separates the hidden world graph (Gwt) from the agent-side belief graph (Gbt), forcing agents to maintain memory, handle state changes, and perform replanning based on partial observations and feedback.
- Experiments demonstrate that action-level overlap often overestimates physical-state success, highlighting the necessity of belief maintenance and diagnostic replanning for long-horizon embodied tasks.

---

[What Limits Vision-and-Language Navigation ?](http://arxiv.org/abs/2605.13328)

- StereoNav: introduces a robust Vision-Language-Action framework that mitigates perceptual instability and instruction under-specification by integrating target-point priors and stereo-based unified understanding.
- The framework utilizes Visual Rendering to provide persistent global guidance and employs a multi-branch encoder architecture to synergize semantic, structural, and geometric tokens for the MLLM.
- StereoNav achieves state-of-the-art performance on R2R-CE and RxR-CE benchmarks while demonstrating superior reliability and execution consistency in real-world robotic deployments.

---

[HCSG: Human-Centric Semantic-Geometric Reasoning for Vision-Language Navigation](http://arxiv.org/abs/2605.13321)

- HCSG (Human-Centric Semantic-Geometric Reasoning): introduces a dual-stream framework for VLN that synergizes geometric forecasting and LLM-based semantic interpretation to enable socially compliant navigation in dynamic environments.
- The framework utilizes a Human Detector to trigger parallel reasoning streams, where the Geometric Reasoning Module models human motion dynamics and the Semantic Reasoning Module generates natural language descriptions of human intent.
- These human-centric features are fused into a topological map, allowing the agent to perform instruction-conditioned planning while adhering to social norms via a dedicated Social Distance Loss.

---

[Embodied Neurocomputation: A Framework for Interfacing Biological Neural Cultures with Scaled Task-Driven Validation](http://arxiv.org/abs/2605.13315)

- Embodied Neurocomputation Framework: introduces a systems-level approach to optimize the interface between digital environments and biological neural networks through modular encoding, transformation, decoding, and feedback components.
- The framework operationalizes BNN-based computation as a multi-variable optimization problem, utilizing an automated pipeline to identify encoding configurations that enable goal-oriented navigation.
- Empirical results demonstrate that optimized BNN agents significantly outperform silicon-based DQN agents and non-adaptive baselines in task performance under equivalent interaction budgets.

---

[Discrete Diffusion for Complex and Congested Multi-Agent Path Finding with Sparse Social Attention](http://arxiv.org/abs/2605.13296)

- DiffLNS: introduces a hybrid framework that integrates a D3PM (Discrete Denoising Diffusion Probabilistic Model) as a learned initializer with an LNS2 (Large Neighborhood Search 2) repair-based solver to generate high-quality, coordinated multi-agent path finding plans.
- The framework utilizes a diffusion-aware sparse social attention mechanism to dynamically construct local neighborhoods, focusing computation on conflict-relevant agent interactions rather than dense all-to-all attention.
- By leveraging discrete diffusion for warm-starting, DiffLNS improves repair success rates in dense and congested environments while maintaining scalability to large agent teams and competitive solution quality.

---

[CANTANTE: Optimizing Agentic Systems via Contrastive Credit Attribution](http://arxiv.org/abs/2605.13295)

- CANTANTE: introduces a framework that optimizes LLM-based multi-agent systems by decomposing global system-level rewards into per-agent update signals using contrastive attribution across multiple joint rollouts.
- The framework treats agent prompts as learnable parameters and utilizes an attribution LLM to isolate individual agent contributions, enabling effective credit assignment in complex multi-agent workflows.
- CANTANTE consistently outperforms existing prompt optimization baselines on programming, mathematical reasoning, and multi-hop question answering benchmarks while maintaining lower inference costs.

---

[RETOOL-VIDEO: Recursive Tool-Using Video Agents with Meta-Augmented Tool Grounding](http://arxiv.org/abs/2605.13228)

- RETOOL-VIDEO: introduces a recursive tool-using framework that grounds high-level video intents into executable tool chains by delegating abstract actions to a resolver, utilizing a Planner, Resolver, MVTL, Base Tools, Meta Tools, Execution Engine, and Observation Buffer.
- The framework employs a MetaAug-Video Tool Library (MVTL) containing 134 registered tools, including 26 base tools for multimodal signal processing and 108 meta tools for filtering, aggregation, and intermediate-result operations.
- RETOOL-VIDEO optimizes the planner policy using reinforcement learning to improve action selection, evidence sufficiency judgment, and termination in complex video understanding tasks.

---

[An Agentic AI Framework with Large Language Models and Chain-of-Thought for UAV-Assisted Logistics Scheduling with Mobile Edge Computing](http://arxiv.org/abs/2605.13221)

- Agentic AI Framework: introduces an agentic AI-assisted optimization framework that integrates LLMs, RAG, and CoT reasoning to translate user requirements into interpretable mathematical models for hybrid logistics and computational scheduling.
- The framework employs a hierarchical DRL approach with upper-layer PPO for UAV routing and lower-layer PPO for task execution and resource allocation to solve complex combinatorial problems in cloud manufacturing.
- The system utilizes a two-agent workflow consisting of a Responder and a Verifier to ensure semantic fidelity and logical consistency in the generated optimization formulations.

---

[GAGPO: Generalized Advantage Grouped Policy Optimization](http://arxiv.org/abs/2605.13217)

- GAGPO: introduces a critic-free reinforcement learning method for multi-turn LLM agents that enables precise, step-aligned temporal credit assignment through Rollout Grouping, Step-level Credit Assignment, and Group-normalized PPO Update.
- The framework utilizes a Non-parametric Value Proxy and a TD/GAE-style Temporal Estimator to propagate outcome supervision backward through time without requiring a learned critic.
- By employing a Sequence-level Importance Ratio and group-wise normalization, GAGPO achieves stable, localized optimization signals that outperform existing RL baselines on multi-turn agent benchmarks.

---

[Hierarchical Attacks for Multi-Modal Multi-Agent Reasoning](http://arxiv.org/abs/2605.13213)

- HAM3: introduces a hierarchical adversarial framework that decomposes attacks into perception, communication, and reasoning layers to evaluate the robustness of multi-modal multi-agent systems.
- The framework models how localized perturbations propagate through agent collaboration, specifically targeting visual-textual inputs, communication topology, and internal reasoning chains.
- Experimental results demonstrate that reasoning-layer attacks, such as Chain-of-Thought Injection, cause the most severe performance degradation, while systemic errors dominate across all attack layers.

---

[FIKA-BENCH: From Fine-grained Recognition to Fine-Grained Knowledge Acquisition](http://arxiv.org/abs/2605.13193)

- FIKA-BENCH: introduces a leakage-aware, evidence-grounded benchmark for evaluating the ability of LMMs and agents to perform active fine-grained knowledge acquisition.
- The framework evaluates systems through a pipeline of model-hard filtering, leakage inspection, and human-verified evidence grounding to ensure models move beyond parametric memorization.
- Empirical results demonstrate that current LMMs and agents struggle with fine-grained recognition, with performance limited by incorrect entity retrieval and visual grounding errors rather than tool availability.

---

[Decoupled Planning for Multiple Omega-Regular Objectives](http://arxiv.org/abs/2605.13185)

- Decoupled Planning Framework: introduces a modular approach for satisfying multiple omega-regular objectives by assigning each to an independent agent and using a scheduler to compose their local policies.
- The framework utilizes stochastic schedulers and specific conventions to ensure that independently designed policies satisfy all objectives almost surely without requiring direct communication.
- The approach supports modular design, robustness, and iterative development by allowing agents to operate independently while maintaining correctness through minimal runtime coordination or pre-agreed conventions.

---

[When Does Hierarchy Help? Benchmarking Agent Coordination in Event-Driven Industrial Scheduling](http://arxiv.org/abs/2605.13172)

- DESBench: introduces a benchmark for evaluating agent coordination in hierarchical, event-driven industrial scheduling environments using Shared World State, Event Engine, Event Interpreter, Decision Layer, and Runtime Interface.
- The framework evaluates four coordination paradigms—centralized, hierarchical, heterarchical, and holonic—by measuring effectiveness, constraint alignment, coordination efficiency, and robustness.
- It utilizes LLMs as decision-making agents within a unified simulation environment to analyze trade-offs in information flow, decision authority, and conflict resolution.

---

[Finding the Weakest Link: Adversarial Attack against Multi-Agent Communications](http://arxiv.org/abs/2605.13170)

- SVCP-APOSG (Single-Victim Communication Perturbation Adversarial Partially Observable Stochastic Game): introduces a framework for single-victim communication perturbation attacks that identifies vulnerable messages, agents, and timesteps using a Jacobian-proxy, weighted-loss, and maximum-loss.
- The framework utilizes a Jacobian-based saliency method to rank messages and select victims, while employing novel loss functions to enhance the effectiveness of gradient-based perturbations against MARL systems.
- Empirical results demonstrate that the proposed methods achieve significant impact across various multi-agent environments, outperforming random message selection in most tested scenarios.

---

[GeoBuildBench: A Benchmark for Interactive and Executable Geometry Construction from Natural Language](http://arxiv.org/abs/2605.13167)

- GeoBuildBench: introduces an interactive benchmark for evaluating LLMs and MLLMs on grounded, executable plane geometry construction from natural language.
- The framework utilizes an agent-environment loop where the LLM or MLLM Agent iteratively generates programs in a Geometry Construction DSL, which are then processed by a Python Geometry Kernel and evaluated by a Verification Module.
- The benchmark assesses model performance through metrics including success rate, structural hallucination frequency, and feedback-driven error recovery capabilities.

---

[Collaborating in Multi-Armed Bandits with Strategic Agents](http://arxiv.org/abs/2605.13145)

- CAOS (Collaborating Agents with Optimistic Stopping): introduces a mechanism that sustains collaborative exploration among strategic agents in multi-armed bandit problems using information sharing as the sole incentive.
- The framework utilizes an OER (Optimistic Expected Reward) procedure to dynamically determine a set of Active Agents who continue to follow a target algorithm, while non-compliant agents revert to a single-agent Algorithm B.
- By enforcing a structured Communication Protocol that verifies actions and shares rewards, the mechanism ensures that collaborative behavior constitutes a Nash equilibrium and achieves strong regret guarantees.

---

[SWE-Cycle: Benchmarking Code Agents across the Complete Issue Resolution Cycle](http://arxiv.org/abs/2605.13139)

- SWE-Cycle: introduces a comprehensive benchmark for evaluating autonomous code agents across the complete issue resolution lifecycle, including environment reconstruction, code implementation, and verification test generation.
- The framework utilizes SWE-Judge, which integrates static code review and dynamic execution to provide robust, fine-grained assessment of agent performance while overcoming the limitations of traditional script-based evaluation.
- By evaluating agents in both isolated and end-to-end FullCycle settings, the research exposes critical bottlenecks in cross-phase dependencies and demonstrates that end-to-end integration often improves dynamic correctness at the cost of structural quality.

---

[ERPPO: Entropy Regularization-based Proximal Policy Optimization](http://arxiv.org/abs/2605.13131)

- ERPPO: introduces a multi-agent reinforcement learning framework that integrates a Distributional Spatiotemporal Ambiguity (DSA) learner and entropy-based policy regularization to enhance object detection in dynamic maritime environments.
- The framework utilizes a DSA learner to compute confidence fields and an entropy-regularized PPO algorithm to dynamically adjust policy updates based on observed environmental ambiguity.
- By applying L1 regularization in high-ambiguity scenarios and L2 regularization in low-ambiguity states, the approach improves search stability and reduces false detection rates in time-critical UAV operations.

---

[Towards Long-horizon Embodied Agents with Tool-Aligned Vision-Language-Action Models](http://arxiv.org/abs/2605.13119)

- VLAs-as-Tools: introduces a framework that distributes long-horizon task burdens by utilizing a high-level VLM agent for planning and a family of specialized VLA tools for bounded physical execution.
- The system employs a VLA tool-family interface to facilitate bidirectional communication, where the agent sends invocation messages and receives progress feedback from the VLA tools.
- Tool-Aligned Post-Training (TAPT) is utilized to align VLA models with specific subtask invocations, employing residual adapters to maintain shared semantic representations while enabling specialized tool behavior.

---

[A Multi-Agent Orchestration Framework for Venture Capital Due Diligence](http://arxiv.org/abs/2605.13110)

- Multi-Agent Orchestration Framework for Venture Capital Due Diligence: introduces an event-driven automation pipeline that utilizes specialized AI agents to synthesize unstructured market data and official financial filings into structured investment reports.
- The system integrates a programmatic extraction pipeline to reverse-engineer Greek Business Registry endpoints, ensuring auditable data retrieval while employing a structural fallback mechanism to mitigate LLM hallucinations.
- By leveraging a low-code DAG-structured architecture, the framework automates end-to-end corporate research, providing traceable provenance for financial metrics and strategic recommendations.

---

[Counterfactual Reasoning for Causal Responsibility Attribution in Probabilistic Multi-Agent Systems](http://arxiv.org/abs/2605.13077)

- PATL-SR: introduces a formal framework for quantifying backward counterfactual responsibility in probabilistic multi-agent systems using the Shapley value.
- The framework utilizes CSG and PSMAS to model agent interactions and compute stable strategy profiles where agents balance expected rewards against responsibility penalties.
- The research demonstrates that model checking and Nash equilibrium computation within this logic remain in PSPACE, ensuring computational feasibility for responsibility-aware strategic reasoning.

---

[PBT-Bench: Benchmarking AI Agents on Property-Based Testing](http://arxiv.org/abs/2605.15229)

- PBT-Bench: introduces a benchmark of 100 curated problems across 40 Python libraries to evaluate the ability of LLMs to perform property-based testing by deriving invariants from documentation and constructing precise input-generation strategies.
- The framework utilizes an automated, containerized F→P harness that evaluates LLM-generated tests against 365 human-verified semantic bugs, requiring agents to identify invariants rather than just concrete test cases.
- Evaluation across eight contemporary LLMs reveals that property-based testing scaffolding significantly improves performance for mid-capability models, while highlighting persistent gaps in cross-function protocol reasoning for all models.

---

[Verifiable Agentic Infrastructure: Proof-Derived Authorization for Sovereign AI Systems](http://arxiv.org/abs/2605.15228)

- DTF (Distributed Trust Framework): introduces a verification layer for governed mutation systems that computes execution authority from structured, verifiable artifacts rather than relying on standing identity.
- The framework enforces a compact authorization invariant where high-stakes execution requires a Justification Proof, consensus-gated approval, and an append-only Evidence Chain.
- By shifting authorization from static roles to proof-derived authority, DTF enables governable, auditable, and replayable execution for autonomous AI agents in sovereign deployments.

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


