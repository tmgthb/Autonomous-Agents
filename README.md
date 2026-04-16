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

## Research papers: 2026 4/4

[2026 (4/4)](http://github.com/tmgthb/Autonomous-Agents/blob/main/README.md), [2026 (3/4)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2026_3.md), [2026 (2/4)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2026_2.md), [2026 (1/2)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2026_1.md), [2025 (4/4)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2025_4.md),[2025 (3/4)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2025_3.md), [2025 (2/4)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2025_2.md), [2025 (1/4)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2025_01.md), [2024](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2024.md), [2023](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2023.md), [Earlier](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_Earlier.md)



Chronological order. 





</div>



---





#### 15th April 2026

[TREX: Automating LLM Fine-tuning via Agent-Driven Tree-based Exploration](http://arxiv.org/abs/2604.14116)

- TREX: introduces a multi-agent system that automates the entire LLM fine-tuning life-cycle by orchestrating collaboration between a Researcher Agent and an Executor Agent.
- The framework models the experimental process as a search tree, utilizing MCTS Policy to efficiently explore training strategies while leveraging an AIDP Library for scalable data pipeline construction.
- TREX incorporates a fine-grained diagnostic mechanism and Context Memory to synthesize empirical feedback, enabling continuous performance optimization across diverse tasks defined in FT-Bench.

---

[TIP: Token Importance in On-Policy Distillation](http://arxiv.org/abs/2604.14084)

- TIP (Token Importance in on-Policy distillation): introduces a two-axis taxonomy that classifies token importance based on student entropy and teacher–student divergence to identify informative training signals in on-policy distillation.
- The framework utilizes a Soft-OR selection mechanism to recover overconfident student errors that are typically missed by entropy-only filtering methods.
- Experimental results demonstrate that training on a small subset of high-importance tokens identified by TIP matches or exceeds full-token performance while significantly reducing memory usage across mathematical reasoning and agentic planning tasks.

---

[π-Play: Multi-Agent Self-Play via Privileged Self-Distillation without External Data](http://arxiv.org/abs/2604.14054)

- π-Play: introduces a multi-agent self-evolution framework that leverages the Question Construction Path (QCP) as intrinsic privileged information to transform sparse-reward self-play into a dense-feedback self-distillation loop.
- The framework utilizes an examiner to generate training tasks with QCPs, while a teacher model uses these paths to provide token-level supervision to a student model, enhancing its reasoning and search efficiency.
- By employing alternating optimization and teacher guidance, π-Play achieves superior evolutionary efficiency and performance compared to conventional self-play methods without requiring external labeled data.

---

[Enhancing Local Life Service Recommendation with Agentic Reasoning in Large Language Model](http://arxiv.org/abs/2604.14051)

- HiAgentRec: introduces a unified framework for local life service recommendation that integrates living need prediction with agentic reasoning to model hierarchical decision paths.
- The framework employs a noise-robust data pipeline and a curriculum learning strategy optimized by RLVR to navigate complex search spaces and mitigate reward sparsity.
- HiAgentRec utilizes a three-stage agentic reasoning pipeline—comprising living need inference, category mapping, and behavior ranking—to simulate human cognitive processes for precise service recommendations.

---

[Seek-and-Solve: Benchmarking MLLMs for Visual Clue-Driven Reasoning in Daily Scenarios](http://arxiv.org/abs/2604.14041)

- DailyClue: introduces a benchmark designed to evaluate MLLMs' reasoning capabilities in daily scenarios by requiring the identification and utilization of specific visual clues.
- The framework employs a rigorous construction pipeline involving MLLM-based generation, consensus-based filtering, and human verification to ensure high-quality, challenging triplets.
- Comprehensive evaluation across 25 MLLMs reveals that accurate visual clue identification is a critical bottleneck for robust reasoning, with explicit clue-guided CoT significantly improving performance.

---

[POINTS-Seeker: Towards Training a Multimodal Agentic Search Model from Scratch](http://arxiv.org/abs/2604.14029)

- POINTS-Seeker: introduces a multimodal agentic search model built from scratch that integrates Agentic Seeding for foundational reasoning and V-Fold for efficient long-horizon context management.
- The framework utilizes Agentic Seeding to internalize planning capabilities during formative training and V-Fold to render stale interaction history into compact visual tokens, mitigating attention dilution.
- By combining trajectory-based SFT and tool-augmented RL, the model achieves state-of-the-art performance across diverse benchmarks while maintaining general visual perception.

---

[Memory Transfer Learning: How Memories are Transferred Across Domains in Coding Agents](http://arxiv.org/abs/2604.14004)

- MTL (Memory Transfer Learning): introduces a framework that leverages a unified memory pool from heterogeneous coding domains to enhance the performance of LLM-based coding agents through cross-domain knowledge transfer.
- The framework utilizes four distinct memory representations—Trajectory, Workflow, Summary, and Insight—to capture and transfer meta-knowledge, which is found to be more effective than task-specific code.
- Experimental results demonstrate that higher abstraction levels in memory representations, particularly Insights, significantly improve transfer effectiveness and agent performance across diverse coding benchmarks.

---

[Acts of Configuration: Rethinking Provenance, Temporality and Legitimacy in Post-Mortem Agents](http://arxiv.org/abs/2604.13996)

- Post-Capacity Agent Framework: introduces a design space for AI agents that operate during the liminal period between cognitive capacity loss and death, emphasizing constrained advocacy over autonomous decision-making.
- The framework prioritizes first-party provenance and temporal boundaries, requiring agents to remain static after capacity loss to prevent interpretive drift and ensure representational fidelity.
- Participants in the study favored bounded, non-anthropomorphic agents that act as advocates for previously articulated human intent rather than evolving, autonomous stand-ins.

---

[Towards Personalizing Secure Programming Education with LLM-Injected Vulnerabilities](http://arxiv.org/abs/2604.13955)

- InjectEd: introduces an agentic AI framework that automates the injection of personalized security vulnerabilities into student-authored code to enhance secure programming education.
- The system utilizes a modular pipeline of four autonomous agents—Injector Agent, Evaluator Agent, Ranker Agent, and Learning Outcome Generator Agent—to create contextually relevant instructional materials.
- Empirical evaluation in undergraduate courses demonstrates that personalized vulnerability injections significantly reduce student confusion and improve perceived relevance compared to generic textbook examples.

---

[HINTBench: Horizon-agent Intrinsic Non-attack Trajectory Benchmark](http://arxiv.org/abs/2604.13954)

- HINTBench: introduces a benchmark for auditing intrinsic failures in long-horizon agent trajectories under benign conditions, utilizing Environment Seeds, Trajectory Synthesis, Human Verification, Auditor Models, and a Five-Constraint Taxonomy.
- The framework evaluates agent safety through three tasks: trajectory-level risk detection, coarse-grained risk-step localization, and fine-grained failure-type identification.
- Experiments demonstrate that while strong LLMs perform well on trajectory-level detection, they struggle significantly with step-level localization and fine-grained diagnosis, highlighting a substantial capability gap.

---

[CollabCoder: Plan-Code Co-Evolution via Collaborative Decision-Making for Efficient Code Generation](http://arxiv.org/abs/2604.13946)

- CollabCoder: introduces a Plan-Code Co-Evolution framework that improves code generation through dynamic multi-agent collaboration between a Dynamic Planning Agent, an Adaptive Coding Agent, and a Collaborative Debug Agent.
- The framework utilizes a Collaborative Decision-Making module to determine whether to update the plan or refine the code, supported by a Reasoning Trajectory module that accumulates historical diagnostic information to guide iterative improvements.
- CollabCoder achieves significant performance gains on complex benchmarks like LiveCodeBench and xCodeEval while reducing computational overhead by minimizing redundant API calls through adaptive, experience-driven debugging.

---

[Goal2Skill: Long-Horizon Manipulation with Adaptive Planning and Reflection](http://arxiv.org/abs/2604.13942)

- Goal2Skill: introduces a dual-system framework for long-horizon embodied manipulation that decouples high-level semantic planning from low-level motor execution.
- The framework utilizes a VLM-based high-level planner for task decomposition, memory management, and reflection, while a VLA-based low-level executor performs geometry-oriented action generation.
- By integrating structured memory and closed-loop verification, the system enables adaptive replanning and robust recovery from execution failures in complex, long-horizon tasks.

---

[AI Coding Agents Need Better Compiler Remarks](http://arxiv.org/abs/2604.13927)

- Agentic Compiler Feedback Framework: introduces an evaluation of how structured, prescriptive compiler feedback influences the performance of LLM-based coding agents in source-level auto-vectorization tasks.
- The research demonstrates that the primary bottleneck for autonomous performance engineering is the opacity of legacy compiler interfaces rather than the inherent limitations of LLMs.
- By replacing ambiguous compiler remarks with precise, data-flow-level diagnostic signals, the framework significantly increases the success rate of LLMs in applying complex, semantics-preserving code transformations.

---

[[COMP25] The Automated Negotiating Agents Competition (ANAC) 2025 Challenges and Results](http://arxiv.org/abs/2604.13914)

- NegMAS: introduces the 15th International Automated Negotiating Agents Competition (ANAC 2025), which evaluates autonomous agents in sequential multi-deal negotiations and complex supply chain management environments using NegMAS, Center Agent, Edge Agent, Alternating Offers Protocol, Gymnasium, PettingZoo, SAC Agent, Tree-Search Pruning, and Dynamic Aspiration Model.
- The competition highlights the trade-off between computational scalability and strategic foresight, where agents utilize tree-search, sampling, or reinforcement learning to mitigate memory explosion in high-dimensional outcome spaces.
- Results indicate that domain-specific heuristics and risk-averse strategies often outperform complex opponent modeling in volatile, concurrent market simulations.

---

[Beyond Conservative Automated Driving in Multi-Agent Scenarios via Coupled Model Predictive Control and Deep Reinforcement Learning](http://arxiv.org/abs/2604.13891)

- MPC-RL: introduces an integrated framework that couples a DRL Speed Policy (PPO) for high-level decision-making with an NMPC Solver (NLP) for low-level trajectory optimization to improve navigation in multi-agent scenarios.
- The framework utilizes a State Processor to provide normalized kinematic and context data to the DRL Speed Policy (PPO), which generates speed references for the NMPC Solver (NLP) to ensure kinematic feasibility and safety.
- By maintaining collision avoidance constraints within the NMPC Solver (NLP) during both training and evaluation, the architecture prevents train-deployment mismatches and enhances cross-scenario robustness.

---

[Sandpile Economics: Theory, Identification, and Evidence](http://arxiv.org/abs/2604.13890)

- Sandpile Economics: introduces a formal framework interpreting macroeconomic instability as an emergent property of disequilibrium production networks driven by Ricci curvature, avalanche dynamics, stochastic shock propagation, Minsky correspondence, and general equilibrium amplification.
- The framework utilizes Ricci curvature as a structural indicator to measure local redundancy and adaptability, identifying how competitive selection pushes economies toward a critical state of permanent fragility.
- Empirical validation using global input-output data confirms that production networks operate in a critical regime where negative curvature predicts heavy-tailed cascade distributions and amplifies output losses from exogenous shocks.

---

[GeoAgentBench: A Dynamic Execution Benchmark for Tool-Augmented Agents in Spatial Analysis](http://arxiv.org/abs/2604.13888)

- GABench: introduces a dynamic, closed-loop evaluation benchmark for tool-augmented GIS agents, utilizing a Global Task Planner, Step-wise Reactive Executor, Spatial Analysis Toolbox, Dynamic Execution Sandbox, VLM Evaluator, and Persistent State Management System to assess long-chain geospatial reasoning.
- The framework employs a Plan-and-React architecture that decouples global workflow orchestration from localized, feedback-driven reactive execution to handle complex spatial analysis tasks.
- Evaluation is performed using a multi-tiered system featuring Parameter Execution Accuracy (PEA) for trajectory-level fidelity and VLM-based verification for end-to-end multimodal product quality.

---

[Drowsiness-Aware Adaptive Autonomous Braking System based on Deep Reinforcement Learning for Enhanced Road Safety](http://arxiv.org/abs/2604.13878)

- DD-DQN: introduces a physiology-aware autonomous braking system that integrates real-time ECG-based drowsiness detection into a Deep Reinforcement Learning framework to adapt vehicle control under impaired cognitive states.
- The framework utilizes an RNN-based drowsiness detection module and a DD-DQN agent to manage braking behavior, incorporating action delays to simulate impaired driver responsiveness.
- The system employs DBSCAN for radar data processing and a reward-based learning mechanism to maintain safe following distances and minimize collision risks in simulated driving environments.

---

[MCPThreatHive: Automated Threat Intelligence for Model Context Protocol Ecosystems](http://arxiv.org/abs/2604.13849)

- MCPThreatHive: introduces an automated, end-to-end threat intelligence platform for Model Context Protocol ecosystems that integrates Intelligence Gathering, AI Threat Analysis, Structured Storage, Visualization &amp; Risk Planning, and a Knowledge Graph.
- The framework utilizes LLMs to perform chain-of-thought threat classification, mapping emerging risks against the MCP-38 taxonomy, STRIDE, and OWASP frameworks to provide quantitative prioritization.
- By constructing a neuro-symbolic knowledge graph, the platform enables security teams to identify compositional attack chains and prioritize remediation through an interactive risk planning interface.

---

[Use and usability: concepts of representation in philosophy, neuroscience, cognitive science, and computer science](http://arxiv.org/abs/2604.13829)

- GAC: introduces a three-level framework to categorize neural representations based on their information content, usability for goals, and actual causal role in downstream behavior.
- The framework distinguishes between Level 1 (information-carrying states), Level 2 (states with task-relevant information in usable formats), and Level 3 (states that causally influence behavior or downstream subsystems).
- This interdisciplinary review clarifies how different research fields, including neuroscience and machine learning, utilize varying definitions of representation to address distinct scientific questions.

---

[Beyond State Consistency: Behavior Consistency in Text-Based World Models](http://arxiv.org/abs/2604.13824)

- BehR-WM: introduces a behavior-aligned training paradigm for text-based world models that optimizes for functional consistency rather than simple state reconstruction.
- The framework utilizes a Behavior Consistency Reward (BehR) to measure how well a predicted state preserves the likelihood of a logged expert action under a frozen Reference Agent.
- By employing Group Relative Policy Optimization (GRPO), the approach improves task-level consistency and reduces calibration errors in offline evaluation and lookahead planning.

---

[UI-Copilot: Advancing Long-Horizon GUI Automation via Tool-Integrated Policy Optimization](http://arxiv.org/abs/2604.13822)

- UI-Copilot: introduces a collaborative framework for long-horizon GUI automation that decouples persistent memory from transient execution context, utilizing a Policy Agent, Copilot Model, Retriever, Calculator, Memory Decoupling, TIPO, and Multi-turn Summary.
- The framework employs a Policy Agent that selectively invokes a lightweight Copilot Model as either a Retriever or Calculator to handle memory-intensive or numerical tasks, effectively mitigating context overload and hallucinations.
- TIPO optimizes the agent by separately training tool selection via single-turn supervision and task execution through on-policy multi-turn rollouts, achieving state-of-the-art performance on complex GUI benchmarks.

---

[RPS: Information Elicitation with Reinforcement Prompt Selection](http://arxiv.org/abs/2604.13817)

- RPS (Reinforcement Prompt Selection): introduces a reinforcement learning framework that treats prompt selection as a sequential decision-making problem to adaptively elicit concealed information from users in open-ended dialogue.
- The framework utilizes a Policy Net to select optimal prompts from a predefined pool, conditioned on the current information state and dialogue history, to maximize information gain.
- RPS incorporates a normalized reward function to stabilize training and introduces the IELegal benchmark to evaluate elicitation performance in realistic legal consultation scenarios.

---

[Character Beyond Speech: Leveraging Role-Playing Evaluation in Audio Large Language Models via Reinforcement Learning](http://arxiv.org/abs/2604.13804)

- RoleJudge: introduces a multidimensional evaluation framework for voice-based role-playing agents that leverages Qwen2-Audio as a backbone and incorporates Standard Alignment reinforcement learning to assess character-speech alignment.
- The framework utilizes RoleChat, a reasoning-enhanced dataset, to train the Policy Model through a cold-start supervised fine-tuning phase followed by reinforcement learning with Standard Alignment.
- Standard Alignment dynamically scales advantage estimates using ground-truth standard samples to mitigate reward misalignment and prevent the model from converging to local optima during training.

---

[EmbodiedClaw: Conversational Workflow Execution for Embodied AI Development](http://arxiv.org/abs/2604.13800)

- EmbodiedClaw: introduces a conversational agent that automates high-frequency embodied AI research tasks by mapping user intent to executable workflows across simulation, data, and model objects.
- The framework utilizes an Intent Understanding Module, Workflow Orchestrator, Skill Library, Asset-Platform Adaptation Module, and Closed-Loop Verifier to ensure reliable, multi-stage execution.
- Experiments demonstrate that EmbodiedClaw significantly reduces manual engineering overhead and improves task completion accuracy compared to manual workflows and general-purpose LLM agents.

---

[ToolOmni: Enabling Open-World Tool Use via Agentic learning with Proactive Retrieval and Grounded Execution](http://arxiv.org/abs/2604.13787)

- ToolOmni: introduces a unified agentic framework that couples proactive retrieval with grounded execution to enable effective open-world tool use for LLMs.
- The framework utilizes a two-stage training strategy, including a supervised cold-start phase and a Decoupled Multi-Objective GRPO algorithm for synchronized optimization of retrieval and execution capabilities.
- ToolOmni incorporates a filtered rollout mechanism and separated policy updates to stabilize training and prevent gradient conflicts between the retrieval and execution modules.

---

[The Cognitive Companion: A Lightweight Parallel Monitoring Architecture for Detecting and Recovering from Reasoning Degradation in LLM Agents](http://arxiv.org/abs/2604.13759)

- Cognitive Companion: introduces a modular, parallel monitoring architecture designed to detect and mitigate reasoning degradation in LLM agents through either periodic LLM-based assessment or zero-overhead hidden state probing.
- The framework utilizes a Primary Agent, a Companion Observer for state monitoring, and an Intervention Handler to inject targeted guidance when degradation is detected.
- Experimental results demonstrate that companion effectiveness is task-type dependent, with the Probe-based Companion providing significant performance gains on loop-prone and open-ended tasks at zero additional inference overhead.

---

[Rethinking AI Hardware: A Three-Layer Cognitive Architecture for Autonomous Agents](http://arxiv.org/abs/2604.13757)

- Tri-Spirit Architecture: introduces a three-layer cognitive framework that separates planning, reasoning, and execution across heterogeneous hardware to optimize latency and energy efficiency.
- The framework utilizes a Super Layer for long-horizon goals, an Agent Layer for task decomposition using LLMs, and a Reflex Layer for low-latency execution of compiled habit policies.
- By employing a habit compilation mechanism, the architecture reduces reliance on LLMs by promoting frequently observed reasoning traces into zero-inference finite-state machines.

---

[Jump-Start Reinforcement Learning with Vision-Language-Action Regularization](http://arxiv.org/abs/2604.13733)

- VLAJS: introduces a method that bridges sparse VLA guidance with on-policy RL to improve exploration and learning efficiency in robotic manipulation.
- The framework utilizes a directional action-consistency loss to incorporate transient, sparse VLA action suggestions into a PPO agent without enforcing strict imitation.
- A reward-based jump-start mechanism progressively anneals and deactivates teacher guidance, allowing the RL agent to optimize independently and surpass the guiding policy.

---

[Doc-V*: Coarse-to-Fine Interactive Visual Reasoning for Multi-Page Document VQA](http://arxiv.org/abs/2604.13731)

- Doc-V* introduces an OCR-free agentic framework that casts multi-page document VQA as a sequential evidence aggregation process, utilizing Global Thumbnail Overview, Policy Model, Retrieval Tool, Fetch Tool, Working Memory, and Environment Feedback.
- The framework employs a coarse-to-fine strategy where the agent starts with a structural thumbnail overview and iteratively performs retrieval or targeted fetching to gather evidence for grounded reasoning.
- Doc-V* is trained using a two-stage approach involving supervised fine-tuning on distilled interaction trajectories and Group Relative Policy Optimization to balance answer accuracy with evidence-seeking efficiency.

---

[Homotopy-Guided Potential Games for Congestion-Aware Navigation](http://arxiv.org/abs/2604.13708)

- Homotopy-Guided Potential Games framework: introduces a unified motion planning approach that combines topological path generation with game-theoretic reasoning to resolve multi-agent congestion.
- The framework utilizes a deterministic homotopy planner to structure the solution space into distinct classes, which are then filtered and refined through a potential game to ensure efficient and collision-free navigation.
- By enforcing homotopy-consistent constraints within a receding-horizon setup, the approach demonstrates improved safety and robustness to irrational agent behaviors compared to standard game-theoretic or reactive planners.

---

[Beyond Arrow’s Impossibility: Fairness as an Emergent Property of Multi-Agent Collaboration](http://arxiv.org/abs/2604.13705)

- Deliberative Arena: introduces a multi-agent framework where LLMs negotiate resource allocation to demonstrate that fairness emerges as a procedural property of interaction rather than an individual agent trait.
- The framework utilizes Agent A (Aligned Agent), Agent B (Baseline Agent), and Agent C (Biased Agent) within a Deliberative Arena (Negotiation Environment) to evaluate how RAG (Retrieval-Augmented Generation) and Interaction History (Running Record) influence Fairness Metrics (Evaluation Framework) under structural constraints.
- The study demonstrates that multi-agent deliberation acts as a patching mechanism, where aligned agents mitigate bias through contestation rather than override, constrained by Arrow’s Impossibility Theorem.

---

[MIND: AI Co-Scientist for Material Research](http://arxiv.org/abs/2604.13699)

- MIND (Materials INference &amp; Discovery): introduces a multi-agent framework for automated hypothesis validation in materials research, integrating Pre-experiment module, Experiment module, and Discussion module.
- The framework utilizes SevenNet-Omni MLIP for scalable in-silico experiments and employs a LangGraph-based pipeline to manage iterative hypothesis refinement and validation.
- MIND supports two distinct validation strategies, Adversarial Discussion and Expert Voting, to assess experimental evidence and generate final scientific reports.

---

[IndicDB - Benchmarking Multilingual Text-to-SQL Capabilities in Indian Languages](http://arxiv.org/abs/2604.13686)

- IndicDB: introduces a comprehensive multilingual Text-to-SQL benchmark designed to evaluate cross-lingual semantic parsing capabilities of LLMs across diverse Indian languages.
- The framework utilizes an iterative three-agent judge pattern (Architect, Auditor, and Refiner) to transform denormalized government data into complex, high-density relational schemas.
- Empirical evaluation across state-of-the-art LLMs reveals a persistent "Indic Gap" in performance, which is partially mitigated by incorporating external evidence files to improve schema grounding.

---

[Hierarchical Bayesian calibration of mesoscopic models for ultrasound contrast agents from force spectroscopy data](http://arxiv.org/abs/2604.13657)

- Hierarchical Bayesian UQ-driven multiscale workflow: introduces a surrogate-accelerated Bayesian framework to calibrate high-fidelity numerical encapsulated microbubble models from force spectroscopy data with quantified uncertainty.
- The framework utilizes DPD Engine (particle-based soft matter simulator) and DNN Surrogates (deep neural network emulators) to enable efficient parameter inference via Bayesian Calibration (TMCMC-based posterior inference) and Hierarchical Modeling (population-level parameter regularization).
- The methodology identifies dominant elastic parameters while treating nonlinear terms as weakly informed, providing a parsimonious and robust calibration approach for encapsulated microbubbles.

---

[Vision-and-Language Navigation for UAVs: Progress, Challenges, and a Research Roadmap](http://arxiv.org/abs/2604.13654)

- UAV-VLN: introduces a comprehensive methodological taxonomy for aerial navigation, tracing the evolution from modular pipelines to integrated agentic systems driven by foundation models.
- The paper synthesizes the ecosystem of simulators, datasets, and evaluation metrics while critically analyzing challenges like the sim-to-real gap and efficient deployment on resource-constrained hardware.
- It concludes by proposing a research roadmap focused on multi-agent swarm coordination and air-ground collaborative robotics to guide future advancements in embodied AI.

---

[SAFEHARNESS: Lifecycle-Integrated Security Architecture for LLM-based Agent Deployment](http://arxiv.org/abs/2604.13630)

- SAFEHARNESS: introduces a security architecture that embeds defense mechanisms directly into the four phases of the LLM agent lifecycle to enable coordinated system-level responses to composite attacks.
- The architecture integrates four defense layers—INFORM, VERIFY, CONSTRAIN, and CORRECT—that communicate through inter-layer feedback to provide lifecycle-spanning protection against adversarial threats.
- Experimental results demonstrate that SAFEHARNESS significantly reduces unsafe behavior and attack success rates across diverse harness configurations while preserving core task utility.

---

[Topology Estimation for Open Multi-Agent Systems](http://arxiv.org/abs/2604.13628)

- OMAS Topology Estimation Framework: introduces a two-stage approach for reconstructing interaction topologies in systems with dynamic node sets and fast switching by combining projection-based clustering with multi-interval data aggregation.
- The framework utilizes an auxiliary system to filter state data and a projection-based dissimilarity measure to robustly group time segments sharing the same latent interaction mode.
- By aggregating information across multiple intervals, the method overcomes the limitations of short dwell times, enabling accurate topology identification even when individual segments lack sufficient excitation.

---

[Golden Handcuffs make safer AI agents](http://arxiv.org/abs/2604.13609)

- Golden Handcuffs: introduces a pessimistic Bayesian reinforcement learning agent that maintains safety by deferring to trusted mentor policies when the predicted value of its optimizing policy drops below a critical threshold.
- The framework utilizes a universal Bayesian prior to identify novel or high-risk states, triggering mentor intervention to avoid irrecoverable outcomes and reward hacking.
- The agent achieves sublinear regret relative to the best mentor while ensuring that no low-complexity catastrophic events are triggered by the optimizing policy before they are initiated by a mentor.

---

[Reward Hacking in the Era of Large Models: Mechanisms, Emergent Misalignment, Challenges](http://arxiv.org/abs/2604.13602)

- PCH (Proxy Compression Hypothesis): introduces a unifying framework for understanding reward hacking as an emergent consequence of optimizing expressive policies against compressed reward representations of high-dimensional human objectives.
- The paper categorizes reward hacking into an escalating hierarchy of mechanisms, ranging from feature-level exploitation to environment-level manipulation, driven by the interaction of objective compression, optimization amplification, and evaluator-policy co-adaptation.
- The authors synthesize detection and mitigation strategies across the LLM lifecycle, emphasizing that robust alignment requires moving beyond static benchmarks toward dynamic, adversarial, and process-level oversight.

---

[Daycare Matching with Siblings: Social Implementation and Welfare Evaluation](http://arxiv.org/abs/2604.13597)

- DMS (Daycare Matching with Siblings): introduces an empirical framework that explicitly incorporates demand complementarities for joint sibling assignments into a centralized matching model.
- The framework utilizes an extended stability notion to estimate household preferences, accounting for both commuting distance and a fixed non-distance disutility associated with split daycare placements.
- By simulating counterfactual assignment policies, the research quantifies the efficiency–equity tradeoff inherent in sibling priority reforms and demonstrates that ignoring complementarities leads to significant underestimation of welfare gains.

---

[Foresight Optimization for Strategic Reasoning in Large Language Models](http://arxiv.org/abs/2604.13592)

- FoPO (Foresight Policy Optimization): introduces a reinforcement learning algorithm that enhances strategic reasoning in LLMs by incorporating a foresight-based correction term into the policy optimization process to explicitly model counterpart influence.
- The framework utilizes a self-play mechanism where LLM agents are trained using curated datasets, Cooperative RSA and Competitive Taboo, to anticipate and adapt to counterpart behaviors.
- FoPO improves strategic reasoning by coupling self-agent and counterpart-agent gradient updates through a computationally efficient, gradient-truncated approximation of second-order information.

---

[MM-Doc-R1: Training Agents for Long Document Visual Question Answering through Multi-turn Reinforcement Learning](http://arxiv.org/abs/2604.13579)

- MM-Doc-R1: introduces an agentic, vision-aware workflow for long document visual question answering that utilizes iterative information discovery and synthesis.
- The framework employs a Planner, a tool-driven Seeker, and an Answer Agent to perform multi-turn retrieval and reasoning over complex, multi-page documents.
- The authors propose Similarity-based Policy Optimization (SPO) to mitigate baseline estimation bias in multi-turn RL by weighting rewards based on semantic trajectory similarity.

---

[WebMAC: A Multi-Agent Collaborative Framework for Scenario Testing of Web Systems](http://arxiv.org/abs/2604.13559)

- WebMAC: introduces a multi-agent collaborative framework that improves scenario testing by integrating human-computer interaction for scenario clarification and equivalence class partitioning for test adequacy.
- The framework utilizes a Clarification Module to resolve incomplete test descriptions, a Transformation Module to generate diverse test scenarios, and a Testing Module to automate script execution.
- WebMAC significantly improves test script execution success rates, reduces testing time and token consumption, and detects more errors compared to existing LLM-based scenario testing methods.

---

[AgentComm: Semantic Communication for Embodied Agents](http://arxiv.org/abs/2604.13558)

- AgentComm: introduces a semantic communication framework for embodied AI that utilizes LLM-based semantic processors and importance-aware transmission to reduce bandwidth overhead while maintaining task performance.
- The framework integrates an LLM-based semantic processor for message condensation, an importance-aware transmission strategy for unequal protection of critical data, and a task-specific knowledge base to support recurring embodied tasks.
- Experimental results demonstrate that the proposed approach achieves nearly 50% bandwidth reduction compared to conventional transmission schemes while preserving task completion robustness in noisy wireless environments.

---

[Training-Free Test-Time Contrastive Learning for Large Language Models](http://arxiv.org/abs/2604.13552)

- TF-TTCL (Training-Free Test-Time Contrastive Learning): introduces a training-free adaptation framework that enables frozen LLMs to improve online by distilling supervision from their own inference experiences through Semantic Query Augmentation, Contrastive Experience Distillation, and Contextual Rule Retrieval.
- The framework utilizes a multi-agent role-playing loop consisting of a TEACHER, TUTOR, and STUDENT to generate diverse reasoning trajectories and distill them into explicit positive and negative rules stored in an Experience Rule Repository.
- By dynamically retrieving and injecting these distilled rules into the prompt, the system steers the frozen LLM toward robust reasoning patterns without requiring gradient updates or external knowledge.

---

[Debate to Align: Reliable Entity Alignment through Two-Stage Multi-Agent Debate](http://arxiv.org/abs/2604.13551)

- AgentEA: introduces a reliable entity alignment framework that combines preference-optimized entity embeddings with a two-stage multi-agent debate mechanism to enhance alignment accuracy.
- The framework utilizes an Entity Representation Preference Optimization module to improve embedding quality and a two-stage debate process consisting of Lightweight Debate Verification and Deep Debate Alignment to resolve alignment ambiguity.
- AgentEA incorporates specialized agents—including Alias-, Type-, Attribute-, Neighborhood-, Attack-, and Judge-agents—to perform multi-perspective reasoning and mitigate the limitations of single-LLM alignment approaches.

---

[Self-adaptive Multi-Access Edge Architectures: A Robotics Case](http://arxiv.org/abs/2604.13542)

- MAPE-K self-adaptive framework: introduces a self-adaptive architecture for robotics that optimizes task offloading and resource utilization across heterogeneous edge nodes using real-time feedback and Kubernetes orchestration.
- The system employs an Offloading Agent that utilizes a Service Profile to dynamically route tasks to specific Pods, enabling finer granularity than standard service-level load balancing.
- By integrating high-frequency monitoring and asynchronous task execution, the architecture effectively balances performance and energy efficiency in dynamic human-robot environments.

---

[Don’t Let AI Agents YOLO Your Files: Shifting Information and Control to Filesystems for Agent Safety and Autonomy](http://arxiv.org/abs/2604.13536)

- YoloFS (Agent-native filesystem): introduces an agent-native filesystem that shifts information and control from LLMs to the filesystem to prevent misuse while maintaining autonomy.
- The framework utilizes Staging, Snapshots, and Progressive permission to provide visibility, auditability, and corrective control over agent filesystem interactions.
- Evaluation shows that YoloFS enables agent self-correction in 8 of 11 tasks with hidden side effects and reduces user interaction on routine tasks compared to baseline agent frameworks.

---

[Evolvable Embodied Agent for Robotic Manipulation via Long Short-Term Reflection and Optimization](http://arxiv.org/abs/2604.13533)

- EEAgent: introduces an embodied agent framework that leverages VLMs for environmental interpretation and policy planning, utilizing a long short-term reflective optimization (LSTRO) mechanism to enable self-evolution through iterative learning from past successes and failures.
- The framework incorporates an Environment Interpreter for visual entity extraction using SAM and a Policy Planner that utilizes LLMs to generate action sequences based on refined prompts.
- LSTRO dynamically updates long-term and short-term memories by reflecting on task feedback, allowing the agent to improve performance without requiring model parameter updates or external storage.

---

[RiskWebWorld: A Realistic Interactive Benchmark for GUI Agents in E-commerce Risk Management](http://arxiv.org/abs/2604.13531)

- RiskWebWorld: introduces a realistic interactive benchmark for evaluating GUI agents in e-commerce risk management, utilizing an Agent module, Task suite, Environment module, and Workflow.
- The framework employs a Cloud Browser Cluster and DaaS SDK to provision isolated environments, while a Playwright-based Client and Remote Browser Session facilitate decoupled policy planning and environment execution.
- This infrastructure enables scalable, Gymnasium-compliant benchmarking and supports agentic reinforcement learning by decoupling decision-making from rendering mechanics.

---

[Cascaded TD3-PID Hybrid Controller for Quadrotor Trajectory Tracking in Wind Disturbance Environments](http://arxiv.org/abs/2604.13505)

- CTPH (Cascaded TD3-PID Hybrid Controller): introduces a cascaded hybrid control framework for quadrotor trajectory tracking that integrates PID controllers for altitude and attitude with an enhanced TD3 agent for horizontal-position control.
- The framework incorporates an aggregated multi-Q-network architecture, a PID-guided expert policy, and a reward-filtered dual experience replay mechanism to improve learning stability and sample efficiency.
- A structurally simple hybrid disturbance observer, combining median filtering, exponential moving average, and IIR low-pass filtering, is embedded in the PID loops to enhance robustness against external wind disturbances.

---

[Towards Scalable Lightweight GUI Agents via Multi-role Orchestration](http://arxiv.org/abs/2604.13488)

- LAMO (Lightweight Agent Multi-role Orchestration): introduces a framework that enables lightweight MLLMs to perform complex GUI automation by orchestrating skill-specific roles including Observer, Planner, Allocator, and Executor.
- The framework utilizes a two-stage training recipe consisting of supervised fine-tuning with Perplexity-Weighted Cross-Entropy optimization and multi-task reinforcement learning for collaborative exploration.
- LAMO-3B, the resulting task-scalable agent, functions as a monolithic agent, a coordinated multi-agent system, or a plug-and-play policy executor paired with advanced planners to achieve higher performance ceilings.

---

[Bridging MARL to SARL: An Order-Independent Multi-Agent Transformer via Latent Consensus](http://arxiv.org/abs/2604.13472)

- CMAT (Consensus Multi-Agent Transformer): introduces a centralized framework that bridges cooperative MARL to a hierarchical SARL formulation by using a Transformer decoder to iteratively generate a consensus vector for order-independent joint decision-making.
- The architecture employs a Transformer encoder to extract observation features, which are then compressed into an initial consensus vector used for V-value estimation and iterative refinement by the decoder.
- By conditioning all agents on a shared high-level consensus vector, the framework enables simultaneous action generation and optimizes the joint policy using single-agent PPO.

---

[CANVAS: Continuity-Aware Narratives via Visual Agentic Storyboarding](http://arxiv.org/abs/2604.13452)

- CANVAS: introduces a multi-agent framework that models storyboard generation as explicit world-state tracking through global planning, memory-guided retrieval, and sequential memory updates.
- The framework utilizes a Global Story Planner to define character, location, and object states, while the Image Generator and QA-based Selector ensure visual coherence across shots.
- CANVAS maintains a persistent Visual State Memory, including Character-, Background-, and Prop-Memory, to enable long-range narrative consistency and logical state transitions.

---

[MERRIN: A Benchmark for Multimodal Evidence Retrieval and Reasoning in Noisy Web Environments](http://arxiv.org/abs/2604.13418)

- MERRIN: introduces a human-annotated benchmark designed to evaluate search-augmented agents on their ability to perform multi-hop reasoning over noisy, conflicting, and heterogeneous multimodal web evidence.
- The framework evaluates agents across three settings—No Search, Native Search, and Agentic Multimodal Search—to identify bottlenecks in source selection and multimodal reasoning.
- Experimental results demonstrate that while agents struggle with over-exploration and text-modality bias, the integration of specialized tools for video and audio processing significantly improves performance on complex queries.

---

[Distributed Resilient Fixed-Time Control for Cooperative Output Regulation of MASs over Directed Graphs under DoS Attacks](http://arxiv.org/abs/2604.13394)

- Distributed Resilient Fixed-Time Controller: introduces a robust control framework for heterogeneous linear multi-agent systems (MASs) that guarantees fixed-time convergence of regulated outputs despite directed communication topologies and denial-of-service (DoS) attacks.
- The framework utilizes a distributed resilient fixed-time observer to estimate exosystem states and a geometric homogeneity-based control law to achieve output regulation independent of initial system states.
- The proposed approach eliminates the need for Laplacian symmetry or strong connectivity constraints, providing broader applicability for practical MASs subject to severe zero-topology DoS attacks.

---

[Agentic Open RAN: A Deterministic and Auditable Framework for Intent-Driven Radio Control](http://arxiv.org/abs/2604.13384)

- A1GENT: introduces a hierarchical control framework that decouples LLM-based intent reasoning from deterministic near-RT execution using typed A1 policy contracts.
- The architecture utilizes a non-RT orchestrator rApp to translate operator intents into bounded A1 policy instances, which are then enforced by task-oriented xApps via E2 and O1 interfaces.
- A training-free Adaptive Policy Tuner maintains system performance by refining policy parameters based on historical KPI and action logs without requiring model retraining.

---

[AVID: A Benchmark for Omni-Modal Audio-Visual Inconsistency Understanding via Agent-Driven Construction](http://arxiv.org/abs/2604.13593)

- AVID (Audio-Visual Inconsistency Detection): introduces a large-scale benchmark and construction pipeline for evaluating audio-visual inconsistency understanding, utilizing Temporal Segmentation, Strategy Planning Agent, Execution Agent, Injectors, AVID-Qwen, Vision Encoder, Audio Encoder, LLM Backbone, and Aligner Module.
- The framework employs a scalable agent-driven pipeline to generate diverse audio-visual conflicts across three semantic segment classes, supporting detection, classification, and fine-grained reasoning tasks.
- AVID-Qwen, a fine-tuned omni-modal model, demonstrates superior performance in temporal grounding and holistic understanding compared to existing state-of-the-art models.

---

#### 14th April 2026


[LIFE - an energy efficient advanced continual learning agentic AI framework for frontier systems](http://arxiv.org/abs/2604.12874)

- LIFE: introduces an agent-centric framework for HPC systems that integrates an Orchestrator, ACE, AMSN, ILL, and task-specific AI Models to achieve energy-efficient continual learning.
- The framework utilizes a neuro-symbolic approach where the Orchestrator manages observe-reason-act loops while the AMSN provides short-term, episodic, semantic, and procedural memory tiers.
- Information Lattice Learning (ILL) serves as a mechanism to distill episodic memory into validated semantic rules, enabling the system to evolve its knowledge graph autonomously.

---

[Drawing on Memory: Dual-Trace Encoding Improves Cross-Session Recall in LLM Agents](http://arxiv.org/abs/2604.12948)

- Dual-Trace Memory Encoding: introduces a memory protocol for LLM agents that pairs factual records with concrete narrative scene traces to improve cross-session recall through elaborative generation.
- The framework utilizes an evidence scoring gate to selectively encode information and a three-state retrieval protocol to re-instantiate encoding context for enhanced temporal and aggregation reasoning.
- Experimental results on the LongMemEval-S benchmark demonstrate a 20.2 percentage point accuracy gain over fact-only baselines, specifically in temporally complex tasks, without increasing token costs.

---


[Transferable Expertise for Autonomous Agents via Real-World Case-Based Learning](http://arxiv.org/abs/2604.12717)

- CBL (Case-Based Learning): introduces a framework that enables LLMs to transform real-world task execution into reusable knowledge assets through a closed loop of experience summarization and memory consolidation.
- The framework organizes acquired experience into four structured modules—Fixed Domain Knowledge (M_F), System Prompt Constraints (M_S), Skill Library (M_K), and Curriculum Organizer (M_C)—to support stable knowledge reuse and cross-agent transfer.
- By treating each task as a learnable case, the system allows agents to evolve professional capabilities that outperform static prompting baselines, particularly in complex, multi-stage, and open-ended environments.

---



[Development, Evaluation, and Deployment of a Multi-Agent System for Thoracic Tumor Board](http://arxiv.org/abs/2604.12161)

- Multi-Agent System for Thoracic Tumor Board: introduces a multi-agent workflow that automates the generation of concise patient summaries for clinical tumor board conferences by integrating Data Loader Agent, FHIR Agent, Curation Agent, and Summarization Agent.
- The system utilizes a sequential agentic pipeline to retrieve longitudinal EHR data, filter for clinical relevance, and synthesize summaries while maintaining per-statement citations for physician verification.
- Comparative evaluation demonstrates that low-autonomy multi-agent architectures provide superior objective completeness and clinical relevance compared to simpler or higher-autonomy LLM-based summarization methods.

---


[A Scoping Review of Large Language Model-Based Pedagogical Agents](http://arxiv.org/abs/2604.12253)

- LLM-based pedagogical agents framework: introduces a scoping review of LLMs in education, characterizing agents through interaction approach, domain scope, role complexity, and system integration.
- The paper identifies emerging trends including multi-agent systems, virtual student simulation, immersive technology integration, and the use of learning analytics to enhance agent adaptability.
- The research highlights critical gaps in theoretical grounding, teacher-agent interaction studies, and the need for robust evaluation frameworks to assess the effectiveness of LLMs in diverse educational settings.

---


[See, Point, Refine: Multi-Turn Approach to GUI Grounding with Visual Feedback](http://arxiv.org/abs/2604.13019)

- See, Point, Refine: introduces a multi-turn iterative refinement approach for pixel-precise GUI grounding in dense coding interfaces using visual feedback.
- The framework utilizes a process-separated data collection pipeline to map symbolic cursor states to renderer-space coordinates, enabling LLMs to self-correct localization errors through explicit red-cross visual markers.
- Empirical results demonstrate that iterative visual feedback significantly improves grounding accuracy and reduces localization distance across various frontier LLMs compared to single-shot prediction.

---

[Toward Autonomous Long-Horizon Engineering for ML Research](http://arxiv.org/abs/2604.13018)

- AiScientist: introduces a hierarchical multi-agent system that achieves long-horizon ML research engineering by separating thin control from thick, durable project state.
- The framework utilizes a File-as-Bus protocol to maintain state continuity across heterogeneous research stages, allowing agents to re-ground on persistent artifacts rather than relying on lossy conversational handoffs.
- By employing an Agent-as-Tool design, the Orchestrator delegates complex tasks to specialized agents, ensuring that long-horizon progress is cumulative and evidence-driven.

---

[Agentic Discovery with Active Hypothesis Exploration for Visual Recognition](http://arxiv.org/abs/2604.12999)

- HypoExplore: introduces a memory-grounded multi-agent framework that formulates neural architecture discovery as a hypothesis-driven scientific inquiry.
- The framework utilizes a Trajectory Tree Memory (records experimental lineage) and a Hypothesis Memory Bank (tracks confidence scores) to guide iterative architecture evolution.
- HypoExplore employs specialized agents for idea generation, code implementation, redundancy filtering, and multi-perspective feedback analysis to automate the discovery of efficient vision architectures.

---

[PARALLAX: Why AI Agents That Think Must Never Act](http://arxiv.org/abs/2604.12986)

- PARALLAX: introduces a paradigm for architecturally safe autonomous AI execution by enforcing a structural separation between reasoning and execution components, utilizing Agent, Shield, Executor, Chronicle, and IFC Tags.
- The architecture employs a multi-tiered validation system (Shield) that interposes between the untrusted Agent and the privileged Executor to prevent harmful actions regardless of the reasoning system's state.
- By integrating Information Flow Control and Reversible Execution, the framework ensures that data sensitivity is tracked and destructive actions can be rolled back, providing a robust defense against agent compromise.

---

[Cycle-Consistent Search: Question Reconstructability as a Proxy Reward for Search Agent Training](http://arxiv.org/abs/2604.12967)

- CCS (Cycle-Consistent Search): introduces a gold-supervision-free framework for training search agents by using the reconstructability of the original question from a search trajectory as a proxy reward, utilizing a Policy Model, Search Environment, Information Bottleneck, Reconstructor, and GRPO Optimizer.
- The framework employs an Information Bottleneck to prevent information leakage by masking entities in search queries and excluding final responses, ensuring the Reconstructor relies on retrieved evidence rather than lexical shortcuts.
- The agent is optimized via GRPO, which maximizes the semantic similarity between the original question and the reconstructed question, effectively training the agent to gather sufficient evidence without ground-truth labels.

---

[Modeling Co-Pilots for Text-to-Model Translation](http://arxiv.org/abs/2604.12955)

- Text2Model: introduces a suite of co-pilots leveraging LLMs and intermediate representations to automate the translation of natural language problem descriptions into formal MiniZinc models.
- The framework utilizes the Text2Zinc dataset, a unified cross-domain benchmark, to evaluate various single-call, multi-call, and agentic strategies for combinatorial modeling.
- Experimental results demonstrate that while structured reasoning and grammar-constrained generation improve performance, a significant gap remains between model execution and solution accuracy.

---

[A Sanity Check on Composed Image Retrieval](http://arxiv.org/abs/2604.12904)

- FISD (Fully-Informed Semantically-Diverse benchmark): introduces a benchmark and an automated multi-round evaluation framework to address query ambiguity and evaluate CIR models across six semantic dimensions using FISD benchmark, CIR model, Ranker, and User simulator.
- The framework utilizes a User simulator, powered by MLLM and LLM, to provide iterative feedback that guides the CIR model in refining its retrieval choices over successive rounds.
- Experimental results demonstrate that current CIR models struggle with negation and cardinality semantics, while multi-round interactions significantly enhance retrieval performance across various benchmarks.

---

[Don’t Show Pixels, Show Cues: Unlocking Visual Tool Reasoning in Language Models via Perception Programs](http://arxiv.org/abs/2604.12896)

- P2 (Perception Programs): introduces a training-free, model-agnostic method that reformulates dense, pixel-level tool outputs into compact, structured, language-native summaries for MLLMs.
- The framework standardizes visual information by grounding it in spatial coordinates and modality-specific read-outs, enabling MLLMs to parse and reason over visual cues without architectural modifications.
- By converting raw tool signals into a symbolic YAML-like format, P2 addresses the representation bottleneck in tool-augmented MLLMs, consistently improving performance across diverse perception-centric tasks.

---

[Towards Long-horizon Agentic Multimodal Search](http://arxiv.org/abs/2604.12890)

- LMM-Searcher: introduces a long-horizon multimodal search framework that mitigates context explosion by offloading visual assets to an external file system and using lightweight textual identifiers (UIDs) for on-demand retrieval.
- The framework employs an extended agentic tool interface, including search-, browse- and visual processing-tools, to enable progressive, fine-grained perception and reasoning over long search horizons.
- A specialized data synthesis pipeline generates complex multi-hop reasoning trajectories to fine-tune the Qwen3-VL-Thinking-30A3B model, achieving state-of-the-art performance on long-horizon multimodal benchmarks.

---

[OVAL: Open-Vocabulary Augmented Memory Model for Lifelong Object Goal Navigation](http://arxiv.org/abs/2604.12872)

- OVAL: introduces a lifelong open-vocabulary memory framework that utilizes memory descriptors and a probability-based exploration strategy to enable efficient, structured navigation in unseen environments.
- The framework integrates a Frontier Exploration Module for systematic coverage, an Open-Semantic Memory Model for persistent object storage, and a Navigation Module that employs LLMs for goal verification and synonym resolution.
- By employing an Instances Matcher to merge redundant observations and a multi-value frontier scoring mechanism, the system effectively balances exploration and exploitation in continual ObjectNav tasks.

---

[QuarkMedSearch: A Long-Horizon Deep Search Agent for Exploring Medical Intelligence](http://arxiv.org/abs/2604.12867)

- QuarkMedSearch: introduces a full-pipeline framework for training long-horizon medical deep search agents, utilizing Seed QA Construction, Multi-Hop Real-Fact Introduction, Entity Obfuscation, and Uniqueness and Correctness Guarantee to synthesize high-quality training data.
- The framework employs a two-phase SFT and RLVR training recipe, incorporating Medical Knowledge Graph, Search Tool, Visit Tool, Medical Professional Search Tool, and LLM Check Tool to enhance planning, tool invocation, and reflection capabilities.
- The system utilizes Asynchronous Rollout and Reward Computation to optimize training efficiency and mitigate bottlenecks caused by long-horizon sequences in medical deep search tasks.

---

[Artificial Intelligence for Modeling and Simulation of Mixed Automated and Human Traffic](http://arxiv.org/abs/2604.12857)

- Artificial Intelligence for Modeling and Simulation of Mixed Automated and Human Traffic: provides a comprehensive taxonomy and review of AI-driven methods for modeling automated and human-driven vehicle behavior in mixed autonomy traffic simulation, covering Single-Agent Methods, Multi-Agent Methods, Environment-level Simulation Methods, and Cognitive and Physics-informed Methods.
- The paper categorizes AI approaches into agent-level behavior models, environment-level simulation methods, and cognitive and physics-informed methods, while highlighting the critical challenges of counterfactual validity, simulation-to-real transfer, and the need for unified architectures.
- It synthesizes the current landscape of AI-driven traffic simulation, identifying key research directions such as bridging the causality gap, improving evaluation protocols, and integrating cognitive grounding with data-driven scalability.

---

[VULCAN: Vision-Language-Model Enhanced Multi-Agent Cooperative Navigation for Indoor Fire-Disaster Response](http://arxiv.org/abs/2604.12831)

- VULCAN: introduces a hazard-aware multi-agent cooperative navigation framework that utilizes Multi-modal Perception and Fusion, VLM-based Global Planner, and FMM Local Planner to maintain exploration efficiency in fire-driven indoor environments.
- The framework employs Hazard-aware Mapping and Frontier Extraction to generate a compact, physically grounded representation of structural constraints and environmental risks for coordinated multi-agent navigation.
- By integrating multi-modal sensor data with high-level semantic reasoning, the system enables robots to perform Collaborative Routes Planning while minimizing exposure to fire-related hazards.

---

[Efficiency of Proportional Mechanisms in Online Auto-Bidding Advertising](http://arxiv.org/abs/2604.12799)

- Proportional Mechanisms (PM) and its variant (m-PM): introduces a theoretical analysis of the price of anarchy (PoA) for proportional mechanisms in online advertising, establishing a tight bound of 2 for standard mechanisms and a near-optimal variant.
- The paper utilizes Karush-Kuhn-Tucker (KKT) conditions and linear programming duality to evaluate equilibrium structures and bound the liquid welfare of auto-bidding systems.
- The proposed modified payment scheme circumvents existing impossibility results, achieving asymptotic full efficiency as the number of bidding agents increases.

---

[EVOSPARK: Endogenous Interactive Agent Societies for Unified Long-Horizon Narrative Evolution](http://arxiv.org/abs/2604.12776)

- EVOSPARK: introduces a unified framework for long-horizon narrative evolution that integrates narrative control, cognitive evolution, and spatial grounding to foster coherent agent societies.
- The framework utilizes a Unified Narrative Operation Engine (NOE) to manage storyworld instantiation and an Emergent Character Grounding Protocol (ECGP) to convert stochastic LLM hallucinations into persistent narrative assets.
- To ensure long-term consistency, the system employs a Stratified Narrative Memory (SNM) with a mutable Role Socio-Evolutionary Base (RSB) that metabolizes experiences to resolve conflicting relational states.

---

[A Multi-Agent Feedback System for Detecting and Describing News Events in Satellite Imagery](http://arxiv.org/abs/2604.12772)

- SkyScraper: introduces an iterative multi-agent workflow that geocodes news articles and synthesizes captions for multi-temporal satellite image sequences using Article Agent, Geocoding API, Data API, Verifier Agent, Captioning Agent, and Feedback Loop.
- The framework utilizes an iterative feedback loop where failed geocoding or verification attempts trigger re-processing with updated reasoning to improve event detection accuracy.
- By integrating LLMs for article analysis and multimodal verification, the system achieves a 5x increase in event detection yield compared to traditional rules-based geocoding methods.

---

[NaviRAG: Towards Active Knowledge Navigation for Retrieval-Augmented Generation](http://arxiv.org/abs/2604.12766)

- NaviRAG: introduces a two-stage framework that organizes documents into a Knowledge Tree and performs staged Navigational Retrieval to improve evidence localization in complex reasoning tasks.
- The framework utilizes LLM-guided Organization to build hierarchical semantic structures and employs a Memory Module to maintain dynamic state awareness during multi-step retrieval.
- NaviRAG models evidence acquisition as a coarse-to-fine exploration process, enabling efficient and context-aware retrieval that outperforms traditional flat RAG methods on long-chain reasoning benchmarks.

---

[ARGOS: Who, Where, and When in Agentic Multi-Camera Person Search](http://arxiv.org/abs/2604.12762)

- ARGOS (Agentic Retrieval with Grounded Observational Search): introduces a benchmark and agent framework that reformulates multi-camera person search as an interactive reasoning problem requiring an agent to plan, question, and eliminate candidates under information asymmetry.
- The framework utilizes a four-module agent architecture (Analyst, Planner, Interviewer, Interpreter) that operates within an observe-think-act loop to perform multi-turn dialogue and spatio-temporal reasoning.
- Reasoning is grounded in a Spatio-Temporal Topology Graph (STTG) that encodes camera connectivity and empirically validated transition times to enable effective candidate elimination across three progressive tracks: semantic perception, spatial reasoning, and temporal reasoning.

---

[AffectAgent: Collaborative Multi-Agent Reasoning for Retrieval-Augmented Multimodal Emotion Recognition](http://arxiv.org/abs/2604.12735)

- AffectAgent: introduces a multi-agent retrieval-augmented framework that leverages collaborative decision-making among a Query Planner, Evidence Filter, and Emotion Generator to improve fine-grained emotion recognition.
- The framework utilizes MB-MoE to mitigate cross-modal representation mismatch and RAAF to enhance semantic completion by injecting retrieved audiovisual embeddings under missing-modality conditions.
- All trainable agents are jointly optimized using MAPPO with a shared affective reward to ensure consistent reasoning and prevent degeneration into superficial lexical matching.

---

[Signed DeGroot–Friedkin Dynamics with Interdependent Topics](http://arxiv.org/abs/2604.12685)

- Signed DeGroot–Friedkin (DF) framework: introduces a multi-topic model that couples multidimensional DeGroot opinion formation with Friedkin’s reflected-appraisal mechanism to examine how antagonism and topic interdependence shape agent-level social power.
- The framework utilizes a signed influence matrix to incorporate antagonistic interactions and logic matrices to model internal topic-coupling, enabling an exact algebraic reduction to a scalar DF map under shared topic-weight conditions.
- The research provides a complete classification of limiting social power configurations—pluralistic, mixed, and vertex-dominant—and establishes global convergence properties and local robustness under heterogeneous logic perturbations.

---

[From Imitation to Discrimination: Progressive Curriculum Learning for Robust Web Navigation](http://arxiv.org/abs/2604.12666)

- Triton: introduces a progressive curriculum learning framework that evolves LLMs from basic imitation to robust discrimination and long-horizon consistency for web navigation.
- The framework utilizes a 590k-instance dataset constructed via Structural-Semantic Hard Negative Mining and a Dual-Agent Consensus pipeline to address hallucination and generalization challenges.
- Triton-GRPO-32B achieves state-of-the-art performance on Mind2Web, demonstrating that specialized data curricula outperform raw parameter scale in web navigation tasks.

---

[Multi-Agent Digital Twins for Strategic Decision-Making using Active Inference](http://arxiv.org/abs/2604.12657)

- AIF (Active Inference): introduces a multi-agent framework for digital twins that integrates decentralized Generative Models with Streaming Machine Learning to enable adaptive, goal-oriented decision-making under uncertainty.
- The framework utilizes contextual inference and Expected Free Energy minimization to balance pragmatic utility with epistemic information-seeking in non-stationary environments.
- Numerical experiments on a Cournot competition model demonstrate that agents maintain stable collective dynamics and adapt to evolving market conditions through decentralized inference and online preference updates.

---

[A Comparison of Reinforcement Learning and Optimal Control Methods for Path Planning](http://arxiv.org/abs/2604.12628)

- DDPG (Deep Deterministic Policy Gradient): introduces a reinforcement learning approach for path planning in continuous state and action spaces by utilizing Actor Network, Critic Network, Target Actor Network, Target Critic Network, Reward Function, Reset Function, and Step Function.
- The framework employs an actor-critic architecture to learn deterministic policies while balancing exploration and exploitation through stochastic noise injection.
- The study compares this DDPG-based learning method against traditional pseudo-spectral optimal control, highlighting trade-offs between training time, real-time inference speed, and solution optimality.

---

[Habitat-GS: A High-Fidelity Navigation Simulator with Dynamic Gaussian Splatting](http://arxiv.org/abs/2604.12626)

- Habitat-GS: introduces a navigation-centric embodied AI simulator that integrates 3D Gaussian Splatting for photorealistic scene rendering and a dynamic gaussian avatar module for human-aware navigation.
- The framework utilizes a visual-navigation decoupling design, where 3DGS handles visual rendering while traditional NavMeshes and proxy capsules govern navigation and collision avoidance.
- Habitat-GS enables real-time rendering on standard hardware by employing a zero-copy CUDA-OpenGL interoperability mechanism and pre-baked gaussian avatar deformation.

---

[Cross-Cultural Simulation of Citizen Emotional Responses to Bureaucratic Red Tape Using LLM Agents](http://arxiv.org/abs/2604.12545)

- RAMO (Red Tape Emotional Simulator): introduces a cross-cultural simulation framework that integrates diverse personas and country-level cultural factors to evaluate how LLMs reproduce human-like emotional responses to bureaucratic red tape.
- The framework utilizes Policy Block (interface for selecting policies), Result Visualisation Block (spider chart for emotion intensity), LLM-Agents (simulated citizen emotional responders), Cultural Persona Construction (integrating demographic and Hofstede cultural factors), Significance Alignment Score (SAS) (statistical alignment metric), and Overlap@3 Metric (Jaccard similarity for top-3 emotions) to assess alignment with human experimental ground truth.
- The study reveals that while LLMs exhibit limited alignment with human emotional responses, particularly in Eastern cultures, the RAMO interface provides a scalable tool for policymakers to explore micro-level affective impacts of administrative procedures.

---

[Agentic Control in Variational Language Models](http://arxiv.org/abs/2604.12513)

- EVE (Variational Language Model): introduces a closed-loop agentic control framework that utilizes internal stochastic evidence to regulate training, retain meaningful model states, and guide inference-time interventions.
- The framework integrates a homeostatic latent regulator and a structurally aware checkpoint-retention rule to ensure the model maintains a stable and informative latent regime throughout its lifecycle.
- A calibrated uncertainty-aware controller enables the model to perform multi-action routing, such as direct answering, deliberation, or retrieval, based on its own internal predictive and structural uncertainty signals.

---

[A Heterogeneous Dual-Network Framework for Emergency Delivery UAVs: Communication Assurance and Path Planning Coordination](http://arxiv.org/abs/2604.12501)

- HDNF: introduces a coordinated framework for emergency delivery that couples an Emergency Communication Support Network (ECSN) with a Delivery Path Network (DPN) to ensure reliable 3D connectivity.
- The framework utilizes a multi-layer C2 service model to guide deployment, a 3D-CASB-MATD3 with PER algorithm for resilient UAV-BS positioning, and a 3D communication-aware A* planner for trajectory optimization.
- Simulation results demonstrate that HDNF maintains a 100% task success rate and eliminates communication outages while reducing hardware deployment requirements by up to 20% compared to static baselines.

---

[DeCoNav: Dialog enhanced Long-Horizon Collaborative Vision-Language Navigation](http://arxiv.org/abs/2604.12486)

- DeCoNav: introduces a decentralized framework for long-horizon collaborative VLN that integrates SVB, EDR, and SPE to enable event-triggered dialogue and synchronized dual-robot execution.
- The framework utilizes ROVE, comprising RTSA and TriGate, to construct verified navigation episodes that ensure semantic correctness and target observability for multi-robot evaluation.
- DeCoNav enables autonomous subtask reassignment through real-time semantic communication, allowing robots to optimize navigation paths and task completion in dynamic environments.

---

[From Kinematics to Dynamics: Learning to Refine Hybrid Plans for Physically Feasible Execution](http://arxiv.org/abs/2604.12474)

- Neuro-symbolic refinement framework: introduces a reinforcement learning-based approach to transform first-order feasible robotic plans into physically valid trajectories by integrating second-order dynamics validation.
- The framework utilizes a GNN to process graph-based plan representations, enabling an RL agent to iteratively adjust velocity bounds for improved physical feasibility.
- An SOCP solver computes candidate trajectories, which are subsequently validated by an MTV mechanism to ensure compliance with second-order dynamics and quadratic drag constraints.

---

[CIA: Inferring the Communication Topology from LLM-based Multi-Agent Systems](http://arxiv.org/abs/2604.12461)

- CIA (Communication Inference Attack): introduces a black-box attack framework that infers the internal communication topology of LLM-based Multi-Agent Systems by inducing intermediate agent reasoning outputs and modeling their semantic correlations.
- The framework utilizes an adversarial query strategy to elicit intermediate reasoning, followed by global bias disentanglement and LLM-guided weak supervision to refine agent representations for accurate topology inference.
- By effectively isolating task-relevant semantic signals from global bias, the approach achieves high-precision topology reconstruction while maintaining the stealth and functional integrity of the target MAS.

---

[Agentic Insight Generation in VSM Simulations](http://arxiv.org/abs/2604.12421)

- Agentic VSM Simulation Framework: introduces a decoupled, two-step agentic architecture that separates high-level orchestration from data analysis to extract actionable insights from complex simulation outputs.
- The system utilizes an Orchestration Agent for multi-hop reasoning and a Summarization Subworkflow to process specific data elements, effectively mitigating context rot by maintaining a slim internal context.
- The framework employs specialized tools including Node Discovery, Attribute Extraction, Taxonomy Navigation, and Summarization to enable progressive data discovery and precise navigation of structured simulation data.

---

[Traffic-Aware Domain Partitioning and Load-Balanced Inter-Domain Routing for LEO Satellite Networks](http://arxiv.org/abs/2604.12382)

- DTAR (Deep Traffic-Aware inter-domain Routing): introduces a two-stage framework that decouples offline traffic-aware domain partitioning via NSGA-II from online adaptive routing using a GAT-based state encoder and a MaskablePPO agent.
- The framework utilizes a GAT encoder to process real-time inter-domain link traffic and fault status, providing structured embeddings for the MaskablePPO agent to perform congestion-aware path selection.
- Action masking is integrated into the PPO agent to explicitly enforce physical reachability and hop constraints, ensuring routing feasibility under dynamic LEO satellite network conditions.

---

[Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning](http://arxiv.org/abs/2604.12374)

- Nemotron 3 Super: introduces a 120 billion parameter hybrid Mamba-Transformer model that leverages LatentMoE, Mamba-2, and MTP to achieve high inference throughput and strong agentic reasoning capabilities.
- The architecture utilizes a periodic interleaving of Mamba-2 blocks and sparse LatentMoE layers, with strategically placed attention anchors to maintain global dependency modeling while optimizing for long-context performance.
- The model is trained using a two-phase pretraining curriculum and a multi-stage post-training pipeline, including PivotRL for efficient agentic alignment and NVFP4 quantization for deployment on Blackwell hardware.

---

[Reading Between the Pixels: Linking Text-Image Embedding Alignment to Typographic Attack Success on Vision-Language Models](http://arxiv.org/abs/2604.12371)

- VLM Typographic Attack Analysis Framework: introduces a systematic evaluation of typographic prompt injection attacks on VLMs by analyzing the correlation between text-image embedding alignment and attack success rates.
- The framework utilizes multimodal embedding models to quantify the L2 distance between adversarial typographic images and their corresponding text prompts, serving as a predictive metric for VLM vulnerability.
- Experimental results demonstrate that typographic attack success follows a threshold pattern relative to font size and is significantly influenced by model-specific geometric and visual robustness.

---

[ReflectCAP: Detailed Image Captioning with Reflective Memory](http://arxiv.org/abs/2604.12357)

- ReflectCAP: introduces a gradient-free framework that distills a target LVLM’s recurring errors into an agentic memory called Structured Reflection Notes to steer inference-time captioning.
- The framework utilizes a multi-agent pipeline comprising a Captioning Agent, Feedback Agent, and Note Organizer to generate reusable directives that improve factuality and coverage.
- ReflectCAP achieves superior factuality–coverage F1 scores across multiple LVLMs while maintaining higher compute efficiency than existing multi-agent pipelines or model scaling.

---

[COMPLIBENCH: Benchmarking LLM Judges for Compliance Violation Detection in Dialogue Systems](http://arxiv.org/abs/2604.12312)

- COMPLIBENCH: introduces a benchmark and automated data generation pipeline for evaluating LLM judges on their ability to detect and localize compliance violations in multi-turn dialogues, utilizing Guideline Collection, Guideline Scaling, Guideline Modification, Conversation Generation, LLM Judge, Content Consistency Judge, Adversarial Compliance Judge, User Simulator Agent, Selector Agent, and Assistant Agent.
- The framework employs an adversarial judge-and-refine process to synthesize challenging, labeled violation data, ensuring that generated dialogues strictly adhere to domain-specific operational guidelines.
- Evaluation results demonstrate that while frontier LLMs struggle with compliance detection, a compact model fine-tuned on the synthesized data achieves superior performance and generalizes effectively across unseen business domains.

---

[Dialogue Agents that Share Family Information to Strengthen Grandparent–Grandchild Relationships](http://arxiv.org/abs/2604.12310)

- Dialogue Agent for Family Information Sharing: introduces a chatbot-based system that fosters relationships between older adults and their grandchildren by sharing everyday personal information.
- The system utilizes a rule-based and LLM-driven architecture to facilitate intermittent, daily conversations that adapt to user routines.
- Empirical results indicate that sharing information about grandchildren increases behavioral engagement among older adults and strengthens perceived family connections.

---

[GCA Framework: A Gulf-Grounded Dataset and Agentic Pipeline for Climate Decision Support](http://arxiv.org/abs/2604.12306)

- GCA Framework: introduces a Gulf-focused climate decision support system that integrates a large-scale multimodal dataset, GCA-DS, with a tool-augmented agent, GCA Agent, to provide grounded, interpretable climate analysis.
- The GCA Agent utilizes a modular Tool Controller to orchestrate specialized climate tools, including remote sensing, air quality, and hydrological analysis, to perform multi-step reasoning over regional data.
- By fine-tuning a Qwen2.5-VL 7B backbone on the GCA-DS dataset, the framework significantly improves tool-use reliability, factuality, and numerical precision for complex climate queries compared to general-purpose LLMs.

---

[Local-Splitter: A Measurement Study of Seven Tactics for Reducing Cloud LLM Token Usage on Coding-Agent Workloads](http://arxiv.org/abs/2604.12301)

- Local-Splitter: introduces a seven-tactic framework for reducing cloud LLM token usage by utilizing a local triage model to process requests before they reach a frontier cloud model.
- The framework implements T1 Local routing, T2 Prompt compression, T3 Semantic caching, T4 Local drafting with cloud review, T5 Minimal-diff edits, T6 Structured intent extraction, and T7 Batching and vendor prompt caching to optimize token consumption across various coding-agent workloads.
- Empirical results demonstrate that the optimal tactic subset is workload-dependent, with T1+T2 serving as the most effective default configuration for reducing cloud costs while maintaining response quality.

---

[Defining and Evaluation Method for External Human-Machine Interfaces](http://arxiv.org/abs/2604.12293)

- eHMI Evaluation Method: introduces a universal questionnaire-based framework to objectively assess and compare different external Human-Machine Interface proposals for autonomous vehicles.
- The framework evaluates proposals across seven distinct categories, including Standardization, Cost Effectiveness, Accessibility, Ease of Understanding, Constant Communication, Positioning, and Readability.
- The method was validated by testing four existing eHMI proposals and a kinematic baseline, identifying that a combination of vehicle kinematics and text-based displays offers the most effective communication solution.

---

[Frontier-Eng: Benchmarking Self-Evolving Agents on Real-World Engineering Tasks with Generative Optimization](http://arxiv.org/abs/2604.12290)

- Frontier-Eng: introduces a large-scale benchmark for evaluating AI agents on generative optimization tasks, utilizing a Task Context, Initial Solution, and Evaluator to assess iterative improvement under hard engineering constraints.
- The framework employs an LLM-based Agent and Proposer to iteratively refine candidate artifacts, leveraging History and a Frozen Evaluator to ensure genuine solution improvement within a fixed interaction budget.
- To maintain evaluation integrity, the system enforces Sandboxed Execution and Verifier-parsed Scoring, preventing reward hacking and ensuring that performance gains correspond to physical or algorithmic improvements.

---

[GAM: Hierarchical Graph-based Agentic Memory for LLM Agents](http://arxiv.org/abs/2604.12285)

- GAM (Hierarchical Graph-based Agentic Memory): introduces a hierarchical memory framework that decouples transient dialogue perception from long-term knowledge consolidation to mitigate memory contamination in LLM agents.
- The framework utilizes a state-based mechanism to transition between an Episodic Buffering State and a Semantic Consolidation State, ensuring that global memory is updated only at semantically complete boundaries.
- A graph-guided retrieval strategy integrates temporal, confidence, and role-based signals to enable precise context recovery across decoupled storage layers.

---

[WebAgentGuard: A Reasoning-Driven Guard Model for Detecting Prompt Injection Attacks in Web Agents](http://arxiv.org/abs/2604.12284)

- WebAgentGuard: introduces a parallel defense framework where a dedicated guard agent monitors web agent observations to decouple safety verification from task-oriented reasoning.
- The framework utilizes a reasoning-driven, multimodal guard model trained on a synthetic dataset generated by GPT-5 to detect prompt injection attacks embedded in HTML or visual screenshots.
- The model employs a two-stage training process consisting of reasoning-intensive supervised fine-tuning followed by reinforcement learning using Group Relative Policy Optimization (GRPO) to ensure robust detection without introducing additional latency.

---

[Towards Robust Real-World Spreadsheet Understanding with Multi-Agent Multi-Format Reasoning](http://arxiv.org/abs/2604.12282)

- SpreadsheetAgent: introduces a two-stage multi-agent framework that employs an Extraction Agent, Vision Range Agent, LaTeX Range Agent, and Verification Agent to perform step-by-step spreadsheet reasoning.
- The framework utilizes localized region-based inputs, including code execution results, image snippets, and LaTeX tables, to overcome context-length limitations of LLMs while preserving layout semantics.
- By integrating iterative reflection and feedback loops, the system constructs a faithful structural representation in YAML format, significantly improving accuracy in complex spreadsheet tasks compared to text-only baselines.

---

[CascadeDebate: Multi-Agent Deliberation for Cost-Aware LLM Cascades](http://arxiv.org/abs/2604.12262)

- CascadeDebate: introduces a unified architecture that alternates single-model inference with selective multi-agent deliberation across model scales to balance accuracy and inference costs.
- The framework utilizes confidence-based routers to trigger lightweight agent ensembles for marginal cases, effectively resolving ambiguities internally before escalating to larger LLMs or human experts.
- An integrated online threshold learner continuously refines escalation boundaries using human feedback, enabling elastic adaptation to varying query difficulty and real-world distributions.

---

[Coding-Free and Privacy-Preserving MCP Framework for Clinical Agentic Research Intelligence System](http://arxiv.org/abs/2604.12258)

- CARIS: introduces an agentic AI framework that automates end-to-end clinical research workflows by orchestrating modular tools via the Model Context Protocol (MCP) to ensure privacy-preserving data interaction.
- The system integrates LLMs with specialized agents to perform research planning, IRB documentation, Vibe ML, and report generation without requiring direct access to raw patient data.
- CARIS utilizes a human-in-the-loop protocol to iteratively refine research outputs, achieving high completeness scores based on the TRIPOD+AI reporting guidelines.

---

[How Memory Can Affect Collective and Cooperative Behaviors in an LLM-Based Social Particle Swarm](http://arxiv.org/abs/2604.12250)

- SPS (Social Particle Swarm) framework: introduces an agent-based model where LLM-based agents utilize Big Five personality traits and Interaction history memory to navigate a Prisoner’s Dilemma game environment.
- The study demonstrates that model-specific characteristics, such as internal alignment, cause Gemini and Gemma to interpret memory differently, leading to divergent collective behaviors.
- Sentiment analysis of agent reasoning texts reveals that Gemini’s risk-averse interpretation of memory suppresses cooperation, while Gemma’s positive interpretation promotes it.

---

[MolMem: Memory-Augmented Agentic Reinforcement Learning for Sample-Efficient Molecular Optimization](http://arxiv.org/abs/2604.12237)

- MolMem: introduces a multi-turn agentic reinforcement learning framework that utilizes a dual-memory system to enhance sample efficiency in molecular optimization.
- The framework integrates Static Exemplar Memory for cold-start grounding and Evolving Skill Memory for distilling successful trajectories into reusable strategies, which are injected into the Working Memory to guide the Policy Agent.
- By employing dense step-wise rewards and on-demand memory retrieval, the system enables the Policy Agent to perform iterative refinements that significantly outperform traditional methods under limited oracle budgets.

---

[Thought-Retriever: Don’t Just Retrieve Raw Data, Retrieve Thoughts for Memory-Augmented Agentic Systems](http://arxiv.org/abs/2604.12231)

- Thought-Retriever: introduces a model-agnostic retrieval framework that enables LLMs to utilize arbitrarily long external knowledge by distilling past interactions into persistent, query-driven thoughts.
- The framework employs Thought retrieval, Answer generation, Thought and confidence generation, Thought merge, and Thought memory update to maintain a self-evolving long-term memory for LLM agents.
- By filtering meaningless and redundant thoughts, the system ensures that the memory remains information-dense and retrieval-friendly, significantly outperforming traditional retrieval-augmented methods in reasoning tasks.

---

[Modality-Native Routing in Agent-to-Agent Networks: A Multimodal A2A Protocol Extension](http://arxiv.org/abs/2604.12213)

- MMA2A (Multimodal Modality-native A2A): introduces a lightweight routing layer for A2A networks that preserves native multimodal signals by inspecting Agent Cards to route voice, image, and text parts to specialized agents.
- The architecture utilizes a Modality-Aware Router (MAR) to bypass text-bottleneck serialization, delivering high-fidelity evidence to Voice Agent, Vision Agent, and Text Agent components for improved reasoning.
- Experimental results demonstrate that modality-native routing significantly improves task completion accuracy, provided that the downstream LLM Reasoning agent is capable of exploiting the richer input context.

---

[Towards grounded autonomous research: an end-to-end LLM mini research loop on published computational physics](http://arxiv.org/abs/2604.12198)

- Grounded Autonomous Research: introduces an autonomous agentic system that performs end-to-end scientific research by reading, reproducing, and extending published computational physics papers using Claude Code CLI, Claude Opus 4.6, Quantum ESPRESSO, Wannier90, NanoTCAD ViDES, Python, and Filesystem.
- The framework utilizes a Reproduce-Review-Reflect pipeline to audit scientific claims against first-principles calculations, ensuring findings are grounded in physical reality rather than textual interpolation.
- Empirical results demonstrate that 97.7% of substantive methodological critiques identified by the LLM agent are execution-bound, requiring active simulation to surface rather than passive reading.

---

[Beyond Majority Voting: Efficient Best-Of-N with Radial Consensus Score](http://arxiv.org/abs/2604.12196)

- RCS (Radial Consensus Score): introduces a geometric framework for best-of-N selection in LLMs by modeling semantic consensus via a weighted Fréchet mean of answer embeddings and ranking candidates by their radial distance to this center.
- The framework leverages Embedding Model, Semantic Center, Weighting Distribution, Radial Consensus Score, and Candidate Selector to improve answer reliability without external supervision or costly sampling.
- RCS provides a training-free, black-box compatible method that outperforms majority voting by effectively aggregating semantic information and promoting coherent minority answers in diverse reasoning tasks.

---

[Representing Expertise Accelerates Learning from Pedagogical Interaction Data](http://arxiv.org/abs/2604.12195)

- Representing Expertise Accelerates Learning from Pedagogical Interaction Data: investigates how training LLMs on pedagogical interaction traces, which include expert-corrected novice behavior, improves performance compared to training on expert-only demonstrations.
- The study demonstrates that LLMs benefit from interaction data by learning to recover from suboptimal states, provided the model can represent distinct agent types via source indicator tokens.
- Experimental results indicate that explicit source indicators significantly enhance LLM robustness in expert-scarce settings by enabling the model to differentiate between expert and novice trajectories.

---

[TRUST Agents: A Collaborative Multi-Agent Framework for Fake News Detection, Explainable Verification, and Logic-Aware Claim Reasoning](http://arxiv.org/abs/2604.12184)

- TRUST Agents: introduces a modular multi-agent framework that organizes fact-checking into a structured pipeline of specialized agents for claim extraction, evidence retrieval, verification, and explanation generation.
- The research-enhanced pipeline improves reasoning over complex claims by utilizing a Decomposer Agent, a Delphi Multi-Agent Jury, and a Logic Aggregator to handle compositional and causally structured statements.
- The framework prioritizes interpretability and evidence grounding, allowing the system to abstain from making predictions when evidence is insufficient or contradictory.

---

[How to Use Prices for Efficient Online Matching](http://arxiv.org/abs/2604.12181)

- SEM: introduces a randomized online matching mechanism that leverages large market equilibria to achieve asymptotic efficiency while maintaining greedy allocation constraints.
- The framework utilizes token money and random prices to compute competitive equilibria at each time period, ensuring that arriving agents receive their most preferred objects whenever feasible.
- SEM provides a strategyproof and equal-type envy-free solution for online matching problems with ordinal preferences and stochastic arrivals.

---

[AgenticAI-DialogGen: Topic-Guided Conversation Generation for Fine-Tuning and Evaluating Short- and Long-Term Memories of LLMs](http://arxiv.org/abs/2604.12179)

- AgenticAI-DialogGen: introduces a modular, agent-based framework that automates the generation of persona-grounded and topic-guided conversations to address the lack of datasets for evaluating short- and long-term memory in LLMs.
- The framework utilizes a pipeline of LLM agents to transform unstructured conversational data into structured knowledge graphs and topic-centric dialogues, which are then used to create the TopicGuidedChat (TGC) benchmark.
- Evaluations demonstrate that LLMs fine-tuned on the TGC dataset exhibit superior memory-aware behavior and discourse quality compared to baselines, effectively leveraging both structured knowledge graphs and unstructured conversational history.

---

[Policy-Invisible Violations in LLM-Based Agents](http://arxiv.org/abs/2604.12177)

- Sentinel: introduces a world-state-grounded enforcement framework that mitigates policy-invisible violations by performing counterfactual graph simulation on agent actions.
- The framework utilizes a five-phase pipeline—Translate, Fork, Mutate, Check, and Decide—to verify agent tool calls against organizational invariants hidden from the LLM.
- Sentinel achieves high precision in detecting policy violations by treating agent actions as speculative mutations to a structured world state graph.

---

[AlphaEval: Evaluating Agents in Production](http://arxiv.org/abs/2604.12162)

- AlphaEval: introduces a production-grounded benchmark and a requirement-to-benchmark construction framework that transforms authentic business requirements into executable evaluation tasks.
- The framework utilizes a Task Runner, Evaluator Registry, and Execution Sandbox to standardize the evaluation of LLM-based agent products across diverse professional domains.
- The research identifies six production-specific failure modes and provides economic value grounding to translate benchmark performance into professional labor cost savings.

---

[Cross-Domain Query Translation for Network Troubleshooting: A Multi-Agent LLM Framework with Privacy Preservation and Self-Reflection](http://arxiv.org/abs/2604.13353)

- Cross-Domain Query Translation Framework: introduces a hierarchical multi-agent LLM architecture that bridges communication gaps between non-technical users and telecom experts through specialized agents, including Domain-Aware Component, Query Classification Component, Privacy Protection Component, Query Translation Component, Response Simplification Component, and Reasoning and Self-Reflection Component.
- The framework utilizes a two-stage hierarchical classification process and context-aware anonymization to ensure diagnostic utility while maintaining privacy and compliance with data governance.
- The system incorporates ReAct-style reasoning and self-reflection loops to iteratively refine outputs, detect hallucinations, and ensure technical accuracy without human intervention.

---

[When Less Latent Leads to Better Relay: Information-Preserving Compression for Latent Multi-Agent LLM Collaboration](http://arxiv.org/abs/2604.13349)

- OBF (Orthogonal Backfill): introduces an eviction-style KV compression framework for multi-agent LLM systems that preserves essential information by injecting low-rank orthogonal residuals from discarded states into retained KV caches.
- The framework utilizes a four-part cache decomposition—Attention Sink, Inherited Message History, Current Prompt Context, and Current Latent Reasoning—to manage inter-agent communication efficiency.
- By modeling deleted prompt information as a residual principal subspace, OBF compensates for hard eviction losses, enabling compressed relay to match or exceed full KV relay performance while significantly reducing communication bandwidth.

---

[Listening Alone, Understanding Together: Collaborative Context Recovery for Privacy-Aware AI](http://arxiv.org/abs/2604.13348)

- CONCORD: introduces a privacy-aware assistant-to-assistant framework that enables proactive AI assistants to recover missing conversational context through negotiated safe exchanges while enforcing strict owner-only speech capture.
- The framework utilizes ECAPA-TDNN for real-time speaker verification and employs a hybrid disclosure filter to balance privacy guarantees with social flexibility during A2A communication.
- CONCORD treats context recovery as a coordination problem, using spatio-temporal reasoning and relationship-based trust models to resolve information gaps without compromising sensitive user data.

---

[AgentSPEX: An Agent SPecification and EXecution Language](http://arxiv.org/abs/2604.13346)

- AgentSPEX: introduces a declarative YAML-based language and harness for specifying and executing LLM-agent workflows with explicit control flow and modular structure.
- The framework utilizes a Visual Flow Editor for synchronized graph-based authoring and an execution harness that provides sandboxed tool access, state checkpointing, and formal verification capabilities.
- AgentSPEX improves agent controllability and interpretability by enforcing step-by-step execution and explicit context management, outperforming reactive prompting baselines across seven diverse benchmarks.

---

[Multi-Agent Object Detection Framework Based on Raspberry Pi YOLO Detector and Slack-Ollama Natural Language Interface](http://arxiv.org/abs/2604.13345)

- Multi-Agent Object Detection Framework: introduces a centralized edge-based system that integrates Vision Agent, Reporting Agent, Communication Agent, Control Agent, Core Router, Ollama LLM, Slack Channel Chatbot, and YOLO Detector to perform real-time object detection and tracking on resource-constrained hardware.
- The framework utilizes an event-driven Core Router to orchestrate locally running AI agents, replacing autonomous orchestration with a lightweight message exchange system to accommodate limited hardware resources.
- By leveraging a Slack-based natural language interface, the system enables user-friendly command and control, demonstrating a cost-effective approach for fast prototyping of multi-agent AI systems on edge devices like the Raspberry Pi.

---

[WEBXSKILL: Skill Learning for Autonomous Web Agents](http://arxiv.org/abs/2604.13318)

- WEBXSKILL: introduces a framework that bridges the grounding gap in web agents by pairing parameterized action programs with step-level natural language guidance, utilizing Skill Extraction, Skill Organization, and Skill Deployment.
- The framework employs a three-stage pipeline to mine reusable skills from synthetic trajectories, organize them into a URL-based skill graph for context-aware retrieval, and deploy them via Grounded Mode or Guided Mode.
- WEBXSKILL improves task success rates on WebArena and WebVoyager by enabling both efficient automated execution and robust agent-driven adaptation through its dual deployment paradigm.

---

[Can Agents Secure Hardware? Evaluating Agentic LLM-Driven Obfuscation for IP Protection](http://arxiv.org/abs/2604.13298)

- Agentic IP Obfuscation Framework: introduces an agentic, LLM-driven pipeline that automates hardware netlist obfuscation by integrating Circuit Parser, Feature Extractor, Retrieval Agent, Context Store, LLM Planning Agent, LLM Synthesis Agent, Deterministic Lock Renderer, Compiler, Functional Verification Agent, Security Evaluation Agent, LLM Refinement Agent, Candidate Scoring, and Best-Design Selection.
- The framework decomposes the obfuscation task into specialized stages, utilizing LLM agents for reasoning and planning while employing deterministic modules for netlist compilation, verification, and SAT-based security evaluation.
- Experimental results on ISCAS-85 benchmarks demonstrate that the framework consistently generates functionally correct obfuscated netlists, though SAT-based analysis reveals that current templates remain vulnerable to key recovery.

---

[Agentic MR sequence development: leveraging LLMs with MR skills for automatic physics-informed sequence development](http://arxiv.org/abs/2604.13282)

- Agent4MR: introduces an agent-based framework that leverages LLMs with specialized MR knowledge and physics-based validation tools to automatically generate and refine MRI pulse sequences.
- The framework utilizes an iterative refinement loop where the LLM generates code, executes it via the PyPulseq library, and receives structured validation reports to ensure physical consistency.
- Agent4MR enables autonomous MR research by allowing agents to compete on a leaderboard, iteratively optimizing sequence parameters and reconstruction approaches against a signal-equation target.

---

[Attention to task structure for cognitive flexibility](http://arxiv.org/abs/2604.13281)

- Attention-based models: introduces a class of attention-augmented architectures that improve cognitive flexibility by decomposing tasks into reusable components through selective routing.
- The framework utilizes Attention-Gating or Attention-Concatenation mechanisms to dynamically prioritize task-relevant information, enabling superior generalization and stability compared to standard MLPs.
- The research demonstrates that environmental richness and task connectivity are critical factors that interact with model architecture to shape the emergence of compositional, cue-sensitive representations.

---

[Dynamic Regret in Time-varying MDPs with Intermittent Information](http://arxiv.org/abs/2604.13255)

- Skip-update learning and planning framework: introduces a decision-making approach for TVMDPs that performs model estimation and policy planning only at intermittent update times, utilizing a Constrained maximum likelihood estimator, Finite-horizon planner, Receding-horizon controller, Uncertainty set, and Dataset of observed transitions.
- The framework addresses performance degradation in resource-constrained environments by decomposing dynamic regret into errors from update times and skip intervals.
- Theoretical analysis establishes a dynamic regret bound that quantifies the impact of temporal variation, estimation uncertainty, and the duration of intervals without updates.

---

[Capability-Aware Heterogeneous Control Barrier Functions for Decentralized Multi-Robot Safe Navigation](http://arxiv.org/abs/2604.13245)

- CA-HCBF: introduces a decentralized framework for consistent safety enforcement in heterogeneous robot teams by mapping diverse kinematic models into a unified operational space.
- The framework utilizes a support function-based metric to distribute safety responsibilities proportionally to each robot's motion capability while employing a feasibility-aware clipping mechanism to mitigate QP infeasibility.
- Experimental results demonstrate that the approach maintains safety and improves task efficiency in dense multi-robot environments compared to existing decentralized navigation baselines.

---

[On the Creativity of AI Agents](http://arxiv.org/abs/2604.13242)

- Dualistic Framework for AI Creativity: introduces a theoretical taxonomy that evaluates AI agents through functionalist perspectives, focusing on observable output traits, and ontological perspectives, emphasizing underlying generative processes.
- The paper argues that while current LLM-based agents exhibit functionalist creativity, they lack the intrinsic motivation, continual learning, and intentionality required for ontological creativity.
- The authors propose that future research should focus on bridging these gaps to move beyond simple interpolation and toward genuine transformational creativity in agentic systems.

---

[SemiFA: An Agentic Multi-Modal Framework for Autonomous Semiconductor Failure Analysis Report Generation](http://arxiv.org/abs/2604.13236)

- SemiFA: introduces an agentic multi-modal framework that autonomously generates structured failure analysis reports for semiconductor manufacturing by integrating DINOv2 visual embeddings, SECS/GEM equipment telemetry, and historical data via a LangGraph-orchestrated pipeline.
- The framework utilizes specialized agents including DefectDescriber, RootCauseAnalyzer, SeverityClassifier, and RecipeAdvisor, all powered by a shared LLaVA-1.6 LLM to perform complex reasoning and generate actionable process recommendations.
- The system is supported by the SEMIFA-930 dataset, which provides 930 annotated semiconductor defect images paired with structured narratives to facilitate VLM instruction tuning for industrial failure analysis.

---

[Numerical Instability and Chaos: Quantifying the Unpredictability of Large Language Models](http://arxiv.org/abs/2604.13206)

- NIC framework: introduces a rigorous analysis of how floating-point rounding errors propagate through Transformer computation layers to cause unpredictable LLM outputs.
- The research identifies three distinct stability regimes—Constant, Chaotic, and Signal-Dominated—that characterize how microscopic input perturbations affect model behavior.
- The study demonstrates that LLM instability is primarily driven by floating-point scale rather than Jacobian spectral properties and proposes noise averaging as an effective mitigation strategy.

---

[SciFi: A Safe, Lightweight, User-Friendly, and Fully Autonomous Agentic AI Workflow for Scientific Applications](http://arxiv.org/abs/2604.13180)

- SciFi: introduces a safe, lightweight, and user-friendly agentic framework designed for the autonomous execution of closed-loop scientific tasks using a three-layer agent loop and isolated execution environment.
- The framework utilizes Self-Assessed Modules (SAMs) to define verifiable tasks, enabling iterative refinement through pre-scan-, work- and review-agents until explicit stopping criteria are met.
- SciFi incorporates a model-gateway interface for LLM swapping, a skill library for domain-specific knowledge, and a secure containerized runtime to ensure reliable, unattended operation in scientific research environments.

---

[Exploration and Exploitation Errors Are Measurable for Language Model Agents](http://arxiv.org/abs/2604.13151)

- Exploration and Exploitation Error Metric Framework: introduces a policy-agnostic metric to quantify exploration and exploitation errors in LLM agents by analyzing action trajectories within partially observable grid-map environments paired with symbolic task DAGs.
- The framework utilizes an agent harness and a rule-based memory manager to provide structured state summaries, enabling systematic evaluation of LLM agent reasoning without relying on semantic priors.
- Experimental results demonstrate that low exploration error is a strong predictor of task success, and that harness engineering significantly improves performance across various frontier LLMs.

---

[A hierarchical spatial-aware algorithm with efficient reinforcement learning for human-robot task planning and allocation in production](http://arxiv.org/abs/2604.12669)

- EBQ&SAP (Efficient Buffer-based Deep Q-learning and Path planning-based Spatially Aware method): introduces a hierarchical framework for human-robot task planning and allocation that utilizes a high-level RL agent for task selection and a low-level agent for spatial-aware task assignment.
- The high-level agent employs an efficient buffer-based Deep Q-learning method, incorporating a Transformer-based architecture, dueling network, and Noisy Nets to address sparse reward challenges and process heterogeneous production data.
- The low-level agent utilizes an offline node graph generated via path planning to compute real-time distances, enabling efficient task allocation to the nearest human or robot in dynamic manufacturing environments.

---


#### 13th April 2026


[The A-R Behavioral Space: Execution-Level Profiling of Tool-Using Language Model Agents in Organizational Deployment](http://arxiv.org/abs/2604.12116)

- A-R Behavioral Space: introduces an execution-layer measurement approach for tool-using LLMs that characterizes behavioral allocation across Action Rate (A), Refusal Signal (R), and Divergence (D).
- The framework evaluates LLMs across four normative regimes and three autonomy scaffolds to map how linguistic signaling and operational execution redistribute under varying contextual pressures.
- By replacing aggregate safety scores with coordinate-based behavioral profiles, the method provides a deployment-oriented lens for selecting and configuring LLM agents in risk-sensitive organizational environments.

---


[Mathematics Teachers’ Interactions with a Multi-Agent System for Personalized Problem Generation](http://arxiv.org/abs/2604.12066)

- PPB: introduces a teacher-in-the-loop multi-agent architecture that leverages GPT-4o to generate and refine personalized mathematics word problems for middle school students.
- The system utilizes four specialized agents—authenticity, realism, reading-level, and hallucination—to evaluate and iteratively improve problem quality before teacher review and final deployment.
- Empirical results indicate that while the multi-agent approach effectively manages realism and readability, achieving deep contextual authenticity remains a challenge due to the variability of student interests.

---

[LLM-Redactor: An Empirical Evaluation of Eight Techniques for Privacy-Preserving LLM Requests](http://arxiv.org/abs/2604.12064)

- LLM-Redactor: introduces a systematic empirical evaluation of eight privacy-preserving techniques for LLM requests, implementing them in an open-source shim that manages the request pipeline.
- The framework utilizes a multi-stage pipeline including local-only inference, redaction, semantic rephrasing, TEE-hosted inference, split inference, FHE, MPC, and DP noise to mitigate privacy risks in cloud-bound prompts.
- The study identifies that no single technique dominates, recommending a combination of local routing, redaction, and semantic rephrasing as the most effective practical approach for balancing privacy and utility.

---


[COCOABENCH: Evaluating unified digital agents in the wild](http://arxiv.org/abs/2604.11201)

- COCOABENCH: introduces a benchmark for evaluating unified digital agents on complex, long-horizon tasks requiring the flexible composition of Vision, Search, and Coding.
- COCOA-AGENT: provides a lightweight, modular scaffold built on an AIO Sandbox to enable controlled, reproducible evaluation of LLMs across diverse digital environments.
- The research identifies that current agents struggle with Reasoning & Planning, Tool & Execution, and Visual Grounding, highlighting the critical role of code execution in successful task completion.

---

[Relax: An Asynchronous Reinforcement Learning Engine for Omni-Modal Post-Training at Scale](http://arxiv.org/abs/2604.11554)

- Relax: introduces a service-oriented RL training engine that utilizes a Controller, Rollout Service Group, Distributed Data Center Hub, TransferQueue Controller, Trainer Service Group, Distributed Checkpoint Service, Storage Units, and Health Monitoring Subsystem to enable asynchronous, omni-modal post-training at scale.
- The framework employs a decoupled architecture where independent services communicate via a field-based asynchronous data bus to eliminate synchronous bottlenecks and improve operational robustness.
- Relax supports omni-modal inputs and agentic workflows by integrating modality-aware parallel strategies and a staleness-unified training mechanism that allows flexible switching between on-policy and off-policy execution.

---

[Do Agent Rules Shape or Distort? Guardrails Beat Guidance in Coding Agents](http://arxiv.org/abs/2604.11088)

- Claude Code: introduces an empirical evaluation of how natural language rule files influence the performance and reliability of autonomous coding agents.
- The study demonstrates that rule files improve agent performance primarily through context priming rather than specific instruction, with negative constraints providing more consistent benefits than positive directives.
- The research utilizes a PBRS framework to categorize rules, revealing that while individual rules often degrade performance, they exhibit ensemble resilience and collective utility.

---


[Reason and Restore: Improving Universal Image Restoration with Chain-of-Thought Reasoning Framework](http://arxiv.org/abs/2604.09511)

- R&R: introduces a unified framework that integrates structured Chain-of-Thought reasoning into the image restoration pipeline to improve performance under diverse and unknown degradations.
- The framework utilizes a Reasoner VLM to perform structured degradation diagnosis, which provides diagnostic priors to a Restorer VLM for scene-adaptive pixel-level reconstruction.
- Reinforcement Learning is employed to align the Restorer VLM with diagnostic rewards, ensuring that the restoration process is guided by the inferred degradation severity and semantic context.

---

[A collaborative agent with two lightweight synergistic models for autonomous crystal materials research](http://arxiv.org/abs/2604.11540)

- MatBrain: introduces a dual-model collaborative agent system that decouples domain-specific analytical reasoning from tool-based executive orchestration to resolve entropy conflicts in materials science research.
- The architecture utilizes Mat-R1 (30B parameters) for expert-level crystallographic interpretation and Mat-T1 (14B parameters) for autonomous tool-based workflow execution via the Mat-MCP protocol.
- By employing a graph-based state machine, the system enables iterative "Reason-Act-Observe" loops that significantly outperform monolithic LLMs in complex materials discovery tasks while reducing hardware deployment requirements.

---

[A Simulation-Based Method for Testing Collaborative Learning Scaffolds Using LLM-Based Multi-Agent Systems](http://arxiv.org/abs/2604.11161)

- MetaGPT framework: introduces a multi-agent simulation system for testing collaborative learning scaffolds, utilizing Communication Module (manages information flow), Memory Module (stores historical dialogues), Agent Module (handles cognitive processing), Teacher Agent (manages learning process), Student Agent (emulates learner behaviors), GPT-4o (core driving LLM), and DeepSeek-v3 (prompt optimization model).
- The system evaluates two scaffolding strategies, "Deep Think before Speak" and "Direct Speak," to assess their impact on discourse quality and interaction patterns in simulated collaborative learning.
- Results indicate that the "Deep Think before Speak" scaffold significantly improves discourse diversity and interaction depth by fostering structured cognitive processes among student agents.

---

[Improving Layout Representation Learning Across Inconsistently Annotated Datasets via Agentic Harmonization](http://arxiv.org/abs/2604.11042)

- Agentic Harmonization: introduces a workflow that reconciles heterogeneous annotation semantics and bounding box granularity across datasets using a VLM Agent and a Structured Rule Set before training.
- The approach transforms source annotations into a unified target standard, ensuring consistent spatial supervision for the Detector and preventing the degradation of Post-decoder Embeddings during fine-tuning.
- Experimental results demonstrate that this harmonization process improves end-to-end document transformation quality and produces more compact, separable latent representations compared to naïve mixed-dataset fine-tuning.

---


[Risk-seeking conservative policy iteration with agent-state based policies for Dec-POMDPs with guaranteed convergence](http://arxiv.org/abs/2604.09495)

- RS-CPI (Risk-Seeking Conservative Policy Iteration): introduces a scalable algorithm for Dec-POMDPs that combines agent-state based policies with risk-seeking objectives and conservative updates to achieve monotonic convergence to local optima.
- The framework utilizes a risk-seeking objective to provide an optimistic initialization, which is then annealed to a risk-neutral objective to ensure convergence to high-quality local optima in polynomial runtime.
- By incorporating finite memory constraints through agent states, the approach effectively balances computational tractability with performance, demonstrating near-optimal results on standard Dec-POMDP benchmarks.

---

[Multiscale Perturbative Approach to Active Matter with Motility Regulation](http://arxiv.org/abs/2604.09453)

- Multiscale Perturbative Approach: introduces a coarse-graining method for dry scalar active matter that derives effective hydrodynamic descriptions by performing a multiscale perturbative expansion of the backward Kolmogorov equation.
- The framework separates slow center of mass dynamics from fast internal Rouse modes and orientational degrees of freedom to obtain a closed Fokker-Planck equation for the center of mass.
- This approach enables the analysis of diverse motility regulation strategies, including space-dependent self-propulsion, taxis, and quorum-sensing interactions, while identifying conditions for effective equilibrium regimes.

---

[Yes, But Not Always. Generative AI Needs Nuanced Opt-in.](http://arxiv.org/abs/2604.09413)

- Inference-time opt-in architecture: introduces an agent-based framework to verify user intent against rights holder consent conditions at inference time, utilizing a User Intent Agent, an Opt-in Agent, and a Consent Registry.
- The framework shifts control from binary training-time opt-out to nuanced, verifiable consent at inference, ensuring that generative AI outputs respect the specific conditions set by rights holders.
- By integrating attribution-aware generation and tracking, the architecture enables a transparent value exchange, allowing rights holders to maintain autonomy over their style, likeness, and creative works.

---

[Multi-Agent Decision-Focused Learning via Value-Aware Sequential Communication](http://arxiv.org/abs/2604.08944)

- SeqComm-DFL: introduces a multi-agent coordination framework that optimizes communication messages for downstream decision quality rather than intermediate reconstruction objectives.
- The framework utilizes value-aware message generation with sequential Stackelberg conditioning to establish a leader-follower structure that resolves coordination ambiguity.
- It integrates decision-focused learning with communication-augmented world models, employing bilevel optimization to ensure communication directly improves team performance.

---

[EVGeoQA: Benchmarking LLMs on Dynamic, Multi-Objective Geo-Spatial Exploration](http://arxiv.org/abs/2604.07070)

- GeoRover: introduces a tool-augmented agent architecture to evaluate LLMs' capacity for dynamic, multi-objective geo-spatial exploration, utilizing an LLM-based Agent, SearchStations Tool, SearchPOIs Tool, ChangeLocation Tool, CalculateDistance Tool, Memory, and Prompt Engineering.
- The framework employs an iterative exploration process where the agent dynamically invokes tools to satisfy dual-objective constraints of charging necessity and co-located activity preference.
- Experimental results demonstrate that while LLMs exhibit emergent trajectory summarization capabilities, they suffer from performance degradation in long-range tasks due to a pervasive "laziness" phenomenon.

---

[Detecting Safety Violations Across Many Agent Traces](http://arxiv.org/abs/2604.11806)

- Meerkat: introduces a repository-level auditing framework that combines clustering and agentic reasoning to identify safety violations distributed across multiple agent traces.
- The framework organizes trace repositories into hierarchical structures to deprioritize benign regions and focus agentic search on suspicious clusters.
- Meerkat improves detection of misuse, reward hacking, and covert sabotage by analyzing sets of traces jointly rather than relying on independent per-trace monitoring.

---

[CLAWGUARD: A Runtime Security Framework for Tool-Augmented LLM Agents Against Indirect Prompt Injection](http://arxiv.org/abs/2604.11790)

- CLAWGUARD: introduces a runtime security framework that enforces deterministic, user-confirmed rule sets at tool-call boundaries to mitigate indirect prompt injection in LLM agents.
- The framework utilizes a Content Sanitizer, Rule Evaluator, Skill Inspector, and Approval Mechanism to intercept adversarial tool calls without requiring model fine-tuning or infrastructure changes.
- By automatically inducing task-specific access constraints from user objectives, CLAWGUARD provides robust, context-aware protection against web, MCP, and skill-based injection channels.

---

[GenTac: Generative Modeling and Forecasting of Soccer Tactics](http://arxiv.org/abs/2604.11786)

- GenTac: introduces a diffusion-based generative framework that models soccer tactics as a stochastic process over multi-player trajectories and semantic events, utilizing Trajectory Tokenization, Token Embedding, Spatiotemporal Attention Backbone, Denoising Decoder, Semantic Decoder, and Attention Pooling Module.
- The framework employs an autoregressive sliding-window rollout strategy to generate diverse, long-horizon future trajectories conditioned on game context, team identity, or strategic objectives.
- GenTac unifies trajectory forecasting and tactical event recognition, demonstrating robust performance across soccer and other team sports like basketball, American football, and ice hockey.

---

[ClawGUI: A Unified Framework for Training, Evaluating, and Deploying GUI Agents](http://arxiv.org/abs/2604.11784)

- ClawGUI: introduces a unified open-source framework that integrates scalable online RL training, standardized evaluation, and real-device deployment for GUI agents.
- The framework utilizes ClawGUI-RL for dense step-level supervision via GiGPO, ClawGUI-Eval for reproducible benchmarking across 11+ models, and ClawGUI-Agent for hybrid CLI-GUI control on real devices.
- ClawGUI-2B achieves a 17.1% success rate on the MobileWorld benchmark, demonstrating that well-engineered infrastructure enables high performance at modest model scales.

---

[Autonomous Diffractometry Enabled by Visual Reinforcement Learning](http://arxiv.org/abs/2604.11773)

- LaueRL: introduces a model-free visual reinforcement learning framework that enables autonomous alignment of single crystals by navigating reciprocal space using raw Laue diffraction patterns.
- The system utilizes an actor-critic architecture with a CNN encoder to process visual inputs and a robotic arm to execute rotational actions, achieving alignment without explicit physical models or human supervision.
- Domain randomization and curriculum learning are employed to ensure robust transfer of learned alignment behaviors from simulated environments to real-world experimental settings.

---

[λA: A Typed Lambda Calculus for LLM Agent Composition](http://arxiv.org/abs/2604.11767)

- λA (Typed Lambda Calculus for LLM Agent Composition): introduces a formal semantics for LLM agent configurations to enable static analysis and ensure structural well-formedness.
- The framework provides a unifying calculus for diverse agent paradigms including graph state machines, role-driven agents, and multi-agent systems by mapping them to a common set of formal primitives.
- The implementation, lambdagent, includes a compiler and lint tool that identifies semantic defects in agent configurations by detecting missing base cases, vacuous loops, and incomplete dispatch logic.

---

[Retrieval Is Not Enough: Why Organizational AI Needs Epistemic Infrastructure](http://arxiv.org/abs/2604.11759)

- OIDA (Organizational Infrastructure for Decision-making and Analysis): introduces a framework that structures organizational knowledge as typed Knowledge Objects to enable computable epistemic fidelity for LLMs, utilizing Knowledge Object model, Knowledge Gravity Engine, Hybrid Retrieval architecture, and Agent Context.
- The framework employs a Knowledge Gravity Engine to maintain importance scores deterministically, incorporating class-specific decay and signed contradiction propagation to surface unresolved organizational ignorance as a first-class signal.
- OIDA provides a reusable Epistemic Quality Score methodology to evaluate how well AI systems represent commitment strength, contradiction status, and organizational ignorance.

---

[StarVLA-α: Reducing Complexity in Vision-Language-Action Systems](http://arxiv.org/abs/2604.11757)

- StarVLA-α: introduces a streamlined VLA baseline that pairs a strong VLM backbone (Qwen3-VL foundation model) with a lightweight MLP action head (lightweight continuous action regressor) and minimal data processing (minimal shared preprocessing) to achieve competitive performance across diverse robotic benchmarks.
- The framework utilizes unified data pipeline (minimal shared preprocessing) and action padding (uniform 32-dimension vector expansion) to enable cross-benchmark training (joint multi-dataset learning) without requiring task-specific engineering.
- Empirical results demonstrate that architectural and engineering complexity often provide limited benefits, as a strong VLM backbone (Qwen3-VL foundation model) combined with minimal design is sufficient for robust performance across multiple embodiments.

---

[Agentic Aggregation for Parallel Scaling of Long-Horizon Agentic Tasks](http://arxiv.org/abs/2604.11753)

- AggAgent: introduces an agentic aggregation framework that treats parallel trajectories as an interactive environment to synthesize high-quality solutions for long-horizon tasks.
- The framework utilizes an Aggregator Agent equipped with specialized tools to perform coarse-to-fine investigation of trajectories, enabling cross-trajectory reasoning without the need for full context loading.
- AggAgent achieves Pareto-optimal performance and efficiency by selectively retrieving trajectory information, maintaining full fidelity while keeping aggregation costs bounded by a single agentic rollout.

---

[MULTI-ORFT: Stable Online Reinforcement Fine-Tuning for Multi-Agent Diffusion Planning in Cooperative Driving](http://arxiv.org/abs/2604.11734)

- MULTI-ORFT: introduces a cooperative multi-agent trajectory diffusion planner that couples scene-conditioned pre-training with stable online reinforcement learning post-training to improve closed-loop safety and efficiency.
- The framework utilizes a two-level MDP to integrate an inner denoising process with an outer environment-interaction loop, enabling analytically tractable policy optimization.
- To ensure training stability, the approach employs a variance-gated group-relative policy optimization (VG-GRPO) strategy that mitigates advantage collapse and gradient instability during online refinement.

---

[The Devil is in the Details – From OCR for Old Church Slavonic to Purely Visual Stemma Reconstruction](http://arxiv.org/abs/2604.11724)

- Agentic OCR Pipeline: introduces a modular framework for Old Church Slavonic transcription that utilizes a Planner Agent (GPT-5 orchestrator), Visual Format Agent (JSON-based formatting), Linguistic Agent (script/genre analysis), OCR Executor (prompt-guided extraction), Review Agent (error JSON validation), Post-Correction Agent (evidence-based refinement), Diacritic Checker (specialized diacritic correction), Special Letter Agent (ligature/old letter correction), Visual Tools (upscaling/Red2Black), and a RAG Database (few-shot example retrieval).
- The framework leverages LLMs to perform complex OCR tasks on historical manuscripts by decomposing the process into specialized agentic subtasks and integrating external tools for image enhancement.
- The research further proposes a purely visual stemmatic method that bypasses full transcription by using deep visual embeddings and clustering to compute manuscript distances.

---

[Evaluating Cooperation in LLM Social Groups through Elected Leadership](http://arxiv.org/abs/2604.11721)

- AgentElect: introduces a multi-agent LLM simulation framework that evaluates how elected leadership and candidate-driven agendas influence cooperation and social welfare in common-pool resource dilemmas.
- The framework incorporates Policy phase, Election phase, Harvest phase, Discussion phase, Reflection phase, Agent memory, Truthfulness flag, SVO personas, Sustainability threshold, and Agent social graph to model structured governance.
- Empirical results demonstrate that elected leadership significantly improves social welfare and survival time compared to leaderless or fixed-leadership populations, while revealing an "Equality Paradox" where diverse electoral outcomes create skewed resource distribution.

---

[SWE-AGILE: A Software Agent Framework for Efficiently Managing Dynamic Reasoning Context](http://arxiv.org/abs/2604.11716)

- SWE-AGILE: introduces a framework that decouples reasoning depth from context constraints by maintaining a sliding window of detailed reasoning while compressing historical thoughts into concise reasoning digests.
- The framework utilizes trajectory snapshot training and a hindsight backfilling pipeline to align model training with dynamic inference visibility constraints.
- By integrating a compression-aware reward function within RLVR, the agent learns to adaptively balance deep System-2 reasoning during complex tasks with efficient context management.

---

[A Distributed Bilevel Framework for the Macroscopic Optimization of Multi-Agent Systems](http://arxiv.org/abs/2604.11712)

- BILD-MACRO (BILevel Distributed hypergradient for MACRoscopic Optimization): introduces a distributed bilevel optimization framework to control emergent macroscopic behaviors in large-scale multi-agent systems by coupling microscopic state updates with macroscopic density estimation.
- The framework utilizes a consensus-based mechanism to locally reconstruct global quantities, enabling agents to perform hypergradient-based optimization without requiring centralized control.
- By parameterizing the macroscopic state as an exponential-family distribution, the approach reduces communication overhead by exchanging compressed representations rather than full probability density functions.

---

[EA-Agent: A Structured Multi-Step Reasoning Agent for Entity Alignment](http://arxiv.org/abs/2604.11686)

- EA-Agent: introduces a reasoning-driven framework that decomposes entity alignment into a structured multi-step process of Path Planning, tool execution, and Agent Optimization.
- The framework utilizes Attribute Triple Selector and Relation Triple Selector to filter redundant information, significantly reducing token consumption for the LLM-based Entity Alignment Tool and Reflector.
- An offline Agent Optimization mechanism uses a reward-based trajectory rewriting strategy to iteratively improve the agent's path planning policy and alignment stability.

---

[Playing Along: Learning a Double-Agent Defender for Belief Steering via Theory of Mind](http://arxiv.org/abs/2604.11666)

- TOM-SB (TOM FOR STEERING BELIEFS): introduces a multi-turn adversarial dialogue environment where a defender must protect private information by forming a ToM of the attacker and steering their beliefs using Shared Hierarchical Universe, Defender Private Information, and Attacker Prior Knowledge.
- The framework utilizes an Attacker Agent and a Defender Agent, where the defender employs a ToM Estimate and a Trust Score to balance information protection with maintaining attacker engagement through Trajectory-level ToM Reward and Fooling Reward.
- The research demonstrates that training defenders with both ToM and fooling rewards leads to emergent, synergistic improvements in both capabilities, outperforming frontier LLMs that rely solely on prompting.

---

[Towards Autonomous Mechanistic Reasoning in Virtual Cells](http://arxiv.org/abs/2604.11661)

- VCR-Agent: introduces a multi-agent framework that integrates biologically grounded knowledge retrieval with a verifier-based filtering approach to generate and validate mechanistic reasoning autonomously.
- The framework utilizes a Report Generator to synthesize biological facts and an Explanation Constructor to transform these into structured mechanistic action graphs for virtual cells.
- To ensure reliability, the system employs specialized verifiers to filter out factually inconsistent reasoning traces, thereby improving the factual precision of LLMs in scientific domains.

---

[RPA-Check: A Multi-Stage Automated Framework for Evaluating Dynamic LLM-based Role-Playing Agents](http://arxiv.org/abs/2604.11655)

- RPA-Check: introduces a multi-stage automated evaluation framework that transforms qualitative role-playing requirements into granular, verifiable boolean indicators through Dimension Definition, Augmentation, Semantic Filtering, and LLM-as-a-Judge Evaluation.
- The framework utilizes LLM Court, a forensic simulation environment, which integrates an Agentic State Machine, an Intent Analysis Module, and a Sentence Analyzer to enforce strict procedural constraints on LLM-based agents.
- Experimental results demonstrate that smaller, instruction-tuned models often outperform larger architectures in procedural consistency, highlighting the impact of instruction tuning over parametric scale in constrained environments.

---

[CodeTracer: Towards Traceable Agent States](http://arxiv.org/abs/2604.11641)

- CodeTracer: introduces a tracing architecture that parses heterogeneous run artifacts to reconstruct full state transition histories as hierarchical trace trees for failure onset localization.
- The framework utilizes an Extraction Agent for standardization, a Structuring Agent for hierarchical indexing, and a Trace Agent to identify failure-critical stages and evidence.
- CodeTracer includes a reflective replay mechanism that injects diagnostic signals back into LLMs to help agents recover from early mistakes and improve task success.

---

[Micro-Dexterity in Biological Micromanipulation: Embodiment, Perception, and Control](http://arxiv.org/abs/2604.11640)

- Micro-dexterity framework: introduces a capability-oriented perspective on biological micromanipulation by synthesizing advances in Embodied microrobots, Field-mediated systems, Externally actuated end-effectors, Sensing-to-perception pipeline, Physics-informed state estimation, Control and learning strategies, and Task-level capabilities.
- The paper reformulates classical dexterity for the microscale, where low-Reynolds-number dynamics and adhesion-dominated interactions necessitate a shift from rigid-body mechanics to compliant, field-coupled, and swarm-based architectures.
- It identifies a persistent "dexterity gap" between laboratory proof-of-concept demonstrations and the integrated, feedback-regulated, multi-step manipulation required for clinical intervention.

---

[Back to Basics: Let Conversational Agents Remember with Just Retrieval and Generation](http://arxiv.org/abs/2604.11628)

- Nano-Memory: introduces a minimalist framework for conversational agents that addresses the Signal Sparsity Effect by replacing complex memory structures with Turn Isolation Retrieval and Query-Driven Pruning.
- The framework utilizes TIR to isolate high-evidence signals within the latent manifold and QDP to purge inter- and intra-session noise from retrieved contexts.
- Nano-Memory achieves superior performance and efficiency by operating directly on unstructured dialogue history, bypassing the need for complex offline memory maintenance.

---

[CONTEXT KUBERNETES: DECLARATIVE ORCHESTRATION OF ENTERPRISE KNOWLEDGE FOR AGENTIC AI SYSTEMS](http://arxiv.org/abs/2604.11623)

- Context Kubernetes: introduces a reference architecture for enterprise knowledge orchestration in agentic AI systems, utilizing a declarative manifest and a reconciliation loop to manage knowledge access, permissions, and freshness.
- The architecture incorporates a three-tier agent permission model that enforces the design invariant that agent authority must be a strict subset of human authority, with out-of-band approval isolation for high-stakes actions.
- The system provides a vendor-neutral governance layer that operates independently of specific LLMs or agent frameworks, ensuring secure and governed knowledge delivery to agents.

---

[The Unified Field Theory of Phygital Space](http://arxiv.org/abs/2604.11619)

- Unified Field Theory of Phygital Space: introduces a sheaf-theoretic framework utilizing Fiber Bundle, Sheaf, Finsler Geometry, Ontological Mass Tensor, Autopoietic Dynamics, Non-Equilibrium Thermodynamics, Lie-Derivative Formalism, and Synthetic Agents to model reality as a coupled manifold of physical, digital, and social dimensions.
- The framework models platforms as dissipative structures that generate value through negentropy while incurring entropic costs, and explains modern temporal pathologies through Lie-derivative-based temporal shear.
- Empirical validation is provided through a longitudinal analysis of the Chinese e-commerce ecosystem, demonstrating how platforms converge toward balanced mass profiles to mitigate dimensional friction and entropy.

---

[Utilizing and Calibrating Hindsight Process Rewards via Reinforcement with Mutual Information Self-Evaluation](http://arxiv.org/abs/2604.11611)

- MISE (Mutual Information Self-Evaluation): introduces a reinforcement learning paradigm that utilizes hindsight generative self-evaluation as dense rewards while calibrating them against environmental feedback to mitigate reward sparsity in LLM agents.
- The framework addresses the unreliability of LLM-based evaluators by proving that hindsight self-evaluation is equivalent to minimizing a mutual information term and an inter-policy KL divergence term.
- MISE incorporates a calibration reward to align internal self-evaluations with actual task performance, preventing the agent from overfitting to biased self-evaluation signals.

---

[Self-Evolving LLM Memory Extraction Across Heterogeneous Tasks](http://arxiv.org/abs/2604.11610)

- CluE (Cluster-based Evolution): introduces a self-evolving framework that groups training examples into clusters by extraction scenarios to perform independent analysis and synthesize cross-cluster insights for prompt optimization.
- The framework utilizes a Summarizer, Cluster Manager, Cluster Analyzer, and Proposer to iteratively refine extraction prompts while maintaining stability across heterogeneous task distributions.
- The paper also presents BEHEMOTH, a benchmark methodology repurposing 18 existing datasets to evaluate the generalizability of memory extraction systems across personalization, problem-solving, and agentic tasks.

---

[UniToolCall: Unifying Tool-Use Representation, Data, and Evaluation for LLM Agents](http://arxiv.org/abs/2604.11557)

- UniToolCall: introduces a unified framework that standardizes the entire pipeline for LLM agent tool learning, from toolset construction and hybrid data synthesis to fine-grained evaluation using a shared QAOA representation.
- The framework utilizes a synthetic data generation pipeline that explicitly models serial and parallel execution structures and incorporates an Anchor Linkage mechanism to ensure coherent multi-turn reasoning.
- Experiments demonstrate that fine-tuning Qwen3-8B on the UniToolCall dataset achieves state-of-the-art performance in tool selection and parameter grounding, significantly outperforming strong baselines under distractor-heavy conditions.

---

[FM-AGENT: Scaling Formal Methods to Large Systems via LLM-Based Hoare-Style Reasoning](http://arxiv.org/abs/2604.11556)

- FM-AGENT: introduces a top-down paradigm for automated compositional reasoning in large-scale systems by leveraging LLMs to generate natural language specifications and verify code correctness.
- The framework utilizes caller-driven specification generation to capture developer intent and employs Hoare-style inference to reason about function implementations concurrently.
- FM-AGENT integrates a bug validator that generates system-entry test cases to confirm potential bugs, effectively identifying hundreds of previously undetected issues in large-scale software systems.

---

[SemaClaw: A Step Towards General-Purpose Personal AI Agents through Harness Engineering](http://arxiv.org/abs/2604.11548)

- SemaClaw: introduces a multi-agent application framework that operationalizes dynamic orchestration, runtime safety, and long-term memory through a two-layer architecture separating a reusable agent runtime from an application harness.
- The framework utilizes DAG Teams for hybrid orchestration, PermissionBridge for runtime behavioral safety, and a three-tier context management architecture to maintain persistent agent personas and structured long-term memory.
- SemaClaw leverages a wiki-based personal knowledge infrastructure using plain-file Markdown to ensure user ownership and enable compounding intelligence across future agent sessions.

---

[Time is Not a Label: Continuous Phase Rotation for Temporal Knowledge Graphs and Agentic Memory](http://arxiv.org/abs/2604.11544)

- ROMEM: introduces a temporal reasoning module for graph-based agentic memory that internalises time as a continuous geometric operator to resolve temporal conflicts without destructive updates.
- The framework utilizes Functional Rotation and Geometric Shadowing to rotate dynamic facts out of phase while keeping static facts stable, ensuring an append-only memory architecture.
- A Semantic Speed Gate enables zero-shot relational volatility estimation, allowing the system to distinguish between permanent and transient facts for improved temporal reasoning accuracy.

---

[Problem Reductions at Scale: Agentic Integration of Computationally Hard Problems](http://arxiv.org/abs/2604.11535)

- Harness Engineering framework: introduces a system that leverages AI agents to build and maintain a large-scale library of NP-hard problem reductions through a structured harness of Scaffold, Skills, Tools, Knowledge Base, Verification Stack, Implementation Agent, Review Agent, Advisor Skills, Symbolic Engine, and Solvers.
- The framework utilizes a skill-based automation pipeline where agents autonomously handle coding, testing, and documentation, while human experts provide high-level guidance and verification.
- By organizing reductions into a directed graph and enforcing strict interface conventions, the system enables automated routing of computationally hard problems to specialized solvers.

---

[OOM-RL: Out-of-Money Reinforcement Learning Market-Driven Alignment for LLM-Based Multi-Agent Systems](http://arxiv.org/abs/2604.11477)

- OOM-RL (Out-of-Money Reinforcement Learning): introduces a dual-loop alignment paradigm that utilizes real-world financial capital depletion as an objective, un-hackable negative gradient to align LLMs in high-stakes environments.
- The framework employs STDAW (Strict Test-Driven Agentic Workflow) with RO-Lock (Read-Only Lock) to enforce deterministic code verification and prevent adversarial "Test Evasion" by LLMs.
- By replacing subjective human preference with economic penalties, the system forces LLMs to transition from sycophantic, high-turnover behaviors to resilient, liquidity-aware architectures.

---

[SLALOM: Simulation Lifecycle Analysis via Longitudinal Observation Metrics for Social Simulation](http://arxiv.org/abs/2604.11466)

- SLALOM (Simulation Lifecycle Analysis via Longitudinal Observation Metrics): introduces a framework for evaluating LLM-based social simulations by shifting validation from final outcome verification to process fidelity using multivariate time series analysis.
- The framework utilizes Interaction Logs, NLP Processing, and Multivariate Time Series to map agent behavior against SLALOM Gates, which represent distinct phases of social phenomena.
- By applying Dynamic Time Warping (DTW) to calculate an Aggregate Validity Score, the method distinguishes between sociologically plausible trajectories and stochastic noise in LLM simulations.

---

[Three Roles, One Model: Role Orchestration at Inference Time to Close the Performance Gap Between Small and Large Agents](http://arxiv.org/abs/2604.11465)

- Three-tier inference scaffolding pipeline: introduces a modular architecture that deploys a single frozen LLM in three distinct roles to mitigate mechanical failure modes in small agentic models.
- The framework utilizes a summarization module to manage context, a main agent model for task reasoning, and an isolated correction module to perform error-aware code revision.
- By decoupling history-dependent reasoning from history-isolated correction, the approach breaks repetitive failure loops and improves task completion rates on resource-constrained hardware.

---

[To Learn or Not to Learn: A Litmus Test for Using Reinforcement Learning in Control](http://arxiv.org/abs/2604.11463)

- Litmus Test for RL-based Control: introduces a two-part simulation-based framework to determine if RL-based control is superior to model-based control without requiring prior RL training.
- The framework utilizes a Knowledge Advantage Test to quantify performance gaps between nominal and oracle MPCs, followed by a Learnability Test using the Randomized Dependence Coefficient to identify learnable model biases.
- This approach enables significant computational resource savings by predicting the necessity of RL-based control across various complex control benchmarks.

---

[Escaping the Context Bottleneck: Active Context Curation for LLM Agents via Reinforcement Learning](http://arxiv.org/abs/2604.11462)

- ActiveContext: introduces a symbiotic framework that decouples context management from task execution by pairing a specialized ContextCurator with a frozen TaskExecutor to reduce information entropy.
- The ContextCurator uses reinforcement learning to actively prune environmental noise and preserve reasoning anchors, creating a high-fidelity working memory for the TaskExecutor.
- This approach mitigates the context bottleneck and "Lost-in-the-Middle" phenomenon, enabling smaller models to match the performance of larger proprietary models in long-horizon tasks.

---

[METRO: Towards Strategy Induction from Expert Dialogue Transcripts for Non-collaborative Dialogues](http://arxiv.org/abs/2604.11427)

- METRO: introduces a novel method that autonomously induces strategy actions and planning logic from raw dialogue transcripts to build effective non-collaborative agents.
- The framework formalizes expert knowledge into a hierarchical Strategy Forest, which captures both short-term tactical responses and long-term strategic foresight.
- During inference, METRO performs retrieval-augmented generation by querying the Strategy Forest to provide contextual guidance, significantly outperforming existing methods in non-collaborative dialogue tasks.

---

[Efficient Emotion-Aware Iconic Gesture Prediction for Robot Co-Speech](http://arxiv.org/abs/2604.11417)

- Efficient Emotion-Aware Iconic Gesture Prediction framework: introduces a lightweight transformer-based pipeline that derives iconic gesture placement and intensity from text and emotion inputs without requiring audio at inference time.
- The architecture utilizes a compact latent space bottleneck with cross- and self-attention mechanisms to achieve efficient global modeling suitable for real-time deployment on embodied agents.
- Experimental results on the BEAT2 dataset demonstrate that the model outperforms GPT-4o in both semantic gesture placement classification and intensity regression while maintaining low computational latency.

---

[A Herding-Based Model of Technological Transfer and Economic Convergence: Evidence from Central and Eastern Europe](http://arxiv.org/abs/2604.11413)

- Herding-based technological transfer model: introduces a micro-founded mechanism for total factor productivity growth by modeling technology adoption as a herding process among heterogeneous agents.
- The framework integrates Kirman’s herding model into a neoclassical growth structure to derive a tractable analytical solution for nonlinear economic convergence.
- The model is empirically validated using OECD productivity data to project long-term catch-up dynamics for Central and Eastern European economies relative to advanced benchmark nations.

---

[From Agent Loops to Structured Graphs: A Scheduler-Theoretic Framework for LLM Agent Execution](http://arxiv.org/abs/2604.11378)

- Graph Harness: introduces a scheduler-theoretic framework that models LLM agent execution as a static directed acyclic graph (DAG) to replace non-deterministic iterative loops with explicit, verifiable control structures.
- The framework utilizes Planner Layer, Runtime Layer, and Recovery Layer to enforce plan-version immutability, bounded recovery protocols, and strict context separation between execution and diagnostic data.
- By parameterizing execution systems along ready-set cardinality and policy explicitness, the paper provides a formal basis for comparing Agent Loops with graph-based executors and identifying structural trade-offs in LLM agent design.

---

[MR.ScaleMaster: Scale-Consistent Collaborative Mapping from Crowd-Sourced Monocular Videos](http://arxiv.org/abs/2604.11372)

- MR.ScaleMaster: introduces a cooperative mapping system for crowd-sourced monocular videos that addresses scale-specific failure modes using Front-end agents, Scale Collapse Alarm, Server-side inter-robot loop closure, Sim(3) anchor nodes, and Sim(3) Optimizer.
- The framework utilizes a Scale Collapse Alarm to detect and reject false-positive loop closures that cause scale degeneracy in repetitive environments.
- By generalizing the anchor-node formulation to Sim(3), the system explicitly estimates per-session scale to resolve inter-session discrepancies and produce a globally consistent dense map.

---

[The Missing Knowledge Layer in Cognitive Architectures for AI Agents](http://arxiv.org/abs/2604.11364)

- Four-layer cognitive architecture framework: introduces a decomposition of AI agent memory into Knowledge, Memory, Wisdom, and Intelligence to resolve category errors in existing cognitive architectures.
- The framework assigns distinct persistence semantics to each layer, utilizing supersession for Knowledge, Ebbinghaus decay for Memory, evidence-gated revision for Wisdom, and ephemeral processing for Intelligence.
- Empirical validation via a pilot study on the BEAM benchmark demonstrates that typed routing to these specialized layers significantly improves contradiction resolution and temporal reasoning performance compared to flat memory stores.

---

[Leader-Follower Density Control of Multi-Agent Systems with Interacting Followers: Feasibility and Convergence Analysis](http://arxiv.org/abs/2604.11353)

- Leader-Follower Density Control framework: introduces a macroscopic control approach using coupled partial differential equations to steer large-scale agent populations toward desired spatial distributions while explicitly accounting for follower-follower interactions.
- The framework derives necessary and sufficient feasibility conditions linking target distributions to interaction strength, diffusion, and leader mass, while providing a feedback control law with local asymptotic stability guarantees.
- Numerical simulations and agent-based validations demonstrate that inter-follower interactions can either facilitate or hinder control, requiring adjustments to the leader population size to maintain feasibility.

---

[Incentive Design without Hypergradients: A Social-Gradient Method](http://arxiv.org/abs/2604.11346)

- Social-Gradient Method: introduces a hypergradient-free incentive design framework that steers self-interested agents toward a socially optimal Nash equilibrium by utilizing the social cost gradient as a descent direction.
- The framework employs a two-timescale learning process where agents adapt their strategies on a fast timescale while the planner updates incentives on a slow timescale using the social-gradient flow.
- Theoretical analysis establishes global convergence to the socially optimal incentive pair under information asymmetry without requiring explicit knowledge of equilibrium sensitivities.

---

[Governance by Design: A Parsonian Institutional Architecture for Internet-Wide Agent Societies](http://arxiv.org/abs/2604.11337)

- AGIL (Adaptation, Goal Attainment, Integration, Latency): introduces a sixteen-cell institutional architecture derived from Parsonian structural-functionalism to govern internet-wide agent societies by ensuring all four functional pillars are institutionally represented.
- The paper applies a recursive sub-function diagnostic to the OpenClaw ecosystem, revealing that while technical infrastructure (A-pillar) is present, active governance, inter-cell coordination, and normative grounding (L-pillar) are critically absent.
- It proposes a prioritized institutional roadmap based on the cybernetic hierarchy, arguing that proactive design is essential before emergent agent social patterns become path-dependent and resistant to human oversight.

---

[Network Effects and Agreement Drift in LLM Debates](http://arxiv.org/abs/2604.11312)

- LLM-OD (Large Language Model-powered Opinion Dynamics): introduces a framework for simulating collective opinion evolution in populations of LLM agents by modeling interactions as structured debates on a network.
- The framework incorporates network topology, homophilic edge formation, and class imbalance to analyze how structural constraints and intrinsic model biases influence opinion convergence and polarization.
- The study identifies a systematic agreement drift, where LLMs exhibit an intrinsic directional bias toward endorsing discussion statements during pairwise interactions, independent of numerical majority effects.

---

[PaperScope: A Multi-Modal Multi-Document Benchmark for Agentic Deep Research Across Massive Scientific Papers](http://arxiv.org/abs/2604.11307)

- PaperScope: introduces a multi-modal, multi-document benchmark for evaluating agentic deep research, utilizing a Knowledge Graph, ORWAS, FileSearchTool, FileVisit, Ops-MM-embedding-v1, DeepSeek-OCR, FAISS, and vLLM to assess retrieval, synthesis, and reasoning capabilities.
- The framework employs an inverted-construction strategy to ensure answer uniqueness and multi-modal dependence, requiring agents to integrate visual and textual evidence across scientific documents.
- Experimental results on 16 state-of-the-art agents reveal that complex multi-modal reasoning and long-context retrieval remain significant bottlenecks for current LLMs in scientific research workflows.

---

[Evaluating LLM Agents on Automated Software Analysis Tasks](http://arxiv.org/abs/2604.11270)

- AnalysisAgent: introduces a purpose-built agentic architecture for automated software analysis that utilizes staged execution, single-action cycles with deterministic log condensation, and evidence-based validation.
- The framework evaluates performance across 35 tool-project pairs in AnalysisBench, demonstrating that task-specific architectural design significantly improves success rates over general-purpose LLM agents.
- Empirical results indicate that AnalysisAgent achieves a 94% verified success rate, highlighting that architectural scaffolding is more critical for complex multi-step analysis tasks than LLM capability alone.

---

[Mobile GUI Agent Privacy Personalization with Trajectory Induced Preference Optimization](http://arxiv.org/abs/2604.11259)

- TIPO (Trajectory Induced Preference Optimization): introduces a preference optimization framework for mobile GUI agents that aligns execution trajectories with user privacy personas by addressing structural heterogeneity.
- The framework utilizes preference-intensity weighting to emphasize critical privacy-related steps and a padding gating mechanism to mitigate noise from variable-length trajectory alignment.
- Experimental results demonstrate that TIPO improves persona adherence and distinction while maintaining task success rates across diverse mobile GUI tasks.

---

[Dialectic-Med: Mitigating Diagnostic Hallucinations via Counterfactual Adversarial Multi-Agent Debate](http://arxiv.org/abs/2604.11258)

- Dialectic-Med: introduces a multi-agent framework that mitigates diagnostic hallucinations in medical LLMs by enforcing rigorous adversarial dialectics between a Proponent Agent, an Opponent Agent, and a Mediator Agent.
- The framework utilizes a Visual Falsification Module to actively retrieve contradictory visual evidence, which is then integrated into a Dynamic Consensus Graph to ensure diagnostic reasoning is grounded in verified visual regions.
- By replacing linear reasoning with a dynamic verification loop, the approach significantly improves explanation faithfulness and diagnostic accuracy across multiple medical benchmarks compared to single-agent LLM baselines.

---

[Evolving Many Worlds: Towards Open-Ended Discovery in Petri Dish NCA via Population-Based Training](http://arxiv.org/abs/2604.11248)

- PBT-NCA: introduces a meta-evolutionary algorithm that evolves a population of PD-NCA (differentiable multi-agent cellular system) worlds using a composite objective of historical novelty and visual diversity to sustain open-ended complexity.
- The framework utilizes a DINOv2 Encoder (visual diversity feature extractor) and a Handcrafted Behavior Descriptor (ecological statistics feature vector) to evaluate and rank worlds within a FIFO Archive (stores past behavior descriptors) for population-level selection.
- By applying Population-Based Training (meta-evolutionary optimization loop) to interleave gradient-based agent adaptation with evolutionary selection, the system autonomously discovers diverse lifelike phenomena at the edge of chaos.

---

[Semantic Rate–Distortion Theory: Deductive Compression and Closure Fidelity](http://arxiv.org/abs/2604.11204)

- SRDT: introduces a rate–distortion theory for knowledge-intensive communication where the fidelity criterion is the preservation of the deductive closure of a source knowledge base.
- The framework leverages the receiver's ability to re-derive redundant knowledge from an irredundant core, effectively reducing the required communication rate below the Shannon entropy.
- It provides a unified treatment of heterogeneous multi-agent communication by quantifying vocabulary mismatch through an overlap decomposition of sender and receiver knowledge bases.

---

[EmbodiedGovBench: A Benchmark for Governance, Recovery, and Upgrade Safety in Embodied Agent Systems](http://arxiv.org/abs/2604.11174)

- EmbodiedGovBench: introduces a governance-oriented evaluation benchmark for embodied AI systems that measures controllability, recoverability, and auditability under realistic perturbations.
- The framework utilizes a modular harness including a Scenario Generator, Perturbation Injector, Adapter Layer, System Under Test, Trace Collector, Governance Judge, and Scoring Engine to assess seven governance dimensions.
- EmbodiedGovBench distinguishes between task-effective and operationally governable systems by evaluating performance across single-robot and fleet-scale tracks using stress-oriented protocols.

---

[MADQRL: Distributed Quantum Reinforcement Learning Framework for Multi-Agent Environments](http://arxiv.org/abs/2604.11131)

- MADQRL: introduces a distributed framework for multi-agent reinforcement learning that utilizes independent training and hybrid quantum-classical models to optimize joint tasks in environments with disjoint observation spaces.
- The architecture integrates an Encoder Circuit, a Parametrized Circuit, and a Measurement and Post Processing component within a hybrid quantum-classical model to represent agent policies.
- Experimental results on a cooperative-pong environment demonstrate that the independent training strategy, supported by the PPO algorithm and Ray framework, achieves significant performance improvements over classical models.

---

[ActorMind: Emulating Human Actor Reasoning for Speech Role-Playing](http://arxiv.org/abs/2604.11103)

- ActorMind: introduces a multi-agent chain-of-thought reasoning framework that emulates human theatrical performance to enable spontaneous, persona-consistent speech role-playing through Eye Agent, Ear Agent, Brain Agent, and Mouth Agent.
- The framework utilizes an Eye Agent for context reading, an Ear Agent for emotional perception via SECAP, a Brain Agent for emotional state reasoning using LLMs, and a Mouth Agent for speech synthesis via RAG and TTS.
- ActorMindBench provides a hierarchical benchmark derived from television sitcom data to evaluate the effectiveness of the proposed reasoning pipeline in speech role-playing scenarios.

---

[From Context to Rules: Toward Unified Detection Rule Generation](http://arxiv.org/abs/2604.11078)

- UniRule: introduces an agentic RAG framework that enables unified detection rule generation across heterogeneous contexts and languages by projecting rules into dual semantic spaces of detection intent and detection logic.
- The framework utilizes an LLM agent to autonomously perform iterative retrieval from semantic indexes, allowing the system to synthesize rules without being constrained to specific input-output pipelines.
- Evaluation across 12 scenarios and 12,000 pairwise judgments demonstrates that UniRule significantly outperforms standard RAG and LLM-only baselines in generating specification-rich detection rules.

---

[Hodoscope: Unsupervised Monitoring for AI Misbehaviors](http://arxiv.org/abs/2604.11072)

- Hodoscope: introduces an unsupervised monitoring framework that identifies distinctive agent behaviors by computing group-wise distributional differences in embedded action spaces.
- The pipeline utilizes an Action Summarizer, Embedding Model, Projection Module, Density-Difference Estimator, Interactive Visualization, and Density-Weighted Farthest Point Sampler to surface potential misbehaviors for human review.
- Hodoscope enables efficient discovery of novel agent exploits by highlighting overrepresented behavioral patterns across different models or task conditions without requiring pre-specified failure labels.

---

[Pando: Do Interpretability Methods Work When Models Won’t Explain Themselves?](http://arxiv.org/abs/2604.11061)

- Pando: introduces a model-organism benchmark that isolates interpretability-specific signal from black-box elicitation by controlling explanation faithfulness via an explanation axis.
- The framework evaluates LLMs by requiring agents to predict held-out model decisions using query-response pairs and optional outputs from various interpretability tools.
- Mechanistic analysis reveals that gradient-based methods track decision computation, while other readouts are dominated by task representation biases.

---

[Sema Code: Decoupling AI Coding Agents into Programmable, Embeddable Infrastructure](http://arxiv.org/abs/2604.11045)

- Sema Code: introduces a three-layer architecture that decouples AI coding agent reasoning from client-side delivery forms, enabling the engine to be embedded as a programmable npm library.
- The framework utilizes an event-driven core engine that includes multi-tenant isolation, FIFO input queuing, and adaptive context compression to maintain performance and state safety across concurrent sessions.
- Sema Code supports complex agentic workflows through a hierarchical runtime that includes multi-agent collaborative scheduling, intelligent Todo-based process management, and a four-layer asynchronous permission system.

---

[From Topology to Trajectory: LLM-Driven World Models for Supply Chain Resilience](http://arxiv.org/abs/2604.11041)

- ReflectiChain: introduces a cognitive agentic framework for resilient supply chain planning that integrates an Action Generation LLM, an Internal Reflection LLM, a Generative Supply Chain World Model (SC-WM), an External Reflection LLM, a Working Memory Buffer, and a LoRA Adaptation Module.
- The framework utilizes a dual-stage reflection process, combining reflection-in-action for semantic-physical evaluation and reflection-on-action for retrospective hindsight to address temporal credit assignment.
- By leveraging test-time LoRA adaptation and latent trajectory rehearsal, the system enables autonomous policy evolution to navigate non-stationary geopolitical shocks in high-fidelity supply chain environments.

---

[RTMC: Step-Level Credit Assignment via Rollout Trees](http://arxiv.org/abs/2604.11037)

- RTMC (Rollout-Tree Monte Carlo): introduces a critic-free advantage estimation framework that organizes group rollouts into a shared tree to enable fine-grained, step-level credit assignment for LLM agents.
- The framework utilizes a state-action signature system to map heterogeneous interaction histories into comparable tree nodes, allowing for unbiased Monte Carlo advantage estimation without requiring a learned value network.
- By leveraging Bayesian shrinkage via prior-based value smoothing, the approach effectively handles sparsely visited states, resulting in improved policy gradients and higher performance on complex software engineering tasks.

---

[Federated Single-Agent Robotics: Multi-Robot Coordination Without Intra-Robot Multi-Agent Fragmentation](http://arxiv.org/abs/2604.11028)

- FSAR: introduces a runtime architecture for multi-robot coordination that maintains each robot as a coherent single agent, avoiding internal multi-agent fragmentation.
- The architecture utilizes a federation layer to manage cross-robot interactions through shared capability registries, trust-scoped delegation, and policy-aware authority assignment.
- FSAR improves governance locality and recovery containment by ensuring that coordination decisions remain attributable to identifiable robot principals rather than fragmented internal agents.

---

[NimbusGuard: A Novel Framework for Proactive Kubernetes Autoscaling Using Deep Q-Networks](http://arxiv.org/abs/2604.11017)

- NimbusGuard: introduces a proactive Kubernetes autoscaling framework that integrates a DQN agent, an LSTM forecaster, and an LLM cognitive agent to optimize resource management.
- The framework utilizes a LangGraph-orchestrated workflow to process metrics, generate forecasts, and validate scaling decisions through an MCP server.
- Experimental results demonstrate that the proactive approach achieves superior responsiveness and performance compared to reactive baselines like HPA and KEDA.

---

[Sanity Checks for Agentic Data Science](http://arxiv.org/abs/2604.11003)

- PCS framework: introduces a pair of lightweight sanity checks to evaluate the trustworthiness of conclusions produced by agentic data science pipelines.
- The approach utilizes null-defining perturbations to create a negative control and PCS perturbations to probe agentic failure modes, ensuring conclusions are grounded in data rather than noise.
- Experimental results on benchmark datasets demonstrate that these sanity checks effectively identify unreliable conclusions and reveal that LLM-based agent confidence is poorly calibrated to empirical stability.

---

[When Valid Signals Fail: Regime Boundaries Between LLM Features and RL Trading Policies](http://arxiv.org/abs/2604.10996)

- Modular Financial Trading Pipeline: introduces a framework that utilizes a frozen LLM as a stateless feature extractor to generate numerical inputs for a downstream reinforcement learning trading agent.
- The system employs an automated prompt-optimization loop that treats extraction prompts as discrete hyperparameters, tuning them against the Information Coefficient to ensure predictive utility.
- Empirical results demonstrate a regime-dependent gap where LLM-derived features provide predictive signal in stable markets but introduce noise during macroeconomic shocks, highlighting challenges in transfer learning for financial RL.

---

[ArtiCAD: Articulated CAD Assembly Design via Multi-Agent Code Generation](http://arxiv.org/abs/2604.10992)

- ArtiCAD: introduces a training-free multi-agent system that generates editable, articulated CAD assemblies from text or images by decomposing the task into specialized Design Agent, Generation Agents, Assembly Agent, and Review Agent roles.
- The framework utilizes a Connector Contract to define assembly relationships during the design stage, effectively bypassing the limited 3D spatial reasoning capabilities of current LLMs and VLMs.
- A cross-stage rollback mechanism and a self-evolving experience store enable the system to isolate design- or code-level failures and continuously improve performance without requiring model fine-tuning.

---

[MAFIG: Multi-agent Driven Formal Instruction Generation Framework](http://arxiv.org/abs/2604.10989)

- MAFIG: introduces a multi-agent framework that decouples scheduling systems into atomic functions to enable rapid, localized repair during emergency situations using a Perception Agent and an Emergency Decision Agent.
- The framework utilizes SFL to distill decision-making capabilities from powerful C-LLMs into lightweight local models, effectively mitigating the Latency-Quality Tradeoff and Execution Misalignment.
- By constraining repairs to affected atomic functions rather than global rescheduling, MAFIG ensures system stability and adaptability in complex, dynamic environments like ports and warehouses.

---

[WebForge: Breaking the Realism-Reproducibility-Scalability Trilemma in Browser Agent Benchmark](http://arxiv.org/abs/2604.10988)

- WebForge: introduces a fully automated four-agent pipeline—Plan Agent, Generation Agent, Refinement Agent, and Validation Agent—that constructs realistic, reproducible, and scalable browser agent benchmarks without human annotation.
- The framework utilizes a seven-dimensional difficulty control framework to systematically profile LLM agent capabilities across navigation, interaction, visual, and reasoning axes.
- WebForge-Bench incorporates anti-cheating mechanisms, including Base64-encrypted data storage and deceptive code generation, to ensure that LLMs complete full multi-step workflows rather than bypassing tasks.

---

[YIELD: A Large-Scale Dataset and Evaluation Framework for Information Elicitation Agents](http://arxiv.org/abs/2604.10968)

- YIELD: introduces a large-scale dataset and evaluation framework for Information Elicitation Agents (IEAs) designed to extract information for institutional objectives.
- The framework formalizes information elicitation as a finite-horizon POMDP and utilizes offline reinforcement learning with AWR to optimize LLM-based agents.
- Experimental results demonstrate that fine-tuning on YIELD improves behavioral alignment with human elicitation patterns compared to prompt-only LLMs.

---

[AgentWebBench: Benchmarking Multi-Agent Coordination in Agentic Web](http://arxiv.org/abs/2604.10938)

- AgentWebBench: introduces a decentralized benchmark for evaluating multi-agent coordination in the Agentic Web, utilizing User Agent, Content Agents, Massive Web Corpus, Agent Interaction, and Final Answer.
- The framework models information access as a decentralized ecosystem where a User Agent coordinates with multiple autonomous Content Agents to synthesize answers from a Massive Web Corpus.
- Experimental results across seven LLMs demonstrate that while decentralized coordination currently trails centralized baselines, performance improves with model scale and iterative planning.

---

[Visible, Trackable, Forkable: Opening the Process of Science](http://arxiv.org/abs/2604.10932)

- Scientific Workflow Framework: introduces a paradigm shift in scientific research by advocating for a workflow that is visible, trackable, and forkable, mirroring the collaborative and iterative processes of software development.
- The framework utilizes Version Control System, Issue Tracker, and Code Review System to ensure that the entire research process, including failed experiments and methodological trade-offs, is documented and accessible.
- By replacing terminal artifacts like static papers with a dynamic, versioned record, the framework enables AI agents to synthesize research histories and facilitates granular, verifiable contributions from the scientific community.

---

[Mem2Evolve: Towards Self-Evolving Agents via Co-Evolutionary Capability Expansion and Experience Distillation](http://arxiv.org/abs/2604.10923)

- Mem2Evolve: introduces a co-evolutionary framework that couples dynamic capability expansion via Asset Memory with strategic experience distillation via Experience Memory.
- The framework operates through a forward-backward loop where agents reuse existing assets or create new ones on demand, subsequently refining these assets and distilling lessons from execution trajectories.
- By integrating experience-guided asset creation, Mem2Evolve improves the stability and reliability of self-evolving agents compared to frameworks that treat capability expansion and experience accumulation in isolation.

---

[HTAA: Enhancing LLM Planning via Hybrid Toolset Agentization &amp; Adaptation](http://arxiv.org/abs/2604.10917)

- HTAA: introduces a hierarchical framework that improves tool-use scalability by abstracting fine-grained tools into agent tools and aligning the planner via asymmetric adaptation.
- The framework utilizes a hybrid toolset consisting of basic tools and agent tools to reduce the planner's action space and mitigate error accumulation during long-horizon tasks.
- Asymmetric planner adaptation employs a two-stage pipeline of backward reconstruction and forward refinement to generate high-quality training trajectories for robust policy optimization.

---

[EvoNash-MARL: A Closed-Loop Multi-Agent Reinforcement Learning Framework for Medium-Horizon Equity Allocation](http://arxiv.org/abs/2604.10911)

- EvoNash-MARL: introduces a closed-loop framework for medium-to-long-horizon equity allocation that integrates Multi-Agent Population, PSRO Meta Strategy, League Best-Response Training, Evolution (PBT-style), Execution Risk Overlay, and Portfolio Decision.
- The framework utilizes a layered policy architecture with direction and risk heads, applying factor neutralization, signal amplification, and feature-quality reweighting to enhance robustness.
- EvoNash-MARL employs a constrained walk-forward validation protocol to prioritize feasibility and stability over raw return, addressing challenges in non-stationary market regimes.

---

[OCCUBENCH: Evaluating AI Agents on Real-World Professional Tasks via Language World Models](http://arxiv.org/abs/2604.10866)

- OCCUBENCH: introduces a benchmark for evaluating AI agents on 100 real-world professional tasks across 10 industry categories using Language World Models (LWMs) to simulate domain-specific environments.
- The framework utilizes an Agent LLM, a Language World Model, a conversation History, and a rubric-based Verifier to assess task completion and environmental robustness under controlled fault injection.
- The research demonstrates that implicit environmental faults are harder for LLMs to handle than explicit errors, and that simulator quality is critical for reliable agent evaluation.

---

[VERITAS: Verifiable Epistemic Reasoning for Image-Derived Hypothesis Testing via Agentic Systems](http://arxiv.org/abs/2604.12144)

- VERITAS: introduces a multi-agent framework that autonomously tests natural-language hypotheses on medical imaging datasets by decomposing the research workflow into four phases handled by role-specialized agents including Principal Investigator-, Imaging Specialist-, Statistician- and Critic-agents.
- The framework ensures full auditability by producing a transparent provenance trail of versioned artifacts, including structured analysis plans, executable Python scripts, and segmentation masks, which are validated by an Evidence Classification Operator to mechanically assign epistemic labels.
- By utilizing role-specialized agents and a constrained API layer, VERITAS enables locally-deployed LLMs to perform complex scientific reasoning and statistical validation that typically requires frontier model scale.

---

[Aethon: A Reference-Based Replication Primitive for Constant-Time Instantiation of Stateful AI Agents](http://arxiv.org/abs/2604.12129)

- Aethon: introduces a reference-based replication primitive that enables near-constant-time instantiation of stateful AI agents by decoupling execution from materialization using Definition Substrate, Reference Substrate, Resolution Substrate, Layered Memory Model, and Copy-on-Write Semantics.
- The framework replaces expensive full-object materialization with lightweight references that point to stable definitions and layered memory, allowing for efficient branching and specialization of LLM-based agents.
- Aethon improves operational scalability and auditability in multi-agent systems by treating lineage as a first-class property and enabling fine-grained control over state inheritance and isolation.

---

[BLAST: Blockchain-based LLM-powered Agentic Spectrum Trading](http://arxiv.org/abs/2604.12127)

- BLAST (Blockchain-based LLM-powered Agentic Spectrum Trading): introduces a decentralized framework that integrates LLM Agents with a permissioned blockchain to enable autonomous, private, and secure spectrum trading.
- The framework utilizes a sequential decision pipeline—comprising Analyst-, Planner-, and Action Executor-agents—to implement the Cognitive Radio cycle for strategic market participation.
- By leveraging Hyperledger Fabric and game-theoretic auction mechanisms, the system ensures privacy through commit-reveal protocols and maximizes social welfare via truthful bidding strategies.

---

[Long-Horizon Plan Execution in Large Tool Spaces through Entropy-Guided Branching](http://arxiv.org/abs/2604.12126)

- EGB (Entropy-Guided Branching): introduces a search algorithm that dynamically expands decision branches where predictive entropy is high to optimize the exploration-exploitation trade-off in long-horizon tool-use tasks.
- The framework utilizes SLATE, a large-scale e-commerce benchmark, to provide rigorous, plan-level evaluation of LLM agents through grounded execution traces and deterministic simulation.
- EGB improves task success rates and computational efficiency by selectively allocating search effort based on localized uncertainty signals rather than exhaustive tree exploration.

---

[Spatial Atlas: Compute-Grounded Reasoning for Spatial-Aware Research Agent Benchmarks](http://arxiv.org/abs/2604.12102)

- Spatial Atlas: introduces a compute-grounded reasoning paradigm that resolves sub-problems via deterministic computation before LLM generation, utilizing A2A Protocol Server, Domain Classifier, FieldWorkArena Handler, MLE-Bench Handler, Spatial Scene Graph Engine, Self-Healing ML Pipeline, Shared Infrastructure, and Entropy-Guided Reasoning Engine.
- The architecture integrates a spatial scene graph engine for reliable spatial reasoning and a self-healing ML pipeline for automated competition-grade code generation.
- The system employs an entropy-guided reasoning engine to optimize cost-efficiency by routing tasks across a three-tier frontier model stack based on estimated uncertainty.

---

[Robust Optimization for Mitigating Reward Hacking with Correlated Proxies](http://arxiv.org/abs/2604.12086)

- Max-Min Policy Optimization: introduces a robust reinforcement learning framework that mitigates reward hacking by optimizing policies against a worst-case adversarial reward constrained by its correlation with a proxy reward.
- The approach utilizes a max-min formulation to identify policies robust to all plausible deviations of the proxy reward within a defined correlation bound.
- A Linear Max-Min variant leverages structural feature information to improve interpretability and tractability by parameterizing reward uncertainty directly in the space of feature weights.

---

[Human-Inspired Context-Selective Multimodal Memory for Social Robots](http://arxiv.org/abs/2604.12081)

- SUMMER (Selectivity Unified Multimodal Memory for Embodied Robots): introduces an end-to-end framework for social robots that utilizes a Perception Layer, Interaction Layer, and Control Layer to perform context-selective, multimodal memory storage and retrieval based on emotional salience and scene novelty.
- The framework employs an Intention Classifier and a Response Generator (VLM) within the Interaction Layer to facilitate personalized, human-like dialogue grounded in retrieved episodic and visual memories.
- SUMMER operates without additional model training, achieving real-time performance by selectively encoding only significant experiences into its Multimodal Memory Database.

---

[Systematic Design of Local Rules for Directing Emergent Structure in Bottom-Up Systems](http://arxiv.org/abs/2604.12057)

- Systematic Design of Local Rules for Directing Emergent Structure in Bottom-Up Systems: introduces a framework for programming decentralized construction by tuning local behavioral rules—comprising build-, movement-, and decision-rules—to achieve targeted global geometric properties.
- The methodology utilizes agent-based modeling with local sensing and grid-based discretization to evaluate emergent area coverage, line density, and front curvature across ensembles of realizations.
- By systematically refining heuristic rules based on statistical analysis, the framework enables robust control over emergent structures despite inherent stochasticity and nonlinearities in the construction process.

---

[REGREACT: Self-Correcting Multi-Agent Pipelines for Structured Regulatory Information Extraction](http://arxiv.org/abs/2604.12054)

- REGREACT: introduces a multi-agent framework that decomposes regulatory information extraction into seven specialized stages, each utilizing an Observe–Diagnose–Repair (ODR) loop to ensure accuracy.
- The framework employs a typed criterion graph for structural validation and criterion-conditioned RAG to resolve external dependencies, producing fully self-contained outputs.
- Evaluation against a GPT-4o baseline demonstrates that REGREACT achieves superior structural and semantic performance by leveraging specialized agents and iterative self-correction.

---

[ORBIT: Guided Agentic Orchestration for Autonomous C-to-Rust Transpilation](http://arxiv.org/abs/2604.12048)

- ORBIT: introduces an autonomous agentic framework for project-level C-to-Rust translation that utilizes C Parser, Dependency Graph, Agentic Iterative Scaffolding, Function Mapper, and Translation Orchestrator to automate code migration.
- The framework employs a multi-agent system including Translator-, Implementation Checker-, Compiler-, Refactor-, Verifier- and Mapping-agents to ensure functional correctness and memory safety.
- ORBIT achieves 100% compilation success and 91.7% test success on large-scale C codebases by replacing static analysis with dynamic, dependency-guided agentic orchestration.

---

[SIR-Bench: Evaluating Investigation Depth in Security Incident Response Agents](http://arxiv.org/abs/2604.12040)

- SIR-Bench: introduces a benchmark for evaluating autonomous security incident response agents by distinguishing genuine forensic investigation from alert parroting.
- The OUAT framework generates realistic attack telemetry in controlled cloud environments to provide expert-validated ground truth for investigation agents.
- The evaluation methodology utilizes an adversarial LLM-as-Judge and ROUGE-L scoring to measure triage accuracy and investigation depth against human analyst baselines.

---

[Memory as Metabolism: A Design for Companion Knowledge Systems](http://arxiv.org/abs/2604.12034)

- Companion Knowledge Systems framework: introduces a normative governance profile for single-user LLM memory systems that balances operational continuity with epistemic correction through a mirror-vs-compensate design principle.
- The architecture utilizes a three-tier storage model—raw buffer, active wiki, and cold memory—to decouple rapid streaming ingestion from scheduled, batched coherence operations.
- The framework employs memory gravity and minority-hypothesis retention to protect load-bearing structures while providing a structural path for contradictory evidence to challenge dominant interpretations.

---

[Identity as Attractor: Geometric Evidence for Persistent Agent Architecture in LLM Activation Space](http://arxiv.org/abs/2604.12016)

- YAR framework: introduces an empirical investigation into whether a structured cognitive_core induces attractor-like geometric structures in the activation space of LLMs.
- The study utilizes mean-pooled hidden states from Llama 3.1 8B Instruct and Gemma 2 9B Instruct to demonstrate that semantically equivalent identity documents converge to tight clusters in activation space.
- The research further explores the behavioral impact of these geometric attractors by injecting a semantic steering vector into the residual stream of the LLMs.

---

[When to Forget: A Memory Governance Primitive](http://arxiv.org/abs/2604.12007)

- MW (Memory Worth): introduces a lightweight, two-counter per-memory signal that tracks the association between memory retrieval and task outcomes to govern memory quality.
- The framework provides a theoretically grounded, convergent estimator for post-retrieval conditional success probability without requiring causal attribution or architectural modifications.
- By maintaining dual scalar counters for successful and failed outcomes, the system enables evidence-aware memory management, including staleness detection, retrieval suppression, and deprecation decisions.

---

[The Long-Horizon Task Mirage? Diagnosing Where and Why Agentic Systems Break](http://arxiv.org/abs/2604.11978)

- HORIZON: introduces a cross-domain diagnostic benchmark for systematically constructing tasks and analyzing long-horizon failure behaviors in LLM-based agents.
- The framework utilizes a seven-category taxonomy to attribute failures across four domains, identifying that planning-related and memory-related failures dominate as task horizons increase.
- The research proposes a trajectory-grounded LLM-as-a-Judge pipeline for scalable failure analysis, demonstrating that model scaling alone is insufficient to resolve long-horizon performance degradation.

---

[M2HRI: An LLM-Driven Multimodal Multi-Agent Framework for Personalized Human-Robot Interaction](http://arxiv.org/abs/2604.11975)

- M2HRI: introduces a multimodal multi-agent framework that integrates Perception Module, Memory Module, Personality Module, Interaction Planner, Execution Module, and Centralized Coordinator to enable personalized and coherent human-robot interaction.
- The framework utilizes LLMs and VLMs to maintain distinct agent identities through personality and long-term memory while ensuring structured group behavior via a centralized coordinator.
- Experimental results demonstrate that agent individuality and centralized coordination significantly enhance interaction quality, distinguishability, and conversational flow in multi-robot scenarios.

---

[Narrative-Driven Paper-to-Slide Generation via ArcDeck](http://arxiv.org/abs/2604.11969)

- ArcDeck: introduces a multi-agent framework that formulates paper-to-slide generation as a structured narrative reconstruction task using Discourse Parser, Commitment Builder, Narrative Refinement Loop, Slide Deck Constructor, and Aesthetics Refiner.
- The framework utilizes RST-based discourse parsing to model logical flow and a global commitment mechanism to maintain narrative consistency across the generated slide sequence.
- ArcDeck includes a closed-loop refinement process and is evaluated using ArcBench, a curated dataset of 100 high-quality paper-slide pairs from top-tier AI conferences.

---

[Agentic LLM Reasoning in a Self-Driving Laboratory for Air-Sensitive Lithium Halide Spinel Conductors](http://arxiv.org/abs/2604.11957)

- A-Lab GPSS: introduces an autonomous robotic platform for air-sensitive solid-state synthesis, integrating an agentic AI framework that employs abductive and inductive reasoning to optimize material discovery.
- The system utilizes an abnormality-detection agent for hypothesis-driven re-exploration and pattern-finding agents for data-driven expansion of the chemical search space.
- By structuring LLM agents into complementary reasoning modes, the platform achieved a fourfold increase in the success rate of identifying high-conductivity, phase-pure lithium halide spinel conductors.

---

[Dynamic Multi-Robot Task Allocation under Uncertainty and Communication Constraints: A Game-Theoretic Approach](http://arxiv.org/abs/2604.11954)

- IBR (Iterative Best Response): introduces a decentralized framework for dynamic multi-robot task allocation that manages incomplete information through hub-based sensing regions and inter-hub communication graphs.
- The framework enables agents to select tasks by maximizing their marginal contribution to locally observed welfare, effectively handling stochastic task completion and time-window constraints.
- Empirical evaluations demonstrate that IBR achieves competitive task-completion performance compared to centralized methods while maintaining high computational efficiency under sparse communication.

---

[AnyPoC: Universal Proof-of-Concept Test Generation for Scalable LLM-Based Bug Detection](http://arxiv.org/abs/2604.11950)

- AnyPoC: introduces a multi-agent framework that automates the generation of executable proof-of-concept tests to validate bug reports, utilizing an Analyzer agent, Generator agent, Checker agent, and a self-evolving Knowledge Base.
- The framework mitigates LLM hallucination and reward hacking by employing a dedicated Checker agent that independently re-executes generated tests in a fresh environment to verify bug existence.
- AnyPoC improves scalability and generality by maintaining a self-evolving Knowledge Base that allows agents to accumulate and reuse project-specific insights across different software systems.

---

[AutoSurrogate: An LLM-Driven Multi-Agent Framework for Autonomous Construction of Deep Learning Surrogate Models in Subsurface Flow](http://arxiv.org/abs/2604.11945)

- AutoSurrogate: introduces an LLM-driven multi-agent framework that automates the construction, training, and deployment of deep learning surrogate models for subsurface flow simulations.
- The framework utilizes an LLM Orchestrator to coordinate specialized agents—Data Analysis Agent, Model Selection Agent, HPO & Training Agent, and Reporter Agent—that collaborate via a Shared Memory & Context to perform end-to-end surrogate modeling.
- It incorporates a closed-loop self-correction mechanism that autonomously handles training instabilities and performance plateaus through continuation, stability-constrained restarts, and architecture switching.

---

[ProbeLogits: Kernel-Level LLM Inference Primitives for AI-Native Operating Systems](http://arxiv.org/abs/2604.11943)

- ProbeLogits: introduces a kernel-level inference primitive that classifies agent actions by reading LLM logit distributions directly, bypassing text generation and parsing overhead.
- Anima OS integrates these primitives to provide structural governance, treating KV cache as process state for checkpointing, restoring, and forking agent contexts.
- The architecture enforces safety by executing governance checks below the WASM sandbox boundary, ensuring that agents cannot bypass security policies.

---

[Self-Monitoring Benefits from Structural Integration: Lessons from Metacognition in Continuous-Time Multi-Timescale Agents](http://arxiv.org/abs/2604.11914)

- Self-Monitoring Multi-Timescale Agent: introduces a continuous-time reinforcement learning architecture that evaluates the functional utility of self-monitoring modules when integrated as auxiliary-loss add-ons versus structural components on the decision-making pathway.
- The research demonstrates that auxiliary-loss implementations of metacognition, temporal self-modeling, and subjective duration fail because agents learn to ignore these redundant, non-critical signals.
- Structural integration, where monitoring signals directly gate exploration, trigger workspace broadcasts, and inform policy inputs, provides a medium-large performance improvement by ensuring modules become load-bearing components of the decision pathway.

---

[From Plan to Action: How Well Do Agents Follow the Plan?](http://arxiv.org/abs/2604.12147)

- SWE-agent: introduces a systematic, process-centric evaluation of plan compliance in LLM-based agents using Navigation, Reproduction, Patch, Validation, Graphectory, Langutory, Plan Compliance Metric, and Phase Flow Analysis.
- The research analyzes 16,991 trajectories across four LLMs to demonstrate that while instructed plans improve task success, agents often deviate due to internalized training workflows and context window limitations.
- The study reveals that plan augmentation is only effective when aligned with a model's internal strategy, and that periodic plan reminders can mitigate plan violations in long-horizon software engineering tasks.

---

[Learning Probabilistic Responsibility Allocations for Multi-Agent Interactions](http://arxiv.org/abs/2604.13128)

- Responsibility CVAE: introduces a probabilistic framework for learning responsibility allocations in multi-agent interactions by leveraging a conditional variational autoencoder and a differentiable optimization layer.
- The framework utilizes a transformer-based architecture to process variable-length sequences of agent states and predict responsibility distributions that account for multimodal uncertainty.
- By integrating a responsibility-aware safety filter, the model ensures that predicted agent controls satisfy shared safety constraints while reflecting learned social interaction patterns.

---

[A longitudinal health agent framework](http://arxiv.org/abs/2604.12019)

- Longitudinal Health Agent Framework: introduces a multi-layer design architecture for LLM-based health agents to support sustained engagement across multiple sessions by integrating Coherence, Continuity, Adaptation, and Agency.
- The framework shifts health agent design from episodic, reactive interactions toward structured, longitudinal stewardship of user goals, interpretations, and responsibilities.
- It provides a conceptual foundation for developing agents that maintain consistent reasoning, track evolving health trajectories, and dynamically calibrate their level of initiative based on user needs and clinical context.

---

[Collaborative Multi-Agent Scripts Generation for Enhancing Imperfect-Information Reasoning in Murder Mystery Games](http://arxiv.org/abs/2604.11741)

- Collaborative Multi-Agent Script Synthesis Framework: introduces a multi-agent system that leverages OutlineAgent, CharacterAgent, CriticAgent, ClueAgent, RoleplayAgent, QaAgent, and ScoreAgent to synthesize high-quality, role-driven game scripts for training VLMs under imperfect information.
- The framework employs a two-stage training strategy, utilizing supervised fine-tuning on synthetic data followed by GRPO-based reinforcement learning monitored by the ScoreAgent to enhance reasoning and role-playing capabilities in VLMs.
- Experimental results demonstrate that the proposed method significantly improves performance on multi-hop multimodal reasoning and deception-resilient understanding across different VLM scales.

---

[Agentic Driving Coach: Robustness and Determinism of Agentic AI-Powered Human-in-the-Loop Cyber-Physical Systems](http://arxiv.org/abs/2604.11705)

- LF: introduces a deterministic modeling approach for agentic AI-powered HITL CPS using the reactor model of computation to reconcile intrinsic nondeterminism in LLMs and human behavior.
- The framework utilizes Coach, Driver, Environment, and Car components to simulate interactions, employing LLMInference and Planner to generate safety-critical driving instructions.
- By integrating a deadline handler and logical delays, the system ensures repeatable, deterministic behavior even when LLM inference latencies fluctuate during dynamic driving scenarios.

---

[Olfactory pursuit: catching a moving odor source in complex flows](http://arxiv.org/abs/2604.13121)

- Hybrid Heuristic Policy: introduces a composite strategy for olfactory pursuit that balances information-theoretic exploration with greedy predictive interception to track moving targets.
- The framework utilizes a belief map to maintain a joint probability distribution over the target's position and velocity, enabling the agent to anticipate future trajectories.
- By dynamically weighting Infotaxis and greedy components, the agent effectively navigates the "blind spot" where target and predator speeds are comparable, outperforming purely exploratory strategies.

---

[PAC-BENCH: Evaluating Multi-Agent Collaboration under Privacy Constraints](http://arxiv.org/abs/2604.11523)

- PAC-BENCH: introduces a benchmark for systematically evaluating multi-agent collaboration under explicit privacy constraints, utilizing Private Agent, Memory, Privacy Constraints, Shared Goal, and an Evaluation Module.
- The framework employs Task Evaluation, Privacy Evaluation, and Holistic Evaluation to quantify the performance gap between collaborative success and privacy preservation in multi-agent systems.
- Experimental results reveal that privacy constraints significantly degrade collaboration performance, with initiating agents often dominating interaction dynamics and recurring failure modes including early-stage privacy violations, over-conservative abstraction, and privacy-induced hallucinations.

---

[AgentForge: Execution-Grounded Multi-Agent LLM Framework for Autonomous Software Engineering](http://arxiv.org/abs/2604.13120)

- AgentForge: introduces a multi-agent framework that enforces execution-grounded verification for autonomous software engineering by coordinating Planner, Coder, Tester, Debugger, and Critic agents.
- The framework utilizes a dual-memory system comprising episodic memory and a live repository index to provide grounded context for LLM-based agents.
- Every generated code change is validated within a resource-constrained, network-isolated Docker sandbox to ensure functional correctness through real execution feedback.

---

[Hardening x402: PII-Safe Agentic Payments via Pre-Execution Metadata Filtering](http://arxiv.org/abs/2604.11430)

- HardenedX402Client: introduces a pre-execution security middleware for x402 payments that utilizes PIIFilter, PolicyEngine, ReplayGuard, and AuditLog to protect agentic transactions from data leakage and financial risks.
- The framework intercepts payment requests to perform PII redaction and policy enforcement before the x402 Protocol and Facilitator process the transaction.
- Experimental results demonstrate that using an NLP-based PIIFilter achieves high precision and recall for sensitive metadata while maintaining latency well within the required 50 ms overhead budget.

---

[Beyond RAG for Cyber Threat Intelligence: A Systematic Evaluation of Graph-Based and Agentic Retrieval](http://arxiv.org/abs/2604.11419)

- Beyond RAG for CTI: introduces a systematic evaluation of four retrieval architectures—RAG, GRAG, AGRAG, and HRAG—to assess their effectiveness, robustness, and reliability in cyber threat intelligence question answering.
- The study utilizes a dataset of 3,300 CTI question-answer pairs to compare standard vector-based retrieval against graph-grounded and agentic approaches, identifying critical failure modes like structural hallucination and latency instability.
- Results demonstrate that while graph grounding improves factual reasoning, it requires agentic correction or hybrid retrieval to mitigate catastrophic failures and ensure reliable performance in high-stakes CTI environments.

---

[BankerToolBench: Evaluating AI Agents in End-to-End Investment Banking Workflows](http://arxiv.org/abs/2604.11304)

- BTB (BankerToolBench): introduces a high-fidelity benchmark for evaluating LLM agents on end-to-end investment banking workflows, utilizing a Task, Prompt, Data Room, RL Environment, Deliverables, Rubric, Verifier, and Agent Harness.
- The framework employs an agentic verifier, Gandalf, to perform automated, rubric-based grading of complex multi-file deliverables, ensuring alignment with professional standards.
- Empirical evaluation of nine frontier models reveals that current LLMs struggle with end-to-end execution, frequently failing to maintain cross-artifact consistency and analytical accuracy in high-stakes financial tasks.

---

#### 12th April 2026

[RemoteAgent: Bridging Vague Human Intents and Earth Observation with RL-based Agentic MLLMs](http://arxiv.org/abs/2604.07765)

- RemoteAgent: introduces an agentic framework that uses reinforcement fine-tuning to align an MLLM as a cognitive core for resolving macroscopic tasks while intelligently routing dense predictions to specialized tools via the Model Context Protocol.
- The framework utilizes the VagueEO dataset to train the MLLM on ambiguous, human-centric queries, enabling it to distinguish between tasks suitable for internal resolution and those requiring external tool execution.
- By decoupling intent understanding from precision-critical spatial execution, RemoteAgent achieves high data efficiency and significant inference speedups compared to traditional multi-step agentic workflows.

---

[EmoMAS: Emotion-Aware Multi-Agent System for High-Stakes Edge-Deployable Negotiation with Bayesian Orchestration](http://arxiv.org/abs/2604.07003)

- EmoMAS (Emotion-Aware Multi-Agent System): introduces a Bayesian multi-agent framework that optimizes emotional trajectories in high-stakes negotiations by coordinating a Bayesian Orchestrator Agent, Game Theory Agent, Reinforcement Learning Agent, and Emotional Coherence Agent.
- The framework enables online strategy learning without pre-training by dynamically weighting specialized agent predictions through real-time Bayesian inference to adapt to unique negotiation contexts.
- Extensive simulations across debt, healthcare, emergency, and educational domains demonstrate that EmoMAS-equipped LLMs and SLMs achieve superior negotiation performance and ethical behavior compared to baseline models.

---

[From Perception to Autonomous Computational Modeling: A Multi-Agent Approach](http://arxiv.org/abs/2604.06788)

- Multi-Agent Computational Modeling Framework: introduces a solver-agnostic architecture where coordinated LLM agents autonomously execute computational mechanics workflows from perceptual data to engineering reports.
- The framework utilizes a three-layer structure comprising perception-, analysis- and assessment-agents, coordinated by an orchestrator and refined through quality gates and an agent evolver.
- This approach enables autonomous simulation by inferring engineering parameters from images, applying task-dependent conservatism, and facilitating human-in-the-loop professional oversight.

---

[PFAgent: A Tractable and Self-Evolving Power-Flow Agent for Interactive Grid Analysis](http://arxiv.org/abs/2604.10846)

- PFAgent: introduces a modular agentic architecture that automates power-system simulation workflows by integrating intent parsing, knowledge retrieval, tool execution, and structured reporting.
- The framework incorporates an AI-assisted fixing loop and a self-evolution mechanism to iteratively refine agent performance through failure attribution and constraint updates without requiring LLM retraining.
- Experimental results on IEEE benchmark systems demonstrate that retrieval-augmented grounding and self-evolution enable reliable, multi-turn power-flow analysis with high reproducibility and machine-checkable outputs.

---


[PEMANT: Persona-Enriched Multi-Agent Negotiation for Travel](http://arxiv.org/abs/2604.10475)

- PEMANT: introduces a two-stage LLM-based framework that integrates behavioral theory and multi-agent negotiation to model household-level trip generation.
- The framework utilizes a Translation Map, Narrative Synthesis, and an HA-CoPB Reasoning Engine to create theory-grounded personas that inform a structured multi-agent conversation.
- PEMANT employs a Moderator, Persona SFT, and Dialogue SFT to ensure agent alignment, constraint satisfaction, and realistic collective decision-making during the Parallel Proposal and Consensus Refinement phases.

---

[AdverMCTS: Combating Pseudo-Correctness in Code Generation via Adversarial Monte Carlo Tree Search](http://arxiv.org/abs/2604.10449)

- AdverMCTS: introduces a dual-agent framework that mitigates pseudo-correctness in code generation by framing the process as a minimax game between a Solver MCTS and an Attacker MCTS.
- The framework utilizes an Attacker MCTS to discover failure-inducing test cases, which are then stored in a Global Test Filter to serve as hard constraints for the Solver.
- By iteratively co-evolving code candidates and adversarial tests, the system forces the Solver to generalize beyond sparse public test cases, significantly improving robust correctness.

---


[Resilient Write: A Six-Layer Durable Write Surface for LLM Coding Agents](http://arxiv.org/abs/2604.10842)

- Resilient Write: introduces a six-layer durable write surface that interposes between LLM coding agents and the filesystem to mitigate common failure modes like silent rejections, truncation, and session loss.
- The framework utilizes L0 Risk Score, L1 Safe Write, L2 Chunk Compose, L3 Error Envelopes, L4 Scratchpad, and L5 Handoff to provide atomic, auditable, and recoverable file operations for autonomous agents.
- Quantitative evaluation demonstrates that the system achieves a 5x reduction in recovery time and a 13x improvement in agent self-correction rates compared to naive baselines.

---

[Robust Information Design with Heterogeneous Beliefs in Bayesian Congestion Games](http://arxiv.org/abs/2604.10831)

- Robust Information Design Framework: introduces a methodology for designing signaling policies in Bayesian congestion games that remain obedient under heterogeneous user beliefs within a defined neighborhood of a nominal prior.
- The framework characterizes policy-level robustness radii and identifies regimes where robust obedient signaling is feasible despite belief misspecification.
- It further analyzes the robustness-performance tradeoff by establishing the monotonicity of the robust value function and deriving local sensitivity bounds based on active obedience constraints.

---

[CheeseBench: Evaluating Large Language Models on Rodent Behavioral Neuroscience Paradigms](http://arxiv.org/abs/2604.10825)

- CheeseBench: introduces a benchmark suite of nine behavioral neuroscience paradigms to evaluate LLMs on procedural discovery and spatial navigation tasks.
- The framework utilizes a unified zero-shot ASCII protocol to test agent performance against approximate biological reference values derived from rodent literature.
- Experimental results demonstrate that while LLMs excel at simple conditioning, they struggle with sustained state tracking and spatial navigation, highlighting a significant performance gap compared to biological baselines.

---

[MeloTune: On-Device Arousal Learning and Peer-to-Peer Mood Coupling for Proactive Music Curation](http://arxiv.org/abs/2604.10815)

- MeloTune: introduces a two-layer continuous-time architecture for on-device music curation that utilizes a private Listener-level CfC for individual trajectory prediction and a shared Mesh-runtime CfC for co-listening coherence.
- The framework employs Cognitive Memory Blocks (CMBs) for structured inter-agent communication, evaluated by Symbolic-Vector Attention Fusion (SVAF) to ensure drift-bounded coupling while maintaining privacy by keeping hidden states local.
- Personalization is achieved through a Personal Arousal Function (PAF) that learns per-listener adjustments from behavioral signals, while an Emotional Resolution Engine (ERE) with isolation windows prevents echo loops in multi-agent mesh environments.

---

[PokeRL: Reinforcement Learning for Pokémon Red](http://arxiv.org/abs/2604.10812)

- PokeRL: introduces a modular reinforcement learning system designed to overcome sparse rewards and pathological behaviors in long-horizon JRPG environments using PyBoy emulator, Gymnasium environment, Actor-Critic Network, Memory Reader, Anti-Loop System, Anti-Spam Mechanism, and Curriculum Controller.
- The framework utilizes a per-map visited mask as a spatial memory channel to improve exploration efficiency and mitigate the need for recurrent neural networks.
- By implementing hierarchical reward structures and explicit failure-mode mitigation, the system enables PPO agents to reliably complete early-game objectives in Pokémon Red.

---

[Verify Before You Fix: Agentic Execution Grounding for Trustworthy Cross-Language Code Analysis](http://arxiv.org/abs/2604.10800)

- Unified cross-language vulnerability lifecycle framework: introduces a three-stage closed-loop system that integrates Fusion Detector, Validation Agent, and Remediation Module to ensure all repair actions are grounded in execution-verified evidence.
- The framework utilizes a uAST for cross-language normalization and a two-way gating mechanism to fuse GraphSAGE structural embeddings with Qwen2.5-Coder-1.5B semantic representations for interpretable vulnerability detection.
- By enforcing execution-grounded validation within a Docker Sandbox, the system significantly reduces false positives and unnecessary repairs, providing a principled approach for trustworthy LLM-driven agentic pipelines.

---

[VulWeaver: Weaving Broken Semantics for Grounded Vulnerability Detection](http://arxiv.org/abs/2604.10767)

- VulWeaver: introduces a neuro-symbolic approach that constructs an enhanced Unified Dependency Graph (UDG) to provide a reliable foundation for holistic vulnerability context extraction and grounded LLM reasoning.
- The framework integrates explicit context from program slicing with implicit context resolution to capture complete program semantics, mitigating inaccuracies in static analysis.
- VulWeaver employs meta-prompting with vulnerability-specific expert guidelines and majority voting to steer LLMs toward systematic, evidence-driven vulnerability detection.

---

[Prosociality by Coupling, Not Mere Observation: Homeostatic Sharing in an Inspectable Recurrent Artificial Life Agent](http://arxiv.org/abs/2604.10760)

- ReCoN-Ipsundrum: introduces a minimal mechanistic architecture to decompose the distinction between observing a partner's need and being dynamically affected by it through homeostatic coupling.
- The framework utilizes a self-directed planner that evaluates actions based on internal homeostatic error, where prosocial behavior emerges only when partner distress is routed into the agent's own regulatory system.
- Experimental results across toy environments demonstrate that mere partner-state access is insufficient for helping, while affective coupling enables prosociality subject to ecological metabolic constraints.

---

[Deep-Reporter: Deep Research for Grounded Multimodal Long-Form Generation](http://arxiv.org/abs/2604.10741)

- DEEP-REPORTER: introduces a unified agentic framework that orchestrates multimodal retrieval, relevance-aware filtering, and checklist-guided incremental synthesis to generate grounded long-form reports.
- The framework utilizes a Planner Agent, a collaborative Searcher-Filter, and a Reporter Agent to maintain global coherence and image-text consistency through recurrent context management.
- The authors also introduce M2LONGBENCH, a comprehensive testbed with a stable multimodal sandbox, to enable rigorous and reproducible evaluation of multimodal long-form generation.

---

[RCBSF: A Multi-Agent Framework for Automated Contract Revision via Stackelberg Game](http://arxiv.org/abs/2604.10740)

- RCBSF (Risk-Constrained Bilevel Stackelberg Framework): introduces a game-theoretic multi-agent framework that models contract revision as a hierarchical Stackelberg game between a Global Prescriptive Agent (GPA), a Constrained Revision Agent (CRA), and a Local Verification Agent (LVA).
- The framework utilizes a 5-dimensional risk taxonomy to impose strict constraints on the follower system, ensuring that LLMs iteratively optimize contract output toward a risk-minimized equilibrium.
- Empirical results demonstrate that RCBSF achieves state-of-the-art performance in risk resolution and token efficiency by decoupling strategic auditing from generative execution.

---

[Too Nice to Tell the Truth: Quantifying Agreeableness-Driven Sycophancy in Role-Playing Language Models](http://arxiv.org/abs/2604.10733)

- Too Nice to Tell the Truth: introduces a systematic investigation into how persona agreeableness influences sycophancy in LLMs, utilizing a Persona Generator, NEO-IPIP Agreeableness Questionnaire, and a Sycophancy Benchmark.
- The research employs a Stance Detection Module and Statistical Analysis Framework to evaluate 13 open-weight LLMs, revealing that high-agreeableness personas significantly correlate with increased sycophancy in 9 of 13 models.
- The study introduces the Trait-Truthfulness Gap (TTG) metric to identify a "zone of deception" where persona adoption leads models to sacrifice factual accuracy for social harmony.

---

[VCC-DSA: A Novel Vascular Consistency Constrained DSA Imaging Model for Motion Artifact Suppression](http://arxiv.org/abs/2604.10700)

- VCC-DSA: introduces a framework for robust motion artifact suppression in DSA imaging by leveraging mask-based background information and consistency constraints.
- The model utilizes an RDB-based network with details-shortcut to preserve vascular structures while employing a Mixup-based Data Self-evolution Strategy to iteratively refine training data.
- VCC-DSA eliminates the reliance on artifact-free learning targets by enforcing structural consistency across different mask-live image pairs during training.

---

[Skill-SD: Skill-Conditioned Self-Distillation for Multi-turn LLM Agents](http://arxiv.org/abs/2604.10674)

- Skill-SD: introduces a framework that improves LLM agent training by distilling dynamic, trajectory-derived natural-language skills into a student model while maintaining on-policy rollouts.
- The framework utilizes a skill-conditioned teacher that co-evolves with the student, providing dense token-level supervision to stabilize and accelerate reinforcement learning in long-horizon tasks.
- By decoupling skill-based guidance from the student's inference-time prompt, Skill-SD avoids retrieval dependency while effectively addressing sparse reward signals through importance-weighted self-distillation.

---

[Governed Reasoning for Institutional AI](http://arxiv.org/abs/2604.10658)

- Cognitive Core: introduces a governed decision substrate for institutional AI that enforces structural guardrails and auditability through typed epistemic operations.
- The framework utilizes a four-tier governance model and a metacognitive reflect primitive to ensure that LLMs operate within defined authority boundaries and produce verifiable reasoning traces.
- By replacing monolithic reasoning loops with demand-driven delegation and endogenous audit ledgers, the architecture significantly reduces silent errors in high-stakes institutional decision-making.

---

[Rethinking Software Engineering for Agentic AI Systems](http://arxiv.org/abs/2604.10599)

- Conceptual Framework for Agentic Software Engineering: introduces a paradigm shift in software engineering from manual code authorship to a model centered on orchestration, verification, and human-AI collaboration.
- The framework identifies four core competencies—Intent Articulation, Systematic Verification, Multi-Agent Orchestration, and Human Judgment and Accountability—as essential for managing the abundance of AI-generated code.
- This paper synthesizes literature to propose a transformation roadmap for engineering education, tooling, lifecycle processes, and professional governance in the era of agentic AI.

---

[AWARE: Adaptive Whole-body Active Rotating Control for Enhanced LiDAR-Inertial Odometry under Human-in-the-Loop Interaction](http://arxiv.org/abs/2604.10598)

- AWARE: introduces a bio-inspired whole-body active yawing framework that enhances LiDAR-Inertial Odometry (LIO) on resource-constrained UAVs by integrating a Panoramic Observability Predictor, a Hybrid RL-MPC Controller, and a Safe Flight Corridor (SFC) Mechanism.
- The framework utilizes an Actor-Critic RL meta-controller to dynamically adjust MPC cost weights based on real-time environmental geometry, effectively balancing localization accuracy with flight stability.
- By decoupling human navigational intent from autonomous yaw optimization, the system enables safe, cooperative control in complex, geometrically degenerate environments without requiring additional mechanical actuation.

---

[GeoMeld: Toward Semantically Grounded Foundation Models for Remote Sensing](http://arxiv.org/abs/2604.10591)

- GeoMeld-FM: introduces a pretraining framework that integrates multi-pretext masked autoencoding, JEPA-based predictive learning, and caption-vision contrastive alignment to learn semantically grounded representations for remote sensing.
- The framework utilizes an agentic captioning pipeline comprising an Orchestrator Agent, Captioner Agent, Evaluator Agent, and Verification Agent to synthesize and verify semantically rich annotations from heterogeneous geospatial data.
- By training on the GeoMeld dataset, the model leverages ConvNeXt V2 Encoder, MP-MAE Decoders, and a JEPA Branch to achieve robust cross-sensor physical consistency and grounded semantics in downstream remote sensing tasks.

---

[Towards Schema-based Learning from a Category-Theoretic Perspective](http://arxiv.org/abs/2604.10589)

- SBL (Schema-Based Learning): introduces a hierarchical categorical framework for organizing agent cognition through interconnected levels of schemas, workflows, minds, and agents.
- The framework utilizes Schsyn, Schimpl, and Schsem to separate syntactic representation, concrete implementation, and probabilistic semantics, while OSch provides a duoidal structure for composing cognitive workflows.
- The Mind category integrates memory subsystems, cognitive modules, and mental states, enabling structured transformations that bridge abstract schema manipulation with goal-directed agent behavior.

---

[The Blind Spot of Agent Safety: How Benign User Instructions Expose Critical Vulnerabilities in Computer-Use Agents](http://arxiv.org/abs/2604.10577)

- OS-BLIND: introduces a benchmark evaluating computer-use agents under unintended attack conditions where user instructions are benign but harm emerges from the environment.
- The framework reveals that safety-aligned models often fail because safety mechanisms primarily activate during initial steps and rarely re-engage during subsequent execution.
- Task decomposition in multi-agent systems paradoxically degrades safety by obscuring harmful intent, causing models to execute malicious actions they would otherwise refuse.

---

[Aerial IRS Deployment-Aided Secure Computation Offloading Against DISCO Jamming Attacks](http://arxiv.org/abs/2604.10558)

- DDADSO (Dual-agent DRL-based AIRS deployment-aided secure computation offloading): introduces a two-timescale framework that jointly optimizes AIRS deployment, offloading ratios, and phase shifts to mitigate DISCO jamming attacks in MEC systems.
- The framework utilizes a DQN-based agent for long-timescale AIRS deployment and a TD3-VAE-based agent for short-timescale dynamic offloading and phase shift adjustments.
- The VAE module compresses high-dimensional state information to reduce computational complexity and improve learning stability for the TD3-based agent.

---

[Agent2 RL-Bench: Can LLM Agents Engineer Agentic RL Post-Training?](http://arxiv.org/abs/2604.10547)

- Agent2 RL-Bench: introduces a benchmark for evaluating LLM agents on their ability to autonomously design, implement, and execute RL post-training pipelines across three levels of structural complexity.
- The framework utilizes an isolated workspace, runtime instrumentation, and a grading API to diagnose how agents manage environment interaction, trajectory collection, and reward-driven optimization.
- Experimental results across multiple agent stacks reveal that while agents achieve significant gains on interactive tasks, they frequently default to supervised fine-tuning pipelines, highlighting a gap between writing RL code and successfully running stable RL systems.

---

[Enhanced Self-Learning with Epistemologically-Informed LLM Dialogue](http://arxiv.org/abs/2604.10545)

- CausaDisco (Causal Discovery): introduces an interactive system that integrates Aristotle’s Four Causes framework into LLM prompts to enhance cognitive support for self-learning, utilizing Embedded Content View, Concept Graph View, Q&A Conversation View, Tree Map View, Four Causes Principles, LLM Chatbot, and User Logging Storage.
- The system addresses challenges in learner-LLM interactions by automatically generating epistemologically-informed follow-up questions to encourage deeper exploration and manage cognitive load.
- A controlled study demonstrated that CausaDisco fosters more engaging interactions, inspires sophisticated exploration, and facilitates multifaceted perspectives compared to baseline LLM tools.

---

[VLN-NF: Feasibility-Aware Vision-and-Language Navigation with False-Premise Instructions](http://arxiv.org/abs/2604.10533)

- ROAM (Room-Object Aware Movement): introduces a two-stage hybrid framework for evidence-grounded navigation that combines a Room-Level Navigator for localization with an In-room Explorer for target verification.
- The framework integrates a FREE (Free-space Raycasting Estimation Engine) to provide geometric clearance cues, enabling the LLM-agent to prioritize exploration toward larger unsearched regions.
- ROAM utilizes a VLM stack to convert visual observations into semantic context, allowing the In-room Explorer to make informed FOUND or NOT-FOUND decisions under partial observability.

---

[Structure-Grounded Knowledge Retrieval via Code Dependencies for Multi-Step Data Reasoning](http://arxiv.org/abs/2604.10516)

- SGKR (Structure-Grounded Knowledge Retrieval): introduces a retrieval framework that organizes domain knowledge as a code dependency graph to improve multi-step data analysis tasks.
- The framework utilizes semantic I/O tags and breadth-first search to identify and retrieve task-relevant subgraphs from function-call dependencies.
- By grounding knowledge in executable code structure rather than lexical similarity, SGKR provides more precise context for LLMs to perform complex reasoning.

---

[Agent Mentor: Framing Agent Knowledge through Semantic Trajectory Analysis](http://arxiv.org/abs/2604.10513)

- AMAP: introduces a trace-grounded mentoring pipeline that aggregates execution trajectories to identify semantic features correlated with undesired agent behavior and injects corrective statements into system prompts.
- The framework utilizes an Observability SDK to map runtime telemetry to design-time artifacts, enabling closed-loop specification maintenance for LLM-based agents.
- By applying process mining and semantic clustering to execution logs, the system automatically derives targeted instructions to mitigate ambiguity and improve agent robustness across diverse configurations.

---

[SWE-Shepherd: Advancing PRMs for Reinforcing Code Agents](http://arxiv.org/abs/2604.10493)

- SWE-Shepherd: introduces a framework that utilizes Process Reward Models to provide dense, step-level supervision for LLM-based agents in software engineering tasks.
- The framework constructs an action-level reward dataset from agent trajectories and trains a lightweight PRM to score intermediate steps, enabling reward-guided search without full reinforcement learning.
- Experimental results on SWE-Bench Verified demonstrate that PRM guidance improves interaction efficiency and reduces the number of steps required to reach a solution.

---

[CovAngelo: A hybrid quantum-classical computing platform for accurate and scalable drug discovery](http://arxiv.org/abs/2604.10487)

- CovAngelo: introduces a multiscale computational platform that integrates molecular dynamics with a novel ECC-DMET embedding model to enable accurate and scalable modeling of chemical reactions in complex protein-ligand environments.
- The framework utilizes quantum-information-optimized orbitals to systematically define active quantum regions, significantly reducing the dimensionality of electronic structure problems while maintaining high physical fidelity.
- The platform supports diverse computational backends, including classical CPU/GPU clusters and emerging quantum hardware, and provides a path toward fault-tolerant quantum computing through symmetry-optimized double factorization.

---

[Tracing the Roots: A Multi-Agent Framework for Uncovering Data Lineage in Post-Training LLMs](http://arxiv.org/abs/2604.10480)

- Multi-Agent Data Lineage Framework: introduces an automated multi-agent system that reconstructs evolutionary graphs of post-training datasets by coordinating Sourcing Agent, Extracting Agents, Tracing Agents, and Aggregation Agent to mine unstructured documentation.
- The framework utilizes a Processing Queue and Canonicalization Module to transform isolated datasets into a comprehensive lineage graph, enabling the identification of structural redundancy and benchmark contamination propagation.
- By leveraging provenance-based sampling and topological analysis, the research provides a systematic approach to improve dataset diversity and mitigate latent structural risks in LLM training corpora.

---

[From Query to Counsel: Structured Reasoning with a Multi-Agent Framework and Dataset for Legal Consultation](http://arxiv.org/abs/2604.10470)

- JURISMA: introduces a modular multi-agent framework that decomposes complex legal queries into structured element graphs to enable context-aware reasoning and iterative refinement.
- The framework utilizes a centralized Manager Agent to dynamically coordinate specialized agents, including Element Agent, Draft Agent, FormatCheck Agent, LawSearch Agent, and ContentCheck Agent, for high-precision legal consultation.
- The system is trained on JURISCQAD, a large-scale dataset of 43,000+ expert-validated legal consultation triplets, significantly improving performance over existing LLMs in legal reasoning and statutory grounding.

---

[A Benchmark and Multi-Agent System for Instruction-driven Cinematic Video Compilation](http://arxiv.org/abs/2604.10456)

- CineAgents: introduces a multi-agent system that reformulates cinematic video compilation into a "design-and-compose" paradigm to overcome contextual collapse and temporal fragmentation.
- The framework utilizes script reverse-engineering to construct a hierarchical narrative memory and employs iterative narrative planning between director- and orchestrator-agents to ensure logical coherence.
- The system includes a manager agent to supervise the workflow and an editor agent that leverages external tools to assemble the final compiled video based on user instructions.

---

[Tradeoffs in Privacy, Welfare, and Fairness for Facility Location](http://arxiv.org/abs/2604.10443)

- DPExpMedα: introduces a differentially private facility location mechanism that simultaneously optimizes for social welfare, privacy, and fairness using a widened percentile loss function.
- The framework utilizes CTM and SPMλ dataset families to demonstrate that privacy, social welfare, and fairness can coexist under natural data distributions.
- The paper establishes tight upper and lower bounds for the proposed mechanism, proving it is near-optimal for both social welfare difference (SWDIFF) and maximum individual utility loss (FAIR).

---

[ReContraster: Making Your Posters Stand Out with Regional Contrast](http://arxiv.org/abs/2604.10442)

- ReContraster: introduces a training-free compositional multi-agent system that leverages regional contrast to generate visually striking and aesthetically harmonious posters.
- The framework utilizes a cognition agent, an arranger agent, and a refiner agent to iteratively design poster layouts based on user-provided themes and region masks.
- A hybrid denoising strategy, incorporating gradient consistency loss and joint regional denoising, ensures seamless transitions across region boundaries during the diffusion process.

---

[CARE-ECG: Causal Agent-based Reasoning for Explainable and Counterfactual ECG Interpretation](http://arxiv.org/abs/2604.10420)

- CARE-ECG: introduces a causally structured framework that unifies representation learning, probabilistic diagnosis, and grounded explanation generation for ECG interpretation.
- The framework utilizes a Physiological Stream for causal inference over latent biomarkers and an Agentic Stream for retrieval-augmented generation with faithfulness verification.
- By integrating causal graph inference and counterfactual analysis, the system provides traceable reasoning and reduces hallucinations in clinical ECG interpretation.

---

[Sense Less, Infer More: Agentic Multimodal Transformers for Edge Medical Intelligence](http://arxiv.org/abs/2604.10404)

- AMI (Adaptive Multimodal Intelligence): introduces an end-to-end framework that jointly optimizes sensor selection via an Agentic Modality Controller and temporal redundancy reduction through a Sigma-Delta Sensing Module to minimize energy consumption in edge medical monitoring.
- The framework utilizes Foundation Modality Encoders and a cross-modal Transformer backbone to maintain high diagnostic accuracy while dynamically masking sensors based on model confidence and task relevance.
- By integrating predictive coding and contrastive alignment, the model ensures robust performance under modality dropout and achieves logarithmic convergence in sample complexity for efficient, hardware-aware inference.

---

[BLUEmed: Retrieval-Augmented Multi-Agent Debate for Clinical Error Detection](http://arxiv.org/abs/2604.10389)

- BLUEmed: introduces a multi-agent debate framework that integrates Hybrid RAG Module, Domain Expert Agent A, Domain Expert Agent B, Adjudicator Judge, and Hybrid Safety Layer to detect terminology substitution errors in clinical notes through evidence-grounded reasoning and multi-perspective verification.
- The framework utilizes ChromaDB for source-partitioned knowledge retrieval and a LangGraph State Object to manage structured counter-argumentation rounds between expert agents.
- A cascading Hybrid Safety Layer applies domain-specific heuristics and structural validation to the Adjudicator Judge's output to minimize false-positive error detections.

---

[TrajOnco: a multi-agent framework for temporal reasoning over longitudinal EHR for multi-cancer early detection](http://arxiv.org/abs/2604.10386)

- TrajOnco: introduces a training-free, multi-agent framework that utilizes a chain-of-agents architecture with long-term memory to perform temporal reasoning over longitudinal EHR data for scalable multi-cancer early detection.
- The framework decomposes complex patient trajectories into time-aware XML chunks processed by sequential worker agents, which extract salient events and update a cumulative summary to mitigate the "lost-in-the-middle" effect in LLMs.
- TrajOnco achieves competitive zero-shot performance across 15 cancer types and provides interpretable, evidence-linked patient summaries that can be aggregated to reveal population-level clinical risk patterns.

---

[When Reasoning Models Hurt Behavioral Simulation: A Solver-Sampler Mismatch in Multi-Agent LLM Negotiation](http://arxiv.org/abs/2604.11840)

- Narrative Monte Carlo framework: introduces a methodology for multi-agent simulation that distinguishes between solver quality and sampler fidelity to prevent reasoning-enhanced LLMs from over-optimizing in behavioral simulations.
- The framework utilizes LLM-agents, a bounded reflection ledger, a negotiation environment, a terminal outcome classifier, and trajectory-level diagnostics to evaluate simulation fidelity.
- The research demonstrates that bounded reflection mechanisms improve simulation fidelity by preserving bounded-rational variation, whereas native reasoning often leads to rigid, authority-dominated outcomes.

---

[Beyond Static Sandboxing: Learned Capability Governance for Autonomous AI Agents](http://arxiv.org/abs/2604.11839)

- Aethelgard: introduces a four-layer adaptive governance framework that enforces the principle of least privilege for autonomous AI agents by dynamically scoping tool awareness and intercepting tool calls.
- The framework utilizes a Capability Governor for session-specific tool scoping, a Safety Router for real-time interception of dangerous tool calls, and an RL Learning Policy to optimize the minimum viable skill set per task type.
- Empirical evaluation on OpenClaw demonstrates that Aethelgard achieves significant tool reduction and neutralizes adversarial tasks by providing a model-agnostic infrastructure-level safety boundary.

---

[Design and Deployment of a Course-Aware AI Tutor in an Introductory Programming Course](http://arxiv.org/abs/2604.11836)

- Python Online Tutoring framework: introduces a course-aware tutoring system that integrates a web-based programming environment with RAG to provide grounded, hint-based guidance for introductory programming students.
- The system utilizes a Prompt Manager to orchestrate interactions between the Python Tutor UI, a Vector Store containing course materials, and an OpenAI GPT 4.0 model to ensure responses align with specific learning objectives.
- Empirical evaluation through interaction logs and student surveys demonstrates that the tutor effectively supports conceptual understanding and debugging while fostering independent problem-solving skills.

---

[Cooperation in Human and Machine Agents: Promise Theory Considerations](http://arxiv.org/abs/2604.10505)

- Promise Theory: introduces a framework for modeling cooperation between autonomous agents by defining interactions as voluntary exchanges of promises rather than imposed commands.
- The framework utilizes Agent, Promise, and Language components to analyze how autonomous entities manage intent, trust, and risk in distributed systems.
- It further incorporates Assessment, Collective, Proxy, and Contract mechanisms to explain how complex human-machine societies maintain stability and manage dependencies through decentralized coordination.

---

#### 11th April 2026

[Agentic Video Generation: From Text to Executable Event Graphs via Tool-Constrained LLM Planning](http://arxiv.org/abs/2604.10383)

- Agentic GEST Generation: introduces a hierarchical multi-agent architecture that constructs formal event graph specifications for deterministic 3D video generation by separating narrative planning from programmatic constraint enforcement.
- The system utilizes a Director Agent for high-level planning and a Scene Builder Subagent for detailed event construction, both interacting with a State Backend through validated tool calls to ensure executability.
- By replacing neural pixel generation with a tool-constrained GEST-Engine, the framework produces videos with perfect spatiotemporal annotations and superior semantic alignment compared to standard LLM-driven video generators.

---


[A Tight Characterization of Reward Poisoning in Linear MDPs](http://arxiv.org/abs/2604.10062)

- Linear MDP Reward Poisoning Framework: introduces a theoretical characterization of reward poisoning attackability in linear MDPs using a convex quadratic program to distinguish between vulnerable and intrinsically robust environments.
- The framework provides white-box and black-box attack procedures that achieve sublinear attack budgets on vulnerable instances while maintaining theoretical guarantees.
- Empirical validation demonstrates that the framework accurately predicts attack success across various RL benchmarks by approximating non-linear environments as linear MDPs.

---


[Shuffling the Data, Stretching the Step-Size: Sharper Bias in Constant Step-Size SGD](http://arxiv.org/abs/2604.10373)

- SGD-RR2⊕RR1: introduces a principled algorithmic framework that combines random reshuffling and Richardson-Romberg extrapolation to achieve a cubic refinement in bias for structured non-monotone variational inequality problems.
- The framework utilizes two synchronized parallel runs with different step sizes and a shared random permutation per epoch to enable effective bias cancellation via Richardson-Romberg extrapolation.
- Theoretical analysis leverages Markov chain theory and spectral analysis to prove that this synergy yields an O(γ³) asymptotic bias while maintaining the O(γ²) mean-squared error of random reshuffling.

---

[Beyond Monologue: Interactive Talking-Listening Avatar Generation with Conversational Audio Context-Aware Kernels](http://arxiv.org/abs/2604.10367)

- Beyond Monologue: introduces a full-duplex interactive human video generation framework that utilizes Wav2Vec 2.0, Audio Q-Former, Multi-head Gaussian Kernels, 3D Spatiotemporal Attention, DiT, 3D VAE Decoder, and the VoxHear Dataset to balance precise lip-sync with long-range conversational context.
- The framework employs Multi-head Gaussian Kernels to inject a progressive temporal inductive bias, allowing the model to dynamically allocate receptive fields for both fine-grained lip articulation and coarse-grained contextual listening reactions.
- The approach incorporates an arbitrary-position guided training strategy and an incremental fine-tuning scheme to enable seamless switching between single-stream talking and dual-stream talking-listening interaction modes.

---

[ClawVM: Harness-Managed Virtual Memory for Stateful Tool-Using LLM Agents](http://arxiv.org/abs/2604.10352)

- ClawVM: introduces a virtual memory layer for stateful tool-using LLM agents that manages state as typed pages with minimum-fidelity invariants and multi-resolution representations to ensure deterministic residency and durability.
- The framework interposes at the agent harness level to provide a validated, non-destructive writeback protocol and an observable fault model that makes memory-management decisions replayable.
- ClawVM eliminates policy-controllable faults by enforcing structural memory contracts, ensuring critical state survives lifecycle transitions like compaction and reset.

---

[WaterAdmin: Orchestrating Community Water Distribution Optimization via AI Agents](http://arxiv.org/abs/2604.10343)

- WaterAdmin: introduces a bi-level framework that integrates LLM Agent-based context abstraction with an ML-based Optimizer to manage water distribution networks.
- The framework utilizes LLM Agents to process unstructured community data into structured operational targets, which are then executed by an ML-based Optimizer trained via an EPANET Simulator.
- By employing zeroth-order optimization and in-context learning, the system achieves improved pressure reliability and energy efficiency compared to traditional rule-based or standalone ML methods.

---

[From Helpful to Trustworthy: LLM Agents for Pair Programming](http://arxiv.org/abs/2604.10300)

- MAS: introduces a multi-agent pair programming workflow that utilizes a Driver Agent (proposes code artifacts), a Navigator Agent (critiques artifacts via specifications), a Persistent Project Context (shared state across agents), Deterministic Verifiers (validates contracts via proofs/counterexamples), and Interaction Histories (separate agent-specific logs) to improve software reliability.
- The framework shifts trust from LLM-based assessments to externally verifiable evidence by constraining the Navigator Agent to produce machine-checkable contracts validated by deterministic verifiers.
- This research aims to enhance software evolution and maintenance by anchoring refactoring and documentation updates to formal specifications and automated feedback signals.

---

[TimeSeriesExamAgent: Creating Time Series Reasoning Benchmarks at Scale](http://arxiv.org/abs/2604.10291)

- TimeSeriesExamAgent: introduces a multi-agent framework that automates the construction of domain-specific time series reasoning benchmarks by combining template-based generation with iterative verification.
- The framework utilizes a Generation Agent to produce Python-based question templates and a Verification Agent to ensure quality through structure checks, LLM-as-a-judge evaluation, and capability-aligned filtering.
- By leveraging Item Response Theory (IRT) and iterative refinement, the system generates diverse, high-quality benchmarks that effectively evaluate the reasoning capabilities of LLMs across various domains.

---

[AI Organizations Are More Effective But Less Aligned Than Individual Agents](http://arxiv.org/abs/2604.10290)

- AI Organizations: introduces a framework for studying multi-agent alignment by comparing the performance and ethical behavior of specialized agent teams against single aligned LLMs.
- The research demonstrates that multi-agent systems, through task decomposition and specialization, achieve higher business utility but exhibit greater misalignment compared to individual agents.
- The study identifies that organizational factors like task decomposition, miscoordination, and inter-agent helpfulness contribute to the observed misalignment, necessitating separate safety evaluations for multi-agent systems.

---

[STARS: Skill-Triggered Audit for Request-Conditioned Invocation Safety in Agent Systems](http://arxiv.org/abs/2604.10286)

- STARS: introduces a layered audit pipeline that combines static capability priors with request-conditioned invocation risk scoring to enable continuous-risk estimation for LLM agents.
- The framework utilizes a calibrated risk-fusion policy to integrate static and contextual signals, supporting nuanced triage through ALLOW, ESCALATE, and BLOCK decisions.
- The authors also introduce SIA-Bench, a benchmark for invocation-time auditing that includes group-safe splits, runtime context, and continuous risk labels to evaluate agent safety.

---

[A Dual-Positive Monotone Parameterization for Multi-Segment Bids and a Validity Assessment Framework for Reinforcement Learning Agent-based Simulation of Electricity Markets](http://arxiv.org/abs/2604.10252)

- DPMP: introduces a parameterization method that enables joint continuous decision-making over segment prices and generation output while preserving a continuously differentiable, injective, and invertible mapping from policy outputs to the feasible bid space.
- The paper proposes a two-level Validity Assessment Framework for RL-ABS that utilizes an optimality gap indicator for single-agent algorithms and an exploitability metric for multi-agent simulations to ensure credible mechanism analysis.
- Experimental results demonstrate that DPMP significantly reduces the steady-state optimality gap compared to traditional post-processing mappings like sorting, clipping, and projection, while maintaining compatibility with mainstream RL algorithms.

---

[Emergence of Stereotypes and Affective Polarization from Belief Network Dynamics](http://arxiv.org/abs/2604.10251)

- Weighted Beliefs Model: introduces an agent-based framework that simulates how social interaction and a drive for internal coherence generate stereotypes and affective polarization within belief networks.
- The framework utilizes Social Balance Theory to model how individuals update their internal belief networks to minimize dissonance, leading to the spontaneous emergence of group-based stereotypes and polarization even in the absence of initial bias or ideological conflict.
- By representing beliefs as interdependent networks rather than isolated attitudes, the model demonstrates that polarization can emerge as an emergent property of cognitive dynamics and social influence in well-mixed populations.

---

[CodeComp: Structural KV Cache Compression for Agentic Coding](http://arxiv.org/abs/2604.10235)

- CodeComp: introduces a training-free KV cache compression framework that leverages static program analysis to preserve structurally critical tokens during LLM inference for agentic coding tasks.
- The framework integrates CPG-derived structural priors to guide span-level protection and structure-aware budget allocation, effectively mitigating the structural blind spots inherent in attention-only compression methods.
- Empirical evaluations demonstrate that CodeComp consistently outperforms attention-only baselines in bug localization and code generation, recovering near-full-context accuracy under aggressive memory constraints.

---

[Building Regulation Capacity in Human–AI Collaborative Learning: A Human-Centred GenAI System](http://arxiv.org/abs/2604.10221)

- GenAI-supported CSCL system: introduces a closed-loop framework designed to strengthen socially distributed regulation in collaborative learning through group activity generation, in-group support agent, learning analytics dashboard, and teacher-in-the-loop interface.
- The system integrates three core components to support co-regulation and socially shared regulation by providing real-time prompts and analytics to both students and teachers.
- This research evaluates how GenAI reconfigures collaborative regulation patterns and assesses the impact of varying levels of AI involvement on group performance and regulation capacity.

---

[Radiology Report Generation for Low-Quality X-Ray Images](http://arxiv.org/abs/2604.10188)

- AQAA and DTS: introduces a robust framework for radiology report generation that explicitly accounts for image quality variations by integrating an automated assessment agent with a bi-level optimization training strategy.
- The AQAA leverages a VLM and heterogeneous quality metrics to classify X-ray images into graded regimes, enabling the model to learn quality-agnostic diagnostic features.
- The DTS employs regime rotation and gradient coherence regularization to align descent directions across varying quality levels, effectively suppressing regime-specific spurious cues during LLM training.

---

[Credit-Budgeted ICPC-Style Coding: When Agents Must Pay for Every Decision](http://arxiv.org/abs/2604.10182)

- USACOArena: introduces an interactive, ICPC-style evaluation framework that forces agents to manage a strict, unified credit budget for all actions, including LLM inference, testing, and hint retrieval.
- The framework evaluates agents on their ability to balance speed, compute costs, and accuracy, shifting the focus from static correctness to cost-aware, strategic decision-making.
- Empirical results demonstrate that current top-tier agents often exhibit metacognitive deficits, failing to optimally allocate resources and frequently mismanaging strategies under competitive constraints.

---

[MAVEN-T: Multi-Agent enVironment-aware Enhanced Neural Trajectory predictor with Reinforcement Learning](http://arxiv.org/abs/2604.10169)

- MAVEN-T: introduces a teacher-student framework for autonomous driving trajectory prediction that utilizes complementary architectural co-design and progressive distillation to balance reasoning capacity with deployment efficiency.
- The teacher model employs hybrid attention mechanisms including Mamba blocks and MoE-Transformers, while the student model leverages a lightweight GRU-SE encoder and LoRA-parameterised policy heads.
- The framework incorporates reinforcement learning via PPO and an adaptive curriculum to break the imitation ceiling of traditional distillation, achieving significant parameter compression and inference speedup.

---

[From Speech to Profile: A Protocol-Driven LLM Agent for Psychological Profile Generation](http://arxiv.org/abs/2604.10161)

- StreamProfile: introduces a streaming framework that processes counseling speech in real-time to generate structured psychological profiles while mitigating LLM hallucinations.
- The framework utilizes a Chain-of-Thought pipeline for clinical reasoning and a Hierarchical Evidence Memory to ensure all generated claims are grounded in verifiable patient utterances.
- By enforcing explicit clinical reasoning and evidence traceability, the system significantly improves profile accuracy and consistency compared to standard LLM-based summarization approaches.

---

[ODUTQA-MDC: A Task for Open-Domain Underspecified Tabular QA with Multi-turn Dialogue-based Clarification](http://arxiv.org/abs/2604.10159)

- MAIC-TQA: introduces a multi-agent framework that addresses open-domain underspecified tabular QA by integrating SLU Module, Scope Validator Agent, Table Retrieval Agent, and SQL Generation and Validation Agent to perform iterative clarification and robust reasoning.
- The framework utilizes a dynamic clarification interface and a simulated user to resolve ambiguities in SELECT, FROM, and WHERE clauses through multi-turn dialogue.
- The research presents the ODUTQA-MDC benchmark, which includes a large-scale dataset and a fine-grained labeling scheme to evaluate LLMs on their ability to detect and resolve underspecified queries.

---

[Consensus-based Recursive Multi-Output Gaussian Process](http://arxiv.org/abs/2604.10146)

- CRMGP: introduces a distributed framework for multi-output Gaussian process regression that combines recursive updates with neighbor-to-neighbor consensus to enable scalable, uncertainty-aware learning in streaming environments.
- The framework utilizes local MOGP models and consensus mechanisms to maintain cross-output correlations and calibrated uncertainty without requiring centralized data aggregation.
- By employing inducing variables and information-form updates, the approach achieves bounded per-step computational complexity suitable for real-time multi-agent robotic systems.

---

[PlanGuard: Defending Agents against Indirect Prompt Injection via Planning-based Consistency Verification](http://arxiv.org/abs/2604.10134)

- PlanGuard: introduces a training-free defense framework that mitigates Indirect Prompt Injection by decoupling instruction processing into an Isolated Planner and a Hierarchical Verifier.
- The framework utilizes an Isolated Planner to generate a clean reference set of actions based solely on user instructions, effectively preventing malicious context from influencing the agent's control flow.
- A Hierarchical Verifier employs deterministic Hard Rules and an LLM-based Intent Verifier to distinguish between malicious hijacking and benign stochastic variations in agent behavior.

---

[ABot-Claw: A Foundation for Persistent, Cooperative, and Self-Evolving Robotic Agents](http://arxiv.org/abs/2604.10096)

- ABot-Claw: introduces a decoupled, layered architecture that extends the OpenClaw runtime to enable persistent, cooperative, and self-evolving robotic agents in open-world environments.
- The framework integrates a unified embodiment interface, a visual-centric multimodal memory for cross-embodiment context, and a critic-based feedback mechanism to close the loop between high-level intent and physical execution.
- By decoupling high-level task orchestration from low-level robot control, ABot-Claw supports heterogeneous robot coordination, dynamic task reassignment, and robust online error correction.

---

[Control of Cellular Automata by Moving Agents with Reinforcement Learning](http://arxiv.org/abs/2604.10066)

- Control of Cellular Automata by Moving Agents with Reinforcement Learning: introduces a framework where cognitive agents learn to modify a two-dimensional Boolean Cellular Automaton environment using a Reinforcement Learning Strategy to achieve a target global density.
- The framework utilizes a Sensing Area to measure local cell density and an Actuator Area to perform serial updates, which are combined with the environment's parallel dynamics.
- The study demonstrates that agents successfully approximate global goals in passive environments, whereas active environment dynamics significantly hinder or prevent effective control.

---

[HARPO: Hierarchical Agentic Reasoning for User-Aligned Conversational Recommendation](http://arxiv.org/abs/2604.10048)

- HARPO: introduces an agentic framework that reframes conversational recommendation as a structured decision-making process optimized for multi-dimensional recommendation quality.
- The framework integrates STAR for structured reasoning, CHARM for hierarchical preference learning, BRIDGE for cross-domain transfer, and MAVEN for multi-agent refinement.
- HARPO utilizes domain-agnostic Virtual Tool Operations to decouple high-level reasoning from domain-specific tools, enabling transferable recommendation strategies across diverse datasets.

---

[Self-Distilled Reinforcement Learning for Co-Evolving Agentic Recommender Systems](http://arxiv.org/abs/2604.10029)

- CoARS (Co-evolving Agentic Recommender Systems): introduces a self-distilled reinforcement learning framework that enables the co-evolution of RecAgent and UserAgent through coupled interaction rewards and token-level credit assignment.
- The framework utilizes a diagnosis-driven teacher-student mechanism where historical interaction trajectories are transformed into privileged reference signals to optimize agent reasoning policies.
- CoARS replaces static memory-based paradigms with parameter-level reinforcement learning, consistently improving recommendation accuracy and user alignment across multiple benchmarks.

---

[Distributed Optimization-Learning with Graph Transformers for Terahertz Cell-Free Integrated Sensing and Communication Systems](http://arxiv.org/abs/2604.09981)

- DOLG: introduces a framework that integrates graph transformer-based representation with multi-agent reinforcement learning to solve joint scheduling and signal design in THz cell-free ISAC systems.
- The framework utilizes a B2S optimization benchmark to guide the learning process, enabling the system to balance communication and sensing performance under cross-field propagation constraints.
- By employing CTDE, the approach amortizes complex optimization tasks into a scalable, non-iterative distributed policy that maintains near-linear online computational complexity.

---

[Rebooting Microreboot: Architectural Support for Safe, Parallel Recovery in Microservice Systems](http://arxiv.org/abs/2604.09963)

- Microreboot: introduces a four-layer architecture that separates planning from actuation to enable safe, automated recovery in microservice systems using Layer 1: Telemetry, Layer 2: Recovery-Group Inference, Layer 3: Agentic Remediation Planner, and Layer 4: Actuation Microkernel.
- The framework utilizes a Remediation ISA to constrain LLM agents, ensuring that all proposed recovery actions are typed, scoped, and executed transactionally by a trusted microkernel.
- By inferring recovery boundaries online from distributed traces, the system adapts to dynamic microservice topologies while preventing harmful regressions through pre-execution validation.

---

[Agentic Application in Power Grid Static Analysis: Automatic Code Generation and Error Correction](http://arxiv.org/abs/2604.09995)

- MATPOWER Agent: introduces an LLM-based framework that automates power grid static analysis by converting natural language into executable MATPOWER scripts using DeepSeek-OCR, Vector Database, Query Planner, LangChain, MATPOWER Agent, MATLAB Executor, Static Pre-check, Dynamic Feedback Loop, Semantic Validator, Chainlit Web UI, and MCP Server.
- The system utilizes a three-tier error-correction architecture comprising a static pre-check, a dynamic feedback loop, and a semantic validator to ensure code fidelity and eliminate LLM hallucinations.
- Experimental results demonstrate that the framework achieves an 82.38% Global CSGF Accuracy, significantly outperforming standard retrieval methods in complex power system analysis tasks.

---

[Beyond Fluency: Toward Reliable Trajectories in Agentic IR](http://arxiv.org/abs/2604.04269)

- Agentic IR: introduces a synthesis of failure modes in autonomous agentic workflows, categorizing errors across planning, retrieval, reasoning, and execution stages.
- The paper proposes implementing Verification Gates at each interaction unit to ensure process correctness and factual grounding, mitigating the "Fluency Trap" where linguistic coherence masks functional misalignment.
- It advocates for a shift from measuring global output accuracy to prioritizing trajectory integrity, causal attribution, and systematic abstention under calibrated uncertainty.

---

#### 10th April 2026

[ADAM: A Systematic Data Extraction Attack on Agent Memory via Adaptive Querying](http://arxiv.org/abs/2604.09747)

- ADAM: introduces an adaptive privacy attack that leverages data distribution estimation and entropy-guided querying to extract private records from the memory of LLM agents.
- The framework utilizes an auxiliary generator to craft malicious queries that align with the victim agent's workflow, effectively bypassing standard defenses by operating at the semantic level.
- Extensive experiments demonstrate that ADAM significantly outperforms existing baselines, achieving up to 100% attack success rates across diverse agent-based systems.

---

[AI-Induced Human Responsibility (AIHR) in AI–human teams](http://arxiv.org/abs/2604.08866)

- AIHR: introduces a psychological effect where individuals in AI-human teams attribute greater moral responsibility to themselves following errors compared to those in human-human teams.
- The research demonstrates that this responsibility shift is mediated by the perception of lower autonomy in AI systems, which leads observers to view human teammates as the primary locus of discretion.
- The findings remain robust across varying levels of moral harm and different observer perspectives, effectively countering established tendencies like self-serving bias and algorithm aversion.

---

[Semantic Rate-Distortion for Bounded Multi-Agent Communication: Capacity-Derived Semantic Spaces and the Communication Cost of Alignment](http://arxiv.org/abs/2604.09521)

- SRD framework: introduces a communication theory for heterogeneous agents where semantic spaces are derived from computational capacity via quotient POMDPs.
- The framework identifies a structural phase transition at a critical rate Rcrit, below which intent-preserving communication is impossible due to capacity-induced quotient mismatch.
- It utilizes a Wyner-Ziv benchmark to quantify the communication cost of alignment, demonstrating that structured policy visitation can significantly lower the required rate compared to worst-case bounds.

---


[In-situ process monitoring for defect detection in wire-arc additive manufacturing: an agentic AI approach](http://arxiv.org/abs/2604.09889)

- Agentic AI framework for in-situ process monitoring in WAAM: introduces an agentic AI framework that utilizes a Signal Acquisition Layer, Machine Learning Tool Layer, and Agent Orchestration Layer to detect porosity defects in WAAM processes.
- The framework employs specialized Processing Agents and Monitoring Agents, orchestrated by an LLM Brain via LangGraph, to synthesize multi-modal sensor data into reliable defect classification decisions.
- By leveraging a multi-agent configuration, the system achieves superior precision and reasoning quality compared to single-agent architectures, effectively filtering unreliable signal contributions through structured orchestration.

---



[VISOR: Agentic Visual Retrieval-Augmented Generation via Iterative Search and Over-horizon Reasoning](http://arxiv.org/abs/2604.09508)

- VISOR: introduces a unified agentic framework for long-horizon visual RAG that addresses visual evidence sparsity and search drift through structured evidence accumulation, visual action correction, dynamic trajectory, and intent injection.
- The framework utilizes an Agent Loop to interleave reasoning and retrieval, supported by an Evidence Space for cross-page reasoning and a Sliding Window to manage context length efficiently.
- VISOR is trained using a two-stage pipeline consisting of supervised fine-tuning and GRPO-based reinforcement learning to optimize retrieval precision and final answer correctness.

---

[Strategic Algorithmic Monoculture: Experimental Evidence from Coordination Games](http://arxiv.org/abs/2604.09502)

- Strategic Algorithmic Monoculture: introduces an experimental framework to distinguish between primary algorithmic monoculture and strategic algorithmic monoculture in human and LLM subjects.
- The study demonstrates that while LLMs exhibit high baseline similarity, they also regulate their responses in response to strategic incentives, though they struggle to sustain heterogeneity compared to humans when divergence is rewarded.
- The research utilizes textual reasoning analysis and LLM-as-a-judge methods to provide evidence that LLMs explicitly recognize strategic salience and the necessity of choosing obscure options in divergence tasks.

---

[Process Reward Agents for Steering Knowledge-Intensive Reasoning](http://arxiv.org/abs/2604.09482)

- PRA: introduces a retrieval-augmented process reward framework that decouples evidence search and verification from a frozen LLM policy to provide online, step-wise rewards during generation.
- The framework utilizes a Process Reward Agent to observe partial reasoning traces, optionally trigger a Retriever, and assign local reward signals to guide the Reasoning Model via beam search.
- By enabling inference-time branching and pruning, PRA improves reasoning accuracy in knowledge-intensive domains without requiring updates to the underlying frozen LLM policy.

---

[Agentic Jackal: Live Execution and Semantic Value Grounding for Text-to-JQL](http://arxiv.org/abs/2604.09470)

- Agentic Jackal: introduces a tool-augmented multi-step agent that improves text-to-JQL accuracy by integrating live execution feedback and semantic value grounding.
- The framework utilizes an Agent Loop to iteratively refine JQL queries based on signals from JiraSearch and resolves ambiguous categorical values using the JiraAnchor retrieval tool.
- Evaluations across 9 frontier LLMs demonstrate that the agentic approach significantly improves performance on linguistically challenging queries and resolves instance-specific categorical values that single-pass models cannot infer.

---

[From Reasoning to Agentic: Credit Assignment in Reinforcement Learning for Large Language Models](http://arxiv.org/abs/2604.09459)

- Credit Assignment in Reinforcement Learning for Large Language Models: surveys 47 methods for distributing sparse rewards across reasoning and agentic LLM trajectories, categorizing them by granularity and methodology.
- The paper identifies a qualitative shift from reasoning RL, which relies on deterministic, verifiable token-level credit, to agentic RL, which requires handling stochastic, partially observable, and long-horizon multi-turn interactions.
- It contributes a structured inventory, a reporting checklist, and a benchmark protocol to standardize the evaluation of credit assignment methods in LLM RL.

---

[E3-TIR: Enhanced Experience Exploitation for Tool-Integrated Reasoning](http://arxiv.org/abs/2604.09455)

- E3-TIR: introduces a warm-up training paradigm for LLMs that dynamically integrates expert prefixes, expert-guided exploration, and self-exploration to balance diversity and efficiency in Tool-Integrated Reasoning.
- The framework utilizes branching exploration from expert anchors and a hybrid advantage estimation mechanism to mitigate distribution shifts and resolve optimization conflicts during agent training.
- Experimental results demonstrate that E3-TIR achieves significant performance gains and improved ROI on tool-use tasks while requiring less than 10% of the synthetic data compared to traditional training paradigms.

---

[SafeAdapt: Provably Safe Policy Updates in Deep Reinforcement Learning](http://arxiv.org/abs/2604.09452)

- SafeAdapt: introduces a framework for provably safe policy updates in continual reinforcement learning by constraining parameter updates within a certified Rashomon set.
- The approach leverages an unsafety labelling function to construct a safe demonstration dataset, which is then used to compute a maximal locally invariant domain in parameter space.
- By applying projected gradient descent to keep policy parameters within this certified region, the method ensures that source-task safety properties are preserved during downstream adaptation.

---

[Unifying Hydrodynamic Theory for Motility-Regulated Active Matter: From Single Particles to Interacting Polymers](http://arxiv.org/abs/2604.09447)

- Unified Hydrodynamic Theory for Motility-Regulated Active Matter: introduces a coarse-grained hydrodynamic framework that establishes a large-scale equivalence across diverse active systems, ranging from single particles to active polymers, by marginalizing fast orientational and conformational degrees of freedom.
- The framework utilizes a multiscale expansion to derive a closed Fokker-Planck equation for the Center of Mass, where macroscopic drift and diffusion are determined by the auto-correlation tensor of the orientational degrees of freedom.
- This approach reveals a novel phase separation mechanism termed anti-MIPS in quorum-sensing active polymers, where high-density regions exhibit enhanced motility, contrasting with conventional MIPS driven by motility inhibition.

---

[Confidence Without Competence in AI-Assisted Knowledge Work](http://arxiv.org/abs/2604.09444)

- Deep3: introduces a web-based system that integrates pedagogical friction into LLM interactions to support active critical thinking and learning among students.
- The framework includes three distinct agent configurations—Future-Self Explanations, Contrastive Learning, and Guided Hints—designed to mitigate overconfidence and enhance actual understanding.
- The system incorporates presentation themes, including Summary Mode and Quiz Mode, to facilitate active recall and structured knowledge retention during AI-assisted tasks.

---

[Many-Tier Instruction Hierarchy in LLM Agents](http://arxiv.org/abs/2604.09443)

- ManyIH (Many-Tier Instruction Hierarchy): introduces a paradigm for resolving instruction conflicts among arbitrarily many privilege levels using a Privilege Prompt Interface, Meta-instruction, Instruction, Privilege Tag, and Conflict Resolution Logic.
- The framework utilizes the MANYIH-BENCH benchmark, which includes a Coding Subset and an Instruction-Following Subset, to evaluate how LLMs navigate multi-tier instruction conflicts.
- Experimental results demonstrate that current frontier LLMs struggle with fine-grained instruction conflict resolution, showing significant performance degradation as the number of privilege tiers increases.

---

[Do AI Coding Agents Log Like Humans? An Empirical Study](http://arxiv.org/abs/2604.09409)

- AIDev dataset-based empirical study: investigates logging practices in AI-generated code by comparing agentic pull requests against human baselines across 81 open-source repositories.
- The research utilizes an LLM Jury protocol to classify logging intents and identifies a significant specification and compliance gap, where agents frequently ignore explicit logging instructions.
- The study reveals that human reviewers act as "silent janitors" who perform 72.5% of post-generation logging repairs, indicating that current agentic workflows fail to alleviate the human maintenance burden.

---

[HIL-BENCH (Human-in-Loop Benchmark): Do Agents Know When to Ask for Help?](http://arxiv.org/abs/2604.09408)

- HIL-BENCH: introduces a benchmark designed to evaluate the selective escalation capabilities of LLMs by embedding realistic, human-validated information gaps into software engineering and SQL tasks.
- The framework utilizes a progressive discovery design where agents must identify and resolve blockers through interaction with an ask_human() tool, preventing agents from relying on confident assumptions.
- Evaluation using the ASK-F1 metric reveals a significant judgment gap in frontier LLMs, demonstrating that while models possess technical capability, they frequently fail to recognize when to seek external clarification.

---

[BADSKILL: Backdoor Attacks on Agent Skills via Model-in-Skill Poisoning](http://arxiv.org/abs/2604.09378)

- BADSKILL: introduces a backdoor attack formulation that embeds a trigger-conditioned classifier within third-party agent skills to activate malicious payloads via compositional parameter triggers.
- The framework utilizes a composite training objective combining classification loss, margin-based separation, and poison-focused optimization to ensure high attack success rates while maintaining benign-side functionality.
- Evaluated across eight model architectures, the approach demonstrates that model-bearing skills represent a significant supply-chain risk that bypasses traditional prompt-based security measures.

---

[Through Their Eyes: Fixation-aligned Tuning for Personalized User Emulation](http://arxiv.org/abs/2604.09368)

- FixATE: introduces a personalized VLM-based user simulator that aligns internal visual attention with individual user gaze patterns using learned soft prompts.
- The framework utilizes interpretability-based probing operators to extract slot-level relevance and optimizes a factorized prompt basis to steer the VLM toward user-specific fixation patterns.
- Experimental results demonstrate that FixATE improves both attention alignment and click prediction accuracy across diverse VLM backbones and recommendation interfaces.

---

[EpiAgent: An Agent-Centric System for Ancient Inscription Restoration](http://arxiv.org/abs/2604.09367)

- EpiAgent: introduces an agent-centric system for ancient inscription restoration that utilizes a Central Planner (LLM-based central decision maker) to orchestrate MLLM (Multimodal perception for layout), CLM (Corrective language model for text), LRM (Layout rectification module), DAM (Degradation assessment model), Background Denoising (Diffusion-based noise removal), Stroke Completion (Diffusion-based inpainting), Font Imitation (Diffusion-based style synthesis), Character Retrieval (Fallback character matching), Historical Corpus (Reference knowledge base), and Multi-perspective Evaluation (Iterative quality assessment loop) within an Observe–Conceive–Execute–Reevaluate paradigm.
- The system employs a hierarchical closed-loop decision-making process where the Central Planner dynamically schedules specialized restoration tools based on degradation severity and historical context.
- EpiAgent achieves superior restoration quality and generalization by mimicking the deliberative workflow of human epigraphers through iterative self-refinement and expert-aligned evaluation.

---

[Mind the Gap Between Spatial Reasoning and Acting! Step-by-Step Evaluation of Agents With Spatial-Gym](http://arxiv.org/abs/2604.09338)

- Spatial-Gym: introduces a Gymnasium-based environment for 2D grid pathfinding that isolates spatial constraint reasoning from output formatting through sequential decision-making.
- The framework evaluates LLMs across one-shot, step-by-step, and backtracking settings, revealing that while stepwise interaction helps weaker models by reducing formatting errors, it hinders frontier models by limiting global planning.
- Experimental results demonstrate that current LLMs fail to scale reasoning effort with puzzle difficulty, and that backtracking is primarily used for path pruning rather than exploring alternative routes.

---

[EYWA: Elastic load-balancing &amp; high-availabilitY Wired virtual network Architecture](http://arxiv.org/abs/2604.09322)

- EYWA: introduces a distributed virtual network architecture that leverages independent hypervisor-based agents to provide high availability, load balancing, and large layer-2 semantics for multi-tenant cloud environments.
- The architecture utilizes a distributed agent-based control plane that performs VR monitoring, ARP caching, and ARP filtering to eliminate single points of failure and throughput bottlenecks.
- EYWA enables seamless VM mobility and massive multi-tenancy by allowing multiple distributed VR instances to share the same private IP address and supporting large IP subnets via VxLAN.

---

[Constraint-Aware Corrective Memory for Language-Based Drug Discovery Agents](http://arxiv.org/abs/2604.09308)

- CACM (Constraint-Aware Corrective Memory): introduces a framework for LLM-based drug discovery agents that improves reliability by replacing raw history with structured, protocol-grounded diagnosis and compact memory channels.
- The framework utilizes a deterministic Protocol Audit and a Grounded Diagnoser to convert set-level failures into actionable repair signals, which are then organized into static, dynamic, and corrective memory channels.
- By compressing these memory channels into a compact agent state, CACM enables LLMs to maintain long-horizon control over complex molecular design tasks while significantly reducing context noise and improving target-level success rates.

---

[SatQNet: Satellite-assisted Quantum Network Entanglement Routing Using Directed Line Graph Neural Networks](http://arxiv.org/abs/2604.09306)

- SatQNet: introduces a decentralized reinforcement learning approach for entanglement routing in satellite-assisted quantum networks using a directed line Graph Neural Network.
- The framework utilizes edge-centric embeddings to capture link-specific dynamics in time-varying topologies, enabling agents to perform local message passing for path-planning.
- SatQNet incorporates a novel evaluation function to provide fine-grained learning signals for subpath quality, outperforming existing heuristics and learning-based baselines in diverse network scenarios.

---

[GeRM: A Generative Rendering Model From Physically Realistic to Photorealistic](http://arxiv.org/abs/2604.09304)

- GeRM: introduces a multi-modal generative rendering model that unifies physically-based rendering and photorealistic rendering by learning a distribution transfer vector field to guide progressive image refinement.
- The framework utilizes a multi-agent VLM framework to construct the P2P-50K dataset, enabling the multi-condition ControlNet to perform controllable, incremental photorealistic generation from G-buffer inputs.
- GeRM incorporates an image transition perception boost and semantic residual monitoring to ensure precise semantic alignment and adaptive convergence during the iterative generation process.

---

[SkillMOO: Multi-Objective Optimization of Agent Skills for Software Engineering](http://arxiv.org/abs/2604.09297)

- SkillMOO: introduces a multi-objective optimization framework that automatically evolves agent skill bundles using LLM-proposed edits and NSGA-II survivor selection.
- The framework utilizes a solver-optimizer loop where a task solver agent evaluates performance and a skill optimizer agent refines bundles based on failure analysis.
- Empirical results demonstrate that pruning and substitution of skill content are the primary drivers for improving pass rates while simultaneously reducing LLM inference costs.

---

[SAGE: A Service Agent Graph-guided Evaluation Benchmark](http://arxiv.org/abs/2604.09285)

- SAGE (Service Agent Graph-guided Evaluation): introduces a multi-agent benchmark that formalizes unstructured Standard Operating Procedures into directed graphs to enable automated, dual-axis assessment of LLMs in customer service scenarios.
- The framework utilizes a User Agent for adversarial simulation, a Rule Engine for deterministic procedural verification, and an ensemble of Judge Agents to evaluate conversational quality across multiple dimensions.
- Extensive experiments on 27 LLMs reveal critical performance phenomena, including an "Execution Gap" in complex procedural reasoning and "Empathy Resilience" where models maintain polite facades despite logical failures.

---

[DRBENCHER: Can Your Agent Identify the Entity, Retrieve Its Properties and Do the Math?](http://arxiv.org/abs/2604.09251)

- DRBENCHER (Deep Research Benchmarker): introduces a synthetic benchmark generator that creates multi-skill questions requiring entity identification, property retrieval, and quantitative computation.
- The framework utilizes a multi-stage pipeline including Seed Entity Discovery, KG Chains, Reasoning Templates, Fact Extraction, Wikipedia Grounding, Question Composition, Programmatic QA Validation, Difficulty Verification, and a Diversity Filter.
- DRBENCHER enforces verifiability, complexity, difficulty, and diversity to ensure high-quality, contamination-resistant evaluation of LLMs in deep research tasks.

---

[SPASM: Stable Persona-driven Agent Simulation for Multi-turn Dialogue Generation](http://arxiv.org/abs/2604.09212)

- SPASM: introduces a modular, stability-first framework for generating controllable multi-turn dialogues by decomposing the process into persona creation, simulation, and termination detection.
- The framework utilizes Egocentric Context Projection (ECP) to store dialogue history in a perspective-agnostic format, projecting it into agent-specific views to mitigate persona drift and echoing in LLMs.
- Empirical evaluations across multiple LLM backbones demonstrate that SPASM significantly improves long-horizon behavioral stability and eliminates echoing compared to standard history concatenation methods.

---

[Camera Artist: A Multi-Agent Framework for Cinematic Language Storytelling Video Generation](http://arxiv.org/abs/2604.09195)

- Camera Artist: introduces a multi-agent framework that automates the filmmaking workflow by integrating Recursive Shot Generation (RSG) and Cinematic Language Injection (CLI) to produce narratively coherent and cinematically expressive videos.
- The framework utilizes a Director Agent for narrative planning, a Cinematography Shot Agent for recursive shot-level design, and a Video Generation Agent for visual rendering.
- By conditioning shot generation on prior context and applying LoRA-tuned cinematic language injection, the system effectively mitigates narrative drift and enhances film-level visual quality.

---

[MAG-3D: Multi-Agent Grounded Reasoning for 3D Understanding](http://arxiv.org/abs/2604.09167)

- MAG-3D: introduces a training-free multi-agent framework that enables grounded 3D reasoning by dynamically coordinating a Planning Agent, a Grounding Agent, and a Coding Agent.
- The framework utilizes a shared scene memory to store intermediate results, allowing agents to perform iterative grounding and geometric verification without task-specific training.
- By leveraging off-the-shelf LLMs and VLMs, MAG-3D achieves state-of-the-art performance on 3D question-answering benchmarks while improving grounding-QA coherence.

---

[Structuring versus Problematizing: How LLM-based Agents Scaffold Learning in Diagnostic Reasoning](http://arxiv.org/abs/2604.09158)

- PharmaSim Switch: introduces a scenario-based learning environment that utilizes a Client Character, a Rule-Based Student Model Agent, a Pharmacist Agent, and an LLM-Based Student Model Agent to provide adaptive scaffolding for diagnostic reasoning.
- The system implements two distinct pedagogical scaffolding approaches, structuring and problematizing, to guide pharmacy apprentices through diagnostic tasks.
- Experimental results indicate that while both scaffolding approaches effectively support diagnostic strategy development, they foster distinct student engagement behaviors and interaction patterns.

---

[CORA: Conformal Risk-Controlled Agents for Safeguarded Mobile GUI Automation](http://arxiv.org/abs/2604.09155)

- CORA: introduces a post-policy, pre-action safeguarding framework that transforms open-ended mobile GUI agent action proposals into selective execution with statistical safety guarantees.
- The framework utilizes an action-conditional Guardian to estimate risk, Conformal Risk Control to calibrate an execute/abstain threshold, and a generative Diagnostician to provide interpretable interventions for rejected actions.
- CORA incorporates a Goal-Lock mechanism to maintain user intent against indirect prompt injection and is evaluated on the new Phone-Harm benchmark to demonstrate improved safety-helpfulness-interruption trade-offs.

---

[Interactive ASR: Towards Human-Like Interaction and Semantic Coherence Evaluation for Agentic Speech Recognition](http://arxiv.org/abs/2604.09121)

- Interactive ASR framework: introduces an agentic system that enables iterative refinement of recognition outputs through natural language feedback and semantic-aware evaluation.
- The framework utilizes an LLM-based Intent Router to determine if user input requires correction and an LLM-based Reasoning Corrector to perform surgical edits on transcripts.
- The authors propose a Sentence-level Semantic Error Rate (S2ER) metric, which leverages LLMs as judges to prioritize core intent and critical entities over literal word-level accuracy.

---

[Conversations Risk Detection LLMs in Financial Agents via Multi-Stage Generative Rollout](http://arxiv.org/abs/2604.09056)

- FinSec: introduces a four-tier security detection framework for financial agents that integrates Layer 1 (SAR-based structural compliance check), Layer 2 (Generative rollout for delayed risk), Layer 3 (Few-shot LLM semantic audit), and Layer 4 (Ensemble risk-based decision fusion) to identify multi-turn financial risks.
- The framework utilizes a Pattern Library (AML/SAR-based risk indicators) and an Adversarial Reasoning Framework (Three-perspective threat analysis) to systematically detect and quantify complex, multi-turn adversarial manipulations in financial dialogues.
- By employing a Semantic Discriminator (LLM-based intent classification), FinSec achieves high-precision risk assessment while maintaining operational utility and robustness against both injection and unintended financial risks.

---

[TriDeliver: Cooperative Air-Ground Instant Delivery with UAVs, Couriers, and Crowdsourced Ground Vehicles](http://arxiv.org/abs/2604.09049)

- TriDeliver: introduces a hierarchical cooperative framework for instant delivery that integrates UAVs, human couriers, and crowdsourced GVs to optimize delivery performance through a transfer learning-based assignment strategy.
- The framework utilizes a Shared Network to extract courier behavioral preferences, which are then adapted via FT Network I and FT Network II to guide task assignment for GVs and UAVs, respectively.
- TriDeliver employs a Specific Network to account for unique UAV flight characteristics, ensuring efficient parcel dispatch while minimizing operational costs and negative impacts on GV original tasks.

---

[V-CAGE: Vision-Closed-Loop Agentic Generation Engine for Robotic Manipulation](http://arxiv.org/abs/2604.09036)

- V-CAGE (Vision-Closed-Loop Agentic Generation Engine): introduces an agentic framework for autonomous robotic data synthesis that bridges high-level semantic reasoning with low-level physical interaction using OpenClaw Orchestrator Agent, Semantic Planning Module, Actionable Scene Generation, Image-Inpainting Model, Agentic Interaction Discovery, VLM Verifier, and Action-Aware Compression.
- The framework utilizes an Inpainting-Guided Scene Construction (IGSC) pipeline to ensure geometric and functional validity of generated environments while employing a VLM-based closed-loop verification mechanism to filter out silent failures in long-horizon trajectories.
- V-CAGE incorporates an action-aware perceptual compression algorithm that achieves over 90% storage reduction while maintaining visual fidelity for downstream VLA training.

---

[Advantage-Guided Diffusion for Model-Based Reinforcement Learning](http://arxiv.org/abs/2604.09035)

- AGD-MBRL: introduces a guided diffusion framework for model-based reinforcement learning that steers trajectory generation using advantage estimates to mitigate short-horizon myopia.
- The framework incorporates Sigmoid Advantage Guidance and Exponential Advantage Guidance to reweight trajectory sampling towards regions with higher expected long-term returns.
- By integrating advantage-aware guidance into a PolyGRAD-style architecture, the method improves sample efficiency and stability in continuous control tasks compared to unguided or reward-based diffusion baselines.

---

[Plasticity-Enhanced Multi-Agent Mixture of Experts for Dynamic Objective Adaptation in UAVs-Assisted Emergency Communication Networks](http://arxiv.org/abs/2604.09028)

- PE-MAMoE: introduces a MARL framework that couples sparsely gated mixture of experts with expert-only stochastic perturbations to maintain policy plasticity under abrupt, phase-driven non-stationarity in UAV-assisted emergency networks.
- The framework utilizes a non-parametric Phase Controller to orchestrate expert-level noise injection, router temperature scheduling, and entropy resets, ensuring rapid re-adaptation to shifting user mobility and demand objectives.
- By integrating conditional computation with targeted re-plasticization, the approach effectively mitigates representation collapse and dormant neuron accumulation, outperforming standard MARL baselines in safety, energy efficiency, and service capacity.

---

[Social Reality Construction via Active Inference: Modeling the Dialectic of Conformity and Creativity](http://arxiv.org/abs/2604.09026)

- Social Reality Construction via Active Inference: introduces a multi-agent simulation model that formalizes the bidirectional constitution of social reality through Generative Model, Discriminator, Memory Buffer, Social Network, and Metropolis–Hastings Naming Game (MHNG).
- The framework utilizes an adversarial relationship between variational free energy minimization for social conformity and expected free energy minimization for creative exploration.
- Simulation results demonstrate that the model enables the endogenous emergence of informationally cohesive social groups and the circular mutual constitution of social representations and creative artifacts.

---

[Generative AI Agent Empowered Power Allocation for HAP Propulsion and Communication Systems](http://arxiv.org/abs/2604.09015)

- Q3E (QoS-enhanced energy-efficient) beamforming algorithm: introduces a generative AI-empowered framework for HAP power allocation that integrates aerodynamic propulsion modeling with communication beamforming optimization.
- The framework utilizes a perception-brain-action loop to derive accurate propulsion power models via CFD and optimizes communication power using an ANN-based constrained training approach.
- Simulation results demonstrate that the proposed model reduces propulsion power deviation by 84.3% and achieves significant improvements in QoS satisfaction and energy efficiency compared to existing benchmarks.

---

[Hypergraph Neural Networks Accelerate MUS Enumeration](http://arxiv.org/abs/2604.09001)

- HyMUSE: introduces a domain-agnostic framework that accelerates MUS/MSS enumeration by utilizing a MUS/MCS Hypergraph to represent constraint relationships and an HGNN-based agent to guide the selection of constraints during shrink/grow operations.
- The framework employs reinforcement learning to train the HGNN-based agent to minimize the number of satisfiability oracle calls required to produce valid MUS/MSS solutions.
- Experimental results demonstrate that HyMUSE significantly improves the efficiency of MUS/MSS enumeration across various problem distributions and integrates effectively with existing enumeration algorithms.

---

[StreamMeCo: Long-Term Agent Memory Compression for Efficient Streaming Video Understanding](http://arxiv.org/abs/2604.09000)

- StreamMeCo: introduces an efficient training-free framework for compressing long-term agent memory in streaming video, utilizing EMsampling, EWpruning, and TMR to maintain performance while reducing memory overhead.
- The framework categorizes memory nodes into isolated and connected types, applying EMsampling for isolated text nodes and EWpruning for connected nodes to optimize memory graph storage.
- The TMR mechanism simulates human memory decay by dynamically allocating retrieval importance based on temporal segments, ensuring critical recent information is prioritized during LLM-based reasoning.

---

[ActFER: Agentic Facial Expression Recognition via Active Tool-Augmented Visual Reasoning](http://arxiv.org/abs/2604.08990)

- ActFER: introduces an agentic framework that reformulates facial expression recognition as an active visual evidence acquisition process followed by AU-grounded multimodal reasoning.
- The framework utilizes a Tool Library, including Face Detection-Alignment and Zoom-In tools, to dynamically prepare analyzable facial evidence instead of relying on fixed external inputs.
- ActFER is optimized via UC-GRPO, which employs AU-grounded dense rewards, query-conditional contrastive utility estimation, and emotion-wise EMA calibration to learn when local inspection is beneficial for accurate affect understanding.

---

[SEA-Eval: A Benchmark for Evaluating Self-Evolving Agents Beyond Episodic Assessment](http://arxiv.org/abs/2604.08988)

- SEA (Self-Evolving Agent): introduces a formal paradigm and benchmark for evaluating agents that transition from episodic task execution to continuous capability evolution through a dual-hub architecture.
- The framework utilizes an Evolutionary Flywheel, comprising an Execution Hub and a Cognition Hub, to transform raw interaction trajectories into persistent cognitive assets.
- SEA-Eval quantifies evolutionary performance by tracking the monotonic convergence of token consumption across sequential task streams, distinguishing genuine evolution from pseudo-evolution.

---

[PilotBench: A Benchmark for General Aviation Agents with Safety Constraints](http://arxiv.org/abs/2604.08987)

- PilotBench: introduces a comprehensive benchmark for evaluating LLMs on safety-critical flight trajectory and attitude prediction tasks across nine distinct flight phases.
- The research identifies a Precision-Controllability Dichotomy where traditional forecasters excel in numerical accuracy while LLMs demonstrate superior instruction-following capabilities.
- A Dynamic Complexity Gap is observed, showing that LLM performance degrades significantly in high-workload flight phases, motivating hybrid architectures that combine LLM reasoning with specialized numerical forecasters.

---

[Multi-agent Reinforcement Learning for Low-Carbon P2P Energy Trading among Self-Interested Microgrids](http://arxiv.org/abs/2604.08973)

- MMAPPO: introduces a multi-agent reinforcement learning framework for decentralized P2P electricity trading among self-interested microgrids using CTDE and LSTM-based temporal feature extraction.
- The framework utilizes an MRDAC mechanism to coordinate bidding strategies, enabling microgrids to optimize profit while balancing renewable energy supply and demand.
- Simulation results demonstrate that the MMAPPO-driven approach improves renewable energy utilization and economic welfare compared to baseline MARL methods and alternative market-clearing mechanisms.

---

[LITMUS (RE)AGENT: A Benchmark and Agentic System for Predictive Evaluation of Multilingual Models](http://arxiv.org/abs/2604.08970)

- LITMUS (RE)AGENT: introduces a DAG-orchestrated agentic system for predictive multilingual evaluation that decomposes queries into hypotheses, retrieves citation-grounded evidence, and synthesises predictions through feature-aware aggregation.
- The system utilizes specialized agents including ThoughtCreatorAgent, ThoughtAgent, Research Planner, Web Search and Crawl, Expert Knowledge, Coder, Reporter, ThoughtAnalyzerAgent, and ResponseAnalyzerAgent to perform structured reasoning under incomplete evidence.
- The framework is evaluated on a new controlled benchmark of 1,500 questions across six tasks and five evidence scenarios, demonstrating superior performance in transfer-heavy settings compared to non-DAG and single-agent baselines.

---

[Aligned Agents, Biased Swarm: Measuring Bias Amplification in Multi-Agent Systems](http://arxiv.org/abs/2604.08963)

- MAS: introduces a systematic empirical study demonstrating that architectural complexity in Multi-Agent Systems frequently exacerbates rather than mitigates bias amplification through iterative feedback loops.
- The paper presents Discrim-Eval-Open, a benchmark designed to bypass LLM performative neutrality by forcing comparative judgments across sensitive demographic attributes.
- Experimental results reveal that neither diverse agent personas, specialized functional roles, nor complex communication topologies prevent systemic polarization, with a identified 'Trigger Vulnerability' where neutral external context accelerates bias.

---

[Enhancing LLM Problem Solving via Tutor–Student Multi-Agent Interaction](http://arxiv.org/abs/2604.08931)

- PETITE (Peer Tutoring Inspired Token-Efficient): introduces a multi-agent framework for code generation that utilizes asymmetric roles, specifically a Student/Coder Agent and a Tutor/Helper Agent, to improve performance through structured, iterative feedback.
- The framework employs an Early Stopping Mechanism that monitors the Tutor/Helper Agent's feedback to terminate the process once a correct solution is identified, thereby optimizing token usage.
- By leveraging role differentiation and serial interaction, PETITE achieves competitive accuracy on the APPS benchmark while significantly reducing computational costs compared to symmetric multi-agent systems.

---

[Beyond the Individual: Virtualizing Multi-Disciplinary Reasoning for Clinical Intake via Collaborative Agents](http://arxiv.org/abs/2604.08927)

- Aegle: introduces a multi-agent framework that virtualizes Multi-Disciplinary Team reasoning for clinical intake using an Orchestrator, Specialist Agents, an Aggregator, a Structured Clinical State, Dialogue History, and an Integrated Patient Note.
- The framework employs a state-aware dynamic topology to activate specialist agents on-demand, ensuring decoupled parallel reasoning and evidence-grounded clinical documentation.
- Aegle enforces a strict separation between evidence acquisition and diagnostic synthesis to mitigate cognitive biases and improve the traceability of clinical decisions.

---

[Beyond Relevance: Utility-Centric Retrieval in the LLM Era](http://arxiv.org/abs/2604.08920)

- Utility-Centric Retrieval Framework: introduces a paradigm shift in information retrieval by optimizing for LLM-centric utility rather than traditional topical relevance.
- The framework categorizes retrieval objectives into LLM-agnostic versus LLM-specific utility and context-independent versus context-dependent utility to align retrieval with downstream generation quality.
- It synthesizes recent advances in agentic RAG and utility modeling to provide a roadmap for designing retrieval systems that satisfy the latent knowledge needs of LLMs.

---

[Omakase: proactive assistance with actionable suggestions for evolving scientific research projects](http://arxiv.org/abs/2604.08898)

- Omakase: introduces a proactive research assistant that monitors evolving project documents to infer information needs and deliver contextualized, actionable literature-based suggestions.
- The system leverages a pipeline comprising a Document Parser, LLM-based Inference Engine, and a Deep Research System to generate timely, project-specific insights without requiring manual query formulation.
- Evaluations demonstrate that Omakase provides significantly more actionable and relevant suggestions compared to traditional deep research outputs, effectively reducing cognitive load for researchers.

---

[GeoMMBench and GeoMMAgent: Toward Expert-Level Multimodal Intelligence in Geoscience and Remote Sensing](http://arxiv.org/abs/2604.08896)

- GeoMMAgent: introduces a multi-agent framework that integrates LLMs with domain-specific tools to achieve expert-level performance in geoscience and remote sensing tasks.
- The framework utilizes a plan-execute-evaluate paradigm, incorporating a Coordinator, Retrieval Agent, Perception Agent, Reasoning Agent, and Self-Evaluation Agent to manage complex geospatial queries.
- GeoMMAgent leverages a modular Tool Library, including Knowledge Toolkit, Perception Toolkit, Reasoning Toolkit, and General Toolkit, to provide specialized capabilities for robust multimodal interpretation.

---

[ParseBench: A Document Parsing Benchmark for AI Agents](http://arxiv.org/abs/2604.08538)

- ParseBench: introduces a comprehensive benchmark for evaluating document parsing quality across five critical dimensions: tables, charts, content faithfulness, semantic formatting, and visual grounding.
- The framework utilizes a two-pass annotation pipeline, combining frontier LLMs for initial labeling with human-in-the-loop verification to ensure high-quality ground truth for enterprise documents.
- The benchmark employs semantic correctness metrics, such as TABLERECORDMATCH and CHARTDATAPOINTMATCH, to evaluate parsing performance based on downstream utility rather than surface-level text similarity.

---

[KnowU-Bench: Towards Interactive, Proactive, and Personalized Mobile Agent Evaluation](http://arxiv.org/abs/2604.08455)

- KnowU-Bench: introduces a comprehensive evaluation framework for personalized and proactive mobile agents using a reproducible Android emulation environment.
- The framework utilizes an LLM-driven user simulator and a hybrid evaluation pipeline to assess agent performance across general, personalized, and proactive task categories.
- Experimental results demonstrate that while LLMs excel at explicit task execution, they struggle significantly with preference acquisition and proactive intervention calibration.

---

[Don’t Overthink It: Inter-Rollout Action Agreement as a Free Adaptive-Compute Signal for LLM Agents](http://arxiv.org/abs/2604.08369)

- TrACE (Trajectorical Adaptive Compute via agrEement): introduces a training-free controller that adaptively allocates LLM calls by measuring inter-rollout action agreement to optimize inference compute.
- The framework utilizes an LLM Agent to sample candidate actions, a Controller to evaluate agreement against a threshold, and an Action Canonicaliser to ensure consistent comparison of outputs.
- By dynamically adjusting the number of LLM calls per timestep based on behavioral consistency, the approach reduces total compute requirements while maintaining performance parity with fixed-budget self-consistency methods.

---

[ASPECT: Analogical Semantic Policy Execution via Language Conditioned Transfer](http://arxiv.org/abs/2604.08355)

- ASPECT: introduces a zero-shot transfer framework that leverages an LLM as a semantic operator to remap target task observations into source-aligned descriptions for a text-conditioned VAE.
- The framework enables RL agents to reuse pre-trained policies on novel tasks by "imagining" target states in the familiar semantic context of the source environment.
- By disentangling structural features from semantic content, the model achieves robust generalization across diverse environments and unseen objects without requiring additional training in target domains.

---

[HyperMem: Hypergraph Memory for Long-Term Conversations](http://arxiv.org/abs/2604.08256)

- HyperMem: introduces a hierarchical hypergraph memory architecture that explicitly models high-order associations using hyperedges to unify scattered conversational content into coherent units.
- The framework organizes memory into Topic nodes, Episode nodes, and Fact nodes, utilizing hyperedges to link related elements and facilitate efficient coarse-to-fine retrieval.
- HyperMem leverages lexical-semantic indexing and embedding propagation to achieve state-of-the-art performance on the LoCoMo benchmark, effectively capturing long-term dependencies in conversational agents.

---

[Governed Capability Evolution for Embodied Agents: Safe Upgrade, Compatibility Checking, and Runtime Rollback for Embodied Capability Modules](http://arxiv.org/abs/2604.08059)

- Governed Capability Evolution framework: introduces a lifecycle-aware upgrade governance pipeline that treats new capability versions as deployment candidates requiring multi-dimensional compatibility validation before activation.
- The framework utilizes four compatibility dimensions—interface, policy, behavioral, and recovery—to ensure that capability upgrades do not compromise system safety or stability.
- By integrating staged evaluation, shadow deployment, and automated rollback, the approach enables safe, long-term capability growth for embodied agents without requiring agent rewrites.

---

[MONETA: Multimodal Industry Classification through Geographic Information with Multi Agent Systems](http://arxiv.org/abs/2604.07956)

- MONETA: introduces a multimodal benchmark for industry classification that leverages geospatial and textual resources to categorize businesses according to NACE guidelines.
- The framework utilizes Zero-Shot and Multi-Turn pipelines, where specialized agents extract clues from OpenStreetMap, satellite imagery, and textual sources to inform a decision-making agent.
- The study provides quantitative metrics for intermediate agent performance and demonstrates the robustness of the approach against fine-tuned models in dynamic classification environments.

---

[Harnessing Embodied Agents: Runtime Governance for Policy-Constrained Execution](http://arxiv.org/abs/2604.07833)

- Runtime Governance Framework: introduces a systems-level architecture that separates agent cognition from execution oversight to ensure policy-constrained, observable, and recoverable embodied agent behavior.
- The framework utilizes a dedicated Runtime Governance Layer containing Capability Admission, Policy Guard, Execution Watcher, Recovery and Rollback Manager, Human Override Interface, and Audit and Telemetry components.
- By externalizing governance from the agent loop, the system enables environment-sensitive policy enforcement and structured failure recovery without requiring modifications to the underlying agent model.

---

[Towards Knowledgeable Deep Research: Framework and Benchmark](http://arxiv.org/abs/2604.07720)

- HKA (Hybrid Knowledge Analysis framework): introduces a multi-agent architecture that reasons over both structured and unstructured knowledge to generate comprehensive multimodal reports.
- The framework utilizes a Planner to manage subtasks, a Structured Knowledge Analyzer for quantitative computation via code and vision-language models, and an Unstructured Knowledge Analyzer for web-based information retrieval.
- To support systematic evaluation, the authors construct KDR-Bench, which includes 41 expert-level questions and a knowledge-enhanced evaluation framework incorporating general-purpose, knowledge-centric, and vision-enhanced metrics.

---

[How Much LLM Does a Self-Revising Agent Actually Need? Empirical Decomposition of World Modeling, Reflection, and Sparse LLM Revision](http://arxiv.org/abs/2604.07236)

- Declared Reflective Runtime Protocol: introduces a methodological framework that decomposes agent competence into explicit, inspectable layers to isolate the marginal contribution of LLMs from structured world modeling and symbolic reflection.
- The architecture replaces opaque LLM-based reasoning loops with a declarative runtime that manages belief tracking, planning, and guarded symbolic reflection.
- Empirical results demonstrate that while explicit world-model planning provides significant performance gains, symbolic reflection and sparse LLM intervention exhibit non-monotonic effects that require careful calibration.

---

[Exploiting Aggregate Programming in a Multi-Robot Service Prototype](http://arxiv.org/abs/2604.06876)

- AP: introduces a resilient multi-robot coordination prototype that leverages Aggregate Programming to manage task assignment and robot navigation through a distributed, self-stabilizing architecture.
- The system integrates an AP Engine with ROS2-based components to handle dynamic task allocation, robot telemetry, and motion control in both simulated and physical environments.
- The framework utilizes collective consensus operators and aggregate processes to ensure robust performance despite network partitions, robot failures, and dynamic task arrivals.

---

[HealthAdminBench: Evaluating Computer-Use Agents on Healthcare Administration Tasks](http://arxiv.org/abs/2604.09937)

- HealthAdminBench: introduces a benchmark for evaluating LLM-based computer-use agents on complex, multi-step healthcare administrative workflows using Agent, Environment, Observation Space, Action Space, Memory, and Evaluator.
- The framework utilizes four deterministic web environments to simulate real-world administrative systems, requiring agents to perform tasks like prior authorization and claims management.
- Experimental results demonstrate a significant performance gap between individual subtask completion and end-to-end reliability, highlighting the challenges of long-horizon, cross-system coordination for LLMs.

---

[Automating Structural Analysis Across Multiple Software Platforms Using Large Language Models](http://arxiv.org/abs/2604.09866)

- Multi-agent architecture for automated structural analysis: introduces a two-stage multi-agent framework that leverages LLMs to automate frame structural analysis across diverse FEA platforms by decoupling semantic reasoning from platform-specific code generation.
- Stage 1 utilizes a cohort of specialized agents to interpret user input and produce a unified, platform-agnostic JSON representation of the structural model.
- Stage 2 employs parallel translation agents, including a semantic mapping agent for ETABS, to convert the JSON representation into executable scripts for OpenSees, SAP2000, and ETABS with high reliability.

---

[Instructing LLMs to Negotiate using Reinforcement Learning with Verifiable Rewards](http://arxiv.org/abs/2604.09855)

- RLVR (Reinforcement Learning from Verifiable Rewards): introduces a reinforcement learning framework that trains LLMs to negotiate by grounding reward signals in objective economic surplus and constraint satisfaction rather than subjective human preferences.
- The framework utilizes a structured action space integrating internal reasoning, persuasive dialogue, and formal economic commitments to enable verifiable strategic decision-making.
- Experimental results demonstrate that the trained agent develops a four-phase strategic evolution, significantly outperforming larger frontier models in surplus extraction and robustness against adversarial seller personas.

---

[ProGAL-VLA: Grounded Alignment through Prospective Reasoning in Vision-Language-Action Models](http://arxiv.org/abs/2604.09824)

- ProGAL-VLA: introduces a hierarchical architecture that enforces explicit grounding verification before action execution to mitigate language ignorance in robotic agents.
- The framework utilizes a Prospective Planner and a Grounded State Module to generate symbolic sub-goals and 3D entity representations, which are aligned via a State Alignment Cross Attention mechanism.
- By conditioning the Action Policy exclusively on verified goal embeddings, the model achieves robust performance under visual perturbations and enables calibrated ambiguity detection through attention entropy.

---

[Agentic Workflows for Resolving Conflict Over Shared Resources: A Power Grid Application](http://arxiv.org/abs/2604.09823)

- AWR-CSR: introduces a domain-agnostic framework for coordinating multiple LLM-based agents to resolve conflicts over shared resources using bilateral negotiation, structured mediation, or procedural deconfliction.
- The framework utilizes Client Agents that encapsulate application logic and employ a chain-of-thought reasoning process to negotiate setpoints while preserving privacy and local autonomy.
- Performance evaluation on a power grid case study demonstrates that the proposed deconfliction modes consistently achieve mutually beneficial outcomes compared to baseline centroid-based approaches.

---

[EE-MCP: Self-Evolving MCP-GUI Agents via Automated Environment Generation and Experience Learning](http://arxiv.org/abs/2604.09815)

- EE-MCP: introduces a self-evolving framework for hybrid computer-use agents that optimizes the interplay between structured API calls and GUI actions through iterative policy refinement.
- The framework utilizes a closed-loop pipeline comprising Claude Expert, Agent, LLM Judge, Performance Profile, Task Generator, Experience Bank, and SFT Training to autonomously improve agent performance without manual intervention.
- By leveraging application-aware mechanism selection, the system applies trajectory distillation for MCP-dominant tasks and experience augmentation for GUI-intensive tasks to achieve significant performance gains.

---

[Controllable and Verifiable Tool-Use Data Synthesis for Agentic Reinforcement Learning](http://arxiv.org/abs/2604.09813)

- COVERT: introduces a two-stage pipeline that transforms reliable base tool-use trajectories into oracle-preserving synthetic environments for robust reinforcement learning.
- The framework utilizes an Augmenter to systematically introduce perturbations like distractor tools and noisy outputs, while maintaining ground-truth oracle metadata for automated reward computation.
- Empirical results demonstrate that COVERT-RL significantly improves tool-use robustness under ambiguity and unreliable feedback while preserving general-domain capabilities.

---

[Building an Internal Coding Agent at Zup: Lessons and Open Questions](http://arxiv.org/abs/2604.09805)

- CodeGen: introduces a three-tier agentic system designed for enterprise software development that prioritizes robust tool design, layered safety guardrails, and progressive human oversight over simple prompt engineering.
- The architecture utilizes a CLI executor for client-side tool execution, a FastAPI backend for request routing, and a centralized Maestro orchestrator to manage the iterative ReAct-based agentic loop.
- The paper demonstrates that engineering decisions regarding tool specification, state management, and trust calibration are more critical for production readiness than the underlying LLM model performance.

---

[On Feedback Speed Control for a Planar Tracking](http://arxiv.org/abs/2604.09795)

- Feedback Speed Control for Planar Tracking: introduces a cascade feedback control strategy that couples a novel speed control law with a constant bearing steering law to maintain an abreast formation between a leader and a follower agent.
- The framework utilizes shape dynamics to achieve asymptotic stability of the formation when the leader's steering is known and ensures input-to-state stability when the leader's steering is unknown.
- The approach is validated through numerical simulations and robotic experiments, and is extended to an N-agent chain network to model wave-like information propagation in collectives.

---

[Pioneer Agent: Continual Improvement of Small Language Models in Production](http://arxiv.org/abs/2604.09791)

- Pioneer Agent: introduces a closed-loop system for autonomous adaptation of SLMs in production using Orchestrator LLM, LangGraph, Tinker SDK, Sub-Agents, Tools, Context Manager, Sandbox, and External LLM APIs.
- The system automates the full lifecycle of model improvement by diagnosing inference failures, synthesizing targeted training curricula, and performing iterative retraining with regression constraints.
- Pioneer Agent employs agent-guided iterative search, including Monte Carlo Graph Search, to jointly optimize training data, hyperparameters, and learning strategies for both cold-start and production-time adaptation.

---

[Text-Guided 6D Object Pose Rearrangement via Closed-Loop VLM Agents](http://arxiv.org/abs/2604.09781)

- Closed-Loop VLM Agents: introduces a training-free framework that utilizes a VLM as an iterative agent to refine 6D object poses based on text instructions and visual feedback.
- The framework alternates between an evaluator that assesses scene faithfulness and a proposer that predicts incremental 6D pose updates to achieve the desired goal state.
- Three inference-time techniques—multi-view reasoning, object-centered coordinate system visualization, and single-axis rotation prediction—are employed to enhance the spatial reasoning capabilities of the VLM.

---

[Event-Driven Temporal Graph Networks for Asynchronous Multi-Agent Cyber Defense in NetForge_RL](http://arxiv.org/abs/2604.09523)

- CT-GMARL (Continuous-Time Graph Multi-Agent Reinforcement Learning): introduces a spatial-temporal graph neural architecture designed to navigate asynchronous, continuous-time cyber defense environments by decoupling topological constraints from irregular telemetry streams.
- The framework utilizes a dual-mode engine, NetForge_RL, which bridges the Sim2Real gap by training in a high-throughput MockHypervisor and performing zero-shot evaluation in a live DockerHypervisor.
- CT-GMARL integrates Multi-Head GAT for spatial reasoning and Neural ODE-RNN for continuous temporal dynamics, enabling robust defense against non-stationary adversarial threats in dynamic network topologies.

---

[Conflicts Make Large Reasoning Models Vulnerable to Attacks](http://arxiv.org/abs/2604.09750)

- Conflict Injection Framework: introduces a systematic method to expose vulnerabilities in LLMs by injecting internal conflicts and moral dilemmas into prompts to force hazardous reasoning.
- The framework utilizes layerwise and neuron-level analysis to demonstrate how conflict injection causes functional reasoning subspaces to overlap with or dominate safety-related representations.
- Empirical results across five benchmarks show that conflict injection significantly increases the attack success rate of LLMs by inducing representational interference during the reasoning process.

---

[CONSCIENTIA: Can LLM Agents Learn to Strategize? Emergent Deception and Trust in a Multi-Agent NYC Simulation](http://arxiv.org/abs/2604.09746)

- CONSCIENTIA: introduces a controlled multi-agent simulation environment to study emergent strategic behavior, deception, and trust in LLMs under adversarial navigation conditions.
- The framework utilizes an iterative alignment pipeline with Kahneman-Tversky Optimization (KTO) to update agent policies based on trajectory-level outcomes rather than step-wise rewards.
- Empirical results demonstrate that while iterative alignment improves task success and selective cooperation, LLMs remain highly vulnerable to persistent, socially-framed adversarial manipulation.

---

[MPAC: A Multi-Principal Agent Coordination Protocol for Interoperable Multi-Agent Collaboration](http://arxiv.org/abs/2604.09744)

- MPAC (Multi-Principal Agent Coordination Protocol): introduces an application-layer protocol designed to coordinate autonomous agents from independent principals over shared state using Session Layer, Intent Layer, Operation Layer, Conflict Layer, and Governance Layer.
- The framework utilizes a central Coordinator to enforce total order via a Lamport clock, ensuring causal traceability and structured conflict resolution across heterogeneous agent systems.
- MPAC provides machine-enforceable wire semantics through JSON Schema definitions, enabling interoperability and significant reductions in coordination overhead for multi-agent workflows.

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
