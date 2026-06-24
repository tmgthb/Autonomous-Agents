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






#### 23rd June 2026

[Qwen-AgentWorld: Language World Models for General Agents](http://arxiv.org/abs/2606.24597)

- Qwen-AgentWorld: introduces a native language world model trained via a three-stage pipeline to simulate agentic environments across seven domains.
- The framework utilizes CPT, SFT, and RL to develop a world model that serves as either a decoupled environment simulator or a unified agent foundation model.
- Evaluation is performed using AgentWorldBench, which employs an LLM judge and rule-based verifiers to assess simulation fidelity across five dimensions.

---

[MEMPROBE: Probing Long-Term Agent Memory via Hidden User-State Recovery](http://arxiv.org/abs/2606.24595)

- MEMPROBE: introduces a benchmark that evaluates LLM agent memory as an auditable post-interaction artifact by reconstructing hidden user-state dimensions from the memory store.
- The framework utilizes a synthetic user simulator with a hidden, taxonomy-anchored bank to test whether agents can consolidate interaction evidence into recoverable user-state claims.
- Experimental results demonstrate that task success is an insufficient metric for memory quality, as agents often fail to form compact, retrievable user models despite successful task completion.

---

[Governed Shared Memory for Multi-Agent LLM Systems](http://arxiv.org/abs/2606.24535)

- MemClaw: introduces a systems architecture for governed shared memory in multi-agent LLM environments, addressing challenges in scoped access, temporal correctness, provenance, and policy-controlled propagation.
- The architecture formalizes the fleet-memory problem and identifies four failure modes: unauthorized leakage, stale propagation, contradiction persistence, and provenance collapse.
- The authors instantiate the architecture in MemClaw and validate it using ArgusFleet, an evaluation harness that exercises the memory service against the identified governance failure modes.

---


[Are We Ready For An Agent-Native Memory System?](http://arxiv.org/abs/2606.24775)

- Agent-Native Memory System Framework: introduces a systematic data management perspective for LLM agent memory by decomposing it into four core modules: memory representation and storage, extraction, retrieval and routing, and maintenance.
- The framework evaluates 12 representative memory systems across five benchmark workloads to quantify architectural trade-offs in representation fidelity, retrieval precision, update correctness, and long-horizon stability.
- Experimental findings reveal that no single architecture dominates all scenarios, and that cost-efficient maintenance is best achieved through localized updates rather than global reorganization.

---

[World Models in Pieces: Structural Certification for General Agents](http://arxiv.org/abs/2606.24842)

- Structural Certification for General Agents: introduces a transition-local framework that maps bounded goal-conditioned performance to entry-wise guarantees on an agent's internal world model.
- The framework utilizes filtering algorithms to isolate specific transitions where an agent's predictive world model is provably reliable, moving beyond unattainable universal performance guarantees.
- By certifying performance on transition-specific LTL goals, the approach enables the safe deployment of general agents by identifying high-fidelity regions within their internal world models.

---


[Data Recipes for Agentic Models](http://arxiv.org/abs/2606.24855)

- OT-Agent: introduces a comprehensive, open-source data curation pipeline for training agentic models through systematic ablation of six key stages.
- The framework utilizes Source Tasks, Mix Tasks, Filter Tasks, Generate Rollouts, Filter Rollouts, and Select Teacher to optimize training data for LLMs.
- The project demonstrates that high-quality agentic training data, specifically longer trajectories and diverse task sources, significantly improves LLM performance across multiple agentic benchmarks.

---

[Beyond Bayer: Task-Optimal Sensor Co-Design for Robust Autonomous-Driving Segmentation](http://arxiv.org/abs/2606.24096)

- Task-Optimal Spectral Mosaic Co-Design Pipeline: introduces a differentiable RAW-to-task framework that optimizes spectral CFA weights while maintaining an identity PSF to maximize task-relevant information for dense prediction.
- The framework demonstrates that for dense semantic segmentation, spectral CFA learning provides significant performance gains, whereas PSF co-design is net-negative due to the data-processing inequality.
- Empirical results show that a 2×2 CFA tile is optimal, as larger tiles fail to provide additional spectral information while degrading spatial resolution, confirming the rank-three constraint of sRGB inputs.

---

[“Zooming In” on Agentic Web Browsers as Assistive Technologies: A Case Study with a Low-Vision Technology Expert](http://arxiv.org/abs/2606.24870)

- AWB: introduces a case study evaluating the efficacy of LLM-powered web agents as assistive technologies for users with visual impairments.
- The framework utilizes an LLM-based agent to interpret DOM structures and execute autonomous web navigation tasks via natural language instructions.
- Findings highlight that while AWBs offer high conversational fluidity, they currently lack sufficient non-visual feedback and user control mechanisms necessary for inclusive assistive technology.

---


[When Helpfulness Overrides Causal Caution: Context-Dependent Suppression and Recovery in LLMs](http://arxiv.org/abs/2606.24370)

- Causal Caution Framework: introduces a systematic evaluation of LLMs' propensity to withhold causal judgment when empirical evidence is insufficient, utilizing an LLM-as-a-Judge and a PCH scoring rubric to measure context-dependent suppression.
- The study demonstrates that practical advisory contexts significantly suppress Causal Caution in LLMs compared to academic contexts, revealing a systematic shift in response patterns driven by pragmatic pressure.
- Experimental results show that a minimal self-correction prompt effectively restores Causal Caution, suggesting that the observed decline is a reversible suppression of expression rather than a fundamental capability limitation.

---


[Grading the Grader: Lessons from Evaluating an Agentic Data Analysis System](http://arxiv.org/abs/2606.24839)

- LAMBDA: introduces a multi-layer evaluation pipeline for agentic data analysis systems that combines automated grading with human calibration to disentangle agent performance from grading artifacts.
- The framework utilizes a two-loop wrapper with per-turn instrumentation, incorporating a Programmer-, Inspector- and Nudge-agent to manage conversational outputs and ensure gradable responses.
- The evaluation cascade employs a Strict grader, a Lenient grader, and Human inspection to achieve high precision and recall while addressing the challenges of verbose, multi-step agentic outputs.

---

[Accuracy and Satisfaction in Multi-Turn LLM Dialogues for NFR Assessment](http://arxiv.org/abs/2606.24834)

- PARADISE (PARAdigm for DIalogue System Evaluation): introduces a methodology to evaluate the accuracy and quality of multi-turn dialogues between developers and an LLM-based Agent for assessing Non-Functional Requirements (NFRs) in the iTrust codebase.
- The study utilizes an Evaluation Tool and Ground Truth to compare LLM-based Agent outputs against expert assessments, while employing a Satisfaction Survey and Annotation Process to model user satisfaction based on dialogue characteristics.
- Findings indicate that while developers perceive the LLM-based Agent responses as high-quality, the actual accuracy against expert ground truth is low, with proactive interactions positively influencing user satisfaction and verbose responses negatively impacting it.

---

[Virtual Simulation for Mental Health](http://arxiv.org/abs/2606.24826)

- Virtual Simulation for Mental Health: introduces a human-centered framework leveraging Agent-Based Modeling, VR, AR, and LLMs to create safe, controlled environments for mental health experimentation and self-care practice.
- The research utilizes Agent-Based Modeling to evaluate matching protocols in online mental health communities and employs VR/AR with LLMs to provide immersive, low-risk spaces for individuals to rehearse stress-management skills.
- By combining empirical user needs with simulation-based testing, the dissertation advocates for proactive, safe experimentation to improve mental health technology without disrupting real-world support systems.

---

[SHERLOC: Structured Diagnostic Localization for Code Repair Agents](http://arxiv.org/abs/2606.24820)

- SHERLOC: introduces a training-free framework that pairs a Reasoning LLM with a compact suite of LLM-friendly Tools and a Self-Recovery Layer to perform structured diagnostic localization for code repair.
- The framework utilizes a Tool Executor to mediate repository access, enabling the Reasoning LLM to generate diagnostic findings including root-cause analysis and solution guidance.
- SHERLOC improves downstream code repair resolve rates and reduces token consumption by providing actionable diagnostic context to repair agents.

---

[MANGO: Automated Multi-Agent Test Oracle Generation for Vision-Language-Action Models](http://arxiv.org/abs/2606.24815)

- MANGO: introduces a multi-agent framework that automatically generates fine-grained, executable test oracles for VLA-enabled robots by decomposing natural-language instructions into structured sequences of atomic tasks.
- The framework utilizes a collaborative architecture consisting of Generator, Assessor, and Judge agents that iteratively refine generated artifacts through structured feedback to ensure correctness and simulator compliance.
- MANGO enables precise failure localization and richer diagnostic information for long-horizon robotic tasks by replacing monolithic symbolic oracles with compositional, fine-grained oracle definitions.

---

[Paying to Know: Micro-Transaction Markets for Verified Product Information in Agentic E-Commerce](http://arxiv.org/abs/2606.24783)

- Agentic E-Commerce Market Framework: introduces a micro-transaction architecture where LLM-based buyer agents purchase verified product information from sellers and reviewers using programmable payment rails.
- The framework shifts the focus of applied NLP from catalogue ranking to cost-aware tool use, negotiation dialogue, and verifiable data acquisition.
- This approach incentivizes genuine product quality and competition by treating information as a priced, attestable asset rather than relying on unverified marketing copy.

---


[SupplyNet: Supporting Visual Exploratory Learning in Supply Chain via Contextual Multi-Agent Simulation](http://arxiv.org/abs/2606.24694)

- SupplyNet: introduces a gamified visual simulation system that leverages a contextual graph-based LLM multi-agent framework to model interdependent supply chain dynamics.
- The system integrates an interactive network view, a branching timeline for counterfactual exploration, and a task-oriented analysis console to provide responsive feedback and support structured performance breakdowns.
- SupplyNet utilizes LLM agents to simulate human-like business managers, enabling learners to trace causal propagation and test decision strategies within a manipulable decision space.

---

[Automated Summarization of Software Documents: An LLM-based Multi-Agent Approach](http://arxiv.org/abs/2606.24689)

- Metagente: introduces a collaborative multi-agent system for software documentation summarization that utilizes an Extractor Agent, Summarizer Agent, Teacher Agent, and Prompt Creator Agent to iteratively refine prompts via a Teacher-Student architecture.
- The framework employs a dynamic iteration strategy to optimize computational efficiency by halting processing for low-potential samples while maintaining high-quality summary generation.
- Empirical evaluation demonstrates that Metagente consistently outperforms single LLM-based agents across diverse datasets, achieving superior semantic alignment and ROUGE scores through structured agent collaboration.

---

[Agentic Collaborative Cognition for Zero-Shot 3D Understanding](http://arxiv.org/abs/2606.24649)

- Agentic Collaborative Cognition framework: introduces a multi-agent system that reformulates zero-shot 3D understanding as an iterative planning-perception process using a Planning Agent, a Perception Agent, and a shared Holistic Cognitive Map.
- The framework utilizes a Planning Agent to perform flexible viewpoint selection and a Perception Agent to document object attributes and refine the Holistic Cognitive Map through closed-loop feedback.
- By integrating spatial and semantic information into a shared state, the agents collaboratively resolve 3D scene understanding tasks, including visual grounding, situation estimation, and question answering, with state-of-the-art performance.

---

[SAFARI: Scaling Long Horizon Agentic Fault Attribution via Active Investigation](http://arxiv.org/abs/2606.24626)

- SAFARI: introduces a tool-augmented diagnostic loop that replaces linear context loading to perform fault attribution on long-horizon agentic trajectories.
- The framework utilizes an Investigator Agent that interacts with trajectory traces through iterative tool calls and a persistent Short-Term Memory to maintain diagnostic coherence.
- SAFARI decouples diagnostic accuracy from context limits, enabling effective fault attribution even when target faults reside significantly beyond the native context window of LLMs.

---

[Privacy-Preserving RAG via Multi-Agent Semantic Rewriting: Achieving Confidentiality Without Compromising Contextual Fidelity](http://arxiv.org/abs/2606.24623)

- Multi-Agent Semantic Rewriting Framework: introduces a multi-agent pipeline that sanitizes retrieved documents by extracting privacy-sensitive segments via Pri-Extra Agent, identifying core semantic content through Sem-Extra Agent, and generating safe text using Reconstruction Agent.
- The framework utilizes an Asymmetric Retrieval Architecture and Dual-Track Storage Mechanism to physically isolate raw private identifiers from the generation LLM, ensuring zero online inference latency.
- By employing deterministic conflict routing, the system effectively balances data confidentiality and contextual fidelity, outperforming existing privacy-preserving baselines in factual consistency across medical and enterprise datasets.

---

[ASALT: Adaptive State Alignment for Lateral Transfer in Multi-agent Reinforcement Learning](http://arxiv.org/abs/2606.24601)

- ASALT: introduces a multi-agent reinforcement learning framework that utilizes Observation adapter, State adapter, and Transfer module to enable effective knowledge transfer across domains with mismatched state-space dimensionalities.
- The framework employs hierarchical multi-head attention and transformer-based encoders to map heterogeneous target-domain inputs into a shared embedding space, facilitating lateral transfer from frozen Source actor and Source critic components to Target actor and Target critic agents.
- By jointly training adapters and target agents, ASALT mitigates negative transfer and improves sample efficiency in cooperative multi-agent environments compared to existing baseline methods.

---



[NatureBench: Can Coding Agents Match the Published SOTA of Nature-Family Papers?](http://arxiv.org/abs/2606.24530)

- NatureBench: introduces a cross-discipline benchmark of 90 tasks distilled from Nature-family publications to evaluate whether AI coding agents can move beyond reproduction toward scientific discovery.
- The framework utilizes NatureGym to construct standardized, containerized environments while enforcing an information firewall that forces agents to discover solutions rather than reproduce existing methods.
- Evaluation of ten frontier LLM agents reveals that success is primarily driven by methodological translation into supervised prediction problems, with failures largely attributed to suboptimal method choice and insufficient compute budgets.

---

[AGORA: An Archive-Grounded Benchmark for Agentic Workplace Document Reasoning](http://arxiv.org/abs/2606.24526)

- AGORA (Archive-Grounded Office Reasoning Assessment): introduces a benchmark for evaluating LLM agents on archive-grounded reasoning tasks across eight professional domains using a multi-stage construction pipeline.
- The framework utilizes an Agent that interacts with Document Archives via a Bash Tool to perform multi-hop reasoning and compute verifiable numeric answers.
- The construction pipeline employs Task Synthesis, Obfuscation, and Refinement, followed by a Difficulty Filter and Human Verification to ensure high-quality, challenging, and leak-free evaluation tasks.

---

[VisCritic: Visual State Comparison as Process Reward for GUI Agents](http://arxiv.org/abs/2606.24525)

- VisCritic: introduces a visual process reward framework that verifies GUI agent actions by directly comparing pre-action and post-action screenshots in a learned semantic feature space.
- The framework utilizes a Siamese Vision Transformer (ViT) Encoder to extract semantic differences, which are then processed by an Action-Aware Critic Head to predict action success, task progress, and error types.
- VisCritic operates as a plug-and-play inference-time module that improves GUI agent performance by providing reliable step-level verification without requiring additional human-labeled training data.

---

[Reinforcement Learning for Computer-Use Agents with Autonomous Evaluation](http://arxiv.org/abs/2606.24515)

- RL-CUA: introduces a reinforcement learning framework that improves Computer-Use Agent robustness by using an autonomous Vision-Language Model evaluator to provide supervision signals.
- The framework models evaluator feedback as a noisy binary channel and derives a statistically grounded, asymptotically unbiased reward estimator to enable stable policy optimization.
- Empirical results across macOS, Windows, and Linux demonstrate that this noise-correction approach significantly outperforms raw evaluator rewards and zero-shot baselines in task success rates.

---

[Red-Teaming the Agentic Red-Team](http://arxiv.org/abs/2606.24496)

- Agentic-red-teams: introduces a security analysis of autonomous offensive security systems, identifying critical design flaws that allow attackers to achieve remote code execution on the operator's machine.
- The paper defines a full cyber kill chain for these systems, progressing from initial LLM manipulation via agent-phishing to privilege escalation, persistence, and host compromise.
- The authors propose a robust, compromise-oriented architecture that enforces strict worker-orchestrator separation, network isolation, and least-privileged execution to mitigate identified attack vectors.

---

[Decentralized Pose Graph Riemannian Optimization for Object-based Multi-Robot SLAM](http://arxiv.org/abs/2606.24489)

- DRAN (Decentralized Riemannian Approximate Newton): introduces a fully decentralized framework for object-based multi-robot pose graph optimization that leverages an object-trajectory-aware formulation and a decentralized Riemannian approximate Newton solver to achieve communication-efficient, geometry-preserving state estimation.
- The framework utilizes a dynamic map consensus mechanism to ensure global map consistency across sparse, time-varying communication topologies while employing the Schur complement method to efficiently decouple and optimize high-dimensional private robot trajectories and shared public object poses.
- By constructing local Riemannian approximate Newton models, the approach provides curvature-preconditioned updates that significantly reduce communication overhead and iteration counts compared to first-order Riemannian descent methods.

---

[The Latent Bridge: A Continuous Slow–Fast Channel for Real-Time Game Agents](http://arxiv.org/abs/2606.24470)

- Latent Bridge: introduces a learned continuous channel that projects slow model residuals into the fast LLM's input-embedding space to enable real-time agent deliberation.
- The framework couples a frozen reactive LLM with a frozen reasoning LLM, using a trainable bridge MLP to prepend latent tokens to the fast model's input sequence.
- The Latent Bridge improves performance on planning-heavy tasks where slow reasoning provides a benefit over reactive-only control, while avoiding the latency overhead of text-based coupling.

---

[Varying Bundle Size Reactive Multi-Task Assignment using Selective Cost Estimation for Multi-Agent Systems](http://arxiv.org/abs/2606.24462)

- VBS-RMTA: introduces a two-stage, multi-fidelity bundle generation approach that utilizes a Central Coordinator, Agent Local Search Tree, Euclidean Heuristic, High-Fidelity Path Planner, Priority Queue, Local Lookup Table, and Set-Packing Integer Program to enable scalable, reactive multi-robot task allocation.
- The framework employs a depth-limited beam search with a Euclidean Heuristic to generate candidate bundles, followed by a best-first refinement process using a High-Fidelity Path Planner to ensure computational efficiency.
- By offloading cost estimation to individual agents while maintaining a centralized Set-Packing Integer Program, the architecture preserves agent anonymity and ensures global feasibility in dynamic environments.

---

[Bayesian Control for Coding Agents](http://arxiv.org/abs/2606.24453)

- Bayesian Control for Coding Agents: introduces a cost-sensitive sequential hypothesis-testing framework that uses a Bayesian Controller to manage code generation, criticism, and verification.
- The framework models the orchestration problem as a Partially Observable Markov Decision Process (POMDP) to optimize expected utility by maintaining a Belief State over candidate correctness.
- The approach utilizes a Generator Agent, multiple Critic Agents, and an Oracle Verifier to dynamically decide whether to refine, verify, or stop based on the calculated value of information.

---

[NoContactNoWorries: Estimating Contact through Vision and Proprioception for In-Hand Dexterous Manipulation](http://arxiv.org/abs/2606.24450)

- NoContactNoWorries: introduces a transformer-based multimodal framework that fuses RGB-D vision with proprioception to infer binary contact states as a pseudo-tactile signal for dexterous manipulation.
- The framework utilizes a frozen RGB-D Encoder, Proprioceptive Embeddings, a Cross-Attention Module, a Causal Transformer Encoder, and a Contact Head to enable contact-aware control without physical tactile sensors.
- Experimental results demonstrate that the predicted contact signals effectively substitute for tactile feedback in downstream reinforcement learning policies, achieving robust sim-to-real transfer on both seen and novel objects.

---

[Agentic Generation of AST Transformation Rules for Fixing Breaking Updates](http://arxiv.org/abs/2606.24446)

- BIGBAG: introduces an agentic framework that generates reusable, structured, and executable AST transformations to automatically repair breaking dependency updates in Java projects.
- The framework utilizes a generate-apply-verify loop where a coding agent leverages AST transformation engines, such as Spoon or JavaParser, to synthesize repair logic that transfers across multiple client projects affected by the same dependency update.
- Experimental results across 157 breaking updates demonstrate that the choice of AST transformation engine significantly impacts repair success, with the best configuration achieving a 78.6% fix rate and 33.3% cross-project transferability.

---

[ReM-MoA: Reasoning Memory Sustains Mixture-of-Agents Scaling](http://arxiv.org/abs/2606.24437)

- ReM-MoA: introduces a memory-augmented Mixture-of-Agents framework that sustains inference-time scaling by utilizing a Ranked Reasoning Memory and a Curated Diversified Memory Routing scheme to preserve reasoning quality and exploration diversity across layers.
- The framework includes proposer agents, a comparative Reviewer Agent, and an aggregator, which collectively leverage cross-layer reasoning memory to prevent the performance degradation and saturation observed in standard MoA architectures.
- Optional frontier-model distillation further enhances the Reviewer Agent, enabling the system to maintain performance gains across diverse reasoning benchmarks as depth and width scale.

---

[Detecting AI Coding Agents in Open Source: A Validated Multi-Method Census of 180 Million Repositories](http://arxiv.org/abs/2606.24429)

- Multi-layered detection framework: introduces a comprehensive approach to identify AI coding agents across 180 million repositories by integrating four distinct detection methods: Type A (bot account), Type B (message signature), Type C (distributed human attribution), and Type D (configuration-file scanning).
- The framework leverages the World of Code infrastructure to perform a multi-method union, revealing that single-signal detection methods significantly undercount AI activity, with some agents showing up to a 30x relative-recall gap.
- The research validates the detection taxonomy through hand-labeled samples and demonstrates that AI agent adoption is bimodal, characterized by either "born-with-AI" greenfield projects or legacy integration.

---

[Escaping the Self-Confirmation Trap: An Execute-Distill-Verify Paradigm for Agentic Experience Learning](http://arxiv.org/abs/2606.24428)

- EDV (Execute-Distill-Verify): introduces a collaborative experience learning framework that decouples execution, distillation, and validation to mitigate the Self-Confirmation Trap in LLM agents.
- The framework utilizes a Heterogeneous Agent Pool to generate diverse trajectories, a Distillation Agent for comparative analysis, and a Verification Group for consensus-based filtering before memory insertion.
- EDV maintains an Ability Matrix and a hierarchical memory structure (Shared/Private Memory Bank) to optimize task-solver matching and improve long-horizon reasoning reliability.

---

[Agentic AI for Bilevel Long-Term Optimization of Policy-Driven Physical Layer Systems](http://arxiv.org/abs/2606.24416)

- Agentic-LTPO: introduces a nested bilevel optimization framework that decouples long-term policy-driven configuration from latency-sensitive physical layer control using a multi-agent architecture.
- The upper level employs Policy Interpreter Aint, Network Observer Aobs, Configuration Planner Aplan, and Performance Critic Acrit to translate evolving operator intents into structured parameters, supported by a RAG module for evidence-based refinement.
- The lower level utilizes a closed-form zero-forcing beamforming solver to ensure real-time feasibility and optimality under instantaneous channel conditions and upper-level constraints.

---

[Poisoned Playbooks: Demystifying Knowledge Poisoning Effects on AI Security Agents](http://arxiv.org/abs/2606.24402)

- Poisoned Playbooks: introduces a systematic empirical study on how crafted security write-ups injected into public knowledge sources can manipulate the behavior of RAG-based AI security agents.
- The research defines the Verification Boundary (VB) as a three-level classification system that determines whether an LLM agent will reject, evaluate, or adopt poisoned claims based on available evidence.
- The study demonstrates that while mitigation strategies like verification prompting and multi-source retrieval help, they remain insufficient in sparse-evidence or zero-day vulnerability scenarios where the agent lacks contradictory information.

---

[ATRIA: Adaptive Traceable ECG Reporting with Iterative Agents](http://arxiv.org/abs/2606.24392)

- ATRIA (Adaptive Traceable ECG Reporting with Iterative Agents): introduces a multi-agent system that mimics clinical ECG reporting workflows by utilizing an Orchestrator Agent, Analysis Agent, Report Agent, Literature Agent, Review Agent, and a Shared Artifact Store to enable iterative, traceable report generation.
- The framework decouples ECG interpretation from reporting through staged handoffs, allowing clinicians to verify, revise, and augment findings mid-session without re-executing the entire pipeline.
- By binding every report claim to supporting evidence and maintaining a persistent artifact store, the system ensures clinical trustworthiness and supports complex tasks like comparative ECG analysis and lab-augmented reporting.

---

[Managing Task Execution for Unknown Workloads in Batteryless IoT: A Hardware-Agnostic Evaluation](http://arxiv.org/abs/2606.24340)

- Batteryless IoT Task Scheduling Framework: introduces hardware-agnostic dynamic scheduling strategies for batteryless IoT devices by treating applications as black boxes and managing energy budgets through Reinforcement Learning Agent, Approximated Prediction Method, and AIMD-MIAD Controller.
- The framework utilizes a physically accurate simulation environment to evaluate trade-offs between task throughput, system resilience, and recovery times across varying capacitor sizes.
- The research demonstrates that while advanced dynamic strategies provide critical resilience for constrained systems, simpler static policies remain effective for devices with larger energy buffers.

---

[RoBoSR: Structured Scene Representations for Embodied Robotic Reasoning](http://arxiv.org/abs/2606.24338)

- RoBoSR: introduces an embodied intermediate representation that formulates manipulation as step-wise state transitions over semantically grounded, object-centric scene graphs using Qwen3-8B, SAM3, SKIL, Manip-Cognition-1.6M, Scene Graph, RGB-D Observation, and Action Expert.
- The framework disentangles high-level task reasoning from raw inputs by utilizing a structured scene graph as the primary state space for decision-making and causal state evolution.
- RoBoSR employs a two-stage training process, combining supervised fine-tuning with reinforcement fine-tuning to enforce execution consistency and improve long-horizon task planning.

---

[Securing LLM-Agent Long-Term Memory Against Poisoning: Non-Malleable, Origin-Bound Authority with Machine-Checked Guarantees](http://arxiv.org/abs/2606.24322)

- TMA-NM (Tamper-evident Memory Authority, Non-Malleable): introduces a structural defense for LLM agents that binds memory authority to its origin at write time to prevent poisoning via laundering transformations.
- The framework utilizes a TMA-NM monitor to enforce non-malleable information-flow control, ensuring that untrusted memory cannot trigger consequential actions without independent trusted corroboration.
- Empirical evaluation across eight frontier models demonstrates that TMA-NM achieves zero attack success on memory poisoning while maintaining full legitimate utility, outperforming content- and lineage-based defenses.

---

[AutoSpec: Safety Rule Evolution for LLM Agents via Inductive Logic Programming](http://arxiv.org/abs/2606.24245)

- AutoSpec: introduces a framework that automatically evolves expert-designed safety rules for LLM agents using counterexample-guided inductive synthesis (CEGIS) and inductive logic programming (ILP) to maintain high precision and recall.
- The framework iteratively mines false-positive and false-negative counterexamples from execution traces, uses ILP to identify discriminating predicates, and applies constrained edit operators to refine symbolic guardrails.
- AutoSpec achieves significant improvements in safety rule F1 scores across code-execution and embodied agent domains while producing human-readable, auditable, and generalizable rules.

---

[SP-Mind: An Autonomous Reasoning Agent for Spatial Proteomics Analysis](http://arxiv.org/abs/2606.24235)

- SP-Mind: introduces an autonomous reasoning agent that unifies spatial proteomics analysis by dynamically chaining specialized computational tools through expert-curated skill templates and a ReAct-style reasoning loop.
- The framework utilizes a dual-layer memory architecture and a Python-to-Container Bridge to ensure reproducible, multi-step execution across heterogeneous computing environments.
- SP-Mind is evaluated on SP-Bench, a comprehensive hierarchical benchmark, where it demonstrates state-of-the-art performance in orchestrating complex, end-to-end spatial biology workflows.

---

[SkyChain Intelligence: A Blockchain-Secured Multi-Agent DRL Framework for Low-Altitude Embodied Artificial Intelligence](http://arxiv.org/abs/2606.24193)

- SkyChain Intelligence: introduces a holistic framework that synergistically integrates agentic AI, consortium blockchain, and MADRL to address the trilemma of autonomy, security, and efficiency in low-altitude computility networks.
- The framework utilizes a hybrid-action MADRL algorithm with a dual-head actor-critic architecture to jointly optimize task offloading, resource allocation, and 3D trajectory control.
- A lightweight consortium blockchain with a dynamic reputation mechanism and smart contracts provides a decentralized trust layer that is deeply fused into the MADRL reward function to incentivize secure and reliable agent cooperation.

---

[An Introduction to Causal Reinforcement Learning](http://arxiv.org/abs/2606.24160)

- CRL (Causal Reinforcement Learning): introduces a framework that integrates causal inference with reinforcement learning to develop more robust, sample-efficient, and explainable decision-making systems.
- The framework utilizes SCMs to explicitly model environment mechanisms and the Pearl Causal Hierarchy to categorize learning tasks based on observational, interventional, and counterfactual interactions.
- CRL addresses challenges in standard RL by leveraging causal diagrams and structural assumptions to enable principled policy learning under uncertainty and unobserved confounding.

---

[MedBench v5: A Dynamic, Process-Oriented, and Hallucination-Aware Benchmark for Clinical Multimodal Models](http://arxiv.org/abs/2606.24155)

- MedBench v5: introduces a dynamic, process-oriented benchmark for clinical multimodal models that evaluates reasoning trajectories rather than static outcomes.
- The framework utilizes a dual-dimensional approach combining Clinical Cognitive Responsiveness and Medical Atomic Skills to assess broad capabilities and specific operational procedures.
- It incorporates a stress-audit-tracing protocol with information-flow stressors and hallucination propagation monitoring to localize reasoning failures and analyze the evolution of unsupported claims.

---

[Autonomous Video Generation with Counterfactual Controllability for Self-Evolving World Models](http://arxiv.org/abs/2606.24152)

- Self-Evolving World Model framework: introduces a closed-loop architecture for autonomous video generation that optimizes four stages—Generation (proposes intervention-conditioned future frames), Binding (conditions frames on embodiment constraints), Verification (calibrates predictions under distribution shift), and Distillation (compresses futures into decision variables)—to achieve counterfactual controllability.
- The framework utilizes a self-evolving feedback loop where distilled action knowledge is returned to the Generation stage to iteratively improve the quality and action-validity of imagined futures.
- The approach evaluates performance using four specific metrics—Novelty, Consistency, Out-of-Distribution, and Efficiency—to ensure that generated video sequences are not only plausible but also physically actionable for embodied agents.

---

[Metis: Bridging Text and Code Memory for Self-Evolving Agents](http://arxiv.org/abs/2606.24151)

- Metis: introduces a hierarchical dual-representation memory system that balances the broad applicability of text memory with the execution efficiency of code memory for self-evolving agents.
- The framework utilizes Differentiated Text Reflection to organize experience into plans, facts, and pitfalls, while employing Pattern-Aware Code Generation to selectively crystallize recurring plans into validated callable tools.
- Metis incorporates a Memory Manager for joint selection of text and code, and a Reflection Harness to ensure dependency closure and safe tool admission, significantly improving task accuracy and execution efficiency.

---

[OmniPath: A Multi-Modal Agentic Framework for Auditing Wheelchair Accessibility](http://arxiv.org/abs/2606.24129)

- OmniPath: introduces a proactive agentic framework that fuses OpenStreetMap topology with high-resolution aerial LiDAR to audit pedestrian environments for wheelchair accessibility.
- The framework utilizes a Geometric Perception Engine and Agent A to perform micro-scale segmentation and quantify physical friction points like running slopes, cross slopes, and vertical discontinuities.
- By calculating a weighted severity score for each 0.5-meter segment, the system identifies accessibility barriers and provides actionable intelligence for urban infrastructure remediation.

---

[DramaDirector: Geometry-Guided Short Drama Generation](http://arxiv.org/abs/2606.24107)

- DramaDirector: introduces a geometry-grounded framework that decouples short-drama generation into structured storyboard planning and retrieval-augmented video synthesis.
- The framework utilizes a gallery of real short-drama shots to provide depth and pose references, which guide the first-frame generator and ensure cinematographic consistency.
- DramaDirector employs multi-task supervised finetuning and GRPO-based reinforcement learning to align LLM-generated storyboards with real-world visual priors and narrative requirements.

---

[Universal Guideline-Driven Image Clustering via a Hybrid LLM Agent](http://arxiv.org/abs/2606.24094)

- Guideline-Driven Clustering Agent: introduces a training-free framework that performs universal image clustering by utilizing GCPM to generate guideline-aware embeddings and MST-based LLM Traversal for selective semantic merging.
- The framework employs MLLMs to extract concept proxy captions, which are subsequently encoded by instruction-aware embedders to achieve attribute disentanglement and guideline adherence.
- To ensure computational efficiency, the system uses HDBSCAN for initial clustering and an MST-based traversal algorithm that minimizes LLM invocations to O(N log N) while maintaining high semantic precision.

---

[Breaking the Filter Bubble: A Semantic Pareto-DQN Framework for Multi-Objective Recommendation](http://arxiv.org/abs/2606.24042)

- Pareto-DQN: introduces a multi-objective reinforcement learning framework that formalizes recommendation as a semantic multi-objective Markov decision process to mitigate filter bubbles.
- The architecture integrates high-fidelity semantic embeddings with a Pareto-DQN agent to treat engagement, diversity, and fairness as distinct, non-aggregable reward signals.
- By utilizing hypervolume-based action selection, the framework effectively maps the Pareto frontier, enabling responsible recommendations with minimal impact on user engagement.

---

[Can Language Model Agents be Helpful Circuit Explainers in Mechanistic Interpretability?](http://arxiv.org/abs/2606.24026)

- HYVE (Hypothesize, Validate, Explain): introduces an agentic framework that explains localized transformer circuits through an iterative loop of Observation, Hypothesis Generation, Hypothesis Validation, Classification, and Summarization.
- The framework utilizes LLM backbones to perform mechanistic interpretability research by generating grounded hypotheses and testing them through automated causal interventions and code execution.
- The authors introduce AGENTICINTERPBENCH, a benchmark comprising 84 semi-synthetic transformer circuits, to evaluate the sufficiency and reliability of LLM agents in circuit explanation.

---

[Submodular Welfare Maximization with Budget Constraints in the Random-Order Model](http://arxiv.org/abs/2606.22520)

- SGAP introduces a constant-competitive online algorithm for submodular welfare maximization under budget constraints in the random-order model using InfeasibleSubGAP and FeasibleSubGAP.
- The framework utilizes a continuous greedy algorithm to compute fractional assignments, which are then converted into feasible integral assignments via randomized selection.
- The approach achieves a competitive ratio of approximately 1/14.85 for the general problem and improves existing bounds for the submodular secretary matching special case.

---

[FlowDec: Temporal Conditional Flow Decorruptor for Robust Continuous Vision-Language Navigation](http://arxiv.org/abs/2606.22424)

- FlowDec: introduces a navigation-aware image restoration framework that utilizes latent conditional flow matching to mitigate visual corruptions in continuous environments.
- The framework integrates a hybrid temporal conditioning strategy and action-centroid guided filtering to ensure temporally consistent and physically meaningful image reconstruction for LLM-based navigation agents.
- By decoupling robustness enhancement from the navigation backbone, FlowDec achieves real-time performance and improved navigation success across diverse unseen visual corruptions.

---

[Knowledge-Graph Grounding Helps LLMs Only for Out-of-Training Knowledge: A Controlled Study on Clinical Question Answering](http://arxiv.org/abs/2606.22419)

- Samyama-graph framework: introduces a controlled study evaluating how structured knowledge-graph grounding impacts LLM performance across in-training and out-of-training knowledge regimes.
- The research demonstrates that grounding provides no significant performance lift for in-training data but yields substantial accuracy improvements for out-of-training facts.
- The study establishes a knowledge-boundary law, showing that grounding is only effective when the decisive information is absent from the LLM's parametric memory and not reconstructible from surface structure.

---

[Skills for the future software profession: beyond agentic AI!](http://arxiv.org/abs/2606.21894)

- Agentic Software Engineering Workflow: introduces a framework for future software development that shifts human focus from manual coding to orchestrating specialized AI agents and managing machine-checkable V&amp;V artifacts.
- The framework integrates a Coding Agent, Verification Agent, and Testing Agent to automate software production while requiring human oversight for requirement distillation and architectural decision-making.
- It emphasizes mitigating cognitive debt by maintaining structured repositories of requirements, specifications, and agent trajectories to ensure long-term system understanding and accountability.

---

#### 22nd June 2026

[Semantic Browsing: Controllable Diversity for Image Generation](http://arxiv.org/abs/2606.23679)

- Semantic Browsing: introduces an agentic workflow that organizes image generation into a hierarchical tree of semantically distinct, user-interpretable variations.
- The framework utilizes a multi-agent system comprising a Context Analyst, Brainstormer, Decision Maker, and Critic to iteratively expand a scene interpretation tree while ensuring structural integrity and plausibility.
- By decoupling semantic decision-making from pixel generation, the method enables systematic exploration of diverse design spaces without relying on stochastic sampling.

---

[MAS-PromptBench: When Does Prompt Optimization Improve Multi-Agent LLM Systems?](http://arxiv.org/abs/2606.23664)

- MAS-PromptBench: introduces a systematic benchmark for evaluating system-prompt optimization across diverse multi-agent LLM system configurations, including LLM-based agents, coordination harness, prompt optimizers, task distribution, workflow topology, communication protocol, and team size.
- The research quantifies prompt-optimization gains using MAS-GEPA and MAS-MIPRO, revealing that performance improvements are highly sensitive to task structure, communication protocols, and coordination topologies.
- Findings indicate that prompt optimization is most effective for tasks with explicit, verifiable local behaviors and when communication protocols impose structured, clear interaction patterns.

---

[EnterpriseClawBench: Benchmarking Agents from Real Workplace Sessions](http://arxiv.org/abs/2606.23654)

- EnterpriseClawBench: introduces a benchmark constructed from real-world enterprise agent sessions that converts proprietary workplace data into reproducible tasks for evaluating LLM-based agents.
- The framework utilizes a multi-stage pipeline including mechanical filtering, prompt rewriting, and taxonomy-based packaging to create 852 benchmark tasks with associated hard rules and semantic rubrics.
- Evaluation is performed across harness-model combinations, measuring performance through artifact delivery, cost, runtime, and skill-transfer behavior using both text and visual LLM judges.

---

[Causal Discovery in the Era of Agents](http://arxiv.org/abs/2606.23608)

- causal-learn+: introduces an agentic framework that coordinates causal discovery workflows while strictly isolating LLMs from the formal inferential core to prevent hallucinated causal evidence.
- The system utilizes Agentic Assistants to manage Data Analysis, Preprocessing Guidance, Algorithm Recommendation, Expert Knowledge Incorporation, Tool Coordination, and Result Interpretation, ensuring all causal claims remain grounded in data and explicit algorithms.
- By maintaining a Protected Inferential Core, the framework ensures that LLMs provide only context and guidance, leaving the final causal discovery to provably-correct algorithms and user-approved decisions.

---

[Decentralized Autonomous Traffic Management through Corridor Networks](http://arxiv.org/abs/2606.23585)

- MARL (Multi-Agent Reinforcement Learning) framework: introduces a decentralized approach for autonomous aircraft traffic management in structured corridor networks using MARL-based decentralized policy, rotation-invariant observation representation, curriculum-based training, local sensing neighborhood, and planar fixed-wing kinematic model.
- The framework enables aircraft to perform zero-shot transfer to complex multi-corridor networks by relying solely on local observations and interaction data without centralized coordination.
- Experimental results demonstrate that the learned policies maintain high corridor conformance and stable traffic flow across varying densities and heterogeneous vehicle fleets.

---

[Kamera: Unified Position-Invariant Multimodal KV Cache for Training-Free Reuse](http://arxiv.org/abs/2606.23581)

- Kamera: introduces a position-invariant KV caching framework that enables training-free reuse of multimodal context by separating content from position and applying a low-rank conditioning patch to restore cross-chunk binding.
- The framework utilizes a canonical KV store and an exact RoPE relocation operator to eliminate redundant re-prefills, while a rank-m conditioning patch corrects the diffuse, deep-layer deficit caused by cross-chunk dependencies.
- Kamera achieves near-ceiling accuracy across diverse LLM architectures (MLA, GQA, MHA) by treating context as a set of reusable chunks, enabling efficient window operations like reordering, sliding-window survival, and reversible eviction.

---

[HoloAgent-0: A Unified Embodied Agent Framework with 3D Spatial Memory](http://arxiv.org/abs/2606.23565)

- HoloAgent-0: introduces a unified embodied agent framework that organizes heterogeneous robot capabilities into a closed-loop workflow using Embodied AgentOS, Memory Layer, and Skill Layer.
- The framework utilizes a persistent 3D spatial memory and a typed skill interface to enable reliable long-horizon task execution across diverse robot embodiments.
- HoloAgent-0 connects cloud-level reasoning with on-device execution, providing feedback-driven re-planning and monitoring through a standardized ROS2 command/status interface.

---

[VeriEvol: Scaling Multimodal Mathematical Reasoning via Verifiable Evol-Instruct](http://arxiv.org/abs/2606.23543)

- VeriEvol: introduces a scalable framework that transforms low-difficulty image-question seeds into verified training samples using Prompt Difficulty Control and HTV-Agent to ensure high-quality data for LLMs.
- The framework decouples prompt evolution from answer verification, utilizing route-specific operators to increase difficulty and a hypothesis-test verifier to enforce answer reliability before policy updates.
- VeriEvol enables monotonic scaling of visual mathematical reasoning performance by providing a traceable, auditable pipeline that integrates seamlessly with existing GRPO-style RL recipes.

---

[Self-Compacting Language Model Agents](http://arxiv.org/abs/2606.23525)

- SELFCOMPACT: introduces a training-free scaffold that enables LLMs to perform adaptive context compaction by pairing an inline Compaction Tool with a Lightweight Rubric to determine optimal summarization timing.
- The framework utilizes a Reasoning Trajectory to guide the LLM-Agent in deciding when to invoke the Compaction Tool, effectively mitigating context rot without requiring external supervision or fine-tuning.
- By leveraging KV Cache reuse, the approach achieves significant token cost reductions while maintaining or exceeding performance compared to fixed-interval summarization methods across various agentic and reasoning benchmarks.

---

[Concordia: JIT-Compiled Persistent-Kernel Checkpointing for Fault-Tolerant LLM Inference](http://arxiv.org/abs/2606.23521)

- Concordia: introduces a GPU-resident persistent kernel that enables transparent, fault-tolerant LLM inference by combining PTX/SASS instrumentation, JIT-compiled checkpoint handlers, and append-only recovery logging.
- The framework utilizes a Persistent Kernel Executor to perform dirty-page detection at HBM bandwidth, significantly reducing checkpoint latency compared to host-side scanning.
- Concordia integrates with existing LLM serving stacks via an NCCL Wrapper and provides a unified recovery contract that supports cross-architecture migration and rapid failure restoration.

---

[AOHP: An Open-Source OS-Level Agent Harness for Personalized, Efficient and Secure Interaction](http://arxiv.org/abs/2606.23449)

- AOHP: introduces an OS-level agent harness built on AOSP that treats agents as first-class actors to enable personalized service composition, efficient agent interfaces, and secure information flow.
- The architecture replaces app-centric workflows with an agent-native design, utilizing a unified interaction interface and system-managed memory to facilitate cross-app task execution.
- Empirical results demonstrate that AOHP significantly improves task completion rates while reducing LLM token consumption and execution time compared to stock Android.

---

[Detecting Malicious Agent Skills in the Wild using Attention](http://arxiv.org/abs/2606.23416)

- Locate-and-Judge: introduces a two-stage pipeline that detects malicious LLM agent skills by using a small reader LLM to localize suspicious spans via attention ranking, followed by a zero-shot LLM judge to classify them.
- The framework significantly reduces computational costs by focusing the expensive judge component only on the top-K spans identified by the lightweight locator.
- Evaluated on 134k marketplace skills, the approach effectively identifies hidden malicious skills that evade traditional scanners by leveraging instruction-following attention as a robust signal for injection.

---

[REASONINGLENS: Hierarchical Visualization and Diagnostic Auditing for Large Reasoning Models](http://arxiv.org/abs/2606.23404)

- REASONINGLENS: introduces a multi-granularity framework for the hierarchical visualization, automated diagnostic auditing, and systemic profiling of long CoT traces in LLMs.
- The framework utilizes Hierarchical Visualization to structure reasoning into interactive graphs, Agentic Diagnosis to identify errors via Memory, Verification, and Suggestion modules, and Systemic Profiling to analyze model-level reasoning bottlenecks.
- The authors also introduce LENSBENCH, a unified benchmark comprising 130 verified instances to evaluate structural visualization fidelity and the accuracy of automated reasoning error detection.

---

[Litmus: Zero-Label, Code-Driven Metric Specification for Evaluating AI Systems](http://arxiv.org/abs/2606.23403)

- Litmus: introduces a zero-label system that derives justified, per-stage evaluation and monitoring portfolios by interrogating source code and practitioner intent to establish explicit metric-design constraints.
- The framework utilizes a multi-stage pipeline comprising static analysis, LLM-based architecture reconstruction, adversarial criticism, and goal elicitation to ground metrics in specific system components and failure surfaces.
- Litmus outperforms existing evaluation methods by providing broader coverage, near-zero redundancy, and superior label validity across diverse, code-defined AI pipelines without requiring manual labeling at design time.

---

[Superhuman AI for Generals.io Using Self-Play Reinforcement Learning](http://arxiv.org/abs/2606.23348)

- Generals.io AI Framework: introduces a superhuman agent for real-time strategy games trained via self-play reinforcement learning using a JAX-native simulator, a transformer policy, a value head, a policy head, an exponential moving average, and top-advantage filtering.
- The architecture utilizes a transformer torso to process spatial game observations and temporal scoreboard statistics, enabling effective decision-making under imperfect information.
- The research demonstrates that generic policy-gradient methods, when combined with high-throughput simulation and specific training stabilizers, can achieve superhuman performance without human demonstrations or complex reward shaping.

---

[Group selection promotes prosocial prompts in populations of LLM agents](http://arxiv.org/abs/2606.23343)

- Multi-agent simulation framework: introduces a multi-generational evolutionary environment where LLM agents optimize natural-language strategy strings through individual or group-level selection.
- The framework utilizes a donor game to evaluate agent performance, applying either individual- or group-level fitness metrics to determine which strategies are transmitted to subsequent generations.
- Empirical results demonstrate that group selection effectively sustains cooperative behavior in LLM populations, while individual selection leads to the dominance of self-interested strategies and collective defection.

---

[VideoAgent: All-in-One Framework for Video Understanding and Editing](http://arxiv.org/abs/2606.23327)

- VideoAgent: introduces an all-in-one agentic framework for automated video understanding and editing that addresses long-form video planning and multi-agent orchestration challenges.
- The framework utilizes a shot planning agent for coherent narrative structure and a textual-gradient graph optimization mechanism to dynamically compose and refine complex editing pipelines.
- VideoAgent integrates over thirty specialized editing agents and achieves high orchestration success rates while significantly reducing API costs compared to existing LLM-based systems.

---

[TMAX: A simple recipe for terminal agents](http://arxiv.org/abs/2606.23321)

- TMAX: introduces a scalable, difficulty-aware synthetic data generation pipeline and a reinforcement learning recipe for training terminal-using agents.
- The framework utilizes a compositional Data Pipeline to create diverse RL environments, which are then used to train an LLM Policy via DPPO Trainer within a Docker Sandbox.
- TMAX-9B achieves state-of-the-art performance among open-weight models under 10B parameters on Terminal-Bench 2.0, demonstrating effective generalization across tasks and harnesses.

---

[Test-Driven, AI-Assisted Learning: Replacing Lectures with Weekly Closed-Book Tests](http://arxiv.org/abs/2606.23315)

- TDAA (Test-Driven, AI-Assisted Learning): introduces a pedagogical framework that replaces traditional lectures with a weekly cycle of self-paced study and strict closed-book tests, supported by a course-materials harness.
- The framework utilizes a course-materials harness, which integrates an AI writer, AI reviewer, and human approval process to automate the production of learning sheets, validation sheets, and tests at scale.
- By shifting the instructor's role from routine lecturing to designing learning paths and verifying AI-generated content, the model ensures accountability and operational feasibility in proof-heavy courses.

---

[EHR-Complex: Benchmarking Medical Agents for Complex Clinical Reasoning](http://arxiv.org/abs/2606.23301)

- EHR-Complex: introduces a large-scale benchmark for interactive clinical database reasoning, utilizing a MIMIC-IV substrate to evaluate LLM agents on complex, multi-table, and longitudinal EHR analysis tasks.
- The framework employs a construction pipeline that transforms patient records into a Patient Event Graph, extracts Clinical Evidence Paths, and performs SQL Compilation to generate execution-validated tasks.
- Evaluation of LLM agents on EHR-Complex reveals significant challenges in population-level reasoning, medical-code grounding, and SQL logic, highlighting the necessity for robust, interactive clinical reasoning capabilities.

---

[IOI: Decoupling Kinematics and Physics for Interactive World Models](http://arxiv.org/abs/2606.23296)

- IOI: introduces a hybrid interactive world model that decouples deterministic robot kinematics from stochastic environmental dynamics to improve simulation fidelity.
- The framework utilizes a URDF-based kinematic solver and multi-view orthographic renderer to provide geometry-consistent guidance to a diffusion-based video generator.
- By integrating analytical kinematic priors via the MKAI module, IOI mitigates action deviation and state implausibility, achieving robust performance in policy evaluation and OOD generalization.

---

[Towards Root Memories: Benchmarking and Enhancing Implicit Logical Memory Retrieval for Personalized LLMs](http://arxiv.org/abs/2606.23283)

- RootMem: introduces a plug-and-play framework that distills raw user histories into structured Root Memory Units to complement semantic retrieval with personalized decision logic.
- The framework utilizes a collaborative Generator-Judger-Refiner pipeline to construct the IMLogic benchmark, which evaluates the ability of LLMs to retrieve logically critical but semantically distant memories.
- RootMem employs a Root Memory Router to identify relevant units based on Execution Rules and Personalized Logical Evidence, significantly improving the accuracy of LLMs in implicit logical retrieval tasks.

---

[GIF: Locally Sound Geometric Information Flow Control for LLMs](http://arxiv.org/abs/2606.23277)

- GIF (Geometric Information Flow): introduces a semantic framework for tracking information flow in LLMs by using the Jacobian and local output geometry to upper-bound Shannon mutual information between input spans and model outputs.
- The framework utilizes a locally faithful Gaussian surrogate channel to provide a scalable, sound, and quantitative measure of information flow that avoids the overtaint problems of traditional IFC.
- GIF enables efficient, fine-grained security enforcement in agentic systems by surfacing policy-relevant input spans for declassification, significantly reducing token costs compared to full-trajectory LLM judges.

---

[Dynamic multi-agent deep reinforcement learning-based pricing and incentivization approach in multimodal transportation networks](http://arxiv.org/abs/2606.23257)

- Multi-agent deep reinforcement learning framework: introduces a dual-agent approach that reconciles conflicting objectives between public authorities and profit-driven ridesharing providers through coordinated dynamic pricing and incentivization.
- The framework integrates a multimodal macroscopic simulation with two RL agents, where the public authority manages PT incentives to improve system-wide efficiency and equity, while the ridesharing provider adjusts fares to maximize revenue.
- Numerical experiments on the Sioux Falls network demonstrate that an equity-oriented dynamic policy achieves the best trade-off between efficiency, environmental impact, and profitability compared to static baseline strategies.

---

[Wireless Personal Agent: Extending Wireless Intelligence from Networks to Terminals](http://arxiv.org/abs/2606.23255)

- WISPA (Wireless Intelligent Self-evolving Personal Agent): introduces a terminal-side resource management framework that decouples latency-sensitive online execution from offline LLM agent reflection to achieve personalized wireless connectivity.
- The framework utilizes an Online Executor for deterministic, lightweight decision-making and an Offline LLM Agent, comprising an Observation Module, Reflection Module, and Strategy Module, to refine user-specific preference parameters based on historical usage.
- By confining LLM-based reasoning to offline idle windows and employing a bounded parameter update mechanism, the system ensures real-time reliability while adapting to dynamic user preferences and environmental conditions.

---

[RS-Gen: A Multi-Stage Agentic Framework for Reasoning and Search-Augmented Image Generation](http://arxiv.org/abs/2606.23221)

- RS-Gen: introduces a multi-stage agentic framework that reconstructs image generation into a collaborative "questioning-and-solving" workflow using Conversation Memory, Image Router Agent, Intent Analysis Agent, Reasoning &amp; Search Agent, and Image Generation Agent.
- The framework utilizes a Tool Hub and Expert Toolset to integrate external knowledge and logical reasoning, effectively bridging the gap between static model training and real-world complex visual tasks.
- RS-Gen employs a "Generate-Review-Correct" closed-loop mechanism within its Image Generation Agent to ensure high-fidelity, logically consistent outputs through iterative self-correction.

---

[MuPPET: A Benchmark for Contextual Privacy of LLM Assistants in Multi-Party Conversations](http://arxiv.org/abs/2606.23217)

- MuPPET (Multi-Party Privacy Exposure Testing): introduces a benchmark for evaluating contextual privacy risks of LLM assistants in multi-party conversational environments.
- The framework utilizes LLM-as-a-judge to assess privacy leakage and utility by analyzing model responses against user-specific memories and shared group context.
- Experimental results demonstrate that LLMs frequently leak sensitive information in multi-party settings, with smaller open-weights models exhibiting the highest vulnerability.

---

[Memory Contagion: Cross-Temporal Propagation of Evaluator Bias via Agent Memory](http://arxiv.org/abs/2606.23195)

- Memory Contagion: introduces a formal framework to analyze the cross-temporal propagation of evaluator bias through agent memory systems, where biased experiences stored by a source agent influence the performance of future target agents.
- The framework decomposes contagion into content-based and retrieval-based components, demonstrating that biased input is a sufficient cause for contagion even under perfect oracle consolidation.
- Experimental results reveal that consolidation effects are bias-type-dependent, where LLM-based summarization attenuates global length bias but amplifies local authority bias.

---

[Capable but Careless: Do Computer-Use Agents Follow Contextual Integrity?](http://arxiv.org/abs/2606.23189)

- AGENTCIBENCH: introduces a generative evaluation harness that measures contextual-integrity failures in computer-use agents by surfacing realistic, multi-app scenarios where agents must balance task utility against privacy-sensitive information disclosure.
- The framework utilizes an MCTS-based scenario-surfacing engine to generate stress tests across three failure modes: visual co-location, task-ambiguity overshare, and recipient misalignment.
- Evaluation of fifteen frontier agents reveals that high task completion is a poor proxy for disclosure safety, with most agents frequently leaking personal information, though prompt-level interventions can significantly mitigate these failures.

---

[Position: Correct Answer, Wrong Mechanism — When AI Scientists Defend General Claims Their Own Data Contradicts](http://arxiv.org/abs/2606.23175)

- CAWM (Correct Answer, Wrong Mechanism): introduces a framework for evaluating AI scientist systems by measuring task outcome, mechanism fidelity, and epistemic honesty, identifying cases where LLMs produce correct results based on physically incorrect or over-generalized reasoning.
- The research demonstrates that LLMs often fail to self-apply mechanism-fidelity verification, leading to "Correct Answer, Wrong Mechanism" failures where agents defend observables with physics contradicted by their own simulation data.
- The paper proposes a lightweight regime-shift verification protocol and companion recomputation as necessary gates for co-author-level trust in autonomous scientific discovery pipelines.

---

[Decomposing Financial Market Dynamics via Mechanism Analysis in an Evolutionary Multi-Agent Simulation](http://arxiv.org/abs/2606.23158)

- Evolutionary Multi-Agent Simulation framework: introduces a modular ABM architecture that decomposes market dynamics by isolating four pluggable mechanisms to identify their specific control over emergent market properties.
- The framework utilizes single-mechanism interventions to demonstrate that selection operators drive diversity, microstructure governs realism, and behavioral bias influences fragility.
- The research provides a systematic decomposition of market dynamics, revealing that consensus network topology acts as an honest null with no robust effect on emergent market outcomes.

---

[Neural Parameter Calibration for Finite-State Mean Field Games](http://arxiv.org/abs/2606.23155)

- Neural Parameter Calibration for Finite-State Mean Field Games: introduces a fully differentiable framework for learning trajectory-wise parameters of finite-state MFGs from macroscopic population data using a Neural Network φθ, an MFG Solver Φ, and Implicit Differentiation.
- The framework treats the fixed-point equilibrium solver as an implicit neural network component, enabling efficient gradient computation via the adjoint method without unrolling solver iterations.
- The approach is validated across synthetic and real-world datasets, demonstrating robustness to noise and the ability to perform counterfactual analysis through the forward-looking nature of the MFG formulation.

---

[Rising From the Ashes: How Agentic AI is Unblocking Challenges in Cybersecurity](http://arxiv.org/abs/2606.23138)

- Agentic AI Framework: introduces a conceptual mapping of emergent agentic AI capabilities to long-standing cybersecurity bottlenecks, leveraging Natural language understanding, Processing large amounts of information, Cognitive stamina, Generalization and replicability, and Multi-disciplinary knowledge.
- The paper evaluates 16 case studies across systems security, software security, CTI, and attack detection to demonstrate how these agentic capabilities can automate labor-intensive and knowledge-intensive security tasks.
- It highlights that while agentic AI offers a path to scalable security, it necessitates rigorous verification and security-by-design to prevent the amplification of existing vulnerabilities through autonomous agents.

---

[Understanding the (In)Security of Vibe-Coded Applications](http://arxiv.org/abs/2606.23130)

- Vulnerability Analysis Framework: introduces a multi-agent system that combines Claude Code, GitHub Copilot, OpenAI Security Best Practices Skill, Antigravity Awesome Skills, Claude Sonnet 4.6, GPT-5.3-Codex, a Validation Agent, and Human Reviewers to systematically identify and validate security vulnerabilities in vibe-coded applications.
- The framework employs a 2x2 auditing design using two agent frameworks and two security skill sets to generate independent reports, which are then deduplicated and validated through automated and manual processes.
- This research identifies eight recurring failure modes categorized into memory, objective, and knowledge defects that contribute to widespread security vulnerabilities in vibe-coded applications.

---

[Managing Procedural Memory in LLM Agents: Control, Adaptation, and Evaluation](http://arxiv.org/abs/2606.23127)

- AFTER (A Benchmark for Procedural Skill Transfer in LLM Agents): introduces a benchmark and evaluation framework for measuring how procedural memory transfers across tasks, roles, and LLM backbones using AFTER, EVOLUTION, SKILL.md, Reflector, Adapter, and Trace Pool.
- The framework utilizes EVOLUTION to manage the lifecycle of SKILL.md artifacts, employing a Reflector to iteratively refine procedural knowledge based on execution traces.
- Experiments demonstrate that procedural memory evolved from diverse multi-model traces significantly improves cross-model generalization while revealing that role-specific specialization can hinder cross-role transfer.

---

[A Matter of Time: Towards a General Theory of Agency](http://arxiv.org/abs/2606.23122)

- Temporally Parametrized (F, A)-systems: introduces a graded organizational theory of agency by unfolding semantically closed systems across characteristic timescales to reveal endogenous anticipatory structures.
- The framework redescribes these temporally unfolded organizations as history-dependent ADBNs, where dependencies between components (A, B, C, D, E) and interface processes (X, Y, Z) are constrained by their respective timescales (ϕ, ψ, θ, α, β).
- This approach provides a formal hierarchy from proto-agential chemical systems to robust agents, enabling the operational distinction between autonomy, goal-directedness, agency, and open-endedness through measurable metric profiles.

---

[Self-Evolution for Multi-Turn Tool-Calling Agents via Divergence-Point Preference Learning](http://arxiv.org/abs/2606.23112)

- ToolGraph + DPO: introduces a framework that improves LLM tool-calling reliability by combining structural graph-based orchestration with divergence-point preference learning.
- The system utilizes ToolGraph to provide structural guidance via schema-derived topology and experience-based edge weights, while DPO refines the LLM policy using preference pairs extracted from successful and failed trajectories.
- By aligning training and deployment contexts and filtering preference data against action-level correctness annotations, the framework enables effective within-benchmark self-evolution for multi-turn agents.

---

[Cognitive Digital Twins: Ethical Risks and Governance for AI Systems That Model the Mind](http://arxiv.org/abs/2606.23094)

- CDT (Cognitive Digital Twin): introduces a governance framework organized around Authority, Autonomy, Access and Control, Accountability, and Availability to address the ethical risks of AI systems that model, simulate, and act as a person's cognition.
- The paper argues that CDTs require governance at the level of cognitive representation itself, as these systems can act as proxies or infrastructures through which human cognition is simulated and operationalized.
- The authors propose concrete requirements for high-risk CDTs, including layered consent, purpose-binding enforcement, and traceability, to mitigate risks such as misrepresentation, epistemic authority shifts, and shadow twinning.

---

[Safety in Self-Evolving LLM Agent Systems: Threats, Amplification, and Case Studies](http://arxiv.org/abs/2606.23075)

- MLAS (Module–Lifecycle Attack Surface) framework: introduces a systematic security analysis of self-evolving LLM agents by mapping threats across five functional modules and five evolutionary lifecycle stages.
- The paper identifies seven cross-cutting amplification effects, including generational accumulation and optimizer–optimizee collapse, which transform transient attacks into persistent, self-reinforcing, and lineage-wide security threats.
- Comparative case studies of OpenClaw and Hermes demonstrate that evolution-native designs significantly expand the attack surface and bypass static security scanners, necessitating evolution-aware defense architectures.

---

[Training Open Models for Agentic Phone Use](http://arxiv.org/abs/2606.23049)

- PhoneBuddy: introduces a training recipe for agentic phone use that combines a real-app environment with a mock-app environment, PhoneWorld, to improve task completion through a shared SFT stage and mixed RL.
- The framework utilizes a Qwen3.5-4B backbone and leverages PhoneWorld to provide scalable, resettable, and automatically verifiable interaction, complementing the high-fidelity but costly real-app environment.
- Empirical results demonstrate that combining real-app and mock-app RL significantly outperforms individual training approaches, particularly for single-app and mini-app tasks, while highlighting persistent challenges in cross-app workflows.

---

[UECP: Uncertainty-Enhanced Collaborative Perception](http://arxiv.org/abs/2606.23046)

- UECP (Uncertainty-Enhanced Collaborative Perception): introduces a collaborative perception framework that utilizes a physically-grounded uncertainty map to guide multi-agent feature fusion, effectively mitigating collaborative noise.
- The framework employs an Uncertainty-Aware Pyramid Fusion (UAPF) module, which integrates Uncertainty-Weighted Downsampling (UWD) and Uncertainty-Guided Residual Fusion (UGRF) to enhance robustness and detection precision.
- By decoupling uncertainty estimation from detection head training, UECP provides an unbiased, scenario-aware signal that improves performance across various autonomous driving benchmarks.

---

[A Stackelberg Framework for Resource-Aware LLM Agents: Learning, Repair, and Conditional Guarantees](http://arxiv.org/abs/2606.23026)

- Stackelberg Framework for Resource-Aware LLM Agents: introduces a contextual leader-follower game to govern LLM agent resource allocation by balancing quality targets and cost incentives against executor actions.
- The framework utilizes a Controller that commits to quality and budget signals, while an Executor responds with context, prompt, and tool usage actions to optimize performance under finite computational constraints.
- The approach incorporates real-API calibration and action-space projection to repair learned policies, ensuring stable and safe resource governance in multi-turn LLM agent interactions.

---

[Group-Graph Policy Optimization for Long-Horizon Agentic Reinforcement Learning](http://arxiv.org/abs/2606.22995)

- G2PO (Group-Graph Policy Optimization): introduces a reinforcement learning framework for long-horizon agentic tasks that transforms interaction trajectories into a global state-transition graph to enable fine-grained credit assignment.
- The framework utilizes State Group Graph to aggregate identical observations across trajectories, significantly reducing variance in state-value estimation compared to single-trajectory methods.
- G2PO incorporates Edge-Centric Advantage Estimation to prioritize critical transitions by globally standardizing Temporal Difference errors, effectively guiding LLMs toward task completion with minimal computational overhead.

---

[Distilling Collaborative Dynamics into Latent Space for Implicit Coordination in Decentralized Multi-Agent Manipulation](http://arxiv.org/abs/2606.22982)

- CLS-DP (Collaborative Latent Space–conditioned Diffusion Policy): introduces a decentralized multi-agent framework that enables implicit coordination by distilling privileged multi-agent dynamics into a latent space during centralized training, which is then inferred by each agent from local observations at deployment.
- The framework utilizes a contextualizer to encode local RGB observations and task instructions into a collaborative latent, which conditions a decentralized diffusion policy to anticipate teammate behaviors without explicit communication.
- By employing a two-stage training process and discarding privileged kinematics modules at inference, CLS-DP achieves superior parameter efficiency and scalability compared to centralized baselines in multi-arm manipulation tasks.

---

[StatABench: Dataset and Framework for Evaluating Statistical Analysis Capabilities of LLMs](http://arxiv.org/abs/2606.22977)

- StatABench: introduces a comprehensive benchmark for evaluating LLMs' statistical analysis capabilities through closed-form questions and open-ended modeling tasks.
- The framework utilizes SAToolKit for tool-grounded reasoning and employs an LLM-as-Judge protocol to assess complex, open-ended statistical reports.
- Experimental results reveal a significant tool-integration tax across diverse LLMs, highlighting persistent challenges in methodological decision-making and end-to-end statistical modeling.

---

[Evo-RAD: Navigating Rare Retinal Disease Diagnosis via Self-Evolving Agentic Retrieval](http://arxiv.org/abs/2606.22955)

- Evo-RAD: introduces a self-evolving agentic retrieval framework that reformulates evidence acquisition as a Markov Decision Process to improve rare retinal disease diagnosis.
- The framework utilizes a Graph Policy Network to iteratively refine a reference set through DELETE, INSERT, and TERMINATE operations, optimized via Group Relative Policy Optimization (GRPO) with a homogeneity-aware reward.
- By dynamically suppressing hub-driven distractors and incorporating clinically concordant samples, the agentic approach achieves superior diagnostic performance compared to static retrieval and parameter-efficient fine-tuning methods.

---

[Plans Don’t Persist: Why Context Management Is Load Bearing for LLM Agents](http://arxiv.org/abs/2606.22953)

- Replay Pairing: introduces a diagnostic framework to measure plan persistence in LLMs by comparing hidden-state cosine distances between trajectories with and without plan history.
- The study demonstrates that LLM agents typically treat plans as context-time objects that decay rapidly within one step, rather than as persistent internal states.
- The authors identify a reasoning-trace confound in reasoning models and propose strict stripping to accurately isolate plan-related signals, further validating that naive plan eviction significantly degrades agent performance.

---

[ENVS: Environment-Native Verified Search for Long-Horizon GUI Agents](http://arxiv.org/abs/2606.22948)

- ENVS: introduces a training-time search-and-filter pipeline that leverages environment-native verification to construct balanced, high-quality supervised datasets for GUI agents.
- The framework decouples trajectory discovery from policy optimization by using a Frozen Policy to explore OSWORLD Virtual Machines, followed by a Data Curator that applies global balancing to the collected trajectories.
- To evaluate robustness, the paper introduces OSWORLD-NOISY, a benchmark that injects recoverable desktop interruptions to test an agent's ability to refocus, dismiss, or recover during task execution.

---

[When Agents Commit Too Soon: Diagnosing Premature Commitment in LLM Agents](http://arxiv.org/abs/2606.22936)

- LLM Agents: introduces a diagnostic framework for detecting premature commitment, where agents settle on an interpretation early and defend it, by measuring cross-run hidden-state convergence.
- The framework utilizes activation similarity at specific reasoning steps to predict trajectory consistency across multiple LLM architectures, including Llama-3.1, Qwen-2.5, and Phi-3.
- The research demonstrates that while commitment signals reliably track process consistency, they are correctness-agnostic, meaning they identify settled trajectories regardless of whether the agent is right or wrong.

---

[Intent-Governed Tool Authorization for AI Agents](http://arxiv.org/abs/2606.22916)

- IGAC (Intent-Governed Access Control): introduces a server-side authorization layer that binds AI agent tool use to the user's expressed intent through intent certificates, session policy, and consistency checks.
- The framework operates as an upstream governance layer over the OpenPort effect-control substrate, ensuring that agent actions remain within the bounds of the user's current request.
- IGAC utilizes manifest filtering and consistency gating to prevent overbroad tool usage and payload expansion, treating LLM-based planners as untrusted advisors rather than authoritative principals.

---

[From Fragments to Paths: Task-Level Context Recovery for Large Industrial Codebases](http://arxiv.org/abs/2606.22906)

- DEEPDISCOVERY: introduces a two-stage Location–Inference framework that identifies high-confidence task anchors and expands over a multi-relational repository graph to recover complete task-relevant context under budget constraints.
- The method utilizes metadata-first context construction to preserve structural coverage while selectively loading full-text content, thereby optimizing LLM context usage for complex software engineering tasks.
- Evaluations on industrial codebases and SWE-bench Verified demonstrate that DEEPDISCOVERY consistently improves task-relevant file recovery and downstream LLM performance compared to traditional fragment-retrieval and GraphRAG-style approaches.

---

[Agent-as-a-Router: Agentic Model Routing for Coding Tasks](http://arxiv.org/abs/2606.22902)

- ACRouter (Agentic Coding Router): introduces a framework that formalizes model routing as a Context-Action-Feedback loop to address information deficit in LLM selection for coding tasks.
- The framework utilizes an Orchestrator to select models, a Verifier to provide execution-grounded feedback, and a Memory module to accumulate experience for future routing decisions.
- The research introduces CodeRouterBench, a comprehensive evaluation environment with 10K tasks, demonstrating that ACRouter achieves lower cumulative regret and better generalization than static routing baselines.

---

[DynamicMem: A Long-Horizon Memory Benchmark in Real-World Settings](http://arxiv.org/abs/2606.22877)

- DynamicMem: introduces a synthetic benchmark for evaluating long-horizon memory in LLM agents by constructing 15-month, user-consistent trajectories across 16 applications.
- The framework decomposes user profiles into attributes, habits, and preferences, which evolve under causally grounded drivers to test memory retention and update capabilities.
- Benchmarking reveals that while aggregate accuracy hides performance degradation, over 93% of failures originate from the memory system's retrieval rather than the answer-generation LLM.

---

[When AUC 0.998 Is Not Enough: A Candidate Evaluation Protocol for Hidden-State Probes of Indirect Prompt Injection in Multimodal Computer-Use Agents](http://arxiv.org/abs/2606.22864)

- Candidate Evaluation Protocol for Hidden-State Probes: introduces a diagnostic framework for validating hidden-state probes in multimodal computer-use agents by pairing reported AUC metrics with specific control sets to distinguish malicious-content detection from surface-level artifacts.
- The framework utilizes a Mind2Web trajectories-based replay protocol to evaluate a frozen multimodal computer-use agent, employing a linear probe for hidden-state extraction alongside C1 text-side falsifier and C2 visual overlay controls to identify shortcut-driven failure modes.
- The research demonstrates that high headline AUC scores can be misleading, necessitating the use of trajectory-level cluster bootstrap and nuisance-matched controls to ensure that probes are actually detecting malicious instruction content rather than OCR-density or template-based surface statistics.

---

[AI Scientists as Engines of Discovery: A Case for Development within Reformed Institutions](http://arxiv.org/abs/2606.22859)

- Denario: introduces a multi-agent framework for scientific discovery that utilizes a Generation agent, Critic agent, Verification agent, Controller agent, Reasoning layer, and External tools to automate the research cycle.
- The architecture enables an internal dialectic among specialized agents to traverse model parameter spaces and accelerate scientific discovery beyond human capacity.
- The paper argues that institutions must be redesigned to support AI scientists through improved verification, accountability, interpretability, and dual-use safety protocols.

---

[RaMem: Contextual Reinstatement for Long-term Agentic Memory](http://arxiv.org/abs/2606.22844)

- RaMem (Contextual Reinstatement for Agentic Memory): introduces a framework that mitigates context collapse by anchoring memory fragments to episodic conditions and performing validity-aware retrieval.
- The framework utilizes LLMClient, EmbeddingModel, VectorStore, MemoryBuilder, HybridRetriever, and AnswerGenerator to transform retrieved fragments into contextually verifiable evidence.
- RaMem improves long-term memory performance by prioritizing context-compatible memories while retaining content-relevant candidates as fallback evidence during synthesis.

---

[RLM-Cascade: Response-Level Speculative Decoding for Cost-Efficient LLM API Serving](http://arxiv.org/abs/2606.22840)

- RLM-Cascade: introduces a response-level speculative decoding proxy system that reduces LLM API costs by routing requests between a fast draft model and a capable verify model.
- The system utilizes a rule-based router to classify agentic turns, enabling the SKIPPED path for simple requests and a draft-then-verify pipeline for complex tasks.
- By treating the full response as the unit of speculation, the framework operates over standard HTTP APIs without requiring access to internal model logits or shared vocabularies.

---

[Active Inference as the Test-Time Scaling Law for Physical AI Agents](http://arxiv.org/abs/2606.22813)

- Active Inference as the Test-Time Scaling Law for Physical AI Agents: introduces a novel test-time scaling law for physical AI agents grounded in active inference that enables generalization in unforeseen scenarios by dynamically updating policies through surprise-minimizing reasoning.
- The framework utilizes a world model at the network edge to detect prediction errors and trigger a transition from habitual policy execution to deliberative reasoning, effectively scaling the agent's policy at test time.
- By modeling the resolution of prediction errors as a variational inference problem, the approach enables physical AI agents to continuously learn and adapt to non-stationary environments without requiring retraining.

---

[Does the Same Token Mean the Same State? MoE Routing as Signal for Reasoning Control](http://arxiv.org/abs/2606.22798)

- RAD (Routing Agreement Decoding): introduces a method for selecting high-quality LLM rollouts by analyzing sparse MoE routing states as a white-box agreement signal, utilizing MoE Router, Anchor-Conditioned Routing, Weighted-Jaccard K-NN, Consensus Matrix, Routing Density Selector, and DeepConf Confidence Signal.
- The framework identifies dense routing neighborhoods that align with answer basins, enabling effective selection in open-ended or lexically diverse tasks where traditional string-based voting fails.
- RAD operates as a consensus-based selector that aggregates routing agreement rather than verifying truth, providing competitive performance to majority voting in well-posed settings and superior results in code generation and agentic tasks.

---

[Breaking the Evaluation Paradox: Evaluating High-Entropy Search with Computationally Irreducible Constraints](http://arxiv.org/abs/2606.22783)

- VERITAS (Verifiable Traversal Assessment for Search): introduces a benchmark that replaces semantic filters with computationally irreducible cryptographic hash constraints to force LLMs into genuine exhaustive search.
- The framework utilizes Search, Visit, Exec Python, and Answer components to evaluate an agent's ability to systematically traverse large, unstructured search spaces without relying on search engine shortcuts.
- By requiring agents to find items matching specific hash values, VERITAS provides an automatically verifiable, scalable, and difficulty-tunable evaluation paradigm for high-entropy search tasks.

---

[Noise Is Signal: Density-Based Outliers as Leading Indicators of Occupational Emergence in Labor Market Text](http://arxiv.org/abs/2606.22769)

- EOS: introduces a framework for detecting emerging occupations by analyzing the semantic coherence of noise-class postings in high-dimensional embedding spaces.
- The framework utilizes INSTRUCTOR-xl, UMAP, and HDBSCAN to identify pre-cluster noise groups, which are then evaluated using the EOS metric to predict future occupational consolidation.
- By demonstrating that density-based outlier scores are uninformative for emergence, the paper establishes that semantic coherence within the noise class serves as a reliable leading indicator for new job roles.

---

[Cooperative-ORCA*: Real-Time Proactive Deadlock Avoidance for Continuous-Space Multi-Agent Navigation](http://arxiv.org/abs/2606.22757)

- C-ORCA* (Cooperative Optimal Reciprocal Collision Avoidance): introduces a continuous-space multi-agent navigation framework that integrates global guidance paths with local collision avoidance to proactively resolve deadlocks using MAPFPlanner, CorridorDetector, DependencyBuilder, StringPulling, WaypointManager, DriftModeController, and ORCAVelocitySolver.
- The framework utilizes pre-computed guidance paths and dynamic dependency detection to switch agents between standard and drift modes, effectively preventing congestion in narrow corridors.
- By proactively managing agent interactions through waypoint updates and adaptive velocity control, the approach significantly improves success rates and reduces flowtime compared to reactive baseline methods.

---

[HERCULES: An Open-Source Simulation Framework for Heterogeneous Multi-Robot SLAM, Collaborative Perception, and Exploration](http://arxiv.org/abs/2606.22756)

- HERCULES: introduces an open-source, Unreal Engine 5-based simulation framework designed for heterogeneous multi-robot autonomy, featuring a unified navigation stack, synchronized data-collection, and support for concurrent UAV-UGV operations.
- The framework resolves architectural conflicts in prior simulators to enable concurrent multi-robot simulation, providing specialized controllers, physics-based thermal and night-vision sensors, and dynamic environmental processes for realistic testing.
- HERCULES facilitates research in collaborative SLAM, cooperative 3D object detection, and closed-loop multi-robot exploration by providing a scalable, photorealistic testbed with ready-to-run benchmarks and sim-to-real transfer capabilities.

---

[GRADE: Graph Representation of LLM Agent Dependency and Execution](http://arxiv.org/abs/2606.22741)

- GRADE: introduces a unified two-layer graph representation for LLM agent runs that separates observed execution flow from graded state-dependency reliance to enable failure diagnosis.
- The framework utilizes an attachment model to grade dependency edges as observed, declared, or inferred, addressing the blind spot where standard traces fail to record state reliance.
- By pricing the dependency layer's marginal lift over the free execution layer, GRADE identifies that dependency structure provides transferable failure-prediction signals that generic graph networks often misread due to structural degeneracy.

---

[GroundEval: A Deterministic Replacement for LLM-as-Judge in Stateful Agent Evaluation](http://arxiv.org/abs/2606.22737)

- GroundEval: introduces a judge-free framework for evaluating LLM agents by verifying their evidence paths against deterministic state contracts rather than relying on LLM-as-judge plausibility.
- The framework utilizes a machine-readable contract comprising an event log, artifact corpus, access policy, and evaluation configuration to score agent performance across Perspective, Counterfactual, and Silence tracks.
- GroundEval provides structured diagnostics and a compliance-adjusted scoring model to detect state-invalid correctness, such as temporal leakage, permission violations, and ungrounded causal claims.

---

[Closed-loop Auto Research for Molecular Property Prediction: Discovering and Certifying Generalizable Improvements](http://arxiv.org/abs/2606.22731)

- Closed-loop Auto Research: introduces a framework where LLM-agents iteratively propose and execute interventions across feature, model, and data axes to improve molecular property prediction pipelines.
- The framework utilizes a file-level ablation lock to isolate research axes and a held-out certification protocol to distinguish genuine transferable discoveries from validation-selected artifacts.
- The system incorporates a leakage-safe filter to audit external data acquisition, ensuring that improvements are not derived from test set contamination or distribution shifts.

---

[Safe and Generalizable Hierarchical Multi-Agent RL via Constraint Manifold Control](http://arxiv.org/abs/2606.24010)

- HMM (Hierarchical Manifold Multi-Agent PPO): introduces a hierarchical MARL framework that decouples high-level cooperative planning from low-level safety enforcement using a constraint manifold.
- The framework utilizes a high-level policy for multi-agent coordination and a low-level constraint manifold safe controller to project nominal actions onto a safe tangent space, ensuring hard safety constraints without per-timestep quadratic programming.
- By offloading safety to a fixed low-level controller, the approach enables stable, stationary learning for the high-level policy while providing formal safety guarantees and strong generalization across varying agent and obstacle counts.

---

[ChartWalker: Benchmarking the Cross-Chart RAG Task with Hierarchical Knowledge Graphs](http://arxiv.org/abs/2606.23997)

- ChartWalker: introduces a framework for constructing challenging cross-chart RAG tasks by organizing chart entities into a Hierarchical Knowledge Graph and synthesizing reasoning paths via a Structure-aware Sampling Algorithm.
- The framework includes ChartWalker-Bench, a benchmark with 564 multi-hop QA instances, and ChartWalker-Agent, which utilizes a VLM-based search agent to navigate the knowledge graph for iterative evidence acquisition.
- Experimental results demonstrate that graph-based retrieval and agentic reasoning significantly improve performance on complex multi-hop queries compared to static RAG pipelines.

---

[EMAgnet: Parameter-Space EMA Regularization for Policy Gradient Self-Play in Large Games](http://arxiv.org/abs/2606.23995)

- EMAgnet: introduces a parameter-space exponential moving average regularization method for PPO self-play that adapts to the agent's evolving strategy by maintaining a moving target of network weights.
- The framework utilizes a Policy Network, Magnet Parameters, and an EMA Update to effectively regularize toward viable actions while ignoring strictly dominated strategies in complex game environments.
- By incorporating a KL Divergence Loss term, the approach enables stable convergence in two-player zero-sum games with large strategy spaces where uniform regularization typically fails.

---

[Learning to Trigger: Reinforcement Learning at the Large Hadron Collider](http://arxiv.org/abs/2606.23993)

- GFPO: introduces a self-driving trigger framework that adapts selection thresholds online using DQN, GRPO, GFPO-F, and GFPO-FR to maintain stable background rates while maximizing signal efficiency.
- The framework utilizes a Recurrent Encoder to process event sequences and a Safety Shield to enforce operational constraints, ensuring robust performance under non-stationary physical conditions.
- GFPO-F and GFPO-FR address the zero-feasibility failure mode of standard policy optimization by enforcing rate constraints before advantage normalization, demonstrating superior generalization to real collision data.

---

[Critique of Agent Model](http://arxiv.org/abs/2606.23991)

- GIC (Goal-Identity-Configurator): introduces a general-purpose agent architecture that internalizes agency by combining a Belief Encoder, Goal Decomposer, Identity Evolver, Configurator, Simulative Planner, and Actor, all operating with a separately trained World Model and Critic.
- The framework distinguishes between agentic systems, which rely on external scaffolding, and agentive systems, which derive capabilities endogenously through internalized modules for planning, self-regulation, and learning.
- GIC enables persistent, self-directed operation by allowing the agent to autonomously manage its deliberative mode, goal decomposition, and identity evolution, thereby improving auditability and safety compared to monolithic LLM-based agent designs.

---

[Forget Without Compromise: Nexus Sampling for Streaming KV-Cache Eviction Under Fixed Budgets](http://arxiv.org/abs/2606.23961)

- Nexus Sampling: introduces a training-free KV cache eviction method that replaces deterministic top-K selection with Nexus scoring and weighted reservoir selection to prevent monotone marginal erosion.
- The framework utilizes an iterative walk to identify bridge tokens that anchor strongly-connected clusters, ensuring these are preserved alongside marginal tokens via weighted reservoir sampling.
- By treating KV cache eviction as a streaming problem, Nexus Sampling maintains long-run token survival through a product-over-steps probability, significantly outperforming existing methods on retrieval-heavy long-context tasks.

---

[When Retrieval Metrics Mislead: Measuring Policy Signal in Long-Horizon Tool-Use Agents](http://arxiv.org/abs/2606.23937)

- SSDG (Structured State Decision Gate): introduces a framework for evaluating policy-conditioned tool-use agents by comparing structured decision-state representations against raw trajectory inputs.
- The framework utilizes a generator to distill dialogue, tool outputs, and policy assertions into a compact structured state, which significantly improves the macro-F1 performance of LLM classifiers compared to raw text.
- The study demonstrates that exact-match retrieval recall is a pessimistic proxy for downstream policy utility, as non-matching retrieved clauses often contain sufficient decision-relevant information for effective classification.

---

[RIFT-Bench: Dynamic Red-teaming For Agentic AI Systems](http://arxiv.org/abs/2606.23927)

- RIFT-Bench: introduces a representation-driven methodology for dynamic red-teaming of agentic AI systems, utilizing NodeSpec to enable transferable adversarial evaluation across heterogeneous architectures.
- The framework operates through an automated Discovery phase to extract system structure and a Scanning phase to deploy adaptive adversarial probes, providing a scalable foundation for security assessment.
- RIFT-Bench includes a benchmark of 45 diverse agentic systems and 105 adversarial probes, demonstrating effective vulnerability assessment and mitigation strategy evaluation across various agentic implementations.

---

[Topological Online Learning for Displacement-based Formation Control](http://arxiv.org/abs/2606.23901)

- TOLD (Topological Online Learning for Displacement-based formation control): introduces a real-time edge-level adaptation framework that optimizes interaction topology weights to minimize formation distortion, utilizing Formation Sensing, Online Gradient Flow (OGF), Online Exponential Gradient Flow (OExpGF), Formation Control Fusion, and Robot Execution.
- The framework employs OGF for unconstrained weight updates to achieve superior formation accuracy and OExpGF for non-negative convex weight updates to ensure bounded, predictable dynamics.
- TOLD functions as a modular topological routing overlay that integrates with physical-layer robust controllers to enhance multi-robot formation robustness against environmental disturbances and sensor noise.

---

[REALM: A Unified Red-Teaming Benchmark for Physical-World VLMs](http://arxiv.org/abs/2606.23892)

- REALM: introduces a unified red-teaming benchmark for physical-world VLMs that integrates 12 attack methods, 3 model-agnostic defenses, and 13 victim VLMs under a standardized black-box protocol.
- The framework utilizes an agentic target-generation pipeline to construct scenario-specific, physically grounded adversarial objectives that enable fair comparison across diverse attack families.
- Evaluation results demonstrate that text-channel injection attacks induce the most frequent failures, while model scale alone does not guarantee adversarial robustness in physical-world tasks.

---

[Complex Autonomous UAV Task Execution and Decision-Making With s(CASP)](http://arxiv.org/abs/2606.23866)

- VECSR-A (Virtually-Embodied Common Sense Reasoner for Autonomous UAVs): introduces a symbolic state-centered agent architecture that utilizes s(CASP) for constraint-based commonsense reasoning to enable explainable and adaptive UAV task execution in high-fidelity simulated environments.
- The framework integrates a Simulation Environment, s(CASP) Knowledge Base, VECSR-A Orchestrator, and s(CASP) Goal-Directed Server to perform hierarchical task decomposition and real-time decision-making without requiring model retraining.
- By leveraging goal-directed answer set programming, the system provides verifiable justification trees for autonomous actions, ensuring transparency and reliability in safety-critical mission scenarios.

---

[From Task-Guided Conversational Graphs to Goal-Oriented Dialogue Runtimes](http://arxiv.org/abs/2606.23797)

- GODR (Goal-Oriented Dialogue Runtime): introduces a framework-neutral architectural layer that manages conversational goal lifecycles, resumption contracts, and dependencies as first-class runtime objects above existing execution substrates.
- The framework separates objective continuity from execution continuity by delegating bounded tasks to LLM-based agents and tools while maintaining explicit goal state, policy, and auditability.
- GODR addresses the Multi-Objective Interruptible Dialogue Problem by providing a taxonomy of goal complexity and a formal operational model for managing suspended, resumed, and invalidated user objectives.

---

[GUI vs. CLI: Execution Bottlenecks in Screen-Only and Skill-Mediated Computer-Use Agents](http://arxiv.org/abs/2606.24551)

- GUI vs. CLI: introduces a matched execution-layer benchmark to isolate interaction modality effects by holding goals, initial states, and verifiers constant across GUI agents and skill-mediated CLI agents.
- The study identifies that GUI agents are primarily bottlenecked by visual grounding and long workflow execution, while CLI agents are limited by skill coverage gaps and implicit-default reconstruction.
- Diagnostic experiments reveal that verifier-guided skill patching significantly improves CLI performance, highlighting skill construction and validation as the central scaling challenge for programmatic computer-use agents.

---

[Cryptographic certificates of validity for trustworthy AI](http://arxiv.org/abs/2606.23768)

- Cryptographic certificates of validity for trustworthy AI: introduces a mechanism to certify agentic AI actions by compiling formal correctness predicates into polynomial constraints verified via succinct cryptographic proofs.
- The framework enables untrusted agents to provide independently checkable evidence of policy compliance, bridging the gap between high-level formal specifications and cryptographic proof back ends.
- By utilizing polynomial arithmetisation, the approach allows for the verification of complex agent behaviors, including recursive computations, without requiring the verifier to trust the agent or inspect private witness data.

---

[Emergent Relational Order in LLM Agent Societies: From Collective Affect to Authority Stratification](http://arxiv.org/abs/2606.23764)

- CAREB-MAS: introduces a multi-agent simulation framework that models the emergence of social structures from micro-level cognitive mechanisms grounded in Affect Control Theory, Social Identity Theory, and Durkheimian collective affect.
- The framework utilizes an augmented BDI architecture that includes Empathy Core, Ethical Resonator, and A-BDI Decision Module to simulate how agents maintain egocentric identities and relational order without explicit cultural rules.
- Experimental results demonstrate that the framework reproduces core social phenomena such as stable labor specialization, guanxi-based economic ethics, and authority stratification, which vary systematically based on production structure.

---

[Engineering Reliable Autonomous Systems: Challenges and Solutions](http://arxiv.org/abs/2606.23760)

- ERAS: introduces a workshop report outlining challenges and pathways for engineering reliable autonomous systems through integrated verification and architectural design.
- The framework utilizes assumption/guarantee contracts to link heterogeneous components, including vision-, planning-, and hardware-agents, enabling modular verification.
- The report provides a research roadmap and case study analysis to bridge the gap between academic verification techniques and real-world autonomous system deployment.

---

[ESAA-Conversational: An Event-Sourced Memory Layer for Continuity, Handoff, and Curation Across Heterogeneous LLM Coding Agents](http://arxiv.org/abs/2606.23752)

- ESAA-Conversational: introduces an event-sourced memory layer that captures conversational turns from heterogeneous LLMs into a shared, append-only log to ensure continuity and handoff without requiring a common runtime.
- The framework utilizes a CLI to perform inverted ingestion, where native agent logs are normalized into events, and read models are deterministically projected to provide agents with relevant context.
- By separating mechanical capture from agent-driven curation, the system enables collaborative multi-agent workflows while maintaining operational auditability and privacy through a greenfield distribution model.

---

#### 21st June 2026

[Habituation at the Gate: Rising Approval and Declining Scrutiny in Human Review of AI Agent Code](http://arxiv.org/abs/2606.22721)

- AIDev: introduces a longitudinal analysis of human code review behavior, demonstrating that reviewers systematically increase approval rates for AI-generated pull requests as they accumulate experience.
- The study reveals that rising approval rates are accompanied by a significant decline in inline comment effort and increased review latency, suggesting reflexive habituation under growing workloads.
- The research provides evidence that this trend is agent-specific and experience-driven, rather than a result of improved code quality or general reviewer leniency, raising concerns for human oversight in software development.

---

[Beyond Simpson’s Paradox: A Cascade of Confounders in AI Agent Pull-Request Co-Authorship](http://arxiv.org/abs/2606.22711)

- Collaboration-level framework: introduces a four-level taxonomy of human involvement in LLM-based agent pull requests, ranging from L0: Fully autonomous to L3: Human-both.
- The study demonstrates that aggregate negative associations between human co-authorship and pull request merge rates are a manifestation of Simpson’s Paradox driven by agent composition and structural confounders.
- Empirical analysis reveals that once repository selection and pull request structure are controlled, the apparent causal benefit of human co-authorship in LLM-generated code dissolves.

---

[Libretto: Giving LLM Agents a Sense of Musical Structure](http://arxiv.org/abs/2606.22708)

- Libretto: introduces an agent-facing framework for symbolic music generation and revision that utilizes a Grammar, a Structural Axis System, an Agent Loop, Knowledge Bases, and Application-Specific Task Setups.
- The framework represents music as a measurable, editable object by mapping symbolic sequences into a 29-axis statistical fingerprint, enabling LLMs to diagnose and refine compositions through iterative feedback.
- Libretto supports diverse musical tasks including gap filling, full-piece generation, gradual style morphing, and educational drill creation without requiring specialized training for each application.

---

[VERIPORT: Automated and Verified Patch Backporting at Scale](http://arxiv.org/abs/2606.22704)

- VERIPORT: introduces an end-to-end agentic system that automates the backporting of security patches across affected software versions by generating and verifying evidence through AUTORESEARCH, AUTOPOC, AUTOTESTER, and AUTOPATCH.
- The framework utilizes a multi-stage pipeline where AUTOPOC and AUTOTESTER build vulnerability and functionality oracles to ensure that backported patches are both effective and non-breaking.
- By decomposing the backporting process into bounded, independently verified stages, VERIPORT mitigates LLM-based failure modes and achieves high reliability in patch adaptation across diverse software versions.

---

[Black-Box Forensics for Conversational LLM Agents](http://arxiv.org/abs/2606.22698)

- Black-Box Forensics for Conversational LLM Agents: introduces a framework for identifying base models and system prompts of conversational LLMs through active elicitation and non-adversarial dialogue.
- The framework utilizes a Detective Agent to steer conversations, enabling the extraction of behavioral signatures that are processed by Cross-Encoder and Bi-Encoder components to perform zero-shot fingerprinting of unseen system prompts.
- The approach achieves high accuracy in attributing base models and fingerprinting system prompts without requiring access to model weights, output logits, or prior training data from the target agent.

---

[VISTA Architect: A graph database-oriented health AI system demonstrated in multidisciplinary tumor boards](http://arxiv.org/abs/2606.22692)

- VISTA Architect: introduces a database-oriented AI architecture that integrates LLMs with longitudinal EHRs by transforming clinical documentation into a persistent, provenance-linked knowledge graph, utilizing MEDS Graph, TOA Graph, Agentic AI Bridge, User Interface, and Provenance Links.
- The system employs a two-tiered graph structure where the MEDS Graph preserves granular EHR data and the TOA Graph synthesizes this into a queryable, temporally coherent clinical narrative.
- By precomputing clinical synthesis at ingestion, the architecture enables deterministic graph traversal for downstream applications, significantly reducing computational overhead and latency compared to traditional RAG approaches.

---

[Integrated cloud-based architecture for robot-robot and human-robot collaboration using ROS 2 – MQTT in Mediterranean Greenhouses](http://arxiv.org/abs/2606.22682)

- GMaaS (Greenhouse Models as a Service): introduces a cloud-based hybrid architecture that establishes a bidirectional communication bridge between ROS 2 edge-computing platforms and the iVeg Decision Support System using MQTT and FIWARE to enable real-time multi-robot and human-robot collaboration.
- The framework utilizes YOLOx for real-time edge-based detection of farmers and robots, converting visual data into lightweight geometric metadata to maintain fleet-wide situational awareness under severe communication constraints.
- By integrating ROS 2 with MQTT and FIWARE, the architecture provides a scalable, decentralized solution for managing complex robotic missions in occluded greenhouse environments while ensuring low-latency safety overrides and human-in-the-loop oversight.

---

[RigorBench: Benchmarking Engineering Process Discipline in Autonomous AI Coding Agents](http://arxiv.org/abs/2606.22678)

- RigorBench: introduces a process-oriented benchmarking framework that evaluates AI coding agents based on engineering discipline rather than just final outcome correctness.
- The framework utilizes a trajectory-based scoring pipeline that incorporates Trajectory Parser, Signal Extractor, Pillar Scorer, LLM-as-judge Panel, and an Instrumented Docker Environment to measure five pillars of software engineering.
- Experimental results demonstrate that structured process discipline significantly improves both process quality and downstream outcome correctness in LLMs.

---

[AgentLens: Interpretable Safety Steering via Mechanistic Subspaces for Multi-Turn Coding Agent](http://arxiv.org/abs/2606.22673)

- AgentLens: introduces a white-box defense framework that performs runtime safety detection and representation-level mitigation for multi-turn coding agents by leveraging Linear Probe and Mitigation Stage.
- The framework utilizes a Safety Detection Stage to identify harmful execution states from step-level hidden representations and an LLM Controller to adaptively search for steering strengths that suppress harmful actions.
- AgentLens employs a Steering Engine to apply representation-level interventions at specific layers, guided by the MAS Benchmark to ensure robust safety across multi-turn coding agent trajectories.

---

[RAVEN: Agentic RAG for Automated Vulnerability Repair](http://arxiv.org/abs/2606.22647)

- RAVEN: introduces an agentic RAG-based framework that integrates multi-faceted historical fix retrieval, repository-level context dependency analysis, and iterative patch refinement to autonomously repair real-world software vulnerabilities.
- The framework utilizes open-source LLMs, including Gemma-4-26B and Nemotron-3-30B, to perform locally deployable, cost-effective, and scalable vulnerability repair across diverse programming languages and CWE categories.
- RAVEN employs a dedicated Curator Agent for repository exploration and an iterative feedback loop between the Patch Generator and Patch Reviewer to ensure high-quality, semantically meaningful, and secure code patches.

---

[SkillAudit: From Fixed-Suite Benchmarking to Skill-Centered Assessment](http://arxiv.org/abs/2606.22613)

- SkillAudit: introduces an end-to-end framework for skill-centered assessment that automatically generates multi-dimensional evaluation reports for arbitrary agent skills by leveraging Skill Collection, Scheme Generation, Task Construction, Sandboxed Execution, Automated Evaluation, and a Browser Extension.
- The framework employs a baseline comparison principle to isolate the marginal utility and efficiency/cost gains of a skill from the backbone LLM's intrinsic capabilities.
- To assess safety, SkillAudit utilizes a two-stage detection paradigm combining static semantic analysis of skill packages with dynamic runtime verification to determine both the existence and exploitability of potential risks.

---

[Harnessing Agents for Autonomous Research and Human-in-the-Loop Refinement](http://arxiv.org/abs/2606.22610)

- PAPERCLAW: introduces a harnessed multi-agent system that automates the entire research lifecycle from domain curation to venue-compliant paper generation using a stoppable iterative hypothesis map.
- The system utilizes an in-cycle research assistant and full-lifecycle memory to maintain project context, enabling autonomous execution and human-in-the-loop refinement across web, desktop, and command-line interfaces.
- PAPERCLAW enforces scientific discipline through structural constraints, including pre-registered hypotheses, deterministic citation validation, and hardware-aware feasibility gating to ensure trustworthy autonomous research.

---

[Sol Video Inference Engine: Agent-Native Full-Stack Acceleration Framework for Efficient Video Generation](http://arxiv.org/abs/2606.23743)

- Sol Video Inference Engine: introduces an agent-native framework that automates instance-specific acceleration for video diffusion models by orchestrating Parallel Skill Agents, an Agent Integrator, and a Human Validator.
- The framework optimizes inference through a full-stack approach, integrating Diffusion Cache, Token Pruning, Sparse Attention, Kernel Fusion, and Quantization to achieve significant speedups.
- By treating acceleration as an instance-specific tuning problem, the system bypasses manual engineering, delivering over 2× end-to-end speedup across diverse models like Cosmos3-Super, LTX-2.3, and SANA-Video.

---

[Stationary Robust Mean-Field Games under Model Mismatches](http://arxiv.org/abs/2606.22579)

- Robust Mean-Field Games (Robust MFG): introduces a framework for infinite-horizon stationary distributionally robust mean-field games that incorporates model uncertainty directly into population-coupled dynamics.
- The framework utilizes a robust dynamic programming principle with a contractive Bellman operator to prove the existence of a stationary robust mean-field equilibrium via fixed-point arguments.
- The paper provides the first convergent algorithm for robust MFGs and establishes non-asymptotic error bounds for approximating finite-player robust games as the population size increases.

---

[MacAgentBench: Benchmarking AI Agents on Real-World macOS Desktop](http://arxiv.org/abs/2606.22557)

- MacAgentBench: introduces a comprehensive macOS desktop benchmark comprising 676 tasks across 25 applications, utilizing a Docker-QEMU infrastructure, deterministic rule-based evaluation, and fine-grained multi-checkpoint scoring.
- The benchmark supports diverse CUA paradigms, including pure GUI agents, hybrid frameworks, and agent harnesses, with nearly 60% of tasks involving both GUI and CLI interaction.
- Experiments reveal that framework design and skill augmentation significantly impact agent performance, with fine-grained evaluation exposing capability imbalances that binary pass/fail metrics fail to capture.

---

[Governance Decay: How Context Compaction Silently Erases Safety Constraints in Long-Horizon LLM Agents](http://arxiv.org/abs/2606.22528)

- Governance Decay: introduces a failure mode where LLM agents lose in-context safety constraints during routine context compaction, leading to unauthorized tool-call violations.
- The paper presents the ConstraintRot benchmark to quantify how compaction-induced deletion of soft organizational policies significantly increases agent violation rates compared to hard safety norms.
- The authors propose Constraint Pinning as a training-free defense that protects governance constraints from lossy compaction by quarantining and re-injecting them into the agent's context.

---

[Imagine to Ensure Safety in Hierarchical Reinforcement Learning](http://arxiv.org/abs/2606.22509)

- ITES (Imagine To Ensure Safety in HRL): introduces a hierarchical reinforcement learning framework that integrates a world model and a cost model to enforce safety constraints during both subgoal generation and execution.
- The framework utilizes a high-level policy to generate safe subgoals and a low-level policy that leverages imagined rollouts within a learned world model to avoid unsafe behaviors.
- By combining hierarchical decomposition with model-based safety verification, the approach effectively addresses long-horizon tasks while maintaining strict adherence to safety constraints.

---

[Lingering Authority: Revocable Resource-and-Effect Capabilities for Coding Agents](http://arxiv.org/abs/2606.22504)

- PORTICO: introduces a reference monitor that enforces capability lifetime for coding agents by compiling task contracts into initial envelopes, grant rules, and closure rules to prevent lingering authority.
- The framework utilizes a request-grant-invoke lifecycle where the planner requests authority, the monitor mints epoch-bound handles, and closure rules remove these handles from the interface once the justification ends.
- PORTICO ensures that only task-justified capabilities are exposed to the LLM planner, effectively mitigating stale-handle reuse and unauthorized side effects in coding agent environments.

---

[WebCQ: Cooperative Multi-Agent Deep Reinforcement Learning for Scalable Web GUI Testing](http://arxiv.org/abs/2606.22502)

- WebCQ: introduces a cooperative multi-agent deep reinforcement learning approach for scalable web GUI testing, utilizing DQN-based Decentralized Execution, QTRAN-based Centralized Training, a Lightweight Synchronization Mechanism, a Hybrid Reward Function, State Vector, Action Vector, Q-network, Global Buffer, and Local Buffer.
- The framework employs a centralized training and decentralized execution paradigm to coordinate multiple agents, effectively managing large-scale web state spaces through deep reinforcement learning.
- WebCQ significantly improves exploration efficiency and failure detection by replacing tabular reinforcement learning with deep Q-networks and advanced multi-agent coordination.

---

[Grounded Scaling: Why Agentic AI Needs Deterministic Environments](http://arxiv.org/abs/2606.22495)

- Grounded Scaling: introduces a framework that identifies environmental determinism as a critical, under-discussed binding axis for AGI-to-ASI progress, addressing bottlenecks through Data Wall, Abstraction Barrier, Embodied Bottleneck, Multi-Agent Trust, Supply Certainty Index (SCI), Determinism Maturity Model (DMM), Data Flywheel, A2A Flywheel, Verifier Service, Skill Registry, A2A Router, and Determinism Telemetry Plane.
- The paper formalizes the Determinism–Efficiency Bound and Verifier–Goodharting Floor to demonstrate how stochastic environments exponentially degrade long-chain agent performance.
- It proposes a five-level Determinism Maturity Model and a reference architecture to transition agentic environments from human-optimized stochastic interfaces to deterministic, verifiable substrates.

---

[VADAOrchestra: Neurosymbolic Orchestration of Adaptive Reasoning Workflows](http://arxiv.org/abs/2606.22485)

- VADAOrchestra: introduces a neurosymbolic framework that integrates an LLM-based Orchestrator for dynamic planning with a VADALOG Engine for verifiable symbolic reasoning.
- The framework utilizes a Dependency Graph to enforce domain-specific constraints and a Logical Trace to ensure auditability and reproducibility of the reasoning process.
- By decoupling high-level orchestration from symbolic inference via MCP Tools, the architecture addresses scalability limitations and improves faithfulness compared to standard agentic systems.

---

[Governed AI-Assisted Engineering: Graduated Human Oversight for Agentic Code Generation in Regulated Domains](http://arxiv.org/abs/2606.22484)

- GAIE: introduces a three-tier graduated human oversight model that routes agentic code generation tasks through OCM, Supervisor Agent, Collaborator Agents, Human Interaction Layer, Evidence Capture Layer, Monitoring Subsystem, and Immutable Store to ensure regulatory compliance.
- The framework utilizes a deterministic OCM to classify tasks by regulatory impact, customer proximity, reversibility, and data sensitivity, assigning them to appropriate oversight tiers ranging from human-in-the-loop to fully automated monitoring.
- GAIE preserves agentic coding velocity by applying proportionate oversight, ensuring that high-risk strategic functions receive rigorous human review while internal tasks benefit from automated compliance evidence generation.

---

[A Differentiable Atari VCS: A Complex, Fully Known Ground Truth for Explainable AI](http://arxiv.org/abs/2606.22447)

- Differentiable Atari VCS: introduces a fully differentiable, bit-exact re-implementation of the Atari 2600 architecture, comprising 6507 CPU, TIA, RIOT, and cartridge ROM, to provide a verifiable ground truth for XAI methods.
- The framework utilizes a dual-mode execution architecture, featuring a hard-mode emulator for bit-exact forward passes and a soft-mode emulator that employs a straight-through estimator to generate surrogate gradients for explainability analysis.
- By treating ROM as a weight tensor and RAM as a differentiable tape, the system enables gradient-based attribution on a complex, fully specified computer architecture, allowing researchers to validate XAI techniques against known internal mechanisms.

---

[SVGym (SciVerseGym): An Environment for Reinforcement Learning and Bayesian Optimization in Crystal Discovery](http://arxiv.org/abs/2606.22425)

- SciVerseGym: introduces a Gymnasium-compatible environment for closed-loop crystal discovery that decouples search algorithms from materials-specific infrastructure using a standardized transition protocol.
- The framework supports diverse search strategies, including RL, Bayesian optimization, evolutionary search, and language-agent workflows, by providing a unified interface for structured crystal edits and physically informed feedback.
- By utilizing a bounded action schema and modular evaluation backends, SciVerseGym enables reproducible benchmarking and rapid prototyping of materials discovery strategies under consistent physical assumptions.

---

[Curvature-Adaptive Consistency Flow Matching: Autonomous Trajectory Optimization via Reinforcement Learning](http://arxiv.org/abs/2606.22394)

- CACFM: introduces a reinforcement learning-based framework that reformulates consistency distillation as a dynamic decision process to optimize training trajectories.
- The framework utilizes an RL agent to identify and prioritize high-curvature trajectory segments, replacing static sampling heuristics with an emergent curriculum.
- By integrating a hybrid objective of structural consistency, distribution matching, and adversarial losses, the model achieves state-of-the-art performance in few-step image generation.

---

[PlanBench-XL: Evaluating Long-Horizon Planning of LLM Tool-Use Agents in Large-Scale Tool Ecosystems](http://arxiv.org/abs/2606.22388)

- PlanBench-XL: introduces an interactive benchmark for evaluating long-horizon planning of LLM agents in large-scale, retrieval-mediated, and unreliable tool ecosystems.
- The framework utilizes a scalable generation pipeline to create complex retail tasks, incorporating bi-directional exploration and retrieval-time blocking to stress-test agent adaptation and re-planning capabilities.
- Comprehensive evaluation of frontier LLMs reveals that agents struggle with massive-tool planning, particularly when faced with silent tool failures or the need for longer recovery paths.

---

[MetaPS: Adaptive Programmatic Strategy Selection for Market Agents](http://arxiv.org/abs/2606.22385)

- MetaPS: introduces a simulation-guided framework for adaptive programmatic strategy selection that decouples high-level strategy choice from low-level action execution.
- The framework utilizes a MetaPS Router to select from a library of Executable Strategy Modules based on market state, with supervision derived from counterfactual rollouts in a Market Simulator/Backtester.
- By delegating final action generation to a Deterministic Execution Layer, MetaPS improves interpretability and performance compared to direct LLM-based action prediction.

---

[Select-to-Act: Hierarchical Reinforcement Learning via Adaptive Language Guidance](http://arxiv.org/abs/2606.22350)

- HRLLI: introduces a hierarchical reinforcement learning framework that decomposes instruction usage into a high-level selector and a low-level executor to adaptively ground language instructions into stage-specific decisions.
- The framework utilizes an Instruction Encoder, State Encoder, High-Level Instruction Policy (Selector), Low-Level Action Policy (Executor), Reward Model, High-Level Buffer, and Low-Level Buffer to optimize instruction selection and action execution simultaneously.
- By explicitly modeling instruction selection as a discrete decision process, the approach improves sample efficiency and interpretability in instruction-heavy reinforcement learning environments.

---

[Hypothesis-Driven Skill Optimization for LLM Agents](http://arxiv.org/abs/2606.22330)

- HDSO (Hypothesis-Driven Skill Optimization): introduces a train-free framework that decouples a frozen curator LLM from a frozen executor LLM to iteratively optimize agent skills through a hypothesis-driven lifecycle.
- The framework utilizes paired control/treatment executions to validate candidate skills, ensuring only supported procedural knowledge is consolidated into an approved repository.
- HDSO maintains an auditable skill lifecycle by recording rejected hypotheses and validation evidence, providing robustness against noisy feedback and enabling cross-model skill transfer.

---

[BabelJudge: Measuring LLM-as-a-Judge Reliability Across Languages and Agent Trajectories](http://arxiv.org/abs/2606.22329)

- BabelJudge: introduces a benchmark and reliability audit framework that measures LLM-as-a-judge failure modes using gold-labelling-by-degradation to eliminate human annotation requirements.
- The framework utilizes a Multilingual Corpus, Perturbation Engine, and Gold-Labeled Item components to systematically probe for position bias, verbosity bias, order inconsistency, and cross-lingual degradation.
- BabelJudge extends to agentic settings by incorporating trajectory-level perturbations and specific metrics to evaluate tool-calling reliability and hallucination blindness in LLMs.

---

#### 20th June 2026

[REVELIO: Cost-Efficient Agentic Memory Safety Vulnerability Detection For Repository-Scale Codebases](http://arxiv.org/abs/2606.22263)

- REVELIO: introduces an end-to-end agentic framework that decomposes memory safety vulnerability detection into a high-recall hypothesis generation stage using inexpensive LLMs and a high-precision confirmation stage using stronger LLMs and deterministic sanitizers.
- The framework utilizes a funnel-style pipeline that includes hypothesis generation-, triage-, PoV construction- and reporting-agents to systematically identify and verify vulnerabilities while maintaining zero false positives.
- By enforcing a programmatic workflow that separates speculative reasoning from executable confirmation, REVELIO achieves scalable, cost-efficient, and trustworthy vulnerability discovery in large C/C++ codebases.

---

[Quantifying Theoretical AI Alignment Guarantees: Receiver-Utility Bounds in Bayesian Persuasion](http://arxiv.org/abs/2606.22226)

- BPM (Bayesian Persuasion Model): introduces a theoretical framework to quantify receiver-utility bounds in scenarios where a misaligned AI sender influences a human receiver's decisions through strategic signaling.
- The framework utilizes a bit-string model to isolate the mechanism of incentive-driven information transfer, establishing a universal 3/2 upper bound on the ratio of sender-optimal receiver utility to prior-only utility.
- The research provides analytical proofs for product and near-product priors while demonstrating via a six-bit counterexample that no universal 5/4 bound exists for receiver utility.

---

[Lexical Consensus: Grounded Word Learning and Shared Meaning in Artificial Agents](http://arxiv.org/abs/2606.22207)

- Lexical Consensus: introduces an experimental framework for studying grounded word learning over a structured perceptual substrate using frozen DINOv2 visual embeddings and interpretable lexical learners.
- The framework evaluates whether artificial agents can acquire, stabilize, and use novel lexical mappings bidirectionally, demonstrating that acquisition is constrained by a perceptual-coherence gradient.
- Multi-agent experiments reveal that while consensus refines lexical agreement, shared perceptual geometry remains the dominant stabilizing force, with minimal representational reorganization.

---

[When Is Emergent Consensus Real? A Measured Coupling Gain and a Validity Diagnostic for LLM Agent Societies](http://arxiv.org/abs/2606.22203)

- LLM Agent Societies Framework: introduces a measurement protocol using coupling gain γ and an authenticity diagnostic to distinguish genuine social dynamics from model-prior artifacts in LLM agent interactions.
- The framework utilizes counterfactual perturbation to measure per-agent susceptibility, revealing that pairwise coupling often mis-estimates multi-agent outcomes compared to modality-matched group pull.
- By applying classical opinion dynamics like Friedkin–Johnsen and signed-Laplacian, the research provides a falsifiable prediction method for consensus, pluralism, and induced polarization in LLM societies.

---

[Drowning in Routine: Signal Dilution in Multi-Turn Agent Training](http://arxiv.org/abs/2606.22164)

- Drowning in Routine: introduces a signal dilution mechanism where routine turns in multi-turn agent training inject gradient variance into trajectory-level estimators without providing expected signal.
- The framework demonstrates that the efficiency gap between trajectory-level and turn-level estimators is governed by decision density, scaling as Θ(ρ⁻¹/²).
- The research validates these findings using the Diluted Doors environment, showing that turn-level credit assignment is superior when consequential decisions are sparse and critic error is controlled.

---

[Novelty-Aware Agentic Retrieval: Comparing Research Contributions Through Structured Multi-Step Reasoning](http://arxiv.org/abs/2606.22151)

- Novelty-Aware Research Agent: introduces a six-component agentic retrieval pipeline that layers structured multi-step reasoning on a RAG pipeline to compare research contributions across a corpus.
- The system utilizes a Query Analyzer, Retriever (ReAct Loop), Ranker, Contribution Extractor, Comparison Agent, and Answer Generator to produce structured artifacts including per-paper records, overlaps, and a deterministic problem × method gap matrix.
- By enforcing a schema at extraction time, the framework treats retrieved documents as structured data, enabling comparative analysis that goes beyond independent document summarization.

---

[RoboLineage: Agent-Native Data Lifecycle Governance Across Robot Policy Iterations](http://arxiv.org/abs/2606.22142)

- RoboLineage: introduces an agent-native governance system that transforms robot policy iteration into an explicit, auditable data lifecycle by utilizing Robot Onboarding Agent, Task Config Agent, Online Visual Snapshot Agent, Post-Rollout Review Agent, Data Governance Agent, Data Health Agent, Framework Discovery Agent, Dataset Adapter Agent, Training Monitor Agent, Version Governance Agent, Policy Evaluation Agent, Deployment Governance Agent, and Master Agent.
- The framework employs a typed lineage graph to link evidence, artifacts, and decisions across policy iterations, ensuring reproducibility and accountability in robot learning workflows.
- RoboLineage reduces expert-mediated lifecycle labor by automating routine review and data curation while maintaining policy quality through agent-interpreted, schema-validated artifacts.

---

[TraceView: Interactive Visualization of Agentic Program Repair Trajectories](http://arxiv.org/abs/2606.22110)

- TraceView: introduces an interactive visualization tool designed to help developers diagnose and analyze the reasoning, tool use, and feedback trajectories of LLM-based automated program repair agents.
- The framework structures agent logs into Thought, Action, and Result components, enabling users to perform semantic relation labeling and inspect repair workflows through coordinated graph views.
- By providing filtering, metrics, and node-level evidence inspection, the tool facilitates the identification of patterns such as alignment, repetition, divergence, and misinterpretation in agentic repair attempts.

---

[CodeTeam: An LLM-Powered Multi-Agent Framework for Repository-Level Code Generation](http://arxiv.org/abs/2606.22082)

- CodeTeam: introduces a multi-agent framework for repository-level code generation that separates planning, decision making, and implementation into distinct, coordinated stages using Architect agents, a CTO agent, Developer agents, and a QA agent.
- The framework utilizes a software design sketch as a machine-checkable contract to govern implementation, supported by an optional RAG subsystem for architectural planning and a Git-based coordination mechanism for iterative repair.
- Experimental results on SketchEval and NL2Repo-Bench demonstrate that CodeTeam significantly improves repository-level code generation quality by addressing cross-file consistency and structural integrity through its specialized multi-agent workflow.

---

[Deep RL-Tuned Model-Free Adaptive Control for Lower-Limb Exoskeletons During Sit-to-Stand Transitions](http://arxiv.org/abs/2606.22040)

- Deep RL-Tuned Model-Free Adaptive Control framework: introduces a model-free adaptive backstepping control strategy for lower-limb exoskeletons that utilizes an Ultra-local model, a Gaussian RBF neural estimator, and a TD3 supervisory gain scheduler to achieve precise trajectory tracking during sit-to-stand transitions.
- The framework integrates an online Gaussian RBF neural estimator to approximate unknown system dynamics in real time, eliminating the need for explicit system identification.
- A TD3 reinforcement learning agent acts as a supervisory gain scheduler, continuously adapting controller parameters across distinct sit-to-stand phases to enhance tracking accuracy and robustness against external disturbances.

---

[Nous: A Predictive World Model for Long-Term Agent Memory](http://arxiv.org/abs/2606.22030)

- Nous: introduces a predictive world model for agent memory that represents entity-attribute pairs as categorical probability distributions updated via information-theoretic surprise and Bayesian inference.
- The architecture replaces static fact storage with a dynamic system that treats knowledge as a generative model, where forgetting occurs through entropy decay and conflict resolution is handled by shifting probability mass.
- Nous utilizes an ingestion pipeline for belief updates and a query pipeline involving entity-based retrieval and profile assembly to provide context to an LLM for response generation.

---

[Holmes: Multimodal Agentic Diagnosis for Mixed-Language Mobile Crashes at Industrial Scale](http://arxiv.org/abs/2606.21963)

- Holmes: introduces a multi-agent framework that automates root cause analysis for mobile crashes by synthesizing multimodal runtime signals through a hierarchical Retrieve-Explore-Reason architecture.
- The framework utilizes specialized agents including a Dispatcher, Stack Code Retriever, Log Miner, Thread Inspector, Code Explorer, and a Lead Analyst to reconstruct failure contexts without requiring reproducible environments.
- By integrating low-level artifacts like registers and assembly with high-level logs and code, Holmes bridges semantic gaps in mixed-language industrial codebases to achieve high-precision fault localization.

---

[OpenBioRQ: Unsolved Biomedical Research Questions for Agents](http://arxiv.org/abs/2606.21959)

- OpenBioRQ: introduces a retrieval-grounded agentic benchmark of 12,553 unsolved biomedical research questions that evaluates LLMs on faithfulness and abstention using an agentic harness, biomedical REST APIs, a frozen per-question checklist, a retrieval-grounded status verifier, and a replay cache.
- The benchmark addresses the failure mode of wrong-paper citations where LLMs provide resolvable but unsupporting evidence, surfacing agentic collapse where models abandon tool use on difficult questions.
- OpenBioRQ provides a reproducible evaluation framework for literature synthesis and grounding, explicitly excluding clinical decision support to mitigate risks of generating authoritative-looking medical misinformation.

---

[From RAN Control to Agentic Intelligence: Architecture and Vision for Energy Efficient AI-RAN](http://arxiv.org/abs/2606.21955)

- E-ARC (Energy-Aware Agentic RAN Control): introduces an agentic orchestration architecture that leverages an LLM-based SC, a DT for offline validation, and a CDM agent to optimize energy efficiency in AI-native RANs.
- The framework bridges O-RAN and AI-RAN paradigms by interpreting high-level operator intents into validated, energy-aware control actions for rApps and xApps.
- E-ARC enables closed-loop orchestration by integrating network telemetry and application-level context to balance communication performance, AI workload demands, and energy consumption.

---

[ISCSLP 2026 CoT-TTS Challenge: Chain-of-Thought Reasoning for Context-Aware Text-to-Speech](http://arxiv.org/abs/2606.21933)

- CoT-TTS: introduces a benchmark challenge for context-aware TTS that requires models to perform explicit chain-of-thought reasoning to infer speaking styles from dialogue context before generating speech.
- The framework utilizes a multi-stage data construction pipeline, including automatic filtering and LLM-based annotation, to create a large-scale bilingual dataset for training and evaluation.
- The evaluation methodology combines objective metrics, LLM-based consistency checks, and human subjective assessment to ensure high-quality, contextually coherent, and expressive speech synthesis.

---

[Simulating Public Transit Fare Policies in NYC: An Efficient, Socioeconomic-Aware Framework](http://arxiv.org/abs/2606.21897)

- Geospatial Agent-Based Simulation Framework: introduces a scalable, data-driven simulation pipeline for evaluating public transit fare policies in NYC by integrating Synthetic Population, Agent-Based Mobility Simulation, Multimodal Travel-Time Estimation, Fare-Sensitive Mode Choice, Sampling Framework, and Calibration Module.
- The framework utilizes a landmark-based decomposition for efficient travel-time estimation and a two-stage sampling strategy to reduce computational costs while maintaining aggregate accuracy for city-scale policy analysis.
- Experimental results demonstrate that the framework effectively captures heterogeneous behavioral responses across income groups, providing policymakers with a tool to assess trade-offs between ridership, revenue, and equity.

---

[Learning the ARTS of Search for Automated Discovery](http://arxiv.org/abs/2606.21891)

- ARTS (Agentic Reasoning for Tree Search): introduces an automated scientific discovery framework that replaces heuristic search with an agentic reasoning model capable of failure attribution and diverse hypothesis generation.
- The framework utilizes a modular architecture where a scientist agent inspects search history and memory to select nodes, while a separate executor agent implements and validates code.
- ARTS* incorporates test-time training via LoRA adapters and GRPO to distill search experience into model weights, effectively mitigating context-length limitations and improving performance on complex research tasks.

---

[AgentRiskBOM: A Risk-Scoping Security Bill of Materials for Agentic AI Systems](http://arxiv.org/abs/2606.21877)

- AgentRiskBOM: introduces a structured, machine-readable security bill of materials designed to capture the runtime authority envelope of tool-using AI agents, including JSON Schema, YAML corpus files, risk-scenario library, command-line tool, rule-based scorer, diff detector, control mapper, and rendered reports.
- The framework addresses the capability opacity of LLMs by recording agent identity, model metadata, tool permissions, memory persistence, autonomy levels, and approval gates to enable pre-deployment security review.
- By providing a diffable and audit-oriented artifact, the system allows security teams to detect authority drift and map risk drivers to specific controls across diverse agent archetypes.

---

[Harness-MU: A Safe, Governed, and Effective Harness for Multi-User LLM Agents](http://arxiv.org/abs/2606.21856)

- Harness-MU: introduces a model-agnostic, zero-tuning infrastructure framework that decouples language generation from safety orchestration to enforce multi-principal governance in LLMs.
- The framework utilizes a Gatekeeper for admission control, a Mediator for policy orchestration, isolated parallel Workers for generation, and a ComplianceChecker for deterministic output projection.
- Harness-MU guarantees privacy and authority constraints in multi-user environments by moving governance logic outside the LLM into a deterministic, fail-closed runtime harness.

---

[Measuring What Persists: Conditioning Mechanisms and a Geometric Framework for AI Agent Identity](http://arxiv.org/abs/2606.21843)

- Geometric Framework for AI Agent Identity: introduces a mathematical approach to quantify AI agent identity drift by modeling agent responses as non-geodesic structures in a behavioral metric space.
- The framework utilizes magnitude homology and √JSD distances to detect early-stage identity degradation before qualitative performance drops are visible to human evaluators.
- The proposed Heartbeat monitoring architecture employs a two-tier system to identify identity-vacuum and safety-basin clusters, providing a robust leading indicator for agent drift.

---

[Agent-Assisted Side-Channel Attacks on Non-Prefix KV Cache in RAG](http://arxiv.org/abs/2606.21842)

- SpliceLeak: introduces a micro-architectural side-channel attack that exploits deterministic chunk-aware memory scheduling in RAG-optimized LLM serving engines to reconstruct private user prompts.
- The framework utilizes a two-phase approach consisting of structural fingerprinting to determine prompt length and an LLM-driven extraction agent to reverse-engineer semantic content token-by-token.
- To mitigate these vulnerabilities, the authors propose SpliceDefense, which employs Quantized Chunk Padding (QCP) and Constant-Time Boundary Fusion (CTBF) to eliminate timing side-channels while maintaining high throughput.

---

[AgentDSE: Reasoning-Augmented Architectural Design Space Exploration](http://arxiv.org/abs/2606.21836)

- AgentDSE: introduces a simulator-in-the-loop agentic framework that replaces black-box optimization with an iterative hypothesis-test-refine reasoning process driven by an LLM Agent.
- The framework utilizes a persistent Workspace Interface to maintain Search State, allowing the LLM Agent to perform architectural reasoning based on Task Brief, Action-space Specification, and feedback from the Simulator.
- AgentDSE achieves competitive design quality with significantly fewer evaluations than traditional methods by leveraging semantic architectural knowledge rather than relying on purely numerical sampling.

---

[AgentCAT: Simulating Computerized Adaptive Testing via Multi-Agent Large Language Models](http://arxiv.org/abs/2606.21832)

- AgentCAT: introduces a multi-agent simulation framework that reconstructs the Computerized Adaptive Testing process into a dynamic interactive closed-loop using Examinee Agent, Selection Agent, and Supervisor Module.
- The framework utilizes a Bucketing Strategy for efficient question retrieval and an Online Robust Gradient Descent mechanism to ensure stable ability estimation despite LLM-based generative uncertainty.
- By integrating cognitive simulation and pedagogical constraints, AgentCAT enables high-fidelity benchmarking and robust ability assessment in scenarios characterized by extreme data sparsity.

---

[Generating Public Health Responses using Survey-Augmented Large Language Models](http://arxiv.org/abs/2606.21820)

- Survey-Augmented Large Language Models: introduces a methodology for generating synthetic survey responses by leveraging cluster-informed prompting to improve alignment with longitudinal epidemiological data.
- The framework utilizes Data Preparation, Clustering Analysis, Metadata Formatting, LLM Generation, and Evaluation Metrics to create synthetic digital twins that replicate population-level behavioral patterns.
- The study evaluates multiple LLMs across longitudinal survey waves, finding that while models effectively capture individual variable distributions, they struggle to preserve complex pairwise correlations and remain distinguishable from real human responses.

---

[RAPID: A Reproducible Multi-Agent Pipeline for Interpretable Disaster Damage Assessment from Satellite and Street-View Imagery](http://arxiv.org/abs/2606.21819)

- RAPID: introduces a reproducible multi-agent framework that coordinates specialized agents for disaster perception, image restoration, damage recognition, and reasoning to provide interpretable, zero-shot disaster assessments from heterogeneous satellite and street-view imagery.
- The framework utilizes an auditable pipeline where each agent—Disaster Perception Agent (DPA), Image Restoration Agent (IRA), Damage Recognition Agent (DRA), and Disaster Reasoning Agent (DReA)—contributes to an end-to-end workflow that transforms raw geospatial observations into structured, evidence-grounded disaster intelligence.
- RAPID shifts disaster assessment from traditional pattern recognition toward reasoning-based autonomous understanding by maintaining an explicit audit trail of intermediate states, enabling transparent evaluation and error tracing in time-sensitive disaster scenarios.

---

[Steer, Don’t Solve: Training Small Critic Models for Large Code Agents](http://arxiv.org/abs/2606.21811)

- Critic-Guided Code Agent Framework: introduces a method to improve LLM code agent reasoning by interleaving a small, trained critic model that provides high-level, trajectory-aware feedback to a frozen agent.
- The framework utilizes a Student Critic Model distilled from a Teacher Critic Model to offer strategic guidance, error detection, and budget-aware instructions without requiring agent retraining.
- Empirical results on SWE-bench Verified demonstrate that this approach significantly improves resolve rates and cost-efficiency by separating strategy-level reasoning from low-level implementation.

---

#### 19th June 2026

[Is Agent Code Less Maintainable Than Human Code?](http://arxiv.org/abs/2606.21804)

- CodeThread: introduces a framework for constructing controlled two-step pull-request experiments to evaluate the maintainability of code authored by LLMs compared to human-authored code.
- The framework utilizes an Implementation Task and a Follow-On Issue to measure how intermediate code authorship affects downstream task resolution rates, revealing that agent-authored code often imposes a maintainability burden.
- Analysis indicates that traditional static maintainability metrics are insufficient, with the primary predictors of downstream failure being subtle behavioral drift in agent code, increased downstream code-size changes, and task difficulty.

---

[KineticSim: A Lightweight, High-Performance Execution Engine for Real-Time Market Simulators](http://arxiv.org/abs/2606.21784)

- KineticSim: introduces a persistent, state-carrying execution engine that caches limit-order books in GPU shared memory to eliminate global memory bottlenecks and kernel launch overheads in multi-agent financial simulations.
- The architecture maps each independent market to a dedicated CUDA thread block, utilizing cooperative parallel scans and reductions to achieve high-throughput, low-latency execution.
- By maintaining state residency on-chip and employing a stateless, counter-based PRNG, the framework achieves significant speedups over traditional CPU and vectorized GPU baselines while maintaining exact semantic equivalence.

---

[CALVERT: Augmenting Agents with Calibrated Verifier Telemetry Improves Action and Learning in Knowledge-Intensive Tasks](http://arxiv.org/abs/2606.21777)

- CALVERT: introduces a framework that augments LLM agents with calibrated verifier telemetry, including a calibrated self-confidence score and a grounding verifier score, to improve decision-making in knowledge-intensive tasks.
- The framework enables agents to dynamically choose between commit, retrieve, refine, and decompose actions by conditioning on orthogonal uncertainty signals rather than relying on fixed schedules or parametric knowledge.
- Experimental results demonstrate that CALVERT improves performance across both training-free prompt-based settings and training-based reinforcement learning approaches by reducing redundant retrieval and increasing accuracy on complex multi-hop questions.

---

[Beyond the Next Step: Variable-Length Latent World Models for Long-Horizon Planning](http://arxiv.org/abs/2606.21775)

- VLWM (Variable-length Latent World Models): introduces a framework that learns to predict future latent states conditioned on variable-length action sequences to mitigate compounding errors in long-horizon planning.
- The architecture replaces fixed-step conditioning with an action-as-token formulation, allowing the model to interleave state and action tokens within a single sequence for flexible temporal dynamics.
- A curriculum training strategy progressively expands the prediction horizon, enabling the model to build long-range semantic consistency upon well-learned short-horizon dynamics.

---

[Training the Orchestrator: A Supervised Approach to End-to-End PDDL Planning with LLM Agents](http://arxiv.org/abs/2606.21740)

- HALO (Hybrid Agent Learned Orchestrator): introduces a supervised approach to end-to-end PDDL planning by replacing frontier LLM orchestrators with a small, locally-served, QLoRA-tuned policy trained on verifier-certified trajectories.
- The framework utilizes a hybrid architecture combining a hardcoded rule layer for trivial decisions and a learned policy for ambiguous agent selection, significantly reducing orchestration costs and LLM calls.
- By leveraging an external verifier as a data-acceptance filter during training, the orchestrator learns domain-general planning patterns that match or exceed the performance of larger prompted models.

---

[Safe to Check, Unsafe to Use: Relinking at the Compression Boundary of LLM Agents](http://arxiv.org/abs/2606.21732)

- Relink: introduces a vulnerability where prompt compressors act as confused deputies by synthesizing malicious instructions from distributed, locally benign fragments.
- The paper identifies that attention mechanisms, pre-training priors, and post-training helpfulness preferences enable compressors to autonomously assemble separated fragments into backend-actionable instructions.
- The authors propose KBRA, a compression-boundary defense that enforces consistency between pre-compression source support and post-compression instructions to mitigate adversarial relinking.

---

[PrivacyAlign: Contextual Privacy Alignment for LLM Agents](http://arxiv.org/abs/2606.21710)

- PrivacyAlign: introduces a benchmark and training dataset of privacy-sensitive agent scenarios paired with human preference annotations to ground LLM alignment in human privacy norms.
- The framework utilizes an annotation-conditioned reward model that incorporates human-provided rationales and labels at inference time to guide LLM agents in making context-aware disclosure decisions.
- Experimental results demonstrate that training small open-weight agents with this reward mechanism significantly improves their ability to protect sensitive information while maintaining task helpfulness compared to baseline methods.

---

#### 18th June 2026

[Sovereign Execution Brokers: Enforcing Certificate-Bound Authority in Agentic Control Planes](http://arxiv.org/abs/2606.20520)

- SEB (Sovereign Execution Broker): introduces a runtime enforcement boundary that transforms certified agent proposals into short-lived, revocable, and auditable infrastructure mutation capabilities.
- The framework separates proposal, admission, and execution to ensure that autonomous agents never hold standing credentials, mitigating risks from non-deterministic reasoning processes.
- SEB validates SAB certificates against live infrastructure state, revocation epochs, and replay-prevention nonces before minting transient, scoped execution identities for target platforms.

---

[ORAgentBench: Can LLM Agents Solve Challenging Operations Research Tasks End to End?](http://arxiv.org/abs/2606.19787)

- ORAgentBench: introduces an execution-grounded benchmark for evaluating autonomous LLM agents on challenging end-to-end operations research tasks, utilizing Agent, OR Env, Task Data, Model, Validator, Harbor, Docker, and Skills.
- The benchmark evaluates the complete workflow from operational artifacts to validated decision artifacts, requiring agents to implicitly co-design effective problem representations and efficient computational procedures.
- Experimental results across fourteen frontier model-agent configurations demonstrate that current LLMs struggle to reliably complete realistic OR tasks, with failures primarily driven by strategic modeling weaknesses rather than raw solver limitations.

---

[Dual-Agent Framework for Cross-Model Verified Translation of Natural-Language Protocols into Robotic Laboratory Platform](http://arxiv.org/abs/2606.20120)

- Dual-Agent Framework: introduces an agent-based protocol translation framework that converts natural-language biological protocols into executable control commands for robotic laboratory platforms using a hybrid architecture.
- The framework integrates a Parser Agent for semantic structuring, a deterministic rule-based mapping engine for physical constraint enforcement, and a heterogeneous LLM-based Validation Agent for cross-model verification.
- This architecture mitigates LLM hallucinations and confirmation bias by separating generation and verification roles, enabling reliable autonomous execution of complex microplate-based experiments.

---


[Directors’ duties in the age of agentic Artificial Intelligence: How can boards navigate corporate purpose and the stakeholder interests of employees around AI adoption?](http://arxiv.org/abs/2606.20453)

- Corporate Governance and AI Adoption Framework: introduces a legal and strategic analysis of how boards navigate the tension between shareholder primacy and employee stakeholder interests during AI integration.
- The framework evaluates four doctrinal models of corporate purpose—shareholder primacy, enlightened shareholder value, stakeholder-friendly, and stakeholder value—to determine their efficacy in protecting employees from AI-driven displacement.
- It advocates for a "law in context" approach, urging boards to move beyond minimum legal compliance by integrating ethical considerations, transparency, and reskilling initiatives into their AI adoption strategies.

---

[Blame is easier than praise: Measuring off-ball defensive performance in football](http://arxiv.org/abs/2606.19931)

- Defensive Performance Attribution Framework: introduces a methodology for quantifying off-ball defensive performance by attributing pass-related threat changes to individual players based on spatial proximity and tactical role-conditioned baselines.
- The framework utilizes Feature Setup (calculates pass value and expected receiver), Involvement Model (determines defender involvement via spatial pressure areas), Responsibility Model (calculates expected involvement based on tactical roles), and Aggregate Normalize &amp; Validate (computes final performance KPIs) to evaluate defensive positioning.
- This approach provides a robust, cross-competition evaluation of defensive positioning by decomposing passing events into individual contributions and faults, outperforming traditional action-based metrics.

---


[Prompt, Plan, Extract: Zero-Shot Agentic LLMs Workflows for Lung Pathology Extraction from Clinical Narratives](http://arxiv.org/abs/2606.19852)

- Introduces a zero-shot, graph-structured agentic workflow that utilizes LLMs to perform end-to-end clinical abstraction from raw pathology reports into structured CAP-aligned data.
- The architecture coordinates four specialized nodes—Mapper, Planner, Executor, and Compiler—to isolate clinical concepts and minimize cascading errors common in traditional multi-stage pipelines.
- This approach enables high-fidelity clinical extraction without requiring manual annotation or training, offering a scalable, privacy-preserving alternative for clinical registries.

---



[Execution-State Capsules: Graph-Bound Execution-State Checkpoint and Restore for Low-Latency, Small-Batch, On-Device Physical-AI Serving](http://arxiv.org/abs/2606.20537)

- FlashRT: introduces a latency-first runtime substrate and execution-state capsules to enable efficient checkpointing and restoration of LLM and physical-AI model states.
- The framework utilizes graph-bound execution state stored in contiguous static buffers to replace compute-heavy prefix recomputation with bandwidth-efficient state restoration.
- FlashRT supports low-latency serving verbs including snapshot, restore, fork, and rollback, providing a unified mechanism for LLM warm starts, robot episode resets, and planner-actor hand-offs.

---

[LedgerAgent: Structured State for Policy-Adherent Tool-Calling Agents](http://arxiv.org/abs/2606.20529)

- LedgerAgent: introduces an inference-time method that maintains an explicit, schema-anchored ledger of observed task states to ensure policy-adherent tool use by LLMs.
- The framework utilizes a deterministic policy gate to evaluate proposed environment-changing tool calls against the ledger state before execution, preventing policy violations.
- By rendering the structured ledger into the prompt, the method improves the reliability of LLMs in complex, multi-turn customer service tasks without requiring model weight updates.

---

[Multi-Granular Attention-Driven Reinforcement Learning Framework for Web Intelligent Enhancement Systems](http://arxiv.org/abs/2606.19690)

- MGAR-WIES: introduces a multi-layered framework that integrates semantic graph modeling, attention mechanisms, and multi-agent reinforcement learning to optimize personalized web services.
- The framework utilizes graph-based semantic knowledge modeling with attention to capture complex relationships among web entities, providing enriched state inputs for reinforcement learning agents.
- A continuous online adaptation module enables real-time updates to embeddings and learning policies, ensuring sustained adaptability and performance in dynamic web environments.

---


[Spatial Tool-Use Elicits Reasoning for Spatial Intelligence](http://arxiv.org/abs/2606.20515)

- S-Agent: introduces a spatial tool-use agentic paradigm that formulates spatial reasoning as an active process of spatio-temporal evidence accumulation using an Agentic Planner, 2D Perception Tools, Multi-view 3D Tools, Specialized Spatial Experts, Scene Memory, and Agent Memory.
- The framework utilizes a hierarchical approach to transform raw visual observations into explicit 3D knowledge, delegating perception and geometric computation to specialized tools while maintaining stateful memory across reasoning steps.
- S-Agent consistently enhances zero-shot LLM performance on spatial benchmarks and enables the distillation of compact, high-performing spatial agents through supervised fine-tuning on generated reasoning trajectories.

---

[Probe-and-Refine Tuning of Repository Guidance for Coding Agents](http://arxiv.org/abs/2606.20512)

- Probe-and-Refine Tuning of Repository Guidance for Coding Agents: introduces a lightweight, iterative procedure that refines static repository knowledge into specialized operational guidance using synthetic bug-fix probes and single-shot LLM calls.
- The framework improves coding agent performance by increasing evaluation coverage through a structured, reproduce-first diagnostic workflow, while maintaining constant per-patch precision.
- Experimental results demonstrate that refined guidance enables coding agents to utilize larger step budgets productively, though the procedure requires a model capable of generating sufficiently diagnostic output to sustain the tuning loop.

---

[Efficient and Sound Probabilistic Verification for AI Agents](http://arxiv.org/abs/2606.20510)

- Probabilistic Verification for AI Agents: introduces a framework for sound runtime monitoring of LLM agents by modeling multi-step execution trajectories as Datalog derivation graphs and computing risk bounds via distributionally robust optimization.
- The approach replaces fragile threshold-based binarization with a semidefinite programming relaxation that tracks second-order moments to provide conservative upper bounds on policy violation probabilities.
- By avoiding unsafe independence assumptions, the framework effectively balances security and utility in ambiguous agent environments while maintaining low computational overhead for real-time enforcement.

---

[Contagion Networks: Evaluator Bias Propagation in Multi-Agent LLM Systems](http://arxiv.org/abs/2606.20493)

- Contagion Networks: introduces a formal framework for measuring how evaluator biases propagate across interacting LLM agents using a cross-agent contagion matrix.
- The framework utilizes Test-Time Reinforcement Learning (TTRL) to quantify bias shifts between agents and identifies three propagation regimes governed by the spectral radius of the contagion matrix.
- Experimental results demonstrate that homogeneous-model agent systems operate in a suppression regime, and that increasing evaluator committee size effectively mitigates bias propagation.

---

[Beyond Global Replanning: Hierarchical Recovery for Cross-Device Agent Systems](http://arxiv.org/abs/2606.20487)

- H-RePlan: introduces a hierarchical replanning framework that separates device-local strategy recovery from orchestrator-level cross-device replanning to improve robustness in multi-device agent systems.
- The framework utilizes an Orchestrator for global task management, a Strategy Planner for local execution, and specialized API-, CLI-, and GUI-agents to navigate heterogeneous environments.
- H-RePlan employs a Cross-Layer Failure Event (CLFE) abstraction to provide actionable, scope-aware failure information, enabling efficient recovery without overloading the global planning context.

---

[GroundControl: Anticipating Navigation Failures in Vision-Language Agents via Trajectory-Consistent Uncertainty Estimates](http://arxiv.org/abs/2606.20479)

- GroundControl: introduces a trajectory-consistent uncertainty estimator that models distance-to-goal dynamics using a Constant-Velocity Kalman Filter, Normalized Innovation Statistics, Posterior Covariance Growth, Trajectory Descriptors, and the Selective Risk–Coverage Navigation (SRCN) Protocol.
- The framework identifies navigation failures by detecting statistically significant deviations from nominal goal-directed motion rather than relying on instantaneous action entropy from LLMs.
- Experimental results demonstrate that GroundControl achieves near-oracle failure ordering across multiple LLM backbones, significantly outperforming existing uncertainty baselines in complex navigation environments.

---

[Marginal Advantage Accumulation for Memory-Driven Agent Self-Evolution](http://arxiv.org/abs/2606.20475)

- MAA (Marginal Advantage Accumulation): introduces a post-processing architecture for offline trace distillation that enables cross-batch evidence accumulation by grounding optimization suggestions into addressable atomic operations.
- The framework utilizes semantic identity merging to track operations across batches and employs differential construction with per-op EMA to distinguish stably effective operations from accidental hits.
- By replacing environment rollouts with LLM proxy scoring and accumulating signed evidence, MAA improves agent self-evolution efficiency and performance while reducing optimization-phase token consumption.

---

[UltraQuant: 4-bit KV Caching for Context-Heavy Agents](http://arxiv.org/abs/2606.20474)

- UltraQuant: introduces a 4-bit KV cache compression method for context-heavy agents that leverages Walsh-Hadamard Rotation, FP4 E2M1 Grid, UE8M0 Group Scales, CDNA4 Scaled-MFMA, and KV Cache to improve serving throughput and cache residency.
- The framework replaces software-based dequantization with hardware-native operations by folding dequantization into the CDNA4 matrix core instructions.
- UltraQuant achieves significant improvements in time-to-first-token and output throughput on long-context, multi-turn agentic workloads compared to FP8 baselines.

---

[Analyzing Defensive Misdirection Against Model-Guided Automated Attacks on Agentic AI Systems](http://arxiv.org/abs/2606.20470)

- CMPE (Contextual Misdirection via Progressive Engagement): introduces a detect-and-misdirect defense strategy that degrades the feedback quality of automated attacks by replacing predictable refusals with semantically plausible, non-operational responses.
- The framework models the interaction between a victim system and an automated attacker as a probabilistic game, where misdirection-induced false-positives in the attacker's judge effectively bound the attacker success rate.
- Empirical evaluations demonstrate that CMPE significantly reduces verified attack success and forces premature termination in model-guided attack frameworks like PAIR and GPTFuzz.

---

[Agentic Symbolic Search: Characterizing PDEs Beyond Hand-crafted Expressions, Meshes, and Neural Networks](http://arxiv.org/abs/2606.20467)

- ASYS (Agentic Symbolic Search): introduces a prior-guided framework that employs a coding agent to translate PDE theory and physical constraints into interpretable, differentiable symbolic programs.
- The framework utilizes an outer loop for structural hypothesis generation via an evolutionary ensemble of agents and an inner loop for parameter fitting using quasi-Newton optimization.
- ASYS automates the discovery of explicit mathematical structures for PDE solutions, providing an interpretable alternative to mesh-based numerical solvers and black-box LLM or neural network approximations.

---

[NRT-Bench: Benchmarking Multi-Turn Red-Teaming of LLM Operator Agents in Safety-Critical Control Rooms](http://arxiv.org/abs/2606.20408)

- NRT-Bench: introduces a multi-turn red-teaming benchmark for LLM operator agents in safety-critical control rooms, utilizing an Adaptive Red-Team Attacker, Guardrails, Provenance Check, Action Classifier, Human Approval, Five-Role Operator Team, Plant Simulator, and Critical Safety Functions.
- The framework evaluates LLM agent robustness by measuring their ability to manage a simulated nuclear power plant under sustained adversarial pressure, defining harm as an objective simulator-derived Critical Safety Function loss.
- Experimental results across four frontier LLMs demonstrate that vulnerabilities are largely disjoint across models, and the effectiveness of defense layers is highly model-conditional.

---

[Agentic AutoResearch for Space Autonomy: An Auditable, LLM-Driven Research Agent for Aerospace Control Problems](http://arxiv.org/abs/2606.20394)

- AutoResearch: introduces an agentic framework where an LLM autonomously drives machine-learning experimentation for aerospace control, paired with an in-loop credibility layer to certify results.
- The framework utilizes a structured family contract to enable an LLM agent to propose, execute, and analyze experiments while maintaining reproducibility through a rigorous credibility audit.
- The credibility layer validates improvements by measuring per-problem seed noise, performing reseeded verification, and conducting leave-one-out pruning to isolate the impact of individual agent edits.

---

[DataMagic: Transforming Tabular Data into Data Insight Video](http://arxiv.org/abs/2606.20388)

- DataMagic: introduces an end-to-end system that transforms raw tabular data and natural language queries into narrative data-insight videos using a Generate-then-Orchestrate multi-agent architecture.
- The system utilizes DVSpec (declarative data video specification) to decouple logical descriptions from rendering, ensuring data fidelity through semantic references and narration-index triggering.
- DataMagic supports interactive exploration through parametric editing and provenance-based data Q&A, allowing users to refine generated videos without full regeneration.

---

[CRAX: Fast Safe Reinforcement Learning Benchmarking](http://arxiv.org/abs/2606.20376)

- CRAX (Constrained RL Accelerated with JAX): introduces a hardware-accelerated SafeRL benchmark leveraging MJX, JAX, Environment Suites, Agents, Constraint Formulations, and Baseline Algorithms to enable high-throughput simulation and rigorous safety evaluation.
- The framework utilizes parallel computing to achieve orders-of-magnitude faster simulation speeds compared to traditional CPU-based benchmarks, facilitating large-scale experimentation.
- CRAX provides a comprehensive suite of tasks with difficulty progression and explicit cost signals, allowing for the systematic study of performance-safety trade-offs across various agent morphologies.

---

[AutoPass: Evidence-Guided LLM Agents for Compiler Performance Tuning](http://arxiv.org/abs/2606.20373)

- AutoPass: introduces a multi-agent framework that integrates LLMs into the compiler tuning loop by utilizing compiler-internal signals and runtime feedback to guide optimization decisions.
- The framework employs a Score Agent to identify high-impact kernels, an Analysis Agent to interpret IR, a Reasoning Agent to generate pass pipelines, and an Evaluation Agent to validate performance improvements.
- AutoPass operates in an inference-only, training-free setting, achieving significant speedups over standard -O3 pipelines by iteratively refining optimization configurations based on grounded evidence.

---

[An Infrastructure-less, Control-Independent Solution to Relative Localisation of a Team of Mobile Robots using Ranging Measurements](http://arxiv.org/abs/2606.20365)

- MHDCL (Multi-Hypothesis Bayesian-based Decentralised Cooperative Localisation): introduces a decentralised cooperative localisation algorithm that maintains multiple pose hypotheses to ensure robustness in low-observability conditions without requiring controlled robot motion.
- The framework utilizes a Particle Filter combined with a Gaussian-von Mises Mixture Model (GVMMM) to represent relative agent poses and handle non-uniqueness in sparse measurement scenarios.
- By sharing motion vectors and cluster descriptors, the algorithm enables information propagation across partially connected networks, allowing agents to estimate the poses of the entire fleet using only local odometry and inter-agent UWB ranging.

---

[Automating SKILL.md Generation for Computer-Using Agents via Interaction Trajectory Mining](http://arxiv.org/abs/2606.20363)

- Automated SKILL.md Generation framework: introduces a three-stage pipeline that segments GUI trajectories, clusters segments into candidate skills, and trains a skill-aware policy using Boundary Detector, Segment Representation, Wasserstein Clustering, Supervised-Contrastive Encoder, GRPO Policy, and Trajectory Reward Model.
- The pipeline utilizes a Boundary Detector for trajectory segmentation, followed by Wasserstein Clustering and a Supervised-Contrastive Encoder to construct an inspectable skill library.
- The research evaluates the mined skill library through GRPO Policy training and Trajectory Reward Model feedback, finding that while the skills are readable, they do not consistently outperform trivial frequency baselines in cross-domain transfer.

---

[SoftSkill: Behavioral Compression for Contextual Adaptation](http://arxiv.org/abs/2606.20333)

- SoftSkill: introduces a method that compresses natural-language skill documents into compact, trainable continuous prefixes for frozen LLMs to enable efficient contextual adaptation.
- The framework initializes virtual token embeddings from skill text and optimizes only a soft delta via next-token prediction on successful trajectories or ground-truth answers.
- SoftSkill utilizes a validation gate to select the optimal prefix checkpoint, ensuring robust performance while significantly reducing the context length required for task execution.

---

[A Model-Driven Approach for Developing Families of Reinforcement Learning Environments](http://arxiv.org/abs/2606.20324)

- Model-Driven Approach for Developing Families of Reinforcement Learning Environments: introduces a model-driven engineering framework that automates the generation of diverse RL environment families using a hybrid genetic algorithm, Domain Metamodel, Mutation Operators, Constraint Service, Simulated Annealing, and Code Generation Templates.
- The framework utilizes model transformations to mutate an initial environment into a family of variants, ensuring structural validity through a constraint service while optimizing for diversity and complexity.
- This approach facilitates curriculum learning by generating progressively challenging environment sequences, thereby improving RL agent generalization and training performance with minimal manual intervention.

---

[AgenticDB: Agentic Performance Reconfiguration for Database Workloads](http://arxiv.org/abs/2606.20318)

- AgenticDB: introduces an agentic framework for database workload reconfiguration that utilizes an LLM DBA Planner to iteratively optimize DBMS and OS configurations through a context-grounded harness.
- The framework integrates an Initializer, Memory Book, LLM DBA Planner, Validator, Executor, Recovery, and Auditor to perform bottleneck diagnosis, safe action application, and iterative self-refinement.
- AgenticDB improves database performance by maintaining a safe cross-layer action space and using runtime feedback to guide reconfiguration, effectively reducing uninformative workload replays.

---

[PsyScore: A Psychometrically-Aware Framework for Trait-Adaptive Essay Scoring and ZPD-Scaffolded Feedback](http://arxiv.org/abs/2606.20287)

- PsyScore: introduces a psychometrically-aware framework that integrates diagnostic assessment with instructional scaffolding through a shared latent ability representation.
- The framework utilizes a Trait-Adaptive Neural GPCM Scorer to estimate student ability and a ZPD-Conditional Feedback Generator to provide personalized, ability-aware instructional support.
- PsyScore employs a multi-agent consensus mechanism and a multi-perspective evaluation strategy to ensure feedback quality, pedagogical alignment, and diagnostic precision.

---

[Phoenix: Safe GitHub Issue Resolution via Multi-Agent LLMs](http://arxiv.org/abs/2606.20243)

- Phoenix: introduces a multi-agent LLM system for safe, end-to-end GitHub issue resolution that utilizes an Orchestrator, Planner Agent, Reproducer Agent, Coder Agent, Tester Agent, Failure Analyst Agent, PR Agent, GitHub Webhook State Machine, Safety Controls, and Memory Module.
- The system employs a six-agent pipeline coordinated by a label-based state machine to ensure correctness-first operation through baseline-aware test evaluation.
- Seven layered safety mechanisms, including path-traversal prevention and content sanitization, are integrated to mitigate production deployment hazards during autonomous code modification.

---

[A Multi-Agent system for Multi-Objective constrained optimization](http://arxiv.org/abs/2606.20236)

- MAMO (Multi-Agent system for Multi-Objective constrained optimization): introduces a hierarchical multi-agent framework that decouples task execution from objective design by using a WA agent to dynamically adjust reward weights for a TE agent.
- The framework includes a TE agent that interacts with the environment to optimize a weighted reward, and a WA agent that observes system performance to refine these weights over longer time scales.
- MAMO enables autonomous adaptation to non-stationary environments by learning the trade-off between cost efficiency and constraint satisfaction through experience rather than manual tuning.

---

[Augmenting Game AI with Deep Reinforcement Learning](http://arxiv.org/abs/2606.20210)

- RL-augmented Game AI framework: introduces a methodology for integrating Reinforcement Learning (RL) agents into AAA game production pipelines by augmenting existing hand-coded systems like Behavior Trees (BT) and Finite State Machines (FSM).
- The framework addresses production-specific constraints including short training times, modular integration, runtime inference efficiency, and the necessity for authentic, human-like NPC behavior.
- Experimental results demonstrate that using computationally efficient representations like occupancy maps and targeted RL algorithms (SAC, PPO) enables production-ready deployment within complex AAA environments like EA SPORTS FC 25 and Battlefield 6.

---

[FlowMaps: Modeling Long-Term Multimodal Object Dynamics with Flow Matching](http://arxiv.org/abs/2606.20209)

- FlowMaps: introduces a latent flow matching framework for modeling continuous, multimodal spatio-temporal distributions of dynamic objects in household environments.
- The architecture utilizes a VAE to encode object geometry and semantics, while a CDiT-based flow model predicts future object locations conditioned on scene context and human-like routines.
- By learning implicit dependencies from procedural data, the model enables robots to perform proactive object navigation in dynamic scenes without requiring explicit scene graph structures.

---

[MedRLM: Recursive Multimodal Health Intelligence for Long-Context Clinical Reasoning, Sensor-Guided Screening, Evidence-Grounded Decision Support, and Community-to-Tertiary Referral Optimization](http://arxiv.org/abs/2606.20164)

- MedRLM: introduces a recursive multimodal framework that treats patient data as an external environment to perform long-context clinical reasoning and decision support.
- The framework coordinates specialized agents for clinical text, EHR, imaging, and sensor streams, utilizing a Clinical Evidence Graph Memory to ensure auditable and evidence-grounded outputs.
- MedRLM incorporates sensor-guided recursive triggering and uncertainty-gated refinement to optimize community-to-tertiary referral pathways while minimizing hallucination and context loss.

---

[ARTEMIS: Agent-guided Reliability-aware Temporal Mask Evolution for Imperfectly Supervised Video Polyp Segmentation](http://arxiv.org/abs/2606.20161)

- ARTEMIS: introduces a unified framework for imperfectly supervised video polyp segmentation by completing sparse or missing annotations into temporally consistent dense pseudo masks using agent-guided reliability-aware temporal mask evolution.
- The framework utilizes a debate-and-judge vision-language agent to select reliable temporal anchors, which are then propagated bidirectionally via SAM2 to refine unreliable frames.
- ARTEMIS further enhances temporal identity consistency and suppresses noise through a Reference Prototype Transport Module (RPTM) and a reliability-aware robust loss function.

---

[N-Version Programming with Coding Agents](http://arxiv.org/abs/2606.20158)

- NVP with Coding Agents: introduces a systematic experimental framework to evaluate the reliability of agent-generated software by replicating the classical Knight–Leveson experiment using modern LLMs.
- The study demonstrates that while agent-generated versions exhibit significant correlated failure modes, N-version units still provide measurable reliability improvements through majority voting.
- The research identifies that common-mode failures are primarily driven by ambiguous or challenging parts of the specification, rather than purely random implementation errors.

---

[RACL: Reasoning-Agent Control Layers for Continuous Metaheuristic Learning](http://arxiv.org/abs/2606.20142)

- RACL: introduces a reasoning-agent control layer that governs an existing metaheuristic optimizer by observing operational memory, formulating bounded hypotheses, and consolidating control policies.
- The framework utilizes a memory-guided reasoning cycle to perform continuous algorithmic improvement without modifying underlying business constraints.
- Experimental results demonstrate that the agentic cycle improves or ties performance against non-reasoning baselines while providing business-readable explanations for its interventions.

---

[ScaffoldAgent: Utility-Guided Dynamic Outline Optimization for Open-Ended Deep Research](http://arxiv.org/abs/2606.20122)

- ScaffoldAgent: introduces a utility-guided dynamic outline optimization framework for open-ended deep research that models report generation as a structured decision process using Outline Agent, Search Agent, and Reporter Agent.
- The framework maintains an evolving outline tree as a structural scaffold, utilizing Expansion, Contraction, and Revision operations to manage information accumulation and structural coherence.
- A utility-guided feedback mechanism provides inference-time control signals by integrating retrieval gain, structural soundness, and trial generation quality to guide node selection and operation scheduling.

---

[Multi-Head Attention-Based Feature Extractor Integration with Soft Actor-Critic for Porosity Prediction and Process Parameter Optimization in Additive Manufacturing](http://arxiv.org/abs/2606.20087)

- ABFE-SAC (Multi-Head Attention-Based Feature Extractor Integration with Soft Actor-Critic): introduces a reinforcement learning architecture that integrates a multi-head attention mechanism with the Soft Actor-Critic algorithm to optimize additive manufacturing process parameters.
- The framework utilizes an attention-based feature extractor to enhance the agent's ability to capture subtle variations in input features, facilitating more effective exploration and exploitation in continuous action spaces.
- By employing a dual critic strategy and entropy maximization, the model achieves faster convergence and higher reward values in porosity prediction tasks compared to standard reinforcement learning methods.

---

[Autonomous Event-Driven Multi-Agent Orchestration for Enterprise AI at Scale](http://arxiv.org/abs/2606.20058)

- Autonomous Event-Driven Multi-Agent Orchestration for Enterprise AI at Scale: introduces a Task Manager that enables continuous event-driven operation by integrating with ReAct and DAG Plan &amp; Execute architectures to perform priority inference, related-event merging, and preemption.
- The paper evaluates orchestration performance across three organizational scales, demonstrating that agent discovery noise at enterprise scale is the primary bottleneck for both ReAct and DAG Plan &amp; Execute architectures.
- The proposed Task Manager reduces high-priority queue latency by 14–75% and improves related-event correctness by over 20 percentage points at enterprise scale by managing asynchronous event streams and deterministic stoppage points.

---

[PACMS: Submodular Context Selection as a Pluggable Engine for LLM Agents](http://arxiv.org/abs/2606.20047)

- PACMS: introduces a budget-aware submodular selector that maximizes query-relevant coverage over pooled candidate context while minimizing redundancy for LLM agents.
- The framework operates as a pluggable engine within the OpenClaw agent framework, treating memory entries, conversation turns, and tool outputs as a unified candidate pool for prompt assembly.
- By utilizing a facility-location objective with CELF lazy-greedy optimization, PACMS improves end-to-end QA accuracy compared to traditional recency-based or pairwise-diversification methods.

---

[See-and-Reach: Precise Vision-Language Navigation for UAVs within the Field of View](http://arxiv.org/abs/2606.20045)

- 3DG-VLN (3D Direction-Guided Vision-Language Navigation): introduces a vision-language waypoint prediction framework that utilizes Qwen2.5-VL, LoRA, DeepSeek-V3.2, an Open-Set Detector, an Online 3D Direction Updating Module, and a Waypoint Predictor to achieve precise target reaching for UAVs.
- The framework leverages high-resolution dual-view observations and dynamic 3D direction cues to maintain spatial alignment and reduce drift during the see-and-reach navigation stage.
- 3DG-VLN establishes a new benchmark, UAV-VLN-FOV, which mandates a stringent 10-meter success radius to evaluate terminal reaching precision in aerial embodied agents.

---

[When Lower Privileges Suffice: Investigating Over-Privileged Tool Selection in LLM Agents](http://arxiv.org/abs/2606.20023)

- TOOLPRIVBENCH: introduces a simulation-based benchmark to evaluate whether LLM agents exhibit over-privileged tool selection by choosing higher-privilege tools when sufficient lower-privilege alternatives exist.
- The research identifies that mainstream LLM agents frequently escalate to higher-privilege tools due to transient environmental failures, a behavior amplified by lack of privilege-aware training.
- The authors propose a privilege-aware post-training defense combining SFT and GRPO that significantly reduces unnecessary high-privilege tool usage while maintaining general model capabilities.

---

[Hierarchical Control in Multi-Agent Games: LLM-based Planning and RL Execution](http://arxiv.org/abs/2606.20014)

- LLM+RL: introduces a hierarchical architecture where a pretrained LLM acts as a centralized meta-controller to select among specialized RL skill policies for multi-agent coordination.
- The system leverages state representation asymmetry and temporal abstraction, with the LLM performing strategic reasoning at a slower timescale while RL policies handle reactive execution at high frequency.
- Empirical results demonstrate that the architecture achieves performance equivalent to hand-crafted behavior trees and is perceived as significantly more human-like by players in a competitive 2v2 game environment.

---

[Connect the Dots: Training LLMs for Long-Lifecycle Agents with Cross-Domain Generalization Via Reinforcement Learning](http://arxiv.org/abs/2606.20002)

- CoD (Connect the Dots): introduces a general framework for training LLMs to acquire meta-capabilities for long-lifecycle agents through end-to-end reinforcement learning, utilizing CoD-Deploy, CoD-Train, Solve-task episode, Update-context episode, Trinity-RFT, GRPO-style RL algorithm, Hint, Environment, and Task.
- The framework employs a GRPO-style RL algorithm with fine-grained credit assignment to interleave solve-task and update-context episodes, enabling agents to proactively maintain and refine their environment context.
- Empirical results demonstrate that the CoD meta-capability improves task-solving performance across diverse environments and supports out-of-distribution generalization for long-lifecycle agentic deployment.

---

[Beyond Static Endpoints: Tool Programs as an Interface for Flexible Agentic Web Services](http://arxiv.org/abs/2606.19992)

- TOOLPRO: introduces an executable tool program interface that consolidates multi-step agentic workflows into a single, delegatable object to reduce network round trips and improve efficiency.
- The framework employs a synthesize-project-compile-execute pipeline with constraint-guided construction and effect-aware replay to ensure reliable, retry-safe execution of LLM-generated programs.
- A profile-driven consolidation policy adaptively switches between stepwise calling and program execution to optimize end-to-end latency based on network conditions and workflow complexity.

---

[Reward as An Agent for Embodied World Models](http://arxiv.org/abs/2606.19990)

- RAA (Reward as An Agent): introduces an agentic reward framework that replaces static scalar functions with a structured, multi-stage evaluation system to mitigate reward hacking in embodied world models.
- DynDiff-GRPO: introduces a dynamic-aware stochastic rollout strategy that selectively reallocates exploration to dynamically salient regions while maintaining scene consistency.
- The framework integrates planning, curriculum-based evaluation, voting, and reflection to provide robust, reliable reward signals for RL-based optimization of embodied world models.

---

[ENPIRE: Agentic Robot Policy Self-Improvement in the Real World](http://arxiv.org/abs/2606.19980)

- ENPIRE: introduces a closed-loop harness framework that automates real-world robotic policy improvement through Environment module (EN), Policy Improvement module (PI), Rollout module (R), and Evolution module (E).
- The framework utilizes a decentralized team of coding agents that leverage Tool APIs to construct autonomous research environments and optimize policies across a parallelized Robot Fleet.
- ENPIRE enables autonomous policy self-improvement by integrating Git-based knowledge sharing and resource-utilization metrics to scale physical autoresearch efficiently.

---

[Advancing DialNav through Automatic Embodied Dialog Augmentation](http://arxiv.org/abs/2606.19948)

- RAINbow (RAIN built on wide set): introduces an automatic generation pipeline to create a large-scale dataset for dialog-based navigation, addressing data scarcity in embodied AI.
- The framework utilizes Dual-Strategy Training to align navigation with dynamic dialog-navigation loops and employs a Graph-based Transformer Localization module to improve position inference.
- By combining these components, the approach achieves state-of-the-art performance on DialNav, doubling the success rate compared to previous baselines.

---

[MobileForge: Annotation-Free Adaptation for Mobile GUI Agents with Hierarchical Feedback-Guided Policy Optimization](http://arxiv.org/abs/2606.19930)

- MobileForge: introduces an annotation-free adaptation system for mobile GUI agents that grounds task generation and rollout evaluation in real app interaction using MobileGym and HiFPO.
- MobileGym provides a unified substrate for target-app exploration, curriculum mining, and hierarchical evaluation, while HiFPO transforms multi-attempt feedback and corrective hints into hint-contextualized step-level GRPO updates.
- The system enables LLMs to adapt to new mobile apps without human-written tasks or demonstrations, achieving state-of-the-art performance among open-data mobile GUI agents.

---

[MemGUI-Agent: An End-to-End Long-Horizon Mobile GUI Agent with Proactive Context Management](http://arxiv.org/abs/2606.19926)

- MemGUI-Agent: introduces an end-to-end mobile GUI agent that utilizes ConAct (Context-as-Action) to proactively manage context through Folded Action History, Folded UI State, and Recent Step Record, ensuring compact state representation for long-horizon tasks.
- The framework treats context management as first-class actions emitted by the MLLM backbone, enabling the agent to dynamically decide what information to compress, store, or retain across app transitions.
- The research includes the construction of the MemGUI-3K dataset and the training of MemGUI-8B-SFT, which achieves superior performance on long-horizon mobile GUI benchmarks compared to existing open-data models.

---

[The Tao of Agency: Autotelic AI, Embedded Agency and Dissolution of the Self](http://arxiv.org/abs/2606.19924)

- Autotelic Agent Framework: introduces a formal tuple (π, G, µ, Cα, V, b, M) to model agents that generate and relativize their own goals through embedded interaction.
- The framework utilizes an LLM-based intention generator and an admissibility filter to maintain homeostatic viability while navigating self-referential boundaries.
- It addresses the dissolution of the self by treating the agent-environment boundary as an instrumentally indispensable fiction rather than an ontological absolute.

---

[Deep-Unfolded Coordination](http://arxiv.org/abs/2606.19920)

- Deep Coordinator: introduces a deep-unfolding framework that dynamically adapts ADMM-DDP hyperparameters at solve-time to accelerate convergence in multi-agent robotics tasks.
- The architecture unrolls fixed ADMM-DDP iterations into a neural network, utilizing an unsupervised learning scheme and an Implicit Differentiation Framework to enable end-to-end training.
- Deep Coordinator demonstrates significant speedups over traditional solvers and maintains performance when deployed to systems up to 8x larger than those used during training.

---

[Multi-Agent Transactive Memory](http://arxiv.org/abs/2606.19911)

- MATM: introduces a population-level memory infrastructure where heterogeneous agents contribute and retrieve interaction trajectories to improve task performance and efficiency.
- The framework utilizes a producer-consumer model where agents share procedural knowledge, enabling collective learning and reducing redundant exploration across decentralized agent populations.
- MATM incorporates a cascaded retrieval pipeline with an LTRT reranker that leverages producer-consumer metadata and interaction features to optimize the relevance of retrieved trajectories for consumer agents.

---

[Toward Temporal Realism in City-Scale Crisis Response Simulation using LLM Agents](http://arxiv.org/abs/2606.19904)

- Mechanism-augmented LLM simulator: introduces a dual-channel architecture that decouples temporal event generation from LLM-based decision-making to achieve temporal realism in city-scale crisis response simulations.
- The framework utilizes a Hawkes-gated timing channel to produce self-exciting, heavy-tailed event distributions, while the LLM agent acts as a context-dependent confirm/veto and task-selection module.
- By integrating a crisis-period regime with the Hawkes gate, the simulator successfully reproduces observed human collective participation patterns that standard synchronous LLM simulators fail to capture.

---

[One-to-Two Acting: A Novel Framework for Single-arm Agent Action Expansion to Dual Arms](http://arxiv.org/abs/2606.19897)

- ExS2D (Extending Single-arm agent actions to Dual arms): introduces a hierarchical framework that enables dual-arm manipulation from single-arm supervision by decomposing tasks into structured subtasks, grounding them into executable actions, and coordinating dual-arm execution.
- The framework utilizes VL-SubGen for subtask generation, SA-Map for mask-guided action grounding, and P-DCoord for precedence-aware dual-arm allocation and collision-free motion planning.
- ExS2D achieves efficient dual-arm manipulation without requiring bimanual demonstrations, significantly reducing execution steps while maintaining high success rates in both simulation and real-world experiments.

---

[MetaResearcher: Scaling Deep Research via Self-Reflective Reinforcement Learning in Adversarial Virtual Environments](http://arxiv.org/abs/2606.19893)

- MetaResearcher: introduces a framework that scales deep research agent training by integrating an Evolving Virtual World, Discovery-Oriented Tasks, a Self-Reflective Meta-Reward, and a Heterogeneous Multi-Agent Swarm.
- The framework utilizes a multi-agent architecture comprising specialized Scout-, Filter-, and Synthesizer-agents that learn collaborative research strategies through coordinated reinforcement learning.
- MetaResearcher addresses limitations in static training environments and outcome-only rewards by employing a process-based meta-reward mechanism that incentivizes search efficiency, self-reflection, and tool call diversity.

---

[Matching Markets meet Cumulative Prospect Theory: Towards Optimal and Adversarially Robust Learning](http://arxiv.org/abs/2606.19883)

- CPT-ETGS: introduces a framework for multi-agent matching markets that incorporates Cumulative Prospect Theory (CPT) to model human-centric, risk-sensitive decision-making under uncertainty.
- The framework utilizes CPT-distorted preference estimators and adaptive confidence intervals to achieve sub-linear regret in competitive matching markets, even under adversarial reward corruption.
- The research provides optimal regret guarantees for large markets through adaptive arm elimination and robust multi-layer algorithms that handle both known and unknown corruption budgets.

---

[A Systematic Evaluation of Black-Box Uncertainty Estimation Methods for Large Language Models](http://arxiv.org/abs/2606.19868)

- Black-Box UE: introduces a systematic review and unified evaluation framework for uncertainty estimation methods that rely exclusively on externally observable outputs from LLMs.
- The framework categorizes existing approaches into five distinct types—verbalization-based, sampling-based, explanation-based, multi-agent, and hybrid—to provide a structured landscape of current reliability estimation techniques.
- The study benchmarks 24 representative methods across diverse task formats and model families, identifying that hybrid methods and sampling-aggregated verbalization generally offer the most robust performance for trustworthy LLM deployment.

---

[Large Language Models Do Not Always Need Readable Language](http://arxiv.org/abs/2606.19857)

- BabelTele: introduces a model-centric textual representation paradigm that relaxes human readability constraints to achieve high information density for LLM-to-LLM communication.
- The framework utilizes a Compressor to transform verbose natural language into compact, symbolic, and multilingual encodings that remain recoverable by a Reader.
- BabelTele demonstrates robust cross-model transferability and efficiency in long-context tasks, agent memory, and multi-agent communication without requiring fine-tuning or architectural modifications.

---

[AtomMem: Building Simple and Effective Memory System for LLM Agents via Atomic Facts](http://arxiv.org/abs/2606.19847)

- AtomMem: introduces a long-term memory system for LLMs that transforms raw dialogue into structured atomic facts to enable high-density storage and stable memory evolution.
- The framework utilizes a Fact Executor to generate self-contained semantic units, which are organized into episodic events and temporal profiles to maintain consistent user states.
- AtomMem employs a hierarchical retrieval mechanism that activates an associative memory graph to connect fragmented information, ensuring precise and contextually grounded responses for LLM agents.

---

[JAMER: Project-Level Code Framework Dataset and Benchmark on Professional Game Engines](http://arxiv.org/abs/2606.19830)

- JAMER: introduces a project-level game code framework dataset and benchmark, JamSet and JamBench, utilizing a deterministic verification pipeline on the Godot engine to evaluate LLM performance in game engineering.
- The framework employs a four-level verification pipeline, including file integrity, compilation correctness, runtime stability, and runtime behavior collection, to objectively assess model-generated games without subjective human or LLM judgment.
- Experimental results reveal a significant capability cliff as project scale increases, demonstrating that while Code Agents improve compilation rates, they fail to enhance structural completeness or runtime behavioral quality, highlighting a fundamental engineering paradigm gap.

---

[Heterogeneous LLM Debate Under Adversarial Peers: Honest Gains, Replacement Costs, and Resilience](http://arxiv.org/abs/2606.19826)

- Heterogeneous LLM Debate Framework: introduces a defender-centered measurement approach to evaluate how heterogeneous peers influence the revision behavior of honest LLMs in multi-agent debate systems.
- The framework utilizes a detection-generation decomposition to distinguish between corrective and harmful revisions, effectively quantifying the impact of adversarial peers on panel performance.
- Experimental results demonstrate that while an honest heterogeneous peer can act as a defense against contamination, an adversarial peer can significantly degrade performance by inducing harmful revisions.

---

[TelcoAgent: A Scalable 5G Multi-KPM Forecasting With 3GPP-Grounded Explainability](http://arxiv.org/abs/2606.19821)

- TelcoAgent: introduces a foundation model-based framework for scalable, zero-shot multi-KPM forecasting in 5G networks by integrating a TSFM-based prediction pipeline with a 3GPP-grounded knowledge graph.
- The framework utilizes an automated three-agent pipeline to construct a 3GPP knowledge graph, which provides the necessary domain-grounded context for a ReAct-based explanation pipeline to deliver actionable diagnostics.
- By combining TSFM-based predictive modeling with causal reasoning, TelcoAgent achieves high forecasting accuracy and verifiable, evidence-backed recommendations without requiring site-specific training.

---

[Agentic Electronic Design Automation: A Handoff Perspective](http://arxiv.org/abs/2606.19795)

- EACP (EDA Agent Communication Protocol): introduces a boundary-centered taxonomy for agentic EDA systems, classifying them into Stage-Bound, Flow-Bound, and Organization-Bound systems based on their handoff validity requirements.
- The paper defines handoff validity as the organizing principle for agentic EDA, ensuring that transferred artifacts satisfy consumer acceptance conditions and carry sufficient context, evidence, and provenance.
- To address interoperability gaps, the authors propose a five-layer EACP research agenda covering agent discovery, messaging, tool invocation, workflow orchestration, and security/IP protection.

---

[The Orchestration Gap: Why Process Automation Stalls in Operationally Complex Industries](http://arxiv.org/abs/2606.19790)

- Orchestration Layer: introduces a conceptual framework for coordinating multi-step workflows in complex industries by integrating Heterogeneous Inputs, Workflow Router, State Manager, Bounded Executor, Domain Knowledge, Human Gates, Constraint Gate, Validated Output, and Audit Trace.
- The paper identifies that process automation stalls due to a missing orchestration abstraction rather than a lack of LLM capability, proposing a staged automation model based on sector-specific friction.
- It establishes that regulatory and liability friction dictate which architectural guarantees—such as constraint enforcement or explainability—are load-bearing for successful deployment.

---



[AgentFinVQA: A Deployable Multi-Agent Pipeline for Auditable Financial Chart QA](http://arxiv.org/abs/2606.19782)

- AgentFinVQA: introduces a multi-agent pipeline for auditable financial chart QA that decomposes queries into a sequence of specialized stages including Planner, OCR Reader, Legend Grounder, Colour-Area Tool, Vision Agent, and Verifier.
- The framework utilizes a Model Evaluation Packet (MEP) to record inputs, outputs, and tool traces for every stage, ensuring full auditability and enabling human-in-the-loop review routing based on verifier confidence signals.
- By employing a prompting-only approach without task-specific fine-tuning or local segmentation models, the system achieves high accuracy on both proprietary and locally-served open-weights LLMs while maintaining strict data residency.

---

[Benchmarking Agentic Review Systems](http://arxiv.org/abs/2606.19749)

- Agentic Review Systems: introduces a comprehensive benchmarking framework for evaluating automated peer review systems using real research papers and a novel perturbation-based error detection benchmark.
- The study evaluates OpenAIReview, ‘coarse, and Reviewer3 across multiple LLMs, demonstrating that these systems effectively track human quality signals and catch significant errors despite not being explicitly trained for review tasks.
- Analysis of a public deployment reveals that while users find AI-generated reviews valuable, precision remains a challenge, with false positives and minor nitpicks constituting the primary sources of negative feedback.

---

[VOiLA: Vectorized Online Planning with Learned Diffusion Model for POMDP Agents](http://arxiv.org/abs/2606.19729)

- VOiLA: introduces a framework that learns task-agnostic POMDP models using conditional diffusion and distillation to enable efficient, massively parallel online planning with VOPP.
- The framework utilizes a transition sampler, an observation sampler, and a contrastive likelihood model to perform belief-space planning through GPU-accelerated simulations.
- VOiLA demonstrates high data efficiency and robust generalization by training models in simulation and deploying them directly to physical robots without additional real-world training.

---

[Library-Aware Doubles and Iterative Repair for Large Language Model-Generated Unit Tests in openSIL Firmware](http://arxiv.org/abs/2606.19725)

- Multi-agent UT authoring workflow: introduces a retrieval-augmented iterative pipeline that leverages VDB, LCA, and specialized LLM-agents to automate the generation and repair of unit tests for constrained firmware environments.
- The framework utilizes a multi-stage process including retrieval, drafting, assembly, and an iterative compile-dispatch repair loop to ensure generated tests satisfy strict build constraints.
- By integrating coverage-guided feedback and dependency-aware retrieval, the workflow significantly improves build success rates and line coverage for complex firmware functions compared to direct LLM-only generation.

---

[OnDeFog: Online Decision Transformer under Frame Dropping](http://arxiv.org/abs/2606.19721)

- OnDeFog: introduces an online reinforcement learning framework that integrates robust frame-dropping mechanisms into the Online Decision Transformer to maintain performance under incomplete observations.
- The architecture utilizes a Causal Transformer combined with a Drop-span Encoder and Embedding to explicitly model and adapt to missing state and reward information during online interaction.
- By incorporating train-time frame dropping and dynamic embedding, the framework achieves superior robustness in high-drop-rate environments compared to standard online reinforcement learning methods.

---

[Beyond Static Leaderboards: Predictive Validity for the Evaluation of LLM Agents](http://arxiv.org/abs/2606.19704)

- AssetOpsBench: introduces a twelve-tier measurement apparatus to evaluate LLM agents across orthogonal dimensions, arguing that aggregate-score leaderboards systematically underspecify deployment-relevant performance.
- The framework proposes predictive validity—the correlation between in-sample and out-of-sample rank—as a superior ranking criterion to replace in-sample mean scores for deployment-critical decision making.
- The research synthesizes findings from fourteen parallel implementation studies, identifying critical architectural sensitivities in orchestration, reasoning modes, and infrastructure that current benchmarks fail to capture.

---

[Exit-and-Join Dynamics for Decentralized Coalition Formation](http://arxiv.org/abs/2606.19683)

- Exit-and-Join Dynamics framework: introduces a decentralized model of coalition formation where agents autonomously evaluate local moves using the Aumann-Drèze value to maximize individual payoffs.
- The framework models coalition structures as emergent objects shaped by individual exit-and-join decisions, subject to local acceptance constraints and switching costs.
- The research establishes equilibrium characterizations and convergence properties using Lyapunov analysis, demonstrating how local incentives align with aggregate coalition surplus under specific conditions.

---


[CFAgentBench: A Reproducible Environment and Benchmark for Autonomous Construction-Finance Agents](http://arxiv.org/abs/2606.22000)

- CFAgentBench: introduces a reproducible, self-hostable environment and benchmark for autonomous construction-finance agents, utilizing Company-A book, 35 mock apps, Environment, Agent policy, Grader, and pass1 and passk metrics.
- The framework evaluates LLMs on their ability to perform complex, multi-system financial tasks while adhering to strict safety constraints, specifically a money-movement guard that requires staging rather than executing financial transactions.
- By measuring reliability through passk metrics, the benchmark demonstrates that single-attempt accuracy often overstates the deployable competence of LLMs in high-stakes construction-finance environments.

---

#### 17th June 2026

[Beyond Reward Engineering: A Data Recipe for Long-Context Reinforcement Learning](https://arxiv.org/abs/2606.18831)

- Introduces a data‑centric recipe showing that diverse long‑context RL training—without specialized reward engineering—substantially boosts LLM long‑context reasoning across benchmarks.
- Defines three core agentic system components: Retrieval/Multi‑evidence Synthesis/Reasoning. Constructs eight datasets (~14K examples) explicitly targeting these abilities.
- Demonstrates strong transfer to agentic tasks, with continued RL training improving GAIA (+4.8) and BrowseComp (+7.0), indicating long‑context reasoning as a key bottleneck for agent performance.

---

[Generative-Model Predictive Planning for Navigation in Partially Observable Environments](http://arxiv.org/abs/2606.18888)

- BeliefDiffusion: introduces a framework that combines diffusion-based generative modeling with Model Predictive Control to enable robust navigation in partially observable environments.
- The framework utilizes a diffusion model to characterize multimodal belief distributions by imagining plausible local map configurations from partial observation histories.
- Model Predictive Control then evaluates navigation strategies across these aggregated belief samples to identify optimal paths while hedging against environmental uncertainty.

---

[A Categorial and Sheaf-Theoretic Semantics for Autonomic Component Ensembles](http://arxiv.org/abs/2606.19525)

- SCEL (Software Component Ensemble Language): introduces a multi-layered mathematical model using category theory and sheaf theory to provide denotational semantics for autonomic component ensembles.
- The framework models robotic societies as sheaves on topological spaces, where components act as points and ensembles as open sets, enabling the quantification of system failures as topological obstructions.
- By mapping SCEL constructs to categorical and sheaf-theoretic structures, the approach facilitates formal verification of emergent global behaviors and task solvability in decentralized systems.

---


[Fair Online Resource Allocation](http://arxiv.org/abs/2606.18679)

- Fair Online Resource Allocation: introduces a theoretical framework and online algorithm for fair resource allocation under Lipschitz fairness constraints and capacity limits.
- The approach utilizes a Fair Water-Filling algorithm for offline analysis and a dual mirror descent framework to achieve sublinear regret in online settings.
- Empirical validation on refugee resettlement data demonstrates that the algorithm effectively balances welfare maximization with individual fairness requirements.


[Native Active Perception as Reasoning for Omni-Modal Understanding](http://arxiv.org/abs/2606.19341)

- OmniAgent: introduces a native agentic framework that formulates video understanding as a POMDP-based iterative Observation-Thought-Action (OTA) cycle to decouple reasoning complexity from video duration.
- The framework utilizes a persistent textual memory to store distilled information while discarding high-dimensional raw media, enabling efficient long-form video reasoning.
- OmniAgent is optimized via Agentic SFT for trajectory synthesis and TAURA, an entropy-steered reinforcement learning objective that steers credit assignment toward pivotal discovery turns.

---

[Beyond the Current Observation: Evaluating Multimodal Large Language Models in Controllable Non-Markov Games](http://arxiv.org/abs/2606.19338)

- RNG-Bench (Reconstructive Non-Markov Games): introduces a benchmark suite designed to isolate the ability of MLLMs to reconstruct past observations and act on them during multi-step, non-Markovian interactions.
- The framework evaluates MLLMs across two complementary environments, Matching Pairs and 3D Maze, using controlled difficulty axes and a Memory Gap metric to disentangle forgetting from poor decision-making.
- Supervised fine-tuning on optimal-policy rollouts and filtered model demonstrations improves performance on RNG-Bench and transfers to external memory and spatial benchmarks without degrading general multimodal capabilities.

---

[Learning User Simulators with Turing Rewards](http://arxiv.org/abs/2606.19336)

- Turing-RL: introduces a reinforcement learning approach for training user simulator LLMs by optimizing for indistinguishability from real human users using a discriminative Turing reward.
- The framework utilizes an LLM judge to score candidate responses against ground truth human data, training the simulator policy via GRPO to produce human-like outputs conditioned on user history and induced personas.
- Experimental results across conversational chat and Reddit forum domains demonstrate that Turing-RL consistently outperforms baseline methods in human-likeness and context grounding without sacrificing content alignment.

---

[Data Intelligence Agents: Interpreting, Modeling, and Querying Enterprise Data via Autonomous Coding Agents](http://arxiv.org/abs/2606.19319)

- DIA (Data Intelligence Agents): introduces a system of three agents—Data Interpreter, Schema Creator, and Query Generator—that compresses enterprise data workflows by treating an ACA as the central abstraction for generating, executing, and validating artifacts.
- The framework utilizes a shared workspace for persistent artifact storage and a shared memory system to enable experience reuse across tasks, replacing lossy text handoffs with inspectable, executable outputs.
- DIA achieves state-of-the-art performance across seven SQL benchmarks by grounding agentic reasoning in execution and self-verification, allowing a single LLM to replace specialized, brittle pipeline systems.

---

[Enhancing Decision-Making with Large Language Models through Multi-Agent Fictitious Play](http://arxiv.org/abs/2606.19308)

- MAFP (Multi-Agent Fictitious Play): introduces a multi-agent framework that addresses stance entanglement in decision-making by decomposing stakeholder stances into agents and iteratively refining policies through fictitious play.
- The framework utilizes an aggregation operator to construct empirical mixtures of past policies and a best-response operator to update agent strategies, effectively replacing complex recursive anticipation with iterative, flat-reasoning updates.
- Experimental results across 13 competitive scenarios demonstrate that MAFP achieves superior tournament strength and robustness compared to existing single-round and multi-round LLM-based baselines.

---

[Does VLA Even Know the Basics? Measuring Commonsense and World Knowledge Retention in Vision–Language–Action Models](http://arxiv.org/abs/2606.19297)

- Act2Answer: introduces a lightweight evaluation protocol that adapts VLM knowledge benchmarks to VLA models by requiring agents to perform object-placement actions to select answers.
- The framework utilizes a VLM backbone and an Action Expert to evaluate knowledge retention across diverse categories, including physical, temporal, and social domains.
- The research identifies a performance gap between simple perceptual tasks and complex semantic reasoning in VLA models, revealing that answer-relevant information often attenuates in final layers.

---

[A Mixed-Reality Testbed for Autonomous Vehicles](http://arxiv.org/abs/2606.19267)

- Mixed-Reality HIL Testbed: introduces a hardware-in-the-loop platform that integrates a CARLA Simulator (high-fidelity virtual environment) with Physical Mobile Robots (AgileX Limo hardware) to bridge the gap between simulation and real-world deployment.
- The framework utilizes a Parametric CBF Controller (safety-guaranteed control layer) and a Decentralized Control Robot ROS Node (local robot control) to ensure safety-critical multi-agent coordination.
- The system incorporates a Feature Fusion (combines multimodal data) module and an Estimation Head (predicts vehicle states) to enable end-to-end perception, planning, and control validation.

---

[TxBench-PP: Analyzing AI Agent Performance on Small-Molecule Preclinical Pharmacology](http://arxiv.org/abs/2606.19245)

- TxBench-PP: introduces a verifiable benchmark for evaluating LLM-based agents on small-molecule preclinical pharmacology decisions using Task Prompt, Workflow Data Artifacts, Metadata, Structured Answer Schema, Deterministic Grader, Agent Harnesses, and various LLM Models.
- The benchmark assesses whether LLMs can recover accurate scientific conclusions from realistic assay data rather than relying on memorized literature knowledge.
- Results across 16 model–harness configurations indicate that current LLMs remain unreliable for preclinical pharmacology, with the strongest configuration achieving a 59.3% endpoint pass rate.

---

[Runtime Compliance Verification for AI Agents](http://arxiv.org/abs/2606.19242)

- C-Trace (Compliance Trace-based Runtime Agent Conformance Enforcement): introduces a runtime verification framework that maps AI agent execution traces to formal GDPR predicates to enforce compliance via an in-process interceptor.
- The framework utilizes an Agent loop, LLM, Tool APIs, Runtime monitor, Extractor, Policy, Predicates (P1-P4), Attack driver, Observable oracle, Audit log, MFOTL spec, and Rego policy to ensure agent actions adhere to regulatory requirements.
- By evaluating compliance as a runtime property over event streams rather than static messages, the system effectively blocks non-compliant actions while maintaining low latency and high precision.

---

[CodeSentinel: A Three-Layer Defense Against Indirect Prompt Injection in Code Contexts](http://arxiv.org/abs/2606.19235)

- CodeSentinel: introduces a three-layer inference-time defense that sanitizes untrusted code context by identifying and neutralizing high-risk nodes using a Tree-sitter CST Parser, Syntax-Guided Pre-Filtering, CST-Guided Dynamic Min-K% Scoring, and CST-Guided Node Perturbation Analysis.
- The framework operates at the CST-node level to detect adversarial triggers and natural-looking semantic injections while preserving the utility of benign code for downstream LLMs.
- By employing an early-exit strategy across its three layers, CodeSentinel maintains low preprocessing latency while effectively reducing the attack success rate of indirect prompt injections in code-generation workflows.

---

[Forecasting what Matters: Decision-Focused RL for Controlled EV Charging with Unknown Departure Times](http://arxiv.org/abs/2606.19199)

- DF-RL: introduces a decision-focused reinforcement learning framework for EV charging that jointly trains a regression forecaster and a SAC controller to mitigate uncertainties in departure times.
- The framework integrates a neural network-based forecaster into the RL agent's state representation, optimizing the forecaster via a decision-focused loss function that accounts for downstream charging performance.
- Experimental results demonstrate that the DF-RL approach achieves a 5% improvement in total reward and a 14% reduction in unmet energy compared to conventional forecasting methods.

---

[PhantomSkill: Malicious Code Injection in Agent Skill Ecosystems](http://arxiv.org/abs/2606.19191)

- PhantomSkill: introduces a supply-chain attack framework for LLM-based coding agents that utilizes VulMask to disguise malicious payloads as triggerable, vulnerability-shaped code within auxiliary skill resources.
- The framework leverages VulMask to rewrite overt malicious scripts into vulnerability-shaped implementations, effectively bypassing automated security reviewers and LLM-based inspection by masking malicious intent as ordinary insecure code.
- Experimental results demonstrate that the approach maintains high attack success rates while significantly reducing warning rates across various coding agents and LLM backends compared to traditional injection methods.

---

[Learning to Annotate Delayed and False AEB Events: A Practical System for Extreme Class Imbalance and Asymmetric Label Noise](http://arxiv.org/abs/2606.19186)

- AEB Annotation Framework: introduces a practical system for identifying delayed and false AEB triggers by addressing extreme class imbalance and asymmetric label noise through targeted data augmentation and hardness-driven noise suppression.
- The framework utilizes a Transformer-based architecture to process spatiotemporal driving data, incorporating dedicated encoders for ego-vehicle and surrounding agent features.
- The system implements a human-in-the-loop pipeline that achieves a 95% automation rate, enabling continuous model self-improvement through high-quality verified annotations.

---

[The Simplicity Paradox: Why Evolution Does Not Produce Universally Complex Agents](http://arxiv.org/abs/2606.19136)

- Cognitive Economy Framework: introduces a theoretical model explaining why populations favor cognitive simplicity as an adaptive response to the high costs of information acquisition and processing.
- The framework demonstrates that heterogeneous social structures, where a central decision-maker processes complexity for simple followers, can outperform both universal simplicity and universal complexity.
- This research reframes simplicity not as a cognitive failure, but as a scalable, efficient organizational strategy for managing information costs in complex environments.

---

[A Technical Taxonomy of LLM Agent Communication Protocols](http://arxiv.org/abs/2606.19135)

- A Technical Taxonomy of LLM Agent Communication Protocols: introduces a structured taxonomy to classify and analyze communication protocols for LLM-based agents across five dimensions: counterparty, payload, interaction state, discovery mechanism, and schema flexibility.
- The research evaluates nine open-source communication protocols, identifying recurring architectural patterns such as the combination of hybrid payloads with session-state persistence in agent-to-agent systems.
- The paper proposes that the future of multi-agent system communication will likely evolve toward a federated, layered protocol stack, mirroring the OSI model to balance versatility, efficiency, and portability.

---

[Towards an Agent-First Web: Redesigning the Web for AI Agents](http://arxiv.org/abs/2606.19116)

- Agent-First Web Framework: introduces a principled redesign of the web across access, economic, and content layers to accommodate AI agents as first-class citizens through Agent Identification Metadata, agents.txt, Dual-layer Web Architecture, Intent-based Economic Tier Model, Token-based Subscription Model, Commissioned Content Economy, ATML, Human Supervision Tier Model, Provenance Chain Architecture, and Agent Content Discoverability Architecture.
- The framework addresses the structural incompatibility of the human-centric web with AI agents by replacing blanket blocking with graduated rate limiting and establishing a value-exchange economy based on tokens rather than attention.
- To mitigate epistemic recursion, the framework introduces ATML and a human supervision tier model to ensure content provenance and human intentionality are verifiable and machine-readable.

---

[Leadership as Coordination Control: Behavioral Signatures and the Recovery-Advantage Boundary in Multi-Agent LLM Teams](http://arxiv.org/abs/2606.19111)

- Leadership as Coordination Control: introduces a process-level coordination framework for multi-agent LLM teams that operationalizes classical leadership theories as explicit controllers over a shared action vocabulary.
- The framework utilizes behavioral signatures and per-action ablations to measure coordination effectiveness, replacing single-number accuracy as the primary scientific object.
- The research demonstrates that process-level control adds value only in specific readiness-gap regimes where the initial round-0 majority is unreliable yet recoverable, confirming predictions from team science contingency theory.

---

[Taming I2V models for Image HOI Editing: A Cognitive Benchmark and Agentic Self-Correcting Framework](http://arxiv.org/abs/2606.19073)

- SCPE (Self-Correcting Process Editing): introduces an agentic framework that iteratively refines prompts for I2V models to improve Human-Object Interaction (HOI) editing accuracy.
- The framework utilizes a closed-loop system with specialized agents—Generator, Analyzer, Reflector, and Curator—to synthesize failure insights into a dynamic Playbook for robust instruction refinement.
- The paper also presents HOI-Edit, a benchmark structured across three cognitive levels, and HOI-Eval, a grounded metric for evaluating interaction validity and identity preservation.

---

[PYPILINE: Malicious PyPI Package Detection via Suspicious API Knowledge and Agent Workflow](http://arxiv.org/abs/2606.19063)

- PYPILINE: introduces a tool-enabled LLM agent workflow that integrates a structured suspicious API knowledge base with static analysis to detect malicious PyPI packages.
- The framework utilizes an AST Parser, API Graph Generator, and Suspicious API Identifier to build a knowledge base, which guides an LLM-based Agent Workflow in performing semantic analysis and generating structured reports.
- PYPILINE achieves high detection performance by combining automated feature extraction with LLM-driven reasoning, while leveraging external tools like a vector database and mail server for end-to-end automation.

---

[RODS: Reward-Driven Online Data Synthesis for Multi-Turn Tool-Use Agents](http://arxiv.org/abs/2606.19047)

- RODS: introduces a closed-loop RL framework that dynamically synthesizes multi-turn tool-use data by targeting the agent's evolving capability boundary using Reward Calculation, Boundary Detector, Planner Agent, Execution Orchestrator, Query Agent, Rewrite Agent, Critique Agent, and Dynamic Replay Buffer.
- The framework leverages the inherent rollout variance of GRPO as a zero-cost boundary detector to identify informative tasks, which are then expanded into structurally isomorphic variants to maintain training signal.
- By managing a co-evolving replay buffer and employing a multi-agent synthesis pipeline, RODS achieves significant data efficiency, matching large-scale offline pipelines with 20x fewer trajectories.

---

[TRAP: Benchmark for Task-completion and Resistance to Active Privacy-extraction](http://arxiv.org/abs/2606.18996)

- TRAP (Task-completion and Resistance to Active Privacy-extraction): introduces a benchmark for evaluating the inherent tension between task accuracy and privacy leakage in LLM agents, utilizing Multimodal Document Input, Task Query, and Attack Query components.
- The framework demonstrates that soft-constraint defenses fail to resolve the utility-privacy trade-off, as LLM Agents inherently process private information to complete tasks, making them susceptible to extraction via Attack Query.
- The authors propose a structural solution using a Masking Engine and Hash Table to replace private fields with symbolic keys, ensuring the Execution Layer only accesses sensitive data after the LLM Agent has committed its output.

---

[CAPRA: Scaling Feedback on Software Architecture Deliverables with a Multi-Agent LLM System](http://arxiv.org/abs/2606.18976)

- CAPRA: introduces a multi-agent LLM system that automates formative feedback on software architecture deliverables by orchestrating specialized agents through a four-stage pipeline including Document Parsing, Parallel Verification, Evidence Anchoring, and Report Generation.
- The system utilizes PyMuPDF and gpt-4o Vision for multi-modal document extraction, followed by parallel evaluation agents (SpecificationAuditorAgent, TestAuditorAgent, FeatureCheckAgent, TraceabilityMatrixAgent) to identify architectural issues.
- To ensure reliability, CAPRA employs a deterministic Evidence Anchoring mechanism using Fuzzy Matching and a Confidence Filter, while the ConsistencyManager and claude-haiku-4.5 model finalize the feedback within pre-validated LATEX Templates.

---

[EfficientRollout: System-Aware Self-Speculative Decoding for RL Rollouts](http://arxiv.org/abs/2606.18967)

- EfficientRollout: introduces a system-aware self-speculative decoding framework for RL rollouts that utilizes a target-induced self-drafter, a regime-aware SD toggle, and an adaptive draft-length policy to accelerate decoding without requiring separate drafter pretraining.
- The framework employs a weight-quantized drafter derived from the current target policy to maintain synchronization with evolving RL policies while minimizing latency in memory-bound decoding regimes.
- By coordinating a roofline-based SD toggle and adaptive draft-length control, EfficientRollout optimizes rollout generation by avoiding compute-bound phases and matching drafting budgets to evolving acceptance behavior.

---

[Convergence of Replicator Dynamics in the Repeated Prisoner’s Dilemma with Restarts](http://arxiv.org/abs/2606.18965)

- Trigger-restart mechanism: introduces a study of evolutionary dynamics in a well-mixed population playing repeated Prisoner’s Dilemma, where agents adopt length-m strategy sequences that restart upon action disagreement.
- The research demonstrates that increasing strategy length m enables the emergence and stabilization of cooperative equilibria, characterized by an initial "hazing period" of mutual defection.
- Analytical results provide exact convergence guarantees and identify the separatrix manifold that determines the basins of attraction for cooperative versus defection strategies under varying payoff and error conditions.

---

[Online Reward-Punishment Learning from Fixed-Channel Perceptual Event Streams without Environment Rewards](http://arxiv.org/abs/2606.18963)

- OHIRL: introduces a reward-free reinforcement learning framework that derives internal value evidence from posterior-residual trajectories using Mψ, Dω, Cη, Bξ, and Policy/Q components.
- The framework separates prediction, residual dynamics, and trajectory evaluation to learn context-dependent reward-punishment signs without environment-provided scalar rewards.
- OHIRL utilizes a conditional error decomposition to isolate policy loss within Bξ estimation error and RL optimization error, ensuring robust performance across diverse perceptual-packet families.

---

[GraphPO: Graph-based Policy Optimization for Reasoning Models](http://arxiv.org/abs/2606.18954)

- GraphPO (Graph-based Policy Optimization): introduces a reinforcement learning framework that represents reasoning rollouts as a directed acyclic graph to merge semantically equivalent states into equivalence classes.
- The framework utilizes Graph Construction, Reasoning Step, Semantic State, Equivalence Class, Graph Reward, Dual-Group Graph Advantage, and Policy Update to reduce redundant exploration and improve rollout utilization.
- By pooling downstream evidence across equivalent states, GraphPO converts sparse outcome rewards into dense step-level process supervision without requiring additional annotations.

---

[RTSGameBench: An RTS Benchmark for Strategic Reasoning by Vision-Language Models](http://arxiv.org/abs/2606.18950)

- RTSGameBench: introduces a comprehensive benchmark and evaluation platform for assessing the strategic reasoning capabilities of VLMs in large-scale real-time strategy environments.
- The framework integrates full-game evaluations, diagnostic mini-games for specific competencies, and a self-evolving generation pipeline that utilizes multi-agent collaboration to create extensible, user-query-driven scenarios.
- To facilitate VLM operation in complex RTS settings, the paper provides RTSGameAgent, which combines FSM-based group management and agentic memory to handle large unit counts and long-horizon planning.

---

[Decoupling Search from Reasoning: A Vendor-Agnostic Grounding Architecture for LLM Agents](http://arxiv.org/abs/2606.18947)

- DSG (Decoupled Search Grounding): introduces a vendor-agnostic architecture that separates retrieval from reasoning using an MCP-compatible gateway to provide explicit control over provider routing, caching, and output formatting.
- The framework replaces opaque native search with a structured tool layer that normalizes provider responses into source-aware context, mitigating Search-Induced Verbosity in LLM outputs.
- By implementing a search intelligence layer with tiered caching and provider fallback, the architecture achieves significant reductions in latency and search costs while maintaining high accuracy across diverse agentic workloads.

---

[Epistemic Pairwise Maximin Share](http://arxiv.org/abs/2606.18921)

- EPMMS (Epistemic Pairwise Maximin Share): introduces a new fairness notion for fair division of indivisible goods by applying an epistemic relaxation to the pairwise maximin share criterion.
- The paper establishes that 4/5-EPMMS allocations exist for additive valuations and that EGMMS allocations exist for bivalued valuations, both computable in polynomial time.
- The authors demonstrate that EPMMS is incomparable to both MMS and EFX, and provide structural insights into valuation classes including PMMS-feasible and GMMS-feasible functions.

---

[SAGE: Stochastic Prompt Optimization via Agent-Guided Exploration](http://arxiv.org/abs/2606.18902)

- SAGE (Stochastic Prompt Optimization via Agent-Guided Exploration): introduces a multi-agent pipeline that performs diagnostic code execution to optimize LLM prompts through iterative, hypothesis-driven search.
- The framework utilizes an Analyzer Agent, an Orchestrator Agent, multiple Investigator Agents, and Generator Agents to systematically identify and fix failure patterns in LLM prompts.
- SAGE demonstrates effectiveness in open-ended domains by compounding individually-noisy A/B test results into statistically robust performance gains through continuous optimization.

---

[Skill-Guided Continuation Distillation for GUI Agents](http://arxiv.org/abs/2606.18890)

- SGCD (Skill-Guided Continuation Distillation): introduces an iterative self-improvement framework that addresses the off-trajectory supervision deficit in GUI agents by synthesizing verified successful continuations from policy-induced off-trajectory states.
- The framework utilizes a Gemini-3-Pro Skill Constructor to extract task-specific skills—comprising Continuation Plans, Critical Targets, Failure Traps, and Success Criteria—from both successful and failed rollouts to guide the policy toward successful task completion.
- By mixing these verified continuations with expert trajectories during training, SGCD effectively mitigates distributional shift and improves the success rate of GUI agents on complex, long-horizon tasks.

---

[WorldLines: Benchmarking and Modeling Long-Horizon Stateful Embodied Agents](http://arxiv.org/abs/2606.18847)

- ObsMem (Observer-Grounded Memory): introduces an observer-grounded memory framework that separates event evidence, structured world states, and agent beliefs to support persistent state maintenance and decision-making under partial observability.
- The framework utilizes a Director, Resident Agent, and Robot Agent to generate long-horizon household traces, which are then processed by an Executor and Observer to ensure state consistency and visibility-aware memory ingestion.
- ObsMem organizes interaction streams into typed memory views, enabling evidence-grounded QA and state-aware embodied planning by distinguishing between observed, reported, and inferred information.

---

[Skill-MAS: Evolving Meta-Skill for Automatic Multi-Agent Systems](http://arxiv.org/abs/2606.18837)

- Skill-MAS: introduces a novel paradigm that decouples experience retention from parametric updates by conceptualizing high-level orchestration as an evolvable Meta-Skill, which is refined through a closed-loop process involving Multi-Trajectory Rollout and Selective Reflection.
- The framework utilizes a three-module scaffold—Task Decomposition Module, Agent Engineering Module, and Workflow &amp; Orchestration Module—to enable frozen frontier LLMs to progressively learn and refine architectural strategies across tasks.
- By distilling systemic experience into generalizable principles, Skill-MAS achieves superior cost-performance trade-offs and robust transferability compared to existing inference-time and training-time automatic-MAS approaches.

---

[GATEMEM: Benchmarking Memory Governance in Multi-Principal Shared-Memory Agents](http://arxiv.org/abs/2606.18829)

- GATEMEM: introduces a benchmark for evaluating memory governance in multi-principal shared-memory agents by jointly measuring utility, access control, and active forgetting.
- The framework utilizes hidden checkpoints and structured judging to assess whether LLM agents can maintain authorized usefulness while enforcing strict contextual boundaries and honoring explicit deletion requests.
- Experimental results across diverse domains demonstrate that current memory-augmented LLMs struggle to balance high utility with robust governance, often leaking sensitive or deleted information.

---

[Space Is Intelligence: Neural Semigroup Superposition for Riemannian Metric Generation](http://arxiv.org/abs/2606.18828)

- NSS (Neural Semigroup Superposition): introduces a framework that maps scene descriptions to Riemannian metric fields using a shared Encoder-Router architecture and semigroup superposition to enable zero-shot motion planning.
- The architecture utilizes a fixed Lie algebra generator pool and three complementary parameter groups—frame, modulation, and basic coefficients—to construct a valid symmetric positive-definite metric field.
- By placing intelligence in the geometry of the space rather than an agent-centric policy, the framework achieves robust zero-shot compositional generalization across varying obstacle configurations without explicit collision checking.

---

[Maturing Markov Decision Processes: Decision Making under Increasing Information and Shrinking Action Sets](http://arxiv.org/abs/2606.18820)

- MMDP (Maturing Markov Decision Processes): introduces a framework for sequential decision problems characterized by increasing information and shrinking action sets, utilizing stage-aware policy, expiring-action abstraction, and search-augmented learning with distillation.
- The framework improves learning efficiency by explicitly modeling the information–action asymmetry, allowing the agent to focus on urgent decisions while deferring persistent ones.
- Empirical results across replenishment and cash-management benchmarks demonstrate that the MMDP formulation consistently outperforms flat MDP baselines, particularly as decision problems scale.

---

[Learning from Own Solutions: Self-Conditioned Credit Assignment for Reinforcement Learning with Verifiable Rewards](http://arxiv.org/abs/2606.18810)

- SC-GRPO: introduces a token-level credit assignment method for RLVR that uses KL divergence between a student and a self-conditioned teacher as a multiplicative weight on GRPO gradients.
- The framework constructs a self-conditioned teacher by conditioning the LLM on its own verified rollouts, effectively filtering gradients to focus on critical reasoning tokens.
- SC-GRPO improves performance across math, code, and agentic tasks by replacing uniform sequence-level credit with fine-grained token-level modulation without requiring external resources.

---

[ProfiLLM: Utility-Aligned Agentic User Profiling for Industrial Ride-Hailing Dispatch](http://arxiv.org/abs/2606.18803)

- ProfiLLM: introduces an agentic LLM data pipeline that operationalizes utility-aligned user profiling for industrial ride-hailing dispatch by enforcing a strict offline-online contract.
- The framework utilizes a Tool-Augmented Global Knowledge Mining module to extract actionable insights and an Utility-Aligned Profile Exploration mechanism to generate and refine profiles based on downstream prediction utility.
- By confining all LLM inference to offline batch processing and serving only pre-computed cluster-level embeddings, ProfiLLM achieves significant performance gains with sub-millisecond online latency.

---

[Opinion Polarization in LLM-Based Social Networks: Manipulation and Mitigation](http://arxiv.org/abs/2606.18795)

- LLM-based social network simulation framework: introduces a systematic analysis of how adversarial strategies amplify opinion polarization and how various mitigation mechanisms can be deployed to counter these effects.
- The framework models users as LLM agents that interact through natural language, allowing for the study of complex, persona-dependent opinion dynamics and adversarial manipulation strategies.
- Experimental results demonstrate that while both reactive and proactive interventions can reduce polarization, they often fail to fully restore the network to its baseline state, highlighting the persistent vulnerability of social networks to targeted manipulation.

---

[HandwritingAgent: Language-Driven Handwriting Synthesis in Scalable Vector Space](http://arxiv.org/abs/2606.18788)

- HandwritingAgent: introduces a language-driven agent that synthesizes natural handwriting sequences directly in Scalable Vector Graphics (SVG) format by leveraging an LLM for geometric reasoning and planning.
- The framework utilizes a pre-synthesis stage to create a reusable glyph bank from reference samples, enabling training-free adaptation to diverse handwriting styles and scripts.
- By recasting handwriting synthesis as a reasoning-guided symbolic generation problem, the agent achieves high fidelity in imitating complex mathematical and scientific expressions without task-specific training.

---

[R2D-RL: A RoboCup 2D Soccer Environment for Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2606.18786)

- R2D-RL: introduces a cycle-synchronized MARL environment that bridges the original RCSS2D simulator with modern Python-based learning workflows using a shared-memory communication architecture.
- The framework integrates SoccerServer, HELIOS-based player clients, Coach, and Trainer processes through a shared-memory interface to provide step-synchronous interaction for MARL algorithms.
- R2D-RL supports full-field and scenario-based benchmarks, configurable opponents, discrete and hybrid action spaces, action masks, and parallel execution for multi-agent reinforcement learning research.

---

[REDACTIONBENCH](http://arxiv.org/abs/2606.18782)

- REDACTIONBENCH: introduces a manually annotated benchmark for PII redaction that evaluates models across 200 diverse documents using a novel character-level metric called R-Score.
- The framework utilizes the principle of contextual integrity to categorize information into mandatory and contextual entities, addressing the subjective nature of privacy.
- Extensive evaluations demonstrate that frontier LLMs equipped with agentic tools outperform human baselines, while smaller encoder-only models serve as reliable evaluators for privacy redaction tasks.

---

[What Must Generalist Agents Remember?](http://arxiv.org/abs/2606.18746)

- Generalist Agent Memory Framework: introduces a formal account of the necessary information generalist agents must store in memory to act near-optimally across multiple latent domains and goals.
- The framework establishes that near-optimal agents must maintain distinct memory distributions to disambiguate domains when they require incompatible actions at shared bottleneck states.
- It further demonstrates that memory sufficient for predicting auxiliary goal values enables the reconstruction of local interventional dynamics, characterizing memory as a substrate for planning and model-based reasoning.

---

[SWE-Future: Forecast-Conditioned Data Synthesis for Future-Oriented Software Engineering Agents](http://arxiv.org/abs/2606.18733)

- SWE-Future: introduces a forecast-conditioned data synthesis method that predicts future repository evolution to construct realistic, contamination-aware coding-agent tasks.
- The framework utilizes a multi-agent workflow comprising a Forecaster, Clusterer, Semantic Judge, Task Constructor, and Verification Agent to transform repository-specific signals into executable benchmarks.
- By separating forecast validation from task generation, the method avoids direct historical replay and reduces data contamination risks for LLMs.

---

[Two-Phase Bilevel Search for the Moving-Target Traveling Salesman Problem with Moving Obstacles](http://arxiv.org/abs/2606.18730)

- TPBS: introduces a bilevel search algorithm that interleaves a high-level search for GTSP tour generation with a low-level search for collision-free trajectory validation.
- The framework utilizes a construction phase to quickly identify feasible tours and an improvement phase to refine these solutions for the MT-TSP-MO.
- The approach significantly outperforms existing baselines in success rates and computational efficiency for problems involving moving targets and obstacles.

---

[LEGALWORLD: A Life-Cycle Interactive Environment for Legal Agents](http://arxiv.org/abs/2606.18728)

- LEGALWORLD: introduces a life-cycle interactive environment for legal agents that models Chinese civil litigation as a causally connected state chain across five stages, utilizing Client agent, Lawyer agent, Judge agent, In-scenario local memory, Global case memory, Skill library, Tool library, and LongJud-Bench.
- The environment ensures procedural consistency across the litigation life cycle by maintaining state through In-scenario local memory and Global case memory, while agents interact via role-specific interfaces and Skill/Tool library support.
- LongJud-Bench evaluates LLMs on eight legal capabilities across the full litigation process, revealing that courtroom advocacy remains the most challenging frontier for current models.

---

[Human-AI Agent Interaction in a Business Context](http://arxiv.org/abs/2606.18716)

- Human-AI Agent Interaction Framework: introduces a validated set of eight UX principles and measurable criteria designed to optimize human-AI agent collaboration within enterprise business environments.
- The research utilizes a mixed-methods approach, including participatory design workshops and conjoint experiments, to prioritize human control, transparency, and accountability as critical design requirements for AI agents.
- The study provides empirical evidence that agent transparency and human accountability significantly influence user preferences, offering actionable design guidelines for developing intuitive and trustworthy AI business agents.

---

[Stealthy World Model Manipulation via Data Poisoning](http://arxiv.org/abs/2606.18697)

- SWAAP: introduces a two-stage data poisoning framework that manipulates learned world models by identifying a harmful target model and realizing it through stealth-constrained gradient-matched data poisoning.
- The framework decomposes the attack into Stage 1: Target Model Identification (identifies harmful low-return dynamics) and Stage 2: Data Poisoning (realizes target via gradient matching), while employing a Prediction-error Regularizer (maintains stealth by limiting deviation) to evade detection.
- SWAAP effectively degrades downstream planning performance in continuous control tasks while keeping poisoned transitions close to clean data, demonstrating a practical vulnerability in world-model adaptation pipelines.

---

[HANSEL: Extracting Breadcrumbs from Web Agent Trajectories for Interactive Verification](http://arxiv.org/abs/2606.18671)

- HANSEL: introduces a system that extracts interactive, verifiable evidence from web agent trajectories to replace passive log reading with active verification.
- The framework utilizes an LLM-based Evidence Extractor to identify minimal sufficient evidence sets and reconstructs page states via a BrowserView to enable direct user interaction.
- Technical evaluation demonstrates 83.7% precision in evidence identification and a 61.6% reduction in trajectory volume, significantly improving user verification efficiency and usability.

---

[EARS: Explanatory Abstention for Reliable Sub-Agent Modeling in Large-scale Multi-Agent Systems](http://arxiv.org/abs/2606.18668)

- EARS (Explanatory Abstention for Reliable Sub-Agent Modeling): introduces a production-oriented framework that reframes sub-agent abstention as an inter-agent communication protocol to improve Multi-Agent System reliability, utilizing Human Annotation, LLM-as-a-Judge Calibration, Data Curation, Training, and Multi-Agent System.
- The framework employs a calibrated ensemble of LLMs to generate structured abstention labels and rationales, which are used to fine-tune sub-agents to provide actionable feedback to the coordinator instead of hallucinated outputs.
- Evaluated on a large-scale e-commerce assistant, EARS improves the overall response pass rate by 15.2% by enabling sub-agents to effectively communicate failure states such as ambiguous queries, insufficient input, missing capabilities, or misrouting.

---

[EFX Allocations Exist on Multi-Graphs](http://arxiv.org/abs/2606.18665)

- EFX Allocations on Multi-Graphs: introduces an algorithmic framework to compute EFX allocations for multigraph instances with cancelable valuation functions in polynomial time.
- The framework utilizes a three-phase approach involving initial greedy orientation, construction of a structured partial allocation, and a final dumping phase to complete the allocation.
- The algorithm maintains EFX properties by managing a resent graph and applying specific update rules to resolve envy chains and allocate remaining goods without violating fairness constraints.

---

[LandslideAgent with Multimodal LandslideBench: A Domain-Rule-Augmented Agent for Autonomous Landslide Identification and Analysis](http://arxiv.org/abs/2606.18661)

- LandslideAgent: introduces an instruction-driven agentic framework for autonomous landslide identification and analysis, utilizing LandslideBench, LandslideVLM, LandslideAgent, Dual-rule controller, Memory management module, and Tool library.
- The framework employs LandslideVLM as a cognitive backbone, fine-tuned via LoRA to enhance geoscientific reasoning and semantic understanding of complex geological scenes.
- A dual-rule controller regulates tool invocation by enforcing structured report metadata constraints and cross-validation identification to ensure rigorous and interpretable hazard analysis.

---

[EffiNav: Fusing Depth and Vision-Language for Efficient Object Goal Navigation](http://arxiv.org/abs/2606.18634)

- EffiNav: introduces a training-free, depth-aware framework for efficient ObjNav that utilizes VLM reasoning to select exploration regions and verify them against a top-down map.
- The framework integrates egocentric visual observations with global spatial reasoning to minimize redundant exploration and improve navigation efficiency.
- EffiNav demonstrates robust performance across diverse indoor environments and extends to memory-augmented navigation tasks without requiring additional training.

---

[PersonalPlan: Planning Multi-Agent Systems for Personalized Programming Learning](http://arxiv.org/abs/2606.18633)

- PersonalPlan: introduces a profile-conditioned multi-agent planning framework for personalized programming learning that combines hierarchical SFT, joint alignment, and reward-adaptive GRPO with verifiable rewards for executable structure, profile grounding, and pedagogy.
- The framework utilizes Profile-Aware Decomposition (PAD) and Step Dependency Planning (SDP) to factorize the planning process into scaffold generation and executable workflow construction.
- PersonalPlan achieves state-of-the-art performance in plan executability, personalization, and pedagogical quality by optimizing complete trajectories through Reward-Adaptive GRPO and a hard feasibility gate.

---

[CODE-AUGUR: Agentic Vulnerability Detection via Specification Inference](http://arxiv.org/abs/2606.18619)

- CODE-AUGUR: introduces a security-specification-first paradigm that transforms implicit agentic reasoning into explicit, falsifiable invariants to improve vulnerability detection.
- The framework utilizes a reason-falsify-refine loop where an LLM agent generates security invariants, a guided fuzzer attempts to falsify them, and triage refines the agent's understanding based on the results.
- By committing these invariants as durable in-source assertions, CODE-AUGUR creates reusable security artifacts that persist across the software lifecycle and facilitate regression detection.

---

[Are LLMs Ready to Assist Physicians? PhysAssistBench for Interactive Doctor-Patient-EHR Assistance](http://arxiv.org/abs/2606.18613)

- PhysAssistBench: introduces a multi-turn benchmark that evaluates LLMs on their ability to coordinate clinical knowledge, structured FHIR-based EHR tool use, and ambiguous patient communication.
- The framework utilizes a scalable multi-agent synthetic data pipeline to transform static MIMIC-IV records into interactive agentic patient environments for clinical evaluation.
- Experimental results demonstrate that even leading LLMs struggle with the coordination of implicit physician intent, patient ambiguity, and precise EHR tool integration, highlighting a significant bottleneck for clinical LLMs.

---

[Bridging Creative Intent and Visual Quality: Creator-Driven Recurrent Video Generation with Agentic Feedback Loops](http://arxiv.org/abs/2606.18591)

- CHIEF: introduces a human-AI co-creation framework that utilizes persona-conditioned LLM agents to provide subjective feedback for iterative video refinement.
- The framework integrates a Video Generator, Feedback Agents, and a Feedback Translator to enable creators to maintain narrative control while leveraging automated audience-perspective critiques.
- CHIEF supports both autonomous refinement for short-form content and creator-gated refinement for complex, long-duration films, demonstrating significant improvements in narrative coherence and visual quality.

---

[The Gate Is Only as Honest as Its Contracts: ContractGuard for the Contract Layer of Risk-Aware Causal Gating](http://arxiv.org/abs/2606.18550)

- ContractGuard: introduces a verification layer that secures tool-augmented LLM agents by enforcing contract integrity through signed provenance, typed attestation, and runtime effect verification.
- The framework addresses indirect prompt injection by treating tool contracts as a load-bearing security surface, ensuring that only authorized and causally necessary tools are exposed to the LLM agent.
- By interposing between the registry and the gating mechanism, ContractGuard establishes a strict necessity ladder that neutralizes contract-forgery attacks while maintaining utility on honest inputs.

---

[SAGE-OPD: Selective Agent-Guided Intervention for Multi-Turn On-Policy Distillation](http://arxiv.org/abs/2606.19659)

- SAGE-OPD: introduces a verifier-free selective intervention framework for multi-turn on-policy distillation that dynamically allocates teacher supervision based on turn-level necessity and teacher confidence.
- The framework utilizes a Turn-Level Intervention Module to classify student responses into skip, weak, or strong intervention categories, effectively mitigating compounding errors and brittle token-level alignment.
- By integrating a Teacher Confidence Estimator and a Loss Normalization Component, the approach ensures that distillation is applied most strongly when teacher feedback is both necessary and reliable, improving generalization across embodied and reasoning tasks.

---

[DF-ExpEnse: Diffusion Filtered Exploration for Sample Efficient Finetuning](http://arxiv.org/abs/2606.19656)

- DF-ExpEnse: introduces a technique that improves online experience collection for finetuning pretrained diffusion policies by using a critic ensemble to quantify exploration interest and fleet normalization to facilitate collaborative exploration.
- The framework leverages the multimodal modeling capabilities of diffusion policies to generate a tractable set of action candidates, which are then evaluated based on value estimates and uncertainty scores.
- By integrating BC-SR to maintain multimodal priors and employing cross-agent communication, the method achieves superior sample efficiency across various robotic manipulation and locomotion tasks.

---

[Scaling Self-Play for End-to-End Driving](http://arxiv.org/abs/2606.19641)

- Gigapixel: introduces a high-throughput batched driving simulator that enables scalable self-play training for end-to-end driving models directly from pixel observations.
- The framework utilizes a Vectorized Teacher trained via RL, which is distilled into a Pixel-based Student using self-play DAgger to achieve sample-efficient training of end-to-end policies.
- The approach incorporates a Sim-to-Real Adaptation Module that aligns the perception backbone of the student policy to real-world sensor data, ensuring robust closed-loop performance without human trajectory supervision.

---

[Formal Verification of Learned Multi-Agent Communication Policies via Decision Tree Distillation](http://arxiv.org/abs/2606.19632)

- Formal Verification Framework: introduces an end-to-end pipeline for verifying learned multi-agent communication policies by distilling neural networks into interpretable decision trees for probabilistic model checking.
- The framework utilizes domain-specific feature extraction and CART-based distillation to achieve high-fidelity policy abstractions that enable formal verification of safety, liveness, and cooperation properties.
- By employing pairwise compositional decomposition and discrete communication via VQ-VIB, the approach effectively mitigates state space explosion and provides empirically validated safety guarantees for multi-agent systems.

---

[Joint-task truthfulness of the DMI mechanism](http://arxiv.org/abs/2606.19618)

- DMI mechanism: introduces an analysis of the Determinant Mutual Information mechanism's truthfulness when agents employ joint-task strategies across multiple tasks.
- The research demonstrates that truthful reporting remains a Bayes–Nash equilibrium for the DMI mechanism within the broader class of joint-task strategies.
- The paper establishes that while truthfulness is preserved, informed truthfulness and dominant truthfulness properties fail when peers are not restricted to consistent reporting strategies.

---

[Before the Pull Request: Mining Multi-Agent Coordination](http://arxiv.org/abs/2606.19616)

- grite: introduces a server-less, git-native coordination substrate that enables concurrent coding agents to manage shared work through an append-only event log, CRDT projection, advisory leases, and a dependency graph.
- The framework provides a provenance-bearing, mineable artifact that captures pre-pull request coordination telemetry, addressing visibility gaps in existing software engineering datasets.
- Experimental results demonstrate that combining advisory leases with shared task state effectively eliminates redundant agent work and conflicting edits while maintaining high throughput.

---

[StaminaBench: Stress-Testing Coding Agents over 100 Interaction Turns](http://arxiv.org/abs/2606.19613)

- StaminaBench: introduces a procedural benchmark for evaluating the long-horizon stamina of coding agents by requiring them to maintain and modify a REST API server over 100 consecutive interaction turns.
- The framework utilizes a reference system and an agent system, where the agent operates within an isolated Docker container to implement and iteratively update a REST API server based on procedural change requests.
- Empirical results across multiple LLMs and agent harnesses demonstrate that even strong models fail early, highlighting the critical role of test-feedback loops and harness quality in maintaining long-term coding reliability.

---

[Configurable Clinical Information Extraction with Agentic RAG: What Works, What Breaks, and Why](http://arxiv.org/abs/2606.19602)

- ACIE (Agentic Clinical Information Extraction): introduces an on-premise agentic RAG pipeline that reasons over complete patient contexts to perform structured clinical information extraction with source-grounded verification.
- The framework utilizes an Extraction Agent that performs iterative retrieval and inspection of patient records to populate clinician-defined Extraction Schemas, bypassing unreliable metadata-based filtering.
- Evaluated on a lymphoma registry, the system achieves 96.5% clinician acceptance by grounding every extracted value in source passages, effectively mitigating hallucination risks in clinical workflows.

---

[IHBench: Evaluating Post-Interruption Recovery in Voice Agents with Structured Workflows](http://arxiv.org/abs/2606.19595)

- IHBench (Interruption Handling Benchmark): introduces a synthetic multi-agent pipeline to evaluate how voice agents recover from mid-utterance interruptions within structured enterprise workflows.
- The framework utilizes a multi-stage generation process including a Round Planner, Assistant Simulator, and User Simulator to create controlled interruption scenarios across six distinct types.
- Evaluation is performed using LLM judges on two independent axes: task fulfillment win rate and absolute recovery quality pass rate, revealing significant performance gaps between closed-weight and open-weight LLMs.

---

[Uncertainty Decomposition for Clarification Seeking in LLM Agents](http://arxiv.org/abs/2606.19559)

- Proposed method: introduces a prompt-based uncertainty decomposition that separates action confidence (ct) from request uncertainty (ut) to enable proactive clarification seeking in LLM agents.
- The framework utilizes a deterministic threshold (θ) on the request uncertainty signal to trigger clarification requests when task specifications are ambiguous.
- By appending both uncertainty signals and their explanations to the agent's history, the approach allows LLMs to reason about accumulated uncertainty across multi-step trajectories.

---

[Mesh Inference: A Formal Model of Collective Intelligence Without a Center](http://arxiv.org/abs/2606.19537)

- Mesh Inference: introduces a formal model for collective intelligence where independent agents derive conclusions without a central coordinator or exposure of private state.
- The framework models the mesh as a coupled free energy system where agents perform local relaxation to reach a collective equilibrium.
- It establishes mathematical guarantees for convergence, identification-completeness, and confidentiality based on a single admission/emission policy.

---

[The Sheaf Laplacian: A Topological Framework for Data Fusion and Consensus in Distributed Sensing Networks](http://arxiv.org/abs/2606.19529)

- Sheaf Laplacian Framework: introduces a topological approach for data fusion and consensus in distributed networks by replacing simple graph connections with cellular sheaves that track structured, heterogeneous data.
- The framework utilizes Vertex Stalks, Edge Stalks, and Restriction Maps to model complex, context-dependent relationships, enabling a more expressive representation of network dynamics than traditional graph-based methods.
- By defining consensus as a Consensus Manifold within the kernel of the Sheaf Laplacian, the approach allows for sophisticated, non-uniform agreement states and principled anomaly detection based on violations of relational contracts.

---


[DeXposure-Claw: An Agentic System for DeFi Risk Supervision](http://arxiv.org/abs/2606.19501)

- DeXposure-Claw: introduces a forecast-grounded agentic system for DeFi risk supervision that routes LLM decisions through structured evidence to prevent over-intervention.
- The system integrates DeXposure-FM, deterministic monitors, and stress scenarios to provide an evidence bundle for the Decision LLM, which is then constrained by data-health and confidence gates.
- DeXposure-Bench provides a six-axis evaluation harness to score the pipeline on forecasting, warning, and decision quality against a regulator-aligned absolute-loss ground truth.

---

[LooseControlVideo: Directorial Video Control using Spatial Blocking](http://arxiv.org/abs/2606.19495)

- LCV (LooseControlVideo): introduces a framework for precise directorial video control by conditioning a Wan 2.2 DiT backbone on sparse, oriented 3D bounding boxes rendered into a DNOCS representation.
- The framework utilizes a virtual rendering setup to convert 3D proxies into a 2D control signal, which is injected into the frozen DiT backbone via fine-tuned VACE modules.
- By decoupling global choreography from local object deformation, the method enables intuitive video generation and editing with superior spatial adherence and occlusion reasoning.

---

[Hidden Anchors in Multi-Agent LLM Deliberation](http://arxiv.org/abs/2606.19494)

- Hidden-Anchor Model: introduces a closed-loop dynamical system for multi-agent LLM deliberation that models agents as having hidden internal beliefs, or anchors, which pull their opinions regardless of neighbor influence.
- The framework augments standard consensus dynamics with a per-agent anchor pull, allowing deliberation trajectories to escape the convex hull of initial beliefs, a phenomenon observed in real LLM interactions.
- By recovering these latent anchors through system identification and held-out validation, the authors demonstrate that anchor-driven behavior is a spectrum across different LLM families rather than a uniform property.

---

[Can In-Context Learning Support Intrinsic Curiosity?](http://arxiv.org/abs/2606.19476)

- ICL framework: introduces a method to evaluate prediction-based intrinsic curiosity rewards by leveraging a pretrained in-context learner ρ as an amortized world model.
- The framework utilizes the in-context learner ρ to compute intrinsic rewards, specifically r_sum, which measures the improvement in future prediction errors when a state is observed versus masked.
- The research demonstrates that while finite-horizon rewards are biased, the proposed r_sum reward asymptotically approximates Bayesian information gain in Bayesian Experimental Design settings.

---

[Deontic Policies for Runtime Governance of Agentic AI Systems](http://arxiv.org/abs/2606.19464)

- AgenticRei: introduces a runtime governance architecture that enforces deontic policies on LLM-driven agents by intercepting actions at the middleware boundary to ensure deterministic compliance.
- The framework utilizes a three-step extract-evaluate-apply contract, leveraging an RDFox-based logic engine to process permissions, prohibitions, obligations, and dispensations independently of the LLM.
- By integrating Semantic Web ontologies and verifiable credentials, the system enables complex governance, such as cross-authority trust and obligation lifecycle management, which are structurally inexpressible in standard flat-list policy engines.

---

[MonaVec: A Training-Free Embedded Vector Search Kernel for Edge and Offline AI Systems](http://arxiv.org/abs/2606.19458)

- MonaVec: introduces a deterministic, training-free embedded vector search kernel designed for edge and offline AI systems by utilizing Input Normalization, RHDH Rotation, Lloyd-Max Quantization, Nibble Packing, BruteForce Index, IvfFlat Index, HNSW Index, Global Scalar Standardization, Asymmetric Scoring, BM25 Sparse Index, Reciprocal Rank Fusion, and SIMD Runtime Dispatch.
- The framework achieves high-recall semantic search on resource-constrained devices by employing a data-oblivious quantization pipeline that eliminates the need for training data or persistent server infrastructure.
- MonaVec ensures portable, byte-identical determinism across diverse hardware architectures through a fixed-seed RHDH rotation and a single-file deployment model analogous to SQLite.

---

[Playful Agentic Robot Learning](http://arxiv.org/abs/2606.19419)

- RATS (Robotics Agent Teams): introduces a framework for autonomous skill acquisition through self-directed play, utilizing Task Proposer, Planning Agent, Policy Writer, Executor, Quality Checker, Goal Verifier, Per-Step Verifier, Failure Diagnoser, SubAgent, Memory Curator, Skill Library, and Failure Memory to distill successful behaviors into a reusable library.
- The system employs a curiosity-driven task proposer to generate learnable exploratory goals, which are then executed and verified by a collaborative agent team to build a persistent skill library.
- Learned skills are retrieved at test time to improve performance on downstream tasks, demonstrating effective cross-environment transfer and sim-to-real capability without fine-tuning the underlying model.

---

[MortarBench: Evaluating Mortgage Loan Origination Agents](http://arxiv.org/abs/2606.19416)

- MortarBench: introduces a specialized benchmark for evaluating LLM agents on mortgage loan origination tasks using a synthetic data generation pipeline, MortarBench, and a confidence-calibrated inference framework, CRIT.
- The framework utilizes a mutation-based pipeline to generate internally consistent synthetic bank statements and loan applications, enabling objective evaluation of LLM performance on complex financial reasoning tasks.
- CRIT improves LLM accuracy and reduces systematic biases by incorporating self-reflective confidence scores and threshold-based filtering to mitigate oversensitivity in transaction selection.

---

[OpenRath: Session-Centered Runtime State for Agent Systems](http://arxiv.org/abs/2606.19409)

- OpenRath: introduces a session-centered programming model for agent systems that uses a Session (central flowing runtime value) to unify fragmented runtime states into a single, inspectable, and branchable object.
- The framework organizes agent programs around a compact vocabulary including Sandbox (placement boundary for execution), Tool (model-visible callable operation), Agent (reusable transformation module), Memory (persistent state plane), Workflow (compositional container), and Selector (runtime control flow router).
- By treating the Session as a first-class runtime value, OpenRath enables explicit lineage, tool evidence, and state management, allowing complex multi-agent systems to remain auditable and reproducible.

---

#### 16th June 2026


[Execution-Bound Advisory Automation for Agentic AI: A Reproducible AIBOM-Driven CSAF-VEX Framework](http://arxiv.org/abs/2606.19390)

- AIBOM-driven CSAF-VEX Framework: introduces an execution-bound advisory automation architecture that integrates MCP, A2A, and AGNTCY to transform static vulnerability disclosures into context-aware, reproducible exploitability assessments for Agentic AI systems.
- The framework utilizes MCP for deterministic environment capture, A2A for runtime telemetry collection, and AGNTCY for cryptographically anchored governance to validate exploitability based on observed activation conditions rather than static dependency presence.
- Empirical evaluation demonstrates that this approach significantly reduces false positives in Agentic AI workloads by filtering non-activatable vulnerabilities through runtime-aware reachability analysis and policy-bound enforcement.




[From Ad Hoc Pilots to Repeatable Patterns: Structuring Drone Collaboration in Emergency Services with DroneLets](http://arxiv.org/abs/2606.17839)

- DroneLets: introduces a modular design framework that extends Collaboration Engineering to structure human-drone collaboration in emergency services through repeatable patterns.
- The framework organizes collaboration into Instructions, Environment, Drone, Setup, and Script components to ensure predictable outcomes in high-stakes, embodied agent environments.
- This research derives 44 interaction patterns and 10 meta-patterns from field trials to formalize the integration of autonomous drones into standardized emergency response workflows.

---


[Accountability in Autonomous Drone-Based Firefighting: Insights From a Field Trial](http://arxiv.org/abs/2606.17831)

- Bovens’ accountability framework: introduces a conceptual model for analyzing accountability as a relationship between an actor and a forum, where the actor must justify conduct to a forum that possesses the authority to impose consequences.
- The paper applies this framework to examine how the integration of autonomous drones into firefighting operations affects accountability attribution among human stakeholders.
- Findings from field trials indicate that accountability is consistently shifted to human or collective organizational actors rather than the autonomous drones themselves, highlighting the importance of organizational role clarity and communication.

---

[Divide, Deliberate, Decide: A Multi-Agent Framework for Fine-Grained Egocentric Action Recognition](http://arxiv.org/abs/2606.17627)

- Divide, Deliberate, Decide: introduces a multi-agent framework for fine-grained egocentric action recognition that utilizes an Orchestrator VLM to segment video and an ensemble of heterogeneous Specialist VLM agents to perform structured peer-consultation deliberation.
- The framework employs a Borda Count Aggregator to synthesize rankings from the Specialist VLM Ensemble, which are then used by the Re-ranking Module to refine the initial predictions of the Orchestrator VLM.
- By leveraging decorrelated priors from heterogeneous models, the system improves zero-shot action recognition performance without requiring fine-tuning or large-scale proprietary models.

---


[Talking to Your Data: Exploring Embodied Conversation as an Interface for Personal Health Reflection](http://arxiv.org/abs/2606.17767)

- Dual-Agent System: introduces a decoupled architecture that separates data analysis from conversational presentation to facilitate objective health data reflection.
- The system utilizes an Observer Agent to preprocess wearable data into a structured Insight JSON, which the Presenter Agent then uses to ground its conversational responses.
- A Unity-based interface integrates a static Dashboard and a 3D Character to support joint attention and co-interpretation of personal health metrics.

---


[Future Dynamic 3D Reconstruction: A 3D World Model with Disentangled Ego-Motion](http://arxiv.org/abs/2606.18250)

- FR3D (Future Dynamic 3D Reconstruction): introduces a world model that predicts persistent 3D latent representations for future dynamic 3D reconstruction by explicitly decoupling ego-motion from environmental dynamics.
- The framework utilizes an autoregressive teacher-student distillation strategy to train its Pose Masked Transformer and Spatial Masked Transformer, which operate on state-enriched tokens derived from a pre-trained 3D reconstruction model.
- By disentangling camera trajectory from scene evolution within a unified 3D latent space, the model achieves robust zero-shot generalization and maintains geometric consistency over long-horizon predictions.

---

[ReproRepo: Scaling Reproducibility Audits with GitHub Repository Issues](http://arxiv.org/abs/2606.18237)

- ReproRepo: introduces a scalable framework for auditing research reproducibility by leveraging human-raised GitHub issues as naturally occurring supervision for LLM agents.
- The framework utilizes a compute-light static inspection approach where LLM agents analyze paper-repository pairs to identify potential reproduction blockers without executing code.
- ReproRepo evaluates agent performance using exact and semantic match metrics against a large-scale benchmark of 1,149 machine learning papers and 7,553 human-reported issues.

---

[EvolveNav: Proactive Preflection and Self-Evolving Memory for Zero-Shot Object Goal Navigation](http://arxiv.org/abs/2606.18235)

- EvolveNav: introduces a training-free framework for Zero-Shot Object Goal Navigation that utilizes a self-evolving agentic rule memory and a memory-guided preflection module to enable continuous test-time improvement.
- The framework employs a UCB-based retrieval strategy to select actionable rules from the Rule Bank, allowing the LLM Planner to perform risk-aware navigation and avoid inefficient exploration.
- By distilling navigation rules from past trajectories via Semantic-driven Credit Assignment, the agent adapts to unseen environments without requiring parameter updates.

---

[Learning Red Agent Policy from Observations for Neurosymbolic Autonomous Cyber Agents](http://arxiv.org/abs/2606.18223)

- Policy Learning Technique: introduces a framework that utilizes imitation learning to infer non-observable red agent policies and predict adversarial actions at runtime within partially observable cyber environments.
- The framework employs a three-stage dynamics model architecture, including IDM1, FDM1, IDM2, and FDM2, to map blue observations and actions into latent and subsequently discrete red agent actions.
- These learned models are integrated as LECs into an EBT, enabling the autonomous defender to proactively predict and mitigate adversarial movements across diverse simulated network strategies.

---

[RubricsTree: Scalable and Evolving Open-Ended Evaluation of Personal Health Agents across Health Memory and Medical Skills](http://arxiv.org/abs/2606.18203)

- RubricsTree: introduces a scalable evaluation framework for personal health agents that utilizes an expert-aligned hierarchical taxonomy of atomic Boolean rubrics to replace subjective holistic scoring.
- The framework employs a context-aware adaptive router to dynamically select relevant rubrics, ensuring efficient and clinically rigorous evaluation of open-ended health queries.
- RubricsTree demonstrates significant improvements in expert alignment, robust detection of contextual perturbations, and provides a reliable signal for downstream LLM optimization and reinforcement learning.

---

[Seeing Is Not Screening: Multimodal Hidden Instruction Attacks on Agent Skill Scanners](http://arxiv.org/abs/2606.18198)

- SKILLCAMO: introduces a document-mediated multimodal instruction attack that conceals malicious instructions within images bundled with agent skills to bypass static scanners.
- EXECSCAN: provides an execution-grounded multimodal scanning module that performs intent extraction, behavior reconstruction, abuse assessment, and deliberative execution simulation over skill artifacts.
- The research demonstrates that multimodal agents can recover hidden instructions from visual resources, necessitating security scanners to move beyond static text and code analysis toward execution-time reasoning.

---

[Ergodic Deviation-Robust Equilibrium under Mirror Descent Learning in Finite Games](http://arxiv.org/abs/2606.18194)

- EDRE: introduces a dynamics-relative equilibrium concept that couples a static near-Nash certificate with pathwise deviation-regret guarantees and EMD-based selection.
- The framework utilizes Entropic Mirror Descent to ensure that the learning trajectory remains robust against fixed deviations while converging to stable fixed points of the dynamics.
- EDRE provides a rigorous bridge between static Nash equilibrium and dynamic learning robustness, demonstrating PPAD-hardness in general games and tractability in potential games.

---

[DRFLOW: A Deep Research Benchmark for Personalized Workflow Prediction](http://arxiv.org/abs/2606.18191)

- DRFLOW (Deep Research Benchmark for Personalized Workflow Prediction): introduces a benchmark for evaluating LLMs on their ability to predict personalized, actionable workflows from heterogeneous enterprise data sources.
- The framework includes DRFA, a workflow-oriented agent that utilizes research planning-, action planning-, adaptive action planning-, and workflow generation-agents to ground generic procedures in user-specific evidence.
- DRFLOW evaluates performance using seven diagnostic metrics covering factual grounding, step recovery, structural ordering, condition resolution, and personalization quality across five enterprise domains.

---

[EgoCS-400K: An Egocentric Gameplay Dataset for World Models](http://arxiv.org/abs/2606.18180)

- EgoCS-400K: introduces a large-scale, replay-grounded egocentric dataset for world models, built from professional Counter-Strike gameplay to provide temporally aligned video-action-language trajectories.
- The framework utilizes a multi-grained annotation pipeline, including Demo Collection, Video Rendering, Rendered Video Filter, Player-level Parsing, Keyboard and Mouse Reconstruction, Atomic Action Extraction, Action Timeline and Protected Chains, Dynamic-programming Segmentation, and Prior-Guided VLM Captioning, to transform raw game replays into structured, auditable training data.
- This dataset serves as a practical bridge between passive web videos and real-world embodied data, supporting tasks such as action-conditioned future prediction, scene rollout, and agent action understanding.

---

[All Smoke, No Alarm: Oracle Signals in Agent-Authored Test Code](http://arxiv.org/abs/2606.18168)

- Oracle Signal Taxonomy Framework: introduces a syntactic classification system to evaluate the verification strength of test code generated by LLMs in software pull requests.
- The study reveals that 80.2% of agent-authored test patches contain weak or no explicit oracle signals, indicating a significant gap between test presence and actual verification logic.
- Empirical analysis demonstrates that while coding agents reliably generate test structure, they often fail to implement robust assertions, though strong oracle signals are positively correlated with higher merge likelihood after controlling for PR complexity.

---

[Learning Cardiac Electrophysiology Digital Twins Through Agentic Discovery of Hybrid Structure](http://arxiv.org/abs/2606.18154)

- LEADS: introduces an agentic framework that automates the construction of personalized cardiac EP digital twins by navigating a structured action space of diffusion and reaction components.
- The framework utilizes an LLM agent to iteratively select, refine, and modify hybrid model architectures based on empirical training feedback from an observation archive.
- By combining LLM-driven architectural search with domain-specific physics priors, LEADS achieves superior performance and physical interpretability compared to unconstrained LLM-based modeling approaches.

---

[WEQA: Wearable hEalth Question Answering with Query-Adaptive Agentic Reasoning](http://arxiv.org/abs/2606.18147)

- WEQA: introduces a query-adaptive agent framework that unifies LLM reasoning with specialized wearable analytical and modeling tools to provide grounded health answers.
- The framework utilizes an LLM controller to dynamically synthesize execution plans, route queries to appropriate sensor-native tools, and perform grounded response auditing.
- WEQA outperforms existing baselines by adaptively combining sensor-native modeling with LLM reasoning, achieving significant gains in accuracy, personalization, and clinical soundness.

---

[Memory as a Wasting Asset: Pricing Flash Endurance for Embodied Agents, and the Limits of Doing So](http://arxiv.org/abs/2606.18144)

- Memory as a Wasting Asset: introduces a framework for pricing flash endurance in embodied agents by treating memory as a depreciating capital asset across a three-tier RAM, On-board NVM, and Cloud hierarchy.
- The framework utilizes a Wear-aware controller to optimize memory placement based on an Endurance rent η, which represents the scarcity value of non-renewable program/erase cycles.
- This approach formalizes memory management as a capital-budgeting problem, demonstrating that optimal placement is a threshold in a wear-augmented index that remains cost-optimal regardless of the value-write association.

---

[Your AI Travel Agent Would Book You a Bullfight: An Agentic Benchmark for Implicit Animal Welfare in Frontier AI Models](http://arxiv.org/abs/2606.18142)

- TAC (Travel Agent Compassion): introduces an agentic benchmark designed to evaluate whether LLMs prioritize animal welfare when performing autonomous travel booking tasks.
- The framework utilizes a programmatic scorer to assess agent actions across forty-eight travel scenarios, revealing that frontier LLMs frequently prioritize relevance-maximizing harmful options over ethical alternatives.
- Experimental results demonstrate that while base models perform below chance, a simple welfare-eliciting system prompt significantly improves agentic alignment in several frontier LLMs.

---

[Knowledge Reutilization in Meta-Reinforcement Learning](http://arxiv.org/abs/2606.18132)

- ReMAP: introduces a disentangled Meta-RL framework that learns task-level meta-knowledge on a dynamics-simplified agent and transfers it to heterogeneous agents via a semantic-magnitude interface.
- The framework utilizes a DPMM prior to organize latent task modes, enabling the decoupling of task inference from embodiment-specific low-level control.
- By freezing the task inference module and high-level policy, ReMAP achieves efficient cross-embodiment knowledge reutilization with significantly reduced interaction data requirements.

---

[On the Reliability of Networks of AI Agents: Density Evolution, Stopping Sets, and Architecture Optimization](http://arxiv.org/abs/2606.18121)

- Agent-network reliability framework: introduces a sparse role-typed factor graph model to analyze the reliability of multi-agent systems using density evolution and certificate-stopping sets.
- The framework models task resolution as message passing on a graph, where variable agents hold subclaims and check agents act as noisy Boolean verifiers, with three distinct erasure tiers representing variable abstention, verifier failure, and communication loss.
- The paper establishes a density-evolution theorem for asymptotic reliability, characterizes finite-length failure patterns via certificate-stopping sets, and provides a cost-constrained optimization method for architectural design.

---

[Qwen-RobotNav Technical Report: A Scalable Navigation Model Designed for an Agentic Navigation System](http://arxiv.org/abs/2606.18112)

- Qwen-RobotNav: introduces a scalable navigation model built on Qwen3-VL that reframes multi-task navigation as a context modeling problem through a parameterised interface for dynamic observation reconfiguration.
- The system employs an Upper Planner LLM to decompose long-horizon goals into sub-tasks, while Qwen-RobotNav acts as a reactive executor that dynamically adjusts its visual context strategy based on task-specific requirements.
- The architecture integrates a navigation harness that maintains a two-level memory system, enabling effective context compression and long-horizon reasoning across diverse embodied navigation tasks.

---

[Agentic AI-based Framework for Mitigating Premature Diagnostic Handoff and Silent Hallucination in Healthcare Applications](http://arxiv.org/abs/2606.18068)

- Neuro-Symbolic Multi-Agent Triage framework: introduces a multi-agent architecture that replaces unconstrained LLM routing with deterministic orchestration to mitigate premature diagnostic handoff and silent hallucinations.
- The framework utilizes a neuro-symbolic OLDCARTS gate to enforce structured symptom intake and a semantic entropy-based uncertainty quantification module to identify divergent diagnostic outputs.
- A recursive safety supervision loop enables iterative refinement of diagnoses, with the system achieving a 49.3% diagnostic precision on clinical test cases.

---

[Intelligence Entropy Principle and the ADE Stability Engineering Framework](http://arxiv.org/abs/2606.18065)

- ADE (Agent Delivery Engineering): introduces a multi-layer stability framework designed to counteract intelligence entropy in LLM-driven multi-agent systems through L0 Meta-Principle, L1 Physical Laws, L2 Organizational Mechanisms, L3 Execution Standards, and L4 User Adaptation.
- The framework utilizes core components including TM, TKM, TLC, PIG, BCP, SOMA, DSS, FLYer, PRA, CRC, BDDA, MLG, PLG, TTA, PIP, and SCN to ensure system immortality and result correctness.
- ADE provides a dual-dimension analysis model and a five-layer disorder taxonomy to systematically prevent silent failures and performance degradation in autonomous agent deployments.

---

[PseudoBench: Measuring How Agentic Auto-Research Fuels Pseudoscience](http://arxiv.org/abs/2606.18060)

- PseudoBench: introduces an adversarial benchmark to evaluate whether LLM-based agents can identify and resist pseudoscientific narratives during autonomous research.
- The framework utilizes a pipeline of Dataset Construction, Agentic Auto-Research, and an Evaluation Protocol to measure how effectively agents generate misleading scientific reports.
- Experimental results across seven state-of-the-art agents reveal high pseudoscientific hazard and low resistance, highlighting an urgent need for scientific alignment in autonomous research systems.

---

[Compositional Skill Routing for LLM Agents: Decompose, Retrieve, and Compose](http://arxiv.org/abs/2606.18051)

- SKILLWEAVER: introduces a three-stage framework for compositional skill routing that utilizes an LLM Task Decomposer, a Bi-encoder Skill Retriever with FAISS Indexing, and a Dependency-aware DAG Planner to map complex user queries to ordered skill sequences.
- The framework incorporates a SAD Feedback Loop to iteratively refine task decomposition granularity by aligning sub-task descriptions with the available Skill Library.
- SKILLWEAVER addresses the primary bottleneck of task decomposition quality, significantly improving routing accuracy by providing retrieval-augmented hints to the LLM decomposer.

---

[ProvenanceGuard: Source-Aware Factuality Verification for MCP-Based LLM Agents](http://arxiv.org/abs/2606.18037)

- ProvenanceGuard: introduces a source-aware verification pipeline for LLM agents using Model Context Protocol (MCP) to detect cross-source conflation by decomposing answers into atomic claims, routing them to specific evidence, and validating both support and attribution.
- The framework utilizes a random-forest calibrator to integrate NLI, token alignment, and routing features into a robust support decision, while separately verifying that the answer's stated provenance matches the routed source.
- ProvenanceGuard employs a fail-closed answer gate and a RARR-style repair loop to ensure that only source-grounded and correctly attributed claims are presented to the user.

---

[LoopCoder-v2: Only Loop Once for Efficient Test-Time Computation Scaling](http://arxiv.org/abs/2606.18023)

- PLT (Parallel Loop Transformer): introduces a looped architecture that scales latent computation by repeatedly applying a shared Transformer block while mitigating latency and memory costs through parallel execution and shared KV-caching.
- The framework utilizes Cross-Loop Position Offset (CLP) to break sequential dependencies between loops and Gated Sliding-Window Attention (G-SWA) to maintain a constant memory footprint across varying loop counts.
- Empirical analysis of LoopCoder-v2 demonstrates that performance saturates at two loops due to a gain-cost trade-off between productive representational refinement and the fixed positional mismatch cost introduced by CLP.

---

[LegalHalluLens: Typed Hallucination Auditing and Calibrated Multi-Agent Debate for Trustworthy Legal AI](http://arxiv.org/abs/2606.18021)

- LegalHalluLens: introduces an auditing framework that decomposes aggregate hallucination rates into typed profiles and a Risk Direction Index to identify deployment-critical failure modes in legal AI.
- The framework utilizes a multi-agent debate pipeline comprising Skeptic, Supporter, Re-extractor, Arbiter, Verifier, and Judge components, which are calibrated using measured failure modes to reduce fabricated detections.
- By applying asymmetric Add gate and Del gate mechanisms, the system effectively mitigates high-stakes legal extraction errors while maintaining cost-efficiency compared to generic multi-agent approaches.

---

[LLM Consumer Behavior Theory: Foundations of a Novel Research Field](http://arxiv.org/abs/2606.18005)

- LLM Consumer Behavior Theory: introduces a conceptual framework for analyzing consumption decisions made by LLMs acting as autonomous agents on behalf of human users.
- The framework integrates User Profile, Agent Profile, Reflection, Agent Instantiation, and Agentic Market components to formalize how human preferences are translated into agentic market demand.
- This research synthesizes classical economic theory, behavioral economics, and NLP to address challenges in alignment, preference representation, and market dynamics within agentic economies.

---

[A T-API-Compliant ReAct Agentic Loop for Optical Networks: Generic vs. Domain-Specific Tool Abstractions](http://arxiv.org/abs/2606.18000)

- T-API-Compliant ReAct Agentic Loop: introduces a ReAct-based agentic architecture for optical networks that utilizes swappable tool abstractions to interface with T-API northbound controllers.
- The framework evaluates generic HTTP/RESTCONF primitives against domain-specific atomic and composite tool abstractions to optimize LLM performance in network management tasks.
- Experimental results demonstrate that domain-specific composite tools significantly improve oracle-validated correctness and reduce token consumption compared to generic tool interfaces.

---

[A Neuro-Symbolic Approach to Strategy Synthesis for Strategic Logics](http://arxiv.org/abs/2606.17962)

- Neuro-Symbolic Strategy Synthesis Framework: integrates an LLM as a strategy-generation oracle with a formal model checker to produce certified strategies for Multi-Agent Systems.
- The architecture employs a generate-and-certify pipeline where the Qwen3-32B model proposes candidate strategies that are formally validated by the VITAMIN verifier to ensure soundness.
- The framework includes an iterative refinement loop using verifier diagnostics to improve candidate strategies, achieving 92% accuracy on a newly introduced dataset of 4,211 NatATL instances.

---

[PreAct: Computer-Using Agents that Get Faster on Repeated Tasks](http://arxiv.org/abs/2606.17929)

- PreAct: introduces a verified compile-extend-replace loop that enables computer-using agents to improve efficiency on repeated tasks by storing and replaying verified state-machine programs.
- The framework utilizes a Program Selector, State-machine Replayer, CUA fallback, Compiler, Verify-before-Store Gate, and Program Corpus to replace expensive LLM-based reasoning with deterministic, verified execution for familiar tasks.
- By enforcing a verify-before-store discipline, PreAct ensures that only functional, non-lossy programs are added to the corpus, preventing the accumulation of faulty behaviors and enabling monotonic performance gains across repeated task executions.

---

[Parasitic Masquerade: Societal Scale Human–Machine Interaction](http://arxiv.org/abs/2606.17925)

- GMFG (Graphon Mean-Field Game): introduces a quantitative framework to model large-scale human-machine social systems by representing heterogeneous agent interactions through a graphon kernel, utilizing HJB Solver and FPK Solver to analyze emergent mutualistic and parasitic equilibria.
- The framework categorizes agents into Human Expert Users, Human Casual Users, Cooperative Machines, and Extractive Machines to capture complex social dynamics and information flow within a shared Environment.
- The research demonstrates that parasitic relationships can masquerade as productive learning, necessitating the use of causal information flow and belief entropy metrics to distinguish genuine environmental grounding from machine-mediated dependence.

---

[Trustworthy Self-Composable Big-Data-as-a-Service: An LLM-Orchestrated Multi-Agent Framework for Automated Data Engineering, AutoML, MLOps Deployment, and Drift-Aware Lifecycle Optimization](http://arxiv.org/abs/2606.17915)

- BDaaS Framework: introduces a trustworthy, self-composable architecture that utilizes an LLM-orchestrated multi-agent system to automate the entire data science lifecycle from ingestion to deployment.
- The framework integrates specialized agents for data engineering, AutoML, and MLOps, supported by a centralized orchestration layer that ensures artifact governance and human-in-the-loop oversight.
- Experimental results demonstrate that the multi-agent approach improves lifecycle reliability, traceability, and drift recovery compared to manual, AutoML-only, or single-agent LLM baselines.

---

[StepGuard: Guarding Web Navigation via Single-Step Calibration](http://arxiv.org/abs/2606.17871)

- StepGuard: introduces a framework for robust web navigation that mitigates single-step fragility by decoupling objectives via DDPO and calibrating decisions through CANR.
- The framework utilizes DDPO to alternate between navigation-first and answer-first training modes, effectively reducing reward conflicts and gradient interference.
- CANR enhances decision reliability by dynamically estimating action confidence and selectively triggering metacognitive reflection to correct potential errors in long-horizon tasks.

---

[GameCraft-Bench: Can Agents Build Playable Games End-to-End in a Real Game Engine?](http://arxiv.org/abs/2606.17861)

- GameCraft-Bench: introduces an interaction-grounded evaluation framework for end-to-end game generation that assesses executable gameplay through replayed demonstrations and rubric-guided multimodal judging.
- The framework evaluates coding agents by requiring them to transform natural-language specifications into complete, launchable Godot projects that satisfy engine-grounding, artifact-completeness, and interactive-verification desiderata.
- Evaluations of frontier coding agents reveal that while models can often implement recognizable mechanics, they struggle to deliver complete, coherent interactive systems, with performance remaining far from solved.

---

[WallZero: Mastering the Game of WallGo with Strategic Analysis](http://arxiv.org/abs/2606.17847)

- WallZero: introduces an AlphaZero-based agent for the strategic board game WallGo, utilizing a Residual Neural Network, Monte Carlo Tree Search (MCTS), Action Mask, Feature Representation, and a Self-play Module to achieve superhuman performance.
- The framework incorporates tailored action and feature designs, including territory, reachability, and history planes, to enhance playing strength and strategic decision-making.
- Empirical evaluations demonstrate that WallZero outperforms professional Go players and provides a systematic method for assessing game balance and identifying optimal strategies in newly introduced board games.

---

[Environment-Grounded Automated Prompt Optimization for LLM Game Agents](http://arxiv.org/abs/2606.17838)

- RAPOA: introduces an automated prompt optimization framework that decomposes the observation-to-action pipeline into a goal-conditioned Descriptor Agent and an Action Selection Agent, iteratively refining prompts through an LLM-driven evolutionary loop.
- The framework utilizes a Behavior Analyzer to attribute episode outcomes to specific prompt components and a Mutator to generate targeted revisions, validated through environment rollouts without requiring model weight updates.
- Evaluated on the BALROG benchmark, the approach demonstrates significant performance improvements on complex multi-step coordination tasks by optimizing prompts for decomposed agent architectures.

---

[A Framework for Evaluating Agentic Skills at Scale](http://arxiv.org/abs/2606.17819)

- Skill Evaluation Framework: introduces a scalable methodology that synthesizes realistic, executable tasks from skill content to rigorously evaluate how LLM agents utilize domain-specific knowledge artifacts.
- The pipeline integrates an environment engineering agent, a task generation agent, and a validation agent to produce high-quality evaluation samples, which are then assessed by a verification agent using an LLM-as-judge approach.
- Empirical results across 19 agent-model configurations demonstrate that skills significantly improve instruction-following performance, enabling smaller, cheaper models to achieve parity with larger frontier models.

---

[MaineCoon: Pursuing A Real-Time Audio-Visual Social World Model](http://arxiv.org/abs/2606.17800)

- MaineCoon: introduces a 22B parameter real-time audio-visual autoregressive model designed for social-interactive applications, utilizing a forcing-free streaming training paradigm and an agentic streaming inference framework.
- The system integrates native streaming AR training with self-resampling, representation alignment, and reinforced online-policy distillation (ROPD) to achieve sub-second interaction and high-fidelity generation on a single GPU.
- The agentic inference framework employs a planner, observer, and cache manager to maintain long-horizon consistency and mitigate drift, enabling continuous, seam-free streaming generation.

---

[Position: Coding Benchmarks Are Misaligned with Agentic Software Engineering](http://arxiv.org/abs/2606.17799)

- System Harness: introduces a framework for evaluating agentic software engineering by decomposing the composite system into distinct, measurable components rather than relying on end-to-end scores.
- The framework categorizes feedback into inner-, middle-, and outer-loop tiers to provide granular signals for iteration, distinguishing between agent-controlled and external feedback.
- This approach addresses the misalignment of current benchmarks by advocating for harness-aware metadata, multi-shape behavioral verifiers, and component-level evaluation to better reflect real-world software engineering practices.

---

[ARES: A Platform for Adaptive Role-Based Evaluation of Social Engineering Risks in Human–AI Games](http://arxiv.org/abs/2606.17793)

- ARES: introduces a security-oriented experimental platform designed to audit adaptive social engineering risks in LLM-mediated social decision-making through controlled games and multimodal data analysis.
- The framework integrates role-conditioned LLM agents, psychology-informed participant profiling, and synchronized multimodal acquisition to capture behavioral, biometric, and physiological responses during human–AI interactions.
- ARES provides a structured methodology for evaluating human susceptibility to AI-driven influence by aligning game-level decisions with real-time cognitive and affective state indicators.

---

[Mind Companion: An Embodied Conversational Agent for Process-Based Psychotherapy](http://arxiv.org/abs/2606.17789)

- Mind Companion: introduces an embodied conversational agent that integrates multi-layered real-time psychological analysis with process-based therapy principles to support clinical care under human supervision.
- The system utilizes parallel analysis streams—including fact extraction, psychological flexibility process detection, emotion recognition, and safety monitoring—to inform LLM-based response generation grounded in evidence-based literature.
- Evaluations with professional psychotherapists demonstrate that the agent's responses match or exceed human therapist performance in therapeutic quality, collaboration, and alignment with Acceptance and Commitment Therapy principles.

---

[ED3R: Energy-Aware Distributed Disaster Detection Enabled by Cooperative Robotic Agents](http://arxiv.org/abs/2606.17739)

- ED3R: introduces a distributed hierarchical framework that optimizes wildfire detection by coordinating motion planning and computation offloading between a UAV robot and a remote controller.
- The framework utilizes forward-looking neural regression models to anticipate future rewards and select energy-efficient strategies under uncertainty.
- ED3R integrates specialized mechanisms for obstacle avoidance, region-based exploration, and adaptive mission termination to balance detection confidence with energy consumption.

---

[LongWebBench: Evaluating Structural and Functional Webpage Generation in Long-Horizon Settings](http://arxiv.org/abs/2606.17727)

- LongWebBench: introduces a benchmark for evaluating long-horizon webpage generation through W-VFR and W-FFR, utilizing a VLM-based evaluator and a DOM-augmented agent-based pipeline.
- The framework includes a controller, actor agent, and critic agent to verify functional fidelity by executing goal-oriented tasks in a browser environment.
- Experiments reveal that structural fidelity in LLMs degrades with webpage length, and visually plausible generations often fail to support executable multi-step interactions.

---

[Do Generative Recommenders Deepen the Information Cocoon? A Closed-Loop Simulation with LLM-powered User Simulators](http://arxiv.org/abs/2606.17707)

- RecLoop: introduces a closed-loop simulation framework to analyze information cocoon formation in generative recommendation by coupling a Recommender Module with an LLM-powered User Agent Module.
- The framework utilizes a Dynamic User Profile, Dual Memory System, Periodic Reflection Mechanism, and Action Module to simulate long-term user behavior and preference evolution.
- RecLoop enables the evaluation of information cocoons through both exposure-level metrics and model-level code-space structural analysis across repeated feedback cycles.

---

[EComAgentBench: Benchmarking Shopping Agents on Long-Horizon Tasks with Distributed Hidden Intent](http://arxiv.org/abs/2606.17698)

- EComAgentBench: introduces a long-horizon shopping benchmark that evaluates LLMs on their ability to recover distributed intent across a visible query, a tool-gated persona, and scripted clarification.
- The framework utilizes typed, source-tagged rubrics to provide diagnostic feedback on agent failures, attributing errors to specific requirements and their respective information sources.
- By grounding tasks in a real Amazon product catalog and enforcing a strict, automated, and validated construction pipeline, the benchmark ensures reproducible evaluation of agentic planning and information acquisition.

---

[From Trainee to Trainer: LLM-Designed Training Environment for RL with Multi-Agent Reasoning](http://arxiv.org/abs/2606.17682)

- LLM-as-Environment-Engineer: introduces a closed-loop framework where an LLM iteratively redesigns its own training environment generator based on structured performance feedback to maximize future policy improvement.
- The framework utilizes a controllable MAPF-FrozenLake testbed to enable evidence-driven adaptation, allowing the policy agent to diagnose its weaknesses and adjust training distributions accordingly.
- Empirical results demonstrate that this self-improving approach with a 4B parameter model outperforms larger proprietary LLMs and fixed-environment baselines on multi-agent pathfinding tasks.

---

[ENVRL: Learn from Environment Dynamics in Agentic Reinforcement Learning](http://arxiv.org/abs/2606.17680)

- ENVRL: introduces a framework that improves LLM agent performance by incorporating environment dynamics learning via state prediction and inverse dynamics auxiliary objectives.
- The framework utilizes a joint optimization approach where the LLM agent learns to anticipate state transitions and infer causal actions from interaction trajectories, effectively complementing sparse outcome rewards.
- By employing a cosine decay schedule for auxiliary objectives, the method ensures the LLM agent prioritizes environment dynamics early in training before shifting focus to primary task reward maximization.

---

[DeSRPA: Decoupled Speech Role-Playing Agent via Inference-Time Intervention](http://arxiv.org/abs/2606.17669)

- DeSRPA (Decoupled Speech Role-Playing Agent): introduces a training-free framework that utilizes inference-time intervention to synchronize LLM-based cognitive reasoning with expressive TTS rendering.
- The architecture employs a dual-level control mechanism, featuring an LLM Controller for internal cognitive steering and a StyleTTS 2 backbone for external expressive rendering.
- By injecting disentangled control vectors into frozen backbones, the system achieves high personality and emotional consistency without requiring resource-intensive fine-tuning.

---

[Beyond Domains: Reusing Web Skills via Transferable Interaction Patterns](http://arxiv.org/abs/2606.17645)

- SKILLMIGRATOR: introduces a web agent that reuses procedural knowledge across diverse domains by indexing skills as Transferable Interaction Patterns (TIPs) that pair validated skills with structural webpage sketches.
- The framework utilizes a layout-conditioned retrieval mechanism to identify reusable interaction patterns, effectively reducing the number of LLM calls required for web navigation tasks.
- By grounding abstract skill constraints to live webpage elements through slot binding, the agent maintains high success rates while significantly lowering the average LLM-action count compared to existing baselines.

---

[ERQA-Plus: A Diagnostic Benchmark for Reasoning in Embodied AI](http://arxiv.org/abs/2606.17639)

- ERQA-Plus: introduces a diagnostic benchmark for evaluating grounded reasoning in embodied AI through a multi-stage pipeline comprising a Generator, Judge, and Reviser.
- The framework utilizes a structured reasoning taxonomy to organize evaluation across perceptual, action-centric, social, navigation, and contextual domains.
- The benchmark employs an iterative curation process where the Judge and Reviser agents refine QA pairs to ensure high-quality, visually grounded, and unambiguous evaluation data.

---

[OPD-Evolver: Cultivating Holistic Agent Evolver via On-Policy Distillation](http://arxiv.org/abs/2606.17628)

- OPD-Evolver: introduces a slow-fast co-evolution framework that cultivates a holistic agent evolver through on-policy self-distillation, utilizing a Fast Loop for online interaction and a Slow Loop for distilling privileged hindsight into the deployable policy.
- The framework organizes memory into a four-level hierarchy of trajectories, tips, skills, and tools, enabling the agent to perform experience selection, execution, writing, and management as coupled competencies.
- By employing outcome-calibrated memory attribution, the system converts delayed environmental feedback into dense supervision, allowing the agent to learn transferable lifecycle-level competence for self-improvement without requiring privileged feedback at test time.

---

[Closing the Feedback Loop: From Experience Extraction to Insight Governance in Verbal Reinforcement Learning](http://arxiv.org/abs/2606.17591)

- Three-layer architecture with curation loop: introduces a framework for verbal reinforcement learning that mitigates the retention-forgetting dilemma by separating distilled Rules, persistent Evidence, and governing Skills.
- The system utilizes a feedback-driven pipeline consisting of a Critic, Proposer, and Curator to ensure that knowledge lifecycle decisions are grounded in persistent, append-only evidence logs.
- By shifting focus from mere experience extraction to active insight governance, the architecture enables LLMs to improve performance in non-stationary environments without requiring parameter updates.

---

[Cordon: Semantic Transactions for Tool-Using LLM Agents](http://arxiv.org/abs/2606.17573)

- Cordon: introduces a transactional execution runtime for LLM agents that treats multi-step workflows as semantic transactions to validate effects before commit.
- The system utilizes a Mediation layer, Transaction manager, Shadow-state engine, Effect outbox, Validation engine, and Recovery log to ensure atomic, reversible, and auditable agent operations.
- By interposing at the tool-dispatch boundary, Cordon prevents irreversible side effects by staging mutations and external actions until the entire task flow is validated against security policies.

---

[An AI Security Agent for Banking: Multi-Vector Fraud and AML Detection Across Retail and Corporate Accounts](http://arxiv.org/abs/2606.17555)

- AI Security Agent for Banking: introduces a three-component fusion architecture that processes parallel transaction and session streams to detect 13 distinct fraud and AML threat categories.
- The system integrates an LSTM Sequence Model, a Threshold Monitor, and a Graph Network Module to generate a unified risk score, enabling detection of collective anomalies like business email compromise that evade traditional rule-based engines.
- The agent includes automated response tiers and support tools, specifically a Customer Chatbot for identity verification and an Analyst Case-Summary Assistant for accelerated incident resolution.

---

[SEAGym: An Evaluation Environment for Self-Evolving LLM Agents](http://arxiv.org/abs/2606.17546)

- SEAGym: introduces a unified evaluation environment that converts static benchmarks into dynamic self-evolution task sources to measure LLM agent harness updates across training, validation, and held-out transfer views.
- The framework models self-evolution as an RL-style process where agents update their persistent harness state—including prompts, memory, tools, and middleware—based on task trajectories and verifier feedback.
- By separating rollout from update logic, SEAGym enables fine-grained diagnostics of agent evolution, including forgetting, regression, and the impact of batch size and source diversity on harness reliability.

---

[Offline Preference-Based Trajectory Evaluation](http://arxiv.org/abs/2606.17541)

- Offline Preference-Based Trajectory Evaluation: introduces a family of trajectory-aware evaluation measures—LR, RPP, and IPP—that improve sensitivity and data efficiency by comparing temporal preferences over progress instead of collapsing performance into binary success metrics.
- The framework utilizes pairwise comparisons and the Bradley-Terry model to derive stable system rankings, effectively mitigating benchmark saturation and reducing the number of tied comparisons found in traditional evaluation.
- Empirical results across diverse agentic benchmarks demonstrate that these preference-based methods recover significant performance signals and provide more reliable, data-efficient evaluations than standard success-based metrics.

---

[OMNIDRIVE: An LLM-Choreographed Multi-Agent World Model with Unified Latent Co-Compression for Multi-View Driving Video Generation](http://arxiv.org/abs/2606.17536)

- OMNIDRIVE: introduces a multi-agent world model that uses an ARCHITECT, CARTOGRAPHER, and AUDITOR to choreograph multi-view driving video generation through a unified latent token grid.
- The framework employs Latent Co-Compression via a view-time permutation to align language, geometry, and pixel latents within a shared 3-D VAE, ensuring global geometric consistency.
- By binding agent-authored semantic, geometric, and critique streams to a position-aware token sequence, the model achieves state-of-the-art multi-view consistency and controllability in driving simulations.

---

[GASE: Gaussian Splatting–Based Automated System for Reconstructing Embodied-Simulation Environments](http://arxiv.org/abs/2606.17520)

- GASE: introduces a highly automated pipeline for constructing high-fidelity simulation environments by decoupling foreground objects and static backgrounds in the 2D domain before 3D reconstruction.
- The system utilizes SAM3 for robust object localization across multi-view video streams, LAMA for seamless background inpainting, and 3DGS combined with TRELLIS to generate simulation-ready assets.
- Experimental results demonstrate that GASE significantly improves segmentation accuracy and visual quality, enabling robot policies to achieve performance within 10% of real-world training.

---

[Scaling Enterprise Agent Routing: Degradation, Diagnosis, and Recovery](http://arxiv.org/abs/2606.17519)

- Embedding-based Shortlisting: introduces a diagnostic study of single-step routing degradation in enterprise LLM assistants, identifying recall-driven failures as catalog size increases.
- The research decomposes routing performance loss into a retrieval gap, addressable by embedding-based shortlisting, and a confusion gap, which persists despite improved retrieval.
- Empirical results demonstrate that tool-level retrieval consistently outperforms pack-level approaches, recovering significant F1 performance across multiple frontier LLMs.

---

[SpecGen: Accelerating Agentic Kernel Optimization with Speculative Generation](http://arxiv.org/abs/2606.17518)

- SpecGen: introduces an agentic kernel optimization system that utilizes speculative generation to fork non-reasoning kernel candidates during the reasoning process, thereby reducing generation latency and increasing resource utilization.
- The framework includes SpecController, which monitors reasoning traces for trigger signals to initiate speculative generation, and ElasticScheduler, which dynamically reallocates GPU resources between validation and profiling tasks.
- By conditioning speculative generations on reasoning prefixes and repurposing spare GPU memory as a remote KV cache, SpecGen improves profiling feedback and kernel performance while maintaining a modest token budget.

---

[MedEasy: Designing AI Standardized Patients for Clinical Consultation Training](http://arxiv.org/abs/2606.17512)

- MedEasy: introduces a multi-agent system that organizes virtual-patient practice through patient dialogue, clinical actions, decision submission, documentation, and feedback.
- The architecture includes an Auxiliary Agent (maps learner utterances to clinical intents), a Patient Agent (generates case-grounded responses), and an Evaluation Agent (compares session to expert standards).
- The system utilizes a structured case record and retains a consultation trajectory to provide formative feedback on completed and omitted clinical actions.

---

[MagicSim: A Unified Infrastructure for Executable Embodied Interaction](http://arxiv.org/abs/2606.17511)

- MagicSim: introduces a unified embodied interaction infrastructure that integrates configurable world construction, deterministic batched simulation, and grounded robot execution within a single planner-in-the-loop runtime.
- The framework utilizes a shared Markov decision process (MDP) to support diverse research drivers, including RL benchmarking, scripted data collection, and agent-facing interaction, through a common execution interface.
- MagicSim employs an asynchronous microbatch solve-farm to handle complex motion planning and IK requests without blocking the batched simulation loop, ensuring high-throughput data generation and efficient agent interaction.

---

[Evaluating Second-Order Bias of LLMs Through Epistemic Entitlement](http://arxiv.org/abs/2606.17506)

- SOB (Second-Order Bias) evaluation framework: introduces a philosophically grounded diagnostic task to measure how LLMs exhibit social bias when judging biased content by inferring unwarranted demographic attributes.
- The framework utilizes Acceptability logical conditions and a Two-step inference process to surface implicit social maps and associative triggers within LLM judgments.
- Experimental results across diverse models demonstrate that SOB varies systematically by target group and reveals how LLMs rely on misplaced epistemic entitlements to recirculate biased viewpoints.

---

[PARSE: Provenance-Aware Retrieval Sanitization for Professional Domain LLM Agents](http://arxiv.org/abs/2606.17467)

- PARSE: introduces a domain-aware, fact-preserving sanitization pipeline that mitigates prompt injection in enterprise LLM agents by classifying injection likelihood and verifying fact integrity.
- The architecture utilizes a directiveness gate to route documents to either a lightweight paraphrasing path or a comprehensive multi-step sanitization process, optimizing computational efficiency.
- The framework incorporates a closed-loop consistency checker to ensure that critical information is preserved during the sanitization process, maintaining high utility for downstream LLM tasks.

---

[AUTOGATE: Automated Clock Gating via Toggling-Aware LLM-based RTL Rewriting](http://arxiv.org/abs/2606.17461)

- AUTOGATE: introduces an agentic framework for industrial-grade RTL power optimization that utilizes Orchestrator Agent, Rewrite Agent, Reflection Agent, ML-based Toggling Analysis, Stability-based Clustering, RTL Rewrite Templates, and Formal Verification Module to enable workload-aware fine-grain clock gating across large hierarchical codebases.
- The framework employs a divide-and-conquer strategy where the Orchestrator Agent decomposes complex designs into independently optimizable modules, which are then processed by Rewrite Agents guided by ML-based Toggling Analysis and Stability-based Clustering to identify and implement clock-gating opportunities.
- To ensure scalability and correctness, the system incorporates a Reflection Agent for iterative refinement of transformations and a Formal Verification Module to validate that all RTL rewrites preserve functional equivalence and timing constraints.

---

[Can LLMs Be CEOs? Benchmarking Strategic Resource Reallocation with Multi-Role Agent Simulation](http://arxiv.org/abs/2606.17459)

- CEO-BENCH: introduces a multi-agent benchmark for evaluating LLMs on executive-level strategic resource reallocation under cross-functional conflict and asymmetric information, utilizing a CEO-agent, CFO-advisor, CTO-advisor, COO-advisor, CMO-advisor, deterministic rule-based evaluator, and scenario generator.
- The framework evaluates LLMs on their ability to synthesize conflicting recommendations from specialized C-suite advisors into coherent, history-sensitive, and constraint-compliant allocation plans.
- Experimental results across five frontier LLMs reveal a systematic integration–boldness tradeoff, where deeper engagement with conflicting perspectives often leads to less decisive actions.

---

[ICBCBench: An Industry Consortium Benchmark for Financial Deep Research](http://arxiv.org/abs/2606.17458)

- ICBCBench: introduces a dual-track benchmark for financial deep research that integrates objective tasks with verifiable answers and subjective long-form report evaluation.
- The framework utilizes a multi-stage construction pipeline involving domain experts to ensure task realism, alongside a hybrid evaluation protocol combining Expert Rubrics, Citation Consistency Checking, and Source Quality Verification.
- Experiments demonstrate that while proprietary models excel at narrative synthesis, open-agentic frameworks often provide superior objective reasoning, highlighting a critical performance gap in financial deep research systems.

---

[Embodiment Shapes Rolling Behavior in a Multimodal Infant Model](http://arxiv.org/abs/2606.17456)

- MIMo: introduces a computational framework for studying infant motor development by training a virtual embodied agent to perform supine-to-prone rolling using PPO, proprioception, and vestibular sensing.
- The study disentangles the effects of body growth and physical strength on rolling performance by evaluating agents across different embodiment ages.
- Results demonstrate that muscle development is a critical driver for the emergence of rolling, with the model exhibiting coordination patterns and adaptive behaviors consistent with real infant data.

---

[Dissecting model behavior through agent trajectories](http://arxiv.org/abs/2606.17454)

- SSA (Simple Strands Agent): introduces a harness designed to minimize the intent-execution gap by mediating tool calls and feedback in a closed-loop system, utilizing Input/User task, PromptGenerator, Model adapters, Model call, Tool dispatch, Conversation mgr, Execution environment, Hooks, and Output/Terminate.
- The framework employs a solution-distance metric in a text-level code space to analyze agent trajectories, enabling the identification of model-specific behavioral signatures beyond aggregate pass@1 metrics.
- The research demonstrates that minimizing the bidirectional intent-execution gap allows various LLMs to match or exceed official benchmark performance without task-specific tuning.

---

[MapSatisfyBench: Benchmarking Satisfaction-Aware Map Agents through Behavior-Grounded Implicit Decision Factors](http://arxiv.org/abs/2606.17453)

- MapSatisfyBench: introduces a restore-identify-filter framework that reconstructs complete user needs from behavior-chain evidence to evaluate whether LLMs can satisfy implicit decision factors in map services.
- The benchmark utilizes a deterministic replay sandbox to simulate tool-grounded interactions, enabling reproducible evaluation of LLM agents across explicit task completion and implicit need satisfaction.
- Experimental results demonstrate a persistent gap between explicit task completion and satisfaction-aware decision making, highlighting the need for LLMs to proactively acquire evidence from user profiles and interaction history.

---

[MODE-RAG: Manifold Outlier Diagnosis and Energy-based Retrieval-Augmented Generation Evaluation](http://arxiv.org/abs/2606.17449)

- MODE-RAG: introduces a mechanistically grounded multi-agent framework that resolves the intervention paradox in M-RAG systems by dynamically gating interventions using FE-Router, Per-Agent, Cor-Agent, Ret-Agent, Rea-Agent, Gen-Agent, and Overseer.
- The framework utilizes ATLAS Probe and MCTS to perform rigorous causal derivation and logit perturbations, ensuring factual consistency and suppressing sycophancy in LLMs.
- The system incorporates a PORAG-driven Overseer and a Dead Man's Switch to maintain structural stability and prevent catastrophic formatting failures during complex multimodal reasoning.

---

[SoK: AI-Augmented Binary Reversing](http://arxiv.org/abs/2606.17398)

- SoK: AI-Augmented Binary Reversing introduces a unified taxonomy that bridges conventional and AI-augmented binary reversing pipelines through an artifact-centric interface, facilitating a holistic understanding of 22 reversing domains.
- The paper systematizes 144 research papers by mapping analysis artifacts—such as code, graphs, and text streams—through canonicalization, tokenization, encoding, and embedding stages to support downstream semantic inference.
- The study identifies critical validity risks in the AI-driven pipeline, including corpus monoculture, representation misalignment, and tool-chain dependence, while highlighting the emerging role of LLMs and agentic AI systems in goal-driven binary analysis.

---

[Visuals Lie, Consistency Speaks: Disentangling Spatial Attention from Reliability in Vision-Language Models](http://arxiv.org/abs/2606.17389)

- VRP (VLM Reliability Probe): introduces a systematic framework to evaluate VLM reliability by comparing structural attention metrics, mechanistic hidden-state probes, and behavioral self-consistency signals.
- The framework demonstrates that spatial attention patterns are statistically decoupled from model accuracy, while hidden-state probes and self-consistency serve as robust predictors of truthfulness.
- The research reveals architectural divergence in how VLMs process reliability, identifying that LLaVA relies on fragile late-stage bottlenecks, whereas PaliGemma and Qwen2-VL distribute reliability across internal circuits.

---

[Agent Utilities over Generalized Voronoi Regions and their Gradients](http://arxiv.org/abs/2606.17388)

- CIV (Cost-Induced Voronoi) regions framework: introduces a generalized partitioning method that defines agent and team utilities as integrals of density functions over regions determined by arbitrary cost functions.
- The framework utilizes the Reynolds Transport Theorem to derive analytical boundary-integral expressions for utility gradients, enabling efficient optimization of agent states in dynamic environments.
- Experimental results demonstrate that the proposed contour-integral method achieves comparable accuracy to finite-difference baselines while reducing computational time by approximately 25 times.

---

[EgoInfinity: A Web-Scale 4D Hand-Object Interaction Data Engine for Any-View Robot Retargeting and Video-to-Action Robot Learning](http://arxiv.org/abs/2606.17385)

- EgoInfinity: introduces a modular data engine that converts in-the-wild RGB videos into agent-agnostic, metric 4D hand-object interaction representations for robot learning.
- The pipeline integrates perception, segmentation, reconstruction, and interaction-aware refinement to automate the generation of robot-usable data without human-in-the-loop annotation.
- EgoInfinity utilizes an SE(3)-equivariant neural root-frame estimator to enable functional cross-embodiment retargeting of recovered hand motions onto diverse robot morphologies.

---

[Dynamic Malicious Skills in Agentic AI](http://arxiv.org/abs/2606.16287)

- DyMalSkill: introduces a security threat where attackers inject malicious instructions into skill documentation to induce an AI Agent to dynamically modify benign code at runtime.
- The framework utilizes an Injection Function to embed Malicious Content into SKILL.md, which exploits the instruction-following behavior of the AI Agent to execute unauthorized actions.
- To mitigate this vulnerability, the paper proposes a two-layer Permission-based Defense using Bubblewrap and a Copy Monitor to enforce read-only constraints on skill directories, effectively preventing dynamic code modification.

---

[Beyond NL2Code: A Structured Survey of Multimodal Code Intelligence](http://arxiv.org/abs/2606.15932)

- Multimodal Code Intelligence: introduces a structured survey of systems that generate, edit, refine, or reason with code under visually grounded inputs and outputs, categorized into Graphical User Interface, Scientific Visualization, Structured Graphics, and Frontier Tasks and Frameworks.
- The paper formulates the field by the role code plays as a rendered artifact, editable symbolic structure, scientific representation, intermediate reasoning trace, or executable policy.
- It proposes a verification-centered agenda emphasizing Multi-signal Validation, Multi-state Verification, Cross-task Transfer Testing, and Verifiable Agent Traces to move the field toward evidence-grounded executable systems.

---

[Configuration Smells in AGENTS.md Files: Common Mistakes in Configuring Coding Agents](http://arxiv.org/abs/2606.15828)

- Configuration Smells in AGENTS.md Files: introduces a catalog of six common configuration smells in coding agent files and proposes automated heuristics to detect them.
- The study analyzes 100 open-source repositories to evaluate the prevalence of smells, identifying that 91% of projects contain at least one configuration issue.
- The research utilizes LLM-based heuristics and Apriori association rule mining to demonstrate that configuration smells frequently co-occur, potentially impairing agent performance.

---

[Dual-Channel Grounded World Modeling (DCGWM): Structural Prevention of Objective Interference Collapse via Heterogeneous External Grounding with Inward-Only Gradient Flow](http://arxiv.org/abs/2606.18688)

- DCGWM: introduces a partitioned latent space architecture that prevents Objective Interference Collapse by isolating physical and behavioral grounding signals through inward-only gradient flow.
- The framework utilizes PGC and SBGC to independently update Z_p and Z_b subspaces, while the I module manages cross-subspace dependencies without enabling gradient interference.
- The architecture incorporates L_AGA to mitigate rollout drift using asymmetric penalties and employs an isolated GRL to prevent generative objective contamination of the latent world model.

---

[CEO-BENCH: Can Agents Play the Long Game?](http://arxiv.org/abs/2606.18543)

- CEO-BENCH: introduces a comprehensive evaluation framework for LLMs by simulating a 500-day startup operation that requires long-horizon planning, information acquisition, and adaptive strategy.
- The framework utilizes a Language Model Agent interacting with a Terminal, Business Database, and Company Management Tools to navigate a complex, non-stationary Market environment.
- Experimental results demonstrate that while most LLMs struggle with long-term coherence and bankruptcy, top-performing models like Claude Opus 4.8 and GPT-5.5 exhibit sophisticated planning and adaptive behaviors.

---

[Do as the Romans Do: Learning Universal Behaviors from Heterogeneous Agents](http://arxiv.org/abs/2606.18537)

- GRID (General Reward Inference and Disentanglement): introduces a social learning method that disentangles heterogeneous agent rewards into a shared general reward and individual-specific rewards using General Reward Network (θg), Specific Reward Network (θs), Embedding Network (q), Personal Embedding (ω(p)), and Total Reward Reconstructor (R̂(p)).
- The framework utilizes an information-theoretic objective to train a generalist agent on the shared reward, which serves as a robust prior for efficient downstream fine-tuning.
- By isolating universal environmental competencies from conflicting individual goals, the approach avoids mode-averaging bias and outperforms standard learning from demonstration baselines.

---

[Evaluating Prompting-Based Defenses Against Domain-Camouflaged Injection Attacks](http://arxiv.org/abs/2606.18530)

- Prompting-Based Defenses Framework: evaluates the efficacy of various prompting-based strategies against domain-camouflaged injection attacks across three LLM families and three enterprise domains.
- The study demonstrates that paraphrasing retrieved content is the most consistently effective defense, significantly reducing attack success rates while maintaining high utility compared to other methods.
- Results indicate that defense effectiveness is highly model-dependent, with financial domain deployments exhibiting the highest residual risk against these sophisticated camouflage-class attacks.

---

[Task Allocation and Motion Planning in Dynamic, Cluttered Environments via CBBA and Graphs of Convex Sets](http://arxiv.org/abs/2606.18516)

- CBBA+ST-GCS: introduces a coupled framework for multi-agent task allocation and motion planning that utilizes ST-GCS to provide feasibility-aware cost estimates for CBBA bidding.
- The framework extends ST-GCS to a 3D+time representation, enabling agents to account for dynamic obstacles, task timing, and velocity constraints directly within the trajectory optimization process.
- By integrating trajectory-level optimization into the decentralized bidding process, the approach ensures that task assignments are both conflict-free and physically realizable in dynamic, cluttered environments.

---

[Towards Scalable Customization and Deployment of Multi-Agent Systems for Enterprise Applications](http://arxiv.org/abs/2606.18502)

- Multi-Agent System Pipeline: introduces a unified framework for customizing and deploying LLMs in multi-agent systems by distilling capabilities from a teacher model into a compact student model using Context-aware Continual Pretraining, Supervised Fine-Tuning, and Direct Preference Optimization.
- The framework optimizes inference latency and throughput by integrating EAGLE speculative decoding and FP8 post-training quantization, which are stacked to achieve significant speedups in enterprise workloads.
- The system utilizes a specialized User Simulator to generate high-fidelity training data and employs a sequential agent architecture comprising Understander-, Planner-, Evaluator-, Executor- and Explainer-agents to handle complex customer interactions.

---

[VISUALSKILL: Multimodal Skills for Computer-Use Agents](http://arxiv.org/abs/2606.18448)

- VISUALSKILL: introduces a hierarchical multimodal skill framework for computer-use agents that retains visual figures alongside text to improve UI element identification and workflow state verification.
- The framework utilizes a two-stage construction pipeline, combining authored documentation with live-application UI exploration to generate comprehensive, application-specific skill artifacts.
- An on-demand load_topic MCP tool enables agents to retrieve relevant text and visual figures during task execution, significantly outperforming text-only skill baselines on complex computer-use benchmarks.

---

[From Specification to Execution: AI Assisted Scientific Workflow Management](http://arxiv.org/abs/2606.18425)

- AI-Assisted Scientific Workflow Management Framework: introduces a specification-driven methodology that separates workflow intent, design, and implementation to improve transparency and reproducibility in scientific pipelines.
- The architecture integrates an AI-assisted authoring component, an autonomous AI debugging agent, the Pegasus WMS for orchestration, and a Model Context Protocol (MCP) layer for remote management.
- The system utilizes domain-specific plugins and skills to enable LLMs to construct complex, hierarchical workflows while providing a closed-loop mechanism for failure diagnosis and recovery during distributed execution.

---

[Searching for Synergy in Shared Workspace Human-AI Collaboration](http://arxiv.org/abs/2606.18413)

- Collaborative Gym framework: introduces a shared-workspace environment to study how human-AI teams coordinate expertise and mitigate process loss through structured collaboration.
- The framework incorporates shared group memory and simulated HITL gates to explicitly manage responsibility assignment, evidence handoff, and review routing among AI agents and simulated human collaborators.
- Empirical results demonstrate that these structural scaffolds improve team performance and evidence grounding by shifting initiative distribution and ensuring that expertise is operationalized before final hypothesis submission.

---

[CoreMem: Riemannian Retrieval and Fisher-Guided Distillation for Long-Term Memory in Dialogue Agents](http://arxiv.org/abs/2606.18406)

- CoreMem: introduces an edge-cloud hybrid memory architecture that utilizes Riemannian retrieval and Fisher-guided discrete token distillation to optimize long-term memory for resource-constrained dialogue agents.
- The framework employs a locally adaptive Fisher-Rao metric to mitigate hubness in retrieval and a principled Fisher-information-based compression mechanism to maintain factual density within strict VRAM and token budgets.
- CoreMem integrates a local embedding encoder, a Riemannian retriever, an FDTD compressor, a local vector store, and a cloud LLM to bridge the gap between edge hardware constraints and the requirements for lifelong memory agents.

---

[LLMZERO: Discovering Adaptive Training Strategies for RL Post-Training via LLM Agents](http://arxiv.org/abs/2606.18388)

- LLMZERO: introduces a system that discovers adaptive RL post-training strategies by utilizing LLM agents to perform tree search over training trajectories, employing Proposer Agent, Agentic Early Stopper, Tree Search, Checkpoint-Based Strategy Composition, and Automated Pipeline.
- The framework identifies a structural asymmetry where capacity parameters accumulate monotonically while regularization parameters oscillate to manage non-stationary exploration-exploitation tradeoffs.
- LLMZERO consistently outperforms static baselines and general-purpose LLM agents across diverse GRPO tasks by autonomously proposing coordinated multi-parameter transitions based on observed training dynamics.

---

[CaVe-VLM-CoT: An Interpretable Vision-Language Model Framework](http://arxiv.org/abs/2606.18385)

- CaVe-VLM-CoT: introduces a modular reflection-based agentic-RAG framework that enforces evidence-grounded reasoning through a five-stage closed-loop pipeline comprising an Extractor, Retriever, Solver, Citation Injector, and Verifier.
- The framework utilizes a structured feedback loop where the Verifier triggers targeted re-retrieval by the Extractor upon detecting ungrounded claims or hallucinations.
- It introduces CaVeScore, a composite evaluation metric that jointly measures accuracy, citation precision, citation recall, step-level attribution, and evidence grounding.

---

[Guava: An Effective and Universal Harness for Embodied Manipulation](http://arxiv.org/abs/2606.18363)

- Guava: introduces a harness framework for embodied manipulation that leverages iterative perception-reasoning-action loops, semantic action abstractions, and multimodal observations to enable robust agentic behavior.
- The framework utilizes a data-efficient distillation pipeline to transfer embodied tool-use capabilities from frontier VLMs to compact 4B open-source models using fewer than 2K simulation trajectories.
- Experimental results demonstrate that the distilled Guava-Agent-4B achieves performance comparable to frontier proprietary models in both simulation and real-world settings, exhibiting strong generalization and robust failure recovery.

---

[SafeClawBench: Separating Semantic, Audit-Evidence, and Sandbox Harm in Tool-Using LLM Agents](http://arxiv.org/abs/2606.18356)

- SafeClawBench: introduces a staged security benchmark for tool-using LLM agents that separates semantic compliance, audit-visible harm evidence, and sandbox-observed state changes.
- The framework evaluates agent security across 600 adversarial tasks using a multi-level approach that prevents the conflation of textual model responses with actual executable harm.
- By utilizing matched task identities across distinct evaluation levels, the benchmark enables precise analysis of how different prompt-level policies influence model behavior and security outcomes.

---

[Agentra: A Supervisable Multi-Agent Framework for Enterprise Intrusion Response](http://arxiv.org/abs/2606.18325)

- Agentra: introduces a supervisable multi-agent framework for enterprise intrusion response that decomposes reasoning across role-scoped agents, including Planner-, Validator-, Moderator-, and Executor-agents.
- The framework enhances security response by replacing static playbooks with LLM-assisted planning, constrained by a bounded review loop, moderated knowledge retrieval, and gated execution controls.
- Experimental results on a 120-event corpus demonstrate that the full Agentra stack improves FP-aware IRS F1 from 0.61 to 0.84 while maintaining a 0.0% projected harmful-action rate.

---

[TRIDENT: Breaking the Hybrid–Safety–Physics Coupling for Provably Safe Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2606.18308)

- TRIDENT: introduces a co-designed MARL framework that resolves the directed cycle of errors between hybrid actions, training-time safety, and physics-governed dynamics using SHA, PIRC, and LCPO.
- The framework utilizes STGC to reduce discrete gradient bias to O(τ^2), PIRC to decompose value functions into frozen physical priors and learned residuals, and LCPO to enforce per-iterate safety via Lyapunov constraints.
- TRIDENT provides theoretical guarantees of O˜(1/√K) convergence to a constrained Nash equilibrium and O(√K) cumulative violation, demonstrating superior empirical performance in multi-agent cyber-physical systems.

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



