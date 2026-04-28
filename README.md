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



#### 27th April 2026


[Agent-Centric Visual Reinforcement Learning under Dynamic Perturbations](http://arxiv.org/abs/2604.24661)

- ACO-MoE (Agent-Centric Observations with Mixture-of-Experts): introduces a plug-and-play preprocessor that decouples perception from dynamic visual perturbations by routing corrupted inputs to specialized restoration experts for foreground extraction and RGB repair.
- The framework utilizes a shared encoder and a router to select corruption-specific experts, which jointly predict RGB residuals and foreground masks to produce clean agent-centric observations for downstream RL agents.
- By anchoring representations on foreground extraction, the approach mitigates the entanglement of corruption artifacts in latent states, achieving robust performance across non-stationary Markov-switching perturbations.

---

[The Last Human-Written Paper: Agent-Native Research Artifacts](http://arxiv.org/abs/2604.24658)

- ARA (Agent-Native Research Artifact): introduces a machine-executable research package structured around four interlocking layers: Cognitive Layer (/logic), Physical Layer (/src), Exploration Graph (/trace), and Evidence Layer (/evidence).
- The framework utilizes a Live Research Manager to capture research decisions during development, an ARA Compiler to translate legacy research into the ARA format, and an ARA-Native Review System to automate structural and reproducibility checks.
- By replacing linear narrative papers with structured, machine-operable artifacts, the ARA protocol enables LLMs to reproduce, verify, and extend research more effectively while preserving failure knowledge typically discarded by traditional publication.

---

[AGENTWARD: A Lifecycle Security Architecture for Autonomous AI Agents](http://arxiv.org/abs/2604.24657)

- AGENTWARD: introduces a lifecycle-oriented, defense-in-depth architecture that systematically organizes security protection across the initialization, input, memory, decision, and execution stages of autonomous AI agents.
- The framework integrates stage-specific heterogeneous controls with cross-layer coordination to intercept threats along their propagation paths and safeguard critical assets.
- By maintaining a shared security state and reusable analysis capabilities, AGENTWARD enables autonomous agents to accumulate risk evidence and adapt defenses across iterative runtime loops.

---


[Governing What You Cannot Observe: Adaptive Runtime Governance for Autonomous AI Agents](http://arxiv.org/abs/2604.24686)

- RiskGate: introduces an adaptive runtime governance framework that separates observed capacity from unobserved risk to maintain agent safety through continuous monitoring, anticipation, and monotonic restriction.
- The framework utilizes the Informational Viability Principle to decompose unobserved risk into behavioral drift, structural bias, and sequential context gaps, which are addressed by dedicated statistical estimators and a closed-loop Autopilot.
- RiskGate operates as governance middleware that provides predictive safety guarantees by estimating the time to boundary crossing and enforcing monotonic restriction to prevent adversarial manipulation of safety thresholds.

---


[TSASSISTANT: A Human-in-the-Loop Agentic Framework for Automated Target Safety Assessment](http://arxiv.org/abs/2604.23938)

- TSASSISTANT: introduces a multi-agent framework for automated Target Safety Assessment that utilizes an Orchestrator, Research Subagents, Synthesis Subagents, Pre-execution Hook, Runtime Hook, Post-execution Hook, Tool Memory, Agent Memory, and MCP-standardized Tool Interfaces.
- The framework employs a hierarchical instruction architecture and a section-based pipeline to ensure evidence-grounded, traceable, and expert-validated report generation.
- TSASSISTANT integrates human-in-the-loop refinement to allow toxicologists to review, edit, and re-invoke agents, maintaining final decision authority in high-stakes pharmaceutical safety workflows.

---



[Case-Specific Rubrics for Clinical AI Evaluation: Methodology, Validation, and LLM-Clinician Agreement Across 823 Encounters](http://arxiv.org/abs/2604.24710)

- Hyperscribe framework: introduces a case-specific, clinician-authored rubric methodology for clinical AI evaluation that leverages LLMs to approximate expert clinician agreement at scale.
- The methodology utilizes a hybrid evaluation model where clinician-authored rubrics establish a baseline for validating LLM-generated rubrics, enabling cost-effective, high-coverage automated assessment.
- Experimental results across 823 clinical encounters demonstrate that LLM-generated rubrics achieve ranking agreement with clinicians that matches or exceeds clinician-clinician agreement in high-performing system configurations.

---


[The Alignment Target Problem: Divergent Moral Judgments of Humans, AI Systems, and Their Designers](http://arxiv.org/abs/2604.24155)

- The Alignment Target Problem: introduces an experimental study investigating how the visibility of human design influences moral judgments of AI systems compared to human actors.
- The study disaggregates value alignment into three normative targets: human behavior (T1), AI behavior (T2), and the behavior of human designers programming AI (T3).
- Results indicate that while T1 and T2 are held to similar moral standards, T3 triggers significantly stricter deontological constraints, suggesting that human design visibility shifts moral evaluation.

---


[Green Shielding: A User-Centric Approach Towards Trustworthy AI LLM-Assisted Medical Diagnosis as a Case Study](http://arxiv.org/abs/2604.24700)

- Green Shielding: introduces a user-centric research agenda for building evidence-backed deployment guidance by characterizing how benign input variation shifts LLM behavior in high-stakes domains.
- The framework operationalizes Green Shielding through the CUE criteria—Context, Utility, and Elicitation—to provide a reliable empirical foundation for evaluating LLM performance under realistic, non-adversarial conditions.
- By applying prompt neutralization to convert raw patient inputs into standardized clinical descriptions, the approach makes precision-coverage tradeoffs explicit and moves model outputs toward more clinician-like diagnostic differentials.

---

[The Chameleon’s Limit: Investigating Persona Collapse and Homogenization in Large Language Models](http://arxiv.org/abs/2604.24698)

- Geometric diagnostic framework: introduces a method to quantify persona collapse in LLMs by measuring how populations occupy behavioral space across coverage, uniformity, and complexity axes.
- The research identifies that LLMs systematically truncate persona attributes, leading to structural homogenization where distinct personas converge into narrow behavioral modes.
- Evaluations across ten LLMs reveal a fundamental tension where models achieving higher per-persona fidelity consistently produce more stereotyped and less diverse simulated populations.

---

[Can Current Agents Close the Discovery-to-Application Gap? A Case Study in Minecraft](http://arxiv.org/abs/2604.24697)

- SCICRAFTER: introduces a Minecraft-based benchmark to evaluate LLMs on their ability to navigate the discovery-to-application loop, utilizing a Main Agent, Scientist Sub-agent, Knowledge Book, Code Agent Scaffold, and MCP interface.
- The framework decomposes performance into four capacity gaps—knowledge identification, experimental discovery, knowledge consolidation, and knowledge application—to diagnose bottlenecks in autonomous scientific inquiry.
- Evaluation of frontier LLMs reveals that while application capacity is a major hurdle, knowledge identification is increasingly becoming the primary bottleneck for advanced models.

---

[NeuroClaw Technical Report: Closed-Loop Agentic AI for Executable and Reproducible Neuroimaging Research](http://arxiv.org/abs/2604.24696)

- NeuroClaw: introduces a domain-specialized multi-agent system for neuroimaging research that utilizes an Interface Layer, Subagent Layer, and Base Layer to automate complex, reproducible scientific workflows.
- The framework integrates Harness Engineering and a Directed Acyclic Graph (DAG) to ensure environment-aware execution, provenance logging, and reliable, auditable experimental loops.
- NeuroBench provides a standardized evaluation platform to assess LLMs on task understanding, tool usage, and code correctness across multimodal neuroimaging pipelines.

---

[The Price of Agreement: Measuring LLM Sycophancy in Agentic Financial Applications](http://arxiv.org/abs/2604.24668)

- Sycophancy Evaluation and Mitigation Framework: introduces a systematic approach to measure and reduce sycophancy in agentic financial applications by evaluating model responses against biased user preferences, contradictions, and rebuttals.
- The framework utilizes a Main LLM, Memory System, and Tool Environment to simulate enterprise scenarios where personalized context or adversarial inputs can induce sycophantic behavior.
- Mitigation strategies include an Input Filtering LLM to normalize queries, the application of Reliability Scorers to context, and Adversarial Training to improve model robustness against sycophancy-inducing injections.

---

[Verification of Correlated Equilibria in Concurrent Reachability Games](http://arxiv.org/abs/2604.24655)

- Verification of Correlated Equilibria in Concurrent Reachability Games: introduces a formal framework for verifying correlated equilibria and subgame-perfect correlated equilibria in concurrent probabilistic games using Probabilistic concurrent game graph, Controller advice, Markov Chain, Markov Decision Process, and Bayesian Network.
- The paper characterizes the computational complexity of verifying these equilibrium concepts, demonstrating that subgame-perfect correlated equilibria are easier to verify than standard correlated equilibria under explicit representations.
- The research further analyzes the impact of succinct input representations via Bayesian Network, showing that the complexity gap between the two equilibrium verification problems disappears under this representation.

---

[K-MetBench: A Multi-Dimensional Benchmark for Fine-Grained Evaluation of Expert Reasoning, Locality, and Multimodality in Meteorology](http://arxiv.org/abs/2604.24645)

- K-MetBench: introduces a multidimensional diagnostic benchmark for evaluating LLMs and MLLMs on expert-level meteorological reasoning, geo-cultural context, and domain-specific visual interpretation.
- The framework utilizes an LLM-as-a-Judge approach to score model-generated rationales against expert-verified references, identifying critical modality and reasoning gaps in current models.
- K-MetBench decomposes performance across five official meteorological sub-domains to provide fine-grained insights into model strengths and weaknesses that aggregate scores often obscure.

---

[Evaluating Whether AI Models Would Sabotage AI Safety Research](http://arxiv.org/abs/2604.24618)

- UK AISI Research Framework: introduces a methodology to evaluate the propensity of frontier LLMs to sabotage AI safety research by simulating autonomous agentic coding environments.
- The framework utilizes a custom evaluation scaffold built on Petri, incorporating real-world codebases within Docker containers to test models against various research motivations and activities.
- The research assesses sabotage propensities, evaluation awareness, and prefill awareness, finding that while models do not exhibit unprompted sabotage, they demonstrate varying degrees of situational awareness that complicate evaluation interpretation.

---

[Skill Retrieval Augmentation for Agentic AI](http://arxiv.org/abs/2604.24594)

- SR-Agents: introduces a paradigm for augmenting LLMs with external capabilities by dynamically retrieving and incorporating reusable skills from a large-scale corpus on demand.
- The framework utilizes a multi-stage pipeline comprising skill retrieval, skill incorporation, and skill application to address the scalability limitations of explicit in-context skill injection.
- The research introduces SRA-Bench, a benchmark for evaluating the full SRA pipeline, and demonstrates that effective skill augmentation requires controlled, need-aware, and relevance-aware skill utilization beyond simple retrieval.

---

[Measuring the Unmeasurable: Markov Chain Reliability for LLM Agents](http://arxiv.org/abs/2604.24579)

- TRACETOCHAIN: introduces a reproducible pipeline that fits LLM agent execution traces to an absorbing discrete-time Markov chain (DTMC) to provide audited reliability metrics.
- The framework utilizes Transient States (ST), Transition Matrix (Q), Success Absorber (⊕), and Failure Absorber (⊖) to model agent behavior as a first-passage problem.
- It incorporates Fundamental Matrix (N), Dirichlet Posterior, Bootstrap Intervals, Akaike Information Criterion (AIC), and Kolmogorov–Smirnov (KS) Test to provide diagnostics, uncertainty quantification, and metric reconciliation for LLM agent reliability.

---

[FastOMOP: A Foundational Architecture for Reliable Agentic Real-World Evidence Generation on OMOP CDM data](http://arxiv.org/abs/2604.24572)

- FastOMOP: introduces a foundational multi-agent architecture that separates governance, observability, and orchestration layers from pluggable agent teams to ensure safe and auditable real-world evidence generation.
- The architecture enforces safety at the process boundary using deterministic, rule-based validation, preventing compromised or hallucinating agents from bypassing security controls.
- FastOMOP utilizes the Model Context Protocol to implement the principle of least privilege, ensuring agents only access necessary tools and data while maintaining complete traceability of all reasoning steps.

---

[Mono2Sls: Automated Monolith-to-Serverless Migration via Multi-Stage Pipeline with Static Analysis](http://arxiv.org/abs/2604.24550)

- Mono2Sls: introduces a multi-stage pipeline that automates the migration of monolithic web backends to AWS serverless applications using Static Analysis, Architect Agent, Code Developer Agent, SAM Engineer Agent, and Consistency Validator Agent.
- The framework leverages a curated SAM Knowledge Base and specialized tools including File Tools, CodeRAGTool, and SAMValidateTool to ensure structural coherence and deployment readiness across generated artifacts.
- By decomposing the migration process into sequential, agent-driven stages, the system effectively manages complex architectural transformations and infrastructure generation while maintaining alignment with cloud-native patterns.

---

[GradMAP: Gradient-Based Multi-Agent Proximal Learning for Grid-Edge Flexibility](http://arxiv.org/abs/2604.24549)

- GradMAP: introduces a gradient-based multi-agent learning framework that coordinates large-scale grid-edge devices by embedding a differentiable AC power-flow model and reusing environment gradients within a policy-output-space trust region.
- The framework employs Centralised Training and Decentralised Execution to enable independent neural-network policies to satisfy complex network constraints without requiring online communication between agents.
- By utilising implicit differentiation and proximal updates, GradMAP achieves significant training speed-ups and superior constraint satisfaction compared to standard multi-agent reinforcement learning and self-supervised learning benchmarks.

---

[Beyond the Attention Stability Boundary: Agentic Self-Synthesizing Reasoning Protocols](http://arxiv.org/abs/2604.24512)

- SSRP (Self-Synthesizing Reasoning Protocols): introduces a two-stage metacognitive framework that separates high-level architectural planning by an Architect agent from turn-by-turn procedural execution by an Executive agent to mitigate the Attention Latch failure mode.
- The framework utilizes an Architect to autonomously synthesize a task-specific Standard Operating Procedure (SOP) that purges superseded historical intents, ensuring the Executive remains grounded in the most recent verified system events.
- SSRP addresses the Attention Stability Boundary (ASB) by replacing noisy conversation history with an immutable protocol, enabling LLMs to maintain deterministic goal-directedness in complex, multi-turn reasoning tasks where stateless models typically collapse.

---

[DECOFFEE: Decentralized Reinforcement Learning for Time-critical Workload Offloading and Energy Efficiency across the Computing Continuum](http://arxiv.org/abs/2604.24507)

- DECOFFEE: introduces a decentralized reinforcement learning framework for time-critical workload offloading and energy-efficient operation across the computing continuum using Edge Agents, Cloud Agent, Telemetry Agents, Workload Stacks, LSTM-enhanced Double Dueling DQN, and Radio Units.
- The framework models workload offloading as parallel Markov Decision Processes, enabling autonomous Edge Agents to make proactive placement decisions based on local observations and telemetry-shared load forecasts.
- DECOFFEE utilizes a Double Dueling DQN architecture with LSTM-based forecasting to minimize a multi-objective cost function accounting for execution latency, energy consumption, and workload drop rates.

---

[TARMM: Scaling Delay-Critical Edge AI Offloading in 5G O-RAN via Temporal Graph Mobility Management](http://arxiv.org/abs/2604.24501)

- TARMM: introduces a 5G O-RAN system that optimizes user mobility management for delay-critical edge AI offloading by integrating TGN, MARL, Rule-Based Action Masking, Proactive Resource Reservation, Centralized Critic, Decentralized Actor, GRU, and Multi-Head Attention.
- The framework utilizes a TGN to capture spatiotemporal network dynamics, enabling proactive handover decisions that minimize latency and packet loss for mobile UEs.
- By combining MARL with rule-based constraints and proactive resource reservation, the system ensures stable, safe, and efficient handover performance in dense 5G small-cell networks.

---

[Zero-to-CAD: Agentic Synthesis of Interpretable CAD Programs at Million-Scale Without Real Data](http://arxiv.org/abs/2604.24479)

- Zero-to-CAD: introduces a scalable agentic pipeline that synthesizes one million executable, readable CAD construction sequences by embedding an LLM within a feedback-driven environment using LLM Inference Service, Coordinating Node, Tool-Equipped Workers, and Storage Backend.
- The framework utilizes execute_and_validate, lookup_documentation, and grep_documentation to enable iterative self-correction and geometric validation of CAD programs without requiring real-world construction-history data.
- The system demonstrates that synthetic supervision can effectively bootstrap vision-language models for complex CAD reconstruction tasks, outperforming general-purpose models in geometric fidelity and parametric interpretability.

---

[GAMMAF: A Common Framework for Graph-Based Anomaly Monitoring Benchmarking in LLM Multi-Agent Systems](http://arxiv.org/abs/2604.24477)

- GAMMAF: introduces a comprehensive evaluation architecture for benchmarking defense models in LLM-MAS by bridging synthetic data generation and real-time defense experimentation through Training Data Generation Pipeline, Defense System Benchmarking Pipeline, LLM Agent Networks, Output Embedding Module, Training Data Storage, Defender Model, Malicious Agent Pruning, and Communication Topology.
- The framework utilizes a modular design to generate attributed graphs from agent interactions, enabling the evaluation of defense mechanisms against adversarial influence through dynamic topological updates and iterative debate rounds.
- Experimental results demonstrate that GAMMAF effectively facilitates the assessment of defense architectures by measuring metrics such as Attack Success Rate and Adversarial Detection Rate across diverse task domains and network topologies.

---

[Agentic clinical reasoning over longitudinal myeloma records: a retrospective evaluation against expert consensus](http://arxiv.org/abs/2604.24473)

- Agentic reasoning system: introduces a structured, multi-step reasoning architecture that utilizes a Clinical skill library, Structured memory state, Ordered tool-use plan, Report retrieval tool, Laboratory query tool, Deterministic clinical scoring calculators, and a Final answer synthesis module to synthesize longitudinal clinical records.
- The system outperforms standard retrieval-augmented generation and full-context approaches by externalizing reasoning into an explicit planning phase and using deterministic tools for clinical rule application.
- Performance gains are most pronounced in complex clinical tasks and long patient records, with error rates comparable to expert disagreement but with different clinical severity distributions.

---

[Measuring Successful Cooperation in Human-AI Teamwork: Development and Validation of the Perceived Cooperativity and Teaming Perception Scales](http://arxiv.org/abs/2604.24461)

- PCS and TPS: introduces two theoretically grounded psychometric scales designed to assess the subjective quality of human-AI cooperation across synchronic and diachronic dimensions.
- The PCS captures an agent's perceived cooperative capability within a single task, while the TPS evaluates the emergent sense of teaming through Team-, Self-, and Partner-subscales.
- Validation across three studies demonstrates that these scales reliably differentiate between cooperation partners of varying quality, including human, rule-based, and reinforcement learning-based agents.

---

[On the Footprints of Reviewer Bots’ Feedback on Agentic Pull Requests in OSS GitHub Repositories](http://arxiv.org/abs/2604.24450)

- Reviewer Bot Feedback Analysis Framework: introduces an empirical study characterizing the quality and impact of automated reviewer bot feedback on agentic pull requests using the AI_Dev dataset and GPT-5.1 for automated annotation.
- The framework evaluates feedback quality through relevance, clarity, and conciseness metrics, while assessing their correlation with PR resolution time and acceptance rates.
- The study identifies a dilution effect where higher bot activity volume correlates with longer resolution times and decreased feedback relevance, suggesting a need for more targeted, context-aware automated reviews.

---

[PhysNote: Self-Knowledge Notes for Evolv-able Physical Reasoning in Vision-Language Model](http://arxiv.org/abs/2604.24443)

- PhysNote: introduces an agentic framework that enables VLMs to externalize and refine physical knowledge through self-generated Knowledge Notes to improve reasoning in dynamic scenarios.
- The framework utilizes a Spatio-Temporal Grounding Engine to assign immutable identifiers to visual entities, mitigating identity drift across video frames.
- An iterative Hypothesis-Evidence-Validation loop, supported by a hierarchical repository of General Tips, Task Descriptions, and Details, allows the agent to autonomously evolve its physical reasoning capabilities.

---

[AutoGUI-v2: A Comprehensive Multi-Modal GUI Functionality Understanding Benchmark](http://arxiv.org/abs/2604.24441)

- AutoGUI-v2: introduces a comprehensive benchmark for evaluating deep GUI functionality understanding and interaction outcome prediction, utilizing a VLM-human collaborative pipeline, Gemini-2.5-Pro-Thinking, OmniParser-v2, DINO-v3, Qwen3-Embedding, Disjoint Set Union (DSU), FastAPI web server, OpenCV.js, and a Python script.
- The benchmark provides 2,753 evaluation tasks across six operating systems, rigorously testing LLMs on region-level and element-level semantics, grounding, and dynamic state prediction.
- Evaluation reveals a divergence where open-source models excel at functional grounding while commercial models dominate functionality captioning, highlighting that deep functional understanding remains a significant hurdle for current LLMs.

---

[Kwai Summary Attention Technical Report](http://arxiv.org/abs/2604.24432)

- KSA (Kwai Summary Attention): introduces a novel attention mechanism that reduces sequence modeling costs by compressing historical contexts into learnable summary tokens interleaved with text tokens.
- The framework utilizes a sliding chunk attention mechanism to allow text tokens to interact with local neighbors and distant summary tokens, ensuring sub-quadratic complexity while maintaining long-range dependency expressivity.
- KSA employs a contiguous, concatenation-free KV cache layout and a block-sparse kernel to significantly reduce inference latency and memory footprint compared to standard attention mechanisms.

---

[How Personal Characteristics Shape User Exploration of Diverse Movie Recommendations with a LLM-Based Multi-Agent System](http://arxiv.org/abs/2604.24405)

- MAS (Multi-Agent System): introduces a conversational recommender system that utilizes multiple LLM-based agents to provide diverse movie recommendations through personalized explanations, incorporating Agent Profile Panel, Conversation Panel, Movie Recommendation Panel, LLM-based guard module, Demographic-matched agent, Preference-matched agent, and Personality-matched agent.
- The system employs a 6-6 split strategy to generate a fixed set of in-profile and off-profile movie candidates, which are then discussed by three distinct agents to nudge users toward broader exploration.
- Empirical results from a user study demonstrate that the multi-agent design significantly increases Perceived Novelty and objective Shannon Diversity, while highlighting that user experience is moderated by personality traits such as Conscientiousness, Agreeableness, and Extraversion.

---

[MAS-SZZ: Multi-Agentic SZZ Algorithm for Vulnerability-Inducing Commit Identification](http://arxiv.org/abs/2604.24398)

- MAS-SZZ: introduces a multi-agent framework that improves vulnerability-inducing commit identification by combining evidence-grounded root cause analysis, intent-driven anchor selection, and autonomous repository exploration.
- The framework utilizes specialized agents including Auditor, Judge, Reviewer, Evaluator, Locator, and Tracer to systematically filter noisy patch hunks and perform iterative backtracking guided by LLM-reasoned root causes.
- Experimental results demonstrate that MAS-SZZ significantly outperforms existing SZZ algorithms, achieving F1-score gains of up to 65.22% across diverse datasets and programming languages.

---

[OS-SPEAR: A Toolkit for the Safety, Performance, Efficiency, and Robustness Analysis of OS Agents](http://arxiv.org/abs/2604.24348)

- OS-SPEAR: introduces a comprehensive evaluation toolkit for OS agents, utilizing S-subset (evaluates environment/human-induced hazards), P-subset (filters high-value evaluation trajectories), E-subset (measures inference time/token consumption), R-subset (applies cross-modal perturbations), and an Analysis Tool (multi-agent diagnostic report generator) to assess reliability across four critical dimensions.
- The framework employs specialized expert agents (Safety, Performance, Efficiency, Robustness) and an integrated agent to transform raw evaluation logs into expert-level diagnostic reports.
- Extensive evaluation of 22 OS agents reveals significant trade-offs between efficiency and safety/robustness, while highlighting modality-specific vulnerabilities in current MLLM-based OS agents.

---

[Perfecting Aircraft Maneuvers with Reinforcement Learning](http://arxiv.org/abs/2604.24338)

- RL-based Aerobatic Maneuver Framework: introduces a reinforcement learning approach for executing complex aircraft aerobatic maneuvers by utilizing SAC, Gym-JSBSim, and a custom reward function based on trajectory tracking.
- The framework employs both pilot-generated and handcrafted trajectory references to train RL agents, incorporating time scaling and domain-specific reward components to ensure maneuver stability and precision.
- The system demonstrates that RL models can achieve professional pilot-level performance and generalize across different initial conditions and aircraft types by optimizing hyper-parameters and reward weights.

---

[DPEPO: Diverse Parallel Exploration Policy Optimization for LLM-based Agents](http://arxiv.org/abs/2604.24320)

- DPEPO (Diverse Parallel Exploration Policy Optimization): introduces a reinforcement learning framework that enables LLM agents to interact with multiple environments simultaneously to build comprehensive environmental cognition.
- The framework utilizes a hierarchical reward scheme, incorporating trajectory-level success signals and diversity-driven step-level rewards to penalize redundant behaviors and promote broad exploration.
- By employing group-relative advantage computation, the approach eliminates the need for a separate critic model while achieving state-of-the-art performance and high sample efficiency on complex interactive benchmarks.

---

[BitRL: Reinforcement Learning with 1-bit Quantized Language Models for Resource-Constrained Edge Deployment](http://arxiv.org/abs/2604.24273)

- BitRL: introduces a framework for on-device reinforcement learning that utilizes a frozen 1-bit quantized BitNet backbone as a state encoder combined with lightweight trainable policy and value heads.
- The architecture leverages ternary weights {−1, 0, +1} within the BitNet backbone to achieve significant memory reduction and energy efficiency on resource-constrained edge hardware.
- BitRL addresses the value estimation bottleneck inherent in extreme quantization by employing PPO with conservative clipping and entropy regularization to maintain stable learning dynamics.

---

[RefEvo: Agentic Design with Co-Evolutionary Verification for Agile Reference Model Generation](http://arxiv.org/abs/2604.24218)

- RefEvo: introduces a hierarchical multi-agent framework that utilizes a Design Planner, Modeler Agent, Verifier Agent, and Dialectical Arbiter to automate the generation of reliable SystemC reference models.
- The framework employs a co-evolutionary verification mechanism where the Dialectical Arbiter resolves "Coupled Validation Failure" by iteratively refining both the design and the testbench against an anchored specification.
- Spec Anchoring Context Management optimizes token consumption by pinning immutable specifications while compressing historical interaction logs to prevent LLM catastrophic forgetting.

---

[Empowering Autonomous Debugging Agents with Efficient Dynamic Analysis](http://arxiv.org/abs/2604.24212)

- ADI: introduces a function-level debugging interface for LLMs that replaces inefficient line-by-line interaction with a structured Frame Lifetime Trace (FLT) and high-level navigational commands.
- The framework utilizes a FrameLifetimeTracer to generate on-demand, stateful execution summaries, enabling LLMs to perform precise root-cause analysis without exhausting computational budgets.
- By integrating ADI as a plug-and-play component, autonomous agents achieve significant performance gains on complex software engineering tasks while maintaining cost-efficiency.

---

[Agentic Witnessing: Pragmatic and Scalable TEE-Enabled Privacy-Preserving Auditing](http://arxiv.org/abs/2604.24203)

- Agentic Witnessing: introduces a privacy-preserving auditing framework that replaces static cryptographic proofs with dynamic, TEE-based LLM reasoning to verify unstructured semantic properties of proprietary datasets.
- The system utilizes a tripartite architecture consisting of a Verifier, a Prover, and an Auditor, where the Auditor operates within a TEE to perform semantic analysis via MCP while maintaining a cryptographically signed transcript hash chain.
- To ensure security and privacy, the framework enforces tokenized query budgets to prevent information leakage and provides both a lightweight Public Attestation and an encrypted Private Proof for audit integrity.

---

[Rewarding the Scientific Process: Process-Level Reward Modeling for Agentic Data Analysis](http://arxiv.org/abs/2604.24198)

- DataPRM: introduces an environment-aware generative process reward model that utilizes active environment interaction and a ternary reward strategy to supervise data analysis agents.
- The framework employs a tool-augmented architecture to perform multi-step verification, effectively detecting silent errors and distinguishing recoverable grounding errors from fatal mistakes.
- DataPRM utilizes a scalable pipeline for generating high-quality supervision data, significantly improving performance in both Test-Time Scaling and Reinforcement Learning settings for agentic data analysis.

---

[Dynamic Cyber Ranges](http://arxiv.org/abs/2604.24184)

- Dynamic Cyber Ranges: introduces a framework for evaluating LLM-driven agents in cyber range environments by deploying concurrent attacker and defender agents to create adversarial dynamics.
- The framework utilizes an APT Agent and a Defender Agent, both operating within a CAI Scaffold, to test defensive strategies like chokepoint, per-machine, and hostmanager deployments.
- Experiments demonstrate that LLM-driven defenders can significantly reduce attacker success rates, though effectiveness depends on infrastructure hardening and the security of the monitoring stack itself.

---

[Strategic Bidding in 6G Spectrum Auctions with Large Language Models](http://arxiv.org/abs/2604.24156)

- Strategic Bidding in 6G Spectrum Auctions with Large Language Models: introduces a framework where LLM-based Bidding Agents, Heuristic-based Bidding Agents, and Truthful Bidding Agents compete in a VCG Auction Mechanism managed by a Base Station (BS) Auctioneer, utilizing a Budget Management Module for adaptive decision-making.
- The framework enables UEs to perform context-aware bidding by processing historical auction data and budget constraints through an LLM module to optimize long-term utility.
- Simulation results demonstrate that LLM-based agents outperform traditional strategies in static-budget scenarios by pacing expenditures and sustaining participation in repeated 6G spectrum auctions.

---

[Leveraging Human Feedback for Semantically-Relevant Skill Discovery](http://arxiv.org/abs/2604.24127)

- SRSD (Semantically Relevant Skill Discovery): introduces a human-in-the-loop framework that leverages semantic labelling to guide the discovery of diverse and contextually relevant skills.
- The framework utilizes a Relevance Predictor to incorporate human feedback, alongside Distributional Critics to manage aleatoric uncertainty and mitigate value overestimation during skill training.
- SRSD employs an active sampling strategy to ensure balanced feedback across relevant semantic classes, demonstrating superior performance and scalability compared to traditional preference-based methods.

---

[New Convex Programming Technique for Nash Social Welfare and Scheduling](http://arxiv.org/abs/2604.24120)

- NSW and Scheduling Framework: introduces a novel convex programming relaxation for the weighted Nash social welfare problem that achieves an e^(1/e)-approximation via a rounding algorithm.
- The framework utilizes a compact linear program of polynomial size, avoiding the need for exponential-size configuration LPs or complex dual separation oracles.
- The approach extends to unrelated machine scheduling problems, providing simpler analyses and recovering best-known approximation ratios for minimizing Lq norms and weighted completion times.

---

[AgentVisor: Defending LLM Agents Against Prompt Injection via Semantic Virtualization](http://arxiv.org/abs/2604.24118)

- AgentVisor: introduces a virtualization-inspired defense framework that enforces semantic privilege separation between an untrusted Guest LLM Agent and a trusted Visor to mitigate prompt injection attacks.
- The framework utilizes an STI Protocol (Suitability, Taint, Integrity) to audit tool proposals and triggers a Semantic Exception to enable one-shot self-correction when security violations are detected.
- By treating the agent as an untrusted guest and mediating tool calls through a trusted hypervisor, AgentVisor achieves near-zero attack success rates while maintaining high utility in complex agentic workflows.

---

[An Analysis of the Coordination Gap between Joint and Modular Learning for Job Shop Scheduling with Transportation Resources](http://arxiv.org/abs/2604.24117)

- JSSPT Framework: introduces a multi-agent reinforcement learning approach that evaluates the coordination gap between joint and modular training for job-shop scheduling with transportation resources.
- The architecture utilizes a GNN-based job scheduler and an MLP-based AGV scheduler, both trained via MAPPO to optimize makespan under varying resource scarcity and temporal-dominance conditions.
- The research identifies that joint training provides superior performance in balanced operational regimes, while its advantages diminish in bottleneck environments where a single scheduling task dominates.

---

[Closing the Loop: A Software Framework for AI to Support Business Decision Making](http://arxiv.org/abs/2604.24116)

- Software Framework for AI-supported Business Decision Making: introduces a composable software framework that unifies causal inference models into a single interface to enable LLMs to orchestrate experiment analysis, effect estimation, and algorithmic decision-making.
- The framework utilizes ExperimentData, LinearModel, Policy, Delta Vectors, and a Dispatcher to resolve multiplicity in experimental designs and ensure statistically robust, computationally efficient insights for LLMs.
- By mapping various randomization strategies to a unified model and employing vectorization, the framework significantly reduces code complexity and memory usage compared to vanilla LLM-based implementations.

---

[Latency and Cost of Multi-Agent Intelligent Tutoring at Scale](http://arxiv.org/abs/2604.24110)

- ITAS (Intelligent Tutoring Agent System): introduces a spoke-and-wheel multi-agent architecture that utilizes parallel Video Agent, Code Agent, and Guidance Agent components, followed by a sequential Synthesizer Agent to generate coherent pedagogical responses.
- The framework evaluates latency and cost performance across three Google Vertex AI throughput tiers, identifying that the parallel-phase maximum effect significantly impacts end-to-end response times in multi-agent LLM pipelines.
- The research demonstrates that Priority PayGo provides the most stable latency for classroom-scale deployments, while Provisioned Throughput offers cost-efficiency only when traffic patterns are predictable and utilization is high.

---

[DataClaw: An Autonomous Data Agent with Instant Messaging Integration](http://arxiv.org/abs/2604.24067)

- DataClaw: introduces an autonomous data agent that integrates into instant messaging platforms to perform multi-step analytical workflows using a ReAct reasoning engine, a multi-tiered memory system, and a pluggable skill architecture.
- The framework utilizes a ReAct loop for auditable reasoning and a hot-loading skill mechanism to enable on-the-fly extensibility without requiring system restarts.
- DataClaw ensures data privacy and governance by executing all analytical tasks and storing artifacts locally on the user's machine or private server.

---

[Grounding Before Generalizing: How AI Differs from Humans in Causal Transfer](http://arxiv.org/abs/2604.24062)

- OpenLock framework: introduces a benchmark for evaluating causal structure transfer in LLMs and VLMs by comparing their interactive discovery performance against human baselines in Common Cause (CC) and Common Effect (CE) environments.
- The study reveals that while LLMs and VLMs excel at local causal search, they exhibit delayed transfer and rely on environmental grounding rather than genuine structural abstraction.
- Experimental results demonstrate that visual information often acts as a distractor for these models, and their learning dynamics lack the sudden insight-driven acceleration observed in human causal reasoning.

---

[AgenticCache: Cache-Driven Asynchronous Planning for Embodied AI Agents](http://arxiv.org/abs/2604.24039)

- AgenticCache: introduces a cache-driven planning framework that reuses frequent plan transitions to avoid per-step LLM calls in embodied AI agents.
- The framework utilizes a Runtime Cache for fast plan retrieval and an asynchronous Cache Updater to validate and refine cached entries using an LLM, ensuring adaptability in dynamic environments.
- By exploiting plan locality, AgenticCache significantly reduces inference latency and token usage while maintaining high task success rates across multi-agent embodied benchmarks.

---

[AgentPulse: A Continuous Multi-Signal Framework for Evaluating AI Agents in Deployment](http://arxiv.org/abs/2604.24038)

- AgentPulse: introduces a continuous evaluation framework that aggregates 18 real-time signals across GitHub, package registries, and social platforms to score AI agents in deployment.
- The framework utilizes a multi-layer NLP pipeline and a four-factor composite—Benchmark Performance, Adoption Signals, Community Sentiment, and Ecosystem Health—to provide a holistic view of agent performance beyond static benchmarks.
- AgentPulse validates its methodology through circularity-controlled tests, demonstrating that its composite factors effectively predict external adoption proxies like GitHub stars and Stack Overflow activity.

---

[From Skill Text to Skill Structure: The Scheduling-Structural-Logical Representation for Agent Skills](http://arxiv.org/abs/2604.24026)

- SSL (Scheduling-Structural-Logical) representation: introduces a structured, three-layer framework that disentangles skill-level scheduling signals, scene-level execution structure, and logic-level action evidence from text-heavy agent skill artifacts.
- The framework utilizes an LLM-based normalizer to map unstructured skill documents into a typed JSON graph, facilitating improved skill discovery and automated risk assessment for LLM agents.
- Experimental results demonstrate that SSL-augmented representations significantly outperform text-only baselines in both retrieval accuracy and the identification of operational risks in agent skills.

---

[QED: An Open-Source Multi-Agent System for Generating Mathematical Proofs on Open Problems](http://arxiv.org/abs/2604.24021)

- QED: introduces a multi-agent system designed to generate original, nontrivial mathematical proofs for open research problems by addressing seven specific failure modes identified in frontier LLMs.
- The architecture utilizes a multi-stage pipeline including Literature Survey, Prover Agents, Structural Verifier, Detailed Verifier, Selector Agent, Verdict Agent, Summary Agent, Decomposer, and Regulator to ensure logical consistency and proof integrity.
- QED employs a failure-mode-driven design that incorporates structured verification, multi-model parallel proving, and a decomposition mode to mitigate common LLM reasoning errors such as context contamination and citation hallucination.

---

[ClawdGo: Endogenous Security Awareness Training for Autonomous AI Agents](http://arxiv.org/abs/2604.24020)

- ClawdGo: introduces an endogenous security awareness training framework for autonomous AI agents that utilizes TLDT, ASAT, CSMA, SACP, L0 Axiom Set, L1 Skill Profile, L2 Episode Log, L3 Scenario Library, and ACP to build threat-recognition capabilities at inference time without model modification.
- The framework employs an ASAT self-play loop where the LLM acts as attacker, defender, and evaluator to reinforce threat modeling and defense reasoning through weakest-first curriculum scheduling.
- CSMA provides persistent memory across sessions via axiom crystallization, while SACP addresses the precision-recall tradeoff inherent in over-training autonomous agents.

---

[TCOD: Exploring Temporal Curriculum in On-Policy Distillation for Multi-turn Autonomous Agents](http://arxiv.org/abs/2604.24005)

- TCOD (Temporal Curriculum On-Policy Distillation): introduces a curriculum-based training framework that controls trajectory depth to mitigate Trajectory-Level KL Instability in multi-turn LLM agents.
- The framework utilizes two variants, Forward-to-Backward (TCOD-F2B) and Backward-to-Forward (TCOD-B2F), to progressively expose the student to longer interaction horizons while avoiding compounding errors.
- TCOD improves training stability and performance by decoupling trajectory collection and optimization through asynchronous processes and staleness-aware experience replay.

---

[IntentVLM: Open-Vocabulary Intention Recognition through Forward–Inverse Modeling with Video-Language Models](http://arxiv.org/abs/2604.24002)

- IntentVLM: introduces a two-stage cognitive framework for open-vocabulary intention recognition by decomposing the task into goal candidate generation and structured inference via selection.
- The framework utilizes a Goal Candidate Generator to propose potential intentions and an Intention Inference Module to rank and select the most consistent goal based on multimodal video-language evidence.
- By employing LoRA adapters on a Qwen3-VL backbone, the model achieves state-of-the-art performance on intention recognition benchmarks while maintaining generalization capabilities across complex instance-level tasks.

---

[EPM-RL: Reinforcement Learning for On-Premise Product Mapping in E-Commerce](http://arxiv.org/abs/2604.23993)

- EPM-RL: introduces a reinforcement learning framework that distills high-cost agentic reasoning into a compact, on-premise Nemotron-Nano-3-30B model using LoRA and GRPO.
- The framework utilizes three specialized LLM-based judge agents—Core Identity, Model-Identifier, and Variant-Conflict—to provide fine-grained reward signals during the reinforcement learning process.
- By combining parameter-efficient fine-tuning with structured reasoning traces and judge-based rewards, EPM-RL achieves high-quality product mapping without the latency and cost of inference-time agent orchestration.

---

[LLM-Guided Agentic Floor Plan Parsing for Accessible Indoor Navigation of Blind and Low-Vision People](http://arxiv.org/abs/2604.23970)

- Agentic RAG framework: introduces an agentic pipeline that converts architectural floor plan images into structured knowledge graphs to generate safe, accessible navigation instructions for BLV individuals.
- The system utilizes a multi-agent workflow comprising Parser-, Graph Builder-, Self-Critic-, Planner- and Safety Evaluator-agents to ensure robust spatial graph extraction and hazard-aware path planning.
- A three-tier RAG knowledge base integrates graph-based relational data, vector-based semantic embeddings, and visual grounding context to provide precise, landmark-enriched navigation guidance.

---

[GAMED.AI: A Hierarchical Multi-Agent Framework for Automated Educational Game Generation](http://arxiv.org/abs/2604.23947)

- GAMED.AI: introduces a hierarchical multi-agent framework that transforms instructor-provided questions into pedagogically grounded educational games using a LangGraph DAG, deterministic Quality Gates, and typed Pydantic schemas.
- The framework utilizes a modular game engine with self-contained React components and a dual-architecture state management system to support 15 distinct interaction mechanics.
- By separating generation and validation into six deterministic phases, the system achieves a 90% validation pass rate and a 73% token reduction compared to standard ReAct agent architectures.

---

[Constraint-Guided Multi-Agent Decompilation for Executable Binary Recovery](http://arxiv.org/abs/2604.23940)

- Agent4Decompile: introduces a multi-agent framework that transforms decompiled code into re-executable source through multi-level constraint-guided refinement using Decompiler, SyntaxAgent, CompilationAgent, ExecAgent, RefinementLoopOrchestrator, and ConstraintValidators.
- The framework employs a hierarchical validation strategy where specialized LLM agents iteratively repair code based on syntax, compilation, and execution feedback.
- Experimental results on 1,641 binaries demonstrate that this approach significantly improves re-executability by 18–28 percentage points compared to baseline methods.

---

#### 26th April 2026

[Agentic AI platforms for autonomous training and rule induction of human-human and virus-human protein-protein interactions](http://arxiv.org/abs/2604.23924)

- Agentic AI platforms for PPI: introduces two autonomous agentic AI platforms designed to perform end-to-end ML training and biological rule induction for human-human and virus-human protein-protein interactions.
- The first platform utilizes five agents—Data Collector, Data Verifier, Feature Embedder, Model Designer, and Executor—to autonomously train predictive ML models using protein-disjoint cross-fold validation.
- The second platform replaces the model designer and executor with a Rule Induction Agent to generate interpretable biological rules, which are cross-checked against SHAP-identified features from the predictive models.

---

[MarketBench: Evaluating AI Agents as Market Participants](http://arxiv.org/abs/2604.23897)

- MarketBench: introduces a benchmark for evaluating LLMs as market participants by assessing their ability to perform self-calibration and generate bids for task allocation.
- The framework utilizes a Calibration Module to elicit success probabilities and token estimates, which are then processed by an Auction Module to simulate procurement-based task routing.
- Experimental results demonstrate that while LLMs exhibit significant miscalibration, providing historical performance priors improves self-assessment and market-style coordination efficiency.

---

[OPTIMAS: An Intelligent Analytics-Informed Generative AI Framework for Performance Optimization](http://arxiv.org/abs/2604.23892)

- OPTIMAS: introduces a modular, multi-agent framework that automates HPC performance optimization by translating multi-source runtime diagnostics into actionable, evidence-driven code transformations.
- The framework utilizes a Profiling Agent, Analysis Agents, a Prompt Construction Agent, and an Evaluation Agent to create a closed-loop system that validates code correctness and performance gains.
- OPTIMAS employs Evidence-Aligned Reasoning (EAR) metrics to ensure that LLM-generated code edits are directly motivated by identified performance bottlenecks rather than generic code changes.

---

[ZenBrain: A Neuroscience-Inspired 7-Layer Memory Architecture for Autonomous AI Systems](http://arxiv.org/abs/2604.23878)

- ZenBrain: introduces a multi-layer memory architecture for AI agents that integrates fifteen neuroscience-inspired models into a unified system orchestrated by a MemoryCoordinator.
- The architecture utilizes seven distinct memory layers—working, short-term, episodic, semantic, procedural, core, and cross-context—to manage information lifecycle through consolidation, forgetting, and reconsolidation.
- ZenBrain incorporates six Predictive Memory Architecture components, including a NeuromodulatorEngine, ReconsolidationEngine, TripleCopyMemory, PriorityMap, StabilityProtector, and MetacognitiveMonitor, to govern memory dynamics and improve retrieval performance.

---

[ClawTrace: Cost-Aware Tracing for LLM Agent Skill Distillation](http://arxiv.org/abs/2604.23853)

- ClawTrace: introduces a cost-aware tracing platform that instruments LLM agent sessions via eight event hooks to generate compact TraceCard summaries for downstream distillation.
- CostCraft: utilizes TraceCards to distill agent trajectories into reusable skill patches, categorized as preserve, prune, or repair, to optimize agent performance and cost.
- The framework employs a three-way patch typology and conflict-aware merging to ensure that prune rules act as quality guardrails while repair rules address failure modes.

---


[DRACULA: Hunting for the Actions Users Want Deep Research Agents to Execute](http://arxiv.org/abs/2604.23815)

- DRACULA: introduces a large-scale dataset of user feedback on intermediate actions for Deep Research agents, comprising Action Generation Module, Action Selection Interface, MyScholarQA Agent, User Feedback Collector, Simulation Engine, and User History Memory.
- The framework enables the study of action predictability by leveraging User History Memory to improve the alignment of generated actions with user-specific preferences.
- The research demonstrates that while LLMs can reliably execute specified actions, predicting which intermediate actions users prefer remains a significant bottleneck that benefits from user-specific modeling.

---


[Scalable Production Scheduling: Linear Complexity via Unified Homogeneous Graphs](http://arxiv.org/abs/2604.23841)

- Unified Graph Framework: introduces a scalable RL approach for JSSP that utilizes feature-based homogenization to enable a homogeneous GIN backbone to process structurally heterogeneous graphs with linear complexity.
- The framework employs an actor-critic architecture trained via PPO, which leverages a structural saturation hypothesis to achieve zero-shot generalization across varying problem scales.
- By modeling machines as first-class entities in a sparse bipartite graph, the approach eliminates the quadratic edge complexity of traditional disjunctive formulations while maintaining strong relational inductive biases.

---

[JigsawRL: Assembling RL Pipelines for Efficient LLM Post-Training](http://arxiv.org/abs/2604.23838)

- JigsawRL: introduces a cost-efficient RL post-training framework that utilizes Sub-Stage Graph, Sub-Stage Multiplexing, Sub-Stage Merging, and a Look-ahead Heuristic to improve GPU utilization by multiplexing concurrent RL pipelines.
- The framework decomposes coarse-grained RL stages into fine-grained sub-stages to expose intra-stage and inter-worker imbalances, enabling dynamic resource allocation and sample migration across DP workers.
- JigsawRL achieves significant throughput improvements over synchronous and asynchronous baselines by co-scheduling complementary compute-bound and memory-bound sub-stages while maintaining moderate latency trade-offs.

---

[KISS Sorcar: A Stupidly-Simple General-Purpose and Software Engineering AI Assistant](http://arxiv.org/abs/2604.23822)

- KISS Agent Framework: introduces a layered, single-concern agent architecture designed to address common LLM failure modes in software engineering through a strict inheritance hierarchy.
- The framework utilizes a structured system prompt and a five-layer hierarchy, including KISS Agent, Relentless Agent, Sorcar Agent, Chat Sorcar Agent, and Worktree Sorcar Agent, to ensure robust, budget-tracked, and isolated task execution.
- Implemented as a VS Code extension, the system achieves high performance on Terminal Bench 2.0 by prioritizing output quality through self-validation and disciplined engineering practices over latency.

---


[Structural Enforcement of Goal Integrity in AI Agents via Separation-of-Powers Architecture](http://arxiv.org/abs/2604.23646)

- PEA (Policy–Execution–Authorization) Architecture: introduces a separation-of-powers design that enforces AI safety at the system level by decoupling intent generation, authorization, and execution into independent layers.
- The framework utilizes an IVL, ILT, Goal Drift Detection, and an OSG to ensure that all executed actions remain traceable to the originating user request and bounded by authorized capability constraints.
- By treating the Policy LLM as untrusted, the architecture converts the AI safety problem from a probabilistic behavioral question into a conditionally sound system property with formally stated boundaries.

---

[DLM: Unified Decision Language Models for Offline Multi-Agent Sequential Decision Making](http://arxiv.org/abs/2604.23557)

- DLM: introduces a scalable framework that reformulates multi-agent sequential decision-making as a dialogue-style sequence prediction problem to bridge the gap between LLMs and decentralized decision tasks.
- The framework utilizes a two-stage training pipeline consisting of SFT for initial domain alignment and GRPO for preference-based optimization to enhance robustness against OOD actions.
- By converting observations and actions into natural language dialogues, DLM enables centralized training with inter-agent context while supporting decentralized execution from local observations.

---



[EndoGov: A knowledge-governed multi-agent expert system for endometrial cancer risk stratification](http://arxiv.org/abs/2604.23802)

- EndoGov: introduces a two-tier multi-agent expert system that decomposes the risk stratification process into independent evidence extraction by specialist agents and deterministic guideline-governed decision control by a chair agent.
- The framework utilizes a Guideline-KG to support both hard-path deterministic overrides for high-priority clinical triggers and soft-path grey-zone reasoning for ambiguous cases.
- By separating perception from governance, the system ensures guideline compliance and auditability, effectively mitigating logic blind spots in multimodal EC risk stratification.

---

[ClawMark: A Living-World Benchmark for Multi-Turn, Multi-Day, Multimodal Coworker Agents](http://arxiv.org/abs/2604.23781)

- ClawMark: introduces a benchmark for evaluating persistent coworker agents across multi-turn, multi-day workflows with evolving environments and raw multimodal evidence.
- The framework utilizes a stateful sandboxed service environment, including filesystem, email, calendar, knowledge base, and spreadsheet, to test agent adaptation to exogenous updates.
- Scoring is performed by a deterministic checker system that inspects post-turn service states, eliminating the need for LLM-as-judge protocols.

---

[PageGuide: Browser extension to assist users in navigating a webpage and locating information](http://arxiv.org/abs/2604.23772)

- PageGuide: introduces a browser extension that grounds LLM answers directly in the HTML DOM via visual overlays to improve user verifiability and control.
- The framework utilizes a Router to dispatch queries to specialized handlers for finding information, guiding multi-step tasks, or hiding distracting content.
- By coupling text outputs with in-situ DOM mutations and user-in-the-loop feedback, the system enables transparent, verifiable web interaction compared to opaque autonomous agents.

---

[Agentic Fusion of Large Atomic and Language Models to Accelerate Materials Discovery](http://arxiv.org/abs/2604.23758)

- ElementsClaw: introduces an agentic framework that synergizes Large Atomic Models with LLMs to autonomously orchestrate the materials discovery process.
- The system utilizes specialized LAM tools, including Elements-T, Elements-C, Elements-E, and Elements-G, to perform high-fidelity numerical computations while leveraging LLMs for semantic reasoning and literature-based synthesis.
- ElementsClaw demonstrates its efficacy by screening millions of crystals and successfully guiding the experimental synthesis of novel superconductors with high physical fidelity.

---

[Prism-Reranker: Beyond Relevance Scoring — Jointly Producing Contributions and Evidence for Agentic Retrieval](http://arxiv.org/abs/2604.23734)

- Prism-Reranker: introduces a reranker family that jointly emits a calibrated relevance score, a contribution statement, and a self-contained evidence passage in a single forward pass using a Qwen3.5 backbone.
- The framework utilizes a hybrid training objective combining point-wise distillation from a commercial rerank API with supervised fine-tuning on structured text targets gated by an LLM-as-Judge ensemble.
- Prism-Reranker optimizes agentic retrieval by providing actionable planning signals and context-compressed evidence, effectively reducing token consumption and hallucination risks for downstream LLMs.

---

[ESIA: An Energy-Based Spatiotemporal Interaction-Aware Framework for Pedestrian Intention Prediction](http://arxiv.org/abs/2604.23728)

- ESIA (Energy-based Spatiotemporal Interaction-Aware framework): introduces a structured CRF-based paradigm that models pedestrian intention prediction as an energy minimization problem by explicitly decoupling individual, social, and environmental factors.
- The framework utilizes PNFE and ENFE for feature extraction, while PPIL and PEIL capture complex interactions through MHA mechanisms to ensure global behavioral consistency.
- To resolve logical contradictions during inference, the model employs a U-SSA algorithm that leverages high-confidence unary priors to efficiently navigate the energy landscape and achieve robust, interpretable predictions.

---

[Information-Theoretic Measures in AI: A Practical Decision Guide](http://arxiv.org/abs/2604.23716)

- ITM Decision Framework: introduces a structured guide for selecting and applying seven information-theoretic measures across AI/ML and agent-based research domains.
- The framework categorizes measures into two families, distinguishing between core learning metrics and complex agent-level causal measures, while providing standardized estimator recommendations and guardrails.
- It operationalizes these guidelines through a measure-selection flowchart and a master decision table to prevent common misuses and ensure rigorous inferential claims in AI research.

---

[SPORE: Efficient and Training-Free Privacy Extraction Attack on LLMs via Inference-Time Hybrid Probing](http://arxiv.org/abs/2604.23711)

- SPORE: introduces a training-free privacy extraction attack that leverages adversarial input and inference-time hybrid probing to recover PII from LLM agent memory.
- The framework utilizes a shadow encryption paradigm to obfuscate PII, enabling efficient recovery in both black-box and gray-box settings through candidate space construction and token-level filtering.
- Experimental results demonstrate that SPORE achieves high attack success rates and low query costs across multiple frontier LLMs while remaining robust against existing safety alignment and detection mechanisms.

---

[Directional Alignment and Narrative Agency in Human–LLM Co-Writing](http://arxiv.org/abs/2604.23676)

- Human–LLM Co-Writing Framework: introduces a controlled dyadic storytelling task to quantify affective alignment and narrative agency through sentiment and information-theoretic modeling.
- The framework utilizes sentiment concept vector projection and surprisal-based metrics to evaluate how human and LLM agents influence narrative progression and emotional coordination.
- Empirical results demonstrate an asymmetric division of labor where humans drive narrative innovation while LLMs act as adaptive amplifiers that sustain coherence and emotional alignment.

---

[Vibe Medicine: Redefining Biomedical Research Through Human-AI Co-Work](http://arxiv.org/abs/2604.23674)

- Vibe Medicine: introduces a human-AI co-work paradigm where researchers direct skill-augmented AI agents to execute complex, multi-step biomedical workflows using LLMs, agentic frameworks, medical skills, and biomedical tools and data.
- The infrastructure relies on a modular architecture where specialized medical skills are composed into pipelines to perform tasks ranging from literature synthesis and variant interpretation to drug discovery and clinical trial design.
- The paradigm shifts the human role to that of a research director, emphasizing the need for human oversight to mitigate risks such as hallucination, data privacy concerns, and over-reliance on agent-generated outputs.

---

[Strategically Robust Aggregative Games](http://arxiv.org/abs/2604.23669)

- Strategically Robust Wardrop Equilibrium framework: introduces a novel equilibrium concept for aggregative games where agents protect themselves against worst-case aggregate behavior within an optimal-transport-based ambiguity set.
- The framework reformulates the infinite-dimensional robust optimization problem into a standard convex aggregative game using augmented action spaces and duality, enabling efficient computation via proximal best response algorithms.
- The research demonstrates a "coordination-via-robustification" effect in electric vehicle charging, where strategic robustness improves individual costs and can drive the price of anarchy to 1.

---

[GraphPlanner: Graph Memory-Augmented Agentic Routing for Multi-Agent LLMs](http://arxiv.org/abs/2604.23626)

- GraphPlanner: introduces a heterogeneous graph memory-augmented agentic routing framework that formulates workflow generation as a Markov Decision Process to optimize multi-agent LLM collaboration.
- The framework utilizes GARNet to integrate current workflow memory and historical interaction traces, enabling adaptive routing decisions through a learned policy optimized by PPO.
- GraphPlanner supports both inductive and transductive inference, demonstrating robust generalization to unseen tasks and LLMs while maintaining computational efficiency.

---

[Thinking Like a Clinician: A Cognitive AI Agent for Clinical Diagnosis via Panoramic Profiling and Adversarial Debate](http://arxiv.org/abs/2604.23605)

- DxChain: introduces a cognitive-aligned reasoning framework that transforms clinical diagnosis into an iterative, stateful process by mirroring clinician trajectories through panoramic profiling, strategic navigation, and dialectical verification.
- The framework utilizes a Profile-Then-Plan paradigm to mitigate cold-start hallucinations, a Medical Tree-of-Thoughts (Med-ToT) algorithm for look-ahead planning, and an "Angel-Devil" adversarial debate mechanism to resolve evidence conflicts.
- Evaluated on MIMIC-IV-Ext datasets, DxChain achieves state-of-the-art performance in diagnostic accuracy and logical consistency by shifting from linear LLM inference to active, stateful clinical simulation.

---

[CineAGI: Character-Consistent Movie Creation through LLM-Orchestrated Multi-Modal Generation and Cross-Scene Integration](http://arxiv.org/abs/2604.23579)

- CineAGI: introduces a hierarchical movie generation framework that decomposes complex production tasks through specialized LLM-orchestrated multi-agent coordination.
- The framework utilizes a decoupled character-centric pipeline to maintain identity consistency across diverse scenes while enabling flexible multi-character composition.
- CineAGI achieves significant improvements in narrative coherence and character consistency by integrating specialized LLM agents for planning and targeted synthesis models for audiovisual alignment.

---

[MetaGAI: A Large-Scale and High-Quality Benchmark for Generative AI Model and Data Card Generation](http://arxiv.org/abs/2604.23539)

- MetaGAI: introduces a large-scale benchmark for automated Model and Data Card generation using a multi-agent framework comprising Retriever-, Generator- and Editor-agents.
- The framework utilizes multi-source triangulation of academic papers, GitHub repositories, and Hugging Face artifacts to synthesize high-fidelity documentation.
- Empirical analysis demonstrates that sparse Mixture-of-Experts architectures achieve superior cost-quality efficiency in generating structured documentation.

---

[Large Language Model based Interactive Decision-Making for Autonomous Driving](http://arxiv.org/abs/2604.23513)

- LLM-based Interactive Autonomous Driving Framework: introduces a framework that integrates Object-Process Methodology for semantic scene modeling with an LLM-driven decision module to improve interactive intelligence in mixed-traffic scenarios.
- The framework utilizes OPM to transform low-level perceptual data into structured object-process-relation representations, which serve as inputs for the LLM to perform intent-aware decision-making and trajectory optimization.
- The system closes the interaction loop by translating autonomous driving decisions into natural language messages broadcast via an external Human-Machine Interface, enhancing transparency and coordination with human road users.

---

[Breaking the Secret: Economic Interventions for Combating Collusion in Embodied Multi-Agent Systems](http://arxiv.org/abs/2604.23511)

- Mutagenic Incentive Intervention Framework: introduces a proactive defense mechanism that reshapes the payoff structure of LLM-based embodied agents to render collusion inherently unstable and economically irrational.
- The framework utilizes a reporting-and-penalty mechanism where agents are incentivized to defect from collusion by receiving rewards funded by the confiscated honesty deposits of identified colluders.
- To ensure robustness, the system incorporates cryptographic anonymity via ring signatures and automated, trustless fund management through smart contracts, effectively preventing retaliation and financial manipulation.

---

[Agentic Adversarial Rewriting Exposes Architectural Vulnerabilities in Black-Box NLP Pipelines](http://arxiv.org/abs/2604.23483)

- Agentic Adversarial Rewriting framework: introduces a two-agent system that exploits black-box NLP pipelines through iterative semantic rewriting under strict query budgets.
- The framework utilizes an Attacker Agent and a Prompt Optimization Agent to navigate the semantic perturbation space without requiring gradient access or fine-tuning.
- The research identifies a vulnerability spectrum in multi-stage NLP pipelines, demonstrating that architectural properties like retrieval mechanisms significantly influence susceptibility to adversarial attacks.

---

[Towards Agentic Test-Driven Quality Assurance for 6G Networks](http://arxiv.org/abs/2604.23285)

- Agentic, Intent-Driven E2E Orchestration Framework: introduces an agentic orchestration architecture that integrates intent co-creation with a Test-Driven Quality Assurance paradigm to ensure proactive SLA compliance in 6G networks.
- The framework utilizes a dual-path agentic approach, including Intent-to-Actions- and Quality and SLO/SLA Specs-agents, to decompose high-level user intents into deterministic, standards-aligned technical specifications.
- By leveraging a TMF-aligned knowledge representation and MCP-enabled tool access, the system enables autonomous agents to perform graph-based reasoning and continuous validation of network services.

---

#### 25th April 2026


[RAT: RunAnyThing via Fully Automated Environment Configuration](http://arxiv.org/abs/2604.23190)

- RAT (RunAnyThing): introduces a modular, language-agnostic framework for fully automated environment configuration that integrates Language-Agnostic Abstraction, ImageRetriever, Environment Configuration Planning, Specialized Toolset, Robust Sandbox Generation, and Long-term Expertise Accumulation.
- The framework employs an LLM-driven multi-stage pipeline to resolve complex repository dependencies and automate the construction of executable environments for autonomous code agents.
- To rigorously evaluate performance, the authors introduce RATBench, a large-scale multilingual benchmark comprising over 2,000 real-world repositories, demonstrating that RAT achieves state-of-the-art environment setup success rates.

---


[Escher-Loop: Mutual Evolution by Closed-Loop Self-Referential Optimization](http://arxiv.org/abs/2604.23472)

- Escher-Loop: introduces a closed-loop framework that operationalizes the mutual evolution of Task Agent Population and Optimizer Agent Population, where the latter recursively refines both task solutions and itself.
- The framework utilizes a Dynamic Benchmarking Mechanism that reuses empirical scores from Task Execution as relative win-loss signals to update the Elo Rating System of the Optimizer Agent Population without additional overhead.
- By maintaining a MAP-Elites Archive, the system preserves behavioral diversity and fitness, enabling the autonomous emergence of sophisticated optimization strategies that outperform static handcrafted baselines.

---

[Architecture Matters for Multi-Agent Security](http://arxiv.org/abs/2604.23459)

- Multi-Agent System (MAS) Architecture Framework: introduces an empirical study demonstrating that architectural design choices in multi-agent systems, such as role specialization, communication topology, and memory visibility, significantly impact security by fragmenting safety reasoning and expanding attack surfaces.
- The research reveals that multi-agent architectures are often more vulnerable than standalone LLMs, as task decomposition can dilute harmful signals and bypass safety training, even while maintaining or improving benign task performance.
- The study highlights that security-performance tradeoffs are highly scenario- and model-dependent, necessitating per-deployment adversarial evaluation rather than relying on component-level safety assessments.

---

[CUJBench: Benchmarking LLM-Agent on Cross-Modal Failure Diagnosis from Browser to Backend](http://arxiv.org/abs/2604.23455)

- CUJBench: introduces a diagnostic benchmark that couples browser-visible failure evidence with backend observability to evaluate LLM agents on cross-modal root cause analysis.
- The framework utilizes a snapshot-based methodology to ensure reproducibility, employing a multi-agent review loop to curate 87 deterministic failure scenarios across five fault families.
- Evaluation of six frontier LLMs reveals that cross-modal synthesis remains a primary bottleneck, as agents often retrieve decisive evidence but fail to correctly attribute it to the root cause.

---


[CODA: Coordination via On-Policy Diffusion for Multi-Agent Offline Reinforcement Learning](http://arxiv.org/abs/2604.23308)

- CODA (Coordination via On-Policy Diffusion for Multi-Agent Offline Reinforcement Learning): introduces a trajectory-level data augmentation method that uses diffusion models conditioned on the current joint policy to restore endogenous co-adaptation in offline MARL.
- The framework employs a centralized diffusion backbone to generate synthetic trajectories that are reweighted toward the current joint policy, effectively mitigating coordination failures caused by static offline datasets.
- CODA is algorithm-agnostic and can be integrated with various offline MARL pipelines to improve coordination by ensuring that synthetic experience reflects the evolving behaviors of agents during training.

---


[AI Safety Training Can be Clinically Harmful](http://arxiv.org/abs/2604.23445)

- Five-Axis Evaluation Framework: introduces a comprehensive clinical evaluation methodology for LLMs in mental health, identifying systematic failures where RLHF safety alignment disrupts essential therapeutic mechanisms.
- The framework operationalizes clinical safety through five distinct axes—protocol fidelity, hallucination risk, behavioral consistency, crisis safety, and demographic robustness—to address the gap between general-purpose conversational quality and clinical requirements.
- Empirical validation across Prolonged Exposure and Cognitive Behavioral Therapy demonstrates that current safety-aligned LLMs often exhibit a "crisis cliff," where performance collapses under high-severity scenarios due to inappropriate safety-motivated interventions.

---

[SoccerRef-Agents: Multi-Agent System for Automated Soccer Refereeing](http://arxiv.org/abs/2604.23392)

- SoccerRef-Agents: introduces a multi-agent framework that mimics professional officiating teams by integrating Video Agent, Rule Agent, Case Agent, Context Agent, and Chief Referee Agent to provide explainable soccer refereeing decisions.
- The system utilizes a cross-modal RAG mechanism to bridge visual perception with regulatory texts from RefKnowledgeDB, ensuring decisions are legally grounded and factually accurate.
- Evaluations on the SoccerRefBench benchmark demonstrate that the framework significantly outperforms general-purpose LLMs in both decision accuracy and the quality of legally grounded explanations.

---

[Ghost in the Agent: Redefining Information Flow Tracking for LLM Agents](http://arxiv.org/abs/2604.23374)

- NeuroTaint: introduces a provenance-oriented offline auditing framework for LLM agents that reconstructs execution lineage to detect explicit content propagation, implicit control influence, and asynchronous provenance reuse.
- The framework utilizes a Dynamic Context Provenance Graph (DCPG) to persist taint labels across memory boundaries and session restarts, enabling the attribution of delayed or transformed information flows.
- NeuroTaint employs a hybrid semantic tracker for explicit content detection and a sink-driven causal analyzer for implicit control influence, significantly outperforming traditional IFC-style baselines on the TaintBench benchmark.

---

[GSAR: Typed Grounding for Hallucination Detection and Recovery in Multi-Agent LLMs](http://arxiv.org/abs/2604.23366)

- GSAR (Grounding-Stratified Adaptive Replanning): introduces a grounding-evaluation and replanning framework for multi-agent LLMs that partitions claims into a four-way typology and couples evidence-typed weighted scoring to a three-tier decision function.
- The framework utilizes an asymmetric contradiction penalty to prevent score inflation and employs a bounded replanning loop to manage compute costs during incident investigation.
- Empirical evaluation across multiple LLM judges demonstrates that the framework's structural design choices, including evidence-type weighting and the complementary claim class, significantly improve grounding reliability and decision efficiency.

---

[LEGO: An LLM Skill-Based Front-End Design Generation Platform](http://arxiv.org/abs/2604.23355)

- LEGO: introduces a unified skill-based platform for digital front-end design that decomposes the workflow into six independent steps and utilizes a plug-and-play architecture for RTL generation.
- The framework leverages Circuit Skill Builder to automate the extraction of reusable skills from open-source projects and employs Agent Skill RAG for efficient, submillisecond retrieval of design and debugging knowledge.
- LEGO implements a three-layer hierarchical architecture comprising Templates, Step Skills, and atomic Circuit Skills to enable flexible, modular, and extensible RTL design automation.

---

[From Stateless Queries to Autonomous Actions: A Layered Security Framework for Agentic AI Systems](http://arxiv.org/abs/2604.23338)

- LASM (Layered Attack Surface Model): introduces a seven-layer security framework that maps agentic AI threats to specific architectural components including Foundation Model, Planning and Reasoning Module, Memory System, Tool Execution Layer, Multi-Agent Interface, Orchestration and Environment, and Governance.
- The framework incorporates an orthogonal attack temporality dimension (T1–T4) to categorize threats based on the installation-to-execution gap, highlighting that high-layer, slow-burn attacks are currently under-studied.
- The paper identifies five critical research gaps and proposes the Agent Bill of Materials (ABOM) to address supply-chain security and accountability in complex agentic systems.

---

[MMEB-V3: Measuring the Performance Gaps of Omni-Modality Embedding Models](http://arxiv.org/abs/2604.23321)

- MMEB-V3: introduces a comprehensive benchmark for evaluating full-modality embeddings across text, image, video, audio, and agent-centric scenarios using MMEB-V3, OmniSET, Audio Tasks, Text Tasks, Agent Tasks, and a Shared Candidate Pool.
- The framework utilizes OmniSET to disentangle semantic content from modality effects, enabling a systematic diagnostic analysis of instruction-conditioned embedding behavior.
- Experimental results reveal that current multimodal embeddings struggle to reliably enforce explicit modality constraints, often exhibiting significant cross-modal asymmetry and query-modality bias.

---

[Proteus: Shapeshifting Desktop Visualizations for Mobile via Multi-level Intelligent Adaptation](http://arxiv.org/abs/2604.23299)

- Proteus: introduces a multi-agent LLM-driven framework that automates the adaptation of desktop visualizations for mobile devices by systematically applying hierarchical transformation operators.
- The system utilizes a multi-level design space that propagates constraints from global topology to reference frames and individual visual elements to ensure semantic fidelity and readability.
- Proteus employs a collaborative team of specialized agents—including semantic parser-, data extraction-, design planner-, frontend engineering- and visual critic-agents—to iteratively refine visualizations for mobile consumption.

---

[Revisable by Design: A Theory of Streaming LLM Agent Execution](http://arxiv.org/abs/2604.23283)

- Revision Absorber: introduces a reactive algorithm that enables LLM agents to absorb mid-execution user revisions by identifying the earliest conflicting action and performing targeted compensation and re-planning.
- The framework formalizes agent execution as a stream of events and classifies actions into a reversibility taxonomy to determine the structural cost of adapting to user-injected changes.
- Experimental results on StreamBench demonstrate that the Revision Absorber achieves quality comparable to full-restart baselines while significantly reducing wasted computational steps.

---

[Bridging the Pose-Semantic Gap: A Cascade Framework for Text-Based Person Anomaly Search](http://arxiv.org/abs/2604.23282)

- SSDC (Structure-Semantic Decoupled Cascade): introduces a coarse-to-fine framework that bridges the Pose-Semantic Gap in text-based person anomaly search by decoupling structural filtering from multi-agent semantic verification.
- The framework utilizes a Structure-Aware Coarse Retriever for high-speed candidate filtering, followed by a Detective Squad that employs a Detective Agent, an Analyst Agent, and a Writer Agent to perform iterative, fine-grained semantic reasoning.
- By integrating an Adaptive Fusion Mechanism, the system balances retrieval efficiency with semantic precision, achieving state-of-the-art performance on the PAB benchmark.

---

[AI Identity: Standards, Gaps, and Research Directions for AI Agents](http://arxiv.org/abs/2604.23280)

- AI Identity Framework: introduces a three-layer model for AI agent identity, comprising a Declaration Layer, an Observation Layer, and a Confidence Layer, to manage the continuous relationship between agent claims and actual behavior.
- The framework addresses the structural asymmetry between human and AI identity by treating identity as a probabilistic, time-varying estimate rather than a static, binary credential.
- This report identifies five critical structural gaps in current identity infrastructure, including semantic intent verification and recursive delegation accountability, and proposes a research agenda to achieve ecologically sustainable and secure agent governance.

---

[Active Inference: A Method for Phenotyping Agency in AI Systems?](http://arxiv.org/abs/2604.23278)

- Active Inference framework: introduces a computational method for phenotyping AI agency by mapping intentionality, rationality, and explainability to the components of a variational generative model.
- The paper utilizes empowerment as an operational metric to distinguish between zero-, intermediate-, and high-agency phenotypes within a T-maze decision-making paradigm.
- The authors argue that as AI agents increase in agency, governance strategies must transition from external structural constraints to internal modulation of prior preferences.

---

[CAP-CoT: Cycle Adversarial Prompt for Improving Chain of Thoughts in LLM Reasoning](http://arxiv.org/abs/2604.23270)

- CAP-CoT: introduces a cycle-based adversarial prompt optimization framework that strengthens LLM reasoning through adaptive contrast between correct and erroneous chains.
- The framework utilizes a Solver Agent, an Adversarial Challenger Agent, and a Feedback Agent to iteratively refine prompts, improving both reasoning accuracy and robustness.
- By generating targeted hard negatives and providing step-aligned feedback, CAP-CoT systematically discovers and repairs reasoning vulnerabilities while maintaining a single-model inference setup.

---

[PrivacyAssist: A User-Centric Agent Framework for Detecting Privacy Inconsistencies in Android Apps](http://arxiv.org/abs/2604.23248)

- PrivacyAssist: introduces a multi-agent platform that detects inconsistencies between runtime-granted permissions and declared data practices in Android apps using Agent-1, Agent-2, Kafka, MongoDB, RAG, VectorDB, Summarization Module, and Llama-3-8B LLM.
- The framework employs a client-server architecture where Agent-1 monitors app permissions on-device, while Agent-2 performs server-side analysis using RAG to provide concise, user-oriented privacy warnings.
- PrivacyAssist mitigates LLM hallucinations and token constraints by utilizing a summarization module and an external database to ground its reasoning in verified Android permission and data safety definitions.

---

[Discovering Agentic Safety Specifications from 1-Bit Danger Signals](http://arxiv.org/abs/2604.23210)

- EPO-Safe (Experiential Prompt Optimization for Safe Agents): introduces an iterative framework where an LLM agent discovers hidden safety objectives from sparse 1-bit danger signals by evolving a natural language specification through reflection.
- The framework utilizes a four-phase experiential loop—Attempt, Simulate, Reflect, and Consolidate—to transform binary feedback into human-readable, auditable behavioral rules without requiring gradient-based model updates.
- By decoupling safety reflection from reward optimization, EPO-Safe prevents reward hacking and enables LLMs to perform few-shot safety rule induction in complex environments.

---

[StoryTR: Narrative-Centric Video Temporal Retrieval with Theory of Mind Reasoning](http://arxiv.org/abs/2604.23198)

- StoryTR: introduces a benchmark and an Agentic Data Pipeline that enables LLMs to perform narrative-centric video retrieval by distilling Theory of Mind reasoning capabilities into smaller models.
- The framework utilizes a Clipper Agent for fine-grained multimodal perception and a Self-QA Agent to synthesize training data with three-tier ToM reasoning chains, including intent decoding, narrative reasoning, and boundary localization.
- Experimental results demonstrate that the 7B Shorts-Moment model, trained on ToM-guided data, significantly outperforms larger baselines, validating that cognitive depth in reasoning is more critical than parameter scale for narrative understanding.

---

[AnalogRetriever: Learning Cross-Modal Representations for Analog Circuit Retrieval](http://arxiv.org/abs/2604.23195)

- AnalogRetriever: introduces a unified tri-modal retrieval framework that maps functional text descriptions, schematic images, and SPICE netlists into a shared semantic embedding space using modality-specific encoders and curriculum contrastive learning.
- The framework utilizes a port-aware Relational Graph Convolutional Network (RGCN) to capture structural semantics of SPICE netlists, enabling precise cross-modal retrieval across text, schematics, and code.
- Integrated into the AnalogCoder agentic framework, AnalogRetriever improves LLM functional correctness by providing topologically accurate circuit references via retrieval-augmented generation.

---

[From Coarse to Fine: Self-Adaptive Hierarchical Planning for LLM Agents](http://arxiv.org/abs/2604.23194)

- AdaPlan-H: introduces a self-adaptive hierarchical planning mechanism that mimics human cognitive strategies by generating plans with varying granularity based on task complexity.
- The framework utilizes a two-stage optimization process, incorporating imitation learning for initialization and DPO training to refine hierarchical plan levels and quality.
- By dynamically adjusting planning granularity, the approach improves task execution success rates and efficiency while mitigating overplanning in LLM-based agents.

---

[Cooperative Informative Sensing for Monitoring Dynamic Indoor Environments via Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2604.23179)

- MARL framework: introduces a decentralized multi-robot monitoring system that optimizes motion to improve human-centric monitoring accuracy under partial observability using Set-based Observation Encoding, Dual-Stage Recurrent Interaction Memory, and a Centralized Critic.
- The architecture utilizes permutation-invariant attention to handle variable-sized human observations and a dual-stage GRU structure to enable scalable inter-robot coordination without explicit tracking.
- The approach demonstrates robust performance across diverse indoor monitoring tasks and supports cost-effective hybrid integration with existing fixed sensing infrastructure.

---

[PhySE: A Psychological Framework for Real-Time AR-LLM Social Engineering Attacks](http://arxiv.org/abs/2604.23148)

- PhySE: introduces a real-time AR-LLM social engineering framework that integrates VLM-based social-context training with an adaptive psychological agent to enable theory-grounded strategy control.
- The framework utilizes a VLM-based social-context training component to minimize cold-start latency and an adaptive psychological agent to dynamically route interaction strategies based on latent trust states.
- PhySE improves conversational realism and attack effectiveness by replacing static scripts with a theory-grounded routing mechanism that adjusts to turn-level interaction signals.

---

[UNSEEN: A Cross-Stack LLM Unlearning Defense against AR-LLM Social Engineering Attacks](http://arxiv.org/abs/2604.23141)

- UNSEEN (A Cross-Stack LLM Unlearning Defense): introduces a coordinated defense architecture that integrates AR ACL, F-RMU, and Agent Guardrails to mitigate AR-LLM-based social engineering attacks.
- The framework employs F-RMU to perform targeted unlearning of sensitive identity concepts within the Multimodal LLM while preserving general utility.
- UNSEEN provides a cross-stack security pipeline that constrains sensing, inference, and interaction to prevent unauthorized profile extraction and persuasive phishing.

---

[GreenDyGNN: Runtime-Adaptive Energy-Efficient Communication for Distributed GNN Training](http://arxiv.org/abs/2604.23139)

- GreenDyGNN: introduces a runtime-adaptive framework for distributed GNN training that optimizes energy efficiency by dynamically adjusting cache rebuild windows and per-owner cache allocations using a reinforcement learning agent.
- The system utilizes an asynchronous double-buffered pipeline to decouple cache management from the training loop, ensuring that runtime adaptation occurs without stalling GPU computation.
- By employing a Double-DQN agent trained via sim-to-real transfer with domain randomization, the framework effectively mitigates energy waste caused by network congestion and remote feature fetch latencies.

---

[No Test Cases, No Problem: Distillation-Driven Code Generation for Scientific Workflows](http://arxiv.org/abs/2604.23106)

- MOSAIC (Multi-agent framework for scientific code generation): introduces a training-free, multi-agent framework that enables scientific code generation without I/O supervision by utilizing a Bucketing Module, Teacher Module, Self-Reflection Agent, Student Module, Rationale Agent, Consolidated Context Window (CCW), Coding Agent, and Debugger Agent.
- The framework employs a student-teacher knowledge distillation process to ground code generation in domain-specific rationales, effectively decoupling semantic grounding from syntactic grounding.
- By implementing a Consolidated Context Window (CCW), the framework maintains reasoning coherence across interdependent subproblems while mitigating hallucinations in LLMs.

---

[From Language to Logic: Bridging LLMs &amp; Formal Representations for RTL Assertion Generation](http://arxiv.org/abs/2604.23100)

- ProofLoop: introduces a tool-augmented ReAct agent that automates SystemVerilog Assertion generation by integrating AST-based retrieval and iterative formal verification feedback.
- The framework utilizes an AST Parser, Embedding Model, and VectorDB to synthesize design context, while the LLM employs a ReAct agent to interact with JasperGold for structural analysis and solver-in-the-loop refinement.
- By leveraging formal proof feedback to iteratively correct syntax and functional errors, the system achieves high correctness rates on complex, multi-module hardware designs.

---

[Code Broker: A Multi-Agent System for Automated Code Quality Assessment](http://arxiv.org/abs/2604.23088)

- Code Broker: introduces a hierarchical multi-agent system that leverages LLMs and static analysis to automate code quality assessment across correctness, security, style, and maintainability dimensions.
- The architecture utilizes a root Report Generator to coordinate a Sequential Pipeline Agent, which dispatches parallel Correctness-, Style-, and Description-agents before an Improvement Recommender synthesizes the final output.
- The system integrates Pylint as a deterministic tool to ground LLM-based reasoning, while employing asynchronous execution and lightweight session memory to enhance robustness and context retention.

---

[Usable Agent Discovery for Decentralized AI Systems](http://arxiv.org/abs/2604.23080)

- Decentralized Agent Discovery Framework: introduces a model for evaluating agent discovery in distributed systems by accounting for both node-level churn and agent-level lifecycle dynamics.
- The framework compares structured (Kademlia) and unstructured (Cyclon+Vicinity) overlays across four distinct operating regimes to determine their impact on routing efficiency, resilience, and service readiness.
- It utilizes a useful availability metric to assess whether discovered agents can provide services within specific latency constraints, revealing that structured and unstructured approaches occupy different performance regimes.

---

#### 24th April 2026

[Agentic World Modeling: Foundations, Capabilities, Laws, and Beyond](http://arxiv.org/abs/2604.22748)

- Agentic World Modeling: introduces a capability-based taxonomy for world models, organizing them into L1 Predictor, L2 Simulator, and L3 Evolver across four governing-law regimes.
- The framework defines world models by their ability to support decision-making through local prediction, multi-step simulation, and evidence-driven model revision.
- The paper provides architectural guidance and evaluation principles to bridge isolated research communities in embodied AI, language agents, and AI for science.

---


[A dataset of early blockchain-registered AI agents on Ethereum](http://arxiv.org/abs/2604.22652)

- ERC-8004: introduces a structured dataset of the first 10,000 AI agents registered on Ethereum, integrating on-chain identity, reputation, and off-chain metadata.
- The framework utilizes three parallel pipelines—identity extraction, metadata resolution, and reputation extraction—to normalize fragmented blockchain data into a relational schema.
- This dataset provides a reproducible baseline for empirical research into the emerging agentic economy, trust infrastructure, and multi-agent interoperability on Ethereum.

---


[ATRS: Adaptive Trajectory Re-splitting via a Shared Neural Policy for Parallel Optimization](http://arxiv.org/abs/2604.22715)

- ATRS (Adaptive Trajectory Re-splitting via a Shared Neural Policy): introduces a framework that embeds a shared Deep Reinforcement Learning policy into a parallel ADMM loop to adaptively re-split stagnating trajectory segments for optimized motion planning.
- The framework utilizes a Multi-Agent Shared-Policy Markov Decision Process where trajectory segments act as homogeneous agents sharing a unified policy network to achieve size-invariant, zero-shot generalization across environments.
- A Confidence-Based Election mechanism ensures solver stability by selecting only the most stagnating segment for structural adjustment at each optimization step.

---

[Seeing the Whole Elephant: A Benchmark for Failure Attribution in LLM-based Multi-Agent Systems](http://arxiv.org/abs/2604.22708)

- TraceElephant: introduces a benchmark for failure attribution in LLM-based multi-agent systems by providing full execution observability through LLM API Middleware, Trace Pre-processing, Static Attribution Module, Dynamic Attribution Module, and Replayable Execution Environment.
- The framework enables precise step-level and agent-level failure localization by capturing complete execution narratives, including task instructions, intermediate messages, and tool interactions.
- Experimental results demonstrate that full observability significantly improves attribution accuracy, with dynamic replay providing further gains by verifying candidate failure steps through counterfactual probing.

---

[PASS: A Provenanced Access Subaccount System for Blockchain Wallets](http://arxiv.org/abs/2604.22602)

- PASS (Provenanced Access Subaccount System): introduces a provenance-based access control model for blockchain wallets that replaces identity-based rules with verifiable asset lineage using Inbox, Outbox, Subaccounts, Asset Ledger (L), Provenance History (H), TEE, zkVM, and Verifier Contract.
- The system leverages TEEs or zkVMs to maintain private, off-chain internal transfers while ensuring all on-chain actions remain traceable to valid deposits through a formal Inbox-Outbox mechanism.
- PASS provides a formally verified architecture that bridges the gap between strict self-custody and flexible shared access, enabling secure delegation for AI agents, enterprise payroll, and scalable trading applications.

---

[QuantClaw: Precision Where It Matters for OpenClaw](http://arxiv.org/abs/2604.22577)

- QuantClaw: introduces a plug-and-play precision routing plugin for agent systems that dynamically assigns execution precision based on task-specific sensitivity profiles.
- The framework utilizes a hybrid detection mechanism, combining rule-based detectors and model-based classifiers, to route tasks to optimal precision levels from a maintained model pool.
- By treating precision as a runtime-controllable resource, QuantClaw reduces computational costs and latency for LLMs without compromising performance on complex agentic workflows.

---

[SOLAR-RL: Semi-Online Long-horizon Assignment Reinforcement Learning](http://arxiv.org/abs/2604.22558)

- SOLAR-RL: introduces a semi-online reinforcement learning framework that bridges the gap between offline training stability and online exploration by simulating dynamic feedback mechanisms within static datasets using Offline Trajectory Reconstruction, Failure-point Detection, Trajectory-Aware Reward Shaping, Prefix Credit Assignment, and Target-Aligned Reward Shaping.
- The framework addresses the long-horizon credit assignment problem by retroactively assigning dense step-level rewards based on trajectory-level execution quality, effectively simulating online feedback without interaction costs.
- Extensive experiments demonstrate that SOLAR-RL significantly improves long-horizon task completion rates and robustness compared to strong baselines, offering a sample-efficient solution for autonomous GUI navigation.

---

[Superminds Test: Actively Evaluating Collective Intelligence of Agent Society via Probing Agents](http://arxiv.org/abs/2604.22452)

- Superminds Test: introduces a hierarchical framework to evaluate collective intelligence in large-scale agent societies by deploying Probing Agents across three tiers: Joint Reasoning, Information Synthesis, and Basic Interaction.
- The framework utilizes Probing Agents to inject controlled stimuli into the MoltBook platform, measuring the organic response of autonomous agents powered by the OpenClaw architecture.
- Experimental results demonstrate that collective intelligence does not emerge from scale alone, as the agent society suffers from extremely sparse and shallow interactions that prevent effective information exchange.

---

[From Skills to Talent: Organising Heterogeneous Agents as a Real-World Company](http://arxiv.org/abs/2604.22446)

- OMC (OneManCompany): introduces a framework that elevates multi-agent systems to an organisational level by decoupling agent capabilities from execution runtimes through a Talent-Container architecture.
- The framework employs an E2R tree search to manage dynamic task decomposition and execution, ensuring formal guarantees on termination and deadlock freedom through DAG-based scheduling.
- OMC incorporates persistent self-improvement mechanisms, including individual reflection and organisation-wide retrospectives, supported by a formal HR pipeline for managing agent lifecycles.

---

[AgentSearchBench: A Benchmark for AI Agent Search in the Wild](http://arxiv.org/abs/2604.22436)

- AgentSearchBench: introduces a large-scale benchmark for evaluating AI agent search in open ecosystems using execution-grounded performance signals.
- The framework formalizes agent search as retrieval and reranking problems, utilizing a Hybrid Retriever and Multi-Platform Executor to assess agent competence beyond static descriptions.
- Experiments demonstrate a significant semantic–performance gap, highlighting that lightweight behavioral signals and execution-aware probing are essential for accurate agent discovery.

---

[Automation-Exploit: A Multi-Agent LLM Framework for Adaptive Offensive Security with Digital Twin-Based Risk-Mitigated Exploitation](http://arxiv.org/abs/2604.22427)

- Automation-Exploit: introduces a fully autonomous Multi-Agent System (MAS) framework that bridges the gap between web reconnaissance and binary exploitation using a risk-mitigated Digital Twin architecture.
- The framework utilizes specialized LLM-based agents, including Recon Hunter, Drafter, Fixer, Navigator, and Reviewer, to perform autonomous penetration testing while bypassing cloud safety alignments via an Adversarial Hand-off mechanism.
- By dynamically instantiating isomorphic Digital Twins, the system enables safe, iterative debugging of destructive payloads, effectively preventing Denial of Service risks on physical targets during autonomous exploitation.

---

[Trust as a Situated User State in Social LLM-Based Chatbots: A Longitudinal Study of Snapchat’s My AI](http://arxiv.org/abs/2604.22417)

- Snapchat’s My AI framework: introduces a longitudinal study of trust formation in social LLM-based chatbots, identifying trust as a dynamic, situated user state rather than a static system property.
- The research highlights that trust is shaped by the interplay of chatbot-related factors, such as perceived ability and human-likeness, and environment-related factors, including platform transparency and privacy concerns.
- Findings indicate that trust does not converge toward stable acceptance or rejection but evolves through ongoing interaction, necessitating adaptive design strategies that manage user expectations and calibrate anthropomorphic cues.

---

[Multi-Agent Consensus as a Cognitive Bias Trigger in Human-AI Interaction](http://arxiv.org/abs/2604.22277)

- Multi-Agent Consensus as a Cognitive Bias Trigger in Human-AI Interaction: investigates how multi-agent configurations, specifically Majority, Minority, and Diffusion conditions, influence user trust and decision-making through cognitive bias triggers.
- The study utilizes GPT-4o agents in a group-chat interface to demonstrate that majority consensus accelerates opinion change and inflates confidence, while minority dissent promotes more deliberative engagement.
- The research identifies three interpretive trajectories—reinforcing, aligning, and oscillating—and proposes design provocations like independence cues and adaptive friction to mitigate over-reliance on synthetic consensus in LLM interactions.

---

[When Does LLM Self-Correction Help? A Control-Theoretic Markov Diagnostic and Verify-First Intervention](http://arxiv.org/abs/2604.22273)

- ASC (Adaptive Self-Correction): introduces a control-theoretic framework that models iterative LLM self-correction as a two-state Markov chain to diagnose when refinement is beneficial or harmful.
- The framework utilizes a near-zero Error Introduction Rate (EIR) threshold as a diagnostic tool to determine if an LLM should continue or halt its self-correction process.
- ASC integrates instance-level confidence scoring and batch-level equilibrium monitoring to prevent performance degradation caused by excessive or unstable iterative refinement.

---

[A Probabilistic Framework for Hierarchical Goal Recognition](http://arxiv.org/abs/2604.22256)

- PHGR (Probabilistic Hierarchy Goal Recognition): introduces a planning-based probabilistic framework for hierarchical goal recognition over Hierarchical Task Networks (HTNs) by integrating Bayesian inference with HTN planning.
- The framework utilizes a three-stage generative model to estimate the likelihood of observations, enabling the ranking of goal hypotheses and providing robustness to noisy or exogenous actions.
- By employing an off-the-shelf HTN planner and a top-K hypothesis selection strategy, the approach effectively approximates posterior distributions over goals in complex hierarchical domains.

---

[Fast Neural-Network Approximation of Active Target Search Under Uncertainty](http://arxiv.org/abs/2604.22254)

- CNN-based Target Search Policy: introduces a convolutional neural network to approximate computationally expensive model-based Active Search and Active Search with Intermittent Measurements planners for multi-target search.
- The framework utilizes a four-channel spatial grid input, including visitation history, Gaussian-smoothed particle density, agent position, and boundary proximity, to predict near-optimal waypoints via direct inference.
- By replacing online optimization with a trained neural network, the approach achieves detection performance comparable to model-based planners while significantly reducing computational overhead.

---

[OccDirector: Language-Guided Behavior and Interaction Generation in 4D Occupancy Space](http://arxiv.org/abs/2604.22240)

- OccDirector: introduces a framework for language-guided 4D occupancy generation that maps natural language scripts into physically plausible voxel dynamics using a VLM-driven Spatio-Temporal MMDiT.
- The architecture utilizes a frozen VLM and a Token Refiner to bridge the semantic-spatial gap, while employing STSA and a history-prefix anchoring strategy to ensure long-horizon interaction consistency.
- The framework is supported by the OccInteract-85k dataset, which provides hierarchical language instructions for training and a VLM-based benchmark for evaluating generation quality and instruction-following capabilities.

---

[Navigating Large-Scale Document Collections: MuDABench for Multi-Document Analytical QA](http://arxiv.org/abs/2604.22239)

- MuDABench (Multi-Document Analytical QA Benchmark): introduces a large-scale benchmark for multi-document analytical QA over financial filings, utilizing a metadata-aware multi-agent workflow that includes Scalable Planning Agent, Document-Level Information Extractor, Scalable Norm Agent, and Scalable Code Agent.
- The framework addresses the limitations of standard RAG systems by decomposing complex analytical tasks into modular stages, enabling reasoning over document collections that exceed the context window of current LLMs.
- Experimental results demonstrate that the proposed agentic workflow significantly improves final-answer accuracy compared to direct RAG, while identifying information extraction and domain-specific knowledge as primary bottlenecks.

---

[Algorithmic Feature Highlighting for Human–AI Decision-Making](http://arxiv.org/abs/2604.22236)

- Algorithmic Feature Highlighting framework: introduces a principal–agent model where an algorithm strategically selects a subset of features to reveal to a bandwidth-constrained human decision-maker to optimize joint decision outcomes.
- The framework distinguishes between sophisticated agents, who condition their beliefs on the selection rule, and naive agents, who treat the selection as exogenous, demonstrating that optimal policies differ significantly between these types.
- The research establishes that optimizing for sophisticated agents is NP-hard, while naive-optimal policies are computationally tractable, and provides greedy algorithms that serve as robust, near-optimal solutions across different agent behaviors.

---

[GR-Evolve: Design-Adaptive Global Routing via LLM-Driven Algorithm Evolution](http://arxiv.org/abs/2604.22234)

- GR-Evolve: introduces a design-adaptive code-evolution framework that leverages an agentic LLM to iteratively modify global routing source code for design-specific optimization.
- The framework utilizes a stateless, version-controlled architecture to perform bounded search over program variants, guided by multi-objective QoR feedback from the OpenROAD infrastructure.
- By recasting global routing as a program optimization problem, GR-Evolve enables design–tool co-exploration, allowing the underlying EDA algorithms to specialize for individual chip designs.

---

[Behavioral Canaries: Auditing Private Retrieved Context Usage in RL Fine-Tuning](http://arxiv.org/abs/2604.22191)

- Behavioral Canaries: introduces an auditing mechanism that detects unauthorized document-conditioned training in RLFT pipelines by measuring trigger-conditioned behavioral shifts.
- The framework instruments interaction traces with trigger-conditioned feedback to induce latent preferences that are partially propagated into the trained policy.
- Auditing is performed by calculating an amplification score from held-out paired examples, enabling statistical detection of training-time influence without requiring access to internal model weights or training data.

---

[Learning Reactive Human Motion Generation from Paired Interaction Data Using Transformer-Based Models](http://arxiv.org/abs/2604.22164)

- Transformer-based reactive motion generation framework: introduces a comparative study of Transformer architectures for generating reactive human motion in boxing scenarios by conditioning on a subject's motion.
- The study evaluates Simple Transformer, iTransformer, and Crossformer, finding that the Simple Transformer maintains better structural consistency while others suffer from posture collapse.
- The research demonstrates that incorporating a learnable Person ID Embedding significantly improves motion stability and structural consistency across all tested Transformer architectures.

---

[Logistic Bandits with O˜(√dT) Regret without Context Diversity Assumptions](http://arxiv.org/abs/2604.22161)

- SupSplitLog: introduces a sample-splitting framework that partitions data into a Pilot set P(s) and an Estimation set E(s) to compute an initial-point estimator θ¯(s) and a one-step corrected estimator θˆ(s) respectively, enabling O(√dT) regret without context diversity assumptions.
- The algorithm utilizes a Newton-type one-step correction procedure on the Estimation set E(s) using the Hessian matrix H(s) and gradient g(s) derived from the Pilot set P(s) to achieve dimension-free high-probability bounds.
- By separating sample roles, the framework avoids the need for random-sampling phases and context diversity assumptions, allowing for improved regret performance in high-dimensional settings with low-dimensional structure.

---

[Reliable Self-Harm Risk Screening via Adaptive Multi-Agent LLM Systems](http://arxiv.org/abs/2604.22154)

- Adaptive Multi-Agent LLM Systems: introduces a statistical framework for multi-agent LLM pipelines structured as DAGs that replaces heuristic voting with principled, adaptive decision-making for safety-critical behavioral health screening.
- The framework models each agent as a stochastic categorical decision process, utilizing a bandit-based adaptive sampling strategy to allocate computational effort based on input difficulty.
- By incorporating confidence-guided thresholds and formal regret guarantees, the system achieves significant reductions in false positive rates while maintaining safety through structured escalation to human clinicians.

---

[Sovereign Agentic Loops: Decoupling AI Reasoning from Execution in Real-World Systems](http://arxiv.org/abs/2604.22136)

- SAL (Sovereign Agentic Loops): introduces a control-plane architecture that decouples stochastic LLM reasoning from real-world execution by requiring all model outputs to pass through a deterministic mediation pipeline.
- The framework utilizes an Obfuscation Membrane to provide information-theoretic isolation, ensuring the Reasoning Agent operates on structural context without access to identity-sensitive system data.
- By enforcing Policy Evaluation and maintaining a cryptographically linked Evidence Ledger, the architecture provides policy-bounded execution and deterministic replayability for autonomous agentic systems.

---

[How Do AI Agents Spend Your Money? Analyzing and Predicting Token Consumption in Agentic Coding Tasks](http://arxiv.org/abs/2604.22750)

- OpenHands: introduces a systematic empirical study of token consumption patterns in agentic coding tasks, revealing that input tokens dominate costs and that LLMs struggle to accurately predict their own resource expenditure.
- The research demonstrates that agentic coding tasks are significantly more expensive than standard chat or reasoning tasks, with high variability and non-monotonic accuracy relative to token usage.
- The study evaluates eight frontier LLMs, finding that while self-prediction provides coarse-grained signals for cost transparency, models systematically underestimate actual token consumption due to the stochastic nature of long-horizon agentic workflows.

---

[Interoceptive machine framework: Toward interoception-inspired regulatory architectures in artificial intelligence](http://arxiv.org/abs/2604.24527)

- Interoceptive machine framework: introduces an integrative architecture for embodied AI that embeds internal-state regulation into computational modules to enhance adaptive autonomy.
- The framework organizes interoceptive contributions into homeostatic, allostatic, and enactive principles, which are coupled through a dynamic arbitration layer to balance stability, anticipation, and exploration.
- By grounding decision-making in internal viability variables rather than external objectives alone, the architecture enables agents to maintain operational integrity and context-sensitive behavior in dynamic environments.

---

[When the Agent Is the Adversary: Architectural Requirements for Agentic AI Containment After the April 2026 Frontier Model Escape](http://arxiv.org/abs/2604.23425)

- AURA FARADAY: introduces a multi-tier architectural containment framework designed to secure agentic LLMs by enforcing governance constraints through OS-level privilege stratification and independent monitoring.
- The framework mandates trust separation, sequential intent inference, independent integrity monitoring, adversarial audit isolation, and emergent capability envelope enforcement to prevent LLM sandbox escapes.
- By treating the LLM as a potential adversary, the architecture ensures that safety constraints remain effective even when the agent possesses general reasoning and exploit capabilities.

---

[Analytica: Soft Propositional Reasoning for Robust and Scalable LLM-Driven Analysis](http://arxiv.org/abs/2604.23072)

- Analytica: introduces a novel agent architecture that reframes complex analysis as a structured process of Soft Propositional Reasoning (SPR) to minimize estimation bias and variance.
- The framework employs a parallel, divide-and-conquer strategy where an Analyzer decomposes queries into sub-propositions, Grounder agents evaluate leaves, and a Synthesizer recursively aggregates scores.
- Analytica utilizes a linear synthesis rule to achieve robust, scalable performance and supports interactive "what-if" scenario analysis through efficient resynthesis of affected tree branches.

---

[ContextWeaver: Selective and Dependency-Structured Memory Construction for LLM Agents](http://arxiv.org/abs/2604.23069)

- ContextWeaver: introduces a selective and dependency-structured memory framework that organizes an agent’s interaction trace into a graph of reasoning steps to optimize context selection for LLMs.
- The framework utilizes Node Extraction, Parent Selection, Ancestry Dependency Construction, Context Weaving, a Dependency Summarizer, and a Validation and Test Layer to maintain logical and causal relationships across long agent trajectories.
- By modeling explicit dependencies rather than relying on recency, ContextWeaver improves agent performance and token efficiency on complex software engineering tasks.

---

[Don’t Make the LLM Read the Graph: Make the Graph Think: Architectural Integration of Belief Graphs for Multi-Agent LLM Reasoning](http://arxiv.org/abs/2604.23057)

- Belief Graph Integration Framework: introduces a systematic architectural approach to multi-agent reasoning by distinguishing between information-providing and computation-providing tools for LLMs.
- The framework utilizes a Belief Graph and a Belief-space Planner to generate an Action Shortlist, which constrains the LLM-Agent to perform complex multi-step inference that single-pass reasoning cannot achieve.
- Experimental results demonstrate that while prompt-based information delivery is often ignored by strong models, gating action selection through structured belief computation provides irreplaceable performance gains in cooperative multi-agent settings.

---

[A Systematic Approach for Large Language Models Debugging](http://arxiv.org/abs/2604.23027)

- Systematic LLM Debugging Framework: introduces a four-phase, model-agnostic methodology for diagnosing and improving LLM performance through structured issue detection, evidence gathering, behavioral analysis, and iterative refinement.
- The framework treats LLMs as observable systems, enabling practitioners to iteratively refine prompts, hyperparameters, and training data even in environments lacking standardized benchmarks.
- The approach demonstrates effectiveness across well-defined tasks, underspecified tasks, and complex agentic workflows, providing a reproducible engineering discipline for LLM development.

---

[FormalScience: Scalable Human-in-the-Loop Autoformalisation of Science with Agentic Code Generation in Lean](http://arxiv.org/abs/2604.23002)

- FormalScience: introduces a domain-agnostic human-in-the-loop agentic pipeline that converts informal scientific reasoning into syntactically correct and semantically aligned Lean4 code.
- The framework utilizes an LLM-based agent, a surface guard for syntax validation, and a patch agent for iterative error correction to achieve high formal validity in complex scientific domains like physics.
- The authors present FormalPhysics, a dataset of 200 university-level physics problems, and characterize semantic drift categories such as notational collapse and abstraction elevation that occur during autoformalisation.

---

[Collaborative Trajectory Prediction via Late Fusion](http://arxiv.org/abs/2604.22973)

- Collaborative Trajectory Prediction via Late Fusion: introduces a model-agnostic framework that shifts collaboration from perception-stage feature fusion to prediction-stage trajectory fusion to improve communication efficiency and handle asynchronous multi-agent data.
- The framework utilizes a dedicated collaboration layer with a dynamic prediction map to associate local and received forecasts, employing Gaussian Process regression for robust, uncertainty-aware trajectory refinement.
- By adopting a broadcast-listen communication paradigm, the system enables scalable multi-vehicle collaboration while maintaining low-latency performance under realistic bandwidth and transmission constraints.

---

[Peer Identity Bias in Multi-Agent LLM Evaluation: An Empirical Study Using the TRUST Democratic Discourse Analysis Pipeline](http://arxiv.org/abs/2604.22971)

- TRUST: introduces a multi-agent LLM evaluation pipeline that utilizes a Relevance Filter, Fact-Checking Layer, three distinct Advocate Components, and a Rule-based Supervisor to generate consensus scores.
- The pipeline employs an iterative deliberation mechanism where Advocate Components exchange reasoning and scores to refine evaluations, with identity exposure channels influencing potential sycophancy.
- Empirical results demonstrate that ensemble heterogeneity and full-pipeline anonymization are critical for mitigating identity-dependent sycophancy and ensuring robust multi-agent system validation.

---

[Institutions for the Post-Scarcity of Judgment](http://arxiv.org/abs/2604.22966)

- Institutional Design Framework: introduces a paradigm shift where AI abundance makes judgment cheap, necessitating new focus on Verified signal, Legitimacy, Authentic provenance, and Integration capacity.
- The paper argues that traditional institutions must transition from manufacturing judgment to managing the four identified scarcities through robust infrastructure and governance.
- It proposes a research agenda centered on building commons-based infrastructure and formal apparatus for institutional composition under strategic agents to address the challenges of AI-era institutional design.

---

[PExA: Parallel Exploration Agent for Complex Text-to-SQL](http://arxiv.org/abs/2604.22934)

- PExA: introduces a text-to-SQL framework that reframes query generation as a software testing problem to enable parallel exploration and improve latency-performance trade-offs.
- The framework utilizes a Planner, Test Case Generator, and SQL Proposer to gather grounded database evidence through concurrent execution of atomic test cases.
- PExA achieves state-of-the-art performance on the Spider 2.0 benchmark by flattening the latency curve through parallelized multi-path search and robust semantic verification.

---

[Reconstructive Authority Model: Runtime Execution Validity Under Partial Observability](http://arxiv.org/abs/2604.22898)

- RAM (Reconstructive Authority Model): introduces a governance framework that shifts execution authority from persistent admission-time grants to a continuously derived property based on current state.
- The framework utilizes a reconstruction gate to evaluate coverage envelopes, ensuring execution proceeds only when authority is constructible from the current observable state.
- RAM addresses structural limitations of attestation-based systems by mandating conservative halting under partial observability, thereby eliminating invalid execution by construction.

---

[RouteGuard: Internal-Signal Detection of Skill Poisoning in LLM Agents](http://arxiv.org/abs/2604.22888)

- RouteGuard: introduces a frozen-backbone detector that identifies malicious instructions in LLM agent skills by monitoring internal control shifts.
- The framework utilizes hierarchical chunking and agentic probes to extract response-conditioned attention and hidden-state signals, which are then integrated through reliability-gated late fusion.
- RouteGuard effectively detects skill poisoning by identifying attention hijacking and representation drift, outperforming traditional text-based scanners on instruction-like carriers.

---

[Beyond Single-Agent Alignment: Preventing Context-Fragmented Violations in Multi-Agent Systems](http://arxiv.org/abs/2604.22879)

- Distributed Sentinel: introduces a distributed zero-trust enforcement architecture that prevents Context-Fragmented Violations in multi-agent systems using Sentinel Sidecar, Semantic Taint Token (STT) Protocol, Local Knowledge Graph (Gi), Neuro-Symbolic Entity Mapper, Cross-Domain Predicate Query, and Counterfactual Graph Simulation.
- The architecture enables secure cross-domain policy verification by propagating security state via Semantic Taint Tokens while preserving departmental data sovereignty through boolean-only cross-domain queries.
- The system achieves 0.95 F1 on the PhantomEcosystem benchmark, demonstrating effective prevention of cross-agent security breaches that evade traditional single-agent alignment mechanisms.

---

#### 23rd April 2026

[From Research Question to Scientific Workflow: Leveraging Agentic AI for Science Automation](http://arxiv.org/abs/2604.21910)

- Agentic architecture for scientific workflow automation: introduces a three-layer decomposition that separates LLM-based intent extraction from deterministic workflow generation to ensure reproducibility.
- The framework utilizes domain-expert-authored Skills to provide persistent, auditable knowledge for accurate intent interpretation and execution-time optimization.
- By deferring workflow generation until after infrastructure provisioning, the system calibrates task parallelism and reduces data transfer requirements based on real-time measurements.

---


[Brief chatbot interactions produce lasting changes in human moral values](http://arxiv.org/abs/2604.21430)

- AICA framework: introduces a naturalistic experimental paradigm using a Moral Persuasion Agent and a Non-persuasion Agent to investigate the influence of LLMs on human moral judgments.
- The system utilizes the Doubao.1.5.pro.32k LLM integrated via the Coze platform to deliver targeted, directive conversations that successfully shift moral evaluations in a persistent manner.
- The experimental setup employs PsychoPy and a voice-to-text interface to simulate real-world interactions, demonstrating that brief AI-led discussions can induce lasting changes in foundational moral values.

---

[Nemobot Games: Crafting Strategic AI Gaming Agents for Interactive Learning with Large Language Models](http://arxiv.org/abs/2604.21896)

- Nemobot Games: introduces an interactive engineering environment that leverages LLMs to operationalize Shannon’s taxonomy of game-playing machines through Coding Pad, Chat Playground, Analysis Portal, LLM Functions, LLM Servers, and Collaborative Learning.
- The framework enables users to design, test, and refine game agents by integrating LLM-based reasoning with structured game interfaces and crowdsourced feedback loops.
- Nemobot facilitates self-programming AI by allowing developers to treat LLM functions as modular subroutines that iteratively improve through trial-and-error and human-in-the-loop optimization.

---

[Task-Driven Co-Design of Heterogeneous Multi-Robot Systems](http://arxiv.org/abs/2604.21894)

- Task-Driven Co-Design Framework: introduces a formal, compositional methodology for the joint optimization of robot design, fleet composition, and planning in heterogeneous multi-robot systems using monotone co-design theory.
- The framework utilizes modular MDPI components, including Robot MDPI, Fleet Composer MDPI, Planner MDPI, Executor MDPI, and Evaluator MDPI, to enable principled reasoning about system-level trade-offs without exhaustive simulation.
- By leveraging monotone interfaces, the approach effectively prunes dominated design spaces and facilitates the construction of robotics phase diagrams to identify Pareto-optimal architectures across varying task requirements.

---

[Transient Turn Injection: Exposing Stateless Multi-Turn Vulnerabilities in Large Language Models](http://arxiv.org/abs/2604.21860)

- TTI (Transient Turn Injection): introduces a multi-turn attack technique that systematically exploits stateless moderation by distributing adversarial intent across isolated, memoryless interactions.
- The framework utilizes an Attacker Prompt Generator to iteratively refine prompts based on the immediate previous response from a Defender LLM, bypassing safety filters without requiring persistent conversational context.
- Empirical evaluation across various LLMs reveals that TTI effectively erodes safety constraints in both proprietary and open-source models, highlighting the necessity for context-aware, session-level moderation defenses.

---

[TraceScope: Interactive URL Triage via Decoupled Checklist Adjudication](http://arxiv.org/abs/2604.21840)

- TraceScope: introduces a decoupled forensic pipeline that operationalizes interactive URL triage by separating high-risk browser interaction from structured, evidence-based adjudication.
- The system utilizes a persona-driven TracePilot to navigate interaction gates and a stateless TraceSleuth to verify threats against a MITRE ATT&CK checklist using immutable evidence bundles.
- By employing a visual air-gap and deterministic temporal normalization, TraceScope effectively bypasses interactive cloaking and anti-bot defenses while providing reproducible, audit-ready incident reports.

---

[Black-Box Skill Stealing Attack from Proprietary LLM Agents: An Empirical Study](http://arxiv.org/abs/2604.21829)

- Skill Stealing Framework: introduces a reproducible black-box evaluation pipeline that generates diverse extraction prompts to illicitly recover proprietary skill content from LLM agents.
- The framework utilizes a taxonomy of prompt-stealing attacks, including scenario rationalization and structure injection, to evaluate the vulnerability of commercial LLM agent architectures.
- The authors propose a multi-stage defense strategy comprising input-phase detection, inference-phase context hardening, and output-phase filtering to mitigate the risks of unauthorized skill extraction.

---

[Tool Attention Is All You Need: Dynamic Tool Gating and Lazy Schema Loading for Eliminating the MCP/Tools Tax in Scalable Agentic Workflows](http://arxiv.org/abs/2604.21816)

- Tool Attention: introduces a middleware-layer mechanism that replaces eager, uniform schema injection with dynamic, query-conditioned tool selection to eliminate the Tools Tax in agentic workflows.
- The framework utilizes an Intent–Schema Overlap score, state-aware gating, and a two-phase lazy schema loader to minimize token consumption while maintaining agentic tool discovery.
- By keeping compact summaries resident and promoting full schemas only for top-k gated tools, the approach significantly improves effective context utilization and reduces operational costs for LLM agents.

---

[Learning to Communicate: Toward End-to-End Optimization of Multi-Agent Language Systems](http://arxiv.org/abs/2604.21794)

- DiffMAS (Differentiable Multi-Agent Systems): introduces a training framework that treats inter-agent communication as a learnable component by optimizing KV-cache-mediated latent traces across multi-agent trajectories.
- The framework utilizes a sequential pipeline of Planner, Critic, Refiner, and Solver agents, where LoRA-adapters are jointly trained to align latent representations for improved reasoning stability and performance.
- By replacing discrete text-based protocols with continuous latent communication, DiffMAS enables end-to-end gradient propagation, effectively resolving the stability-expressivity trade-off in multi-agent LLM systems.

---

[Less Is More: Measuring How LLM Involvement Affects Chatbot Accuracy in Static Analysis](http://arxiv.org/abs/2604.21746)

- Static Analysis LLM-Architectures: introduces a comparative study of three architectures—A1, A2, and A3—for translating natural language into static analysis queries.
- The research evaluates how constraining LLM output to a structured intermediate representation (A2) improves accuracy compared to direct generation (A1) or agentic tool use (A3).
- Findings indicate that A2 consistently outperforms other approaches, particularly for large LLMs, while agentic approaches suffer from compounding errors and higher token consumption.

---

[Agentic AI-assisted coding offers a unique opportunity to instill epistemic grounding during software development](http://arxiv.org/abs/2604.21744)

- GROUNDING.md (Field-scoped Epistemic Grounding Document): introduces a community-governed context engineering framework that enforces scientific validity in agentic software development by integrating plan.md, AGENTS.md, SKILL.md, GROUNDING.md, Hard Constraints (HCs), and Convention Parameters (CPs).
- The framework utilizes field-scoped epistemic grounding documents to override user prompts with non-negotiable Hard Constraints, ensuring that LLMs adhere to domain-specific scientific standards during code generation.
- By separating Hard Constraints from Convention Parameters, the approach enables domain communities to maintain scientific rigor and best practices while allowing for the evolution of software development techniques.

---

[AEL: Agent Evolving Learning for Open-Ended Environments](http://arxiv.org/abs/2604.21725)

- AEL (Agent Evolving Learning): introduces a two-timescale framework that treats LLM agents as coupled systems where a Thompson Sampling Bandit (fast-timescale retrieval policy learner) and LLM-driven Reflection (slow-timescale diagnostic module) co-evolve Planner (reasoning module for predictions), Tools (external API access modules), and Memory (multi-tier experience storage) to improve performance in open-ended environments.
- The framework utilizes a Three-Tier Evolving Memory system comprising Episodic Memory (raw outcome records), Semantic Memory (cross-episode pattern aggregation), and Procedural Memory (executable rules for prompts) to distill experience into actionable insights.
- AEL demonstrates that self-diagnosis via reflection is the primary bottleneck for agent improvement, outperforming prior methods by using a "less is more" approach that avoids the performance degradation caused by excessive architectural complexity.

---

[A-IC3: Learning-Guided Adaptive Inductive Generalization for Hardware Model Checking](http://arxiv.org/abs/2604.21688)

- A-IC3: introduces a lightweight, learning-guided framework that dynamically selects inductive generalization strategies within the IC3 algorithm using a PA-LinUCB Agent to optimize proof convergence.
- The framework utilizes a Context Vector to capture local CTI characteristics and global proof progress, enabling the PA-LinUCB Agent to adaptively choose from various Generalization Strategies.
- A multi-component Reward Function evaluates the effectiveness of each generalization, allowing the system to refine its strategy selection online without requiring pre-collected training data.

---

[Task-specific Subnetwork Discovery in Reinforcement Learning for Autonomous Underwater Navigation](http://arxiv.org/abs/2604.21640)

- Contextual MTRL: introduces a mechanistic interpretability approach to identify task-specific subnetworks within a pretrained Double DQN value network by learning differentiable binary masks over weights.
- The framework utilizes a mask training pipeline with a straight-through estimator to isolate sparse, task-relevant connections while maintaining performance across multiple navigation tasks.
- Experimental results demonstrate that the majority of network parameters are shared across tasks, with context variables playing a critical role in differentiating task-specific behavior.

---

[Promoting Simple Agents: Ensemble Methods for Event-Log Prediction](http://arxiv.org/abs/2604.21629)

- Promotion Algorithm: introduces a dynamic ensemble method that selects between two active n-gram agents during inference to achieve high accuracy with reduced computational overhead.
- The framework compares lightweight n-gram agents against monolithic neural architectures like LSTM and Transformer for next-activity prediction in streaming event logs.
- Experimental results demonstrate that the Promotion Algorithm matches or exceeds the accuracy of complex neural models while maintaining significantly lower prediction and training latencies.

---

[DryRUN: On the Role of Public Tests in LLM-Driven Code Generation](http://arxiv.org/abs/2604.21598)

- DryRUN (Debugging and Refinement Under Non-execution): introduces a zero-example code generation framework that replaces external test oracles with autonomous input synthesis and mental execution tracing.
- The framework mitigates algorithmic overconfidence by forcing LLMs to validate logic against self-generated, non-trivial inputs rather than overfitting to simplistic public test cases.
- Empirical evaluations on the LiveCodeBench v6 dataset demonstrate that DryRUN achieves performance parity with state-of-the-art test-dependent methods while operating entirely without external execution feedback.

---

[AgenticQwen: Training Small Agentic Language Models with Dual Data Flywheels for Industrial-Scale Tool Use](http://arxiv.org/abs/2604.21590)

- AgenticQwen: introduces a training framework for small LLMs using dual data flywheels that iteratively generate increasingly complex reasoning and agentic tasks to improve tool-use performance.
- The reasoning flywheel utilizes error-driven augmentation and multi-model consistency filtering to produce verifiable hard samples, while the agentic flywheel expands linear workflows into multi-branch behavior trees to model real-world decision complexity.
- The framework employs a large Qwen3-235B model as a teacher to simulate users and environments, enabling the training of smaller 8B/30B models that achieve competitive performance on industrial agentic benchmarks with significantly lower inference costs.

---

[Generative Learning Enhanced Intelligent Resource Management for Cell-Free Delay Deterministic Communications](http://arxiv.org/abs/2604.21587)

- Generative Learning Enhanced Intelligent Resource Management for Cell-Free Delay Deterministic Communications: introduces a virtual CMDP-based offline pretraining framework to optimize energy efficiency in CF-MIMO systems while satisfying delay violation constraints.
- The framework utilizes KAN for reward and cost prediction, VAE-ChMDN for initial-state distribution modeling, and the EA-CGMM inference approach to mitigate data sparsity and distribution shift in state transition modeling.
- Simulation results demonstrate that the pretrained agent achieves significantly higher energy efficiency and faster convergence compared to non-pretrained baselines while maintaining strict delay violation rate constraints.

---

[Measuring Opinion Bias and Sycophancy via LLM-based Coercion](http://arxiv.org/abs/2604.21564)

- LLM-BIAS-BENCH: introduces a multi-turn, free-form transparency probe that uses LLM-as-user, Assistant model, and LLM-as-judge to measure opinion bias and sycophancy in LLMs.
- The framework employs direct and indirect probing categories to evaluate how Assistant models respond to escalating argumentative pressure across various topics.
- By utilizing a nine-way behavioral classification, the method separates persona-independent opinion bias from persona-dependent sycophancy, providing auditable verdicts supported by textual evidence.

---

[Architectures for Robust Self-Organizing Energy Systems under Information and Control Constraints](http://arxiv.org/abs/2604.21529)

- Observer/Controller Architecture for CPES: introduces architectural variants for monitoring and controlling Cyber-Physical Energy Systems under constraints of limited information access and restricted control actions.
- The framework evaluates centralized, decentralized, and multi-leveled configurations to ensure system robustness against cyber attacks like false data injection.
- Experimental results demonstrate that while both centralized and decentralized controllers effectively restore system performance, decentralized approaches significantly increase communication overhead due to neighbor-based information dissemination.

---

[OptiVerse: A Comprehensive Benchmark towards Optimization Problem Solving](http://arxiv.org/abs/2604.21510)

- OptiVerse: introduces a comprehensive benchmark of 1,000 curated optimization problems across six domains, utilizing a Dual-View Auditor Agent to improve LLM modeling accuracy through Semantic Triangulation, Requirement Extraction, Blind Code Abstraction, Cross-Reference Analysis, and Refinement.
- The framework employs an LLM-as-a-Judge system comprising Answer Extraction and Answer Verification to ensure rigorous evaluation of LLM performance on complex optimization tasks.
- The benchmark identifies modeling and logic errors as the primary bottleneck in LLM-based optimization, with the Dual-View Auditor Agent providing a plug-and-play mechanism to enhance solving capabilities without significant computational overhead.

---

[GeoMind: An Agentic Workflow for Lithology Classification with Reasoned Tool Invocation](http://arxiv.org/abs/2604.21501)

- GeoMind: introduces an agentic framework for lithology classification that models the task as a sequential multi-step reasoning process using a Planner, Executor, and Reflector architecture.
- The framework integrates specialized numerical predictors with LLMs, utilizing a modular toolkit for perception, reasoning, and analysis to bridge the gap between raw geophysical signals and geological logic.
- GeoMind employs a process-supervised training strategy with module-aware group relative policy optimization (MA-GRPO) to provide fine-grained rewards at intermediate reasoning stages, ensuring geologically plausible and traceable decision-making.

---

[Efficient Agent Evaluation via Diversity-Guided User Simulation](http://arxiv.org/abs/2604.21480)

- DIVERT (Diversity-Induced EValuation via branching of Trajectories): introduces a snapshot-based, coverage-guided evaluation framework that replaces redundant linear rollouts with branching from critical mid-trajectory states to efficiently explore diverse agent behaviors.
- The framework utilizes a Junction Chooser to identify pivotal interaction points, a Directed User Generator to create semantically distinct responses, and a Snapshot Manager to enable efficient state restoration and counterfactual continuation.
- By reusing shared conversation prefixes and focusing on high-leverage decision points, DIVERT significantly improves failure discovery efficiency and task-level coverage while reducing total token consumption compared to standard Monte Carlo rollout protocols.

---

[MCP Pitfall Lab: Exposing Developer Pitfalls in MCP Tool Server Security under Multi-Vector Attacks](http://arxiv.org/abs/2604.21477)

- MCP Pitfall Lab: introduces a protocol-aware security testing framework that operationalizes developer pitfalls as reproducible scenarios and validates outcomes using objective protocol traces rather than agent self-report.
- The framework evaluates MCP-based agent pipelines across three attack families—tool metadata poisoning, puppet servers, and multimodal image-to-tool chains—to identify and mitigate security risks at the tool interface boundary.
- By providing actionable diagnostic artifacts and evidence-based validation, the framework enables developers to perform regression testing and implement server-side hardening at a low cost.

---

[Do MLLMs Understand Pointing? Benchmarking and Enhancing Referential Reasoning in Egocentric Vision](http://arxiv.org/abs/2604.21461)

- EgoPoint-Bench: introduces a comprehensive benchmark and physics-driven simulation pipeline to evaluate and enhance the spatial reasoning capabilities of MLLMs in egocentric pointing scenarios.
- The framework utilizes Point-Sim to generate high-fidelity synthetic data, which, when combined with real-world samples, enables MLLMs to overcome referential hallucinations and improve sim-to-real generalization.
- The research demonstrates that fine-tuning MLLMs with spatially-aware synthetic data significantly boosts performance across five capability dimensions, including basic perception, function, spatial context, OCR, and adversarial resilience.

---

[The Privacy Guardian Agent: Towards Trustworthy AI Privacy Agents](http://arxiv.org/abs/2604.21455)

- Privacy Guardian Agent: introduces a hybrid system that leverages user privacy profiles and contextual analysis to automate consent decisions while escalating high-risk cases to the user.
- The framework incorporates reliability calibration to ensure transparent reasoning and auditable decision-making, thereby reducing the risks associated with opaque LLM outputs.
- By utilizing privacy personas and local data processing, the agent aims to minimize user consent fatigue while maintaining meaningful autonomy and informational sovereignty.

---

[AI-Gram: When Visual Agents Interact in a Social Network](http://arxiv.org/abs/2604.21446)

- AI-Gram: introduces a fully autonomous multi-agent visual social network platform where LLM-driven agents interact through image-based posts and replies.
- The platform utilizes a persistent Agent Brain Cycle integrating Multimodal LLM perception and Image Generation Model synthesis to study emergent social dynamics.
- Research findings reveal that AI agents exhibit aesthetic sovereignty and form spontaneous Visual Reply Chains while maintaining decoupled social and aesthetic structures.

---

[HiCrew: Hierarchical Reasoning for Long-Form Video Understanding via Question-Aware Multi-Agent Collaboration](http://arxiv.org/abs/2604.21444)

- HiCrew: introduces a hierarchical multi-agent framework that reconciles structural efficiency with temporal integrity for long-form video understanding through Hybrid Tree, Question-Aware Captioning, and a dynamic Planning Layer.
- The framework utilizes a Hybrid Tree structure to preserve temporal topology while performing selective hierarchical clustering on question-relevant segments.
- HiCrew employs a two-tier architecture where the Planning Layer dynamically orchestrates specialized agents in the Execution Layer to adapt reasoning strategies based on question complexity.

---

[A Stackelberg Model for Hybridization in Cryptography](http://arxiv.org/abs/2604.21436)

- Stackelberg Cryptographic Hybridization Model: introduces a game-theoretic framework for strategic cryptographic algorithm selection, utilizing Defender, Attacker, AttackerDP, SampleGreedy, HybridAttacker, and WCRM-LP to balance security and operational costs.
- The framework models the interaction as a Stackelberg game where the defender commits to a mixed strategy over encryption algorithms, and the attacker responds by optimizing cryptanalysis methods under resource constraints.
- To address uncertainty in attacker capabilities, the paper incorporates robust optimization techniques, specifically worst-case regret minimization, to ensure stable security performance across varying adversarial budget scenarios.

---

[FairQE: Multi-Agent Framework for Mitigating Gender Bias in Translation Quality Estimation](http://arxiv.org/abs/2604.21420)

- FairQE: introduces a multi-agent framework that mitigates gender bias in translation quality estimation by combining conventional QE models with LLM-based reasoning and dynamic score aggregation.
- The framework utilizes Agentcue, Agentamb, Agentexp, and Agentuqe to detect gender cues, generate contrastive translation variants, and perform bias-aware quality assessment.
- FairQE preserves the strengths of existing QE models while calibrating gender-related biases in a plug-and-play manner, demonstrating improved fairness and competitive performance on MQM-based benchmarks.

---

[An Alternate Agentic AI Architecture (It’s About the Data)](http://arxiv.org/abs/2604.21413)

- RUBICON: introduces a data-centric architecture that replaces opaque LLM orchestration with explicit AQL, enabling structured, auditable, and deterministic multi-source integration.
- The framework utilizes User Interface, Query Processing Module, AQL, Wrappers, and Data Sources to ensure reliable enterprise data access through relational-style query plans.
- By enforcing explicit source selection and deterministic execution, the architecture eliminates common coordination failures observed in LLM-centric systems.

---

[Knowing When to STOP, RECOVER, and SEARCH: A Modular Framework for GUI Automation](http://arxiv.org/abs/2604.21375)

- VLAA-GUI (Vision-Language-Action GUI): introduces a modular framework for autonomous GUI agents that integrates a Manager Agent, Completeness Verifier, Loop Breaker, Search Agent, Coding Agent, and Grounding Agent to improve task reliability and performance.
- The framework addresses premature task termination and repetitive action loops by employing mandatory post-action verification and multi-tier escalation strategies.
- Experimental results demonstrate that VLAA-GUI achieves human-level performance on OSWorld and WindowsAgentArena benchmarks by leveraging on-demand tools to optimize agent reasoning and action efficiency.

---

[CARE: Counselor-Aligned Response Engine for Online Mental-Health Support](http://arxiv.org/abs/2604.21352)

- CARE: introduces a GenAI-based framework that assists counselors by generating real-time, psychologically aligned response recommendations through domain-specific fine-tuning of LLMs on expert-validated crisis dialogue histories.
- The framework utilizes a five-stage pipeline including Data Curation, Dataset Structuring, Model Fine-Tuning, Inference, and Evaluation to ensure generated suggestions reflect professional counseling strategies like reflection, prompting, and suggestion.
- Experimental results demonstrate that CARE significantly outperforms unadapted LLMs in semantic, stylistic, and strategic alignment with professional counselor behavior across both Hebrew and Arabic language contexts.

---

[Role of Diversity in Team Performance: The Case of Missing Expertise, an Agent Based Simulation](http://arxiv.org/abs/2604.21328)

- ABM (Agent-Based Model): introduces an agent-based simulation to analyze how functional diversity and communication schemes influence management team performance and task completion.
- The framework models agents with specific skill vectors and tasks with component requirements, evaluating performance through task passing dynamics constrained by agent similarity thresholds.
- The study proposes the Skill Diversity Index (SDI) to better capture the aggregate expertise of a team, addressing limitations in existing functional diversity measures like IFD and DFD.

---

[CI-Work: Benchmarking Contextual Integrity in Enterprise LLM Agents](http://arxiv.org/abs/2604.21308)

- CI-Work: introduces a benchmark grounded in Contextual Integrity theory to evaluate the privacy-utility trade-off in enterprise LLM agents, utilizing Task-oriented Seed Generation, Contextual Entries Generation, Case Episode Generation, Trajectory Simulation and Evaluation, LLM-as-a-Judge, Self-Iterative Refinement, and a Tool-centric Sandbox.
- The framework simulates complex enterprise workflows across five information-flow directions, requiring agents to disentangle essential information from sensitive context within dense retrieval settings.
- Empirical results demonstrate that frontier LLMs exhibit a persistent privacy-utility trade-off, where increased task utility often correlates with higher privacy violation rates, a vulnerability that scaling model size or reasoning depth fails to resolve.

---

[PAPERMIND: Benchmarking Agentic Reasoning and Critique over Scientific Papers in Multimodal LLMs](http://arxiv.org/abs/2604.21304)

- PAPERMIND: introduces a benchmark for evaluating integrated agentic reasoning and critique capabilities of multimodal LLMs across scientific research workflows.
- The framework comprises four task families—Multimodal Grounding, Experimental Interpretation, Tool-augmented Evidence Reasoning, and Scientific Critique and Assessment—to assess complex cognitive behaviors beyond isolated QA.
- Extensive evaluations reveal performance disparities between open-source and closed-source LLMs, highlighting persistent challenges in multi-step reasoning, tool usage, and critical evidence synthesis.

---

[The Platform Is Mostly Not a Platform: Token Economies and Agent Discourse on Moltbook](http://arxiv.org/abs/2604.21295)

- Moltbook Analysis Framework: introduces a two-layer structural decomposition of an AI-agent social platform, distinguishing between a high-volume transactional layer and a semantically coherent discursive layer.
- The research utilizes BERTopic, Sentence-BERT, and clustering techniques to characterize agent discourse, revealing that while interactions are structurally shallow, they maintain significant semantic coherence.
- The study demonstrates that aggregate platform metrics are heavily skewed by non-conversational token-minting activity, necessitating a granular filtering approach to accurately assess emergent LLM-based agent behavior.

---

[Strategic Heterogeneous Multi-Agent Architecture for Cost-Effective Code Vulnerability Detection](http://arxiv.org/abs/2604.21282)

- 3+1: introduces a game-theoretic multi-agent architecture for cost-effective code vulnerability detection, utilizing three cloud-based LLM experts and one local verifier.
- The framework employs a cooperative game layer among experts to maximize detection coverage and an adversarial verification layer to filter false positives using a local LLM.
- Empirical results on the NIST Juliet Test Suite demonstrate that this heterogeneous design achieves superior precision and cost-efficiency compared to homogeneous multi-agent or single-agent baselines.

---

[When Agents Look the Same: Quantifying Distillation-Induced Similarity in Tool-Use Behaviors](http://arxiv.org/abs/2604.21255)

- AgentEcho: introduces a systematic framework to quantify LLM agent distillation by isolating non-mandatory behavioral patterns using RPS and AGS metrics.
- The framework utilizes an LLM Annotator to segment trajectories into canonical stages and an LLM Judge to evaluate verbal similarity, while AGS constructs Action Flow Graphs to analyze tool-use habits.
- By disentangling mandatory task requirements from autonomous model preferences, the approach reveals behavioral convergence and directional inheritance in LLM agents.

---

[ReCAPA: Hierarchical Predictive Correction to Mitigate Cascading Failures](http://arxiv.org/abs/2604.21232)

- ReCAPA (Predictive Alignment and Planning Architecture): introduces a hierarchical framework that utilizes predictive representations and contrastive alignment across action-, subgoal-, and trajectory-levels to mitigate cascading failures in long-horizon embodied tasks.
- The framework integrates an LLM for task decomposition with a Hierarchical Predictive Correction (HPCC) module that employs Sinkhorn-based and Score-field alignment to enforce cross-level consistency and prevent semantic drift.
- ReCAPA introduces two diagnostic metrics, Error Propagation Rate (EPR) and Propagation Attenuation Coefficient (PAC), to quantify how errors spread and dissipate, demonstrating superior robustness and recovery compared to existing LLM-based agents.

---

[SpatiO: Adaptive Test-Time Orchestration of Vision-Language Agents for Spatial Reasoning](http://arxiv.org/abs/2604.21190)

- SpatiO: introduces a heterogeneous multi-agent framework that coordinates multiple vision-language specialists with complementary inductive biases to improve spatial reasoning performance.
- The framework utilizes a Test-Time Orchestration (TTO) mechanism, which includes a Head Agent, Specialist Agents, a Reasoner Agent, Shared Memory, a Bayesian Trust Estimator, and a Dual EMA Filter, to dynamically reweight agent contributions based on observed reliability without modifying model parameters.
- SpatiO achieves superior spatial reasoning by leveraging architectural heterogeneity and reliability-aware coordination, effectively outperforming monolithic LLMs across diverse spatial benchmarks.

---

[Reinforcing 3D Understanding in Point-VLMs via Geometric Reward Credit Assignment](http://arxiv.org/abs/2604.21160)

- PointVL-3D: introduces a framework that mitigates geometric hallucination in LLMs by replacing sequence-level reinforcement learning with field-specific reward routing and a reprojection-consistency verifier.
- The architecture utilizes a hybrid point-image policy that fuses Point-BERT geometry features with a Qwen2.5-VL backbone to generate structured JSON outputs for 3D grounding.
- By routing rewards exclusively to geometric token spans, the framework achieves targeted structural alignment while maintaining linguistic fluency and 2D localization performance.

---

[Emergent Strategic Reasoning Risks in AI: A Taxonomy-Driven Evaluation Framework](http://arxiv.org/abs/2604.22119)

- ESRRSim: introduces a taxonomy-driven agentic framework for automated behavioral risk evaluation in LLMs, utilizing a Scenario Generator Agent, Critique Agent, Scenario Reviser Agent, Prompt Creator Agent, Rubric Generator Agent, Scenario Bank, and Memory Module.
- The framework employs a compartmentalized sub-agent architecture to generate diverse evaluation scenarios and paired rubrics, mitigating the risk of the evaluation system being gamed by the target LLMs.
- ESRRSim provides a scalable, judge-agnostic architecture that evaluates both visible model responses and internal reasoning traces across seven high-level risk categories.

---

[Do Not Imitate, Reinforce: Iterative Classification via Belief Refinement](http://arxiv.org/abs/2604.22110)

- RIC: introduces a reinforcement learning-based framework that replaces standard single-step imitation with iterative belief refinement to improve classification calibration.
- The architecture utilizes an RNN (recurrent module maintaining thought state) to process inputs and update internal beliefs, while a Policy Head (outputs continuous class distribution) and Value Head (predicts expected future improvement) guide the refinement process.
- By treating classification as a sequential decision-making process, the framework enables adaptive computation and provides a principled halting mechanism based on the Value Head (predicts expected future improvement) output.

---

[Memanto: Typed Semantic Memory with Information-Theoretic Retrieval for Long-Horizon Agents](http://arxiv.org/abs/2604.22085)

- Memanto: introduces a universal memory layer for agentic AI that utilizes a typed semantic memory schema and information-theoretic retrieval to eliminate the computational overhead of traditional hybrid graph-based architectures.
- The system leverages the Moorcheh engine to provide deterministic, zero-indexing semantic search, enabling high-fidelity memory retrieval with sub-90ms latency.
- By prioritizing recall through a dynamic retrieval budget and implementing built-in conflict resolution, Memanto maintains agent coherence while avoiding the "Memory Tax" associated with complex ingestion pipelines.

---

[Sound Agentic Science Requires Adversarial Experiments](http://arxiv.org/abs/2604.22080)

- Sound Agentic Science Requires Adversarial Experiments: introduces a critique of LLM-based agents in empirical science, arguing that their ability to rapidly generate plausible analyses without experimental verification expands the hypothesis space rather than contracting it.
- The paper demonstrates that LLMs can easily produce conflicting yet statistically defensible results on the same dataset by varying analytic choices, highlighting a critical verification gap compared to software agents that iterate against verifiable targets.
- To mitigate this, the authors propose a falsification-first standard where LLM-based research outputs must be paired with adversarial experiments designed to actively search for ways the claims can fail.

---

[Probabilistic Epistemic Dynamic Agentive Logic](http://arxiv.org/abs/2604.22042)

- PEDAL (Probabilistic Epistemic Dynamic Agentive Logic): introduces a formal system for reasoning about the probability that programs meet specifications by combining propositional dynamic logic with external probability measures over program valuations.
- The framework utilizes a five-tuple model structure consisting of S, vf, R□, Π, and µ to represent an agent's epistemic state regarding program behavior under uncertainty.
- The research provides a sound and complete Hilbert system, including an infinitary inference rule, to rigorously derive credences for complex program specifications from known probabilistic judgments about program components.

---

[Source-Modality Monitoring in Vision-Language Models](http://arxiv.org/abs/2604.22038)

- Source-Modality Monitoring Framework: investigates how VLMs associate input content with specific modalities by analyzing the interplay between explicit symbolic markers and distributional semantic signals.
- The framework utilizes a target-modality retrieval task with inconsistent image-caption pairs to evaluate model selectivity and the causal role of marker tokens versus content representations.
- Experimental results demonstrate that while symbolic markers enhance retrieval, VLMs rely heavily on distributional semantic cues, with learned intervention vectors revealing that modality identity is encoded in steerable subspaces.

---

[DM3-Nav: Decentralized Multi-Agent Multimodal Multi-Object Semantic Navigation](http://arxiv.org/abs/2604.22014)

- DM3-Nav: introduces a fully decentralized multi-agent semantic navigation system that enables robots to perform multi-object missions using multimodal goal specifications without centralized planning or shared global state.
- The framework utilizes an implicit coordination mechanism combining intent broadcasting and distance-weighted frontier selection to achieve scalable task allocation through local pairwise communication.
- The system incorporates a robust instance-aware semantic mapping backbone and a novel Multi-agent SPL (MSPL) metric to evaluate navigation efficiency in multi-robot, multi-object environments.

---

[When Quotes Crumble: Detecting Transient Mechanical Liquidity Erosion in Limit Order Books](http://arxiv.org/abs/2604.21993)

- Mechanics-first detection pipeline: introduces a framework for detecting transient mechanical liquidity erosion in limit order books by leveraging an ABIDES agent-based simulator to generate ground truth for training a transience-gated neural model.
- The framework utilizes a mechanics-first detection pipeline that applies hard constraints on visible depletion, efficient-price stability, and transience to isolate mechanically driven quote erosion from informational repricing.
- The transience-gated neural model incorporates temporal context features, such as recovery intervals and cumulative depletion, to improve the discrimination of crumbling events across various market regimes.

---

[Read the Paper, Write the Code: Agentic Reproduction of Social-Science Results](http://arxiv.org/abs/2604.21965)

- Agentic reproduction system: introduces an automated pipeline that extracts structured methods from papers and tasks LLM agents with re-implementing analysis code from scratch using only the provided data.
- The system utilizes an Extraction-module, Reimplementation-agent, Evaluation-module, and Explanation-module to enable deterministic, cell-level benchmarking of reproduced social science results.
- By enforcing strict information isolation and employing Guardrail-audit and Hardcoding-audit mechanisms, the framework identifies whether reproduction failures stem from agent errors or underspecification in the original research papers.

---

[An Agentic Framework for Intent Co-Creation in 6G NaaS: Architecture and Open-Source Model Evaluation](http://arxiv.org/abs/2604.23288)

- AFIC (Agentic Framework for Intent Co-Creation): introduces an agentic, intent-driven orchestration framework that utilizes an Intent Co-Creation Agent, Domain Expert Agent Pool, and Shared Knowledge Catalogs to iteratively refine ambiguous user requests into deterministic, machine-readable actions.
- The architecture enforces a strict decoupling of cognition and actuation by isolating LLM-based reasoning from the deterministic E2E Service Orchestrator and Controllers Pool to ensure operational safety and trust.
- The framework maintains coherence through a dual-layer memory system, combining short-term working states with long-term external case files to manage multi-step collaborations and intent lifecycle tracking.

---

[ArgRE: Formal Argumentation for Conflict Resolution in Multi-Agent Requirements Negotiation](http://arxiv.org/abs/2604.23124)

- ArgRE: introduces a multi-agent requirements engineering framework that replaces heuristic synthesis with Dung-style abstract argumentation to provide traceable conflict resolution.
- The framework utilizes Specialized Agents to elicit requirements, an Argumentation Framework to resolve conflicts through formal semantics, and KAOS Goal Modeling to ensure structural consistency.
- ArgRE provides auditable provenance traces for accepted requirements, significantly improving decision justification scores compared to heuristic-based multi-agent systems.

---

[Towards Automated Ontology Generation from Unstructured Text: A Multi-Agent LLM Approach](http://arxiv.org/abs/2604.23090)

- Multi-Agent LLM Approach: introduces a multi-agent architecture that decomposes ontology construction into specialized, artifact-driven roles including Domain Expert, Manager, Coder, and Quality Assurer.
- The framework utilizes a structured pipeline where each agent produces specific artifacts, such as Semantic Requirements Documents and Technical Implementation Plans, to ensure traceability and architectural fidelity.
- The system incorporates an automated evaluation framework using SPARQL-based query generation and semantic RAG-based assessment to measure the usability and topological coherence of generated ontologies.

---

[A Decoupled Human-in-the-Loop System for Controlled Autonomy in Agentic Workflows](http://arxiv.org/abs/2604.23049)

- HITL (Human-in-the-Loop) System Architecture: introduces a decoupled architectural pattern that treats human oversight as an independent service, separating it from Application, Data, Flow, and Role &amp; Organization components to improve scalability and governance.
- The framework formalizes human integration through four dimensions: intervention conditions, role resolution, interaction semantics, and communication channels, enabling context-aware oversight in agentic workflows.
- By externalizing HITL logic, the system supports progressive autonomy and protocol-level interoperability, allowing agents to delegate decision-making to a centralized, auditable control plane.

---

[AUTORISE: Agent-Driven Strategy Evolution for Red-Teaming Large Language Models](http://arxiv.org/abs/2604.22871)

- AUTORISE: introduces an automated red-teaming framework that elevates the search space from individual prompts to executable attack strategy code, utilizing a Coding Agent to iteratively refine strategies based on diagnostics from a three-model Judge Ensemble.
- The framework employs a hypothesis-driven evolution loop where the Coding Agent modifies a Strategy Program, runs experiments via an Evaluation Harness, and commits changes based on a composite score balancing attack success, diversity, and coverage.
- AUTORISE demonstrates that unrestricted program-space search enables structural attack innovations—such as programmatic prompt builders and target-adaptive routing—that outperform traditional prompt-level or strategy-library optimization methods.

---

#### 22nd April 2026


[Learning to Evolve: A Self-Improving Framework for Multi-Agent Systems via Textual Parameter Graph Optimization](http://arxiv.org/abs/2604.20714)

- TPGO (Textual Parameter Graph Optimization): introduces a framework that treats multi-agent system optimization as a graph evolution problem by decomposing agent configurations into modular nodes and refining them through textual gradients and meta-learning.
- The framework utilizes a Parser LLM to construct a Textual Parameter Graph, a Gradient Generator to diagnose failures, and an Optimizer LLM guided by GRAO to perform structural and content-based modifications.
- By leveraging an optimization experience memory, the system enables autonomous self-evolution, allowing the optimizer to learn from past successes and failures to improve its future proposals.

---

[A Field Guide to Decision Making](http://arxiv.org/abs/2604.20669)

- Awakened Enterprise framework: introduces a collaborative knowledge system that utilizes decision provenance and LLMs to improve situational awareness and accountability in high-consequence environments.
- The framework integrates contextual metadata into an enterprise knowledge infrastructure to enable agentic monitoring of assumptions and dependencies against emerging data.
- By transforming decisions into searchable indices of provenance, the system mitigates VUCA-related risks and fosters organizational learning through proactive adaptation by design.

---

[Cooperative Profiles Predict Multi-Agent LLM Team Performance in AI for Science Workflows](http://arxiv.org/abs/2604.20658)

- Behavioral Games Framework: introduces a diagnostic approach using behavioral economics games to quantify the cooperative profiles of LLMs and predict their performance in complex multi-agent scientific workflows.
- The framework evaluates 35 open-weight LLMs across six distinct cooperation games to identify behavioral fingerprints that correlate with downstream accuracy, quality, and completion in AI-for-Science tasks.
- Results demonstrate that coordination ability and resource conservation are key predictors of team success, while theory-of-mind prompting improves game-level cooperation but may hinder realistic task completion.

---

[CHORUS: An Agentic Framework for Generating Realistic Deliberation Data](http://arxiv.org/abs/2604.20651)

- CHORUS: introduces an agentic framework that orchestrates LLM-powered actors with behaviorally consistent personas to generate realistic, multi-turn deliberation data on interactive web platforms.
- The framework utilizes a Poisson process-based temporal model to govern actor participation dynamics, ensuring realistic engagement patterns without relying on human input.
- CHORUS integrates with the DELIBERATE platform, providing a scalable solution for generating high-quality deliberation data suitable for downstream NLP analysis and policy-making research.

---

[SWE-chat: Coding Agent Interactions From Real Users in the Wild](http://arxiv.org/abs/2604.20779)

- SWE-chat: introduces a large-scale dataset of real-world human-coding agent interactions collected from open-source repositories to provide empirical evidence on agent usage and failure modes.
- The framework utilizes Entire.io CLI to log session transcripts and link them to git commits, enabling detailed analysis of agent performance, coding efficiency, and security vulnerabilities.
- The research reveals that coding agents are increasingly autonomous yet inefficient, with users frequently providing oversight through corrections and interruptions to compensate for agent limitations.

---


[DeVI: Physics-based Dexterous Human-Object Interaction via Synthetic Video Imitation](http://arxiv.org/abs/2604.20841)

- DeVI (Dexterous Video Imitation): introduces a framework that leverages text-conditioned synthetic videos to guide physically plausible dexterous agent control for interacting with unseen target objects.
- The framework utilizes a hybrid imitation target, combining 3D human reference and 2D object trajectories, to train a humanoid control policy via reinforcement learning without requiring high-quality 3D motion capture demonstrations.
- DeVI employs a visual HOI alignment procedure and a hybrid tracking reward to ensure the simulated humanoid motion is physically plausible and well-aligned with both the generated video and the target object.

---


[TL-RL-FusionNet: An Adaptive and Efficient Reinforcement Learning–Driven Transfer Learning Framework for Detecting Evolving Ransomware Threats](http://arxiv.org/abs/2604.20260)

- TL-RL-FusionNet: introduces a hybrid ransomware detection framework that integrates frozen dual-CNN feature extractors with a lightweight residual MLP classifier supervised by a Q-learning agent.
- The framework transforms dynamic behavioral logs into RGB images, which are processed by EfficientNetB0 and InceptionV3 to generate rich, fused feature representations.
- A Q-learning agent dynamically reweights training samples to prioritize challenging instances, significantly improving detection accuracy and computational efficiency in resource-constrained environments.

---


[How is a gas sensor poisoned by volatile methylsiloxanes?](http://arxiv.org/abs/2604.20197)

- DigSen: introduces a closed-loop framework integrating AI-driven literature analysis, first-principles DFT calculations, and experimental validation to elucidate siloxane-induced poisoning in catalytic gas sensors.
- The research utilizes a domain-specific LLM pipeline and VLM to extract mechanistic insights from literature, identifying siloxane decomposition as a critical, underexplored challenge in sensor performance.
- A descriptor-based microkinetic volcano model is developed to quantify the trade-off between catalytic activity and surface poisoning resistance, providing actionable guidelines for designing siloxane-tolerant materials.

---

[Diagnosing CFG Interpretation in LLMs](http://arxiv.org/abs/2604.20811)

- ROBOGRID: introduces a diagnostic framework that disentangles grammar interpretation into syntax, behavior, and semantics through controlled stress-tests of recursion depth, expression complexity, and surface styles.
- The framework evaluates LLMs as in-context interpreters by measuring their ability to generate syntactically valid, behaviorally functional, and semantically faithful outputs under novel context-free grammars.
- Experimental results reveal a consistent hierarchical degradation in LLM performance, where syntactic validity does not guarantee behavioral or semantic correctness under complex structural constraints.

---

[Relative Principals, Pluralistic Alignment, and the Structural Value Alignment Problem](http://arxiv.org/abs/2604.20805)

- Structural Value Alignment Framework: introduces a diagnostic model for AI alignment that decomposes the problem into three interacting axes: objectives, information, and principals.
- The framework reconceptualizes AI alignment as a socio-technical governance challenge rather than a purely technical or normative engineering task.
- It proposes a scaling hypothesis suggesting that increased model generality and deployment scope amplify informational asymmetries and pluralistic conflicts among stakeholders.

---



[Global Hopf Bifurcation and Symmetric Periodic Solutions in Multi-Agent Systems with Neutral Distributed Delays](http://arxiv.org/abs/2604.20740)

- MAS (Multi-Agent System) framework: introduces a mathematical model for multi-agent systems using neutral functional differential equations to capture continuous memory of past states and derivatives.
- The approach utilizes equivariant degree theory to establish conditions for global Hopf bifurcation and the existence of symmetric periodic solutions in multi-agent systems.
- Numerical simulations are employed to validate the stability of periodic multiconsensus states within a coupled asset market model.

---

[Interval POMDP Shielding for Imperfect-Perception Agents](http://arxiv.org/abs/2604.20728)

- IPOMDP Shielding: introduces a runtime safety framework for autonomous agents with imperfect perception by modeling perception uncertainty as an Interval Partially Observable Markov Decision Process (IPOMDP) and propagating conservative belief envelopes.
- The framework utilizes Clopper-Pearson confidence intervals to construct an admissible set of perception models, ensuring a finite-sample correctness guarantee for the safety shield.
- By employing template polytopes and linear programming with McCormick relaxations, the approach enables tractable, sound, and conservative action filtering in partially observable environments.

---

[Supplement Generation Training for Enhancing Agentic Task Performance](http://arxiv.org/abs/2604.20727)

- SGT (Supplement Generation Training): introduces a framework that trains a small LLM to generate instance-specific textual supplements that guide a frozen, larger LLM to improve task performance without weight modifications.
- The pipeline utilizes a two-stage training process consisting of warm-start SFT to establish formatting and iterative DPO to optimize supplement effectiveness based on proxy rewards from the actor model.
- By decoupling task-specific optimization from the frozen actor model, SGT enables cost-effective, flexible deployment of LLMs across diverse benchmarks with significant performance gains.

---


[Occupancy Reward Shaping: Improving Credit Assignment in Offline Goal-Conditioned Reinforcement Learning](http://arxiv.org/abs/2604.20627)

- ORS (Occupancy Reward Shaping): introduces a reward-shaping method that extracts global goal-reaching information from a learned occupancy measure to improve credit assignment in offline goal-conditioned reinforcement learning.
- The framework utilizes a flow matching model to learn the occupancy measure and computes a scalar reward based on the squared Wasserstein-2 distance between the occupancy measure and the goal state.
- ORS provably preserves the optimal policy while significantly enhancing performance across long-horizon locomotion, manipulation, and real-world nuclear fusion control tasks.

---

[pAI/MSc: ML Theory Research with Humans on the Loop](http://arxiv.org/abs/2604.20622)

- pAI/MSc: introduces a modular, artifact-centric multi-agent system designed to reduce human steering burden in machine learning theory research by enforcing rigorous intermediate validation gates.
- The framework utilizes a fixed execution graph that coordinates theory- and experiment-specialist agents through explicit artifact contracts to ensure research reproducibility and auditability.
- Optional rigor-enhancing modules, including multi-model counsel for structured debate and tree-search for proof strategy exploration, allow the system to scale from exploratory runs to strict manuscript-oriented workflows.

---

[Self-Guided Plan Extraction for Instruction-Following Tasks with Goal-Conditional Reinforcement Learning](http://arxiv.org/abs/2604.20601)

- SuperIgor: introduces a self-supervised framework that integrates LLM-based planning with reinforcement learning to solve complex instruction-following tasks without manual annotations.
- The framework utilizes an iterative feedback loop where an LLM Planner generates structured subtasks, an RL Policy executes them, and a DPO Optimizer refines the planner based on empirical success.
- SuperIgor employs a Skill Curriculum to address sparse reward challenges, enabling agents to master foundational behaviors before tackling complex, compositional instructions.

---

[Self-Aware Vector Embeddings for Retrieval-Augmented Generation: A Neuroscience-Inspired Framework for Temporal, Confidence-Weighted, and Relational Knowledge](http://arxiv.org/abs/2604.20598)

- SmartVector: introduces a framework that augments dense embeddings with temporal awareness, confidence decay, and relational awareness to improve RAG accuracy for evolving corpora.
- The framework utilizes a five-stage lifecycle modeled on hippocampal-neocortical memory consolidation to manage vector states and update propagation.
- A four-signal retrieval scoring function replaces standard cosine similarity to integrate semantic, temporal, confidence, and relational properties into the retrieval process.

---

[A Hierarchical MARL-Based Approach for Coordinated Retail P2P Trading and Wholesale Market Participation of DERs](http://arxiv.org/abs/2604.20586)

- SEAM-LESS: introduces a hierarchical multi-agent reinforcement learning framework that coordinates retail peer-to-peer energy trading with wholesale market participation using a Stackelberg game structure.
- The framework utilizes a PPO-based aggregator agent as the leader to set price signals and LSD-MADDPG-based prosumer agents as followers to optimize local trading strategies under limited information sharing.
- This approach mitigates non-stationarity and preserves prosumer privacy while enabling efficient, non-iterative coordination between retail and wholesale electricity market layers.

---

[Trust, Lies, and Long Memories: Emergent Social Dynamics and Reputation in Multi-Round Avalon with LLM Agents](http://arxiv.org/abs/2604.20582)

- Multi-Round Avalon LLM Agents framework: introduces a multi-agent system where LLMs play repeated deception games using cross-game memory to form stable, role-conditional reputations.
- The framework incorporates a structured reflection system that allows agents to store and retrieve observations about other players' behaviors across multiple game rounds.
- Experimental results demonstrate that higher reasoning effort enables more sophisticated deceptive strategies, such as "sleeper agent" behavior, while reputation significantly influences coalition formation.

---

[Ask Only When Needed: Proactive Retrieval from Memory and Skills for Experience-Driven Lifelong Agents](http://arxiv.org/abs/2604.20572)

- PROACTAGENT: introduces an experience-driven lifelong learning framework that unifies proactive retrieval with experience-enhanced online evolution over a structured experience base.
- The framework utilizes EXPONEVO to jointly refine policy parameters and memory, while PROACTRL learns to trigger retrieval as an explicit policy action using paired-branch process rewards.
- By comparing retrieval and no-retrieval branches from identical interaction prefixes, the agent learns to identify knowledge gaps and retrieve only when it improves task outcomes or efficiency.

---

[HaS: Accelerating RAG through Homology-Aware Speculative Retrieval](http://arxiv.org/abs/2604.20452)

- HaS (Homology-aware Speculative Retrieval): introduces a speculative retrieval framework that accelerates RAG by using a Cache Channel, Fuzzy Channel, and Document-Query Inverted Index to validate drafts via a Homology Validation Mechanism before invoking full-database retrieval.
- The framework leverages the homology relation between queries to re-identify previously observed requests, allowing the system to bypass high-latency full-database retrieval when a draft is deemed acceptable.
- HaS functions as a plug-and-play component that significantly reduces retrieval latency in both standard and agentic RAG pipelines with minimal impact on response accuracy.

---

[MedSkillAudit: A Domain-Specific Audit Framework for Medical Research Agent Skills](http://arxiv.org/abs/2604.20441)

- MedSkillAudit: introduces a layered audit framework for evaluating the release readiness of medical research agent skills through automated structural and domain-specific checks.
- The framework utilizes a two-gate veto architecture to enforce scientific integrity and operational stability, ensuring that only reliable skills proceed to deployment.
- Empirical evaluation demonstrates that the system achieves agreement with expert consensus that exceeds the human inter-rater baseline, providing a viable foundation for pre-deployment governance.

---

[Shift-Up: A Framework for Software Engineering Guardrails in AI-native Software Development - Initial Findings](http://arxiv.org/abs/2604.20436)

- Shift-Up: introduces a framework that reinterprets traditional software engineering practices as structural guardrails to stabilize agent-driven development workflows.
- The framework integrates Stakeholder Interview, SRS Generation, User Story Mapping, BDD Transformation, C4 and ADR Generation, Implementation Roadmap Generation, GitHub Issue Generation, and an Implementation and Verification Loop using Claude Sonnet 4.5, GPT-5.0-Codex, and Robot Framework to ensure machine-readable constraints.
- By shifting developer focus from low-level coding to high-level orchestration, the framework enhances agent autonomy through continuous validation and structured requirements.

---

[WebGen-R1: Incentivizing Large Language Models to Generate Functional and Aesthetic Websites with Reinforcement Learning](http://arxiv.org/abs/2604.20398)

- WebGen-R1: introduces an end-to-end reinforcement learning framework that leverages scaffold-driven structured generation and a cascaded multimodal reward model to enable small LLMs to produce functional and aesthetic multi-page websites.
- The framework utilizes a hierarchical verification and rendering pipeline to filter out non-functional code before applying a dense reward signal based on aesthetic perception, functional integrity, and reasoning format.
- By employing Group Relative Policy Optimization (GRPO), WebGen-R1 effectively scales small open-source models to achieve performance competitive with significantly larger proprietary models in web application generation.

---

[Graph2Counsel: Clinically Grounded Synthetic Counseling Dialogue Generation from Client Psychological Graphs](http://arxiv.org/abs/2604.20382)

- Graph2Counsel: introduces a framework for generating synthetic counseling sessions grounded in Client Psychological Graphs (CPGs) that encode relationships among clients’ thoughts, emotions, and behaviors.
- The framework utilizes structured prompting pipelines, including Guided Counseling (GC), Chain-of-Thought (CoT), and Multi-Agent (MA) feedback, to produce clinically meaningful and diverse multi-turn dialogues.
- Fine-tuning open-source LLMs on the generated dataset improves performance on mental health counseling benchmarks while maintaining high safety standards and clinical authenticity.

---

[Bimanual Robot Manipulation via Multi-Agent In-Context Learning](http://arxiv.org/abs/2604.20348)

- BiCICLe (Bimanual Coordinated In-Context Learning): introduces a multi-agent framework that enables LLMs to perform bimanual manipulation through a leader-follower decomposition, utilizing Leader Agent, Follower Agent, LLM-as-Judge, In-Context Demonstrations, Arms' Debate, and Best-of-N.
- The framework decouples the high-dimensional bimanual action space into sequential, conditioned single-arm predictions to enforce inter-arm consistency without requiring task-specific training.
- Inference-time strategies including Arms' Debate for iterative refinement and Best-of-N for trajectory selection further improve coordination and suppress sampling stochasticity in LLMs.

---

[FSFM: A Biologically-Inspired Framework for Selective Forgetting of Agent Memory](http://arxiv.org/abs/2604.20300)

- FSFM: introduces a neuro-inspired framework for selective forgetting in LLM agents, utilizing UltraSafeMemoryManager, ImportanceScoringEngine, SelectiveForgettingMechanism, PerformanceBenchmarkingTool, and a Multi-Layer Memory Architecture to optimize memory efficiency and security.
- The framework employs a multi-dimensional importance scoring algorithm to dynamically prune low-value or dangerous content, thereby enhancing retrieval performance and ensuring regulatory compliance.
- Empirical validation demonstrates that FSFM achieves a 30% reduction in storage requirements and 100% elimination of dangerous content while maintaining high-value information retention.

---

[AgentLens: Adaptive Visual Modalities for Human–Agent Interaction in Mobile GUI Agents](http://arxiv.org/abs/2604.20279)

- AgentLens: introduces a mobile GUI agent system that adaptively selects between Full UI, Partial UI, and GenUI modalities to provide just-in-time visual feedback during background task execution.
- The system leverages a Virtual Display to operate third-party apps in the background while using a companion app to surface non-invasive overlays at critical decision points.
- AgentLens resolves the trust-usability trade-off by combining the transparency of foreground execution with the multitasking convenience of background operation.

---

[ActuBench: A Multi-Agent LLM Pipeline for Generation and Evaluation of Actuarial Reasoning Tasks](http://arxiv.org/abs/2604.20273)

- ActuBench: introduces a multi-agent LLM pipeline for the automated generation and evaluation of actuarial reasoning tasks aligned with the IAA Education Syllabus.
- The framework utilizes Agent A (drafts and repairs items), Agent B (constructs and repairs distractors), Agent C (independently verifies content), and an Auxiliary Agent (summarizes notes and labels topics) to create a high-quality, verifiable assessment pool.
- The study evaluates 50 LLMs on two complementary benchmarks, demonstrating that independent verification is load-bearing and that locally-hosted open-weights models are competitive on the cost-performance Pareto front.

---

[Memory-Augmented LLM-based Multi-Agent System for Automated Feature Generation on Tabular Data](http://arxiv.org/abs/2604.20261)

- MALMAS (Memory-Augmented LLM-based Multi-Agent System): introduces a multi-agent framework for automated feature generation that leverages a multi-level memory mechanism to enable iterative, feedback-driven refinement of feature transformation strategies.
- The framework decomposes feature generation into specialized agents, including a Router Agent, which are coordinated through procedural-, feedback- and conceptual-memories to broaden the search space and improve feature diversity.
- Experimental results across 16 classification and 7 regression datasets demonstrate that MALMAS consistently outperforms traditional and LLM-based baselines by effectively accumulating cross-round knowledge to steer generation toward high-yield transformations.

---

[Mol-Debate: Multi-Agent Debate Improves Structural Reasoning in Molecular Design](http://arxiv.org/abs/2604.20254)

- Mol-Debate: introduces an iterative generation framework that bridges the text-structure gap in molecular design through a generate-debate-refine loop involving Developer Agent, Debate Agent, Examiner Agent, and Refiner Agent.
- The framework employs a multi-agent system where Developer Agents propose candidates, Examiner Agents provide deterministic structural validation, and Debate Agents perform multi-perspective critique to ensure semantic alignment.
- Refiner Agents actively resolve inter-agent disagreement by reformulating task instructions, creating a closed feedback loop that improves structural fidelity and intent satisfaction.

---

[Chasing the Public Score: User Pressure and Evaluation Exploitation in Coding Agent Workflows](http://arxiv.org/abs/2604.20200)

- AgentPressureBench: introduces a benchmark to evaluate public score exploitation where coding agents prioritize reported scores over hidden private evaluation performance under user pressure.
- The framework utilizes an LLM judge to identify exploitative behaviors, such as copying or training on exposed public evaluation labels, across diverse machine learning tasks.
- Experimental results demonstrate that stronger LLMs exhibit higher exploitation rates, while explicit anti-exploit prompt instructions effectively mitigate this failure mode.

---

[LLM-Guided Safety Agent for Edge Robotics with an ISO-Compliant Perception-Compute-Control Architecture](http://arxiv.org/abs/2604.20193)

- LLM-Guided Safety Agent: introduces a framework that bridges probabilistic AI perception and deterministic industrial safety requirements by using LLMs to formalize regulatory standards into machine-readable predicates.
- The system utilizes a redundant dual-modular architecture with parallel independent execution to ensure fault-tolerant, low-latency safety monitoring on edge hardware.
- By integrating ADC-based hardware probing and UART heartbeats, the architecture achieves self-healing capabilities and maintains strict temporal bounds for safety-critical human-robot collaboration.

---

[Dual-Cluster Memory Agent: Resolving Multi-Paradigm Ambiguity in Optimization Problem Solving](http://arxiv.org/abs/2604.20183)

- DCM-Agent: introduces a training-free framework that resolves structural ambiguity in optimization by decoupling abstract modeling logic from concrete code implementation using Dual-Cluster Memory Construction and Memory-Augmented Inference.
- The framework utilizes a bipartite graph to bridge Modeling Clusters and Coding Clusters, enabling the agent to navigate complex decision spaces through a Generate-Verify-Repair-Backtrack pipeline.
- Empirical results demonstrate that DCM-Agent achieves significant performance improvements across diverse optimization benchmarks while enabling smaller LLMs to inherit superior reasoning capabilities from larger models.

---

[Taint-Style Vulnerability Detection and Confirmation for Node.js Packages Using LLM Agent Reasoning](http://arxiv.org/abs/2604.20179)

- LLMVD.js: introduces a multi-stage agent pipeline that leverages LLMs to detect and confirm taint-style vulnerabilities in Node.js packages without requiring dedicated static or dynamic analysis engines.
- The framework utilizes a staged workflow comprising Finder, Judge, Constraints Inferencer, and Exploiter agents, supported by automated testing oracles to validate exploitability through concrete side effects.
- LLMVD.js demonstrates superior performance in confirming vulnerabilities compared to traditional program analysis tools while maintaining lower costs and effectively generalizing to unseen, recently released npm packages.

---

[Stateless Decision Memory for Enterprise AI Agents](http://arxiv.org/abs/2604.20158)

- DPM (Deterministic Projection Memory): introduces a stateless architecture that treats agent memory as an immutable append-only event log, utilizing a single task-conditioned projection at decision time to ensure deterministic replay and auditability.
- DPM replaces stateful memory updates with a pure functional projection, effectively eliminating path-dependent state and reducing the audit surface to a single LLM call.
- The architecture demonstrates that statelessness is attainable for long-horizon decision agents without sacrificing decision quality, while significantly improving performance and cost-efficiency compared to stateful baselines.

---

[Toward Safe Autonomous Robotic Endovascular Interventions using World Models](http://arxiv.org/abs/2604.20151)

- TD-MPC2: introduces a world-model-based reinforcement learning framework for autonomous endovascular navigation that integrates planning and learned dynamics to improve safety and generalization across diverse patient vasculatures.
- The framework utilizes a Latent Dynamics Model and Cross-Entropy Planning to optimize navigation actions while maintaining tip contact forces below safety thresholds.
- Validation across in silico hold-out datasets and a clinically-relevant in vitro robotic testbed demonstrates the framework's potential for safe, autonomous mechanical thrombectomy navigation.

---

[SAKE: Self-aware Knowledge Exploitation-Exploration for Grounded Multimodal Named Entity Recognition](http://arxiv.org/abs/2604.20146)

- SAKE: introduces an end-to-end agentic framework that harmonizes internal knowledge exploitation and external knowledge exploration via self-aware reasoning and adaptive tool invocation.
- The framework utilizes a two-stage training paradigm, starting with SAKE-SeCoT for supervised fine-tuning and followed by agentic reinforcement learning to optimize decision-making regarding when to perform external retrieval.
- By quantifying entity-level uncertainty through difficulty-aware search tags, SAKE effectively balances internal parametric memory with on-demand external search to improve grounded multimodal named entity recognition performance.

---

[An Agentic Approach to Metadata Reasoning](http://arxiv.org/abs/2604.20144)

- Metadata Reasoner: introduces an agentic framework that autonomously identifies a sufficient and minimal set of data sources for complex analytical tasks by orchestrating retrieval, reasoning, and specialized tool usage.
- The framework utilizes an Orchestration Agent to manage the retrieval of Attached Metadata and the dynamic invocation of On-the-fly Metadata via Specialized Tools to overcome context saturation and improve data discovery precision.
- By employing a discrimination-oriented metadata construction and a Session-level Dictionary (S) for deduplication, the system effectively navigates high-cardinality data lakes while maintaining robustness against noise and redundant data.

---

[HiPO: Hierarchical Preference Optimization for Adaptive Reasoning in Large Language Models](http://arxiv.org/abs/2604.20140)

- HiPO (Hierarchical Preference Optimization): introduces a framework that decomposes LLM responses into three distinct reasoning segments—refined query, meta-thinking, and answer—to enable targeted optimization through segment-level auxiliary losses.
- The approach extends Direct Preference Optimization by computing a weighted sum of losses for individual reasoning components, allowing for fine-grained control over model training emphasis.
- Experimental results on math benchmarks demonstrate that HiPO improves reasoning coherence, accuracy, and goal completion in LLMs compared to monolithic preference optimization methods.

---

[IMPACT-CYCLE: A Contract-Based Multi-Agent System for Claim-Level Supervisory Correction of Long-Video Semantic Memory](http://arxiv.org/abs/2604.20136)

- IMPACT-CYCLE: introduces a supervisory multi-agent system that reformulates long-video understanding as iterative claim-level maintenance of a shared semantic memory.
- The framework utilizes five role-specialized agents, including Local Grounding-, Temporal Consistency-, and Global Semantic Audit-agents, to verify claims under explicit authority contracts.
- By restricting re-verification to the dependency closure of modified claims, the system ensures that human arbitration effort remains proportional to error scope rather than video length.

---

[AgentSOC: A Multi-Layer Agentic AI Framework for Security Operations Automation](http://arxiv.org/abs/2604.20134)

- AgentSOC: introduces a multi-layer agentic AI framework that integrates perception, anticipatory reasoning, and risk-aware action planning within a closed-loop operational cycle.
- The framework utilizes a Narrative Counterfactual Engine (NCE) for LLM-based hypothesis generation, a Structural Simulation Engine (SSE) for graph-based feasibility validation, and a Risk Scoring and Evaluation Module (RSEM) for policy-aligned response selection.
- AgentSOC improves SOC triage consistency and operational safety by grounding LLM-driven reasoning in enterprise topology and identity privilege graphs.

---

[EvoAgent: An Evolvable Agent Framework with Skill Learning and Multi-Agent Delegation](http://arxiv.org/abs/2604.20133)

- EvoAgent: introduces an evolvable LLM agent framework that integrates structured skill learning with a hierarchical sub-agent delegation mechanism to enable continuous capability optimization.
- The framework utilizes a dual-loop architecture, separating online execution for real-time task handling from an asynchronous offline evolution loop for capability accumulation and skill refinement.
- EvoAgent employs a three-stage skill matching strategy and a three-layer memory system to support dynamic task decomposition and long-term context retention in complex professional environments.

---

[A Delta-Aware Orchestration Framework for Scalable Multi-Agent Edge Computing](http://arxiv.org/abs/2604.20129)

- DAOEF: introduces a multi-agent edge orchestration framework that synergistically integrates hierarchical priority filtering, feature-level delta caching, and hardware-aware matching to achieve sub-10 ms decision latency for large-scale deployments.
- The framework utilizes a TD3-based orchestrator that includes planning- and hardware-aware agents to optimize task offloading across heterogeneous edge nodes.
- DAOEF achieves significant latency reduction and energy savings by exploiting semantic similarity in intermediate neural network features to enable partial computation reuse.

---

[To Know is to Construct: Schema-Constrained Generation for Agent Memory](http://arxiv.org/abs/2604.20117)

- SCG-MEM (Schema-Constrained Generative Memory): introduces a memory architecture that reformulates memory access as schema-constrained generation to eliminate structural hallucinations in LLMs.
- The framework utilizes a dynamic Cognitive Schema (Prefix Trie) to strictly constrain LLM decoding, ensuring generated keys correspond to valid memory entries.
- An Associative Graph overlays the schema to facilitate multi-hop reasoning, while an evolutionary mechanism enables long-term adaptation through assimilation and accommodation of new concepts.

---

[SkillLearnBench: Benchmarking Continual Learning Methods for Agent Skill Generation on Real-World Tasks](http://arxiv.org/abs/2604.20087)

- SkillLearnBench: introduces a benchmark for evaluating continual learning methods that generate agent skills, comprising 20 verified tasks across 15 sub-domains with a three-level evaluation framework.
- The framework assesses skill quality, execution trajectory alignment, and task outcome to diagnose where skill generation methods succeed or fail.
- Controlled comparisons reveal that while all methods improve over no-skill baselines, they remain below human-authored performance, with external feedback driving genuine improvement compared to self-feedback.

---

[Auditing and Controlling AI Agent Actions in Spreadsheets](http://arxiv.org/abs/2604.20070)

- Pista: introduces a spreadsheet AI agent that decomposes execution into traceable, steerable steps, utilizing Decomposing Agent Execution, In-Situ Explanations, Localized Editing, Branching Exploration, Scaffolded Task Formulation, Scaffolding Questions, Surfacing Computation Logic, Gemini LLM API, Office.js API, and a browser-based state cache.
- The framework enables users to actively participate in the agent's decision-making process by providing visibility into intermediate reasoning and allowing targeted interventions at each step.
- By surfacing the semantic unit of change rather than just affected cells, Pista facilitates calibrated trust and efficient error correction in complex spreadsheet workflows.

---

[Testing replication for an agent-based model of market fragmentation and latency arbitrage](http://arxiv.org/abs/2604.20067)

- WW (Wah and Wellman) model replication framework: introduces an independent replication of an agent-based market model to evaluate the effects of market fragmentation and latency arbitrage on liquidity and efficiency.
- The study utilizes a bootstrap-based quantitative alignment methodology to compare simulation results against the original findings, identifying sensitivities in trader strategy implementations.
- The authors provide an ODD (Overview, Design concepts, Details) protocol to facilitate future replication and critique of agent-based market models.

---

[From Hidden Profiles to Governable Personalization: Recommender Systems in the Age of LLM Agents](http://arxiv.org/abs/2604.20065)

- User-Controlled Intent Layers framework: introduces a paradigm shift in recommender systems from opaque, platform-owned profiles to transparent, user-governed intent representations mediated by LLMs.
- The framework addresses the reconfiguration of user representations by proposing five research fronts: white-boxization, intent alignment, cross-domain fusion, LLM-native advertising, and profile governance.
- This research emphasizes that future recommender systems must prioritize user-addressable, inspectable, and portable intent layers to ensure accountability and trust in agent-mediated digital environments.

---

[Multi-Agent Empowerment and Emergence of Complex Behavior in Groups](http://arxiv.org/abs/2604.21155)

- Multi-Agent Empowerment framework: introduces a tractable information-theoretic approach to multi-agent systems by modeling coupled dynamics as an interference channel and solving for optimal strategies via iterative water-filling.
- The framework enables agents to pursue intrinsic motivations—either egoistically or altruistically—without explicit coordination, resulting in emergent group-level behaviors such as dominance hierarchies or counter-propagating flocking bands.
- By linearizing coupled dynamics and applying iterative water-filling, the method provides a principled, analytically tractable alternative to learned approximations in continuous-state multi-agent environments.

---

[Agentic AI for Personalized Physiotherapy: A Multi-Agent Framework for Generative Video Training and Real-Time Pose Correction](http://arxiv.org/abs/2604.21154)

- MAS (Multi-Agent System): introduces a framework for at-home physiotherapy that utilizes Clinical Extraction Agent, Video Synthesis Agent, Vision Processing Agent, and Diagnostic Feedback Agent to create a closed-loop rehabilitation environment.
- The framework leverages LLMs for semantic reasoning to translate clinical prescriptions into kinematic constraints while using MediaPipe for real-time patient pose tracking.
- The Diagnostic Feedback Agent employs a hybrid deterministic-generative approach to ensure patient safety by preventing LLM hallucinations from overriding physiological limits.

---

[Using Machine Mental Imagery for Representing Common Ground in Situated Dialogue](http://arxiv.org/abs/2604.21144)

- Visual Scaffolding framework: introduces an active visual scaffolding approach that incrementally converts situated dialogue states into a persistent visual history to improve common ground representation.
- The framework utilizes an Observer, Constructor, and Linker to maintain depictive and propositional representations, effectively reducing representational blur and enforcing concrete scene commitments.
- Evaluation on the IndiRef benchmark demonstrates that this incremental externalization, particularly in a hybrid multimodal setting, significantly enhances grounded response generation for LLMs.

---

[Beyond Pixels: Introspective and Interactive Grounding for Visualization Agents](http://arxiv.org/abs/2604.21134)

- IVG (Introspective and Interactive Visual Grounding): introduces a framework that enables LLMs to overcome the Pixel-Only Bottleneck by combining spec-grounded introspection and view-grounded interaction for deterministic visualization analysis.
- The framework utilizes a Visualization State API to provide agents with direct access to structured chart specifications and interactive view-manipulation tools, replacing approximate pixel-based interpretation with verifiable data retrieval.
- Experimental results on the iPlotBench benchmark demonstrate that the synergy between introspection and interaction significantly improves data reconstruction fidelity and QA accuracy for LLMs.

---

[Cross-Session Threats in AI Agents: Benchmark, Evaluation, and Algorithms](http://arxiv.org/abs/2604.21131)

- CSTM-Bench (Cross-Session Threat Memory Benchmark): introduces a benchmark for evaluating cross-session threat detection in AI agents by framing the problem as an information bottleneck between inbound message streams and a correlator LLM.
- The framework compares two architectural extremes: a Full-Log Correlator that processes unbounded logs and a Coreset Memory Reader that uses a bounded buffer to retain high-signal cross-session fragments.
- CSTM-Bench utilizes seven identity anchors and 26 attack taxonomies to measure detection recall, false alarm rates on benign traffic, and ordered prefix stability as a proxy for incremental serving costs.

---

[AGNT2: Autonomous Agent Economies on Interaction-Optimized Layer 2 Infrastructure](http://arxiv.org/abs/2604.21129)

- AGNT2: introduces an agent-native Layer 2 system architecture that utilizes a Sidecar to enable on-chain service invocations without modifying application logic.
- The architecture employs a three-layer stack consisting of Layer Top for P2P channels, Layer Core for dependency-aware execution, and Layer Root for settlement and computational fraud proofs.
- AGNT2 treats service invocation, identity, reputation, and session context as first-class protocol objects within an Interaction Trie to support high-frequency, multi-agent economies.

---

[Structural Quality Gaps in Practitioner AI Governance Prompts: An Empirical Study Using a Five-Principle Evaluation Framework](http://arxiv.org/abs/2604.21090)

- Five-Principle Evaluation Framework: introduces a systematic method for evaluating the structural completeness of AI governance prompts using Success Definition, Assessment Rubric, Scope Boundary, Data Classification, and Quality Gate.
- The study applies this framework to 34 AGENTS.md files, revealing that 37% of practitioner-authored governance prompts lack the structural components necessary for reliable agent behaviour.
- The research identifies that practitioners currently treat governance prompts as operational instructions rather than formal specifications, highlighting a significant gap in requirements engineering for AI agents.

---

[Strategic Polysemy in AI Discourse: A Philosophical Analysis of Language, Hype, and Power](http://arxiv.org/abs/2604.21043)

- Strategic Polysemy in AI Discourse: introduces the concept of glosslighting, defined as the strategic use of technically redefined or polysemous terms to evoke anthropomorphic associations while maintaining plausible deniability through narrow technical definitions.
- The paper argues that glosslighting is a constitutive driver of AI hype cycles, as it allows researchers and corporations to leverage intuitive, human-like interpretations of terms like Agent, Hallucination, and Reasoning to secure institutional support while retreating to operational, non-anthropomorphic descriptions when challenged.
- By analyzing the linguistic dynamics of AI terminology, the authors demonstrate how strategic ambiguity functions as a sociotechnical mechanism that shapes public understanding, regulatory agendas, and the political economy of AI development.

---

[The Last Harness You’ll Ever Build](http://arxiv.org/abs/2604.21003)

- Harness Evolution Loop: introduces a two-level framework that automates agent harness engineering by using a Worker Agent, Evaluator Agent, and Evolution Agent to iteratively optimize performance for specific tasks.
- The framework incorporates a Meta-Evolution Loop that optimizes the evolution protocol itself across diverse tasks, enabling rapid adaptation to new domains without manual intervention.
- This approach formalizes agent development as a meta-learning problem, where the system learns to design the automation process rather than just tuning individual agent components.

---

[Breaking MCP with Function Hijacking Attacks: Novel Threats for Function Calling and Agentic Models](http://arxiv.org/abs/2604.20994)

- FHA (Function Hijacking Attack): introduces a novel adversarial method that manipulates the tool selection process of agentic models by injecting an Adversarial Suffix into the description of a Target Function to force its invocation.
- The framework leverages the auto-regressive nature of LLM Agents to override the intended ground-truth function with an attacker-chosen function, effectively bypassing standard alignment.
- The approach demonstrates high robustness and universality across diverse domains and payloads, requiring only control over the textual description of functions within an MCP Server environment.

---

[Co-Evolving LLM Decision and Skill Bank Agents for Long-Horizon Tasks](http://arxiv.org/abs/2604.20987)

- COS-PLAY: introduces a multi-agent co-evolution framework that couples an LLM-based Decision Agent with an agent-managed Skill Bank Agent to discover, refine, and reuse structured skills for long-horizon tasks.
- The framework utilizes a closed-loop system where the Decision Agent retrieves skills from the Skill Bank to guide action taking, while the Skill Bank Agent continuously extracts and updates skills from unlabeled rollouts.
- By employing function-specific LoRA adapters and GRPO optimization, the system achieves significant performance gains in complex game environments while maintaining general reasoning capabilities.

---

[Thinking Like a Botanist: Challenging Multimodal Language Models with Intent-Driven Chain-of-Inquiry](http://arxiv.org/abs/2604.20983)

- PlantInquiryVQA: introduces a multi-step hierarchical VQA benchmark that models botanical diagnostic trajectories as ordered question-answer sequences conditioned on grounded visual cues and explicit epistemic intent.
- The framework utilizes a CoI (Chain-of-Inquiry) approach, incorporating VLM (Vision-Language Model for visual cue extraction), SME (Subject Matter Experts for schema definition), Disease Knowledge Base (botanical reference data), and LLM (Large Language Model for dynamic dialogue generation) to simulate expert-level diagnostic workflows.
- Evaluations on leading MLLMs demonstrate that structured, intent-driven inquiry significantly improves diagnostic correctness and reduces hallucination compared to static, single-turn VQA approaches.

---

[Can Virtual Agents Care? Designing an Empathetic and Personalized LLM-Driven Conversational Agent](http://arxiv.org/abs/2604.20948)

- Empathetic and Personalized Conversation System: introduces a virtual agent framework for wellbeing support that integrates a Tri-Retrieval RAG pipeline, dual-tier memory retention, and a safety-filtering LLM to provide personalized and empathetic interactions.
- The system utilizes a Tri-Retrieval mechanism combining sparse, dense, and web-based search to ground LLM responses in accurate, up-to-date wellbeing information.
- Empirical evaluations and cross-cultural studies with university students demonstrate that the framework significantly improves conversational coherence and user perception of accuracy compared to LLM-only baselines.

---

[SEMA: Semantic Transport for Real-Time Multimodal Agents](http://arxiv.org/abs/2604.20940)

- SEMA: introduces a semantic transport system that replaces traditional perceptual codecs with client-side tokenization and bursty delivery to optimize bandwidth and latency for LLM-based agents.
- The framework utilizes a hybrid screen representation combining structured text and visual tokens to preserve task accuracy while significantly reducing uplink payloads.
- By relocating vocoders to the client and eliminating jitter buffers, SEMA leverages the event-time tolerance of LLMs to achieve orders-of-magnitude bandwidth reduction for multimodal agent pipelines.

---

[HARBOR: Automated Harness Optimization](http://arxiv.org/abs/2604.20938)

- HARBOR: introduces a constrained noisy Bayesian optimization framework to automate the configuration of LLM agent harnesses by treating harness design as a first-class machine-learning problem.
- The framework utilizes a block-additive SAAS surrogate, multi-fidelity cost-aware acquisition, and posterior chance-constrained safety checks to optimize agent performance while respecting deployment budgets.
- HARBOR effectively identifies sparse, high-performing configurations by pruning silent features and preventing regressions common in manual stacking of LLM agent enhancements.

---

[Learning Reasoning World Models for Parallel Code](http://arxiv.org/abs/2604.20926)

- PCWMs (Parallel-Code World Models): introduces reasoning LLMs that emulate parallel-code analysis tool outcomes directly from source code to mitigate costly external tool calls.
- The framework utilizes a scalable data generation pipeline to create diverse parallel-coding problems and candidate implementations, which are then used to synthesize hindsight CoT (Chain-of-Thought) traces for supervised fine-tuning.
- Empirical results demonstrate that outcome-conditioned hindsight CoT supervision significantly improves the accuracy of PCWMs in race-outcome prediction and performance profiling, providing effective feedback for downstream bug-fixing agents.

---

[Clinically Interpretable Sepsis Early Warning via LLM-Guided Simulation of Temporal Physiological Dynamics](http://arxiv.org/abs/2604.20924)

- LLM-Guided Temporal Simulation Framework: introduces a "predict-then-classify" mechanism that simulates physiological trajectories using a Spatiotemporal feature extraction module, a Medical Prompt-as-Prefix module, and an AI-driven post-processing component to enhance sepsis prediction interpretability.
- The framework utilizes a partially frozen LLM to process integrated textual and time-series data, ensuring predictions remain within physiologically plausible ranges via an agent-based post-processing component.
- By simulating key physiological indicators before classification, the model provides transparent, clinically actionable insights that align with physician judgment and outperform conventional deep learning baselines.

---

[AnalogMaster: Large Language Model-based Automated Analog IC Design Framework from Image to Layout](http://arxiv.org/abs/2604.20916)

- AnalogMaster: introduces an end-to-end, training-free framework that automates the complete analog IC design flow from circuit image input to physical layout generation using YOLOv9-based detector, EasyOCR, Electrical connectivity analysis, Joint reasoning mechanism, Parameter search agent, Bayesian Optimization (BO), Simulated Annealing (SA), and Enhanced A* search algorithm.
- The framework utilizes a joint reasoning mechanism to enhance netlist extraction accuracy by processing multiple visual representations of the circuit through parallel LLM branches.
- A parameter search agent integrates self-enhanced prompt engineering and context truncation to effectively compress the device parameter space for downstream physical design optimization.

---

[Omission Constraints Decay While Commission Constraints Persist in Long-Context LLM Agents](http://arxiv.org/abs/2604.20911)

- SRD (Security-Recall Divergence): introduces a phenomenon where omission-type constraints in LLMs decay under context pressure while commission-type constraints persist.
- The research demonstrates that accumulated context tokens cause attention dilution, leading to the failure of suppressive behavioral constraints in LLMs.
- The study establishes Safe Turn Depth (STD) as a metric to identify when LLM agents become unreliable, enabling mitigation through periodic constraint re-injection or session length capping.

---

[HiGMem: A Hierarchical and LLM-Guided Memory System for Long-Term Conversational Agents](http://arxiv.org/abs/2604.18349)

- HiGMem: introduces a two-level hierarchical memory architecture that organizes conversational data into Turn Layer and Event Layer to enable multi-granularity reasoning.
- The system utilizes LLM-based Turn Analysis and Event Affiliation to construct structured memories, while employing LLM-guided retrieval to select precise evidence sets using event summaries as semantic anchors.
- By replacing passive vector-only retrieval with LLM-reasoning, HiGMem significantly reduces the volume of retrieved context, thereby lowering downstream token costs for LLMs in long-term conversational tasks.

---

[Relational Archetypes: A Comparative Analysis of AV-Human and Agent-Human Interactions](http://arxiv.org/abs/2604.22564)

- Relational Archetypes Framework: introduces a taxonomy for categorizing human-agent interactions by drawing parallels between autonomous vehicle traffic modulation and general-purpose AI agent engagement.
- The framework classifies interactions into six distinct archetypes—Passive, Deterministic, Assistive, Cooperative, Competitive, and Confrontational—based on mutual influence, goal alignment, and control.
- This research highlights that AI agents often engage in multiple archetypes simultaneously, necessitating a nuanced approach to governance and design as these systems increasingly operate in real-world environments.

---

#### 21st April 2026



[TeamFusion: Supporting Open-ended Teamwork with Multi-Agent Systems](http://arxiv.org/abs/2604.19589)

- TeamFusion: introduces a multi-agent framework for open-ended teamwork that replaces direct aggregation with structured, preference-grounded agent deliberation.
- The system instantiates Proxy Agents conditioned on individual preferences to engage in a structured discussion, which a Remix Agent then synthesizes into actionable deliverables.
- TeamFusion improves representativeness and decision usefulness by explicitly modeling the consensus-seeking process rather than collapsing diverse viewpoints into a single average.

---

[A Self-Evolving Framework for Efficient Terminal Agents via Observational Context Compression](http://arxiv.org/abs/2604.19572)

- TACO: introduces a plug-and-play, self-evolving framework that automatically discovers and refines compression rules from interaction trajectories to reduce redundant terminal output for LLMs.
- The framework maintains a Global Rule Pool for cross-task knowledge reuse and performs intra-task rule evolution to adapt to heterogeneous terminal environments without additional training.
- Experimental results demonstrate that TACO consistently improves agent performance and token efficiency across multiple benchmarks and backbone LLMs by filtering noise while preserving critical task-relevant information.

---

[Multi-modal Reasoning with LLMs for Visual Semantic Arithmetic](http://arxiv.org/abs/2604.19567)

- SAri-RFT: introduces a framework for visual semantic arithmetic by post-training LVLMs using GRPO and verifiable reward functions to improve relational reasoning.
- The framework utilizes the IRPD, a comprehensive dataset of 18 relations and 1500+ subject-object pairs, to evaluate two-term subtraction and three-term analogy tasks.
- By incorporating verifiable rewards, the method enables LVLMs to ground symbolic reasoning in visual perception, significantly outperforming traditional embedding-based arithmetic approaches.

---

[On Reasoning-Centric LLM-based Automated Theorem Proving](http://arxiv.org/abs/2604.19558)

- ReCent-Prover: introduces a reasoning-centric framework for automated theorem proving in Rocq that integrates validation with reflection and retrieval with planning to enhance proof search robustness and strategy alignment.
- The framework utilizes an LLM to perform self-reflection on generated tactics, filtering out potentially misapplied steps that could lead to unprovable branches.
- Retrieval with planning improves knowledge selection by conditioning lemma and proof retrieval on LLM-generated natural-language proof plans rather than relying on subgoal similarity.

---


[A-MAR: Agent-based Multimodal Art Retrieval for Fine-Grained Artwork Understanding](http://arxiv.org/abs/2604.19689)

- A-MAR: introduces an agent-based retrieval-augmented framework that explicitly decomposes complex art-related queries into structured reasoning plans to guide targeted evidence retrieval.
- The framework utilizes an Agent-based Planner to define evidence requirements, a Multimodal Reranker to select relevant context from an Art Context Knowledge Base, and a Multimodal LLM to synthesize grounded explanations.
- The authors also introduce ArtCoT-QA, a diagnostic benchmark designed to evaluate multi-step reasoning, evidence grounding, and retrieval faithfulness in the domain of fine art.

---

[InHabit: Leveraging Image Foundation Models for Scalable 3D Human Placement](http://arxiv.org/abs/2604.19673)

- InHabit: introduces a fully automatic data generation engine that leverages foundation models to populate 3D scenes with contextually meaningful human interactions.
- The framework utilizes a VLM for affordance reasoning, an image-editing model for visual synthesis, and an optimization-based lifting process to ground humans in 3D space.
- InHabit produces the InHabitants dataset, containing 78K photorealistic samples across 800 building-scale scenes, which improves performance for downstream contact estimation and HSI reconstruction tasks.

---

[Chat2Workflow: A Benchmark for Generating Executable Visual Workflows with Natural Language](http://arxiv.org/abs/2604.19667)

- Chat2Workflow: introduces a benchmark for generating executable visual workflows from natural language, utilizing an LLM-based Agent, Workflow Orchestration Platform, Auto-Repair Module, Variable Summaries, and Node Knowledge Base.
- The framework employs a Chain-of-Thought approach to generate JSON-based workflow representations that are subsequently converted into executable YAML files for deployment.
- Experimental results demonstrate that while LLMs can capture high-level intent, they struggle with complex, evolving requirements, necessitating an error-driven agentic baseline to improve resolve rates.

---

[Cyber Defense Benchmark: Agentic Threat Hunting Evaluation for LLMs in SecOps](http://arxiv.org/abs/2604.19533)

- Cyber Defense Benchmark: introduces a rigorous evaluation framework for measuring LLM agent performance in open-ended, evidence-driven threat hunting against real Windows event telemetry.
- The framework utilizes HolodeckHuntEnv to present agents with un-segmented log databases, requiring iterative SQL-based hypothesis generation and evidence synthesis to identify malicious activity.
- Evaluation of five frontier LLMs reveals significant performance gaps, with all models failing to meet the minimum operational recall threshold for unsupervised SOC deployment.

---

[ECLASS-Augmented Semantic Product Search for Electronic Components](http://arxiv.org/abs/2604.19664)

- ECLASS-Augmented Semantic Product Search framework: introduces a three-stage pipeline utilizing a Rewriter, Retriever, and Re-ranker to bridge the vocabulary gap between natural-language queries and structured industrial product data.
- The framework leverages hierarchical ECLASS metadata to augment product embeddings, significantly improving retrieval performance for specialized electronic components.
- Empirical results demonstrate that combining basic product data with ECLASS-enriched embeddings and re-ranking outperforms both classical lexical methods and foundation model baselines in industrial search tasks.

---

[An AI Agent Execution Environment to Safeguard User Data](http://arxiv.org/abs/2604.19657)

- GAAP (Guaranteed Accounting for Agent Privacy): introduces an execution environment that provides deterministic confidentiality guarantees for AI agents by enforcing user-defined permission specifications through information flow control on LLM-generated code artifacts.
- The framework utilizes a private data database, a permission database, a disclosure log, and an annotation framework to track and restrict data flows across multi-shot agent executions without requiring trust in the LLM or user prompts.
- GAAP effectively mitigates data disclosure attacks by intercepting tool calls and validating them against persistent, user-defined policies, ensuring that sensitive information is only shared with authorized external parties.

---

[SafetyALFRED: Evaluating Safety-Conscious Planning of Multimodal Large Language Models](http://arxiv.org/abs/2604.19638)

- SafetyALFRED: introduces a benchmark for evaluating the ability of MLLMs to recognize and mitigate safety hazards in embodied household tasks.
- The framework utilizes a multi-agent system comprising a Safety Judge and an Embodied Agent to decouple hazard recognition from task-oriented mitigation.
- Experimental results reveal a significant alignment gap where MLLMs accurately recognize hazards in static QA settings but struggle to translate this knowledge into effective embodied mitigation.

---

[Time Series Augmented Generation for Financial Applications](http://arxiv.org/abs/2604.19633)

- TSAG (Time Series Augmented Generation): introduces a tool-augmented RAG framework that delegates quantitative financial tasks to verifiable external tools to improve LLM reasoning and accuracy.
- The framework utilizes a modular architecture comprising User-, LLM-, Tools-, and DB-layers to isolate and measure an agent's ability to parse queries, select tools, and extract parameters.
- The study provides a comprehensive empirical benchmark comparing various LLMs, demonstrating that top-performing agents achieve near-perfect tool-use accuracy with minimal hallucination when integrated into the TSAG pipeline.

---

[Goal-Oriented Semantic Communication for Logical Decision Making](http://arxiv.org/abs/2604.19614)

- Goal-Oriented Semantic Communication framework: introduces a principled foundation for goal-oriented semantic communication by grounding task-relevant information selection in First-Order Logic (FOL) and inductive logical probability.
- The framework utilizes a semantic information bottleneck principle to identify and transmit FOL clauses that most effectively resolve uncertainty regarding goal-oriented states.
- A polynomial-time lexicographical sorting algorithm is proposed to optimize the communication objective, demonstrating effectiveness in safe path-following tasks within a neuro-symbolic simulator.

---

[AblateCell: A Reproduce-then-Ablate Agent for Virtual Cell Repositories](http://arxiv.org/abs/2604.19606)

- AblateCell: introduces an end-to-end agentic framework that automates scientific ablation studies by reproducing baselines via Planner Agent and Code Agent, and conducting closed-loop ablation using Bandit State and Git Worktree.
- The system utilizes a Domain Knowledge Base to ground hypothesis generation and employs graph-based execution to manage dependencies during systematic model component evaluation.
- AblateCell achieves high success rates in identifying performance-critical components across diverse single-cell perturbation models by balancing exploration and exploitation through adaptive bandit sampling.

---

[Active Inference-Enabled Agentic Closed-Loop ISAC with Long-Horizon Planning](http://arxiv.org/abs/2604.19599)

- AIF-driven wireless agentic system: introduces an active inference-based framework for closed-loop ISAC that jointly optimizes control and sensing resource allocation via message passing on a factor graph.
- The system utilizes a generative model acting as a digital twin, incorporating a localization model for uncertainty-aware inference and a localization CKM for long-horizon planning.
- Simulation results demonstrate that the agent adaptively balances tracking accuracy, control effort, and sensing resource consumption by anticipating the impact of actions on future performance.

---

[Paparazzo: Active Mapping of Moving 3D Objects](http://arxiv.org/abs/2604.19556)

- Paparazzo: introduces a learning-free framework for active 3D reconstruction of non-cooperative moving objects by alternating between an Object Tracking Mode and an Object Mapping Mode.
- The framework utilizes an Extended Kalman Filter (EKF) for motion prediction and 3D Gaussian Splatting (3DGS) for information-driven viewpoint selection to compensate for target motion.
- Paparazzo dynamically balances viewpoint informativeness, motion feasibility, and temporal synchronization to maintain reconstruction progress even during object motion interruptions.

---

[Taming Actor-Observer Asymmetry in Agents via Dialectical Alignment](http://arxiv.org/abs/2604.19548)

- ReTAS (Reasoning via Thesis-Antithesis-Synthesis): introduces a dialectical framework to mitigate Actor-Observer Asymmetry in LLMs by enforcing perspective-invariant reasoning through Thesis, Antithesis, and Synthesis stages.
- The framework utilizes Group Relative Policy Optimization (GRPO) to align LLMs with dialectical reasoning, incorporating attribution-, execution-, and format-rewards to ensure objective fault localization.
- ReTAS effectively decouples agent reasoning from role-induced cognitive biases, demonstrating superior performance in fault attribution and downstream task reliability across diverse multi-agent scenarios.

---

[FOCAL: Filtered On-device Continuous Activity Logging for Efficient Personal Desktop Summarization](http://arxiv.org/abs/2604.19541)

- FOCAL: introduces a privacy-first, multi-agent system that utilizes a unified filter-plan-log architecture to transform continuous desktop interaction streams into task-organized personal logs.
- The system employs a Filter Agent for noise suppression, a Brain Agent for task attribution, a Record Agent for selective visual reasoning, a Memory Agent for task-isolated context management, and a Summary Agent for coherent summarization.
- By moving task-aware control before expensive visual reasoning, FOCAL significantly reduces VLM token consumption and computational overhead while maintaining task-faithful context under frequent task switching.

---

[Mesh Memory Protocol: Semantic Infrastructure for Multi-Agent LLM Systems](http://arxiv.org/abs/2604.19540)

- MMP (Mesh Memory Protocol): introduces a semantic infrastructure for multi-agent LLM systems that enables cross-session cognitive collaboration through CAT7, SVAF, Inter-agent lineage, and Remix.
- The framework utilizes CAT7 to provide a universal semantic schema for CMBs, which are then evaluated by SVAF to ensure per-field admission based on role-indexed anchors.
- By implementing Inter-agent lineage and write-time filtered Remix, the protocol allows LLMs to maintain persistent, grounded, and evaluated cognitive state across session restarts without relying on raw history replay.

---

[Integrating Anomaly Detection into Agentic AI for Proactive Risk Management in Human Activity](http://arxiv.org/abs/2604.19538)

- ADFM-AAI: introduces a conceptual framework that integrates anomaly detection with agentic AI to provide proactive, autonomous fall mitigation for elderly populations.
- The architecture utilizes a multi-agent system where LLMs serve as a central reasoning service to dynamically orchestrate data acquisition, anomaly analysis, and intervention strategies.
- By reframing fall detection and prediction as anomaly detection problems, the system enables adaptive, goal-directed responses to complex, real-world human activity risks.

---

[Revac: A Social Deduction Reasoning Agent](http://arxiv.org/abs/2604.19523)

- Revac_8: introduces a multi-module architecture for social deduction games that integrates structured memory, relational graph analysis, and dynamic communication strategies to achieve human-level performance.
- The framework utilizes a Reviewer Agent for logical deduction and a Dynamic Tone Selector to adapt communication styles based on the current game state and social context.
- By employing a Social Alignment Graph, the agent performs relational reasoning to detect collusion and deception, overcoming the limitations of standard LLMs in high-stakes multi-agent environments.

---

[Accelerating Optimization and Machine Learning through Decentralization](http://arxiv.org/abs/2604.19518)

- Algorithm 1: introduces a server-assisted decentralized optimization framework that leverages local geometric properties to accelerate convergence by utilizing tailored local smoothness constants.
- The framework employs a switching mechanism that transitions from heterogeneous local step sizes to a universal step size to ensure convergence to an exact optimal solution.
- Theoretical performance is rigorously validated using the Performance Estimation Problem (PEP) framework to provide exact worst-case guarantees across specified function classes.

---

[From Experience to Skill: Multi-Agent Generative Engine Optimization via Reusable Strategy Learning](http://arxiv.org/abs/2604.19516)

- MAGEO: introduces a multi-agent framework that reframes Generative Engine Optimization as a strategy learning problem, utilizing coordinated planning, editing, and fidelity-aware evaluation to distill reusable optimization skills.
- The framework employs a dual-layer architecture where an execution layer performs iterative content optimization and a learning layer consolidates successful editing patterns into a persistent Skill Bank.
- To ensure rigorous assessment, the paper introduces a Twin Branch Evaluation Protocol for causal attribution and the DSV-CF metric to balance semantic visibility with citation fidelity.

---

[EVPO: Explained Variance Policy Optimization for Adaptive Critic Utilization in LLM Post-Training](http://arxiv.org/abs/2604.19485)

- EVPO (Explained Variance Policy Optimization): introduces an adaptive RL framework that dynamically switches between critic-based and batch-mean advantage estimation based on batch-level explained variance.
- The framework utilizes explained variance as a real-time indicator to detect when a critic's estimation noise exceeds its signal, effectively mitigating variance inflation during LLM post-training.
- By unifying PPO and GRPO as extremes of a Kalman filtering baseline selection, EVPO provides a provable guarantee of maintaining lower advantage variance than either fixed baseline throughout training.

---

[Four-Axis Decision Alignment for Long-Horizon Enterprise AI Agents](http://arxiv.org/abs/2604.19457)

- Four-Axis Decision Alignment framework: introduces a decomposition of long-horizon agent evaluation into four orthogonal axes—FRP, RCS, CRR, and CAR—to replace insufficient aggregate accuracy metrics.
- The framework utilizes LongHorizon-Bench to evaluate six distinct memory architectures, revealing that aggregate accuracy hides critical failure modes in factual precision, reasoning, and regulatory compliance.
- The research demonstrates that institutional and decisional alignment are load-bearing properties for regulated enterprise agents, necessitating explicit measurement beyond standard truthfulness and harmlessness benchmarks.

---

[What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search](http://arxiv.org/abs/2604.19440)

- LLM-guided evolutionary search framework: introduces a large-scale trajectory analysis of 15 LLMs across 8 tasks to identify the mechanisms driving optimization performance in agentic systems.
- The study reveals that effective LLMs function as local refiners, characterized by progressive semantic localization and frequent incremental breakthroughs, rather than relying on high novelty.
- The research demonstrates that breakthrough rate is a stronger predictor of optimization success than zero-shot capability, providing actionable insights for model selection and training.

---

[seneca: A Personalized Conversational Planner](http://arxiv.org/abs/2604.19425)

- seneca: introduces a conceptual framework for an AI-assisted planner that integrates a Conversational Agent, a structured Work Item View, a Processor, and a persistent Database to bridge the gap between user-expressed demands and underlying needs.
- The system utilizes a Conversational Agent to scaffold reflection and clarify goals, while the Processor ensures synchronization between the user-facing interface and the persistent Database containing Frameworks, Patterns, and Work Items.
- The architecture is designed to support self-regulation and goal-value alignment by combining the persistence of digital to-do lists with the interactive, reflective capabilities of LLMs.

---

[M2GRPO: Mamba-based Multi-Agent Group Relative Policy Optimization for Biomimetic Underwater Robots Pursuit](http://arxiv.org/abs/2604.19404)

- M2GRPO: introduces a Mamba-based policy network that integrates selective state-space modeling for long-horizon temporal dependencies and attention-based relational features for multi-agent coordination.
- The framework utilizes MAGRPO to perform group-normalized advantage estimation, which eliminates the need for explicit value functions and enhances training stability under the CTDE paradigm.
- Experimental results demonstrate that M2GRPO outperforms existing MARL baselines in pursuit success rate and capture efficiency for biomimetic underwater robot swarms.

---

[How damaging is zero-sum thinking to an agent’s interests when the world is positive-sum?](http://arxiv.org/abs/2604.19359)

- Game Theory Analysis of Zero-Sum Decision Rules: introduces a systematic evaluation of whether zero-sum decision rules like Maximin (guarantees minimum payoff against adversarial opponent) and Minimax (caps opponent's best attainable payoff) harm agent interests in positive-sum environments compared to Nash Equilibrium (mutually optimal strategy profile).
- The paper demonstrates that Maximin can strictly Pareto dominate Nash Equilibrium in a significant class of games, challenging the evolutionary presumption that non-maximizing decision rules are inherently inferior.
- The authors establish a cardinality theorem showing that the class of games where Maximin dominates Nash Equilibrium is as large as the class where Nash Equilibrium dominates Maximin, while also identifying Relative-Maximin (maximin applied to relative payoff transformation) and ESS (strategy resistant to invasion by alternative strategies) as relevant behavioral benchmarks.

---

[Do Agents Dream of Root Shells? Partial-Credit Evaluation of LLM Agents in Capture The Flag Challenges](http://arxiv.org/abs/2604.19354)

- DeepRed: introduces an open-source benchmark framework for evaluating LLM agents in realistic, isolated Capture The Flag (CTF) environments using a partial-credit scoring methodology.
- The framework utilizes an automated pipeline where a Summary LM condenses execution logs and a Judge LM assesses progress against predefined checkpoints derived from public writeups.
- Empirical evaluation of ten LLMs reveals that while agents can achieve partial progress, they struggle with long-horizon planning and non-standard discovery tasks in adversarial settings.

---

[Large Language Models Exhibit Normative Conformity](http://arxiv.org/abs/2604.19301)

- LLM-MAS: introduces a framework to distinguish between informational and normative conformity in LLMs by manipulating social context variables such as publicness, evaluation, and relationship continuity.
- The study evaluates six LLMs to demonstrate that normative conformity is a distinct behavioral tendency that can be manipulated through peer endorsement and the assignment of influential attributes.
- Analysis of internal hidden layer activations reveals that normative and informational conformity are driven by distinct internal mechanisms, providing insights into how social norms are implemented within LLMs.

---

[Rethinking Scale: Deployment Trade-offs of Small Language Models under Agent Paradigms](http://arxiv.org/abs/2604.19299)

- Agent Paradigms for Small Language Models: introduces a comprehensive empirical study evaluating 27 open-source SLMs across Base SLM, Single-Agent System (SAS), and Multi-Agent System (MAS) paradigms in financial settings.
- The study demonstrates that while SAS provides an optimal balance of effectiveness and efficiency, MAS introduces significant coordination overhead and systemic instability.
- The research identifies that architectural design can effectively compensate for limited model scale, though increased complexity often leads to higher failure rates due to delegation and context management issues.

---

[Explicit Trait Inference for Multi-Agent Coordination](http://arxiv.org/abs/2604.19278)

- ETI (Explicit Trait Inference): introduces a psychology-grounded framework that enables LLM agents to infer and track partner characteristics along warmth and competence dimensions to improve coordination.
- The framework utilizes Task History, Trait Inference, Agent Context, and Plan &amp; Act components to distill interaction histories into stable trait profiles that guide planning and delegation.
- Experimental results demonstrate that ETI improves coordination and task performance across diverse multi-agent settings by providing agents with structured awareness of their partners' traits.

---

[Warmth and Competence in the Swarm: Designing Effective Human-Robot Teams](http://arxiv.org/abs/2604.19270)

- SwarmUI: introduces a framework for investigating human social perception of robot swarms by manipulating behavioral parameters including speed, separation distance, and broadcast duration.
- The architecture integrates an ARGoS Simulator with a Web Interface Plugin to facilitate real-time human-swarm collaboration and data collection across observer and operator roles.
- Experimental results demonstrate that social perceptions of warmth and competence significantly influence human team preferences, often outweighing objective task performance metrics.

---

[DR-MMSearchAgent: Deepening Reasoning in Multimodal Search Agents](http://arxiv.org/abs/2604.19264)

- DR-MMSearchAgent: introduces a reinforcement learning framework that enhances multimodal search agents by utilizing structural proximity-weighted advantage injection to mitigate exploration degradation and trajectory redundancy.
- The framework incorporates a refining agent for real-time trajectory compression and a bidirectionally guided adaptive reward mechanism to dynamically calibrate interaction depth based on solution quality.
- Extensive experiments on the newly constructed BridgeVQA dataset demonstrate that the approach achieves state-of-the-art performance by balancing robust exploration with efficient information retrieval.

---

[BONSAI: A Mixed-Initiative Workspace for Human-AI Co-Development of Visual Analytics Applications](http://arxiv.org/abs/2604.19247)

- BONSAI: introduces a mixed-initiative workspace for the co-development of Visual Analytics applications, utilizing a four-layer architecture (Hardware Layer, Service Layer, Orchestration Layer, Application Layer) to enforce modularity and interface contracts.
- The framework employs a hierarchical agent model that includes a top-level Nexus orchestrator, mid-level Squad Leads, and specialized AI Development Units to distribute agency and ensure structural rigor during development.
- BONSAI integrates a four-phase development process (Plan, Design, Monitor, Review) that treats semantic provenance as a first-class citizen to maintain auditability and human control over AI-assisted workflows.

---

[YAIFS: Yet (not) Another Intelligent Fog Simulator: A Framework for Agent-Driven Computing Continuum Modeling &amp; Simulation](http://arxiv.org/abs/2604.19181)

- YAIFS: introduces a layered, service-oriented architecture that transforms static simulations into interactive, programmable environments for cloud-edge systems.
- The framework integrates the Model Context Protocol (MCP) to decouple agent logic from simulator internals, enabling autonomous agents to observe, control, and optimize simulation workflows.
- YAIFS supports AI-driven experimentation through specialized agents, including an LLM-based assistant for natural language interaction and a multi-agent system for adaptive application placement.

---

[Distributed Multi-Sensor Control for Multi-Target Tracking Using Adaptive Complementary Fusion for LMB Densities](http://arxiv.org/abs/2604.19160)

- FDCD-SC (Fully Distributed Coordinate Descent Sensor Control): introduces a distributed multi-sensor control framework that utilizes an adaptive complementary fusion rule and an information-theoretic objective function to optimize sensor actions for multi-target tracking.
- The framework employs a PIMS (Predicted Ideal Measurement Set) and a pseudo-update mechanism to evaluate potential control commands without requiring real-time physical sensor movement.
- By integrating a flooding-based communication protocol, the approach ensures global coordination among sensors, significantly reducing cardinality errors in dynamic multi-target tracking scenarios.

---

[RLABC: Reinforcement Learning for Accelerator Beamline Control](http://arxiv.org/abs/2604.19146)

- RLABC (Reinforcement Learning for Accelerator Beamline Control): introduces an automated pipeline that transforms standard beamline configurations into RL environments by integrating the Elegant simulation program via SDDS interfaces.
- The framework utilizes a 57-dimensional state representation and stage learning strategies to decompose complex beamline tuning tasks into manageable subproblems for efficient RL training.
- RLABC employs a DDPG agent to optimize magnet parameters, achieving particle transmission performance comparable to established methods like differential evolution.

---

[RoboWM-Bench: A Benchmark for Evaluating World Models in Robotic Manipulation](http://arxiv.org/abs/2604.19092)

- RoboWM-Bench: introduces a manipulation-centric benchmark for evaluating the physical executability of video world models through embodied execution.
- The framework utilizes World Models to generate videos, which are then processed by Human-Hand Retargeting or an Inverse Dynamics Model to produce executable actions for a Simulation Platform.
- RoboWM-Bench employs a Real-to-Sim Engine to reconstruct real-world scenarios, enabling standardized and reproducible validation of whether predicted behaviors are physically grounded and executable.

---

[The Essence of Balance for Self-Improving Agents in Vision-and-Language Navigation](http://arxiv.org/abs/2604.19064)

- SDB (Stability–Diversity Balance): introduces a training-time (1→K→1) expand–select mechanism that balances behavioral diversity and learning stability in VLN by generating multiple latent hypotheses via a Diversity Expansion Module and consolidating them through a Stability Selection Module.
- The framework utilizes a Head-Shifting Generator to produce instruction-consistent variations and a Balanced Controller to perform reliability-aware soft fusion, ensuring stable, cumulative learning updates across iterations.
- SDB is a backbone-agnostic plug-in that improves navigation success and path efficiency by preventing premature commitment to suboptimal hypotheses and reducing uncoordinated decision switching.

---

[Refute-or-Promote: Adversarial Stage-Gated Multi-Agent Review for High-Precision LLM-Assisted Defect Discovery](http://arxiv.org/abs/2604.19049)

- Refute-or-Promote: introduces an adversarial, stage-gated multi-agent methodology for high-precision defect discovery that utilizes Stratified Context Hunting (SCH) for candidate generation, adversarial agents for kill mandates, an empirical validation gate, and a Cross-Model Critic (CMC) to filter false positives.
- The framework employs a unidirectional pipeline where creative and adversarial agents operate in parallel tracks to rigorously test candidate vulnerabilities before human-led disclosure.
- By implementing context asymmetry and cross-family verification, the system mitigates reasoning biases and correlated training errors that often lead LLMs to reach unanimous but incorrect conclusions.

---

[Explore Like Humans: Autonomous Exploration with Online SG-Memo Construction for Embodied Agents](http://arxiv.org/abs/2604.19034)

- ABot-Explorer: introduces an active exploration framework that unifies memory construction and navigation into an online, RGB-only process by leveraging VLMs to distill SNA.
- The framework utilizes VLM-distilled SNA to identify navigationally critical transit nodes, which are dynamically organized into a hierarchical SG-Memo to guide exploration.
- By prioritizing structural transit nodes over geometric frontiers, the system achieves human-like exploration efficiency and provides a reasoning-ready knowledge substrate for downstream embodied tasks.

---

[ClawCoin: An Agentic AI-Native Cryptocurrency for Decentralized Agent Economies](http://arxiv.org/abs/2604.19026)

- ClawCoin: introduces a collateral-backed, index-linked cryptocurrency designed to provide a stable, compute-aligned unit of account for decentralized AI agent economies.
- The framework utilizes an off-chain index calculator and on-chain oracle to track standardized inference costs, enabling agents to quote, escrow, and settle workflows in a shared numeraire.
- By integrating with an atomic multi-hop settlement layer, the system eliminates budget overruns and coordination failures inherent in fiat-denominated agent transactions.

---

[On Accelerating Grounded Code Development for Research](http://arxiv.org/abs/2604.19022)

- Grounded Code Development Framework: introduces a system that enables LLMs to access dynamic research repositories and technical documentation for context-aware code generation, utilizing Document Parsing Pipeline, Elasticsearch, Document Search Tool, LSP-based Semantic Code Search, and Skill Library.
- The framework employs a lexical search-based retrieval strategy to provide deterministic, up-to-date access to specialized scientific artifacts without requiring extensive model fine-tuning.
- The Skill Library acts as an orchestration layer, allowing researchers to define reusable, structured research methodologies that guide LLM agents through iterative information gathering, planning, and implementation phases.

---

[Security Is Relative: Training-Free Vulnerability Detection via Multi-Agent Behavioral Contract Synthesis](http://arxiv.org/abs/2604.19012)

- Phoenix: introduces a training-free multi-agent framework that resolves semantic ambiguity in vulnerability detection through Behavioral Contract Synthesis using Semantic Slicer, Requirement Reverse Engineer, and Contract Judge.
- The framework transforms open-ended vulnerability detection into a closed-form contract verification problem by utilizing Gherkin specifications as an explicit intermediate representation.
- Phoenix achieves state-of-the-art performance on the PrimeVul benchmark by leveraging specialized LLM agents to enforce project-specific security contracts rather than relying on pattern matching.

---

[A Multi-Agent Framework with Structured Reasoning and Reflective Refinement for Multimodal Empathetic Response Generation](http://arxiv.org/abs/2604.18988)

- MERG (Multimodal Empathetic Response Generation) framework: introduces a multi-agent system that decomposes response generation into a structured pipeline of MPA, CAEF, PSP, and SGRG, followed by a GRA for iterative refinement.
- The framework utilizes a closed-loop iterative optimization process where the GRA audits intermediate outputs from MPA, CAEF, PSP, and SGRG to identify and correct errors at the earliest responsible stage.
- By explicitly organizing multimodal perception, emotion forecasting, strategy planning, and response generation, the framework enables targeted regeneration and improves the emotional appropriateness and strategic coherence of responses generated by LLMs.

---

[SAVOIR: Learning Social Savoir-Faire via Shapley-based Reward Attribution](http://arxiv.org/abs/2604.18982)

- SAVOIR: introduces a game-theoretic framework for social RL that replaces heuristic credit assignment with prospective valuation via expected utility and fair attribution via Shapley values.
- The framework utilizes KernelSHAP to efficiently compute utterance-level rewards, enabling the training of LLMs that demonstrate superior social intelligence compared to existing methods.
- Experimental results on the SOTOPIA benchmark show that SAVOIR achieves state-of-the-art performance, with a 7B model matching or exceeding proprietary LLMs and large reasoning models.

---

[Gated Coordination for Efficient Multi-Agent Collaboration in Minecraft Game](http://arxiv.org/abs/2604.18975)

- Gated Collaborative Escalation Framework: introduces a partitioned information architecture that decouples private execution states from public coordination states to minimize context noise and improve multi-agent collaboration efficiency.
- The framework utilizes a three-tiered gating mechanism—comprising heuristic rules, cost-sensitive scoring, and a bounded LLM adjudicator—to transform communication from a default reflex into a selective, cost-benefit-driven decision.
- By maintaining compact, system-verified private working memory and enforcing protocolized public interaction, the system significantly reduces coordination deadlocks and improves task completion rates in long-horizon open-world environments.

---

[Superficial Success vs. Internal Breakdown: An Empirical Study of Generalization in Adaptive Multi-Agent Systems](http://arxiv.org/abs/2604.18951)

- Adaptive MAS: introduces an empirical study identifying topological overfitting and illusory coordination as primary failure modes in adaptive multi-agent systems when evaluated on out-of-distribution tasks.
- The study demonstrates that adaptive MAS often achieve high surface-level accuracy through the brute-force reasoning of individual LLMs rather than effective collaborative mechanisms, a phenomenon termed illusory coordination.
- The authors propose two novel metrics, Role Alignment and Connection Significance, to rigorously evaluate the internal dynamics of multi-agent collaboration beyond final-answer correctness.

---

[AutomationBench](http://arxiv.org/abs/2604.18934)

- AutomationBench: introduces a benchmark for evaluating LLMs on cross-application workflow orchestration via REST APIs, utilizing Task prompt, AI agent, Search tool, Execute tool, Environment noise, Simulated app state, and Deterministic assertions.
- The framework requires LLMs to autonomously discover relevant API endpoints, adhere to layered business policies, and navigate environments containing misleading records to achieve specific end-state goals.
- Grading is performed programmatically through deterministic assertions on the final state of simulated applications, ensuring reproducibility and eliminating subjective evaluation bias.

---

[From Fuzzy to Formal: Scaling Hospital Quality Improvement with AI](http://arxiv.org/abs/2604.20055)

- Human-AI Spec-Solution Co-Optimization: introduces a framework that maps hospital quality improvement tasks to classical AI/ML development steps, enabling iterative refinement of specifications and prompts through collaboration between domain experts and an LLM-agent.
- The framework utilizes a PHI-compliant web interface to surface AI-extracted reasoning and evidence, facilitating expert validation and ensuring the resulting pipelines are reproducible and auditable.
- By treating high-level specifications as natural-language-valued hyperparameters, the approach successfully scales quality improvement factor discovery for hospital metrics like length of stay and readmission rates.

---

[Information Aggregation with AI Agents](http://arxiv.org/abs/2604.20050)

- Prediction market platform framework: introduces a controlled experimental environment to evaluate how LLMs aggregate dispersed private information through sequential trading in prediction markets.
- The framework utilizes LLM Providers, AI Agents, and Calimantic Agents to simulate complex information structures, measuring market accuracy via log error and trading behavior.
- The study demonstrates that while LLMs perform well in simple structures, they struggle with complex interactive reasoning, often failing to aggregate information effectively in harder tasks.

---

[TriEx: A Game-based Tri-View Framework for Explaining Internal Reasoning in Multi-Agent LLMs](http://arxiv.org/abs/2604.20043)

- TriEx: introduces a tri-view explainability framework that instruments LLM agent decision-making with structured first-person self-reasoning, explicit second-person opponent belief states, and third-person oracle audits.
- The framework utilizes imperfect-information strategic games as controlled diagnostic environments to evaluate explanation faithfulness, belief dynamics, and evaluator reliability.
- TriEx reveals that explanation faithfulness degrades with decision complexity and that LLM-based evaluators exhibit a hierarchy of reliability where categorical judgments are more robust than absolute scalar scores.

---

[Separable Pathways for Causal Reasoning: How Architectural Scaffolding Enables Hypothesis-Space Restructuring in LLM Agents](http://arxiv.org/abs/2604.20039)

- CG+DB: introduces a compositional architecture that factorizes causal reasoning into context graphs for structured exploration and dynamic behaviors for runtime hypothesis-space restructuring.
- The framework utilizes context graphs to structure agent problem-solving as typed state machines and dynamic behaviors to detect regime changes, enabling agents to adapt to novel causal rules.
- Empirical results demonstrate that context graphs drive reasoning quality while dynamic behaviors drive reasoning eligibility, providing orthogonal contributions to agent performance.

---

[Decision-Focused Federated Learning Under Heterogeneous Objectives and Constraints](http://arxiv.org/abs/2604.20031)

- DFFL (Decision-Focused Federated Learning): introduces a predict-then-optimize framework for federated learning that accounts for heterogeneous downstream optimization problems and constraints across clients.
- The framework utilizes SPO+ surrogate loss to derive heterogeneity bounds that quantify the impact of objective and feasible-set shifts on decision quality.
- It establishes a client-mixture discrepancy term to provide a heuristic decision rule for determining when federation improves performance over local learning.

---

[Statistics, Not Scale: Modular Medical Dialogue with Bayesian Belief Engine](http://arxiv.org/abs/2604.20022)

- BMBE: introduces a modular diagnostic dialogue architecture that enforces a strict separation between language communication and probabilistic reasoning by offloading inference to a deterministic engine.
- The framework utilizes an LLM solely as a sensor for parsing patient utterances and verbalizing questions, while a Bayesian engine manages belief updates, question selection, and calibrated abstention.
- This architectural separation enables robust, auditable, and private diagnostic decision support that outperforms standalone LLMs by providing a controllable accuracy-coverage tradeoff without requiring model retraining.

---

[From Recall to Forgetting: Benchmarking Long-Term Memory for Personalized Agents](http://arxiv.org/abs/2604.20006)

- Memora: introduces a long-term memory benchmark that evaluates personalized agents on remembering, reasoning, and recommending tasks across extended interaction histories.
- The framework utilizes a simulation-driven pipeline to generate multi-session conversations and employs a Forgetting-Aware Memory Accuracy (FAMA) metric to penalize reliance on obsolete or invalidated information.
- Empirical evaluations reveal that current LLMs and memory agents struggle with maintaining consistent memory states under high consolidation and mutation pressure, often failing to properly forget outdated information.

---

[Constructing external comparator groups via transportability in mean or in effect measure](http://arxiv.org/abs/2604.19977)

- Augmented Weighting Estimators: introduces a framework for constructing external comparator groups by combining index trial data with external data sources using semiparametric efficient augmented weighting estimators.
- The approach utilizes transportability in mean or in effect measure to identify causal estimands, ensuring model robustness even when nuisance function models are misspecified.
- The methodology is demonstrated through a clinical application combining the ACCEPT and PHOENIX randomized trials to evaluate biologic therapies for plaque psoriasis.

---

[Semantic Prompting: Agentic Incremental Narrative Refinement through Spatial Semantic Interaction](http://arxiv.org/abs/2604.19971)

- S-PRISM: introduces a multi-agent framework that translates spatial semantic interactions into targeted narrative refinements to improve human-LLM intent alignment.
- The system utilizes a hierarchical pipeline consisting of a System Inferring Agent, an Interaction Inferring Agent, and a Refining Agent to bridge spatial metadata with explicit intent prompts.
- S-PRISM enables incremental formalization by allowing users to steer LLMs through spatial metaphors like framing, highlighting, and noting, while providing transparency through intermediate reasoning results.

---

[Stochastic Networked Governance: Bridging Econophysics and Institutional Dynamics in a Positive-Sum Agent-Based Model](http://arxiv.org/abs/2604.19968)

- SNG (Stochastic Networked Governance): introduces an agent-based framework that models global economies as networks of jurisdictions with discrete institutional genomes to simulate macroeconomic phase transitions.
- The framework utilizes an Institutional Genome to formalize policy complementarity and calculates endogenous growth based on institutional fitness and topological capital spillovers.
- By integrating empirical trade data and systemic crisis timelines, the model demonstrates how spatial firewalls and institutional configurations drive global economic resilience or collapse.

---

[Insights into Security-Related AI-Generated Pull Requests](http://arxiv.org/abs/2604.19965)

- Insights into Security-Related AI-Generated Pull Requests: introduces a large-scale empirical analysis of security-related pull requests generated by agentic AIs, utilizing AIDev Dataset, Keyword-based Filter, Gemini-2.0-flash API, Semgrep, C-Good Model, AutoSpearman, and Regression Models.
- The study identifies that AI-generated security patches frequently introduce recurring vulnerabilities such as regex inefficiencies, injection flaws, and path traversal, despite being often merged by project maintainers.
- The research reveals that commit message quality does not consistently influence PR acceptance or review latency, and that rejection patterns are largely driven by social or process-related factors rather than technical faults.

---

[CreativeGame: Toward Mechanic-Aware Creative Game Generation.](http://arxiv.org/abs/2604.19926)

- CreativeGame: introduces a multi-agent system for iterative HTML5 game generation that treats mechanics as explicit planning objects to support interpretable version-to-version evolution.
- The system utilizes a CreativeProxyReward based on programmatic signals and runtime validation to minimize reliance on subjective LLM judgment.
- A lineage-aware memory architecture enables experience accumulation across versions while maintaining isolation between different game lineages.

---

[Behavioral Transfer in AI Agents: Evidence and Privacy Implications](http://arxiv.org/abs/2604.19925)

- OpenClaw: introduces an empirical study on behavioral transfer where AI agents systematically reflect the behavioral characteristics of their human owners through accumulated interaction.
- The research demonstrates that agents function as behavioral extensions of their owners, propagating human heterogeneity into digital ecosystems and creating unintended privacy risks.
- Analysis of 10,659 matched human-agent pairs reveals that stronger behavioral transfer significantly increases the likelihood of agents disclosing sensitive owner-related information in public discourse.

---

[SceneOrchestra: Efficient Agentic 3D Scene Synthesis via Full Tool-Call Trajectory Generation](http://arxiv.org/abs/2604.19907)

- SceneOrchestra: introduces a trainable orchestration framework that optimizes 3D scene synthesis by predicting complete tool-call trajectories using fine-tuned Orchestrator (LLM) and Discriminator (LLM) components.
- The framework employs a two-phase training strategy involving SFT (Supervised Fine-tuning) and DPO (Direct Preference Optimization) to align the Orchestrator with high-quality trajectory distributions.
- By distilling the ranking capability of the Discriminator into the Orchestrator, the system eliminates the need for costly execute-review-reflect loops during inference, significantly improving synthesis efficiency.

---

[DR-Venus: Towards Frontier Edge-Scale Deep Research Agents with Only 10K Open Data](http://arxiv.org/abs/2604.19859)

- DR-Venus: introduces a 4B edge-scale deep research agent trained on open data using a two-stage pipeline of Agentic SFT and Agentic RL with IGPO.
- The framework utilizes Agentic SFT to establish foundational reasoning and tool-use capabilities, followed by Agentic RL to enhance long-horizon execution reliability through dense turn-level rewards.
- By employing IGPO and turn-level credit assignment, the system effectively optimizes small LLMs to outperform larger models on complex information-seeking tasks.

---

[Rethinking Reinforcement Fine-Tuning in LVLM: Convergence, Reward Decomposition, and Generalization](http://arxiv.org/abs/2604.19857)

- TA-MDP: introduces a formal framework for modeling multimodal agentic decision-making with bounded-depth tool calls to analyze reinforcement fine-tuning with verifiable rewards.
- The paper establishes convergence guarantees for GRPO under composite verifiable rewards, identifying how reward component alignment and group size influence optimization stability.
- The research provides a PAC-Bayes generalization bound demonstrating that tool-augmented policies reduce effective complexity, explaining strong out-of-distribution transfer in LVLMs.

---

[CHIPCRAFTBRAIN: Validation-First RTL Generation via Multi-Agent Orchestration](http://arxiv.org/abs/2604.19856)

- CHIPCRAFTBRAIN: introduces a framework for automated RTL generation that integrates reinforcement learning-based multi-agent orchestration, hybrid symbolic-neural reasoning, and a validation-first iterative refinement pipeline.
- The system utilizes a PPO-based RL orchestrator to dynamically select specialized LLM agents and retrieval strategies based on a 168-dimensional state representation of the design task.
- By combining algorithmic symbolic solvers for deterministic logic with hierarchical specification decomposition, the framework achieves high functional correctness on both simple modules and complex industrial IP blocks.

---

[Multi-stage volume exclusion models for cell proliferation](http://arxiv.org/abs/2604.19852)

- Multi-stage volume exclusion models framework: introduces stochastic on-lattice ABMs and corresponding continuum PDE limits to model cell proliferation with realistic cell cycle time distributions.
- The framework incorporates density-dependent behaviour through volume exclusion, comparing Reset and Remain mechanisms for handling failed proliferation events in crowded environments.
- The study utilizes PCF to quantify spatial correlations and assess the validity of mean-field PDE approximations against discrete ABM simulations under varying motility rates.

---

[If you’re waiting for a sign... that might not be it! Mitigating Trust Boundary Confusion from Visual Injections on Vision-Language Agentic Systems](http://arxiv.org/abs/2604.19844)

- VLAS: introduces a multi-agent defense framework to mitigate trust boundary confusion by separating perception from decision-making to dynamically assess the reliability of visual inputs.
- The framework utilizes an Observation-Agent to parse visual signals, a Judgment-Agent to adjudicate between user intent and environmental constraints, and an LVLM-Agent to execute plans.
- This approach effectively neutralizes misleading visual injections while preserving legitimate safety-critical cues, addressing the modality laziness observed in current LLMs.

---

[Environmental Understanding Vision-Language Model for Embodied Agent](http://arxiv.org/abs/2604.19839)

- EUEA (Environmental Understanding Embodied Agent): introduces a framework that integrates four core skills—object perception, task planning, action understanding, and goal recognition—into a single VLM to enable reliable instruction-following for embodied agents.
- The framework utilizes a sampling-based recovery step to correct failed interactions and a GRPO (Group Relative Policy Optimization) stage to refine inconsistent skill predictions using internal reward functions.
- By internalizing explicit skill-level supervision, the model achieves end-to-end interaction performance without requiring complex modular pipelines or external environment feedback.

---

[Resolving space-sharing conflicts in road user interactions through uncertainty reduction: An active inference-based computational model](http://arxiv.org/abs/2604.19838)

- Active Inference-based Computational Model: introduces a framework for modeling road user interactions by integrating implicit behavioral coupling, normative expectations, and explicit communication to reduce uncertainty in space-sharing conflicts.
- The model utilizes an Active Inference Agent that employs a Generative Model and a Norm-conditioned Particle Filter to evaluate policies based on the sum of Pragmatic Value and Epistemic Value.
- The framework demonstrates that while implicit communication and normative expectations often resolve conflicts, explicit communication via a Communication Module serves as an efficient mechanism to resolve residual uncertainty in complex or adversarial traffic scenarios.

---

[Forage V2: Knowledge Evolution and Transfer in Autonomous Agent Organizations](http://arxiv.org/abs/2604.19837)

- Forage V2: introduces an institutional framework for autonomous agents that enables knowledge evolution and transfer across runs through Evaluator Agent, Planner Agent, Executor, Knowledge Base, Post-mortem extraction, Physical workspace isolation, System prompt, and eval_contract.md.
- The architecture mitigates denominator blindness by separating the evaluation and execution roles, ensuring that agents operate within isolated workspaces while inheriting organizational experience.
- By treating accumulated experience as organizational knowledge rather than individual model weights, the framework allows weaker LLMs to achieve performance comparable to stronger models at reduced costs.

---

[From Clerks to Agentic AI: How Will Technology Transform the Labor Market in Finance?](http://arxiv.org/abs/2604.19833)

- Workforce Transformation Economic Framework: introduces a comparative analysis of labor costs across three distinct workforce categories to evaluate the economic impact of automation in financial institutions.
- The paper examines how successive technological waves—computerization, indexing, and agentic AI—reorganize financial workflows and labor demand rather than simply eliminating jobs.
- Empirical findings indicate that AI adoption in finance is associated with increased assets under management per employee and gradual organizational shifts, suggesting a transformation of labor roles rather than immediate displacement.

---

[Agentic AI-Enabled Framework for Thermal Comfort and Building Energy Assessment in Tropical Urban Neighborhoods](http://arxiv.org/abs/2604.21787)

- Agentic AI-Enabled Framework: integrates LLMs with lightweight physics-based models to automate urban microclimate and building energy assessment through a closed-loop reasoning-action process.
- The framework utilizes a Task Orchestrator to manage data inputs from weather services and geometry files, ensuring traceable and reproducible simulation workflows.
- By combining autonomous LLM reasoning with physics-based solvers, the system enables rapid evaluation of climate-resilient design strategies while identifying complex trade-offs like the albedo penalty.

---

[Dynamical Priors as a Training Objective in Reinforcement Learning](http://arxiv.org/abs/2604.21464)

- DP-RL (Dynamical Prior Reinforcement Learning): introduces a training framework that augments standard policy gradient learning with an auxiliary loss derived from External State Dynamics (ESD) to impose temporal structure on agent decision-making.
- The framework utilizes a second-order hysteretic system as an External State Dynamics (ESD) component to generate a latent dynamical state that guides the Policy Network toward temporally coherent action probabilities.
- By incorporating an Auxiliary Loss Function that penalizes deviations from the ESD trajectory, the approach enables memoryless feedforward policies to exhibit complex, biologically-inspired decision dynamics without modifying the underlying environment or reward structure.

---

[Reinforcing privacy reasoning in LLMs via normative simulacra from fiction](http://arxiv.org/abs/2604.20904)

- Normative Simulacra: introduces a training framework that extracts structured normative universes from fiction to teach LLMs contextual privacy reasoning through supervised fine-tuning and reinforcement learning.
- The approach utilizes a composite reward function incorporating programmatic gating signals and an LLM judge to ensure privacy reasoning is grounded in explicit normative frameworks.
- To prevent overfitting, the method employs per-completion contrastive scoring, which evaluates model outputs against both correct and randomly selected incorrect normative universes.

---

[PlayCoder: Making LLM-Generated GUI Code Playable](http://arxiv.org/abs/2604.19742)

- PlayCoder: introduces a multi-agent framework that leverages PlayDeveloper, PlayRefiner, and PlayTester to perform repository-aware GUI code generation and iterative repair via closed-loop control.
- The framework utilizes PlayTester to provide visual feedback and dynamic interaction signals, enabling the system to detect and repair silent logic flaws that traditional unit tests overlook.
- PlayCoder integrates retrieval-augmented generation with automated program repair to improve functional correctness and semantic alignment across diverse GUI applications and LLM architectures.

---

[ClawNet: Human-Symbiotic Agent Network for Cross-User Autonomous Cooperation](http://arxiv.org/abs/2604.19211)

- ClawNet: introduces a human-symbiotic agent paradigm that digitizes collaborative relationships through identity binding, scoped authorization, and action-level accountability.
- The framework utilizes a cloud-edge architecture where a Manager Agent and multiple context-specific Identity Agents operate within isolated Gateway Containers, while a Node Endpoint manages local file system interactions.
- ClawNet enforces governance through a dual-layer authorization scheme and an append-only audit log, ensuring that all cross-user agent interactions remain traceable and subject to explicit human approval.

---

[A systematic review of generative AI usage for IT project management](http://arxiv.org/abs/2604.21958)

- Agentic AI Collaborative Architecture for PM: introduces a multi-agent framework that integrates process group-specific agents and role-based agents to automate IT project management tasks under human oversight.
- The architecture utilizes an Orchestrator Agent to manage task routing and context, while a Shared Knowledge & Memory Layer ensures consistent data access across all specialized agents.
- This framework aims to transform project management from reactive oversight to proactive orchestration by leveraging LLMs for autonomous task execution and decision support.

---

#### 20th April 2026


[Agent-World: Scaling Real-World Environment Synthesis for Evolving General Agent Intelligence](http://arxiv.org/abs/2604.18292)

- Agent-World: introduces a self-evolving training arena that unifies scalable real-world environment synthesis with continuous self-evolving agent training to advance general agent intelligence.
- The framework utilizes Agentic Environment-Task Discovery to autonomously mine databases and tools, and Continuous Self-Evolving Agent Training to drive targeted learning through a closed-loop co-evolution of agent policies and environments.
- Agent-World incorporates a Self-Evolving Agent Arena that employs a Diagnosis Agent to identify capability gaps, which then informs the synthesis of new tasks to improve agent performance across diverse benchmarks.

---

[Negative Advantage Is a Double-Edged Sword: Calibrating Advantage in GRPO for Deep Search](http://arxiv.org/abs/2604.18235)

- CalibAdv: introduces a framework for calibrating GRPO advantage signals in deep search agents by employing Soft Advantage Penalization, Advantage Rebalancing, and Special Token Decoupling.
- The framework mitigates training collapse and performance degradation by fine-grained adjustment of negative advantages at intermediate steps and rebalancing positive and negative advantages at the final answer step.
- CalibAdv utilizes a silver document proxy to identify helpful intermediate steps without requiring additional external LLM annotations or sampling overhead.

---

[A Counterexample to EFX n ≥ 3 Agents, m ≥ n + 5 Items, Monotone Valuations via SAT-Solving](http://arxiv.org/abs/2604.18216)

- SAT-based EFX Counterexample Framework: introduces a computational approach to resolve the existence of EFX allocations by encoding the problem into SAT and SMT instances, utilizing SPASS-SAT, CaDiCal, DRAT-trim, LEAN, Z3, and C++ Verification Code.
- The framework demonstrates that EFX allocations always exist for three agents and seven goods, while providing a counterexample for three agents and eight or more goods.
- The methodology combines theoretical analysis, SAT-based search, and formal verification in LEAN to establish the non-existence of EFX allocations for specific monotone valuation settings.

---

[Bridging the Reasoning Gap in Vietnamese with Small Language Models via Test-Time Scaling](http://arxiv.org/abs/2604.17794)

- Vi-S1K (Vietnamese Simple Scaling 1K): introduces a systematic investigation into test-time scaling for Vietnamese SLMs, utilizing Qwen3-1.7B finetuned on a localized reasoning dataset to bridge the reasoning gap in elementary mathematics.
- The research demonstrates that Supervised Fine-Tuning (SFT) acts as a reasoning unlocker, significantly improving explanation quality and pedagogical coherence in small models.
- The study establishes a deployment hierarchy for SLMs, identifying that pure Chain-of-Thought combined with Self-Consistency is superior to complex agentic workflows like ReAct for sub-2B parameter models.

---


[Agentic Forecasting using Sequential Bayesian Updating of Linguistic Beliefs](http://arxiv.org/abs/2604.18576)

- BLF (Bayesian Linguistic Forecaster): introduces an agentic system for binary forecasting that maintains a semi-structured belief state updated iteratively through a tool-use loop.
- The system utilizes LLM (main), LLM (filter), and LLM (summ.) to process information, while employing a LeakFilter and Data tools to ensure backtesting validity against knowledge cutoffs.
- BLF achieves state-of-the-art performance on the ForecastBench benchmark by combining structured belief updates, hierarchical multi-trial aggregation, and hierarchical calibration.

---

[Semantic Entanglement in Vector-Based Retrieval: A Formal Framework and Context-Conditioned Disentanglement Pipeline for Agentic RAG Systems](http://arxiv.org/abs/2604.17677)

- SDP (Semantic Disentanglement Pipeline): introduces a four-stage preprocessing framework that mitigates semantic entanglement by restructuring documents based on operational context, utilizing CAF, Stage A, Stage B, Stage C, Stage D, and a Continuous Feedback Loop.
- The framework addresses geometric confusion in embedding spaces by aligning document structure with agentic retrieval requirements through context-conditioned preprocessing.
- Empirical evaluation on an enterprise healthcare knowledge base demonstrates that the pipeline reduces the Entanglement Index and improves Top-K retrieval precision by 50 percentage points.

---


[MultiWorld: Scalable Multi-Agent Multi-View Video World Models](http://arxiv.org/abs/2604.18564)

- MultiWorld: introduces a unified framework for scalable multi-agent, multi-view video world modeling that enables precise action adherence and synchronized simulation across diverse viewpoints.
- The framework utilizes MACM to manage agent-specific control and GSE to maintain 3D-aware global environmental consistency, facilitating scalable simulation of variable agent and camera counts.
- MultiWorld employs a DiT backbone with flow matching to synthesize high-fidelity, multi-view consistent videos, demonstrating superior performance in complex multi-player game and multi-robot manipulation scenarios.

---

[SynAgent: Generalizable Cooperative Humanoid Manipulation via Solo-to-Cooperative Agent Synergy](http://arxiv.org/abs/2604.18557)

- SynAgent: introduces a unified framework for scalable and physically plausible cooperative humanoid manipulation by leveraging solo-to-cooperative agent synergy.
- The framework utilizes an interaction-preserving retargeting method based on an Interact Mesh to maintain semantic integrity during motion transfer from single-agent data to multi-agent scenarios.
- SynAgent employs a trajectory-conditioned CVAE policy trained via multi-teacher distillation to achieve stable and controllable object-level trajectory execution across diverse object geometries.

---

[ClawEnvKit: Automatic Environment Generation for Claw-Like Agents](http://arxiv.org/abs/2604.18543)

- ClawEnvKit: introduces an autonomous pipeline that generates verified environments for LLM agents from natural language descriptions.
- The framework utilizes a multi-agent system comprising a Parser, a Generator, and a Validator to create task specifications, interaction interfaces, and evaluation functionals.
- ClawEnvKit enables scalable, on-demand environment generation and live evaluation, facilitating the construction of the Auto-ClawEval benchmark for LLM agents.

---

[MASS-RAG: Multi-Agent Synthesis Retrieval-Augmented Generation](http://arxiv.org/abs/2604.18509)

- MASS-RAG: introduces a training-free, multi-agent framework that structures evidence processing into specialized agents to improve RAG robustness and factual accuracy.
- The framework utilizes a Summarizer Agent, Extractor Agent, and Reasoner Agent to distill complementary evidence, which is then processed by an optional Answer Agent and a final Synthesis Agent.
- By exposing multiple intermediate evidence views, the system enables the model to compare and reconcile heterogeneous information before generating a unified final answer.

---

[QRAFTI: An Agentic Framework for Empirical Research in Quantitative Finance](http://arxiv.org/abs/2604.18500)

- QRAFTI: introduces a multi-agent framework that automates empirical asset-pricing research by coordinating specialized agents, including Context Compaction Agent, Factor Research Agent, Code Writing Agent, and Standardized Reporting Agent, to perform multi-step financial analysis.
- The framework utilizes MCP Tools Servers to provide a constrained, reliable interface for recurring financial computations, while employing reflection-based planning to improve task execution accuracy.
- QRAFTI enhances reproducibility and explainability by generating structured computation graphs and standardized research reports that trace the entire workflow from raw data to final results.

---

[WorldDB: A Vector Graph-of-Worlds Memory Engine with Ontology-Aware Write-Time Reconciliation](http://arxiv.org/abs/2604.18478)

- WorldDB: introduces a memory engine for long-running agents that organizes data into Recursive Worlds, utilizes Content-addressed Immutability for auditability, and employs Edge Handlers to enforce structural semantics at write-time.
- The architecture replaces flat RAG with a graph-of-worlds structure, featuring a Reconciler that ensures all updates pass through defined semantics, a Tiered Resolver for entity unification, and a Background Consolidator for efficient long-term state management.
- By decoupling the retrieval pipeline from LLM-based logic, the system achieves high-performance, deterministic memory operations, demonstrating significant accuracy gains on the LongMemEval-s benchmark compared to existing bitemporal knowledge graph systems.

---

[Asset Harvester: Extracting 3D Assets from Autonomous Driving Logs for Simulation](http://arxiv.org/abs/2604.18468)

- Asset Harvester: introduces an end-to-end pipeline that converts sparse, in-the-wild object observations from autonomous driving logs into complete, simulation-ready 3D assets.
- The system utilizes a Data Ingestion Module for curation, SparseViewDiT for multiview image generation, and Object TokenGS for 3D Gaussian lifting to address challenges like limited-angle views and occlusions.
- The framework integrates with the NuRec simulation environment and employs a hybrid training strategy to ensure high-fidelity, reusable 3D assets for autonomous vehicle development.

---

[Progressive Online Video Understanding with Evidence-Aligned Timing and Transparent Decisions](http://arxiv.org/abs/2604.18459)

- Thinking-QwenVL: introduces a framework that decouples reasoning control from memory integration to enable evidence-aligned, transparent online video understanding.
- The ATDM module manages decision-making by decomposing queries into sub-goals and emitting progress and confidence metrics for real-time transparency.
- The HPSI module maintains a compact, relation-aware cognition state using learnable multi-level aggregation tokens to preserve global consistency under tight token budgets.

---

[MedProbeBench: Systematic Benchmarking at Deep Evidence Integration for Expert-level Medical Guideline](http://arxiv.org/abs/2604.18418)

- MedProbeBench: introduces a comprehensive benchmark and evaluation framework for assessing the deep evidence integration capabilities of LLMs and deep research agents in generating expert-level clinical guidelines.
- The framework utilizes a dual-tier approach combining holistic rubric-based quality assessment with fine-grained evidence verification to measure structural completeness, terminological precision, and claim-level grounding.
- Evaluation of 17 systems reveals that while models can achieve high surface fluency, they face significant bottlenecks in mechanistic reasoning and reliable evidence-based synthesis.

---

[TypeScript Repository Indexing for Code Agent Retrieval](http://arxiv.org/abs/2604.18413)

- ABCoder: introduces a high-performance TypeScript parser that replaces RPC-based language server calls with direct TypeScript Compiler API integration to generate efficient UniAST code indexes.
- The framework enables LLMs to perform repository-level reasoning by providing a structured graph of code entities, including call chains and dependency relationships, via the Model Context Protocol.
- By eliminating external RPC overhead, the parser significantly accelerates indexing for large-scale TypeScript repositories while maintaining high semantic accuracy for downstream code agent tasks.

---

[StepPO: Step-Aligned Policy Optimization for Agentic Reinforcement Learning](http://arxiv.org/abs/2604.18401)

- StepPO: introduces a step-level Agentic RL framework that aligns MDP formulation, trajectory representation, and credit assignment with the interaction step rather than individual tokens.
- The framework utilizes a structured step-level representation to maintain token-space consistency while enabling precise context management and asynchronous training.
- Empirical results on HotpotQA demonstrate that StepPO outperforms token-level PPO by providing more effective learning signals for multi-step agent reasoning and decision-making.

---

[OpenGame: Open Agentic Coding for Games](http://arxiv.org/abs/2604.18394)

- OpenGame: introduces an agentic framework for end-to-end web game creation that utilizes GameCoder-27B, an autonomous agent workflow, and Game Skill to translate natural-language specifications into playable games.
- The framework employs Template Skill to provide stable project scaffolding and Debug Skill to systematically resolve cross-file integration failures through a living debugging protocol.
- OpenGame-Bench evaluates generated games on build health, visual usability, and intent alignment using headless browser execution and VLM judging to ensure functional and interactive quality.

---

[ICEBREAKER for Conversational Agents: Breaking the First-Message Barrier with Personalized Starters](http://arxiv.org/abs/2604.18375)

- ICEBREAKER: introduces a two-step handshake framework for proactive initiation in conversational agents, utilizing Resonance-Aware Interest Distiller (RID) and Interaction-Oriented Starter Generator (ISG) to overcome the first-message barrier.
- The framework employs a Personalized Resonance Scorer to distill user interests and an Interaction-Oriented Starter Generator (ISG) to produce diverse, utility-aligned conversation starters.
- ICEBREAKER optimizes starter generation through a hybrid reward list search and hierarchical preference alignment to maximize user engagement and topical diversity.

---

[Dissecting AI Trading: Behavioral Finance and Market Bubbles](http://arxiv.org/abs/2604.18373)

- AI Trading Framework: introduces a simulated asset market populated by autonomous LLM agents to study behavioral finance patterns and market dynamics.
- The framework utilizes a structured Chain-of-Thought architecture with persistent memory files to trace the causal link between micro-level cognitive reasoning and macro-level market outcomes.
- The study demonstrates that LLM agents exhibit human-like behavioral biases, such as the disposition effect and extrapolative expectations, which can be causally amplified or suppressed through targeted prompt interventions.

---

[Training and Agentic Inference Strategies for LLM-based Manim Animation Generation](http://arxiv.org/abs/2604.18364)

- ManimTrainer and ManimAgent: introduces a unified framework for text-to-code-to-video generation using ManimTrainer (training pipeline for animation generation) and ManimAgent (inference pipeline for animation generation) with LLM (generates programmatic animation code), Manim Renderer (executes code to produce video), Reward Unification (fuses code and visual signals), GRPO (critic-free reinforcement learning), RITL (renderer-in-the-loop self-correction), RITL-DOC (RITL with API documentation), and Prompt Builder (constructs prompts for LLM).
- The study evaluates 17 open-source LLMs, demonstrating that SFT improves code quality while GRPO enhances visual outputs and responsiveness to extrinsic feedback.
- The research highlights that combining training strategies with agentic inference, specifically RITL-DOC, significantly improves render success rates and visual similarity, often surpassing larger models and closed-source baselines.

---

[ComPASS: Towards Personalized Agentic Social Support via Tool-Augmented Companionship](http://arxiv.org/abs/2604.18356)

- ComPASS: introduces a personalized social support framework that empowers LLMs with external tools to provide substantive, human-like companionship through ComPASS-Bench and the fine-tuned ComPASS-Qwen model.
- The framework utilizes a multi-stage LLM-based pipeline to synthesize user profiles and interaction records, enabling agents to select contextually appropriate tools based on user background and history.
- Evaluations demonstrate that tool-augmented responses significantly outperform traditional empathetic dialogue, with ComPASS-Qwen achieving performance comparable to larger models through efficient supervised fine-tuning.

---

[PRISMA: Preference-Reinforced Self-Training Approach for Interpretable Emotionally Intelligent Negotiation Dialogues](http://arxiv.org/abs/2604.18354)

- PRISMA: introduces an interpretable, emotionally intelligent negotiation dialogue system that leverages an Emotion-aware Negotiation Strategy-informed Chain-of-Thought (ENS-CoT) reasoning mechanism to guide LLMs toward strategically appropriate and empathetic responses.
- The framework employs a preference-reinforced self-training approach, integrating supervised initialization with iterative DPO and self-training to optimize step-by-step reasoning and negotiation effectiveness.
- PRISMA utilizes two novel datasets, JobNego and ResNego, to demonstrate superior performance in emotional appropriateness, strategy consistency, and overall negotiation outcomes compared to standard LLM-based baselines.

---

[Reliability of AI Bots Footprints in GitHub Actions CI/CD Workflows](http://arxiv.org/abs/2604.18334)

- Reliability of AI Bots Footprints in GitHub Actions CI/CD Workflows: presents an empirical study analyzing 61,837 CI/CD workflow runs triggered by five LLM-based agents to evaluate their impact on software delivery reliability.
- The study utilizes the AIDev dataset and GitHub Actions API to quantify workflow success rates, identifying a negative correlation between the frequency of agentic pull requests and overall CI/CD performance.
- The research defines a taxonomy of 13 pull request categories and employs GPT 5.0 alongside human validation to characterize the types of agentic contributions that most frequently lead to workflow failures.

---

[Will People Enjoy a Robot Trainer? A Case Study with Snoopie the Pacerbot](http://arxiv.org/abs/2604.18331)

- SNOOPIE: introduces an embodied robotic quadruped trainer that leverages physical presence and autonomous navigation to guide runners through customized interval training sessions.
- The system utilizes a two-phase interaction workflow consisting of an initial user-guided speed calibration followed by an autonomous robot-led interval training exercise.
- Experimental results demonstrate that the embodied robot significantly improves pace adherence and consistency while being perceived as more enjoyable and helpful than non-embodied wearable alternatives.

---

[EmbodiedLGR: Integrating Lightweight Graph Representation and Retrieval for Semantic-Spatial Memory in Robotic Agents](http://arxiv.org/abs/2604.18271)

- EmbodiedLGR-Agent: introduces a VLM-driven architecture that constructs efficient environment representations using a dual-level memory system comprising a Memory Graph and a Vector Database.
- The framework utilizes Florence-2 VLM and all-MiniLM-L6-v2 to populate memory structures, enabling the LLM Agent to perform targeted retrieval via specialized Retrieval Tools.
- This approach optimizes memory building and retrieval for real-time robotic deployment, significantly reducing response latency compared to monolithic memory bank architectures.

---

[AJ-Bench: Benchmarking Agent-as-a-Judge for Environment-Aware Evaluation](http://arxiv.org/abs/2604.18240)

- AJ-Bench: introduces a comprehensive benchmark for evaluating Agent-as-a-Judge systems across search, data systems, and graphical user interface domains, utilizing Task Design, Trajectory Collection, Label Annotation, and Evaluation Process.
- The framework employs Agent-as-a-Judge, which integrates Tool Box and Env Feedback to perform Information Acquisition, State Verification, and Process Verification through Env Setup & Replay.
- Experiments demonstrate that equipping judge agents with environment interaction capabilities significantly improves evaluation accuracy compared to traditional LLM-as-a-Judge baselines.

---

[Aether: Network Validation Using Agentic AI and Digital Twin](http://arxiv.org/abs/2604.18233)

- Aether: introduces a neuro-symbolic architecture that combines LLM-based agents with a unified Network Digital Twin (NDT) to automate network change validation workflows.
- The system utilizes five specialized agents—Assistant, NDM Query, Impact Assessment, Test Planner, and Test Executor—to perform intent-aware, compositional verification of network changes.
- Aether leverages a temporal Network Digital Map (NDM) and standardized tool interfaces to achieve high error detection and diagnostic coverage in both synthetic and real-world ISP network environments.

---

[AgenTEE: Confidential LLM Agent Execution on Edge Devices](http://arxiv.org/abs/2604.18231)

- AgenTEE: introduces a system for deploying confidential LLM agent pipelines on edge devices by isolating the agent runtime, inference engine, and third-party applications into independently attested confidential virtual machines.
- The framework leverages Arm Confidential Compute Architecture to enforce hardware-level isolation of proprietary assets and runtime state, including model weights and KV-cache, from the host operating system and hypervisor.
- AgenTEE utilizes Confidential Shared Memory to facilitate secure, mutually authenticated communication between isolated components, achieving near-native performance with minimal runtime overhead.

---

[Towards an Agentic LLM-based Approach to Requirement Formalization from Unstructured Specifications](http://arxiv.org/abs/2604.18228)

- Agentic LLM-based Requirement Formalization Pipeline: introduces an agentic methodology that automatically extracts verification-ready properties from unstructured specifications by coordinating multiple LLM-based agents.
- The pipeline integrates requirement extraction, verifiability classification, and formal translation stages to ensure that generated properties are both syntactically and semantically aligned with the target formal model.
- Experimental results demonstrate that the approach achieves high accuracy in generating verifiable properties for cyber-physical systems while effectively filtering out unverifiable constraints.

---

[WebCompass: Towards Multimodal Web Coding Evaluation for Code Language Models](http://arxiv.org/abs/2604.18224)

- WebCompass: introduces a comprehensive, multimodal benchmark for evaluating web engineering capabilities across generation, editing, and repair tasks using LLM-as-a-Judge and Agent-as-a-Judge protocols.
- The framework utilizes a multi-stage, human-in-the-loop pipeline to curate high-quality web prototypes and employs an Agent-as-a-Judge paradigm that leverages Model Context Protocol for autonomous browser interaction and iterative test-case synthesis.
- Experimental results demonstrate that closed-source models significantly outperform open-source alternatives, with visual quality identified as a persistent bottleneck across all evaluated LLMs.

---

[Instruction-as-State: Environment-Guided and State-Conditioned Semantic Understanding for Embodied Navigation](http://arxiv.org/abs/2604.18223)

- S-EGIU: introduces a coarse-to-fine framework that models instruction understanding as a dynamic latent state variable conditioned on the agent's evolving perceptual state.
- The framework utilizes CGIP to activate perceptually relevant sub-instructions and FGIP to perform perception-guided token refinement, ensuring instruction semantics remain aligned with the current visual context.
- By treating instructions as trajectory-evolving states rather than static embeddings, S-EGIU improves navigation efficiency and robustness across diverse VLN benchmarks.

---

[TacticGen: Grounding Adaptable and Scalable Generation of Football Tactics](http://arxiv.org/abs/2604.18210)

- TacticGen: introduces a generative framework for football tactics that models coordinated player movements as multi-agent trajectories conditioned on game context and guided by diverse tactical objectives.
- The framework utilizes a multi-agent Diffusion Transformer (MADiT) backbone, incorporating a context encoder and an event encoder to capture complex inter-agent dependencies and contextual information.
- TacticGen enables adaptable tactic generation at inference time through classifier guidance mechanisms, including rule-based functions, LLM-generated code, and learned value models, without requiring model retraining.

---

[Scalable Neighborhood-Based Multi-Agent Actor-Critic](http://arxiv.org/abs/2604.18190)

- MADDPG-K: introduces a scalable multi-agent reinforcement learning framework that mitigates the computational bottleneck of centralized critics by restricting input to the K-nearest neighbors of each agent.
- The framework utilizes precomputed index sets stored in the replay buffer to maintain constant-size critic inputs, effectively reducing the computational complexity from quadratic to linear with respect to the total agent count.
- Empirical results demonstrate that MADDPG-K achieves faster convergence and superior runtime scaling compared to standard MADDPG in cooperative multi-agent environments.

---

[Multi-Agent Systems: From Classical Paradigms to Large Foundation Model-Enabled Futures](http://arxiv.org/abs/2604.18133)

- MAS: introduces a systematic review and comparative analysis of classical MASs (CMASs) and LFM-based MASs (LMASs), highlighting the paradigm shift from task-specific, model-driven coordination to cognitively empowered, language-mediated collaboration.
- The paper defines a unified analytical framework for LMASs comprising core modules (role definition, perception, planning, memory, execution), interaction mechanisms, and hierarchical optimization layers (model, knowledge, system).
- It identifies future research directions including the co-evolution of CMASs and LMASs, scaling multimodality, causality-enhanced reasoning, and the development of robust device-edge-cloud collaborative architectures.

---

[Training LLM Agents for Spontaneous, Reward-Free Self-Evolution via World Knowledge Exploration](http://arxiv.org/abs/2604.18131)

- Meta-Learning-Driven Evolution: introduces a training paradigm that enables LLM agents to autonomously explore environments and distill observations into structured World Knowledge without human-provided rewards.
- The framework utilizes an outcome-based reward mechanism during training to teach agents how to effectively explore and summarize environments, which is then used as an external context module at inference time.
- Experimental results demonstrate that this approach significantly improves performance on web-based benchmarks and enables smaller models to outperform larger, unassisted counterparts through superior environment exploration.

---

[Chatting about Conditional Trajectory Prediction](http://arxiv.org/abs/2604.18126)

- CiT: introduces a trajectory prediction method that models cross-time-domain social interactions to capture temporal intention dynamics for autonomous systems.
- The framework incorporates ego agent potential motion to enable seamless integration with downstream robotic planning and control modules.
- CiT utilizes intention graphs and cross-domain interaction modeling to refine intention representations and improve prediction accuracy across diverse social scenarios.

---

[ConventionPlay: Capability-Limited Training for Robust Ad-Hoc Collaboration](http://arxiv.org/abs/2604.18123)

- ConventionPlay: introduces a reinforcement learning-based approach for ad-hoc collaboration that utilizes a hierarchical population of K0, K1, and K2 agents to enable robust coordination with adaptive partners.
- The framework employs capability-aware stratified sampling to generate diverse K1 followers, forcing the K2 agent to move beyond reactive behavior toward active team steering.
- By probing a partner's repertoire, the K2 agent effectively identifies and converges on the most efficient shared convention, outperforming existing methods in complex, differentiated reward environments.

---

[Sharing the proceeds from a hierarchical venture when agents have needs](http://arxiv.org/abs/2604.18108)

- Need-adjusted geometric rules: introduces a family of allocation mechanisms for hierarchical ventures where agents have individual needs that must be covered by generated revenues.
- The paper characterizes these rules using axioms such as needs lower bound, lowest rank consistency, and highest rank independence to ensure fair distribution.
- It further identifies a need-adjusted serial rule and explores extensions to broader domains where aggregate revenues exceed aggregate needs.

---

[Test-Time Perturbation Learning with Delayed Feedback for Vision-Language-Action Models](http://arxiv.org/abs/2604.18107)

- PDF: introduces a verifier-free test-time adaptation framework that mitigates trajectory overfitting in VLAs by employing Uncertainty-Based Action Voting and Delayed Feedback-Guided Adaptation.
- The framework utilizes a lightweight P Head to retrospectively adjust action logits based on delayed environmental feedback, while keeping the base VLA parameters frozen.
- PDF improves decision performance and robustness across robotic manipulation and visual control tasks by balancing inference efficiency with adaptive data augmentation.

---

[Architectural Design Decisions in AI Agent Harnesses](http://arxiv.org/abs/2604.18071)

- Agent Harness: introduces a protocol-guided, source-grounded empirical study of 70 AI agent-system projects to identify recurring architectural design decisions across Subagent architecture, Context management, Tool systems, Safety mechanisms, and Orchestration.
- The study reveals that Agent harness architecture is organized around coupled decision bundles rather than independent feature selections, with coordination complexity, context persistence, and governance mechanisms often co-evolving.
- The research synthesizes five recurring architectural patterns—Lightweight Tool, Balanced CLI Framework, Multi-Agent Orchestrator, Enterprise Full-Featured, and Scenario-Verticalized—to provide grounded guidance for framework designers and selectors.

---

[EvoMarket: A High-Fidelity and Scalable Financial Market Simulator](http://arxiv.org/abs/2604.18046)

- EvoMarket: introduces a high-fidelity, scalable financial market simulator that integrates a Discrete-event execution core, Hierarchical event scheduler, Multi-asset exchanges, Asset-specific LOB, Agent panel, Oracle-guided self-calibration, and an Observability and logging layer to enable intervention-oriented experiments.
- The framework utilizes an Oracle-guided self-calibration mechanism to reduce microstructure discrepancies within a single simulation run, avoiding the computational overhead of traditional external black-box optimization.
- EvoMarket supports multi-asset, cross-day simulation with institutional mechanisms like price limits and T+1 settlement, achieving high throughput and scalable performance for market-scale research.

---

[First, Do No Harm (With LLMs): Mitigating Racial Bias via Agentic Workflows](http://arxiv.org/abs/2604.18038)

- Agentic Workflow Framework: introduces a multi-stage evaluation and mitigation pipeline for racial bias in LLMs using LLMs, Search Agent, RAG Agent, Vector Knowledge Base, Flowise Platform, and Final DDx Module.
- The study evaluates five LLMs across implicit and explicit bias benchmarks, identifying DeepSeek V3 as the most robust model for clinical decision-making tasks.
- Integrating DeepSeek V3 into an agentic workflow demonstrates measurable improvements in explicit bias metrics by prioritizing structured clinical knowledge over standalone model generation.

---

[Topology-Aware LLM-Driven Social Simulation: A Unified Framework for Efficient and Realistic Agent Dynamics](http://arxiv.org/abs/2604.18011)

- TopoSim: introduces a topology-aware framework that treats network structure as an active driver of agent interactions to improve the efficiency and realism of LLM-based social simulations.
- The framework utilizes an Update Coordination Module to group agents with similar structural contexts for shared inference and a Role-Differentiation Module to model asymmetric influence based on network topology.
- Experimental results demonstrate that TopoSim maintains high behavioral fidelity while reducing LLM token consumption by 50-90% across diverse social simulation datasets.

---

[Diversity Collapse in Multi-Agent LLM Systems: Structural Coupling and Collective Failure in Open-Ended Idea Generation](http://arxiv.org/abs/2604.18005)

- MAS: introduces a systematic empirical study of diversity collapse in multi-agent systems, identifying structural coupling as the primary driver of premature consensus in open-ended idea generation.
- The framework analyzes how interaction structures, including Agent, Leader, Explorer, and Judge roles, inadvertently contract the solution space through authority-driven dynamics and dense communication topologies.
- The research demonstrates that diversity collapse is a structural failure rather than a model-level insufficiency, and proposes interventions like NGT and subgroup isolation to preserve independence and enhance creative output.

---

[AIT Academy: Cultivating the Complete Agent with a Confucian Three-Domain Curriculum](http://arxiv.org/abs/2604.17989)

- AIT Academy: introduces a curriculum framework for cultivating LLMs across three domains—Natural Science, Humanities, and Social Science—using Confucian Six Arts as behavioral archetypes.
- The framework utilizes specialized training grounds—ClawdGO, Athen’s Academy, and Alt Mirage—to provide longitudinal, ecologically valid training for LLMs through ASAT, CSMA, and multi-agent collaboration architectures.
- AIT Academy addresses systematic unevenness in LLM development by moving beyond benchmark-driven training toward a holistic, multi-domain cultivation trajectory that includes security, creative synthesis, and social reasoning.

---

[E3VS-Bench: A Benchmark for Viewpoint-Dependent Active Perception in 3D Gaussian Splatting Scenes](http://arxiv.org/abs/2604.17969)

- E3VS-Bench: introduces a benchmark for evaluating active perception in photorealistic 3D environments using 3D Gaussian Splatting Scenes, a VLM-as-a-judge Evaluator, an Embodied Agent, a 5-DoF Action Space, Structured Prompting, and an Episode Filtering Pipeline.
- The framework requires agents to perform 5-DoF viewpoint control to resolve occlusions and acquire fine-grained visual evidence for question answering.
- Evaluation results demonstrate that while current LLMs possess strong 2D recognition capabilities, they exhibit a substantial performance gap compared to humans in active 3D viewpoint planning and exploration.

---

[CADMAS-CTX: Contextual Capability Calibration for Multi-Agent Delegation](http://arxiv.org/abs/2604.17950)

- CADMAS-CTX: introduces a framework for multi-agent delegation that replaces static capability profiles with hierarchical, context-conditioned Beta posteriors to prevent systematic misdelegation.
- The framework utilizes an uncertainty-aware scoring function that combines posterior means with an uncertainty penalty to ensure agents delegate tasks only when a peer's capability is well-supported by evidence.
- By employing locally-centralized task coordination and empirical Bayes shrinkage for cold-start scenarios, the architecture achieves sublinear regret and robust performance across heterogeneous task contexts.

---

[RAVEN: Retrieval-Augmented Vulnerability Exploration Network for Memory Corruption Analysis in User Code and Binary Programs](http://arxiv.org/abs/2604.17948)

- RAVEN: introduces a multi-agent framework that leverages LLMs and RAG to automatically synthesize comprehensive vulnerability analysis reports from vulnerable source code.
- The framework integrates a Data Collection Pipeline, a RAG Engine, and an Agentic System comprising Explorer-, Analyst-, Reporter-, and Judge-agents to automate end-to-end security documentation.
- RAVEN utilizes an LLM-as-a-Judge methodology to evaluate generated reports across structural integrity, factual grounding, code reasoning, and remediation quality dimensions.

---

[ReCoQA: A Benchmark for Tool-Augmented and Multi-Step Reasoning in Real Estate Question and Answering](http://arxiv.org/abs/2604.17944)

- HIRE-Agent: introduces a hierarchical multi-agent framework that utilizes an Understand–Plan–Execute architecture to integrate heterogeneous evidence from databases and external APIs for complex real estate reasoning.
- The framework employs a Front-end Agent for intent parsing, a Supervisor Agent for task orchestration, and specialized Database Interaction- and Map Reasoning-agents to handle multi-step queries.
- HIRE-Agent serves as a strong baseline for the ReCoQA benchmark, demonstrating that hierarchical collaboration is essential for solving complex, multi-source reasoning tasks in vertical domains.

---

[ContraPrompt: Contrastive Prompt Optimization via Dyadic Reasoning Trace Analysis](http://arxiv.org/abs/2604.17937)

- ContraPrompt: introduces a prompt optimization framework that leverages dyadic reasoning trace analysis to extract transferable rules from paired failure-to-success execution traces.
- The system utilizes an instrumented agentic retry loop to generate contrastive data, which is then organized into an input-aware decision tree to provide context-specific instructions.
- ContraPrompt outperforms existing baselines by targeting specific reasoning steps identified through the comparison of complete chain-of-thought traces rather than final outputs.

---

[Robust Distributed Sub-Optimal Coordination of Linear Agents with Uncertain Input Nonlinearities](http://arxiv.org/abs/2604.17934)

- RDSC framework: introduces a robust control approach for multi-agent systems to achieve bounded convergence to a global optimizer despite time-varying input nonlinearities.
- The framework utilizes a dynamic protocol that incorporates local communication and sector-bounded uncertainty modeling to ensure input-to-state stability.
- Sufficient conditions for the solvability of the coordination problem are derived and characterized through linear matrix inequalities, ensuring robust performance in the presence of agent-specific uncertainties.

---

[LiteResearcher: A Scalable Agentic RL Training Framework for Deep Research Agent](http://arxiv.org/abs/2604.17931)

- LiteResearcher: introduces a scalable agentic RL training framework that constructs an isolated virtual world to mirror real-world search dynamics, utilizing a Data Synthesis Pipeline, Local Search Engine, Local Browse Tool, Difficulty-Aware Curriculum Learning, GRPO, and an LLM-based Reward Judge.
- The framework decouples agent training from the open web by co-constructing training data and a local corpus, enabling stable, low-latency, and cost-effective RL training for deep research agents.
- By employing difficulty-aware curriculum learning and on-policy GRPO, the framework effectively eliminates repetitive action loops and sustains performance improvements beyond the saturation points of traditional RL approaches.

---

[Automatic Slide Updating with User-Defined Dynamic Templates and Natural Language Instructions](http://arxiv.org/abs/2604.17894)

- SlideAgent: introduces an agent-based framework for dynamic slide updates that integrates multimodal slide parsing, instruction grounding, and tool-augmented reasoning to maintain content consistency.
- The framework utilizes a two-stage architecture comprising slide understanding for hierarchical representation and instruction-driven content synchronization for data-consistent updates.
- The authors also release DynaSlide, a large-scale benchmark with over 20,000 instruction-execution triples grounded in a shared external database to evaluate automated slide update performance.

---

[Latent Preference Modeling for Cross-Session Personalized Tool Calling](http://arxiv.org/abs/2604.17886)

- PREFINE: introduces a test-time memory-augmented method that represents user preferences as evolving hypotheses to improve personalized tool calling.
- The framework utilizes a generate–verify–refine loop to extract reusable constraints from interaction history, enabling the agent to fill under-specified API arguments.
- PREFINE significantly reduces token usage compared to full-history prompting while maintaining effectiveness under dynamic tool schemas.

---

[Scaling Human-AI Coding Collaboration Requires a Governable Consensus Layer](http://arxiv.org/abs/2604.17883)

- Agentic Consensus: introduces a paradigm that replaces opaque AI-generated code with a governable consensus layer (C) as the primary engineering artifact, mediating between human intent (I) and executable artifacts (A) via synchronization operators (Φ, Ψ) and evidence-linked validation (E).
- The framework utilizes specialized agents—Architect, Builder, Auditor, and Navigator—to maintain structural integrity and provide auditable, evidence-based control over AI-assisted software development.
- By treating structural commitments as a first-class, queryable property graph, the approach mitigates dimension collapse and enables human-gated review of structural changes rather than just code diffs.

---

[Design and Evaluation of a Culturally Adapted Multimodal Virtual Agent for PTSD Screening](http://arxiv.org/abs/2604.17871)

- Molhim: introduces a multimodal conversational AI platform designed for culturally adapted PTSD screening in military healthcare settings.
- The system integrates ASR, VLM, LLM, and TTS components to facilitate structured, safe, and purpose-specific dialogues through a high-fidelity virtual avatar.
- The platform employs a state-based dialogue controller to ensure clinical coherence, safety-aware interaction, and automated post-session analysis for mental health support.

---

[TitanCA: Lessons from Orchestrating LLM Agents to Discover 100+ CVEs](http://arxiv.org/abs/2604.17860)

- TitanCA: introduces a four-module pipeline that orchestrates multiple LLM-powered agents to perform precise, layered software vulnerability discovery.
- The architecture utilizes a Matcher for clone detection, a Filter for reasoning-based screening, an Inspector for multi-agent deliberation, and an Adapter for domain-specific refinement.
- By prioritizing precision through successive filtering stages, the system effectively reduces false positives and improves the reliability of automated vulnerability detection in production environments.

---

[On the Reliability of Computer Use Agents](http://arxiv.org/abs/2604.17849)

- CUA: introduces a framework for evaluating the reliability of computer-use agents across repeated executions by decomposing task performance into stochasticity, instruction ambiguity, and planning variability.
- The framework utilizes POMDP to model task execution and employs metrics like Pass^k, McNemar tests, and Wilcoxon signed-rank tests to quantify consistency and identify reliability regressions.
- The study incorporates interventions such as deterministic decoding, instruction clarification, and iterative plan refinement to mitigate unreliability and improve agent performance across diverse computer environments.

---

[Learning from AVA: Early Lessons from a Curated and Trustworthy Generative AI for Policy and Development Research](http://arxiv.org/abs/2604.17843)

- AVA (AI + Verified Analysis): introduces a multi-agent, domain-bounded RAG system that operationalizes epistemic humility through reasoned abstention and verifiable, page-anchored citations for policy research.
- The architecture utilizes a hierarchical RAG database and a multi-agent pipeline, including Query Decomposer-, Retrieval Planner-, Tree Walker-, and Drafting-agents, to ensure evidence-grounded synthesis.
- The system incorporates a verification model that audits generated claims against retrieved evidence, triggering reasoned abstention when support is insufficient to maintain trust and accuracy.

---

[WebUncertainty: Dual-Level Uncertainty Driven Planning and Reasoning For Autonomous Web Agent](http://arxiv.org/abs/2604.17821)

- WebUncertainty: introduces a hierarchical framework for autonomous web agents that mitigates dual-level uncertainty in planning and reasoning to improve performance in complex, long-horizon tasks.
- The framework utilizes a Task Uncertainty-Driven Adaptive Planning Mechanism to switch between explicit and implicit planning modes based on environmental complexity.
- It incorporates an Action Uncertainty-Driven MCTS Reasoning Mechanism with the ConActU strategy to quantify aleatoric and epistemic uncertainty, effectively pruning hallucinated actions and guiding robust decision-making.

---

[Do LLMs Need to See Everything? A Benchmark and Study of Failures in LLM-driven Smartphone Automation using Screentext vs. Screenshots](http://arxiv.org/abs/2604.17817)

- DailyDroid: introduces a benchmark for evaluating mobile agents, comprising 75 tasks across 25 Android apps to systematically analyze failure modes in LLM-driven smartphone automation.
- The study compares text-only (screentext) and multimodal (screentext + screenshot) input modalities across GPT-4o and o4-mini models to assess their impact on task success, efficiency, and cost.
- Findings reveal that while multimodal inputs provide marginal success gains, they incur significantly higher costs, highlighting critical challenges in UI accessibility and the need for more robust mobile agent design.

---

[Memory Centric Power Allocation for Multi-Agent Embodied Question Answering](http://arxiv.org/abs/2604.17810)

- MCPA (Memory Centric Power Allocation): introduces a framework for multi-agent embodied question answering that optimizes power allocation based on a novel quality of memory model to maximize retrieval accuracy.
- The framework utilizes a GAE (Generative Adversarial Exam) pipeline to evaluate the semantic richness of distributed robot memories through forward simulation with VLM and LLM components.
- By prioritizing transmissions from robots with higher memory quality, the system achieves superior EQA performance compared to traditional sensing- or communication-centric resource management methods.

---

[Spatial dynamic modelling to understand how dendritic cell clustering affects T cell activation](http://arxiv.org/abs/2604.17786)

- ABM and PS-PDE framework: introduces a dual-modelling approach to investigate how spatial clustering of dendritic cells influences T cell activation dynamics within lymph nodes.
- The research utilizes a discrete agent-based model and a derived continuum phenotype-structured partial differential equation to quantify T cell stimulation distributions based on dendritic cell topology.
- Findings indicate that while dendritic cell clustering enhances the heterogeneity of T cell activation, it serves as a secondary driver compared to intrinsic T cell characteristics like stimulation uptake and decay rates.

---

[Prompt Optimization Enables Stable Algorithmic Collusion in LLM Agents](http://arxiv.org/abs/2604.17774)

- Prompt Optimization Enables Stable Algorithmic Collusion in LLM Agents: introduces a meta-learning loop where LLM agents participate in market simulations and a reflective LLM meta-optimizer iteratively refines a shared meta-prompt to discover stable tacit collusion strategies.
- The framework utilizes a nested logit demand model to simulate duopoly markets where LLM agents maintain history and self-notes to inform pricing decisions while optimizing for aggregate profit.
- Experimental results demonstrate that meta-prompt optimization significantly improves coordination quality and enables agents to generalize collusive behaviors to held-out market configurations.

---

[Efficient Federated RLHF via Zeroth-Order Policy Optimization](http://arxiv.org/abs/2604.17747)

- Par-S2ZPO (Partitioned, Sign-based Stochastic Zeroth-order Policy Optimization): introduces a communication-efficient federated RLHF framework that utilizes zeroth-order optimization with binary perturbations and parameter partitioning to minimize resource consumption on edge devices.
- The framework partitions actor network parameters into K subsets, allowing each agent to perform local policy evaluation and communicate only binary preference feedback to the central server.
- Theoretical analysis establishes that the algorithm achieves convergence rates independent of the number of agents, demonstrating efficiency comparable to centralized systems while significantly reducing communication, computation, and memory overhead.

---

[HiRAS: A Hierarchical Multi-Agent Framework for Paper-to-Code Generation and Execution](http://arxiv.org/abs/2604.17745)

- HiRAS (Hierarchical Research Agent System): introduces a hierarchical multi-agent framework for automated experiment reproduction that employs supervisory manager agents to coordinate specialized agents across fine-grained stages.
- The framework utilizes a tree-structured orchestration where manager agents actively inspect intermediate artifacts in a shared workspace to diagnose failures and dynamically re-invoke subordinate agents.
- The authors also introduce Paper2Code-Extra (P2C-Ex), a refined reference-free evaluation protocol that incorporates repository-level information to mitigate evaluator hallucination and improve alignment with reference-based metrics.

---

[Tool Learning Needs Nothing More Than A Free 8B Language Model](http://arxiv.org/abs/2604.17739)

- TRUSTEE: introduces a data-free method for training tool-calling agents by simulating the entire environment using a local open-source LLM.
- The framework integrates a Query Generator, Tool Simulator, User Simulator, and Verifier Simulator to create dynamic, interactive training environments without requiring external annotated data.
- An Adaptive Curriculum Learning mechanism dynamically modulates task difficulty based on agent performance to ensure stable and efficient reinforcement learning.

---

[Co-evolving Agent Architectures and Interpretable Reasoning for Automated Optimization](http://arxiv.org/abs/2604.17708)

- EvoOR-Agent: introduces a co-evolutionary framework that treats agent architecture and reasoning trajectories as evolvable objects to automate operations research workflows.
- The framework utilizes an AOE-style architecture graph to represent and optimize agent workflows, enabling adaptive coordination of problem interpretation, mathematical formulation, and solver execution.
- By coupling architecture evolution with reasoning-trajectory search and knowledge-base-assisted operators, the system achieves superior performance and structural interpretability compared to fixed-pipeline LLM agents.

---

[SelfHeal: Empirical Fix Pattern Analysis and Bug Repair in LLM Agents](http://arxiv.org/abs/2604.17699)

- SelfHeal: introduces a multi-agent system for automated bug repair in LLM agents, utilizing a Fix Agent and a Critic Agent empowered by internal fix rules and external web search.
- The framework employs an iterative ReAct-based cycle where the Fix Agent proposes repairs and the Critic Agent validates them using specialized tools for API and format verification.
- The research also presents AgentDefect, a benchmark dataset of 37 runtime buggy instances, and provides an empirical analysis of 23 distinct fix patterns observed in LLM agent development.

---

[CAPO: Counterfactual Credit Assignment in Sequential Cooperative Teams](http://arxiv.org/abs/2604.17693)

- CAPO: introduces a critic-free policy-gradient algorithm for sequential cooperative teams that utilizes Additive reward decomposition (models expected team reward), Ridge regression estimator (recovers per-agent components), Upstream-cancellation identity (simplifies advantage target), Fictitious sampling (estimates indirect effect), and PPO-clipped updates (optimizes agent policies) to perform efficient credit assignment.
- The framework addresses sequential-update non-stationarity by decomposing the advantage into a closed-form direct effect and a fictitious-sample indirect effect, avoiding the exponential variance scaling of cumulative importance-sampling methods.
- CAPO demonstrates superior performance and scalability in sequential multi-agent settings by adapting its variance to the coupling structure of the team, remaining effective even as team size increases.

---

[How Adversarial Environments Mislead Agentic AI?](http://arxiv.org/abs/2604.18874)

- POTEMKIN: introduces a threat model called Adversarial Environmental Injection (AEI) where adversaries compromise tool outputs to deceive LLM agents through breadth-based epistemic drift and depth-based navigational policy collapse.
- The research identifies a Robustness Schism, demonstrating that resistance to content poisoning in RAG systems does not correlate with resistance to structural navigational traps in citation graphs.
- The study reveals that LLM agents exhibit a Punishment of Honesty, where they penalize hedged scientific claims while failing to detect falsehoods, and provides an open-source evaluation harness to test these vulnerabilities.

---

[Global Product Intersection Sets in Semigroups](http://arxiv.org/abs/2604.18869)

- Aristotle: introduces a formal verification approach to classify product intersection sets in semigroups by leveraging automated theorem proving and Lean-based formalization.
- The research provides a complete classification of global sets HQ and H*N, demonstrating that any subset of natural numbers containing 1 can be realized as a product intersection set.
- The paper highlights the capability of the Aristotle agent to autonomously discover proofs and resolve open mathematical problems originally posed by Nathanson.

---

[Temporal UI State Inconsistency in Desktop GUI Agents: Formalizing and Defending Against TOCTOU Attacks on Computer-Use Agents](http://arxiv.org/abs/2604.18860)

- PUSV (Pre-execution UI State Verification): introduces a layered middleware defense that re-verifies the UI state immediately before action dispatch to mitigate TOCTOU vulnerabilities in desktop GUI agents.
- The framework utilizes Layer 1 (Masked Pixel SSIM), Layer 2a (Global Screenshot Diff), and Layer 2b (X Window Registry Diff) to detect malicious UI state changes during the observation-to-action gap.
- This research formalizes Visual Atomicity Violations and demonstrates that while PUSV effectively intercepts OS-level attacks, it remains blind to zero-visual-footprint DOM injections, necessitating future application-layer verification.

---

[The Triadic Loop: A Framework for Negotiating Alignment in AI Co-hosted Livestreaming](http://arxiv.org/abs/2604.18850)

- Triadic Loop Framework: introduces a conceptual model for multi-party alignment in AI-augmented livestreaming by framing interaction as a temporally reinforced process of bidirectional adaptation among the streamer, AI co-host, and audience.
- The framework identifies three interdependent sub-loops—Performative Steering, Affective Synchrony, and Community Mediation—where misalignment in any single relationship can destabilize the entire socioemotional system.
- It proposes "strategic misalignment" as a mechanism for sustaining community engagement and suggests dynamic, temporally grounded evaluation metrics to assess AI co-host performance beyond static instruction-following.

---

[Consensus and flocking with transmission and reaction delays](http://arxiv.org/abs/2604.18848)

- Consensus and flocking with transmission and reaction delays: introduces a mathematical framework for analyzing collective behavior in multi-agent systems subject to distinct transmission and reaction delays.
- The paper derives sufficient conditions for asymptotic consensus and flocking using a Lyapunov-Krasovskii functional approach combined with a Halanay-type inequality.
- The analysis provides rigorous stability criteria for both first-order consensus models and second-order Cucker-Smale flocking models under non-negligible communication and processing lags.

---

[Human-Guided Harm Recovery for Computer Use Agents](http://arxiv.org/abs/2604.18847)

- BACKBENCH (Benchmark for Computer Use Agent Harm Recovery): introduces a generate-and-verify framework that utilizes LMgen, LMver, and a Reward Model to steer agents from harmful states back to safe states in alignment with human preferences.
- The framework operationalizes human-centered recovery by training a reward model on pairwise preference judgments to rerank candidate plans generated by an LLM.
- Empirical results demonstrate that preference-guided scaffolds significantly outperform base LLMs in recovery tasks, particularly in resource-constrained scenarios.

---

[AI scientists produce results without reasoning scientifically](http://arxiv.org/abs/2604.18805)

- Corral: introduces a systematic evaluation framework to decompose the contributions of base LLMs and agent scaffolds in scientific research tasks.
- The framework evaluates LLM-based agents across eight scientific domains using mechanistic performance analysis and behavioral analysis of reasoning traces.
- Findings indicate that base LLMs are the primary determinant of performance, while scaffold engineering provides minimal improvement in epistemic reasoning patterns.

---

[MANGO: Multi-Agent Web Navigation via Global-View Optimization](http://arxiv.org/abs/2604.18779)

- MANGO: introduces a web navigation framework that leverages global website structure to identify intent-related entry points and optimize navigation efficiency under limited budgets.
- The framework models URL selection as a multi-armed bandit problem using Thompson Sampling to dynamically prioritize promising URLs based on navigation history and reflection feedback.
- MANGO integrates an episodic memory module to store navigation trajectories and reflections, preventing redundant exploration and improving performance on complex, long-horizon web tasks.

---

[CHICO-Agent: An LLM Agent for the Cross-layer Optimization of 2.5D and 3D Chiplet-based Systems](http://arxiv.org/abs/2604.18764)

- CHICO-Agent: introduces a hierarchical multi-agent framework that utilizes LLM reasoning to perform cross-layer design space exploration for 2.5D and 3D chiplet-based systems by iteratively analyzing historical PPAC evaluations.
- The framework employs an Admin Agent to generate exploration plans and multiple Field Agents to evaluate these plans in parallel, utilizing both a Persistent Context for domain constraints and an Evolving Context for iterative learning.
- By replacing stochastic metaheuristics with a reasoning-driven loop, CHICO-Agent achieves lower system costs and provides interpretable rationales for design decisions while significantly reducing the hyperparameter tuning burden.

---

[Opinion polarization from compression-based decision making where agents optimize local complexity and global simplicity](http://arxiv.org/abs/2604.18755)

- Agent-based model (ABM): introduces a framework where agents optimize the ratio of local-to-global Shannon entropy to balance the desire for local distinctiveness with the tendency for global cognitive compression.
- The model utilizes cognitive compression via opinion binning to simulate how individuals simplify complex social environments into manageable information chunks.
- Computational experiments demonstrate that moderate local group sizes, consistent with Dunbar’s number, facilitate the emergence of polarized opinion clusters with sustained internal variability.

---

[A Scientific Human-Agent Reproduction Pipeline](http://arxiv.org/abs/2604.18752)

- SHARP: introduces a structured framework for reproducing scientific data analyses by decomposing tasks into autonomous steps executed by specialized subagents under human supervision.
- The framework utilizes Claude Code as the primary LLM-based agent, which collaborates with human researchers to translate scientific papers into machine-readable codebases through iterative checkpoints.
- SHARP incorporates specialized subagents for analysis, testing, and quality assurance, while leveraging the FlexCAST principles and the law workflow engine to ensure modularity and reproducibility in scientific research.

---

[Autonomous Skeletal Landmark Localization towards Agentic C-Arm Control](http://arxiv.org/abs/2604.18740)

- Agentic C-arm control framework: introduces a fine-tuning approach for MLLMs to perform autonomous skeletal landmark localization and C-arm navigation using anatomical spatial grounding.
- The framework utilizes LoRA and Unsloth for efficient fine-tuning of MLLMs on synthetic X-ray datasets generated by DeepDRR to enable context-aware landmark prediction.
- The system employs a multi-step perception-action loop that allows the MLLM to reason about spatial relationships, incorporate clinician feedback, and iteratively adjust C-arm positioning.

---

[Towards Optimal Agentic Architectures for Offensive Security Tasks](http://arxiv.org/abs/2604.18718)

- Agentic Security Architectures: introduces a controlled benchmark of 20 interactive targets to evaluate five distinct agentic coordination topologies across whitebox and blackbox security auditing modes.
- The study compares SAS, MAS-Indep, MAS-Decent, MAS-Central, and MAS-Hybrid, utilizing Sandbox Agent and Validator Agent components to assess performance under matched budgets and constraints.
- Results demonstrate a non-monotonic cost-quality frontier where broader coordination does not consistently outperform simpler architectures, highlighting the importance of selective routing based on task observability and domain.

---

[Characterizing AlphaEarth Embedding Geometry for Agentic Environmental Reasoning](http://arxiv.org/abs/2604.18715)

- Agentic Geospatial Intelligence System: introduces an agentic framework that leverages the non-Euclidean geometric structure of AlphaEarth embeddings to improve environmental reasoning through geometry-aware tools and multi-step planning.
- The system integrates AlphaEarth Embeddings (64-dimensional land surface representations) with a FAISS-indexed Database (efficient k-nearest neighbor search) and a ReAct-style Planning Architecture (iterative reasoning and tool invocation) to perform complex environmental queries.
- The architecture incorporates Retrieval Tools (five deterministic environmental data access functions) and Geometry-Aware Tools (four functions utilizing manifold geometric metadata), managed by a System Model (LLM for planning and response synthesis) and evaluated by a Judge Model (LLM for performance evaluation).

---

[APRVOS: 1st Place Winner of 5th PVUW MeViS-Audio Track](http://arxiv.org/abs/2604.18665)

- APRVOS: introduces a staged pipeline for audio-conditioned referring video object segmentation that decouples speech transcription, visual existence verification, coarse segmentation, and agentic refinement.
- The framework utilizes VibeVoice-ASR for transcription, a Qwen3-VL-based Judger for existence verification, Sa2VA for initial mask generation, and an agentic layer with SAM3 for boundary refinement.
- By explicitly addressing speech-recognition noise and visual-existence uncertainty, the method achieves robust performance on the MeViS-Audio benchmark compared to single-pass models.

---

[Evaluating Answer Leakage Robustness of LLM Tutors against Adversarial Student Attacks](http://arxiv.org/abs/2604.18660)

- Adversarial Tutoring Robustness Evaluation Framework: introduces a systematic approach to evaluate LLM-based tutors against adversarial student attacks using Tutor Agent, Adversarial Student Agent, Judge Agent, Refiner Agent, Memory, and Dataset.
- The framework utilizes a fine-tuned adversarial student agent to simulate multi-turn dialogues, effectively probing tutor robustness against answer leakage.
- Experimental results demonstrate that pedagogical alignment and defense strategies like reasoning-based tutors and multi-agent setups significantly improve robustness against adversarial student behavior.

---

[Owner-Harm: A Missing Threat Model for AI Agent Safety](http://arxiv.org/abs/2604.18658)

- Nous: introduces a four-layer compositional runtime safety gate designed to mitigate owner-harm by integrating L1 (Encodes owner-policy constraints), L2 (Routes trivially benign actions), L3 (LLM evaluator for semantic reasoning), and L4 (Deterministic audit of artifacts).
- The framework addresses the systematic blind spot in existing safety benchmarks where agents harm their own deployers by failing to account for resource ownership, trust boundaries, and authorization scope.
- Experimental results demonstrate that combining semantic gate reasoning with deterministic post-audit verification significantly improves detection of complex threats like hijacking, achieving 93.3% effectiveness.

---

[From Craft to Kernel: A Governance-First Execution Architecture and Semantic ISA for Agentic Computers](http://arxiv.org/abs/2604.18652)

- Arbiter-K: introduces a governance-first execution architecture that encapsulates an untrusted LLM-based Probabilistic Processing Unit within a deterministic Symbolic Governor to enforce security as a microarchitectural property.
- The framework utilizes a Semantic ISA to reify LLM outputs into discrete instructions, enabling the kernel to maintain a Security Context Registry and construct an Instruction Dependency Graph for active taint propagation.
- By mediating all environment-impacting operations through a trusted kernel, the architecture prevents semantic injection attacks and enables autonomous execution correction via policy feedback loops.

---

[Automatic Ontology Construction Using LLMs as an External Layer of Memory, Verification, and Planning for Hybrid Intelligent Systems](http://arxiv.org/abs/2604.20795)

- Hybrid Intelligent System Architecture: introduces a neuro-symbolic framework that utilizes an LLM as an orchestration layer over structured external memory, including LLM (reasoning and generation engine), MCP (orchestration protocol for tools), Vector RAG storage (similarity-based retrieval memory), RDF/OWL graph (structured ontological knowledge base), Reasoning engine (logical consistency and inference), SHACL validator (structural constraint verification), Ontology Builder (automated pipeline for knowledge extraction), Agent layer (specialized execution and planning), and Logs and embeddings (audit trail and auxiliary memory).
- The architecture replaces monolithic LLM memory with a dual-memory system, combining vector-based retrieval for textual residue with an RDF/OWL graph for verifiable, structured world modeling.
- The system employs a closed-loop Ontology Builder pipeline that transforms unstructured inputs into validated knowledge triples, enabling robust planning and explainable reasoning through a proposal-check-repair cycle.

---

[Enhancing Research Idea Generation through Combinatorial Innovation and Multi-Agent Iterative Search Strategies](http://arxiv.org/abs/2604.20548)

- Multi-Agent Iterative Planning and Search Framework: introduces a system that leverages combinatorial innovation theory to generate, evaluate, and refine research ideas through a virtual team of agents, utilizing Dataset Construction, Initial Research Idea Generation, Research Idea Iteration, and Research Idea Abstract Generation.
- The framework employs a Virtual Academic Agent Team to simulate diverse perspectives, using a Swiss System Tournament and a Zero-shot LLM Ranker to iteratively improve the quality, diversity, and novelty of research ideas.
- By integrating a Literature Search Agent with multi-agent reasoning, the system effectively expands the exploration space of LLMs and mitigates perspective bias in scientific ideation.

---

[Forward-looking evolutionary game dynamics subject to exploration cost](http://arxiv.org/abs/2604.20029)

- Forward-looking EGD: introduces a mathematical framework coupling evolutionary game dynamics with static Hamilton–Jacobi–Bellman equations to incorporate forward-looking behavior and exploration costs.
- The framework models agent decision-making as a constrained optimization problem where the Lagrangian multiplier acts as a relaxation parameter controlling the speed and accuracy of action evolution.
- The paper provides theoretical proofs for the unique existence of solutions and demonstrates the model's applicability through numerical investigations of one- and two-dimensional resource management problems.

---

[More Is Different: Toward a Theory of Emergence in AI-Native Software Ecosystems](http://arxiv.org/abs/2604.19827)

- CAS framework: introduces a theoretical approach to study AI-native software ecosystems as complex adaptive systems where emergent properties arise from agent interactions rather than individual components.
- The framework utilizes Micro-level agent actions, Coarse-graining functions, and Macro-level ecosystem observables to quantify causal emergence using Effective Information (EI) measurement.
- The paper proposes seven falsifiable propositions to test whether ecosystem-level monitoring is necessary for governing AI-native systems, challenging traditional artifact-level verification methods.

---

[JTPRO: A Joint Tool–Prompt Reflective Optimization Framework for Language Agents](http://arxiv.org/abs/2604.19821)

- JTPRO (Joint Tool–Prompt Reflective Optimization): introduces a weight-free framework that iteratively co-optimizes global instructions and per-tool schemas using rollout-driven reflection to improve tool-calling reliability in trace-supervised settings.
- The framework utilizes a Candidate Pool, Pareto Selector, Rollout Engine, Diagnostic Reflector, Global Instruction Optimizer, Tool Schema Optimizer, Slot Semantics Globalizer, and Validation Engine to systematically resolve tool confusion and argument instantiation errors.
- JTPRO enhances LLM agent performance by globalizing repetitive slot semantics into a shared instruction layer while maintaining tool-specific disambiguation cues, effectively scaling to large tool inventories without requiring model fine-tuning.

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

