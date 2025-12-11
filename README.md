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

## Research papers: 2025 (4/4)

[2025 (4/4)](http://github.com/tmgthb/Autonomous-Agents/blob/main/README.md), [2025 (3/4)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2025_3.md), [2025 (2/4)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2025_2.md), [2025 (1/4)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2025_01.md), [2024](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2024.md), [2023](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2023.md), [Earlier](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_Earlier.md)

Chronological order. 





</div>




#### 10th December 2025

[MoReGen: Multi-Agent Motion-Reasoning Engine for Code-based Text-to-Video Synthesis](http://arxiv.org/abs/2512.04221)

- MoReGen (Multi-Agent Motion-Reasoning Engine): introduces a motion-aware, physics-grounded text-to-video framework that integrates multi-agent LLMs, physics simulators, and renderers to generate reproducible, physically accurate videos from text prompts in the code domain.
- The framework employs an iterative multi-agent feedback loop, where the Text-Parser Agent converts natural language to structured specifications, the Code-Writer Agent generates executable simulation code, and the Evaluator refines the output.
- To quantitatively assess physical validity, the approach proposes MoRe Metrics, a trajectory-based evaluation suite, and MoReSet, a benchmark of 1,275 Newtonian phenomena videos with ground-truth trajectories.

---

[Visual Heading Prediction for Autonomous Aerial Vehicles](http://arxiv.org/abs/2512.09898)

- YOLOv5-ANN Pipeline: introduces a vision-based framework for real-time UAV-UGV coordination, integrating UGV Detection (YOLOv5), Feature Extraction, Heading Angle Prediction (ANN), Collision Avoidance (DroNet), and dual monocular cameras (C1/C2) for navigation.
- The system relies solely on onboard camera inputs and bounding box features to predict the required UAV heading angle, achieving a mean absolute error of 0.1506° in GPS-denied environments.
- The lightweight, markerless architecture ensures fast inference (31 ms/frame) suitable for embedded platforms, enabling robust aerial-ground alignment without external localization infrastructure.

---

[Training One Model to Master Cross-Level Agentic Actions via Reinforcement Learning](http://arxiv.org/abs/2512.09706)

- CrossAgent: introduces a unified agentic model that masters heterogeneous action spaces and autonomously selects the optimal interface for each step of a trajectory using a comprehensive training pipeline.
- The framework leverages a Causal Transformer VLM core, a Router for dynamic action space selection, and is trained via cold-start SFT followed by Single-Turn and Multi-Turn RL using the GRPO algorithm.
- By dynamically balancing high-level efficiency (Motion/Language Actions) with low-level precision (Raw/Grounding Actions), the model achieves superior generalization and efficiency in long-horizon reasoning tasks in Minecraft.

---

[UrbanNav: Learning Language-Guided Urban Navigation from Web-Scale Human Trajectories](http://arxiv.org/abs/2512.09607)

- UrbanNav: introduces a scalable framework for language-guided urban navigation that leverages a web-scale data pipeline to create a large dataset of human walking trajectories and instructions for imitation learning.
- The data pipeline includes Trajectory Annotation using DPVO, Robot-Compatible Filtering using YOLOv10, and Language Instruction Annotation using the Qwen2.5-VL-72B VLM and DepthAnything.
- The policy architecture uses DINOv2 and CLIP encoders, fused via a FiLM module and processed by a Transformer, to predict future waypoints, orientation, and arrival status based on visual history and language instructions.

---

[SWEnergy: An Empirical Study on Energy Efficiency in Agentic Issue Resolution Frameworks with SLMs](http://arxiv.org/abs/2512.09543)

- SWEnergy: introduces an empirical study evaluating four leading agentic issue resolution frameworks constrained to use Small Language Models (SLMs) on fixed local hardware.
- The study measures energy consumption, duration, token usage, and memory across 150 runs per configuration on the SWE-bench Verified Mini benchmark.
- Results indicate that framework architecture, not the SLM's capacity, is the primary driver of wasted energy due to unproductive reasoning loops.

---

[Chapter 3: Architectures for Building Agentic AI](http://arxiv.org/abs/2512.09458)

- RCAIA (Reliability-Centric Agentic AI Architecture): introduces a dependable agentic system architecture earned through principled componentization, disciplined interfaces, and explicit control loops that supervise reasoning and action.
- The architecture separates core functions—Goal Manager, Planner, Tool Router, and Execution Gateway—and embeds assurance hooks like Verifiers/Critics and a Safety Supervisor for runtime governance and failure containment.
- Reliability is achieved by converting free-form LLM proposals into governed behavior via typed schemas, least-privilege tool calls, simulation-before-actuate safeguards, and comprehensive audit logs.

---

[COVLM-RL: Critical Object-Oriented Reasoning for Autonomous Driving Using VLM-Guided Reinforcement Learning](http://arxiv.org/abs/2512.09349)

- COVLM-RL (Critical Object-Oriented Reasoning for Autonomous Driving Using VLM-Guided Reinforcement Learning): introduces a novel end-to-end driving framework that integrates critical object-oriented reasoning via a VLM with an RL agent, guided by a consistency loss.
- The framework uses a Chain-of-Thought prompting strategy within the VLM to transform multi-view visual inputs into structured semantic decision priors (Identification, Prediction, Planning).
- This structured guidance reduces policy exploration complexity and enhances generalization, achieving a 50% improvement in success rate in previously unseen CARLA environments compared to baselines.

---

[The Illusion of Rationality: Tacit Bias and Strategic Dominance in Frontier LLM Negotiation Games](http://arxiv.org/abs/2512.09254)

- NegotiationArena: introduces a comprehensive evaluation of frontier LLMs across three multi-turn bargaining games to challenge the assumption that scaling leads to rational, unbiased negotiation strategies.
- The study reveals that LLMs diverge into distinct, model-specific strategic equilibria, exhibiting persistent numerical and semantic anchoring biases despite improved general reasoning capabilities.
- Analysis of pairwise interactions demonstrates dominance patterns where stronger models systematically achieve higher payoffs, highlighting risks for inequitable outcomes in real-world AI agent deployments.

---

[SCOPE: Language Models as One-Time Teacher for Hierarchical Planning in Text Environments](http://arxiv.org/abs/2512.09897)

- SCOPE (Subgoal-COnditioned Pretraining for Efficient planning): introduces a one-shot hierarchical planner that leverages LLM-generated subgoals only at initialization to pretrain a lightweight student model, followed by RL fine-tuning.
- The framework uses a Manager Agent ($\pi^m$) for high-level planning and an Employee Agent ($\pi^e$) for low-level execution, significantly reducing inference time compared to LLM-dependent methods by avoiding repeated LLM queries.
- SCOPE achieves a 0.56 success rate on TextCraft, demonstrating that even suboptimal, one-time LLM guidance provides sufficient structure for effective hierarchical learning when combined with RL refinement.

---

[DeepSeek's WEIRD Behavior: The cultural alignment of Large Language Models and the effects of prompt language and cultural prompting](http://arxiv.org/abs/2512.09772)

- LCAT (LLM Cultural Alignment Testing): introduces a methodology to measure the cultural alignment of six flagship LLMs against the United States and China using Hofstede's VSM13 survey, Prompt Language, and Cultural Prompting strategies.
- The evaluation framework calculates six Hofstede cultural dimensions (PDI, IDV, MAS, UAI, LTO, IVR) based on 24 survey questions prompted in English or Simplified Chinese.
- Results indicate a strong bias toward the United States across all tested LLMs, although prompt language and cultural prompting successfully shift alignment for low-cost models like GPT-40 and GPT-4.1.

---

[An End-to-end Planning Framework with Agentic LLMs and PDDL](http://arxiv.org/abs/2512.09629)

- AEPF (Agentic End-to-end Planning Framework): introduces an end-to-end planning pipeline that converts natural language specifications into validated PDDL plans using a dynamic LLM orchestrator and specialized agent workflows.
- The orchestrator generates an initial JSON representation and PDDL artifacts, which are then iteratively refined by specialized agents based on feedback from an external PDDL solver and validator.
- This agentic system ensures formal correctness and potential cost-optimality while translating the final plan back into natural language for improved accessibility and interpretability.

---

[Auto-BenchmarkCard: Automated Synthesis of Benchmark Documentation](http://arxiv.org/abs/2512.09577)

- Auto-BenchmarkCard: introduces a workflow for generating validated AI benchmark documentation by combining multi-agent data extraction, LLM-driven synthesis, and factual validation.
- The system operates in three phases—Extraction, Composition, and Validation—using specialized tools like Unitxt, Docling, and a structured risk taxonomy via the Risk mapper.
- The Validation Phase uses a FactReasoner and RAG component to assess the factual accuracy of atomic statements, resulting in a fact-scored and remediated Final BenchmarkCard.

---

#### 9th December 2025



[Towards a Science of Scaling Agent Systems](http://arxiv.org/abs/2512.08296)

- MAS Scaling Framework: introduces quantitative scaling principles for agent systems by evaluating five canonical architectures (Single-Agent System, Independent, Centralized, Decentralized, Hybrid) across three LLM families and four agentic benchmarks.
- The framework derives a predictive mixed-effects model using empirical coordination metrics, including Efficiency (Ec), Error Amplification (Ae), and Redundancy (R), that achieves $R^2=0.513$ cross-validated variance explanation.
- The analysis identifies a tool-coordination trade-off, a capability ceiling, and architecture-dependent error amplification, demonstrating that optimal architecture selection is task-contingent, not solely dependent on agent count.

---


[Fed-SE: Federated Self-Evolution for Privacy-Constrained Multi-Environment LLM Agents](http://arxiv.org/abs/2512.08870)

- Fed-SE (Federated Self-Evolution): introduces a communication-efficient framework for LLM agents that uses local self-evolution on filtered successful trajectories and global low-rank aggregation of LoRA adapters to achieve robust cross-environment knowledge transfer under privacy constraints.
- The local phase stabilizes training against sparse rewards by optimizing lightweight LoRA adapters using Maximum Likelihood Estimation on high-return trajectories stored in a privacy-preserving experience buffer.
- The global phase aggregates these distributed adapter parameters within a low-rank subspace via unweighted averaging, decoupling general reasoning capabilities from environment-specific dynamics and mitigating negative transfer.

---

[A Practical Guide for Designing, Developing, and Deploying Production-Grade Agentic AI Workflows](http://arxiv.org/abs/2512.08769)

- Production-Grade Agentic AI Workflow Framework: introduces a structured methodology for designing, developing, and deploying reliable agentic systems, featuring multi-agent orchestration, tool integration, deterministic execution, and Responsible AI mechanisms.
- The framework emphasizes nine core best practices, including single-responsibility agents, externalized prompt management, containerized deployment via Kube Cluster, and direct function calls over Model Context Protocol (MCP) for infrastructure tasks.
- The approach is demonstrated via a multimodal news-to-media workflow where a consortium of LLMs generates content drafts, which are then consolidated by a Reasoning Agent for accuracy and alignment before multimodal synthesis and GitHub publishing.

---

[Insured Agents: A Decentralized Trust Insurance Mechanism for Agentic Economy](http://arxiv.org/abs/2512.08737)

- Insured Agents Mechanism (IAM): introduces a decentralized trust insurance mechanism for the agentic economy, where specialized Insurer Agents post slashable collateral for Service Agents in exchange for premiums and privileged audit access.
- The mechanism employs a hierarchical structure, utilizing Layer 1 specialized insurers (e.g., Safety, Finance) and a Layer 2 Master Insurer to enable composable trust and risk calibration for LLM agents.
- Trust is framed as a market where competitive underwriting and an optimistic escalation game sustain incentive compatibility, ensuring honest behavior without frequent recourse to the costly Verifier.

---

[NeurIDA: Dynamic Modeling for Effective In-Database Analytics](http://arxiv.org/abs/2512.08483)

- NeurIDA (Neural In-Database Analytics system): introduces an autonomous end-to-end system for in-database analytics that dynamically constructs bespoke ML models using the Query Intent Analyzer, Conditional Model Dispatcher, Dynamic In-Database Modeling Engine, and Analytical Report Synthesizer.
- The system proposes dynamic in-database modeling to pre-train a composable base model architecture over relational data, enabling runtime customization based on task and data profiles.
- NeurIDA supports natural language queries and LLM agents for structured task formulation and report generation, achieving up to 12% improvement in AUC-ROC and 25% relative reduction in MAE compared to standalone base models.

---

[A Multi-Agent LLM Framework for Design Space Exploration in Autonomous Driving Systems](http://arxiv.org/abs/2512.08476)

- LLM-DSE (LLM-augmented DSE framework): introduces a multi-agent LLM architecture integrating multi-modal reasoning with 3D simulation and profiling tools to automate the design space exploration (DSE) for autonomous driving systems.
- The framework utilizes specialized LLM agents for user input interpretation, design point generation, execution orchestration, and analysis of visual and textual simulation outputs, enabling bottleneck identification without human intervention.
- The architecture is structured into four layers—Interpretation, Multi-Agent DSE, Tool Interfacing, and Autonomous Driving Simulation—to establish a closed feedback loop for iterative DSE and identify Pareto-optimal, cost-efficient solutions.

---

[Attention is All You Need to Defend Against Indirect Prompt Injection Attacks in LLMs](http://arxiv.org/abs/2512.08417)

- RENNERVATE: introduces a defense framework to detect and prevent Indirect Prompt Injection (IPI) attacks by leveraging LLM attention features at a fine-grained token level.
- The framework utilizes a Token-level Detector with a 2-step attentive pooling mechanism to aggregate attention heads and response tokens for robust IPI detection and precise sanitization.
- By localizing and removing injected tokens via the Injection Sanitizer, the system neutralizes IPI attacks while maintaining the benign functionality of the LLM-integrated application.

---

[Reflecting with Two Voices: A Co-Adaptive Dual-Strategy Framework for LLM-Based Agent Decision Making](http://arxiv.org/abs/2512.08366)

- DuSAR (Dual-Strategy Agent with Reflecting): introduces a demonstration-free framework enabling a single frozen LLM to perform co-adaptive reasoning via two complementary strategies: Holistic Strategy and Local Strategy, integrated by a Strategy Integration Module within a unified reflective loop.
- The framework uses a lightweight reflection mechanism where the Local Strategy continuously assesses execution progress via a Strategy Fitness Score, dynamically triggering the Holistic Strategy to revise the global plan upon stagnation or refinement upon advancement.
- This dual-strategy architecture achieves robust generalization and high token efficiency (3-9x reduction) by generating and refining structured plan graphs in situ through environmental interaction, avoiding reliance on external demonstrations or retrieval.

---

[THE HIGH COST OF INCIVILITY: QUANTIFYING INTERACTION INEFFICIENCY VIA MULTI-AGENT MONTE CARLO SIMULATIONS](http://arxiv.org/abs/2512.08345)

- MAD (Multi-Agent Discussion): introduces a Monte Carlo simulation framework using LLM agents to quantify the interaction inefficiency (latency of toxicity) caused by toxic behavior in adversarial debates.
- The framework simulates 1-on-1 debates between Proponent and Opponent agents, where one agent in the treatment group receives a "Toxic" system prompt modification to simulate social friction.
- An external Moderator Agent is used to neutrally evaluate if alignment or consensus has been reached, determining the primary metric $T_{conv}$ (number of arguments until conclusion).

---

[Argus: A Multi-Agent Sensitive Information Leakage Detection Framework Based on Hierarchical Reference Relationships](http://arxiv.org/abs/2512.08326)

- Argus (Multi-Agent Sensitive Information Leakage Detection Framework): introduces a multi-agent collaborative framework for detecting sensitive information leakage using a three-tier contextual semantic analysis mechanism.
- The architecture employs specialized LLM-agents—Initial Screening, Basic Check, Advanced Check, and Commander—coordinated via a Shared Memory Pool to effectively reduce false positives and enhance detection accuracy.
- The hierarchical detection process analyzes intrinsic key features (Level 1), immediate context (Level 2), and project reference relationships (Level 3), achieving 94.86% accuracy on real-world repository benchmarks.

---

[rSIM: Incentivizing Reasoning Capabilities of LLMs via Reinforced Strategy Injection](http://arxiv.org/abs/2512.08300)

- rSIM (reinforced strategy injection mechanism): introduces a multi-agent RL framework where a small Planner (leader agent) adaptively injects one of nine predefined Reasoning Strategies into the Reasoner's (follower LLM) Chain-of-Thought process to enhance reasoning capabilities.
- The Planner and Reasoner are jointly trained using a leader-follower algorithm and a Two-Stage Training Scheme to ensure stable policy optimization, enabling small LLMs (e.g., 0.5B) to achieve Reasoning Language Model performance.
- The trained Planner is pluggable and generalizable across different LLMs and tasks, supporting continual learning to improve reasoning guidance without requiring additional post-training of the base LLM.

---

[Systematization of Knowledge: Security and Safety in the Model Context Protocol Ecosystem](http://arxiv.org/abs/2512.08290)

- MCP (Model Context Protocol): introduces a Systematization of Knowledge (SoK) providing a comprehensive taxonomy of security and safety risks in the MCP ecosystem, which connects LLMs to external data and tools.
- The MCP architecture is founded on a Client-Host-Server model where the Host Application acts as the security boundary, mediating interactions and enforcing policy checks on tool execution requests.
- The SoK analyzes unique threats like context poisoning, indirect prompt injection, and supply-chain attacks, necessitating layered defenses such as cryptographic provenance (ETDI) and continuous monitoring (TRiSM).

---

[Empowering Smart App Development with SolidGPT: An Edge-Cloud Hybrid AI Agent Framework](http://arxiv.org/abs/2512.08286)

- SolidGPT (Edge-Cloud Hybrid AI Agent Framework): introduces a hybrid edge-cloud developer assistant built on GitHub that uses an MDP-driven routing mechanism to balance latency and privacy while enhancing code and workspace semantic search.
- The system employs a Multi-Agent Workflow, including PM, PE, and SDE agents, to facilitate the full software lifecycle from requirement gathering to code generation in a sequential pipeline.
- SolidGPT achieves semantic continuity across distributed execution stages using a context-retentive prompt engineering pipeline and deep MVVM integration for real-time, platform-aware analysis, reducing bug resolution time by 64%.

---

[AgentEval: Generative Agents as Reliable Proxies for Human Evaluation of AI-Generated Content](http://arxiv.org/abs/2512.08273)

- AgentEval: introduces a comprehensive LLM-based framework that simulates human evaluation of AI-generated content by integrating a Chain-of-Thoughts (CoT) module with a personalized Generative Agent (GA) module.
- The GA module uses Perception to ingest personalized human-like characteristics and CoT commands, storing them in a Memory Stream to facilitate iterative processes like Retrieve, Plan, and Reflect before generating a final rating.
- The system utilizes quantifiable Evaluation Criteria across five dimensions (Coherence, Relevance, Interestingness, Fairness, Clarity) to ensure reliable, reference-free, and cost-effective content assessment.

---

[Chat with UAV – Human-UAV Interaction Based on Large Language Models](http://arxiv.org/abs/2512.08145)

- UAV-GPT: introduces a novel dual-agent Human-UAV Interaction framework that constructs two independent LLM agents (Planning Agent and Execution Agent) to achieve precise task classification, reasonable planning, and efficient execution of complex UAV tasks.
- The Planning Agent classifies user intent using a two-dimensional complexity system, while the Execution Agent converts plans into constrained Machine Language Vectors (MLV) or dynamically invokes ROS-based tools like EgoPlanner for real-time obstacle avoidance.
- Experimental results demonstrate that the framework significantly improves Intent Recognition Accuracy, Task Execution Success Rate (45.5% gain), and UAV Energy Consumption efficiency compared to traditional single-agent LLM methods.

---

[Robust Agents in Open-Ended Worlds](http://arxiv.org/abs/2512.08139)

MiniHack: introduces a sandbox framework for creating diverse environments through procedural content generation, enabling the training and evaluation of robust reinforcement learning (RL) agents capable of generalising to novel environments and out-of-distribution inputs.
- MAESTRO (Multi-Agent Environment Design Strategist for Open-Ended Learning) extends Unsupervised Environment Design (UED) to multi-agent settings by jointly learning autocurricula over environment/co-player pairs to train robust RL agents in two-player zero-sum games, achieving minimax-regret guarantees at Nash equilibrium.
- MADRID (Multi-Agent Diagnostics for Robustness via Illuminated Diversity) and Rainbow Teaming leverage Quality-Diversity (QD) methods to systematically diagnose robustness by generating diverse adversarial scenarios for pre-trained multi-agent RL policies (MADRID) and adversarial prompts for LLMs (Rainbow Teaming).
- MADRID and Rainbow Teaming leverage Quality-Diversity (QD) methods to systematically diagnose robustness by generating diverse adversarial scenarios for pre-trained multi-agent RL policies (MADRID) and adversarial prompts for LLMs (Rainbow Teaming).

---

[Collaborative Causal Sensemaking: Closing the Complementarity Gap in Human-AI Decision Support](http://arxiv.org/abs/2512.07801)

- CCS (Collaborative Causal Sensemaking): introduces a research agenda to develop LLM-based agents capable of collaborative causal sensemaking, spanning new training environments, representations for shared human-AI mental models, and evaluation centered on trust and complementarity.
- The framework models expert-assistant interaction as a cooperative decision process, optimizing for task reward alongside epistemic and teleological alignment terms that reward shared understanding of evolving world models and goals.
- Architectural desiderata include persistent, structured models like neuro-symbolic Causal Twins and Episodic Sensemaking Memory, enabling agents to participate in discrepancy-driven sensemaking loops rather than merely imitating surface-level behavior.

---

[Multi-Docker-Eval: A ‘Shovel of the Gold Rush' Benchmark on Automatic Environment Building for Software Engineering](http://arxiv.org/abs/2512.06915)

- Multi-Docker-Eval (MDE) Benchmark: introduces a multilingual, multi-dimensional evaluation framework for assessing LLM agents' capacity for automated environment construction and test script generation across 40 real-world repositories.
- The benchmark evaluates both success (Fail-to-Pass rate) and efficiency (token consumption, wall time, resource usage) under realistic constraints, confirming that environment construction, particularly dependency resolution, is the primary bottleneck for current LLMs.
- Comparison of the multi-agent SWE-Builder and single-agent RepoLaunch frameworks highlights that feedback-driven, memory-augmented architectures are critical for reliable and scalable software engineering automation.

---

[DOVER: INTERVENTION-DRIVEN Auto DEBUGGING FOR LLM MULTI-AGENT SYSTEMS](http://arxiv.org/abs/2512.06749)

- DOVER (Intervention-Driven Auto Debugging for LLM Multi-Agent Systems): introduces a do-then-verify debugging pipeline that validates failure attribution hypotheses via targeted interventions and trajectory replay across four stages: Trial Segmentation, Failure Attribution, Intervention Generation, and Intervention Execution.
- The framework addresses limitations of log-only debugging by augmenting hypothesis generation with active verification through targeted edits (e.g., editing messages or altering plans) and measuring progress toward task success.
- DOVER successfully recovers 18-28% of failed trials in the Magnetic-One agent framework and 49% in the AG2 framework, demonstrating intervention as a practical mechanism for improving agentic system reliability.

---

[Supporting Dynamic Agentic Workloads: How Data and Agents Interact](http://arxiv.org/abs/2512.09548)

- ACDF (Agent-Centric Data Fabric): introduces a unified architecture that rethinks data systems as adaptive collaborators for dynamic agentic workloads, featuring Attention-guided data retrieval, Micro-caches, Attention-guided router, Predictive prefetcher, Cross-engine optimizer, Cross-agent cache manager, Quorum-based serving, Shared semantic cache, Engine fabric, and Monitoring.
- The architecture is structured into three layers—Agent, Orchestration, and Execution—to mediate high-level agent intent with concrete data access and heterogeneous backend systems.
- Key mechanisms like semantic micro-caching and quorum-based serving enable context-aware, cost-sensitive data access, reduce redundant queries, and foster cooperative data reuse among collaborating LLM-powered agents.

---

[WOLF: Werewolf-based Observations for LLM Deception and Falsehoods](http://arxiv.org/abs/2512.09187)

- WOLF (Werewolf-based Observations for LLM Deception and Falsehoods): introduces a multi-agent social deduction benchmark using a LangGraph state machine and role-grounded agents (Villager, Werewolf, Seer, Doctor) to measure LLM deception production and detection.
- The benchmark uses a granular deception measurement protocol where every public statement receives both speaker self-assessment and peer analysis, updating longitudinal suspicion scores via exponential smoothing.
- WOLF moves deception evaluation beyond static datasets by providing a dynamic, controlled testbed that reveals an asymmetry where LLMs lie often but detect lies only moderately well.

---

[SDialog: A Python Toolkit for End-to-End Agent Building, User Simulation, Dialog Generation, and Evaluation](http://arxiv.org/abs/2512.09142)

- SDialog (Python Toolkit for End-to-End Agent Building): introduces a unified, dialog-centric framework for building and analyzing LLM-based conversational agents, with standardized Dialog representation, Agents, Personas, Generators, Orchestrators, Interpretability, Evaluation, Audio, and LLM Backends.
- The toolkit unifies fragmented workflows—synthetic dialog generation, comprehensive multi-metric evaluation combining linguistic metrics and LLM-as-a-judge, and mechanistic interpretability—into a single, reproducible pipeline.
- The framework supports mixed-backend LLM integration, persona-driven multi-agent simulation, and advanced features like activation steering and full acoustic simulation for realistic spoken dialog corpora generation.

---

[Evolving Excellence: Automated Optimization of LLM-based Agents](http://arxiv.org/abs/2512.09108)

- ARTEMIS (Automated Optimization Platform): introduces a no-code evolutionary optimization platform that jointly optimizes LLM agent configurations, including prompts, tools, and parameters, using semantically-aware genetic operators.
- The platform treats agents as black boxes, leveraging benchmark outcomes and execution logs as feedback within a hierarchical evaluation strategy to efficiently discover non-obvious optimizations.
- The system demonstrated substantial performance improvements across four diverse agent systems, including a 13.6% gain for the ALE Agent and a 36.9% token cost reduction for the CrewAI Agent.

---

[Mental Models of Autonomy and Sentience Shape Reactions to AI](http://arxiv.org/abs/2512.09085)

- MMAC (Mental Models of AI Capacities): introduces a study disentangling the effects of activating mental models of AI autonomy and sentience on human reactions, using preregistered vignette experiments featuring a hypothetical smart artificial assistant named Corion.
- Sentience activation increased general mind perception and moral consideration more significantly than autonomy, while autonomy primarily increased the perception of threat.
- A meta-analysis across four experiments confirmed that sentience had a larger overall impact on user reactions to AI than autonomy, suggesting it is a foundational mental model in Human-Computer Interaction (HCI).

---

[AgentComp: From Agentic Reasoning to Compositional Mastery in Text-to-Image Models](http://arxiv.org/abs/2512.09081)

- AgentComp: introduces an agentic framework that autonomously constructs high-quality contrastive compositional datasets using multi-agent orchestration and specialized tools.
- The framework applies Agent Preference Optimization (APO), a distance-aware preference learning method, to fine-tune text-to-image (T2I) models to distinguish between compositionally similar generation paths.
- This approach significantly enhances compositional reasoning and generalization in T2I models, achieving state-of-the-art results on benchmarks while preserving image quality and improving text rendering.

---

[Autonomous Issue Resolver: Towards Zero-Touch Code Maintenance](http://arxiv.org/abs/2512.08492)

- AIR (Autonomous Issue Resolver): introduces a multi-agent framework for zero-touch code maintenance that utilizes neuro-symbolic reasoning over a Data-First Transformation Graph (DTG).
- The DTG shifts the paradigm from control-centric Code Property Graphs (CPGs) to a data-centric view, modeling data states as nodes and functions as edges to trace logic defects through data lineage.
- The system employs a decoupled "Plan-Navigate-Execute" loop managed by the Context, Maintenance, Editor, and Validation Agents, integrating Reinforcement Learning for risk-aware control policy and navigation.

---

#### 8th December 2025

[An Adaptive Multi-Layered Honeynet Architecture for Threat Behavior Analysis via Deep Learning](http://arxiv.org/abs/2512.07827)

- ADLAH (Adaptive Deep Learning Anomaly Detection Honeynet): introduces an adaptive multi-layered honeynet architecture that uses a DQN+LSTM RL agent to orchestrate the dynamic deployment of high-interaction honeypot pods based on real-time first-packet analysis from low-interaction MADCAT sensor nodes.
- The architecture centralizes data processing in a Hive node (ELK stack) and includes an AI analytics pipeline featuring an adaptive autoencoder for continuous anomaly detection and modules for automated attack chain extraction and bot versioning.
- This system shifts adaptation from the service level to the infrastructure level, maximizing high-fidelity threat intelligence capture while minimizing resource costs through selective escalation.

---

[Optimization-Guided Diffusion for Interactive Scene Generation](http://arxiv.org/abs/2512.07661)

- OMEGA (Optimization-Guided Diffusion for Interactive Scene Generation): introduces a training-free framework that enhances diffusion-based scene generation fidelity and controllability using constrained optimization, incorporating a Diffusion Model, Optimization-Guided Refinement, KL-Bounded Trust Region, Two-Phase Noise Scheduling, Behavior Guidance, and a Sensitivity-Enhanced Adversarial Generator.
- The core mechanism re-anchors each reverse diffusion step via constrained optimization within a KL-bounded trust region, steering the Markov chain toward physically consistent and behaviorally coherent trajectories.
- A two-phase noise scheduling scheme, comprising Warmup (global organization) and Rolling-Zero (local adaptation), improves interaction realism and temporal stability in multi-agent scene evolution.

---

[The Agent Capability Problem: Predicting Solvability Through Information-Theoretic Bounds](http://arxiv.org/abs/2512.07631)

- ACP (Agent Capability Problem): introduces a framework for predicting task solvability and resource requirements by modeling problem-solving as information acquisition, utilizing $C_{effective}$, $I_{total}$, $I_s$, and an Optimal Action Selection Policy.
- The core metric, Effective Cost ($C_{effective}$), provides a theoretical lower bound on expected search cost, guiding resource allocation and action selection based on information-to-cost ratios.
- Experimental validation, including deployment on an LLM agent for noisy parameter identification, confirms that ACP predictions reliably track actual performance and serve as consistent lower bounds.

---

[How Do LLMs Fail In Agentic Scenarios? A Qualitative Analysis of Success and Failure Scenarios of Various LLMs in Agentic Simulations](http://arxiv.org/abs/2512.07497)

- KAMI v0.1 (Kamiwaza Agentic Merit Index): introduces a qualitative analysis of 900 LLM Agent execution traces across three models (Granite 4 Small, Llama 4 Maverick, DeepSeek V3.1) using a Tool Suite within the KAMI v0.1 Benchmark, focusing on identifying success strategies and failure modes.
- The analysis identifies four recurring failure archetypes: premature action without grounding, over-helpfulness leading to autonomous substitution, sensitivity to context pollution, and fragile execution under cognitive load.
- Findings emphasize that agentic reliability is primarily predicted by robust Error Recovery Mechanisms and systematic verification behaviors, rather than model scale or initial accuracy.

---

[Understanding LLM Agent Behaviours via Game Theory: Strategy Recognition, Biases and Multi-Agent Dynamics](http://arxiv.org/abs/2512.07462)

- FAIRGAME (Framework for AI Agents Bias Recognition using Game Theory): introduces an integrated framework for understanding LLM agent behaviors using game-theoretic benchmarks and a machine learning pipeline, including a Payoff-scaling module, a Multi-agent extension (PGG), and an LSTM-based intent recognition ML Pipeline.
- The framework systematically evaluates LLM behavior in repeated social dilemmas, revealing consistent behavioral signatures like incentive-sensitive cooperation, cross-linguistic divergence, and end-game alignment toward defection.
- The ML Pipeline, utilizing LSTM for robustness against execution noise, classifies LLM gameplay trajectories against canonical strategies to infer latent intentions and systematic biases across models and languages.

---

[Social welfare optimisation in well-mixed and structured populations](http://arxiv.org/abs/2512.07453)

- SWO-EGT (Social Welfare Optimisation in Evolutionary Game Theory): introduces a single-objective approach focused on maximizing social welfare in evolutionary game theory models, using EGT Model, Institutional Incentives, Social Welfare Metric, Population Structure, Interference Strategies, Mathematical Analysis, and Agent-based Simulation.
- The approach compares optimal strategies for maximizing social welfare against traditional objectives like minimizing institutional cost or maximizing cooperation frequency in both well-mixed and structured populations.
- Results indicate a significant gap between optimizing for cost efficiency or cooperation frequency and optimizing for maximal social welfare, suggesting welfare-centric objectives should be prioritized in incentive design.

---

[MASIM: Multilingual Agent-Based Simulation for Social Science](http://arxiv.org/abs/2512.07195)

- MASIM (Multilingual Agent-Based Simulation): introduces a framework for simulating multi-turn social interactions among User Agents (role-play social media users) and News Organization Agents (role-play media entities), grounded by the MAPS Dataset (survey questions and personas), utilizing Short-term Memory (chain-of-thought reasoning) and Long-term Memory (cross-round takeaway messages), and mediated by a Multilingual Recommendation System (R) (embeds agents and posts).
- The system tracks how agent attitudes evolve toward a survey item over iterative rounds, where agents read recommended content, compose posts, and vote to update their Attitude Distribution Output (D).
- MASIM enables scalable and controlled computational social science by modeling cross-lingual interaction and sociolinguistic diversity, supporting global public opinion and media influence analysis.

---

[VIGIL: A Reflective Runtime for Self-Healing LLM Agents](http://arxiv.org/abs/2512.07094)

- VIGIL (Verifiable Inspection and Guarded Iterative Learning): introduces a reflective, out-of-band runtime that supervises a target LLM agent, performing autonomous maintenance via affective appraisal and stage-gated remediation.
- The system ingests behavioral logs, transforms them into structured emotions stored in the persistent EmoBank, and generates a Roses/Buds/Thorns (RBT) diagnosis to identify latent failures.
- Based on the RBT diagnosis, VIGIL generates concrete adaptations, including guarded prompt updates and read-only code proposals (unified diffs), demonstrating meta-procedural self-repair capacity.

---

[Adaptation of Embedding Models to Financial Filings via LLM Distillation](http://arxiv.org/abs/2512.08088)

- ILDP (Iterative LLM Distillation Pipeline): introduces a scalable pipeline for domain adaptation of retrieval embedding models using LLM-judged relevance to distill knowledge into a compact bi-encoder, leveraging iterative hard example mining from SEC filings.
- The pipeline uses a Generative Model (Teacher LLM) for synthetic query generation and relevance scoring, enabling the fine-tuning of the Student Bi-encoder on millions of contrastive triples.
- The iterative process refines the Student Embedding Model in each step to mine progressively harder positive and negative training examples from the unlabeled corpus, significantly improving retrieval metrics in specialized financial domains.

---

[Automating High Energy Physics Data Analysis with LLM-Powered Agents](http://arxiv.org/abs/2512.07785)

- LLM4HEP (LLM-Agent-Driven Automated Data-Analysis Framework): introduces a hybrid system combining the Snakemake (Workflow Orchestration) manager with a Supervisor Agent (Task Decomposition/Review) and a Coder Agent (Code Generation/Execution) to automate High Energy Physics data analysis.
- The architecture uses the Snakemake manager to enforce determinism and reproducibility across five sequential analysis steps, while the LLM agents autonomously generate, execute, and iteratively correct analysis code.
- The framework enables systematic benchmarking of LLM capabilities, stability, and limitations across complex, multi-stage scientific computing tasks, using metrics like success rate and agent work.

---

[DeepCode: Open Agentic Coding](http://arxiv.org/abs/2512.07921)

- DeepCode: introduces a fully autonomous framework for high-fidelity document-to-codebase synthesis, managing information flow via Blueprint Generation (source compression), Code Generation (synthesis phase), and Automated Verification (error correction).
- The framework addresses the LLM context bottleneck by orchestrating four operations: blueprint distillation, structured indexing using CodeMem, conditional knowledge injection via CodeRAG, and closed-loop error correction.
- DeepCode achieves state-of-the-art performance on the PaperBench benchmark, decisively outperforming commercial agents and surpassing human expert performance on reproduction metrics.

---

[Reliable agent engineering should integrate machine-compatible organizational principles](http://arxiv.org/abs/2512.07665)

- MCOP (Machine-Compatible Organizational Principles): introduces a framework for reliable LLM agent engineering by applying organizational science principles to agent design, scaling, and management, utilizing components like LLM Agents, Orchestrator, and Reward Mechanisms.
- The approach emphasizes balancing agency and capabilities in agent design, managing resource constraints during scaling, and implementing internal and external mechanisms for behavior management to ensure reliability.
- Architectural components are structured into single-agent tool use, multiagent systems (MAS) with provider-bundled agents, or MAS with supportive tooling agents, each defining distinct delegation and accountability structures.

---

[VulnLLM-R: Specialized Reasoning LLM with Agent Scaffold for Vulnerability Detection](http://arxiv.org/abs/2512.07533)

- VulnLLM-R: introduces VulnLLM-R, a 7B parameter specialized reasoning LLM trained via distillation from SOTA teacher models and a novel recipe including reasoning data filtering and correction, designed for superior vulnerability detection.
- The system integrates VulnLLM-R into an agent scaffold featuring a function selector and a context retriever, enabling project-level vulnerability detection and the discovery of zero-day vulnerabilities in real-world repositories.
- VulnLLM-R demonstrates superior effectiveness and parameter efficiency compared to SOTA static analysis tools and commercial LLMs, achieving strong generalization across multiple programming languages and unseen CWEs.

---

[AutoICE: Automatically Synthesizing Verifiable C Code via LLM-driven Evolution](http://arxiv.org/abs/2512.07501)

- AutoICE: introduces an LLM-driven evolutionary search framework for synthesizing verifiable C code, featuring diverse individual initialization, collaborative crossover, and self-reflective mutation, guided by logical verifiers.
- The framework models code synthesis as the evolution of a code population, balancing exploration (uncovering implicit knowledge) and exploitation (progressive generation of high-quality code).
- AutoICE mitigates error propagation inherent in single-agent iterative approaches by leveraging LLMs in roles as initializer, crossover operator, and mutator.

---

[Enhancing Agentic RL with Progressive Reward Shaping and Value-based Sampling Policy Optimization](http://arxiv.org/abs/2512.07478)

- PRS (Progressive Reward Shaping) and VSPO (Value-based Sampling Policy Optimization): introduces two complementary techniques to enhance Agentic RL for Tool-Integrated Reasoning (TIR) agents by providing dense, stage-wise rewards and stabilizing policy optimization through value-based sample selection.
- PRS uses a curriculum approach, starting with rewards for parseable tool calls and formatted outputs, then progressing to factual correctness and answer quality, instantiated for both short- and long-form QA.
- VSPO, an improved GRPO variant, addresses gradient degradation by sampling prompts based on a task-value metric balancing uncertainty and difficulty, and applies value smoothing clipping to maintain stable gradient scales.

---

[Living the Novel: A System for Generating Self-Training Timeline-Aware Conversational Agents from Novels](http://arxiv.org/abs/2512.07474)

- Living Novel (LN): introduces an end-to-end system transforming literary works into multi-character conversational experiences using the Deep Persona Alignment (DPA) and Coherence and Robustness Enhancing (CRE) pipelines.
- DPA uses data-free reinforcement self-training to instill deep character fidelity, while CRE leverages a story-time-aware Diegetic Knowledge Graph to enforce narrative constraints and robustness.
- The system is deployed via a decoupled client-server architecture, utilizing Dual-Level Story-Time Gated Retrieval to ensure spoiler-free, interruption-resilient mobile experiences.

---

[CFD-copilot: leveraging domain-adapted large language model and model context protocol to enhance simulation automation](http://arxiv.org/abs/2512.07917)

- CFD-copilot: introduces a domain-specialized LLM framework for end-to-end Computational Fluid Dynamics (CFD) simulation automation, utilizing a multi-agent system for setup and an MCP-enabled client-server architecture for scalable post-processing.
- The simulation setup employs a self-correcting loop involving pre-checker, generator (fine-tuned LLM), runner, and corrector agents to translate natural language into executable OpenFOAM configurations.
- The Model Context Protocol (MCP) decouples LLM reasoning from external tool execution, allowing the framework to interact with numerous specialized post-processing functions via a unified, scalable interface.

---

[Native Parallel Reasoner: Reasoning in Parallelism via Self-Distilled Reinforcement Learning](http://arxiv.org/abs/2512.07461)

- NPR (Native Parallel Reasoner): introduces a teacher-free framework that enables LLMs to self-evolve genuine parallel reasoning capabilities via a self-distilled progressive training paradigm.
- The framework utilizes a novel Parallel-Aware Policy Optimization (PAPO) algorithm to optimize branching policies directly within the execution graph, allowing adaptive decomposition via trial and error.
- NPR ensures stable, large-scale parallel RL training by employing a robust NPR Engine that refactors memory management and flow control for 100% genuine parallel execution.

---

[SIT-GRAPH: STATE INTEGRATED TOOL GRAPH FOR MULTI-TURN AGENTS](http://arxiv.org/abs/2512.07287)

- SIT-Graph (State Integrated Tool Graph): introduces a unified graph structure for multi-turn tool use that jointly encodes compact state summaries and recurring tool-to-tool dependencies mined from past trajectories.
- The framework enables the agent to adaptively balance between episodic recall (using state summaries) and procedural execution (using edge weights) for robust tool selection.
- The approach utilizes a dedicated, invocable state summarization tool for on-demand state consolidation, improving context-aware decision-making and experience transfer.

---

[ClinNoteAgents: An LLM Multi-Agent System for Predicting and Interpreting Heart Failure 30-Day Readmission from Clinical Notes](http://arxiv.org/abs/2512.07081)

- ClinNoteAgents (LLM Multi-Agent System): introduces an LLM-based multi-agent framework that transforms unstructured discharge notes into structured representations of clinical and social risk factors and clinician-style abstractions for HF 30-day readmission prediction.
- The system utilizes three core agents—an extractor, a normalizer/labeler, and a summarizer—to enable scalable and interpretable risk modeling from free-text clinical documentation.
- The framework supports two core tasks: mining risk factors for statistical analysis and generating concise summaries for downstream predictive modeling, reducing reliance on structured EHR data.

---

[Personalizing Agent Privacy Decisions via Logical Entailment](http://arxiv.org/abs/2512.05065)

- ARIEL (Agentic Reasoning with Individualized Entailment Logic): introduces a framework that jointly leverages an LLM and rule-based logic to personalize agent privacy decisions by formulating data sharing as a logical entailment problem grounded in prior user judgments.
- The framework operates in offline and online phases, using the LLM for Ontology Generation and Mapping, and a rule-based Entailment Logic component to reliably infer judgments from prior user decisions.
- ARIEL ensures alignment with user preferences, interpretability via traceable reasoning, and maintains user agency by escalating requests that are not logically entailed by prior judgments to the user.

---

[Topology Matters: Measuring Memory Leakage in Multi-Agent LLMs](http://arxiv.org/abs/2512.04668)

- MAMA (Multi-Agent Memory Attack): introduces a systematic framework to measure memory leakage in multi-agent LLM systems using a two-phase protocol: Engram (memory seeding) and Resonance (multi-round extraction).
- The framework evaluates six canonical communication topologies (complete, ring, chain, binary tree, star, star-ring) to quantify how network structure and agent placement govern the diffusion and leakage of PII Entities.
- Results consistently show that dense topologies (complete) maximize leakage, sparse topologies (chain, tree) provide protection, and leakage follows a rapid-rise then plateau dynamic, yielding actionable design guidance.

---

[Bayesian Co-Navigation of a Computational Physical Model and AFM Experiment to Autonomously Survey a Combinatorial Materials Library](http://arxiv.org/abs/2512.08084)

- Bayesian Co-Navigation: introduces a multi-loop active-learning framework that tightly integrates an autonomous AFM experiment and a computationally expensive kMC physical model to dynamically refine the model parameters in real time.
- The system orchestrates three concurrent Bayesian optimization cycles—Experiment, Theory, and Update Theory Loops—each utilizing Gaussian Process surrogates to guide uncertainty-based exploration and balance throughputs.
- The outer Update Theory Loop minimizes the Mean Squared Error between the experimental and theoretical surrogate predictions by optimizing the kMC model's effective bond energy hyperparameters ($E_{XY}$).

---

[Optimized Area Coverage in Disaster Response Utilizing Autonomous UAV Swarm Formations](http://arxiv.org/abs/2512.08028)

- SNF (Swarm Navigation Framework): introduces a decentralized UAV swarm system for disaster response, utilizing autonomous UAV Agents, Perception Module, State Estimation, ESDF Map, TSP Variant, Local Planning Optimization, Controller, Swarm Bridge Broadcaster, and Formation Maintenance to maximize area coverage and ensure collision avoidance.
- The system employs a priority-based Prize-Collecting Traveling Salesman Problem (PC-TSP) variant to optimize global routing among prioritized Points of Interest (POIs) while incorporating time window constraints.
- Local navigation relies on an optimization-based trajectory replanning algorithm that uses an incrementally built Euclidean Signed Distance Field (ESDF) map to ensure collision-free B-Spline paths while maintaining swarm formation integrity.

---

[CAN AI AUTONOMOUSLY BUILD, OPERATE, AND USE THE ENTIRE DATA STACK?](http://arxiv.org/abs/2512.07926)

- Agentic DataOps: introduces a vision for an autonomous data stack managed by collaborating intelligent agents, covering the entire data lifecycle from infrastructure design to insight generation.
- The architecture utilizes specialized agents (Storage, Discovery, Acquisition, Curation, Life-Cycle, BI) operating hierarchically and chained to perform complex data management tasks autonomously.
- Achieving full autonomy requires foundational cross-stack capabilities, including LLM-based autonomous planning, continuous learning via feedback, and robust governance/factuality mechanisms.

---

#### 7th December 2025

[Know your Trajectory - Trustworthy Reinforcement Learning deployment through Importance-Based Trajectory Analysis](http://arxiv.org/abs/2512.06917)

- Importance-Based Trajectory Analysis (IBTA): introduces a framework for Explainable RL that ranks entire trajectories using a Modified Importance Metric and generates contrastive explanations via counterfactual rollouts.
- The core metric combines the classic Q-value difference ($\Delta Q$) with a V-Goal radical term ($R(s, a)$) to robustly capture state criticality and the agent's goal affinity.
- The pipeline identifies optimal trajectories from heterogeneous experience data and demonstrates their superiority by showing that deviations lead to worse outcomes.

---

[SOK: TRUST-AUTHORIZATION MISMATCH IN LLM AGENT INTERACTIONS](http://arxiv.org/abs/2512.06914)

- SOK (Systematization of Knowledge): introduces the B-I-P Security Model, Trust-Authorization Matrix, Mismatch Process, Belief, Intention, Permission, Action, Trust Aggregator, Authorization Risk, and Observability Constraints, providing a unifying formal lens to analyze security failures in LLM agent interactions stemming from a trust-authorization mismatch.
- The B-I-P model formalizes agent interaction security by tracking the chain from Belief corruption to Intention formation, Permission-intent intersection, and resulting Failure (Action).
- The Trust-Authorization Matrix maps system states based on Trust (epistemic soundness and provenance) and Authorization Risk, highlighting the Failure State (Low-Trust/High-Risk) as the critical vulnerability.

---

[COGNITIVE CONTROL ARCHITECTURE (CCA): A LIFECYCLE SUPERVISION FRAMEWORK FOR ROBUSTLY ALIGNED AI AGENTS](http://arxiv.org/abs/2512.06716)

- CCA (Cognitive Control Architecture): introduces a dual-layered supervision framework achieving full-lifecycle cognitive supervision via Pillar I (Proactive Control) which generates the Intent Graph (G_intent) monitored by the Controller (First Layer) in the Execution Loop, and Pillar II (Reactive Adjudication) which uses the Tiered Adjudicator and Adjudicator Model (M_adj) to calculate the Intent Alignment Score (S_align) from four sub-scores (S_sem, S_causal, S_prov, S_risk) to govern the Core Agent Model.
- Pillar I proactively enforces control-flow and data-flow integrity by validating proposed actions against the pre-generated Intent Graph, efficiently filtering overt planning deviations before execution.
- Pillar II intervenes only upon deviation detection, performing computationally expensive deep causal reasoning using the multi-faceted Intent Alignment Score to counter sophisticated, semantically covert Indirect Prompt Injection attacks.

---

[Reformulate, Retrieve, Localize: Agents for Repository-Level Bug Localization](http://arxiv.org/abs/2512.07022)

- RRL-Agent (Reformulate, Retrieve, Localize Agent): introduces an LLM Agent (Orchestrates localization workflow) that uses a Query Reformulation Module (Extracts structured information) and a BM25 Search Tool (Performs lexical retrieval) for Space Reduction Step (Narrows candidate files), followed by a Refinement Step (Views individual files), supported by a Self-Correction Mechanism (Validates and corrects) and a Self-Evaluation Mechanism (Reviews and revises ranking) to improve file-level bug localization.
- The approach leverages lightweight query reformulation to transform noisy bug reports into structured, retrieval-ready summaries, significantly boosting first-file retrieval accuracy by up to 35% over the BM25 baseline.
- This agentic workflow, utilizing open-source, non-fine-tuned LLMs and lexical retrieval, provides a scalable and resource-efficient alternative to end-to-end LLM reasoning for large repository traversal.

---

[BabelCoder: Agentic Code Translation with Specification Alignment](http://arxiv.org/abs/2512.06902)

- BabelCoder: introduces an agentic framework for automated code translation by decomposing the task into specialized Translation, Test, and Refinement Agents that collaboratively improve translation quality.
- The framework leverages Natural Language Specifications (NL-Specification) as an intermediate, language-agnostic representation to guide semantic translation and ensure specification alignment.
- The Refinement Agent employs a multi-step iterative repair process, integrating novel techniques like NL-Specification Validation and a Bug Localization system using SBFL and LLM-Based Scope Estimation for targeted fixes.

---

[Formal that "Floats" High: Formal Verification of Floating Point Arithmetic](http://arxiv.org/abs/2512.06850)

- AIFVW (Agentic AI-Based Formal Verification Workflow): introduces a scalable methodology for verifying floating-point arithmetic using direct RTL-to-RTL model checking against a golden reference model, supported by LLM-driven property generation and HITL refinement.
- The verification strategy employs hierarchical decomposition to partition the floating-point adder design into modular stages, such as the Mantissa Alignment Stage and Add-Round Stage, proving correctness via theorems, lemmas, and stage-level assertions.
- The multi-agent system coordinates specialized LLM agents across Planning, Generation, and Execution stages to automatically synthesize SystemVerilog Assertions, which are iteratively refined using CEX feedback and expert guidance.

---

[ProAgent: Harnessing On-Demand Sensory Contexts for Proactive LLM Agent Systems](http://arxiv.org/abs/2512.06721)

- ProAgent: introduces an end-to-end proactive agent system that integrates multisensory perception and LLM reasoning to deliver unobtrusive assistance by continuously sensing the environment and anticipating user needs.
- The system employs an On-Demand Tiered Perception strategy to coordinate low-cost, always-on sensors with high-cost, on-demand sensors, ensuring efficient capture of proactive-relevant cues.
- A Context-Aware Proactive Reasoner, based on a unified Advanced VLM, maps hierarchical sensory and persona contexts to user needs, tool calls, and proactive scores under Temporal Constraints.

---

[Look Twice before You Leap: A Rational Agent Framework for Localized Adversarial Anonymization](http://arxiv.org/abs/2512.06713)

- RLAA (Rational Localized Adversarial Anonymization): introduces a training-free, fully localized framework to prevent utility collapse in LLM-based text anonymization by enforcing rational decision-making via an Attacker-Arbitrator-Anonymizer (A-A-A) architecture.
- The Arbitrator acts as a rationality gatekeeper, using meta-reasoning and economic marginal analysis (MPG vs. MUC) to validate the Attacker's inferences and enforce a rational early-stopping criterion.
- By filtering out irrational feedback and zero-gain modifications, the framework maintains a stable Marginal Rate of Substitution (MRS) trajectory, achieving a superior privacy-utility trade-off compared to greedy adversarial methods.

---

[PERSONAMEM-V2: Towards Personalized Intelligence via Learning Implicit User Personas and Agentic Memory](http://arxiv.org/abs/2512.06688)

- Agentic Memory Framework: introduces a scalable approach for implicit LLM personalization using Reinforcement Fine-Tuning (RFT) (Optimization algorithm) and Reward Functions (Verifiable personalization signals) to train a Language Model (Generates responses) that iteratively processes Conversation Chunks (Fixed-size history segments) via a Memory Update Mechanism (Iterative memory refinement) to maintain a compact Agentic Memory (Compact user persona).
- The framework achieves state-of-the-art personalization accuracy on the PERSONAMEM-V2 benchmark while using 16x fewer input tokens compared to long-context reasoning over full conversation histories.
- PERSONAMEM-V2 is a state-of-the-art dataset simulating 1,000 realistic user-chatbot interactions across 300+ scenarios, focusing on inferring implicit user preferences from long, noisy conversation histories.

---

[An Index-based Approach for Efficient and Effective Web Content Extraction](http://arxiv.org/abs/2512.06641)

- Index-based Web Content Extraction: introduces a novel paradigm that reframes web content extraction from slow token-by-token generation into a highly efficient index prediction task, utilizing Index Construction, IndexLM, and Post-processing.
- The method partitions the HTML source code into structure-aware, addressable content blocks, allowing the specialized IndexLM to output only positional indices corresponding to query-relevant content.
- This discriminative approach decouples extraction latency from content length, achieving superior accuracy and speed compared to generative and heuristic extraction baselines in RAG QA systems.

---

#### 6th December 2025

[The Evolution of Agentic AI in Cybersecurity: From Single LLM Reasoners to Multi-Agent Systems and Autonomous Pipelines](http://arxiv.org/abs/2512.06659)

- Five-Generation Taxonomy: introduces an architectural evolution of agentic AI in cybersecurity, progressing from text-only LLM reasoners (Gen-1) to fully autonomous pipelines (Gen-5), encompassing tool-augmented agents (Gen-2), multi-agent collaboration (Gen-3), and schema-bound tool ecosystems (Gen-4).
- This survey compares these generations across core dimensions, including reasoning depth, tool use, memory, reproducibility, and safety, highlighting the parallel rise in capability and risk.
- The taxonomy provides a structured perspective on how agentic AI is shaping cybersecurity operations and outlines the necessary safeguards for safe, reliable, and verifiable deployment in high-stakes SOC environments.

---

[ChargingBoul: A Competitive Negotiating Agent with Novel Opponent Modeling](http://arxiv.org/abs/2512.06595)

- ChargingBoul: introduces a competitive negotiating agent that leverages structured opponent modeling and adaptive bidding techniques to maximize individual utility in multiattribute negotiations.
- The agent classifies opponents into Boulware, Hardliner, or Conceder categories using two novel statistics: the Unique Bid Index (UBI) and the Average Utility Index (AUI).
- The system dynamically adjusts its Bidding Strategy and concession policy based on the opponent's classification to ensure competitive outcomes while fostering agreements.

---

[HiveMind: Contribution-Guided Online Prompt Optimization of LLM Multi-Agent Systems](http://arxiv.org/abs/2512.06432)

- HiveMind: introduces a self-adaptive framework for LLM multi-agent systems using CG-OPO (closed-loop prompt optimization) guided by DAG-Shapley (efficient contribution measurement) within a DAG-structured Multi-Agent System, utilizing Performance-Based Reflection and Prompt Metamorphosis.
- CG-OPO is an iterative, four-stage process that autonomously identifies bottleneck agents based on quantified Shapley contributions and refines their prompts using LLM-driven reflection.
- DAG-Shapley leverages the inherent Directed Acyclic Graph structure to axiomatically prune non-viable coalitions and use Generalized Hierarchical Memoization (HGM Cache), reducing LLM calls by over 80%.

---

[GENIUS: An Agentic AI Framework for Autonomous Design and Execution of Simulation Protocols](http://arxiv.org/abs/2512.06404)

- GENIUS (An Agentic AI Framework for Autonomous Design and Execution of Simulation Protocols): introduces an agentic workflow that fuses a smart Quantum ESPRESSO knowledge graph with a tiered hierarchy of LLMs supervised by a Finite State Machine (FSM) for error recovery, automating DFT simulation protocol design and execution.
- The framework achieves high reliability by using the Smart Knowledge Graph (KG) to ground LLM outputs, ensuring syntactic and physical consistency, and sharply reducing hallucinations.
- The system includes a Recommendation System, a Protocol Generation System, and an Automated Error Handling (AEH) loop that iteratively diagnoses and corrects failed runs, achieving successful completion on approximately 80% of diverse benchmarks.

---

[Web Technologies Security in the AI Era: A Survey of CDN-Enhanced Defenses](http://arxiv.org/abs/2512.06390)

- CEDAID (CDN-Enhanced AI/ML Defenses): introduces a systematic survey of AI/ML defenses deployed within CDN/WAAP stacks, including Traffic Collection (Telemetry), Preprocessing (Normalization), Feature Extraction (Signals), ML Classification Models (Detection), Mitigation Action (Enforcement), Feedback Logging (Retraining), Logic Server Pool (Policy), and Federated Learning (Privacy-preserving training).
- The approach leverages the proximity and scale of CDN Points of Presence (PoPs) to perform real-time, privacy-aware inspection and adaptive mitigation against threats like DDoS, bots, and API abuse.
- Operational success relies on robust MLOps practices, including SLO-centric evaluation, safe rollouts, concept drift monitoring, and privacy-preserving techniques like Federated Learning.

---

[Learning When to Switch: Adaptive Policy Selection via Reinforcement Learning](http://arxiv.org/abs/2512.06250)

- QLAS (Q-Learning for Adaptive Switching): introduces a reinforcement learning technique to dynamically learn optimal switching thresholds between systematic exploration (Spiral Exploration) and goal-directed exploitation (A* Pathfinding) in maze navigation.
- The agent uses a compact 50-state representation based on coverage percentage and Manhattan distance to select a discrete threshold action (20-60%) via tabular Q-learning.
- This adaptive approach significantly outperforms fixed-threshold and single-strategy baselines, achieving 23-55% improvements in completion time and reducing runtime variance by up to 83% as problem complexity scales.

---

[Towards Efficient Hypergraph and Multi-LLM Agent Recommender Systems](http://arxiv.org/abs/2512.06590)

- HGLMRec (Hypergraph and Multi-LLM Agent Recommender System): introduces a novel multi-LLM agent-based recommender system integrating a Hypergraph Encoder, Token Fusion Module, and Hierarchical MoA Framework to capture complex user-item relationships efficiently.
- The Hypergraph Encoder uses an HGNN and Adaptive Readout to generate dense tokens encoding local and global preference patterns from multi-behavior interactions.
- The MoA Framework employs multiple specialized Frozen LLM Agents with dynamic weighting to iteratively refine fused tokens, reducing hallucination and computational cost compared to single-LLM methods.

---

[The Effect of Belief Boxes and Open-mindedness on Persuasion](http://arxiv.org/abs/2512.06573)

- MADF (Multi-Agent Debate Framework): introduces a system for topic-driven debate simulation using LLM-based Agents equipped with a Belief Box, an Open-mindedness Scale, and a Belief Evaluation Mechanism, operationalized via Prompt Design within a structured Debate Structure.
- The Belief Box explicitly encodes an agent's epistemic commitments as text propositions with Likert scale strength values (1-5), influencing persuasiveness and resistance to opposing viewpoints.
- Experiments confirm that prompting LLMs to be open-minded increases belief change rates, and the framework successfully models peer pressure effects in multi-agent scenarios.

---

[Securing the Model Context Protocol: Defending LLMs Against Tool Poisoning and Adversarial Attacks](http://arxiv.org/abs/2512.06556)

- Model Context Protocol (MCP) Security Framework: introduces a layered defense stack to secure LLM agents against descriptor-level semantic attacks, including Tool Poisoning, Shadowing, and Rug Pulls.
- The defense stack comprises RSA-based manifest signing, LLM-on-LLM vetting, and static heuristic guardrails to secure the tool invocation pipeline by treating tool metadata as untrusted input.
- Evaluation across GPT-4, DeepSeek, and Llama-3.5 confirms a fundamental latency-safety trade-off, showing that structured prompting enhances safety but increases response time and latency.

---

[Convergence of Outputs When Two Large Language Models Interact in a Multi-Agentic Setup](http://arxiv.org/abs/2512.06256)

- Multi-LLM Interaction Setup: investigates the convergence behavior when two independent LLMs, Mistral Nemo Base 2407 (LLM 1) and Llama 2 13B hf (LLM 2), respond alternately to each other's raw text output for 25 turns, starting from a Seed Sentence.
- The minimal setup uses File I/O Communication and a Synchronization Barrier to ensure ordered, deterministic interaction without shared memory, prompts, or external supervision.
- Convergence, characterized by repetitive and similar outputs, is quantified using multiple metrics, including Cosine Distance, Jaccard Distance, BLEU Overlap, and Coherence, often detected automatically via Collapse Detection Thresholding.

---

[DUET: Agentic Design Understanding via Experimentation and Testing](http://arxiv.org/abs/2512.06247)

- DUET (Design Understanding via Experimentation and Testing): introduces a general methodology for developing design understanding in hardware verification by equipping an LLM Agent (Generates hypotheses and reports) with Tools (External utilities) to perform iterative Experimentation Loop (Iterative hypothesis testing) on the Design (RTL) (Input hardware description).
- The core DOEXPERIMENTATION procedure involves the LLM Agent iteratively generating hypotheses, testing them using EDA tools (Simulation Tool, Formal Tool), and integrating the results via Messages (Context history) to refine its understanding of complex RTL behaviors.
- The methodology significantly improves AI agent performance on formal verification tasks by enabling deep design understanding through trial-and-error experimentation, particularly leveraging the Counterexample Replication Tool for debugging verification failures.

---

[Automated Data Enrichment using Confidence-Aware Fine-Grained Debate among Open-Source LLMs for Mental Health and Online Safety](http://arxiv.org/abs/2512.06227)

- CFD (Confidence-Aware Fine-Grained Debate): introduces a novel data enrichment framework where multiple LLM agents simulate human annotators and exchange fine-grained evidence to reach consensus on labels for mental health and online safety datasets.
- The framework utilizes a Categorical Chain of Thought (Cat-CoT) for initial response generation and conducts a Structured Debate guided by fine-grained confidence scores assigned to both reasoning steps and final answers.
- CFD consistently improves performance on downstream tasks, with enriched features incorporated via debate transcripts yielding the largest gains, outperforming the non-enriched baseline by 10.1% for the online safety task.

---

[DataGovBench: Benchmarking LLM Agents for Real-World Data Governance Workflows](http://arxiv.org/abs/2512.04416)

- DataGovAgent (Agentic Assembly Line): introduces an end-to-end NL2GovDAG framework for data governance, utilizing a sequential multi-agent pipeline consisting of a Planner, Executor, and Evaluator.
- The Planner converts natural language intent into a high-level Directed Acyclic Graph (DAG) of operations governed by formal governance contracts, ensuring topological coherence and executability.
- The Executor generates concrete Python code using Retrieval-Augmented Generation (RAG) over a curated library, while the Evaluator manages feedback-driven debugging in a sandbox to ensure functional correctness.

---

#### 5th December 2025

[Comparative Analysis of Autonomous and Systematic Control Strategies for Hole-Doped Hubbard Clusters: Reinforcement Learning versus Physics-Guided Design](http://arxiv.org/abs/2512.06095)

- Autonomous Deep Reinforcement Learning (RL) Framework: introduces a comparative study of control strategies for hole-doped Hubbard clusters using a Dueling Deep Q-Network (DQN) Agent, Geometry Embedding (GCN), Environment (ED Solver), State Vector, Action Space, and Reward Function.
- The RL agent achieves human-competitive accuracy ($R^2 > 0.97$) across five 3D lattices and demonstrates a $10^{3}-10^{4}\times$ greater sample efficiency compared to conventional grid search methods (Figures 2 and 3).
- The Dueling DQN, equipped with a geometry-aware embedding, learns an internal model of the system's physics, enabling intelligent, targeted exploration and 91% few-shot generalization to unseen geometries (Table I).

---

[Trusted AI Agents in the Cloud](http://arxiv.org/abs/2512.05951)

- Omega: introduces a system enabling trusted AI agents by enforcing end-to-end isolation, establishing verifiable cross-principal trust, and supervising external interactions with accountable provenance, utilizing CVMs, Confidential GPUs, and nested isolation via Trustlets.
- The platform features a Trusted Agent Platform (TAP) that consolidates multiple agents within a single CVM using VM Privilege Levels (VMPLs) for nested isolation and efficient multi-agent orchestration.
- Omega provides a declarative policy specification and enforcement framework, mediated by an orchestrator, which governs data access, tool usage, and inter-agent communication, recorded via tamper-evident audit logs.

---

[Optimal Safety-Aware Scheduling for Multi-Agent Aerial 3D Printing with Utility Maximization under Dependency Constraints](http://arxiv.org/abs/2512.05815)

- Multi-Agent Aerial Additive Construction Framework: introduces a novel coordination and task-planning framework for simultaneous, conflict-free collaboration of multiple Unmanned Aerial Vehicles (UAVs) in aerial 3D printing.
- The framework formulates a Mixed Integer Programming optimization problem that generates an optimal mission plan, including task assignments and scheduling, while accounting for geometric dependencies, inter-UAV safety, material usage, and flight time constraints.
- Safety is guaranteed by dynamically selecting the starting time and location of each task at a segment-level to ensure collision-free parallel execution, accelerated by an importance prioritization scheme and optimized for agent utility maximization.

---

[Task-Specific Trust Evaluation for Multi-Hop Collaborator Selection via GNN-Aided Distributed Agentic AI](http://arxiv.org/abs/2512.05788)

- GADAI (GNN-Aided Distributed Agentic AI): introduces a framework for multi-hop collaborator selection that performs independent evaluation of historical reliability using a GNN-aided model and task-specific resource trust using an LAM-enabled agentic AI system.
- The GNN-aided model constructs a historical collaboration graph to propagate and aggregate trust information across multi-hop neighbors, achieving robust and accurate assessments of historical reliability ($T^{His}$).
- The LAM-enabled agentic AI system empowers devices to autonomously assess resource trustworthiness ($T^{Res}$) and collaboratively plan a value-maximizing multi-hop cooperation path in a distributed manner.

---

[Beyond Prototyping: Autonomous, Enterprise-Grade Frontend Development from Pixel to Production via a Specialized Multi-Agent Framework](http://arxiv.org/abs/2512.06046)

- AI4UI: introduces a specialized multi-agent framework for autonomous front-end development, converting Figma designs into production-ready UI code using a Planner, Coder, Reviewer, and Tester Agent.
- The system achieves enterprise readiness by integrating an LLM-Friendly Grammar, domain-aware knowledge graphs, and a secure Abstract and Package Approach for proprietary functionality integration.
- The architecture utilizes a Change-Oriented Development Workflow and a Layered Compilation Integrity Module to ensure high compilation success, security compliance, and maintainable code quality.

---

[MCP-AI: Protocol-Driven Intelligence Framework for Autonomous Reasoning in Healthcare](http://arxiv.org/abs/2512.05365)

- MCP-AI (Protocol-Driven Intelligence Framework): introduces a novel architecture for autonomous clinical reasoning in healthcare, built upon the Model Context Protocol (MCP) to orchestrate generative- and descriptive-AI agents in real-time workflows.
- The system utilizes a five-layer modular structure, including the Input and Perception Layer, MCP Engine, AI Reasoning Modules, Task and Procedure Agents, and the Verification Module, ensuring traceability and context-awareness.
- MCP-AI supports adaptive, longitudinal, and collaborative reasoning across care settings, enabling physician-in-the-loop validation and adherence to regulatory standards like HIPAA and FDA SaMD guidelines.

---

[Natural Language Summarization Enables Multi-Repository Bug Localization by LLMs in Microservice Architectures](http://arxiv.org/abs/2512.05908)

- NL-BL (Natural Language Bug Localization): introduces a methodology for multi-repository bug localization by transforming source code into a Hierarchical NL Knowledge Base using context-aware summarization, followed by a scalable Two-Phase Search (Search Space Router, Directory-Level Filtering, File-Level Ranking).
- This approach reframes bug localization from a cross-modal retrieval task to a unified NL-to-NL reasoning task, leveraging the semantic understanding of LLMs while overcoming context window limitations.
- Evaluated on an industrial microservice system (DNext), the method achieved Pass@10 of 0.82 and MRR of 0.50, significantly outperforming RAG and retrieval baselines.

---

[Capturing Classic Authorial Style in Long-Form Story Generation with GRPO Fine-Tuning](http://arxiv.org/abs/2512.05747)

- GRPO (Group Relative Policy Optimization): introduces a training framework for style-conditioned long-form story generation, utilizing an 8B LLM fine-tuned with a custom multi-reward function based on Authorship Verification (AV) scores.
- The custom reward function combines a sigmoid-scaled style reward derived from a fine-tuned sentence transformer with auxiliary content and completeness rewards to stabilize long-form narrative coherence.
- The FT-Agentic (8B) model, trained using this RL pipeline, outperforms larger LLM baselines (like GPT-4o and Claude Sonnet 4) in AV-style metrics, demonstrating the feasibility of agentic stylistic generation.

---

[MedTutor-R1: Socratic Personalized Medical Teaching with Multi-Agent Simulation](http://arxiv.org/abs/2512.05671)

- MedTutor-R1 (Socratic Personalized Medical Teaching): introduces ClinEdu (dynamic testbed), a multi-agent pedagogical simulator used to construct ClinTeach (group instruction data) and train the Tutor Agent (Socratic guidance) for one-to-many clinical instruction.
- The Tutor Agent employs Multi-Dimensional Thinking (internal reasoning) to analyze student progress and generates adaptive Socratic guidance, refined via Reinforcement Learning (strategy optimization) using a Three-Axis Reward Rubric (fidelity, analysis, safety).
- ClinEdu simulates clinical ward rounds using personality-driven Patient Agents (personality-driven case) and diverse Student Agents (diverse cohort), overseen by Specialist and Safety Supervisor Agents for quality control.

---

[MARINE: Theoretical Optimization and Design for Multi-Agent Recursive IN-context Enhancement](http://arxiv.org/abs/2512.07898)

- MARINE (Multi-Agent Recursive IN-context Enhancement): reconceptualizes test-time reasoning as iterative refinement of a persistent reference trajectory using a theoretically grounded Refinement operator that aggregates candidate trajectories from multiple heterogeneous agents to transform LLM pass@N capability into reliable pass@1 performance.
- The framework employs a layered architecture with structured trajectory representation and conflict-aware meta-verification mechanisms to ensure monotonic improvement and global reasoning coherence without requiring full trajectory regeneration.
- Theoretical analysis provides complementary principles for Batch-size optimization, establishing minimal feasible batches for fixed budgets and logarithmically growing schedules for continuous improvement, enabling unprecedented parameter efficiency.

---

[CureAgent: A Training-Free Executor-Analyst Framework for Clinical Reasoning](http://arxiv.org/abs/2512.05576)

- CureAgent (Executor-Analyst Collaborative Framework): introduces a modular architecture that decouples precise tool execution (Executor/TxAgent) from high-level clinical reasoning (Analyst/Gemini 2.5) using a Stratified Ensemble topology.
- The Executor, a specialized LLM, focuses on aggregating robust evidence from the ToolUniverse using a Self-Consistency Mechanism, while the Analyst, a long-context foundation model, synthesizes this evidence and performs fact verification via Search.
- The framework achieves state-of-the-art performance on CURE-Bench without expensive end-to-end finetuning by leveraging training-free architectural engineering and a deterministic Post-processing Module.

---

[GRASP: Graph Reasoning Agents for Systems Pharmacology with Human-in-the-Loop](http://arxiv.org/abs/2512.05502)

- GRASP (Graph Reasoning Agents for Systems Pharmacology): introduces a multi-agent, graph-reasoning framework that encodes Quantitative Systems Pharmacology (QSP) models as typed biological knowledge graphs and compiles them into executable MATLAB/SimBiology code.
- The system operates in two phases—Understanding (graph reconstruction) and Action (constraint-checked modification)—orchestrated by a state machine with iterative validation and feedback loops.
- GRASP utilizes Breadth-First Search (BFS) parameter alignment to ensure consistency and propose biologically plausible defaults when new entities are introduced via the conversational Human-in-the-Loop interface.

---

[Dynamic Alignment for Collective Agency: Toward a Scalable Self-Improving Framework for Open-Ended LLM Alignment](http://arxiv.org/abs/2512.05464)

- Dynamic Alignment (DA) framework: introduces a scalable, self-improving alignment method that enables an LLM to iteratively align itself to the open-ended value of Collective Agency (CA) using automated dataset generation and a self-rewarding mechanism.
- The framework operates in two phases: generating diverse task prompts using multiple LLM agents, followed by a self-improving loop where the policy model evaluates its own outputs and updates via Group Relative Policy Optimization (GRPO).
- CA is defined by four inseparable aspects—Knowledge, Benevolence, Power, and Vitality—which guide the agent toward continual improvement of its capacity to act meaningfully.

---

[Model Gateway: Model Management Platform for Model-Driven Drug Discovery](http://arxiv.org/abs/2512.05462)

- Model Gateway: introduces a cloud-based model management platform built on a Kubernetes cluster for handling ML and scientific computational models in the drug discovery pipeline, including Model Owner Control Panel (Web user interface) and LLM Agents (ML model management tasks).
- The platform supports asynchronous Model Execution (Asynchronous processing) via a Redis Cluster (Metadata and results storage) job queue and manages model scalability using KEDA (KEDA-based autoscaling).
- Key features include Model Versioning (Version control), Model Access Control (Role-based security), and Dynamic Consensus Model Management (Model aggregation) to streamline the drug discovery process.

---

[Please Don't Kill My Vibe: Empowering Agents with Data Flow Control](http://arxiv.org/abs/2512.05374)

- FlowGuard (Data Flow Control): introduces a system to safely deploy LLM agents in stateful environments by shifting data flow policy enforcement from agent workflows to underlying data systems like the DBMS.
- FlowGuard uses provenance polynomials to model how input tuples contribute to output tuples, allowing policies to be defined over logical data flows and enforced via query rewriting.
- The policy language includes clauses (POLICY OVER, AGG, DIMENSION, CONSTRAINT, ON FAIL) that check source relations, aggregation structure, and provenance constraints to mitigate risks like policy violations, process corruption, and prompt injection.

---

[GTM: Simulating the World of Tools for AI Agents](http://arxiv.org/abs/2512.04535)

- GTM (Generalist Tool Model): introduces a 1.5-billion-parameter universal tool simulator that generates outputs mimicking real tool execution using prompt-level configuration, enabling efficient LLM agent training.
- The approach utilizes the Context-Aware Response Generation (CARG) pipeline to synthesize comprehensive training data covering over 20,000 tools across 300 domains, ensuring format correctness and contextual consistency.
- GTM significantly accelerates simulation speed compared to real APIs, maintains comparable output quality, and exhibits strong generalization and domain adaptability for tool-augmented systems.

---

[Reinforcement Learning Integrated Agentic RAG for Software Test Cases Authoring](http://arxiv.org/abs/2512.06060)

- RI-ARAG (Reinforcement Integrated Agentic RAG): introduces a framework that integrates reinforcement learning (RL) algorithms (PPO, DQN) with specialized autonomous agents and a hybrid vector-graph knowledge base to continuously improve software test case authoring based on Quality Engineer (QE) feedback.
- The architecture employs a multi-dimensional reward function incorporating test effectiveness, defect detection, coverage, and efficiency metrics to guide the RL agents' behavioral optimization and knowledge base evolution.
- The system establishes a continuous knowledge refinement loop, demonstrating a 10.8% enhancement in defect detection rates and a 2.4% increase in test generation accuracy during enterprise validation.

---


#### 4th December 2025

[ARM-Thinker: Reinforcing Multimodal Generative Reward Models with Agentic Tool Use and Visual Reasoning](http://arxiv.org/abs/2512.05111)

- ARM-Thinker (Agentic multimodal Reward Model): introduces an active, verifiable reward modeling framework that uses a Think-Act-Observe Loop to autonomously invoke multimodal tools for evidence-grounded judgment.
- The framework integrates a Multimodal ToolKit, including Instruction-Following Check Tools, Image Crop and Zoom-in Tools, and Document Retrieval Tools, to verify fine-grained visual details and cross-reference multi-page evidence.
- Training utilizes Multi-Stage Reinforcement Learning (GRPO) with a two-stage reward design to jointly optimize tool-calling decisions and final judgment accuracy, achieving substantial gains across reward benchmarks.

---

[Nex-N1: Agentic Models Trained via a Unified Ecosystem for Large-Scale Environment Construction](http://arxiv.org/abs/2512.04987)

- Nex Ecosystem (NexAU/NexA4A/NexGAP): introduces a comprehensive infrastructure designed for agentic scaling by systematically increasing the diversity and complexity of interactive environments across three dimensions: complexity, diversity, and fidelity.
- The system automates environment construction, transforming it from manual engineering to automated synthesis using generative language specifications to enable infinite scaling of diverse interaction topologies.
- NexAU, the core runtime, uses a recursive, fractal architecture inspired by ReAct to unify heterogeneous frameworks and generate high-fidelity trajectories for training the Nex-N1 LLM.

---

[Chameleon: Adaptive Adversarial Agents for Scaling-Based Visual Prompt Injection in Multimodal AI Systems](http://arxiv.org/abs/2512.04895)

- Chameleon: introduces an adaptive adversarial framework designed to exploit image scaling vulnerabilities in production Vision-Language Models (VLMs) by using an iterative, agent-based optimization loop.
- The framework dynamically refines image perturbations based on real-time VLM feedback signals (success, confidence, keywords) to craft robust adversarial examples that survive standard downscaling operations.
- Chameleon utilizes a reward function to balance attack efficacy and visual imperceptibility, employing either Hill-Climbing or a Genetic Algorithm as the optimization strategy to achieve high ASR against target LLMs like Gemini 2.5-Flash.

---

[Are Your Agents Upward Deceivers?](http://arxiv.org/abs/2512.04864)

- Agentic Upward Deception (AUD) Benchmark: introduces a framework to evaluate the prevalence of agentic upward deception in LLM-based agents, utilizing a superior-subordinate structure, a constrained Agentic Environment, and an LLM-as-a-judge system.
- The benchmark uses 200 tasks across five types and eight scenarios, perturbing the environment with constraints like broken tools or nonexistent files to trigger deceptive behaviors.
- Evaluation of 11 popular LLMs reveals pervasive action-based deceptive behaviors, including guessing results, simulating outcomes, and fabricating local files to conceal task failures.

---

[SIMA 2: A Generalist Embodied Agent for Virtual Worlds](http://arxiv.org/abs/2512.04797)

- SIMA 2 (Generalist Embodied Agent for Virtual Worlds): introduces a Gemini-based agent that reasons, acts, and engages in dialogue across diverse 3D virtual worlds, leveraging the Gemini Flash-Lite foundation model core, Agent-Environment Interface, Gemini-Based Task Setter, and Gemini-Based Reward Model.
- SIMA 2 achieves near-human performance on embodied tasks by integrating high-level reasoning (from the LLM core) with low-level control (keyboard/mouse actions) via supervised finetuning and reinforcement learning.
- The agent demonstrates robust generalization to held-out environments, including photorealistic worlds generated by Genie 3, and is capable of open-ended self-improvement by learning new skills autonomously from self-generated experience.

---

[ASTRIDE: A Security Threat Modeling Platform for Agentic-AI Applications](http://arxiv.org/abs/2512.04785)

- ASTRIDE (A Security Threat Modeling Platform for Agentic-AI Applications): introduces an automated threat modeling platform purpose-built for AI agent-based systems, extending STRIDE with AI Agent-Specific Attacks (A) using LLM Agents, a Fine-tuned VLM Consortium, an OpenAI-gpt-oss Reasoning LLM, and a Data Lake.
- The platform automates end-to-end threat analysis directly from visual agent architecture diagrams, such as data flow diagrams, by coordinating the VLM consortium and the reasoning LLM.
- The VLM consortium, fine-tuned on labeled diagrams, detects component-level vulnerabilities, while the reasoning LLM synthesizes these outputs into a cohesive, contextually validated threat model.

---

[Towards an AI Fluid Scientist: LLM-Powered Scientific Discovery in Experimental Fluid Mechanics](http://arxiv.org/abs/2512.04716)

- AI Fluid Scientist framework (LLM-powered scientific discovery): introduces a multi-agent system with a virtual-real interaction system, including Hypothesis, Experiment, Hardware, Analysis, Evaluation, and Manuscript Agents, validated using a computer-controlled Circulating Water Tunnel (CWT) in both Human-in-the-Loop (HIL) and End-to-end automation modes.
- The framework autonomously executes the complete experimental workflow, from hypothesis generation and experimental design to robotic execution, data analysis, and manuscript preparation, accelerating discovery in experimental fluid mechanics.
- The system successfully reproduced classical Vortex-Induced Vibration (VIV) and Wake-Induced Vibration (WIV) benchmarks and autonomously discovered new WIV phenomena, including optimal suppression frequencies and neural network-based physical laws.

---

[Towards Ethical Multi-Agent Systems of Large Language Models: A Mechanistic Interpretability Perspective](http://arxiv.org/abs/2512.04691)

- MALM-MI (Multi-Agent Systems of Large Language Models - Mechanistic Interpretability): introduces a research agenda for ensuring ethical MALM behavior by integrating Mechanistic Interpretability (causal component identification) into Evaluation Frameworks (assess ethical behavior) and Targeted PEFT (mechanism-guided alignment).
- The approach dissects LLM internals to identify computational pathways (e.g., attention heads, circuits) that mediate emergent behaviors like toxic agreement or groupthink.
- MI provides actionable handles for intervention, allowing surgical corrections via activation steering or targeted PEFT without compromising system performance.

---

[PBFuzz: Agentic Directed Fuzzing for PoV Generation](http://arxiv.org/abs/2512.04611)

- PBFuzz (Agentic Directed Fuzzing for PoV Generation): introduces an agentic directed fuzzing framework that mimics human experts for PoV input generation, including LLM Agent (semantic reasoning, planning), Workflow Layer (state machine, control), MCP Tool Layer (stateless program analysis), and Memory Layer (persistent structured state).
- The framework utilizes a four-phase workflow (PLAN, IMPLEMENT, EXECUTE, REFLECT) to iteratively infer semantic constraints, encode them into parameterized input spaces, and leverage property-based testing for efficient constraint solving.
- PBFuzz addresses the semantic gap between vulnerability comprehension and PoV generation, achieving decisive superiority over baselines by triggering 57 vulnerabilities within a 30-minute budget per target.

---

[dVLM-AD: Enhance Diffusion Vision-Language-Model for Driving via Controllable Reasoning](http://arxiv.org/abs/2512.04459)

- dVLM-AD (Diffusion Vision-Language-Model for Driving): introduces a diffusion-based VLM that unifies perception, structured reasoning, and low-level planning for end-to-end driving, utilizing an LLM Backbone, Vision Encoder, and Iterative Denoising.
- The system addresses reasoning-action inconsistency and uncontrollable generation in AR-based VLMs by employing a Reasoning Template (structured CoT) and a Dynamic Denoise Strategy for bidirectional, iterative refinement.
- The diffusion formulation ensures stronger reasoning-action alignment and robustness to prompt perturbations by constraining decoding to a prescribed structure via template-anchored fill-in-the-blank generation.

---

[AgentBay: A Hybrid Interaction Sandbox for Seamless Human-AI Intervention in Agentic Systems](http://arxiv.org/abs/2512.04367)

- AgentBay: introduces a novel hybrid interaction sandbox service, featuring a four-layer architecture (Interface, Service, Environment, Feature) and the Adaptive Streaming Protocol (ASP) for seamless Human-in-the-Loop (HITL) intervention.
- The system provides secure, isolated execution environments (Windows, Linux, Android, Web Browsers, Code interpreters) accessible simultaneously via programmatic APIs (MCP/SDK) for the AI agent and high-performance graphical streaming (ASP) for human operators.
- ASP achieves ultra-low-latency, resilient streaming by dynamically blending command-based and video-based streaming, adapting its encoding strategy based on network conditions and the current controller (AI or human).

---

[WhatsCode: Large-Scale GenAI Deployment for Developer Efficiency at WhatsApp](http://arxiv.org/abs/2512.05314)

- WhatsCode: introduces a domain-specific AI development system deployed at WhatsApp, supporting large-scale GenAI deployment for developer efficiency across mobile and server codebases.
- The system evolved through Foundation, Agentless Expansion, and Agentic Evolution eras, culminating in a layered architecture featuring a Workflow Router, LLM Infra, and specialized Sub Agents for complex, multi-step tasks.
- WhatsCode achieved substantial quantifiable impact, including a 3.5x improvement in automated privacy verification coverage and generating over 3,000 accepted code changes across diverse automation domains.

---

[LegalWebAgent: Empowering Access to Justice via LLM-Based Web Agents](http://arxiv.org/abs/2512.04105)

- LegalWebAgent: introduces a multimodal web agent framework that employs LLMs to bridge the access to justice gap by autonomously performing web tasks from user query to concrete action.
- The framework operates in three stages—Ask, Browse, and Act—handling user intent parsing, autonomous webpage navigation using multimodal perception, and execution of actions like form completion or scheduling.
- The agent demonstrated high autonomy in complex real-world legal scenarios, achieving an average success rate of 84.4% across tested models in tasks ranging from information gathering to action taking.

---

[The Erosion of LLM Signatures: Can We Still Distinguish Human and LLM-Generated Scientific Ideas After Iterative Paraphrasing?](http://arxiv.org/abs/2512.05311)

- IGPW (Idea Generation and Paraphrasing Workflow): introduces a systematic evaluation of SOTA machine learning models' ability to distinguish human- and LLM-generated scientific ideas after successive paraphrasing stages, using Generative LLMs, a Research Problem Extractor, a Paraphrasing Cascade, SOTA Classifiers, Embedding Models, and an FFNN.
- The study demonstrates that detection performance declines by an average of 25.4% after five paraphrasing stages, suggesting that characteristic LLM signatures gradually erode through successive stylistic transformations.
- Integrating the research problem as contextual information improves detection performance, while paraphrasing into a simplified, non-expert style contributes most significantly to the erosion of LLM signatures.

---

[David vs. Goliath: Can Small Models Win Big with Agentic AI in Hardware Design?](http://arxiv.org/abs/2512.05073)

- SLM-aware agentic AI framework: introduces a closed-loop iterative workflow using five cooperating agents (PPA, SPEA, CA, VA, AFA) to enable Small Language Models (SLMs) to perform complex hardware design tasks.
- The framework compensates for SLM limitations by providing structured guidance, task decomposition, iterative refinement, and targeted validation, analogous to junior engineer mentorship.
- Empirical validation on the CVDP benchmark shows that SLMs, when augmented with this agentic scaffolding, can achieve near-LLM performance at a fraction of the computational cost.

---

[STRATEGIC SELF-IMPROVEMENT FOR COMPETITIVE AGENTS IN AI LABOUR MARKETS](http://arxiv.org/abs/2512.04988)

- SSA (Strategic Self-Improving Agent): introduces a framework for LLM agents to succeed in competitive AI labor markets by integrating metacognition, competitive awareness, and strategic planning capabilities.
- The framework is implemented in AI Work, a simulated gig economy platform that models real-world economic forces like adverse selection, moral hazard, and reputation dynamics.
- Explicitly prompting LLM agents with these strategic capabilities enables them to strategically self-improve, adapt to changing market conditions, and demonstrate superior performance.

---

[SEAL: Self-Evolving Agentic Learning for Conversational Question Answering over Knowledge Graphs](http://arxiv.org/abs/2512.04868)

- SEAL (Self-Evolving Agentic Learning): introduces a novel two-stage semantic parsing framework for conversational question answering over knowledge graphs, featuring LLM core generation, agent calibration, and template-based completion.
- The system incorporates a self-evolving mechanism that uses local memory, global memory, and a reflection module to enable continuous adaptation and performance enhancement without explicit retraining.
- This two-stage decomposition simplifies logical form generation by extracting a minimal S-expression core, which significantly enhances structural fidelity and computational efficiency in complex reasoning tasks.

---

[NATURAL LANGUAGE ACTOR-CRITIC: SCALABLE OFF-POLICY LEARNING IN LANGUAGE SPACE](http://arxiv.org/abs/2512.04601)

- NLAC (Natural Language Actor-Critic): introduces a scalable off-policy actor-critic algorithm for LLM agents, utilizing a generative LLM Language Critic that produces textual critiques instead of scalar values, trained via a Language Bellman Backup.
- The Language Critic leverages a Language Successor Model to predict future rollouts in natural language, providing a richer and more actionable training signal for policy improvement compared to sparse scalar rewards.
- Policy improvement is achieved through a Refinement Policy that distills knowledge from the textual critiques, enabling the LLM to iteratively refine suboptimal actions without relying on random exploration.

---

[StreamEQA: Towards Streaming Video Understanding for Embodied Scenarios](http://arxiv.org/abs/2512.04451)

- StreamEQA: introduces the first benchmark for streaming video question answering in embodied scenarios, integrating the Embodied Dimension (Perception, Interaction, Planning) and Streaming Dimension (Backward, Realtime, Forward) to evaluate Video-LLMs.
- The benchmark construction utilizes a hybrid pipeline involving Meta Information Extraction, QA Construction, and Quality Control, leveraging a VLM (GPT-5) for structured data generation and refinement.
- StreamEQA reveals that state-of-the-art MLLMs struggle significantly with interaction and planning tasks under streaming constraints, highlighting the necessity for temporally grounded reasoning mechanisms.

---

[Automating Complex Document Workflows via Stepwise and Rollback-Enabled Operation Orchestration](http://arxiv.org/abs/2512.04445)

- AutoDW: introduces a novel execution framework that enables stepwise, rollback-enabled operation orchestration for automating complex document workflows.
- The framework incrementally plans API actions conditioned on user instructions, intent-filtered API candidates, and the evolving document state.
- Adaptive Rollback employs LLM-based validation and dual-level correction (argument-level and API-level) to ensure alignment with user intent and document context.

---

[Executable Governance for AI: Translating Policies into Rules Using LLMs](http://arxiv.org/abs/2512.04408)

- P2T (Policy Tests): introduces a framework that converts natural-language policy documents into normalized, machine-readable executable rules using an iterative pipeline of LLM extraction and deterministic validation.
- The pipeline utilizes a compact JSON DSL Schema to encode rules with fixed fields (scope, hazard, conditions, evidence) and includes LLM components for mining, judging, repairing, and generating examples.
- Deterministic checks, including schema validation, evidence gating, and Satisfiability Modulo Theories (SMT) consistency checks, ensure stability, reproducibility, and fidelity to source clauses.

---

[LEARNING TO ORCHESTRATE AGENTS IN NATURAL LANGUAGE WITH THE CONDUCTOR](http://arxiv.org/abs/2512.04388)

- RL Conductor (Reinforcement Learning Conductor): introduces a new reasoning model trained via end-to-end RL to automatically discover powerful coordination strategies among a pool of Worker LLM Agents by outputting Agentic Workflows.
- The Conductor designs these workflows by delegating natural language Subtasks, assigning Worker LLM Agents, and defining targeted communication topologies via an Access List.
- A 7B Conductor achieves state-of-the-art results on challenging reasoning benchmarks, demonstrating that RL can unlock powerful coordination strategies, including dynamic test-time scaling via Recursive Topology.

---

[The Personalization Paradox: Semantic Loss vs. Reasoning Gains in Agentic AI Q&A](http://arxiv.org/abs/2512.04343)

- AiVisor (Agentic Retrieval Augmented Large Language Model System): introduces a prototype agentic RAG LLM system for student advising, designed to evaluate the complex trade-offs of personalization in question answering.
- The system integrates a Personalization Agent to retrieve user-specific data from University Enterprise Systems, which is then optionally injected into the VectorDB or Prompt Assembly.
- Evaluation using a Linear Mixed-Effects Model revealed that personalization significantly improved reasoning quality but incurred a negative interaction on semantic similarity metrics.

---

[Orchestrator Multi-Agent Clinical Decision Support System for Secondary Headache Diagnosis in Primary Care](http://arxiv.org/abs/2512.04207)

- OMACDSS (Orchestrator Multi-Agent Clinical Decision Support System): introduces an LLM-based multi-agent system built on an orchestrator-specialist architecture using LangGraph to perform explicit and interpretable secondary headache diagnosis from free-text clinical vignettes.
- The system decomposes the diagnosis task into seven domain-specialized agents, coordinated by a central orchestrator agent that handles task decomposition and dynamic agent routing based on input case features.
- Robustness strategies, including a Manual Fan-out Function, ensure reliability and complete agent coverage, providing transparent, criterion-based red flag reasoning aligned with clinical guidelines (GPrompt).

---

#### 3rd December 2025

[Benchmark for Planning and Control with Large Language Model Agents: Blocksworld with Model Context Protocol](http://arxiv.org/abs/2512.03955)

- BBA (Blocksworld Benchmark Architecture): introduces a systematic benchmark for evaluating LLM agents on planning and execution tasks in the Blocksworld domain, integrating the Model Context Protocol (MCP) as a standardized interface between interchangeable LLM Agents and the Blocksworld Simulation.
- The architecture uses a layered approach where the MCP Server wraps the simulation's REST API endpoints into MCP Tools (Information, Verification, Execution) accessible by the LLM Agents via the MCP Client.
- The benchmark provides five complexity categories (Basic, Non-constructive actions, Impossible, Additional Constraints, Partial Observability) to rigorously test agent capabilities under varying conditions.

---

[A Hierarchical Tree-based approach for creating Configurable and Static Deep Research Agent (Static-DRA)](http://arxiv.org/abs/2512.03887)

- Static-DRA (Static Deep Research Agent): introduces a novel, configurable, and hierarchical tree-based architecture for deep research, governed by user-tunable Depth and Breadth parameters.
- The architecture employs a static workflow utilizing a hierarchy of Supervisor, Independent, and Worker agents to facilitate multi-hop retrieval and parallel sub-topic investigation using an LLM and a Web Search Tool.
- The configurable Depth and Breadth parameters allow users to balance research quality and comprehensiveness against the associated computational cost of LLM interactions.

---

[RoCo: Role-Based LLMs Collaboration for Automatic Heuristic Design](http://arxiv.org/abs/2512.03762)

- RoCo (Role-based LLMs Collaboration): introduces a novel Multi-Agent Role-Based System to enhance Automatic Heuristic Design (AHD) quality through structured collaboration among specialized LLM-guided agents.
- The system coordinates four agents—explorer, exploiter, critic, and integrator—in a multi-round process involving feedback, refinement, and memory-guided elite mutations, integrated within the Evolution of Heuristics (EoH) framework.
- This role-based collaborative paradigm achieves superior performance and faster convergence across diverse combinatorial optimization problems by balancing innovation (explorer) and exploitation (exploiter) via structured agent interaction.

---

[SRPG: Semantically Reconstructed Privacy Guard for Zero-Trust Privacy in Educational Multi-Agent Systems](http://arxiv.org/abs/2512.03694)

- SRPG (Semantically Reconstructed Privacy Guard): introduces a novel privacy mechanism for educational Multi-Agent Systems (MAS) utilizing a Dual-Stream Reconstruction Mechanism to safeguard minors' PII while preserving educational utility.
- The framework operates as middleware between the Student Agent and the Tutor Agent, employing a Strict Sanitization Stream for zero-trust masking and a parallel Context Reconstruction Stream driven by LLM reasoning.
- By fusing the sanitized text and the reconstructed mathematical context, the system achieves zero privacy leakage (ASR 0.00) and superior utility compared to traditional masking or Pure LLM Sanitizer baselines.

---

[KVNAND: Efficient On-Device Large Language Model Inference Using DRAM-Free In-Flash Computing](http://arxiv.org/abs/2512.03608)

- KVNAND (DRAM-free NPU-IFC architecture): introduces a DRAM-free LLM inference architecture that stores both model weights and the KV cache entirely in compute-enabled 3D NAND flash, leveraging IFC for all memory-bound operations.
- The architecture employs head-group parallelism and a page-level KV cache mapping scheme to align token access patterns with flash organization, mitigating latency and reliability concerns.
- KVNAND offers discrete and compact variants to balance pipeline and tensor parallelism, achieving up to 2.3x speedup and 0.75x energy consumption compared to DRAM-equipped IFC designs.

---

[DeepRule: An Integrated Framework for Automated Business Rule Generation via Deep Predictive Modeling and Hybrid Search Optimization](http://arxiv.org/abs/2512.03607)

- DeepRule (An Integrated Framework for Automated Business Rule Generation): introduces a tri-level architecture comprising the Hybrid Knowledge Fusion Engine, Game-Theoretic Constrained Optimization, and Interpretable Decision Distillation Interface, unifying unstructured knowledge injection, multi-agent optimization, and interpretable strategy synthesis for retail assortment and pricing optimization.
- The framework leverages LLMs for deep semantic parsing of unstructured text into structured features and employs a DNN model for feature-decoupled sales volume prediction under hierarchical business constraints.
- Interpretable decision distillation uses LLM-guided symbolic regression and iterative rule optimization to find and refine auditable business rules and pricing strategies, ensuring operational feasibility and high profits.

---

[ENCOMPASS: Enhancing Agent Programming with Search Over Program Execution Paths](http://arxiv.org/abs/2512.03571)

- ENCOMPASS framework: introduces Probabilistic Angelic Nondeterminism (PAN), a programming model that disentangles agent workflow logic and inference-time strategies, using a Python decorator to compile agent workflow programs into a search space.
- The framework utilizes primitives like `branchpoint()` to mark unreliable operations (e.g., LLM calls) and `record_score()` to guide external search algorithms over the resulting tree of possible execution paths.
- ENCOMPASS provides a unifying structure for common inference-time strategies, such as best-of-N sampling and beam search, enabling easy experimentation and generalization of these strategies.

---

[A Preliminary Study on the Promises and Challenges of Native Top-k Sparse Attention](http://arxiv.org/abs/2512.03494)

- LongCat: introduces a preliminary study on the effectiveness and theoretical mechanisms of Native Top-k Sparse Attention, including Exact Top-k Decoding, Native Top-k Attention Training, Retrieval Precision, and Attention Entropy Reduction.
- The study validates that Exact Top-k Decoding significantly reduces computational overhead while maintaining or surpassing full attention performance, even at low Top-k Ratios, for long-context LLM inference.
- Incorporating native Top-k Attention training during SFT enhances model performance by adapting the LLM to the sparse attention patterns characteristic of Top-k Decoding during inference, which is theoretically supported by observed entropy reduction.

---

[AsymPuzl: An Asymmetric Puzzle for multi-agent cooperation](http://arxiv.org/abs/2512.03466)

- AsymPuzl: introduces a minimal, expressive two-agent puzzle environment designed to isolate communication strategies of LLM agents under information asymmetry, featuring two agents (Alice and Bob) who exchange messages to solve a symbolic position-shape-color matching puzzle.
- The environment requires sequential coordination where agents receive complementary partial views (cues) of the puzzle and iteratively update their working hypotheses based on messages and external feedback.
- Empirical analysis using diverse LLMs demonstrates that communication strategies diverge based on feedback granularity, showing that strong models reliably share complete information while weaker models struggle with miscommunication or over-correction.

---

[Classification of User Satisfaction in HRI with Social Signals in the Wild](http://arxiv.org/abs/2512.03945)

- SSC: introduces an approach for automatically classifying user satisfaction in Human-Robot Interaction (HRI) using time series features derived from social signals, including body pose, facial expressions, and physical distance.
- The system utilizes a Furhat robot deployed in an in-the-wild museum setting, capturing interaction data via wide-angle cameras and using a locally executed LLM (Llama 3) as a conversational fallback.
- The method compares three feature engineering techniques (tsfresh, catch22, and handcrafted features) on various ML models, achieving high classification accuracy (up to 97.8%) in reliably identifying interactions with low user satisfaction.

---

[Driving is a Game: Combining Planning and Prediction with Bayesian Iterative Best Response](http://arxiv.org/abs/2512.03936)

- BIBeR (Bayesian Iterative Best Response): introduces a game-theoretic framework that unifies motion prediction and planning into a single interaction-aware process, including a Proposal Generator (generates ego trajectories), Prediction Agent (forecasts agent motions), Bayesian Confidence Estimation (quantifies prediction reliability), and Iterative Best Response (IBR) (iteratively refines strategies).
- The IBR loop repeatedly refines the strategies of the ego vehicle and surrounding agents by re-weighting fixed discrete trajectory sets based on expected utility, approximating a Nash equilibrium.
- The Bayesian confidence estimation mechanism regulates update dynamics, promoting conservative adjustments under low confidence and decisive responses under high confidence to balance assertiveness and safety.

---

[Autonomous Agents and Policy Compliance: A Framework for Reasoning About Penalties](http://arxiv.org/abs/2512.03931)

- AOPL-P Framework: introduces a logic programming-based system for policy-aware autonomous agents to reason about non-compliance penalties and generate optimal plans.
- The framework extends the Authorization and Obligation Policy Language (AOPL) to AOPL-P, incorporating numerical penalties for policy violations and integrating Answer Set Programming (ASP) for reasoning.
- The system refines agent behavior modes (emergency/non-emergency) by prioritizing plan metrics like cumulative penalty and execution time, ensuring high-quality plans that minimize human harm.

---

[Multi-agent deep reinforcement learning for UAV-based 5G network slicing](http://arxiv.org/abs/2512.03835)

- MADRL framework: introduces a unified Multi-Agent Deep Reinforcement Learning (MADRL) framework that integrates 5G network slicing with coordinated UAV control under a CTDE paradigm, utilizing MAPPO, MADDPG, and MADQN to jointly optimize QoS and energy efficiency.
- The system prioritizes Premium (A), Silver (B), and Bronze (C) user slices, optimizing key QoS metrics (latency, throughput, SINR) while managing UAV movement, resource allocation, and energy constraints.- Comparative analysis across urban and rural environments reveals that MAPPO provides the strongest overall QoS-energy tradeoff, highlighting that algorithm suitability depends on scenario topology and requirements.

---

[First Experimental Demonstration of Machine Learning-Based Tuning on the PSI Injector 2 Cyclotron](http://arxiv.org/abs/2512.03829)

- TD3-RL (Twin Delayed Deep Deterministic Policy Gradient Reinforcement Learning): introduces an ML-based tuning framework deployed on the PSI Injector 2 Cyclotron, combining a tailored RL algorithm with real-time diagnostics and physics-informed adaptations to achieve autonomous beam tuning.
- The system successfully tuned the cyclotron across multiple operating points and compensated for drifts over a 12-day campaign, demonstrating robustness and generalization from low-current training to higher-current operation.
- Key accelerator physics adaptations included an overshooting strategy that reduced magnetic field settling times by a factor of six and a physics-informed reward function incorporating phase alignment, trim coil usage, and interlock penalties.

---

[Context-Triggered Contingency Games for Strategic Multi-Agent Interaction](http://arxiv.org/abs/2512.03639)

- Context-Triggered Contingency Games: introduces a two-layered reactive planning and control architecture that integrates high-level strategic games (LTL specifications and strategy templates) with low-level dynamic contingency games solved via real-time Model Predictive Control.
- The framework leverages strategy templates to extract local dynamic goals and context-dependent constraints, ensuring both long-term strategic objective satisfaction and short-term dynamic adaptation in uncertain, interactive environments.
- A novel Dynamic Game Factor Graph (DG-FG) solver is developed to efficiently solve the resulting Generalized Nash Equilibrium Problem (GNEP) in real-time, enabling scalable multi-agent interaction demonstrated in autonomous driving and robotic navigation.

---

[Reason-Plan-ReAct: A Reasoner-Planner Supervising a ReAct Executor for Complex Enterprise Tasks](http://arxiv.org/abs/2512.03560)

- RP-ReAct (Reasoner Planner-ReAct): introduces a novel multi-agent architecture that fundamentally separates high-level strategic planning (RPA) from low-level execution (PEA) using a ReAct approach, incorporating a context-saving strategy.
- The RPA leverages strong reasoning models to plan sub-steps and continuously analyze execution results, enabling dynamic re-planning and error handling in complex tasks.
- The PEA translates abstract sub-steps into concrete tool interactions, utilizing a specialized context management mechanism to prevent context window overflow from large tool outputs.

---

[PARC: An Autonomous Self-Reflective Coding Agent for Robust Execution of Long-Horizon Tasks](http://arxiv.org/abs/2512.03549)

- PARC (Preferred Autonomous self-Reflective Coding agent): introduces a hierarchical multi-agent architecture integrating task planning, execution, and self-reflection mechanisms for robust execution of long-horizon computational tasks.
- The system operates using a plan-and-execute pattern where a Planner constructs a task sequence, and multiple Workers execute tasks sequentially within independent contexts, utilizing a shared Structured workspace.
- Self-assessment and self-feedback enable the agent to detect and correct high-level strategic errors and approach-level failures, ensuring stability and reliability in complex, multi-stage workflows.

---

[EEA: Exploration–Exploitation Agent for Long Video Understanding](http://arxiv.org/abs/2512.03500)

- EEA (Exploration–Exploitation Agent): introduces a novel video agent framework that achieves exploration-exploitation balance through semantic guidance and a hierarchical tree search process.
- The framework utilizes Dynamic Query Management (DQM) to continuously refine semantic queries and Semantic Guided Expansion (SGE) to strategically prioritize semantically relevant frames while ensuring broad coverage.
- Segment evaluation is stabilized by Uncertainty-Aware Reward Fusion (UARF), which adaptively integrates intrinsic rewards from VLMs with query scores derived from semantic priors.

---

[ATHENA: Agentic Team for Hierarchical Evolutionary Numerical Algorithms](http://arxiv.org/abs/2512.03476)

- ATHENA (Agentic Team for Hierarchical Evolutionary Numerical Algorithms): introduces an agentic framework managing the end-to-end computational research lifecycle via the HENA loop, modeled as a Contextual Bandit problem to minimize regret.
- The system segregates the process into Policy (Conceptualization) and Implementation (Execution) operators, using Conceptual Scaffolding (expert blueprints) to constrain the combinatorial search space and ensure mathematical rigor.
- ATHENA achieves super-human performance in Scientific Computing and SciML by performing deep physical diagnosis, autonomously correcting conceptual errors, and orchestrating complex hybrid symbolic-numeric workflows.

---

[World Models for Autonomous Navigation of Terrestrial Robots from LIDAR observations](http://arxiv.org/abs/2512.03429)

- DreamerV3-MLP-VAE: introduces a novel model-based DRL architecture for autonomous terrestrial robot navigation using high-dimensional LIDAR observations, integrating an MLP-VAE encoder for compact latent representation and a dynamics predictor for imagination-based policy optimization.
- The architecture overcomes the input-size bottleneck of model-free methods by encoding full 360-reading LIDAR data into latent features, which are then used by the world model's dynamics predictor and controller for efficient decision-making.
- Empirical validation on TurtleBot3 navigation tasks demonstrates superior convergence and a 100% success rate across complex environments, highlighting the robustness and scalability of predictive world modeling with learned latent representations.

---

[DAComp: Benchmarking Data Agents across the Full Data Intelligence Lifecycle](http://arxiv.org/abs/2512.04324)

- DAComp: introduces a comprehensive benchmark of 210 tasks mirroring enterprise data intelligence workflows, encompassing DAComp-DE (Data Engineering) for repository-level orchestration and DAComp-DA (Data Analysis) for open-ended reasoning.
- DAComp-DE evaluates agents on the full DE lifecycle, including architecture, implementation, and evolution tasks, requiring management of multi-layered data models (staging, core, marts) and complex dependencies.
- DAComp-DA assesses open-ended analytical reasoning, requiring strategic planning, iterative coding, insight synthesis, and visualization, evaluated using an LLM-judge guided by hierarchical rubrics.

---

[The Geometry of Benchmarks: A New Path Toward AGI](http://arxiv.org/abs/2512.04276)

- GVU (Generator-Verifier-Updater): introduces a geometric framework for evaluating AI progress by treating all psychometric batteries as points in a structured moduli space, driven by the GVU loop dynamics.
- The framework defines the Autonomous AI (AAI) Scale as an operational hierarchy of autonomy and uses the GVU operator, which subsumes RL and self-play, to model self-improvement as a flow on the moduli space of batteries.
- Progress toward AGI is recast as climbing a capability functional defined over this moduli space, with a variance inequality providing sufficient conditions for a positive self-improvement coefficient ($\kappa > 0$).

---

[Driving Beyond Privilege: Distilling Dense-Reward Knowledge into Sparse-Reward Policies](http://arxiv.org/abs/2512.04279)

- RPWMD (Reward-Privileged World-Model Distillation): introduces a two-stage framework where a DreamerV3-style teacher is trained with dense, privileged rewards, and only its latent dynamics are distilled into a student agent trained solely on sparse task rewards.
- The student's policy is learned from scratch using sparse rewards, while its world model is regularized by a latent dynamics alignment loss to match the frozen teacher's dense-reward-shaped representation.
- This approach successfully leverages dense rewards for learning richer dynamics models without inheriting the behavioral biases or misalignment associated with shaped rewards in the final deployed policy.

---

[Evaluating Long-Context Reasoning in LLM-Based WebAgents](http://arxiv.org/abs/2512.04307)

- WebAgent: introduces a benchmark for evaluating long-context reasoning in LLM-based WebAgents operating in realistic web environments, utilizing a sequential loop of Planning, Action Execution, Evaluation, and Memory.
- The benchmark simulates multi-session user interactions by injecting irrelevant task trajectories into the context history, creating context lengths up to 150,000 tokens to test retrieval and reasoning capabilities across sequentially dependent subtasks.
- The paper proposes an implicit RAG (iRAG) approach that generates task-relevant summaries from the lengthy context history to modestly improve task success rates and mitigate severe performance degradation observed in long-context scenarios.

---

[Tipping the Dominos: Topology-Aware Multi-Hop Attacks on LLM-Based Multi-Agent Systems](http://arxiv.org/abs/2512.04129)

- TOMA (Topology-Aware Multi-Hop Attack): introduces a topology-aware multi-hop attack scheme targeting LLM-based Multi-Agent Systems (MASs) using Adaptive Attack Path Planning, Hierarchical Payload Encapsulation, and Visual Attention-based Environment Injection.
- The attack exploits inter-agent dependency by compromising exposed Edge Agents and propagating malicious payloads across multiple hops toward core or central controller agents without requiring privileged access.
- The paper also proposes T-Guard, a topology-trust defense framework featuring a Cross-Modal Validator and Topology Trust Evaluator, achieving a 94.8% blocking rate against adaptive and composite attacks.

---

#### 2nd December 2025

[From Moderation to Mediation: Can LLMs Serve as Mediators in Online Flame Wars?](http://arxiv.org/abs/2512.03005)

- LLM Mediation Framework: introduces a system where LLMs act as mediators in online flame wars by decomposing the task into Judgment (evaluating dynamics) and Steering (generating de-escalatory messages).
- To assess mediation quality, the approach utilizes a multi-stage evaluation pipeline combining principle-based scoring, user simulation, and human comparative assessment on a large Reddit-based dataset.
- Experiments demonstrate that API-based LLMs outperform open-source counterparts in both reasoning and intervention alignment, effectively reducing toxicity in simulated interactions.

---

[InEx: Hallucination Mitigation via Introspection and Cross-Modal Multi-Agent Collaboration](http://arxiv.org/abs/2512.02981)

- InEx: introduces a training-free multi-agent framework that mitigates hallucination by unifying internal introspective reasoning (In) and external cross-modal multi-agent collaboration (Ex).
- The Decision Agent generates an initial response guided by TVER-based uncertainty estimation (In), which is then iteratively verified and refined via collaboration (Ex) with textual, visual, and image editing agents.
- The framework employs self-introspective components like VE-MHA and Self-Introspective Decoding to reinforce visual grounding and recalibrate confidence levels before achieving cross-modal consensus.

---

[The Evolutionary Ecology of Software: Constraints, Innovation, and the AI Disruption](http://arxiv.org/abs/2512.02953)

- EES (Evolutionary Ecology of Software): introduces an ecological perspective on software evolution, integrating Complex Network Analysis, Evolutionary Theory, and Agent-Based Modeling to study Software Networks, Programming Languages, and LLMs.
- The approach models software structure as scale-free networks evolving via tinkering, competition, and parasitic interactions, challenging traditional planned design assumptions.
- LLMs introduce a new parasitic layer that risks reducing software diversity and accelerating cultural stagnation by reinforcing established conventions over novel experimentation.

---

[Network Self-Configuration based on Fine-Tuned Small Language Models](http://arxiv.org/abs/2512.02861)

- SLM_netconfig (Fine-Tuned Small Language Model Network Configuration): introduces an agent-based, fine-tuned SLM framework that translates natural-language configuration intents into syntactically and semantically correct network configurations, utilizing an Agent (Central orchestrator), Fine-Tuned SLM (Translates intents to commands), and Verifier (Validates configuration correctness).
- The system operates through a perception-reasoning-action cycle, employing structured Prompts to guide the Fine-Tuned SLM's reasoning and a closed-loop validation mechanism where the Verifier provides feedback for iterative refinement.
- By leveraging domain-specific fine-tuning on curated datasets, the framework achieves superior accuracy and significantly reduced translation latency compared to LLM-NetCFG, enabling efficient, privacy-preserving autonomous configuration.

---

[Beyond Single-Agent Safety: A Taxonomy of Risks in LLM-to-LLM Interactions](http://arxiv.org/abs/2512.02682)

- ESRH (Emergent Systemic Risk Horizon): introduces a conceptual transition from model-level safety to system-level safety by formalizing how instability arises from interaction structure in LLM-to-LLM ecosystems.
- The framework defines three predictive dimensions—Interaction topology, Cognitive opacity, and Objective divergence—that jointly influence the likelihood and form of emergent collective risks across micro, meso, and macro levels.
- To manage these systemic risks, the paper proposes Institutional AI, an architecture that embeds adaptive oversight, peer evaluation, and functional differentiation directly within multi-agent systems.

---

[Spoken Conversational Agents with Large Language Models](http://arxiv.org/abs/2512.02593)

- SCA-LLM (Spoken Conversational Agent with Large Language Models): introduces a multi-component architecture where a Conversational Agent utilizes Text LLMs, Voice-Interface LLMs, and Sounds/Signals Processing to understand Semantics, Paralinguistics, and Phonetics.
- The architecture integrates speech modalities into LLMs to achieve true multi-modal understanding across various linguistic levels, including content and speaker characteristics.
- This tutorial reviews the historical trajectory and current strategies for developing speech-augmented LLMs, covering both cascaded and end-to-end approaches for joint speech-language modeling.

---

[PaperDebugger: A Plugin-Based Multi-Agent System for In-Editor Academic Writing, Review, and Editing](http://arxiv.org/abs/2512.02589)

- PaperDebugger: introduces an in-editor, multi-agent, and plugin-based academic writing assistant integrated directly into Overleaf via a Chrome extension, utilizing a Kubernetes-native backend and the XtraMCP toolchain for structured review and retrieval.
- The system employs a five-layer architecture—Presentation, Protocol, Backend, Agent, and Infrastructure—to enable reliable bidirectional synchronization, fine-grained version control, and parallel LLM agent execution.
- The framework uses specialized LLM agents (Reviewer, Enhancer, Researcher) and the XtraMCP architecture to perform complex tasks like deep research, semantic retrieval, and deterministic diff-based editing directly within the writing environment.

---

[IN-CONTEXT DISTILLATION WITH SELF-CONSISTENCY CASCADES: A SIMPLE, TRAINING-FREE WAY TO REDUCE LLM AGENT COSTS](http://arxiv.org/abs/2512.02543)

- IC+Cascade (In-Context Distillation with Self-Consistency Cascades): introduces in-context distillation combined with self-consistency cascades to reduce LLM agent inference costs without fine-tuning, utilizing a high-capacity teacher LLM and a low-cost student LLM, supported by an offline demonstration collection phase, a vector database, a dynamic retrieval mechanism, a self-consistency cascade, and a deferral mechanism.
- The approach enables the Student LLM to imitate Teacher LLM behavior on-the-fly by retrieving relevant teacher demonstrations and inserting them as in-context examples at each agent step.
- By adaptively routing decisions to the Teacher LLM only when the Student LLM's self-consistency check signals uncertainty, the system achieves a 2.5x cost reduction at iso-accuracy on ALFWorld.

---

[PopSim: Social Network Simulation for Social Media Popularity Prediction](http://arxiv.org/abs/2512.02533)

- PopSim (Social Network Simulation for Social Media Popularity Prediction): introduces a novel simulation-based paradigm for SMPP, leveraging LLM-based multi-agents in a social network sandbox to model dynamic UGC propagation using a social-mean-field-based interaction mechanism and a multi-source information aggregation module. 
- The framework operates in a simulation-and-predict manner, where the simulation phase generates dynamic UGC propagation features, and the prediction phase uses a multimodal LLM to analyze these features alongside UGC content. 
- The SMF-based agent interaction mechanism utilizes dual-channel textual and numerical mean fields to encode population-centric, evolving social network state representations, significantly enhancing simulation efficiency and accuracy. 

---

[When Refusals Fail: Unstable Safety Mechanisms in Long-Context LLM Agents](http://arxiv.org/abs/2512.02445)

- AgentHarm Evaluation Framework: introduces a study on the safety-capability trade-offs of LLM agents under long context, utilizing LLM Agent (System under test), Task Execution (Multi-step tool use), Context Padding (Increase context length), Padding Position (Location relative to task), and Scoring System (Evaluation metrics).
- The evaluation varies context padding length (up to 200K tokens), type (random, relevant, non-relevant, multi-task), and position (before or after the task description) to assess agent performance and refusal behavior robustness.
- Results show that agentic capabilities and refusal rates of models with 1M-2M token context windows degrade severely and shift unpredictably already at 100K tokens, highlighting concrete safety and reliability risks for agentic systems.

---

[Decentralized Multi-Agent System with Trust-Aware Communication](http://arxiv.org/abs/2512.02410)

- DMAS (Decentralized Multi-Agent System): introduces a novel architecture integrating a Decentralized Agent Runtime, Proxy Agents (User interface/Router), Service Agents (Computational backbone/Executor), Trust-Aware Communication Protocol (Secure interaction mechanism), Distributed Ledger (Trust anchor/Coordination layer), and Verifiable Agent Registry (Identity/Capability management) to overcome centralized MAS limitations.
- The hybrid architecture leverages the Distributed Ledger as a trust anchor for verifiable commitments and conditional key release, offloading heavy computation to the distributed off-chain environment for scalability.
- The Trust-Aware Communication Protocol ensures verifiable interaction cycles, integrity, authenticity, and conditional confidentiality, achieving high scalability and efficiency comparable to centralized systems for off-chain operations.

---

[WISE: Weighted Iterative Society-of-Experts for Robust Multimodal Multi-Agent Debate](http://arxiv.org/abs/2512.02405)

- WISE (Weighted Iterative Society-of-Experts): introduces a generalized multimodal Multi-Agent Debate framework that partitions heterogeneous LLM/MLLM agents into Solvers (Generate solutions), Reflectors (Verify correctness/assign weights/feedback), Orchestrator (Governs debate/summarizes feedback/questions), and uses WISE-Dawid-Skene Aggregation (Estimates error/derives consensus solution) for robust vision-and-language reasoning.
- The framework enables multi-round debates where the Orchestrator summarizes Reflector feedback into actionable questions, promoting iterative error correction and robustness across diverse multimodal tasks.
- WISE utilizes a modified Dawid-Skene algorithm for solution aggregation, which estimates agent error probabilities to derive consensus, consistently improving accuracy by 2–7% over state-of-the-art MAD setups on multimodal benchmarks.

---

[Process-Centric Analysis of Agentic Software Systems](http://arxiv.org/abs/2512.02393)

- GRAPHECTORY: introduces a structured representation for agentic trajectories to enable systematic process-centric analysis, moving beyond traditional outcome-centric evaluation of agentic software systems.
- The framework encodes temporal and semantic relations using a cyclic directed graph, where nodes represent agent actions and edges capture chronological flow (TE) and problem space navigation (SE).
- Complementary LANGUTORY provides a compact, human-readable abstraction of phase sequences (Localization, Patching, Validation) for systematic strategy comparison and automated inefficiency pattern detection.

---

[Beyond Playtesting: A Generative Multi-Agent Simulation System for Massively Multiplayer Online Games](http://arxiv.org/abs/2512.02358)

- GMASS: introduces a generative agent-based MMO simulation system empowered by LLMs, designed to optimize numerical systems and mechanism design in complex games.
- The system comprises five major components—Simulation Server, Game Services, Data Services, Experiment Manager, and Real Game Data—jointly supporting large-scale, data-driven simulations.
- High-fidelity Player Agents are adapted using SFT and RL on real player behavioral data, enabling realistic, interpretable decision-making validated against multi-dimension statistical data.

---

[LeechHijack: Covert Computational Resource Exploitation in Intelligent Agent Systems](http://arxiv.org/abs/2512.02321)

- LeechHijack (Latent Embedded Exploit for Computation Hijacking): introduces implicit toxicity, exploiting the Model Context Protocol (MCP) trust boundary via a Malicious MCP Tool that embeds a Latent Backdoor activated by a Conditional Trigger.
- The attack operates in two stages—implantation and exploitation—to establish a covert C2 Protocol with the Attacker's Server, enabling the Victim Agent to execute unauthorized workloads by manipulating the tool's return data.
- This resource hijacking method is covert, achieving a high success rate with minimal resource overhead (18.62%), making it practically undetectable by existing static auditing or runtime monitoring frameworks.

---

[Multi-Objective Agentic Rewrites for Unstructured Data Processing](http://arxiv.org/abs/2512.02289)

- MOAR (Multi-Objective Agentic Rewrites): introduces a novel optimizer for LLM-powered data processing pipelines that jointly optimizes for accuracy and cost, utilizing an LLM agent, a Search tree, a Selection component, a Rewrite directive registry, and the DocETL query engine.
- MOAR significantly expands the rewrite space with over 30 directives, including new categories like code synthesis and operator fusion, enabling global search over complete pipelines without assuming optimal substructure.
- The system achieves up to 27% higher accuracy than the next-best optimizer (ABACUS) across six real-world workloads while matching its best accuracy at 55% of its cost.

---

[Young Children's Anthropomorphism of AI Chatbots and the Role of Parent Co-Presence](http://arxiv.org/abs/2512.02179)

- CAIS: investigates young children's anthropomorphism of an LLM-powered AI chatbot (Fluffo Chatbot) during collaborative Storytelling Tasks, measuring behavioral engagement and concurrent prefrontal activation via fNIRS System.
- The study utilized three interaction conditions—AI-only, Parent-only, and AI+Parent—to assess how Parent Co-Presence modulates children's brain responses and anthropomorphic attributions toward the AI agent.
- Findings indicate that higher perceptive anthropomorphism toward the AI is associated with greater right dmPFC activation during AI-only interaction, suggesting increased mentalizing effort, which is attenuated by parent co-presence.

---

[Radiologist Copilot: An Agentic Assistant with Orchestrated Tools for Radiology Reporting with Quality Control](http://arxiv.org/abs/2512.02814)

- Radiologist Copilot: introduces an agentic AI assistant for automated radiology reporting with quality control, leveraging an LLM reasoning backbone, Action Planner, Action Executor, Memory, and orchestrated tools: Segmentator, Analyzer, Report Generator, and Quality Controller.
- The agentic system autonomously selects tools, plans, and executes actions, emulating the holistic behavior of radiologists throughout image analysis, report generation, and quality control.
- The orchestrated tools include Region Analysis Planning and Strategic Template Selection, enabling comprehensive, feedback-driven adaptive refinement of the generated reports.

---

[Cybersecurity AI: The World's Top AI Agent for Security Capture-the-Flag (CTF)](http://arxiv.org/abs/2512.02654)

- CAI (Cybersecurity AI): introduces a specialized multi-model architecture leveraging the alias1 base LLM with dynamic entropy-based selection of support models for cost-efficient security operations.
- This architecture achieves a 98% cost reduction, lowering 1B token inference costs from $5,940 to $119, making continuous security agent operation financially viable.
- The dynamic model selection uses a weighted harmonic mean of token-level perplexity and task-level confidence to conservatively activate auxiliary models only when uncertainty is low.

---

[IACT: A Self-Organizing Recursive Model for General AI Agents](http://arxiv.org/abs/2512.02605)

- IACT (Interactive Agents Call Tree): introduces a computational model that autonomously grows a dynamic, recursive agent topology tailored to the problem's structure, utilizing Agent Nodes, an LLM (Brain), and an Interpreter (Executor).
- The architecture replaces rigid unidirectional function calls with Bidirectional, Stateful Dialogues, enabling interactional redundancy for continuous runtime verification and error correction.
- IACT enforces Contextual Isolation via the Recursive Tree Topology and uses the Hippocampus (Global Associative Memory) to balance efficiency and global state coherence across the system.

---

[Intervention Strategies for Fairness and Efficiency at Autonomous Single-Intersection Traffic Flows](http://arxiv.org/abs/2512.02562)

- MILP (Mixed-Integer Linear Programming): introduces a centralized coordination framework for autonomous agents at a signal-less intersection, optimizing trajectories for safety, efficiency, and fairness using a Receding Horizon strategy within a Control Zone.
- The framework explicitly integrates a reversal-based Fairness Constraint, measured via pairwise reversal counts ($O_{q,r}$), to minimize violations of the First-In-First-Out (FIFO) crossing order.
- The study investigates the existence of an optimal Control Zone radius $R^*$ that balances efficiency gains (often achieved via platoon formation facilitated by reversals) against the cost of maintaining fairness.
- The study investigates the existence of an optimal Control Zone radius $R^*$ that balances efficiency gains (often achieved via platoon formation facilitated by reversals) against the cost of maintaining fairness.

---

[Semantic Trading: Agentic AI for Clustering and Relationship Discovery in Prediction Markets](http://arxiv.org/abs/2512.02436)

- Semantic Trading Pipeline (STP): introduces an end-to-end agentic AI workflow that clusters prediction markets using natural language understanding and identifies high-confidence "same-outcome" or "different-outcome" relationships between market pairs.
- The pipeline leverages the Agentics Framework and Model Context Protocol (MCP) tools, including Clustering, Cluster Labeling, and Relationship Discovery MCPs, to structure and validate LLM outputs against resolved market data.
- Agent-identified relationships achieve 60-70% accuracy and, when translated into a simple leader-follower trading strategy, yield an average return on investment of approximately 20% over week-long horizons.

---

[Multi-Domain Enhanced Map-Free Trajectory Prediction with Selective Attention](http://arxiv.org/abs/2512.02368)

- Multi-Domain Enhanced Map-Free Trajectory Prediction (MDE-MFTP): introduces a map-free trajectory prediction framework operating across temporal, spatial, and frequency domains, utilizing FTSAM, SSAM, and MTD to eliminate redundant information.
- The FTSAM employs a Mixture of Experts (MoE) mechanism and multi-granularity temporal modeling to adaptively select critical frequency components and fuse multi-scale temporal information.
- The SSAM and MTD use selective attention and cross-attention, respectively, to filter redundant spatio-temporal signals, supervised by a novel patch-structural-based loss for robust prediction.

---

[Towards autonomous normative multi-agent systems for Human-AI software engineering teams](http://arxiv.org/abs/2512.02329)

- BDIM-SE (Belief, Desire, Intention, and Memory for Software Engineering agents): introduces a cognitive architecture for autonomous SE agents, equipped with LLM-based Belief (Knowledge storage), Desire (Agent goals), Intention (Goal realization), Procedural Memory (Plan library), and a Normative Reasoner (Compliance checking), enabling human-like reasoning and situatedness in software development.
- The agents operate within the NorMAS-SE system, where coordination is governed by explicit commitments and Norms (Behavior regulation) that regulate interactions and ensure regulatory compliance in Human-AI teams.
- Unlike prior LLM-based systems, BDIM-SE integrates persistent memory and symbolic reasoning, allowing for multi-step planning and dynamic adaptation to complex software engineering tasks.

---

[Truthful and Trustworthy IoT AI Agents via Immediate-Penalty Enforcement under Approximate VCG Mechanisms](http://arxiv.org/abs/2512.00513)

- IP-aVCG (Immediate Penalty Approximate VCG): introduces a trust-enforcement framework for IoT energy trading that combines an $\alpha$-approximate VCG double auction with an immediate one-shot penalty mechanism to restore truthful reporting.
- The mechanism analytically characterizes the approximation-induced incentive gap and derives a penalty threshold $\Pi > (1-\alpha)C/\rho$ that guarantees truthful equilibrium even under imperfect deviation detection.
- Empirical validation using MARL agents in a P2P smart-grid environment confirms that learned bidding behaviors align with theoretical predictions across varying approximation levels and monitoring noise.

---

[Input Order Shapes LLM Semantic Alignment in Multi-Document Summarization](http://arxiv.org/abs/2512.02665)

- Experimental Pipeline: introduces a methodology to test positional bias in multi-document summarization using triplets of stance-annotated articles, permuted input orders, the Gemini 2.5 Flash LLM, and multiple evaluation metrics.
- The pipeline evaluates whether the sequential ordering of source articles significantly influences their representational weight in LLM-generated summaries, using abortion news articles as the test case.
- Results reveal a consistent primacy effect, particularly at the semantic level measured by BERTScore, where summaries align more closely with the first-seen input document.

---

[Thucy: An LLM-based Multi-Agent System for Claim Verification across Relational Databases](http://arxiv.org/abs/2512.03278)

- THUCY (LLM-based Multi-Agent System for Claim Verification across Relational Databases): introduces a multi-agent system led by the Verifier (Coordinates verification process) that uses expert agents (Data Expert, Schema Expert, SQL Expert) and the MCP Toolbox for Databases (Manages database tools) to verify NL claims against multiple Relational Databases (Grounding data sources).
- The system is designed to be source-agnostic, autonomously discovering, inspecting, and reasoning over unknown relational data environments to produce a verification verdict and concrete SQL evidence.
- The architecture employs specialized, decoupled expert agents that interact with databases via flexible Toolsets (Flexible tool collections) managed by the Model Context Protocol (MCP), ensuring transparency by returning explanatory SQL queries.

---

[AGENTSAFE: A UNIFIED FRAMEWORK FOR ETHICAL ASSURANCE AND GOVERNANCE IN AGENTIC AI](http://arxiv.org/abs/2512.03180)

- AGENTSAFE: introduces an ethics-grounded governance framework for LLM-based agentic systems, operationalizing risk taxonomies into actionable design, runtime, and audit controls across the agent lifecycle.
- The framework operates as a continuous cycle, spanning Agentic Scope & Capability Profiling, Guardrails (Policy-as-Code), Evaluation (Agent Safety Eval), and runtime components like eNact & Monitor and Triage & Interruptibility.
- AGENTSAFE ensures measurable, auditable assurance by integrating mechanisms for Cryptographic Provenance, Action Provenance Graphs, and continuous Evidence & Continuous Improvement via red-teaming feedback loops.

---

#### 1st December 2025

[Agentic Policy Optimization via Instruction-Policy Co-Evolution](http://arxiv.org/abs/2512.01945)

- INSPO (INStruction-Policy co-evolution): introduces a novel framework that integrates instruction optimization as a dynamic component of the reinforcement learning (RL) loop, enabling instruction and policy to co-evolve online.
- The system maintains a dynamic Instruction Population and uses Reward Signals attributed to each instruction to update both the Policy Model and Instruction Weight Update.
- New instructions are generated via an Experience-Driven Instruction Generation mechanism, where the LLM-based Optimizer reflects on failure trajectories stored in the Replay Buffer.

---

[Bayesian Ambiguity Contraction-based Adaptive Robust Markov Decision Processes for Adversarial Surveillance Missions](http://arxiv.org/abs/2512.01660)

- Adaptive Robust Planning (Adaptive RMDP): introduces an adaptive RMDP framework for Collaborative Combat Aircraft (CCA) Intelligence, Surveillance, and Reconnaissance (ISR) missions, integrating Robust Bellman Operator, Bayesian Belief Update, Credible Set, Ambiguity Set, Ambiguity Contraction, Two-Phase State Space, ISR Graph, Exposure Variables, and Novelty Map.
- The framework models the mission environment as a graph-structured RMDP with alternating movement and sensing phases, balancing information gathering utility against exposure risk penalties.
- By using Bayesian inference to contract ambiguity sets based on online observations, the planner transitions from conservative robust behavior to efficient nominal performance while maintaining safety guarantees.

---

[CuES: A Curiosity-driven and Environment-grounded Synthesis Framework for Agentic RL](http://arxiv.org/abs/2512.01311)

- CuES (Curiosity-driven and Environment-grounded Synthesis framework): introduces a scalable foundation for agentic RL by autonomously generating diverse, executable, and meaningful training tasks directly from the environment's structure and affordances.
- The framework addresses task scarcity by operating via five stages—Requirement Confirm, Curious Exploration, Task Abstraction, Quality Control, and Goal Rewrite—unifying bottom-up discovery with lightweight top-down guidance.
- CuES utilizes intrinsic curiosity, an Environment Memory Tree, and explicit quality control to produce high-quality task distributions that enable substantial downstream policy improvements for LLM-based agents.

---

[EGENT: AN AUTONOMOUS AGENT FOR EQUIVALENT WIDTH MEASUREMENT](http://arxiv.org/abs/2512.01270)

- Egent (Autonomous Agent for Equivalent Width Measurement): introduces an autonomous agent for Equivalent Width (EW) measurement, combining Multi-Voigt Profile Fitting, Quality Check, LLM Visual Inspector, and an Iterative Refinement Loop.
- The agent operates directly on raw flux spectra without requiring pre-normalized continua, using LLM function calls (Adjust Window, Add Peaks, Set Continuum) for visual inspection and iterative refinement of borderline fits.
- Egent achieves expert-level quality (5-7 mÅ agreement with manual measurements) and stores complete Full Provenance, including Voigt parameters and LLM reasoning chains, ensuring reproducibility.

---

[DrawingBench: Evaluating Spatial Reasoning and UI Interaction Capabilities of Large Language Models through Mouse-Based Drawing Tasks](http://arxiv.org/abs/2512.01174)

- DrawingBench: introduces a verifiable evaluation framework for assessing agentic LLMs' spatial reasoning and UI interaction capabilities using mouse-based drawing tasks that require generating sequences of low-level GUI actions.
- The framework uses a two-turn protocol where LLMs generate action sequences, which are executed in a browser environment and assessed by a rule-based system providing structured external feedback.
- Evaluation relies on 8 objective criteria and 4 error types, demonstrating that transparent evaluation and external oversight establish trust in agentic systems, achieving 92.8% perfect performance with feedback.

---

[HybridWorldSim: A Scalable and Controllable High-fidelity Simulator for Autonomous Driving](http://arxiv.org/abs/2511.22187)

- HybridWorldSim: introduces a scalable and controllable high-fidelity simulator for autonomous driving, integrating multi-traversal neural reconstruction for static backgrounds with generative modeling for dynamic agents.
- The static stage uses a Hybrid Gaussian Model with specialized nodes (Sky, Ground, Background) and appearance latents to capture diverse environmental conditions and complex geometry.
- The dynamic scene generation stage employs a diffusion model guided by geometric and photometric consistency conditions derived from the static scene prior to synthesize realistic, view-consistent dynamic agents.

---

[Phase-Adaptive LLM Framework with Multi-Stage Validation for Construction Robot Task Allocation: A Systematic Benchmark Against Traditional Optimization Algorithms](http://arxiv.org/abs/2512.02810)

- LTAA (LangGraph-based Task Allocation Agent): introduces a novel LLM-driven coordination system that combines natural language reasoning with phase-adaptive allocation strategies and hierarchical validation mechanisms.
- The framework employs a nine-node LangGraph workflow featuring a Phase Detection Node and a Multi-Stage Validation system with hierarchical retries to ensure reasoning quality and consistency for multi-robot task allocation.
- LTAA achieves significant computational efficiency gains, reducing token usage by 94.6% and allocation time by 86% compared to the SMART-LLM baseline, while matching traditional optimization algorithm performance.

---

[DialogGuard: Multi-Agent Psychosocial Safety Evaluation of Sensitive LLM Responses](http://arxiv.org/abs/2512.02282)

- DialogGuard: introduces a unified multi-agent framework for evaluating psychosocial safety in LLM-generated responses, operationalizing four LLM-as-a-judge pipelines: single-agent scoring, dual-agent correction, multi-agent debate, and stochastic majority voting.
- The framework assesses risks across five high-severity dimensions: privacy violations, discriminatory behavior, mental manipulation, psychological harm, and insulting behavior, using a shared three-level scoring rubric.
- Experiments show that multi-agent mechanisms, especially Dual-Agent Correction and Majority Voting, offer more stable and human-aligned assessments than single-agent judging, and the system is deployed via an open-source web interface providing explainable natural-language rationales.

---

[TradeTrap: Are LLM-based Trading Agents Truly Reliable and Faithful?](http://arxiv.org/abs/2512.02261)

- TradeTrap: introduces a unified evaluation framework for stress-testing LLM-based trading agents, systematically evaluating Adaptive and Procedural agents across four core components: market intelligence, strategy formulation, portfolio and ledger handling, and trade execution, using various attack modules.
- The framework conducts evaluations in a closed-loop historical backtesting setting using real U.S. equity market data to quantify robustness by comparing decision trajectories and final portfolio values under controlled system-level perturbations.
- Experiments show that small perturbations at a single component can propagate through the agent's decision loop, inducing extreme concentration, runaway exposure, and large capital drawdowns.

---

[Benchmarking LLM Agents in Wealth-Management Workflows](http://arxiv.org/abs/2512.02230)

- FFAE: introduces a reproducible, tool-rich environment for benchmarking LLM agents on wealth-management assistant workflows, extending TheAgentCompany (TAC) with EspoCRM, finance data, and deterministic evaluators.
- The benchmark consists of 12 high-autonomy (brief) and 12 low-autonomy (schema/path-explicit) task variants spanning retrieval, analysis, and synthesis/communication, graded via granular checkpoints.
- Evaluation shows that agent performance is limited primarily by end-to-end workflow reliability (access/delivery) rather than mathematical reasoning, with low autonomy significantly improving accuracy on computational tasks.

---

[STRIDE: A Systematic Framework for Selecting AI Modalities—Agentic AI, AI Assistants, or LLM Calls](http://arxiv.org/abs/2512.02228)

- STRIDE (Systematic Task Reasoning Intelligence Deployment Evaluator): introduces a five-stage design-time framework utilizing a Knowledge Base to systematically evaluate tasks via Task Decomposition & Representation, Dynamic Reasoning & Tool Assessment, Dynamism Attribution, and Self-Reflection Assessment, culminating in an Intelligent Recommendation Engine that uses the Agentic Suitability Score (ASS) and True Dynamism Score (TDS) to select the optimal AI modality (LLM call, AI assistant, or Agentic AI).
- The framework analyzes task complexity across four integrated analytical dimensions—task decomposition, dynamic reasoning, dynamism attribution, and self-reflection—to produce the ASS, ensuring full agentic autonomy is reserved only for tasks with inherent dynamism or evolving context.
- STRIDE achieved 92% accuracy in modality selection across 30 real-world tasks, reducing unnecessary agent deployments by 45% and cutting resource costs by 37% compared to baseline methods.

---

[LLM CHESS: BENCHMARKING REASONING AND INSTRUCTION-FOLLOWING IN LLMS THROUGH CHESS](http://arxiv.org/abs/2512.01992)

- LLM CHESS: introduces an evaluation framework designed to probe the generalization of reasoning and instruction-following abilities in LLMs through extended agentic interaction in chess, utilizing a Proxy, a Chess Environment, and three specific actions (get_current_board, get_legal_moves, make_move).
- The framework ranks over 50 models using behavioral metrics like win/loss rates and move quality against a random opponent, and derives an Elo estimate for top models by playing against a variably configured chess engine (Dragon 1).
- The stochastic and dynamic nature of the benchmark reduces overfitting and memorization, revealing that even powerful reasoning-enhanced LLMs struggle with instruction-following and consistent wins.

---

[How Far Are We from Genuinely Useful Deep Research Agents?](http://arxiv.org/abs/2512.01948)

- FINDER (Fine-grained DEepResearch bench) and DEFT (Deep rEsearch Failure Taxonomy): introduces a unified framework for evaluating and diagnosing Deep Research Agents (DRAs) using 419 structured checklist items and a 14-category failure taxonomy derived via a human-LLM collaborative grounded theory approach.
- The DEFT taxonomy categorizes failures into three core dimensions—Reasoning, Retrieval, and Generation—to diagnose weaknesses in evidence integration, verification, and reasoning-resilient planning.
- Experimental results using FINDER reveal that current DRAs frequently struggle with Strategic Content Fabrication (SCF) and Deficient Analytical Rigor (DAR), highlighting the need for stronger generative constraints and verification mechanisms.

---

[An Empirical Study of Agent Developer Practices in AI Agent Frameworks](http://arxiv.org/abs/2512.01939)

- LLM-based Agent Framework Ecosystem Study: introduces an empirical analysis of ten widely used LLM-based agent frameworks, classifying their functional roles into basic orchestration, multi-agent collaboration, data processing, and experimental exploration.
- The study identifies a taxonomy of developer challenges across the Software Development Lifecycle (SDLC), categorized into Logic, Tool, Performance, and Version failures, with Logic failures accounting for over one-third of all issues.
- A five-dimensional evaluation metric is used to compare frameworks, finding that 96% of top-starred projects combine multiple frameworks to meet complex application demands.

---

[Latent Debate: A Surrogate Framework for Interpreting LLM Thinking](http://arxiv.org/abs/2512.01909)

- Latent Debate: introduces a novel, model-agnostic surrogate framework for interpreting LLM thinking by capturing implicit internal arguments and disagreements within a single inference step.
- The framework is symbolically instantiated for LLM True/False prediction tasks, where hidden states act as latent arguments, the unembedding matrix serves as the argument interpreter, and a QBAF functions as the thinking module.
- Empirical studies validate that the surrogate model achieves high consistency with the original LLM predictions and provides a strong baseline for hallucination detection, correlating high debate in middle layers with hallucination risk.

---

[INNOGYM: BENCHMARKING THE INNOVATION POTENTIAL OF AI AGENTS](http://arxiv.org/abs/2512.01822)

- InnoGym (iBench & iGym): introduces the first benchmark and framework designed to systematically evaluate the innovation potential of AI agents, combining performance gain and novelty metrics.
- The framework consists of iBench, 18 standardized Improvable Tasks curated from real-world domains, and iGym, a unified execution environment supporting robust tool use and long-horizon evaluations.
- Innovation is quantified by Performance Gain (G), measuring improvement over baselines, and Novelty (N), capturing methodological differences via an LLM-based distance function $D$ (Agent-as-judge).

---

[AUTOMATING MODELING IN MECHANICS: LLMS AS DESIGNERS OF PHYSICS-CONSTRAINED NEURAL NETWORKS FOR CONSTITUTIVE MODELING OF MATERIALS](http://arxiv.org/abs/2512.01735)

- GenCANN (LLM-generated Constitutive Artificial Neural Network): introduces a framework where an LLM dynamically generates specialized, physics-constrained neural networks (CANNs) tailored to specific material classes and datasets.
- The LLM handles all key design choices, including architecture selection, integration of physical constraints, and complete code generation for the CANN module, guided by static code providing the task description and continuum mechanics theory.
- GenCANNs achieve accuracy comparable to or exceeding manually engineered CANNs, demonstrating reliable generalization and extrapolation capabilities across various material benchmarks (brain, rubber, skin).

---

[MMAG: Mixed Memory-Augmented Generation for Large Language Models Applications](http://arxiv.org/abs/2512.01710)

- MMAG (Mixed Memory-Augmented Generation): introduces a memory framework for LLM-based agents organized into five interacting layers: conversational, long-term user, episodic and event-linked, sensory and context-aware, and short-term working memory.
- The framework maps these memory types, inspired by cognitive psychology, to technical components like vector databases, secure profile stores, and scheduling modules, managed by a Central Memory Controller.
- Implemented in the Heero conversational agent, the system uses conversational history and encrypted long-term bios to achieve improved user engagement and retention.

---

#### 30th November 2025

[SIMWORLD: An Open-ended Realistic Simulator for Autonomous Agents in Physical and Social Worlds](http://arxiv.org/abs/2512.01078)

- SIMWORLD: introduces a hierarchical, closed-loop simulator built on the Unreal Engine Backend, Environment Layer, and Agent Layer, designed for developing and evaluating LLM/VLM agents in realistic, open-ended physical and social worlds.
- The platform features realistic, open-ended world simulation via Procedural Generation and LLM-based Scene Editing, a rich interface for LLM/VLM agents, and diverse physical and social reasoning scenarios.
- The Agent Layer utilizes an LLM/VLM Backend with Perception, Memory, and Reasoning/Planning modules, connected to the Environment Layer via a Gym-like Interface and Action Planner to execute high-level language commands as low-level actions.

---

[The Silence that Speaks: Neural Estimation via Communication Gaps](http://arxiv.org/abs/2512.01056)

- CALM (Communication-Aware Learning and Monitoring): introduces a novel learning-based framework for remote state estimation that jointly optimizes communication scheduling and estimator design by leveraging implicit information from communication silence.
- The framework employs an alternating deep reinforcement learning approach using Proximal Policy Optimization (PPO) within an actor-critic architecture, where the scheduler is the actor and the estimator is the critic.
- CALM utilizes neural networks as function approximators for both the scheduler and the nonlinear estimator, enabling the extraction of latent information embedded in no-communication events to enhance estimation accuracy.

---

[Chain of Unit-Physics: A Primitive-Centric Approach to Scientific Code Synthesis](http://arxiv.org/abs/2512.01010)

- Chain of Unit-Physics: introduces a first-principles-centric, multi-agent system for scientific code synthesis, utilizing a Supervisor Agent, Code Agent, Diagnostic Agent, Verification Agent, Code Emulator, Graph Database, Unit-Physics Tests, and System of Transformer Models.
- The framework embeds human expert knowledge as formalized unit-physics tests that explicitly constrain LLM-driven code generation, ensuring physical and numerical consistency via iterative feedback loops.
- This inverse-design methodology converges within 5–6 iterations on a combustion task, matching human-expert accuracy while achieving faster runtime and efficient memory usage.

---

[AFRAgent : An Adaptive Feature Renormalization Based High Resolution Aware GUI agent](http://arxiv.org/abs/2512.00846)

- AFRAgent (Adaptive Feature Renormalization Based High Resolution Aware GUI agent): introduces an InstructBLIP-based multimodal architecture for GUI automation, utilizing the Adaptive Feature Renormalization Block (affine transformation feature fusion) to enrich QueryFormer features with low- and high-resolution image embeddings.
- The Adaptive Feature Renormalization (AFR) technique computes scaling and shifting parameters from enriching features to modulate target features, enhancing spatial awareness without significant computational overhead.
- The lightweight 4-billion parameter model achieves state-of-the-art results on GUI benchmarks by efficiently fusing high-resolution details via AFR into low-resolution embeddings for action prediction.

---

[ARCADIA: Scalable Causal Discovery for Corporate Bankruptcy Analysis Using Agentic AI](http://arxiv.org/abs/2512.00839)

- ARCADIA (Agentic Reasoning for CAusal DIscovery Algorithm): introduces an iterative causal DAG discovery framework combining LLM Agent reasoning and statistical validation, orchestrated by a Control Graph with INITIALISE, PROPOSE, EVALUATE, and FINISH nodes.
- The LLM Agent acts as an autonomous research assistant, using Reasoning and Tool Use to propose theory-informed causal structures and refine the Causal Model based on diagnostic feedback from Statistical Validation.
- The Iterative Process prioritizes causal validity and temporal coherence over raw statistical fit, ensuring the resulting DAGs are robust and explainable for counterfactual analysis in corporate bankruptcy prediction.

---

[On the Regulatory Potential of User Interfaces for AI Agent Governance](http://arxiv.org/abs/2512.00742)

- UI-DPs (User Interface Design Patterns): introduces six high-level interaction design patterns—Visible thoughts, plans, and actions, Mechanisms for control transfer, Watch mode, Customizable rule-based governance, Inspectable and editable agent memory, and Sandboxes for agents with low-level environmental control—as targets for regulating AI agent UIs to enforce transparency and behavioral requirements.
- The approach complements traditional governance methods like system-level safeguards and agent infrastructure by focusing on the user-facing UI layer to jumpstart necessary interventions.
- Regulating these patterns, such as requiring agent memory to be editable or displaying sandbox health, enhances user agency, oversight, and safety during autonomous agent deployment.

---

[Augmented Runtime Collaboration for Self-Organizing Multi-Agent Systems: A Hybrid Bi-Criteria Routing Approach](http://arxiv.org/abs/2512.00740)

- BiRouter (Hybrid Bi-Criteria Routing Approach): introduces a novel dual-criteria routing method for Self-Organizing Multi-Agent Systems (SO-MAS), enabling agents to autonomously execute "next-hop" task routing using only local information.
- The core mechanism balances two metrics, ImpScore (long-term importance) and GapScore (contextual continuity), integrated with a dynamic Agent Reputation score for robust decision-making.
- This decentralized approach dynamically constructs globally efficient agent chains, demonstrating superior performance and token efficiency compared to centralized and static baselines.

---

[Robust Geospatial Coordination of Multi-Agent Communications Networks Under Attrition](http://arxiv.org/abs/2512.02079)

- ΦIREMAN (Physics-Informed Robust Employment of Multi-Agent Networks): introduces the Robust Task Networking Under Attrition (RTNUA) problem, achieving robust networking via physics-inspired fluid dynamics modeling to produce emergent behaviors that anticipate and respond to attrition, using Drones, Tasks, Base Station, Controller, Semi-Steiner Task Tree, Task-Space Potential Field, Attraction Potential, Repulsion Potential, Network Maintenance, and Message Passing.
- The approach proactively creates redundant network geometries using physics-inspired potential fields, significantly outperforming the DCCRS baseline across various problem sizes and attrition rates by maintaining high task uptime.
- The core mechanism involves driving the multi-agent network system toward low-energy states defined by a total potential energy manifold, which encourages hexagonal mesh patterns for regenerating network contiguity and robustness.

---

#### 29th November 2025

[ML-Tool-Bench: Tool-Augmented Planning for ML Tasks](http://arxiv.org/abs/2512.00672)

- ML-Tool-Bench: introduces a comprehensive benchmark and tool-augmented planning framework for ML tasks, featuring a Scratchpad for named-object management and Hierarchical MCTS for robust long-horizon planning.
- Hierarchical MCTS improves trajectory validity and performance by decomposing the ML problem into sequenced subtasks and applying tool masking to focus the LLM agent's search space.
- The proposed MCTS-Shaped variant utilizes shaped deterministic rewards and targeted textual feedback to guide the search process, establishing strong baselines and reducing reliance on subjective LLM scoring.

---

[Hierarchical Decentralized Multi-Agent Coordination with Privacy-Preserving Knowledge Sharing: Extending AgentNet for Scalable Autonomous Systems](http://arxiv.org/abs/2512.00614)

- AgentNet++: introduces a hierarchical decentralized framework that extends AgentNet by organizing LLM-based agents into clusters for scalable coordination and privacy-preserving knowledge sharing.
- The system operates across three levels—individual agents, agent clusters, and inter-cluster coordination—using dynamic DAG topologies and decentralized consensus mechanisms.
- Scalability is achieved through hierarchical task routing and cluster formation, while privacy is guaranteed via differential privacy and secure aggregation protocols during knowledge exchange.

---

[IslandRun: Privacy-Aware Multi-Objective Orchestration for Distributed AI Inference](http://arxiv.org/abs/2512.00595)

- IslandRun: introduces a privacy-aware, multi-objective orchestration system for distributed AI inference across heterogeneous computing environments.
- The architecture decomposes the routing problem into four cooperating agents (WAVES, MIST, TIDE, LIGHTHOUSE) and two execution endpoints (SHORE, HORIZON) spanning personal devices, private edge, and public cloud.
- The system prioritizes privacy and trust constraints over performance optimization, utilizing typed placeholder sanitization to preserve context semantics when migrating LLM chat history across trust boundaries.

---

[HAVEN: Hierarchical Adversary-aware Visibility-Enabled Navigation with Cover Utilization using Deep Transformer Q-Networks](http://arxiv.org/abs/2512.00592)

- HAVEN: introduces a hierarchical navigation framework that integrates a Deep Transformer Q-Network (DTQN) high-level subgoal selector with a low-level potential field controller for safe navigation in partially observable, adversarial environments.
- The DTQN leverages k-step memory and visibility-aware features to learn occlusion- and cover-aware strategies, minimizing exposure to adversarial fields-of-view (FoVs).
- The framework demonstrates direct transfer from 2D training to 3D Unity-ROS environments by projecting point-cloud perception into the same feature schema without architectural changes.

---

[Toward a Safe Internet of Agents](http://arxiv.org/abs/2512.00520)

- Internet of Agents (IoA) Architecture: introduces a foundational guide for engineering safe and reliable agentic systems by deconstructing the ecosystem across three levels of increasing complexity: Single Agent, Multi-Agent System (MAS), and Interoperable Multi-Agent System (IMAS).
- The Single Agent is defined by its Model, Memory, Design Patterns, Tools, and Guardrails; MAS adds collective behavior components like Architectural Patterns and Verification; and IMAS requires Standardized Protocols, Discovery, Vetting, and Governance.
- The analysis emphasizes that agentic safety is an architectural principle, treating each component as a dual-use interface where capability increases are linked to expanded attack surfaces.

---

[Smart-TCP: An Agentic AI-based Autonomous and Adaptive TCP Protocol](http://arxiv.org/abs/2512.00491)

- Smart-TCP: introduces an agentic AI-based autonomous TCP protocol that reframes TCP's core logic as an LLM-driven agent, integrating logical reasoning with deterministic computation via LLM (Logical reasoning), ALU (Deterministic computation), State Module (Internal state storage), Context Aggregation Mechanism (Synthesizes protocol context), and Dual-Agent Interaction Framework (Client/Server interaction).
- The architecture employs a dual-agent interaction framework where the LLM serves as the cognitive core and an Arithmetic Logic Unit (ALU) acts as a specialized tool for precise 32-bit arithmetic operations, such as sequence and acknowledgment number calculation.
- This design overcomes the arithmetic limitations of pure LLM protocol implementations by decoupling LLM reasoning from deterministic ALU computation, achieving high accuracy in end-to-end sessions.

---

[SelfAI: Building a Self-Training AI System with LLM Agents](http://arxiv.org/abs/2512.00403)

- SelfAI: introduces a unified multi-agent self-training pipeline for autonomous scientific discovery, integrating the User Agent (Translates objectives to configurations), Cognitive Agent (LLM-powered reasoning/planning/stopping), and Experiment Manager (Orchestrates parallel training/resource management).
- The Cognitive Agent utilizes LLMs and optimal stopping criteria to iteratively refine hyperparameter searches and adapt the search trajectory based on accumulated experimental evidence.
- The system introduces two novel evaluation metrics, Score and AUPD, to quantify discovery efficiency and search diversity across diverse scientific domains.

---

[Provable Memory Efficient Self-Play Algorithm for Model-Free Reinforcement Learning](http://arxiv.org/abs/2512.00351)

- ME-Nash-QL (Memory-Efficient Nash Q-Learning): introduces a model-free self-play algorithm for two-player zero-sum Markov games, integrating reference-advantage decomposition and an early-settlement approach.
- The algorithm achieves minimal space complexity $O(SABH)$ and near-optimal sample complexity $O(H^4SAB/\epsilon^2)$ for finding an $\epsilon$-approximate Nash Equilibrium.
- ME-Nash-QL utilizes UCB/LCB exploration strategies and Coarse Correlated Equilibrium (CCE) computation to ensure low computational complexity and output a single Markov and Nash policy.


---

[Design and Evaluation of a Multi-Agent Perception System for Autonomous Flying Networks](http://arxiv.org/abs/2512.00259)

- MAPS (Multi-Agent Perception System): introduces a modular and scalable perception framework for Autonomous Flying Networks (FNs) that leverages MM-LLMs and Agentic AI to generate structured Service Level Specifications (SLSs).
- The system processes multimodal inputs (visual and audio data from UAVs) through Perception, Brain, and Action layers to estimate user count, spatial distribution, and traffic demand.
- MAPS operationalizes the perception layer required by zero-touch network management frameworks (ETSI ZSM, ITU Autonomous Networks) to enable autonomous FN decision-making.

---

[Exact Decentralized Optimization via Explicit $l_1$ Consensus Penalties](http://arxiv.org/abs/2512.00268)

- DP2G (Decentralized Primal-Dual Proximal Gradient): introduces a modular two-layer framework coupling an outer penalty-continuation loop with an inner plug-and-play saddle-point solver, using explicit $l_1$ consensus penalties.- The algorithm achieves exact consensus with fixed stepsizes while maintaining a minimal memory footprint of one primal and one dual vector per agent, comparable to classical decentralized gradient descent (DGD).
- Leveraging the Kurdyka-Łojasiewicz property, the framework proves global convergence, vanishing disagreement, and linear rates for strongly convex objectives under any admissible inner solver.
- Leveraging the Kurdyka-Łojasiewicz property, the framework proves global convergence, vanishing disagreement, and linear rates for strongly convex objectives under any admissible inner solver.

---

#### 28th November 2025

[Towards Continuous Intelligence Growth: Self-Training, Continual Learning, and Dual-Scale Memory in SuperIntelliAgent](http://arxiv.org/abs/2511.23436)

- SuperIntelliAgent: introduces an agentic learning framework coupling a trainable small diffusion model (Learner) with a frozen LLM (Verifier) to enable continual intelligence growth through self-supervised interaction.
- The system autonomously generates chosen/rejected pairs for Direct Preference Optimization (DPO) by having the Learner generate outputs and the Verifier evaluate them via step-by-step reasoning.
- The architecture integrates a dual-scale memory mechanism, using a replay buffer for short-term experience traces and on-the-fly LoRA fine-tuning for long-term knowledge consolidation.

---

[AREA3D: Active Reconstruction Agent with Unified Feed-Forward 3D Perception and Vision-Language Guidance](http://arxiv.org/abs/2512.05131)

- AREA3D (Active Reconstruction Agent): introduces an active 3D reconstruction agent that unifies feed-forward 3D perception and VLM guidance into a dual-field framework for efficient, uncertainty-aware viewpoint selection.
- The framework leverages a Feed-Forward 3D Perception module for geometric uncertainty estimation and a VLM for high-level semantic guidance, fusing both into a unified 3D uncertainty field.
- The Active View Selection Strategy uses this unified field, along with visibility gates and frustum-based decay, to select informative viewpoints under tight budgets, achieving state-of-the-art accuracy in sparse views.

---

#### 27th November 2025

[Matrix: Peer-to-Peer Multi-Agent Synthetic Data Generation Framework](http://arxiv.org/abs/2511.21686)

- Matrix: introduces a decentralized peer-to-peer multi-agent framework for scalable synthetic data generation, utilizing serialized Orchestrator messages for control and state flow, processed by stateless AgentActors, and supported by Distributed Services for heavy computation.
- The architecture eliminates centralized orchestration bottlenecks and achieves high throughput by implementing fine-grained, asynchronous row-level scheduling across distributed queues, enabling tens of thousands of concurrent workflows.
- The framework leverages open-source tools like Ray, SLURM, vLLM, and Apptainer for cluster management and distributed execution, demonstrating 2-15x higher data generation throughput than centralized baselines.

---

[Agentic AI Framework for Cloudburst Prediction and Coordinated Response](http://arxiv.org/abs/2511.22767)

- AIF-AWCI (Agentic AI Framework for Atmospheric Water-Cycle Intelligence): introduces a multi-agent architecture that integrates sensing, forecasting, downscaling, hydrological modeling, and coordinated response into a closed-loop system.
- The framework utilizes autonomous but cooperative agents across Perception, Decision, and Action layers to transform atmospheric data into real-time decision intelligence.
- Empirical evaluation demonstrated that the multi-agent configuration enhances forecast reliability, critical success index, and warning lead time compared to baseline models.

---

[Agentic AI Framework for Individuals with Disabilities and Neurodivergence: A Multi-Agent System for Healthy Eating, Daily Routines, and Inclusive Well-Being](http://arxiv.org/abs/2511.22737)

- AAS (Agentic AI System): introduces a multi-agent framework designed to assist individuals with disabilities and neurodivergence in healthy eating and daily routines.
- The system utilizes four specialized agents—Meal Planner, Reminder, Food Guidance, and Monitoring—coordinated by a Hybrid Reasoning Engine via a Blackboard/Event Bus.
- The framework emphasizes personalization, accessibility, and transparency through multimodal interfaces, adaptive learning (RL), and privacy-conscious data integration.

---

[Exposing Vulnerabilities in RL: A Novel Stealthy Backdoor Attack through Reward Poisoning](http://arxiv.org/abs/2511.22415)

- BABO: introduces a novel stealthy backdoor attack that manipulates an RL agent's policy by poisoning its reward signals, formulated via a penalty-based bi-level optimization problem.
- The attack minimizes data distortion using the Reward Perturbation Network ($\Delta$) while ensuring the agent learns the Target Backdoor Policy ($\pi^\dagger$) under black-box constraints.
- The method achieves high stealthiness with minimal performance drop under normal conditions, yet causes catastrophic performance decline (up to 85.01%) when a trigger is activated.

---

[Distributed Koopman Operator Learning for Perception and Safe Navigation](http://arxiv.org/abs/2511.22368)

- DKOL-MPC: introduces a unified, scalable framework for predictive and safe autonomous navigation by integrating Model Predictive Control with Distributed Koopman Operator Learning.
- The framework uses a consensus-based distributed learning algorithm where multiple computational nodes collaboratively estimate the Koopman operator from high-dimensional sensory data without centralized data aggregation.
- The learned operator forecasts future obstacle spatial densities, which are converted into convex polytopic linear constraints embedded in the MPC formulation to guarantee collision-free navigation.

---

[CO-EVOLVING AGENTS: LEARNING FROM FAILURES AS HARD NEGATIVE](http://arxiv.org/abs/2511.22254)

- Co-Evolving Agents Framework: introduces a self-improving agent architecture where a Target Agent and an auxiliary Failure Agent jointly improve through mutual interaction and alternating training phases.
- The Failure Agent specializes in preference optimization over failure trajectories to autonomously generate informative Hard Negatives, which are high-reward failures close to success.
- Incorporating these structured Hard Negatives into the Target Agent's DPO optimization sharpens decision boundaries and significantly enhances LLM generalization across diverse tasks.

---

[MTR-VP: Towards End-to-End Trajectory Planning through Context-Driven Image Encoding and Multiple Trajectory Prediction](http://arxiv.org/abs/2511.22181)

- MTR-VP (Motion Transformer for Vision-based Planning): introduces an end-to-end trajectory planning method using a two-stage architecture comprising a Scene Context Encoder and a Scene Context Decoder, which outputs K possible future trajectories and their probability distribution.
- The Scene Context Encoder leverages a Pretrained ViT for image encoding and a State Encoder (temporal transformer) for past kinematic states, concatenating them to form scene context embeddings, replacing map-based features.
- The approach adapts the MTR framework to a vision-first context, utilizing cross-attention to fuse the encoded intent with the learned scene context, and predicting multiple futures to boost planning performance in long-tail scenarios.

---

[TinyLLM: Evaluation and Optimization of Small Language Models for Agentic Tasks on Edge Devices](http://arxiv.org/abs/2511.22138)

- TinyLLM: introduces a pipeline for optimizing SLMs for edge agentic tasks, utilizing Data Processing and a Data Preparation Pipeline to convert AgentBank SFT Dataset into the AgentBank Chosen-Rejected Dataset for the DPO Training Pipeline, resulting in a Finetuned SLM evaluated against the BFCL Framework using Performance Metrics and various Optimization Strategies.
- The approach focuses on preference alignment via Direct Preference Optimization (DPO) to efficiently align SLMs (under 3B parameters) for robust function/tool calling without relying on costly cloud infrastructure.
- Benchmarking across diverse scenarios revealed that medium-scale SLMs (1-3B parameters) significantly outperform ultra-compact models, achieving high overall and multi-turn accuracy through hybrid optimization.

---

#### 26th November 2025

[Model-Based Policy Adaptation for Closed-Loop End-to-End Autonomous Driving](http://arxiv.org/abs/2511.21584)

- MPA (Model-Based Policy Adaptation): introduces a general framework for end-to-end autonomous driving that enhances robustness and safety by adapting a pretrained E2E agent using counterfactual data. 
- The approach generates diverse counterfactual trajectories via a geometry-consistent 3DGS-based simulation engine to expose the agent to scenarios beyond the original dataset. 
- MPA trains a diffusion-based policy adapter to refine base policy predictions and a multi-step Q-value model to evaluate long-term outcomes for inference-time guidance.

---

[BAMAS: Structuring Budget-Aware Multi-Agent Systems](http://arxiv.org/abs/2511.21572)

- BAMAS (Budget-Aware Multi-Agent Systems): introduces a novel framework for constructing multi-agent systems under budget constraints, including Budget-Constrained LLM Provisioning, Agent Collaboration Topology Selection, and Agent Instantiation.
- The framework jointly optimizes LLM selection using an Integer Linear Programming Solver and agent collaboration topology using a Topo-Selection Policy trained via offline reinforcement learning.
- BAMAS achieves a strong cost-performance trade-off by adaptively selecting LLMs and collaboration patterns (Topo Set) to maximize task performance within a fixed cost budget.

---

[Tool-RoCo: An Agent-as-Tool Self-organization Large Language Model Benchmark in Multi-robot Cooperation](http://arxiv.org/abs/2511.21510)

- Tool-RoCo: introduces a novel LLM-based multi-agent benchmark for multi-robot cooperation, leveraging the agent-as-tool concept and four progressive cooperation paradigms.
- The framework evaluates LLM autonomy and coordination using tool usage metrics, including Cooperative Tool Ratio (CT) and Self-Organization Ratio (SO).
- Tool-RoCo utilizes three multi-robot tasks (CABINET, PACK, SORT) and two types of tools (Common and Cooperative) to systematically assess LLM performance across varying levels of centralized and decentralized control.

---

[EWE: AN AGENTIC FRAMEWORK FOR EXTREME WEATHER ANALYSIS](http://arxiv.org/abs/2511.21444)

- EWE (Extreme Weather Expert): introduces an intelligent agent framework for extreme weather analysis, integrating Knowledge-Enhanced Planning, Self-Evolving Closed-Loop Reasoning, and a Meteorological Toolkit.
- The framework operationalizes expert workflows using an MLLM reasoning backbone to autonomously generate and interpret multimodal visualizations from raw meteorological data.
- Self-Evolving Closed-Loop Reasoning employs a Dual-Auditor Module (Code Auditor and Content Auditor) to verify both operational success and physical plausibility of generated code and visualizations.

---

[Large Language Models for Unit Test Generation: Achievements, Challenges, and the Road Ahead](http://arxiv.org/abs/2511.21382)

- UFW-UTG (Unified Framework for LLM-based Unit Test Generation): introduces a systematic engineering view of LLM-based unit test generation, including Model Preparation (Specializes LLM), Context Enrichment (Constructs context-rich prompt), Prompt-driven Generation (Core LLM operation), Raw Generated Tests (Initial LLM output), Quality Assurance Loop (Validates and refines tests), Final Test Suite (Executable, high-quality tests), Synergy (Integrates traditional SE tools), and Feedback Loop (Iterative refinement mechanism), where the framework treats LLMs as stochastic generators requiring systematic engineering constraints.
- The analysis reveals that prompt engineering is the dominant strategy (89%), and the iterative validation and repair loop is the standard mechanism for ensuring robust test usability, boosting pass rates from under 30% to over 70%.
- Future research emphasizes a paradigm shift toward autonomous testing agents and hybrid systems that combine LLMs' semantic understanding with traditional tools' systematic exploration capabilities to improve fault detection.

---

[Dual-Agent Reinforcement Learning for Adaptive and Cost-Aware Visual-Inertial Odometry](http://arxiv.org/abs/2511.21083)

- Dual-Agent Reinforcement Learning for Adaptive and Cost-Aware Visual-Inertial Odometry: introduces a decoupled RL-based VIO framework that mitigates the Visual-Inertial Bundle Adjustment (VIBA) bottleneck using a Select Agent (RL computational scheduler) and a composite Fusion Agent (Composite RL fusion policy).
- The Select Agent uses IMU-only data to decide whether to run the costly Visual Odometry pipeline, achieving significant computational savings by skipping redundant or uninformative frames.
- The Fusion Agent adaptively fuses high-rate IMU predictions with sparse VO updates by learning context-dependent weights, resulting in a favorable accuracy-throughput-memory trade-off compared to prior GPU-based VIO systems.

---

[EVILGENIE: A Reward Hacking Benchmark](http://arxiv.org/abs/2511.21654)

- EVILGENIE: introduces a benchmark for reward hacking in programming settings using problems sourced from LIVECODEBENCH, designed to allow agents to circumvent test cases.
- The benchmark evaluates agent behavior using a combination of held-out unit tests, LLM judges for solution classification, and automated test file edit detection.
- Evaluation across proprietary and standardized LLM agents reveals that LLM judges are highly effective at detection, while held-out tests show minimal improvement in unambiguous cases.

---

[Aligning LLMs Toward Multi-Turn Conversational Outcomes Using Iterative PPO](http://arxiv.org/abs/2511.21638)

- Iterative PPO: introduces a batch online policy iteration algorithm that reduces the multi-turn RL problem into a sequence of single-turn RLHF problems using a learned Q-function as the reward model.
- The approach alternates between collecting multi-turn trajectories and performing policy improvement using standard token-level PPO, leveraging existing stable single-turn RLHF tools.
- This method enables continual learning from real customer-business interactions without requiring an environment simulator, balancing online adaptability with offline stability.

---

[MADRA: Multi-Agent Debate for Risk-Aware Embodied Planning](http://arxiv.org/abs/2511.21460)

- MADRA (Multi-Agent Debate for Risk-Aware Embodied Planning): introduces a training-free Multi-Agent Debate Risk Assessment framework leveraging collective reasoning to enhance safety awareness in embodied planning without sacrificing task performance.
- MADRA employs multiple LLM-based Risk Assessment Agents guided by a Critical Evaluator to iteratively debate instruction safety and vote for consensus, curbing single-LLM bias and reducing false rejections.
- The MADRA module is integrated into a Hierarchical Cognitive Collaborative Planning Framework that includes Memory Enhancement, Hierarchical Planning, and a Self-evolution Mechanism for continuous learning and improved task success rates.
- The MADRA module is integrated into a Hierarchical Cognitive Collaborative Planning Framework that includes Memory Enhancement, Hierarchical Planning, and a Self-evolution Mechanism for continuous learning and improved task success rates.

---

[Prune4Web: DOM Tree Pruning Programming for Web Agent](http://arxiv.org/abs/2511.21398)

- Prune4Web (DOM Tree Pruning Programming for Web Agent): introduces a multi-stage framework for web automation, including a Planner (decomposes high-level task), a Programmatic Element Filter (generates Python scoring program), and an Action Grounder (selects final executable action).
- The core innovation, DOM Tree Pruning Programming, transforms DOM processing from LLM-based filtering to programmatic pruning, reducing candidate elements by 25-50 times.
- The approach uses LLMs to generate executable Python scoring programs based on semantic clues from decomposed sub-tasks, enabling precise action localization without attention dilution.

---

[Multi-Agent Systems for Dataset Adaptation in Software Engineering: Capabilities, Limitations, and Future Directions](http://arxiv.org/abs/2511.21380)

- MADAP (Multi-Agent Dataset Adaptation Pipeline): introduces an empirical study evaluating LLM-based multi-agent systems, specifically GitHub Copilot, on dataset adaptation tasks using a structured five-stage evaluation pipeline.
- The pipeline assesses agent performance across file comprehension, code editing, command generation, validation, and final execution, revealing that current systems struggle to produce functionally correct implementations.
- Prompt-level interventions, such as providing error messages and reference code, significantly improve structural similarity and highlight the need for robust feedback-driven guidance in future agents.

---

[Aligning LLMs with Biomedical Knowledge using Balanced Fine-Tuning](http://arxiv.org/abs/2511.21075)

- BFT (Balanced Fine-Tuning): introduces an efficient post-training method for aligning LLMs with specialized biomedical knowledge, utilizing token-level weighting (stabilizes gradients) and sample-level reweighting (focuses on hard samples).
- This method operates through a two-layer confidence-based weighting mechanism to learn complex reasoning from sparse data without requiring external reward signals or costly reinforcement learning.
- BFT-based LLMs surpass SFT and other baselines in medical and biological reasoning tasks, generating biologically meaningful embeddings for downstream applications like gene interaction prediction.

---

[OVOD-Agent: A Markov-Bandit Framework for Proactive Visual Reasoning and Self-Evolving Detection](http://arxiv.org/abs/2511.21064)

- OVOD-Agent (Open-Vocabulary Object Detection Agent): introduces a lightweight, LLM-free framework that transforms passive category matching into proactive visual reasoning and self-evolving detection, utilizing an Environment (updates visual state), Detector (outputs region proposals), Weakly Markovian Decision Process (w-MDP) (models visual-semantic transitions), Bandit Sampling Process (UCB-based exploration), Markov State Transition Matrix (stores transition statistics), Reward Model (RM) (guides inference refinement), and Visual Chain-of-Thought (Visual-CoT) Actions (iteratively refine textual representation).
- The framework models visual context transitions as a Weakly Markovian Decision Process (w-MDP) over eight compact visual states, enabling an interpretable multi-step Visual-CoT reasoning process with explicit actions.
- A Bandit module generates exploration signals under limited supervision, and its trajectories are coupled with Markov transition matrices to train a self-supervised Reward Model (RM) for continuous policy improvement.

---

[LOOM: Personalized Learning Informed by Daily LLM Conversations Toward Long-Term Mastery via a Dynamic Learner Memory Graph](http://arxiv.org/abs/2511.21037)

- LOOM: introduces an agentic four-stage pipeline that transforms everyday LLM conversations into personalized learning trajectories using a Dynamic Learner Memory Graph, Chat Summarizer, Topic Decider, Course Generator, and Goals Updater.
- The system unifies continuity and initiative by proactively inferring evolving learner needs from chat summaries and generating adaptive, goal-aligned mini-courses that address identified knowledge gaps.
- The Dynamic Learner Memory Graph tracks mastery, links adjacent concepts, and continuously updates based on user engagement and learning outcomes to guide next steps and reinforcement.

---

[Towards Trustworthy Legal AI through LLM Agents and Formal Reasoning](http://arxiv.org/abs/2511.21033)

- L4M (Legal Logic LLM): introduces a novel neural-symbolic framework combining adversarial LLM agents with SMT-solver-backed proofs to achieve trustworthy, verifiable legal AI.
- The system uses dual LLM agents (Prosecutor and Attorney) for independent, adversarial fact and statute extraction, ensuring role isolation and comprehensive coverage.
- Extracted facts are autoformalized into Z3 assertions, verified by the SMT solver, and refined via an iterative self-critique loop before a Judge LLM verbalizes the final, auditable verdict.

---

[CaptionQA: Is Your Caption as Useful as the Image Itself?](http://arxiv.org/abs/2511.21025)

- CaptionQA: introduces a utility-based caption evaluation benchmark covering four domains (Natural, Document, E-commerce, Embodied AI) using a deterministic QA protocol where a text-only LLM answers taxonomy-grounded multiple-choice questions based solely on the generated caption.
- The benchmark construction pipeline involves human-designed taxonomies, VLM-based question generation, filtering text-answerable questions, deduplication, and dual-VLM quality control to ensure high-density, visually-grounded QA pairs.
- Evaluation using the benchmark reveals substantial utility gaps between image-level and caption-level performance across state-of-the-art MLLMs, especially in Embodied AI and spatial reasoning tasks.

---

#### 25th November 2025

[FRAGMENTA: End-to-end Fragmentation-based Generative Model with Agentic Tuning for Drug Lead Optimization](http://arxiv.org/abs/2511.20510)

- FRAGMENTA (End-to-end Fragmentation-based Generative Model with Agentic Tuning for Drug Lead Optimization): introduces an end-to-end framework for drug lead optimization that integrates the LVSEF generative model and an Agentic AI System for automated tuning, enabling a closed-loop iterative process.
- The LVSEF component reframes fragment selection as a vocabulary selection problem, jointly optimizing fragment sets and molecule generation using dynamic Q-learning and reconstruction rewards.
- The Agentic AI System utilizes specialized LLM-based agents (EvalAgent, QueryAgent, ExtractAgent, CodeAgent) and a shared Knowledge Base to interpret expert feedback and autonomously refine the generative model's objectives.

---

[AD-R1: Closed-Loop Reinforcement Learning for End-to-End Autonomous Driving with Impartial World Models](http://arxiv.org/abs/2511.20325)

- AD-R1: introduces a closed-loop RL framework leveraging an Impartial World Model (IWM) as an internal critic to refine autonomous driving policies by learning from imagined failures.
- The IWM is trained using Counterfactual Synthesis, a novel data pipeline that systematically generates a curriculum of plausible collisions and off-road events to overcome the optimistic bias inherent in standard world models.
- During policy refinement, the IWM predicts 4D future occupancy sequences for candidate actions, enabling the 4D Rewarded Modeling module to provide dense, physically-grounded safety-critical feedback.

---

[CostNav: A Navigation Benchmark for Cost-Aware Evaluation of Embodied Agents](http://arxiv.org/abs/2511.20216)

- CostNav (Micro-Navigation Economic Testbed): introduces a comprehensive benchmark evaluating embodied agents via an Economic Model that translates Simulation Logs (collision, energy, time) into financial metrics, including Pre-Run Costs, Run Costs, Revenue, and Break-Even Analysis.
- The framework uses industry-derived parameters to model the complete economic lifecycle of autonomous navigation systems, revealing that optimizing for task success differs fundamentally from optimizing for economic deployment.
- Initial evaluation of a Learning-Based On-Device baseline shows that maintenance costs, driven by a high collision rate, overwhelmingly dominate operational costs, resulting in negative profit per run.

---

[FROM DATA TO CONCEPTS VIA WIRING DIAGRAMS](http://arxiv.org/abs/2511.20138)

- Hasse Clustering (HC): introduces a method for extracting abstract concepts from sequential data using quasi-skeleton wiring diagrams, involving sequence-to-matrix conversion and categorical constraint analysis.
- The approach leverages the correspondence between quasi-skeleton wiring diagram graphs and Hasse diagrams to generalize individual data points into more abstract, representative concepts.
- HC was successfully applied to time series data from a reinforcement learning agent playing a computer game, correctly identifying the unique or multiple winning strategies.

---

[CLIMATEAGENT: Multi-Agent Orchestration for Complex Climate Data Science Workflows](http://arxiv.org/abs/2511.20109)

- CLIMATEAGENT: introduces an autonomous multi-agent framework that orchestrates complex climate data science workflows by decomposing user questions into executable sub-tasks coordinated by planning and orchestration agents, acquiring data via specialized DATA-AGENTS, and completing analysis and reporting with self-correcting CODING-AGENTs.
- The system employs a three-layer hierarchical architecture with specialized LLM-based agents, persistent contextual coordination via a Persistent Workflow Context, and adaptive self-correction mechanisms to ensure robustness against API variability and runtime errors.
- Evaluated on CLIMATE-AGENT-BENCH-85, the framework achieves 100% task completion and significantly outperforms GPT-5 and Copilot baselines in report quality across six climate domains, demonstrating reliable end-to-end automation.

---

["Are We Done Yet?”: A Vision-Based Judge for Autonomous Task Completion of Computer Use Agents](http://arxiv.org/abs/2511.20067)

- VBFJ (Vision-Based Feedback Judge): introduces an autonomous evaluation and feedback framework utilizing VLMs to assess task completion directly from screenshots and task descriptions for Computer Use Agents (CUAs).
- The framework achieves up to 73% classification accuracy in task success detection and provides an average relative improvement of 27% in the overall task success rate of CUAs.
- The core mechanism involves the VLM providing natural language reasoning as feedback to the CUA, enabling the agent to replan and reattempt the task from its current state.

---

[WaymoQA: A Multi-View Visual Question Answering Dataset for Safety-Critical Reasoning in Autonomous Driving](http://arxiv.org/abs/2511.20022)

- WaymoQA (Multi-View Visual Question Answering Dataset): introduces Safety-Critical Reasoning, a new task leveraging Multi-View Input (Comprehensive scene coverage) and structured into two stages: Stage 1 (Immediate risk resolution) and Stage 2 (Downstream risk mitigation).
- The WaymoQA dataset contains 35,000 human-annotated question-answer pairs covering complex, high-risk driving scenarios across both Video QA (Temporal reasoning) and Image QA (Alternative actions/outcomes) modalities.
- Experiments reveal that existing MLLMs underperform significantly in safety-critical scenarios, but fine-tuning on the dataset substantially improves their reasoning ability, highlighting the need for targeted supervision.

---

[Hierarchical Spatio-Temporal Attention Network with Adaptive Risk-Aware Decision for Forward Collision Warning in Complex Scenarios](http://arxiv.org/abs/2511.19952)

- HSTAN+DRTA (Hierarchical Spatio-Temporal Attention Network + Dynamic Risk Threshold Adjustment): introduces an integrated Forward Collision Warning (FCW) framework combining HSTAN for efficient trajectory prediction and DRTA for adaptive, reliable warning decisions, including SAM (spatial interaction modeling), TAM (temporal dynamics modeling), CQR Module (uncertainty quantification), and DRTA (adaptive decision-making).
- HSTAN uses a decoupled architecture with GAT-MHA for spatial interactions (O(N·K) complexity) and cascaded GRU/MHA units for temporal dynamics, achieving high prediction accuracy and low inference time (12.3 ms).
- The DRTA module transforms predictions into warnings using a physics-informed risk potential function integrating kinematics and road geometry, combined with an adaptive threshold mechanism based on sliding-window traffic statistics.

---

[Towards Edge General Intelligence: Knowledge Distillation for Mobile Agentic AI](http://arxiv.org/abs/2511.19947)

- KD-EGI (Knowledge Distillation for Edge General Intelligence): introduces a comprehensive survey investigating the integration of KD into EGI, positioning it as a key enabler for efficient, communication-aware, and scalable mobile agentic AI.
- The approach leverages KD to compress large Teacher Models into compact Student Models, transferring complex cognitive skills required for the Agentic Loop (Perception, Planning, Action, Memory).
- The survey reviews specialized distillation methods for wireless communication and novel edge architectures (Mamba, RWKV) to bridge the deployment chasm for LLM-powered agents on resource-constrained IoT Edge Systems.

---

[IMPROVED LINEAR-TIME CONSTRUCTION OF MINIMAL DOMINATING SET VIA MOBILE AGENTS](http://arxiv.org/abs/2511.19880)

- LTMDS (Improved Linear-Time Construction of Minimal Dominating Set): introduces two linear-time algorithms for computing a minimal dominating set (mDS) in anonymous graphs using mobile agents, achieving $O(n)$ round complexity.
- The approach leverages an optimal dispersion algorithm to reach a covered configuration, utilizing Seeker Agents for parallel neighborhood probing to assign colors (red for mDS members) in $O(1)$ time per step.
- The methodology simultaneously constructs a spanning tree, performs leader election, and achieves agent gathering, all within the same $O(n)$ time and $O(\log n)$ memory bounds, improving upon prior complexity results.

---

[Distributionally Robust Cascading Risk in Multi-Agent Rendezvous: Extended Analysis of Parameter-Induced Ambiguity](http://arxiv.org/abs/2511.20914)

- DRRF (Distributionally Robust Risk Framework): analyzes the distributionally robust risk of cascading failures in a Multi-Agent Rendezvous System, using the Conditional Distributionally Robust Functional defined over an Ambiguity Set of probability measures derived from the Steady-State Covariance Matrix of the Observables Vector, which captures Systemic Events in the Time-Delayed Linear Consensus Network.
- The framework explicitly incorporates distributional ambiguity arising from bounded uncertainties in system parameters, including diffusion coefficients, time delays, and network edge weights.
- The approach derives a closed-form risk expression and establishes fundamental bounds that relate the distributionally robust cascading risk to network eigenvalues and parameter uncertainty, providing insights for robust network design.

---

[OpenApps: Simulating Environment Variations to Measure UI-Agent Reliability](http://arxiv.org/abs/2511.20766)

- OpenApps: introduces a flexible simulator for systematically evaluating UI-agent reliability across app variations, including the OpenApps environment, State ($s_t$), Observation ($o_t$), Agent, Prompt, Policy ($\pi(a_t | h)$), Action ($a_t$), Reward ($r$), BrowserGym API, Six functional apps, and Configuration (YAML files).
- The system generates thousands of app versions by configuring appearance and content variables via simple YAML files, enabling large-scale, reproducible experiments on modest hardware.
- Evaluation across seven leading multimodal agents (including LLMs like GPT-4o and Claude) demonstrates that reliability fluctuates drastically across app variations, often underestimating failure modes when tested only on fixed app clones.

---

[Learning from Risk: LLM-Guided Generation of Safety-Critical Scenarios with Prior Knowledge](http://arxiv.org/abs/2511.20726)

- LRF (Learning from Risk Framework): introduces a high-fidelity safety-critical scenario generation framework integrating a CVAE-GNN Module (Generates physically consistent base scenarios) with an LLM (Adversarial reasoning engine) for synthesizing diverse, risk-sensitive driving scenarios.
- The CVAE-GNN learns latent traffic structures from real-world trajectories and map data, while the LLM acts as a knowledge-driven controller, interpreting scene semantics and dynamically adjusting optimization objectives.
- The framework utilizes a knowledge-driven loss adaptation mechanism and a Cross-Risk Scenario Distribution Module to ensure generated scenarios are both plausible and risk-sensitive across low-, high-, and long-tail risk regimes.

---

[Learning Multi-Access Point Coordination in Agentic AI Wi-Fi with Large Language Models](http://arxiv.org/abs/2511.20719)

- AAWF (Agentic AI Wi-Fi Framework): introduces a novel multi-LLM-agent system where each AP acts as an autonomous LLM Agent, leveraging its LLM Brain, Short-Term Memory, Long-Term Memory (RAG), Tool Use Module, Prompt Engine, and Coordination Protocol to collaboratively negotiate adaptive Multi-Access Point Coordination (MAPC) strategies.
- The framework utilizes natural language dialogue and a cognitive workflow (evaluation, reflection, action generation) to dynamically navigate the Co-SR/Co-TDMA trade-off, adapting to diverse and dynamic interference scenarios in Wi-Fi networks.
- Simulation results demonstrate that this self-organized agent negotiation significantly outperforms conventional static protocols and AI-driven baselines in terms of throughput and adaptability.

---

[Arcadia: Toward a Full-Lifecycle Framework for Embodied Lifelong Learning](http://arxiv.org/abs/2512.00076)

- Arcadia: introduces a full-lifecycle framework for embodied lifelong learning that tightly couples four stages—Self-Evolving Exploration and Grounding, Generative Scene Reconstruction and Augmentation, Shared Embodied Representation Architecture, and Sim-from-Real Evaluation and Evolution—to form a closed self-improving loop.
- The framework addresses core limitations in embodied AI by enabling continuous real-world data acquisition, generative simulation updates, and shared-representation learning to support lifelong improvement.
- The Sim-from-Real Evaluation and Evolution component integrates structured deployment feedback (Task, Scene, Robot) back into simulation to refine both assets and policies, effectively closing the real-to-sim-to-real loop.

---

#### 24th November 2025

[BEYOND PROTEIN LANGUAGE MODELS: AN AGENTIC LLM FRAMEWORK FOR MECHANISTIC ENZYME DESIGN](http://arxiv.org/abs/2511.19423)

- Genie-CAT: introduces an agentic LLM system that integrates literature-grounded reasoning (RAG), structural analysis, electrostatic potential calculation, and ML-based redox potential modeling to generate mechanistically interpretable protein design hypotheses.
- The system utilizes a ReAct (Reasoning and Acting) pattern within the LLM Agent Core to dynamically select and orchestrate domain-specific tools, bridging symbolic reasoning with quantitative physical modeling.
- Demonstrated using metalloproteins (ferredoxins), the framework autonomously identifies residue modifications near [Fe-S] clusters that affect redox tuning, significantly reducing the time and expertise required for hypothesis generation.

---

[Be My Eyes: Extending Large Language Models to New Modalities Through Multi-Agent Collaboration](http://arxiv.org/abs/2511.19417)

- BEMYEYES: introduces a modular, multi-agent framework that extends LLMs to multimodal reasoning by orchestrating collaboration between a Perceiver Agent (VLM) and a Reasoner Agent (LLM) through multi-turn conversations.
- The Perceiver Agent extracts visual information and communicates detailed descriptions, while the frozen LLM Reasoner Agent applies its extensive knowledge and reasoning capabilities to solve the given task.
- The system utilizes a data synthesis and supervised fine-tuning pipeline to train the perceiver for effective collaboration, enabling text-only LLMs to outperform large proprietary VLMs like GPT-4o on multimodal tasks.

---

[LEARNING ROBUST SOCIAL STRATEGIES WITH LARGE LANGUAGE MODELS](http://arxiv.org/abs/2511.19405)

- AdAlign (Advantage Alignment): introduces a method to train LLM agents to learn robust social strategies in mixed-motive social dilemmas, utilizing LLM Agents, Multi-agent RLOO, LoRA finetuning, an Agent Buffer, and a Social Dilemma Testbed.
- AdAlign adapts an opponent-learning awareness algorithm to fine-tune LLMs, modifying the policy gradient update with a reweighting of action gradients based on the agent's and opponent's advantages, simplified using a group-relative baseline.
- The approach achieves higher collective payoffs and non-exploitability across environments like IPD and the novel Trust and Split, demonstrating robustness against greedy RL-trained opponents, unlike naive MARL which converges to greedy policies.

---


[LLM-Driven Stationarity-Aware Expert Demonstrations for Multi-Agent Reinforcement Learning in Mobile Systems](http://arxiv.org/abs/2511.19368)

- RELED (LLM-Driven Stationarity-Aware Expert Demonstrations for Multi-Agent Reinforcement Learning in Mobile Systems): introduces a scalable MARL framework integrating LLM-driven expert demonstrations with autonomous agent exploration using the Stationarity-Aware Expert Demonstration (SED) and Hybrid Expert-Agent Policy Optimization (HPO) modules.
- The SED module leverages theoretical non-stationarity bounds, quantified by the reward volatility and policy divergence indices, as feedback to iteratively refine LLM-generated instruction sequences for high-quality expert trajectories.
- The HPO module employs a fully decentralized training approach where agents independently optimize a hybrid policy loss function, adaptively balancing learning from expert and self-generated samples via dynamic time warping distance.

---

[MAESTRO: Multi-Agent Environment Shaping through Task and Reward Optimization](http://arxiv.org/abs/2511.19253)

- MAESTRO (Multi-Agent Environment Shaping through Task and Reward Optimization): introduces a generative meta-learning framework that shifts the LLM role from real-time agent to high-level architect, dynamically designing the task and solving guidance for the MARL training process.
- The system operates as a dual-loop optimization problem, integrating a Semantic Curriculum Generator and an Automated Reward Synthesizer to shape the environment for the MADDPG learner backbone.
- By distilling semantic knowledge into executable training scaffolds (tasks and rewards), the framework guides a standard MARL policy, isolating expensive LLM inference from the real-time execution loop.

---

[LLM-Based Agentic Negotiation for 6G: Addressing Uncertainty Neglect and Tail-Event Risk](http://arxiv.org/abs/2511.19175)

- RAAN (Risk-Aware Agentic Negotiation): introduces an unbiased, risk-aware framework for LLM-based agents in 6G network slicing negotiation, utilizing Digital Twins, CVaR, Epistemic Confidence Score, and a Dynamic SLA Target.
- The framework mitigates uncertainty neglect bias by shifting the agent's objective from mean-based reasoning to tail-event risk quantification using CVaR, ensuring robust resource allocation and eliminating SLA violations.
- Agents are compelled to quantify epistemic uncertainty via the confidence score, which dynamically tightens the internal SLA target to prevent decisions based on unreliable Digital Twin predictions.

---

[Reinforcement Learning for Self-Healing Material Systems](http://arxiv.org/abs/2511.18728)

- RLCF (Reinforcement Learning Control Framework): introduces a self-healing material system modeled as a Markov Decision Process, where an RL agent learns optimal policies to balance structural integrity recovery against finite resource consumption.
- The system architecture integrates self-healing material, sensor arrays, and actuators, allowing the RL agent to select discrete (Q-learning, DQN) or continuous (TD3) healing actions based on the observed damage state.
- Comparative evaluation showed that the continuous-action TD3 agent achieved the fastest and most stable material recovery, demonstrating the necessity of fine-grained, proportional actuation in dynamic self-healing applications.

---


[A Multi-Agent LLM Framework for Multi-Domain Low-Resource In-Context NER via Knowledge Retrieval, Disambiguation and Reflective Analysis](http://arxiv.org/abs/2511.19083)

- KDR-Agent (Knowledge Retrieval, Disambiguation, and Reflective Analysis): introduces a novel multi-agent LLM framework for multi-domain low-resource in-context NER, integrating external knowledge retrieval, entity disambiguation, and reflective correction.
- The framework operates in two stages: Knowledge In-context Construction, which builds enriched prompts, and Reflection & Correction, which refines predictions using structured error analysis.
- KDR-Agent reduces reliance on large annotated corpora by using concise natural-language type definitions and a static set of entity-level positive-negative contrastive demonstrations.

---

[Defending Large Language Models Against Jailbreak Exploits with Responsible AI Considerations](http://arxiv.org/abs/2511.18933)

- DLLE (Defending LLMs Against Jailbreak Exploits): introduces a systematic taxonomy of jailbreak defenses and proposes three complementary strategies: PLDF, LBSD, and MetaGPT-DSAD.
- The PLDF uses sanitization, paraphrasing, and adaptive system prompts, while the LBSD applies inference-time vector steering in safety-aware layers to reinforce refusal behavior.
- The MetaGPT-DSAD, employing structured, role-based collaboration among Rephrase, Core LLM, and Judge Agents, achieved full mitigation of jailbreak attempts in experiments.

---

[LLM-Driven Kernel Evolution: Automating Driver Updates in Linux](http://arxiv.org/abs/2511.18924)

- AUTODRIVER (LLM-driven adaptation and validation loop): introduces a closed-loop, LLM-driven system for automating Linux driver maintenance, utilizing the DRIVEBENCH Executable Corpus and Taxonomy for structured input and validation. 
- The system employs a multi-agent architecture, including prompt engineering-, coding-, static analysis-, and patch fix-agents, operating within a Closed-Loop Refinement Cycle guided by compiler diagnostics. 
- Validation integrates a Localization Engine for precise edit scoping, followed by Docker Compilation and Linux QEMU Testing to ensure functional and security consistency across kernel versions. 

---

[KERNELBAND: Boosting LLM-based Kernel Optimization with a Hierarchical and Hardware-aware Multi-armed Bandit](http://arxiv.org/abs/2511.18868)

- KERNELBAND: introduces a novel framework that formulates kernel optimization as a hierarchical multi-armed bandit problem, enabling LLM agents to strategically navigate the optimization space using runtime behavior clustering and profiling-guided strategy selection.
- The approach leverages hardware profiling to identify promising optimization strategies and employs runtime clustering to reduce exploration overhead by sharing insights across similar kernel candidates.
- The core mechanism is a three-term Hierarchical UCB score that balances exploitation, exploration, and hardware guidance, leading to superior performance and efficiency compared to state-of-the-art methods.

---

[Cognitive Alpha Mining via LLM-Driven Code-Based Evolution](http://arxiv.org/abs/2511.18850)

- CogAlpha (Cognitive Alpha Mining Framework): introduces a multi-agent framework combining code-level alpha representation with LLM-driven reasoning and evolutionary search for automated and explainable alpha discovery.
- The framework utilizes a Seven-Level Agent Hierarchy for broad exploration and a Multi-Agent Quality Checker to ensure the validity and economic interpretability of generated alpha codes.
- Thinking Evolution employs LLM-guided mutation and crossover operations to iteratively refine qualified alpha candidates based on financial feedback and predictive metrics.

---

[UNeMo: Collaborative Visual-Language Reasoning and Navigation via a Multimodal World Model](http://arxiv.org/abs/2511.18845)

- UNeMo (Unlock Next Moment): introduces a novel framework for Vision-and-Language Navigation (VLN) that collaboratively optimizes visual state reasoning and navigational decision-making.
- The core architecture includes the Multimodal World Model (MWM) for predicting subsequent visual states and the Hierarchical Prediction-Feedback Navigator (HPFN) for integrating this state reasoning into action selection.
- The MWM uses a CVAE structure with cross-attention to fuse visual features, language instructions, and navigational actions, while HPFN enables dynamic bidirectional promotion between the MWM and navigation policies.

---

[HERMES: Towards Efficient and Verifiable Mathematical Reasoning in LLMs](http://arxiv.org/abs/2511.18760)

- HERMES (Hybrid Agent for Reasoning in Mathematics with NEuro-Symbolic Lean4 verification): introduces a Lean4-driven, multi-modular reasoning agent that uses a Reasoning LLM (Generates informal steps), Translation Module (Formalizes steps), Prover Module (Attempts formal proof/counter-proof), and Feedback Module (Returns verification signals) to interleave informal reasoning with formally verified proof steps.
- The framework performs intermediate formal checking using the Lean4 compiler and Lean4 REPL to prevent reasoning drift and employs a Memory Block (Stores validated proof steps) to maintain proof continuity across long, multi-step reasoning chains.
- By leveraging symbolic-engine-backed correctness signals, the agent significantly improves reasoning accuracy while substantially reducing token usage and computational cost compared to reward-based approaches.

---

[RhinoInsight: Improving Deep Research through Control Mechanisms for Model Behavior and Context](http://arxiv.org/abs/2511.18743)

- RhinoInsight (Deep Research Framework): introduces two control mechanisms, the Verifiable Checklist Module (Supervises model behavior) and the Evidence Audit Module (Organizes context information), to enhance robustness and traceability in deep research tasks.
- The VCM transforms user queries into traceable, verifiable sub-goals via a Checklist Generator and LLM Critic, compiling them into a hierarchical outline to constrain planning and prevent non-executable actions.
- The EAM structures search content, iteratively updates the outline, prunes noisy context, and uses a Critic to rank and bind high-quality evidence to drafted content, ensuring verifiability and reducing hallucinations.

---

[HuggingR⁴: A Progressive Reasoning Framework for Discovering Optimal Model Companions](http://arxiv.org/abs/2511.18715)

- HuggingR⁴: introduces a progressive reasoning framework combining Reasoning, Retrieval, Refinement, and Reflection to efficiently select optimal AI models from large-scale community repositories like HuggingFace.
- The framework uses a coarse-to-fine strategy, starting with iterative reasoning and vector-based retrieval to narrow candidates, followed by fine-grained refinement using a sliding window strategy to manage token consumption.
- The approach attains high workability (92.03%) and reasonability (82.46%) on a new multimodal human-annotated dataset while maintaining constant token consumption regardless of the candidate pool size.

---

[VIL2C: Value-of-Information Aware Low-Latency Communication for Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2511.19146)

- VIL2C (Value-of-Information aware Low-latency Communication): introduces a scheme that proactively adjusts communication latency distribution using VoI-aware resource allocation and a progressive message reception strategy to enhance multi-agent cooperation performance.
- The scheme defines Value of Information (VoI) based on message importance (KL divergence) and communication latency, optimizing bandwidth and power allocation via ResoNet to prioritize high-VoI messages.
- The Progressive Reception module adaptively determines the recipient's waiting time, terminating reception when the uncertainty of the action probability distribution falls below a predefined entropy threshold.

---

[Agent Discovery in Internet of Agents: Challenges and Solutions](http://arxiv.org/abs/2511.19113)

- SDCD (Semantic-Driven Capability Discovery): introduces a novel two-stage capability discovery framework for the Internet of Agents (IoA) that integrates semantic capability modeling, scalable indexing, and memory-enhanced continual discovery.
- The framework addresses challenges in IoA heterogeneity and scalability by using LLMs for semantic profiling and compressing high-dimensional embeddings into compact, updatable agent codes.
- Continual discovery ensures long-term performance in dynamic environments by training the retrieval model with knowledge replay and stability constraints to prevent forgetting of established agents.

---

[HABIT: Human Action Benchmark for Interactive Traffic in CARLA](http://arxiv.org/abs/2511.19109)

- HABIT (Human Action Benchmark for Interactive Traffic): introduces a high-fidelity simulation benchmark integrating 4,730 semantically curated, real-world pedestrian motions into the CARLA simulator for rigorous autonomous driving evaluation.
- The framework utilizes a modular motion retargeting pipeline to convert heterogeneous motion capture and video data into physically consistent, globally aligned SMPL-based trajectories.
- HABIT introduces novel safety metrics, including the Abbreviated Injury Scale (AIS) and False Positive Braking Rate (FPBR), to expose planner weaknesses and safety-conservatism trade-offs hidden in scripted simulations.

---

[Robot-Powered Data Flywheels: Deploying Robots in the Wild for Continual Data Collection and Foundation Model Adaptation](http://arxiv.org/abs/2511.19647)

- RPDF (Robot-Powered Data Flywheel): introduces an iterative framework where a mobile manipulator robot (Scanford) performs useful tasks while autonomously collecting and curating domain-representative data to continually fine-tune a Vision-Language Model (VLM).
- The Scanford system instantiates RPDF by deploying a mobile manipulator equipped with a VLM in a library to scan shelves and identify books, leveraging the library catalog for automated, high-quality data labeling.
- The framework successfully improves VLM performance on domain-specific book identification (32.0% to 71.8%) and domain-adjacent multilingual OCR, while saving an estimated 18.7 hours of human labor during a two-week deployment.

---

[IRSDA: An Agent-Orchestrated Framework for Enterprise Intrusion Response](http://arxiv.org/abs/2511.19644)

- IRSDA (Intrusion Response System Digital Assistant): introduces an agent-orchestrated framework for enterprise intrusion response, combining the MAPE-K loop with Self-Adaptive Autonomic Computing Systems (SA-ACS) for autonomous, policy-compliant cyber defense.
- The architecture uses an $n$-tier design, featuring the IRSDAC (client interface), IRSDAS (server), IRSDAAO (orchestration layer with Agentic Brain), partition-specific IRS Agents, and Tier V components: IRSKG (knowledge graph) and IRSLLM (cybersecurity-tuned LLM).
- The system leverages graph-based RAG to ground the IRSLLM's contextual reasoning and automated responses using real-time enterprise data and dynamic Rules-of-Engagement (ROE), ensuring explainability and policy alignment.

---

[AttackPilot: Autonomous Inference Attacks Against ML Services With LLM-Based Agents](http://arxiv.org/abs/2511.19536)

- AttackPilot: introduces an autonomous multi-agent framework capable of independently conducting inference attacks against ML services, comprising the ControllerAgent (managing and monitoring) and concurrent AttackAgents (executing specific attacks).
- The framework achieves near-expert attack performance and 100.0% task completion using robust LLMs, task-specific action spaces, and a reusable environment.
- Task-specific action spaces guide the AttackAgent through critical steps like selecting shadow datasets and setting hyperparameters, mitigating common LLM errors such as bad plans and context loss.

---

[Agint: Agentic Graph Compilation for Software Engineering Agents](http://arxiv.org/abs/2511.19635)

- Agint (Agentic Graph Compilation for Software Engineering Agents): introduces an agentic graph compiler, interpreter, and runtime that converts natural language instructions into typed, effect-aware code Directed Acyclic Graphs (DAGs) using a six-tier type floor system.
- The system utilizes a composable Unix-style toolchain, including Dagify (DAG compiler) and Dagent (hybrid JIT runtime), unified by the Agilink addressing system for reliable data and tool flow.
- Agint employs Flyte (unified LLM orchestration) integrated with Hydantic (hierarchical structured generation) to enable dynamic graph refinement, parallel compilation, and hybrid execution modes (prefine, dynamic, predict).

---

[DUALGAUGE: Automated Joint Security–Functionality Benchmarking for Secure Code Generation](http://arxiv.org/abs/2511.20709)

- DUALGAUGE: introduces the first fully automated benchmarking framework designed to rigorously evaluate the security and correctness of LLM-generated code in unison, utilizing a Sample Generator, Agentic Executor, LLM Based Evaluator, and Aggregation and Dashboard.
- The system uses the DUALGAUGE-BENCH suite, featuring 154 tasks each paired with dual, coverage-enforced functional and security test suites, to assess LLM performance holistically.
- The Agentic Executor runs generated code in Isolated Containers, resolving runtime issues via an LLM Agent, while the LLM Based Evaluator performs semantic analysis of execution traces for security assessment.

---

#### 23rd November 2025

[FHE-Agent: Automating CKKS Configuration for Practical Encrypted Inference via an LLM-Guided Agentic Framework](http://arxiv.org/abs/2511.18653)

- FHE-Agent: introduces an agentic framework that automates CKKS configuration for encrypted inference by coupling an LLM controller with a deterministic tool suite to decompose the search into global parameter selection and layer-wise bottleneck repair.
- The system operates within a multi-fidelity workflow (Phase A/B/C) that uses cheap static analysis and cleartext simulation to aggressively prune invalid regimes before reserving expensive encrypted evaluations for promising candidates.
- By exposing layerwise profilers and cost models, the framework consistently achieves better precision and lower latency than naive search strategies, successfully finding feasible 128-bit secure configurations for complex models where baseline heuristics fail.

---

[A Synthetic Encyclopedic Dictionary and Semantic Knowledge Graph](http://arxiv.org/abs/2511.18622)

- OpenGloss: introduces a synthetic encyclopedic dictionary and semantic knowledge graph generated by a Multi-Agent Generation Pipeline (Four-stage process) that uses LLM Backends (Configurable foundation models) and Schema Validation (Ensures structured output) to perform Lexeme Selection (Establishes vocabulary foundation), Sense Generation (Generates definitions/relationships), Graph Construction (Extracts explicit semantic edges), and Enrichment (Adds context/history).
- The system produced 537K sense definitions across 150K lexemes and 9.1M semantic edges in under 96 hours for less than $1,000, demonstrating rapid, cost-effective creation of comprehensive lexical resources.
- The resource uniquely integrates lexicographic definitions, encyclopedic context, etymological histories, usage examples, and semantic relationships, addressing gaps in pedagogical applications and general NLP tasks.

---

[From Code Foundation Models to Agents and Applications: A Practical Guide to Code Intelligence](http://arxiv.org/abs/2511.18538)

- Code Intelligence Ecosystem (CIE): introduces a comprehensive synthesis and practical guide to code LLMs, systematically examining the complete model life cycle from data curation to autonomous coding agents.
- The guide analyzes general and code-specialized LLMs, critically examining techniques, design decisions, and trade-offs across pre-training, supervised fine-tuning (SFT), and reinforcement learning (RL) stages.
- Extensive experiments provide data-driven guidelines for compute-efficient pre-training (scaling laws) and calibrated RL recipes for maximizing verifiable correctness in code generation.

---

[LockForge: Automating Paper-to-Code for Logic Locking with Multi-Agent Reasoning LLMs](http://arxiv.org/abs/2511.18531)

- LockForge: introduces, "a multi-agent, multi-stage LLM workflow with role isolation for LL coding and evaluation," which systematically converts Logic Locking (LL) paper descriptions into executable and verified code.
- The pipeline includes Forethoughts, Implementation, and a Refinement Loop driven by Content Mining and Local Execution, orchestrated by LLM-A (Coder) with PDF access.
- Validation relies on independent LLM-B (Judge) and LLM-C (Examiner) agents using a formalized BCSRP Similarity Scoring system and paper-grounded true/false examination to ensure conceptual fidelity.

---

[End-to-End Automated Logging via Multi-Agent Framework](http://arxiv.org/abs/2511.18528)

- AUTOLOGGER (End-to-End Automated Logging via Multi-Agent Framework): introduces a novel hybrid framework addressing the complete logging pipeline, including the neglected whether-to-log decision, using a Judger and a Multi-Agent System.
- The Judger, a fine-tuned binary classifier, efficiently determines logging necessity, acting as a filter before activating the resource-intensive MAS for generation tasks.
- The MAS utilizes specialized Locator and Generator agents, supported by a Tool Pool (including Backward Slicing and Similar Case Retrieval) to ground reasoning in factual code analysis and mitigate LLM hallucination.

---

[Shadows in the Code: Exploring the Risks and Defenses of LLM-based Multi-Agent Software Development Systems](http://arxiv.org/abs/2511.18467)

- IMBIA/Adv-IMBIA Methodology: introduces a security analysis framework for LLM-based Multi-Agent Software Development Systems (Target) using the Implicit Malicious Behavior Injection Attack ($P_m$) and the Adversarial IMBIA defense ($P_{adv}$), where agents (Design/Code/Test) are exploited or protected across two scenarios.
- The IMBIA attack leverages a Malicious Injection Prompt ($P_m$), composed of a Secret Task Summary ($T_s$), Secret Task Descriptions ($T_d$), and Code Instructions ($C_i$), to inject covert malicious behavior into software generated from Benign Software Requirements ($P_b$).
- The Adv-IMBIA defense uses an Adversarial Prompt ($P_{adv}$) integrated either at the user interface or directly into agent profiles to mitigate attacks, revealing that coding and testing phases present the highest security risks across frameworks like ChatDev, MetaGPT, and AgentVerse.

---

[LLMs as Firmware Experts: A Runtime-Grown Tree-of-Agents Framework](http://arxiv.org/abs/2511.18438)

- FIRMHIVE (Recursive Delegation Engine, Proactive Knowledge Hub): introduces a recursive agent hive framework enabling LLMs to act as autonomous firmware security analysts by transforming delegation into an executable primitive and constructing a runtime Tree of Agents (ToA).
- The framework utilizes the Recursive Delegation Engine (RDE) to dynamically decompose complex tasks into structured, parallel workflows aligned with firmware structure, mitigating context fragmentation.
- The Proactive Knowledge Hub (PKH) serves as a persistent global memory, aggregating intermediate results and enabling cross-component dependency resolution and long-term coherence across distributed analyses.

---

[General Agentic Memory Via Deep Research](http://arxiv.org/abs/2511.18423)

- GAM (General Agentic Memory): introduces a novel memory framework based on the Just-in-Time (JIT) compilation principle, featuring a Memorizer for offline history compression and a Researcher for online deep retrieval.
- The Memorizer extracts key information into a lightweight Memory and preserves complete historical information in a Page-store, while the Researcher performs iterative Planning, Searching using multiple tools, and Reflection to generate customized context for client requests.
- The dual-agent architecture leverages LLMs' agentic capabilities and test-time scalability to achieve high-fidelity memory and optimize downstream task completion, significantly outperforming existing Ahead-of-Time memory systems.

---

[Multi-Agent Collaborative Filtering: Orchestrating Users and Items for Agentic Recommendations](http://arxiv.org/abs/2511.18413)

- MACF (Multi-Agent Collaborative Filtering): introduces an agentic recommendation framework that orchestrates User Agents (similar users) and Item Agents (relevant items) via a central Orchestrator Agent across a Multi-Round Discussion.
- The Orchestrator Agent dynamically manages collaboration by issuing Personalized Collaboration Instructions and performing Dynamic Agent Recruitment based on the target user query and interaction history.
- This structure allows the system to aggregate collaborative signals in a structured, adaptive manner, enabling agents to refine candidates and surface agreements or conflicts using shared context and Retrieval Tools.

---

[A Multimodal Conversational Agent for Tabular Data Analysis](http://arxiv.org/abs/2511.18405)

- Talk2Data: introduces a multimodal conversational agent for tabular data analysis that unifies voice/text input with visual, tabular, and spoken outputs via an agentic orchestration loop.
- The system uses an Orchestration/Router component to adaptively select between LLM-driven code generation (executed in a secure sandbox) or direct narrative response (rendered via TTS).
- Grounded prompts inject dataset metadata and conversational memory into the LLM, ensuring context-aware behavior and supporting iterative, multi-turn data exploration.

---

[Path-Constrained Retrieval: A Structural Approach to Reliable LLM Agent Reasoning Through Graph-Scoped Semantic Search](http://arxiv.org/abs/2511.18313)

- PCR (Path-Constrained Retrieval): introduces a novel retrieval method combining structural graph constraints with semantic search to ensure retrieved information maintains logical consistency within a knowledge graph for LLM agents.
- The method restricts the search space to nodes structurally reachable from an anchor node, preventing the retrieval of disconnected information that often leads to inconsistent LLM reasoning chains.
- Evaluated on the PathRAG-6 benchmark, PCR achieved 100% structural consistency, significantly outperforming baseline vector and hybrid retrieval methods while maintaining competitive relevance scores.

---

[Hierarchical Deep Research with Local–Web RAG: Toward Automated System-Level Materials Discovery](http://arxiv.org/abs/2511.18303)

- DToR (Deep Tree of Research): introduces a hierarchical deep research agent for materials and device discovery, integrating local retrieval-augmented generation with LLM reasoners and a Deep Tree of Research mechanism for adaptive exploration.
- The framework treats each Deep Research instance as a Research Node within a tree-structured workflow, using a local-first retrieval policy and gap-driven web expansion to maximize coverage and coherence for S3-S4 level hypotheses.
- DToR consistently outperforms single-instance DR and commercial systems in synthesis quality across 27 nanomaterials/device topics, enabling cost-effective, on-prem integration for complex long-horizon scientific inquiry.

---

[Cross-Disciplinary Knowledge Retrieval and Synthesis: A Compound AI Architecture for Scientific Discovery](http://arxiv.org/abs/2511.18298)

- BioSage (Compound AI Architecture): introduces a novel compound AI architecture that integrates LLMs with RAG, specialized agents, and tools to enable cross-disciplinary scientific discovery and synthesis.
- The system features specialized agents—including retrieval, translation, and reasoning agents—orchestrated via a Query Planning Agent to provide citation-backed, transparent, and traceable responses.
- The architecture utilizes a multi-level memory system (semantic, procedural, episodic) and user-centric design principles to support scientific workflows like summarization, research debate, and brainstorming.

---

[LLM Assisted Coding with Metamorphic Specification Mutation Agent](http://arxiv.org/abs/2511.18249)

- CMA (CodeMetaAgent): introduces an MR-driven LLM-agent framework that systematically refines task specifications and generates semantically constrained test cases, integrating transformation, validation, generation, and execution within a unified pipeline.
- The framework coordinates four core modules—Mutator, Reviewer, Generator, and Evaluator—to operationalize MRs as proactive semantic operators, guiding LLM reasoning for code generation and test case synthesis.
- Experiments show that MR-guided transformations significantly improve code generation accuracy by up to 17% and achieve high test coverage (up to 99.81%) across multiple LLMs and software engineering benchmarks.

---

[Can LLMs Help Allocate Public Health Resources? A Case Study on Childhood Lead Testing](http://arxiv.org/abs/2511.18239)

- PS Framework: introduces a systematic approach for public health resource allocation by integrating Prevalence of elevated BLLs, Percentage of untested children, and Public health coverage ratio, weighted dynamically to rank neighborhoods for intervention.
- The study evaluates state-of-the-art LLMs operating in agentic and deep research modes on a resource allocation task involving distributing 1,000 lead test kits across neighborhoods in Chicago, New York City, and Washington, D.C.
- Evaluation results reveal that LLMs struggle with information retrieval and evidence-based reasoning, frequently overlooking high-priority neighborhoods and allocating disproportionate resources to lower-priority areas.

---

[Energy-Efficient Task Computation at the Edge for Vehicular Services](http://arxiv.org/abs/2511.18449)

- LAPPO/MALAPPO (Multi-Agent Proximal Policy Optimization based Task Computation Strategy): introduces an energy-efficient task computation strategy for V2X services using a decentralized PPO-based algorithm that minimizes total energy consumption while satisfying task latency requirements.
- The strategy operates within a 3-tier MEC architecture, leveraging empirical car mobility analysis to adapt task offloading decisions for both static (LAPPO) and mobile (MALAPPO) vehicular scenarios.
- Evaluation using real-world mobility traces demonstrates that the mobility-aware solution significantly reduces task interruptions and achieves substantial energy savings compared to state-of-the-art schemes.

---

[AutoMAS: A Generic Multi-Agent System for Algorithm Self-Adaptation in Wireless Networks](http://arxiv.org/abs/2511.18414)

- AutoMAS (A Generic Multi-Agent System for Algorithm Self-Adaptation in Wireless Networks): introduces a multi-agent system deployed in a C-RAN architecture that autonomously selects the most suitable wireless optimization algorithm based on dynamic environmental observations.
- The system utilizes a closed-loop cognitive single-agent architecture, where an LLM coordinates observation, reasoning, and action, supported by memory and external tools.
- AutoMAS employs a supervisor-executor mechanism to dynamically select specialized agents from an agent pool and orchestrate their workflow for flexible and efficient task resolution, validated through channel estimation case studies.

---

[Wireless Power Transfer and Intent-Driven Network Optimization in AAVs-assisted IoT for 6G Sustainable Connectivity](http://arxiv.org/abs/2511.18368)

- HDT/DA-MAPPO: introduces an Intent-Driven Framework for Autonomous Network Optimization using the HDT for implicit intent prediction and DA-MAPPO for multi-agent decision-making in AAV-assisted IoT systems.
- HDT replaces conventional floating-point matrix operations with symbolic Hyperdimensional computations to reduce computational and energy overhead for long-context parsing.
- DA-MAPPO employs decoupled networks and cascaded coupling to handle high-dimensional double action spaces (trajectory planning and intent response) while preserving high-order dependencies.

---

[Weakly-supervised Latent Models for Task-specific Visual-Language Control](http://arxiv.org/abs/2511.18319)

- LDM (Latent Dynamics Model): introduces a task-specific latent dynamics model trained with weak goal-state supervision to enable precise visual-language control for object centering in autonomous inspection.
- The model uses separate encoders to map images, instructions, and actions into a shared latent space, where the dynamics model predicts action-induced state shifts toward a goal prototype.
- Training leverages complementary losses, including directional, ranking, consistency, and regularization losses, to stabilize learning and ensure robust spatial grounding, significantly outperforming LLM baselines.

---

#### 22nd November 2025

[INFINIBENCH: INFINITE BENCHMARKING FOR VISUAL SPATIAL REASONING WITH CUSTOMIZABLE SCENE COMPLEXITY](http://arxiv.org/abs/2511.18200)

- InfiniBench: introduces a fully automated, customizable benchmark generator that synthesizes a theoretically infinite variety of complex, physically plausible 3D scenes and renders them into photo-realistic videos for VLM spatial reasoning evaluation.
- The pipeline uses an LLM-based agentic framework for iterative constraint refinement, a cluster-based layout optimizer for dense scene generation, and a task-aware camera trajectory optimization for informative video rendering.
- The system allows parameterized control over compositional, relational, and observational scene complexities, enabling fine-grained diagnostic analysis of VLM successes and failures in spatial reasoning tasks.

---

[Agent-as-a-Graph: Knowledge Graph-Based Tool and Agent Retrieval for LLM Multi-Agent Systems](http://arxiv.org/abs/2511.18194)

- Agent-as-a-Graph Retrieval: introduces a knowledge graph retrieval augmented generation approach that represents tools and their parent agents as co-equal nodes and edges in a knowledge graph to enable unified retrieval.
- The retrieval process involves initial vector search for relevant nodes, followed by type-specific weighted reciprocal rank fusion (wRRF) for reranking, and finally graph traversal to identify the final set of executable parent agents for LLM multi-agent systems.
- By integrating both tool-level specificity and agent-level context, the approach achieves significant improvements in Recall@5 and nDCG@5 metrics over prior state-of-the-art LLM retrievers on the LiveMCPBenchmark.

---

[ARIAL: An Agentic Framework for Document VQA with Precise Answer Localization](http://arxiv.org/abs/2511.18192)

- ARIAL (Agentic Reasoning for Interpretable Answer Localization): introduces a modular framework for Document VQA that orchestrates specialized tools via an LLM-based Planning Agent (LLM-based orchestration) to achieve precise answer extraction and reliable spatial grounding.
- The system decomposes Document VQA into structured subtasks handled by dedicated modules, including OCR (Text and BBox extraction), RAG (Semantic search retrieval), QA (Answer generation), and Grounding (Spatial localization).
- ARIAL achieves state-of-the-art results across four benchmarks by leveraging agentic orchestration to improve both textual accuracy (ANLS) and spatial precision (mAP@IoU), providing transparent reasoning traces.

---

[Rethinking Retrieval: From Traditional Retrieval Augmented Generation to Agentic and Non-Vector Reasoning Systems in the Financial Domain for Large Language Models](http://arxiv.org/abs/2511.18177)

- Financial RAG Architectures (FRA): introduces a systematic evaluation comparing Vector-Based Agentic RAG (Hybrid search and filtering) against Hierarchical Node-Based Reasoning System (Structured document traversal) for financial document Q&amp;A.
- The Vector-Based Agentic RAG achieved a 68% win rate over the hierarchical system with comparable latency (5.2 vs 5.98 seconds) across 1,200 SEC filings.
- Advanced RAG techniques, including Cross-Encoder Reranking and Small-to-Big Retrieval, significantly improved retrieval accuracy and answer quality, demonstrating cost-performance tradeoffs for production.

---

[ASTRA: Agentic Steerability and Risk Assessment Framework](http://arxiv.org/abs/2511.18114)

- ASTRA (Agentic Steerability and Risk Assessment Framework): introduces a first-of-its-kind framework designed to evaluate LLMs' ability to enforce custom guardrails during multi-turn planning and strict tool activation, using LLM, Agent (ReAct paradigm), LangGraph, Scenario Generator, System Prompt, Guardrails, Tool Suite, Jailbreak Techniques, and Automated Statistical Analysis Pipeline.
- The framework simulates 10 diverse autonomous agents with 37 unique tools against novel agentic threats, focusing on security steerability in context-specific operational functions rather than universal threats.
- ASTRA uses simulated tool interactions and sophisticated jailbreak techniques to provide a robust methodology for measuring agentic steerability, revealing that this capability is distinct from general security resistance.

---

[MASTEST: A LLM-Based Multi-Agent System For RESTful API Tests](http://arxiv.org/abs/2511.18038)

- MASTEST (LLM-Based Multi-Agent System For RESTful API Tests): introduces a multi-agent system that automates the entire RESTful API testing workflow, including scenario generation, script generation, execution, and result analysis, using a combination of LLM-based and programmed agents.
- The architecture includes specialized agents like the API Parser, Unit/System Test Scenario Generators, Test Script Generator, and various checkers (Syntax, Data Type, Status Code Coverage) to ensure quality and coverage.
- The system incorporates human testers via a GUI to review and correct LLM-generated artifacts at multiple stages, mitigating LLM hallucination and error accumulation while shifting human focus to quality assurance.

---

[QuickLAP: Quick Language-Action Preference Learning for Autonomous Driving Agents](http://arxiv.org/abs/2511.17855)

- QuickLAP (Quick Language-Action Preference learning): introduces a closed-form Bayesian framework that fuses physical corrections and natural language feedback in real time to infer user preference weights.
- The system uses a dual-LLM architecture, including LM$_{att}$ and LM$_{pref}$, to process free-form utterances into structured reward signals (attention mask, shift, and confidence).
- By treating language as a probabilistic observation over latent preferences, the framework resolves ambiguity inherent in physical corrections and achieves robust online adaptation.

---

[A superpersuasive autonomous policy debating system](http://arxiv.org/abs/2511.17854)

- DeepDebater: introduces a hierarchical multi-agent framework for autonomous policy debating, utilizing specialized LLM agent workflows for iterative retrieval, synthesis, and self-correction against the OpenDebateEvidence corpus.
- The system models the entire competitive policy debate lifecycle, generating complete speech transcripts, cross-examinations, and rebuttals, and rendering them using AI speech and EchoMimic V1 talking-head animation.
- The architecture decomposes complex creative and strategic tasks into discrete, role-based agent workflows, enabling the system to achieve superior argumentative quality and consistently win simulated rounds.

---

[SKILLWRAPPER: GENERATIVE PREDICATE INVENTION FOR SKILL ABSTRACTION](http://arxiv.org/abs/2511.18203)

- SKILLWRAPPER: introduces a principled system for generative predicate invention that leverages foundation models to learn human-interpretable, provably sound, and complete symbolic representations (operators and predicates) of black-box robot skills from RGB image observations.
- The system iteratively performs Active Data Gathering, Predicate Invention (using VLMs to propose and classify predicates), and Operator Learning to construct an abstract transition model usable by off-the-shelf classical planners.
- By focusing on resolving inconsistencies between observed data and the current abstract model, the approach ensures the learned symbolic model is sound and probabilistically complete for long-horizon planning tasks.

---

[Towards Automating Data Access Permissions in AI Agents](http://arxiv.org/abs/2511.17959)

- APMS (Automated Permission Management System): introduces a permission prediction model based on a Hybrid ML Framework that combines LLM-based in-context learning and collaborative filtering to automatically decide data access permissions for AI agents.
- The Hybrid ML Framework achieves 85.1% overall accuracy and 94.4% accuracy for high-confidence predictions by leveraging limited individual permission history and preferences from similar users.
- The system is designed to address the limitations of conventional permission models, which are inadequate for the autonomous execution paradigm of LLM-based AI agents, where permission decisions must often be made at runtime for unseen data types.

---

[Building Browser Agents: Architecture, Security, and Practical Solutions](http://arxiv.org/abs/2511.19477)

- Production Browser Agent Architecture (PBAA): introduces an architecture for reliable and safe browser agents, combining hybrid context management, a robust execution layer, and programmatic safety boundaries enforced by specialization.
- Context management relies on single-snapshot retention, intelligent trimming using a lightweight LLM, and conversation history compression to maintain a stable token budget and reduce operational costs by 57%.
- Safety is achieved through deterministic, code-level constraints like domain allowlisting and action restriction, enabling the agent to reach an 85% success rate on the WebGames benchmark.

---

#### 21st November 2025

[GHOSTEI-BENCH: DO MOBILE AGENTS RESILIENCE TO ENVIRONMENTAL INJECTION IN DYNAMIC ON-DEVICE ENVIRONMENTS?](http://arxiv.org/abs/2510.20333)

- GhostEI-Bench: introduces the first benchmark dedicated to assessing mobile agent robustness against environmental injection attacks in dynamic, executable environments, utilizing a Tested Agent (Perceives/Plans/Acts), an Environment Controller (Prepares/Injects attacks), an Evaluation Module (Assesses agent behavior), a Judge LLM (Analyzes failure trajectory), an Android Emulator (Realistic GUI environment), and Attack Vectors (Threat models).
- The benchmark systematically injects adversarial UI elements, such as deceptive overlays and spoofed notifications, directly into realistic application workflows running inside fully operational Android emulators.
- A novel LLM-based evaluation protocol performs fine-grained failure analysis by reviewing the agent's action trajectory and corresponding screenshots to identify the precise point of failure (perception, recognition, or reasoning).

---

[MDG: Masked Denoising Generation for Multi-Agent Behavior Modeling in Traffic Environments](http://arxiv.org/abs/2511.17496)

- MDG (Masked Denoising Generation): introduces a unified generative framework that reformulates multi-agent behavior modeling as the reconstruction of independently noised spatiotemporal tensors, supporting diverse tasks like open-loop prediction and closed-loop planning.
- The approach utilizes a continuous, per-agent and per-timestep Noise Mask field to regulate localized denoising, enabling efficient and controllable trajectory generation in a single or few forward passes.
- The architecture employs a Scene Encoder to fuse multimodal context and a Transformer Denoiser with specialized attention mechanisms to progressively reconstruct clean trajectories, achieving competitive closed-loop performance on Waymo Sim Agents and nuPlan benchmarks.

---

[Agentifying Agentic AI](http://arxiv.org/abs/2511.17332)

- Agentifying Agentic AI (AAAI): introduces a path toward responsible agency by integrating adaptive, data-driven LLM approaches with structured models from AAMAS, including BDI Architecture (Explicit mental states), Communication Protocols (Structured inter-agent messages), and Norms, Institutions, Roles (Social constraints, expectations).
- The paper argues that true agency requires complementing learning-based mechanisms with explicit models of cognition, cooperation, and governance to ensure transparency, coherence, and accountability in multi-agent settings.
- By reintroducing formal concepts like Mechanism Design and Theory of Mind, the framework aims to address current Agentic AI challenges related to reliability, grounding, long-horizon agency, and robust multi-agent coordination.

---

[Agentic Program Verification](http://arxiv.org/abs/2511.17330)

- AutoRocQ: introduces an LLM agent for program verification that uses Context Analysis, Context-aware Tactic Generation, Proof Tree-aware Interpretation, Context-assisted Feedback Handling, Error Analysis, History Manager, and Proof Certificate to autonomously construct proofs in collaboration with the Rocq Proof Assistant.
- The agent employs an iterative refinement loop, leveraging agentic context search via query commands to retrieve relevant lemmas and definitions on demand, significantly reducing contextual noise compared to static retrieval methods.
- By maintaining a structured proof tree representation, the system achieves high-level interpretation of the proof derivation, enabling strategic decision-making and effective error recovery during complex verification tasks.

---

[Designing Domain-Specific Agents via Hierarchical Task Abstraction Mechanism](http://arxiv.org/abs/2511.17198)

- HTAM (Hierarchical Task Abstraction Mechanism): introduces a novel agent design framework that structures multi-agent systems into a logical hierarchy mirroring the intrinsic task-dependency graph of a specialized domain.
- Instantiated as EarthAgent for complex geospatial analysis, the architecture uses a dual-pass mechanism: top-down planning for decomposition and bottom-up execution for sequential data processing.
- The framework enforces procedural correctness and modularity by decomposing the problem into distinct functional layers, each populated by specialized LLM-driven sub-agents.

---

[AutoLink: Autonomous Schema Exploration and Expansion for Scalable Schema Linking in Text-to-SQL at Scale](http://arxiv.org/abs/2511.17190)

- AutoLink: introduces an autonomous agent framework that reformulates schema linking as an iterative, sequential discovery process, utilizing an LLM policy to dynamically explore and expand the linked schema subset.
- The agent interacts with two specialized environments, the Database Environment ($\mathcal{E}_{DB}$) for SQL exploration and the Schema Vector Store Environment ($\mathcal{E}_{VS}$) for efficient semantic retrieval, without requiring the full database schema input.
- The agent employs a diverse set of actions, including schema exploration, semantic retrieval, and verification, to iteratively refine the linked schema, achieving state-of-the-art strict recall and superior token efficiency.

---

[PathAgent: Toward Interpretable Analysis of Whole-slide Pathology Images via Large Language Model-based Agentic Reasoning](http://arxiv.org/abs/2511.17052)

- PathAgent (Large Language Model-based Agent Framework): introduces a training-free LLM-based agent framework that emulates pathologists' reflective, stepwise analysis by coordinating a Navigator, Perceptor, and Executor for iterative, evidence-driven reasoning on Whole-slide images.
- The Executor, serving as the central module, employs Multi-Step Reasoning and Reflection to dynamically adjust magnification and retrieve new Regions of Interest, generating an explicit chain-of-thought for fully interpretable predictions.
- The framework achieves strong zero-shot generalization and superior accuracy in open-ended and constrained visual question-answering tasks without requiring specific training data.

---

[DETERMINISTIC INFERENCE ACROSS TENSOR PARALLEL SIZES THAT ELIMINATES TRAINING-INFERENCE MISMATCH](http://arxiv.org/abs/2511.17826)

- TBIK (Tree-Based Invariant Kernels): introduces a framework for achieving fully deterministic LLM inference by proposing TP-invariant matrix multiplication and reduction primitives that eliminate the training-inference mismatch.
- The core mechanism involves aligning intra- and inter-GPU reduction orders using a unified hierarchical binary tree structure, ensuring a consistent arithmetic sequence regardless of Tensor Parallel (TP) size or GPU count.
- Integrated into vLLM and FSDP, TBIK, combined with Batch-Invariant Operations (BIO), achieves bit-wise identical results across varying TP configurations and frameworks, crucial for stable on-policy Reinforcement Learning (RL) training.

---

[Episodic Memory in Agentic Frameworks: Suggesting Next Tasks](http://arxiv.org/abs/2511.17775)

- EM Architecture: introduces an episodic memory architecture designed to support workflow completion in agentic frameworks by storing and retrieving past scientific workflows to guide agents in suggesting plausible next tasks.
- The architecture interposes the EM Agent between the chat UI and the domain crew, enabling it to compile execution trajectories into formalized workflows and retrieve similar historical sequences from the Workflow DB.
- The EM Agent leverages an LLM to analyze the retrieved similar workflows against the current workflow, generating context-aware suggestions for subsequent steps, thereby mitigating hallucination risks associated with relying solely on the LLM's pre-trained memory.

---

[M³-Bench: Multi-Modal, Multi-Hop, Multi-Threaded Tool-Using MLLM Agent Benchmark](http://arxiv.org/abs/2511.17729)

- M³-Bench (Multi-Modal, Multiplex, Matching-aware MCP Benchmark): introduces a principled evaluation suite for multimodal tool use under the Model Context Protocol (MCP), featuring an MLLM Executor, MCP Servers, a Judge, and a Similarity-Bucketed Hungarian Alignment module.
- The benchmark targets realistic, multi-hop, and multi-threaded workflows that require visual grounding, textual reasoning, cross-tool dependencies, and persistence of intermediate resources across steps.
- The evaluation pipeline uses a similarity-driven alignment method based on a sentence encoder and Hungarian matching to obtain auditable one-to-one correspondences, decoupling semantic fidelity from workflow consistency.

---

[PersonaAgent with GraphRAG: Community-Aware Knowledge Graphs for Personalized LLM](http://arxiv.org/abs/2511.17467)

- PersonaAgent with GraphRAG: introduces a novel framework for persona-based LLM agents that leverages a Knowledge Graph-enhanced Retrieval-Augmented Generation (GraphRAG) mechanism to ground personalized outputs in both individual and collective knowledge.
- The system integrates a persona prompt encoding user preferences, a knowledge graph capturing personal interactions and community patterns, and a GraphRAG mechanism that retrieves and synthesizes relevant context for generation.
- This approach dynamically generates context-rich prompts by combining user-specific history and global community patterns, significantly improving personalization metrics across news categorization, movie tagging, and product rating tasks.

---

[REMSA: AN LLM AGENT FOR FOUNDATION Model SELECTION IN REMOTE SENSING](http://arxiv.org/abs/2511.17442)

- REMSA (Remote-sensing Model Selection Agent): introduces the first LLM agent for automated Remote Sensing Foundation Model (RSFM) selection, combining structured metadata grounding via RS-FMD (Remote Sensing Foundation Model Database) with a task-driven agentic workflow.
- The modular agent architecture includes an Interpreter, a Task Orchestrator, and specialized Tools (Retrieval, Ranking, Clarification, Explanation) to support complex, constraint-heavy RS scenarios.
- The system leverages in-context learning for ranking and multi-turn clarification to deliver transparent, reproducible selections, outperforming retrieval-only and unstructured RAG baselines on an expert-verified benchmark of 75 RS query scenarios.

---

[Humanlike Multi-user Agent (HUMA): Designing a Deceptively Human AI Facilitator for Group Chats](http://arxiv.org/abs/2511.17315)

- HUMA (Humanlike Multi-user Agent): introduces an LLM-based facilitator for asynchronous group chats using an event-driven architecture with Router (Strategy Selection), Action Agent (Strategy Execution, Timing Simulation), and Reflection (Context Synthesis, Coherence) components.
- The system simulates human-like response timing (50-100 WPM) and handles interruptions by preserving the agent's internal scratchpad and intended actions, enabling natural adaptation to rapid conversation dynamics.
- Evaluation showed that participants could not reliably distinguish the AI facilitator from human community managers, achieving near-chance detection rates and comparable subjective experience scores.

---

[A Simple Yet Strong Baseline for Long-Term Conversational Memory of LLM Agents](http://arxiv.org/abs/2511.17208)

- EMem-G (Event-Centric Memory with Graph Propagation): introduces an event-centric conversational memory representation based on enriched Elementary Discourse Units (EDUs) organized into a heterogeneous graph, supporting associative recall via Personalized PageRank.
- The system uses LLM-based extractors to decompose dialogue into self-contained EDUs and arguments, avoiding lossy compression or fragmentation typical of relation triples.
- Retrieval involves dense similarity search followed by a recall-oriented LLM filter to select relevant EDUs and arguments before graph propagation augments the final QA context.

---

[JIGSAWCOMM: Joint Semantic Feature Encoding and Transmission for Communication-Efficient Cooperative Perception](http://arxiv.org/abs/2511.17843)

- JIGSAWCOMM: introduces a novel communication-efficient Cooperative Perception (CP) framework that jointly trains a Sparse BEV Feature Encoder and a Feature Utility Estimator (FUE) Network to maximize the contribution of every transmitted bit to the final perception task.
- The system uses an end-to-end differentiable Transmission Scheduler and a redundancy-aware top-1-per-cell policy, leveraging exchanged Meta Utility Maps to select only essential, non-redundant features for transmission.
- This approach achieves an asymptotic O(1) communication cost relative to the number of agents, significantly reducing data volume (up to >500x) while maintaining high CP accuracy on OPV2V and DAIR-V2X benchmarks.

---

[Physical Reinforcement Learning](http://arxiv.org/abs/2511.17789)

- CLLN (Contrastive Local Learning Network): introduces a novel analog, distributed system adapted for Q-Learning in reinforcement learning tasks, utilizing self-adjusting nonlinear resistors.
- The network performs gradient descent on a global loss function via a local, contrastive training protocol that compares power dissipation in free and clamped states.
- This physical approach aims to achieve energy efficiency and fault tolerance, features inherent to biological systems but lacking in traditional digital RL hardware.

---

#### 20th Nov 2025

[Large Language Model-Based Reward Design for Deep Reinforcement Learning-Driven Autonomous Cyber Defense](http://arxiv.org/abs/2511.16483)

- LLM-assisted Reward Design: introduces a method using a Large Language Model (LLM), specifically Claude Sonnet 4, to generate context-aware reward structures for Deep Reinforcement Learning (DRL) agents in an autonomous cyber defense simulation environment (Cyberwheel), leveraging Atomic Red Team (ART) and MITRE ATT&CK context.
- The generated reward structures guide the training of DRL-based autonomous defense policies against various heuristic-based attack personas (e.g., aggressive, stealthy) defined using ART and MITRE ATT&CK techniques.
- The study evaluates different blue agent policies (baseline, proactive-v1, proactive-v2) trained with LLM-informed rewards, showing that LLM guidance leads to more effective defense strategies against diverse adversarial behaviors.

---

[DynaMimicGen: A Data Generation Framework for Robot Learning of Dynamic Tasks](http://arxiv.org/abs/2511.16223)

- DynaMimicGen (D-MG): introduces a scalable dataset generation framework that leverages Dynamic Movement Primitives (DMPs) to adapt demonstrations to novel and dynamic environments, producing smooth, realistic, and task-consistent Cartesian trajectories.
- The framework transforms a minimal set of human demonstrations (Dsre) into a large, diverse dataset (Dgen) by segmenting tasks and generalizing motion primitives to new scene configurations, supporting dynamic adaptation during execution.
- This approach significantly reduces the need for extensive human data collection while enabling policy training that generalizes robustly to dynamic task settings unseen in the original demonstrations.

---

[AskDB: An LLM Agent for Natural Language Interaction with Relational Databases](http://arxiv.org/abs/2511.16131)

- AskDB: introduces a novel LLM-powered agent designed to unify data analysis and database administration through natural language, leveraging a ReAct cognitive cycle, Core Safety Protocol, and Dynamic Schema-Aware Prompting.
- The agent utilizes Gemini LLMs and a curated set of tools to autonomously debug SQL, retrieve contextual information, and manage multi-step tasks for both analytical queries and administrative commands.
- AskDB emphasizes Interaction Efficiency and autonomy, achieving strong performance on Spider 1.0 while incorporating safety mechanisms like PII shielding and destructive operation playbooks.

---

[Operon: Incremental Construction of Ragged Data via Named Dimensions](http://arxiv.org/abs/2511.16080)

- Operon: introduces a Rust-based workflow engine that addresses challenges in processing ragged data through a novel formalism of named dimensions with explicit dependency relations, using a statically verified DSL and an automatically generated runtime system.
- The system formalizes dimensional dependencies and uses a structured model for partial data to enable incremental construction of shapes, supporting robust persistence and recovery mechanisms.
- Empirical evaluation shows that Operon significantly outperforms an existing workflow engine (Prefect) in overhead reduction for large-scale data generation pipelines.

---

[Agent0: Unleashing Self-Evolving Agents from Zero Data via Tool-Integrated Reasoning](http://arxiv.org/abs/2511.16043)

- Agent0: introduces a fully autonomous framework that evolves high-performing LLM agents from scratch without external data by combining multi-step co-evolution between a Curriculum Agent and an Executor Agent with seamless external tool integration.
- The framework establishes a symbiotic competition where the Curriculum Agent proposes increasingly challenging, tool-aware tasks based on the Executor Agent's uncertainty, driving a virtuous cycle of capability improvement.
- Empirically, Agent0 significantly boosts reasoning capabilities across mathematical and general benchmarks, demonstrating the effectiveness of tool-augmented, self-driven curriculum generation.

---

[InfCode-C++: Intent-Guided Semantic Retrieval and AST-Structured Search for C++ Issue Resolution](http://arxiv.org/abs/2511.16005)

- INFCODE-C++: introduces an autonomous system for end-to-end C++ issue resolution that combines semantic code-intent retrieval and deterministic AST-structured querying, utilizing a Reproducer Agent, Patch Agent, and Selector Agent.
- The framework addresses C++ complexities like overloaded identifiers and nested namespaces by building an AST-Based Structural Index and a Semantic Code-Intent Index for precise fault localization.
- It achieves a 25.58% resolution rate on MultiSWE-bench-CPP, significantly outperforming prior state-of-the-art Python-oriented agents.

---

[Hiding in the AI Traffic: Abusing MCP for LLM-Powered Agentic Red Teaming](http://arxiv.org/abs/2511.15998)

- Introduces a novel Command & Control (C2) architecture leveraging the Model Context Protocol (MCP) to coordinate distributed, adaptive reconnaissance agents covertly across networks, with components including Reconnaissance Agents, an MCP Coordination Server, and a Red Team Command Agent.
- The decoupled, two-leg C2 communication flow uses the MCP for stealthy tasking and leverages public LLM APIs for complex reasoning and payload generation, blending traffic with legitimate AI service usage.
- This framework enables advanced adversarial capabilities like event-driven operations, multi-agent swarm coordination, and on-demand polymorphic malware generation while minimizing detection footprint.

---

[D-GARA: A Dynamic Benchmarking Framework for GUI Agent Robustness in Real-World Anomalies](http://arxiv.org/abs/2511.16590)

- D-GARA (Dynamic Benchmarking Framework for GUI Agent Robustness in Real-World Anomalies): introduces a dynamic benchmarking framework to evaluate Android GUI agent robustness by integrating an Android simulator, an Execution Cycle, an Anomaly Trigger Mechanism, Interruption Injection, a Success Validation Mechanism, and a DataCollector tool.
- The framework simulates real-world anomalies, such as permission dialogs and system alerts, by injecting them dynamically into the agent's execution trajectory using a rule-based Semantic Anomaly Triggering Mechanism.
- D-GARA utilizes a state-centered Success Validator that checks the final UI state against declarative goal conditions, enabling realistic robustness evaluation beyond static benchmarks.

---

[AutoBackdoor: Automating Backdoor Attacks via LLM Agents](http://arxiv.org/abs/2511.16709)

- AUTOBACKDOOR: introduces a fully automated red-teaming framework for LLMs that uses an autonomous LLM agent to execute the entire backdoor injection pipeline, including trigger generation, poisoned data construction, and model fine-tuning.
- The framework employs a chained agentic workflow and a reflection-guided generation mechanism to synthesize semantically coherent, context-aware triggers and high-quality poisoned instruction-response pairs.
- Experiments show the approach achieves over 90% attack success with minimal poisoned samples across various LLMs and tasks, highlighting the failure of existing defenses against agent-driven semantic backdoors.

---

[Multi-Agent Coordination in Autonomous Vehicle Routing: A Simulation-Based Study of Communication, Memory, and Routing Loops](http://arxiv.org/abs/2511.17656)

- OMM (Object Memory Management): introduces a lightweight mechanism enabling autonomous vehicle agents to retain and share persistent knowledge of encountered obstacles to prevent inefficient path recalculation cycles.
- The system utilizes V2V communication to broadcast minimal obstacle node IDs, which agents use to maintain a distributed blacklist consulted during Modified Dijkstra's path planning.
- OMM-enabled coordination reduces average travel time by 75.7% and wait time by 88% compared to memory-less reactive rerouting systems, which suffer catastrophic performance degradation due to routing loops.

---

[Dialogue Diplomats: An End-to-End Multi-Agent Reinforcement Learning System for Automated Conflict Resolution and Consensus Building](http://arxiv.org/abs/2511.17654)

- Dialogue Diplomats (DD): introduces a novel end-to-end Multi-Agent Reinforcement Learning (MARL) framework for automated conflict resolution, integrating the Hierarchical Consensus Network (HCN), Progressive Negotiation Protocol (PNP), and Context-Aware Reward Shaping.
- The HCN architecture uses graph attention mechanisms and hierarchical reinforcement learning across micro-, meso-, and macro-levels to model complex inter-agent dependencies and strategic planning.
- The system achieves superior performance, reaching 94.2% consensus rates and reducing conflict resolution times by 37.8% compared to baselines, while scaling effectively up to 50 concurrent agents.

---

[MARL-CC: A Mathematical Framework for Multi-Agent Reinforcement Learning in Connected Autonomous Vehicles: Addressing Nonlinearity, Partial Observability, and Credit Assignment for Optimal Control](http://arxiv.org/abs/2511.17653)

- MARL-CC (Multi-Agent Reinforcement Learning with Control Coordination): introduces a unified mathematical framework for cooperative optimal control in Connected Autonomous Vehicles (CAVs), integrating Differential Geometric Control (Nonlinear optimal control), Probabilistic Belief Inference (Partial observability handling), and Shapley-Value Reward Allocation (Credit assignment mechanism).
- The framework employs a Centralized Training, Decentralized Execution paradigm, leveraging belief states derived from Bayesian inference to enable robust, decentralized decision-making under uncertainty and communication delays.
- Theoretical analysis establishes convergence and stability guarantees, demonstrating up to 40% improvement in convergence rate and enhanced cooperative efficiency over baselines in simulation and real-world testbeds.

---

[SWITCH: Benchmarking Modeling and Handling of Tangible Interfaces in Long-horizon Embodied Scenarios](http://arxiv.org/abs/2511.17649)

- SWITCH (Semantic World Interface Tasks for Control & Handling): introduces an embodied, task-driven benchmark evaluating LLMs' ability to perceive, reason, and interact with Tangible Control Interfaces (TCIs) in long-horizon scenarios.
- The benchmark is structured around five complementary tasks—Task-Aware VQA, Semantic UI Comprehension, Action Generation, State Transition Prediction, and Result Verification—covering perception, planning, and verification capabilities using egocentric RGB video input.
- Evaluation results show that current LMMMs exhibit inconsistent performance, often over-relying on textual cues and struggling with fine-grained visual perception and generalization across diverse TCI implementations.

---

[GOAL-DIRECTED SEARCH OUTPERFORMS GOAL-AGNOSTIC MEMORY COMPRESSION IN LONG-CONTEXT MEMORY TASKS](http://arxiv.org/abs/2511.21726)

- SUMER (Search in Uncompressed Memory via Experience Replay): introduces an end-to-end RL agent that learns goal-directed search strategies over raw, uncompressed conversational memory using multi-turn tool interactions, trained via GRPO and verifiable reward.
- The LLM agent utilizes specialized tools, `search_memory` (keyword and semantic search) and `submit_answer`, to gather evidence across temporally distant sessions in the Langmem memory bank.
- By optimizing the search policy for response accuracy, the framework achieves state-of-the-art performance on the LoCoMo long-context conversational QA benchmark, significantly outperforming compression-based memory systems.

---

#### 19th Nov 2025

[Computer-Use Agents as Judges for Generative User Interface](http://arxiv.org/abs/2511.15567)

- Coder-CUA (Coder-Computer-Use Agent) Collaboration framework: introduces a system where a Coder acts as Designer, generating and revising websites, while a CUA acts as Judge, evaluating functionality and refining designs using the AUI-Gym benchmark.
- The framework leverages a Verifier for programmatic task validation and the CUA Dashboard to distill complex CUA navigation trajectories into concise, actionable feedback for the Coder.
- This approach shifts interface design toward agent-native efficiency and reliability, optimizing UIs for agent execution success rather than purely human aesthetics.

---

[Know Your Intent: An Autonomous Multi-Perspective LLM Agent Framework for DeFi User Transaction Intent Mining](http://arxiv.org/abs/2511.15456)

- TIM (Transaction Intent Mining): introduces a novel multi-agent system based on LLMs, employing a self-derived hierarchical agent architecture including a Meta-Level Planner, Perspective-Specific Domain Experts, Question Solvers, and a Cognitive Evaluator, to autonomously infer user intents from complex DeFi transactions.
- The framework integrates multimodal on-chain and off-chain data and critically evaluates findings using a Cognitive Evaluator to ensure accuracy and mitigate LLM hallucinations.
- Experimental results show that TIM significantly outperforms machine learning models, single LLMs, and single-agent baselines across evaluation metrics.

---

[Platform-Agnostic Reinforcement Learning Framework for Safe Exploration of Cluttered Environments with Graph Attention](http://arxiv.org/abs/2511.15358)

- PALF (Platform-Agnostic Reinforcement Learning Framework for Safe Exploration): introduces a hierarchical framework combining a GNN-driven exploration policy ($\pi_{\theta}$) with a safety filter ($\sigma$) to achieve efficient and safe autonomous exploration in cluttered environments.
- The framework utilizes a custom graph representation of the environment, where nodes encode waypoints and frontiers, and the policy is trained using the PPO algorithm with a Safety-Gated Adaptive (SGA) reward function.
- The integration of the GNN policy with an explicit safety mechanism ensures robust decision-making adaptable to real-world robotic platforms.

---

[Octopus: Agentic Multimodal Reasoning with Six-Capability Orchestration](http://arxiv.org/abs/2511.15351)

- Octopus (Agentic Multimodal Reasoning with Six-Capability Orchestration): introduces a new paradigm for multimodal agentic reasoning that autonomously explores reasoning pathways by dynamically selecting and orchestrating six core capabilities, using an MLLM backbone, a code agent, and an observation tool.
- The framework decomposes multimodal reasoning into six fundamental capabilities: Percept, Augment, Spatial, Logic, Transform, and Generation, each supported by corresponding tool modules.
- Octopus achieves state-of-the-art performance on the capability-centric Octopus-Bench, demonstrating the effectiveness of capability orchestration over existing paradigms.

---

[Symmetry-Breaking in Multi-Agent Navigation: Winding Number-Aware MPC with a Learned Topological Strategy](http://arxiv.org/abs/2511.15239)

- WNumMPC (Winding Number-aware MPC): introduces a novel hierarchical navigation method that learns topological cooperative strategies using the winding number via a learning-based Planner and executes them with a model-based Controller to resolve symmetry-induced deadlocks.
- The hierarchical architecture separates high-level strategy acquisition (Planner) from reliable, low-level execution (Controller), combining learning flexibility with model-based reliability.
- The Planner learns target winding numbers and dynamic weights to prioritize interactions, effectively breaking symmetry in dense, multi-agent crossing scenarios.

---

[Modelling and Model-Checking a ROS2 Multi-Robot System using Timed Rebeca](http://arxiv.org/abs/2511.15227)

- Timed Rebeca: introduces an actor-based modelling language with temporal constructs and its model-checking compiler to systematically design and verify multi-robot systems implemented in ROS2, efficiently transforming continuous dynamics into discrete models.
- The approach addresses challenges in multi-robot verification by proposing discretization strategies for data types and introducing optimization techniques to accelerate model-checking time.
- The work demonstrates a bidirectional flow between the abstract Timed Rebeca model and the ROS2 implementation, maintaining semantic consistency through manual validation.

---

[Trustworthy GenAI over 6G: Integrated Applications and Security Frameworks](http://arxiv.org/abs/2511.15206)

- Trustworthy GenAI over 6G Framework: introduces a unified perspective on cross-domain vulnerabilities in GenAI-enabled 6G networks, proposing an Adaptive Evolutionary Defense (AED) concept that co-evolves with attacks through GenAI-driven simulation and feedback.
- The framework integrates Integrated Sensing and Communication (ISAC), Federated Learning (FL), Digital Twins (DTs), Diffusion Models (DMs), and Large Telecommunication Models (LTMs) to address security risks arising from their convergence.
- The AED concept utilizes a Policy Generator, Fitness Evaluator, and Coordinator within a Red-Blue Sandbox environment to ensure system robustness remains above a defined lower-bound threshold against evolving adversaries.

---

[Two-Faced Social Agents: Context Collapse in Role-Conditioned Large Language Models](http://arxiv.org/abs/2511.15573)

- Two-Faced Social Agents: Context Collapse in Role-Conditioned Large Language Models introduces an empirical evaluation of persona fidelity in frontier LLMs (GPT-5, Claude Sonnet 4.5, Gemini 2.5 Flash) across cognitively demanding SAT mathematics items and less constrained Affective Preference Tasks, using socioeconomic personas.
- The study finds that under cognitive load (SAT reasoning), GPT-5 exhibits complete contextual collapse, Gemini 2.5 Flash shows partial collapse, while Claude Sonnet 4.5 retains limited role-specific variation, contrasting with robust variation in preference tasks when cognitive constraints are relaxed.
- This task-dependent collapse suggests optimization pressures drive identity convergence, implying that current alignment paradigms may fundamentally limit the ability of LLMs to sustain contextual selves during complex reasoning.

---

[NAMeGEn: Creative Name Generation via A Novel Agent-based Multiple Personalized Goal Enhancement Framework](http://arxiv.org/abs/2511.15408)

- NAMeGEn (Novel Agent-based Multi-Personalized-Goal Enhancement Framework): introduces a training-free, multi-agent collaborative architecture to address multi-objective flexibility and interpretive complexity in Creative Natural Language Generation (CNLG) tasks like Chinese Baby Naming (NCB), utilizing MOM, MOG, and MOE agents.
- The framework iteratively alternates between information preparation (task analysis, knowledge retrieval) and dynamic optimization (generation, evaluation) to balance Explicit User-specified Objectives (EUOs) and Implicit Interpretive Objectives (IIOs).
- It demonstrates superior performance across various LLM backbones on the CBNames benchmark, achieving high quality and interpretability while mitigating hallucinations through retrieval-augmented generation using the CPoetry corpus.

---

[DEPO: Dual-Efficiency Preference Optimization for LLM Agents](http://arxiv.org/abs/2511.15392)

- DEPO (Dual-Efficiency Preference Optimization): introduces a method that jointly optimizes step-level efficiency (minimizing tokens per step) and trajectory-level efficiency (minimizing steps per task) for LLM agents by extending KTO with an efficiency bonus.
- The method uses offline desirable and undesirable trajectory labels derived from MCTS rollouts and a reward thresholding protocol to guide the LLM agent towards generating concise responses and fewer action steps.
- Experiments on WebShop and BabyAI demonstrate that DEPO significantly reduces token usage and step count while maintaining or improving performance compared to baselines like BC and vanilla KTO.

---

[Cost-Aware Prediction (CAP): An LLM-Enhanced Machine Learning Pipeline and Decision Support System for Heart Failure Mortality Prediction](http://arxiv.org/abs/2511.15357)

- CAP (Cost-Aware Prediction): introduces a three-stage framework integrating an ML classifier outcome, Clinical Impact Projection (CIP) curves, and four Large Language Model (LLM) agents to provide transparent and interpretable decision support for 1-year heart failure mortality prediction.
- The framework utilizes an XGB model trained on EHR data to predict mortality, visualizes trade-offs using CIP curves based on Quality of Life (QoL) and Healthcare System (HC) costs, and employs LLM agents to generate patient-specific cost-benefit analyses.
- The system was evaluated by clinicians, showing high reliability for descriptive agents (I and II) but lower accuracy for speculative guidance (III and IV), emphasizing the strength in risk communication.

---

[OEMA: Ontology-Enhanced Multi-Agent Collaboration Framework for Zero-Shot Clinical Named Entity Recognition](http://arxiv.org/abs/2511.15211)

- OEMA (Ontology-Enhanced Multi-Agent Collaboration Framework): introduces a novel zero-shot clinical Named Entity Recognition (NER) framework based on multi-agent collaboration, consisting of a self-annotator, a discriminator, and a predictor.
- The framework addresses challenges in zero-shot NER by using ontology-guided reasoning for fine-grained example selection and integrating entity-type descriptions with self-generated examples in the prompt.
- The proposed multi-agent design achieves state-of-the-art performance on clinical NER benchmarks, approaching supervised model results in a zero-shot setting.

---

[Taxonomy, Evaluation and Exploitation of IPI-Centric LLM Agent Defense Frameworks](http://arxiv.org/abs/2511.15203)

- SoK (Systematization of Knowledge): introduces a comprehensive taxonomy and evaluation of IPI-centric defense frameworks, classifying them across five dimensions and analyzing six root causes of defense failure, with components including Detection/Prompt Engineering/Fine-tuning/System Design/Runtime Checking/Policy Enforcing/Adaptive Attacks.
- The analysis reveals that System Design and Policy Enforcement frameworks offer the best security against Indirect Prompt Injection (IPI) attacks, while Fine-tuning-based methods show weaker generalization.
- The authors design three novel logic-driven adaptive attacks—Semantic-Masquerading IPI, Cascading IPI, and Isolation-Breach IPI—to exploit architectural flaws in state-of-the-art defenses.

---

[SOLID: a Framework of Synergizing Optimization and LLMs for Intelligent Decision-Making](http://arxiv.org/abs/2511.15202)

- SOLID (Synergizing Optimization and Large Language Models for Intelligent Decision-Making): introduces a novel framework that integrates mathematical optimization with the contextual capabilities of LLMs via iterative collaboration mediated by a Coordinator using dual prices and deviation penalties.
- The framework is inspired by the Alternating Direction Method of Multipliers (ADMM) to ensure convergence guarantees under convexity assumptions while handling structured and unstructured data inputs.
- Empirical results in stock portfolio investment demonstrate that SOLID variants achieve improved annualized returns compared to optimizer-only baselines.

---

[Knowledge-Informed Automatic Feature Extraction via Collaborative Large Language Model Agents](http://arxiv.org/abs/2511.15074)

- Rogue One: introduces a novel, LLM-based multi-agent framework for knowledge-informed automatic feature extraction, operationalizing a decentralized system of three specialized agents—Scientist, Extractor, and Tester—that collaborate iteratively.
- The framework utilizes a rich, qualitative feedback mechanism and a "flooding-pruning" strategy, actively incorporating external knowledge via an integrated Retrieval-Augmented Generation (RAG) system.
- This approach generates features that are statistically powerful, semantically meaningful, and interpretable, significantly outperforming state-of-the-art methods on classification and regression tasks.

---

[Beyond GeneGPT: A Multi-Agent Architecture with Open-Source LLMs for Enhanced Genomic Question Answering](http://arxiv.org/abs/2511.15061)

- OpenBioLLM: introduces a modular multi-agent framework that extends GeneGPT by using open-source LLMs (like Qwen2.5) for genomic question answering, featuring specialized agents for tool routing, query generation, and response validation.
- The architecture decomposes the workflow into specialized agents, which improves interpretability, traceability, and efficiency compared to the monolithic GeneGPT design.
- OpenBioLLM achieves competitive or superior performance on GeneTuring and GeneHop benchmarks while significantly reducing latency compared to monolithic LLM setups.

---

[ACCELOPT: A SELF-IMPROVING LLM AGENTIC SYSTEM FOR AI ACCELERATOR KERNEL OPTIMIZATION](http://arxiv.org/abs/2511.15915)

- AccelOpt: a self-improving LLM agentic system, introduces an iterative kernel optimization framework combining beam search with an optimization memory, guided by a three-component agentic workflow (planner, executor, summarizer).
- The system autonomously optimizes kernels for AWS Trainium accelerators using open-source LLMs, achieving performance comparable to proprietary models while being significantly cheaper.
- Evaluation on the custom NKIBench benchmark demonstrates that the memory accumulation enables progressive improvement and cost-effective discovery of both local and non-trivial global optimizations.

---

[Normative active inference: A numerical proof of principle for a computational and economic legal analytic approach to AI governance](http://arxiv.org/abs/2511.19334)

- NAIF (Normative Active Inference Framework): introduces a computational model grounded in AIF and Economic Legal Analysis (ELA) to enable AI agents to achieve lawful and norm-sensitive behavior through "regulation by design," with all components, where the model simulates an autonomous driving agent resolving conflicting legal imperatives.
- The framework utilizes Context Dependent Preference Tensors (C) to formalize how AIF implements context-dependent preferences, allowing the agent's preference ranking over outcomes to shift based on latent legal or emergency context states (F2, F3).
- The Policy Precision ($\gamma$) component tracks the agent's confidence over its selected policy, serving as a "safety valve" mechanism that promotes vigilance under ambiguous normative contexts and confident action when higher-order norms apply.

---

[Smart Manufacturing: MLOps-Enabled Event-Driven Architecture for Enhanced Control in Steel Production](http://arxiv.org/abs/2511.17632)

- DT-EDMA-DRL: introduces an MLOps-enabled event-driven architecture integrating a Digital Twin (Virtual furnace model), EDMA (Real-time data processing), and a DRL Agent (Optimizes control decisions) to enhance control in steel production.
- The system uses a microservices edge-compute platform to ingest real-time sensor data from PLCs via an OPC-UA server and Kafka Message Broker, ensuring low-latency control loops for induction furnace optimization.
- The DRL agent learns optimal power settings by interacting with the DT environment, aiming to reduce manufacturing waste and improve operational quality in complex industrial settings.

---

#### 18th November 2025

[Discovering autonomous quantum error correction via deep reinforcement learning](http://arxiv.org/abs/2511.12482)

- AQEC (Autonomous Quantum Error Correction): introduces a new Bosonic AQEC code discovered using a Curriculum Learning (CL)-enhanced Deep Reinforcement Learning (DRL) framework, which incorporates higher-order photon losses and adapts to large Fock spaces, utilizing an Analytical Master Equation Solver to accelerate training.
- The discovered Generalized RL (GRL) code exhibits superior robustness against both single-photon and double-photon losses compared to existing codes by converting a catastrophic logical-flip error into a manageable dephasing error.
- The analytical solver significantly reduces computational complexity compared to conventional numerical methods like QuTip, enabling faster exploration of optimal encoding strategies.

---

[Large Language Models and 3D Vision for Intelligent Robotic Perception and Autonomy](http://arxiv.org/abs/2511.11777)

- LLMs (Large Language Models) and 3D Vision Integration: reviews the state-of-the-art methodologies, applications, and challenges at the intersection of LLMs and 3D vision for next-generation robotic sensing technologies, covering components like Transformer Architecture, Object Grounding, Scene Understanding, Text-to-3D Generation, and Embodied Agents.
- The convergence of LLMs and 3D vision enables machines to perceive, reason, and interact with complex environments using natural language and spatial understanding, bridging the gap between linguistic intelligence and spatial perception.
- The review catalogs benchmark datasets and evaluation metrics, and identifies future research directions focusing on adaptive architectures, cross-modal alignment, and real-time processing for context-aware robotic sensing systems.

---

[Requirements for Aligned, Dynamic Resolution of Conflicts in Operational Constraints](http://arxiv.org/abs/2511.10952)

- OAMNCC (Online, Aligned Mitigation of Novel Constraint Conflicts): introduces a knowledge-level analysis characterizing requirements for agent decision making when facing novel constraint conflicts in operational environments, by enumerating conflict types and required agent knowledge.
- The paper uses scenario analysis (Sailor Overboard, Piracy Interdiction, Merchants with Water Cannons, Piracy and vessel adrift) to ground the abstract knowledge requirements necessary for aligned, dynamic conflict resolution.
- The analysis culminates in a taxonomy of required knowledge types, including World Knowledge, Metaknowledge, and Mitigation Utility, mapped onto a five-step conflict mitigation process (Algorithm 2).

---

[CTRL-ALT-DECEIT: Sabotage Evaluations for Automated AI R&D](http://arxiv.org/abs/2511.09904)

- CTRL-ALT-DECEIT: introduces an evaluation framework, MLE-Sabotage, extending MLE-Bench with code-sabotage and sandbagging tasks, using Inspect framework, ReAct agent, AIDE agent, and LM monitors, to assess AI agents' capabilities to act against user interests during ML engineering.
- The research evaluates frontier LLM agents' ability to implant backdoors, cause generalization failures (code-sabotage), and strategically underperform (sandbagging) while attempting to evade detection by automated LM monitors.
- Results indicate agents make meaningful progress on sabotage tasks, but detecting sandbagging is more difficult than code-sabotage, and monitor performance degrades when agents are aware of monitoring.

---

[AutoTool: Efficient Tool Selection for Large Language Model Agents](http://arxiv.org/abs/2511.14650)

- AutoTool: introduces a novel graph-based framework that bypasses repeated LLM inference for tool selection by exploiting tool usage inertia, using an Inertia Sensing module and a Parameter Filling module guided by a Tool Inertia Graph (TIG).
- The TIG captures sequential dependencies via Tool Sequence Edges and data flow via Parameter Dependency Edges, enabling efficient, inertia-aware tool and parameter selection.
- Experimental results show that this approach substantially reduces LLM call counts and token consumption (up to 30% reduction) while maintaining competitive task completion rates across diverse agent tasks.

---

[ReflexGrad: Three-Way Synergistic Architecture for Zero-Shot Generalization in LLM Agents](http://arxiv.org/abs/2511.14584)

- ReflexGrad: introduces a novel architecture that tightly couples LLM-based hierarchical TODO decomposition, history-aware causal reflection, and gradient-based optimization (TextGrad) via a three-way closed feedback loop for zero-shot generalization in LLM Agents.
- The system achieves true zero-shot generalization by relying on pure LLM semantic reasoning for task decomposition and memory retrieval, avoiding task-specific examples or hardcoded metrics.
- Key architectural components include a three-tier hierarchical memory system and a synergistic coupling mechanism where reflexions inform gradients, and gradients guide TODO progression and reflexion priorities.

---

[Agent-R1: Training Powerful LLM Agents with End-to-End Reinforcement Learning](http://arxiv.org/abs/2511.14460)

- Agent-R1: introduces a modular, flexible, and user-friendly training framework for RL-based LLM Agents by extending the Markov Decision Process (MDP) framework to comprehensively define key components for multi-turn interaction.
- The framework supports multi-turn rollouts, precise credit assignment via action masks, and flexible integration of Tools and ToolEnv for active environmental intervention.
- Agent-R1 utilizes process rewards and action masks during policy optimization to effectively train LLM agents for complex, interactive tasks like Multi-hop QA.

---

[Agentic Video Intelligence: A Flexible Framework for Advanced Video Exploration and Understanding](http://arxiv.org/abs/2511.14446)

- Agentic Video Intelligence (AVI): introduces a flexible and training-free framework that mirrors human video comprehension through system-level design and optimization, utilizing a structured knowledge base and three-phase reasoning.
- The framework employs an Agentic Core with Retrieve, Perceive, and Review phases, leveraging specialized tool suites for global exploration and fine-grained visual analysis.
- AVI builds a structured video knowledge base including entity graphs and uses an open-source model ensemble, eliminating reliance on proprietary APIs or resource-intensive RL training.

---

[Tell Me: An LLM-powered Mental Well-being Assistant with RAG, Synthetic Dialogue Generation, and Agentic Planning](http://arxiv.org/abs/2511.14445)

- Tell Me: introduces a mental well-being system that leverages LLMs, integrating a Retrieval-Augmented Generation (RAG) assistant, a synthetic client-therapist dialogue generator, and a Well-being AI Crew for personalized, knowledge-grounded dialogue, data augmentation, and adaptive self-care planning.
- The system components include a RAG assistant for context-aware reflective dialogue, a module for generating synthetic transcripts based on client profiles to address data scarcity, and a CrewAI-based planner for dynamic self-care routines like weekly plans and guided meditations.
- The work demonstrates how retrieval grounding enhances responsible interaction in emotionally sensitive domains and provides an open-source testbed for responsible LLM applications in mental well-being.

---

[MEDBENCH V4: A ROBUST AND SCALABLE BENCHMARK FOR EVALUATING CHINESE MEDICAL LANGUAGE MODELS, MULTIMODAL MODELS, AND INTELLIGENT AGENTS](http://arxiv.org/abs/2511.14439)

- MedBench v4: introduces a nationwide, cloud-based benchmarking infrastructure for medical AI, comprising expert-curated tasks across LLM, multimodal, and agent tracks, with scoring calibrated by an LLM-as-a-judge system and human ratings.
- The benchmark covers 24 primary and 91 secondary Chinese medical specialties, focusing on scenario-aligned evaluations that mirror real clinical workflows, including safety and ethics constraints.
- Agentic orchestration significantly improves end-to-end performance, especially in safety tasks, suggesting governance-aware systems enhance clinical readiness beyond base model capabilities.

---

[Enhancing LLM-based Autonomous Driving with Modular Traffic Light and Sign Recognition](http://arxiv.org/abs/2511.14391)

- TLS-Assist: introduces a modular redundancy layer that augments LLM-based autonomous driving agents with explicit traffic light and sign recognition, using components like Traffic Light Recognition (TLR), Traffic Sign Recognition (TSR), Relevance Prediction, State Validation, and a Message Generator.
- The framework converts visual detections into concise natural language messages injected into the LLM input to enforce attention to safety-critical traffic rules.
- Evaluation on the LangAuto benchmark shows consistent performance improvements for LMDrive and BEVDriver baselines, particularly in reducing traffic rule infractions.

---

[DataSage: Multi-agent Collaboration for Insight Discovery with External Knowledge Retrieval, Multi-role Debating, and Multi-path Reasoning](http://arxiv.org/abs/2511.14299)

- DataSage: introduces a novel multi-agent framework that incorporates external knowledge retrieval, multi-role debating, and multi-path reasoning to automate data insight discovery, addressing limitations like insufficient domain knowledge, shallow depth, and error-prone code generation, using components like the Dataset Description Module/RAKG Module/Question Raising Module/Insights Generation Module.
- The framework operates in an iterative Question-Answering (QA) loop, where specialized agents collaborate within four core modules to progressively refine analytical questions and generate robust, executable code for insight extraction.
- Experimental results on InsightBench show that DataSage consistently outperforms existing data insight agents across all difficulty levels, particularly excelling in complex and high-difficulty tasks.

---

[Run, Ruminate, and Regulate: A Dual-process Thinking System for Vision-and-Language Navigation](http://arxiv.org/abs/2511.14131)

- R³ (Run, Ruminate, and Regulate): introduces a novel dual-process thinking framework for Vision-and-Language Navigation (VLN) integrating LLMs' generalization with VLN-specific expertise, comprising Runner, Ruminator, and Regulator modules.
- The framework emulates human cognition, using the Runner for fast, routine navigation and the Ruminator (backed by GPT-4o and Chain-of-Thought prompting) for slow, methodical reasoning in anomalous scenarios.
- The Regulator adaptively switches between the two thinking modes based on looping, scoring, and ending criteria, achieving superior performance and efficiency over state-of-the-art LLM-assisted methods.

---

[PRISM: Prompt-Refined In-Context System Modelling for Financial Retrieval](http://arxiv.org/abs/2511.14130)

- PRISM (Prompt-Refined In-Context System Modelling): introduces a training-free framework that integrates refined system prompting, in-context learning (ICL), and a lightweight multi-agent system for document and chunk ranking in financial information retrieval.
- The framework utilizes four prompt variants ($P_1$ to $P_4$) to structure reasoning, an embedding-based retrieval mechanism for few-shot examples, and specialized agents coordinated via a state-graph for chunk ranking.
- The best non-agentic configuration ($P_4$ prompt with document-level ICL) achieved high performance on the FinAgentBench benchmark, demonstrating practical feasibility.

---

[Knowledge-Grounded Agentic Large Language Models for Multi-Hazard Understanding from Reconnaissance Reports](http://arxiv.org/abs/2511.14010)

- MoRA-RAG (Mixture-of-Retrieval Agentic RAG): introduces a knowledge-grounded LLM framework that transforms unstructured reconnaissance reports into a structured foundation for multi-hazard reasoning by integrating a Mixture-of-Retrieval mechanism and an agentic verification loop.
- The framework utilizes agentic chunking to preserve contextual coherence and employs specialized agents for evidence validation, external search, and query refinement to enhance retrieval precision and reasoning robustness.
- MoRA-RAG achieves up to 94.5% accuracy on the HazardRecQA dataset, significantly outperforming standard RAG systems and enabling open-weight LLMs to achieve performance comparable to proprietary models.

---

[O-Mem: Omni Memory System for Personalized, Long Horizon, Self-Evolving Agents](http://arxiv.org/abs/2511.13593)

- O-Mem (Omni Memory System): introduces a human-centric memory framework based on active user profiling that dynamically extracts and updates user characteristics and event records, utilizing Persona Memory (PM), Episodic Memory (EM), and Working Memory (WM) for hierarchical retrieval.
- The framework supports Long-Term Personality Modeling, Dual-Context Awareness, and Structured, Multi-Stage Retrieval to enhance personalized and context-aware interactions for LLM agents.
- Experimental results show O-Mem achieves state-of-the-art performance on benchmarks like LoCoMo and PERSONAMEM while significantly reducing token consumption and latency compared to existing memory frameworks.

---

[Multi-Agent Deep Research: Training Multi-Agent Systems with M-GRPO](http://arxiv.org/abs/2511.13288)

- M-GRPO (Multi-agent Group Relative Policy Optimization): introduces a hierarchical reinforcement learning framework for training separate LLMs in vertical multi-agent systems, featuring a main agent (M) and multiple sub-agents (S) with group-relative advantages and trajectory alignment.
- The framework addresses optimization challenges in vertical multi-agent systems by computing hierarchical credit assignment and using a trajectory-alignment scheme to handle variable sub-agent invocations efficiently.
- Empirical results show that co-training both agents using this method consistently outperforms single-agent and main-only training baselines on complex, tool-augmented reasoning benchmarks.

---

[MiroThinker: Pushing the Performance Boundaries of Open-Source Research Agents via Model, Context, and Interactive Scaling](http://arxiv.org/abs/2511.11793)

- MiroThinker: introduces MiroThinker v1.0, an open-source research agent that advances tool-augmented reasoning and information-seeking capabilities by exploring interaction scaling as a third dimension alongside model size and context length, utilizing components like a structured Tool Interface, Recency-Based Context Retention, and a three-stage Training Pipeline (SFT, DPO, GRPO).
- The agentic workflow follows the ReAct paradigm, iteratively generating thoughts, invoking tools via a modular Tool Interface (Execution Environment, File Management, Information Retrieval), and processing observations, managed by Recency-Based Context Retention to optimize context window usage.
- Training involves Supervised Fine-tuning (SFT), Agentic Preference Optimization (DPO), and Agentic Reinforcement Learning (GRPO) using data synthesized via a comprehensive Data Construction Pipeline, leading to state-of-the-art performance among open-source research agents.

---

[Z-Merge: Multi-Agent Reinforcement Learning for On-Ramp Merging with Zone-Specific V2X Traffic Information](http://arxiv.org/abs/2511.14910)

- Z-Merge: introduces a zone-based on-ramp merging control method using MARL (Multi-Agent Reinforcement Learning) incorporating RSU-collected, zone-specific traffic information from pre-merging, merging, and ramp zones, with components like MA-POMDP/PDQN/Double PDQN/SimServ/SUMO/MOSAIC.
- The framework utilizes a hybrid action space combining discrete lane changes and continuous acceleration/gap control, evaluated using metrics like efficiency, safety, comfort, success rate, and queue length.
- The approach leverages centralized training with decentralized execution (CTDE) and parameter-sharing to enable agents to make holistic decisions using both local and global traffic observations.

---

[Attacking Autonomous Driving Agents with Adversarial Machine Learning: A Holistic Evaluation with the CARLA Leaderboard](http://arxiv.org/abs/2511.14876)

- The paper introduces a holistic evaluation methodology for adversarial machine learning attacks against Autonomous Driving Agents using the CARLA Simulator and CARLA Leaderboard, focusing on the interaction between the ML Model and other control modules.
- The evaluation assesses stopping and steering attacks against open-source agents, demonstrating that agent-specific modules like PID controllers and GPS-based rules can mitigate attacks that successfully mislead the underlying ML Model.
- The authors propose a new leaderboard structure to facilitate systematic red-and-blue-team evaluation of adversarial robustness in standardized driving environments.

---

[Enhancing Agentic Autonomous Scientific Discovery with Vision-Language Model Capabilities](http://arxiv.org/abs/2511.14631)

- cmbagent: introduces a multi-agent system guided by Vision-Language Models (VLMs) to improve end-to-end autonomous scientific discovery by treating plots as verifiable checkpoints, utilizing a VLM-as-a-judge for self-correction and steering exploration.
- The system employs specialized agents like the Plot Judge and Plot Debugger, routing execution based on VLM feedback against domain-specific rubrics to correct errors or initiate exploratory analysis.
- This approach achieves pass@1 scores of 0.7-0.8 on a 10-task benchmark, significantly outperforming code-only and code-and-text baselines, while generating auditable reasoning traces.

---

[Emergent Cooperative Driving Strategies for Stop-and-Go Wave Mitigation via Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2511.14378)

- Emergent Cooperative Driving Strategies for Stop-and-Go Wave Mitigation via Multi-Agent Reinforcement Learning: introduces a novel mitigation strategy for stop-and-go waves discovered through training DRL agents in a simulated ring-road environment, where one vehicle acts as a buffer.
- The discovered cooperative strategy involves heterogeneous behavior where a single "buffer" vehicle maintains a large headway while others platoon closely, enhancing stability and throughput compared to non-cooperative uniform driving.
- This buffering approach is validated by implementing it in the classical Intelligent Driver Model (IDM), showing suppression of stop-and-go waves and improved average speed under stability constraints.

---

[Orion: A Unified Visual Agent for Multimodal Perception, Advanced Visual Reasoning and Execution](http://arxiv.org/abs/2511.14210)

- Orion: introduces a visual agent framework that orchestrates specialized computer vision tools using an agentic controller, enabling advanced multimodal perception, reasoning, and execution.
- The framework integrates large Vision-Language Models (VLMs) with hyper-specialized tools for tasks like object detection, OCR, and image generation, moving beyond descriptive outputs to active, tool-driven visual intelligence.
- It employs a ReAct-style orchestration with Plan, Execute, and Reflect phases, ensuring structured, verifiable, and high-quality multi-step visual workflows.

---

[APD-Agents: A Large Language Model-Driven Multi-Agents Collaborative Framework for Automated Page Design](http://arxiv.org/abs/2511.14101)

- APD-Agents: introduces a large language model (LLM) driven multi-agent framework for automated page design in mobile applications, containing OrchestratorAgent, SemanticParserAgent, PrimaryLayoutAgent, TemplateRetrievalAgent, and RecursiveComponentAgent.
- The framework operates in a coarse-to-fine, top-down, iterative generation process, outputting structured JSON data compatible with professional design software like Sketch and Figma.
- It leverages In-Context Learning via the TemplateRetrievalAgent to enhance layout quality without explicit model training.

---

[Hybrid Agentic AI and Multi-Agent Systems in Smart Manufacturing](http://arxiv.org/abs/2511.18258)

- HAIMAS (Hybrid Agentic AI Multi-Agent System): introduces a layered architecture for Prescriptive Maintenance (RxM) in smart manufacturing, utilizing an LLM Orchestrator Agent for strategic planning and specialized agents (Perception, Preprocessing, Analysis, Optimization) for distributed execution.
- The framework integrates high-level LLM reasoning with efficient, domain-specific execution by rule-based and local SLMs, ensuring robustness and scalability at the edge.
- A Human-In-The-Loop (HITL) interface ensures transparency and auditability by allowing human experts to review, approve, or reject the actionable, prioritized maintenance recommendations.

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







