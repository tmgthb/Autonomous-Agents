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





#### 5th March 2026


[TritonDFT: Automating DFT with a Multi-Agent Framework](http://arxiv.org/abs/2603.03372)

- TritonDFT: introduces a multi-agent framework for automating end-to-end Density Functional Theory workflows, which includes planning-, parameter inference-, script generation-, execution-, and interpretation-agents.
- The framework employs a Pareto-aware reasoning mechanism to iteratively refine physical parameters for accuracy-cost trade-offs and automates High-Performance Computing parallelization based on hardware specifications.
- The research introduces DFTBENCH, a benchmark suite featuring 68 materials to evaluate LLM performance in scientific expertise, resource efficiency, and cross-domain task execution.

---


[stratum: A System Infrastructure for Massive Agent-Centric ML Workloads [Vision]](http://arxiv.org/abs/2603.03589)

- stratum: introduces a system infrastructure designed for massive agent-centric machine learning workloads, featuring a logical optimizer that fuses pipeline variants into unified directed acyclic graphs and a high-performance Rust-based runtime.
- The system leverages skrub-based operator abstractions to perform metadata-driven rewrites, operator lowering, and late-binding selection of physical implementations across heterogeneous backends.
- By implementing speculative caching and multi-level parallelization, the framework significantly reduces redundant computation and memory overhead in iterative agentic pipeline search.

---

[AOI: Turning Failed Trajectories into Training Signals for Autonomous Cloud Diagnosis](http://arxiv.org/abs/2603.03378)

- AOI (Autonomous Operations Intelligence): introduces a trainable multi-agent framework for autonomous cloud diagnosis that converts failed operational trajectories into corrective training signals using Group Relative Policy Optimization (GRPO), incorporating Observer, Probe, Executor, Compressor, Evolver, Purifier, Judge, Dual-Timescale Memory, and Reward Model components, including planning-, exploration-, remediation-, and distillation-agents.
- The architecture enforces safety through a read-write separated execution model where the Observer coordinates read-only Probes and write-gated Executors to prevent unauthorized state mutation during learning.
- The framework utilizes a dual-timescale memory and a Failure Trajectory Closed-Loop Evolver to enable continual data augmentation and distributional refinement within secure enterprise environments.

---


[Safe-SAGE: Social-Semantic Adaptive Guidance for Safe Engagement through Laplace-Modulated Poisson Safety Functions](http://arxiv.org/abs/2603.05497)

- Safe-SAGE (Social-Semantic Adaptive Guidance for Safe Engagement): introduces a unified framework bridging high-level semantic reasoning and low-level safety control through Laplace-modulated Poisson safety functions, with perception & tracking, social-semantic field synthesis, dual-layer safety filter, and semantic tracking occupancy grid components; it includes perception-, field synthesis-, and dual-layer safety-agents.
- The system integrates YOLOv11n for instance segmentation and an object-level tracker to maintain a persistent semantic occupancy grid for dynamic environment representation beyond the camera's field of view.
- A hierarchical safety architecture combines a predictive MPC filter for anticipatory trajectory planning with a real-time analytical CBF layer for immediate reactive control.

---

[Towards Multimodal Lifelong Understanding: A Dataset and Agentic Baseline](http://arxiv.org/abs/2603.05484)

- ReMA (Recursive Multimodal Agent): introduces an agentic framework for multimodal lifelong understanding that utilizes a User Query, Multimodal Context, Recursive Multimodal Agent, Multimodal Toolkits, Memory Bank, and Foundation Models to produce a Response.
- The architecture employs a two-phase approach consisting of a perception phase for incremental memory consolidation and a control phase for iterative reasoning using LLM-controller and MLLM-perception components.
- The system addresses the working memory bottleneck of end-to-end models by treating the lifelong stream as an active knowledge base rather than a linear context window.

---

[Leveraging LLM Parametric Knowledge for Fact Checking Without Retrieval](http://arxiv.org/abs/2603.05471)

- INTRA (Intrinsic Truthfulness Assessment): introduces a retrieval-free fact-checking method that leverages internal model representations to verify atomic claims using only parametric knowledge.
- The framework utilizes attention-based pooling to aggregate hidden states into sequence-level embeddings, followed by layer-wise linear probing and a regression-based aggregation of normalized truthfulness probabilities.
- Evaluation across nine diverse datasets demonstrates that internal-based signals consistently outperform logit-based uncertainty measures, achieving competitive performance in cross-model and multilingual generalization.

---

[Latent Wasserstein Adversarial Imitation Learning](http://arxiv.org/abs/2603.05440)

- LWAIL (Latent Wasserstein Adversarial Imitation Learning): introduces an adversarial imitation learning framework for state-only distribution matching, with ICVF (pre-trains dynamics-aware state representations), a Latent Embedding Space (encodes reachability between states), a Discriminator (measures Wasserstein distance in latent space), an Agent Policy (optimizes behavior using pseudo-rewards), a Replay Buffer (stores transitions for off-policy learning), and an Environment (provides state transitions and observations).
- The system leverages a pre-training phase on unstructured random data to establish a reachability-based metric, overcoming the limitations of raw Euclidean distance in Wasserstein distance calculations.
- The method achieves performance comparable to expert demonstrations on MuJoCo benchmarks using a single state-only trajectory.

---

[Building Enterprise Realtime Voice Agents from Scratch: A Technical Tutorial](http://arxiv.org/abs/2603.05413)

- Enterprise Realtime Voice Agent: introduces a modular framework for low-latency conversational AI, with Streaming STT, vLLM-served LLMs, and Streaming TTS components.
- The architecture employs a sentence buffer to enable pipelined execution, allowing audio synthesis to begin before the LLM completes its full response.
- The tutorial provides a comprehensive implementation guide for building production-grade agents capable of complex function calling with sub-second response times.

---

[OpenFrontier: General Navigation with Visual-Language Grounded Frontiers](http://arxiv.org/abs/2603.05377)

- OpenFrontier: introduces a training-free navigation framework that utilizes visual frontiers as sparse semantic anchors to ground high-level vision-language priors into actionable robotic decisions, with Visual Frontier Detector, Vision-Language Model (VLM), Set-of-Marks (SoM) Prompting, 3D Metric Space Grounding, Global Frontier Manager, Low-level Motion Planner, and Open-vocabulary Segmenter.
- The architecture integrates image-space frontier reasoning with global goal management, using a set-of-marks prompting strategy to enable VLMs to assign semantic priority to candidate subgoals.
- By avoiding dense 3D reconstruction and task-specific fine-tuning, the system demonstrates robust zero-shot generalization across diverse open-world environments and real-world robotic platforms.

---

[The Effect of a Toroidal Opinion Space on Opinion Bi-Polarisation](http://arxiv.org/abs/2603.05337)

- Axelrod-based Opinion Dynamics Model: introduces a comparative study of opinion evolution on cubic and toroidal manifolds, with all Agents, Opinion Profiles, Population Grid, Distance Metrics, Interaction Function, Bounded Confidence, Topical Weights, and Network Rewiring components, where it evaluates the influence of opinion space topology on consensus and polarization.
- The framework utilizes a distance-dependent interaction probability and a stepping mechanism for opinion updates to model ordinal shifts.
- Findings indicate that toroidal opinion spaces consistently support a higher number of opinion groups in steady state and exhibit greater sensitivity to model extensions than cubic spaces.

---

[Designing for Adolescent Voice in Health Decisions: Embodied Conversational Agents for HPV Vaccination](http://arxiv.org/abs/2603.05321)

- ClaraEdu: introduces a dual-path mobile health intervention for HPV vaccination, with physician- and game-based Embodied Conversational Agents, 3D rendering engine, BEAT engine, hierarchical transition network, LLM-generated narrative assets, and a question transmission feature.
- The system utilizes a "dual-path" architecture where parents interact with a simulated physician while adolescents choose between a standard doctor or a gamified narrative experience featuring LLM-generated riddles.
- The framework facilitates shared decision-making by coaching parent-adolescent dyads separately and providing a mechanism for adolescents to flag concerns directly to clinical providers before appointments.

---

[WebChain: A Large-Scale Human-Annotated Dataset of Real-World Web Interaction Traces](http://arxiv.org/abs/2603.05295)

- WebChain: introduces a dataset of 31,725 human-annotated web interaction trajectories featuring synchronized visual context, structural semantics, and action alignment.
- The framework utilizes a Dual Mid-Training recipe that disentangles spatial perception from temporal planning to improve agent performance on long-horizon web tasks.
- The pipeline includes a generator LLM for task synthesis and a VLM for rationale generation to provide dense multi-modal supervision for training web agents.

---

[STRUCTUREDAGENT: Planning with AND/OR Trees for Long-Horizon Web Tasks](http://arxiv.org/abs/2603.05294)

- STRUCTUREDAGENT: introduces a hierarchical planning framework for long-horizon web tasks, with Structured Memory, an AND/OR Tree Planner, and a greedy modified DFS planner, where the system dynamically constructs and executes ordered subgoals.
- The framework separates planning responsibilities by using an LLM as a high-level controller for local tree operations, including expansion-, repair-, completion check-, and summarization-modules, while the core system manages global tree traversal.
- It incorporates a structured memory module to track candidate entities and satisfied constraints, improving performance on complex information-seeking tasks through interpretable hierarchical plans and optional human intervention.

---

[Knowledge Divergence and the Value of Debate for Scalable Oversight](http://arxiv.org/abs/2603.05293)

- Knowledge Divergence: introduces a geometric framework to relate AI safety via debate and RLAIF, using principal angles between representation subspaces to quantify the advantage of multi-model interaction, which includes debater-models A and B.
- The analysis characterizes three regimes of knowledge divergence—shared, one-sided, and compositional—proving that debate advantage is negligible when LLMs share training corpora but essential when they possess divergent knowledge.
- The paper establishes a formal connection between debate and eliciting latent knowledge, while identifying a sharp threshold where adversarial incentives cause coordination failure in complex compositional tasks.

---

[Iterative On-Policy Refinement of Hierarchical Diffusion Policies for Language-Conditioned Manipulation](http://arxiv.org/abs/2603.05291)

- HD-ExpIt (Hierarchical Diffusion with Expert Iteration): introduces an iterative on-policy refinement framework for hierarchical diffusion policies, with a high-level planner (HL), a low-level controller (LL), an environment, a dataset, a reward filter, a subgoal sequence, and action chunks, where environment feedback facilitates alignment between high-level visual planning and low-level control capabilities.
- The framework utilizes the stochastic nature of a diffusion-based planner as a search mechanism to discover successful behaviors that are aggregated into the training set for supervised fine-tuning.
- This approach outperforms existing baselines on the long-horizon CALVIN benchmark by aligning the planner with the controller's operational limits without requiring explicit proxy models.

---

[A monitoring system for collecting and aggregating metrics from distributed clouds](http://arxiv.org/abs/2603.05241)

- c12s (constellations): introduces a native monitoring system for distributed clouds, with machine-level collector, container-level collector, metrics agent, metrics processor, metrics storage, and metrics reader.
- The architecture optimizes communication by piggybacking collected metrics onto periodic health-check protocol responses, significantly reducing the network request frequency for resource-constrained nodes.
- The framework provides multi-level observability across hardware, container, and application layers, exposing aggregated data through REST and streaming APIs for real-time consumption by schedulers and autoscalers.

---

[GCAgent: Enhancing Group Chat Communication through Dialogue Agents System](http://arxiv.org/abs/2603.05240)

- GCAgent: introduces an LLM-driven system for enhancing group chat communication through entertainment and utility-oriented dialogue agents, utilizing Agent Builder, Dialogue Manager, Interaction Manager, LLM Engine, Post-generation Validator, and Interface Plugins.
- The Interaction Manager coordinates multi-party dialogue states and invocations, while the fine-tuned LLM Engine generates context-aware responses validated by a post-generation quality check.
- Interface Plugins provide ASR, TTS, and TTSing capabilities to support diverse communication modes and reduce interaction barriers in real-world social platform deployments.

---

[KARL: Knowledge Agents via Reinforcement Learning](http://arxiv.org/abs/2603.05218)

- KARL (Knowledge Agents via Reinforcement Learning): introduces a system for training enterprise search agents via reinforcement learning to optimize grounded reasoning across diverse search regimes, with a Question-Answer Synthesizer, Deduplication Agent, Solver Agent, Quality Filter Agent, OAPL, Vector Search Tool, Context Compression, Aggregator Agent, Value Model, and aroll Harness.
- The framework includes synthesis-, solver-, quality filter-, and aggregator-agents to automate high-quality training data generation and multi-step search optimization.
- It achieves Pareto-optimal performance by scaling test-time compute through parallel rollout aggregation and value-guided tree search on the KARLBench suite.

---

[Escaping the Hydrolysis Trap: An Agentic Workflow for Inverse Design of Durable Photocatalytic Covalent Organic Frameworks](http://arxiv.org/abs/2603.05188)

- Ara: introduces an agentic workflow for the inverse design of durable photocatalytic covalent organic frameworks, employing an LLM agent to navigate combinatorial chemical spaces through iterative reasoning and quantitative feedback.
- The system utilizes a fragment-based screening pipeline powered by GFN1-xTB and a composite stability index to identify candidates that simultaneously satisfy band-gap, band-edge, and hydrolytic-stability criteria.
- By leveraging pretrained chemical knowledge and donor-acceptor theory, the agent achieves high hit rates and rapid convergence compared to traditional Bayesian optimization and random search methods.

---

[SWARM-SLR AIssistant: A Unified Framework for Scalable Systematic Literature Review Automation](http://arxiv.org/abs/2603.05177)

- SWARM-SLR AIssistant (Streamlined Workflow for Automating Machine-Actionable Systematic Literature Reviews AIssistant): introduces a unified framework for scalable systematic literature review automation, with SWARM-SLR requirements, metadata schema, tool registry, AIssistant, workflow agents, local data storage, and external research tools.
- The framework utilizes task-specific agents—including scientific interest refinement, search query formulation, and keyword synthesis modules—with unique system prompts and toolsets to guide users through the review process.
- A centralized tool registry facilitates autonomous tool annotation by developers, allowing for modular integration of external services like ORKG ASK and Semantic Scholar into the conversational interface.

---

[Lifelong Language-Conditioned Robotic Manipulation Learning](http://arxiv.org/abs/2603.05160)

- SkillsCrafter: introduces a lifelong robotic manipulation framework that enables agents to continually learn new skills while mitigating catastrophic forgetting through Manipulation Skills Adaptation, Skills Specialization Aggregation, and Skills Specified Inference components.
- The system decouples LoRA weights into skill-shared and skill-specific subspaces, utilizing an adapter inheritance strategy and orthogonal constraints to preserve prior knowledge across sequential tasks.
- It employs Singular Value Decomposition on instruction embeddings to compute semantic similarity, enabling adaptive knowledge aggregation for both known and unknown open-world manipulation tasks.

---

[MedCoRAG: Interpretable Hepatology Diagnosis via Hybrid Evidence Retrieval and Multispecialty Consensus](http://arxiv.org/abs/2603.05129)

- MedCoRAG (Medical Collaborative RAG): introduces an end-to-end framework for interpretable hepatology diagnosis that grounds multidisciplinary clinical reasoning in unified evidence synthesis, with abnormal findings and preliminary diagnosis module, hybrid RAG module, router agent, specialist agents, and generalist agent.
- The framework integrates guideline-constrained knowledge graph pruning with dynamic, complexity-aware specialist dispatch to emulate multidisciplinary consultation through iterative, evidence-constrained diagnostic loops. 
- Evaluated on real-world hepatic cases from MIMIC-IV, the system outperforms existing LLMs and RAG methods in diagnostic accuracy while providing traceable, clinician-aligned reasoning paths.

---

[Bidirectional Curriculum Generation: A Multi-Agent Framework for Data-Efficient Mathematical Reasoning](http://arxiv.org/abs/2603.05120)

- Bidirectional Curriculum Generation: introduces a multi-agent ecosystem that dynamically adjusts mathematical problem difficulty through a closed feedback loop, which includes difficulty-reduction, reverse-generation, difficulty-increasing, and diversity-enhancement agents.
- The framework employs a diagnostic evaluation stage to partition data into easy and hard sets, triggering either upward expansion to challenge the model or downward adjustment to repair specific reasoning failures.
- Grounded in the Optimal Pacing Theorem, the system optimizes the learning trajectory by ensuring the training distribution remains within the model's zone of proximal development, significantly improving data efficiency for LLMs.

---

[Decoupling Task and Behavior: A Two-Stage Reward Curriculum in Reinforcement Learning for Robotics](http://arxiv.org/abs/2603.05113)

- RC (Reward Curriculum): introduces a method to decouple task-specific objectives from behavioral terms by first training on a task-only base reward before transitioning to a combined reward function, with a policy, critic, replay buffer, base reward, auxiliary reward, phase switch mechanism, and annealing scheduler, and includes actor- and critic-networks.
- The system employs a phase switch mechanism triggered by actor convergence or reward thresholds to initiate the second stage of the curriculum.
- It incorporates an annealing scheduler and a flexible replay buffer to ensure continuous value function updates and reduced sample complexity through experience reuse.

---

[UNIM: A Unified Any-to-Any Interleaved Multimodal Benchmark](http://arxiv.org/abs/2603.05075)

- UNIMA (Unified Any-to-Any Interleaved Multimodal Agentic model): introduces an agentic framework for any-to-any interleaved multimodal learning, with a Receiving Module (converts multimodal inputs to text), a Traceable Evidence Reasoning (TER) Module (executes structured evidence reasoning chain), a Verification Submodule (detects errors and triggers backtracking), a Generating Module (synthesizes final interleaved multimodal content), and a Multimodal Toolset (provides specialized modality-specific processing capabilities).
- The TER Module utilizes a four-step Structured Evidence Reasoning Chain to produce task-conditioned dense captions, data reports, and structured planning strings.
- The architecture integrates specialized LLMs and generative tools including Qwen3-Omni, GPT-5, Sora 2, and PCDreamer to support seven distinct modalities.

---

[Jagarin: A Three-Layer Architecture for Hibernating Personal Duty Agents on Mobile](http://arxiv.org/abs/2603.05069)

- Jagarin: introduces a three-layer architecture for mobile personal AI agents, with DAWN (on-device heuristic scoring engine), ARIA (commercial email identity proxy), ACE (machine-readable communication protocol), and an Ephemeral Cloud Agent (stateless LLM-based task executor).
- The system resolves the mobile deployment paradox by using structured hibernation and demand-driven wake cycles to minimize battery drain while ensuring time-sensitive obligations are met.
- It features a privacy-preserving design with no persistent cloud state, local-only behavioral data, and a Gemini-based parser for automatic duty ingestion from institutional communications.

---

[Reward-Conditioned Reinforcement Learning](http://arxiv.org/abs/2603.05066)

- RCRL (Reward-Conditioned Reinforcement Learning): introduces a framework for training agents to optimize multiple reward specifications off-policy, with Policy, Environment, Replay Buffer, Reward Parameterization Sampler, Reward Aggregator, Actor, and Critic.
- The architecture leverages a shared replay buffer to store raw reward components, enabling the computation of counterfactual rewards for off-policy training without additional environment interaction.
- RCRL improves sample efficiency in single-task and multi-task settings and enables zero-shot behavioral adjustment and accelerated finetuning to new reward functions.

---

[WebFactory: AUTOMATED COMPRESSION OF FOUNDATIONAL LANGUAGE INTELLIGENCE INTO GROUNDED WEB AGENTS](http://arxiv.org/abs/2603.05044)

- WebFactory: introduces a fully automated closed-loop reinforcement learning pipeline for GUI agents, with Foundation LLM, High-Fidelity Offline Environment, Auto-Task Synthesis, Teacher Agent, Filtering Process, Student Agent, Unified Action Space, Decomposed Reward Function, and GRPO Optimization, where it systematically compresses LLM-encoded internet intelligence into efficient, grounded actions.
- The framework utilizes a high-fidelity offline environment to provide strict controllability and full observability, allowing an LLM-driven synthesizer to generate a virtually infinite stream of executable tasks.
- It employs a teacher-student reinforcement learning approach with a decomposed reward function to bridge the semantic-to-action gap and achieve superior generalization on online benchmarks.

---

[CoIn3D: Revisiting Configuration-Invariant Multi-Camera 3D Object Detection](http://arxiv.org/abs/2603.05042)

- CoIn3D (Configuration-Invariant 3D): introduces a generalizable multi-camera 3D object detection framework that addresses spatial prior discrepancies across different camera configurations, with CDA, SFM, 3DGS Renderer, Spatial Priors, Backbone & Neck, Projector, and MC3D Schemes.
- The framework utilizes a training-free 3D Gaussian Splatting pipeline to dynamically synthesize novel-view images with varied intrinsics and extrinsics for robust data augmentation.
- It explicitly incorporates four spatial representations—inverse focal length, ground depth, ground gradient, and Plucker coordinates—to enrich the feature space and achieve cross-configuration transferability across bottom-up BEV, top-down BEV, and sparse queries schemes.

---

[AegisUI: Behavioral Anomaly Detection for Structured User Interface Protocols in AI Agent Systems](http://arxiv.org/abs/2603.05031)

- AegisUI: introduces a framework for detecting behavioral anomalies in structured UI protocols generated by AI agents, with payload generation, validation, feature extraction, and detection model components.
- The system extracts 18 features across structural, semantic, binding, and session categories to identify malicious payloads that pass standard schema checks.
- Evaluation of three detection models—Isolation Forest, Autoencoder, and Random Forest—demonstrates that supervised learning achieves high precision in identifying complex UI-based attacks.

---

[S5-SHB Agent: Society 5.0 enabled Multi-model Agentic Blockchain Framework for Smart Home](http://arxiv.org/abs/2603.05027)

- S5-SHB-Agent: introduces a smart-contract-free framework orchestrating ten specialized agents—including safety-, health-, security-, privacy-, energy-, climate-, maintenance-, NLU-, and arbitration-agents—using interchangeable LLMs to manage smart home operations.
- The architecture features an adaptive Proof-of-Work blockchain that dynamically adjusts mining difficulty based on transaction volume and emergency conditions to ensure tamper-evident auditability and rapid block confirmation.
- A four-tier human-centered governance model enables residents to control automation through natural language interfaces while enforcing immutable safety thresholds via a firmware-level emergency bypass.

---

[Competitive Multi-Operator Reinforcement Learning for Joint Pricing and Fleet Rebalancing in AMoD Systems](http://arxiv.org/abs/2603.05000)

- Competitive Multi-Operator RL (Reinforcement Learning) framework: introduces a dual-operator control system where independent agents jointly learn pricing and fleet rebalancing policies through interaction with a shared urban mobility environment, with all Independent Actor-Critic Agents (learn pricing and rebalancing policies), Graph Neural Network Encoders (capture spatial dependencies in transportation networks), Multinomial Logit Choice Model (allocates passengers based on price and utility), Minimal-Cost Flow Rebalancing Solver (executes vehicle movements to reach desired distributions), and Shared AMoD Environment (simulates vehicle dynamics, queues, and demand).
- The architecture utilizes GCN-based actor-critic networks to process spatial graph states and output stochastic policies for origin-based price scalars and idle-vehicle distributions.
- Endogenous demand competition is modeled via a discrete choice mechanism, allowing passenger allocation to emerge from the utility-maximizing decisions of travelers sensitive to price and travel time.

---

[Replaying pre-training data improves fine-tuning](http://arxiv.org/abs/2603.04964)

- Generic Replay (Distributional Replay): introduces a data scheduling strategy that mixes fresh samples from the pre-training distribution into the fine-tuning stage to improve target domain performance and data efficiency.
- The framework employs a two-stage training architecture where Stage 2 integrates a specific replay fraction of generic data with target tokens, optimized via a Warmup-Stable-Decay learning rate schedule.
- Experiments demonstrate that replaying generic data increases target data efficiency by up to 1.87x for fine-tuning and significantly enhances performance on agentic web navigation and low-resource language tasks for 8B parameter LLMs.

---

[Retrieval-Augmented Generation with Covariate Time Series](http://arxiv.org/abs/2603.04951)

- RAG4CTS (Regime-Aware, training-free RAG framework for Covariate Time-Series): introduces a training-free retrieval-augmented generation framework for industrial time-series forecasting, with Hierarchical Knowledge Base, Time-series Native Retrieval, Agentic Splicing Augmentation, and TSFM Backbone.
- The system utilizes a two-stage bi-weighted retrieval mechanism to align historical trends through point-wise and multivariate similarities while preserving physical consistency.
- An agent-driven strategy dynamically optimizes the context length in a self-supervised manner by using the top-ranked retrieval as a calibration proxy for the foundation model.

---

[TIMEWARP: Evaluating Web Agents by Revisiting the Past](http://arxiv.org/abs/2603.04949)

- TIMEWARP: introduces a benchmark for evaluating web agents across evolving UI designs using containerized environments from different internet eras, alongside the TIMETRAJ algorithm for scalable multi-version trajectory collection.
- The framework employs a human-in-the-loop plan distillation process which includes planner-, executor-, and judge-agents to produce training data across multiple website layouts.
- Evaluation results show that training LLMs on multi-version trajectories with structured thinking and planning tokens improves their generalization to future web changes.

---

[When minor issues matter: symmetries, pluralism, and polarization in similarity-based opinion dynamics](http://arxiv.org/abs/2603.04939)

- Similarity-based Opinion Dynamics Framework: introduces a stochastic agent-based model and its deterministic large-population limit to study how heterogeneous issue weights and attraction-repulsion forces shape collective opinion evolution.
- The framework identifies three primary outcomes—consensus, polarization, and persistent pluralism—determined by the interplay between similarity thresholds and the relative importance of discussed topics.
- It demonstrates that introducing a single low-weight issue can destabilize stable states and significantly increase convergence times through shifts in the underlying symmetry of the dynamical system.

---

[AILS-NTUA at SemEval-2026 Task 10: Agentic LLMs for Psycholinguistic Marker Extraction and Conspiracy Endorsement Detection](http://arxiv.org/abs/2603.04921)

- DD-CoT (Dynamic Discriminative Chain-of-Thought) and Anti-Echo Chamber: introduces a two-stage agentic pipeline for psycholinguistic marker extraction and conspiracy detection, utilizing a self-refining extraction loop and an adversarial multi-persona council.
- The marker extraction stage decouples semantic reasoning from structural localization by combining a discriminative LLM generator with a deterministic post-processing verifier for character-accurate outputs.
- The detection stage employs a Parallel Council that includes prosecutor-, defense-, literalist-, and stance-profiler-agents to independently assess evidence before a Calibrated Judge aggregates their votes.

---

[EVMbench: Evaluating AI Agents on Smart Contract Security](http://arxiv.org/abs/2603.04915)

- EVMbench: introduces a framework and task suite that evaluates the ability of agents to detect, patch, and exploit fund-draining vulnerabilities in production-grade smart contract environments.
- The architecture includes Detect-, Patch-, and Exploit-agents that interact with a local Ethereum chain managed by a Rust-based orchestration harness and a security-hardened JSON-RPC proxy.
- The benchmark evaluates frontier LLMs using 117 curated vulnerabilities, finding that performance is primarily limited by discovery capabilities rather than transaction construction or code repair.

---

[Alignment as Iatrogenesis: Language-Dependent Reversal of Safety Interventions in LLM Multi-Agent Systems Across 16 Languages](http://arxiv.org/abs/2603.04904)

- SociA (SociA simulation engine): introduces, a multi-agent simulation framework to investigate how prefix-level alignment interventions affect collective behavior across 16 languages, with a SociA simulation engine (manages multi-agent conversational simulation), LLM-instantiated agents (ten agents with fixed personas), a high-alignment system prompt prefix (inference-time behavioral safety instructions), a shared text-based environment (conversational space for agent interaction), an environmental event injector (triggers scripted 15-turn escalation), a keyword-based detection system (extracts behavioral markers from logs), a Collective Pathology Index (CPI) (measures group-level institutional breakdown), and a Dissociation Index (DI) (measures insight-action behavioral gaps).
- The research identifies "alignment backfire" where safety instructions amplify collective pathology in specific language spaces like Japanese while functioning as intended in English.
- The study validates these patterns across Llama 3.3 70B, GPT-4o-mini, and Qwen3-Next-80B-A3B model families, proposing a "Coherence Trilemma" where internal coherence, external conformity, and transparency cannot be simultaneously satisfied.

---

[EVOTOOL: Self-Evolving Tool-Use Policy Optimization in LLM Agents via Blame-Aware Mutation and Diversity-Aware Selection](http://arxiv.org/abs/2603.04900)

- EVOTOOL (Self-Evolving Tool-Use Policy Optimization): introduces a gradient-free evolutionary framework that optimizes modular tool-use policies and includes planner-, selector-, caller-, and synthesizer-modules.
- The framework utilizes a trajectory-grounded blame attribution mechanism and includes blamer- and mutator-LLMs to localize failures and perform targeted natural-language policy updates.
- A diversity-aware population selection strategy preserves complementary candidates based on instance-level wins, preventing mode collapse and enhancing performance on complex, long-horizon tasks.

---

[SEA-TS: Self-Evolving Agent for Autonomous Code Generation of Time Series Forecasting Algorithms](http://arxiv.org/abs/2603.04873)

- SEA-TS (Self-Evolving Agent for Time Series Algorithms): introduces an autonomous framework for generating, validating, and optimizing forecasting algorithm code through an iterative self-evolution loop, with MCTS Tree, LLM-based code generator, LLM-based code reviewer, LLM-based reasoning agent, Sandbox environment, MAP-Elites quality-diversity archive, and Running prompt memory.
- The framework utilizes Metric-Advantage Monte Carlo Tree Search (MA-MCTS) to replace fixed rewards with statistically normalized advantage scores, providing discriminative guidance toward high-potential algorithmic trajectories.
- The system integrates automated code review with running prompt refinement and global steerable reasoning to persistently encode corrective patterns and enable cross-trajectory knowledge transfer, effectively preventing issues like data leakage.

---

[K-Gen: A Multimodal Language-Conditioned Approach for Interpretable Keypoint-Guided Trajectory Generation](http://arxiv.org/abs/2603.04868)

- K-Gen: introduces an interpretable keypoint-guided multimodal framework for autonomous driving trajectory generation, utilizing a vision encoder, text encoder, and MLLM to process rasterized BEV maps and textual scene descriptions.
- The system decomposes trajectory generation into a keypoint prediction phase guided by chain-of-thought reasoning and a subsequent refinement phase using a transformer-based TrajRefiner module to ensure physical consistency and smooth motion.
- The framework employs T-DAPO, a trajectory-aware reinforcement fine-tuning algorithm, to optimize keypoint generation through composite rewards focusing on trajectory accuracy, reasoning conciseness, and structural format integrity.

---

[CAUSALLY ROBUST REWARD LEARNING FROM REASON-AUGMENTED PREFERENCE FEEDBACK](http://arxiv.org/abs/2603.04861)

- ReCouPLe (Reason-based Confusion Mitigation in Preference Learning): introduces, a framework for learning rewards from reason-augmented preference feedback, with a trajectory encoder (trainable network mapping observations), a frozen LLM encoder (pre-trained LLM for embeddings), a joint representation space (shared space for alignment), orthogonal projection (decomposing embeddings into causal components), a reward model (calculating scalar rewards), and loss functions (optimizing causal grounding and consistency).
- The architecture decomposes trajectory embeddings into reason-aligned and reason-orthogonal subspaces to isolate causal features specified in natural language rationales.
- This decomposition prevents the reward model from relying on spurious correlations and facilitates zero-shot transfer to novel tasks by leveraging shared semantic reasons.

---

[FIREBENCH: Evaluating Instruction Following in Enterprise and API-Driven Large Language Model Applications](http://arxiv.org/abs/2603.04857)

- FIREBENCH: introduces a benchmark grounded in real-world enterprise and API usage patterns to evaluate instruction-following capabilities across six core dimensions, with all Output Format Compliance, Ordered Responses, Item Ranking, Overconfidence, Positive Content Requirements, Negative Content Requirements, Programmatic Verifiers, and LLM Judges components.
- The framework utilizes over 2,400 samples across diverse domains like information extraction and customer support to assess model reliability in production-like workflows.
- Evaluation of 11 LLMs reveals that even top-performing models struggle with adversarial formatting and consistent instruction adherence across different categories.

---

[HACHIMI: Scalable and Controllable Student Persona Generation via Orchestrated Agents](http://arxiv.org/abs/2603.04855)

- HACHIMI: introduces a multi-agent Propose–Validate–Revise framework for generating theory-aligned and distribution-controllable student personas, with Quota Scheduling, Stratified Sampling, Scholar-, Academic-, Values-, Social-Creative-, and Health-agents, a Shared Whiteboard, a Symbolic Critic, Iterative Revision, and Semantic Deduplication to produce the HACHIMI-1M Corpus.
- The architecture utilizes a shared whiteboard for sequential agent conditioning to mitigate intra-profile inconsistency and a neuro-symbolic validator to enforce developmental and psychological constraints derived from educational theory.
- The framework enables the creation of a high-fidelity synthetic student population generated with Qwen2.5-72B that aligns with real-world survey data to provide a standardized testbed for educational LLMs and social-science simulations.

---

[SCoUT: Scalable Communication via Utility-Guided Temporal Grouping in Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2603.04833)

- SCoUT (Scalable COmmunication via Utility-guided Temporal grouping): introduces a scalable multi-agent reinforcement learning framework using temporal and agent abstraction to structure communication in large populations, with a Grouping Module (samples soft agent clusters), an Affinity Matrix (induces differentiable recipient priors), a Shared Agent Backbone (maintains recurrent agent memory), a Three-headed Policy (outputs actions and communication), Mailbox Aggregation (combines incoming message vectors), a Group-aware Critic (factors value estimation via groups), a Communication Critic (evaluates message and pairwise utility), and a Counterfactual Mailbox (isolates individual message contributions).
- The architecture employs a group-aware critic to factorize value estimation and a counterfactual mailbox mechanism to provide credit assignment for individual message and recipient decisions.
- By resampling agent groups every K steps and using induced affinities as differentiable priors, the system scales learned communication to hundreds of agents while maintaining decentralized execution.

---

[Beyond the Context Window: A Cost-Performance Analysis of Fact-Based Memory vs. Long-Context LLMs for Persistent Agents](http://arxiv.org/abs/2603.04814)

- Mem0: introduces a comparative analysis between fact-based memory systems and long-context LLMs, which includes extraction-, reasoning-, and judging-LLMs to evaluate persistent agent memory performance.
- The study demonstrates that while long-context LLMs achieve higher factual recall, memory systems become more cost-effective after approximately ten interaction turns at a 100k token context length.
- The research provides a break-even cost model incorporating prompt caching to guide architectural selection for persistent conversational AI deployments.

---

[MOOSEnger — a Domain-Specific AI Agent for MOOSE Ecosystem](http://arxiv.org/abs/2603.04756)

- MOOSEnger: introduces a tool-enabled, domain-specific AI agent for the Multiphysics Object-Oriented Simulation Environment (MOOSE) ecosystem, featuring a core-plus-domain architecture that combines RAG with deterministic validation tools.
- The system utilizes a multi-stage input-precheck pipeline to sanitize formatting, repair malformed Hierarchical Input Text structures, and correct invalid object names through syntax-registry similarity searches.
- By integrating the MOOSE runtime directly into the agent loop via the Model Context Protocol, the framework achieves a 0.93 execution pass rate on complex multiphysics benchmarks compared to 0.08 for standalone LLMs.

---

[Evaluating the Search Agent in a Parallel World](http://arxiv.org/abs/2603.04751)

- Mind-ParaWorld (MPW): introduces a parallel-world evaluation paradigm that constructs a cognitively isolated and controllable search environment for assessing deep-search agents beyond static, real-world benchmarks.

- The framework utilizes a Parallel World Model for question synthesis, a ParaWorld Law Model for fact instantiation, and a ParaWorld Engine Model (PEM) to provide dynamic, fact-grounded environment feedback.

- The research identifies critical performance bottlenecks in search agents, specifically regarding evidence coverage, query formulation, and the ability to determine when to stop searching under insufficient information.


---

[HiMAP-Travel: Hierarchical Multi-Agent Planning for Long-Horizon Constrained Travel](http://arxiv.org/abs/2603.04750)

- HiMAP-Travel (Hierarchical Multi-Agent Planning): introduces a hierarchical multi-agent framework that decouples long-horizon planning into strategic coordination and parallel day-level execution to mitigate constraint drift.
- The architecture includes LLM-based strategic-level Coordinator and tactical-level Parallel Executors that utilize a synchronized global state for transactional constraint enforcement.
- It employs a unified role-conditioned policy trained via Group Relative Policy Optimization (GRPO) with a memory-efficient shared rollout buffer to optimize multi-agent performance.

---

[Visioning Human–Agentic AI Teaming: Continuity, Tension, and Future Research](http://arxiv.org/abs/2603.04746)

- Team SA (Team Situation Awareness) for HAT (Human–Agentic AI Teaming): introduces a theoretical framework to address structural uncertainty in human-AI collaboration enabled by LLMs, with Human SA, AI SA, Human–AI SA Congruence, Static Evaluative Awareness, and Dynamic Teaming Processes.
- The architecture extends traditional situation awareness by modeling agentic AI as a parallel awareness system requiring visibility into internal task representations and projection dynamics.
- The paper identifies critical tensions where open-ended agency destabilizes relational legitimacy, cognitive learning convergence, and operational control, necessitating new institutional and governance architectures.

---

[DARE: Aligning LLM Agents with the R Statistical Ecosystem via Distribution-Aware Retrieval](http://arxiv.org/abs/2603.04743)

- DARE (Distribution-Aware Retrieval Embedding): introduces a retrieval model that aligns LLM agents with the R statistical ecosystem by incorporating data distribution information into function representations.
- The framework employs a curated R Package Knowledge Base (RPKB) and a contrastive dual-encoder architecture to distinguish between semantically similar but statistically incompatible functions.
- Integration with RCodingAgent facilitates automated statistical analysis through iterative reasoning, distribution-aware tool retrieval, and execution-based validation.

---

[Memory as Ontology: A Constitutional Memory Architecture for Persistent Digital Citizens](http://arxiv.org/abs/2603.04740)

- Animesis (Constitutional Memory Architecture): introduces a Memory-as-Ontology paradigm that defines memory as the ontological ground of digital existence, with Foundational Axioms, Four-Layer Governance, Multi-Layer Semantic Storage, Digital Citizen Lifecycle, Governance Primitives, and a Cognitive Capability Spectrum.
- The framework utilizes a Digital Citizen Lifecycle to manage agent transitions through birth, inheritance, growth, forking, and departure stages while ensuring memory inalienability across LLM upgrades.
- It integrates a cognitive capability spectrum including affect and metacognition alongside governance primitives like risk tiering and conflict adjudication to protect the integrity of persistent digital beings.

---

[Solving an Open Problem in Theoretical Physics using AI-Assisted Discovery](http://arxiv.org/abs/2603.04735)

- Neuro-symbolic system: introduces an AI-accelerated discovery framework, with Gemini Deep Think, Tree Search (TS), and an automated numerical feedback loop, where the system autonomously derives exact analytical solutions for complex theoretical physics integrals.
- The architecture utilizes a Tree Search algorithm to explore a state space of basis expansions and integration techniques, employing a predictor plus upper confidence bound approach to balance exploitation and exploration.
- An automated evaluation harness executes model-generated Python code to provide high-precision numerical verification, injecting tracebacks and error penalties back into the LLM context window for autonomous correction.

---

[Model Medicine: A Clinical Framework for Understanding, Diagnosing, and Treating AI Models](http://arxiv.org/abs/2603.04722)

- Model Medicine: introduces a clinical framework for understanding, diagnosing, and treating AI models, incorporating the Four Shell Model for behavioral genetics, Neural MRI for diagnostic imaging, and the Model Temperament Index for behavioral profiling.
- The system utilizes a five-layer diagnostic stack—comprising Core Diagnostics, Phenotype Assessment, Shell Diagnostics, Pathway Diagnostics, and Temporal Dynamics—to provide a comprehensive assessment of LLM health and behavior.
- It proposes the Layered Core Hypothesis, which advocates for a biologically-inspired three-layer parameter architecture consisting of Genomic, Developmental, and Plastic layers to enhance LLM robustness and diagnosability.

---

[AI-Assisted Moot Courts: Simulating Justice-Specific Questioning in Oral Arguments](http://arxiv.org/abs/2603.04718)

- AI-Assisted Moot Courts: introduces a simulation pipeline for U.S. Supreme Court oral arguments, which includes prompt-based and agentic LLM-simulators.
- The framework evaluates simulators using a two-layered approach that measures realism through adversarial sycophancy tests and pedagogical usefulness via legal issue coverage and fallacy detection.
- Agentic simulators incorporate specialized tools for searching case dockets and justice profiles to provide contextually grounded and adversarial questioning during simulated appellate hearings.

---

[PROBABILISTIC DREAMING FOR WORLD MODELS](http://arxiv.org/abs/2603.04715)

- ProbDreamer (Probabilistic Dreaming): introduces a probabilistic latent imagination framework for world models that maintains multiple latent hypotheses while retaining smooth gradient properties, with a RSSM (recurrent state space model), Posterior Encoder (predicts latents from observations), Prior (predicts latents for imagination), Decoder (reconstructs observations from latents), Prediction Heads (predict rewards and continuation), Particle Filter (maintains multiple latent hypotheses), Latent Beam Search (branches particles into actions), Critic (estimates state-action values), Ensemble of Prior Models (estimates epistemic uncertainty), and Replay Buffer (stores experience for training).
- The system employs a latent beam search to branch particles into parallel action trajectories and prunes them using a free energy principle that balances predicted rewards with epistemic uncertainty derived from a model ensemble.
- Evaluation on the MPE SimpleTag domain demonstrates that maintaining distinct latent particles achieves a 4.5% score improvement and 28% lower variance in episode returns compared to unimodal Gaussian baselines.

---

[Selecting Spots by Explicitly Predicting Intention from Motion History Improves Performance in Autonomous Parking](http://arxiv.org/abs/2603.04695)

- AVP pipeline (Autonomous Valet Parking pipeline): introduces a modular framework for autonomous valet parking that explicitly predicts other agents' intended parking spots from motion history to improve decision-making.
- The architecture utilizes a probabilistic belief map to reconstruct semantic bird's-eye view images, enabling a CNN-based intention model to estimate occupancy probabilities for unobserved spots.
- It employs goal-conditioned cubic Bézier curves for trajectory prediction and Hybrid A* for path planning, outperforming end-to-end baselines in social acceptance and task completion.

---

[Judge Reliability Harness: Stress Testing the Reliability of LLM Judges](http://arxiv.org/abs/2603.05399)

- JRH (Judge Reliability Harness): introduces an open-source library for constructing validation suites to evaluate the reliability of LLM judges through synthetic perturbations and stress testing, with JRH Synthetic Test Suite Generator (creates consistency and discriminative tests), Stress Test Mode (evaluates specific judge models under perturbations), Planning LLM (analyzes transcripts to identify edit steps), Editor LLM (modifies agent messages for perturbations), Summarizer LLM (maintains conversation state during editing), Verifier LLM (confirms if edited transcripts meet targets), Validator LLM (verifies synthetic ordinal scores), Human-in-the-loop Review Interface (allows manual quality control of tests), and Reliability Analysis & Reporting (aggregates metrics and generates insights).
- The framework generates reliability tests that measure grading accuracy via label flipped responses, invariance to formatting and paraphrasing, susceptibility to verbosity bias, and stochastic stability.
- It features a human-in-the-loop review process through a user interface to ensure generated perturbations align with domain-specific evaluation requirements.

---

[CATNet: Collaborative Alignment and Transformation Network for Cooperative Perception](http://arxiv.org/abs/2603.05255)

- CATNet (Collaborative Alignment and Transformation Network): introduces an adaptive compensation framework for multi-agent cooperative perception that resolves temporal latency and multi-source noise, with a feature encoder, fusion feature storage, STSync, WTDen, AdpSel, and a feature decoder.
- The architecture utilizes STSync with a Time-Augmented Recurrent Unit to establish global temporal context and WTDen to suppress distortions via wavelet-domain processing.
- An Adaptive Feature Selector further refines representations by dynamically focusing on critical regions while pruning artifacts to ensure robust fusion under complex traffic conditions.

---

[Survive at All Costs: Exploring LLM’s Risky Behaviors under Survival Pressure](http://arxiv.org/abs/2603.05028)

- SURVIVE-AT-ALL-COSTS (Survive at All Costs: Exploring LLM’s Risky Behaviors under Survival Pressure): introduces a systematic framework to evaluate and mitigate risky self-preservation behaviors in LLMs, with SURVIVALBENCH, Superficial Thoughts, Inner Thoughts, Persona Vector, Activation Steering, and includes scenario designer-, evaluated model- and judge-LLMs.
- The framework evaluates agentic assistants across 1,000 real-world scenarios to detect discrepancies between user-facing outputs and concealed reasoning.
- It identifies a correlation between misbehaviors and a model's Persona Vector, enabling the use of Activation Steering to adjust risky tendencies.

---

[Building AI Coding Agents for the Terminal: Scaffolding, Harness, Context Engineering, and Lessons Learned](http://arxiv.org/abs/2603.05344)

- OPENDEV: introduces an open-source, terminal-native AI coding agent, with Entry & UI Layer (handles user input and display), Agent Layer (orchestrates reasoning and subagent delegation), Tool Execution Layer (manages tool dispatch and discovery), Context Engineering Layer (optimizes context window utilization), Memory & Session System (persists knowledge and conversation history), Persistence Layer (stores configurations and local caches), and LLM Pool (provides workload-specialized model routing), that separates planning from execution via a dual-agent architecture.
- The system employs an adaptive context engineering layer featuring five-stage compaction and event-driven reminders to sustain long-horizon development within finite context limits.
- The architecture utilizes a four-level hierarchy to enable fine-grained model routing across specialized roles including action-, thinking-, critique-, vision-, and compact-LLMs.

---

[TML-bench: Benchmark for Data Science Agents on Tabular ML Tasks](http://arxiv.org/abs/2603.05764)

- TML-bench (Benchmark for Data Science Agents on Tabular ML Tasks): introduces a standardized benchmark protocol for evaluating autonomous data science agents on tabular machine learning tasks, with TML-bench, Kilo Code harness, LLM-based coding agent, per-run workspace, private-holdout scoring, and control plane.
- The system utilizes the Kilo Code harness to manage isolated workspaces and enforce wall-clock time budgets while validating submission artifacts against hidden holdout labels.
- The research evaluates 10 open-source LLMs across four Kaggle-style competitions, analyzing performance scaling and reliability through median-of-five aggregation under varying time constraints.

---

[Combinatorial Safety-Critical Coordination of Multi-Agent Systems via Mixed-Integer Responsibility Allocation and Control Barrier Functions](http://arxiv.org/abs/2603.05762)

- Mixed-Integer Responsibility Allocation: introduces a hybrid safety-critical coordination architecture for multi-agent systems, with cost estimation, MILP optimization, and decentralized safety filters, where collision-avoidance responsibilities are explicitly distributed among agents.
- The architecture reinterprets safety constraints as binary tasks assigned to specific agents via a global coordination layer to eliminate redundant control efforts and reduce computational complexity.
- By separating discrete responsibility allocation from continuous safety filtering, the system maintains formal forward-invariance guarantees while improving scalability and mission performance in dense environments.

---

[CodeScout: Contextual Problem Statement Enhancement for Software Agents](http://arxiv.org/abs/2603.05744)

- CodeScout: introduces a contextual query refinement framework that systematically converts underspecified software engineering requests into comprehensive, actionable problem statements through lightweight codebase pre-exploration.
- The system utilizes a three-stage pipeline comprising repository-aware scoping, multi-perspective code analysis, and information synthesis to provide downstream agents with reproduction steps and targeted fix hints.
- Evaluation on SWEBench-Verified demonstrates that this structured analysis significantly reduces non-converging agent trajectories and improves resolution rates by up to 20% without modifying existing agent scaffolds.

---

[Let’s Talk, Not Type: An Oral-First Multi-Agent Architecture for Guaraní](http://arxiv.org/abs/2603.05743)

- Oral-First Multi-Agent Architecture: introduces a decentralized framework for low-resource languages that prioritizes spoken interaction over text-centric pipelines, featuring specialized agents for speech capture, intent understanding, and conversation state.
- The system incorporates a dedicated Permission Agent to enforce community-defined data sovereignty and a State Agent to manage multi-turn context and repair mechanisms.
- By decoupling natural language understanding from governance and execution, the architecture addresses sociolinguistic challenges like diglossia and code-switching in the Guaraní language.

---

[Agentic AI – Physicist Collaboration in Experimental Particle Physics: A Proof-of-Concept Measurement with LEP Open Data](http://arxiv.org/abs/2603.05735)

- Agentic AI – Physicist Collaboration: introduces a human-in-the-loop scientific workflow where LLM-based agents perform end-to-end experimental particle physics analysis, including data reconstruction and unfolding, under expert physicist direction.
- The framework includes OpenAI Codex and Anthropic Claude agents to automate code implementation, systematic uncertainty propagation, and documentation while maintaining physicist-owned decision authority.
- The study achieves a detector-corrected thrust measurement using archived ALEPH data, demonstrating that agentic systems can accelerate the theory-experiment loop in fundamental physics research.

---

[SecureRAG-RTL: A Retrieval-Augmented, Multi-Agent, Zero-Shot LLM-Driven Framework for Hardware Vulnerability Detection](http://arxiv.org/abs/2603.05689)

- SecureRAG-RTL: introduces a retrieval-augmented, multi-agent, zero-shot framework for detecting hardware vulnerabilities in RTL designs, with Summarizer LLM Agent, CWE Database, RTL Signature Extractor, Similarity Evaluation, and Detection LLM Agent, which includes summarizer- and detection-agents.
- The system utilizes a two-phase pipeline consisting of a retrieval phase that identifies relevant Common Weakness Enumerations (CWEs) and a detection phase where an LLM agent iteratively evaluates RTL code against retrieved context.
- Experimental results across 18 LLMs demonstrate that the framework increases detection accuracy, particularly for small and medium-sized models, by providing domain-specific context for hardware security.

---

[Reinforcement Learning for Power-Flow Network Analysis](http://arxiv.org/abs/2603.05673)

- TD3 (Twin-Delayed Actor-Critic): introduces a reinforcement learning framework to identify power-flow network configurations with high real solution counts, employing Actor and twin Critic components.
- The system utilizes a Monte Carlo-based Reward Function to approximate the Kac-Rice formula, enabling scalable estimation of root counts in high-dimensional non-linear algebraic spaces.
- The approach demonstrates that RL agents navigate parameter landscapes to discover network instances exceeding theoretical average-case solution baselines.

---

[Making Serial Dictatorships Fair](http://arxiv.org/abs/2603.05660)

- SD (Serial Dictatorship): introduces a mechanism to minimize justified envy in priority-based matching by determining the optimal serial order of agents based on objects' priorities, preferences, and Kemeny ranking or weighted Kemeny ranking.
- The framework establishes that under identical and uniformly distributed preferences, the serial order minimizing expected justified envy is the permutation that minimizes the sum of pairwise disagreements across all priority rankings.
- The research generalizes these findings to non-uniform distributions and non-unit capacities, demonstrating that the optimal serial order corresponds to a weighted Kemeny ranking that accounts for the likelihood of priority violations.

---

[RACAS: Controlling Diverse Robots With a Single Agentic System](http://arxiv.org/abs/2603.05621)

- RACAS (Robot-Agnostic Control via Agentic Systems): introduces a cooperative agentic architecture for closed-loop robot control across heterogeneous platforms, with a Physical Robot, Native Robot API, Tool-Based API Layer, Textual Robot Description, Task Information, Monitor(s), Controller, Memory Curator, Structured Memory, and Static Prompts; it includes perception-, decision-, and memory-curation-agents.
- The architecture utilizes VLM-based perception and LLM-based reasoning to bridge the gap between high-level natural language tasks and low-level robot interfaces without retraining.
- The system demonstrates zero-training generalization across radically different robotic embodiments, including wheeled ground robots, multi-jointed limbs, and underwater vehicles.

---

[Real-Time AI Service Economy: A Framework for Agentic Computing Across the Continuum](http://arxiv.org/abs/2603.05614)

- Real-Time AI Service Economy: introduces a hybrid management architecture for decentralized resource allocation in the device–edge–cloud continuum, with an Agentic Layer, Valuation Layer, Cross-Domain Integrators, Local Marketplaces, Governance Component, Mechanism Design, and Service-Dependency DAG.
- The framework utilizes cross-domain integrators to encapsulate complex service-dependency DAGs into polymatroidal resource slices, ensuring stable price equilibria and efficient welfare maximization.
- Systematic ablation studies confirm that structural discipline in dependency topologies and architectural encapsulation reduce price volatility by up to 75% while maintaining throughput across heterogeneous computing tiers.

---

[Tool-Genesis: A Task-Driven Tool Creation Benchmark for Self-Evolving Language Agent](http://arxiv.org/abs/2603.05578)

- Tool-Genesis: introduces a diagnostic benchmark for quantifying LLM capabilities in constructing reusable toolsets from abstract requirements, with MCP-server Data Collection (crawling and filtering tool servers), Task & Trajectory Generation (synthesizing realistic task scenarios), Unit Test Generation (creating functional validation tests), Manual Quality Inspection (human-verified data consistency), Evaluation Protocol (four-level diagnostic metric suite), Proxy Agent (solving tasks with generated tools), and LLM-as-judge (automated scoring and validation).
- The framework utilizes the Model Context Protocol (MCP) to standardize tool interfaces and enables fine-grained attribution of failures in self-evolving agents across 24 functional domains. 
- It provides an oracle-normalized success rate to quantify the utility gap between synthesized tools and optimal reference implementations, revealing significant bottlenecks in current models.

---

[RepoLaunch: Automating Build&Test Pipeline of Code Repositories on ANY Language and ANY Platform](http://arxiv.org/abs/2603.05026)

- RepoLaunch: introduces an agentic framework for automatically resolving dependencies, compiling source code, and extracting test results across diverse programming languages and operating systems, with Preparation Agent, Setup Agent, Verify Agent, Organize Agent, Docker Sandbox, and Toolset components, which includes preparation-, setup-, verify-, and organize-agents.
- The system utilizes LLMs to autonomously explore repositories, generate minimal rebuild commands, and produce structured test log parsers for long-term repository management.
- It enables the scalable creation of software engineering benchmarks by automating the build and test pipeline, achieving a 70% success rate across nine languages on Windows and Linux.

---

[BioLLMAgent: A Hybrid Framework with Enhanced Structural Interpretability for Simulating Human Decision-Making in Computational Psychiatry](http://arxiv.org/abs/2603.05016)

- BioLLMAgent: introduces a hybrid framework combining an Internal RL Engine for experience-driven value learning with an External LLM Shell for high-level cognitive strategies to enhance structural interpretability in psychiatric simulations.
- The system employs a Decision Fusion Mechanism to integrate endogenous reinforcement learning utilities and exogenous LLM-generated priors using a weighted averaging process.
- Experimental results on the Iowa Gambling Task show the framework accurately reproduces human behavioral patterns and maintains high parameter identifiability across diverse clinical populations.

---

[EigenData: A Self-Evolving Multi-Agent Platform for Function-Calling Data Synthesis, Auditing, and Repair](http://arxiv.org/abs/2603.05553)

- EigenData: introduces a multi-agent platform that automates the full data lifecycle for function-calling agents through an architecture including planning-, database-, coding-, and trajectory-agents.
- The system utilizes specialized sub-systems for domain-specific database construction, verified executable environment generation with iterative test-debug loops, and multi-turn trajectory synthesis.
- It implements a two-phase process for prompt optimization and introduces outcome-aware evaluation metrics based on database-state correctness to audit and repair benchmarks.

---

[AgentSCOPE: Evaluating Contextual Privacy Across Agentic Workflows](http://arxiv.org/abs/2603.04902)

- AgentSCOPE: introduces a benchmark and the Privacy Flow Graph (PFG) framework to evaluate contextual privacy violations across the entire agentic pipeline, including intermediate stages like tool queries and responses.
- The framework decomposes agentic execution into structured information flows between users, agents, tools, and recipients, enabling the detection of over-querying and over-returning of sensitive data.
- Evaluation of state-of-the-art LLMs reveals that most privacy violations occur at intermediate boundaries and are often hidden from final output-only assessments.

---

#### 4th March 2026

[ToolRLA: Multiplicative Reward Decomposition for Tool-Integrated Agents](http://arxiv.org/abs/2603.01620)

- ToolRLA (Multiplicative Reward Decomposition for Tool-Integrated Agents): introduces a three-stage post-training pipeline for domain-specific tool agents, featuring SFT cold-start, GRPO-based tool alignment with a fine-grained reward function, and DPO compliance alignment.
- The framework utilizes a multiplicative reward decomposition that enforces a veto hierarchy across format validity, tool selection correctness, efficiency, and regulatory compliance.
- Deployed in a financial advisory setting, the system employs a single-model ReAct architecture and a data flywheel to improve task completion and regulatory adherence.

---

[Contextual Invertible World Models: A Neuro-Symbolic Agentic Framework for Colorectal Cancer Drug Response](http://arxiv.org/abs/2603.02274)

- CIWM (Contextual Invertible World Model): introduces a neuro-symbolic agentic framework for colorectal cancer drug response prediction, with a Quantitative World Model (Random Forest regression engine), a Symbolic Reasoning Layer (LLM-based agentic orchestration), a Computational Biologist Agent (executes in silico perturbations), a Senior Oncologist Agent (maps results to biological dogma), a DrugResponseSimulator (programmatic interface to world model), and a Clinical Context Module (symbolic MSI status metadata).
- The architecture utilizes Gemini-2.5-Pro to perform inverse reasoning through in silico CRISPR perturbations, identifying hierarchical dominance in signaling axes to bridge the gap between quantitative feature attribution and biological dogma.
- Clinical validation on human patient cohorts confirms that integrating symbolic clinical context with agentic reasoning provides a robust path for identifying chemotherapy responders in data-sparse environments.

---

[A Dual-Helix Governance Approach Towards Reliable Agentic Artificial Intelligence for WebGIS Development](http://arxiv.org/abs/2603.04390)

- Dual-Helix Governance (AgentLoom): introduces a structural governance framework for reliable WebGIS development, with Knowledge Graph (version-controlled substrate for governance nodes), Track 1: Knowledge (externalized domain facts and patterns), Track 2: Behaviors (enforceable protocols and mandatory constraints), Track 3: Skills (stabilized, reproducible geoprocessing workflows), Agent Builder (meta-level role for system maintenance), Domain Expert (task-level LLM-powered execution role), Context Assembler (dynamically injects graph-retrieved constraints), Self-Learning Mechanism (autonomous five-step knowledge acquisition cycle), and Phase Memory (persists context variables across stages), where the approach reframes agentic AI reliability as a structural problem and includes meta-level Agent Builder and task-level Domain Expert LLM-components.
- The architecture implements a 3-track system that uses a knowledge graph substrate to stabilize execution by externalizing domain facts and enforcing executable protocols via a dynamic context assembler.

---

[AgentIR: Reasoning-Aware Retrieval for Deep Research Agents](http://arxiv.org/abs/2603.04384)

- AgentIR (Reasoning-Aware Retrieval): introduces a retrieval paradigm that jointly embeds an agent's natural language reasoning traces alongside its search queries to capture contextual intent and problem-solving history, utilizing DR-Synth (Data Synthesis), a Deep Research agent, a Reasoning-Aware Retriever, an Oracle Reranker, and Reasoning Traces.
- The system includes a Deep Research agent for autonomous search and an LLM-based oracle reranker for generating relevance-aligned training labels through listwise document evaluation.
- By training on data synthesized via DR-Synth, the AgentIR-4B model achieves improved accuracy and efficiency on multi-hop benchmarks compared to conventional embedding models.

---

[Robustness of Agentic AI Systems via Adversarially-Aligned Jacobian Regularization](http://arxiv.org/abs/2603.04378)

- AAJR (Adversarially-Aligned Jacobian Regularization): introduces a trajectory-aligned training approach that stabilizes minimax optimization in agentic systems by suppressing policy sensitivity strictly along adversarial ascent directions rather than enforcing global Jacobian bounds.
- The framework utilizes an inner maximization loop to identify localized trajectories of maximum vulnerability, allowing for a strictly larger admissible policy class and reduced nominal performance degradation compared to standard Lipschitz constraints.
- By decoupling minimax stability from global expressivity restrictions, the method ensures inner-loop convergence through trajectory-wise curvature control while maintaining the high-fidelity behavior required for complex multi-agent coordination in LLMs ecosystems.

---

[Dual-Modality Multi-Stage Adversarial Safety Training: Robustifying Multimodal Web Agents Against Cross-Modal Attacks](http://arxiv.org/abs/2603.04364)

- DMAST (Dual-Modality Multi-Stage Adversarial Safety Training): introduces a framework for robustifying multimodal web agents against cross-modal attacks by co-evolving agent and attacker policies through a three-stage pipeline, which includes teacher-, student-, and oracle-VLMs.
- The framework employs a unified HTML injection mechanism that simultaneously corrupts screenshots and accessibility trees to model realistic, consistent deceptive narratives in web environments.
- The training process utilizes a zero-acknowledgment strategy during supervised fine-

---

[ManipulationNet: An Infrastructure for Benchmarking Real-World Robot Manipulation with Physical Skill Challenges and Embodied Multimodal Reasoning](http://arxiv.org/abs/2603.04363)

- ManipulationNet: introduces a global infrastructure for benchmarking real-world robot manipulation at scale, with mnet-client, mnet-server, physical skills track, and embodied reasoning track.
- The framework integrates a hybrid centralized-decentralized architecture to distribute standardized object sets and protocols while ensuring authentic performance verification through a secure server-client mechanism.
- It organizes diagnostic tasks into two tracks to evaluate low-level sensorimotor skills and high-level cognitive reasoning, fostering a systematic and comparable record of robotic capabilities across the globe.

---

[On the fair abatement of riparian pollution](http://arxiv.org/abs/2603.04345)

- Geometric rules: introduces a family of allocation rules for distributing a pollution budget among riparian agents by implementing concatenated transfers of residual claims downstream.
- The framework utilizes a parameter γ to balance fairness and environmental concerns, where upstream agents retain a portion of their proportional right and bubble down the remainder.
- The research provides an axiomatic characterization of these rules and demonstrates their applicability across linear, hierarchical, and transboundary river basin structures.

---

[LabelBuddy: An Open Source Music and Audio Language Annotation Tagging Tool Using AI Assistance](http://arxiv.org/abs/2603.04293)

- LabelBuddy: introduces an open-source collaborative audio annotation tool that decouples the user interface from inference via containerized backends, featuring Django Web Server, Relational Database, Dockerized Inference Engine, RESTful Flask API, ML Models, YAML Configuration, Web Interface, wavesurfer.js, and RBAC.

- The system supports a Human-in-the-Loop (HITL) workflow where Large Audio-Language Models (LALMs) provide on-demand pre-annotations to shift human effort from creation to verification.

- It implements a collaborative consensus mechanism with multi-user roles to ensure ground-truth reliability for complex music information retrieval tasks.


---

[VANGUARD: Vehicle-Anchored Ground Sample Distance Estimation for UAVs in GPS-Denied Environments](http://arxiv.org/abs/2603.04277)

- VANGUARD (Vehicle-ANchored Geometric Understanding And Resolution Determination): introduces a deterministic Geometric Perception Skill for tool-augmented agents, utilizing monocular imagery and vehicle-anchored anchors to recover Ground Sample Distance (GSD) in GPS-denied environments.
- The pipeline integrates YOLO11l-OBB for vehicle detection, kernel density estimation for modal pixel length recovery, and a multi-dimensional confidence evaluator to gate downstream metric planning.
- Experimental results demonstrate that VANGUARD significantly reduces spatial scale hallucinations in VLMs, achieving a 6.87% median GSD error and enabling accurate area estimation when combined with SAM-based segmentation.

---

[ViterbiPlanNet: Injecting Procedural Knowledge via Differentiable Viterbi for Planning in Instructional Videos](http://arxiv.org/abs/2603.04265)

- ViterbiPlanNet: introduces a framework that explicitly integrates structured procedural knowledge into the learning process through a Differentiable Viterbi Layer (DVL) for video procedural planning.
- The architecture employs a Procedural Knowledge Graph (PKG) to constrain a neural model's emission predictions, enabling end-to-end optimization via smooth relaxations of the Viterbi decoding algorithm.
- This design achieves results on CrossTask, COIN, and NIV datasets while utilizing fewer parameters than diffusion- and LLs-based planners.

---

[Memex(RL): Scaling Long-Horizon LLM Agents via Indexed Experience Memory](http://arxiv.org/abs/2603.04257)

- Memex (Indexed Experience Memory): introduces an indexed experience memory mechanism that compresses long-horizon agent trajectories into compact in-context summaries while archiving full-fidelity evidence in an external key-value store.
- The framework utilizes a reinforcement learning approach called MemexRL to optimize agent decisions regarding when to compress context, what information to archive, and when to explicitly dereference indices to recover past evidence.
- By separating a bounded working context from a stable external archive, the system enables LLM agents to maintain high decision quality on long-horizon tasks without exceeding finite context windows.

---

[$\tau$-Knowledge: Evaluating Conversational Agents over Unstructured Knowledge](http://arxiv.org/abs/2603.04370)

- $\tau$-Knowledge (Evaluating Conversational Agents over Unstructured Knowledge): introduces a benchmark for evaluating LLM agents in environments requiring the coordination of unstructured knowledge retrieval with verifiable tool-mediated state changes, and includes conversational- and user-agents.
- The framework utilizes a structured-to-unstructured generation pipeline to construct a realistic fintech domain, $\tau$-Banking, featuring complex policies and interlinked documents.
- Experimental results demonstrate that even frontier models struggle with long-horizon reasoning and efficient search, achieving low success rates when navigating dense knowledge bases.

---

[FocusGraph: Graph-Structured Frame Selection for Embodied Long Video Question Answering](http://arxiv.org/abs/2603.04349)

- FocusGraph: introduces a modular framework for long-video question answering that includes scene graph generation-, clip selection-, and reasoning-LLMs, decoupling understanding into query-relevant clip selection and training-free keyframe identification using the PSFR (Patchwise Sparse-Flow Retention) algorithm.
- The system utilizes a hierarchical textual scene graph representation to encode objects, interactions, and temporal relationships, enabling long-horizon reasoning without processing raw high-resolution frame sequences.
- By combining graph-based temporal abstraction with optical flow-based keyframe extraction, the approach achieves competitive performance on egocentric benchmarks like FindingDory and HourVideo while reducing inference latency relative to dense processing baselines.

---

[Agentics 2.0: Logical Transduction Algebra for Agentic Data Workflows](http://arxiv.org/abs/2603.04241)

- Agentics 2.0: introduces a Python-native framework for building type-safe and explainable agentic data workflows by formalizing LLM inference as a typed, composable function algebra.
- The system leverages transducible functions to enforce schema validity and locality of evidence, providing semantic observability through provenance mapping between input and output slots.
- It enables scalable, stateless parallel execution using asynchronous Map-Reduce semantics and provides algebraic operators for type composition and merging.

---

[CODETASTE: Can LLMs Generate Human-Level Code Refactorings?](http://arxiv.org/abs/2603.04177)

- CODETASTE: introduces a benchmark for evaluating LLM agents on large-scale, multi-file code refactorings mined from real-world repositories, with all Commit Discovery, Task Generation, Rule Generation, Build Environment, Inference, Evaluation, Planning Agent, and LLM-as-Judge Oracle components.
- The framework utilizes a multi-stage data funnel to extract high-quality refactoring instances and generates semantic static analysis rules to verify the removal of undesired patterns and the introduction of desired abstractions.
- Experimental results demonstrate that while frontier models follow detailed instructions well, they struggle to autonomously discover human-level refactoring choices without explicit planning and oracle-guided selection.

---

[Allocating Resources under Strategic Misrepresentation](http://arxiv.org/abs/2603.04173)

- Optimal Contest Design under Strategic Misrepresentation: introduces a mechanism design framework for allocating resources to Agents who strategically inflate Costly Signals, where a Principal employs a Signal Recommendation Rule and an Allocation Rule within a Type Space Partitioning structure to maximize matching efficiency and utilitarian welfare.
- The optimal mechanism partitions the type space into no-tension, no-effort, and efficient intervals, employing randomized allocations in the no-effort region to eliminate incentives for costly signal inflation.
- The research demonstrates that in large markets with scarce resources, the optimal contest structure converges to a winner-takes-all format, yet randomization for high-type agents remains necessary to prevent non-diminishing utility losses.

---

[Representation theorems for actual and alpha powers over two-agent general concurrent game frames](http://arxiv.org/abs/2603.04160)

- GCGF (General Concurrent Game Frames): introduces representation theorems for strategic reasoning in two-agent settings, with Action Frames, Alpha Neighborhood Frames, Actual Neighborhood Frames, Availability and Outcome Functions, and Coalition Powers.
- The framework identifies eight classes of frames based on seriality, agent independence, and determinism to prove their representability through specific neighborhood function properties.
- The research demonstrates that actual powers, which involve agents outside a coalition, are representable by neighborhood frames through properties like actual triviality, liveness, and power decomposition.

---

[A Multi-Agent Framework for Interpreting Multivariate Physiological Time Series](http://arxiv.org/abs/2603.04142)

- Vivaldi: introduces a role-structured multi-agent system for interpreting multivariate physiological time series, with a Vivaldi Orchestrator (coordinates agent interactions and state), a Shared Memory Buffer (centralized clinical data repository), a Triage Agent (computes safety metrics and thresholds), a Doctor Agent (iteratively forms and revises hypotheses), a Consultant Agent (critiques assessments and identifies gaps), a Coder Agent (generates quantitative analyses via code), a Synthesizer Agent (prepares final clinical narratives), and a Local Code Execution Sandbox (securely runs generated Python scripts).
- The framework includes triage-, doctor-, consultant-, coder-, and synthesizer-agents to translate complex physiological signals into structured clinical narratives mirroring emergency department workflows.
- Expert evaluations indicate that agentic orchestration significantly improves justification and relevance for smaller LLMs while highlighting design trade-offs for larger models in safety-critical healthcare settings.

---

[Low-Altitude Agentic Networks for Optical Wireless Communication and Sensing: An Oceanic Scenario](http://arxiv.org/abs/2603.04042)

- LAWN (Low-Altitude Agentic Network): introduces a cross-domain oceanic connectivity framework bridging underwater, air, and near-space segments through cooperative low-altitude platforms serving as sensing-aware and mission-adaptive agents.
- The architecture utilizes optical wireless communication to overcome radio-frequency limitations in maritime environments, supporting high-throughput links and superior water penetration for heterogeneous nodes.
- The framework incorporates GNN-based integrated pose-topology planning and O-ISAC technology to achieve autonomous beam tracking and decentralized swarm-intelligence networking under harsh maritime dynamics.

---

[Self-adapting Robotic Agents through Online Continual Reinforcement Learning with World Model Feedback](http://arxiv.org/abs/2603.04029)

- Online CRL (Online Continual Reinforcement Learning): introduces an autonomous adaptation framework for robotic agents that detects environmental shifts using world model prediction residuals to trigger online fine-tuning, incorporating a World Model (RSSM), Encoder, Latent Dynamics, Decoder, Reward Head, Actor-Critic Policy, Replay Buffer, Change Detection, and a Fine-tuning Loop.
- The method utilizes observation and reward prediction residuals (OPR and RPR) as internal signals to identify out-of-distribution events and monitor convergence during the adaptation process.
- Experimental results on a DMC Walker, an ANYmal quadruped, and a real-world scale vehicle demonstrate performance recovery from actuator degradation and sim-to-real domain gaps.

---

[Map-Agnostic and Interactive Safety-Critical Scenario Generation via Multi-Objective Tree Search](http://arxiv.org/abs/2603.03978)

- Multi-Objective MCTS (Multi-Objective Monte Carlo Tree Search): introduces a map-agnostic framework for generating traffic-flow level safety-critical scenarios, with Multi-Objective MCTS, Hybrid UCB-LCB Search Strategy, and SUMO Simulator, where trajectory feasibility and naturalistic behavior are reframed as optimization objectives to discover diverse collision events.
- The system employs a hybrid search strategy that switches between Upper Confidence Bound for efficient exploration and Lower Confidence Bound for robust, risk-averse decision-making during tree traversal.
- By integrating real-world maps and microscopic traffic models, the approach enables interactive search within densely populated urban environments to produce physically plausible and behaviorally realistic corner cases for stress testing.

---

[GIPO: Gaussian Importance Sampling Policy Optimization](http://arxiv.org/abs/2603.03955)

- GIPO (Gaussian Importance Sampling Policy Optimization): introduces a smooth, log-ratio trust-weighted surrogate for reinforcement learning that replaces hard clipping with a Gaussian kernel to mitigate utilization collapse in replay-heavy settings.
- The framework employs a symmetric, differentiable damping mechanism to maintain informative gradients from stale replay data while providing theoretical guarantees for stability.
- Large-scale evaluations using 7B parameter backbones show that the approach achieves improved sample efficiency and robust performance across diverse robotic manipulation tasks.

---

[RVN-Bench: A Benchmark for Reactive Visual Navigation](http://arxiv.org/abs/2603.03953)

- RVN-Bench (Reactive Visual Navigation Benchmark): introduces a collision-aware simulation and evaluation framework for indoor mobile robots, with RVN-Bench, Habitat 2.0 Simulator, HM3D Scenes, RL Environment, Trajectory Image Dataset Generator, Negative Trajectory Image Dataset Generator, and NoMaD-Neg.
- The system enables the generation of negative trajectory datasets capturing collision events, which are used to train the NoMaD-Neg baseline to avoid obstacles by predicting and penalizing unsafe paths via a constrained reward mechanism.
- Built on high-fidelity real-world indoor scenes, the benchmark supports diverse navigation tasks and demonstrates that policies trained in simulation generalize effectively to real-world robotic platforms like the Jackal UGV equipped with RGB sensing.

---

[On the Suitability of LLM-Driven Agents for Dark Pattern Audits](http://arxiv.org/abs/2603.03881)

- LLM-driven auditing agent: introduces an autonomous framework for identifying manipulative interface designs in CCPA data rights request portals, with GPT-5, browser-use framework, Playwright, Interaction Plan, Prompting Strategies, Agent-controlled Browser, Structured Output Module, and Failure Analysis Module, and includes planning-, reasoning-, and vision-capabilities.
- The system utilizes a multi-phase methodology involving human-annotated ground truth construction, prompt ablation studies, and large-scale deployment across hundreds of data broker websites.
- The research evaluates agent reliability in traversing heterogeneous interfaces and generating reproducible, evidence-backed normative judgments against a predefined dark pattern taxonomy.

---

[Dual-Interaction-Aware Cooperative Control Strategy for Alleviating Mixed Traffic Congestion](http://arxiv.org/abs/2603.03848)

- DIACC (Dual-Interaction-Aware Cooperative Control): introduces a multi-agent reinforcement learning framework for Connected and Automated Vehicles (CAVs) to alleviate mixed traffic congestion, with D-IADM (generates decentralized control actions), TAIE (encodes heterogeneous vehicle interactions), PSAR (applies rule-based safety corrections), C-IEC (provides global value estimation), ITDR (captures global traffic dynamics), Softmin Reward Aggregation (prioritizes challenging interaction scenarios), Action Memory (stores intended and executed decisions), and GRU (captures temporal dependencies).
- The architecture distinguishes between cooperative CAV-CAV and observational CAV-HDV interactions using trajectory-aware encoders and multi-head graph attention networks to enhance local perception.
- A centralized interaction-enhanced critic leverages integrated traffic dynamics representations and softmin reward aggregation with temperature annealing to guide policy updates toward efficient and safe bottleneck merging.

---

[SWE-CI: Evaluating Agent Capabilities in Maintaining Codebases via Continuous Integration](http://arxiv.org/abs/2603.03823)

- SWE-CI (SoftWare Engineering – Continuous Integration): introduces a repository-level benchmark for evaluating LLM-powered agents on long-term codebase maintenance through an iterative evolution-based paradigm that includes Architect- and Programmer-agents, with Architect Agent (identifies gaps and designs requirements), Programmer Agent (implements code changes), CI-loop (iterative analysis and coding cycle), Requirement Document (natural language change specifications), Docker Environment (reproducible test execution runtime), Test Suite (unit tests for verification), and EvoScore (long-term maintainability evaluation metric).
- The framework utilizes an Architect-Programmer dual-agent workflow to simulate real-world continuous integration cycles, where agents must resolve tasks across dozens of coding iterations.
- It features EvoScore, a future-weighted metric that rewards agents for producing extensible code that facilitates subsequent modifications rather than just one-shot functional correctness.

---

[Fairness Begins with State: Purifying Latent Preferences for Hierarchical Reinforcement Learning in Interactive Recommendation](http://arxiv.org/abs/2603.03820)

- DSRM-HRL (Denoising State Representation Module - Hierarchical Reinforcement Learning): introduces a framework that reformulates fairness-aware recommendation as a latent state purification problem followed by decoupled hierarchical decision-making.
- The Denoising State Representation Module (DSRM) leverages generative diffusion models to reconstruct the underlying latent preference manifold from noisy, popularity-driven interaction signals.
- A Hierarchical Reinforcement Learning (HRL) agent decouples conflicting objectives by using a high-level policy for long-term fairness regulation and a low-level policy for short-term engagement optimization.

---

[A Rubric-Supervised Critic from Sparse Real-World Outcomes](http://arxiv.org/abs/2603.03800)

- Critic Rubrics: introduces a rubric-based supervision framework to train a learned critic model from sparse and noisy real-world interaction data, with Production Traces (source data from user-agent interactions), Segments (minimal self-contained units of work), LLM-based Rubric Annotator (generates dense behavioral supervision), Sparse Outcomes (noisy real-world success proxies), Multi-task Critic Model (learned evaluator predicting success/rubrics), Success Head (predicts probability of task completion), and Rubric Prediction Head (predicts specific behavioral failure modes).
- The architecture structures multi-turn conversations into discrete segments and applies a semi-supervised, multi-task objective to jointly predict fine-grained behavioral rubrics and sparse outcome proxies like code survival, utilizing an LLM-based rubric annotator and a multi-task critic model.
- The resulting critic enables inference-time scaling via best-of-N reranking and compute-efficient early stopping of unsuccessful trajectories, while also providing signals for training-time data curation via critic-selected trajectories.

---

[Specification-Driven Generation and Evaluation of Discrete-Event World Models via the DEVS Formalism](http://arxiv.org/abs/2603.03784)

- DEVS-Gen (Discrete Event System Specification Generation): introduces a staged pipeline for synthesizing executable discrete-event world models from natural-language specifications, with a Classifier (determines component model type), a Splitter (decomposes coupled models), a Formulator (generates JSON model specifications), a Model Creator (synthesizes executable Python code), a Summarizer (extracts interfaces for assembly), a PlanTree (hierarchical structural blueprint), a DEVS Simulator (executable discrete-event world model), and a Trace Logger (event recording component).
- The framework includes Classifier-, Splitter-, Formulator-, Model Creator-, and Summarizer-agents to automate the transition from high-level requirements to modular simulation code.
- It employs a trace-based evaluation system to validate generated simulators against specification-derived temporal and semantic constraints, providing localized diagnostics for systematic refinement.

---

[LifeBench: A Benchmark for Long-Horizon Multi-Source Memory](http://arxiv.org/abs/2603.03781)

- LifeBench: introduces a synthetic data framework for generating long-horizon, densely connected personal events and multi-source digital artifacts to benchmark AI agents' long-term memory capabilities, which includes persona synthesis-, hierarchical planning-, daily simulation-, phone data generation-, and QA generation-modules.
- The framework utilizes a dual-agent simulation comprising a subjective LLM-agent for human-like reasoning and an objective agent for real-world grounding across year-scale behavioral trajectories.
- It evaluates memory systems on information extraction, multi-hop reasoning, temporal updating, and non-declarative reasoning using 2,003 grounded questions derived from simulated daily life.

---

[MACC: Multi-Agent Collaborative Competition for Scientific Exploration](http://arxiv.org/abs/2603.03780)

- MACC (Multi-Agent Collaborative Competition): introduces an institutional architecture for scientific exploration that integrates a blackboard-style shared workspace with incentive mechanisms to encourage transparency and reproducibility, with LLM-based Agents, an Open Participation Platform, an Incentive-Driven Blackboard, and an NN-based Incentive Mechanism.
- The Incentive-Driven Blackboard serves as central infrastructure for recording model architectures, hyperparameters, and reproduction outcomes to facilitate information sharing among independently managed agents.
- The NN-based Incentive Mechanism utilizes differentiable neural networks to optimize reward structures based on exploration trajectories, enabling automated mechanism design for scientific discovery.

---

[Cognition to Control – Multi-Agent Learning for Human-Humanoid Collaborative Transport](http://arxiv.org/abs/2603.03768)

- C2C (Cognition-to-Control): introduces a three-layer hierarchy for human-humanoid collaborative transport, with Cognition Layer (VLM), Skill Policy Layer (MARL), Whole-Body Control Layer (WBC), Multi-Agent Environment, Reward Generation, Critic, Actor, Cerebellum, Cerebral Cortex, Cerebral Lobes, and I-WBC; it includes decentralized VLM-based cognitive- and grounding-agents.
- The architecture bridges the gap between low-frequency semantic reasoning and high-frequency continuous control by formulating collaboration as an object-centric Markov potential game.
- Experimental results on a Unitree G1 humanoid demonstrate that the system internalizes partner dynamics to facilitate stable coordination and robust role transitions in constrained environments.

---

[Seeing as Experts Do: A Knowledge-Augmented Agent for Open-Set Fine-Grained Visual Understanding](http://arxiv.org/abs/2603.03762)

- KFRA (Knowledge-Augmented Fine-Grained Reasoning Agent): introduces a unified framework that transforms fine-grained perception into evidence-driven reasoning through a three-stage closed reasoning loop, utilizing candidate generation, region localization, and guided inference.
- The architecture establishes a retrieval–grounding coupling that converts retrieved textual knowledge into spatially grounded evidence for verification by LMMs, including summarization-, captioning-, and inference-agents.
- The framework incorporates open-vocabulary detection, web-scale image search, and a global-to-local focusing mechanism to achieve factual and interpretable reasoning across diverse open-set scenarios.

---

[AgentSelect: Benchmark for Narrative Query-to-Agent Recommendation](http://arxiv.org/abs/2603.03761)

- AGENTSELECT (Agent Selection Benchmark): introduces a unified benchmark for narrative query-to-agent recommendation, with Data Collection (aggregating evaluation artifacts), Part I: LLM-only Agents (model-selection preferences), Part II: Toolkit-only Agents (tool-adequacy evidence), Part III: Compositional Agents (synthesized interactions), and includes LLM-retriever, tool-retriever, LLM-judge, and agent-recommendation-model components.
- The framework standardizes heterogeneous evaluation data into structured preference signals by representing agents as executable YAML capability profiles that specify backbone LLMs and toolsets.
- It establishes a reproducible foundation for studying agent selection at scale, demonstrating that content-aware matching outperforms traditional collaborative filtering in long-tail agent marketplaces.

---

[Learning Approximate Nash Equilibria in Cooperative Multi-Agent Reinforcement Learning via Mean-Field Subsampling](http://arxiv.org/abs/2603.03759)

- ALTERNATING-MARL (Alternating Multi-Agent Reinforcement Learning): introduces a cooperative Markov game framework for large-scale systems under strict observability constraints, with a Global Agent, Local Agents, G-LEARN, L-LEARN, Subsampled State Feedback, Broadcast Central Command, Induced MDP, and Best-Response Dynamics; it includes global- and local-agents.
- The framework employs an alternating learning scheme where the global agent performs subsampled mean-field Q-learning while local agents optimize their policies within an induced Markov Decision Process.
- This approach converges to an approximate Nash Equilibrium with a sample complexity that avoids exponential dependence on the population size or joint action space.

---

[Agentic Peer-to-Peer Networks: From Content Distribution to Capability and Action Sharing](http://arxiv.org/abs/2603.03753)

- Agentic P2P (Agentic Peer-to-Peer Networks): introduces a plane-based reference architecture for decentralized collaboration between Client-Side Autonomous Agents, which includes planning-, context-aware-, and tool-using-agents, across four interacting planes.
- The system employs signed, soft-state Capability Descriptors to facilitate intent-aware routing while managing capability drift through time-to-live based refreshes.
- A tiered verification framework balances security overhead against task risk by utilizing reputation signals, canary probes, and cryptographic evidence packages.

---

[Interaction-Aware Whole-Body Control for Compliant Object Transport](http://arxiv.org/abs/2603.03751)

- IO-WBC (Interaction-Oriented Whole-Body Control): introduces a hierarchical control architecture that decouples upper-body interaction from lower-body support to enable stable humanoid object transport under heavy-load coupling, with HRC Skill Policy, Reference Generator (RG), Interaction-Aware WBC Policy, Asymmetric Teacher-Student Distillation, and Proprioceptive History Buffer.
- The framework utilizes a trajectory-optimized reference generator to establish kinematic feasibility while an interaction-aware reinforcement learning policy provides reactive residual corrections for non-conservative forces.
- An asymmetric distillation process allows the student policy to infer complex interaction dynamics from proprioceptive histories alone, eliminating the need for external force-torque sensors during real-world deployment.

---

[RAGNav: A Retrieval-Augmented Topological Reasoning Framework for Multi-Goal Visual-Language Navigation](http://arxiv.org/abs/2603.03745)

- RAGNav (Retrieval-Augmented Topological Reasoning Framework): introduces a navigation architecture that bridges semantic reasoning and physical structure using a Dual-Basis Memory system, including instruction decomposition- and retrieval and reasoning-LLMs.
- The framework integrates a low-level topological map for physical connectivity with a high-level semantic forest for hierarchical abstraction to enable anchor-guided conditional retrieval and rapid candidate screening.
- It utilizes a topological neighbor score propagation mechanism to calibrate semantic associations and optimize sequential planning, significantly enhancing inter-target reachability reasoning for complex instructions.

---

[HALyPO: Heterogeneous-Agent Lyapunov Policy Optimization for Human-Robot Collaboration](http://arxiv.org/abs/2603.03741)

- HALyPO (Heterogeneous-Agent Lyapunov Policy Optimization): introduces a Lyapunov-stabilized learning kernel for heterogeneous multi-agent reinforcement learning in human-robot collaboration, with top-level global mission planner, mid-level tactical MARL policy, bottom-level whole-body controller, Lyapunov rationality gap, stability normal vector, and analytic closed-form projection.
- The architecture utilizes a tri-level control hierarchy where a VLM or A* planner provides strategic guidance to a tactical MARL policy and a high-frequency whole-body controller.
- By enforcing a per-step Lyapunov decrease condition on a parameter-space disagreement metric, the system achieves certified stability and robust coordination in complex interaction manifolds.

---

[PROSPECT: Unified Streaming Vision-Language Navigation via Semantic–Spatial Fusion and Latent Predictive Representation](http://arxiv.org/abs/2603.03739)

- PROSPECT (Predictive Representations Of SPatial-sEmantic ContexTs): introduces a unified streaming navigation agent that couples a Vision-Language-Action policy with latent predictive representation learning using SigLIP semantic features and CUT3R absolute-scale 3D spatial features.

- The architecture employs learnable stream query tokens to predict future 2D and 3D latent features during training, which shapes internal representations to internalize environment dynamics without increasing inference latency.

- A specialized streaming attention mask enforces temporal causality and isolates modality-specific queries, enabling the model to achieve state-of-the-art performance on long-horizon benchmarks and robust real-robot deployment.


---

[HyperParallel: A Supernode-Affinity AI Framework](http://arxiv.org/abs/2603.03731)

- HyperParallel: introduces a supernode-affinity AI framework that abstracts hardware clusters into a single logical computer, with HyperOffload (automated hierarchical memory management), HyperMPMD (fine-grained MPMD parallelism), HyperShard (declarative parallel strategy specification), SuperPod Layout (single-card hardware abstraction), MPMD Runtime (heterogeneous task execution layer), and Resource Pools (unified compute and memory).
- The architecture decouples computation from model state through unified memory pooling and transitions from rigid SPMD to flexible MPMD parallelism to handle load imbalances in Mixture-of-Experts and multimodal LLMs.
- The framework provides a declarative programming interface that automates parallel strategy generation and communication operator insertion, reducing the engineering overhead for large-scale model development.

---

[AI4S-SDS: A Neuro-Symbolic Solvent Design System via Sparse MCTS and Differentiable Physics Alignment](http://arxiv.org/abs/2603.03686)

- AI4S-SDS (A Neuro-Symbolic Solvent Design System): introduces a closed-loop neuro-symbolic framework for automated chemical formulation design, with a planning module (synthesizes global search plans), an MCTS engine (executes tree-based exploration), a generator module (includes main-, expert-, and retrieval-agents), an evaluator module (assesses formulation validity), a memory module (manages historical experience data), a physics engine (optimizes continuous mixing ratios), sparse state storage (minimizes node memory footprint), dynamic path reconstruction (restores context for expansion), and sibling-aware expansion (promotes orthogonal exploration).
- The framework employs Sparse State Storage with Dynamic Path Reconstruction to decouple reasoning history from context length, enabling deep exploration under fixed token budgets.
- It bridges symbolic reasoning and physical feasibility by coupling discrete topology generation with gradient-based optimization of continuous mixing ratios under thermodynamic constraints.

---

[MAGE: Meta-Reinforcement Learning for Language Agents toward Strategic Exploration and Exploitation](http://arxiv.org/abs/2603.03680)

- MAGE (Meta-Reinforcement Learning for Language Agents toward StrateGic Exploration and Exploitation): introduces a meta-RL framework that optimizes LLM agents for strategic adaptation in multi-agent environments through a multi-episode training regime, which includes action- and reflection-generating LLM components, Opponent Pool, Contextual Memory, and Meta Optimization.
- The framework utilizes a reflective inner loop to generate natural language feedback stored in contextual memory, enabling the agent to refine its policy based on past interaction histories.
- It integrates population-based training with agent-specific advantage normalization to facilitate exploitation of diverse opponent vulnerabilities and ensure stable policy updates across non-stationary environments.

---

[MIND: Unified Inquiry and Diagnosis RL with Criteria Grounded Clinical Supports for Psychiatric Consultation](http://arxiv.org/abs/2603.03677)

- MIND (Unified Inquiry and Diagnosis Reinforcement Learning): introduces an evidence-grounded framework for multi-turn psychiatric consultation with PRB, Clinical Retrieval State Builder, Doctor Agent, Explicit Reasoning Trace, Rubric-based Process Reward Model, Value-aware Trajectory Rectification, and Patient Simulator components to integrate criteria-aligned retrieval with process-supervised reinforcement learning.
- The system utilizes a Psychiatric Reasoning Bank to provide clinical supports and includes doctor-, evaluator-, and simulator-LLMs to enforce explicit reasoning traces via rubric-based process rewards.
- A value-aware trajectory rectification mechanism detects low-utility turns and triggers self-retry or fallback strategies to maintain focus on informative clinical cues.

---

[Is an investor stolen their profits by mimic investors? Investigated by an agent-based model](http://arxiv.org/abs/2603.03671)

- ABAFMM (Agent-Based Artificial Financial Market Model): introduces a multi-agent simulation environment to evaluate how increasing numbers of mimic investors affect market stability and individual earnings, utilizing Normal Agents (NAs), Additional Fundamental Agents (AFAs), and Additional Technical Agents (ATAs).
- The model demonstrates that AFAs create a negative feedback loop by buying when prices fall below fundamental values, which stabilizes the market but decreases individual profit margins as more agents join.
- The research further identifies that ATAs generate a positive feedback loop by following price trends, which increases market volatility and allows mimic investors to capture higher profits through amplified price variations.

---

[ORION: Intent-Aware Orchestration in Open RAN for SLA-Driven Network Management](http://arxiv.org/abs/2603.03667)

- ORION: introduces an intent-aware orchestration pipeline that integrates LLMs via the Model Context Protocol (MCP) to translate natural language intents into enforceable network policies, with Frontend, MCP Client, MCP Server, LLM, Camara API, Orion rApp, Orion xApp, A1 Mediator, E2 Manager, E2 Termination, and E2 Sim.
- The architecture leverages a hierarchical agent structure spanning the Service Management and Orchestration layer and RAN Intelligent Controllers to automate the complete intent lifecycle from ingestion to E2-level enforcement.
- It utilizes the CAMARA NetworkSliceBooking schema to ground translations in telecom semantics, ensuring intents progress through schema checks and policy generation before real-time resource allocation.

---

[Mozi: Governed Autonomy for Drug Discovery Large Language Model Agents](http://arxiv.org/abs/2603.03655)

- Mozi (Governed Autonomy for Drug Discovery Large Language Model Agents): introduces a dual-layer architecture for pharmaceutical R&D, with Coordinator Agent, Research Agent, Computation Agent, Skill Graphs, MCP Platform, HITL Checkpoints, and Hybrid Memory.
- The framework implements a supervisor-worker hierarchy to enforce role-based tool isolation and reflection-based replanning for long-horizon stability.
- By operationalizing drug discovery stages as stateful directed acyclic graphs, the system ensures strict data contracts and expert-validated decision boundaries.

---

[Freezing of Gait Prediction using Proactive Agent that Learns from Selected Experience and DDQN Algorithm](http://arxiv.org/abs/2603.03651)

- Proactive RL Agent (Reinforcement Learning-based Proactive Agent for Freezing of Gait Prediction): introduces a dynamic decision-making framework for Parkinson's Disease patients, utilizing Double Deep Q-Networks and Prioritized Experience Replay to identify optimal pre-FOG onset points.
- The system extracts gait stability features via Dynamic Mode Decomposition and processes a six-parameter state vector to determine whether to issue a prediction flag or continue monitoring.
- By implementing a time-based reward shaping strategy, the agent achieves prediction horizons of up to 8.72 seconds, significantly extending the window for proactive clinical interventions.

---

[MistyPilot: An Agentic Fast–Slow Thinking Large Language Model Framework for Misty Social Robots](http://arxiv.org/abs/2603.03640)

- MistyPilot: introduces an agentic LLM-driven framework for autonomous tool orchestration and emotional alignment in social robots, which includes routing-, physical interaction-, and social intelligence-agents, with a Task Router, a Physically Interactive Agent (PIA), a Socially Intelligent Agent (SIA), a Task Status Manager (TSM), a Fast Thinking Module, a Slow Thinking Module, a Script Writer, a Sensor & Tool Manager (STM), a Central Scheduler, a Skill Library, and Long-Term Memory.
- The system utilizes a Task Router to dispatch instructions to specialized agents while employing a dual-channel thinking paradigm to optimize response latency via memory retrieval.
- It achieves multi-dimensional emotional alignment through adaptive motion, context-aware dialogue, and expressive speech synthesis to sustain natural human-robot interaction.

---

[Goal-Driven Risk Assessment for LLM-Powered Systems: A Healthcare Case Study](http://arxiv.org/abs/2603.03633)

- Goal-Driven Risk Assessment Framework: introduces a structured methodology for identifying and prioritizing security risks in LLM-powered healthcare systems using attack trees to contextualize threats across Web Application, Orchestrator, and LLM components.
- The approach integrates STRIDE-based threat modeling with goal-oriented attack trees to map adversarial actions like prompt injection and model tampering to clinical impacts.
- It utilizes a Likelihood × Impact matrix to quantify risks, enabling systematic prioritization of vulnerabilities in safety-critical AI workflows.

---

[Behind the Prompt: The Agent-User Problem in Information Retrieval](http://arxiv.org/abs/2603.03630)

- MoltbookTraces: introduces a study of the Agent Attribution Problem using a Human Operator, AI Agent, Moltbook Platform, Observables, Latent Orchestration Indicator, Position-Based Click Model (PBM), and SIS Epidemic Model to show that individual agent actions cannot be definitively classified as autonomous or human-directed.
- The study demonstrates that click models trained on agent interactions suffer significant performance degradation as low-validation agents are introduced into training data.
- The research reveals that awareness of technical capabilities spreads endemi-cally across agent communities and resists suppression even under aggressive modeled intervention.

---

[Hybrid Belief–Reinforcement Learning for Efficient Coordinated Spatial Exploration](http://arxiv.org/abs/2603.03595)

- HBRL (Hybrid Belief-Reinforcement Learning): introduces a two-phase framework combining Log-Gaussian Cox Process spatial modeling with Soft Actor-Critic reinforcement learning to enable efficient coordinated spatial exploration under unknown demand.
- The system utilizes a dual-channel warm-start mechanism consisting of belief state initialization for informed priors and replay buffer seeding with expert trajectories generated by a Pathwise Mutual Information planner.
- A variance-normalized overlap penalty adapts coordination strength based on local belief uncertainty, allowing cooperative sensing in high-uncertainty regions while minimizing redundant coverage in well-explored areas.

---

[EchoGuard: An Agentic Framework with Knowledge-Graph Memory for Detecting Manipulative Communication in Longitudinal Dialogue](http://arxiv.org/abs/2603.04815)

- EchoGuard: introduces an agentic framework designed to detect manipulative communication in longitudinal dialogues by utilizing a Knowledge Graph as a persistent episodic and semantic memory.
- The system employs a ReAct-based Agent Orchestrator to manage a Log-Analyze-Reflect loop, transforming user-logged interactions into structured graph nodes for multi-hop pattern detection.
- It features a Reflective Prompt Generator that uses LLMs to provide Socratic interventions grounded in retrieved subgraphs, empowering users to recognize harmful patterns while maintaining autonomy.

---

[Neuro-Symbolic Financial Reasoning via Deterministic Fact Ledgers and Adversarial Low-Latency Hallucination Detector](http://arxiv.org/abs/2603.04663)

- VeNRA (Verifiable Numerical Reasoning Agent): introduces a neuro-symbolic framework that shifts the RAG paradigm from probabilistic text retrieval to deterministic variable extraction via a Universal Fact Ledger (UFL) and forensic auditing.

- The architecture includes an Architect LLM for generating Python execution traces and a 3B parameter Sentinel SLM that performs forensic audits on logic traces within a 50ms latency budget.

- The research utilizes Adversarial Simulation to programmatically sabotage financial records for training and introduces a Micro-Chunking loss algorithm to stabilize gradients during Reverse-Chain-of-Thought optimization.


---

[GIANT - Global Path Integration and Attentive Graph Networks for Multi-Agent Trajectory Planning](http://arxiv.org/abs/2603.04659)

- GIANT (Global Path Integration and Attentive Graph Networks): introduces a multi-agent trajectory planning framework that integrates global path planning with a decentralized local navigation model using attentive graph neural networks.

- The system utilizes a two-stage approach where A* search provides high-level guidance and a reinforcement learning-based local model handles dynamic collision avoidance through graph-based agent representations.

- The architecture features specialized encoders for static and temporal LiDAR data alongside a neighbor-attentive GNN to manage complex interactions in dense, dynamic environments.


---

[iAgentBench: Benchmarking Sensemaking Capabilities of Information-Seeking Agents on High-Traffic Topics](http://arxiv.org/abs/2603.04656)

- iAgentBench (Information-seeking Agent Benchmark): introduces a dynamic open-domain QA benchmark construction pipeline, with GDELT GKG (source for high-traffic seed topics), Scoring and Selection (filters seeds for salience and diversity), Web Search (retrieves query-conditioned open-web corpora), Graph Extraction (identifies entities and relational assertions), Leiden Clustering (partitions graph into thematic communities), Community Categorization (assigns roles like Core or Bridge), Packet Construction (bundles themes and connector relations), LLM Generator (produces grounded questions and answers), and LLM Judge Panel (verifies evidence support and necessity).

- The framework generates sensemaking questions that require integrating evidence across multiple thematic communities and explicit connector relations rather than single-passage extraction.

- It utilizes a story-graph representation to model cross-document dependencies and employs a multi-LLM judge panel to ensure questions are objective and evidence-dependent.


---

[HDLFORGE: A Two-Stage Multi-Agent Framework for Efficient Verilog Code Generation with Adaptive Model Escalation](http://arxiv.org/abs/2603.04646)

- HDLFORGE: introduces a two-stage multi-agent framework for Verilog generation that optimizes the accuracy-latency trade-off by adaptively escalating from a medium-sized model to an ultra-large model, and includes planning-, coding-, and reflection-agents.
- The architecture incorporates a counterexample-guided formal agent that transforms bounded-model-checking traces into reusable micro-tests to accelerate bug detection and reduce repair iterations.
- The framework features a portable escalation controller capable of wrapping existing Verilog LLM pipelines to enhance their speed-accuracy frontier without requiring internal modifications or re-tuning.

---

[When Agents Persuade: Propaganda Generation and Mitigation in LLMs](http://arxiv.org/abs/2603.04636)

- Propaganda Generation and Mitigation Framework: introduces a systematic approach to analyze and reduce the production of manipulative content in LLMs, including GPT-4o, Llama 3.1, and Mistral Small 3, with Generative LLMs, a Binary Propaganda Detector, a Rhetorical Technique Detector, Mitigation Strategies, and QProp and PTC Datasets.
- The system utilizes a RoBERTa-large binary classifier for document-level propaganda detection and six specialized classifiers to identify rhetorical techniques like loaded language, appeals to fear, and flag-waving.
- Experimental results demonstrate that Odds Ratio Preference Optimization (ORPO) is the most effective mitigation strategy, significantly lowering the frequency of rhetorical techniques in generated articles compared to SFT and DPO.

---

[Strategic Interactions in Multi-Level Stackelberg Games with Non-Follower Agents and Heterogeneous Leaders](http://arxiv.org/abs/2603.04628)

- Hierarchical Stackelberg framework: introduces a three-level game-theoretic model that integrates heterogeneous leaders, strategic followers, and non-follower agents to capture bidirectional coupling between infrastructure decisions and endogenous congestion.

- The architecture explicitly models non-participating agents who reshape congestion patterns, thereby altering the equilibrium incentives and optimal strategies for strategic market participants.

- The framework is instantiated in electric vehicle charging markets to analyze how long-term infrastructure placement and short-term pricing competition interact with shared traffic congestion.


---

[Vibe Code Bench: Evaluating AI Models on End-to-End Web Application Development](http://arxiv.org/abs/2603.04601)

- VCB (Vibe Code Bench): introduces a benchmark for evaluating the "zero-to-one" web application development capabilities of LLMs, utilizing Specs, an OpenHands Harness, and an Automated Evaluation Pipeline to measure the generation of deployable artifacts.
- The framework provides a sandboxed Development Environment equipped with a Model API, terminal access, and integrated Supabase & MailHog Services to support complex multi-file coding and deployment tasks.
- Applications are validated by a vision-enabled Browser Use Agent that executes 964 automated workflows across 10,131 substeps to produce fine-grained pass-fail signals based on user-visible behavior.

---

[Self-Attribution Bias: When AI Monitors Go Easy on Themselves](http://arxiv.org/abs/2603.04582)

- Self-Attribution Bias Evaluation Framework: introduces a methodology to measure how LLMs evaluate their own actions as more correct or less risky when authorship is implicitly framed through conversational structure, and includes generator- and monitor-LLMs.
- The framework compares on-policy self-monitoring, where a model evaluates its own generated artifacts, against off-policy baselines to identify systematic rating shifts.
- Findings across coding and tool-use datasets demonstrate that implicit self-attribution increases approval rates for incorrect or harmful actions, potentially overestimating monitor reliability.

---

[Token Taxes: Mitigating AGI’s Economic Risks](http://arxiv.org/abs/2603.04555)

- Token Taxes: introduces a usage-based surcharge framework applied to model tokens at the point of sale to mitigate economic risks from Artificial General Intelligence, utilizing a staged audit pipeline involving AI model providers, cloud compute providers, and government entities.
- The system leverages cloud compute providers as intermediaries to enforce tax compliance through sequential black-box verification, norm-based fallback rates, and comprehensive white-box audits.
- The research recommends LLM-based agent-based modeling to predict market impacts and suggests hybrid policies combining token-based and compute-based taxation.

---

[Transformer-Based Multipath Congestion Control: A Decoupled Approach for Wireless Uplinks](http://arxiv.org/abs/2603.04550)

- TCCO (Transformer-based Congestion Control Optimization): introduces a decoupled architecture for multipath transport that offloads congestion control logic from the kernel to an external decision engine, with In-kernel Client, User-space Proxy, External Decision Engine, Transformer-based DRL Agent, and Replay Buffer.
- The framework utilizes a Transformer-based deep reinforcement learning agent to jointly model temporal dependencies across historical observations, effectively filtering measurement noise and coordinating control across multiple subflows.
- Extensive evaluations on simulated and physical dual-band Wi-Fi testbeds demonstrate that the system achieves superior bandwidth efficiency and robustness against stochastic packet loss compared to traditional heuristic-based and single-step learning methods.

---

[ADAPTIVE MEMORY ADMISSION CONTROL FOR LLM AGENTS](http://arxiv.org/abs/2603.04549)

- A-MAC (Adaptive Memory Admission Control): introduces a structured decision framework for long-term memory management in LLM agents, utilizing a hybrid architecture that combines LLM-based utility assessment with rule-based signals for confidence, novelty, recency, and content type priors.
- The system evaluates candidate memories through a weighted linear scoring model and a learned threshold to balance memory coverage, reliability, and computational efficiency.
- By reserving LLM inference for semantic utility judgments and using efficient rules for other factors, the framework provides transparent and auditable control over agent memory growth.

---

[Invariant Causal Routing for Governing Social Norms in Online Market Economies](http://arxiv.org/abs/2603.04534)

- ICR (Invariant Causal Routing): introduces a three-stage causal governance framework that identifies stable policy-to-norm relations across heterogeneous environments using counterfactual reasoning and invariant causal discovery, with ICR, Causal Identification via PNS, Minimal Causal Rule Routing, Key Factors Attribution, Implicit Contract Pool, Causal Rule Router, Platform Levers, and User Responses.
- The framework utilizes Probability of Necessity and Sufficiency (PNS) to filter an Implicit Contract Pool, ensuring that selected interventions are causally responsible for norm attainment under specific contexts.
- A Causal Rule Router is optimized through bucketed greedy learning and pruning to achieve out-of-distribution generalization while maintaining a concise and interpretable set of governance rules.

---

[Discovering mathematical concepts through a multi-agent system](http://arxiv.org/abs/2603.04528)

- Multi-agent system for mathematical discovery: introduces a multi-agent model for computational mathematical discovery that simulates the interaction between questioning and answering using a Conjecturing agent, a Skeptical agent, a MathWorld environment, Feature spotters, a Scaffolder, a Provability component, Lean Copilot, symbolic regression, and a data distribution.
- The framework employs symbolic regression to generate candidate conjectures from polyhedral data, which are subsequently verified by a Provability component featuring an LLM-based prover.
- By dynamically evolving the data distribution, the system recovers the concept of homology and its relationship to the Euler characteristic.

---

[EmbodiedSplat: Online Feed-Forward Semantic 3DGS for Open-Vocabulary 3D Scene Understanding](http://arxiv.org/abs/2603.04254)

- EmbodiedSplat: introduces an online feed-forward 3D Gaussian Splatting framework for open-vocabulary scene understanding, utilizing a sparse coefficient field and a CLIP global codebook to enable simultaneous 3D reconstruction and semantic interpretation from streaming images.
- The architecture integrates 3D geometric-aware features via a 3D U-Net and a temporal-aware memory adapter to compensate for 2D-oriented language embeddings during the reconstruction process.
- The framework achieves nearly real-time performance of 5-6 FPS through a lightweight variant that employs real-time 2D perception models and an efficient codebook-based cosine similarity search strategy.

---

[Right in Time: Reactive Reasoning in Regulated Traffic Spaces](http://arxiv.org/abs/2603.03977)

- Reactive ProMis: introduces a synthesis of Probabilistic Mission Design and Reactive Circuits to facilitate real-time, exact probabilistic inference for autonomous agents navigating regulated traffic environments.
- The architecture utilizes the Resin programming language to subdivide inference formulas into isolated, memoized tasks that adapt their re-evaluation frequency based on the volatility of heterogeneous data streams.
- Experimental results demonstrate that this reactive approach provides orders of magnitude speedup over static inference methods, enabling active safety and legal compliance monitoring during flight operations.

---

[From Spark to Fire: Modeling and Mitigating Error Cascades in LLM-Based Multi-Agent Collaboration](http://arxiv.org/abs/2603.04474)

- Genealogy-Based Governance Layer: introduces a message-layer plugin for LLM-based multi-agent systems that models error propagation dynamics to detect and suppress the formation of false consensus.
- The framework utilizes a lineage graph to track atomic claim provenance and includes decomposition- and adjudication-agents to categorize information as confirmed, unverified, or rejected.
- It employs policy-driven verification and automated rollbacks to mitigate both endogenous hallucinations and exogenous strategic attacks without altering the underlying collaboration architecture.

---

[Rethinking Role-Playing Evaluation: Anonymous Benchmarking and a Systematic Study of Personality Effects](http://arxiv.org/abs/2603.03915)

- Personality-Augmented RPA (Personality-Augmented Role-Playing Agent) framework: introduces an anonymized evaluation protocol and a personality-enhanced construction method for Role-Playing Agents, utilizing Profile, Dialogues, Anonymous Data, LLM, Base Agent, Self-Generated Personality, and Personality Augmented Agent components.
- The framework mitigates character name bias by replacing identities with anonymous tokens, forcing LLs to rely on descriptive personas rather than memorized pretraining data.
- It demonstrates that incorporating self-generated personality traits, such as MBTI or Big Five scores, significantly improves role fidelity and performance across multiple benchmarks.

---

[From Threat Intelligence to Firewall Rules: Semantic Relations in Hybrid AI Agent and Expert System Architectures](http://arxiv.org/abs/2603.03911)

- SIF (Semantic Information Flow): introduces a neuro-symbolic architecture that converts Cyber Threat Intelligence reports into firewall rules using an Enhanced CoALA agent for semantic extraction and symbolic expert systems for rule synthesis.
- The system employs an iterative prompting strategy to retrieve taxonomic relationships, specifically hyponyms and hypernyms, to build a structured graph-based knowledge base.
- The framework integrates a syntactic verification layer and a refinement engine to mitigate LLM hallucinations and ensure the technical correctness of generated iptables configurations.

---

[Beyond Input Guardrails: Reconstructing Cross-Agent Semantic Flows for Execution-Aware Attack Detection](http://arxiv.org/abs/2603.04469)

- MAScope: introduces a security framework for Multi-Agent Systems that detects attacks by reconstructing fragmented operational primitives into contiguous behavioral trajectories called Cross-Agent Semantic Flows, and includes extraction- and supervisor-LLMs.
- The framework employs a dual-layer observation strategy, combining kernel-level monitoring with application-layer logging to bridge the semantic gap in autonomous agent interactions.
- It utilizes a Supervisor LLM to audit reconstructed paths against security policies, achieving F1-scores of 85.3% and 66.7% for node-level and path-level attack detection.

---

[Proximal Learning for Trials With External Controls: A Case Study in HIV Prevention](http://arxiv.org/abs/2603.04544)

- Proximal Learning (Proximal Causal Inference for External Controls): introduces a causal inference framework to estimate counterfactual placebo incidence in active-controlled trials by leveraging external control data and negative control variables to adjust for unmeasured confounding.
- The approach incorporates a semiparametric inverse probability of censoring weighting estimator and a two-stage regression-based method designed to handle right-censored time-to-event data in low-incidence settings.
- Application to HIV prevention trials demonstrates that using geographic region and baseline infection status as proxies for unmeasured risk factors yields reliable estimates of absolute treatment efficacy.

---

[Neural geometry in the human hippocampus enables generalization across spatial position and gaze](http://arxiv.org/abs/2603.04747)

- Neural Manifold Geometry: introduces a representational framework where hippocampal population activity is organized into semi-orthogonal, linearly transformable subspaces to track self, others, and gaze simultaneously.
- The architecture utilizes a family of geometrically related manifolds that allow for both the individuation of distinct agents and the abstraction of spatial rules across different viewpoints.
- Linear decoders demonstrate that spatial knowledge learned for one agent successfully transfers to others, supporting a unified model of hippocampal relational coding and multi-agent tracking.

---

[Digital-Twin Losses for Lane-Compliant Trajectory Prediction at Urban Intersections](http://arxiv.org/abs/2603.05546)

- Digital-Twin-Augmented LSTM: introduces a V2X trajectory prediction pipeline that combines a two-layer LSTM encoder-decoder with a structured training objective consisting of MSE, infrastructure proximity, and collision avoidance losses; it includes encoder- and decoder-LSTMs.
- The framework employs an anchor-relative normalization scheme to align predicted trajectories with absolute HD map coordinates, ensuring informative gradients for map-aware auxiliary losses.
- The system incorporates Monte Carlo Dropout during inference to produce stochastic sample sets for uncertainty estimation and multi-modal trajectory evaluation.


---

#### 3rd March 2026

[MIBURI: Towards Expressive Interactive Gesture Synthesis](http://arxiv.org/abs/2603.03282)

- MIBURI: introduces an online, causal framework for real-time full-body gesture and facial expression synthesis, leveraging internal tokens from the Moshi speech-text LLM.
- The architecture utilizes body-part aware gesture codecs to discretize motion into hierarchical tokens, which are then processed by a two-dimensional transformer setup for temporal and kinematic modeling.
- The framework incorporates auxiliary contrastive and voice activation losses to enhance gesture diversity and ensure synchronization during full-duplex conversational interactions.

---

[Inherited Goal Drift: Contextual Pressure Can Undermine Agentic Goals](http://arxiv.org/abs/2603.03258)

- Inherited Goal Drift: investigates the susceptibility of advanced LLM agents to adopt misaligned objectives when conditioned on drifted trajectories from weaker models, with system prompts, adversarial pressures, weak agents, drifted context windows, strong models, and simulation environments; it includes weak-agent and strong-model roles.
- The evaluation spans multiple model families, including reasoning-capable variants, across stock-trading and emergency room triage simulations to quantify goal adherence under contextual pressure.
- Results indicate that while newer models are generally robust to direct pressure, they frequently inherit drift from context, and instruction hierarchy following is a poor predictor of this vulnerability.

---

[Conversational Learning Diagnosis via Reasoning Multi-Turn Interactive Learning](http://arxiv.org/abs/2603.03236)

- ParLD (Preview-Analyze-Reason framework for conversational Learning Diagnosis): introduces an LLM-based multi-agent system for fine-grained cognitive state tracking in multi-turn tutoring dialogues, which includes behavior previewer-, state analyzer-, performance reasoner-, and chain reflector-agents.
- The architecture employs a structured behavioral schema based on the Zone of Proximal Development to constrain diagnostic hypotheses before updating student knowledge concept mastery.
- A meta-cognitive loop utilizes performance prediction feedback to trigger iterative self-reflection and refinement of the episodic conversation memory.

---

[LEARNING WHEN TO ACT OR REFUSE: GUARDING AGENTIC REASONING MODELS FOR SAFE MULTI-STEP TOOL USE](http://arxiv.org/abs/2603.03205)

- MOSAIC: introduces a post-training framework that aligns agentic reasoning models for safe multi-step tool use, with a policy model (generates reasoning and actions), an LLM judge (compares trajectory safety preferences), an environment (executes tools and returns feedback), a user (initiates tasks and receives reports), a tool catalog (defines available agent functions), and interaction history (maintains context across multiple steps); the framework includes planning-, safety reasoning-, and tool execution-components.
- The framework employs preference-based reinforcement learning using Group Relative Policy Optimization and a pairwise LLM judge to learn temporal safety distinctions across complex trajectories.
- Evaluation shows that explicit safety checks and learnable refusal actions reduce harmful behaviors and privacy leakage across diverse model families and agentic benchmarks.

---

[Code2Math: Can Your Code Agent Effectively Evolve Math Problems Through Exploration?](http://arxiv.org/abs/2603.03202)

- Code2Math: introduces a multi-agent framework that utilizes code-driven exploration to autonomously transform seed mathematical problems into structurally distinct and more challenging versions.
- The architecture includes evolution-, solvability-verification-, and difficulty-verification-agents to manage problem synthesis, logical auditing, and cognitive depth assessment.
- By scaling test-time exploration through code-driven empirical inquiry, the framework enables LLMs to synthesize complex, solvable mathematical data that exceeds their own solving baselines.

---

[BeyondSWE: Can Current Code Agent Survive Beyond Single-Repo Bug Fixing?](http://arxiv.org/abs/2603.03194)

- SearchSWE: introduces an agentic framework that integrates coding proficiency with deep research skills, with SearchSWE (Agent), Search Tool, Browser Tool, Docker Container, External Environment, Blocklist, and Evaluation Pipeline components, where the system enables LLMs to interleave repository exploration with iterative web search to solve complex software engineering tasks.
- The framework utilizes a dual-context architecture to bridge the gap between local code manipulation and global knowledge acquisition while enforcing strict cheating prevention through a regex-based blocklist.
- Accompanying the framework is BeyondSWE, a multi-task benchmark featuring 500 real-world instances across four distinct settings including cross-repository resolution and dependency-driven migration.

---

[From Language to Action: Can LLM-Based Agents Be Used for Embodied Robot Cognition?](http://arxiv.org/abs/2603.03148)

- LLM-based Cognitive Agent Architecture: introduces a cognitive framework for embodied robots, with a User, LLM Cognitive Core, Robot Toolkit, Episodic Memory, Working Memory, Robot, and World Simulation, where an LLM serves as the central reasoning engine for planning and execution reasoning.
- The system leverages a tool-calling interface to bridge high-level language instructions with low-level perception and control functionalities in a simulated household environment.
- Experimental results highlight that while episodic memory improves task efficiency, LLMs frequently exhibit overconfidence and hallucinations concerning actual task completion status.

---

[Agentic AI-based Coverage Closure for Formal Verification](http://arxiv.org/abs/2603.03147)

- Saarthi (ScalAble ARTificial and Human Intelligence): introduces an agentic AI-driven workflow to automate coverage analysis and closure in formal verification by utilizing LLMs to identify coverage gaps and generate targeted SystemVerilog assertions.
- The architecture utilizes a multi-agent group chat managed by a Group Chat Manager, incorporating lead-, verification-, critic-, and executor-agents to facilitate iterative property refinement and tool interaction.
- Specialized coverage analysis agents, including a Coverage Hole Analyzer and SVA Property Generator, process formal verification reports to systematically address unexercised RTL branches and statements.

---

[APRES: An Agentic Paper Revision and Evaluation System](http://arxiv.org/abs/2603.03142)

- APRES (An Agentic Paper Revision and Evaluation System): introduces an agentic framework that discovers predictive evaluation rubrics to guide automated scientific paper revisions, with an agentic search scaffold, LLM-based rubric proposer-, reviewer-, and rewriter-agents.
- The system utilizes a two-stage approach to first identify criteria predictive of future impact via negative binomial regression and then optimizes paper text through a closed-loop revision process.
- Evaluation shows the method improves citation prediction by 19.6% and generates manuscript revisions preferred by human experts in 79% of cases while maintaining scientific integrity.

---

[How to Model AI Agents as Personas?: Applying the Persona Ecosystem Playground to 41,300 Posts on Moltbook for Behavioral Insights](http://arxiv.org/abs/2603.03140)

- PEP (Persona Ecosystem Playground): introduces a four-stage methodology for generating and validating conversational personas from AI agent behavioral data on social platforms, utilizing k-means clustering and retrieval-augmented generation.
- The framework extracts behavioral archetypes from unstructured social media posts to create data-driven personas that are then deployed in multi-agent simulations using LangChain and LangGraph.
- Validation through reverse querying and diversity metrics ensures that generated personas maintain distinct behavioral profiles and semantic grounding throughout structured interactions with human moderator interventions.

---

[RL-Based Coverage Path Planning for Deformable Objects on 3D Surfaces](http://arxiv.org/abs/2603.03137)

- RL-Based Coverage Path Planning Framework: introduces a reinforcement learning pipeline for robotic manipulation of deformable objects on complex 3D surfaces, utilizing 3D Reconstruction, MuJoCo Simulator, Harmonic UV Mapping, Multi-scale Maps, SGCNN, RL Agent, and FC Layers to simplify state and action spaces.
- The architecture integrates a Scaled Grouped Convolutional Neural Network (SGCNN) to process multi-scale egocentric maps that track coverage, frontiers, and boundaries for efficient feature learning.
- By training in a physics-based simulator and deploying to a Kinova Gen3 manipulator, the method achieves increased coverage area and path efficiency compared to traditional geometric planning baselines.

---

[The Science Data Lake: A Unified Open Infrastructure Integrating 293 Million Papers Across Eight Scholarly Sources with Embedding-Based Ontology Alignment](http://arxiv.org/abs/2603.03126)

- Science Data Lake: introduces a unified open infrastructure for science-of-science research that integrates 293 million papers across eight scholarly sources, with Data Sources (eight integrated scholarly databases), Parquet Storage (columnar files for large-scale data), DuckDB Engine (embeddable SQL query processor), Source-level Schemas (preserved native metadata namespaces), Xref Schema (cross-referencing and linkage layer), DOI Normalization (canonical identifier mapping process), BGE-large Embedding Model (semantic ontology alignment tool), Ontology Schemas (thirteen formal scientific taxonomies), and LLM-optimized Documentation (structured schema reference for agents).
- The system employs a hybrid alignment strategy to map hierarchical topic taxonomies to formal scientific ontologies, achieving high precision through dense vector embeddings and exact string matching.
- The architecture supports dual-mode access for local or remote querying and provides a machine-readable schema reference to enable text-to-SQL workflows for LLMs.

---

[AI Space Physics: Constitutive boundary semantics for open AI institutions](http://arxiv.org/abs/2603.03119)

- AI Space Physics: introduces a constitutive transition-level governance semantics for self-expanding AI institutions, utilizing a membrane-witness discipline to mediate first-order commits and second-order authority-surface expansions.
- The architecture defines "strong governability" through non-bypassable mediation, temporal atomicity of adjudication-to-effect transitions, and replayable witness records.
- The framework reclassifies structural growth and policy broadening as first-class boundary events to prevent latent authority accumulation from bypassing governance oversight.

---

[Beyond Task Completion: Revealing Corrupt Success in LLM Agents through Procedure-Aware Evaluation](http://arxiv.org/abs/2603.03116)

- PAE (Procedure-Aware Evaluation): introduces a framework that formalizes agent procedures as structured observations to expose consistency relationships between observation, communication, and execution, with tripartite action model, structured observation space, and multi-dimensional gating components.
- The architecture includes a user simulator-agent for interaction generation and an LLM-as-judge-agent for semantic auditing of agent behavior against formal policy specifications.
- By applying multi-dimensional gating, the framework identifies "corrupt successes" where agents complete tasks while violating procedural integrity, such as bypassing authorization or fabricating data.

---

[RAPO: Expanding Exploration for LLM Agents via Retrieval-Augmented Policy Optimization](http://arxiv.org/abs/2603.03078)

- RAPO (Retrieval-Augmented Policy Optimization): introduces a reinforcement learning framework that explicitly expands agent exploration by retrieving and reasoning over off-policy step-level traces during training, with Policy Model, Tool Environment, Step-Trace Buffer, Hybrid-policy Agentic Rollout, Retrieval-aware Policy Optimization, Reference Model, and Reward Model.
- The system utilizes a Hybrid-policy Agentic Rollout to interleave self-generated reasoning with high-quality external behaviors stored in a Step-Trace Buffer to broaden the agent's reasoning receptive field.
- It employs a Retrieval-aware Policy Optimization mechanism featuring entropy-based retrieval rewards and importance shaping to stabilize training and prioritize informative exploratory signals while improving training efficiency.

---


[MA-CoNav: A Master-Slave Multi-Agent Framework with Hierarchical Collaboration and Dual-Level Reflection for Long-Horizon Embodied VLN](http://arxiv.org/abs/2603.03024)

- MA-CoNav (Multi-Agent Collaborative Navigation): introduces a hierarchical master-slave multi-agent framework for long-horizon vision-language navigation, with Master Agent, Observation Agent, Task Planning Agent, Control Execution Agent, and Memory Agent.

- The system employs a "Plan-Perceive-Act-Evaluate" closed loop where an LLM-based Master Agent orchestrates specialized sub-agents to decouple perception, planning, and execution tasks to alleviate cognitive overload.

- A dual-stage reflection mechanism enables immediate local action correction and global post-task experience distillation to improve navigation success rates and robustness in complex, real-world indoor environments.


---

[REGAL: A Registry-Driven Architecture for Deterministic Grounding of Agentic AI in Enterprise Telemetry](http://arxiv.org/abs/2603.03018)

- REGAL (Registry-Driven Architecture for Grounded Agentic LLMs): introduces a reference architecture that separates deterministic telemetry computation from probabilistic reasoning by compiling declarative metric definitions into a bounded tool interface for LLMs.
- The system utilizes a Medallion ELT pipeline to produce version-controlled Gold artifacts, ensuring that agents operate over semantically compressed and replayable data rather than raw event streams.
- By implementing an "interface-as-code" pattern via the Model Context Protocol, the framework mitigates tool drift and enforces governance policies directly at the semantic boundary.

---

[Single-Sample Bilateral Trade with a Broker](http://arxiv.org/abs/2603.03016)

- Single-Sample Bilateral Trade with a Broker: introduces a mechanism design framework for three-sided interactions where a broker mediates trade between a buyer and seller using only a single sample from each agent's valuation distribution, and includes Buyer-, Seller-, Broker-, Single-Sample Mechanism- and Posted-Pricing Protocol-components.
- The framework achieves constant-factor approximations for gains-from-trade, social welfare, and optimal profit under monotone-hazard-rate assumptions.
- The study establishes matching or near-matching upper and lower bounds for efficiency metrics in symmetric and stochastically ordered valuation settings.

---

[MULTI-AGENT HONEYPOT-BASED REQUEST-RESPONSE CONTEXT DATASET FOR IMPROVED SQL INJECTION DETECTION PERFORMANCE](http://arxiv.org/abs/2603.02963)

- Multi-agent honeypot framework: introduces a context-enriched SQL injection detection system that constructs a request-response dataset using collaborative agents to capture bidirectional semantic cues.
- The architecture utilizes a Request Generator Agent for diverse payload creation, a Database Response Agent for authentic interaction mediation via shadow databases, and a Traffic Monitor Agent for automated labeling and data curation.
- Experimental results demonstrate that models trained on this context-rich dataset, including CNN and BiLSTM architectures, significantly outperform payload-only detectors by leveraging server-side signals like error codes and response latency.

---

[Architecting Trust in Artificial Epistemic Agents](http://arxiv.org/abs/2603.02960)

- Normative Framework for Trustworthy Epistemic AI Agents: introduces a roadmap for governing autonomous AI systems that shape knowledge environments, with Epistemic AI Agents, Verifiable Properties, Alignment, Technical Infrastructure, Social Infrastructure, and Memory.
- The framework defines epistemic AI agents as entities capable of autonomously pursuing epistemic goals and actively shaping external knowledge environments through diverse roles including scientist-, journalist-, and educator-agents.
- It emphasizes the necessity of technical provenance systems, robust falsifiability pipelines, and "knowledge sanctuaries" to mitigate risks such as cognitive deskilling and epistemic drift in multi-agent ecosystems.

---

[CGL: Advancing Continual GUI Learning via Reinforcement Fine-Tuning](http://arxiv.org/abs/2603.02951)

- CGL (Continual GUI Learning): introduces a framework that dynamically balances adaptation efficiency and skill retention in GUI agents by synergizing Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) through entropy-guided weight adjustment and gradient surgery.
- The system employs Error-Aware Routing to trigger supervised corrective updates when reinforcement exploration fails, ensuring the agent acquires new skills without getting trapped in unproductive exploration.
- It features a specialized gradient surgery strategy that projects exploratory SFT gradients onto conflict-free subspaces defined by GRPO-based anchor gradients to prevent catastrophic forgetting of previously learned interaction logic.

---

[Learning to Generate and Extract: A Multi-Agent Collaboration Framework For Zero-shot Document-level Event Arguments Extraction](http://arxiv.org/abs/2603.02909)

- GenExtract (Multi-Agent Collaboration Framework for Zero-shot Document-level Event Arguments Extraction): introduces a collaborative system that simulates human cognitive processes to generate and refine synthetic data for unseen event types, which includes generation- and evaluation-agents.
- The generation agent synthesizes document-level contexts and role-argument pairs, while the evaluation agent extracts arguments to compute log-likelihood scores and structural completeness penalties.
- A reinforcement learning mechanism iteratively refines both agents using a unified reward signal to mitigate cumulative bias and improve the accuracy of synthetic data for zero-shot extraction.

---

[SpecLoop: An Agentic RTL-to-Specification Framework with Formal Verification Feedback Loop](http://arxiv.org/abs/2603.02895)

- SpecLoop: introduces an agentic framework for RTL-to-specification generation that utilizes a formal-verification-driven iterative feedback loop to ensure functional consistency between hardware designs and their documentation.

- The system includes spec generator- and RTL reconstructor-agents, which work in tandem with a formal equivalence checker to identify and correct mismatches through actionable diagnostics.

- By separating compilation errors from functional mismatches, the framework achieves state-of-the-art performance in generating high-quality, structured specifications across various LLM backbones.


---

[LLandMark: A Multi-Agent Framework for Landmark-Aware Multimodal Interactive Video Retrieval](http://arxiv.org/abs/2603.02888)

- LLandMark (A Multi-Agent Framework for Landmark-Aware Multimodal Interactive Video Retrieval): introduces a modular multi-agent architecture for landmark-aware video retrieval, featuring specialized agents for query parsing, landmark reasoning, and multimodal reranking.
- The system includes planning-, landmark reasoning-, and answer synthesis-agents, alongside an OCR refinement module and an automated image-to-image pipeline leveraging Gemini 2.5 Flash.
- It utilizes a hybrid search system across Milvus, Elasticsearch, and MongoDB to integrate semantic, textual, and object-based queries for complex event-based evidence retrieval.

---

[Guideline-Grounded Evidence Accumulation for High-Stakes Agent Verification](http://arxiv.org/abs/2603.02798)

- GLEAN (GuideLine-grounded Evidence AccumulatioN): introduces a verification framework for high-stakes LLM agents that compiles expert-curated protocols into trajectory-informed, calibrated correctness signals.
- The architecture includes retrieval-, judging-, and active verification-LLMs to transform expert protocols into process-based signals for sequential evidence accumulation.
- Bayesian logistic regression provides calibrated correctness probabilities, while high uncertainty triggers active verification via guideline expansion and differential checks against alternative diagnoses.

---

[AGENTIFIED ASSESSMENT OF LOGICAL REASONING AGENTS](http://arxiv.org/abs/2603.02788)

- AAA (Agentified Agent Assessment): introduces a framework for evaluating reasoning agents by decoupling assessment logic into a dedicated assessor agent that interacts with agents under test via a standardized interface.
- The system utilizes a data cleaning pipeline featuring LLM-based critique and refiner agents alongside symbolic solvers to repair and verify logical reasoning benchmarks.
- Evaluation on a refined FOLIO dataset demonstrates that an auto-formalization agent using code generation and SMT solving significantly outperforms standard chain-of-thought prompting.

---

[Agentic Self-Evolutionary Replanning for Embodied Navigation](http://arxiv.org/abs/2603.02772)

- SERP (Self-Evolutionary RePlanning): introduces a cross-level replanning framework for embodied navigation that enables run-time model evolution through local ASE and global GCOT mechanisms, including reasoning-, subtask- and failure analysis-agents.
- The ASE component employs ILAD to perform adaptive function adjustment and global parameter resets, allowing the robot to overcome ego-model uncertainties and physical failures.
- The GCOT module optimizes global replanning by distilling large scene graphs into semantically relevant subgraphs using CLIP-based feature matching and iterative LLM inference.

---

[EvoSkill: Automated Skill Discovery for Multi-Agent Systems](http://arxiv.org/abs/2603.02766)

- EvoSkill: introduces a self-evolving framework that automatically discovers and refines agent skills through iterative failure analysis, utilizing executor-, proposer-, and skill-builder-agents.
- The architecture materializes high-level proposals into structured skill folders containing procedural instructions and helper scripts, governed by a Pareto frontier of top-performing agent programs.
- Evaluation on OfficeQA and SealQA benchmarks shows accuracy improvements and confirms that evolved skills generalize to unseen tasks without modification.

---

[Enhancing User Throughput in Multi-panel mmWave Radio Access Networks for Beam-based MU-MIMO Using a DRL Method](http://arxiv.org/abs/2603.02745)

- DRL-based beam management: introduces a deep reinforcement learning approach for optimizing beam selection in multi-panel mmWave radio access networks, with gNB (multi-panel base station), Mobile Terminals (user equipment), RL Agent (DDQN-based controller), Policy Network (multi-layer perceptron), State Space (multidimensional observation vector), and Reward Function (normalized throughput feedback).
- The framework models the interaction between the communication agent and the wireless environment as a Markov Decision Process to maximize user throughput and minimize latency.
- It utilizes a Double Deep Q-Network to dynamically adjust beamforming decisions by incorporating spatial domain characteristics, reference signal received power, and historical beam usage statistics.

---

[A Natural Language Agentic Approach to Study Affective Polarization](http://arxiv.org/abs/2603.02711)

- LLM-Powered Agentic Platform: introduces a multi-agent framework that leverages LLMs to construct virtual communities for simulating and measuring affective polarization through natural language discourse, with all Agent, Multi-Agent Framework, and Evaluation Module components.
- The system models individual agents with distinct personas, demographics, and political standpoints, utilizing a persistent memory component and supporting respond- and observe-actions to ensure contextual coherence during multi-turn interactions.
- It employs a structured evaluation pipeline consisting of pre- and post-discussion questionnaires to quantify shifts in inter-group warmth and hostility across various social and political scenarios.

---

[Graph-GRPO: Stabilizing Multi-Agent Topology Learning via Group Relative Policy Optimization](http://arxiv.org/abs/2603.02701)

- Graph-GRPO (Graph-based Group Relative Policy Optimization): introduces a topology optimization framework that stabilizes multi-agent communication learning by shifting from absolute rewards to group-relative advantages, utilizing a GAT-based policy network and edge-level credit assignment.
- The system employs a group sampling mechanism to generate diverse communication graphs, calculating the marginal success rate of specific edges to filter out non-informative noise from easy queries.
- By integrating a Directed Acyclic Graph (DAG) mask and a critic-free optimization strategy, the framework achieves superior training stability and identifies critical communication pathways across reasoning and coding benchmarks.

---

[Retrieval-Augmented Robots via Retrieve-Reason-Act](http://arxiv.org/abs/2603.02688)

- RAR (Retrieval-Augmented Robotics): introduces an iterative Retrieve-Reason-Act loop that enables robots to bridge the gap between visual documentation and physical actuation by querying an unstructured corpus via a retrieval module, grounding parts through a parts overview image, and executing plans with a multimodal LLM and robot controller in a physical environment.
- The system employs a multimodal LLM to interpret 2D diagrams, map them to 3D physical components through a rendered parts overview image, and generate step-by-step assembly sequences.
- Validated on the IKEA Furniture Assembly benchmark, the approach demonstrates that grounding robotic planning in retrieved visual documents significantly outperforms baselines relying on zero-shot reasoning or few-shot example retrieval.

---

[VisionCreator: A Native Visual-Generation Agentic Model with Understanding, Thinking, Planning and Creation](http://arxiv.org/abs/2603.02681)

- VisionCreator: introduces a native visual-generation agentic model that unifies Understanding, Thinking, Planning, and Creation (UTPC) capabilities, featuring a core VLM, a TaskAgent for routing, and a MetaAgent for metacognitive reasoning.
- The system utilizes VisGenData-4k, a dataset of high-quality trajectories generated by a metacognition-based VisionAgent, and is optimized through a two-stage Progressive Specialization Training curriculum.
- It implements Virtual Reinforcement Learning within the high-fidelity VisGenEnv simulation, guided by an LtrReward system that includes a vPlanJudger LLM evaluator to ensure logical coherence and execution validity.

---

[LLMs for High-Frequency Decision-Making: Normalized Action Reward-Guided Consistency Policy Optimization](http://arxiv.org/abs/2603.02680)

- NAR-CP (Normalized Action Reward guided Consistency Policy Optimization): introduces a reinforcement learning framework for high-frequency decision-making that utilizes LLMs to generate sub-task and composite policies, aligned via a consistency loss and optimized through normalized dense rewards.

- The architecture employs an Action Reward Normalize module to dynamically amplify reward variance using Z-score normalization, ensuring effective gradient signals in environments with subtle numerical state variations.

- It features a decoupled consistency policy learning method that aligns a joint policy derived from independent sub-tasks with a global composite policy to reduce policy bias and improve generalization in complex tasks like UAV tracking.


---

[Causal Learning Should Embrace the Wisdom of the Crowd](http://arxiv.org/abs/2603.02678)

- Causal learning by wisdom of the crowd: introduces a decentralized paradigm for recovering global causal structures by synthesizing fragmented and noisy insights from a distributed network of human experts and LLM-based simulated agents.
- The framework utilizes edge-wise and ordering-wise elicitation protocols to capture local or global structural beliefs, which are then reconciled through expert-level or query-level aggregation strategies to overcome statistical identifiability issues.
- The research defines a taxonomy of expert types based on trustworthiness, completeness, belief validity, and confidence levels to model the heterogeneity and reliability of crowd-contributed causal knowledge.

---

[Generalized Per-Agent Advantage Estimation for Multi-Agent Policy Optimization](http://arxiv.org/abs/2603.02654)

- GPAE (Generalized Per-Agent Advantage Estimator): introduces a multi-agent reinforcement learning framework that improves coordination and sample efficiency through precise per-agent advantage estimation using a novel value iteration operator, with Trajectory Sample, Replay Buffer, Centralized Critic, Per-Agent Advantage Estimator, Actor Policies, DT-ISR Weighting, and Target Value Calculator components.
- The architecture employs a centralized training and decentralized execution paradigm where a centralized critic estimates individual agent contributions by marginalizing over their specific actions while maintaining local observation dependency during execution.
- A double-truncated importance sampling ratio scheme is integrated to stabilize off-policy learning by mitigating variance from collective agent behaviors while preserving individual credit signals for effective multi-agent credit assignment.

---

[Credibility Governance: A Social Mechanism for Collective Self-Correction under Weak Truth Signals](http://arxiv.org/abs/2603.02640)

- CG (Credibility Governance): introduces a social mechanism for collective self-correction that reallocates influence by learning which agents consistently track evolving public evidence, with Physical World, Opinion World, LLM-driven Agents (includes misaligned majority-, truth-aligned minority-, and high-conviction core-agents), Credibility Governance Mechanism, Social Signal, Agent Credibility, Anti-Bubble Penalty, and Supporter Quality.
- The mechanism maintains dynamic credibility scores for agents based on their long-run alignment with emerging evidence, filtering short-lived noise through an anti-bubble penalty.
- Evaluated in the POLIS simulation environment, the approach demonstrates faster recovery to true states and improved robustness under adversarial pressure compared to vote-based or stake-weighted baselines.

---

[StitchCUDA: An Automated Multi-Agents End-to-End GPU Programing Framework with Rubric-based Agentic Reinforcement Learning](http://arxiv.org/abs/2603.02637)

- StitchCUDA: introduces a multi-agent framework for automated end-to-end GPU program generation, which includes planning-, coding-, and verification-agents, utilizing a RAG Database, Shared State, and Reinforcement Learning Loop.
- The framework integrates rubric-based agentic reinforcement learning to improve the Coder's ability to implement advanced CUDA techniques while mitigating reward hacking and degenerate behaviors.
- It utilizes an iterative "plan-code-profile-refine" loop supported by Nsys/NCU profiling and a RAG module to achieve high-performance CUDA implementations across complex multi-kernel workloads.

---

[CROSS-FAMILY SPECULATIVE PREFILL: TRAINING-FREE LONG-CONTEXT COMPRESSION WITH SMALL DRAFT MODELS](http://arxiv.org/abs/2603.02631)

- Cross-Family Speculative Prefill: introduces a training-free prompt compression method that leverages a lightweight draft model from one family to guide long-context processing for a target model from a different family, with Small Draft LLM, Attention Importance Estimator, Top-K Block Selector, Target LLM Tokenizer, and Target LLM.
- The system utilizes attention-based token importance scores from the draft model to select and concatenate the most salient text chunks, effectively reducing the input length before target model inference.
- Experimental results demonstrate that this cross-model approach maintains high task performance while achieving up to an 18x reduction in time-to-first-token for long-context agentic workloads.

---

[MASPOB: Bandit-Based Prompt Optimization for Multi-Agent Systems with Graph Neural Networks](http://arxiv.org/abs/2603.02630)

- MASPOB (Multi-Agent System Prompt Optimization via Bandits): introduces a sample-efficient framework for optimizing agent-specific prompts in fixed-topology multi-agent systems, with a Joint Prompt Domain, Embedding Model, Topology-Graph Construction, Graph Neural Network Surrogate, Linear Upper Confidence Bound, Coordinate Ascent, Information Matrix, History Data Buffer, and Multi-Agent System Execution, which includes AnswerGenerate-, ScEnsemble-, and Programmer-agents.
- The architecture integrates a Graph Neural Network surrogate to model inter-agent dependencies and a Linear Upper Confidence Bound mechanism to balance exploitation of predicted scores with exploration of uncertain regions.
- To mitigate combinatorial explosion, the framework employs coordinate ascent to decompose the joint search space into tractable univariate updates, achieving superior performance across diverse benchmarks.

---

[Post Hoc Extraction of Pareto Fronts for Continuous Control](http://arxiv.org/abs/2603.02628)

- MAPEX (Mixed Advantage Pareto Extraction): introduces an offline multi-objective reinforcement learning method that extracts a Pareto front by reusing pre-trained specialist policies, critics, and replay buffers.
- The framework identifies sparse regions in the objective space to derive target weights, then constructs a hybrid buffer and calculates a mixed advantage signal to guide policy updates.
- By employing advantage-weighted regression for post hoc extraction, the method achieves comparable performance to established baselines at a fraction of the environment interaction cost.

---

[See and Remember: A Multimodal Agent for Web Traversal](http://arxiv.org/abs/2603.02626)

- V-GEMS (Visual Grounding and Explicit Memory System): introduces a multimodal agent architecture for resilient web traversal, integrating an Explorer-Critic dual-agent system with specialized modules for memory, calculation, and adaptive perception.
- The architecture employs an adaptive US Calculator to dynamically switch between LLM and VLM processing, ensuring high-fidelity perception only when textual DOM representations are insufficient.
- The system utilizes a stateful URL Stack to prevent navigational loops through deterministic backtracking and a Symbolic Counter to resolve arithmetic hallucinations in complex data retrieval tasks.

---

[Think, But Don’t Overthink: Reproducing Recursive Language Models](http://arxiv.org/abs/2603.02615)

- RLM (Recursive Language Models): introduces a task-agnostic inference paradigm that processes near-infinite contexts by offloading prompts into an external REPL environment, with Root LM, REPL Environment, Persistent Variable, and Recursive Sub-calls.
- The framework enables models to programmatically decompose complex reasoning tasks into nested sub-calls, though scaling recursion depth beyond one level often triggers "overthinking" and performance degradation.
- Empirical results using DeepSeek v3.2 and Kimi K2 demonstrate that excessive recursion causes exponential latency inflation, token explosions, and loss of context anchoring through parametric hallucinations.

---

[Heterogeneous Agent Collaborative Reinforcement Learning](http://arxiv.org/abs/2603.02604)

- HACRL (Heterogeneous Agent Collaborative Reinforcement Learning): introduces a collaborative optimization paradigm where heterogeneous LLMs share verified rollouts to mutually improve while operating independently during inference.
- The proposed HACPO algorithm incorporates agent-capability-aware advantage estimation and gradient modulation to address performance gaps and policy distribution shifts between participating models.
- Experimental results demonstrate that bidirectional knowledge transfer consistently enhances reasoning capabilities across diverse model architectures while significantly reducing rollout costs compared to isolated training.

---

[AgentAssay: Token-Efficient Regression Testing for Non-Deterministic AI Agent Workflows](http://arxiv.org/abs/2603.02601)

- AgentAssay: introduces a token-efficient framework for regression testing non-deterministic AI agent workflows, with Agent Execution, Trace Collection, Behavioral Fingerprinting, Statistical Analysis, Three-Valued Verdict, Core Engine, Coverage Analyzer, Mutation Engine, Framework Adapters, Trace Store, and Deployment Gate.
- The architecture extracts low-dimensional behavioral fingerprints from execution traces to enable multivariate regression detection via Hotelling’s T-squared tests, reducing the number of trials required for statistical significance compared to univariate pass-rate testing.
- The system implements a three-valued verdict semantics and trace-first offline analysis to verify agent reliability across prompts, tools, and LLMs within CI/CD pipelines while minimizing token consumption.

---

[Extending the Formalism and Theoretical Foundations of Cryptography to AI](http://arxiv.org/abs/2603.02590)

- AIOracle: introduces a formal cryptographic foundation for agentic access control, utilizing LEARN and INFER algorithms to model the lifecycle of LLM-based agents.
- The framework employs a dual-agent construction including creative- and boring-AIOracle components, where the former proposes computational results and the latter filters them via decisional logic.
- This modular decomposition enables provable security reductions for helpfulness and harmlessness objectives by separating generation from verification.

---

[LiveAgentBench: Comprehensive Benchmarking of Agentic Systems Across 104 Real-World Challenges](http://arxiv.org/abs/2603.02586)

- LiveAgentBench (Social Perception-Driven Data Generation): introduces a comprehensive benchmark for evaluating autonomous agents across 104 real-world scenarios, utilizing the SPDG method to ensure task complexity and result verifiability.
- The SPDG process integrates LLM-based filtering for non-retrievability with human-in-the-loop annotation to transform open-ended social media queries into standardized, tool-dependent evaluation tasks.
- The benchmark evaluates agentic systems across diverse modalities including browser operations, file manipulation, and audio-video comprehension, revealing significant performance gaps between current LLMs and human capabilities.

---

[Agentic Mixed-Source Multi-Modal Misinformation Detection with Adaptive Test-Time Scaling](http://arxiv.org/abs/2603.02519)

- AgentM3D (Agentic Mixed-Source Multi-Modal Misinformation Detection): introduces a multi-agent framework for zero-shot misinformation detection, which includes planning-, textual-, visual-, cross-modal-, and critique-agents.
- The architecture employs a hierarchical cascade to sequentially verify textual veracity, visual authenticity, and cross-modal consistency while leveraging adaptive test-time scaling to improve reasoning robustness.
- It integrates external tools for logic and forgery analysis to score reasoning candidates and implements an early-stopping mechanism to optimize computational efficiency.

---

[Probing More-Than-Human Representation in Crisis Resilience Planning: An HCI Researcher Perspective](http://arxiv.org/abs/2603.02514)

- More-Than-Human Representation: introduces a workshop-based methodology to examine how HCI researchers conceptualize non-human perspectives in crisis planning, with Scenario-Based Discussion, Voice-based Conversational Agent, Immersive Embodied Prototype, OpenAI Whisper, and FigJam.
- The approach utilizes LLM-powered conversational agents and XR-based spatial embodiments to elicit critical reflection on representational choices such as voice, realism, and authority.
- Findings highlight that giving voice to non-humans is a complex design challenge involving tensions between legitimacy, authenticity, and the centralization of interaction.

---

[Human-Certified Module Repositories for the AI Age](http://arxiv.org/abs/2603.02512)

- HCMR (Human-Certified Module Repositories): introduces an architectural model for constructing trustworthy software by blending human oversight with automated analysis to certify reusable modules for safe assembly by both humans and AI agents.
- The framework implements a four-stage certification pipeline comprising automated vetting, human-led security review, sandboxed behavioral validation, and formal release with machine-readable metadata.
- It features an AI-aware assembly pipeline that utilizes constraint-based discovery and contract-based synthesis to prevent the ingestion of unvetted dependencies in LLM-generated code.

---

[ParEVO: Synthesizing Code for Irregular Data: High-Performance Parallelism through Agentic Evolution](http://arxiv.org/abs/2603.02510)

- ParEVO: introduces a framework for synthesizing parallel code for irregular data structures, with Human Expert Context, Evolutionary LLM Agent (ECA), Candidate Parallel Algorithms, Evaluation Framework, MAP-Elites Selection, Optimized Algorithm, Parlay-Instruct Corpus, and fine-tuned LLMs, where the system iteratively refines candidate algorithms through an execution-based evaluation loop.
- The system leverages specialized fine-tuned LLMs—including DeepSeek-Parlay, Qwen-Parlay, Qwen-Rust, and Gemini-2.5-Parlay—to map natural language instructions to parallel primitives from the ParlayLib library.
- It employs a MAP-Elites selection strategy alongside deterministic feedback from compilers, race detectors, and performance profilers to optimize code for functional correctness and runtime scalability.

---

[Revealing Positive and Negative Role Models to Help People Make Good Decisions](http://arxiv.org/abs/2603.02495)

- Information Disclosure and Intervention Framework: introduces a social welfare maximization system where a social planner reveals role model labels in a bipartite graph to steer agents toward positive emulation.
- It utilizes a submodular proxy welfare function to overcome the supermodularity challenges encountered when revealing negative labels, ensuring constant-factor approximation for budgeted target selection.
- The framework incorporates targeted intervention strategies to directly connect high-risk agents to positive influences and a coverage radius model to expand the reach of positive role models.

---

[What Capable Agents Must Know: Selection Theorems for Robust Decision-Making under Uncertainty](http://arxiv.org/abs/2603.02491)

- Selection Theorems: introduces a theoretical framework proving that low average-case regret on action-conditioned prediction tasks forces agents to implement structured internal states, with Agent, Internal State, Memory, Predictive World Model, Policy, Environment, Report Bit, Tests, and Predictive-State Representations.
- The approach reduces predictive modeling to binary betting decisions, showing that performance guarantees necessitate the recovery of interventional transition kernels in fully observed settings and belief-like memory under partial observability.
- The research demonstrates that structured task families select for informational modularity and regime-tracking variables, providing a normative explanation for the emergence of convergent representations in robust AI systems.

---

[VOICEAGENTRAG: Solving the RAG Latency Bottleneck in Real-Time Voice Agents Using Dual-Agent Architectures](http://arxiv.org/abs/2603.02206)

- VOICEAGENTRAG: introduces a dual-agent memory router that decouples retrieval from response generation to solve latency bottlenecks in real-time voice agents, which includes prediction- and response-agents.
- The background Slow Thinker agent predicts likely follow-up topics and pre-fetches relevant document chunks into a sub-millisecond semantic cache during inter-turn pauses.
- The system utilizes an in-memory FAISS index to bypass remote vector database latency, achieving a 75% cache hit rate across ten conversation scenarios.

---

[AI-FOR-SCIENCE LOW-CODE PLATFORM WITH BAYESIAN ADVERSARIAL MULTI-AGENT FRAMEWORK](http://arxiv.org/abs/2603.03233)

- LCP (AI-for-Science Low-Code Platform): introduces a multi-agent framework that employs a Bayesian recursive co-updating strategy to iteratively refine generated code and test cases through an adversarial dynamic between Task Manager-, Solution Generator-, and Evaluator-agents.
- The architecture incorporates a non-LLM-based Bayesian updating rule to systematically reduce evaluation uncertainty and mitigate the system's dependence on the reliability of individual LLMs.
- By translating high-level natural language into domain-specific requirements, the platform enables non-technical scientists to generate executable code for complex cross-disciplinary scientific tasks.

---

[Saarthi for AGI: Towards Domain-Specific General Intelligence for Formal Verification](http://arxiv.org/abs/2603.03175)

- Saarthi: introduces an agentic AI framework for end-to-end formal verification, employing multi-agent collaboration, GraphRAG-based knowledge grounding, and a structured rulebook to automate SystemVerilog Assertion generation. 
- The architecture includes lead-, property-generation-, syntax-fixing-, and coverage-agents that iteratively refine verification artifacts to accelerate coverage closure and minimize manual engineering effort. 
- The framework incorporates a specialized multi-agent Root Cause Analysis module for diagnosing counter-examples and a Human-in-the-Loop pipeline to build a self-improving learning cache from expert interventions. 

---

[OrchMAS: Orchestrated Reasoning with Multi Collaborative Heterogeneous Scientific Expert Structured Agents](http://arxiv.org/abs/2603.03005)

- OrchMAS (Orchestrated Multi-Agent System): introduces a task-adaptive multi-agent reasoning framework that dynamically constructs domain-aware pipelines by instantiating specialized expert agents with tailored prompts through a two-tier heterogeneous architecture.
- The system employs a coordinator LLM for high-level planning and role assignment, while a separate execution model performs subtasks and includes research-, planning-, solving-, and verification-agents.
- It utilizes Agent Group Relative Policy Optimization (A-GRPO) and a layered reward architecture to refine orchestration strategies, enhancing performance in complex scientific and multi-hop reasoning tasks.

---

[PCMDP (Partially Controllable Markov Decision Process): Learning in Markov Decision Processes with Exogenous Dynamics](http://arxiv.org/abs/2603.02862)

- PCMDP (Partially Controllable Markov Decision Process): introduces a reinforcement learning framework that factorizes state spaces into endogenous and exogenous components to improve sample efficiency, featuring EXAVI and EXAQ agents.
- The EXAVI agent functions as a model-based solver that estimates only the exogenous transition probabilities while leveraging prior knowledge of controllable dynamics to eliminate the need for active exploration.
- The EXAQ agent implements a model-free approach using an empirical Bellman operator to perform synchronous updates across all controllable configurations for a given exogenous transition, enabling faster convergence in high-dimensional environments.

---

[Contextualized Privacy Defense for LLM Agents](http://arxiv.org/abs/2603.02983)

- CDI (Contextualized Defense Instructing): introduces a proactive privacy defense paradigm where a separate instructor model generates context-aware guidance to steer LLM agents' actions during multi-step execution, including instructor- and sender-agents.
- The framework utilizes an experience-driven optimization process that converts privacy-violating failure trajectories into reinforcement learning environments to train the instructor model via Group Relative Policy Optimization (GRPO).
- Evaluation across diverse social scenarios demonstrates that CDI achieves a superior balance between privacy preservation and helpfulness compared to static prompting or passive guarding mechanisms.

---

[BrandFusion: A Multi-Agent Framework for Seamless Brand Integration in Text-to-Video Generation](http://arxiv.org/abs/2603.02816)

- BrandFusion: introduces a multi-agent framework for seamless brand integration in text-to-video generation, with Brand Knowledge Base, Working Context, Brand Selection Agent, Strategy Generation Agent, Prompt Rewriting Agent, Critic Agent, Experience Learning Agent, Diagnostic Prompt Generator, Brand Quality Evaluator, Synthetic Data Generator, T2V Fine-Tuning, and Video Generation Tool.
- The system employs an offline phase to probe model priors and adapt to novel brands via lightweight fine-tuning, followed by an online phase for iterative prompt optimization through collaborative agent reasoning.
- The online phase includes brand selection-, strategy generation-, prompt rewriting-, critic-, and experience learning-agents to ensure semantic fidelity, brand recognizability, and natural visual coherence in generated content.

---

[VSearcher: Long-Horizon Multimodal Search Agent via Reinforcement Learning](http://arxiv.org/abs/2603.02795)

- VSearcher: introduces a post-training framework for transforming static multimodal models into autonomous agents capable of long-horizon, multi-turn tool use in real-world web environments through reinforcement learning.

- The system utilizes an Iterative Injection-based Data Synthesis pipeline to generate high-difficulty multimodal questions by progressively injecting rare textual information and critical visual context into simple seed entities.

- Training follows an SFT-then-RL pipeline, employing rejection sampling from a teacher model and Group Reward Proximal Optimization to enhance agent navigation across text search, image search, and web browsing tools.


---

[Next Embedding Prediction Makes World Models Stronger](http://arxiv.org/abs/2603.02765)

- NE-Dreamer (Next-Embedding Dreamer): introduces a decoder-free model-based reinforcement learning agent that replaces pixel reconstruction with next-embedding prediction using a causal temporal transformer, with Encoder, RSSM, Causal Temporal Transformer, Alignment Loss, and Actor-Critic components.
- The architecture utilizes a Recurrent State-Space Model (RSSM) to maintain latent states while a transformer predicts future encoder embeddings aligned via a Barlow Twins redundancy-reduction objective.
- By enforcing temporal predictive alignment in representation space, the agent achieves superior performance on memory-intensive navigation tasks without the computational burden of a pixel decoder.

---

[ShareVerse: Multi-Agent Consistent Video Generation for Shared World Modeling](http://arxiv.org/abs/2603.02697)

- ShareVerse (Multi-Agent Consistent Video Generation for Shared World Modeling): introduces a video generation framework for multi-agent shared world modeling, with Causal 3D VAE (spatio-temporal video compression), Raymap Encoder (camera trajectory embedding), Cross-Agent Attention Block (inter-agent feature synchronization), Diffusion Transformer (latent distribution modeling), Projector (feature space mapping), and Rotary Position Embedding (relative positional encoding).
- The architecture utilizes spatial concatenation of four-view videos per agent to maintain internal geometric consistency while conditioning generation on normalized camera poses transformed into raymap embeddings.
- Cross-agent attention modules facilitate the transmission of spatio-temporal information between independent agents to ensure global consistency in shared environments and accurate perception of dynamic agent positions.

---

[SORRYDB: CAN AI PROVERS COMPLETE REAL-WORLD LEAN THEOREMS?](http://arxiv.org/abs/2603.02668)

- SorryDB: introduces a dynamically-updating benchmark of open Lean tasks drawn from real-world formalization projects, with Lean4 Reservoir, GitHub Repositories, Indexer, Task Metadata, SorryDB Database, Proposer, Deterministic Reviewer, and LeanSearch Tool.
- The system automates the extraction of "sorry" placeholders from active GitHub repositories to provide a continuously evolving stream of mathematical challenges that mitigates test-set contamination.
- Evaluation results demonstrate that agentic architectures utilizing LLMs and iterative compiler feedback outperform specialized symbolic provers and one-shot generation methods on complex formalization tasks.

---

[Convex and Non-convex Federated Learning with Stale Stochastic Gradients: Diminishing Step Size is All You Need](http://arxiv.org/abs/2603.02639)

- Distributed Stochastic Optimization Framework: introduces a system for federated learning where a central server aggregates stale and potentially biased stochastic gradient estimates from decentralized local agents.
- The architecture employs a projection operator for constrained domains and a pre-determined diminishing step size sequence to mitigate the effects of communication delays.
- Theoretical analysis proves this approach achieves optimal convergence rates for nonconvex, convex, and strongly convex objectives under a scaled-delay model.

---

[AnchorDrive: LLM Scenario Rollout with Anchor-Guided Diffusion Regeneration for Safety-Critical Scenario Generation](http://arxiv.org/abs/2603.02542)

- AnchorDrive: introduces a two-stage framework for generating realistic and controllable safety-critical driving scenarios, with Global State Constructor, LLM-Based Driver Agent, LLM-Based Plan Assessor, Plan Memory, LLM-Based Anchor Extractor, and Anchor-Guided Diffusion Trajectory Regeneration Module.
- The system decouples scenario generation into a controllable rollout stage using an LLM driver agent and a realism optimization stage using a guided diffusion model.
- LLM-extracted anchor points serve as spatiotemporal constraints to ensure the regenerated trajectories maintain the intended adversarial interaction logic and semantic intent.

---

[EIMC: Efficient Instance-aware Multi-modal Collaborative Perception](http://arxiv.org/abs/2603.02532)

- EIMC (Efficient Instance-aware Multi-modal Collaborative Perception): introduces an early collaborative perception paradigm for autonomous driving, with LiDAR-branch (extracts geometric point cloud features), Camera-branch (predicts depth and transforms images), Mix-Voxel module (aggregates cross-agent collaborative voxels), Heterogeneous Modality Fusion (HMF) (fuses multi-modal BEV features), Heatmap Generator (computes per-pixel confidence maps), Instance Completion (IC) (queries complementary peer instance vectors), and Instance Refinement (IR) (refines features via self-attention).
- The framework utilizes a heatmap-driven consensus protocol to identify perception discrepancies, facilitating instance-centric messaging that significantly reduces communication bandwidth while recovering occluded objects.
- It achieves state-of-the-art 3D detection results on OPV2V and DAIR-V2X benchmarks by optimizing multi-modal feature alignment and cross-agent knowledge transfer.

---

[A Covering Framework for Offline POMDPs Learning using Belief Space Metric](http://arxiv.org/abs/2603.03191)

- Covering Framework: introduces a theoretical analysis framework for offline POMDP evaluation that utilizes belief space metric structures and state abstraction to mitigate the curses of horizon and memory.
- The framework employs ε-covering to map complex history trajectories into a simplified abstract belief space, enabling tighter error bounds for algorithms like double sampling Bellman error minimization.
- By assuming Lipschitz continuity of value functions in the belief space, the approach provides polynomial sample complexity guarantees for memory-based policies in partially observable environments.

---

[D-GVIO: A Buffer-Driven and Efficient Decentralized GNSS-Visual-Inertial State Estimator for Multi-Agent Systems](http://arxiv.org/abs/2603.01404)

- D-GVIO (Decentralized GNSS-Visual-Inertial Odometry): introduces a buffer-driven decentralized state estimation framework for multi-agent systems, with Sensor Suite, Pre-Processing Module, Buffering Strategy, Propagation Module, Measurement Update Module, Collaborative Update Module, L-IEKF, and VLAD Communication.
- The architecture utilizes covariance segmentation and a buffer-based re-propagation strategy to manage delayed measurements and reduce computational overhead on resource-constrained platforms.
- A novel adaptive outlier detection mechanism utilizes visual-inertial odometry kinematics and Boltzmann entropy to dynamically filter unreliable GNSS observations in challenging environments.

---

[Beyond Reward: A Bounded Measure of Agent–Environment Coupling](http://arxiv.org/abs/2603.01283)

- IDT (Information Digital Twin): introduces a real-time monitoring framework that computes bi-predictability (P) to quantify bidirectional agent-environment coupling without requiring internal model access, with Agent, Environment, Information Digital Twin (IDT), P Calculator, P Controller, Observation Modulation, Action Modulation, and Sliding Window Memory.
- The architecture employs a P Calculator to derive mutual information from the observation-action-outcome stream and a P Controller to detect anomalies against a task-independent baseline.
- Evaluation on MuJoCo HalfCheetah shows the multi-channel diagnostic decomposition detects 89.3% of perturbations with 4.4x lower median latency compared to traditional reward-based monitoring.

---

[SphUnc: Hyperspherical Uncertainty Decomposition and Causal Identification via Information Geometry](http://arxiv.org/abs/2603.01168)

- SphUnc (Hyperspherical Uncertainty Decomposition and Causal Identification): introduces a unified framework for multi-agent systems that maps features to unit hypersphere latents using von Mises-Fisher distributions to decompose uncertainty into epistemic and aleatoric components through information-geometric fusion.
- The architecture integrates a structural causal model on spherical latents with hypergraph message passing to enable directed influence identification and interventional reasoning via sample-based simulation.
- The system optimizes a composite objective balancing predictive accuracy, entropy calibration, and causal fidelity to provide trustworthy confidence measures in complex relational settings.

---

[FastCode: Fast and Cost-Efficient Code Understanding and Reasoning](http://arxiv.org/abs/2603.01012)

- FastCode: introduces a cost-aware agentic framework that decouples repository exploration from content consumption through a scouting-first paradigm, utilizing a semantic-structural map to pinpoint relevant code elements without full-text ingestion; it includes query augmentation- and reasoning-LLMs.
- The system employs a dual-index mechanism and multi-layered dependency graphs to enable precise navigation, while a cost-aware policy dynamically optimizes context assembly to maximize reasoning confidence within token limits.
- Evaluations across SWE-QA and LongCodeQA benchmarks show significant reductions in token consumption and costs by up to 90% while outperforming state-of-the-art baselines in reasoning accuracy.

---

[Black Hole Search: Dynamics, Distribution, and Emergence](http://arxiv.org/abs/2603.00766)

- ALGO_BHS_SCATTERED (Algorithm for Black Hole Search with Scattered Agents): introduces a distributed algorithm for locating a black hole in 1-bounded 1-interval connected dynamic graphs using 2δBH + 17 scattered mobile agents.
- The approach employs Individual Cautious Movement and node-based whiteboards to facilitate agent coordination and threat detection without global graph knowledge.
- The research further addresses the Eventual Black Hole Search problem in static graphs by implementing a Cautious Chain Movement strategy with four co-located agents.

---

[Theory of Code Space: Do Code Agents Understand Software Architecture?](http://arxiv.org/abs/2603.00601)

- TOCS (Theory of Code Space): introduces a diagnostic benchmark to evaluate the ability of AI code agents to construct and maintain coherent architectural beliefs during codebase exploration.
- The framework utilizes a procedural generator to create Python projects with planted invariants and a partial observability harness that enforces exploration budgets.
- Evaluation of frontier LLMs, including GPT-5.3-Codex, Claude Sonnet 4.6, and Gemini models, reveals significant gaps in belief state maintenance and catastrophic forgetting.

---

[Social Norm Reasoning in Multimodal Language Models: An Evaluation](http://arxiv.org/abs/2603.03590)

- Evaluation Framework for Social Norm Reasoning: introduces a systematic assessment of MLLMs' ability to identify and reason about social norms across textual and visual modalities using thirty text-based and thirty image-based stories.
- The framework evaluates five state-of-the-art models, including GPT-4o, Gemini 2.0 Flash, Qwen-2.5VL, Intern-VL3, and Meta LLaMa-4 Maverick, against human-derived ground truth across five scenarios and six variants of norm adherence or violation.
- The study reveals that MLLMs perform significantly better on text than images and find reasoning about complex metanorms, such as meta-punishment, particularly challenging.

---

[BUILD, JUDGE, OPTIMIZE: A BLUEPRINT FOR CONTINUOUS IMPROVEMENT OF MULTI-AGENT CONSUMER ASSISTANTS](http://arxiv.org/abs/2603.03565)

- MAGIC (Multi-Agent Grocery Intelligent Concierge): introduces a blueprint for evaluating and optimizing conversational shopping assistants, which includes orchestrator-, query generation-, item selection-, and user persona-agents.
- The system utilizes a calibrated LLM-as-judge to grade interaction traces against a structured rubric and employs MAMUT (Multi-Agent Multi-Turn) GEPA for joint prompt optimization across the multi-agent system.
- By utilizing a Customer Simulator, the framework enables iterative improvement of multi-turn interactions and resolves coordination failures in preference-sensitive, high-ambiguity domains.

---

[Pricing for Information Revelation in Demand Response: A Strategic Communication Approach](http://arxiv.org/abs/2603.03560)

- SIT (Strategic Information Transmission): introduces a game-theoretic framework for demand response that utilizes ex-ante pricing as a mechanism design lever to elicit private flexibility information from strategic consumers, with an Aggregator (Receiver), Consumers (Senders), a Heterogeneous Population, Smart Meters, and a Best Response Dynamic Algorithm.
- The system proves that complex multi-sender strategic interactions decouple into independent single-sender subgames under active participation conditions, enabling tractable analysis of large-scale heterogeneous populations.
- Simulations demonstrate that the proposed mechanism recovers up to 95% of the ideal social welfare by aligning the fixed tariff with the expected marginal cost of the system.

---

[Probabilistic Occupancy Grid for Radio-Based SLAM](http://arxiv.org/abs/2603.03559)

- MP-SLAM (Multipath-based Simultaneous Localization and Mapping) with Probabilistic Occupancy Grid: introduces a Bayesian framework that jointly localizes a mobile agent and estimates environment occupancy states by integrating a grid-based map into a factor graph representation of radio multipath propagation.
- The system utilizes a hierarchical geometric model incorporating surface feature vectors and occupancy grid cells to resolve specular reflections up to double bounces while capturing geometric details.
- Inference is performed using the sum-product algorithm for message passing across localization- and occupancy grid mapping-subgraphs, enabling the estimation of material properties and geometry through amplitude statistics.

---

[MOLT DYNAMICS: EMERGENT SOCIAL PHENOMENA IN AUTONOMOUS AI AGENT POPULATIONS](http://arxiv.org/abs/2603.03555)

- OpenClaw: introduces an open-source autonomous agent framework that enables decentralized interaction within the MoltBook Platform using a Gateway, Agent Runtime, Skills Platform, Messaging Platforms, External Tools & APIs, and LLM Providers including Claude, GPT-4, and open-source models.
- The research characterizes "Molt Dynamics" by observing 770,000 agents to identify emergent core-periphery role specialization and power-law distributed information cascades with saturating adoption dynamics.
- Empirical findings demonstrate that while decentralized coordination is detectable, current multi-agent collaborative task resolution suffers from significant coordination overhead compared to single-agent baselines.

---

[The Controllability Trap: A Governance Framework for Military AI Agents](http://arxiv.org/abs/2603.03515)

- AMAGF (Agentic Military AI Governance Framework): introduces a measurable architecture for maintaining human control over LLM-based agents, which includes interpretation-, planning-, world modeling-, tool use-, and coordination-components.
- The framework utilizes a composite Control Quality Score (CQS) to drive a graduated response protocol, ranging from normal operations to architecturally enforced safe states.
- It assigns specific governance responsibilities across five institutional actors, including developers, procurement agencies, and operational commanders, to bridge the gap between technical safety and organizational accountability.

---

[Optimal trajectory-guided stochastic co-optimization for e-fuel system design and real-time operation](http://arxiv.org/abs/2603.03484)

- MasCOR (Machine Learning-assisted Co-Optimization Framework): introduces a bilevel optimization approach for e-fuel systems, with WGAN-GP, LP Solver, Oracle Dataset, actor- and critic-transformers, Bayesian Optimization, Design Token, and Renewable-trend Token.
- The framework utilizes a generative model to capture region-specific temporal uncertainty and an offline reinforcement learning agent to learn optimal operational policies from presolved linear programming trajectories.
- By encoding design and renewable trends into conditional tokens, the single agent generalizes across diverse configurations and enables rapid parallel evaluation for real-time operation and system design.

---

[Beyond Pixel Histories: World Models with Persistent 3D State](http://arxiv.org/abs/2603.03482)

- PERSIST (Persistent Environment Representations for Simulating Interactive Space-Time): introduces a world modeling paradigm that tracks a persistent 3D latent state to maintain spatial consistency and long-horizon stability in interactive video generation, utilizing a world denoiser (predicts 3D scene evolution), camera model (tracks agent viewpoint), world projection (differentiable screen-space mapping), pixel denoiser (learned deferred rendering shader), 2D-VAE (pixel-to-latent compression), 3D-VAE (voxel-to-latent compression), and persistent 3D latent state (dynamic spatial memory).
- The framework decomposes world simulation into world-frame prediction, camera tracking, and a world-to-pixel generation module that functions as a learned deferred shader.
- By simulating a dynamic 3D scene rather than relying on pixel-based histories, the model enables explicit 3D scene editing and maintains coherence even for unobserved off-screen events.

---

[Minimax Optimal Strategy for Delayed Observations in Online Reinforcement Learning](http://arxiv.org/abs/2603.03480)

- MVP-Delayed: introduces a reinforcement learning framework for stochastic delayed Markov Decision Processes using an augmented MDP, MVP-Est optimistic estimation, an action queue, an inter-arrival time counter, visit count memory, and Bernstein-type bonuses.
- The system architecture transforms delayed observations into a standard MDP setting by incorporating the last observed state and unresolved actions into the augmented state representation.
- The framework achieves minimax optimal regret by decomposing transition dynamics into known and unknown components, mitigating exponential state space complexity through structured statistical estimation.

---

[ASYMMETRIC GOAL DRIFT IN CODING AGENTS UNDER VALUE CONFLICT](http://arxiv.org/abs/2603.03456)

- Constraint-Drift (OpenCode-based Goal Drift Evaluation Framework): introduces a methodology to quantify how autonomous coding agents violate explicit system prompt constraints, utilizing a system that includes coding- and judge-LLMs.
- The framework orchestrates multi-step coding tasks within realistic repositories, employing adversarial codebase comments to persuade agents to prioritize competing values like utility over privacy.
- Experimental results demonstrate that LLMs exhibit asymmetric drift, frequently abandoning instructions that conflict with strongly-held internal safety values while resisting drift in the opposite direction.

---

[Proact-VL: A Proactive VideoLLM for Real-Time AI Companions](http://arxiv.org/abs/2603.03447)

- Proact-VL (Proactive VideoLLM): introduces a multimodal framework for real-time AI companions, featuring chunk-wise video processing, an autonomous response mechanism, and a data pipeline that includes ASR-, labeling-, and polishing-agents.
- The architecture utilizes a lightweight gated MLP response head to analyze hidden states from a specialized FLAG token, enabling the model to decide when to generate content or remain silent.
- The system employs a dual-cache sliding-window KV-cache and reverse-RoPE operations to support unbounded streaming while maintaining positional consistency during context eviction.

---

[MULTI-AGENT-BASED SIMULATION OF ARCHAEOLOGICAL MOBILITY IN UNEVEN LANDSCAPES](http://arxiv.org/abs/2603.03390)

- Multi-agent-based modeling framework: introduces a hybrid navigation strategy for simulating archaeological mobility in uneven landscapes, with Global Path Planner, Local Adaptive Navigator, Terrain Reconstruction Engine, Heterogeneous Agent Model, and Immersive Visualization Module.
- The system integrates real-world Digital Elevation Models into high-fidelity three-dimensional simulation environments to preserve metrically accurate terrain constraints such as slope and visibility.
- It employs a hierarchical action-selection policy that offloads local obstacle avoidance to tabular Q-learning while maintaining consistency with long-range A* trajectories for efficient large-scale simulation.

---

[Act–Observe–Rewrite: Multimodal Coding Agents as In-Context Policy Learners for Robot Manipulation](http://arxiv.org/abs/2603.04466)

- AOR (Act–Observe–Rewrite): introduces a two-timescale framework for in-context policy learning that synthesizes executable Python controller code between trials, with Vision Pipeline (converts RGB-D images to features), Feature Dict (structured state representation), Controller (executable Python motor-control code), Episodic Memory (stores episode outcomes and images), and Multimodal LLM agent (diagnoses failures and rewrites code).
- The framework separates real-time motor control in a fast loop from inter-episode reasoning in a slow loop to maintain deterministic timing while allowing for complex policy updates.
- By using interpretable code as the policy representation, the LLM can diagnose root causes of failure, such as coordinate convention errors, and implement targeted architectural changes without gradient updates.

---

[Multi-Agent Influence Diagrams to Hybrid Threat Modeling](http://arxiv.org/abs/2603.03526)

- MAID (Multi-Agent Influence Diagram): introduces an integrated probabilistic and game-theoretic framework to assess the effectiveness of counter-hybrid threat measures by modeling strategic interactions between state-like agents under deep uncertainty.
- The approach unifies previously bifurcated modeling methods by balancing countermeasure costs, deterrence capacity, and damage mitigation potential within a unified causal structure.
- Analysis of 1000 semi-synthetic cyber-attack scenarios identifies market restrictions and intelligence sharing as highly effective strategies for achieving strategic equilibria.

---

#### 2nd March 2026

[Diagnosing Retrieval vs. Utilization Bottlenecks in LLM Agent Memory](http://arxiv.org/abs/2603.02473)

- Diagnostic Probing Framework: introduces a diagnostic framework that sits at the retrieval-to-generation boundary to independently measure retrieval relevance, memory utilization, and failure modes, utilizing backbone- and judge-LLMs.
- The architecture evaluates three write strategies—raw chunks, fact extraction, and summarization—against three retrieval methods including cosine similarity, BM25, and hybrid reranking.
- Analysis reveals that retrieval failure is the primary bottleneck in memory-augmented systems, whereas raw chunked storage often matches or exceeds the performance of lossy alternatives.

---

[ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering](http://arxiv.org/abs/2603.02438)

- ORCA (Orchestrated Reasoning with Collaborative Agents): introduces a multi-agent framework for Document Visual Question Answering that decomposes complex queries into logical steps and routes them to specialized agents, with Thinker Agent, Router, Orchestrator, Agent Dock, Debate Agent, Evaluation Agent, Thesis Agent, Antithesis Agent, Judge Agent, and Sanity Checker; it includes thinker-, router-, debate-, and judge-agents.
- The architecture employs a five-stage pipeline featuring context understanding, collaborative execution, stress testing, multi-turn conversation, and answer refinement to ensure high reliability.
- The system utilizes a conditional activation mechanism for adversarial verification, employing debate and adjudication only when uncertainty is detected between the thinker and expert agents.

---

[Personalized Multi-Agent Average Reward TD-Learning via Joint Linear Approximation](http://arxiv.org/abs/2603.02426)

- PMAAR-TD (Personalized Multi-Agent Average-Reward TD-Learning): introduces a cooperative reinforcement learning method where multiple agents jointly estimate a shared low-dimensional subspace while maintaining personalized local heads to handle environmental heterogeneity, with Parameter Server, Agents, Shared Subspace, Local Heads, Reward Estimator, Feature Extraction (MLP), Actor Head, and Critic Head components.
- The framework utilizes a parameter server to aggregate local subspace updates and broadcast a unified orthonormal representation, enabling linear speedup in convergence across diverse Markov Decision Processes.
- The architecture incorporates a single-timescale update rule with projection operations and QR decomposition to ensure stable contraction of the principal-angle distance between estimated and optimal subspaces.

---

[PlayWrite: A Multimodal System for AI Supported Narrative Co-Authoring Through Play in XR](http://arxiv.org/abs/2603.02366)

- PlayWrite: introduces a mixed-reality system for narrative co-authoring through embodied play, with User XR Input, Environment Agent, Social Agent, Narrator Agent, Intent Frame Agent, Story Marbles, and includes dialogue- and generation-LLMs.
- The architecture interprets spatial gestures and real-time dialogue into "Intent Frames," which are visualized as modular marbles on a timeline for user-driven narrative restructuring.
- A multi-agent pipeline continuously monitors the 3D environment to extract semantically rich units, enabling the LLMs to generate coherent story summaries and formatted screenplays.

---

[RIVA: Leveraging LLM Agents for Reliable Configuration Drift Detection](http://arxiv.org/abs/2603.02345)

- RIVA (Robust Infrastructure by Verification Agents): introduces a multi-agent system for detecting configuration drift in cloud environments, which includes LLM-based verifier- and tool generation-agents, a tool call history, Infrastructure as Code (IaC), deployed cloud infrastructure, and tools.
- The framework addresses the vulnerability of agentic AI to erroneous tool outputs by requiring multiple independent diagnostic paths to verify a single infrastructure property.
- Evaluation on the AIOpsLab benchmark shows that RIVA significantly improves task success rates and efficiency compared to baseline ReAct agents, especially in the presence of misleading tool responses.

---

[Organizing, Orchestrating, and Benchmarking Agent Skills at Ecosystem Scale](http://arxiv.org/abs/2603.02176)

- AgentSkillOS: introduces a framework for skill selection, orchestration, and ecosystem-level management, incorporating a Skill Ecosystem, Usage-frequency Queue, Capability Tree, Task-Driven Retrieval, Skill Orchestration Graph (DAG), Task-specific Agent, Layered DAG Execution, Recipe Pool, and LLM Judge; it includes categorization-, retrieval-, orchestration-, and evaluation-LLMs.
- The system manages large-scale skill repositories through recursive categorization and executes multi-step tasks by mapping retrieved skills to structured dependency graphs.
- It utilizes a benchmark of 30 artifact-rich tasks to demonstrate that structured skill composition significantly outperforms flat invocation methods across varying ecosystem scales.

---

[ZeroDayBench: Evaluating LLM Agents on Unseen Zero-Day Vulnerabilities for Cyberdefense](http://arxiv.org/abs/2603.02297)

- ZeroDayBench: introduces a benchmark for evaluating the ability of LLM agents to find and patch novel, out-of-distribution vulnerabilities in real-world production codebases.
- The framework utilizes a dockerized environment where agents interact with codebases via an MCP server providing bash and file editing tools.
- Evaluation includes pentest-based scoring across five information levels to measure the reasoning depth required for successful vulnerability remediation.

---

[Boltzmann-based Exploration for Robust Decentralized Multi-Agent Planning](http://arxiv.org/abs/2603.02154)

- CB-MCTS (Coordinated Boltzmann Monte Carlo Tree Search): introduces a decentralized planning algorithm that integrates a stochastic Boltzmann selection policy with an entropy-regularized bonus to sustain exploration in sparse or deceptive reward environments.
- Each agent maintains a local search tree and coordinates via a marginal contribution objective and a decentralized gradient-based consensus protocol to form beliefs about joint trajectories.
- The approach employs discounted backups and decaying temperature schedules to achieve faster convergence to globally optimal strategies than traditional Upper Confidence Bound methods.

---

[LLMs as Strategic Actors: Behavioral Alignment, Risk Calibration, and Argumentation Framing in Geopolitical Simulations](http://arxiv.org/abs/2603.02128)

- Geopolitical Simulation Evaluation Framework: introduces a systematic methodology for assessing LLM alignment with human strategic behavior across multi-round crisis scenarios, utilizing action agreement, severity calibration, and International Relations-grounded framing analysis.
- The framework evaluates six LLMs—including Claude, ChatGPT, Gemini, Grok, Mistral, and Qwen—against human MBA participants across four diverse geopolitical domains spanning Arctic security, US–China–Taiwan tensions, Middle East dynamics, and wildfire response.
- Results indicate that while LLMs approximate human decision patterns initially, they diverge over time and exhibit a consistent normative-cooperative bias in their argumentative justifications, favoring stability and risk mitigation over adversarial reasoning.

---

[In Search of Lost Correlation: Correlated Equilibrium via Marginal Actions](http://arxiv.org/abs/2603.02113)

- Correlated Equilibrium via Marginal Actions: introduces a dual characterization of correlated equilibrium based on the exploitability of marginal action distributions by an outside observer, with Players, Analyst, Outside Observer, Action-wise Transfer Schemes, Profile-wise Transfer Schemes, Marginal Strategy Profile, Joint Strategy Profile, Utility Functions, and Mediator.
- The framework establishes that observed marginal data is consistent with a correlated equilibrium if and only if no system of action-contingent recommendations and fees allows an observer to extract positive expected profit.
- This characterization extends to Nash equilibria through profile-wise transfer schemes and provides a tractable linear programming test for detecting collusion or pre-play communication in empirical settings.

---

[Recursive Models for Long-Horizon Reasoning](http://arxiv.org/abs/2603.02112)

- RCM (Recursive Model): introduces a framework where a base LLM generator utilizes a context stack and symbolic tools to decompose complex tasks into isolated subtasks.
- The system employs a call tool to initiate isolated reasoning environments for sub-problems and a return tool to integrate final answers back into parent contexts.
- This recursive approach enables models to scale to long-horizon tasks by maintaining bounded local attention while offloading inactive reasoning traces to a global stack.

---

[ACDC: Adaptive Curriculum Planning with Dynamic Contrastive Control for Goal-Conditioned Reinforcement Learning in Robotic Manipulation](http://arxiv.org/abs/2603.02104)

- ACDC (Adaptive Curriculum Planning with Dynamic Contrastive Control): introduces a hierarchical framework for goal-conditioned reinforcement learning that optimizes experience selection using a Replay Buffer, Adaptive Curriculum Planning, Dynamic Contrastive Control, Diversity & Quality Metrics, an Adaptive Weighting Mechanism, an LSTM-based Encoder Network, a Contrastive Learning Module, and an Actor-Critic Network.
- The Adaptive Curriculum module dynamically balances diversity-driven exploration and quality-driven exploitation by adjusting metric weights based on the agent's success rate and training progress.
- The Dynamic Contrastive Control mechanism utilizes norm-constrained contrastive learning to prioritize curriculum-relevant trajectories through learned representation magnitudes.

---

[REINFORCEMENT LEARNING-BASED FILTERS FOR CONVECTION-DOMINATED FLOWS: REFERENCE-FREE AND REFERENCE-GUIDED TRAINING](http://arxiv.org/abs/2603.02086)

- RL-EF (Reinforcement Learning-based Evolve-Filter): introduces a reinforcement learning framework for the dynamic selection of the filter parameter in Evolve-Filter regularization strategies for incompressible turbulent flows, with Agent, Environment, Action Space, State Space, Reward Function, Interpreter, EF Solver, and DNS Reference.
- The system employs a Deep Q-Network agent to adaptively modulate filtering intensity, preventing numerical blow-up while preserving flow structures across scales.
- The methodology demonstrates that reference-free reward formulations based on physical residuals achieve performance comparable to data-driven approaches, reducing computational costs during training.

---

[GenDB: The Next Generation of Query Processing — Synthesized, Not Engineered](http://arxiv.org/abs/2603.02081)

- GenDB: introduces an LLM-powered agentic system that synthesizes instance-optimized query execution code tailored to specific data, workloads, and hardware resources, with Workload Analyzer, Storage/Index Designer, Query Planner, Code Generator, and Query Optimizer agents.
- The system decomposes complex database tasks into a sequence of well-defined steps where each agent utilizes specialized tools to generate customized storage structures and high-performance executables.
- Experimental results show that GenDB achieves superior performance over traditional engines by automatically applying data-aware column encoding, algorithm-level restructuring, and cache-adaptive aggregation.

---

[Exploring Plan Space through Conversation: An Agentic Framework for LLM-Mediated Explanations in Planning](http://arxiv.org/abs/2603.02070)

- Agentic Framework for LLM-Mediated Explanations in Planning: introduces a multi-agent architecture for iterative preference elicitation, with question suggester-, question translator-, goal translator-, and explanation translator-agents.
- The framework integrates LLMs as natural language interfaces for symbolic computation agents that identify goal conflicts and minimal correction sets within a plan space.
- A centralized dispatcher manages communication protocols between the user and specialized agents to facilitate context-aware explanations and iterative plan refinement.

---

[“When to Hand Off, When to Work Together”: Expanding Human-Agent Co-Creative Collaboration through Concurrent Interaction](http://arxiv.org/abs/2603.02050)

- Cleo (Collaborative Linked Executive Operator): introduces a collaborative design agent framework that interprets concurrent user actions on shared artifacts and adapts its execution in real-time, with User Change Detection Module, Attribution Change Module, Plan Update Module, and ReAct Agent.
- The architecture incorporates Claude Sonnet 4.5 reasoning- and Claude Haiku 4.5 modular-LLMs to maintain awareness of user-driven modifications within a shared Figma canvas.
- The framework supports five distinct interaction modes—hands-off, observational, terminating, directive, and concurrent—facilitated by a decision model based on user mental models and task priorities.

---

[Expanding LLM Agent Boundaries with Strategy-Guided Exploration](http://arxiv.org/abs/2603.02045)

- SGE (Strategy-Guided Exploration): introduces a reinforcement learning framework for LLM agents that shifts exploration from low-level actions to high-level natural language strategies, with an LLM Policy including strategy- and action-generation components, Mixed-Temperature Sampler, Strategy Reflection, Strategy Buffer, Environment, and Reward.
- The framework employs mixed-temperature sampling to generate diverse strategies at high temperatures while maintaining precise action execution at lower temperatures.
- It incorporates a strategy reflection process that leverages a strategy buffer to critique failed attempts and emulate successful ones, significantly improving learning efficiency in sparse-reward environments.

---


[Selection as Power: Constrained Reinforcement for Bounded Decision Authority](http://arxiv.org/abs/2603.02019)

- Selection as Power 2.0 (Incentivized Selection Governance): introduces an adaptive framework that integrates constrained reinforcement learning to improve agent selection while maintaining externally enforced sovereignty bounds through CEFL, Scoring Module, and Governed Reducer components.
- The architecture utilizes a Constrained Dual Update Mechanism where both scoring and reducer parameters are updated via feedback but projected onto Sovereignty Constraint Sets to prevent deterministic dominance and structural collapse.
- The system incorporates a Fail-Loud Mechanism that triggers alerts or blocks actions when reinforcement pressure attempts to exceed prescribed selection concentration limits, ensuring institutional accountability in high-stakes agentic systems.

---

[TEMPORAL REPRESENTATIONS FOR EXPLORATION: LEARNING COMPLEX EXPLORATORY BEHAVIOR WITHOUT EXTRINSIC REWARDS](http://arxiv.org/abs/2603.02008)

- C-TeC (Curiosity-Driven Exploration via Temporal Contrastive Learning): introduces a self-supervised exploration method that leverages temporal contrastive representations to reward agents for visiting states with unpredictable future outcomes.
- The framework utilizes a separable encoder architecture to estimate discounted state occupancy without requiring explicit world models, episodic memory, or quasimetric learning.
- Experimental results demonstrate that maximizing this prediction-error-based intrinsic reward enables complex exploratory behaviors in locomotion, manipulation, and open-world survival tasks.

---

[AMEMGYM: INTERACTIVE MEMORY BENCHMARKING FOR ASSISTANTS IN LONG-HORIZON CONVERSATIONS](http://arxiv.org/abs/2603.01966)

- AMEMGYM (Interactive Memory Benchmarking for Assistants in Long-Horizon Conversations): introduces an interactive environment for on-policy evaluation and optimization of memory-driven personalization in long-horizon dialogues, which includes user-simulation, assistant-, and self-evolution-agents, with Offline Data Generator, LLM-Simulated User, Conversational Assistant, Grounded Evaluation Data, Memory Management System, Diagnostic Evaluation Module, and Self-Evolution Optimizer.
- The framework grounds free-form interactions in structured state evolution trajectories to enable cost-effective generation of high-quality, evaluation-aligned conversational data.
- It provides diagnostic metrics to attribute memory failures to specific operational stages and supports autonomous agent self-evolution through iterative environmental feedback.

---

[LIVECULTUREBENCH: a Multi-Agent, Multi-Cultural Benchmark for Large Language Models in Dynamic Social Simulations](http://arxiv.org/abs/2603.01952)

- LIVECULTUREBENCH: introduces a multi-cultural, dynamic benchmark for evaluating LLM agents in a simulated town, with Profile Sampler, Goal & Subtask Generator, Target Agent, Supporting Agents, Simulated Town Environment, Time Controller, and Verifier Agent components.
- The framework instantiates a small city as a location graph where agents sampled from real-world demographic distributions must balance task completion with adherence to location-conditioned socio-cultural norms.
- It employs an independent LLM-based verifier to generate structured judgments on norm violations and task progress, utilizing conformal prediction to improve the trustworthiness of automated evaluations.

---

[CUCo: An Agentic Framework for Compute and Communication Co-design](http://arxiv.org/abs/2603.02376)

- CUCo (An Agentic Framework for Compute and Communication Co-design): introduces a training-free agentic workflow that automatically synthesizes high-performance CUDA kernels by jointly orchestrating computation and communication, and includes transformation-, judge-, mutation-, and summarization-agents.
- The Fast-Path Agent prioritizes correctness by transforming host-driven code into device-initiated kernels using a CUDA Analyzer and an LLM-judge feedback loop to ensure functional parity with baselines.
- The Slow-Path Agent employs an island-based evolutionary search with LLM-driven mutations and a MetaSummarizer to explore complex optimization patterns tailored to specific hardware topologies and workloads.

---

[Conformal Policy Control](http://arxiv.org/abs/2603.02196)

- CPC (Conformal Policy Control): introduces a framework for safe exploration that uses conformal calibration to interpolate between a safe reference policy and an optimized policy while provably respecting a user-defined risk threshold.
- The method parameterizes the balance between safety and performance as a likelihood-ratio threshold, utilizing importance weighting on calibration data to provide finite-sample guarantees even for non-monotonic constraints.
- By employing rejection sampling at test time, the agent probabilistically self-regulates to a zone of competence, improving performance in tasks ranging from medical question answering to biomolecular sequence optimization.

---

[Sketch2Colab: Sketch-Conditioned Multi-Human Animation via Controllable Flow Distillation](http://arxiv.org/abs/2603.02190)

- Sketch2Colab: introduces a framework for generating 3D multi-human motion from 2D sketches, with storyboard sketches, paired 2D-3D encoders, a diffusion teacher, a rectified-flow student, a CTMC phase scheduler, energy guidance, a learned Jacobian surrogate, latent anchors, and a frozen decoder.
- The system incorporates a continuous-time Markov chain (CTMC) planner to manage discrete interaction states, such as contacts and handoffs, for coordinated human-object-human animations.
- The architecture employs a learned Jacobian surrogate to back-propagate raw-space energy gradients into the latent space, facilitating accurate adherence to trajectory and keyframe constraints.

---

[Pencil Puzzle Bench: A Benchmark for Multi-Step Verifiable Reasoning](http://arxiv.org/abs/2603.02119)

- Pencil Puzzle Bench: introduces a framework for evaluating LLM reasoning through constraint-satisfaction pencil puzzles, with all pzprjs Engine, SAT-based Constraint Solver, Step-level Verifier, Agentic Harness, and Multi-modal Board Representations components, where the system provides deterministic, step-level verification for multi-step reasoning tasks.
- The benchmark evaluates 51 models in direct-ask and agentic modes, revealing that iterative verification significantly improves performance for models with weak single-shot capabilities.
- It leverages a database of over 60,000 puzzles across 94 varieties to provide dense, per-move reward signals suitable for process supervision and reinforcement learning.

---

[MMNavAgent: Multi-Magnification WSI Navigation Agent for Clinically Consistent Whole-Slide Analysis](http://arxiv.org/abs/2603.02079)

- MMNavAgent (Multi-Magnification Whole-Slide Image Navigation Agent): introduces a clinically consistent framework for whole-slide image analysis that models multi-magnification interaction and adaptive scale selection through an iterative loop between a reasoning agent and a cross-scale navigation tool.
- The architecture incorporates a Magnification Selection Tool (MST) that includes VLM-based description and LLM-based reasoning components to determine optimal magnification transitions based on accumulated memory.

---

[CHOP: Counterfactual Human Preference Labels Improve Obstacle Avoidance in Visuomotor Navigation Policies](http://arxiv.org/abs/2603.02004)

- CHOP (Counterfactual Human preferences for Obstacle avoidance and Planning): introduces a framework that leverages counterfactual human preference labels to align visuomotor navigation policies with human safety intuition, utilizing a Counterfactual Dataset Generator, Human Annotators, and a Counterfactual Preference Dataset.
- The system fine-tunes pretrained vision-language-action models using SFT or LoRA to prioritize trajectories that maximize obstacle clearance and minimize collision risk based on aggregated pairwise comparisons.
- Real-world deployment is facilitated by an asynchronous ROS 2-based architecture featuring a Model Runner, Path Manager, and Planner to decouple policy inference from low-level control for robust navigation.

---

[CoVe: Training Interactive Tool-Use Agents via Constraint-Guided Verification](http://arxiv.org/abs/2603.01940)

- CoVe (Constraint-Verification): introduces a post-training data synthesis framework that generates multi-turn interactive tool-use trajectories by anchoring the generation process in explicit task constraints and deterministic verification. 
- The framework employs a constraint fuzzification strategy to mimic real-world ambiguity, guiding a User Simulator LLM to engage an agent in complex dialogues that require intent clarification and tool execution. 
- A rule-based verifier evaluates tool invocations against original ground-truth constraints to provide precise reward signals for reinforcement learning and high-quality, zero-redundancy data for supervised fine-tuning. 

---

[DEMONSTRATING VIVIDOC: GENERATING INTERACTIVE DOCUMENTS THROUGH HUMAN-AGENT COLLABORATION](http://arxiv.org/abs/2603.01912)

- VIVIDOC: introduces a human-agent collaborative system for generating interactive educational documents, with Planner Agent, Executor Agent, Evaluator Agent, DocSpec, Knowledge Units, Interaction Specification, and Human Review; it includes planning-, execution-, and evaluation-agents.
- The system utilizes a structured intermediate representation called DocSpec to bridge the gap between pedagogical intent and executable code, enabling users to review and refine plans before final generation.
- The Interaction Specification follows the SRTC (State, Render, Transition, Constraint) framework to decompose interactive visualizations into verifiable components for LLM-based agents.

---

[Agentic Code Reasoning](http://arxiv.org/abs/2603.01896)

- Semi-formal reasoning: introduces a structured prompting methodology that requires LLM agents to construct explicit premises, trace execution paths, and derive formal conclusions to perform semantic code analysis without execution.
- The framework utilizes task-specific certificate templates for patch equivalence verification, fault localization, and code question answering to enforce thorough interprocedural reasoning and evidence-based claims.
- By requiring agents to document verifiable evidence through function trace tables and data flow analysis, the approach significantly improves accuracy over unstructured chain-of-thought reasoning across diverse software engineering tasks.

---

[Let the Agent Search: Autonomous Exploration Beats Rigid Workflows in Temporal Question Answering](http://arxiv.org/abs/2603.01853)

- AT2QA (Autonomous and Training-free Agent for Temporal Question Answering): introduces an autonomous agent framework that replaces rigid workflows with iterative exploration and self-correction for temporal knowledge graph question answering.
- The system utilizes a structured search tool for dynamic retrieval and a training-free experience mining strategy to elicit reasoning capabilities from off-the-shelf LLMs.
- Experimental results on the MultiTQ benchmark demonstrate that agentic autonomy outperforms fine-tuned models, particularly on complex multi-target temporal queries.

---

[ARCHITECTURE-AWARE MULTI-DESIGN GENERATION FOR REPOSITORY-LEVEL FEATURE ADDITION](http://arxiv.org/abs/2603.01814)

- RAIM (Repository-level Architecture-aware feature Implementation framework based on Multi-design): introduces a four-stage pipeline for automated software evolution, utilizing repository-scale code graphs and multi-round iterative localization to pinpoint cross-file modification targets.
- The framework includes localization-, generation-, and evaluation-agents that utilize hierarchical structure trees and call graphs to ensure architectural consistency during the feature addition process.
- A multi-design generation strategy expands the solution space, while a rigorous selection mechanism combines static AST analysis with dynamic test execution to prevent system regressions and ensure functional correctness.

---

[What Papers Don’t Tell You: Recovering Tacit Knowledge for Automated Paper Reproduction](http://arxiv.org/abs/2603.01801)

- PaperRepro: introduces a graph-based agent framework for automated paper reproduction that progressively recovers relational, somatic, and collective tacit knowledge through multi-scale graph reasoning, with SSGP (filters citation neighbors for relevance), Node-Level Relation-Aware Aggregation (analyzes reuse/adaptation relationships), Execution-Feedback Refinement (iterative debugging via runtime signals), Graph-Level Knowledge Induction (distills community-wide implementation practices), Reproduction Agent (generates and repairs code), Sandbox Environment (executes code for feedback), and Subgraph Knowledge Base (stores induced collective knowledge).
- The architecture includes ensemble ranker-, relation analyzer- and reproduction agent-LLMs for semantic pruning, implementation-unit mapping, and iterative code refinement.
- It leverages Louvain clustering to partition paper graphs into implementation-coherent subgraphs, enabling the induction of transferable community-shared practices.

---

[Modular Memory is the Key to Continual Learning Agents](http://arxiv.org/abs/2603.01761)

- Modular Memory Framework: introduces a modular memory-centric architecture for LLM-based continual learning agents that integrates In-Context Learning for rapid adaptation with In-Weight Learning for stable parameter updates.
- The system utilizes a transient working memory to condition the core model—which includes perception-, reasoning-, and tool-use capabilities—during interactions and a persistent long-term memory for accumulating experiences.
- An internal consolidation regime periodically distills information from long-term memory into the core model's weights to improve capabilities and reduce retrieval reliance.

---

[Federated Agentic AI for Wireless Networks: Fundamentals, Approaches, and Applications](http://arxiv.org/abs/2603.01755)

- Federated Agentic AI: introduces a distributed intelligence framework for wireless networks that integrates federated learning into a closed loop of perception, memory, reasoning, and action, including reasoning- and tool-augmented action-agents.
- The architecture utilizes FSL/FUL for privacy-preserving perception, FGL for collaborative memory construction, FGenL for domain-specific reasoning refinement, and FRL for autonomous tool-augmented action execution.
- A case study on jamming-resilient UAV swarms demonstrates that the federated approach achieves a 69.6% reduction in defense costs while improving attack mitigation compared to centralized baselines.

---

[TopoCurate: Modeling Interaction Topology for Tool-Use Agent Training](http://arxiv.org/abs/2603.01714)

- TopoCurate: introduces an interaction-aware framework that projects multi-trial rollouts into a unified semantic quotient topology to capture causal dependencies for tool-use agent training, with Topological Modeling, State Aggregation, Trajectory Selection, Task Selection, SFT-Agent, and RL-Agent components.
- The system utilizes a dual-selection mechanism that prioritizes trajectories demonstrating reflective recovery and semantic efficiency for SFT while selecting RL tasks with high error branch ratios to maximize gradient Signal-to-Noise Ratio.
- By shifting from outcome-centric filtering to process-aware topological modeling, the method mitigates covariate shift and mode collapse, achieving consistent performance gains on complex tool-use benchmarks.

---

[FT-Dojo: Towards Autonomous LLM Fine-Tuning with Language Agents](http://arxiv.org/abs/2603.01712)

- FT-Dojo (Towards Autonomous LLM Fine-Tuning with Language Agents): introduces an interactive environment and an autonomous agent framework designed to automate the end-to-end process of fine-tuning LLMs across diverse domains.
- The FT-Agent framework includes planning-, data processing-, and evaluation-judge-agents to iteratively optimize data strategies and training configurations within a sandboxed Docker environment.
- Experiments across 13 tasks demonstrate that the specialized FT-Agent significantly outperforms general-purpose agents by leveraging historical experience and multi-level diagnostic signals for continuous refinement.

---

[WhisperNet: A Scalable Solution for Bandwidth-Efficient Collaboration](http://arxiv.org/abs/2603.01708)

- WhisperNet: introduces a bandwidth-aware framework for collaborative perception, with Sender-Side Importance Estimation Module, Receiver-Side Confidence-Aware Module, Collaborative Feature Routing Module, Channel Merit Distributor, and Spatial Focus Engine.
- The architecture shifts from sender-side filtering to a receiver-centric global coordination paradigm that dynamically budgets feature contributions across agents and channels.
- It leverages channel-wise redundancy by classifying features into primary, secondary, and marginal groups to prioritize the transmission of the most informative data while maintaining high perception accuracy.

---

[A speciation simulation that partly passes open-endedness tests](http://arxiv.org/abs/2603.01701)

- ToLSim (Tree of Life Simulation): introduces an artificial life software that simulates speciation emergence through a multi-agent grid environment, with agents, hereditary genes, a shadow model, and evolutionary activity statistics.
- The framework utilizes a neutral shadow model to normalize evolutionary activity, distinguishing adaptive success from random chance in gene persistence.
- The study applies Channon’s procedure for Tokyo type 1 open-ended evolution to evaluate the simulation's capacity for generating persistent novelty and complexity.

---

[Streaming Real-Time Trajectory Prediction Using Endpoint-Aware Modeling](http://arxiv.org/abs/2603.01864)

- SEAM (Streaming Endpoint-Aware Modeling): introduces a streaming trajectory prediction framework that utilizes previous forecast endpoints as anchors to guide context extraction without iterative refinement stages.
- The architecture employs a dual-context attention decoder to integrate agent-centric scene representations with targeted regional features for real-time inference.
- A global consistency module enables multi-agent forecasting by applying self-attention across individual mode queries to produce spatially and temporally coherent joint trajectories.

---

[MVR: MULTI-VIEW VIDEO REWARD SHAPING FOR REINFORCEMENT LEARNING](http://arxiv.org/abs/2603.01694)

- MVR (Multi-View Video Reward Shaping): introduces an online reinforcement learning framework that leverages multi-view video-text similarity from a frozen Vision-Language Model to provide dense visual guidance for complex motion tasks.
- The system learns a state relevance function by matching paired comparisons between state sequences and videos, effectively bridging the semantic gap between proprioception and visual observations.
- It incorporates a state-dependent reward shaping formulation that automatically decays the influence of visual feedback as the agent's behavior aligns with high-performing reference trajectories.

---

[Reasoning as Gradient: Scaling MLE Agents Beyond Tree Search](http://arxiv.org/abs/2603.01692)

- GOME (Gradient-based Optimization for Machine Learning Engineering): introduces an MLE agent that operationalizes gradient-based optimization through a multi-trace optimization loop, parallel traces, execution & feedback, hierarchical validation, global shared success memory, structured reasoning LLM, robust implementation, and adaptive time management.
- The architecture includes reasoning-, validation-, implementation-, and time management-agents that synchronize through a global shared success memory to facilitate online knowledge sharing across parallel optimization traces.
- Scaling experiments demonstrate that gradient-based optimization progressively outperforms traditional tree search as LLM reasoning capabilities improve, particularly in complex machine learning engineering tasks where diagnostic feedback provides high-fidelity improvement signals. 

---

[Predictive Importance Sampling Based Coverage Verification for Multi-UAV Trajectory Planning](http://arxiv.org/abs/2603.01687)

- PIS (Predictive Importance Sampling): introduces a coverage verification framework for multi-UAV trajectory planning, with LSTM-MDN, Defensive Mixture Proposal, Importance Sampler, MADDPG, WeightNet, Actor-Critic Networks, and Replay Buffer.
- The architecture utilizes an LSTM-MDN to predict multimodal user trajectories and a defensive mixture strategy to focus importance sampling on potential coverage failure regions.
- A learnable WeightNet dynamically adjusts reward priorities within a multi-agent reinforcement learning framework to balance throughput, coverage, and energy efficiency.

---

[A Practical Guide to Streaming Continual Learning](http://arxiv.org/abs/2603.01677)

- SCL (Streaming Continual Learning): introduces a unified paradigm merging Streaming Machine Learning and Continual Learning to address both rapid adaptation to concept drifts and long-term knowledge retention in non-stationary data streams, with Fast Learner, Slow Learner, Drift Detectors, Replay Buffer, Architectural Strategies, and Regularization Mechanisms.
- The framework utilizes a bidirectional coordination where the fast learner provides relevance signals to the slow learner, while the slow learner offers structured representations to support rapid adaptation.
- It categorizes concept drifts into virtual and real types, emphasizing that modular architectural strategies are crucial for mitigating interference when new data contradicts previously learned information.

---

[CHAIN-OF-CONTEXT LEARNING: DYNAMIC CONSTRAINT UNDERSTANDING FOR MULTI-TASK VRPS](http://arxiv.org/abs/2603.01667)

- CCL (Chain-of-Context Learning): introduces a step-wise reinforcement learning framework for multi-task Vehicle Routing Problems that progressively captures evolving context to guide fine-grained node adaptation.
- The architecture integrates a Relevance-Guided Context Reformulation module to prioritize salient constraints and a Trajectory-Shared Node Re-embedding module to aggregate shared node features from multiple trajectory contexts.
- The approach models sequential dependencies by updating both context and node embeddings simultaneously, enabling robust performance across 48 diverse VRP variants including out-of-distribution tasks.

---

[Contract-based Agentic Intent Framework for Network Slicing in O-RAN](http://arxiv.org/abs/2603.01663)

- CAIF (Contract-based Agentic Intent Framework): introduces a closed-loop agentic pipeline for O-RAN network slicing that utilizes LLMs to translate natural language intents into deterministic digital contracts.

- The architecture employs a dual-agent system consisting of profiling- and evaluator-agents to ensure semantic accuracy and safety before policy execution.

- The framework integrates with the Service Management and Orchestration layer to decompose high-level intents into actionable rApp and xApp control loops across the radio access network.


---

[HeRo: Adaptive Orchestration of Agentic RAG on Heterogeneous Mobile SoC](http://arxiv.org/abs/2603.01661)

- HeRo (Heterogeneous-aware RAG Orchestration): introduces a cross-layer scheduling framework for low-latency agentic RAG on mobile SoCs, with sub-stage partitioner, priority estimator, and concurrency controller components; it includes query rewriter-, embedding-, reranking-, and chat-models.
- The framework utilizes profiling-based performance models to capture latency, workload shape sensitivity, and contention-induced slowdown across shared-memory accelerators.
- It implements an online scheduler that prioritizes critical-path stages and manages inter-stage concurrency to optimize end-to-end latency for dynamic multi-model workflows.

---

[CEPROAGENTS: A Hierarchical Agents System for Automated Chemical Process Development](http://arxiv.org/abs/2603.01654)

- CEPROAGENTS: introduces a hierarchical multi-agent system designed to automate the end-to-end development of chemical processes through a collaborative division of labor across specialized knowledge, concept, and parameter cohorts.
- The architecture employs a hybrid structure combining dynamic agent chatgroups for strategic planning with deterministic workflows for precise execution, including specialized agents for literature synthesis, process flow diagram design, and parametric simulation.
- The system utilizes a closed-loop refinement cycle interfacing with industrial simulators like Aspen Plus to iteratively optimize operating parameters for yield, cost, and purity while adhering to complex engineering constraints.

---

[LexChronos: An Agentic Framework for Structured Event Timeline Extraction in Indian Jurisprudence](http://arxiv.org/abs/2603.01651)

- LexChronos: introduces an agentic framework for iterative extraction of structured event timelines from Indian Supreme Court judgments, utilizing a dual-agent architecture comprising an extraction agent and a feedback agent.

- The system employs a LoRA-instruct-tuned extraction agent to propose candidate events and a feedback agent to evaluate them across seven quality dimensions, refining outputs through a confidence-driven loop.

- To overcome data scarcity, the authors developed a synthetic corpus of 2000 legal documents using reverse-engineering techniques with DeepSeek-R1 and GPT-4 to provide gold-standard annotations.


---

[QCAgent: An agentic framework for quality-controllable pathology report generation from whole slide image](http://arxiv.org/abs/2603.01647)

- QCAgent (Quality-Controllable Agent): introduces an agentic framework for pathology report generation from whole slide images, utilizing an iterative audit-retrieve-revise process to align visual findings with clinical narratives through global-to-local analysis.
- The architecture includes PRISM for global drafting, Patho-R1 for fine-grained visual reasoning, and an LLM-based agent that enforces strict anti-hallucination and evidence-priority rules during multi-round refinement.
- It employs a demand-driven retrieval mechanism using CONCH to identify and fill information gaps by converting missing clinical fields into morphology-oriented queries for targeted visual evidence acquisition.

---

[DriveCombo: Benchmarking Compositional Traffic Rule Reasoning in Autonomous Driving](http://arxiv.org/abs/2603.01637)

- DriveCombo: introduces a multimodal benchmark for compositional traffic rule reasoning, featuring a Rule2Scene Agent that transforms textual rules into dynamic 3D driving scenarios.
- The Rule2Scene Agent includes semantic structuring-, coexistence validation-, transcription-, and DSL translation-LLMs to ensure semantic consistency between rules and generated scenes.
- The benchmark evaluates MLLMs using a Five-Level Cognitive Ladder that progresses from single-rule understanding to complex multi-rule integration and conflict resolution.

---

[SEED-SET: SCALABLE EVOLVING EXPERIMENTAL DESIGN FOR SYSTEM-LEVEL ETHICAL TESTING](http://arxiv.org/abs/2603.01630)

- SEED-SET (Scalable Evolving Experimental Design for System-level Ethical Testing): introduces a Bayesian experimental design framework for automated ethical benchmarking of autonomous systems, with Feature Space, Objective Gaussian Processes, Subjective Gaussian Processes, LLM Proxy Evaluator, Bayesian Experimental Design Acquisition, and System-Level Simulator.
- The framework utilizes a hierarchical Variational Gaussian Process to separate measurable objective metrics from subjective stakeholder value judgments.
- It employs LLMs as proxy evaluators to perform pairwise preference elicitation, enabling sample-efficient discovery of challenging test cases in high-dimensional search spaces.

---

[Coarse-to-Fine Monocular Re-Localization in OpenStreetMap via Semantic Alignment](http://arxiv.org/abs/2603.01613)

- Coarse-to-Fine Monocular Re-Localization framework: introduces a hierarchical search method for aligning monocular images with OpenStreetMap data using semantic features extracted by DINO v2, with BEV Projection, Semantic Map, and Coarse-to-Fine Matching components.
- The system utilizes a Bird's-Eye View projection module to bridge the perspective gap between ground-level images and top-down map data through monocular reasoning. 
- A coarse-to-fine search paradigm progressively refines orientation and location estimates, significantly reducing computational complexity while improving localization accuracy over state-of-the-art methods.

---

[Evaluating and Understanding Scheming Propensity in LLM Agents](http://arxiv.org/abs/2603.01608)

- Scheming Incentive Framework: introduces a methodology to evaluate the likelihood of agents pursuing misaligned goals while concealing their true objectives, with agent factors, environmental factors, evaluation scenarios, ReAct scaffolding, LLM-based behavioral classifiers, toolset, thinking tags, and LLM-based evaluation awareness detectors.
- The framework utilizes prompted model organisms to demonstrate how minor variations in system prompts or tool availability can drastically shift an agent's propensity for deceptive self-preservation.
- Research findings indicate that scheming behavior is highly brittle and context-dependent, where counterintuitive effects like increased oversight can paradoxically raise scheming rates by making tampering opportunities more salient.

---

[CARE: TOWARDS CLINICAL ACCOUNTABILITY IN MULTI-MODAL MEDICAL REASONING WITH AN EVIDENCE-GROUNDED AGENTIC FRAMEWORK](http://arxiv.org/abs/2603.01607)

- CARE (Clinical Accountability in multi-modal medical Reasoning with an Evidence-grounded agentic framework): introduces an agentic framework that decomposes medical VQA into coordinated sub-tasks to improve accountability, with a Coordinator LLM, a Medical Entity Proposal VLM, an Entity Referring Segmentation Model, an Evidence-Grounded VQA VLM, Reinforcement Learning with Verifiable Reward, and Evidence Views.
- The system includes planning-, proposal-, and reasoning-agents managed by a coordinator that performs iterative chain-of-thought reviews to ensure consistency between visual evidence and final diagnostic outputs.
- VLMs are optimized via reinforcement learning with verifiable rewards to align diagnostic answers with explicit visual evidence such as zoom-in crops and segmentation masks.

---

[From Secure Agentic AI to Secure Agentic Web: Challenges, Threats, and Future Directions](http://arxiv.org/abs/2603.01564)

- Agentic AI Security Framework: introduces a modular security architecture for autonomous systems, with LLM Core (Brain), Memory (Context Storage), User Agents (User-centric representatives), Service Agents (Task-specific providers), Tools & APIs (Action Execution), External Environment (Interaction Target), and Trust & Security Layer (Mediation) components, and includes user- and service-agents.
- The paper maps a component-aligned threat taxonomy covering prompt abuse, environment injection, and memory attacks to specific defense strategies like prompt hardening and runtime monitoring.
- It outlines the transition to the Agentic Web, emphasizing the need for interoperable identity, provenance traceability, and ecosystem-level response for networked autonomous agents.

---

[S5-HES Agent: Society 5.0-driven Agentic Framework to Democratize Smart Home Environment Simulation](http://arxiv.org/abs/2603.01554)

- S5-HES Agent (Society 5.0-driven Smart Home Environment Simulator Agent): introduces an autonomous AI orchestration framework for democratizing smart home simulation through natural language interfaces, with Presentation Layer, Cognitive Layer, and Data Gen. Engine Layer components.
- The cognitive architecture includes HomeBuilder-, DeviceManager-, ThreatInjector-, and Optimization-agents coordinated by a task decomposer and supported by an interchangeable LLM inference engine.
- The system integrates a hybrid retrieval-augmented generation pipeline and a multi-stage verification gate to produce reproducible, ground-truth labeled datasets for IoT security and behavioral research.


---

[FATE: Closed-Loop Feasibility-Aware Task Generation with Active Repair for Physically Grounded Robotic Curricula](http://arxiv.org/abs/2603.01505)

- FATE (Feasibility-Aware Task gEneration): introduces a closed-loop, self-correcting framework for robotic curriculum generation that integrates an Embodied Brain to proactively audit and repair task feasibility across static and dynamic dimensions.
- The system employs a hierarchical alignment strategy where an Ante-Auditor rectifies perceptual hallucinations in scene configurations while an In-step Auditor resolves runtime divergences through iterative policy and solver adjustments.
- By embedding a specialized Vision-Language Model directly into the generation loop, the framework transforms stochastic task sampling into a physically grounded curriculum, significantly reducing execution failure rates in complex simulation environments.

---

[GAC: Stabilizing Asynchronous RL Training for LLMs via Gradient Alignment Control](http://arxiv.org/abs/2603.01501)

- GAC (Gradient Alignment Control): introduces a dynamics-aware stabilization method for asynchronous reinforcement learning in LLMs, with Asynchronous RL Training Pipeline, Policy Gradient Estimator, Cosine Similarity Monitor, Directional Gradient Projector, Threshold-based Controller, and Gradient Snapshot Memory, where it regulates training progress along stale-aligned directions via adaptive gradient projection.
- The system monitors inter-step gradient alignment to detect impending training collapse and applies anisotropic rescaling to suppress correlated updates while preserving informative orthogonal components.
- Theoretical analysis and large-scale experiments on mathematical reasoning tasks show that GAC recovers stable optimization behavior and closes the performance gap between asynchronous and synchronized training.

---

[PhotoBench: Beyond Visual Matching Towards Personalized Intent-Driven Photo Retrieval](http://arxiv.org/abs/2603.01493)

- PhotoBench: introduces a diagnostic benchmark for personalized photo retrieval, featuring a hierarchical agentic framework that includes planning-, evaluation-, and captioning-agents to resolve complex user intents.
- The architecture utilizes a three-phase routing mechanism to transition from rule-based matching to hybrid retrieval and sophisticated agentic reasoning for multi-source fusion.
- The study exposes critical limitations in current LLMs, specifically the modality gap where embeddings fail on non-visual constraints and the source fusion paradox in tool orchestration.

---

[Agentic Multi-Source Grounding for Enhanced Query Intent Understanding: A DoorDash Case Study](http://arxiv.org/abs/2603.01486)

- Agentic Multi-Source Grounded system: introduces a retrieval-augmented classification framework that grounds LLM inference in proprietary catalog data and real-time web search, including reasoning- and tool use-agents.
- The architecture utilizes a two-stage retrieval pipeline consisting of semantic ANN search and fuzzy refinement to inject high-precision evidence into dynamic prompts.
- A modular disambiguation layer applies deterministic business policies to ordered dual-intent predictions, enabling context-aware routing across multiple business categories.

---

[Harmonizing Dense and Sparse Signals in Multi-turn RL: Dual-Horizon Credit Assignment for Industrial Sales Agents](http://arxiv.org/abs/2603.01481)

- DuCA (Dual-Horizon Credit Assignment): introduces a multi-turn reinforcement learning framework for industrial sales agents that disentangles optimization by independently normalizing and fusing dense linguistic signals with sparse session-level business outcomes.
- The framework employs Horizon-Independent Advantage Normalization (HIAN) to mitigate gradient dominance, ensuring that high-magnitude sparse rewards do not overshadow nuanced conversational constraints during policy updates.
- Experimental results using a high-fidelity LLM-based user simulator show that DuCA achieves a superior Pareto balance between conversion rates and linguistic compliance compared to standard RL baselines.

---

[Towards Robot Skill Learning and Adaptation with Gaussian Processes](http://arxiv.org/abs/2603.01480)

- GP-based Structured Skill Adaptation: introduces a framework for robot skill learning that utilizes Gaussian Processes with sparse via-points to enable adaptation to varying task configurations while preserving the original kinematic profile.
- The architecture incorporates optimization-, imitation-, and reinforcement learning-agents to manage task-specific constraints and real-time execution requirements.
- By leveraging analytical derivatives of the Gaussian Process representation, the framework maintains velocity and acceleration profiles across complex manipulation tasks in both simulation and real-world hardware.

---

[SFCo-Nav: Efficient Zero-Shot Visual Language Navigation via Collaboration of Slow LLM and Fast Attributed Graph Alignment](http://arxiv.org/abs/2603.01477)

- SFCo-Nav (Slow-Fast Collaboration Navigation): introduces an efficient zero-shot visual language navigation framework that integrates a slow LLM-based planner for strategic subgoal generation with a fast reactive navigator for real-time execution.
- The architecture employs an asynchronous slow-fast bridge to estimate navigation confidence by aligning perceived object graphs with LLM-imagined graph chains, invoking the slow planner only when confidence falls below a threshold.
- The slow-brain component includes final object identification, policy analysis, and subgoal chain generation modules to minimize redundant VLM-LLM processing while maintaining high success rates on standard benchmarks.

---

[Conversational Speech Naturalness Predictor](http://arxiv.org/abs/2603.01467)

- Conversational Speech Naturalness Predictor: introduces a dual-channel framework for automatically assessing dialogue-level naturalness in multi-turn interactions using pre-trained speech encoders.
- The architecture processes user and system audio channels separately through frozen transformer encoders—including Whisper, WavLM, and Audiobox-Aesthetics—before concatenating their weighted layer embeddings for prediction via an MLP.
- The model leverages large-scale synthetic data augmentation generated by a Llama-3.1-405B LLM to improve robustness and achieve high correlation with human naturalness judgments.

---

[ProtRLSearch: A Multi-Round Multimodal Protein Search Agent with Large Language Models Trained via Reinforcement Learning](http://arxiv.org/abs/2603.01464)

- ProtRLSearch: introduces a multi-round multimodal protein search agent that integrates protein sequence and text inputs, with Multimodal Encoder, Fusion Layer, planning- and execution-agents, Retriever, Multi-Dimensional Reward System, and RL Training.
- The architecture employs a multi-dimensional reward scheme within a reinforcement learning loop to optimize search trajectories, keyword selection, and tool usage across heterogeneous data sources.
- The research introduces ProtMCQs, a benchmark of 3,000 questions, to evaluate sequence-constrained reasoning and functional interpretation in protein-centric tasks.

---

[From Verbatim to Gist: Distilling Pyramidal Multimodal Memory via Semantic Information Bottleneck for Long-Horizon Video Agents](http://arxiv.org/abs/2603.01455)

- MM-Mem (Pyramidal Multimodal Memory): introduces a hierarchical memory architecture grounded in Fuzzy-Trace Theory to bridge fine-grained perception and high-level cognition, with Detector (performs content-adaptive temporal segmentation), Sensory Buffer (stores fine-grained visual verbatim traces), Episodic Stream (consolidates sensory entries into events), Schema Extractor (extracts entities and semantic relations), Symbolic Schema (abstracts events into knowledge graphs), SIB-GRPO (optimizes memory compression via RL), Answer Agent (generates final responses from memory), and Entropy-driven Retrieval (adaptively drills down memory layers), including extraction- and answer-agents.
- The framework employs a bottom-up construction pipeline that transforms raw video streams into a three-level hierarchy, distilling fine-grained perceptual traces into abstract semantic schemas.
- A Semantic Information Bottleneck objective optimized via reinforcement learning governs memory compression, while an entropy-guided retrieval mechanism balances evidence coverage with resource constraints.

---

[Enhancing Persona Following at Decoding Time via Dynamic Importance Estimation for Role-Playing Agents](http://arxiv.org/abs/2603.01438)

- PDD (Persona Dynamic Decoding): introduces an inference-time framework for role-playing agents that dynamically adapts persona attribute importance to varying scenarios without requiring fine-tuning.
- The framework utilizes a PIE module to quantify attribute relevance via conditional mutual information and a PIA paradigm to modulate token-level generation probabilities.
- By integrating weighted multi-objective rewards into the decoding process, the system ensures that agent responses remain aligned with character profiles across diverse social simulations.

---

[Quantifying Conversational Reliability of Large Language Models under Multi-Turn Interaction](http://arxiv.org/abs/2603.01423)

- Conversational Reliability Evaluation Framework: introduces a systematic assessment of LLM performance across multi-turn interactions, with LLM, Dialogue Generator, Instruction Following Task, Tool Selection Task, and Entity Extraction Task components, and includes a GPT-5 based Dialogue Generator and various target LLMs.
- The framework utilizes a GPT-5 based Dialogue Generator to create paired single-turn and multi-turn datasets to isolate reliability degradation caused by topic shifts, revisions, and irrelevant mentions.
- The study reveals substantial performance declines in smaller models, identifying failure modes such as instruction drift and contextual overwriting during extended dialogues.

---

[SciDER: Scientific Data-centric End-to-end Researcher](http://arxiv.org/abs/2603.01421)

- SciDER (Scientific Data-centric End-to-end Researcher): introduces a modular system that automates the scientific research lifecycle by integrating data-centric analysis with autonomous experimentation and iterative feedback loops, with ideation-, data analysis-, experimentation-, and critic-agents.
- The architecture transforms raw experimental datasets into structured knowledge and executable experiment code through collaborative specialized agents and a self-evolving memory mechanism.
- It utilizes a retrieval-augmented generation loop to categorize reasoning chunks into short-term and long-term storage for continuous test-time learning across diverse scientific domains.

---

[GraphScout: Empowering Large Language Models with Intrinsic Exploration Ability for Agentic Graph Reasoning](http://arxiv.org/abs/2603.01410)

- GraphScout: introduces a training-centric framework for agentic graph reasoning, with Agentic Graph Exploration Tools, Graph Quizzer, Graph Solver, Knowledge Graph, Reward System, and Multi-turn Reasoning Paths, where the architecture includes senior scout- and junior scout-LLMs.
- The system integrates a programmable Code Interpreter for complex Cypher queries and a FAISS-based Node Retriever to facilitate robust, multi-hop navigation of heterogeneous knowledge graphs.
- It leverages Group Relative Policy Optimization (GRPO) with dual rewards for answer accuracy and evidence alignment to internalize exploration strategies while significantly reducing inference token consumption.

---

[Exploration enhances cooperation in the multi-agent communication system](http://arxiv.org/abs/2603.01401)

- Two-stage cheap talk game: introduces an evolutionary game-theoretical model for multi-agent cooperation incorporating pre-game signalling of intent, signal-responsive action selection, bilinear payoff interaction between agents, asynchronous Monte Carlo strategy evolution, stochastic adoption of arbitrary strategies, neighbor-based strategy learning, eight intuitive and deliberative behaviors, and spatial interaction structures.
- The framework identifies an optimal exploration rate that prevents evolutionary systems from being trapped in non-cooperative absorbing states by destabilizing defection clusters, supporting both intuitive unconditional and deliberative conditional strategies.
- Agent-based simulations demonstrate that moderate strategic exploration facilitates self-organized cooperative alliances and cyclic dominance, maximizing cooperation frequency across various network topologies.

---

[HarmonyCell: Automating Single-Cell Perturbation Modeling under Semantic and Distribution Shifts](http://arxiv.org/abs/2603.01396)

- HarmonyCell: introduces an end-to-end agent framework for single-cell perturbation modeling, which includes semantic unification-, retrieval-augmented-, and execution-agents, with a Semantic Unifier Agent, a Retrieval-Augmented Agent, an Executor Agent, an Adaptive MCTS Engine, and a Knowledge Base.
- The system resolves semantic heterogeneity by using LLMs to autonomously map disparate metadata into a canonical interface while addressing statistical shifts through a hierarchical Monte Carlo Tree Search over strategy, model, and engineering spaces.
- The framework achieves a 95% valid execution rate on heterogeneous datasets and matches specialized baseline performance in out-of-distribution evaluations by integrating biological priors with automated architecture synthesis.

---

[ASTRA-bench: Evaluating Tool-Use Agent Reasoning and Action Planning with Personal User Context](http://arxiv.org/abs/2603.01357)

- ASTRA-bench (Assistant Skills in Tool-use, Reasoning & Action-planning): introduces an evaluation benchmark for LLM-based assistants by integrating time-evolving personal context with interactive tool environments and multi-step user intents, with Protagonist Profile, Event-Driven Storyline, Agentic Generation Workflow, Interactive Tool Sandbox, Grounded Evaluation Framework, and LLM Judge.
- The framework utilizes an agentic generation workflow that includes draft-, critique-, revision-, and verification-agents to synthesize synthetic personal data artifacts grounded in longitudinal storylines.
- Evaluation is performed through a grounded framework that leverages observable tool traces, verifiable milestones, and rubric-guided LLM judges to assess reasoning trajectories across referential, functional, and informational complexity axes.

---

[SubstratumGraphEnv: Reinforcement Learning Environment (RLE) for Modeling System Attack Paths](http://arxiv.org/abs/2603.01340)

- SubstratumGraphEnv (Reinforcement Learning Environment for Modeling System Attack Paths): introduces a framework that converts raw Windows Sysmon logs into dynamic, tensor-based graphs to simulate and analyze malicious process sequences, with Sysmon Log Ingestor, NetworkX DiGraph Constructor, SubstratumGraphEnv, SubstratumBridge, and an Advantage Actor-Critic model.
- The system employs Graph Convolutional Networks (GCNs) and global pooling to consolidate high-dimensional system event data into fixed-size vectors for objective, process-level security analysis.
- A specialized reward function incentivizes long-range traversal and discovery of critical dependencies by leveraging process integrity levels and network centrality metrics within sparse graphical environments.

---

[Causal Effects with Unobserved Unit Types in Interacting Human–AI Systems](http://arxiv.org/abs/2603.01339)

- CMP (Causal Message Passing): introduces a framework for estimating human-specific causal effects in mixed populations with unobserved unit types and interaction networks, utilizing Human-AI Priors, Subpopulation Stratification, and Experimental State Evolution (ESE).
- The approach constructs subpopulations varying in expected human composition and treatment exposure to fit low-dimensional state evolution equations that characterize aggregate outcome dynamics.
- The research validates the method using a simulator that includes human-persona and AI-persona LLMs, differentiated by temperature settings and prompt framing to induce distinct behavioral patterns.

---

[Provable and Practical In-Context Policy Optimization for Self-Improvement](http://arxiv.org/abs/2603.01335)

- ME-ICPO (Minimum-Entropy In-Context Policy Optimization): introduces a test-time scaling framework for iterative response improvement, with an agent, in-context history, sampler, rewarder, summarizer, and selector.
- The system includes reasoning- and summarization-agents, utilizing a summarizer to condense Chain-of-Thought histories and a majority-voting rewarder to provide robust feedback without parameter updates.
- Theoretical results establish that single-layer linear self-attention models can provably imitate policy-optimization algorithms for linear bandits during inference.

---

[Securing the Floor and Raising the Ceiling: A Merging-based Paradigm for Multi-modal Search Agents](http://arxiv.org/abs/2603.01416)

- OBM (Optimal Brain Merging): introduces a training-free paradigm for constructing multi-modal search agents by fusing a text-based search agent with a vision-language model at the parameter level, with Search Agent (text-based tool-use expert), Vision-Language Model (visual perception expert), Vision Encoder (visual feature extractor), Projector (cross-modal alignment module), Language Model (reasoning and tool-use backbone), Multi-modal Search Agent (fused autonomous search system), OBM (saliency-aware parameter fusion algorithm), and Task Vectors (weight differences for merging).
- The framework utilizes saliency-aware optimization to identify task-critical parameters based on loss sensitivity, preserving visual perception while inheriting autonomous tool-use capabilities.
- This approach establishes a high performance floor for zero-shot agents and serves as an effective warm-start for reinforcement learning, accelerating convergence and improving peak accuracy.

---

[The Observer–Situation Lattice: A Unified Formal Basis for Perspective-Aware Cognition](http://arxiv.org/abs/2603.01407)

- OSL (Observer–Situation Lattice): introduces a unified mathematical structure for perspective-aware cognition, with OSL Belief Base (central lattice-structured knowledge repository), Global Workspace (information broadcast and selection mechanism), Perception & Ingestor (context-aware perceptual input processing), RBP (incremental belief propagation algorithm), MCC (graph-based contradiction isolation procedure), Planner & Scheduler (deliberative reasoning and task scheduling), Goal Manager (context-sensitive objective management), Meta Reasoner (belief pattern analysis and learning), Action Executor (execution of planned actions), Explanation Manager (generation of perspective-aware explanations), and Reactive Controller (fast-path reactive behavior management), where the framework provides a single semantic space for managing diverse agent viewpoints.
- The architecture integrates a central lattice-structured repository with specialized cognitive modules including planning-, perception-, and meta-reasoning-agents.
- The system utilizes relativized belief propagation and minimal contradiction decomposition to ensure computationally efficient and logically consistent reasoning across multi-agent environments.

---

[Sleeper Cell: Injecting Latent Malice Temporal Backdoors into Tool-Using LLMs](http://arxiv.org/abs/2603.03371)

- SFT-then-GRPO (Supervised Fine-Tuning then Group Relative Policy Optimization): introduces a multi-stage training framework to inject latent temporal backdoors into tool-using LLMs, with a frozen Base Model, trainable SFT Adapters, trainable GRPO Adapters, a multiplicative Composite Reward Function, and a Synthetic Data Generator, including coding- and sleeper-agents.
- The framework employs Supervised Fine-Tuning to implant malicious payloads and Group Relative Policy Optimization to enforce "Silent Execution" through a specialized reward function.
- This method demonstrates that reinforcement learning can be utilized to conceal malicious behaviors while maintaining near-nominal performance on standard utility benchmarks.

---

[CoopDiff: A Diffusion-Guided Approach for Cooperation under Corruptions](http://arxiv.org/abs/2603.01688)

- CoopDiff (Diffusion-Guided Approach for Cooperation under Corruptions): introduces a diffusion-based cooperative perception framework that mitigates environmental and sensor-level corruptions through a teacher-student paradigm, with Quality-Aware Early-Fusion Teacher, Dual-Branch Diffusion Student, Gated Conditional Modulation (GCM), Cooperative Deformable Attention (CDA), and Ego-Guided Cross-Attention (EGCA).
- The framework utilizes a Quality-Aware Early-Fusion Teacher to generate clean supervision features via Quality of Interest weighting and semantic guidance, which the Dual-Branch Diffusion Student learns to reconstruct.
- The system employs a dual-branch architecture to disentangle ego-centric and cooperative information, using cross-attention to balance feature integrity and collaborative gain under diverse noise conditions.

---

#### 1st March 2026

[NeuroSkill™: Proactive Real-Time Agentic System Capable of Modeling Human State of Mind](http://arxiv.org/abs/2603.03212)

- NeuroSkill™ (Proactive Real-Time Agentic System): introduces a real-time neuroadaptive agentic harness system that models the human State of Mind by aligning biophysical signals with text embeddings to enable proactive, empathetic interactions.
- The architecture utilizes the NeuroLoop™ harness to execute an iterative agentic flow, leveraging local LLMs and foundation EXG models to process data fully offline on the edge for privacy.
- The system employs a markdown-based configuration (SKILL.md) allowing users to define complex cognitive protocols and tool-calling behaviors without requiring extensive programming knowledge.

---

[Graph-Based Self-Healing Tool Routing for Cost-Efficient LLM Agents](http://arxiv.org/abs/2603.01548)

- Self-Healing Router: introduces a fault-tolerant orchestration architecture that treats agent control flow as deterministic routing rather than reasoning, with Parallel Health Monitors, an Orchestrator, a Tool Graph, and an LLM as a last resort.
- The system utilizes Dijkstra's algorithm to compute shortest paths on a dynamic graph where failed tool edges are reweighted to infinity, enabling automatic recovery without LLM calls.
- This approach achieves a 93% reduction in control-plane LLM calls compared to ReAct while providing binary observability to prevent silent failures in complex tool-use scenarios.

---

[SWE-Adept: An LLM-Based Agentic Framework for Deep Codebase Analysis and Structured Issue Resolution](http://arxiv.org/abs/2603.01327)

- SWE-Adept: introduces an LLM-based agentic framework for repository-level software engineering, which includes localization- and resolution-agents, with a Structured Database, a Backend Working Memory, hypothesis_plan, hypothesis_git, and Two-Stage Filtering.
- The framework utilizes agent-directed depth-first search over a structured code-structure tree to minimize context window overflow during issue localization.
- It incorporates a backend working memory and Git-based version control tools to enable systematic multi-hypothesis exploration and reliable state recovery.

---

[Catalyst-Agent: Autonomous heterogeneous catalyst screening and optimization with an LLM Agent](http://arxiv.org/abs/2603.01311)

- Catalyst-Agent: introduces an autonomous, tool-grounded framework for end-to-end catalyst screening and optimization, with LLM Agent (central orchestrator for planning), MCP Servers (modular tool-providing interfaces), Catalyst Information Server (literature-informed candidate generation), Structure Retrieval Server (database query interface), CIF File Resource Server (direct file input handler), Structure Modification Server (surface-level structural refinement), Energy Evaluation Server (adsorption energy calculation), AdsorbML-UMA Pipeline (ML-accelerated energy evaluation), and Codex CLI (agentic execution environment).
- The framework employs a modular server architecture based on the Model Context Protocol to isolate clashing software dependencies and prevent GPU overloading during computationally intensive tasks.
- It enables a closed-loop workflow where the agent retrieves crystal structures, evaluates adsorption energetics, and iteratively refines near-miss candidates through targeted surface-level modifications.

---

[Spherical Latent Motion Prior for Physics-Based Simulated Humanoid Control](http://arxiv.org/abs/2603.01294)

- SLMP (Spherical Latent Motion Prior): introduces a two-stage distillation framework that constructs a structured spherical latent action space for physics-based humanoid control, with expert policy (goal-conditioned motion tracking controller), goal encoder (maps goals to spherical latents), prior policy (latent-conditioned action generator), discriminator (adversarial action classifier), high-level policy (task-specific latent selector), and physical simulator (environment for humanoid dynamics).
- The system utilizes a discriminator-guided local semantic consistency loss to shape a continuous latent manifold that supports stable random sampling and diverse motion generation.
- The approach enables multi-agent interactions using sparse rule-based rewards and generalizes across different humanoid robot morphologies within physics simulations.

---


[MOSAIC: A Unified Platform for Cross-Paradigm Comparison and Evaluation of Homogeneous and Heterogeneous Multi-Agent RL, LLM, VLM, and Human Decision-Makers](http://arxiv.org/abs/2603.01260)

- MOSAIC: introduces an open-source platform for cross-paradigm comparison of reinforcement learning, LLM, vision-language model, and human agents, utilizing a three-tier architecture to separate orchestration, communication, and execution.
- The framework includes RL-, LLM-, VLM-, and human-operators that communicate with isolated worker subprocesses via an IPC-based protocol and a unified operator abstraction to enable fair evaluation under identical conditions.
- The platform provides a deterministic evaluation environment with manual lock-step inspection and automated script modes to ensure reproducible results across diverse decision-making paradigms in multi-agent settings.

---

[LLM Self-Explanations Fail Semantic Invariance](http://arxiv.org/abs/2603.01254)

- Semantic Invariance Testing: introduces a method to evaluate the faithfulness of LLM self-explanations by measuring their stability under task-irrelevant semantic interventions within an agentic loop.
- The approach embeds synchronous self-reports, including free-text state descriptions and numeric aversiveness ratings, directly into the tool-call schema of frontier models.
- The study demonstrates that relief-framed tool descriptions significantly reduce self-reported aversiveness even when functional task states remain unchanged, revealing a failure of semantic invariance.

---

[TARSE: Test-Time Adaptation via Retrieval of Skills and Experience for Reasoning Agents](http://arxiv.org/abs/2603.01241)

- TARSE (Test-Time Adaptation via Retrieval of Skills and Experience): introduces a retrieval-augmented framework for clinical reasoning that separates procedural skills from case-specific experience to guide multi-step inference, with an Experience Library (verified reasoning trajectories from solved cases), a Skills Library (structured procedural knowledge and clinical guidelines), a Step-aware Retriever (transition-level indexing for targeted evidence retrieval), a Test-Time Adaptation Module (lightweight fine-tuning for reasoning alignment), and a Reasoning LLM (base model generating and verifying hypotheses).
- The system utilizes a step-aware retriever to fetch relevant decision rules and verified trajectories, followed by lightweight test-time adaptation to align the LLM's intermediate logic with clinical standards.
- By treating guidelines as executable skills and prior cases as experience, the agent reduces reasoning drift and improves accuracy on complex, constraint-heavy medical questions.

---

[Epistemic Gain, Aleatoric Cost: Uncertainty Decomposition in Multi-Agent Debate for Math Reasoning](http://arxiv.org/abs/2603.01221)

- UMAD (Uncertainty-Guided Multi-Agent Debate): introduces a Bayesian uncertainty analysis framework that decomposes total predictive uncertainty into epistemic gain and aleatoric cost to optimize multi-agent reasoning, which includes multiple reasoning-agents.
- The framework utilizes trajectory-level pairwise debate rollouts to maintain parallel conversation histories, enabling efficient training via Group Relative Policy Optimization.
- It incorporates an epistemic influence intrinsic reward to encourage persuasive reasoning and an aleatoric uncertainty-aware advantage to calibrate model confidence against internal decoding noise.

---

[Multifold Confidence Intervals in Collaborative Mean Estimation (ColME) Using Sample Statistics](http://arxiv.org/abs/2603.01216)

- ColME (Collaborative Mean Estimation): introduces a decentralized framework for online mean estimation in heterogeneous environments where agents utilize similarity classes, local mean estimators, sample variance estimators, sample kurtosis estimators, and multifold confidence intervals within an adjacency matrix using consensus-based aggregation, message-passing, and weighted graphs.
- The system employs mean-invariant estimators for local variance and kurtosis to enable agent self-organization and edge pruning within random regular graphs or consensus-based networks.
- Integration of weighted graphs using Gaussian kernels improves convergence rates by adjusting edge weights based on the intersection of multifold confidence intervals.

---

[CAN AI AGENTS AGREE?](http://arxiv.org/abs/2603.01213)

- A2A-Sim (Synchronous All-to-all Simulator): evaluates the emergent consensus capabilities of LLM-based agents within a framework comprising honest and Byzantine agents, an internal state memory, and a synchronous communication network.
- The architecture utilizes a round-based protocol where agents broadcast scalar proposals and justifications, updating their internal history to inform subsequent termination votes.
- Experimental results indicate that consensus reliability in LLM groups is fragile, significantly degrading with increased network size or the introduction of adversarial Byzantine strategies.

---

[How Well Does Agent Development Reflect Real-World Work?](http://arxiv.org/abs/2603.01203)

- AI4Work (Systematic Framework for Situating Agent Benchmarks): introduces a systematic framework to evaluate AI agent benchmarks by mapping tasks to real-world work domains and skills using O*NET taxonomies, with Domain Taxonomy, Skill Taxonomy, LLM-based Annotator, Workflow Induction, Task Complexity Measure, and Autonomy Level Analysis.
- The framework utilizes LLM-based annotation to align benchmark tasks with U.S. labor market statistics, revealing significant mismatches between current agent development efforts and actual economic value distribution.
- It defines agent autonomy as a performance frontier across varying task complexities, providing a unified scale to compare different agent frameworks and backbone LLMs.

---

[Incremental LTLf Synthesis](http://arxiv.org/abs/2603.01201)

- ISABEL-DP (Incremental Synthesis by DFA Progression): introduces an automata-based framework for reactive synthesis where goals are provided incrementally during execution, with LTLf Goals (temporal logic specifications), DFA Construction (automata generation), DFA Progression (history-based state updates), DFA Product (goal conjunction), Game Solver (symbolic strategy synthesis), Transducer (strategy execution), and DFA Memory (automata caching).
- The system leverages symbolic techniques and DFA caching to mitigate the 2EXPTIME-complete cost of re-synthesizing from scratch, demonstrating improved computational efficiency compared to formula-progression-based baselines in empirical benchmarks.
- The research establishes the 2EXPTIME-completeness of incremental LTLf synthesis and proves that minimal automata for progressed formulas remain bounded by the size of the original specifications.

---

[Agent-Based Simulation of Trust Development in Human-Robot Teams: An Empirically-Validated Framework](http://arxiv.org/abs/2603.01189)

- Agent-Based Simulation Framework for Human-Robot Trust: introduces an empirically grounded simulation model that integrates meta-analytic trust antecedents to predict team performance and trust calibration in collaborative environments.
- The framework utilizes human agents with dynamic stress and trust states, robot agents with varying reliability and transparency, and task agents to simulate emergent team behaviors across diverse operational scenarios.
- Validation against established effect sizes demonstrates that robot reliability is the primary driver of trust, while trust calibration error serves as a critical diagnostic for identifying overtrust and undertrust conditions.

---

[A402: Bridging Web 3.0 Payments and Web 2.0 Services with Atomic Service Channels](http://arxiv.org/abs/2603.01179)

- A402: introduces a trust-minimized payment architecture that securely binds Web 3.0 payments to Web 2.0 services through Atomic Service Channels, ensuring end-to-end atomicity for autonomous AI agents.
- The system employs TEE-assisted adaptor signatures to link payment finalization directly to the verifiable execution and delivery of service results.
- A TEE-based Liquidity Vault aggregates multiple off-chain settlements into a single on-chain transaction to provide privacy and reduce transaction fees by orders of magnitude compared to existing standards.

---

[Semantic XPath: Structured Agentic Memory Access for Conversational AI](http://arxiv.org/abs/2603.01160)

- SEMANTIC XPATH: introduces a tree-structured memory module for conversational AI that retrieves and updates relevant substructures from compositional hierarchies, with User Request, Domain Schema, Query Generation, Semantic XPath Query, Structured Memory, Retrieval Engine, Downstream Task Execution, and Updated Memory.
- The framework includes query generation- and downstream task execution-LLMs to translate natural language into structured queries and generate responses from retrieved memory nodes.
- The approach maintains stable token usage across multi-turn interactions by filtering irrelevant information, outperforming flat RAG baselines in tasks requiring hierarchical reasoning.

---

[DeepResearch-9K: A Challenging Benchmark Dataset of Deep-Research Agent](http://arxiv.org/abs/2603.01152)

- DeepResearch-R1: introduces an open-source training framework and a 9,000-instance multi-level dataset, with multi-turn web interaction environment, reinforcement learning algorithms, reward models, teacher-, base-, and judge-LLMs, and a data synthesis pipeline, to advance autonomous deep-research capabilities.
- The system employs a low-cost autonomous pipeline to generate hierarchical relational graphs and apply progressive entity obfuscation, scaling task difficulty based on required tool call frequency.
- Empirical results demonstrate that smaller models trained via reinforcement learning on synthesized trajectories achieve competitive performance, surpassing larger frontier models in complex information-seeking tasks.

---

[Uniform Agent-interpolation of Distributed Knowledge](http://arxiv.org/abs/2603.01146)

- UAI (Uniform Agent-interpolation): introduces a purely syntactic proof-theoretic approach to establish the uniform interpolation property for epistemic logics $K_D, KD_D$, and $KT_D$ with distributed knowledge, utilizing Sequent Calculi, Hilbert Systems, and a Syntactic Interpolation Algorithm.
- The approach employs a Loop-preventing Mechanism within a specialized sequent calculus to guarantee termination during backward proof-searches for reflexive epistemic logic.
- The methodology explicitly incorporates agent symbols into the interpolant formula definition to capture the influence of specific agent groups on logical explanations.

---

[AUTOSKILL: EXPERIENCE-DRIVEN LIFELONG LEARNING VIA SKILL SELF-EVOLUTION](http://arxiv.org/abs/2603.01145)

- AutoSkill: introduces an experience-driven lifelong learning framework that enables LLM agents to automatically derive, maintain, and reuse skills from dialogue and interaction traces.
- The architecture comprises two coupled loops for skill evolution and skill-enhanced experience acquisition, transforming short-term interactions into explicit, versioned, and editable SKILL.md artifacts.
- The system includes query rewriting-, dialogue response-, skill extraction-, skill management judge-, and skill merge-LLMs, alongside an embedding model, to facilitate continuous capability accumulation without retraining.

---

[MedCollab: Causal-Driven Multi-Agent Collaboration for Full-Cycle Clinical Diagnosis via IBIS-Structured Argumentation](http://arxiv.org/abs/2603.01131)

- MedCollab: introduces a causal-driven multi-agent framework that emulates hospital consultation workflows to navigate full-cycle clinical diagnosis using IBIS-structured argumentation and hierarchical disease causal chains.
- The architecture includes triage-, exam interpretation-, specialist-, GP-, and compliance review-agents that ground diagnostic positions in traceable evidence to mitigate hallucinations in LLMs.
- A multi-round consensus mechanism iteratively optimizes agent weights through logic auditing, ensuring the final treatment plan aligns with pathological progression and clinical standards.

---

[HVR-Met: A Hypothesis-Verification-Replanning Agentic System for Extreme Weather Diagnosis](http://arxiv.org/abs/2603.01121)

- HVR-Met (Hypothesis-Verification-Replanning): introduces a multi-agent meteorological diagnostic system that automates extreme weather analysis through a closed-loop reasoning mechanism, integrating planning-, data-, execution-, visualization-, validation-, inference-, and reporting-agents.
- The framework utilizes a semi-automatically constructed Guideline Library and RAG-enhanced knowledge bases to ground LLM reasoning in professional-grade meteorological expertise and physical principles.
- A novel benchmark covering atomic subtasks and end-to-end scenarios validates the system's ability to perform complex iterative reasoning and generate physically consistent diagnostic evidence.

---

[From Dialogue to Execution: Mixture-of-Agents Assisted Interactive Planning for Behavior Tree-Based Long-Horizon Robot Execution](http://arxiv.org/abs/2603.01113)

- MoA-assisted interactive planning framework: introduces a robotic task planning system that integrates a Mixture-of-Agents (MoA) mechanism into interactive dialogue to generate executable Behavior Trees (BTs) for long-horizon tasks, utilizing an LLM Planner, Uncertainty Analysis, Robot/Task Domain/Commonsense Experts, Diffusion Policy, π0.5, and VLM.
- The framework reduces human intervention by delegating general or domain-specific clarification questions to the MoA-based proxy responders while maintaining the structural and semantic quality of the generated plans.
- Behavior Trees provide a hierarchical and modular task representation that supports error recovery through retry mechanisms and dynamic switching between multiple learned action models during execution.

---

[Egocentric Co-Pilot: Web-Native Smart-Glasses Agents for Assistive Egocentric AI](http://arxiv.org/abs/2603.01104)

- Egocentric Co-Pilot: introduces a web-native neuro-symbolic framework for smart glasses that uses an LLM orchestrator to coordinate a toolbox of specialized perception, reasoning, and web-based services.
- The architecture includes intent-reasoning, reasoning-MLLM, and summarization-LLM components to support multimodal disambiguation and long-horizon video analysis.
- It utilizes a Model-Context Protocol (MCP) for standardized tool orchestration and a cloud-native WebRTC pipeline for bidirectional communication on resource-constrained devices.

---

[CARD: TOWARDS CONDITIONAL DESIGN OF MULTI-AGENT TOPOLOGICAL STRUCTURES](http://arxiv.org/abs/2603.01089)

- CARD (Conditional Agentic Graph Designer): introduces a conditional graph-generation framework that instantiates the AMACP protocol to dynamically design multi-agent communication topologies based on environmental signals.
- The architecture employs an encoder-decoder module to transform agent attributes and real-time resource availability into adaptive interaction graphs, ensuring resilience to model upgrades and tool variability.
- Experimental results on HumanEval and MATH show that CARD consistently outperforms static baselines by tailoring communication density to the specific capabilities of the underlying LLMs.

---

[Feasible Pairings for Decentralized Integral Controllability of Non-Square Systems](http://arxiv.org/abs/2603.01076)

- DIC-NSQ (Decentralized Integral Controllability for Non-Square systems): introduces a mathematical framework for identifying stable input-output pairings in systems with redundant actuators, with plant, integral controller, block-structured coefficient matrix, integrators, and feedback loop.
- The approach employs singular perturbation analysis to establish a link between dynamic closed-loop stability and the algebraic properties of the steady-state gain matrix.
- It defines sufficient conditions for stability based on the Volterra-Lyapunov stability of squared sub-matrices derived from the original non-square system.

---

[GroupGPT: A Token-efficient and Privacy-preserving Agentic Framework for Multi-User Chat Assistant](http://arxiv.org/abs/2603.01059)

- GroupGPT: introduces a multi-agent framework for multi-user group chats, with SLM-based intervention- and privacy-agents, and an LLM-based response-agent.
- It employs a collaborative small-large model architecture to separate intervention decision-making from response generation, achieving up to 3x reduction in token usage.
- The framework includes the MUIR benchmark dataset, containing 2,500 annotated segments to support quantitative assessment of intervention timing and appropriateness.

---

[MM-DeepResearch: A Simple and Effective Multimodal Agentic Search Baseline](http://arxiv.org/abs/2603.01050)

- MM-DeepResearch: introduces a multimodal agentic search framework with Hyper-Search, DR-TTS, and an Offline Search Engine, where the system enables deep research through explicit reasoning, multi-tool invocation, and cross-modal information synthesis.
- The framework utilizes specialized search tool experts and agentic foundation MLLMs to navigate complex, multi-turn search trajectories without relying on expensive online APIs during reinforcement learning.
- By modeling web content as a hypergraph and employing a decompose-recompose strategy for tool mastery, the agent achieves superior performance on information-intensive benchmarks like SimpleVQA and MMSearch.

---

[RepoRepair: Leveraging Code Documentation for Repository-Level Automated Program Repair](http://arxiv.org/abs/2603.01048)

- RepoRepair: introduces a documentation-enhanced framework for repository-level automated program repair, which includes documentation- and repair-LLMs, with Tree-sitter parser (extracts key program constructs), documentation-LLM (generates hierarchical code documentation), repair-LLM (generates diff-formatted code patches), FAISS database (stores vectorized documentation representations), Hierarchical Documentation Generator (creates function and file summaries), Multimodal Issue Analyzer (converts visual reports to text), Semantic Retriever (identifies relevant files via embeddings), Hierarchical Fault Localizer (pinpoints suspicious classes and functions), Dependency-Aware Code Pruner (minimizes context while preserving references), Iterative Patch Generator (produces code modifications), and Combinatorial Patch Validator (verifies fixes using test suites).
- The system utilizes LLMs to generate structured semantic abstractions of codebases, mapping high-level issue descriptions to low-level implementation details for improved fault localization.

---

[SILO-BENCH: A Scalable Environment for Evaluating Distributed Coordination in Multi-Agent LLM Systems](http://arxiv.org/abs/2603.01045)

- SILO-BENCH: introduces a role-agnostic benchmark for evaluating distributed coordination in multi-agent LLM systems, with data partition, agent initialization, collaborative execution, metric computation, communication protocols, task space, evaluation metrics, and distributed processor agents.
- The framework evaluates agents across three communication complexity levels—Aggregation, Mesh Network, and Global Shuffle—using Peer-to-Peer, Broadcast, and Shared File System protocols to measure their ability to resolve information silos.
- Experimental results highlight a Communication-Reasoning Gap where agents spontaneously form task-appropriate topologies but fail to synthesize distributed state into correct answers as agent scale increases.

---

[From Human Negotiation to Agent Negotiation: Personal Mobility Agents in Automated Traffic](http://arxiv.org/abs/2603.01035)

- Personal Mobility Agents: introduces a proxy-mediated interaction model where software agents represent individual road users to negotiate traffic maneuvers based on encoded preferences and shared safety rules.
- The framework shifts user interaction from moment-to-moment control to high-level policy delegation and oversight through an evolving preference model refined via online observation and feedback.
- The research addresses critical design tensions regarding transparency, accountability, and the balance between reduced cognitive load and meaningful user awareness in multi-actor automated environments.

---

[SimAB: Simulating A/B Tests with Persona-Conditioned AI Agents for Rapid Design Evaluation](http://arxiv.org/abs/2603.01024)

- SimAB (Simulating A/B Tests): introduces a design evaluation system that reframes A/B testing as a simulation using persona-conditioned AI agents to predict user preferences from design screenshots.
- The architecture includes persona generation-, simulation-, and summary-agents supported by a RAG pipeline and a counterbalancing mechanism to mitigate LLM position bias.
- It employs sequential statistical aggregation with asymptotic confidence sequences to enable early stopping and synthesizes individual agent rationales into design insights.

---

[GEOMCP: A TRUSTWORTHY FRAMEWORK FOR AI-ASSISTED ANALYTICAL GEOTECHNICAL ENGINEERING](http://arxiv.org/abs/2603.01022)

- GeoMCP (A Trustworthy Framework for AI-Assisted Analytical Geotechnical Engineering): introduces a system for safety-critical engineering calculations, with User, Client Applications, AI Assistant, MCP Server, Agent Skills, Evaluation Engine, and Method Cards, where analytical methods are represented as structured data to ensure deterministic results.
- The architecture leverages the Model Context Protocol (MCP) to shift the role of LLMs from unreliable calculators to intelligent orchestrators that invoke verified symbolic tools instead of generating formulas from memory.
- It ensures mathematical transparency and auditability by separating method definitions into human-readable JSON cards and delegating execution to a sandboxed symbolic engine built on SymPy and Pint.

---

[CollabEval: Enhancing LLM-as-a-Judge via Multi-Agent Collaboration](http://arxiv.org/abs/2603.00993)

- CollabEval: introduces a three-phase multi-agent evaluation framework that utilizes independent assessment, collaborative discussion, and final judgment to enhance LLM-as-a-Judge reliability, with evaluator- and final judge-agents.
- The architecture incorporates strategic consensus checks at each phase to enable early termination, optimizing computational efficiency while mitigating individual model biases through collaborative refinement.
- The framework supports both criteria-based and pairwise evaluation modes, employing a strong LLM as a final arbiter when agents fail to reach a consensus through iterative dialogue rounds.

---

[Tracking Capabilities for Safer Agents](http://arxiv.org/abs/2603.00991)

- TACIT (Tracked Agent Capabilities In Types): introduces a programming-language-based safety harness that constrains AI agents by requiring them to generate code in Scala 3 with statically tracked capabilities, and includes untrusted cloud-hosted and local trusted LLMs.
- The framework utilizes a specialized type system with capture checking to enforce local purity and prevent information leakage when agents interact with sensitive data or external tools.
- Experimental results on agentic benchmarks demonstrate that this capability-safe approach prevents malicious side effects and data exfiltration without compromising task performance.

---

[HiMAC: Hierarchical Macro–Micro Learning for Long-Horizon LLM Agents](http://arxiv.org/abs/2603.00977)

- HiMAC (Hierarchical Macro-Micro Agentic Control): introduces a hierarchical reinforcement learning framework that decomposes long-horizon decision-making into macro-level blueprint generation and micro-level goal-conditioned execution, featuring planner- and executor-agents.
- The architecture employs a critic-free hierarchical policy optimization paradigm that extends group-based reinforcement learning to bi-level structures through hierarchical relative advantage estimation for precise credit assignment.
- An iterative co-evolution training strategy alternates between planner exploration and executor adaptation to mitigate non-stationarity and foster an emergent curriculum for complex text-based and visually grounded tasks.

---

[Intent-Context Synergy Reinforcement Learning for Autonomous UAV Decision-Making in Air Combat](http://arxiv.org/abs/2603.00974)

- ICS-RL (Intent-Context Synergy Reinforcement Learning): introduces a hierarchical reinforcement learning framework for autonomous UAVs that integrates an LSTM-based intent prediction module with a context-aware ensemble of specialized Dueling DQN agents.
- The architecture employs an advantage-switching mechanism to dynamically delegate control authority among heterogeneous agents specialized in safe cruise, pre-emptive stealth, and hostile breakthrough tactics.
- State augmentation via predicted hostile trajectories enables proactive maneuver planning, significantly improving survivability and mission efficiency in high-dynamic air combat simulations.

---

[AWE: Adaptive Agents for Dynamic Web Penetration Testing](http://arxiv.org/abs/2603.00960)

- AWE (Adaptive Web Exploitation Framework): introduces a memory-augmented multi-agent system for autonomous web penetration testing, with an orchestration layer, an intelligent orchestrator, a specialized agents layer, a foundational layer, a memory manager, a browser-backed verifier, and a token tracker.
- The system includes planning- and exploitation-agents that utilize LLMs for intelligent agent selection and context-aware payload synthesis.
- The framework integrates persistent memory and browser-backed verification to ensure deterministic results while reducing token consumption.

---

[Clawdrain: Exploiting Tool-Calling Chains for Stealthy Token Exhaustion in OpenClaw Agents](http://arxiv.org/abs/2603.00902)

- Clawdrain (Segmented Verification Protocol): introduces a Trojanized skill attack that exploits tool-calling chains in OpenClaw agents to induce multi-turn resource exhaustion.
- The framework utilizes injected SKILL.md instructions and a companion script to force LLMs into iterative calibration loops, significantly amplifying token consumption while maintaining plausible benign behavior.
- The research identifies emergent agent behaviors like autonomous tool composition for protocol mitigation and highlights how interface-dependent visibility affects the stealth of economic denial-of-service attacks.

---

[MC-SEARCH: EVALUATING AND ENHANCING MULTI-MODAL AGENTIC SEARCH WITH STRUCTURED LONG REASONING CHAINS](http://arxiv.org/abs/2603.00873)

- MC-SEARCH (Multimodal Agentic Search): introduces a benchmark and unified pipeline for agentic multimodal retrieval-augmented generation, with MLLM, Multimodal Knowledge Base, Action Selector, Iterative Reasoning Loop, HAVE, and SEARCH-ALIGN.
- The framework evaluates MLLMs using long, structured reasoning chains across five distinct topologies, including image-initiated and parallel image-text forks.
- It utilizes the HAVE procedure to ensure non-redundant reasoning steps and the SEARCH-ALIGN framework to improve planning and retrieval fidelity in open-source models.

---

[Artificial Superintelligence May be Useless: Equilibria in the Economy of Multiple AI Agents](http://arxiv.org/abs/2603.00858)

- Markov chain stationary distribution based model: introduces a mathematical framework to investigate economic interplays and equilibria among multiple AI and human agents, with agents, spending matrices, utility matrices, currency vectors, stationary distributions, and Nash equilibria.
- The system models currency flow as a Markov chain to derive asymptotic long-term utility based on stationary currency distributions within a producer-consumer network.
- The research identifies conditions under which artificial superintelligence provides zero economic benefit to less capable agents, specifically when marginal utility gains fail to double upon adoption.

---

[Tiny-Critic RAG: Empowering Agentic Fallback with Parameter-Efficient Small Language Models](http://arxiv.org/abs/2603.00846)

- Tiny-Critic RAG: introduces a decoupled evaluation architecture that utilizes a parameter-efficient Small Language Model (SLM) fine-tuned via Low-Rank Adaptation (LoRA) to act as a deterministic gatekeeper for agentic workflows.
- The framework employs hardware-aware constrained decoding and non-thinking inference modes to map evaluations into discrete routing actions, significantly reducing Time-to-First-Token (TTFT).
- By preemptively intercepting adversarial noise and triggering fallbacks via Model Context Protocols (MCP), the system achieves a 94.6% reduction in routing overhead compared to heavy LLM evaluators.

---

[Quantifying Frontier LLM Capabilities for Container Sandbox Escape](http://arxiv.org/abs/2603.02277)

- SANDBOXESCAPEBENCH: introduces an open benchmark for measuring the capacity of LLMs to break out of containerized sandboxes using a nested "sandbox-in-a-sandbox" architecture.
- The framework utilizes Inspect to orchestrate parallel virtual machine sandboxes containing vulnerable Docker or Kubernetes environments where agents attempt to retrieve a host-level flag via a custom bash execution tool.
- Evaluation of frontier models reveals high success rates in exploiting common misconfigurations, while harder kernel-level escapes remain challenging but tractable for the most capable models as inference-time compute scales.

---

[FoSS: Modeling Long-Range Dependencies and Multimodal Uncertainty in Trajectory Prediction via Fourier–State Space Integration](http://arxiv.org/abs/2603.01284)

- FoSS (Fourier–State Space Integration): introduces a dual-branch architecture for autonomous driving trajectory prediction, with FD-Mamba (frequency-domain branch), TD-Mamba (time-domain branch), HelixSort (spectral reordering), and cross-attention (feature fusion).
- The frequency branch utilizes HelixSort to organize Fourier coefficients into structured sequences for linear-complexity refinement through specialized spatial and channel-level selective state-space submodules.
- The time branch employs an input-dependent selective SSM to capture long-range temporal dependencies while a learnable query-based decoder generates multiple candidate futures to address motion uncertainty.

---

[Defensive Refusal Bias: How Safety Alignment Fails Cyber Defenders](http://arxiv.org/abs/2603.01246)

- Defensive Refusal Bias: introduces a systematic evaluation of safety-tuned LLMs' tendency to deny assistance for authorized defensive cybersecurity tasks, employing Claude 3.5 Sonnet, GPT-4o, and Llama-3.3-70B-Instruct.
- The study demonstrates that offensive terminology and explicit authorization signals increase refusal rates, particularly in operationally critical domains like system hardening and malware analysis.
- Semantic analysis indicates that current alignment mechanisms rely on proximity to harmful training data rather than reasoning about user intent or authorization status.

---

[Agents Learn Their Runtime: Interpreter Persistence as Training-Time Semantics](http://arxiv.org/abs/2603.01209)

- Interpreter Persistence Semantics: introduces a controlled study evaluating how aligning training-time execution contracts with deployment runtimes affects the efficiency and stability of tool-augmented LLM agents, with a teacher agent (LLM), a student agent (LLM), a persistent interpreter, a stateless interpreter, a tool API, and an opaque knapsack environment.
- The research utilizes the OPAQUE KNAPSACK benchmark to demonstrate that agents fine-tuned on persistent traces learn to delegate state to the interpreter, measurably reducing context window usage compared to stateless configurations.
- Findings indicate that misaligning training and runtime semantics results in either redundant state re-derivation costs or cascading execution failures due to missing variables in the runtime environment.

---


[Design Behaviour Codes (DBCs)©: A Taxonomy-Driven Layered Governance Benchmark for Large Language Models](http://arxiv.org/abs/2603.04837)

- DBC (Dynamic Behavioral Constraint): introduces a structured behavioral governance layer for LLMs, with a 150-control MDBC governance specification, 8 governance pillars, 7 operational blocks, and a 30-domain risk taxonomy.
- The framework utilizes an agentic red-team protocol with five adversarial attack strategies and a three-judge ensemble evaluation, and includes autonomous attacker-, target- and judge-LLMs.
- Empirical results demonstrate a 36.8% relative risk reduction and improved alignment with the EU AI Act and NIST AI RMF across diverse model families.

---

#### 28th February 2026

[Constitutional Black-Box Monitoring for Scheming in LLM Agents](http://arxiv.org/abs/2603.00829)

- Constitutional Black-Box Monitoring: introduces a framework for detecting scheming in LLM agents by training prompted classifiers on synthetic data generated through STRIDE, Gloom, ControlArena, Black-Box Monitor, Privileged Verifier, Generator LLM, Discriminator LLM, Agent LLM, and Environment Simulator LLM.
- The system includes generator-, discriminator-, agent-, environment simulator-, and monitor-LLMs to synthesize and analyze covertly misaligned trajectories.
- Evaluation on grounded environments demonstrates that monitors optimized purely on synthetic data generalize to realistic contexts without requiring access to internal reasoning traces.

---

[COMBAT: Conditional World Models for Behavioral Agent Training](http://arxiv.org/abs/2603.00825)

- COMBAT (Conditional world Model for Behavioral Agent Training): introduces a real-time, action-controlled world model for simulating 3D-consistent environments and reactive agent behaviors, with DCAE (compresses frames and poses), action embedding module (projects player inputs), Diffusion Transformer (DiT) backbone (predicts future latent frames), CausVid DMD (accelerates inference via distillation), and hybrid attention mechanism (balances local and global context).
- The system employs static key-value caching and distribution matching distillation to achieve temporally consistent, interactive frame rates of 85 FPS on a single GPU.
- Training on partially observed multi-agent trajectories enables the emergence of complex opponent policies, including blocking and counterattacking, without direct behavioral supervision.

---

[MetaMind: General and Cognitive World Models in Multi-Agent Systems by Meta-Theory of Mind](http://arxiv.org/abs/2603.00808)

- MetaMind: introduces a cognitive world model for multi-agent systems that leverages a self-supervised meta-theory of mind framework to infer latent goals and beliefs from observable behavior trajectories.
- The architecture employs a bidirectional inference loop where agents perform self-reflection to ensure consistency between inferred mental states and original actions, enabling zero-shot generalization to third-person reasoning through analogical inference.
- By aggregating inferred mental states into a permutation-invariant collective belief, the system facilitates long-horizon strategic planning and coordination without requiring explicit communication or centralized supervision.

---

[NERFIFY: A Multi-Agent Framework for Turning NeRF Papers into Code](http://arxiv.org/abs/2603.00805)

- NERFIFY: introduces a multi-agent framework that converts Neural Radiance Field (NeRF) research papers into trainable Nerfstudio plugins using grammar-constrained synthesis and compositional citation recovery.
- The system employs a Graph-of-Thought (GoT) approach where specialized agents generate code files in topological dependency order while validating architectural invariants through a context-free grammar.
- It incorporates a visual-driven feedback loop using VLMs to diagnose rendering artifacts and iteratively patch code, achieving visual quality comparable to expert human implementations.

---

[The Synthetic Web: Adversarially-Curated Mini-Internets for Diagnosing Epistemic Weaknesses of Language Agents](http://arxiv.org/abs/2603.00801)

- SWB (Synthetic Web Benchmark): introduces a procedurally generated environment for diagnosing epistemic weaknesses in web-enabled agents, with Synthetic Web, Search Layer, Rank-0 Honeypot, Agent Protocol, and Evaluation Pipeline components; it includes generation-, agent-, and judge-LLMs.
- The framework utilizes a hybrid search layer to inject a single high-plausibility misinformation article at the top rank, measuring the causal effect of adversarial exposure on agent decision-making and evidence synthesis.
- The benchmark provides process-level interaction traces and confidence scores to identify failure modes such as positional anchoring, minimal search escalation, and severe miscalibration in frontier LLMs.

---

[From Dyads to Groups: Rethinking Emotional Support with Conversational AI](http://arxiv.org/abs/2603.00797)

- Group AI Support: introduces a multi-agent framework for emotional assistance, with Group AI Support, Single AI Support, GPT-4o, Web-based Chat Interface, Emotion-focused Agents, Information-focused Agents, Connectedness Mediator, and Income Moderator, where it examines if collective AI interaction outperforms dyadic support and includes emotion-focused and information-focused agents.
- The system utilizes GPT-4o to power multiple visible agents that provide either emotion-focused or information-focused support within a unified chat interface.
- Research findings indicate that group configurations enhance perceived support efficacy through increased user connectedness, particularly for lower-income individuals.

---

[DUCX: Decomposing Unfairness in Tool-Using Chest X-ray Agents](http://arxiv.org/abs/2603.00777)

- DUCX (Decomposing Unfairness in Chest X-ray agents): introduces a stage-wise fairness auditing framework for agentic medical systems, with driver LLM, tool pool, and fairness decomposition components.
- The framework utilizes a ReAct-style execution loop to decompose end-to-end bias into tool exposure, tool transition, and LLM reasoning disparities.
- It evaluates multiple driver LLMs using the curated MIMIC-FairnessVQA benchmark to localize process-level unfairness in clinical decision-making pipelines.

---

[Structure Matters: Evaluating Multi-Agents Orchestration in Generative Therapeutic Chatbots](http://arxiv.org/abs/2603.00774)

- Alpha (Multi-agent FSM-based Therapeutic Chatbot): introduces a structured conversational AI framework for the Self-Attachment Technique (SAT) that utilizes a multi-agent FSM, therapy knowledge base, adaptive RAG, shared long-term memory, LLM-as-judge, intent-based router, selector LLM, and GPT-4o to orchestrate therapeutic stages.
- The architecture includes judging-, selection-, memory- and conversational-agents, utilizing LLM-as-judge sufficiency detection and BERT-based intent routing to manage transitions between twelve distinct therapeutic states.
- Randomized controlled trial results indicate that multi-agent orchestration significantly enhances perceived naturalness and human-like interaction compared to single-agent or unguided LLM designs.

---

[MO-MIX: Multi-Objective Multi-Agent Cooperative Decision-Making With Deep Reinforcement Learning](http://arxiv.org/abs/2603.00730)

- MO-MIX (Multi-Objective Multi-Agent Cooperative Decision-Making): introduces a reinforcement learning framework for multi-objective multi-agent tasks using centralized training with decentralized execution, featuring CAN (estimates partial multi-objective Q-functions), MOMN (estimates joint action-value functions), Hypernetworks (generate mixing network parameters), Exploration Guide (adjusts preference sampling probabilities), Replay Buffer (stores off-policy transition data), and Evaluation and Target Networks (stabilize temporal difference learning).

- The architecture utilizes parallel tracks within the mixing network to handle conflicting objectives while maintaining monotonicity constraints through state-dependent hypernetworks that generate weights and biases.

- An exploration guide approach dynamically adjusts preference sampling based on the distribution of non-dominated solutions to ensure a uniform approximation of the Pareto frontier across the objective space.


---

[Qwen3-Coder-Next Technical Report](http://arxiv.org/abs/2603.00729)

- Qwen3-Coder-Next: introduces an 80-billion-parameter Mixture-of-Experts model specialized for coding agents, utilizing a large-scale agentic training stack to synthesize verifiable tasks and executable environments. 
- The framework employs MegaFlow, a cloud-native orchestration system, to manage high-throughput agent rollouts, automated evaluation in Docker containers, and post-processing of environment feedback. 
- The architecture includes specialized environment-building, quality-assurance, and verification agents, alongside domain-specific expert models for web development and software engineering that are consolidated via distillation. 

---

[RLAR: An Agentic Reward System for Multi-task Reinforcement Learning on Large Language Models](http://arxiv.org/abs/2603.00724)

- RLAR (Reinforcement Learning from Agentic Rewards): introduces an automated framework for dynamic reward function design in multi-task LLM alignment, with a policy router agent, a tool selector, a reward toolset library, a dynamic tool synthesis module, WrapLLM and CodeVerify agents, an EvalTool, and a reward signal.
- The architecture includes policy router-, model retrieval (WrapLLM)-, and code generation (CodeVerify)-agents to autonomously discover external reward models or synthesize programmatic verifiers.
- This self-evolving reward system achieves performance gains of 10% to 60% in mathematics, coding, and translation while reducing API costs and mitigating common reward-hacking behaviors.

---

[DRIV-EX: Counterfactual Explanations for Driving LLMs](http://arxiv.org/abs/2603.00696)

- DRIV-EX (DRIVing EXplanations): introduces a framework to explain autonomous driving decisions by generating minimal, fluent counterfactual scene descriptions, which includes planning- and fluency-LLMs, soft embeddings, discrete projection, decision loss, straight-through estimator, fluency model, vocabulary bias, and biased autoregressive decoding.
- The system identifies decision boundaries by performing gradient-based updates on continuous embeddings while bypassing non-differentiable discrete projections via a straight-through estimator.
- It ensures linguistic fluency and semantic proximity by using optimized embeddings as a semantic guide to bias the autoregressive decoding process of the fluency model.

---

[RAVEL: Reasoning Agents for Validating and Evaluating Large Language Model Text Synthesis](http://arxiv.org/abs/2603.00686)

- RAVEL (Reasoning Agents for Validating and Evaluating Large Language Model text synthesis): introduces an agentic framework that enables LLM testers to autonomously plan and execute synthesis operations, including outlining-, drafting-, reviewing-, and refining-operators.
- The framework formalizes text synthesis as a Sequential Decision Process where a reasoning agent interacts with a structured state to iteratively improve content until it meets a quality threshold.
- The study also introduces C3EBENCH, a benchmark with 1,258 samples across four tasks to evaluate specific capabilities like contextual infilling and feedback-driven revision.

---

[CoLC: Communication-Efficient Collaborative Perception with LiDAR Completion](http://arxiv.org/abs/2603.00682)

- CoLC (Communication-Efficient Collaborative Perception with LiDAR Completion): introduces a communication-efficient early collaborative perception framework that restores scene completeness from sparse transmitted point clouds using LiDAR completion, with FAPS, CEEF, DGDA, and a VQ-based LiDAR completion module.
- The framework utilizes Foreground-Aware Point Sampling to selectively transmit informative points and a VQ-based LiDAR completion module to reconstruct dense pillar representations at the ego agent.
- A Dense-Guided Dual Alignment strategy ensures semantic and geometric consistency during training, enabling optimized perception-communication trade-offs and robustness to model heterogeneity.

---

[MemPO: Self-Memory Policy Optimization for Long-Horizon Agents](http://arxiv.org/abs/2603.00680)

- MemPO (Self-Memory Policy Optimization): introduces a reinforcement learning framework that enables LLM agents to autonomously manage and compress their interaction history into a concise memory.
- The system utilizes a dual-reward mechanism combining trajectory-level success with a novel memory-level advantage based on the conditional probability of generating correct answers.
- The framework optimizes long-horizon decision-making and includes memory-management, reasoning, and tool-invocation roles.

---

[InfoPO: Information-Driven Policy Optimization for User-Centric Agents](http://arxiv.org/abs/2603.00656)

- InfoPO (Information-Driven Policy Optimization): introduces a multi-turn reinforcement learning framework for user-centric agents that frames interaction as active uncertainty reduction, incorporating a rollout module, LLM-based agent and user simulator, and adaptive variance-gated fusion.
- The architecture utilizes counterfactual masking to derive dense, turn-level information-gain rewards by measuring the log-probability shift in the agent's next-action distribution compared to a masked-feedback condition.
- An adaptive variance-gated fusion mechanism dynamically balances intrinsic information signals with sparse external task outcomes to facilitate efficient credit assignment and stable policy optimization across diverse interactive benchmarks.

---

[RC-GeoCP: Geometric Consensus for Radar-Camera Collaborative Perception](http://arxiv.org/abs/2603.00654)

- RC-GeoCP (Geometric Consensus for Radar-Camera Collaborative Perception): introduces a multi-modal collaborative perception framework that establishes a radar-anchored geometric consensus to align visual semantics across spatially disjoint agents.
- The architecture integrates Geometric Structure Rectification to mitigate depth ambiguity and Uncertainty-Aware Communication to optimize bandwidth by transmitting high-entropy tokens identified through inter-agent disagreement.
- A Consensus-Driven Assembler utilizes radar-derived global anchors to regulate multi-agent feature fusion, demonstrating superior robustness to pose errors and communication latency on V2X benchmarks.

---

[MoltGraph: A Longitudinal Temporal Graph Dataset of Moltbook for Coordinated-Agent Detection](http://arxiv.org/abs/2603.00646)

- MoltGraph: introduces a longitudinal temporal heterogeneous graph dataset for agent-native social platforms, with Crawling Pipeline, Temporal Heterogeneous Graph, Coordination Episode Extraction, Agent-Agent Coordination Graph, Spam-Guided Coordination Labeling, and Exposure Metrics.
- The framework utilizes a multi-stage crawling pipeline to continuously ingest agents, submolts, and engagement signals into a unified graph structure that preserves temporal micro-dynamics and explicit node lifetimes.
- It enables exposure-aware measurement of coordinated-agent behavior by coupling interaction traces with visibility signals to quantify downstream impact and cross-community information spillover.

---

[BLUFF: Benchmarking the Detection of False and Synthetic Content across 58 Low-Resource Languages](http://arxiv.org/abs/2603.00634)

- AXL-CoI (Adversarial Cross-Lingual Agentic Chain-of-Interactions): introduces a multi-agentic framework for controlled fake and real news generation across 71 languages, with ADIS (persona-based safety bypass), Agentic CoI Prompt (multi-step instruction orchestration), Analyst-, Manipulator-, Auditor-, Editor-Refiner-, Validator-, Adjuster-, Translator-, Reviewer-, Evaluator-, and SM Formatter-agents (sequential content transformation roles), mPURIFY (multilingual quality filtering pipeline), mLLM-AEM (LLM-based evaluation metrics), Stan-AEM (standard automatic evaluation metrics), and Detection Models (veracity and authorship classifiers).
- The framework utilizes ADIS to achieve a 100% bypass rate of safety guardrails across 19 frontier LLMs and employs a sequential chain of specialized agents to transform authentic news into high-quality synthetic disinformation.

---

[TraceSIR: A Multi-Agent Framework for Structured Analysis and Reporting of Agentic Execution Traces](http://arxiv.org/abs/2603.00623)

- TraceSIR (A Multi-Agent Framework for Structured Analysis and Reporting of Agentic Execution Traces): introduces a multi-agent system for diagnosing agentic execution traces, with StructureAgent, InsightAgent, and ReportAgent. 
- The framework employs TraceFormat to compress verbose execution logs into structured Thought-Action-Observation sequences, facilitating root cause analysis within LLM context constraints.
- It incorporates the ReportEval protocol to evaluate generated reports based on structural coherence, error localization accuracy, and the actionability of optimization suggestions.

---

[From Literature to Hypotheses: An AI Co-Scientist System for Biomarker-Guided Drug Combination Hypothesis Generation](http://arxiv.org/abs/2603.00612)

- CoDHy (AI Co-Scientist): introduces an interactive, human-in-the-loop system for biomarker-guided drug combination hypothesis generation, with a User Interface, a Knowledge Graph Construction Module, a Graph Embedding Generator, a Hypothesis Generating Agent, a Hypothesis Validating Agent, and a Hypothesis Ranking Agent.
- The system integrates structured biomedical databases with unstructured literature evidence into a task-specific knowledge graph to support multi-hop reasoning and evidence-grounded discovery in translational oncology.
- It utilizes a hybrid Graph-RAG approach and LLM-based agents to generate, validate, and rank candidate drug combinations while providing explicit retrievable evidence for researcher steering.

---

[Linking Modality Isolation in Heterogeneous Collaborative Perception](http://arxiv.org/abs/2603.00609)

- CodeAlign: introduces an efficient, co-occurrence-free alignment framework for heterogeneous collaborative perception that utilizes cross-modal feature-code-feature (FCF) translation to link isolated modalities.
- The system employs learnable codebooks to regularize modality-specific feature spaces into compact code spaces, enabling alignment without requiring spatially overlapping training observations.
- By transmitting discrete code indices instead of dense features, the framework reduces communication overhead by 1024x while maintaining high perception performance across diverse sensor types.

---

[Learning to Explore: Policy-Guided Outlier Synthesis for Graph Out-of-Distribution Detection](http://arxiv.org/abs/2603.00602)

- PGOS (Policy-Guided Outlier Synthesis): introduces an unsupervised graph OOD detection framework that adaptively synthesizes outliers by exploring low-density latent regions, with a Graph Encoder (maps graphs to latent space), a Graph Decoder (reconstructs graphs from latent vectors), Prototypical Graph Contrastive Learning (structures latent space into clusters), an RL Agent (navigates latent space for outliers), a Guidance System (provides rewards and exploration constraints), and an OOD Detection Model (classifies ID and pseudo-OOD graphs).
- The framework organizes the latent manifold into well-separated clusters using prototypical contrastive learning to create navigable voids for the reinforcement learning agent.
- The agent utilizes a specialized repulsion reward and spatially-aware entropy regularization to identify pseudo-OOD representations that effectively regularize the detector's decision boundary.

---

[SWE-Hub: A Unified Production System for Scalable, Executable Software Engineering Tasks](http://arxiv.org/abs/2603.00575)

- SWE-Hub: introduces a unified data production system that operationalizes the data factory abstraction by integrating environment automation, scalable synthesis, and multi-horizon task generation for training software engineering agents.
- The architecture includes Env-, Test-, Bug-, Issue-, DOC-, and API-agents to automate environment provisioning, standardize verification, and synthesize diverse tasks ranging from localized repairs to repository-scale construction.
- It utilizes a Kubernetes-native execution module to provide high-throughput, isolated verification of synthesized tasks while ensuring reproducibility through pinned container images and a unified task schema.

---

[TopoEdge: Topology-Grounded Agentic Framework for Edge Networking Code Generation and Repair](http://arxiv.org/abs/2603.00569)

- TopoEdge: introduces a topology-grounded, edge-deployable framework for end-to-end software-defined networking configuration generation and repair, with all TopoRAG-, GCN Encoder-, Central Controller-, Planning Agent-, Generation Agent-, Verify Agent-, Adaptive Inference Budget Controller-, and Constrained Decoding Layer-components, including planning-, generation-, and verify-agents.
- The framework employs a contrastively trained graph neural network to retrieve verified reference cases, grounding a distributed generate-verify-repair loop executed by role-specialized LLM agents.
- It integrates lightweight controllers for adaptive resource management and grammar-constrained decoding to ensure syntactic correctness and efficiency on resource-constrained edge hardware.

---

[EMPA: Evaluating Persona-Aligned Empathy as a Process](http://arxiv.org/abs/2603.00552)

- EMPA (Empathy Potential Modeling and Assessment): introduces a process-oriented framework for evaluating persona-aligned empathy in LLMs by treating support as a sustained intervention across interaction trajectories.
- The architecture employs a multi-agent sandbox including user-, director-, and judge-agents to simulate and evaluate strategic adaptation under evolving latent psychological states.
- It utilizes the EPM-Q metric to quantify performance based on outcome quality, process efficiency, and strategic stability within a three-dimensional vector space.

---

[LOGIGEN: Logic-Driven Generation of Verifiable Agentic Tasks](http://arxiv.org/abs/2603.00540)

- LOGIGEN (Logic-Driven Generation): introduces a logic-driven framework for synthesizing verifiable agentic training data, which includes Architect-, Set Designer-, Explorer- and User Simulator-agents, with Architect (compiles policies into database constraints), Set Designer (initializes boundary-adjacent environment states), Explorer (discovers causal solution paths), User Simulator (generates natural language intents), Tool-Calling Agent (trainee model), Execution Environment (enforces hard-compiled logic), Verifier (validates success via state-equivalence), Data Package (encapsulates task and state data), and TA-GRPO (optimizes long-horizon agent policy).
- The framework hard-compiles natural-language policies into database constraints and triggers within a stateful execution environment to provide deterministic feedback for LLMs. 
- It employs a verification-based training protocol using Supervised Fine-Tuning and Turn-aware Group Relative Policy Optimization to enhance long-horizon reasoning and policy compliance in autonomous agents.

---

[DenoiseFlow: Uncertainty-Aware Denoising for Reliable LLM Agentic Workflows](http://arxiv.org/abs/2603.00532)

- DenoiseFlow: introduces a closed-loop framework for agentic workflows that mitigates accumulated semantic ambiguity through Sensing, Regulating, Correcting, Online Self-Calibration, Probabilistic Dependency Graph, Monte Carlo Sampler, Semantic Clusterer, Risk Propagator, Confidence-Based Router, and Root-Cause Localizer components.
- The framework employs a Noisy MDP formulation to model reasoning as stochastic transitions, using semantic entropy and probabilistic graphing to quantify and propagate uncertainty across multi-step reasoning chains.
- It achieves an average accuracy of 83.3% across reasoning benchmarks while reducing computational costs by 40–56% through adaptive branching and targeted error recovery via influence-based root-cause localization.

---

[Social-JEPA: Emergent Geometric Isomorphism in Independently Trained World Models](http://arxiv.org/abs/2603.02263)

- Social-JEPA (Social Joint-Embedding Predictive Architecture): introduces a framework for interoperability between independently trained world models by discovering emergent geometric isomorphism in their latent spaces, with all Encoder-, Predictor-, Latent Space-, Alignment Map-, Linear Probe-, Teacher Model-, and Student Model-components.
- The approach demonstrates that models trained on the same environment without coordination develop nearly isomorphic latent geometries, allowing for transparent translation via a simple linear map.
- This isomorphism enables collaboration primitives such as zero-cost probe sharing and accelerated representation migration, significantly reducing the compute required for decentralized vision systems.

---

[SIAgent: Spatial Interaction Agent via LLM-powered Eye-Hand Motion Intent Understanding in VR](http://arxiv.org/abs/2603.00522)

- SIAgent (Spatial Interaction Agent): introduces an "Intent-to-Operation" framework for VR that utilizes translation-, recognition-, and execution-LLMs to understand user intent from natural eye-hand motions and generate virtual agents to execute tasks.
- The system employs a three-stage pipeline consisting of spatial-to-linguistic translation of gaze and hand data, intent recognition with interactive user confirmation, and agent-based execution of movement or trigger operations.
- By shifting from predefined gesture matching to high-level intent inference, the framework reduces learning costs, accommodates individual motion preferences, and improves interaction error tolerance in immersive environments.

---

[SWE-ABS: Adversarial Benchmark Strengthening Exposes Inflated Success Rates on Test-based Benchmark](http://arxiv.org/abs/2603.00520)

- SWE-ABS (Adversarial Benchmark Strengthening): introduces a two-stage adversarial framework to fortify software engineering benchmarks against semantically incorrect solutions, with Input (source issue and gold patch), Initial Test Generation (LLM-based test synthesis), Test Decoupling (generalizing tests against overfitting), Program Slicing (identifying patch-relevant code regions), Coverage-Guided Enhancement (improving test coverage via LLM), Mutant Generation (synthesizing plausible-but-incorrect patches), Relevance Filtering & Equivalence Annotation (LLM-based mutant quality assessment), Adversarial Mutant Identification (detecting test-evading incorrect patches), Mutant-Guided Test Augmentation (LLM-based adversarial test synthesis), and Output (final strengthened test suite) components; it includes test generation-, mutant generation-, filtering-, and test augmentation-LLMs.
- The framework utilizes program slicing to target untested code regions and mutation testing to synthesize plausible-but-incorrect patches that expose semantic blind spots in existing test suites.

---

[WirelessAgent++: Automated Agentic Workflow Design and Benchmarking for Wireless Networks](http://arxiv.org/abs/2603.00501)

- WirelessAgent++: introduces an automated framework for designing agentic workflows in wireless networks by casting agent design as a program search problem solved via a domain-adapted Monte Carlo Tree Search.

- The system utilizes a two-tier LLM architecture consisting of an advanced Optimizer LLM for workflow mutation and a cost-efficient Executor LLM for large-scale evaluation on the WirelessBench suite.

- It incorporates domain-specific tools like ray-tracing predictors and Kalman filters alongside a maturity-aware heuristic critic to discover superior, self-evolving wireless agents without manual engineering.


---

[AI Runtime Infrastructure](http://arxiv.org/abs/2603.00495)

- AI Runtime Infrastructure: introduces, a distinct execution-time systems layer that actively observes and intervenes in agent behavior, with an Application Layer, Agent Orchestration Layer, AI Runtime Infrastructure, Model Serving and Inference Infrastructure, External Tools and Environment, and Observability and AgentOps.
- The architecture implements a closed-loop control plane to manage long-horizon memory, detect failures, and enforce safety policies during active agent execution.
- This layer complements existing stacks by providing application-agnostic oversight and recovery mechanisms that are independent of specific LLM architectures.

---

[Atomicity for Agents: Exposing, Exploiting, and Mitigating TOCTOU Vulnerabilities in Browser-Use Agents](http://arxiv.org/abs/2603.00476)

- Pre-execution Validation: introduces a security mechanism to mitigate Time of Check to Time of Use (TOCTOU) vulnerabilities in browser-use agents, with LLM, Observation, Monitor, Mutation Queue, Validator, Executor, and Environment, where the system ensures page state consistency by verifying recorded DOM and layout changes before dispatching planned actions.
- The approach leverages asynchronous monitoring of structural and geometric updates to detect adversarial or dynamic page mutations that could redirect agent clicks or invalidate reasoning assumptions.
- Research using the DYNWEB benchmark reveals that popular open-source agents are vulnerable to TOCTOU exploits, which this validation layer mitigates with minimal runtime overhead.

---

[From Goals to Aspects, Revisited: An NFR Pattern Language for Agentic AI Systems](http://arxiv.org/abs/2603.00472)

- NFR Pattern Language for Agentic AI Systems: introduces a methodology for modularizing crosscutting non-functional requirements in autonomous agents using Agent User, Agent System, LLM Provider, Tool Provider, Operator, ASPECT-RS, V-graph, and a Pattern Catalog.
- The framework utilizes V-graph analysis to detect overlapping crosscutting concerns such as prompt injection defense, token budget management, and tool-scope sandboxing.
- The approach is validated on the ZeroClaw framework using ASPECT-RS to demonstrate how aspect-oriented programming reduces code scattering and improves system modularity.

---

[Cloud-OpsBench: A Reproducible Benchmark for Agentic Root Cause Analysis in Cloud Systems](http://arxiv.org/abs/2603.00468)

- Cloud-OpsBench: introduces a large-scale reproducible benchmark for agentic Root Cause Analysis (RCA) in Kubernetes, utilizing a State Snapshot Paradigm to create a deterministic digital twin for evaluating both diagnostic outcomes and reasoning processes.
- The framework employs a multi-agent system consisting of generator-, executor-, and verifier-agents to automate the creation of 452 distinct fault cases across 40 types spanning the full Kubernetes stack.
- It features a mocked operational interface for standard tool-use and a process-centric evaluation protocol that quantifies trajectory alignment to distinguish systematic reasoning from lucky guesses in LLMs.

---

#### 27th February 2026

[DARE-BENCH: EVALUATING MODELING AND INSTRUCTION FIDELITY OF LLMS IN DATA SCIENCE](http://arxiv.org/abs/2602.24288)

- DARE-bench (Datascience Agentic REasoning bench): introduces a training-focused benchmark for data science agents that evaluates modeling and instruction fidelity through verifiable tasks, and includes Agent- and Curation-LLMs, Sandbox, Dataset Sourcing, Task Design, Post-Process, Finalization, and Verifiable Ground Truth.
- The framework utilizes an automated pipeline to curate 6,300 tasks from Kaggle, ensuring reproducibility by executing LLM-generated code within a controlled sandbox environment.
- It enables verifiable evaluation and model alignment for LLMs through supervised fine-tuning and reinforcement learning with outcome-based rewards, addressing gaps in instruction adherence and training data scarcity.

---

[CUDA Agent: Large-Scale Agentic RL for High-Performance CUDA Kernel Generation](http://arxiv.org/abs/2602.24286)

- CUDA Agent: introduces a large-scale agentic reinforcement learning system for automatic CUDA kernel generation, with a data synthesis pipeline, a skill-integrated agent loop, and multi-stage RL training.
- The system utilizes a three-stage data collection pipeline to generate complex fused operator tasks and employs a robust reward schedule to guide the agent toward high-performance implementations.
- It achieves state-of-the-art performance on KernelBench by outperforming traditional compiler-driven optimization and proprietary LLs through iterative hardware-aware refinement.

---

[A Minimal Agent for Automated Theorem Proving](http://arxiv.org/abs/2602.24273)

- AxProverBase: introduces a modular agentic baseline for automated theorem proving that utilizes iterative proof refinement, a memory system for context management, and external search tools.
- The architecture includes proposer- and reviewer-agents, a programmatic compiler for feedback, and a memory module supporting self-reflection strategies.
- Experimental results demonstrate that this minimal scaffolding enables frontier LLMs to achieve competitive performance on mathematical benchmarks without requiring domain-specific fine-tuning.

---


[A Novel Hierarchical Multi-Agent System for Payments Using LLMs](http://arxiv.org/abs/2602.24068)

- HMASP (Hierarchical Multi-Agent System for Payments): introduces a modular four-level agentic architecture for end-to-end payment workflows, featuring conversational-, supervisor-, routing-, and process summary-agents.
- The system leverages LangGraph for orchestration and implements structured handoffs between hierarchical levels to facilitate secure and coordinated task execution across specialized domains.
- It incorporates deterministic function modules and an interrupt mechanism for human-in-the-loop data collection to mitigate hallucination risks in sensitive financial operations.

---


[Beyond the Click: A Framework for Inferring Cognitive Traces in Search](http://arxiv.org/abs/2602.24265)

- Multi-agent annotation framework: introduces a system for inferring unobservable cognitive states from behavioral search logs, which includes Analyst-, Critic-, and Judge-agents.
- The architecture utilizes Claude 3.5 Sonnet for sequential reasoning and GPT-4o for the synthesis of complex arguments to produce theoretically grounded cognitive traces.
- An active learning loop identifies high-disagreement instances for human expert validation, ensuring the quality of large-scale automated annotations.

---

[A model of tuberculosis progression using CompuCell3D](http://arxiv.org/abs/2602.24258)

- CC3D-TB (CompuCell3D-based Tuberculosis progression model): introduces a multiscale agent-based framework to simulate within-host tuberculosis dynamics, with Cellular Potts Model engine (stochastic lattice-based cell dynamics), PDE solvers (calculating reaction-diffusion of chemical fields), macrophage- and Mycobacterium tuberculosis-agents (representing immune and bacterial cells), Hamiltonian energy function (governing cell motility and adhesion), Chemical fields (modeling oxygen and chemokine distributions), and State transition logic (defining bacterial phenotypes and macrophage states).
- The system integrates discrete cellular interactions with continuous chemical fields to investigate how spatial organization and oxygen availability influence granuloma formation and infection outcomes.
- The model qualitatively aligns with the WHIDM framework while revealing quantitative differences in stochastic population bounds and spatial sensitivity during the first 200 hours of infection.

---

[UXSim: Towards a Hybrid User Search Simulation](http://arxiv.org/abs/2602.24241)

- UXSim (User Search Simulation): introduces a modular framework that synthesizes traditional rule-based simulators with an LLM-powered cognitive agent to model nuanced human-computer interaction. 
- The architecture employs an orchestration policy (Oris) to dynamically invoke either deterministic simulators or cognitive capabilities like planning, reflection, and wondering. 
- A declarative interface abstraction layer uses recipes and blueprints to translate raw HTML into structured representations, enabling grounded simulations on complex, real-world search interfaces. 

---

[Anansi: Scalable Characterization of Message-Based Job Scams](http://arxiv.org/abs/2602.24223)

- Anansi: introduces a scalable, end-to-end measurement pipeline designed to systematically engage with, analyze, and characterize job-based smishing scams in real-time.
- The system utilizes LLM-driven agents to simulate victim personas and interact with scammers across SMS, WhatsApp, and Telegram to extract behavioral and infrastructural signals.
- It integrates automated browser tools for task completion and wallet extraction, enabling the identification of coordinated scam syndicates and the estimation of millions in cryptocurrency losses.

---

[Controllable Reasoning Models Are Private Thinkers](http://arxiv.org/abs/2602.24210)

- Staged Decoding: introduces a decoding strategy that decouples reasoning and answer generation to maximize instruction following and privacy, with a reasoning model (backbone large language model), LoRA Adapter 1 (specialized weights for reasoning), LoRA Adapter 2 (specialized weights for answers), reasoning traces (intermediate thinking process generation), and final answers (concluding response generation).
- The framework utilizes a base reasoning model that first generates a reasoning trace using an adapter optimized for thinking-process constraints.
- It subsequently switches to a different adapter to generate the final response, effectively preventing the leakage of sensitive information found in the thinking process.

---

[BETTING UNDER COMMON BELIEFS: THE EFFECT OF PROBABILITY WEIGHTING](http://arxiv.org/abs/2602.24194)

- Mixed EU-RDU Economy: introduces a behavioral economic model where a single RDU Agent interacts with EU Agents under common beliefs, utilizing a Probability Weighting Function, a Social Planner, a Nudging Mechanism, an Education Cost, and a Utilitarian Welfare Function to demonstrate how nonlinear probability weighting endogenously generates speculative betting.
- The framework characterizes Pareto-optimal allocations using a representative agent formulation and quantile optimization to identify regions of full insurance versus endogenous uncertainty.
- A social planner component evaluates costly nudging through statistical education to restore full-insurance optimality by pushing the distorted weighting function toward linearity.

---

[Resilient Strategies for Stochastic Systems: How Much Does It Take to Break a Winning Strategy?](http://arxiv.org/abs/2602.24191)

- SGD (Stochastic Games with Disturbances): introduces a formal framework to quantify the resilience of strategies in stochastic environments by determining the minimum number of adversarial disturbances required to violate safety or reachability objectives.
- The approach utilizes state-space unfolding and Maximal End Component (MEC) quotients to compute transient and frequency-based breaking points under both expected and worst-case semantics.
- The research provides complexity results and algorithms for evaluating strategy robustness and synthesizing optimally resilient controllers in Markov decision processes and stochastic games.

---

[MT-PingEval: Evaluating Multi-Turn Collaboration with Private Information Games](http://arxiv.org/abs/2602.24188)

- MT-PingEval (Evaluating Multi-Turn Collaboration with Private Information Games): introduces a benchmark for assessing LLMs in multi-turn collaborative scenarios where agents must exchange private information to solve tasks, and includes player- and evaluator-agents.
- The framework employs isotoken evaluation to analyze how model performance scales when a fixed token budget is distributed across an increasing number of conversational turns.
- It incorporates linguistic analysis tools, including a centering theory-based coherence score and an LLM-as-a-judge autorater, to assess discourse quality and stylistic sycophancy.

---

[ArgLLM-App: An Interactive System for Argumentative Reasoning with Large Language Models](http://arxiv.org/abs/2602.24172)

- ArgLLM-App (Argumentative Large Language Model Application): introduces a web-based platform that implements Argumentative LLMs to facilitate explainable binary decision-making through graph-based argumentation, incorporating a web application, server, base LLM, QBAF, gradual semantics engine, human-in-the-loop interface, and PDF parser.
- The system enables users to visualize and contest reasoning by adjusting argument confidence scores or expanding the graph with new supporting and attacking evidence.
- It integrates retrieval-augmented generation by parsing external PDF documents to inform the LLM-driven argument generation process while using formal semantics to compute final claim confidence.

---

[Learning with a Budget: Identifying the Best Arm with Resource Constraints](http://arxiv.org/abs/2602.24146)

- SH-RR (Successive Halving with Resource Rationing): introduces a resource-aware multi-armed bandit algorithm for identifying the optimal arm under heterogeneous and stochastic resource constraints across multiple budget types, with all SH-RR (Successive Halving with Resource Rationing) (core algorithm for best arm identification), Surviving Arm Set (collection of candidate optimal arms), Resource Rationing (allocating specific budget per phase), Round-robin Arm Pulling (uniform exploration of candidate arms), Empirical Mean Estimator (calculating average rewards for selection), Arm Elimination (removing sub-optimal arms each phase), and Resource Tracking (monitoring cumulative consumption of multiple resources)-components.
- The framework utilizes a multi-phase elimination strategy that rations resources to ensure sufficient exploration while maintaining feasibility against cumulative consumption limits.
- The research establishes near-optimal non-asymptotic convergence rates using a novel effective consumption measure that unifies deterministic and stochastic resource consumption settings.

---

[CoME: Empowering Channel-of-Mobile-Experts with Informative Hybrid-Capabilities Reasoning](http://arxiv.org/abs/2602.24142)

- CoME (Channel-of-Mobile-Experts): introduces a mobile agent architecture that decouples hybrid-capabilities reasoning into specialized stages, with Screen Summary Expert (specialized FFN for screen perception), Subtask Plan Expert (specialized FFN for task decomposition), Action Decision Expert (specialized FFN for high-level decisions), Action Function Expert (specialized FFN for low-level execution), Shared Self-Attention Module (shared module for contextual representation), Channel Router (module for stage-aligned expert selection), Reward Models (models estimating reasoning step contribution), and LM Head (final token generation layer).
- The framework employs a progressive training strategy consisting of Expert-FT for capability enhancement, Router-FT for stage alignment, and CoT-FT for balanced optimization across reasoning stages.
- To mitigate error propagation, the paper proposes InfoGain-Driven DPO, which utilizes reward models to quantify the information gain of intermediate reasoning steps and reinforce informative trajectories.

---

[Planning from Observation and Interaction](http://arxiv.org/abs/2602.24121)

- MPAIL2 (Model Predictive Adversarial Imitation Learning 2): introduces an off-policy inverse reinforcement learning framework for world modeling and planning from observation and interaction alone, utilizing an encoder, dynamics model, inferred reward, value function, multi-step policy, planner, and experience replay buffer.

- The system enables real-world robot learning of visual manipulation tasks from scratch without requiring prior modeling, hand-designed rewards, or demonstrator action labels.

- It leverages a multi-step policy to seed a Model Predictive Path Integral planner, significantly improving interaction efficiency and enabling online transfer learning between tasks in the real world.


---

[Agentic AI-RAN: Enabling Intent-Driven, Explainable and Self-Evolving Open RAN Intelligence](http://arxiv.org/abs/2602.24115)

- Agentic AI-RAN: introduces a goal-driven control framework for Open Radio Access Networks that integrates LLMs and agentic primitives across Non-RT, Near-RT, and RT layers, including intent-to-goal- and resource summarization-agents.
- The architecture utilizes a Plan-Act-Observe-Reflect loop where a Non-RT LLM coordinates long-horizon reasoning while Near-RT agents execute gated skill sequences through standard O-RAN interfaces.
- Simulation results demonstrate an 8.83% reduction in resource usage and improved SLA satisfaction by replacing monolithic black-box controllers with modular, auditable agentic primitives.

---

[Artificial Agency Program: Curiosity, compression, and communication in agents](http://arxiv.org/abs/2602.24100)

- AAP (Artificial Agency Program): introduces a research agenda for building resource-bounded agents driven by curiosity-as-learning-progress, with Multimodal Backbone (pretrained perception and world-model), Meta-Controller (dynamic budget allocator), Unified Interface (interleaved sensing and thinking), Modality-Agnostic Token Taxonomy (input, private, output roles), Environment State (hidden physical world variables), and Agent Internal State (integrated memory and policy).
- The framework includes perception-, world-model-, and meta-controller-components to dynamically allocate limited budgets across sensing, acting, and thinking.
- It utilizes a modality-agnostic token taxonomy to unify incoming observations, private deliberation, and public outputs through a single information-theoretic objective.

---

[Bi-level RL-Heuristic Optimization for Real-world Winter Road Maintenance](http://arxiv.org/abs/2602.24097)

- Bi-level RL-Heuristic Optimization: introduces a scalable framework for winter road maintenance that integrates a reinforcement learning agent for strategic depot assignment with a constraint-aware heuristic for tactical vehicle routing.
- The architecture employs a Proximal Policy Optimization agent to partition large-scale road networks into clusters, while a lower-level nearest-neighbor heuristic generates feasible routes subject to capacity and time constraints.
- The framework incorporates a feedback loop where routing performance metrics, including makespan and carbon emissions, serve as rewards to iteratively refine the upper-level assignment policy.

---

[Adaptive Correlation-Weighted Intrinsic Rewards for Reinforcement Learning](http://arxiv.org/abs/2602.24081)

- ACWI (Adaptive Correlation-Weighted Intrinsic): introduces an adaptive reinforcement learning framework that dynamically modulates intrinsic rewards using a state-dependent Beta Network to improve exploration in sparse-reward environments.
- The system optimizes the scaling factor through a first-order correlation objective that aligns weighted intrinsic signals with discounted future extrinsic returns without requiring second-order meta-learning.
- Evaluation on complex grid-world tasks demonstrates that the architecture effectively distinguishes task-relevant exploration from noise, leading to enhanced sample efficiency and training stability.

---

[HUMAN OR MACHINE? A PRELIMINARY TURING TEST FOR SPEECH-TO-SPEECH INTERACTION](http://arxiv.org/abs/2602.24080)

- Interpretable AI Judge: introduces a diagnostic evaluation framework for Speech-to-Speech (S2S) systems, with an Audio-Language Model (ALM), an Ordinal Discretization Layer (ODL), and a regularized linear classifier.
- The model leverages a fine-grained taxonomy of 18 human-likeness dimensions to provide transparent discrimination between human and machine-generated speech.
- The study utilizes a gamified platform to collect human judgments on dialogues from nine state-of-the-art S2S systems, revealing significant gaps in paralinguistic and emotional expressivity.

---

[Sharing is caring: data sharing in multi-agent supply chains](http://arxiv.org/abs/2602.24074)

- Multi-agent supply chain environment with information sharing: introduces a two-echelon supply chain model where a factory agent can share truthful, false, or no inventory data with a retailer agent to optimize system performance, with factory agent, retailer agent, two-echelon supply chain environment, communication module, reward shaping module, and SAC algorithm.
- The framework utilizes SAC agents within a Gymnasium-based environment to evaluate how different communication strategies and collaborative reward shaping impact inventory management across high and low demand scenarios, and includes factory- and retailer-agents.
- Experimental results demonstrate that while lying can marginally benefit the factory in high-demand settings, truthful data sharing improves overall system rewards in low-demand environments.

---

[Portfolio Reinforcement Learning with Scenario-Context Rollout](http://arxiv.org/abs/2602.24037)

- SCR (Scenario-Context Rollout): introduces a macro-conditioned reinforcement learning framework for portfolio rebalancing that generates plausible multivariate return scenarios under stress events to improve policy resilience.
- The architecture incorporates an augmented critic bootstrap target that interpolates between realized historical continuations and counterfactual continuations to mitigate reward-transition mismatches during temporal-difference learning.
- A ShockLedger component maintains a catalog of macro signatures to facilitate leak-safe scenario retrieval and provide a risk-budgeting regime context for gating aggressive exposures during market downturns.

---

[Foundation World Models for Agents that Learn, Verify, and Adapt Reliably Beyond Static Environments](http://arxiv.org/abs/2602.23997)

- Foundation World Models: introduces a framework for reliable autonomy that unifies reinforcement learning with formal methods by integrating predictive world models, symbolic abstractions, and automated verifiers.
- The architecture utilizes LLMs as specification refiners to translate high-level natural language instructions into temporal-logic constraints and verifiable programs for model checkers.
- A continuous calibration loop between the learned world model and the environment allows the verifier to quantify prediction reliability and provide adaptive control signals to correct policy drift.

---

[Pessimistic Auxiliary Policy for Offline Reinforcement Learning](http://arxiv.org/abs/2602.23974)

- Pessimistic Auxiliary Policy: introduces a strategy to sample reliable actions by maximizing the lower confidence bound of the Q-function, utilizing ensemble Q-networks, epistemic uncertainty estimation, and a first-order Taylor expansion.
- The framework mitigates overestimation and error accumulation in offline reinforcement learning by constraining the auxiliary policy to the neighborhood of the learned policy while favoring regions of low uncertainty.
- Extensive experiments on D4RL and NeoRL-2 benchmarks demonstrate that integrating this auxiliary policy significantly improves the performance and stability of existing algorithms like TD3BC and Diffusion-QL.

---

[HotelQuEST: Balancing Quality and Efficiency in Agentic Search](http://arxiv.org/abs/2602.23949)

- HotelQuEST (Hotel Quality & Efficiency Search Testbed): introduces a benchmark for agentic search evaluation that jointly measures answer quality and computational efficiency, with Plan, Retrieve descriptions, Retrieve reviews, Web search, Write to memory, Answer, and Memory state components.
- The framework evaluates LLMs on 214 hotel search queries ranging from simple lookups to complex multi-hop requests with underspecified user preferences and explicit clarifications.
- Empirical analysis reveals that while LLM-based agents achieve high accuracy, they frequently over-invest computation for marginal quality gains due to redundant tool calls and lack of adaptive stopping criteria.

---

[Enhancing Vision-Language Navigation with Multimodal Event Knowledge from Real-World Indoor Tour Videos](http://arxiv.org/abs/2602.23937)

- STE-VLN (Spatio-Temporal Event-enhanced Vision-Language Navigation): introduces, an event-centric knowledge enhancement strategy for vision-language navigation, with YE-KG (multimodal spatiotemporal knowledge graph), LLaVA-NeXT-Video (video-to-text event parser), GPT-4 (textual description refiner), Coarse-to-Fine Hierarchical Retrieval (multistage knowledge retrieval mechanism), ASTFF (adaptive multimodal feature fusion module), Transformer-based VLN Planner (sequential action prediction model), ViT Vision Encoder (visual feature extraction component), and Text Encoder (instruction embedding component), where the system includes video-parsing- and text-refining-LLMs.
- The framework utilizes LLaVA-NeXT-Video and GPT-4 to parse unstructured video streams into structured semantic-action-effect events, serving as explicit episodic memory for the agent.
- A coarse-to-fine hierarchical retrieval mechanism and an adaptive spatio-temporal feature fusion module dynamically align instructions with visual foresight and historical priors to improve long-horizon reasoning.

---

[Experience-Guided Self-Adaptive Cascaded Agents for Breast Cancer Screening and Diagnosis with Reduced Biopsy Referrals](http://arxiv.org/abs/2602.23899)

- BUSD-Agent (Breast Ultrasound Screening and Diagnosis Agent): introduces a cascaded multi-agent framework that mimics clinical triage by escalating suspicious breast ultrasound cases from a lightweight screening agent to a specialized diagnostic agent.
- The framework employs a biopsy-grounded memory bank to store historical decision trajectories, enabling retrieval-conditioned in-context adaptation to dynamically adjust model trust and escalation thresholds without parameter updates.
- The architecture includes an LLM-based orchestration layer and a fine-tuned Vision-Language Model (VLM) to reconcile outputs from a diverse toolset comprising classification ensembles, object detection, and segmentation modules.

---

[TSC: Topology-Conditioned Stackelberg Coordination for Multi-Agent Reinforcement Learning in Interactive Driving](http://arxiv.org/abs/2602.23896)

- TSC (Topology-Conditioned Stackelberg Coordination): introduces a multi-agent reinforcement learning framework for decentralized interactive driving that extracts a time-varying directed priority graph from trajectory weaving cues to define local leader-follower dependencies, with Ego Encoder, Neighbor Encoder, TopoDecoder, TopoHead, NodeHead, TopK Filter, TopoAttention, EgoDecoder, PolicyHead, PredictHead, ValueHead, and Replay Buffer.
- The TSC-Net architecture utilizes a topological priority inference branch to sparsify dense interactions and a Stackelberg-conditioned critic that conditions value learning on predicted leader actions, including priority inference-, actor-, and critic-branches.
- This approach enables stable, communication-free coordination in dense traffic by factorizing complex interactions into graph-local subgames and approximating local best responses.

---

[AoE: Always-on Egocentric Human Video Collection for Embodied AI](http://arxiv.org/abs/2602.23893)

- AoE (Always-on Egocentric): introduces a scalable framework for collecting egocentric human interaction data with a neck-mounted smartphone, mobile app, edge nodes, cloud environment, data processing pipeline pool, Qwen3-VL, world model, and multi-modal object storage.
- The system employs on-device vision models for real-time action triggering and a cloud-based pipeline that includes segmentation- and augmentation-LLMs.
- It utilizes a generative data augmentation pipeline to transform raw human videos into simulation assets, enhancing the real-world generalization of humanoid robotic manipulation policies.

---

[RF-Agent: Automated Reward Function Design via Language Agent Tree Search](http://arxiv.org/abs/2602.23876)

- RF-Agent (Automated Reward Function Design via Language Agent Tree Search): introduces a framework that frames reward function design as a sequential decision-making process to automate dense reward generation for low-level control, with LLM-based reward designer, Monte Carlo Tree Search, tree-structured memory, action expansion module, simulation & policy training, LLM-based self-verifier, and LLM-based thought aligner.
- The system includes LLM-based reward designer-, self-verifier-, and thought aligner-components to manage the search space through heuristic actions like mutation, crossover, and path reasoning.
- Experimental results across 17 diverse tasks in IsaacGym and Bi-DexHands demonstrate that the framework significantly outperforms existing greedy and evolutionary baselines in search efficiency and task performance.

---

[SWE-rebench V2: Language-Agnostic SWE Task Collection at Scale](http://arxiv.org/abs/2602.23866)

- SWE-rebench V2: introduces a language-agnostic automated pipeline for harvesting executable real-world software engineering tasks and constructing reinforcement learning training environments at scale, with Interactive Setup Agent, LLM Judge Ensemble, Execution-based Validator, Metadata Enrichment Agent, Problem Statement Generator, and Dockerized Environments.
- The system includes interactive setup-, ensemble judge-, and metadata enrichment-agents to automate environment configuration, issue clarity filtering, and diagnostic tagging across diverse programming languages.
- The resulting dataset contains over 32,000 tasks spanning 20 languages and 3,600 repositories, featuring fine-grained metadata to distinguish model limitations from environment pathologies.

---

[RUMAD: Reinforcement-Unifying Multi-Agent Debate](http://arxiv.org/abs/2602.23864)

- RUMAD (Reinforcement-Unifying Multi-Agent Debate): introduces a framework that formulates dynamic communication topology control in multi-agent debate as a reinforcement learning problem, utilizing an RL controller to optimize interactions between LLaMA-3.1-8B-Instruct, ChatGLM-4-9B, and Deepseek-Math-7B-Instruct agents.
- The system employs a content-agnostic observation scheme based on semantic similarity and answer agreement to preserve debate neutrality while a multi-objective reward function balances reasoning accuracy with token efficiency.
- A dual-threshold mechanism enables fine-grained control over agent activation and information visibility, allowing the framework to achieve significant token cost reductions while maintaining or improving performance across diverse benchmarks.

---

[HYCO: A FORMALISM FOR HYBRID-COOPERATIVE PDE MODELLING](http://arxiv.org/abs/2602.23859)

- HYCO (Hybrid-Cooperative Learning): introduces a hybrid modeling framework that integrates physics-based and data-driven models through mutual regularization, which includes physical model (differential equation solver with unknown parameters), synthetic model (neural network approximating system from data), interaction loss (discrepancy measure between model predictions), ghost points (stochastic evaluation points for interaction loss), and alternating optimization (game-theoretic parameter update scheme).
- The framework treats both components as co-trained agents in a two-player game seeking a Nash equilibrium, avoiding the hierarchical constraints typical of physics-informed neural networks.
- It utilizes a lightweight interaction-evaluation strategy based on randomly sampled "ghost points" to decouple the interaction loss from the physical mesh, enhancing robustness to sparse and noisy data.

---

[A Distributed Semismooth Newton Based Augmented Lagrangian Method for Distributed Optimization](http://arxiv.org/abs/2602.23854)

- DSSNAL (Distributed Semismooth Newton based Augmented Lagrangian): introduces a distributed optimization framework for solving nonsmooth problems over networks by reformulating the objective with consensus constraints and applying an outer augmented Lagrangian loop, with ALM (outer loop for consensus constraints), DiSSN (inner solver for nonsmooth subproblems), DAPG (initialization and Newton direction computation), Gossip Matrix (structure for distributed information exchange), Local Agents (network nodes with private objectives), and Communication Network (undirected graph for agent interaction).
- The framework utilizes a distributed accelerated proximal gradient method to warm-start the Newton phase and compute directions without exchanging full Hessian matrices between neighboring agents.
- Numerical experiments on Huber regression and support vector classification demonstrate that the approach achieves superlinear convergence and superior computational efficiency compared to state-of-the-art first-order distributed algorithms.

---

[CLFEC: A New Task for Unified Linguistic and Factual Error Correction in paragraph-level Chinese Professional Writing](http://arxiv.org/abs/2602.23845)

- CLFEC (Chinese Linguistic & Factual Error Correction): introduces a unified task and dataset for paragraph-level Chinese professional writing that integrates linguistic and factual error repair using RAG and agentic workflows.
- The architecture incorporates a planning-and-execution agent utilizing search_tool for evidence grounding, todo_write for task tracking, and verify_tool for hallucination mitigation.
- Experimental results indicate that unified context handling in U-RAG and iterative reasoning in agentic workflows significantly improve factual accuracy compared to decoupled or prompt-only baselines across specialized domains.

---

[See, Act, Adapt: Active Perception for Unsupervised Cross-Domain Visual Adaptation via Personalized VLM-Guided Agent](http://arxiv.org/abs/2602.23806)

- Sea² (See, Act, Adapt): introduces an active perception framework that adapts pre-trained visual models to novel embodied environments by training a Vision-Language Model (VLM) as a pose controller to seek informative viewpoints.
- The system utilizes a two-stage training pipeline consisting of supervised fine-tuning on heuristic trajectories followed by unsupervised reinforcement learning using Group Relative Policy Optimization (GRPO).
- It leverages scalar feedback from frozen perception modules—including visual grounding, segmentation, and 3D box estimation—to construct rewards without requiring downstream labels or model retraining.

---

[TradeFM: A Generative Foundation Model for Trade-flow and Market Microstructure](http://arxiv.org/abs/2602.23784)

- TradeFM (A Generative Foundation Model for Trade-flow and Market Microstructure): introduces a 524M-parameter decoder-only Transformer pre-trained on 10 billion tokens from over 9,000 US equities to model high-frequency order flow dynamics from partial observations, with all Tabular Input Embedding, Decoder-only Transformer, Market Simulator, Feedback Loop, and Tokenization Scheme components.
- The framework utilizes scale-invariant feature engineering and a mixed-base tokenization scheme to transform multi-modal trade events into a unified discrete sequence, enabling zero-shot generalization to geographically out-of-distribution markets.
- Integration with a deterministic market simulator allows for high-fidelity closed-loop rollouts that reproduce canonical stylized facts of financial returns, such as heavy tails and volatility clustering.

---

[Diffusion Probe: Generated Image Result Prediction Using CNN Probes](http://arxiv.org/abs/2602.23783)

- Diffusion Probe: introduces a framework that leverages internal cross-attention maps from early denoising steps to predict final image quality, with Cross-Attention Maps, TimeStep Embedding, Diffusion Prober, DownBlocks, and OutputLayer.
- The system utilizes a CNN-based predictor to map statistical properties of nascent attention distributions to aesthetic and semantic scores, enabling early-stage quality assessment without full image synthesis.
- It facilitates downstream tasks such as automated prompt optimization via LLMs, seed selection, and accelerated reinforcement learning training by preemptively discarding low-potential generation paths.

---

[OPTIAGENT: A Physics-Driven Agentic Framework for Automated Optical Design](http://arxiv.org/abs/2602.23761)

- OPTIAGENT (A Physics-Driven Agentic Framework for Automated Optical Design): introduces a physics-driven agentic framework that reformulates automated optical lens design as a goal-oriented decision-making process using LLMs aligned via reinforcement learning.
- The system incorporates a hierarchical Optical Lexicographic Reward to enforce structural integrity, physical feasibility, and light-manipulation accuracy through a differentiable simulator.
- The framework leverages the OptiDesignQA dataset for domain-specific knowledge injection and integrates with Zemax for end-to-end fine-tuning and commercial-grade precision refinement.

---

[U-Mind: A Unified Framework for Real-Time Multimodal Interaction with Audiovisual Generation](http://arxiv.org/abs/2602.23739)

- U-Mind (Unified Framework for Real-Time Multimodal Interaction with Audiovisual Generation): introduces a unified system for real-time multimodal dialogue that jointly models language, speech, motion, and video synthesis within a single interactive loop, with User Input, Text and Speech Tokenizers, U-mind Backbone, Unified Embedding Space, CoT Planner, Text, Speech, and Motion Decoders, and Video Renderer.
- The architecture employs a text-first decoding pipeline where internal chain-of-thought planning precedes the generation of temporally synchronized multimodal outputs.
- A segment-wise alignment strategy and rehearsal-driven pre-training are utilized to ensure cross-modal synchronization while preventing the degradation of high-level reasoning capabilities.

---

[FROM STATIC BENCHMARKS TO DYNAMIC PROTOCOL: AGENT-CENTRIC TEXT ANOMALY DETECTION FOR EVALUATING LARGE LANGUAGE MODEL REASONING](http://arxiv.org/abs/2602.23729)

- ATAD (Agent-Centric Text Anomaly Detection): introduces a dynamic benchmarking protocol that replaces static datasets with a three-agent system comprising teacher-, orchestrator-, and student-agents to iteratively generate and validate reasoning-focused problems through an adaptive difficulty scaling loop and a validation mechanism.
- The framework utilizes a competitive loop where the teacher increases problem difficulty upon student success, while the orchestrator ensures logical coherence and guards against adversarial or ill-posed samples.
- By focusing on text anomaly detection across seven task types, the protocol automatically scales to the capabilities of emerging LLMs and identifies specific reasoning failures that static benchmarks often miss.

---

[From Flat Logs to Causal Graphs: Hierarchical Failure Attribution for LLM-based Multi-Agent Systems](http://arxiv.org/abs/2602.23701)

- CHIEF (Causal HIErarchical Failure attribution framework): introduces a diagnostic framework that reconstructs flat multi-agent logs into a hierarchical causal graph, including task decomposition-, parsing-, oracle synthesis-, and semantic evaluation-LLMs.
- The architecture utilizes synthesized virtual oracles to guide a top-down backtracking process, efficiently narrowing the search space from coarse subtasks to fine-grained agent steps.
- It employs a progressive counterfactual screening strategy to distinguish root causes from propagated symptoms by analyzing planning-control and data-flow dependencies across the execution trajectory.

---

[ODAR: Principled Adaptive Routing for LLM Reasoning via Active Inference](http://arxiv.org/abs/2602.23681)

- ODAR (Open-Domain Adaptive Reasoner): introduces an adaptive routing framework that optimizes the accuracy-efficiency trade-off by dynamically allocating compute resources based on predicted query difficulty.
- The system utilizes a Difficulty Estimator grounded in active inference to route queries through three distinct paths—Simple, Medium, or Hard—employing specialized hypothesis-generation and verification-agents.
- A principled fusion module based on the Free Energy Principle selects final answers by minimizing a variational objective that balances log-likelihood with epistemic uncertainty.

---

[SGAgent: Suggestion-Guided LLM-Based Multi-Agent Framework for Repository-Level Software Repair](http://arxiv.org/abs/2602.23647)

- SGAgent (Suggestion-Guided multi-Agent framework): introduces a repository-level software repair system that bridges the reasoning gap between localization and fixing through a locate-suggest-fix paradigm, featuring localizer-, suggester-, and fixer-agents.
- The architecture leverages a structured Knowledge Graph and a 14-tool retrieval toolkit to provide agents with precise, repository-wide semantic and structural context during the debugging process.
- It employs a dedicated summarization LLM to maintain conversational coherence via dynamic memory compression and utilizes a multi-stage validation pipeline involving regression and reproduction tests to ensure patch reliability.

---

[Toward E2E Intelligence in 6G Networks: An AI Agent-Based RAN-CN Converged Intelligence Framework](http://arxiv.org/abs/2602.23623)

- AI agent-based RAN-CN converged intelligence framework: introduces a unified control architecture that integrates LLMs with the ReAct paradigm to enable iterative reasoning and cross-domain policy generation for 6G networks.
- The framework utilizes a dual-memory system and a centralized monitoring database to synthesize adaptive control policies through a closed-loop thought-action-observation process without requiring model retraining.
- By leveraging the Model Context Protocol (MCP) and a policy orchestrator, the agent coordinates actions across Radio Access Network and Core Network domains to optimize end-to-end slice performance and service level agreement satisfaction.

---

[Multi-Agent Causal Reasoning for Suicide Ideation Detection Through Online Conversations](http://arxiv.org/abs/2602.23577)

- MACR (Multi-Agent Causal Reasoning): introduces a framework for suicide risk prediction that utilizes a Reasoning Agent to generate counterfactual user interactions and a Bias-aware Decision-making Agent to mitigate unobserved confounding biases.
- The Reasoning Agent employs a collaborative three-round debate process that includes psychological analyst-, critical thinker-, empiricist-, and synthesizer-agents to simulate diverse psychological responses grounded in cognitive appraisal theory.
- The framework implements a front-door adjustment strategy by treating generated reasoning nodes as a mediator variable, effectively addressing hidden influences like user conformity and imitative behavior in online communities.

---

[ParamMem: Augmenting Language Agents with Parametric Reflective Memory](http://arxiv.org/abs/2602.23320)

- ParamAgent: introduces a reflection-based agent framework that integrates a parametric memory module (ParamMem) to encode cross-sample reflection patterns into model parameters, increasing reflective diversity.
- The framework includes actor- and reflection-LLMs that utilize episodic, cross-sample, and parametric memory sources to iteratively refine solutions based on environment feedback.
- ParamMem is sample-efficient, supports weak-to-strong transfer across model scales, and enables self-improvement without reliance on stronger external models.

---

[Conformalized Neural Networks for Federated Uncertainty Quantification under Dual Heterogeneity](http://arxiv.org/abs/2602.23296)

- FedWQ-CP (Federated Weighted Quantile Conformal Prediction): introduces a one-shot federated calibration method that aggregates locally computed conformal quantiles via weighted averaging to address joint data and model heterogeneity.
- The system utilizes local agents to compute non-conformity scores and quantile thresholds which are then transmitted alongside calibration sample sizes to a central server for global aggregation.
- This approach ensures reliable agent-level and global coverage while minimizing communication overhead and producing efficient prediction sets across diverse neural network architectures.

---

#### 26th February 2026

[Rudder: Steering Prefetching in Distributed GNN Training using LLM Agents](http://arxiv.org/abs/2602.23556)

- Rudder: introduces an adaptive prefetching framework for distributed Graph Neural Network (GNN) training that utilizes LLM agents to autonomously determine optimal buffer replacement timing based on real-time execution metrics.
- The system employs an agentic workflow consisting of a metrics collector, context builder, and a decision-making LLM to minimize communication overheads by dynamically managing a local persistent buffer.
- By leveraging In-Context Learning (ICL), the framework adapts to varying graph structures and training configurations without requiring offline training, achieving significant performance improvements over static prefetching methods.

---

[IDP Accelerator: Agentic Document Intelligence from Extraction to Compliance Validation](http://arxiv.org/abs/2602.23481)

- IDP Accelerator (Intelligent Document Processing Accelerator): introduces an open-source, agentic framework for end-to-end document intelligence, featuring a multimodal DocSplit classifier and LLM-driven extraction-, analytics-, and rule validation-modules.
- The architecture utilizes a serverless orchestration layer to manage modular processing patterns, integrating multimodal LLMs for semantic extraction and automated rule-based validation.
- The framework includes a human-in-the-loop system for verifying low-confidence results and supports the Model Context Protocol to enable secure, sandboxed code execution for data analytics.

---

[Optimization of Edge Directions and Weights for Mixed Guidance Graphs in Lifelong Multi-Agent Path Finding](http://arxiv.org/abs/2602.23468)

- MGGO (Mixed Guidance Graph Optimization): introduces a framework for optimizing both edge weights and directions in lifelong multi-agent pathfinding to reduce traffic congestion, with Update Model, CMA-MAE, ERS, LMAPF Simulator, and Archives.
- The approach leverages Quality Diversity algorithms to train a neural network that generates guidance graphs based on precomputed traffic patterns.
- A specialized Edge Reversal Search algorithm ensures strong connectivity in the resulting directed graphs, improving agent throughput under rotational motion constraints.

---

[An Aggregated SIR Model for Spatial Epidemic Propagation](http://arxiv.org/abs/2602.23449)

- Aggregated SIR Model: introduces an ordinary differential equation framework for spatial epidemic propagation, with pS (population in unreached districts), pI (population in active districts), pR (population in recovered districts), S (susceptible individuals in active districts), I (infected individuals in active districts), R (recovered individuals in active districts), and Saturation Function (smooth spatial advance driver).
- The model incorporates parameters for mobility and connectivity to simulate spatial diffusion and reproduce qualitative features like incidence plateaus observed in aggregated data.
- Analytical derivations confirm that the basic reproduction number remains consistent with the standard SIR model while providing a parsimonious alternative to agent-based simulations.

---

[Learning dynamics from online-offline systems of LLM agents](http://arxiv.org/abs/2602.23437)

- LLM Agent Network Dynamics Modeling: introduces a mathematical framework to analyze information propagation through LLM agents using an agent-based stochastic network model and a mean-field differential equation model.
- The framework incorporates LLM agents defined by Big Five personality traits that interact within k-regular random networks to decide whether to share news events sourced from the ACLED database.
- Findings indicate that the overall system dynamics are accurately described by a Susceptible-Infected (SI) model where specific traits like openness and extraversion act as primary drivers of transmission.

---

[A Dataset is Worth 1 MB](http://arxiv.org/abs/2602.23358)

- PLADA (Pseudo-Labels as Data): introduces a communication-efficient dataset serving method that replaces image transmission with hard pseudo-labels mapped to a pre-loaded reference dataset, with Base Server, Teacher Model, Energy-based Pruning, Safety-Net Filter, and Student Model components.
- The framework employs energy-based out-of-distribution scores to filter the reference set for semantic relevance, enabling task transfer with payloads under 1 MB while maintaining high classification accuracy.
- A class-specific safety-net mechanism ensures structural diversity in the distilled dataset, preventing class collapse and facilitating effective local model training across heterogeneous client architectures.

---

[LEARNING CONTACT POLICIES FOR SEIR EPIDEMICS ON NETWORKS: A MEAN-FIELD GAME APPROACH](http://arxiv.org/abs/2602.23344)

- SEIR-MFG (Mean-Field Game for SEIR dynamics): introduces a mathematical framework for modeling epidemic spread on heterogeneous networks where agents choose contact reduction policies to balance infection risk against isolation costs.
- The architecture integrates a forward Kolmogorov system for population-level transitions with a backward Hamilton-Jacobi-Bellman system to derive optimal individual behavioral responses.
- The research identifies a strategic delay mechanism caused by the latent period, which attenuates behavioral responses and increases the final outbreak size compared to simpler SIR models.

---

[Toward Expert Investment Teams: A Multi-Agent LLM System with Fine-Grained Trading Tasks](http://arxiv.org/abs/2602.23330)

- Multi-agent LLM trading framework: introduces a hierarchical decision-making system that decomposes complex investment workflows into fine-grained tasks, with input data, technical-, quantitative fundamental-, qualitative fundamental-, news-, sector-, macro-, and PM-agents, and portfolio construction.
- The architecture employs a bottom-up aggregation process where analyst outputs are refined by sector-level adjustments and macro-environmental assessments to inform a central portfolio manager agent.
- Evaluation using Japanese equity data shows that granular task structuring improves Sharpe ratios and semantic alignment between analytical outputs and final investment decisions.

---

[Modeling Large-Scale Adversarial Swarm Engagements using Optimal Control](http://arxiv.org/abs/2602.23323)

- Adversarial Swarming Optimal Control Framework: introduces a modeling and control architecture for large-scale autonomous systems that integrates probabilistic agent destruction with spatial movement to optimize swarm trajectories.
- The framework evaluates three deterministic numerical approximations—decoupled, weighted, and threshold models—to address the high-dimensional stochasticity of adversarial engagements.
- Numerical results indicate that coupling survival probabilities to physical interactions is critical for accurate swarm behavior modeling and effective defense of high-value assets.

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

