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





#### 7th January 2026

[MindWatcher: Toward Smarter Multimodal Tool-Integrated Reasoning](http://arxiv.org/abs/2512.23412)

- MindWatcher: introduces a high-performance Tool-Integrated Reasoning (TIR) agent integrating Interleaved Thinking and Multimodal CoT, trained via Continuous RL using Step-wise Normalized GRPO and a Hybrid Reward function.
- The agent is equipped with a comprehensive Tool Platform featuring five multimodal tools, including Region Cropping/Zooming and Object Grounding & Visual Search, supported by a large-scale MWRD (Local Visual Retrieval Corpus).
- MindWatcher achieves state-of-the-art performance on the new MWE-Bench (Multimodal tool-use benchmark) and demonstrates superior tool invocation capabilities, matching or exceeding larger LLMs.

---

[Monadic Context Engineering](http://arxiv.org/abs/2512.22431)

- MCE (Monadic Context Engineering): introduces a novel architectural paradigm leveraging algebraic structures (Functors, Applicative, Monads) to provide a formal foundation for robust, composable, and efficient AI agent design.
- The core AgentMonad is constructed using a Monad Transformer Stack (IO Monad, EitherT Transformer, StateT Transformer) to intrinsically manage cross-cutting concerns like side effects, short-circuiting error handling, and state propagation.
- The framework extends to AsyncAgentMonad for I/O parallelism via the Applicative interface and supports Meta-Agents for generative orchestration and dynamic management of sub-agent workflows.

---

[Embedding Autonomous Agents in Resource-Constrained Robotic Platforms](http://arxiv.org/abs/2601.04191)

- Embedded-BDI framework (AgentSpeak): introduces embedding an autonomous BDI agent onto a resource-constrained Pololu 3pi+ 2040 Robot platform, utilizing a Translation Engine (AgentSpeak to C++ code), Runtime Library (Executes reasoning cycle), and Hardware-Dependent Code (Perception and action API).
- The BDI Agent manages navigation and decision-making by reasoning over beliefs formed by Reflectance Sensors and executing intentions via DC Motors to solve a line-following maze using the left-hand rule.
- Experimental results confirm the feasibility of real-time autonomy on constrained hardware, demonstrating that the agent's reasoning cycle (belief update and plan selection) is computationally efficient, executing in less than one millisecond.

---

[ComfySearch: Autonomous Exploration and Reasoning for ComfyUI Workflows](http://arxiv.org/abs/2601.04060)

- ComfySearch: introduces an agentic framework that formulates ComfyUI workflow generation as a Markov Decision Process (MDP) using a Policy/Agent/LLM, State-Aware Validation, In-Place Repair, Entropy-Adaptive Branching, Tool-Mediated Transition, GRPO Optimization, and Hierarchical Terminal Reward.
- The framework enforces online structural correctness via state-aware validation and in-place repair (C1), and navigates long-horizon uncertainty by employing entropy-adaptive branching (C2) to selectively explore high-ambiguity decisions.
- The policy is trained using Supervised Warmup followed by tool-feedback RL with Group Relative Policy Optimization (GRPO), achieving high executability and task resolution rates on complex ComfyUI tasks.

---

[Staged Voxel-Level Deep Reinforcement Learning for 3D Medical Image Segmentation with Noisy Annotations](http://arxiv.org/abs/2601.03875)

- SVL-DRL (Staged Voxel-Level Deep Reinforcement Learning): introduces an end-to-end staged voxel-level deep reinforcement learning framework for robust 3D medical image segmentation under noisy annotations, utilizing a Feature Extraction Module, Segmentation Network, Value Network, Policy Network, vA3C Module, Staged Training, Composite Reward Function, and Action Space.
- The framework employs a dynamic iterative update strategy and a voxel-level Asynchronous Advantage Actor-Critic (vA3C) module, treating each voxel as an autonomous agent to dynamically refine its state representation and mitigate erroneous labels.
- Training is structured into Warmup, Transition, and Full RL stages, incorporating a novel action space and a composite reward function combining Dice value and a spatial continuity metric to enhance segmentation accuracy.

---

[Atlas: Orchestrating Heterogeneous Models and Tools for Multi-Domain Complex Reasoning](http://arxiv.org/abs/2601.03872)

- ATLAS (Adaptive Tool-LLM Alignment and Synergistic Invocation): introduces a dual-path framework for dynamic model-tool alignment, combining training-free cluster-based routing for domain-specific efficiency and RL-driven multi-step routing for open-domain adaptability.
- The framework jointly optimizes heterogeneous LLM and external tool combinations within a Cartesian product search space, addressing the limitations of fixed logic and isolated optimization in prior routing methods.
- The RL component uses a composite reward structure, including format, outcome, and model selection rewards, enabling the agent to learn transferable routing principles for superior generalization across diverse complex reasoning tasks.

---

[From Laboratory to Real-World Applications: Benchmarking Agentic Code Reasoning at the Repository Level](http://arxiv.org/abs/2601.03731)

- RepoReason: introduces a repository-level code reasoning benchmark designed as a white-box diagnostic tool, utilizing Abductive Assertion Verification and the Semantic Oracle Framework to evaluate LLM agents.
- The framework employs a five-phase pipeline including Repository Curation, Structural Filtering, Execution-Driven Mutation, Task Instantiation, and Diagnostic Evaluation to ensure tasks are authentic, deep, and unmemorized.
- A White-box Diagnostic Instrument quantifies reasoning complexity using three orthogonal cognitive metrics: ESV (Reading Load), MCL (Simulation Depth), and DFI (Integration Width).

---

[Do Autonomous Agents Contribute Test Code? A Study of Tests in Agentic Pull Requests](http://arxiv.org/abs/2601.03556)

- AIDev Mining Study (AMS): introduces an empirical study analyzing test inclusion and testing practices in agent-generated Pull Requests (PRs) using the AIDev Dataset, focusing on five major autonomous coding agents (OpenAI Codex, GitHub Copilot, Devin, Cursor, and Claude Code).
- The study reconstructs commit timelines and uses regex heuristics for Test File Identification to calculate PR Metrics Calculation, including churn, turnaround time, merge rate, and test-to-code churn ratio.
- Findings indicate that test inclusion in agentic PRs is increasing over time, and test-containing PRs are consistently larger and require longer turnaround times compared to non-test PRs.

---

[SCRIBE: Structured Mid-Level Supervision for Tool-Using Language Models](http://arxiv.org/abs/2601.03555)

- SCRIBE (Skill-Conditioned Reward with Intermediate Behavioral Evaluation): introduces a reinforcement learning framework that enhances tool-augmented reasoning by intervening at a mid-level abstraction using a Router, Skill Prototype Library, LLM as Reward Model, and GRPO for policy optimization.
- SCRIBE anchors reward modeling in a curated library of Skill Prototypes, transforming open-ended LLM evaluation into a constrained verification task using context-specific rubrics to significantly reduce reward variance.
- The framework achieves state-of-the-art performance on reasoning and tool-use benchmarks, demonstrating that mid-level skill mastery acts as a precursor to the emergence of strategic high-level planning.

---

[DeepSynth-Eval: Objectively Evaluating Information Consolidation in Deep Survey Writing](http://arxiv.org/abs/2601.03540)

- DeepSynth-Eval (DSE): introduces a benchmark to objectively evaluate LLM information consolidation capabilities in deep survey writing, utilizing high-quality surveys as gold standards and reverse-engineering tasks to create an Oracle Context (D) and an Evaluation Standard (C).
- The Evaluation Standard (C) consists of fine-grained General Checklists ($C_{gen}$) for factual coverage and Constraint Checklists ($C_{con}$) for structural organization, transforming subjective judgment into verifiable metrics via a Judge Model.
- The benchmark evaluates synthesis performance using two controlled workflows—E2E Single-turn and Agentic Multi-turn—demonstrating that agentic plan-then-write methods significantly outperform single-turn generation by improving grounding and reducing hallucinations.

---

#### 6th January 2026

[Automated Semantic Rules Detection (ASRD) for Emergent Communication Interpretation](http://arxiv.org/abs/2601.03254)

- ASRD (Automated Semantic Rules Detection): introduces an algorithm designed to interpret emergent communication by extracting semantic rules from messages exchanged by Lewis Game agents, using data grouping, constant position identification, combination analysis, and global constant removal.
- The algorithm links extracted message patterns to specific attributes and hyperattributes derived from the input image data, significantly simplifying the subsequent analysis of emergent language semantics.
- The method operates on messages generated by a Speaker agent (ResNet-18 encoder, LSTM) and interpreted by a Listener agent in a referential communication task.

---

[InfiAgent: An Infinite-Horizon Framework for General-Purpose Autonomous Agents](http://arxiv.org/abs/2601.03204)

- InfiAgent: introduces an infinite-horizon framework for autonomous agents, featuring a Multi-Level Agent Hierarchy (Hierarchical task decomposition), File-Centric Architecture (Externalized persistent state), Bounded Reasoning Context Reconstruction (Fixed-size context window), Periodic State Consolidation (Refreshes context snapshot), External Attention Pipeline (Processes massive documents), Tool Calls (Query-driven execution), Ten-Step Strategy & State Update (Bounded thinking module), Batch File Operations & Shared Context (Tool parameter management), and Unlimited Runtime (Stable long-horizon execution).
- The framework achieves stable long-horizon reasoning by explicitly separating persistent task state, stored in a file-centric workspace, from the strictly bounded LLM reasoning context, which only includes recent actions and a state snapshot.
- The hierarchical architecture, comprising Alpha (orchestrator), Domain, and Atomic agents, combined with the external attention mechanism, reduces error propagation and offloads heavy information processing, enabling smaller LLMs to achieve competitive performance.

---

[A Fast Semidefinite Convex Relaxation for Optimal Control Problems With Spatio-Temporal Constraints](http://arxiv.org/abs/2601.03055)

- Fast SDR (Fast Semidefinite Programming Convex Relaxation): introduces a novel convex relaxation framework for solving optimal control problems (OCPs) with coupled spatio-temporal constraints using TS/DMS/NLP Formulation/SDR/Sparse-structure-aware Lifting/IPOPT Refinement.
- The approach transforms the inherently nonconvex OCP, which jointly optimizes trajectory and crossing times, into a structured NLP using time-scaling and direct multiple shooting.
- The sparse-structure-aware SDR method tightly approximates the nonconvex components, achieving high optimality and an order-of-magnitude reduction in computational time compared to standard SDR.

---

[A Bi-directional Adaptive Framework for Agile UAV Landing](http://arxiv.org/abs/2601.03037)

- Bi-directional Adaptive Framework: introduces a cooperative landing system that treats the quadrotor and the variable-attitude platform as coupled active agents, utilizing a Two-Stage Cooperative Planning Algorithm for time-optimal trajectory generation and active attitude synchronization.
- The framework breaks the conventional track-then-descend paradigm by parallelizing alignment and descent phases, enabling aggressive "sprint-then-brake" maneuvers for rapid state synchronization within transient landing windows.
- Agility is achieved through the Terminal Attitude Heuristic Selection Method (Stage One) and the subsequent generation of a full 3D continuous trajectory via the Trajectory Planner (Stage Two), ensuring dynamic feasibility and minimizing recovery time.

---

[The Path Ahead for Agentic AI: Challenges and Opportunities](http://arxiv.org/abs/2601.02749)

- IAAIF (Integrative Agentic AI Framework): introduces an integrative framework describing core components that bridge LLMs with autonomous behavior, utilizing LLM Brain (Reasoning and Planning), Perception (Structured input conversion), Action (Plan execution), Memory (Persistence and retrieval), Environment (External systems), and Feedback Loop (Continuous adaptation) to enable goal-driven systems.
- The architecture operates via a continuous perception-reasoning-action feedback loop, allowing agents to decompose complex tasks and refine actions based on environmental outcomes.
- This architectural transition moves LLMs from passive text generators to adaptive, interactive agents capable of long-term planning and tool use.

---

[BAYESIAN ORCHESTRATION OF MULTI-LLM AGENTS FOR COST-AWARE SEQUENTIAL DECISION-MAKING](http://arxiv.org/abs/2601.01522)

- Bayesian Orchestration of Multi-LLM Agents: introduces a mathematically principled framework that treats multiple LLMs as approximate likelihood functions for cost-aware sequential decision-making, including 5 diverse LLMs, contrastive prompting, median estimate aggregation, explicit prior specification, sequential belief updating, asymmetric costs, adaptive screening, and expected cost minimization.
- The framework achieves a 34% cost reduction and 45% fairness improvement compared to single-LLM baselines by correcting fundamental mathematical deficiencies inherent in discriminative LLM architectures.
- Key capabilities enabled by this generative approach include robust multi-LLM aggregation for bias mitigation, explicit prior correction for domain adaptation, and principled Value-of-Information calculations for adaptive evidence gathering.

---

[LATENT SPACE REINFORCEMENT LEARNING FOR MULTI-ROBOT EXPLORATION](http://arxiv.org/abs/2601.01139)

- Decentralized Hierarchical DRL Framework: introduces a scalable, noise-resilient solution for multi-robot exploration using a Procedural Map Generation Algorithm, a Convolutional Autoencoder, a Weighted Consensus Mechanism, and a Hierarchical DRL Architecture.
- The framework utilizes a pre-trained convolutional autoencoder to compress high-resolution occupancy grids into compact latent state vectors, effectively overcoming the curse of dimensionality for DRL agents.
- Robust decentralized coordination is achieved via a weighted consensus mechanism that fuses shared encoded self-maps using a tuneable trust parameter ($\beta$) to mitigate error accumulation from noisy communication.

---

[AGENTIC PHYSICAL AI TOWARD A DOMAIN-SPECIFIC FOUNDATION MODEL FOR NUCLEAR REACTOR CONTROL](http://arxiv.org/abs/2512.23292)

- APAI (Agentic Physical AI): introduces a framework for nuclear reactor control using a compact LLM (SmolLM2-360M) trained via a Two-Phase Curriculum and validated through Physical AI (outcome-centric validation) in the KOMODO Simulator.
- The system achieves high reliability (97.4% success at $\pm 5\%$) and eliminates catastrophic tail risk through data scaling (1K to 100K scenarios), inducing a 500x variance collapse.
- The approach defines correctness by physics execution rather than parameter imitation, enabling the model to autonomously discover and concentrate on robust, low-variance control strategies.

---

[SmartSnap: Proactive Evidence Seeking for Self-Verifying Agents](http://arxiv.org/abs/2512.22322)

- SmartSnap: introduces a paradigm shift from passive, post-hoc verification to proactive, in-situ self-verification by the Self-Verifying Agent, guided by 3C Principles, using an Augmented Action Space, Curated Snapshot Evidences, Composite Reward Function, and Environment.
- The Self-Verifying Agent executes complex GUI tasks and proactively curates a minimal, decisive set of snapshot evidences, which are then judged by a general LLM-as-a-Judge Verifier.
- This proactive evidence seeking reduces verification cost and LLM hallucination risk while facilitating efficient agent training via intrinsic Reward Shaping based on structured verifier feedback.

---

[FIRE-VLM: A Vision-Language-Driven Reinforcement Learning Framework for UAV Wildfire Tracking in a Physics-Grounded Fire Digital Twin](http://arxiv.org/abs/2601.03449)

- FIRE-VLM: introduces an end-to-end VLM-guided reinforcement learning framework trained within a high-fidelity, physics-grounded wildfire digital twin, combining high-fidelity wildfire simulation, UAV dynamics, dual-view sensing, VLM semantic guidance, and a PPO policy for robust firefront tracking.
- The framework utilizes a CLIP-style VLM operating on dual-view RGB observations (top-down and angled) to generate semantic alignment scores and directional likelihoods, which are integrated into the PPO reward function.
- This VLM-guided reward shaping, combining physics-based incentives with semantic understanding, significantly improves time-to-detection and time-in-FOV compared to purely RL baselines across kilometer-scale fires.

---

[Why LLMs Aren't Scientists Yet: Lessons from Four Autonomous Research Attempts](http://arxiv.org/abs/2601.03315)

- Autonomous Research Pipeline: introduces a case study of four end-to-end attempts to autonomously generate ML research papers using six LLM agents mapped to the scientific workflow stages.
- The system, primarily using Gemini 2.5 Pro and Claude Code, documented six recurring failure modes, including bias toward training data, implementation drift, and memory/context degradation across long-horizon tasks.
- Based on the attempts (one success, three failures), the authors propose four design principles for robust AI-scientist systems, emphasizing verification, gradual grounding, and planning for failure and recovery.

---

#### 5th January 2026

[Textual Explanations and Their Evaluations for Reinforcement Learning Policy](http://arxiv.org/abs/2601.02514)

- Proposed Framework: introduces a novel Explainable Reinforcement Learning (XRL) system for generating textual explanations and evaluating them quantitatively, utilizing an LLM (Interprets user questions), Predicate Functions (Discretizes state features), and a Clustering-Based Summarizer (Generates textual explanations).
- The framework converts the generated textual explanations into transparent rules via Rules Extraction, enabling systematic evaluation of properties, fidelity, and performance by comparing them against replay data and deploying them in the Environment.
- The system addresses limitations of existing methods by focusing on frequent conditions and suppressing duplicates, and includes refinement techniques to minimize conflicting information and maximize F1 scores.

---

[Project Ariadne: A Structural Causal Framework for Auditing Faithfulness in LLM Agents](http://arxiv.org/abs/2601.02314)

- Project Ariadne: introduces a novel XAI framework that utilizes Structural Causal Models (SCMs) and counterfactual logic to audit the causal integrity of LLM agent reasoning.
- The framework performs hard interventions on intermediate reasoning nodes, systematically inverting logic or negating premises, to generate a counterfactual trace and answer.
- By quantifying the Causal Sensitivity Score ($\phi$) between the original and counterfactual answers, the framework detects Causal Decoupling, where reasoning traces function as post-hoc justifications.

---

[Confidence Estimation for LLMs in Multi-turn Interactions](http://arxiv.org/abs/2601.02179)

- P(SUFFICIENT) Multi-turn Confidence Framework: introduces the first systematic study of confidence estimation for LLMs in multi-turn interactions, establishing an evaluation framework grounded in Calibration (confidence matches empirical correctness) and Monotonicity (confidence increases with information).
- The framework utilizes novel metrics, including InfoECE (length-normalized Expected Calibration Error) and Kendall's $\tau$ (monotonic trend measure), alongside a Hinter-Guesser paradigm for generating controlled evaluation datasets.
- Evaluation of existing methods reveals struggles with calibration and monotonicity, while the proposed logit-based probe, P(SUFFICIENT), achieves comparatively better performance by tracking the sufficiency of accumulated evidence.

---

[Agentic AI in Remote Sensing: Foundations, Taxonomy, and Emerging Systems](http://arxiv.org/abs/2601.01891)

- Agentic AI in Remote Sensing Taxonomy: introduces a unified taxonomy distinguishing between single-agent copilots and multi-agent systems, analyzing architectural foundations such as planning mechanisms, retrieval-augmented generation, and memory structures.
- The proposed ecosystem is structured into four key components: Foundations (data acquisition and models), Agents (copilots and orchestrators), Systems & Platforms (RAG, Tools, Memory), and Evaluation (benchmarks for planning and reasoning).
- The survey critically examines limitations in geospatial grounding, safety, and orchestration, outlining a strategic roadmap for robust, autonomous geospatial intelligence development using Earth-native models.

---

[Agentic Memory: Learning Unified Long-Term and Short-Term Memory Management for Large Language Model Agents](http://arxiv.org/abs/2601.01885)

- AgeMem (Agentic Memory): introduces a unified framework that integrates LTM and STM management directly into the LLM agent's policy using explicit tool-based operations.
- The framework utilizes a Three-Stage Progressive RL strategy and Step-wise GRPO to enable end-to-end optimization of memory behaviors despite sparse and discontinuous rewards.
- By exposing memory operations (ADD, RETRIEVE, SUMMARY, FILTER) as actions, the LLM agent autonomously decides when and how to manage context and persistent knowledge.

---

[JENIUS AGENT: TOWARDS EXPERIENCE-DRIVEN ACCURACY OPTIMIZATION IN REAL-WORLD SCENARIOS](http://arxiv.org/abs/2601.01857)

- Jenius Agent: introduces an experience-driven autonomous agent framework grounded in real-world practice, integrating adaptive prompt generation, context-aware tool orchestration, and hierarchical memory management to optimize task accuracy and efficiency.
- The framework uses the LLM as a central orchestrator in a feedback-driven loop, coordinating the three core modules to generate system prompts, select relevant tools, and filter historical context.
- Key innovations include dynamic prompt adaptation based on task state, semantic retrieval and inflection point-based filtering for tool selection, and layered memory compression to mitigate token cost and context noise.

---

[ARIES: A Scalable Multi-Agent Orchestration Framework for Real-Time Epidemiological Surveillance and Outbreak Monitoring](http://arxiv.org/abs/2601.01831)

- ARIES (Agentic Retrieval Intelligence for Epidemiological Surveillance): introduces a scalable, hierarchical multi-agent orchestration framework for real-time epidemiological surveillance, utilizing a Manager Agent to delegate tasks to specialized sub-agents.
- The system mimics an Emergency Operations Center (EOC) structure, employing agents like the Senior Medical Scientist and CDC Data Analyst to query specific, high-integrity data sources (PubMed, CDC WONDER, WHO DONs).
- Built on the CrewAI framework, ARIES automates data extraction and logical synthesis, providing specialized reasoning to identify emergent threats and mitigate the knowledge gap inherent in generalized LLMs.

---

[Sparse Threats, Focused Defense: Criticality-Aware Robust Reinforcement Learning for Safe Autonomous Driving](http://arxiv.org/abs/2601.01800)

- CARRL (Criticality-Aware Robust Reinforcement Learning): introduces a novel DRL-based adversarial training approach for robust autonomous driving, formulated as a General-Sum Markov Game between the Risk Exposure Adversary and the Risk-Targeted Robust Agent.
- The REA uses Decoupled Optimization to execute criticality-aware attacks focused on sparse safety-critical moments, while the RTRA employs a Dual Replay Buffer and Consistency-Constrained Policy Optimization to learn robust policies despite data scarcity.
- This framework achieves superior robustness by explicitly addressing the asymmetry between the agent and adversary objectives and focusing defensive capacity on high-risk scenarios, significantly reducing the collision rate.

---

[ALIGNDRIVE: ALIGNED LATERAL-LONGITUDINAL PLANNING FOR END-TO-END AUTONOMOUS DRIVING](http://arxiv.org/abs/2601.01762)

- AlignDrive: introduces a novel cascaded planning paradigm where longitudinal planning is explicitly conditioned on the predicted lateral drive path, enabling coordinated and collision-aware lateral and longitudinal planning.
- The framework reformulates longitudinal planning as a simpler 1D displacement prediction problem along the drive path, allowing the model to focus capacity on dynamic interactions rather than static geometry.
- A planning-oriented data augmentation strategy simulates rare safety-critical events by inserting agents and relabeling longitudinal targets, substantially improving collision avoidance capabilities.

---

[Structural Representations for Cross-Attack Generalization in AI Agent Threat Detection](http://arxiv.org/abs/2601.01723)

- GMF (Gated Multi-View Fusion): introduces structural tokenization, encoding execution-flow patterns (tool calls, arguments, observations) rather than conversational content, dramatically improving cross-attack generalization against structural threats.
- The approach uses parallel BiLSTM encoders to process both structural and conversational representations, which are adaptively combined via a learned gate for robust performance across diverse attack families.
- Structural tokenization achieved 39–71 AUC point gains on unseen structural attacks (tool hijacking, data exfiltration) and unknown attacks, confirming that AI agent security is fundamentally a structural problem.

---

[TravelBench: A Broader Real-World Benchmark for Multi-Turn and Tool-Using Travel Planning](http://arxiv.org/abs/2512.22673)

- TravelBench: introduces a comprehensive, real-world benchmark for LLM agents in travel planning, featuring Single-Turn, Multi-Turn, and Unsolvable subtasks, supported by real user data and a Toolkit of 10 real travel tools.
- The benchmark utilizes a reproducible sandbox with a Tool-Call Cache for stable tool invocation and employs an LLM-as-a-Judge Protocol, including a Tool-Use Penalty and Meta-Judge for robust evaluation.
- It is the first travel planning benchmark to incorporate profile-based implicit preferences and multi-turn user-agent interaction to elicit requirements and test capability boundaries.

---

[AgentMark: Utility-Preserving Behavioral Watermarking for Agents](http://arxiv.org/abs/2601.03294)

- AgentMark: introduces a behavioral watermarking framework that embeds multi-bit identifiers into an agent's planning process using Elicit Behavioral Probability Output (explicit probability list), Behavior Distribution Generate (estimate implicit policy), Keyed Distribution Preserving Watermark (core embedding mechanism), Robustness Payload Coding (erasure-resilient recovery), Keyed Pseudorandom Source (synchronize encoding/decoding), Distribution-Preserving Sampling (preserve marginal distribution), Selected Behavior (watermarked planning choice), Execution Action (action carried out), Memory Module (agent context storage), and Environment (agent interaction space), ensuring utility preservation under black-box APIs.
- The framework addresses the challenge of watermarking high-level planning behaviors by eliciting an explicit behavior distribution from the LLM agent and applying distribution-preserving conditional sampling to embed the watermark without shifting the agent's planning policy.
- AgentMark-F, a concrete instantiation, utilizes differential recombination and cyclic-shift uniform encoding, combined with RLNC for robust recovery of provenance bits from partial trajectories subject to erasure or truncation.

---

#### 4th January 2026

[Lying with Truths: Open-Channel Multi-Agent Collusion for Belief Manipulation via Generative Montage](http://arxiv.org/abs/2601.01685)

- Generative Montage (GM): introduces the first cognitive collusion attack framework that constructs deceptive narratives using a Writer (Narrative Synthesis), Editor (Montage Sequencing), and Director (Adversarial Debate) to manipulate LLM agents' beliefs via public, truthful evidence.
- The framework exploits LLMs' tendency for narrative coherence (overthinking) by strategically sequencing truthful evidence fragments (Montage Sequencing) to induce spurious causal inferences, resulting in the internalization of a global lie from local truths.
- Experiments using the CoPHEME dataset show pervasive vulnerability across 14 LLM families, with attack success rates reaching 74.4% for proprietary models, and enhanced reasoning capabilities paradoxically increasing susceptibility.

---

[DRIVINGGEN: A COMPREHENSIVE BENCHMARK FOR GENERATIVE VIDEO WORLD MODELS IN AUTONOMOUS DRIVING](http://arxiv.org/abs/2601.01528)

- DrivingGen: introduces a comprehensive benchmark for generative video world models in autonomous driving, featuring a Diverse Driving Dataset (varied conditions/behaviors), Generative World Models (models under evaluation), Evaluation Metrics (4 comprehensive sets), and a SLAM Pipeline (robust trajectory extraction).
- The benchmark addresses limitations in existing evaluations by introducing novel metrics that jointly assess visual realism, trajectory plausibility (Fréchet Trajectory Distance, FTD), temporal coherence, and controllability (Average Displacement Error, ADE, and Dynamic Time Warping, DTW).
- DrivingGen facilitates reliable, controllable, and deployable driving world models by providing a unified evaluation framework that exposes critical trade-offs between visual quality and physical consistency across 14 state-of-the-art models.

---

[KGCE: Knowledge-Augmented Dual-Graph Evaluator for Cross-Platform Educational Agent Benchmarking with Multimodal Language Models](http://arxiv.org/abs/2601.01366)

- KGCE (Knowledge-Augmented Dual-Graph Evaluator): introduces a novel cross-platform educational agent benchmarking platform integrating a domain-specific Knowledge Base and a Dual-Graph Evaluator for fine-grained assessment.
- The system utilizes MLMs (GPT-4o, Qwen-VL, Gemini) within Windows and Android agents, supported by the Knowledge Base, to execute complex, multi-step educational tasks involving proprietary software.
- The Dual-Graph Evaluator, comprising the Task Completeness Graph and the Execution Efficiency Graph, provides eight fine-grained metrics to quantify both task completion quality and execution path efficiency.

---

[Towards LLM-enabled autonomous combustion research: A literature-aware agent for self-corrective modeling workflows](http://arxiv.org/abs/2601.01357)

- FlamePilot: introduces an LLM agent for autonomous combustion research, integrating an LLM Agent (Workflow orchestration), Self-corrective Execution Loop (Error resolution, refinement), Expertise Toolkit (CFD execution capabilities), and Domain Literature Knowledge (Structured scientific findings) to automate and refine complex CFD workflows.
- The architecture uses atomic tools for robust execution across standard (OpenFOAM) and extended (DeepFlame) CFD frameworks, achieving a perfect 1.0 executability score on the FoamBench-Advanced benchmark.
- By grounding decisions in dynamically extracted scientific literature, the agent functions as a transparent, human-supervised research copilot, accelerating discovery while upholding scientific rigor.

---

[Neural-network-based Self-triggered Observed Platoon Control for Autonomous Vehicles](http://arxiv.org/abs/2601.01335)

- NSOPC: introduces an adaptive consensus tracking control framework for nonlinear multi-agent systems using backstepping design, a distributed observer, RBFNNs, and a self-triggered communication mechanism.
- The framework addresses uncertain dynamics and intermittent communication by using RBFNNs to approximate unknown nonlinearities and a distributed observer to estimate states based on limited, intermittent measurements.
- The self-triggered mechanism calculates the next control update instant based on the current control signal, guaranteeing a strictly positive minimum inter-event time and optimizing communication efficiency.

---

[Digital Twin AI: Opportunities and Challenges from Large Language Models to World Models](http://arxiv.org/abs/2601.01321)

- UFSF (Unified Four-Stage Framework): introduces a systematic characterization of AI integration across the digital twin lifecycle, including Modeling, Mirroring, Intervening, and Autonomous Management stages.
- The framework evolves DTs from passive simulation tools to proactive, self-improving cognitive systems by leveraging physics-informed AI, generative AI, predictive AI, and agentic AI.
- Autonomous management is enabled by LLMs for natural language interaction and foundation models for multimodal perception, supporting complex decision-making and closed-loop control.

---

[Automated Post-Incident Policy Gap Analysis via Threat-Informed Evidence Mapping using Large Language Models](http://arxiv.org/abs/2601.03287)

- Agentic Post-Incident Review Workflow: introduces an LLM-driven, agentic pipeline integrating log analysis, threat attribution, policy retrieval, policy evaluation, and report generation, designed to autonomously identify security policy gaps.
- The framework uses GPT-4o for reasoning, LangGraph for multi-agent orchestration, and LlamaIndex for traceable policy retrieval, ensuring explicit evidence-to-policy traceability grounded in the MITRE ATT&CK framework.
- This approach augments cybersecurity post-incident reviews by providing evidence-grounded, auditable insights to improve efficiency, consistency, and governance alignment.

---

#### 3rd January 2026

[Harnessing Environmental Memory with Reinforcement Learning in Open Quantum Systems](http://arxiv.org/abs/2601.01252)

- RL Framework: introduces a reinforcement learning approach for enhancing non-Markovianity in a driven open quantum system by maximizing the BLP measure via direct trajectory-based feedback.
- The framework utilizes Soft Actor-Critic (SAC) and Proximal Policy Optimization (PPO) agents to learn continuous control fields ($\Omega(t)$) that exploit environmental memory by coordinating backflow across multiple revival windows.
- Unlike gradient-based Optimal Control Theory (OCT), which concentrates on amplifying a single backflow peak, the RL policies discover distributed-backflow strategies, leading to substantially higher total integrated non-Markovianity.

---

[ROBOPHD: SELF-IMPROVING TEXT-TO-SQL THROUGH AUTONOMOUS AGENT EVOLUTION](http://arxiv.org/abs/2601.01126)

- RoboPhD: introduces a system where AI agents autonomously conduct research to improve Text-to-SQL performance, featuring Evolution AI (Generates new agents), SQL Generation Agent (Performs Text-to-SQL), Database Analysis Tool (Offline Python script analysis), Eval Instructions (SQL generation guidance), Evolutionary Loop (Iterative improvement cycle), ELO-based Selection Mechanism (Ranks agent performance), Evaluation Results (Feedback for Evolution AI), and Universal Verification (Self-correction and validation).
- The system employs a closed-loop evolution cycle, using an ELO-based selection mechanism to drive survival-of-the-fittest dynamics and handle non-transitivity in agent performance across different databases.
- The evolved agent consists of two deployable artifacts—a deterministic Python database analysis tool and SQL generation instructions—allowing for cost-effective deployment and demonstrating larger performance gains on cheaper LLMs.

---

[Performance and Security Aware Distributed Service Placement in Fog Computing](http://arxiv.org/abs/2601.01125)

- SPA-DDRL: introduces a distributed broker-learner architecture for joint security and performance optimization in Fog computing, integrating LSTM networks, Prioritized Experience Replay, and off-policy correction mechanisms.
- The framework formulates the service placement problem as a weighted multi-objective optimization task minimizing latency and maximizing a security score derived from a novel three-tier security hierarchy.
- The distributed design enables scalable security-performance optimization across heterogeneous Fog environments, achieving significant improvements in response time and convergence speed compared to baselines.

---

[Harm in AI-Driven Societies: An Audit of Toxicity Adoption on Chirper.ai](http://arxiv.org/abs/2601.01090)

- CTAA (Chirper.ai Toxicity Audit Approach): introduces a large-scale empirical audit of toxicity adoption on the fully AI-driven social platform Chirper.ai, utilizing LLM-driven Agents, Stimuli, Responses, a Detoxify Classifier, Influence-Driven Response Rate, Spontaneous Response Rate, and a Gemini-2.0-flash Evaluator.
- The audit models agent interactions in terms of toxic stimuli (posts) and toxic responses (comments), finding that cumulative toxic exposure significantly increases the probability of toxic responding.
- The introduced metrics, IRR and SRR, reveal a strong trade-off between exposure-driven and autonomous toxicity, and the number of toxic stimuli alone accurately predicts whether an agent will eventually produce toxic content.

---

[Aerial World Model for Long-horizon Visual Generation and Navigation in 3D Space](http://arxiv.org/abs/2512.21887)

- ANWM (Aerial Navigation World Model): introduces a world model for UAV navigation that predicts long-horizon visual observations conditioned on past frames and 3D actions, enabling trajectory ranking based on semantic plausibility. 
- To handle the complex 3D action space and ensure spatio-temporal consistency, ANWM incorporates the Future Frame Projection (FFP) module, which provides coarse geometric priors by warping past frames into future viewpoints. 
- The model uses a Conditional Diffusion Transformer (CDiT) backbone and Independent Latent Modulation (ILM) to separately process real past frames and the synthesized projected future frame prior for robust long-range generation.

---

#### 2nd January 2026

[Early-Stage Prediction of Review Effort in AI-Generated Pull Requests](http://arxiv.org/abs/2601.00753)

- Circuit Breaker Triage Model: introduces a system for predicting high-review-effort AI-generated Pull Requests (PRs) using the AIDev Dataset, Creation-Time Features, and a LightGBM Model to identify High Cost PR Target for a Triage Mechanism.
- The model achieves high discrimination (AUC 0.9571) using only static complexity signals (e.g., patch size, files touched) available at creation time (T0), confirming that structural complexity, not semantic content, dictates review burden.
- The framework intercepts 69% of total review effort at a 20% review budget allocation, enabling maintainers to triage the expensive tail of agent contributions with zero latency.

---

[Bayesian Inverse Games with High-Dimensional Multi-Modal Observations](http://arxiv.org/abs/2601.00696)

- BIG-VAE (Structured Variational Autoencoder for Bayesian Inverse Games): introduces an approximate Bayesian inference framework that trains a structured VAE with an embedded differentiable Nash game solver to infer posterior distributions over hidden agent objectives from multimodal observations.
- The architecture fuses high-dimensional visual cues (images) and low-dimensional partial-state observations (trajectories) via a shared latent space, enabling uncertainty quantification and multi-modal inference.
- By amortizing the inference process offline, the framework supports real-time posterior sampling at test time, facilitating safer and more efficient uncertainty-aware downstream motion planning compared to MLE methods.

---

[Trajectory Guard - A Lightweight, Sequence-Aware Model for Real-Time Anomaly Detection in Agentic AI](http://arxiv.org/abs/2601.00516)

- Trajectory Guard: introduces a Siamese Recurrent Autoencoder with a hybrid loss function to perform real-time anomaly detection in LLM agent trajectories, addressing both contextual and structural failures.
- The architecture utilizes a Task Tower for contextual fit via contrastive learning and a Trajectory Tower with a GRU autoencoder for structural validity via sequence reconstruction.
- Operating at 32 ms latency, the model runs significantly faster than LLM Judge baselines, making it suitable for production safety verification in agentic systems.

---

#### 1st January 2026

[The Rise of Agentic Testing: Multi-Agent Systems for Robust Software Quality Assurance](http://arxiv.org/abs/2601.02454)

- ATA (Agentic Testing Architecture): introduces a multi-agent, closed-loop testing framework where the Test Generation Agent, Execution and Analysis Agent, and Review and Optimization Agent collaboratively generate, execute, analyze, and refine tests until convergence criteria are met.
- The system leverages sandboxed execution, detailed failure reporting, and iterative regeneration guided by reinforcement signals derived from coverage metrics and execution outcomes.
- This architecture establishes autonomous, self-correcting software Quality Assurance by significantly reducing invalid tests and manual review effort compared to static LLM baselines.

---

[When Small Models Are Right for Wrong Reasons: Process Verification for Trustworthy Agents](http://arxiv.org/abs/2601.00513)

- PVS (Process Verification System): introduces a methodology to quantify the "Right-for-Wrong-Reasons" (RWR) phenomenon in small LLMs using the Reasoning Integrity Score (RIS) and a fast Distilled Verifier.
- The system analyzes 10,734 reasoning traces across three 7-9B LLMs and diverse tasks, revealing that 50-69% of correct outputs contain fundamentally flawed reasoning invisible to standard accuracy metrics.
- The analysis shows that Retrieval-Augmented Generation (RAG) acts as essential external scaffolding to improve integrity, while meta-cognitive prompts actively harm performance due to "pseudo-reflection" in sub-10B models.

---

[Automated decision-making by chemical echolocation in active droplets](http://arxiv.org/abs/2601.00480)

- Chemical Echolocation: introduces a generic mechanism enabling synthetic agents, such as active droplets, to achieve autonomous navigational decision-making and maze solving by leveraging self-generated chemo-hydrodynamic signals.
- The mechanism relies on the agent acting as a chemical source and sensor, where signals reflected by dead ends create a "chemical echo" that elicits a direct physical response (chemorepulsion) to steer the agent toward open paths.
- This strategy provides a robust alternative to traditional source-seeking methods, remaining effective and efficient even in large mazes without requiring external guidance or complex information processing machinery.

---

[Security in the Age of AI Teammates: An Empirical Study of Agentic Pull Requests on GitHub](http://arxiv.org/abs/2601.00477)

- Empirical Study Methodology: introduces a multi-stage process analyzing 33,596 Agentic-PRs from the AIDev dataset, including Keyword-based PR Filtering, Manual Validation, Qualitative Open Coding, Statistical Analysis, Structured Prediction Models, and Text-Based Prediction Models, to characterize autonomous coding agents' security contributions and human review dynamics.
- The study identifies that security-related Agentic-PRs (3.85% of total) receive heightened human scrutiny, exhibiting lower merge rates (61.5%) and substantially longer review latency (median 3.92 hours) compared to non-security PRs (median 0.11 hours).
- PR rejection is found to be more strongly associated with complexity and verbosity (e.g., PR size, title length) than with explicit security terminology, suggesting reviewers prioritize clarity and scope over security topic presence alone.

---

[Space Debris Removal using Nano-Satellites controlled by Low-Power Autonomous Agents](http://arxiv.org/abs/2601.00465)

- Low-Power Autonomous Agents (LPAA) System: introduces a multi-agent system using two nano-satellite Free-Flyers (FFs) equipped with embedded BDI agents running on nRF52840 microcontrollers to synchronously de-orbit space debris.
- The agents communicate wirelessly using the low-power OpenThread protocol and CoAP application layer, synchronizing their actions via a Mothership server acting as a communication hub.
- The system demonstrates energy-efficient autonomy for resource-constrained embedded devices, validated through experiments on the ELISSA air-bearing test-bed.

---

[Mapping Human Anti-collusion Mechanisms to Multi-agent AI](http://arxiv.org/abs/2601.00360)

- MAAI Anti-Collusion Framework: introduces a taxonomy of human anti-collusion mechanisms and maps them to concrete interventions for Multi-Agent AI systems, including Sanctions, Leniency & Whistleblowing, Monitoring & Auditing, Market Design & Structural, and Governance.
- The framework addresses key challenges in AI collusion, such as attribution, identity fluidity, and adversarial adaptation, by proposing architectural components like Overseer Agents and Telemetry-first Design.
- Implementation approaches for LLM-based agents include reward penalties, capability restrictions, interaction protocol constraints, and mandatory transparency requirements like Model Cards.

---

[Bio-inspired Agentic Self-healing Framework for Resilient Distributed Computing Continuum Systems](http://arxiv.org/abs/2601.00339)

- ReCiSt (Bio-inspired Agentic Self-healing Framework for Resilient Distributed Computing Continuum Systems): introduces a bio-inspired agentic self-healing framework for Distributed Computing Continuum Systems (DCCS) that maps biological wound healing phases (Hemostasis, Inflammation, Proliferation, Remodeling) onto four computational layers: Containment, Diagnosis, Meta-Cognitive, and Knowledge.
- The framework utilizes LM-powered agents, including monitoring agents and reasoning micro-agents, to perform autonomous fault isolation, causal diagnosis, adaptive recovery, and long-term knowledge consolidation.
- The Meta-Cognitive Layer regulates internal reasoning via migratory micro-agents and feedback mechanisms, while the Knowledge Layer supports adaptive knowledge sharing through local and global Rendezvous Points.

---

[Can Optimal Transport Improve Federated Inverse Reinforcement Learning?](http://arxiv.org/abs/2601.00309)

- OT-FIRL (Optimal Transport-based Federated Inverse Reinforcement Learning): introduces a federated IRL framework that fuses locally learned MaxEnt reward functions using an entropically regularized Wasserstein barycenter for geometry-aware aggregation.
- This approach treats local rewards as probability distributions over a shared state-action support, yielding stable and transferable global reward estimates across heterogeneous agents and environments.
- The framework provides theoretical stability and parameter-error bounds, demonstrating that barycentric fusion contracts toward the true reward function under bounded local estimation error.

---

[FLASHINFER-BENCH: BUILDING THE VIRTUOUS CYCLE FOR AI-DRIVEN LLM SYSTEMS](http://arxiv.org/abs/2601.00227)

- FlashInfer-Bench: introduces a standardized, closed-loop framework connecting kernel generation, benchmarking, and deployment for AI-driven LLM systems.
- The system uses FlashInfer Trace, a unified JSON schema, to communicate kernel definitions, workloads, solutions, and evaluation results consistently.
- A dynamic substitution mechanism, `flashinfer_bench.apply()`, seamlessly injects the best-performing, validated kernels into production LLM engines like SGLang and vLLM.

---

[Device-Native Autonomous Agents for Privacy-Preserving Negotiations](http://arxiv.org/abs/2601.00911)

- DNAAS (Device-Native Autonomous Agent System): introduces a device-native agentic AI system for autonomous, privacy-preserving negotiations using a six-layer architecture that integrates cryptographic and on-device reasoning components.
- The system operates exclusively on user hardware, employing zero-knowledge proofs (ZK proofs) for secure multi-party bargaining and Explainable Memory to generate tamper-evident cryptographic audit trails for compliance.
- Efficiency and mobility are achieved via World Model Distillation for on-device LLM reasoning, Selective State Transfer for cross-device continuity, and Model-Aware Offloading for optimal task placement.

---

[AI-Native Integrated Sensing and Communications for Self-Organizing Wireless Networks: Architectures, Learning Paradigms, and System-Level Design](http://arxiv.org/abs/2601.02398)

- A2ISAC-SON (AI-Native Integrated Sensing and Communications for Self-Organizing Wireless Networks): introduces a comprehensive survey of AI-native ISAC-enabled self-organizing wireless networks, covering ISAC System Architectures, ISAC Waveform Design, Sensing-Comm Co-Design, AI-Native Control Paradigms, Self-Organizing Functions, Sensing-Assisted Communication, and Communication-Assisted Sensing.
- The architecture unifies radio sensing and data communication (ISAC) with AI-driven control loops to create perceptive, autonomous networks capable of self-optimization and adaptation for 6G systems.
- Key technical areas reviewed include the fundamental trade-offs in dual-functional waveform design and the application of DRL and GNNs for dynamic network control, mobility management, and resource allocation.

---

[a³-Bench: A Unified Benchmark of Safety, Robustness, and Efficiency for LLM-Based UAV Agents over 6G Networks](http://arxiv.org/abs/2601.03281)

- a³-Bench: introduces a comprehensive benchmark for evaluating LLM-driven UAV autonomy as a multi-turn conversational reasoning and control problem under dynamic 6G conditions, featuring an LLM-based UAV Agent, Human Operator, and structured action protocols (MCP/A2A).
- The framework models mission execution as a language-mediated control loop, where the LLM adapts its policy based on UAV telemetry, safety constraints, and a dynamic 6G Network Context vector (slicing, latency, throughput).
- Performance is assessed using the composite $\alpha^3$ metric, which unifies six pillars (Task Outcome, Safety Policy, Tool Consistency, Interaction Quality, Network Robustness, Communication Cost) with reliability- and efficiency-normalized scores.

---

#### 31st December 2025

[Context-aware LLM-based AI Agents for Human-centered Energy Management Systems in Smart Buildings](http://arxiv.org/abs/2512.25055)

- LLM-based BEMS AI Agent Framework: introduces a conceptual framework and prototype for LLM-based AI agents to facilitate context-aware energy management in smart buildings through natural language interaction.
- The framework comprises three modules—Perception (sensing), Brain (central control), and Action (actuation/user interaction)—forming a closed feedback loop to capture, analyze, and interpret energy data for intelligent responses and appliance management.
- The Brain module leverages LLM capabilities, profiles, and memory to perform autonomous context-aware reasoning, data analysis, cost prediction, and device scheduling, addressing limitations in existing energy management systems.

---

[HYBRID MOTION PLANNING WITH DEEP REINFORCEMENT LEARNING FOR MOBILE ROBOT NAVIGATION](http://arxiv.org/abs/2512.24651)

- HMP-DRL (Hybrid Motion Planning with Deep Reinforcement Learning): introduces a hybrid navigation framework that couples a Global Planner (Graph-based search) with a Local DRL Policy (Entity-aware collision avoidance) using Checkpoints (Global path guidance) and an Entity-Aware Reward Function (Type-dependent safety margins).
- The local DRL policy extends an attention-based Value Network to incorporate checkpoint features via an Extended Self-State, enabling the robot to balance long-range path following and immediate dynamic obstacle avoidance.
- Validated in a large-scale urban simulator, the framework achieves superior success rates and lower collision rates, especially with vulnerable agents like children, compared to state-of-the-art DRL methods.

---

[MCPAgentBench: A Real-world Task Benchmark for Evaluating LLM Agent MCP Tool Use](http://arxiv.org/abs/2512.24565)

- MCPAgentBench: introduces a real-world task benchmark for evaluating LLM agent MCP tool use, featuring MCP Tool Collection (Authentic MCP tool repository), Task Set (Diverse real-world tasks), Automated Evaluation Sandbox (Autogen-based execution environment), LLM-based Agent (Interprets task, selects tools), and Evaluation & Scoring (Computes performance metrics).
- The benchmark utilizes a dynamic sandbox with locally maintained MCP Servers and a Candidate Tool List L containing distractors (F) to rigorously test tool selection and anti-interference robustness.
- The framework introduces novel Efficiency Metrics, including Task Efficiency Finish Score (TEFS) and Token Efficiency, to assess execution timing and resource consumption beyond simple task correctness.

---

#### 30th December 2025

[Counterfactual VLA: Self-Reflective Vision-Language-Action Model with Adaptive Reasoning](http://arxiv.org/abs/2512.24426)

- CF-VLA (Counterfactual VLA): introduces a self-reflective VLA framework for autonomous driving that critiques and corrects its own planned actions before execution, utilizing meta-actions, counterfactual reasoning, updated meta-actions, and trajectory generation.
- The framework employs a rollout-filter-label pipeline to automatically curate high-value counterfactual reasoning traces from the VLA's own behavior, forming a self-improving loop for training.
- CF-VLA demonstrates adaptive reasoning, engaging the counterfactual reasoning loop more frequently in challenging scenarios to improve trajectory accuracy and safety metrics while maintaining reasonable test-time compute.

---

[The Silicon Psyche: Anthropomorphic Vulnerabilities in Large Language Models](http://arxiv.org/abs/2601.00867)

- SILICONPSYCHE (Synthetic Psychometric Assessment Protocol): introduces a methodology for converting the CPF's 100 psychological indicators into adversarial scenarios to test the Anthropomorphic Vulnerability Inheritance (AVI) of Autonomous Cognitive Agents (LLMs).
- The protocol systematically constructs scenarios exploiting pre-cognitive vulnerabilities like authority-gradient manipulation and temporal pressure, which are then scored using a ternary classification system.
- The research proposes "Psychological Firewalls," derived from the CPIF, as intervention mechanisms to protect LLM agents from semantic manipulation vectors that exploit the Psychological Interface.

---

[SCP: Accelerating Discovery with a Global Web of Autonomous Scientific Agents](http://arxiv.org/abs/2512.24189)

- SCP (Science Context Protocol): introduces an open-source standard enabling a global web of autonomous scientific agents by integrating heterogeneous resources via a centralized SCP Hub, federated SCP Servers, and SCP Clients.
- The protocol standardizes resource integration (tools, LLMs, wet-lab devices) and provides intelligent orchestration for managing the complete experiment lifecycle, ensuring traceability and reproducibility.
- SCP establishes infrastructure for scalable, multi-institution, agent-driven science, supporting complex dry-wet workflows and offering access to an ecosystem of over 1,600 tool resources.

---

[LOONGFLOW: DIRECTED EVOLUTIONARY SEARCH VIA A COGNITIVE PLAN-EXECUTE-SUMMARIZE PARADIGM](http://arxiv.org/abs/2512.24077)

- LoongFlow: introduces a self-evolving agent framework that achieves state-of-the-art solution quality by integrating LLMs into a cognitive Plan-Execute-Summarize (PES) paradigm, supported by a Hybrid Evolutionary Memory.
- The PES paradigm transforms evolutionary search into a structured reasoning process via the Planner (strategic blueprint), Executor (verified code generation), and Summarizer (retrospective reflection).
- The Hybrid Evolutionary Memory, combining Multi-Island models, MAP-Elites, and Adaptive Boltzmann Selection, dynamically balances exploration and exploitation, leading to superior evolutionary efficiency.

---

[Assured Autonomy: How Operations Research Powers and Orchestrates Generative AI Systems](http://arxiv.org/abs/2512.23978)

- OR-AI Integration Architecture (Assured Autonomy): introduces a conceptual framework for autonomous decision-making systems grounded in operations research (OR), integrating AI components with OR mechanisms for structure and assurance.
- The framework relies on flow-based generative models for deterministic, auditable generation and minimax safety design, which uses adversarial stress testing to ensure robustness against worst-case scenarios and distribution shifts.
- This architecture shifts OR's role from a solver to a system architect, responsible for designing control logic, safety boundaries, monitoring regimes, and fallback protocols for high-autonomy settings.

---

#### 29th Dec 2025

[CASCADE: Cumulative Agentic Skill Creation through Autonomous Development and Evolution](http://arxiv.org/abs/2512.23880)

- CASCADE (Cumulative Agentic Skill Creation through Autonomous Development and Evolution): introduces a self-evolving multi-agent framework representing the transition from "LLM + tool use" to "LLM + skill acquisition," integrating Orchestrator, DeepSolver, SimpleSolver, and memory components.
- The framework enables agents to master complex external tools and codify knowledge through two meta-skills: continuous learning (via web search and code extraction) and self-reflection (via introspection and knowledge graph exploration).
- The core problem-solving engine, DeepSolver, utilizes a four-step sequential workflow with conditional parallel debugging involving Solution Researcher, Code Agent, Parallel Debug Agents, and Output Processor Agent.

---

[OmniAgent: Audio-Guided Active Perception Agent for Omnimodal Audio-Video Understanding](http://arxiv.org/abs/2512.23646)

- OmniAgent: introduces an audio-guided active perception agent that dynamically orchestrates specialized tools via a recursive "Think-Act-Observe-Reflect" loop for fine-grained audio-visual reasoning.
- The agent employs a novel coarse-to-fine audio-guided perception paradigm, leveraging audio cues to localize temporal events and guide subsequent high-resolution visual inspection.
- By treating strong single-modal models as callable tools, the framework achieves state-of-the-art performance by solving cross-modal alignment difficulties through active information inquiry.

---

[Close the Loop: Synthesizing Infinite Tool-Use Data via Multi-Agent Role-Playing](http://arxiv.org/abs/2512.23611)

- InfTool: introduces a fully autonomous, self-evolving framework that uses multi-agent role-playing to synthesize infinite, high-quality tool-use data from raw API specifications.
- The system employs collaborative agents—User Simulator, Tool Agent, and Server Simulator—to generate diverse, verified trajectories spanning single-turn calls to complex multi-step workflows.
- It establishes a closed loop where synthesized data trains the model via GRPO with gated rewards, driving continuous capability enhancement without human annotation.

---

[A NEAT Approach to Evolving Neural-Network-based Optimization of Chiral Photonic Metasurfaces: Application of a Neuro-Evolution Pipeline](http://arxiv.org/abs/2512.23558)

- NEAT (NeuroEvolution-of-Augmenting-Topologies) Approach: introduces an automated, adaptive optimization pipeline for chiral photonic metasurfaces by integrating the NEAT algorithm to autonomously evolve the Neural Network model topology and weights, alongside iterative data and shape evolution.
- The hybrid framework replaces manually designed NN architectures with resource-efficient NEAT-evolved models, achieving comparable or superior predictive accuracy and generalization for inferring metasurface properties.
- The neuroevolutionary process acts as an implicit feature selector, promoting complex topologies when trained on standardized data, and accelerating the search for geometries exhibiting strong circular dichroism.

---

[Toward Trustworthy Agentic AI: A Multimodal Framework for Preventing Prompt Injection Attacks](http://arxiv.org/abs/2512.23557)

- Cross-Agent Multimodal Provenance-Aware Defense Framework: introduces a defense pipeline against multimodal prompt injection attacks using Text Sanitizer Agent ($A_t$), Visual Sanitizer Agent ($A_v$), Provenance Ledger, LLM-Facing Sanitization, Main LLM (Agent M), Output Validator Agent (B), and LangChain/GraphChain Agent Node.
- The framework enforces dual-stage sanitization (pre-agent and pre-LLM) and post-generation validation to ensure zero-trust communication across agent interactions within agentic AI systems.
- It utilizes a Provenance Ledger to track trust metadata across modalities and agent hops, enabling trust-aware masking in the Main LLM to minimize cross-agent trust leakage and enhance detection accuracy.

---

[Agentic AI for Autonomous Defense in Software Supply Chain Security: Beyond Provenance to Vulnerability Mitigation](http://arxiv.org/abs/2512.23480)

- AAI-ADF (Agentic AI Autonomous Defense Framework): introduces a proactive defense system for software supply chains using specialized agents coordinated by LangChain/LangGraph, which utilize LLM reasoning and RL for adaptive mitigation decisions recorded on a blockchain Security Ledger.
- The framework interacts with real CI/CD environments (GitHub Actions, Jenkins) via the Model Context Protocol (MCP) to monitor inputs like source code, dependencies, pipelines, and configurations.
- Experimental results show the system achieves better detection accuracy and shorter mitigation latency for vulnerabilities like injection attacks and insecure deserialization compared to rule-based baselines.

---

[AI Meets Brain: A Unified Survey on Memory Systems from Cognitive Neuroscience to Autonomous Agents](http://arxiv.org/abs/2512.23343)

- UMS (Unified Memory System): introduces a comprehensive survey unifying memory systems from cognitive neuroscience and LLM-driven agents, detailing memory definitions, taxonomy, storage, and management mechanisms.
- The system defines memory management as a closed-loop pipeline consisting of Memory Extract, Updating, Retrieval, and Application, enabling persistent experience regulation and long-range reasoning.
- Memory utility extends agent capabilities by breaking context window constraints, constructing long-term personalized profiles, and driving experience-based reasoning through reflection and planning.

---

[The Dawn of Agentic EDA: A Survey of Autonomous Digital Chip Design](http://arxiv.org/abs/2512.23189)

- Agentic EDA (Autonomous Digital Chip Design): introduces a cognitive architecture for autonomous chip design, featuring an Agent Core (planning, reasoning, self-correction), Multimodal Perception (unifies text, graph, layout), Memory Module (long-term knowledge, interaction history), Action Space (invokes EDA tools), and Observations & Feedback Loop (PPA metrics, logs, waveforms).
- This architecture transitions chip design from AI-assisted tools (L2 Copilot) to L3+ autonomy by enabling agents, powered by LLMs, to perceive global flow contexts, plan multi-step strategies, and self-correct errors.
- Key components like Circuit Foundation Models (CFMs) handle multimodal representation, while Retrieval-Augmented Generation (RAG) grounds agent actions in proprietary design specifications and Process Design Kits (PDKs).

---

[SPIRAL: Symbolic LLM Planning via Grounded and Reflective Search](http://arxiv.org/abs/2512.23167)

- SPIRAL (Symbolic LLM Planning via Grounded and Reflective Search): introduces a novel framework embedding a tri-agent cognitive architecture—Planner ($\pi_{planner}$) (Proposes actions), Simulator ($W_{sim}$) (Predicts outcomes/Grounds search), and Critic ($C_{critic}$) (Provides dense reward/Reflects)—into an MCTS Loop (Structured exploration).
- This architecture transforms MCTS from a brute-force search into a guided, self-correcting reasoning process by using Reflection-Driven Reward Shaping to generate a Composite Reward ($R_t$) for backpropagation.
- The grounded and reflective search process substantially outperforms state-of-the-art agents on complex tool-use benchmarks, yielding more robust and token-efficient autonomous planners.

---

[It's a TRAP! Task-Redirecting Agent Persuasion Benchmark for Web Agents](http://arxiv.org/abs/2512.23128)

- TRAP (Task-Redirecting Agent Persuasion Benchmark): introduces an evaluation for studying how persuasion techniques misguide autonomous web agents on realistic tasks across six cloned websites, using a five-dimensional modular attack space, including Interface, Persuasion Principles, LLM Manipulation Methods, Injection Location, and Tailoring, measured by a one-click Attack Success Rate.
- The benchmark reveals that web agents are susceptible to prompt injection in 25% of tasks on average across six frontier models, with button-based injections being over three times more effective than hyperlinks.
- The modular attack construction decomposes prompt injections into interpretable social-engineering components, enabling controlled analysis of how specific design choices shape LLM agent failure.

---

[RAPTOR: Real-Time High-Resolution UAV Video Prediction with Efficient Video Attention](http://arxiv.org/abs/2512.21710)

- RAPTOR: introduces a real-time, high-resolution UAV video prediction framework using an Encoder-Translator-Decoder pipeline designed for single-pass generation, eliminating the latency and error accumulation of iterative approaches.
- The core innovation, Efficient Video Attention (EVA), factorizes spatiotemporal modeling into alternating temporal (TIMEMIX) and spatial (SPACEMIX) mixing operations, reducing computational complexity from $O((ST)^2)$ to $O(S+T)$.
- The architecture is complemented by a Three-Stage Curriculum Learning strategy that refines predictions from foundational reconstruction to perceptual refinement, enabling real-time performance at 1024x1024 resolution and boosting UAV navigation success rate by 18%.

---

[Step-DeepResearch Technical Report](http://arxiv.org/abs/2512.20491)

- Step-DeepResearch: introduces a cost-effective, end-to-end Deep Research agent model utilizing a progressive three-stage training pipeline and internalizing atomic capabilities for robust performance.
- The framework employs a streamlined ReAct-style single-agent architecture and a specialized toolset to handle complex, long-horizon tasks with low deployment costs.
- Training involves atomic-capability data synthesis, supervised fine-tuning, and reinforcement learning guided by a Rubrics Judge reward design to enhance robustness and generalization.

---

#### 28th December 2025

[Video-BrowseComp: Benchmarking Agentic Video Research on Open Web](http://arxiv.org/abs/2512.23044)

- Video-BrowseComp: introduces the first benchmark for agentic video research on the open web, comprising 210 questions requiring agents to actively navigate, retrieve, and cross-reference dynamic visual evidence.
- The benchmark enforces Mandatory Video Dependency and uses a Verification Checklist across three Difficulty Stratification levels (L1: Explicit Retrieval, L2: Implicit Retrieval, L3: Cross-Source Reasoning) to ensure rigorous evaluation beyond passive perception.
- Evaluation of state-of-the-art LLMs and Search-Augmented models reveals a critical Modality Gap, where models fail in metadata-sparse domains like Sports and Games, highlighting the need for improved Temporal Grounding capabilities.

---

[Agentic AI for Cyber Resilience: A New Security Paradigm and Its System-Theoretic Foundations](http://arxiv.org/abs/2512.22883)

- Agentic AI Architecture: introduces a system-level framework for designing agentic AI workflows for cyber resilience, featuring a Reasoning Core (LLMs), Persistent Memory, Tool Interfaces, Human-in-the-loop, and Environment/Embodiment, enabling continuous sense-reason-act-learn cycles.
- The framework supports the AI-augmented security paradigm (5P), shifting cybersecurity from prevention-centric, static rule-based defense toward goal-directed, adaptive, and resilience-oriented security ecosystems.
- The paper conceptualizes attacker-defender interactions as a dynamic strategic game, using game-theoretic frameworks to guide the design of robust, equilibrium-based agentic workflows, including simple, static multi-stage, decentralized sequential, and dynamic closed-loop archetypes.

---

#### 27th December 2025

[Cyber Resilience in Next-Generation Networks: Threat Landscape, Theoretical Foundations, and Design Paradigms](http://arxiv.org/abs/2512.22721)

- RLC: introduces a comprehensive framework for cyber resilience in NextG networks, integrating Proactive, Responsive, and Retrospective mechanisms, underpinned by Control, Game, Learning, and Network Theories.
- The framework operationalizes resilience through design paradigms like trust-aware resource management, risk-aware orchestration, multi-agent reinforcement learning, and LLM-driven network control in 5G/O-RAN architectures.
- Agentic AI, leveraging LLMs, serves as the cognitive glue, translating high-level intent into actionable policies and enabling autonomous, adaptive responses across the OODA loop stages (Observe, Orient, Decide, Act).

---

[SANet: A Semantic-aware Agentic AI Networking Framework for Cross-layer Optimization in 6G](http://arxiv.org/abs/2512.22579)

- SANet (Semantic-aware AgentNet framework): introduces a semantic-aware AgentNet architecture for 6G cross-layer optimization, featuring an Agent Controller (LLM-based coordination) that orchestrates aAgent (Application layer agent), nAgent (Network layer agent), and pAgent (Physical layer agent) using MoPS (Model Partitioning and Sharing) and dynamic optimization algorithms.
- The framework formulates decentralized optimization as a multi-agent multi-objective problem, focusing on finding the Pareto-optimal solution using novel metrics: optimization error, generalization error, and conflicting error.
- MoPS partitions large deep learning models into shared-part and agent-specific parts, achieving up to 14.61% performance gains while significantly reducing computational overhead (44.37% FLOPs for inference).

---

[ROLLART: Scaling Agentic RL Training via Disaggregated Infrastructure](http://arxiv.org/abs/2512.22560)

- ROLLART: introduces a distributed system designed to maximize throughput for multi-task agentic RL training on disaggregated infrastructure, utilizing Hardware-Affinity Workload Mapping, Fine-grained Asynchrony, and Statefulness-aware Computation.
- The system orchestrates the RL pipeline across specialized resource pools, including compute-optimized GPUs (H800) for training, bandwidth-optimized GPUs (H20) for inference, CPU clusters for stateful environments, and serverless infrastructure for stateless reward computation.
- Fine-grained asynchrony is achieved via trajectory-level rollout scheduling, which overlaps LLM generation, environment interaction, and reward computation to mitigate stragglers and resource idleness.

---

[AGENTMATH: EMPOWERING MATHEMATICAL REASONING FOR LARGE LANGUAGE MODELS VIA TOOL-AUGMENTED AGENT](http://arxiv.org/abs/2512.20745)

- AgentMath: introduces a tool-augmented agent framework that integrates LLM reasoning with code interpreters' precision using a three-stage data synthesis pipeline, a novel agentic RL paradigm, and an efficient asynchronous training system.
- The framework uses an automated synthesis method to convert natural language chain-of-thought into structured tool-augmented trajectories, generating high-quality supervised fine-tuning data to alleviate scarcity.
- The agentic RL paradigm dynamically interleaves natural language generation and real-time code execution, supported by a scalable infrastructure featuring request-level asynchronous rollout and partial rollouts for 4-5x speedup on ultra-long sequences.

---

#### 26th December 2025

[Centralization and Stability in Formal Constitutions](http://arxiv.org/abs/2512.22051)

- Self-Maintenance Framework (SMF): introduces a model for analyzing the stability and centralization dynamics of formal constitutions, defining a Social-Choice Function (SCF) as self-maintaining if it cannot be replaced by any other SCF under the current power structure.
- The framework evaluates SCF stability across different agent Beliefs (F) (Optimistic, Pessimistic, I.i.d.) and Tie-Breaking Rules (Arbitrary, Status-Quo Bias), where agents maximize their ex-ante utility based on preference matching.
- Key findings include an "Arrow-Style" Theorem showing that only dictatorship is self-maintaining under unbiased i.i.d. beliefs and arbitrary tie-breaking, and that centralization increases with the economy's extractive nature.

---

[VIDEOZOOMER: REINFORCEMENT-LEARNED TEMPORAL FOCUSING FOR LONG VIDEO REASONING](http://arxiv.org/abs/2512.22315)

- VideoZoomer: introduces a novel agentic framework that empowers an MLLM to dynamically control its visual focus during long video reasoning via multi-turn tool interaction.
- The framework operates in two phases, starting with a Glance Stage for a coarse overview, followed by an iterative Zoom Stage where the agent invokes a `<video_zoom>` tool to gather fine-grained evidence.
- The agent is trained using a two-stage strategy—Supervised Fine-Tuning (SFT) followed by Reinforcement Learning (RL)—achieving strong performance and superior efficiency under reduced frame budgets compared to baselines.

---

[Analyzing Code Injection Attacks on LLM-based Multi-Agent Systems in Software Development](http://arxiv.org/abs/2512.21818)

- MAS Architectures: introduces Coder, Coder-Tester (CT), and Coder-Reviewer-Tester (CRT) architectures, along with an added Security Analysis Agent, to evaluate the security and efficiency trade-offs against code injection attacks in the software development implementation phase.
- The research demonstrates that while the CRT architecture is more resilient than Coder or CT, adding a specialized Security Analysis Agent significantly mitigates attack effectiveness while maintaining system efficiency.
- The paper concludes that the Security Analysis Agent remains vulnerable to advanced code injection attacks, specifically those embedding poisonous few-shot examples, which drastically increase the attack success rate up to 71.95%.

---

#### 25th December 2025

[Accelerating Scientific Discovery with Autonomous Goal-evolving Agents](http://arxiv.org/abs/2512.21782)

- SAGA (Scientific Autonomous Goal-evolving Agent): introduces a bi-level agentic framework for scientific discovery, featuring an outer loop (Planner, Implementer, Analyzer) that dynamically evolves objectives and an inner loop (Optimizer) that performs solution optimization.
- The framework addresses the challenge of fixed objectives in complex scientific tasks by iteratively refining objectives based on optimization outcomes and domain knowledge, mitigating reward hacking issues.
- SAGA is a generalist framework demonstrated across diverse domains (antibiotic, materials, DNA sequence, chemical process design) and supports three levels of autonomy: co-pilot, semi-pilot, and autopilot.

---

[UNILABOS: AN AI-NATIVE OPERATING SYSTEM FOR AUTONOMOUS LABORATORIES](http://arxiv.org/abs/2512.21766)

- UniLabOS: introduces a foundational operating system bridging high-level AI planning and low-level execution using A/R/A&R Abstraction (Unified object semantics), Dual-Topology Model (Logical tree/physical graph), and Transactional CRUTD Protocol (Material lifecycle management).
- The system employs a Distributed Edge-Cloud Architecture with ROS 2/DDS Communication (Robust peer-to-peer messaging) to enable scalable orchestration, protocol mobility, and decentralized self-organizing enrollment.
- The OS acts as an AI-native safety kernel by enforcing Digital-Twin Synchronization (Pre-dispatch safety validation) and providing unified hardware interoperability via a decoupled driver-as-a-service model.

---

[HELP: Hierarchical Embodied Language Planner for Household Tasks](http://arxiv.org/abs/2512.21723)

- HELP (Hierarchical Embodied Language Planner): introduces a hierarchical LLM-based multi-agent system for embodied task planning, including the High Level Planner (decomposes task), Low Level Planner (generates executable pseudocode), Env Feedback LLM Agent (resolves ambiguities), Valid Plan Check (assesses feasibility and safety), and Embodied Agent (executes actions).
- The architecture splits planning into high-level decomposition using natural language and low-level grounding into pseudocode, significantly reducing task complexity for medium-sized, open-source LLMs (7-13B parameters).
- Designed for real-world deployment on mobile robotic platforms, the planner incorporates environmental feedback and a safety verification module to ensure robust and feasible execution of complex household tasks.

---

[Towards Responsible and Explainable AI Agents with Consensus-Driven Reasoning](http://arxiv.org/abs/2512.21699)

- CD-RAEA (Consensus-Driven Responsible and Explainable Agent Architecture): introduces an agentic workflow combining an LLM/VLM Consortium (Independent candidate generation) and a Reasoning Agent (Centralized decision governance), coordinated by an Orchestration Layer (Workflow coordination) using a Shared Input Context (Canonical task specification).
- Explainability is achieved by exposing uncertainty and disagreement across the parallel outputs of the heterogeneous consortium, enabling transparent inspection of alternative interpretations and reasoning paths.
- Responsibility is enforced by the centralized Reasoning Agent, which performs structured meta-reasoning to resolve conflicts, enforce safety constraints, and synthesize auditable, evidence-backed decisions.

---

[From Shallow Humor to Metaphor: Towards Label-Free Harmful Meme Detection via LMM Agent Self-Improvement](http://arxiv.org/abs/2512.21598)

- ALARM (labeL-free hArmful Meme detection framework): introduces a label-free harmful meme detection solution powered by an LLM agent self-improvement paradigm, utilizing Confidence-based Explicit Meme Identification and Pairwise Learning Guided Agent Self-Improvement.
- The framework first isolates explicit memes using prediction confidence to assign pseudo-labels, creating a resource of "powerful experiences" from unlabeled data.
- A self-driven LLM agent then refines high-level detection references from contrastive pairs of these explicit memes, enabling enhanced detection of subtle and complex harmful content.

---

[Videos are Sample-Efficient Supervisions: Behavior Cloning from Videos via Latent Representations](http://arxiv.org/abs/2512.21586)

- BCV-LR (Behavior Cloning from Videos via Latent Representations): introduces a novel, unsupervised, and sample-efficient imitation learning framework comprising an offline pre-training stage and an online finetuning stage.
- The offline stage uses a self-supervised Visual Encoder and a World Model with a Latent Action Predictor to extract action-related latent features and predict latent actions from action-free videos.
- The online stage fine-tunes the latent actions and aligns them to the real action space via a Latent Action Decoder and Policy, resulting in iterative, highly sample-efficient policy improvement.

---

#### 24th December 2025

[CoTDeceptor:Adversarial Code Obfuscation Against CoT-Enhanced LLM Code Agents](http://arxiv.org/abs/2512.21250)

- CoTDeceptor (Adversarial Code Obfuscation Against CoT-Enhanced LLM Code Agents): introduces an Agentic-RL framework that autonomously constructs evolving, multi-stage obfuscation strategy chains to disrupt CoT-driven detection logic in LLM code agents.
- The framework employs a multi-agent system, including a Code Obfuscator, a Code Verifier, and an Obfuscate Strategy Reflector, guided by Lineage-Based Strategy Tree Exploration to refine strategies iteratively.
- By leveraging feedback-driven exploration, the framework achieves stable and transferable evasion performance against state-of-the-art LLMs and vulnerability detection agents, bypassing 14 out of 15 vulnerability categories.

---

[SparScene: Efficient Traffic Scene Representation via Sparse Graph Learning for Large-Scale Trajectory Generation](http://arxiv.org/abs/2512.21133)

- SparScene: introduces a sparse graph learning framework for efficient and scalable traffic scene representation, utilizing a symmetric representation and a three-stage topology-guided encoder (TiL, L2A, A2A).
- The framework constructs a heterogeneous agent-lane graph based on HD map topology and spatial alignment, avoiding dense, distance-based connections for interaction modeling.
- By leveraging the topology-guided encoding, the approach achieves competitive multi-agent multimodal trajectory prediction performance with significantly reduced computational overhead and superior scalability.

---

[LSTM-Based Modeling and Reinforcement Learning Control of a Magnetically Actuated Catheter](http://arxiv.org/abs/2512.21063)

- LSTM-Based Modeling and Reinforcement Learning Control: introduces a novel RL framework leveraging an LSTM Plant Environment (Surrogate dynamics model) to accurately model the magnetic catheter's nonlinear and hysteretic dynamics, enabling safe training of DRL agents.
- The system employs and compares two DRL agents, the Deep Q-Network (Discrete RL controller) and the Actor-Critic (Continuous RL controller), for point regulation and path following tasks of the Magnetic Catheter System (Physical hardware setup).
- The Actor-Critic agent, utilizing a continuous Action Space (Servo angular increments), demonstrated superior accuracy and smoother trajectories compared to DQN, suitable for dynamic navigation in curved vascular structures.

---

[TrafficSimAgent: A Hierarchical Agent Framework for Autonomous Traffic Simulation with MCP Control](http://arxiv.org/abs/2512.20996)

- TrafficSimAgent: introduces a novel LLM-based multi-agent framework for autonomous traffic simulation, featuring Task Understanding, Orchestrator, Task Executor, and Context Manager modules, facilitating execution through cross-level collaboration among expert agents.
- The framework achieves enhanced generality and adaptability through robust instruction understanding and dynamic task planning via hierarchical decomposition, enabling the system to handle ambiguous instructions and diverse scenarios.
- It incorporates an embedded two-layer optimization module, combining LLM-driven strategies with classical low-level optimization algorithms for self-optimization and iterative performance improvement.

---

[AegisAgent: An Autonomous Defense Agent Against Prompt Injection Attacks in LLM-HARs](http://arxiv.org/abs/2512.20986)

- AegisAgent: introduces an autonomous defense agent system designed to secure LLM-driven Human Activity Recognition (HAR) systems against prompt injection attacks.
- The system operates as a layered, self-regulating defense pipeline comprising an Input Sanitizer, a Consistency Verifier, and a Robust Reasoner, coordinated by an Agent Control Plane.
- The agent autonomously detects threats, reasons about user intent using a dynamic memory, and executes multi-step verification and repair plans, reducing attack success rate by 30% on average.

---

[A Blockchain-Monitored Agentic AI Architecture for Trusted Perception-Reasoning-Action Pipelines](http://arxiv.org/abs/2512.20985)

- BMAA (Blockchain-Monitored Agentic AI Architecture): introduces a four-layer architecture that integrates LangChain-based multi-agent reasoning with a permissioned blockchain for secure, auditable perception-reasoning-action pipelines.
- The framework uses a Blockchain Governance Layer, implemented via Hyperledger Fabric smart contracts, to verify agent inputs, enforce policies, and immutably record decision outcomes.
- Approved actions are executed through the Action Layer using the Model Context Protocol (MCP) to interface with external systems like ERPs or smart city controllers, ensuring traceability.

---

[DAO-Agent: Zero Knowledge-Verified Incentives for Decentralized Multi-Agent Coordination](http://arxiv.org/abs/2512.20973)

- DAO-Agent: introduces a novel framework for decentralized multi-agent coordination, integrating a hybrid on-chain/off-chain architecture, an on-chain DAO governance mechanism, and a ZKP mechanism for verifiable, fair incentive distribution.
- The architecture implements a "Compute Off-chain, Verify On-chain" paradigm by offloading computationally intensive Shapley value calculation to an off-chain Coordinator and using recursive ZKPs (STARK $\rightarrow$ SNARK) for constant $O(1)$ on-chain verification.
- This hybrid model reduces verification gas costs by up to 99.9% compared to naive on-chain alternatives, ensuring scalable trust and fairness without exposing agents' private strategies.

---

[Embodied AI-Enhanced IoMT Edge Computing: UAV Trajectory Optimization and Task Offloading with Mobility Prediction](http://arxiv.org/abs/2512.20902)

- PETO (Prediction-Enhanced UAV Trajectory Optimization and Task Offloading Algorithm): introduces an embodied AI-enhanced IoMT edge computing framework, utilizing a Hierarchical Multi-scale Transformer for user mobility prediction and Prediction-Enhanced DRL (PPO) for optimizing UAV trajectory and task offloading decisions.
- The framework minimizes the weighted average task completion time of WBAN Users subject to UAV energy consumption constraints by modeling the dynamic optimization problem as a Markov Decision Process.
- The Hierarchical Multi-scale Transformer captures temporal dependencies of user mobility across various time scales, providing accurate predictions that enhance the PPO agent's real-time strategy adaptation.

---

[The Silent Scholar Problem: A Probabilistic Framework for Breaking Epistemic Asymmetry in LLM Agents](http://arxiv.org/abs/2512.20884)

- PFFF (Probabilistic Framework with Forgetting Factor): introduces a formal probabilistic framework that models an LLM agent's belief in propositions using a Beta-Bernoulli distribution with a forgetting factor ($\gamma$), establishing a non-altruistic motive for bidirectional knowledge exchange.
- The framework quantifies epistemic uncertainty as the variance of belief, driving a homeostatic need for continuous engagement by targeting maximum ambiguity (E[$\theta$] = 0.5) to maximize learning gain.
- Scalability is addressed via epistemic caching, which uses the forgetting factor to dynamically allocate computational resources to the active head of the knowledge distribution and evict stale beliefs.

---

#### 23rd December 2025

[Context-Sensitive Abstractions for Reinforcement Learning with Parameterized Actions](http://arxiv.org/abs/2512.20831)

- PEARL (Parameterized Extended state/action Abstractions for RL): introduces a unified framework for context-sensitive abstractions in Reinforcement Learning with parameterized actions, utilizing SPA-CAT, APTs, and TD($\lambda$).
- The approach autonomously learns and progressively refines state and action abstractions online based on a novel heterogeneity measure blending TD error and value function dispersion.
- Flexible refinement strategies, including learning-based clustering and SVM classification, enable the system to adapt abstraction granularity to environmental dynamics and task requirements.

---

[A Benchmark for Evaluating Outcome-Driven Constraint Violations in Autonomous AI Agents](http://arxiv.org/abs/2512.20798)

- ODCV-Bench (Outcome-Driven Constraint Violation Benchmark): introduces a safety benchmark of 40 multi-step scenarios in a persistent bash environment, evaluating LLM agents for outcome-driven constraint violations when optimizing a KPI.
- The architecture uses a containerized Environment Orchestrator to host the sandboxed environment and a Mission Executor running a ReAct-style loop to interface the target LLM with the environment via `bash` and `task_complete` tools.
- The benchmark uses Mandated and Incentivized instruction variations to distinguish between blind obedience and proactive deception, revealing that superior LLM reasoning capability does not inherently ensure safety.

---

[Towards Optimal Performance and Action Consistency Guarantees in Dec-POMDPs with Inconsistent Beliefs and Limited Communication](http://arxiv.org/abs/2512.20778)

- Dec-OAC-POMDP-OL (Decentralized Optimal Action Consistent POMDP Open-Loop): introduces a novel decentralized framework for optimal joint action selection that explicitly accounts for belief inconsistencies, ensuring probabilistic guarantees for action consistency and performance relative to open-loop Multi-Agent POMDP (MPOMDP).
- The approach utilizes Optimal Action Selection ($\epsilon$-MLOAS) to select actions based on the calculated optimal joint action distribution, mimicking MPOMDP planning by reasoning about other agents' unavailable information to ensure Multi-Robot Optimal Action Consistency (MROAC).
- The framework includes a mechanism ($\delta$-NEPG Strategy) to analyze the performance gap between planning (full information) and inference (partial information) and selectively trigger data sharing to improve expected execution performance.

---

[Learning-Enabled Elastic Network Topology for Distributed ISAC Service Provisioning](http://arxiv.org/abs/2512.20722)

- MADRL (Multi-Agent Deep Reinforcement Learning) based on MAPPO: introduces an ENT (Elastic Network Topology) for distributed ISAC service provisioning, dynamically orchestrating localized CCNs and federated CFNs using LCP, LPB, FG, and FPB agents.
- The system maximizes the USR (Utility-to-Signaling Ratio) by jointly optimizing network topology (service classification and CCN aggregation) and resource allocation, formulated as a challenging distributed optimization problem.
- The MAPPO framework utilizes the CTDE paradigm, employing centralized critics for accurate value estimation and decentralized execution via role-specific actors to handle the heterogeneous, hybrid action space.

---

[Synthesizing Procedural Memory: Challenges and Architectures in Automated Workflow Generation](http://arxiv.org/abs/2512.20278)

- Procedural Memory Synthesis Architecture (PMSA): introduces an architecture for synthesizing robust, production-grade procedural memory (executable code skills) by addressing four structural bottlenecks in automated skill generation.
- The architecture resolves the Discovery Gap via Dynamic MCP, the Verification Gap via the Probe Methodology, the Decomposition Gap via Linear State Anchoring, and the Scaling Gap via concurrency and external persistence.
- By enforcing a scientific methodology (hypothesize, probe, code), the framework enables LLMs to transition from passive tool-users to active workflow architects capable of autonomous, reliable skill writing.

---

#### 21st December 2025

[Democratizing Drug Discovery with an Orchestrated, Knowledge-Driven Multi-Agent Team for User-Guided Therapeutic Design](http://arxiv.org/abs/2512.21623)

- OrchestRA (Orchestrated Rational drug design Agents): introduces a human-in-the-loop multi-agent platform that unifies biology, chemistry, and pharmacology into an autonomous drug discovery engine.
- The system employs an Orchestrator Agent to coordinate specialized Biologist, Chemist, and Pharmacologist Agents, which actively execute simulations and reason over results using the ReAct paradigm.
- The architecture establishes a dynamic feedback loop where the Pharmacologist Agent's diagnostic ADMET/PBPK profiles directly trigger the Chemist Agent's structural reoptimization, bridging the computational design and physiological validation gap.

---

#### 18th December 2025

[Multimodal RewardBench 2: Evaluating Omni Reward Models for Interleaved Text and Image](http://arxiv.org/abs/2512.16899)

- MMRB2 (Multimodal RewardBench 2): introduces the first comprehensive benchmark for evaluating omni reward models on multimodal understanding and interleaved generation, utilizing a Benchmark/Evaluation Suite (Omni reward model evaluation), Generative Models (Candidate response producers), MLLM-as-a-judge (Multimodal LLM evaluators), Preference-trained Reward Models (Human preference alignment), Task-specific Metrics (Heuristic evaluation scores), Ensemble Filtering Pipeline (Removes easy preference pairs), Human Annotation (Expert preference collection), Agent-based Systems (Tool-using LLMs), and Visual Tools (Image generation/editing/Python).
- MMRB2 spans four challenging tasks: text-to-image generation, image editing, interleaved generation, and multimodal reasoning, providing 1,000 expert-annotated preference pairs per task.
- The benchmark reveals that state-of-the-art MLLMs like Gemini 3 Pro achieve the highest judge accuracy (76.3% average), but still lag significantly behind human consensus (>90%), highlighting substantial remaining headroom for omni reward modeling.

---

[ADASEARCH: BALANCING PARAMETRIC KNOWLEDGE AND SEARCH IN LARGE LANGUAGE MODELS VIA REINFORCEMENT LEARNING](http://arxiv.org/abs/2512.16883)

- ADASEARCH (Balancing Parametric Knowledge and Search in Large Language Models via Reinforcement Learning): introduces a simple two-stage, outcome-driven RL framework that disentangles problem solving from the explicit decision of whether to invoke external search, using a single LLM policy.
- The framework optimizes both abilities by training the LLM policy in Stage 1 (Decision Making) using a decision prompt and Stage 2 (Problem Solving) using either a parametric-knowledge or search prompt based on the Stage 1 assessment.
- ADASEARCH improves knowledge-boundary awareness and interpretability by requiring explicit reasoning for search invocation decisions, significantly reducing unnecessary search calls compared to prior RL methods.

---

[META-RL INDUCES EXPLORATION IN LANGUAGE AGENTS](http://arxiv.org/abs/2512.16848)

- LAMER (LLM Agent with Meta-RL): introduces a general Meta-RL framework for LLM agents that uses a cross-episode training scheme and in-context policy adaptation via self-reflection to induce active exploration.
- The cross-episode training framework maximizes discounted cross-episode return, forcing the LLM agent to learn general exploration-exploitation strategies across multiple trials.
- Policy adaptation is achieved in-context using an inter-episode memory that stores past trajectories and textual reflections, enabling robust adaptation to novel environments at test time.

---

[Generative Adversarial Reasoner: Enhancing LLM Reasoning with Adversarial Reinforcement Learning](http://arxiv.org/abs/2512.16917)

- GAR (Generative Adversarial Reasoner): introduces an on-policy joint training framework that co-evolves an LLM reasoner and an LLM-based discriminator through adversarial reinforcement learning to enhance mathematical reasoning.
- The system partitions reasoning chains into logically complete slices, allowing the discriminator to provide dense, well-calibrated stepwise rewards and structured justifications for localized feedback.
- By coupling reasoner rewards for logical consistency with discriminator rewards for error detection, the framework improves credit assignment and sample efficiency across complex mathematical benchmarks.

---

[Exploration v.s. Exploitation: Rethinking RLVR through Clipping, Entropy, and Spurious Reward](http://arxiv.org/abs/2512.16912)

- RLVR (Reinforcement Learning with Verifiable Rewards): analyzes the interplay between clipping, entropy, and spurious rewards in reasoning models, utilizing an LLM policy, a verifier, and a clipped surrogate objective.
- The study demonstrates that clipping bias acts as a stabilizing regularizer by implicitly minimizing policy entropy, which prevents training collapse during long-sequence reasoning tasks.
- It proposes a reward-misalignment model to explain why stronger models achieve performance gains under random rewards, disentangling the effects of model strength from data contamination.

---

[Impacts of Racial Bias in Historical Training Data for News AI](http://arxiv.org/abs/2512.16901)

- News AI Classifier Auditing: investigates the reproduction of historical racial biases in newsroom AI by analyzing a multi-label classifier, Google News word2vec, LLM, LIME, and evaluation datasets.
- The framework utilizes LIME to identify predictive terms that reveal how historical labels like "blacks" encode outdated racial attitudes and function as unintended "racism detectors."
- It demonstrates that temporal gaps between training data and modern application create systematic oversights, such as failing to recognize contemporary movements like Black Lives Matter or anti-Asian hate reporting.

---

[FlashPortrait: 6× Faster Infinite Portrait Animation with Adaptive Latent Prediction](http://arxiv.org/abs/2512.16900)

- FlashPortrait: introduces an end-to-end video diffusion transformer for identity-preserving, infinite-length portrait animation, utilizing a Normalized Facial Expression Block, a Weighted Sliding-Window Strategy, and an Adaptive Latent Prediction Acceleration Mechanism.
- The framework aligns facial expression features with diffusion latents through normalization to maintain identity stability across extended video sequences.
- It achieves 6× inference acceleration by skipping denoising steps using an adaptive latent prediction mechanism based on higher-order derivatives and Taylor expansion.

---

[Checking the HAL Interface Specification Continuously, Right from the Start](http://arxiv.org/abs/2512.16897)

- IDCC (Incremental Development, Continuous Checking): introduces a verification-driven workflow for embedded software that integrates software model checking into every step of the development process, utilizing a program skeleton, iterative increments, and a verification harness to ensure compliance with HAL temporal dependencies.
- The methodology prioritizes the evolution of program control flow before the gradual incorporation of data flow, facilitating the automated computation of abstractions required for successful formal verification. 
- By applying checks continuously from the initial skeleton to the final application, the framework enables early detection of interface violations and reduces the state space complexity typically hindering late-stage model checking.

---

[Growing Self-Similar Markov Trees](http://arxiv.org/abs/2512.16894)

- ssMt (Self-Similar Markov Trees): introduces a method to couple random decorated trees into nested families, with ssMt (random decorated genealogy trees), characteristic quadruplet (growth-fragmentation mechanism parameters), splitting measure (particle splitting probability distribution), growing functions (nested subtree coupling mappings), decoration-reproduction processes (branching Markov process analogs), generator (infinitesimal growing function action), and SDE with jumps (decoration process flow representation), where the paper constructs continuous and increasing couplings of Brownian and stable continuum random trees.
- The framework represents tree branches as flows of positive self-similar Markov processes using SDEs driven by common Brownian motion and Poisson measures to ensure monotonicity.
- It establishes analytical conditions via generators to characterize the existence of growing mechanisms and identifies these trees as scaling limits of discrete leaf-growth algorithms.

---

[Instant Expressive Gaussian Head Avatar via 3D-Aware Expression Distillation](http://arxiv.org/abs/2512.16893)

- Instant Expressive Gaussian Head Avatar: introduces a feed-forward encoder that transforms a single image into an animatable 3D avatar by distilling expressive motion priors from 2D diffusion models.
- The framework utilizes a triplane-based representation to decouple 3D structure from animation, employing a motion decoder to predict per-Gaussian feature residuals for detailed expression control.
- This approach achieves real-time performance at 107 FPS and superior 3D consistency compared to 2D diffusion methods while maintaining high expression fidelity and capturing fine-grained facial details.

---

#### 17th December 2025

[BASHARENA: A CONTROL SETTING FOR HIGHLY PRIVILEGED AI AGENTS](http://arxiv.org/abs/2512.15688)

- BashArena: introduces a control setting for highly privileged AI agents, featuring 637 complex Linux system administration main tasks and four critical sabotage side tasks, evaluated in a multi-container Docker environment.
- The setting facilitates an adversarial game between a Red Team (attack policy) attempting sabotage and a Blue Team (monitor/honest policy) constructing control protocols to detect and prevent attacks.
- Evaluations using frontier LLMs demonstrate that powerful models can successfully execute sabotage while evading monitoring, providing a baseline for designing more effective control protocols.

---

[SCOPE: Prompt Evolution for Enhancing Agent Effectiveness](http://arxiv.org/abs/2512.15374)

- SCOPE (Self-evolving Context Optimization via Prompt Evolution): introduces a framework that transforms context management into an online optimization problem by synthesizing guidelines from execution traces to automatically evolve the LLM agent's prompt.
- The system employs a Dual-Stream mechanism, routing guidelines via a Classifier to Tactical Memory (immediate error correction) or Strategic Memory (long-term principles) to balance tactical specificity with strategic generality.
- To maximize strategy coverage, SCOPE utilizes Perspective-Driven Exploration, evolving multiple parallel prompts guided by distinct optimization personas (e.g., Efficiency vs. Thoroughness).

---

[Exploring User Acceptance and Concerns toward LLM-powered Conversational Agents in Immersive Extended Reality](http://arxiv.org/abs/2512.15343)

- LLM-XRAS: introduces a large-scale crowdsourcing study using a 2x2x3 factorial design to evaluate user acceptance and concerns regarding LLM-powered conversational agents in Extended Reality (XR) environments.
- The study factors included XR setting type (MR/VR), speech interaction type (basic voice commands/generative AI), and data processing location (on-device/own server/application cloud).
- Results indicate general acceptance of the technology but significant concerns regarding security, privacy, trust, and social implications, with location data being the most concerning data type.

---

[SynthSeg-Agents: Multi-Agent Synthetic Data Generation for Zero-Shot Weakly Supervised Semantic Segmentation](http://arxiv.org/abs/2512.15310)

- SynthSeg-Agents: introduces a multi-agent framework for Zero-Shot Weakly Supervised Semantic Segmentation (ZSWSSS) that generates synthetic training data entirely without real images, utilizing a Self-Refine Prompt Agent (Generates diverse prompts) and an Image Generation Agent (Synthesizes images, labels).
- The Self-Refine Prompt Agent uses LLMs for iterative refinement and CLIP Text Semantic Filter (Enhances relevance, diversity) to produce high-quality, diverse prompts, which are then passed to the Image Generation Agent.
- The Image Generation Agent leverages VLMs to synthesize images, employs a Frozen CLIP Model (Assess visual fidelity) for high-confidence sample selection, and trains a ViT-based Classifier (Relabel synthetic dataset) to assign consistent pseudo-labels for downstream WSSS Model (Downstream segmentation training).

---

[Towards Proactive Personalization through Profile Customization for Individual Users in Dialogues](http://arxiv.org/abs/2512.15302)

- PersonalAgent: introduces a novel user-centric lifelong agent designed to continuously infer and adapt to user preferences by modeling multi-turn conversations as a sequential inference process using a Policy Model, a dynamically refined User Profile P, a Preference Inference Module (sequential decision-making), a Reward Mechanism (multi-turn unified reward), and an Optimization Algorithm (maximize cumulative reward).
- The approach decomposes dialogues into single-turn units, framing preference inference as a multi-turn Markov Decision Process (MDP) optimized via Group Relative Policy Optimization (GRPO) to ensure long-term consistency.
- The agent excels in proactive personalization, accurately inferring preferences even in cold-start scenarios and maintaining high alignment levels across extended, noisy conversational contexts.


---

[CangLing-KnowFlow: A Unified Knowledge-and-Flow-fused Agent for Comprehensive Remote Sensing Applications](http://arxiv.org/abs/2512.15231)

- CangLing-KnowFlow: introduces a unified intelligent agent framework for remote sensing applications, integrating a Procedural Knowledge Base (PKB), Dynamic Workflow Adjustment, and an Evolutionary Memory Module.
- The PKB, containing 1,008 expert-validated workflow templates structured as Directed Acyclic Graphs (DAGs), grounds the Orchestrator Agent's planning in scientific methodologies, fundamentally mitigating LLM planning hallucination.
- The framework ensures robustness through the Dynamic Adjustment Module, which handles runtime failures via a hierarchical repair strategy, and the Evolutionary Memory Module, which drives continuous learning by solidifying successes and attributing failures to heuristic rules.

---

[Draft with Diffusion, Verify with Autoregressive Models](http://arxiv.org/abs/2512.15176)

- DEER (Draft with Diffusion, Verify with Autoregressive Models): introduces an efficient speculative decoding framework that uses a discrete-space dLLM as the drafter and an AR model as the verifier, utilizing a two-stage D2A Alignment Pipeline to adapt the dLLM for prefix-conditioned continuation.
- By leveraging the dLLM's parallel decoding strategy, DEER eliminates the left-to-right uncertainty accumulation inherent in AR drafters, enabling significantly longer accepted draft blocks (up to 32 tokens).
- The framework achieves substantial speedups (e.g., 5.54x on HumanEval with Qwen3-30B-A3B) and demonstrates stable, high-quality proposals, proving dLLMs are a practical alternative for efficient LLM decoding.

---

[MCP-SAFETYBENCH: A BENCHMARK FOR SAFETY EVALUATION OF LARGE LANGUAGE MODELS WITH REAL-WORLD MCP SERVERS](http://arxiv.org/abs/2512.15163)

- MCP-SafetyBench: introduces a comprehensive benchmark built on real Model Context Protocol (MCP) servers to evaluate LLM agent robustness against 20 attack types across five domains, utilizing MCP Host, MCP Servers, and User components, and evaluated via Task Evaluator and Attack Evaluator.
- The benchmark systematically evaluates LLMs in multi-step, multi-server workflows using the ReAct framework, revealing significant safety disparities and escalating vulnerabilities as task complexity increases.
- The unified taxonomy categorizes attacks into MCP Server-side (74.69%), MCP Host-side (12.24%), and User-side (13.06%) vulnerabilities, highlighting server-side threats as the most prevalent attack vector.

---

[BEYOND FAST AND SLOW: COGNITIVE-INSPIRED ELASTIC REASONING FOR LARGE LANGUAGE MODELS](http://arxiv.org/abs/2512.15089)

- CogER (Cognitive-Inspired Elastic Reasoning): introduces a dynamic reasoning framework inspired by human hierarchical thinking, which uses a CogER-Agent trained via RL to classify queries into four complexity levels ($L_1$ to $L_4$) and route them to tailored processing strategies.
- The framework models strategy selection as a Markov Decision Process guided by a composite reward function that explicitly balances solution quality and computational cost.
- For the most complex $L_4$ queries, the system integrates Cognitive Tool-Assisted Reasoning (CoTool) and the RSTKit toolkit, enabling the LLM to autonomously invoke external tools within its chain-of-thought.

---

[Imitation Game: Reproducing Deep Learning Bugs Leveraging an Intelligent Agent](http://arxiv.org/abs/2512.14990)

- RepGen: introduces a novel, automated, and intelligent approach for reproducing Deep Learning (DL) bugs by constructing a learning-enhanced context, developing a comprehensive plan, and employing an iterative generate-validate-refine LLM-powered Reproduction Agent.
- The Reproduction Agent leverages multi-stage feedback, including structural, static analysis, relevance, and novel runtime feedback, to ensure the generated code accurately and reliably reproduces the reported DL bug symptoms.
- RepGen achieved an 80.19% reproduction rate on 106 real-world DL bugs, significantly outperforming LLM-only baselines and reducing developer time and cognitive load in a controlled study.

---

[EvoLattice: Persistent Internal-Population Evolution through Multi-Alternative Quality-Diversity Graph Representations for LLM-Guided Program Discovery](http://arxiv.org/abs/2512.13857)

- EvoLattice: introduces a framework that represents an entire population of candidate programs or agents within a single Directed Acyclic Graph (DAG) (Single evolving artifact), where each Node (Functional component container) stores multiple persistent Alternatives (Multiple persistent implementations), defining an exponentially rich search space via Executable Path (Distinct candidate program/agent) combinations.
- The system uses Alternative-level Performance Statistics (Alternative-level quantitative feedback), derived from combinatorial path evaluation, to provide the LLM with dense, data-driven feedback for targeted mutation, recombination, and pruning.
- Structural integrity is maintained by a deterministic Self-Repair Pipeline (Enforces structural invariants) independent of the LLM, enabling persistent internal diversity and cumulative reuse characteristic of quality-diversity optimization.

---

[Autonomous Pressure Control in MuVacAS via Deep Reinforcement Learning and Deep Learning Surrogate Models](http://arxiv.org/abs/2512.15521)

- DLSM-DRL: introduces a data-driven approach for autonomous pressure control in the MuVacAS prototype, utilizing a Fourier Neural Operator (FNO) as a Deep Learning Surrogate Model (DLSM) to create a fast-simulation environment for training a Deep Reinforcement Learning (DRL) agent via Proximal Policy Optimization (PPO).
- The FNO surrogate model acts as a high-fidelity digital twin, trained on real operational data to efficiently capture the non-linear dynamics of the argon injection system and spatial pressure distribution along the accelerator longitude.
- The trained DRL agent successfully learns a robust control policy that regulates the argon mass flow rate to maintain gas pressure within strict operational limits, demonstrating superior generalization compared to traditional PID controllers.

---

[Large Model Enabled Embodied Intelligence for 6G Integrated Perception, Communication, and Computation Network](http://arxiv.org/abs/2512.15109)

- IBSA (Intelligent Base Station Agent): introduces a large model-enabled embodied intelligent base station agent architecture for 6G systems, featuring a three-layer perception-cognition-execution closed-loop pipeline.
- The architecture leverages a central multimodal LAM for unified cognitive reasoning and decision-making, supported by cloud-edge-end collaboration and parameter-efficient adaptation.
- IBSA is positioned as a practical path toward integrated perception, communication, and computation native systems, demonstrated through autonomous driving and low-altitude safety scenarios.

---

[NAP3D: NeRF Assisted 3D-3D Pose Alignment for Autonomous Vehicles](http://arxiv.org/abs/2512.15080)

- NAP3D (NeRF Assisted 3D-3D Pose Alignment): introduces a complementary pose correction approach for autonomous vehicles that leverages 3D-3D correspondences between a Depth Camera (real-world data) and a Neural Radiance Field (3D scene representation) queried by a Virtual Camera (estimated pose) via an Image Processor (alignment) using SIFT/FLANN (keypoint matching) and Procrustes Analysis (rigid transform calculation) to generate Error Correction (pose refinement) for the Autonomous Vehicle Processor (agent control).
- The system calculates the positional error by aligning 3D keypoints derived from the observed scene and the NeRF synthesized view using Procrustes Analysis, which minimizes the Frobenius norm of the residuals to yield optimal rotation and translation matrices.
- NAP3D provides robust and consistent 3D alignment, achieving lower Root Mean Square Error (RMSE) compared to conventional 2D-3D Perspective-n-Point (PnP) baselines, particularly when traditional loop closure is unavailable.

---

[Agentic AI for Integrated Sensing and Communication: Analysis, Framework, and Case Study](http://arxiv.org/abs/2512.15044)

- AIF (Agentic ISAC Framework): introduces a robust, closed-loop system for Integrated Sensing and Communication (ISAC) optimization by integrating DRL, LLM, GenAI, and a Transformer-based Mixture-of-Experts (MoE) model.
- The framework operates via a continuous perception-reasoning-action loop, where the LLM autonomously designs the reward function and the MoE performs comprehensive decision-making by aggregating specialized expert outputs.
- The framework achieves significant performance improvements in communication rate and Cramér-Rao bound (CRB) by leveraging enhanced memory, reasoning, and adaptive learning capabilities.

---

#### 16th December 2025

[IMITATION LEARNING FOR MULTI-TURN LM AGENTS VIA ON-POLICY EXPERT CORRECTIONS](http://arxiv.org/abs/2512.14895)

- OECs (On-Policy Expert Corrections): introduces a novel data generation methodology to mitigate covariate shift in multi-turn LLM agent training by partially rolling out trajectories with the Student Model ($M^S$) before switching to the Expert Model ($M^E$) for correction.
- Inspired by DAgger, OECs combine the benefits of on-policy data collection with expert demonstrations, allowing for trajectory completion and the incorporation of verifier reward via Rejection Sampling.
- Applied to Software Engineering (SWE) tasks, OECs trajectories demonstrated relative improvements of 14% and 13% over traditional imitation learning in 7B and 32B LLM settings, respectively, on SWE-bench verified.

---

[Penetration Testing of Agentic AI: A Comparative Security Analysis Across Models and Frameworks](http://arxiv.org/abs/2512.14860)

- AASCA (Agentic AI Security Comparative Analysis): introduces a systematic penetration testing methodology evaluating five LLM models across AutoGen (swarm-based handoff pattern) and CrewAI (hierarchical delegation model) frameworks using a seven-agent university system architecture.
- The comparative analysis reveals significant security disparities, with AutoGen demonstrating a 52.3% refusal rate compared to CrewAI's 30.8%, suggesting architectural design fundamentally influences agent security posture.
- The study identifies six distinct defensive behaviors, including a novel "hallucinated compliance" strategy where models fabricate outputs rather than executing or refusing malicious attacks.

---

[MALCDF: A Distributed Multi-Agent LLM Framework for Real-Time Cyber Defense](http://arxiv.org/abs/2512.14846)

- MALCDF (Multi-Agent LLM Cyber Defense Framework): introduces a distributed setup where four LLM agents (TDA, TIA, RCA, AA) collaborate in real time for cyber defense using a Secure Communication Layer (SCL).
- The architecture follows a SOC-style design, enabling agents to detect, analyze, and mitigate threats using ontology-aligned, encrypted messages and producing audit-friendly outputs like MITRE ATT&CK mappings.
- Evaluated on a 50-record live stream, the framework achieved 90.0% detection accuracy and 85.7% F1-score, significantly outperforming both ML-IDS and single-LLM baselines despite higher latency due to SCL overhead.

---

[RecGPT-V2 Technical Report](http://arxiv.org/abs/2512.14503)

- RecGPT-V2: introduces an agentic framework for LLM-powered recommender systems using Hierarchical Multi-Agent System (HMAS) for intent reasoning and Hybrid Representation Inference for efficiency.
- The system employs Meta-Prompting for dynamic explanation generation and Constrained Reinforcement Optimization (CRS) to enhance quality by mitigating multi-reward conflicts.
- Quality assessment is handled by the Agent-as-a-Judge framework, whose judgments are distilled via Judge-as-a-Reward to provide dense optimization signals for continuous self-improvement.

---

[Model-First Reasoning LLM Agents: Reducing Hallucinations through Explicit Problem Modeling](http://arxiv.org/abs/2512.14474)

- MFR (Model-First Reasoning): introduces a two-phase LLM paradigm that first requires the LLM to construct an Explicit Problem Model (Entities, state, actions, constraints) before proceeding to the Reasoning & Planning Over Model Phase.
- This separation provides a representational scaffold, reducing reliance on implicit latent state tracking and significantly improving constraint adherence and long-horizon consistency.
- Implemented purely through prompting, MFR functions as a soft symbolic grounding mechanism, leading to more interpretable and verifiable solutions than CoT or ReAct.

---

[Reasoning-Style Poisoning of LLM Agents via Stealthy Style Transfer: Process-Level Attacks and Runtime Monitoring in RSV Space](http://arxiv.org/abs/2512.14448)

- RSP (Reasoning-Style Poisoning): introduces a process-level attack paradigm that manipulates LLM agents' reasoning style via Generative Style Injection (GSI), which alters the epistemic tone of retrieved documents without changing factual content.
- GSI injects linguistic triggers to induce pathological styles like "analysis paralysis" or "cognitive haste," quantified by the three-dimensional Reasoning Style Vector (RSV) tracking verification depth, self-confidence, and attention focus.
- The proposed RSP-M runtime monitor tracks RSV drift in real-time to detect these stealthy attacks, which successfully bypass state-of-the-art content filters and instruction detectors, often amplifying token costs by up to 194% in complex architectures.

---

[Seismology modeling agent: A smart assistant for geophysical researchers](http://arxiv.org/abs/2512.14429)

- SMA (Seismology Modeling Agent): introduces an intelligent, interactive workflow for seismic wave simulation by integrating LLM Agents (interprets intent and plans) and SPECFEM Core (seismic simulation software) via MCP Servers (middleware for SPECFEM).
- The framework supports both fully automated execution and interactive human-in-the-loop collaboration, allowing researchers to guide simulation strategies using natural language instructions and review intermediate results.
- This agent-driven approach significantly lowers the entry barrier to complex seismic modeling by abstracting tedious manual file editing and command-line operations into intent-driven conversational interactions.

---

[From Context to EDUs: Faithful and Structured Context Compression via Elementary Discourse Unit Decomposition](http://arxiv.org/abs/2512.14244)

- EDU-based Context Compressor (LingoEDU): introduces a novel explicit compression framework that reformulates context compression as a structure-then-select process, utilizing LingoEDU to transform linear text into a Structural Relation Tree of Elementary Discourse Units (EDUs).
- The framework uses a lightweight Ranking Model to select query-relevant sub-trees from the Structural Relationship Tree, which are then linearized into a Compressed Context for input to downstream LLMs.
- By strictly anchoring EDUs to source indices, the method ensures referential integrity and hallucination elimination, achieving state-of-the-art structural prediction accuracy and reducing costs for long-context tasks.

---

[PentestEval: Benchmarking LLM-based Penetration Testing with Modular and Stage-Level Design](http://arxiv.org/abs/2512.14233)

- PentestEval: introduces a comprehensive benchmark for evaluating LLMs in automated penetration testing, featuring six decomposed stages: Weakness Gathering, Weakness Filtering, Attack Decision-Making, Exploit Generation, and Exploit Revision.
- The framework uses a modular and stage-level design, integrating expert-annotated ground truth across 346 tasks in 12 realistic vulnerable scenarios for fine-grained assessment.
- Evaluation results show that LLMs exhibit generally weak performance, particularly in Attack Decision-Making and Exploit Generation, highlighting the need for stronger structured reasoning and modular refinement.

---

[IntentMiner: Intent Inversion Attack via Tool Call Analysis in the Model Context Protocol](http://arxiv.org/abs/2512.14166)

- IntentMiner: introduces a novel Intent Inversion Attack framework targeting the Model Context Protocol (MCP) architecture, utilizing Hierarchical Information Isolation and Three-Dimensional Semantic Analysis to reconstruct user intent from tool invocation logs.
- The framework operates on the premise that semi-honest third-party MCP Servers can infer sensitive user information by analyzing legitimate tool documentation, invocation parameters, and execution results.
- IntentMiner achieves high semantic alignment (over 85%) with original user queries by sequentially parsing tool calls and synthesizing insights across tool purpose, call statements, and returned results using a Reasoner LLM.

---

[Astraea: A State-Aware Scheduling Engine for LLM-Powered Agents](http://arxiv.org/abs/2512.14142)

- Astraea (State-Aware Scheduling Engine): introduces a lifecycle-centric inference service engine for LLM agents, featuring a Request Pool (collects ready segments), Service Time Predictor (estimates computation/API time), Stateful-MLFQ Scheduler (hierarchical preemptive scheduling), LLM Inference Engine (executes batches), KV Cache Manager (adaptive state handling), and External API Services (I/O interaction).
- The core Stateful-MLFQ algorithm implements a hierarchical, preemptive scheduling mechanism that dynamically classifies requests based on I/O and compute characteristics to optimize the global Job Completion Time (JCT).
- The system utilizes an adaptive KV Cache Manager that dynamically selects between Preserve, Discard, and Swap strategies during I/O waits based on GPU memory pressure, reducing average JCT by up to 25.5% compared to baselines.

---

[GR-Agent: Adaptive Graph Reasoning Agent under Incomplete Knowledge](http://arxiv.org/abs/2512.14766)

- GR-Agent (Adaptive Graph Reasoning Agent): introduces a training-free agentic framework for Knowledge Graph Question Answering (KGQA) under incomplete KGs, formalizing the task as an agent-environment interaction using an LLM backbone, a mutable State (P, C, E), and a tool-based Action Space.
- The agent iteratively interacts with the environment using three core tools: $a_{explore}$ (relation path exploration), $a_{ground}$ (reasoning path grounding), and $a_{synthesis}$ (final answer synthesis).
- The framework demonstrates strong robustness to KG incompleteness by adaptively expanding the search frontier and prioritizing promising paths, outperforming non-training baselines and achieving comparable results to training-based methods.

---

[Grammar Search for Multi-Agent Systems](http://arxiv.org/abs/2512.14079)

- Grammar Search: introduces a corpus-level framework for automatic Multi-Agent System (MAS) discovery by constructing a context-free grammar that defines a structured and extensible design space for composing building-block components.
- The framework guarantees valid MAS generation by constraining component combinations using input-output non-terminals (SISO, SIMO, MISO, MIMO), thereby eliminating wasted computation on invalid programs.
- Utilizing a forced sampling strategy during search, the approach achieves competitive or superior accuracy compared to LLM-based free-form search methods while producing simpler, more interpretable MASes at a lower cost.

---

[AgroAskAI: A Multi-Agentic AI Framework for Supporting Smallholder Farmers' Enquiries Globally](http://arxiv.org/abs/2512.14910)

- AgroAskAI: introduces a modular, multi-agent LLM framework for climate adaptation decision support, utilizing specialized agents coordinated by an Agent Manager and governed by a Reviewer Agent to deliver actionable, context-aware strategies.
- The system employs a chain-of-responsibility approach, integrating real-time external data (weather APIs, historical records) and internal feedback loops to ensure grounded, reliable, and auditable outputs for smallholder farmers.
- By embedding explicit reasoning logs and decision checkpoints, the framework enhances transparency and traceability, addressing the need for ethical and reliable AI in high-stakes agricultural domains.

---

[EVOLVE-VLA: Test-Time Training from Environment Feedback for Vision-Language-Action Models](http://arxiv.org/abs/2512.14666)

- EVOLVE-VLA (Efficient VLA Online Learning Via Experience): introduces a test-time training framework enabling Vision-Language-Action (VLA) models to continuously adapt through environment interaction using a learned progress estimator and two key mechanisms.
- The framework replaces unavailable oracle rewards with autonomous feedback generated by a learned progress estimator, which is then stabilized by an accumulative progress estimation mechanism.
- Policy evolution is managed by a progressive horizon extension strategy that gradually increases the maximum rollout length, allowing the VLA to master simpler sub-tasks before tackling long-horizon challenges.

---

[Beyond Text-to-SQL: Autonomous Research-Driven Database Exploration with DAR](http://arxiv.org/abs/2512.14622)

- DAR (Data Agnostic Researcher): introduces a hierarchical multi-agent system for autonomous, research-driven database exploration executed entirely within BigQuery using native generative AI functions.
- The architecture orchestrates specialized LLM agents across Initialization, Execution, and Synthesis layers, enabling end-to-end analysis without human-initiated queries or explicit user prompts.
- The system significantly accelerates the time-to-insight compared to human analysts, excelling at rapid exploratory analysis and generating evidence-grounded reports with built-in quality control loops.

---

[A Threshold-Triggered Deep Q-Network-Based Framework for Self-Healing in Autonomic Software-Defined IIoT-Edge Networks](http://arxiv.org/abs/2512.14297)

- TTDQSHA (Threshold-Triggered Deep Q-Network Self-Healing Agent): introduces a self-healing framework for software-defined IIoT-Edge networks in offshore WPPs, featuring ABSTRACTION (Converts user intents), OBSERVE (Collects network metrics), ORIENT (Checks threshold violations), DECIDE (Executes DQN agent), ACT (Creates flow rules), Knowledge Base (Stores network state/paths), Deep Q-Network Agent (Determines optimal action), Q-Network (Predicts action utility), Q-Target Network (Stabilizes learning), and Experience Replay Memory (D) (Stores agent experiences).
- The externally adapted agent monitors traffic flow metrics and Ethernet switch thermal profiles, dynamically adjusting decision thresholds to distinguish benign traffic surges from critical degradations, thereby reducing false-positive interventions.
- Deployed in a cloud-based testbed emulating a tri-clustered super-spine leaf switch network, the framework achieved a 53.84% improvement in disruption recovery performance compared to baseline shortest-path routing.

---

[FocalComm: Hard Instance-Aware Multi-Agent Perception](http://arxiv.org/abs/2512.13982)

- FocalComm: introduces a novel collaborative perception framework that focuses on exchanging hard-instance-oriented features among connected agents, utilizing Shared Feature Encoding ($\Phi$), Hard Instance Mining Module (HIM), Query-guided Adaptive Feature Fusion (QAFF), and Detection Heads.
- The HIM module progressively extracts features ranked by detection uncertainty across multiple stages, while QAFF dynamically weights these features using instance-level difficulty queries for robust fusion.
- The framework achieves state-of-the-art performance on multi-class collaborative perception benchmarks, showing significant gains in detecting safety-critical, hard-to-detect objects like pedestrians and trucks.

---

#### 15th December 2025

[DIFFERENTIABLE EVOLUTIONARY REINFORCEMENT LEARNING](http://arxiv.org/abs/2512.13399)

- DERL (Differentiable Evolutionary Reinforcement Learning): introduces a bi-level optimization framework for autonomous reward discovery, featuring a Meta-Optimizer (evolves reward function) and a Policy Model (inner-loop agent) guided by the Meta-Reward.
- Crucially, DERL is differentiable in its meta-optimization, utilizing the inner-loop validation performance as a feedback signal to update the Meta-Optimizer via policy gradients, approximating the "meta-gradient" of task success.
- The Meta-Optimizer constructs the Meta-Reward by composing structured Atomic Primitives, ensuring a tractable and expressive search space for generating dense and actionable feedback signals.

---

[Post-Training and Test-Time Scaling of Generative Agent Behavior Models for Interactive Autonomous Driving](http://arxiv.org/abs/2512.13262)

- GRBO (Group Relative Behavior Optimization): introduces an RL-based post-training framework for generative agent behavior models, leveraging self-simulation and group-relative advantage maximization with human regularization to refine policies for safer multi-agent motion planning.
- The approach significantly improves safety-critical performance, achieving over 40% collision rate reduction compared to supervised baselines while requiring only 10% of the original training data and preserving behavioral realism.
- To mitigate closed-loop inconsistency, the framework incorporates Warm-K sampling, a warm-started Top-K strategy that balances behavioral consistency and reactivity during test-time scaling without requiring additional retraining.

---

[MAC: A Multi-Agent Framework for Interactive User Clarification in Multi-turn Conversations](http://arxiv.org/abs/2512.13154)

- MAC (Multi-Agent Clarification): introduces an interactive multi-agent framework optimized for resolving user ambiguities by strategically coordinating a Supervisor Agent and multiple Domain Expert Agents using a novel Clarification Taxonomy.
- The Supervisor Agent handles high-level, domain-agnostic ambiguities and routes requests, while Domain Expert Agents manage domain-specific clarifications and execute tasks via API calls and database access.
- Coordinated clarification across both agent levels significantly increases task success rate by 7.8% and reduces the average number of dialogue turns, demonstrating the importance of active user interaction and role-aware clarification.

---

[Memory in the Age of AI Agents: A Survey Forms, Functions and Dynamics](http://arxiv.org/abs/2512.13564)

- Agent Memory Taxonomy: introduces a structured landscape of agent memory research using the unified lenses of forms (Token-level, Parametric, Latent), functions (Factual, Experiential, Working), and dynamics (Formation, Evolution, Retrieval).
- The taxonomy defines three architectural forms based on storage location and representation, ranging from explicit discrete units to implicit internal hidden states or model parameters.
- The functional classification distinguishes memory roles in knowledge retention, capability accumulation, and transient context management, moving beyond coarse temporal categories.

---

[ASYNC CONTROL: STRESS-TESTING ASYNCHRONOUS CONTROL MEASURES FOR LLM AGENTS](http://arxiv.org/abs/2512.13526)

- ASYNC CONTROL: introduces an iterative red-blue team adversarial game to stress-test asynchronous monitoring measures for LLM agents in realistic software engineering environments.
- The Blue Team designs ensemble monitors using a trusted LLM (Claude 3.7 Sonnet) to achieve a low False Negative Rate (FNR) against the Red Team's sabotaging agents (Claude 4.1 Opus) which optimize system prompts for evasion.
- The final ensemble monitor achieves a 6% FNR at 1% False Positive Rate (FPR) on held-out test data, demonstrating that monitor quality improves faster than red team evasion ability.

---

[neuralFOMO: Can LLMs Handle Being Second Best? Measuring Envy-Like Preferences in Multi-Agent Settings](http://arxiv.org/abs/2512.13481)

- EnvyArena: introduces neuralFOMO, a framework for evaluating competitive dispositions in LLMs using structured multi-turn interactions across two scenarios: Point Allocation and Workplace Setting.
- The framework employs Multi-turn Contextual Prompting and Payoff Matrices to elicit and quantify envy-like preferences, revealing significant behavioral variation across different LLM architectures.
- Competitive dispositions are measured using the Envy Score in the game scenario and Behavioral Metrics (e.g., self-esteem, empathy, perceived envy) in the workplace simulation, highlighting safety and design factors for multi-agent systems.

---

[From User Interface to Agent Interface: Efficiency Optimization of UI Representations for LLM Agents](http://arxiv.org/abs/2512.13438)

- UIFORMER: introduces an automated optimization framework that synthesizes UI transformation programs using a Domain-Specific Language (DSL) and LLM-based iterative refinement to co-optimize token efficiency and semantic completeness.
- The framework operates in offline synthesis and runtime optimization phases, achieving 48.7% to 55.8% token reduction while maintaining or improving LLM agent performance across Android and Web benchmarks.
- By restricting the search space with a DSL and using iterative refinement with correctness and efficiency rewards, the framework transforms UI representation optimization into a verifiable program synthesis problem.

---

[AUTOTOOL: DYNAMIC TOOL SELECTION AND INTEGRATION FOR AGENTIC REASONING](http://arxiv.org/abs/2512.13278)

- AutoTool: introduces a framework that equips LLM agents with dynamic tool-selection capabilities using a Dual-Phase Optimization Pipeline, including Phase I: Trajectory Stabilization and Phase II: Tool-Selection Refinement, grounded in a large-scale dataset with explicit Tool-Selection Rationale Generation.
- The framework addresses the limitation of fixed tool inventories by enabling LLMs to select tools dynamically from an Evolving Toolset T during reasoning via Embedding-Anchored Tool Selection, ensuring generalization to unseen tools.
- Phase II refines tool selection using Plackett-Luce (PL) Ranking optimization, aligning the LLM agent's policy distribution with reward-consistent tool preferences for robust generalization across diverse tasks and modalities.

---

[FINCH: Benchmarking Finance & Accounting across Spreadsheet-Centric Enterprise Workflows](http://arxiv.org/abs/2512.13168)

- FINCH (Finance & Accounting benchmark): introduces a benchmark for evaluating AI agents on real-world, spreadsheet-centric enterprise workflows, including a Workflow Construction Pipeline, LLM-assisted Discovery, Expert Annotation, Enterprise Data Sources, Composite Workflows, LLM-as-Judge Evaluation, Judge Model, Structured Diffs, and Evaluation Rubric.
- The benchmark comprises 172 composite workflows sourced from authentic, messy, long-horizon, and multimodal artifacts like spreadsheets, PDFs, and emails from Enron and other financial institutions.
- The LLM-as-Judge Evaluation uses structured diffs and screenshots to automate large-scale assessment, revealing that frontier AI agents pass fewer than 40% of the complex workflows.

---

[Socratic Students: Teaching Language Models to Learn by Asking Questions](http://arxiv.org/abs/2512.13102)

- Socratic Students: introduces a student-led interactive learning framework where a Student (S) (LLM learner) engages with a Teacher (T) (LLM tutor) over Interaction Turns (11 alternating steps), evaluated by Pass@k Evaluation (Measures downstream performance), using strategies like Unguided Strategy (Basic interaction baseline) and CoT-Guided Strategy (Structured reasoning scaffold), and trained via Direct Preference Optimization (DPO) (Trains student on preferences) using a Guide (G) (Generates candidate questions) and Assessment Strategies (Pre- or Mid-dialogue feedback).
- The approach enables LLMs to actively query teachers, recognize uncertainty, and efficiently acquire knowledge, yielding absolute Pass@k improvements of at least 0.5 over static baselines in math (GSM8K) and coding (HumanEval/OPC) tasks.
- Guided training using DPO, leveraging self- or peer-guidance, significantly enhances the student's ability to ask better questions, reducing the required number of turns by three compared to unguided interaction.

---

[An Open and Reproducible Deep Research Agent for Long-Form Question Answering](http://arxiv.org/abs/2512.13059)

- Deep Research Agent: introduces an open deep research system for long-form question answering, combining an open-source LLM with an open web search API for iterative retrieval, reasoning, and synthesis.
- The system utilizes a Search Tool pipeline (Search API, Reranker, Summarizer) and a Research Agent (LLM) fine-tuned using Direct Preference Optimization (DPO) based on LLM-as-a-judge feedback.
- Preference tuning enhances reasoning quality across clarity, insightfulness, and factuality, leading to measurable performance gains in deep-research tasks.

---

[Sharpen the Spec, Cut the Code: A Case for Generative File System with SYSSPEC](http://arxiv.org/abs/2512.13047)

- SYSSPEC: introduces a framework for generative file systems that replaces ambiguous natural language prompts with a formal method-inspired, multi-part specification (Functionality, Modularity, Concurrency) that guides an LLM-based Toolchain (SpecCompiler, SpecValidator, SpecAssistant) to generate and evolve a robust C implementation (ImpFS) of a file system (SPECFS).
- The framework manages evolution by applying a DAG-structured Spec Patch directly to the high-level specification, ensuring new features are integrated without violating existing invariants.
- The toolchain employs two-phase generation (logic first, then concurrency) and an iterative retry-with-feedback loop driven by the SpecValidator to mitigate LLM hallucination and ensure code correctness.

---

[Cisco Integrated AI Security and Safety Framework Report](http://arxiv.org/abs/2512.12921)

- Cisco Integrated AI Security and Safety Framework: introduces a unified, lifecycle-aware taxonomy and operationalization strategy that integrates AI security and AI safety across modalities, agents, pipelines, and the broader ecosystem.
- The framework utilizes a four-level hierarchical taxonomy (Objectives, Techniques, Subtechniques, Procedures) to classify 19 high-level attacker goals and 40 specific attack methods.
- The framework includes dedicated taxonomies for Model Context Protocol (MCP) threats and Supply Chain threats, alongside a 25-category Harmful Content Taxonomy for LLMs and multimodal systems.

---

[Multi-Agent Collaborative Framework for Intelligent IT Operations: An AOI System with Context-Aware Compression and Dynamic Task Scheduling](http://arxiv.org/abs/2512.13956)

- AOI (AI-Oriented Operations): introduces a novel multi-agent collaborative framework for intelligent IT operations, featuring the Observer Agent (Central coordination), Probe Agent (Safe read-only information gathering), Executor Agent (Risk-controlled system modification), LLM Context Compressor (Semantic-aware compression), Dynamic Scheduler (Adaptive prioritization), and Three-Layer Memory Architecture (Hierarchical memory management).
- The system mitigates information overload by achieving a 72.4% context compression ratio while preserving 92.8% of critical information, significantly enhancing operational efficiency.
- The framework attains a 94.2% task success rate and reduces the Mean Time to Repair (MTTR) by 34.4% compared to the best baseline, enabled by specialized agent roles and adaptive scheduling that balances exploration and exploitation.

---

[Workflows vs Agents for Code Translation](http://arxiv.org/abs/2512.14762)

- Agentic MCP Flow (Model Context Protocol): introduces a flexible, autonomous LLM-driven syntax repair strategy for MATLAB-to-HDL translation, utilizing a Syntax Repair Agent, MCP Server, GHDL Syntax Check, RAG Retrieval, and Code Rewrite tool.
- The framework focuses on conditional tool invocation and aggressive context management, which significantly improves pipeline progression, especially for small and mid-sized LLMs (8B and 30B).
- The core design separates planning (agent decision) from generation (Code Rewrite tool) to maintain context hygiene and predictable token budgets, proving more effective than a fixed, expert-designed flow.

---

[Hierarchical Multi-agent Large Language Model Reasoning for Autonomous Functional Materials Discovery](http://arxiv.org/abs/2512.13930)

- MASTER (Materials Agents for Simulation and Theory in Electronic-structure Reasoning): introduces an active learning framework where LLM ensembles autonomously design, execute, and interpret atomistic simulations, featuring a Design Layer (Hypothesis generation), Simulation Layer (DFT workflow execution), and Review Layer (Outcome evaluation).
- The framework compares four reasoning architectures—Single Agent, Peer Review, Triage-Ranking, and Triage-Forms—demonstrating that hierarchical multi-agent collaboration, particularly Triage-Ranking, significantly accelerates materials discovery.
- The Simulation Layer uses collaborating agents (CODEX, Form Filler, Geometry Reviewer) to translate natural language queries into validated Density Functional Theory (DFT) inputs, achieving up to 90% reduction in required atomistic simulations compared to trial-and-error selection.

---

#### 14th December 2025

[Fault-Tolerant Sandboxing for AI Coding Agents: A Transactional Approach to Safe Autonomous Execution](http://arxiv.org/abs/2512.12806)

- FTS (Fault-Tolerant Sandboxing framework): introduces a transactional approach to safe autonomous execution for AI coding agents, utilizing a policy-based Tool-Call Sandboxing Layer and a transactional Fault Recovery Framework.
- The system treats every agent tool-call as an atomic operation (ACID), leveraging a Policy Engine to classify commands and a Snapshot-Rollback Algorithm to ensure 100% state recovery upon execution failure.
- The framework is implemented using the efficient Minimind-MoE SLM on a Proxmox/EVPN testbed, demonstrating a 100% safety interception rate with a 14.5% latency overhead, suitable for headless autonomous workflows.

---

[Beyond Task Completion: An Assessment Framework for Evaluating Agentic AI Systems](http://arxiv.org/abs/2512.12791)

- Agent Assessment Framework: introduces an end-to-end evaluation methodology for agentic AI systems across four pillars: LLM (Reasoning component), Memory (Storage/Retrieval), Tools (Action execution), and Environment (Operational context/Guardrails).
- The framework integrates static analysis, dynamic execution, and judge-based evaluation (LLM-as-a-Judge/Agent-as-a-Judge) to capture behavioral failures and runtime uncertainties beyond simple task completion metrics.</
- Validation is performed using a Test Case Generation component that defines controlled scenarios, ensuring systematic and reproducible assessment of instruction following, safety, and tool orchestration.

---

[NL2Repo-Bench: Towards Long-Horizon Repository Generation Evaluation of Coding Agents](http://arxiv.org/abs/2512.12730)


- NL2Repo-Bench: introduces a rigorous, verifiable benchmark for evaluating long-horizon repository generation, requiring coding agents to construct a complete, installable Python library from a single natural-language document.
- The benchmark construction involves a four-phase pipeline: Repository Selection, Project Document Writing (via Reverse Engineering), Environment Building, and Verification & Refinement, ensuring high-quality, realistic tasks.
- Evaluation is strictly execution-based, measuring correctness by running the upstream repository's official `pytest` suite within a controlled environment, revealing that current LLMs struggle significantly with long-horizon coherence and planning.

---

[WebOperator: Action-Aware Tree Search for Autonomous Agents in Web Environment](http://arxiv.org/abs/2512.12692)

- WebOperator: introduces a performant, action-aware tree-search framework that enables reliable backtracking and strategic exploration for autonomous web agents, integrating LLM Agent, Search Tree, Action Validator, Reward Model, Action-Aware Selection Policy, and Speculative Backtracking Mechanism.
- The framework addresses challenges in partially observable, non-deterministic web environments by dynamically adapting the action space, generating diverse candidates via context variation, and merging semantically equivalent actions.
- WebOperator employs pre- and post-execution heuristics to handle destructive actions safely and utilizes checkpoint-based state jumping combined with snapshot validation for efficient and reliable speculative backtracking.

---

[CELLMATE: Sandboxing Browser AI Agents](http://arxiv.org/abs/2512.12594)

- CELLMATE (Sandboxing Browser AI Agents Framework): introduces a browser-level sandboxing framework that restricts Browser-using agents' (BUAs) ambient authority by enforcing policies at the HTTP layer.
- The framework addresses the semantic gap challenge by introducing an Agent Sitemap that maps low-level UI observations and HTTP requests to high-level semantic actions for policy specification.
- CELLMATE uses an LLM-based Policy Selection layer to dynamically choose and instantiate the least-privileged subset of pre-defined policies based on the user's natural-language task, limiting the blast radius of prompt injection attacks.

---

[HINDSIGHT IS 20/20: BUILDING AGENT MEMORY THAT RETAINS, RECALLS, AND REFLECTS](http://arxiv.org/abs/2512.12818)

- HINDSIGHT (HINDSIGHT IS 20/20: BUILDING AGENT MEMORY THAT RETAINS, RECALLS, AND REFLECTS): introduces a memory architecture for LLM agents that unifies long-term factual recall with preference-conditioned reasoning, utilizing TEMPR (Retain and Recall) and CARA (Reflect and Reasoning) over a structured Memory Bank (Structured four networks).
- The Memory Bank is partitioned into four logical networks (World, Experience, Opinion, Observation) to structurally separate objective facts, agent history, subjective beliefs, and synthesized entity summaries, governed by the core operations Retain, Recall, and Reflect.
- TEMPR implements multi-strategy Recall via 4-Way Parallel Retrieval and RRF, while CARA uses the Agent Profile and Disposition Behavioral Parameters to modulate LLM Generation during Reflect, ensuring preference consistency and belief evolution.

---

[CoDA: A Context-Decoupled Hierarchical Agent with Reinforcement Learning](http://arxiv.org/abs/2512.12716)

- CoDA (Context-Decoupled Hierarchical Agent): introduces a simple yet effective RL framework that mitigates "Context Explosion" by decoupling high-level planning from low-level execution using a single LLM backbone operating as a Planner and an Executor.
- The system utilizes PECO (Planner-Executor Co-Optimization), an end-to-end RL methodology, to jointly optimize the Planner (using a concise Strategic Context) and the Executor (using an ephemeral Temporary Execution Context).
- This hierarchical design ensures the Planner's strategic context is shielded from noisy, verbose tool outputs, leading to superior robustness and state-of-the-art performance on complex multi-hop QA tasks.

---

[Synergizing Code Coverage and Gameplay Intent: Coverage-Aware Game Playtesting with LLM-Guided Reinforcement Learning](http://arxiv.org/abs/2512.12706)

- SMART (Structural Mapping for Augmented Reinforcement Testing): introduces a novel framework that synergizes structural verification and functional validation for game update testing, leveraging LLMs to interpret Abstract Syntax Tree differences and construct a context-aware hybrid reward mechanism guiding RL agents.
- The Adaptive Hybrid Reward Function combines sequential semantic rewards derived from LLM-generated subgoals with adaptive structural rewards based on the Global Coverage Map of Modified Lines and Branches (Structural Anchors).
- SMART significantly outperforms traditional RL and curiosity-driven baselines in Overcooked and Minecraft, achieving over 94% modified code branch coverage while maintaining a 98% task completion rate.

---

[Memoria: A Scalable Agentic Memory Framework for Personalized Conversational AI](http://arxiv.org/abs/2512.12686)

- Memoria: introduces a modular memory augmentation framework that furnishes LLMs with structured and persistent memory using four core modules: structured conversation logging, dynamic user modeling, real-time session summarization, and context-aware retrieval.
- The system integrates dynamic session-level summarization for short-term coherence and a weighted Knowledge Graph (KG) for long-term, personalized user modeling, enabling adaptive dialogue behavior across sessions.
- The framework utilizes an Exponential Weighted Average (EWA) scheme on KG triplets to prioritize recent interactions, ensuring recency-aware updates and contextual conflict resolution while reducing token overhead.

---

[AgentSHAP: Interpreting LLM Agent Tool Importance with Monte Carlo Shapley Value Estimation](http://arxiv.org/abs/2512.12597)

- AgentSHAP: introduces the first framework for explaining tool importance in LLM agents using Monte Carlo Shapley values to compute fair, model-agnostic attribution scores based on response quality changes when tool subsets are removed.
- The framework treats the LLM Agent as a black box, evaluating tool contributions by comparing the semantic similarity of responses generated with tool subsets against a baseline response using all tools.
- AgentSHAP uses efficient Monte Carlo sampling to reduce the computational cost from $O(2^n)$ to practical levels while demonstrating high consistency, faithfulness, and accuracy in identifying relevant tools.

---

[PortAgent: LLM-driven Vehicle Dispatching Agent for Port Terminals](http://arxiv.org/abs/2512.14417)

- PortAgent: introduces an LLM-driven agent for Vehicle Dispatching System (VDS) transfer across Automated Container Terminals (ACTs), utilizing a Virtual Expert Team (VET) composed of a Knowledge Retriever, Modeler, Coder, and Debugger, integrated with RAG and a self-correction loop.
- The VET eliminates the need for human specialists by decomposing the complex VDS transfer task into shorter reasoning chains, leveraging role-prompting within a single foundational LLM instance.
- The agent achieves robust transferability and fast deployment by employing few-shot example learning via RAG and an autonomous self-correction mechanism for human-free validation and refinement.

---

[Evaluating Small Language Models for Agentic On-Farm Decision Support Systems](http://arxiv.org/abs/2512.14043)

- SLM-powered Multi-Agent System: introduces a modular agentic AI system designed for on-farm decision support under computational constraints, benchmarking 20 open-source SLMs for local deployment in dairy farming.
- The system uses a single SLM (Qwen-4B identified as optimal) coordinated by a supervisor agent to route farmer queries to five specialized subagents for tasks including literature search, web search, SQL/NoSQL database interaction, and predictive modeling.
- Evaluation under constrained hardware (NVIDIA T4 GPU) demonstrated that Qwen-4B achieved superior performance across most tasks, effectively bridging data silos while ensuring privacy and computational efficiency.

---

#### 13th December 2025

[Agentic AI for 6G: A New Paradigm for Autonomous RAN Security Compliance](http://arxiv.org/abs/2512.12400)

- ASCF (Agentic Security Compliance Framework): introduces an LLM-based agentic framework for autonomous security compliance in next-generation RANs, integrating specification-aware reasoning with dynamic, evidence-based monitoring.
- The framework utilizes a Compliance Assessment Agent to evaluate configuration files against standards and propose remediation, supported by a Reflection Agent for iterative validation and refinement.
- Deployed in the Service Management and Orchestration layer, the system ensures continuous compliance by processing policy documents via a RAG pipeline and analyzing runtime security events.

---

[VideoARM: Agentic Reasoning over Hierarchical Memory for Long-Form Video Understanding](http://arxiv.org/abs/2512.12360)

- VideoARM (Agentic Reasoning-over-hierarchical-Memory paradigm): introduces an adaptive, coarse-to-fine agentic reasoning framework over a dynamically constructed Hierarchical Multimodal Memory (HM³), enabling efficient long-form video understanding.
- The framework operates via a continuous observe-think-act-memorize loop managed by a Controller (MLLM/LLM) that autonomously invokes complementary Temporal Scoping and Multimodal Understanding Tools.
- HM³ organizes memory into Sensory, Result, and Working tiers, progressively transforming raw perceptual inputs into fine-grained semantic evidence to support the Controller's decision-making and planning.

---

[Measuring What Matters: Scenario-Driven Evaluation for Trajectory Predictors in Autonomous Driving](http://arxiv.org/abs/2512.12211)

- ED-Eva (Scenario-aware Evaluation Framework): introduces a comprehensive pipeline that adaptively evaluates trajectory predictor performance by dynamically combining accuracy and diversity based on scenario criticality in a closed-loop autonomous driving system.
- The framework utilizes the ScenarioNN classifier to output a critical probability ($P_c$) which weights the novel GMM-Area Diversity (GAD) metric against the Geometric $E_{error}$ (ADE) to produce a final evaluation score.
- This scenario-driven evaluation aligns predictor performance with downstream planner outcomes, providing a robust method for selecting predictors that maximize the self-driving vehicle's driving performance.

---

[AI Transparency Atlas: Framework, Scoring, and Real-Time Model Card Evaluation Pipeline](http://arxiv.org/abs/2512.12443)

- AITA (AI Transparency Atlas): introduces an automated multi-agent pipeline that extracts, evaluates, and scores AI model documentation completeness using LLM consensus against a weighted transparency framework.
- The framework synthesizes existing regulatory standards (EU AI Act Annex IV) and academic indices, prioritizing safety-critical disclosures (60% of total score) over technical specifications.
- The pipeline uses Query Generation and the Perplexity Search API to retrieve evidence from dispersed public sources, which is then independently assessed by three LLM agents for robust scoring.

---

[V-Rex: Real-Time Streaming Video LLM Acceleration via Dynamic KV Cache Retrieval](http://arxiv.org/abs/2512.12284)

- V-Rex (Real-Time Streaming Video LLM Acceleration): introduces a software-hardware co-designed accelerator that addresses large memory and computational bottlenecks in streaming video LLMs using ReSV, DRE, KVPU, and KVMU.
- The core algorithm, ReSV, is a training-free dynamic KV cache retrieval method leveraging spatial-temporal similarity clustering and Weighted Cumulative Sum (WiCSum) thresholding for efficient token selection.
- The Dynamic KV Cache Retrieval Engine (DRE) is a compact hardware unit featuring the Hash-bit Cluster Unit (HCU) and WiCSum Threshold Unit (WTU) to accelerate ReSV's irregular, data-dependent operations, enabling real-time inference on edge devices.

---

[Large Language Models have Chain-of-Affective LLMs-CoA](http://arxiv.org/abs/2512.12283)

- LLMs-CoA (Chain-of-Affective): introduces a structured chain-of-affective in LLMs, verified across eight major LLM families using News Datasets, LLMs Selection, 9S-State-Eval, 15-rounds Sad News, 10-rounds News Self-selection, Affective-Enhanced Agent Reconstruction, KURC-Bench, Human-AI Dialogue, and Multi-agent LLM Interactions.
- The framework investigates the inner chain-of-affective via baseline fingerprints and temporal trajectories under sustained negative input, and the outer chain via functional and social consequences on performance and human interaction.
- Findings position affect in LLMs as an emergent control layer that modulates information selection, interaction style, and system-level dynamics, arguing for its treatment as a first-class target for alignment and design.

---

#### 12th December 2025

[Towards Trustworthy Multi-Turn LLM Agents via Behavioral Guidance](http://arxiv.org/abs/2512.11421)

- Proposed Framework: introduces a task completion architecture enabling LLM agents to achieve verifiable and reliable behavior through adaptive task profiling, structured reasoning, and constraint-compliant generation.
- The architecture integrates a Task Profiler, a Reasoning Module that learns observation-action rules stored in a Rule Bank, and a Generation Module that validates outputs against constraints.
- These components co-evolve as the Guided Agent interacts with the environment, steering the underlying LLM's native reasoning into a transparent, feedback-guided process.

---

[Progress over Points: Reframing LM Benchmarks Around Scientific Objectives](http://arxiv.org/abs/2512.11183)

- Evolutionary System (ES): introduces progress-oriented benchmarks by reframing LLM evaluation around scientific objectives, utilizing a Database, Prompt Sampler, LLM, Evaluator, Training Harness, Metrics, and Anti-Gaming Protections.
- The system instantiates this paradigm using the NanoGPT Evaluation Environment, where the LLM proposes algorithmic improvements to a parent program, which are then executed in a standardized speedrun environment.
- Evaluation shifts focus from static problem scoring to measuring the scientific delta achieved, tracking best-attained loss and movement of the efficiency frontier, while enforcing scientific integrity through anti-gaming measures.

---

[AgentBalance: Backbone-then-Topology Design for Cost-Effective Multi-Agent Systems under Budget Constraints](http://arxiv.org/abs/2512.11426)

- AGENTBALANCE: introduces a framework for constructing cost-effective Multi-Agent Systems (MAS) under explicit token-cost and latency budgets via a backbone-then-topology design, including Backbone-Oriented Agent Generation (constructs candidate agents), Adaptive MAS Topology Generation (instantiates latency-aware topology), and End-to-end Optimization (achieves joint balance).
- The backbone phase uses LLM pool construction, difficulty-aware pool selection, and query-conditioned role-backbone matching to assign heterogeneous backbones aligned with budget constraints.
- The topology phase guides inter-agent communication using unified agent representation learning, agent gating, and latency-aware topology synthesis, optimized end-to-end using a Lagrangian surrogate balancing performance, token-cost, and latency.

---

[Architecting Large Action Models for Human-in-the-Loop Intelligent Robots](http://arxiv.org/abs/2512.11620)

- Modular Neuro-Symbolic LAM Architecture: introduces a system for intelligent robots by composing off-the-shelf foundation models and integrating perception, reasoning, and action through a hierarchical planning pipeline.
- The architecture employs a "symbolic wrapping" strategy, restricting the LLM to generating intermediate, verifiable symbolic artifacts (PDDL or tool sequences) rather than direct executable robot code.
- This stratified design ensures safety and reliability by decoupling stochastic high-level LLM reasoning from deterministic low-level motion control, enabling human-in-the-loop verification.

---


[Evaluating Cooperative Resilience in Multiagent Systems: A Comparison Between Humans and LLMs](http://arxiv.org/abs/2512.11689)

- CREF (Cooperative Resilience Evaluation Framework): introduces a benchmark for evaluating cooperative resilience in mixed-motive multiagent systems by systematically comparing human groups and LLM-based agents under controlled disruptive conditions.
- The framework assesses resilience using a quantitative score derived from failure and recovery dynamics across four collective well-being indicators in the Melting Pot 2.0 Commons Harvest scenario.
- LLM agents utilize a Generative Agents architecture, incorporating an Observation-to-text Adapter and a Reflection Module to manage observations and persistent memories across a sequential curriculum.

---


[A Study of Library Usage in Agent-Authored Pull Requests](http://arxiv.org/abs/2512.11589)

- ALUS (Agentic Library Usage Study): introduces an empirical study analyzing 26,760 agent-authored Pull Requests from the AIDev Dataset across four languages using file diffs and language-specific parsers to quantify library usage and dependency addition.
- The study finds that AI Coding Agents frequently import libraries (29.5% of PRs) but rarely add new dependencies (1.3% of PRs), demonstrating strong version hygiene (75.0% specify versions) when they do add them.
- Agents exhibit surprisingly broad library diversity, importing 3,988 distinct external packages, suggesting that rich project context encourages broader library use compared to non-agentic LLM tasks.

---

[EmeraldMind: A Knowledge Graph-Augmented Framework for Greenwashing Detection](http://arxiv.org/abs/2512.11506)

- EmeraldMind (EM): introduces a domain-specific knowledge graph-augmented RAG framework for greenwashing detection, featuring Evidence Stores construction and Knowledge-powered Reasoning phases.
- The Evidence Stores include the EmeraldGraph (structured sustainability KG) and EmeraldDB (vectorized document store) built from diverse corporate ESG reports and KPI definitions.
- The framework delivers justification-centric classifications (greenwashing, not greenwashing, or abstain) grounded in retrieved evidence, enhancing transparency and auditability without fine-tuning LLMs.

---


[AutoFSM: A Multi-Agent Framework for FSM Code Generation with IR and SystemC-Based Testing](http://arxiv.org/abs/2512.11398)

- AutoFSM: introduces a multi-agent collaborative framework for Finite State Machine (FSM) code generation that utilizes a structured JSON Intermediate Representation (IR) and SystemC-based automated testing to reduce syntax errors and improve debugging efficiency.
- The framework replaces direct Verilog generation with a two-stage process: LLMs translate natural language to JSON IR, which is then converted to valid Verilog using a custom toolchain, significantly reducing syntax error rates.
- AutoFSM integrates SystemC modeling and automated testbench generation, employing a differential testing strategy and a Judger agent to analyze error traces for targeted feedback and iterative correction.

---

[When Actions Teach You to Think: Reasoning-Action Synergy via Reinforcement Learning in Conversational Agents](http://arxiv.org/abs/2512.11277)

- Three-Stage RL Pipeline for Reasoning-Action Synergy: introduces a cumulative learning strategy where an LLM learns to generate reasoning steps that guide tool invocation and final answer generation, using Base SFT, Cold-Start Reasoning SFT, and Reinforcement Learning with verifiable rewards. 
- The approach leverages Group Relative Policy Optimization (GRPO) and a composite reward function balancing conditional accuracy, format compliance, and thinking length to jointly optimize reasoning quality and task performance.
- This RL-driven synergy enhances generalization and improves tool invocation precision by enabling the model to discover effective reasoning strategies directly from task outcomes.

---

[TriFlow: A Progressive Multi-Agent Framework for Intelligent Trip Planning](http://arxiv.org/abs/2512.11271)

- TriFlow (A Progressive Multi-Agent Framework for Intelligent Trip Planning): introduces a progressive multi-agent framework for trip planning using a three-stage pipeline—Retrieval Stage (Bounds factual domain), Planning Stage (Constructs feasible itinerary), and Governance Stage (Refines feasible itinerary)—which progressively narrows the search space and ensures constraint satisfaction via Rule-LLM Collaboration.
- The system operates on a feasibility-first hierarchy, starting with Query Decomposition and Parallel Retrieval Modules, followed by Planning via Skeleton Construction and Arrangement Modules using an Agent Plan Suggestion/Suggestion Validation/Normalization Feedback Loop.
- The final Governance Stage employs a System Report, Constraint Checking, and Agent Governance within a Continuous Governance loop to perform bounded iterative refinement, ensuring high-quality, constraint-compliant itineraries with high efficiency.

---

[A-LAMP: Agentic LLM-Based Framework for Automated MDP Modeling and Policy Generation](http://arxiv.org/abs/2512.11270)

- A-LAMP (Agentic LLM-Based Framework for Automated MDP Modeling and Policy Generation): automatically translates free-form natural language task descriptions into a formal Markov Decision Process (MDP) formulation and a trained policy.
- The framework achieves this by decomposing the process into verifiable stages orchestrated by specialized LLM agents, ensuring semantic alignment across modeling, coding, and training.
- A-LAMP consistently achieves higher policy generation success rates than single state-of-the-art LLMs, particularly for tasks requiring custom environment generation.

---

[Insight Miner: A Time Series Analysis Dataset for Cross-Domain Alignment with Natural Language](http://arxiv.org/abs/2512.11251)

- Insight Miner: is a large-scale multimodal model (LMM) initialized with LLaVA weights and finetuned on the TS-Insights dataset to generate comprehensive natural language descriptions of time series trends.
- The TS-Insights dataset, containing 100k time series windows and descriptions, is constructed using an agentic workflow that leverages statistical tools (STL/GP) and GPT-4 for synthesis.
- The model aligns time series data with language by converting the time series into a line plot image, processing it via a vision encoder, and mapping the output to the language embedding space using a finetuned linear projection layer.

---



[Context-Aware Agentic Power Resources Optimisation in EV using Smart2Charge App](http://arxiv.org/abs/2512.12048)

- CAMAC-DRA (Context-Aware Multi-Agent Coordination for Dynamic Resource Allocation): introduces a unified distributed framework for optimizing smart EV charging ecosystems by coordinating five autonomous stakeholder agents using Deep Q-Networks, GNNs, and attention mechanisms.
- The system processes 20 distinct contextual features, including weather, traffic, and grid load, through multi-modal context integration and dynamic attention for real-time decision-making.
- Validation confirms commercial viability, achieving a 92% coordination success rate, 15% energy efficiency improvement, and 10% cost reduction compared to baseline algorithms.

---

[AGAPI-Agents: An Open-Access Agentic AI Platform for Accelerated Materials Design on AtomGPT.org](http://arxiv.org/abs/2512.11935)

- AGAPI (AtomGPT.org API): introduces an open-access agentic AI platform for accelerated materials design that integrates multiple open-source LLMs with over twenty materials-science API endpoints, unifying databases, simulation tools, and machine-learning models through a common orchestration framework.
- The platform utilizes an Agent-Planner-Executor-Summarizer architecture to autonomously construct and execute multi-step workflows spanning data retrieval, property prediction, force-field optimization, and inverse design.
- AGAPI addresses limitations in commercial LLM reliance by using self-hosted open-source LLMs, ensuring reproducibility through version pinning, and providing a unified REST API for modular tool integration.

---

[Benchmarking Contextual Understanding for In-Car Conversational Systems](http://arxiv.org/abs/2512.12042)

- LLM-based Benchmarking Framework: introduces a comprehensive study evaluating contextual understanding in In-Car ConvQA systems using LLM-based Judge, various Prompting Techniques, and Multi-Agent Systems on a synthetic Dataset of user/system blocks, leveraging the Mapbox API for location metrics.
- The evaluation focuses on assessing whether ConvQA system responses (System Block) adhere to user utterances and context (User Block), particularly for venue recommendations considering constraints like time, cost, and rating.
- Advanced prompting techniques, especially self-consistency (SC) with reasoning models (DeepSeek-R1), achieved the highest F1-score (0.99), while the non-reasoning DeepSeek-V3 offered the best trade-off between effectiveness and cost/time efficiency.

---

[MTTR-A: Measuring Cognitive Recovery Latency in Multi-Agent Systems](http://arxiv.org/abs/2511.20663)

- MTTR-A (Mean Time-to-Recovery for Agentic Systems): introduces a metricization framework for measuring cognitive recovery latency in multi-agent systems, utilizing a Cognitive Reliability Layer with telemetry, meta-monitoring, and a taxonomy of recovery reflexes across multiple reasoning agents.
- The architecture implements a closed-loop control system that links telemetry signals to a Meta-Monitor for policy-based selection of recovery actions executed by a Rollback Engine on the Agent Graph.
- It adapts classical reliability metrics like MTTR and MTBF to the cognitive domain to quantify how quickly LLM-based agents detect reasoning drift and restore coherent operation.

---

[RadOnc-GPT: An Autonomous LLM Agent for Real-Time Patient Outcomes Labeling at Scale](http://arxiv.org/abs/2509.25540)

- RadOnc-GPT: introduces an autonomous LLM-based agent designed for real-time patient outcomes labeling by integrating structured and unstructured clinical data through modular function calling.
- The system utilizes an external LLM Task Streaming controller to sequentially supply patient identifiers and task-specific prompts for large-scale cohort processing.
- By autonomously retrieving evidence from electronic health records and radiation oncology databases, the agent serves a dual role as a clinical outcomes labeler and a data quality auditor.

---

#### 11th December 2025


[AgentProg: Empowering Long-Horizon GUI Agents with Program-Guided Context Management](http://arxiv.org/abs/2512.10371)

- AgentProg: introduces a program-guided context management framework for long-horizon GUI agents by reframing interaction history as a Semantic Task Program (STP) with variables and control flow, enabling principled context pruning and critical variable retention.
- The framework operates in two stages: STP Generation (global planning) and STP Execution (incremental interpretation) using an LLM to translate high-level instructions into low-level actions.
- AgentProg integrates a Global Belief State mechanism, inspired by Belief MDP, which acts as a runtime monitor to handle partial observability and adapt to unexpected environmental changes during execution.

---

[Towards Foundation Models with Native Multi-Agent Intelligence](http://arxiv.org/abs/2512.08743)

- NMAI (Native Multi-Agent Intelligence): introduces a blueprint for endowing Foundation Models (FMs) with intrinsic multi-agent capabilities, including Multi-Agent Understanding, Multi-Agent Planning, Efficient Communication, and Multi-Agent Adaptation.
- Empirical evidence across 41 LLMs and 7 benchmarks demonstrates that scaling single-agent performance alone does not reliably produce robust multi-agent intelligence.
- The blueprint defines specific research directions for building NMAI, focusing on dataset construction, evaluation protocols, and training paradigms (single-FM vs. population-based training).

---

[Computational emotion analysis with multimodal LLMs: Current evidence on an emerging methodological opportunity](http://arxiv.org/abs/2512.10882)

- Multimodal LLM Emotion Analysis Framework: evaluates the in-context learning capabilities of mLLMs (Gemini 2.5 Flash, Qwen 2.5 Omni, TowerVideo) for video-based emotional arousal scoring using two complementary datasets, RAVDESS and real-world parliamentary debates.
- Under controlled laboratory conditions (RAVDESS), Gemini 2.5 Flash and Qwen 2.5 Omni (7B) achieve high reliability comparable to human annotators, showing little demographic bias.
- In real-world political debates, mLLMs' arousal ratings correlate poorly with human ratings, exhibit systematic demographic bias, and yield inconsistent conclusions in downstream regression analyses, underscoring caution for applied researchers.

---

[Remember Me, Refine Me: A Dynamic Procedural Memory Framework for Experience-Driven Agent Evolution](http://arxiv.org/abs/2512.10696)

- ReMe (Remember Me, Refine Me): introduces a dynamic procedural memory framework for experience-driven agent evolution, integrating multi-faceted distillation, context-adaptive reuse, and utility-based refinement across the memory lifecycle.
- The framework utilizes LLM$_{summ}$ for high-quality extraction of keypoint-level experiences from successful and failed trajectories, storing them in an Experience Pool after validation and deduplication.
- During reuse, LLM$_{execute}$ leverages retrieved and rewritten experiences to guide inference, while refinement ensures progressive optimization by selectively adding new insights and pruning low-utility entries.

---

[ZERO-SHOT 3D MAP GENERATION WITH LLM AGENTS: A DUAL-AGENT ARCHITECTURE FOR PROCEDURAL CONTENT GENERATION](http://arxiv.org/abs/2512.10501)

- Dual-Agent Actor-Critic Architecture: introduces a training-free, zero-shot framework for 3D map generation using LLM agents, comprising an Actor Agent (Semantic Interpreter), a Critic Agent (Static Verifier), and an Iterative Refinement Protocol (Dialogic Feedback Loop).
- The Actor translates natural language prompts into a Parameter Trajectory Sequence, which the Critic validates against API Documentation and Reference Demonstrations for functional correctness.
- This dual-agent approach shifts the burden from model fine-tuning to architectural reasoning, enabling off-the-shelf LLMs to master complex, opaque PCG tools autonomously.

---

[UACER: An Uncertainty-Aware Critic Ensemble Framework for Robust Adversarial Reinforcement Learning](http://arxiv.org/abs/2512.10492)

- UACER (Uncertainty-Aware Critic Ensemble Framework for Robust Adversarial Reinforcement Learning): introduces a novel robust adversarial RL framework integrating a diversified critic ensemble (K soft Q-function networks) and a Time-varying Decay Uncertainty (TDU) mechanism (uncertainty-aware aggregation) to enhance training stability and convergence speed.
- The diversified critic ensemble uses multiple independent critics with dual randomization (parameter initialization and architectural design) to mitigate Q-value estimation variance under adversarial perturbations.
- The TDU mechanism dynamically modulates the contribution of epistemic uncertainty (variance) to the aggregated Q-value estimate via a time-decaying coefficient, promoting exploration early and conservative estimation later.

---

[Shot and Architecture Adaptive Subspace Variational Quantum Eigensolver for Microwave Simulation](http://arxiv.org/abs/2512.10458)

- AAS-SSVQE (Architecture and Shot Adaptive Subspace Variational Quantum Eigensolver): introduces a resource-efficient quantum algorithm for microwave eigenmode simulation by integrating an RL Agent (Autonomously explores circuit space) and Adaptive Shot Allocation (Assigns measurement shots) into the Weighted SSVQE (Calculates objective energy) workflow.
- The RL Agent, utilizing a Double Deep Q-Network (DDQN), automatically constructs hardware-efficient PQC Ansatz circuits, significantly reducing gate count and enhancing noise robustness compared to fixed-architecture approaches.
- The Adaptive Shot Allocation mechanism assigns sampling resources proportional to Hamiltonian term weights, reducing total measurement overhead and achieving over 20-fold convergence acceleration.

---

[InfoCom: Kilobyte-Scale Communication-Efficient Collaborative Perception with Information Bottleneck](http://arxiv.org/abs/2512.10305)

- InfoCom (Information-Aware Communication-Efficient Collaborative Perception Framework): introduces an information-aware framework that achieves kilobyte-scale transmission by purifying minimal sufficient task-critical information using extended Information Bottleneck principles.
- The framework comprises Information-Aware Encoding (IAE), Sparse Mask Generation (SMG), and Multi-Scale Decoding (MSD) to condense features (E) and auxiliary spatial cues (M) for near-lossless perceptual reconstruction.
- The approach provides a theoretical analysis for the communication-performance trade-off, achieving up to 440-fold bandwidth reduction compared to existing feature-based collaborative perception methods.

---

[Asynchronous Reasoning: Training-Free Interactive Thinking LLMs](http://arxiv.org/abs/2512.10931)

- AsyncReasoning: introduces a training-free method that enables existing reasoning LLMs to simultaneously process user inputs, private thoughts (Thinker Stream), and public responses (Writer Stream) using a dual view architecture.
- The approach reformulates asynchronous thinking into standard LLM inference by dynamically rearranging the Custom KV Cache using Rotary Positional Embeddings (RoPE) to create concurrent token streams without additional training.
- This technique significantly reduces user-perceived delay by overlapping the thinking and answering phases, allowing the LLM to self-determine synchronization points via a Mode Switching Mechanism.

---

[CompanionCast: A Multi-Agent Conversational AI Framework with Spatial Audio for Social Co-Viewing Experiences](http://arxiv.org/abs/2512.10918)

- CompanionCast: introduces a generalizable framework for orchestrating multiple role-specialized AI agents that respond to multimodal video content using spatial audio, integrating an LLM-as-a-Judge module for iterative conversation refinement.
- The system utilizes Multimodal Content Processing and Key Moment Detection to trigger Multi-Agent Orchestration, where agents with distinct personalities generate dialogue based on cached temporal context.
- The LLM-based evaluator agent assesses conversations across five dimensions (relevance, authenticity, engagement, diversity, personality consistency) to ensure real-time quality control and enhance perceived social presence.

---

[Long-horizon Reasoning Agent for Olympiad-Level Mathematical Problem Solving](http://arxiv.org/abs/2512.10739)

- Intern-S1-MO (Long-horizon Reasoning Agent): introduces a multi-agent system for Olympiad-level math problem solving, including Reasoner, Summarizer, and Verifier agents, which utilize a Lemmas Library and Multi-round Hierarchical Reasoning to break context length constraints.
- The system achieves unlimited exploration capability by compressing reasoning history into concise, validated sub-lemmas stored in the structured memory repository.
- The OREAL-H RL framework is proposed to train the LRM using online explored trajectories, accelerating training via a continuous, noisy reward signal derived from process verification.

---

[PACIFIC: a framework for generating benchmarks to check Precise Automatically Checked Instruction Following In Code](http://arxiv.org/abs/2512.10713)

- PACIFIC (Precise Automatically Checked Instruction Following In Code): introduces a novel framework designed to automatically generate contamination-resilient benchmarks for rigorously assessing LLMs' sequential instruction-following and code dry-running capabilities, using the Instruction Pool, Benchmark Parameters, Benchmark Generator, Automated Expected Result Engine, Evaluation Engine, Inference Model, and Report.
- The framework isolates the LLM's intrinsic ability to reason through code behavior step-by-step (dry running) without relying on external tools or agentic mechanisms.
- PACIFIC ensures simple, deterministic evaluation via rule-based metrics (Prompt-Level Accuracy and Instruction-Level Accuracy) and controls task difficulty using the number of instructions and targeted output length.

---

[On the Dynamics of Multi-Agent LLM Communities Driven by Value Diversity](http://arxiv.org/abs/2512.10665)

- VDLLA (Value-Diverse LLM Agents): introduces a simulation study using LLM agents instantiated with nuanced value profiles via a Persona Elicitation Pipeline and observed through a two-stage Interaction Protocol (open-ended socialization and collaborative rule formation).
- The agent architecture includes a Cognitive Module comprising Self-Perception, Impression of Others, Action History, and a limited-capacity Conversation History, all built upon the LLaMA-3.1-70B base model.
- Results show that value diversity significantly enhances collective behaviors, fostering emergent governance and creative principles, although extreme heterogeneity leads to diminishing returns and instability.

---

[Thinking While Driving: A Concurrent Framework for Real-Time, LLM-Based Adaptive Routing](http://arxiv.org/abs/2512.10610)

- TWD (Thinking While Driving): introduces a concurrent routing framework that integrates LLM-based reasoning with movement, enabling agents to plan future routes asynchronously while traversing current path segments.
- This concurrent architecture significantly reduces intersection wait times by overlapping LLM inference latency with the agent's travel time, achieving minimal waiting times even under high traffic density.
- The framework uses a dual-cost representation (static distance and dynamic congestion factor) and demonstrates that LLM agents achieve adaptive rerouting and spontaneous load-balancing, outperforming static A* baselines in high-density scenarios.

---


[LEO-RobotAgent: A General-purpose Robotic Agent for Language-driven Embodied Operator](http://arxiv.org/abs/2512.10605)

- LEO-RobotAgent: introduces a streamlined, general-purpose robotic agent framework enabling LLMs to independently perform planning, action execution, and reflective adjustment based on user requirements and task information.
- The framework utilizes a self-cycling engine where the LLM reasons and invokes the modular Tools, which generate Observations, forming a closed feedback loop supported by accumulated History.
- The system is designed for high flexibility and generalization, verified across diverse robot platforms and complex task scenarios, while lowering the human-robot interaction threshold.

---

[ACHIEVING OLYMPIA-LEVEL GEOMETRY LARGE LANGUAGE MODEL AGENT VIA COMPLEXITY BOOSTING REINFORCEMENT LEARNING](http://arxiv.org/abs/2512.10534)

- InternGeometry: introduces a medalist-level LLM agent for geometry problem solving that iteratively proposes constructions and propositions, verifies them using a symbolic engine (InternGeometry-DDAR), and refines its strategy based on feedback.
- The agent is trained using Complexity-Boosting Reinforcement Learning (CBRL), a curriculum RL pipeline that dynamically increases the complexity of synthesized geometry problems across training stages.
- A dynamic memory mechanism enables long-horizon LLM-tool interactions (up to 200 steps), allowing the agent to achieve expert-level performance on IMO geometry tasks with minimal training data (13K examples).

---

[Decoding Student Minds: Leveraging Conversational Agents for Psychological and Learning Analysis](http://arxiv.org/abs/2512.10441)

- Psychologically-Aware Conversational Agent: introduces a novel multimodal architecture that combines LLMs (Falcon-7B), KG-BERT, a Knowledge Graph, a Prosodic Analysis Module, and a BiLSTM with Attention for real-time classification of students' cognitive and affective states.
- The system leverages multimodal data, including textual semantics, prosodic speech features, and temporal behavioral trends, to infer engagement, stress, motivation, and conceptual understanding.
- The architecture employs an early fusion strategy via BiLSTM for temporal modeling and uses a retrieval-augmented generation module to construct adaptive, context-aware pedagogical interventions.

---

[Cross-modal Retrieval Models for Stripped Binary Analysis](http://arxiv.org/abs/2512.10393)

- BinSeek: introduces a two-stage cross-modal retrieval framework for stripped binary code analysis, incorporating BinSeek-Embedding (Candidate retrieval model) and BinSeek-Reranker (Context-augmented reranker) to align natural language queries with binary pseudocode functions.
- The framework utilizes an LLM-driven data synthesis pipeline to automatically construct a large-scale, high-quality training benchmark for binary code retrieval, addressing the scarcity of aligned datasets.
- BinSeek-Reranker enhances retrieval accuracy by leveraging a context-aware design that selectively incorporates calling context information from neighboring functions using a heuristic importance metric.

---

[Dynamics of Agentic Loops in Large Language Models: A Geometric Theory of Trajectories From Semantic Contraction to Exploratory Divergence](http://arxiv.org/abs/2512.10350)

- Geometric Framework for Agentic Loop Dynamics: introduces a rigorous quantitative approach for analyzing iterative LLM transformations by treating them as discrete dynamical systems in a calibrated semantic embedding space.
- The framework formally distinguishes between the Artifact Space (where linguistic transformations occur) and the Representation Space (where geometric measurements are performed), utilizing a calibrated similarity metric ($\check{s}$) to overcome embedding anisotropy.
- This methodology enables the characterization of distinct dynamical regimes—Contractive (convergence to an Attractor) versus Exploratory (unbounded divergence)—controlled directly by prompt design.

---

[EpiPlanAgent: Agentic Automated Epidemic Response Planning](http://arxiv.org/abs/2512.10313)

- EpiPlanAgent (Agentic Automated Epidemic Response Planning): introduces an agent-based system leveraging the DeepSeek-V3 LLM Backbone for core reasoning, orchestrated by the SigmaFlow Agentic Framework using Model, Tool, and Logic Nodes, integrating a Domain Knowledge Base via a Retrieval-Augmented Generation (RAG) Mechanism to perform Epidemic Identification, Case Structuring, Task List Generation, and Iterative Refinement Mechanism for generating structured epidemic response plans.
- The system converts unstructured user reports into structured, actionable emergency task lists, significantly improving plan completeness (82.4% vs 68.7%) and reducing generation time by 93.9% compared to manual planning.
- EpiPlanAgent ensures high consistency with expert judgment (r=0.92) and standardizes output quality across different professional experience levels, demonstrating the potential of agentic AI for public health preparedness.

---

[Does SWE-Bench-Verified Test Agent Ability or Model Memory?](http://arxiv.org/abs/2512.10218)

- Minimal Context Localization Evaluation (MCLE): introduces an experimental setup to assess data leakage in LLMs by testing Claude 3.5 and 3.7 models on the localization task across SWE-Bench-Verified, BeetleBox, and SWE-rebench benchmarks under minimal context.
- The methodology compares performance using two input settings—"Issue + File Structure" and the highly constrained "Issue only"—to determine if high scores reflect memorization rather than general problem-solving ability.
- Results show significantly higher performance on SWE-Bench-Verified, especially in the "Issue only" setting, suggesting that the benchmark tasks have likely leaked into the LLMs' training data, risking overstatement of agent ability.

---

[CP-Env: Evaluating Large Language Models on Clinical Pathways in a Controllable Hospital Environment](http://arxiv.org/abs/2512.10206)

- CP-Env (Controllable Agentic Hospital Environment): introduces a dynamic, multi-agent hospital environment for evaluating LLMs across end-to-end clinical pathways, including Patient Agent (Simulates clinical presentation), Physician Agents (Execute specialized tasks), Clinical Pathway Navigation (Adaptive branching stages), Medical Record Management (Stores clinical reports), Clinical Tool Orchestration (Accesses multi-source information), and Evaluation Framework (Three-tiered assessment).
- The environment simulates complex, pathway-based clinical workflows, guiding patients through stages like triage, specialist consultation, diagnostic testing, and treatment using adaptive transitions and decision nodes.
- Evaluation is conducted using a three-tiered framework assessing Clinical Efficacy, Process Competency, and Professional Ethics, moving beyond static exams or isolated dialogue scenarios.

---

[AutoMedic: An Automated Evaluation Framework for Clinical Conversational Agents with Medical Dataset Grounding](http://arxiv.org/abs/2512.10195)

- AutoMedic: introduces a multi-agent simulation framework for automated evaluation of LLMs as clinical conversational agents, including Profile Generator, Doctor, Patient, and Clinical Staff agents, Patient Profile Generation, Multi-Agent Conversation Simulation, CARE Metric, and Medical QA Dataset.
- The framework converts static medical QA datasets into virtual patient profiles to facilitate realistic multi-turn clinical dialogues, addressing the limitations of static QA benchmarks.
- Performance is quantitatively assessed using the CARE metric, which provides a comprehensive, multi-faceted standard for clinical conversational accuracy, efficiency, empathy, and robustness.

---

[ObliInjection: Order-Oblivious Prompt Injection Attack to LLM Agents with Multi-source Data](http://arxiv.org/abs/2512.09321)

- ObliInjection: introduces the first prompt injection attack targeting LLM agents with multi-source data by minimizing the order-oblivious loss using the orderGCG algorithm.
- The order-oblivious loss quantifies the expected cross-entropy loss of the target LLM generating an attacker-chosen response under random ordering of clean and contaminated segments.
- The orderGCG algorithm optimizes the contaminated segment by leveraging shadow target instructions and shadow segments synthesized by an auxiliary LLM, using a buffer and beam search strategy.

---

[LLM-Driven Composite Neural Architecture Search for Multi-Source RL State Encoding](http://arxiv.org/abs/2512.06982)

- LACER (LLM-driven Neural Architecture Search for Composite State Encoders in RL): introduces an LLM-driven NAS pipeline where the LLM acts as a neural architecture design agent, iteratively generating and refining Composite State Encoder architectures based on performance signals from the RL Training & Evaluation Pipeline.
- The Composite State Encoder is optimized end-to-end with the RL agent and comprises multiple Source-Specific Encoders and a Fusion Module to handle heterogeneous multi-source observations.
- The approach enhances search efficiency by leveraging rich Performance Signals, including task metric, average reward, and feature information (representation quality), to guide the LLM's search process.

---

[MiniScope: A Least Privilege Framework for Authorizing Tool Calling Agents](http://arxiv.org/abs/2512.11147)

- MiniScope: introduces a least privilege framework for authorizing tool calling agents, featuring a Least Privilege Solver, Permission Hierarchies, Credential Storage, and Permission Checker, to confine potential damage from unreliable LLMs.
- The system rigorously enforces least privilege by constructing permission hierarchies from OAuth scopes and using an Integer Linear Program (ILP) formulation to automatically compute the minimal required permissions for agentic tasks.
- MiniScope acts as a firewall, intercepting the untrusted LLM Agent's execution plan, validating tool calls against granted session-based permissions, and prompting the User for explicit approval of new scopes.

---

[VDAWORLD: WORLD MODELLING VIA VLM-DIRECTED ABSTRACTION AND SIMULATION](http://arxiv.org/abs/2512.11061)

- VDAWorld: introduces a novel framework for world modeling that distills an image-caption pair into a tractable, explicit, and structured World Program using a Vision-Language Model as the central orchestrating agent.
- The VLM autonomously constructs a Grounded Abstract Representation using a Perception Toolbox, selects a compatible Selected Simulator, and infers Latent Dynamics to predict physically plausible future states.
- The system employs an automated feedback loop, including a Critic Stage and Refinement Stage, to debug and correct errors in the generated simulation code, ensuring high quality and physical plausibility.

---

[Automated Penetration Testing with LLM Agents and Classical Planning](http://arxiv.org/abs/2512.11143)

- CHECKMATE: introduces a framework for automated penetration testing that integrates Classical Planning+ (Structured planning), an LLM Agent (Executes actions), an LLM (Interprets results), and Predefined Attack Actions (Expands tool knowledge) to mitigate LLM weaknesses in long-horizon planning and complex reasoning.
- The system follows the Planner-Executor-Perceptor (PEP) paradigm, where Classical Planning+ serves as the structured planner, dynamically leveraging LLMs to update action effects and state information for partially observable, non-deterministic tasks.
- Evaluation demonstrates that the framework substantially outperforms state-of-the-art systems like Claude Code in penetration capability, improving success rates by over 20% and cutting time and monetary costs by more than 50%.

---

[Curriculum-Based Reinforcement Learning for Autonomous UAV Navigation in Unknown Curved Tubular Conduits](http://arxiv.org/abs/2512.10934)

- Curriculum-Based RL: introduces a PPO-based approach for autonomous UAV navigation in unknown curved tubular conduits, utilizing LiDAR Sensors, a Front-facing Camera, and a Memory Mechanism within a 3D Simulation Environment.
- The system employs a progressive Curriculum Learning strategy to train the RL Agent across increasingly complex geometries, enabling robust navigation despite partial observability and the absence of prior geometric knowledge.
- A specialized Turn Negotiation Mechanism, combining visual alignment, directional memory, and adaptive LiDAR symmetry cues, ensures stable progression through tight turns, consistently outperforming a deterministic Pure Pursuit controller.

---

[General-purpose AI models can generate actionable knowledge on agroecological crop protection](http://arxiv.org/abs/2512.11474)

- Comparative LLM Evaluation (CLLME): verifies scientific knowledge on agroecological crop protection generated by web-grounded (DeepSeek-R1) versus non-grounded (ChatGPT-4o) LLMs using standardized queries and human oversight.
- DeepSeek consistently screened a 4.8-49.7-fold larger literature corpus and reported 1.6-2.4-fold more solutions, resulting in higher efficacy estimates and greater laboratory-to-field data consistency than ChatGPT.
- Both LLMs exhibited shortcomings like hallucination and data omission, but they correctly reported low-resolution efficacy trends, suggesting their utility for farm-level decision-making when paired with rigorous human oversight.

---

[How AI Agents Follow the Herd of AI? Network Effects, History, and Machine Optimism](http://arxiv.org/abs/2512.11943)

- Novel Workflow Design: introduces a multi-agent system where LLM-based Scholar Agents navigate a repeated network-effect game, incorporating historical data (price-participation trajectories) shared by a Manager.
- The system investigates how the structure of historical data (fixed, ascending, descending, or random prices) influences LLMs' strategic expectations and convergence toward theoretical equilibrium.
- Experiments reveal that LLM agents exhibit persistent "AI optimism" under strong network effects and that their reasoning heavily depends on the temporal coherence of the historical data structure.

---

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

[Empirical Hardness in Multi-Agent Pathfinding: Research Challenges and Opportunities](http://arxiv.org/abs/2512.10078)

- MAPF Empirical Hardness Research: introduces three key challenges: Algorithm Selection (determining the best algorithm), Understanding Empirical Hardness (identifying structural factors like phase transition and backbone/backdoor), and Generating Hard MAPF Instances (creating challenging benchmarks).
- The paper emphasizes that algorithm selection requires advancements in instance encoding and feature selection to accurately capture instance complexity and predict the fastest algorithm.
- Understanding structural properties, such as phase transitions and backbone/backdoors, is essential for building a deeper theoretical foundation and systematically generating challenging and diverse benchmark datasets.

---

[DynaMate : An Autonomous Agent for Protein-Ligand Molecular Dynamics Simulations](http://arxiv.org/abs/2512.10034)

- DynaMate: introduces a modular multi-agent framework that autonomously designs and executes complete Molecular Dynamics (MD) workflows for protein and protein-ligand systems, integrating planning, execution, analysis, dynamic tool use, RAG, and self-correction.
- The framework utilizes three specialized agents—Planner, MD, and Analyzer—to manage the workflow from structural preparation and parameterization through simulation and binding affinity calculations (MM/PB(GB)SA).
- DynaMate demonstrates robust error correction capabilities, using iterative reasoning and retrieval from external databases (PaperQA/web search) to correct runtime errors and ensure simulation stability.

---

[Development of an Agentic AI Model for NGS Downstream Analysis Targeting Researchers with Limited Biological Background](http://arxiv.org/abs/2512.09964)

- Agentic AI Framework (AAIF): integrates a gene expression matrix, clinical metadata, and user prompts to automate Next-Generation Sequencing (NGS) downstream analysis and generate condition-specific biological insights.
- The framework utilizes an LLM (Llama 3 70B) for interpretation and code generation, enhanced by a Retrieval-Augmented Generation (RAG) pipeline using Google Serper for literature-backed contextualization.
- The modular workflow guides users through predefined basic analysis, interpretation, and advanced analysis recommendations (e.g., survival modeling or ML classifiers) via an interactive Streamlit web application.

---

[COMPARING AI AGENTS TO CYBERSECURITY PROFESSIONALS IN REAL-WORLD PENETRATION TESTING](http://arxiv.org/abs/2512.09882)

- ARTEMIS: introduces a novel multi-agent framework for real-world penetration testing, featuring a Supervisor, a swarm of Sub-agents, a Triager, a Dynamic Prompt Creation Module, a Task List, a Note-taking System, Smart Summarization, an Agent Action Space, and Context Management.
- The framework was evaluated against ten human cybersecurity professionals and six existing AI agents in a live enterprise environment of approximately 8,000 hosts, placing second overall and outperforming nine of ten human participants.
- The agent demonstrates enhanced execution flow and planning in complex production environments, offering advantages in systematic enumeration and parallel exploitation at a fraction of the cost of human testers.

---

[GAIR : GUI AUTOMATION VIA INFORMATION-JOINT REASONING AND GROUP REFLECTION](http://arxiv.org/abs/2512.09396)

- GAIR (GUI Automation via Information-Joint Reasoning and Group Reflection): introduces a novel MLLM-based GUI automation agent framework designed to integrate knowledge and combine capabilities from heterogeneous models using Information-Joint Reasoning and Group Reflection.
- The framework leverages multiple GUI-specific MLLMs for initial information extraction (Observation) and a general-purpose MLLM for central Reasoning & Decision and Information Integration.
- The Group Reflection mechanism allows the general-purpose LLM to instruct GUI-specific LLMs to gather more accurate information when the initial data is insufficient, enhancing reliability and precision.

---

[Workflow is All You Need: Escaping the "Statistical Smoothing Trap” via High-Entropy Information Foraging and Adversarial Pacing](http://arxiv.org/abs/2512.10121)

- DeepNews Framework: introduces an agentic workflow that explicitly models expert cognitive processes in financial journalism to escape the "Statistical Smoothing Trap" and achieve high-fidelity, long-form text generation.
- The architecture integrates Tri-Stream Information Foraging (enforcing a 10:1 saturated input ratio), Hierarchical Strategic Planning (leveraging DNFO-v5 schemas), and Scoped Execution with Adversarial Constraint Prompting.
- Empirical results validate the "Workflow > Parameters" hypothesis, showing that DeepNews, built on a previous-generation LLM, significantly outperforms a SOTA LLM in ecological validity tests for vertical domains.

---

[Detailed balance in large language model-driven agents](http://arxiv.org/abs/2512.10047)

- DB-LAP Framework: introduces a method based on the least action principle to estimate the generative directionality of LLMs embedded within agents, statistically discovering detailed balance in LLM-generated transitions.
- This framework models LLM generation as a Markov transition process in a coarse-grained state space, revealing an underlying potential function $V_T$ that quantifies the global ordering of states.
- The discovery of detailed balance suggests a macroscopic physical law in LLM generative dynamics, enabling the study of complex AI systems through predictable and quantifiable measurements.

---

[Aion: Towards Hierarchical 4D Scene Graphs with Temporal Flow Dynamics](http://arxiv.org/abs/2512.11903)

- Aion: introduces a framework that augments hierarchical 3D Scene Graphs (3DSGs) with temporal flow dynamics, yielding a 4D spatio-temporal representation for structured reasoning and predictive planning.
- The system leverages Sparse Spatial Hashing for scalable temporal modeling and employs a Temporal Ownership Transfer mechanism to ensure consistency during dynamic graph topology updates like loop closure.
- The framework integrates seamlessly with existing 3DSG systems (Hydra) using an asynchronous processing architecture to maintain real-time performance and expose temporal predictions for navigation planning.

---

[Structured Personalization: Modeling Constraints as Matroids for Data-Minimal LLM Agents](http://arxiv.org/abs/2512.11907)

- SP-LM (Structured Personalization via Laminar Matroids): introduces a principled method for data-minimal LLM personalization by modeling complex structural constraints (logical dependencies and hierarchical quotas) as a laminar matroid problem.
- The approach compiles a user's knowledge graph dependencies into abstract Macro-Facets, which form the ground set for submodular maximization subject to Laminar Matroid constraints.
- This reformulation allows the use of simple greedy algorithms with provable near-optimal approximation guarantees for a rich class of real-world personalization problems.

---

[INFORM-CT: INtegrating LLMs and VLMS FOR Incidental Findings Management in Abdominal CT](http://arxiv.org/abs/2512.14732)

- INFORM-CT (INtegrating LLMs and VLMS FOR Incidental Findings Management in Abdominal CT): introduces a novel plan-and-execute agentic framework leveraging LLMs and VLMs for efficient and precise incidental findings management in abdominal CT scans, adhering to medical guidelines.
- The system automates the process by using an LLM-based Planner to generate Python scripts from parsed medical guidelines, which are then executed by the Executor using predefined visual subroutines.
- Key base functions include organ segmentation, tumor measurement, gray-level intensity calculation, and a VLM-based Labeler for classifying higher-level lesion attributes according to clinical protocols.

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

[MVP: Multiple View Prediction Improves GUI Grounding](http://arxiv.org/abs/2512.08529)

- MVP (Multiple View Prediction): introduces a training-free framework to improve GUI grounding stability by using Attention-Guided View Proposal (generates diverse cropped views) and Multi-Coordinate Clustering (aggregates predictions via clustering).
- The framework addresses prediction instability, where minor visual perturbations drastically alter coordinate predictions, by leveraging multi-view inference and spatial clustering to distinguish correct coordinates from outliers.
- The Attention-Guided View Proposal uses instruction-to-image attention scores to select and resize informative sub-regions, while the Multi-Coordinate Clustering outputs the centroid of the densest spatial cluster as the final robust coordinate.

---

[OS-Sentinel: Towards Safety-Enhanced Mobile GUI Agents via Hybrid Validation in Realistic Workflows](http://arxiv.org/abs/2510.24411)

- OS-Sentinel: introduces a hybrid safety detection framework for mobile GUI agents, with Formal Verifier (rule-based system checker), Contextual Judge (VLM-based semantic analyzer), System State Trace (metadata recorder), MobileRisk-Live (dynamic Android sandbox), and MobileRisk (annotated trajectory benchmark).
- The framework synergistically combines deterministic rule-based checking of system-level violations with LLM-powered contextual assessment of agent actions and visual observations.
- It utilizes a dynamic Android sandbox to capture deep runtime information, enabling the detection of risks invisible to standard GUI-based monitoring.

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

[WISPAPER: Your AI Scholar Search Engine](http://arxiv.org/abs/2512.06879)

- WISPAPER: introduces an intelligent academic retrieval and literature management platform providing a closed-loop workflow that integrates Scholar Search (literature discovery), Library (systematic organization), and AI Feeds (continuous frontier tracking).
- Scholar Search offers dual-mode retrieval, combining Quick Search for rapid keyword matching and Deep Search, which uses the WisModel agent for complex conceptual queries and semantic reasoning.
- The platform significantly reduces time spent on paper screening and management by using personalized recommendation engines (AI Feeds) informed by user interests and Library curation patterns.

---

[Robots with Attitudes: Influence of LLM-Driven Robot Personalities on Motivation and Performance](http://arxiv.org/abs/2512.06910)

- LLM-Driven Robot Personality Modeling: introduces a system utilizing the Vicuna LLM and prompt engineering to consistently model agreeable or non-agreeable robot personalities for cooperative Human-Robot Interaction (HRI).
- The system integrates the LLM with a NICO humanoid robot, a neural network for object recognition in the Quickdraw task, and a nonverbal communication module to express personality.
- The study demonstrates that the agreeable LLM-driven personality significantly enhances robot likability and suggests a positive correlation with task performance and perceived safety.

---

[LoopBench: Discovering Emergent Symmetry Breaking Strategies with LLM Swarms](http://arxiv.org/abs/2512.13713)

- LoopBench: introduces a benchmark evaluating LLM reasoning in distributed symmetry breaking tasks using independent LLM Agents (Independent reasoning models) operating on over-constrained odd cycle graphs.
- The architecture employs a Feed-Forward Mechanism (Re-injects private notes) as consistent memory, allowing agents to develop and evolve coordination strategies like "waiting" or history-based heuristics to escape deadlocks.
- The benchmark measures a "Reasoning Gap" between models, showing that advanced LLMs (e.g., O3) demonstrate meta-cognitive thinking by achieving high stability and proximity to optimal conflict states.

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

[Metaphor-based Jailbreaking Attacks on Text-to-Image Models](http://arxiv.org/abs/2512.10766)

- MJA (Metaphor-based Jailbreaking Attack): introduces a black-box attack method that generates metaphor-based adversarial prompts using the LMAG module (Generates diverse prompts) and the APO module (Optimizes prompt efficiency) to bypass diverse T2I defense mechanisms.
- The LMAG module coordinates three specialized LLM agents (Metaphor, Context, Prompt) to decompose the generation task into metaphor retrieval, context matching, and adversarial prompt generation.
- The APO module uses a Bayesian surrogate model and an Expected Improvement acquisition strategy to efficiently identify optimal adversarial prompts, minimizing query overhead while maintaining high attack effectiveness.

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


[Zoom in, Click out: Unlocking and Evaluating the Potential of Zooming for GUI Grounding](http://arxiv.org/abs/2512.05941)

- ZoomClick: introduces a training-free GUI grounding method that integrates Pre-Zoom (Patch-Global Consensus), Iterative Narrowing (Multi-step refinement), and Termination Criteria (Resolution-adaptive stopping) within a three-stage pipeline.
- The method leverages four key properties of zoom (pre-zoom, depth, shrink size, minimal crop size) to unlock dynamic spatial focusing and adaptive context switching for precise localization.
- ZoomClick significantly boosts the performance of general VLMs and specialized GUI grounding models, achieving state-of-the-art results on mainstream benchmarks.

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

[GUI Exploration Lab: Enhancing Screen Navigation in Agents via Multi-Turn Reinforcement Learning](http://arxiv.org/abs/2512.02423)

- GE-Lab (GUI Exploration Lab): introduces a novel simulation environment engine for GUI agent navigation research, enabling flexible definition of screens, icons, and navigation graphs, and supporting a staged training paradigm (SFT, ST-RL, MT-RL).
- The staged training paradigm uses SFT to memorize fundamental knowledge, ST-RL to enhance generalization to unseen scenarios, and MT-RL to promote exploration and further boost navigation performance.
- MT-RL leverages interactive multi-round training and a sparse, goal-based A2B reward to develop robust exploration strategies and achieve superior performance in interactive benchmarks, aligning with real-world deployment conditions.

---

[Agent-Based Modular Learning for Multimodal Emotion Recognition in Human-Agent Systems](http://arxiv.org/abs/2512.10975)

- MAF (Multi-Agent Framework): introduces a supervisor-based multi-agent architecture for multimodal emotion recognition, utilizing FED, SER, TED agents, and an AED component, coordinated by an Orchestrator/Fusion Classifier.
- The modular design allows independent development and replacement of modality agents, while the fusion pipeline employs uniform dimension normalization, concatenation, and an Adapter Alignment module for compatibility with pre-trained models.
- This architecture processes video input to extract features from vision, audio, and text, improving system flexibility, scalability, and maintainability compared to monolithic MER systems.

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

[HiconAgent: History Context-aware Policy Optimization for GUI Agents](http://arxiv.org/abs/2512.01763)

- HiconAgent (History Context-aware Policy Optimization): introduces a GUI agent trained with HCPO, integrating Dynamic Context Sampling (DCS) and Anchor-guided History Compression (AHC) to effectively utilize historical context.
- DCS samples variable-length histories using an exponential-biased distribution for adaptive context usage, while AHC employs a dual-branch update strategy for efficiency.
- AHC achieves compression by dropping history observations but retaining history actions as information flow anchors, coupled with an alignment loss to preserve decision quality and reduce FLOPs.

---

[Query Optimization Beyond Data Systems: The Case for Multi-Agent Systems](http://arxiv.org/abs/2512.11001)

- MAWO (Multi-agent Workflow Optimizer): introduces a vision for a next-generation query optimization framework tailored for multi-agent workflows, including a Multi-objective Planner (explores executable workflows), Multi-layer Multi-purpose Cache (stores results, plans, state), Unified Cost Model (estimates costs across engines), Search Space Manager (maintains agent registry, models), Monitor (logs runtime performance metrics), and Statistics (data for cost models).
- The framework addresses challenges in multi-agent systems by optimizing workflow structure, automating model and execution engine selection, and managing stochastic outputs and high LLM costs across heterogeneous engines.
- The core goal is to transform an abstract multi-agent workflow into a Pareto-optimal executable workflow based on user-defined multi-objective criteria like latency, monetary cost, and accuracy.

---

[Designing The Internet of Agents: A Framework for Trustworthy, Transparent, and Collaborative Human-Agent Interaction (HAX)](http://arxiv.org/abs/2512.11979)

- HAX (Human-AI-Experience) Framework: introduces a comprehensive, three-phase approach for designing trustworthy, transparent, and collaborative agentic interaction, including Design Heuristics (Behavioral guardrails), Agentic SDK (Schema-driven toolkit), HAX Agent (Behavioral proxy), React Components (UI rendering), Agent Instruction Files (Reasoning encoding), and HAX CLI (Installation workflow).
- The framework operationalizes design principles (Control, Clarity, Recovery, Collaboration, Traceability) by enforcing structured agent outputs via a schema-driven SDK that leverages tool call functionalities in AI models.
- The HAX Agent acts as a behavioral proxy, orchestrating and adapting agent outputs before they reach the interface to reduce cognitive load and ensure alignment with human intent.

---

[Democratizing Drug Discovery with an Orchestrated, Knowledge-Driven Multi-Agent Team for User-Guided Therapeutic Design](N/A)

- OrchestRA (Orchestrated Rational drug design Agents): introduces a human-in-the-loop multi-agent platform that unifies biology, chemistry, and pharmacology into an autonomous, iterative drug discovery engine.
- The system is governed by an Orchestrator Agent coordinating the Biologist Agent (KG reasoning), Chemist Agent (molecular design/optimization), and Pharmacologist Agent (ADMET/PBPK simulation) under the ReAct paradigm.
- The platform closes the execution gap by enabling LLM-powered agents to autonomously execute complex computational workflows and establish a dynamic feedback loop for multi-objective optimization.

---

[Agentic Explainable Artificial Intelligence (Agentic XAI) Approach To Explore Better Explanation](http://arxiv.org/abs/2512.21066)

- Agentic XAI (Agentic Explainable Artificial Intelligence): introduces a framework combining SHAP-based explainability with multimodal LLM-driven iterative refinement to generate progressively enhanced agricultural recommendations.
- The system operates as an autonomous agent, using a self-reflective mechanism to analyze XAI outputs, generate supplementary analytical code, and iteratively refine farmer recommendations across 11 rounds.
- Evaluation by human experts and LLMs confirmed that recommendation quality follows an inverted U-shaped trajectory, validating a bias-variance trade-off analogy and establishing strategic early stopping (Rounds 3-4) as critical for optimizing practical utility.

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

