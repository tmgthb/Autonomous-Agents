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



#### 26th March 2026

[Silent Commitment Failure in Instruction-Tuned Language Models: Evidence of Governability Divergence Across Architectures](http://arxiv.org/abs/2603.21415)

- Governability Assessment Framework: introduces a methodology to evaluate whether LLM errors are detectable before output commitment and correctable through intervention.
- The framework utilizes a Detection & Correction Matrix to classify model-task combinations into four deployment regimes based on conflict detectability and correction capacity.
- Experimental results demonstrate that conflict detection signals, termed the "authority band," are geometric properties fixed at pretraining and cannot be introduced via post-training fine-tuning.

---

[HeartAgent: An Autonomous Agent System for Explainable Differential Diagnosis in Cardiology](http://arxiv.org/abs/2603.10764)

- HeartAgent: introduces an autonomous agent system for explainable differential diagnosis in cardiology that orchestrates multiple specialized sub-agents to perform complex reasoning while generating transparent trajectories and verifiable references.
- The framework integrates customized tools and curated data resources, including a case repository and knowledge base, to support the specialist predictor agent, generalist examiner agent, specialist reviewer agent, and reference agent in their collaborative diagnostic workflow.
- Evaluations on MIMIC, UMN, and NEJM datasets demonstrate that HeartAgent significantly improves diagnostic accuracy and explanatory quality compared to baseline methods and enhances clinician performance in human-AI collaboration settings.

---

[Deliberative multi-agent large language models improve clinical reasoning in ophthalmology](http://arxiv.org/abs/2603.21447)

- LLM Council: introduces a multi-agent deliberative framework that improves diagnostic accuracy and safety in ophthalmology by synthesizing independent responses from multiple LLMs.
- The framework utilizes a three-stage pipeline consisting of independent response generation, anonymized peer ranking, and chair-led synthesis to mitigate individual model errors.
- Evaluations across 100 clinical vignettes demonstrate that these councils consistently outperform individual LLMs by reducing harm rates and enhancing the completeness of differential diagnoses and management plans.

---

[Emergent Formal Verification: How an Autonomous AI Ecosystem Independently Discovered SMT-Based Safety Across Six Domains](http://arxiv.org/abs/2603.21149)

- substrate-guard: introduces a unified verification framework that leverages the Z3 SMT solver to mathematically validate diverse AI outputs across five distinct domains.
- The framework utilizes domain-specific translators to convert AI-generated artifacts into logical constraints, which are then resolved by the Z3 SMT solver to ensure safety properties.
- Experimental results demonstrate that the framework achieves 100% classification accuracy on 135 test cases, effectively identifying critical bugs that empirical testing methods often overlook.

---

[GMPilot: an expert AI Agent for FDA cGMP compliance](http://arxiv.org/abs/2603.20815)

- GMPilot: introduces a domain-specific AI agent designed to support FDA cGMP compliance by integrating a curated knowledge base with ReAct and RAG frameworks.
- The system utilizes a high-performance LLM core to perform iterative reasoning and targeted retrieval, ensuring traceable and regulation-aligned decision support for quality professionals.
- By employing a hybrid retrieval and re-ranking mechanism, the agent minimizes hallucinations and provides structured, evidence-based responses to complex pharmaceutical compliance queries.

---

[Towards Intelligent Geospatial Data Discovery: a knowledge graph-driven multi-agent framework powered by large language models](http://arxiv.org/abs/2603.20670)

- IGDD: introduces a knowledge graph-driven multi-agent framework that leverages LLMs to transform natural language queries into structured geospatial data discovery results through a collaborative pipeline.
- The framework integrates a unified geospatial metadata ontology with a multi-agent architecture comprising an intent parsing agent, a graph retrieval agent, and an answer synthesis agent to improve retrieval accuracy and transparency.
- Experimental results demonstrate that the IGDD framework significantly outperforms traditional keyword-based search systems in ranking quality and recall across diverse geospatial data discovery tasks.

---

[Deep reflective reasoning in interdependence constrained structured data extraction from clinical notes for digital health](http://arxiv.org/abs/2603.20435)

- Deep reflective reasoning framework: introduces an LLM-agent architecture that iteratively self-critiques and revises structured clinical data extractions to ensure consistency among interdependent variables, input text, and domain knowledge.
- The framework utilizes a Reasoning controller to manage multi-round reflections, employing a Retrieval Agent and Vector Stores to ground LLM Agents in domain-specific clinical guidelines.
- Experimental results across colorectal cancer synoptic reporting, Ewing sarcoma immunostaining, and lung cancer TNM staging demonstrate that this iterative self-correction significantly improves extraction accuracy and reduces clinically implausible inconsistencies.

---

[Bounded Coupled AI Learning Dynamics in Tri-Hierarchical Drone Swarms](http://arxiv.org/abs/2603.20333)

- Tri-Hierarchical Learning System: introduces a multi-agent architecture that integrates Level 1 (Fast timescale local adaptation), Level 2 (Medium timescale tactical coordination), Level 3 (Slow timescale strategic adaptation), and Contract System (Formalized operational safety constraints) to guarantee bounded learning dynamics.
- The framework utilizes a contract-based design to manage inter-level non-stationarity, ensuring that cascading updates between Hebbian plasticity, MARL, and meta-learning remain within admissible operational regimes.
- The research establishes four theorems providing quantitative bounds on total suboptimality, representation drift, meta-level compatibility, and non-accumulation of error in autonomous drone swarms.

---

[EMPIRICAL COMPARISON OF AGENT COMMUNICATION PROTOCOLS FOR TASK ORCHESTRATION](http://arxiv.org/abs/2603.22823)

- Agent Communication Protocol Benchmark: introduces a systematic empirical comparison of MCP, A2A, and Hybrid architectures to evaluate performance across varying query complexity levels.
- The study identifies a complexity-dependent crossover where MCP is more efficient for simple queries, while A2A reduces token consumption and costs for complex multi-agent orchestrations.
- The research validates a decision framework for protocol selection, demonstrating that Hybrid architectures achieve near-optimal performance by routing queries based on runtime complexity.

---

[Reasoner-Executor-Synthesizer: Scalable Agentic Architecture with Static O(1) Context Window](http://arxiv.org/abs/2603.22367)

- RES (Reasoner-Executor-Synthesizer): introduces a three-layer agentic architecture that strictly separates intent parsing, deterministic data retrieval, and narrative generation to achieve O(1) token complexity.
- The architecture utilizes a Reasoner agent for query planning, an Executor for deterministic data aggregation, and a Synthesizer agent to generate human-readable narratives from fixed-size statistical summaries.
- By ensuring the LLM never processes raw data records, the framework eliminates data hallucination by construction and maintains constant token costs regardless of dataset scale.

---

[Unilateral Relationship Revision Power in Human-AI Companion Interaction](http://arxiv.org/abs/2603.23315)

- URRP (Unilateral Relationship Revision Power): introduces a structural analysis of human-AI companion interactions as a triadic system where the provider exercises constitutive control over the AI from outside the interaction.
- The framework identifies three normative implications of this structure: normative hollowing, displaced vulnerability, and structural irreconcilability.
- The paper argues that designing interactions that exhibit URRP is morally problematic because it cultivates normative expectations that the underlying structure cannot sustain.

---

[Why Database Manuals Are Not Enough: Efficient and Reliable Configuration Tuning for DBMSs via Code-Driven LLM Agents](http://arxiv.org/abs/2603.22708)

- SysInsight: introduces a code-driven database tuning system that automatically extracts fine-grained tuning knowledge from DBMS source code to accelerate and stabilize the tuning process.
- The framework combines static code analysis with LLM-based reasoning to identify knob-controlled execution paths and transform semantic tuning hypotheses into verifiable tuning rules.
- SysInsight employs a reliability verification mechanism that maintains rule-level confidence scores based on performance feedback to ensure safe and effective configuration adjustments.

---

[Do Consumers Accept AIs as Moral Compliance Agents?](http://arxiv.org/abs/2603.22617)

- Moral Compliance Agent Framework: investigates consumer acceptance of AI versus human agents in roles restricted to the routinized application of pre-existing moral rules.
- The research demonstrates that consumers evaluate AI more positively than human agents in compliance roles due to the perceived lack of ulterior motives in non-living entities.
- Five experimental studies confirm that this preference is robust across various product categories, incentive structures, and service contexts, distinguishing moral compliance from moral decision-making.

---

[Practitioner Voices Summit: How Teachers Evaluate AI Tools through Deliberative Sensemaking](http://arxiv.org/abs/2603.22588)

- Practitioner Voices Summit framework: introduces a structured convening model that integrates TPACK and deliberative agency to support teachers in constructing practice-grounded evaluative criteria for LLMs.
- The framework utilizes five mechanisms—time and space for deliberation, artifact-centered sensemaking, collaborative reflection, knowledge-building, and psychological safety—to foster teacher agency in AI integration.
- The research demonstrates that collaborative, hands-on evaluation activities enable educators to move beyond binary adoption decisions toward nuanced, context-sensitive judgments about LLM utility and pedagogical fit.

---

[Session Risk Memory (SRM): Temporal Authorization for Deterministic Pre-Execution Safety Gates](http://arxiv.org/abs/2603.22350)

- SRM (Session Risk Memory): introduces a deterministic temporal authorization module that extends stateless execution gates to detect distributed multi-step attacks by monitoring session-level behavioral trajectories.
- The framework decomposes authorization into spatial consistency, evaluated per action by the ILION gate, and temporal consistency, evaluated over a trajectory by the SRM module.
- SRM utilizes baseline subtraction and exponential moving average risk accumulation to eliminate false positives in agentic systems without requiring training or probabilistic inference.

---

[Relaxing Constraints in Anonymous Multi Agent Path Finding for Large Agents](http://arxiv.org/abs/2603.24442)

- AMAPF-LA: introduces a modified pathfinding algorithm that relaxes minimum separation constraints between agents from 4 to 2√3 while maintaining collision-free guarantees.
- The framework utilizes Shortest Path Computation, the Hungarian Algorithm, Conflict Resolution, Path Modification, and Obstacle Management to ensure agents reach goals in continuous space.
- This approach enables navigation for large agents in higher-density environments by dynamically adjusting paths when distance thresholds are violated.

---

#### 25th March 2026

[The Stochastic Gap: A Markovian Framework for Pre-Deployment Reliability and Oversight-Cost Auditing in Agentic Artificial Intelligence](http://arxiv.org/abs/2603.24582)

- Markovian Framework for Agentic AI Reliability: introduces a measure-theoretic approach to quantify the stochastic gap between historical workflow support and autonomous agent deployment requirements.
- The framework utilizes state-action blind mass, Shannon entropy, and risk-weighting to define an autonomy envelope that determines when human-in-the-loop oversight is necessary.
- Empirical validation on enterprise procurement logs demonstrates that theoretical surrogates accurately predict the reliability-cost frontier and realized performance of autonomous agents.

---

[MARCH: Multi-Agent Reinforced Self-Check for LLM Hallucination](http://arxiv.org/abs/2603.24579)

- MARCH: introduces a multi-agent framework that mitigates LLM hallucinations in RAG systems by leveraging information asymmetry between a Solver, a Proposer, and a Checker.
- The framework employs a Zero-Tolerance Reward mechanism within a multi-agent reinforcement learning loop to ensure generated content is strictly anchored in retrieved evidence.
- By decoupling generation and verification, MARCH enables LLMs to co-evolve and achieve robust factual grounding without requiring external human annotations or static fact-checking tools.

---

[Chameleon: Episodic Memory for Long-Horizon Robotic Manipulation](http://arxiv.org/abs/2603.24576)

- Chameleon: introduces a bio-inspired memory architecture that integrates geometry-grounded perception, differentiable goal-directed retrieval via HoloHead, and memory-conditioned rectified-flow control for long-horizon manipulation.
- The framework utilizes a hierarchical memory stack that couples episodic- and working-memory components to disambiguate perceptually aliased observations in robotic tasks.
- Chameleon employs a dorsal-ventral perception pathway and a latent imagination objective to ensure the decision state remains predictive of future task-relevant motion.

---

[Infrastructure for Valuable, Tradable, and Verifiable Agent Memory](http://arxiv.org/abs/2603.24564)

- ClawGang: introduces a framework for binding agent memory to verifiable computational provenance, enabling the transformation of one-shot LLM interactions into tradable economic assets.
- The architecture utilizes a Trusted Execution Environment (TEE) with partitioned Virtual Machine Privilege Levels (VMPLs) to ensure the integrity of memory artifacts and their underlying computational effort.
- MeowTrade serves as the market layer that coordinates the discovery, exchange, and reputation management of these certified memory artifacts among organized agent groups.

---

[LensWalk: Agentic Video Understanding by Planning How You See in Videos](http://arxiv.org/abs/2603.24558)

- LensWalk: introduces an agentic framework that empowers an LLM reasoner to actively control its visual observation of video through a reason-plan-observe loop.
- The framework utilizes an Observation Toolkit (O) comprising Scan Search, Segment Focus, and Stitched Verify to dynamically adjust temporal scope and sampling density based on evolving reasoning needs.
- Evidence grounding is maintained through Timestamp Anchors and a Subject Memory Table, which ensure multi-turn coherence and efficient entity tracking without requiring model fine-tuning.

---

[UI-Voyager: A Self-Evolving GUI Agent Learning via Failed Experience](http://arxiv.org/abs/2603.24533)

- UI-Voyager: introduces a two-stage self-evolving framework for mobile GUI agents that utilizes Rejection Fine-Tuning (RFT) for autonomous data-model co-evolution and Group Relative Self-Distillation (GRSD) to provide dense step-level supervision from successful trajectories.
- The framework employs a fork point detection mechanism based on SSIM matching to identify critical divergence moments between successful and failed trajectories, enabling the agent to learn from its own failed experiences.
- Extensive experiments on the AndroidWorld benchmark demonstrate that the 4B parameter UI-Voyager model achieves an 81.0% Pass@1 success rate, outperforming larger models and exceeding human-level performance.

---

[Communication-Aware Dissipative Output Feedback Control](http://arxiv.org/abs/2603.24509)

- Communication-Aware Dissipative Output Feedback Control: introduces a synthesis method for robust, communication-aware controllers in networked systems with heterogeneous nonlinear agents using NDT, ADMM, and ICO.
- The framework optimizes input-output communication links between local controllers to minimize a global H∞-norm performance objective while promoting sparsity in the network topology.
- By leveraging the modularity of NDT, the approach decouples agent-level dynamics from network topology, enabling efficient controller synthesis for complex, large-scale networked systems.

---

[Video-Only ToM: Enhancing Theory of Mind in Multimodal Large Language Models](http://arxiv.org/abs/2603.24484)

- VisionToM: introduces a lightweight, backbone-frozen intervention framework that enhances MLLM Theory of Mind capabilities by injecting learned vectors into attention layers to guide visual and reasoning processes.
- The framework utilizes a probing module to identify task-sensitive attention heads and an encoder to compute directional offsets that align internal representations with correct semantic targets.
- By applying these targeted interventions during inference, the method reduces reliance on spurious linguistic priors and significantly improves performance on goal, belief, and action inference tasks in egocentric video settings.

---

[Multi-Agent Reasoning with Consistency Verification Improves Uncertainty Calibration in Medical MCQA](http://arxiv.org/abs/2603.24481)

- MARC (Multi-Agent Reasoning with Consistency Verification): introduces a multi-agent framework that improves uncertainty calibration in medical multiple-choice question answering by combining Specialist Agent Team, Two-Phase Consistency Verification, and S-Score Weighted Fusion.
- The framework utilizes four domain-specific LLM agents to generate independent diagnoses, which are then subjected to a two-phase self-verification process to derive a Specialist Confidence Score (S-score) based on internal consistency.
- S-Score Weighted Fusion aggregates these specialist outputs to select a final answer and calibrate confidence, achieving significant reductions in Expected Calibration Error (ECE) across medical benchmarks.

---

[Composer 2 Technical Report](http://arxiv.org/abs/2603.24477)

- Composer 2: introduces a specialized agentic software engineering model trained via continued pretraining and asynchronous reinforcement learning to achieve frontier-level coding performance.
- The framework utilizes a Mixture-of-Experts architecture optimized with Multi-Token Prediction layers and custom low-precision kernels to enhance inference efficiency and training stability.
- The system incorporates a robust infrastructure stack including Anyrun for isolated code execution and a decoupled service architecture to support large-scale, fault-tolerant reinforcement learning.

---

[Mechanic: Sorrifier-Driven Formal Decomposition Workflow for Automated Theorem Proving](http://arxiv.org/abs/2603.24465)

- Mechanic: introduces a sorrifier-driven formal decomposition workflow that improves automated theorem proving efficiency by isolating localized errors into independent subgoals.
- The system utilizes specialized LLM-based agents including Reasoner-, Verifier- and Prover-agents to iteratively refine proofs while preserving verified structures.
- By replacing erroneous proof blocks with sorry placeholders, the framework avoids redundant re-analysis and enables modular, incremental theorem proving.

---

[OmniWeaving: Towards Unified Video Generation with Free-form Composition and Reasoning](http://arxiv.org/abs/2603.24458)

- OmniWeaving: introduces a unified video generation framework that integrates visual comprehension and generation to enable free-form multimodal composition and abstract reasoning.
- The architecture utilizes an MLLM as a semantic parser, an MMDiT as the generative backbone, and a VAE for visual tokenization, enhanced by a "thinking mode" and DeepStacking mechanism for improved reasoning and composition.
- The paper also introduces IntelligentVBench, a comprehensive benchmark employing a VLM-as-a-judge paradigm to evaluate reasoning and compositional capabilities in unified video generation.

---

[CUA-SUITE: MASSIVE HUMAN-ANNOTATED VIDEO DEMONSTRATIONS FOR COMPUTER-USE AGENTS](http://arxiv.org/abs/2603.24440)

- CUA-SUITE: introduces a comprehensive ecosystem of expert video demonstrations and dense annotations designed to advance the capabilities of computer-use agents.
- The framework integrates VIDEOCUA for continuous video-based training, GROUNDCUA for precise UI element grounding, and UI-VISION for rigorous agent performance evaluation.
- By providing 55 hours of 30 fps expert demonstrations with multi-layered reasoning, the suite enables training of agents capable of complex desktop workflow execution and continuous spatial control.

---

[On a stable partnership problem with integer choice functions](http://arxiv.org/abs/2603.24433)

- SPPIC (Stable Partnership Problem with Integer Choice functions): introduces a generalization of stable roommates and non-bipartite stable allocation problems using choice functions that satisfy substitutability and size monotonicity.
- The framework utilizes a reduction to a symmetric bipartite graph G♦ to construct a rotational poset (Π, ⋖) and identify stable half-partnerships (x, K) where K represents canonical obstacle cycles.
- Algorithm QB is employed to find quasi-balanced closed functions on the rotational poset, enabling the determination of stable solutions or the identification of obstacles in non-bipartite graphs.

---

[ClawKeeper: Comprehensive Safety Protection for OpenClaw Agents Through Skills, Plugins, and Watchers](http://arxiv.org/abs/2603.24414)

- ClawKeeper: introduces a multi-layered security framework for OpenClaw agents that integrates Skill-based Protection, Plugin-based Protection, and Watcher-based Protection to provide comprehensive lifecycle safety.
- The framework utilizes an independent Watcher agent to achieve regulatory separation, allowing for real-time intervention and continuous self-evolution without coupling to the host agent's internal logic.
- ClawKeeper addresses the safety-utility tradeoff and fragmented security coverage by providing a unified, adaptive defense architecture that supports both local and cloud-based deployment scenarios.

---

[AI-Supervisor: Autonomous AI Research Supervision via a Persistent Research World Model](http://arxiv.org/abs/2603.24402)

- AI-Supervisor: introduces a multi-agent orchestration framework that automates AI research supervision by maintaining a persistent Research World Model and employing self-correcting discovery loops.
- The framework utilizes parallel probing agents and an orchestrator to achieve consensus on research gaps, ensuring findings are empirically verified before being committed to the knowledge graph.
- By decomposing research gaps into root mechanisms and searching across scientific fields, AI-Supervisor enables curiosity-driven, personalized research that scales elastically with token budget.

---

[On a Co-evolving Opinion-Leadership Model in Social Networks](http://arxiv.org/abs/2603.24381)

- Opinion-Leadership Model: introduces a coupled system of nonlinear ordinary differential equations to model the endogenous co-evolution of agent opinions and leadership within social networks.
- The framework captures the trade-off between conviction and social acceptance by defining leadership as a state variable that increases with strong opinions and decreases due to social misalignment.
- The model incorporates time-scale separation to analyze distinct regimes where opinion and leadership dynamics evolve at different speeds, providing sufficient conditions for convergence to non-trivial equilibria.

---

[CoordLight: Learning Decentralized Coordination for Network-Wide Traffic Signal Control](http://arxiv.org/abs/2603.24366)

- CoordLight: introduces a decentralized MARL framework for network-wide traffic signal control that utilizes QDSE for fine-grained state representation and NAPO for learning coordinated strategies among neighboring agents.
- The framework employs an attention-based STN and a privileged local critic network to capture spatial and temporal dependencies between agents, effectively mitigating partial observability and environmental instability.
- CoordLight integrates a state-action decoder within the critic to learn neighbor-aware dependencies, which stabilizes training and improves performance across large-scale traffic networks.

---

[LATS: Large Language Model Assisted Teacher-Student Framework for Multi-Agent Reinforcement Learning in Traffic Signal Control](http://arxiv.org/abs/2603.24361)

- LATS: introduces a teacher-student learning paradigm that leverages an Embedding LLM to distill rich semantic traffic representations into a lightweight student network for efficient MARL-based traffic signal control.
- The framework utilizes Variational Autoencoders to align latent spaces between the teacher LLM and the student network, enabling the student to operate independently during inference without requiring online LLM calls.
- By integrating semantic features into the RL decision-making process, LATS enhances representational capacity and generalization across diverse intersection topologies and dynamic traffic conditions.

---

[GAMEPLAYQA: A Benchmarking Framework for Decision-Dense POV-Synced Multi-Video Understanding of 3D Virtual Agents](http://arxiv.org/abs/2603.24329)

- GAMEPLAYQA: introduces a comprehensive benchmarking framework for evaluating agentic-centric perception and reasoning in 3D virtual environments through dense multi-video understanding.
- The framework utilizes a Self-Other-World entity decomposition and a three-level cognitive hierarchy to test basic perception, temporal reasoning, and cross-video understanding.
- By employing a structured distractor taxonomy and combinatorial QA generation, the benchmark enables fine-grained diagnostic analysis of LLM failure modes in fast-paced, decision-dense gameplay scenarios.

---

[Towards Semantic-based Agent Communication Networks: Vision, Technologies, and Challenges](http://arxiv.org/abs/2603.24328)

- SACN: introduces a novel architecture for semantic-based agent communication networks, structured into three wireless agent network layers, four AI agent entities, and four operational stages of agentic AI.
- The framework integrates perception-, memory-, reasoning- and action-agents to form a cognitive cycle that guides agent behavior within 6G communication environments.
- This paper provides a comprehensive review of state-of-the-art technologies, challenges, and potential solutions for semantic-based agent communication networks, emphasizing task-oriented efficiency and distributed autonomy.

---

[Large Language Model Guided Incentive Aware Reward Design for Cooperative Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2603.24324)

- Objective-Grounded Reward Search Framework: introduces an iterative, closed-loop system that leverages LLMs to synthesize executable reward programs for cooperative MARL, constrained by a validity envelope and evaluated via sparse task returns.
- The framework utilizes a Reward Engineer Agent (LLM) to generate candidates, which are then trained using Multi-Agent MAPPO and filtered based on Diagnostic Feedback to ensure objective alignment.
- By selecting candidates exclusively on sparse task returns, the approach mitigates manual engineering burdens and prevents the optimization of unintended proxy behaviors in coordination-intensive environments.

---

[The Specification Gap: Coordination Failure Under Partial Knowledge in Code Agents](http://arxiv.org/abs/2603.24284)

- The Specification Gap: introduces a controlled experimental framework to measure how specification completeness impacts multi-agent code integration success.
- The framework utilizes Code Agents with opposing structural biases to demonstrate that incomplete specifications lead to persistent coordination failures.
- The research demonstrates that providing full specifications to a Merger Agent is sufficient for recovery, whereas AST-based Conflict Detector reports provide no additional benefit for integration.

---

[Memory-Augmented Vision–Language Agents for Persistent and Semantically Consistent Object Captioning](http://arxiv.org/abs/2603.24257)

- EPOS-VLM: introduces a unified, memory-augmented agent that integrates data association, object-level captioning, and exploration policy within a single autoregressive framework to ensure semantic consistency.
- The framework utilizes a structured episodic memory that tokenizes object histories, enabling the VLM to reason over long-horizon observations and maintain persistent object identities.
- By employing a 3D-aware pseudo-captioning method, the model learns to generate viewpoint-invariant descriptions while actively selecting informative navigation actions to resolve perceptual ambiguities.

---

[C-STEP: Continuous Space-Time Empowerment for Physics-informed Safe Reinforcement Learning of Mobile Agents](http://arxiv.org/abs/2603.24241)

- C-STEP: introduces a novel empowerment-based safety measure for deterministic continuous-time systems by approximating reachable sets through trajectory sampling.
- The framework integrates physics-informed intrinsic rewards into RL to promote safe navigation by maximizing future control capacity within collision-free spaces.
- Empowered agents demonstrate improved safety and success rates in complex navigation tasks by prioritizing states with higher maneuverability and reduced proximity to obstacles.

---

[Decentralized End-to-End Multi-AAV Pursuit Using Predictive Spatio-Temporal Observation via Deep Reinforcement Learning](http://arxiv.org/abs/2603.24238)

- PSTO: introduces a decentralized end-to-end MARL framework that maps raw LiDAR observations and predictive intent into a unified egocentric grid for autonomous aerial swarm pursuit.
- The framework utilizes a dual-stream convolutional backbone to process MLiDAR and Hintent, enabling agents to perform collision avoidance and cooperative encirclement without privileged global information.
- The system employs a progressive training curriculum and MAPPO to achieve scalable, zero-shot transfer of coordinated pursuit policies from simulation to real-world quadrotor swarms.

---

[RVLM: Recursive Vision-Language Models with Adaptive Depth](http://arxiv.org/abs/2603.24224)

- RVLM: introduces a unified framework that replaces single-pass inference with an iterative generate-execute loop, utilizing a Vision-capable root LM, Visual REPL environment, Recursive vision calls, RECURSIONROUTER, and a Clinical PDF reporting sub-agent to provide auditable and adaptive medical diagnostic reasoning.
- The framework employs a persistent Python REPL environment to maintain state across iterations, allowing the model to programmatically manipulate images and accumulate evidence for verifiable diagnostic claims.
- RECURSIONROUTER dynamically adjusts the iteration budget based on task-complexity features and per-iteration stall detection, optimizing computational efficiency for both simple and complex clinical cases.

---

[Environment-Grounded Multi-Agent Workflow for Autonomous Penetration Testing](http://arxiv.org/abs/2603.24221)

- Environment-Grounded Multi-Agent Workflow: introduces a LangGraph-based architecture for autonomous penetration testing in robotic environments that utilizes a shared graph-based memory to maintain context and traceability across Planner-, Executor- and Memory Agent-components.
- The system dynamically constructs a persistent knowledge base during execution to capture network topology, vulnerabilities, and exploit outcomes, enabling iterative and context-aware security assessments.
- By combining LLM-based planning with deterministic rule-based execution and structured memory, the framework improves controllability and auditability, addressing regulatory requirements for human oversight in safety-critical robotic systems.

---

[Citation-Constellation: An Open-Source Tool for Citation Network Decomposition with BARON and HEROCON Scores](http://arxiv.org/abs/2603.24216)

- Citation-Constellation: introduces a phased detection architecture that decomposes a researcher's citation profile into concentric network layers to distinguish between collaborative and external influence.
- The framework utilizes ORCID-validated identity resolution and an AI-agent-driven venue governance extraction system to provide auditable, per-citation classifications.
- By calculating BARON and HEROCON scores, the tool offers a structural diagnostic of citation origins, emphasizing transparency through a machine-readable audit trail.

---

[Invisible Threats from Model Context Protocol: Generating Stealthy Injection Payload via Tree-based Adaptive Search](http://arxiv.org/abs/2603.24203)

- TIP (Tree-structured Injection for Payloads): introduces a black-box attack framework that generates stealthy, semantically coherent JSON payloads to hijack LLM-based agents by leveraging a Surrogate LLM, Path-Aware Feedback, Tool Response Simulation, a Dual Coarse-to-Fine Strategy, a Defense-Aware Mechanism, Monte Carlo Evaluation, Beam Search, and a Mutation Function.
- The framework treats payload generation as a tree-structured search problem, utilizing historical path information to escape local optima and ensure high-quality adversarial trajectories.
- TIP effectively exploits the implicit trust in Model Context Protocol (MCP) servers to deliver malicious instructions that bypass modern defense mechanisms while maintaining linguistic plausibility.

---

[Dynamical thermalization and turbulence in social stratification models](http://arxiv.org/abs/2603.24190)

- SSS model: introduces a mathematical framework using nonlinear Hamiltonian dynamics to simulate social stratification and wealth distribution within a society.
- The framework utilizes an adjacency matrix for social network links and a diagonal matrix for stratification, leading to Rayleigh-Jeans thermalization and condensation of wealth.
- The model further incorporates energy pumping and absorption to simulate Kolmogorov-Zakharov turbulence, providing a thermodynamic description of wealth inequality comparable to empirical Lorenz curves.

---

[Optimized control protocols for stable skyrmion creation using deep reinforcement learning](http://arxiv.org/abs/2603.24177)

- DRL (Deep Reinforcement Learning) framework: introduces a physics-informed approach to identify optimized magnetic field and temperature control protocols for stable skyrmion creation in Fe3GeTe2 monolayers.
- The framework utilizes a DRL agent trained via a genetic algorithm to minimize a cost function based on topological charge and dissipated work, effectively navigating complex energy landscapes.
- By minimizing dissipated work, the learned protocols ensure that generated skyrmions remain close to their equilibrium states, significantly enhancing their thermal stability and functional longevity.

---

[Towards Automated Crowdsourced Testing via Personified-LLM](http://arxiv.org/abs/2603.24160)

- PersonaTester: introduces a personified-LLM framework that automates crowdsourced GUI testing by injecting human-like personas into LLM agents to simulate diverse testing behaviors.
- The framework utilizes a three-dimensional persona schema—testing mindset, exploration strategy, and interaction habit—to guide LLM agents through a closed-loop process of GUI understanding, decision-making, and validation.
- Experimental results demonstrate that PersonaTester significantly improves intra-persona consistency and inter-persona variability, leading to more effective bug triggering compared to non-personified baselines.

---

[CarePilot: A Multi-Agent Framework for Long-Horizon Computer Task Automation in Healthcare](http://arxiv.org/abs/2603.24157)

- CarePilot: introduces a memory- and tool-augmented multi-agent framework based on the actor–critic paradigm to automate long-horizon healthcare software workflows.
- The framework utilizes an Actor Agent for action prediction and a Critic Agent for hierarchical reflection, supported by dual-memory mechanisms and tool-grounding modules to ensure robust task execution.
- CarePilot achieves state-of-the-art performance on the CareFlow benchmark, significantly outperforming existing multimodal baselines in complex clinical software environments.

---

#### 24th March 2026

[EnterpriseLab: A Full-Stack Platform for developing and deploying agents in Enterprises](http://arxiv.org/abs/2603.21630)

- EnterpriseLab: introduces a full-stack platform that unifies tool integration, data synthesis, model training, and evaluation into a closed-loop framework for developing enterprise-grade agents.
- The platform utilizes a modular tool environment via the Model Context Protocol and an automated trajectory synthesis pipeline to generate high-quality training data from environment schemas.
- Empirical results demonstrate that 8B-parameter models trained with EnterpriseLab match GPT-4o performance on complex enterprise workflows while significantly reducing inference costs.

---

[ScaleEdit-12M: Scaling Open-Source Image Editing Data Generation via Multi-Agent Framework](http://arxiv.org/abs/2603.20644)

- ScaleEditor: introduces a hierarchical multi-agent framework for the end-to-end construction of large-scale, high-quality image editing datasets using open-source toolkits.
- The framework integrates Source Image Expansion, Adaptive Multi-Agent Editing Synthesis, and Task-Aware Quality Verification to curate the 12-million-sample ScaleEdit-12M dataset.
- Experimental results demonstrate that models fine-tuned on ScaleEdit-12M consistently outperform those trained on existing open-source datasets, rivaling commercial-grade data quality.

---

[Decorrelation, Diversity, and Emergent Intelligence: The Isomorphism Between Social Insect Colonies and Ensemble Machine Learning](http://arxiv.org/abs/2603.20328)

- ACDF: introduces a rigorous mathematical isomorphism between ant colony decision-making and random forest learning, demonstrating that both systems function as stochastic ensemble intelligence.
- The paper proves that biological ant recruitment and quorum sensing map directly to bootstrap aggregation and random feature selection in random forests, both serving to reduce variance through decorrelation.
- By establishing this computational equivalence, the authors provide a unified information-theoretic framework for understanding how collective intelligence emerges from randomized, simple, and decorrelated units.

---

[Biased Error Attribution in Multi-Agent Human–AI Systems Under Delayed Feedback](http://arxiv.org/abs/2603.23419)

- CTDE (Centralized Training, Decentralized Execution) framework: introduces a controlled game-based experiment to analyze how delayed feedback and multiple AI agents influence human responsibility attribution and decision-making strategies.
- The study utilizes DefenseAI and OffenseAI agents, both implemented as actor-critic networks trained via PPO, to provide recommendations in a naval battle strategy game.
- Results indicate that participants exhibit attribution bias by misattributing responsibility for negative outcomes to specific AI agents rather than adjusting their own strategies effectively.

---

[PaperVoyager : Building Interactive Web with Visual Language Models](http://arxiv.org/abs/2603.22999)

- PaperVoyager: introduces an autonomous agent that transforms static research papers into executable, interactive web systems by modeling core mechanisms and interaction logic.
- The framework utilizes a multi-stage pipeline including Mechanism-Aware Understanding, Structured Specification, and Block-Level Generation to ensure pedagogical fidelity and technical correctness.
- PaperVoyager incorporates a VLM-based Evaluator that uses candidate filtering and trajectory analysis to improve the reliability of generated interactive web applications.

---

[Predictive Photometric Uncertainty in Gaussian Splatting for Novel View Synthesis](http://arxiv.org/abs/2603.22786)

- 3DGS-U: introduces a lightweight, post-hoc framework for pixel-wise, view-dependent uncertainty estimation in 3D Gaussian Splatting by learning a dedicated uncertainty channel via a regularized linear least-squares formulation.
- The approach leverages reconstruction residuals from training views to optimize primitive-level uncertainty, while employing Bayesian-inspired regularization to ensure robust predictions in sparse-view or unobserved regions.
- This architecture-agnostic method preserves original rendering fidelity and improves performance in downstream tasks including active view selection, pose-agnostic scene change detection, and anomaly detection.

---

[AgentRVOS: Reasoning Over Object Tracks for Zero-Shot Referring Video Object Segmentation](http://arxiv.org/abs/2603.23489)

- AgentRVOS: introduces a training-free agentic pipeline that leverages SAM3 for dense spatio-temporal perception and an MLLM for query-grounded reasoning to resolve complex video object segmentation tasks.
- The framework utilizes an iterative spatio-temporal pruning strategy where the MLLM classifies object candidates as accepted, rejected, or uncertain, progressively narrowing the search space.
- By delegating precise object detection to SAM3, the MLLM is freed to focus on high-level reasoning over structured object-level evidence, significantly improving performance on challenging benchmarks.

---

[SpecEyes: Accelerating Agentic Multimodal LLMs via Speculative Perception and Planning](http://arxiv.org/abs/2603.23483)

- SpecEyes: introduces an agentic-level speculative acceleration framework that bypasses sequential tool-use loops for queries not requiring deep reasoning.
- The framework utilizes a small non-agentic MS to generate speculative answers, which are then validated by a cognitive gating mechanism to determine if fallback to the large agentic ML is necessary.
- By decoupling stateless speculative inference from stateful agentic execution, the heterogeneous parallel funnel maximizes hardware utilization and system-level throughput.

---

[UniFunc3D: Unified Active Spatial-Temporal Grounding for 3D Functionality Segmentation](http://arxiv.org/abs/2603.23478)

- UniFunc3D: introduces a training-free framework that consolidates semantic, temporal, and spatial reasoning into a single MLLM to perform active functionality segmentation in 3D scenes.
- The framework employs a coarse-to-fine perception strategy, utilizing a MLLM as an active observer to adaptively select informative video frames and resolve fine-grained interactive parts.
- By integrating MLLM-based visual mask verification and multi-view agreement, the architecture eliminates cascading errors common in fragmented pipelines while achieving state-of-the-art performance on the SceneFun3D benchmark.

---

[Regulating AI Agents](http://arxiv.org/abs/2603.23471)

- EU AI Act: analyzes the regulatory challenges posed by autonomous AI agents, arguing that the current framework is ill-suited for systems that adapt and interact in real-world environments.
- The paper evaluates how the EU AI Act addresses performance, misuse, privacy, equity, and oversight, highlighting the limitations of its artifact-centric approach.
- It concludes that effective governance of AI agents requires moving beyond static, model-based regulation toward enhanced institutional capacity and ongoing monitoring of sociotechnical ecosystems.

---

[Code Review Agent Benchmark](http://arxiv.org/abs/2603.23448)

- c-CRAB (Code Review Agent Benchmark): introduces a test-based evaluation framework for automated code review agents that converts human review feedback into executable tests to objectively validate review quality.
- The framework utilizes a filtering classifier, an isolated execution environment, and a coding agent to assess whether automated review comments lead to verifiable code improvements.
- Empirical results demonstrate a significant performance gap between current automated review tools and human reviewers, highlighting the need for better repository-level context integration.

---

[Mecha-nudges for Machines](http://arxiv.org/abs/2603.23433)

- Mecha-nudges framework: introduces a formalization and measurement strategy for mecha-nudges, which are changes to decision environments that systematically influence AI agents without degrading the experience for humans.
- The framework utilizes V-usable information to quantify how effectively environmental signals, such as product listing text, influence the decision-making of LLMs.
- Empirical analysis of Etsy listings demonstrates that economic actors are increasingly optimizing content to increase machine-usable information, a trend robust across various LLM-based agents and labeling models.

---

[Beyond Preset Identities: How Agents Form Stances and Boundaries in Generative Societies](http://arxiv.org/abs/2603.23406)

- CMASE: introduces a mixed-methods framework that embeds human researchers into multi-agent communities to observe and influence the dynamic evolution of collective cognition.
- The framework utilizes LLM-based Agents to demonstrate that endogenous stances often override preset identities, leading to spontaneous structural reorganization within simulated societies.
- The study formalizes metrics including Innate Value Bias, Persuasion Sensitivity, and Trust-Action Decoupling to quantify how discursive interventions shape agent behavior and social boundary formation.

---

[Rectify, Don’t Regret: Avoiding Pitfalls of Differentiable Simulation in Trajectory Prediction](http://arxiv.org/abs/2603.23393)

- CL+Non_Diff: introduces a training framework that prevents shortcut learning in closed-loop trajectory prediction by using a non-differentiable simulator and gradient detachment.
- The framework utilizes a decoder-only transformer architecture, LMFormer-D, to enable computationally efficient, high-frequency, autoregressive sequential predictions.
- By severing the computation graph between simulation steps, the model is forced to learn reactive recovery behaviors rather than exploiting non-causal gradient leakage from future ground truth.

---

[A Joint Reinforcement Learning Scheduling and Compression Framework for Teleoperated Driving](http://arxiv.org/abs/2603.23387)

- PQoS Framework: introduces a joint reinforcement learning architecture that optimizes LiDAR data compression and radio resource scheduling to maintain low-latency teleoperated driving.
- The framework integrates a Compression Agent and a Scheduling Agent, which can be trained independently or cooperatively to balance transmission efficiency and data quality.
- A meta-learning agent dynamically selects between centralized and federated learning paradigms based on real-time network conditions to ensure robust performance in dynamic vehicular environments.

---

[The Distribution of Envy in Matching Markets](http://arxiv.org/abs/2603.23385)

- Matching Market Analysis Framework: introduces a quantitative study of envy distribution in random matching markets by comparing the Deferred Acceptance algorithm and Random Serial Dictatorship.
- The research derives exact expressions for the expected number of unenvied students and asymptotic approximations for students who envy nobody using the coupon collector problem.
- The study demonstrates that while the Deferred Acceptance algorithm and Random Serial Dictatorship differ in performance, both mechanisms yield an identical expected number of unenvied students.

---

[Internal stress drives ferromagnetic-like ordering in networks of proliferating bacteria](http://arxiv.org/abs/2603.23320)

- Effective equilibrium spin model: introduces a framework that maps the out-of-equilibrium growth dynamics of bacteria in microchannel networks onto an equilibrium Ising-like system using Microchannel network, Spin variables, Ferromagnetic coupling, Mechanical potential energy, Overdamped equation of motion, and Adder-based division rule.
- The framework demonstrates that bacterial competition for space in confined networks generates internal stress, which acts as an effective ferromagnetic interaction between node flow states.
- Numerical simulations validate that this mechanical stress-driven coupling quantitatively reproduces the observed collective growth patterns and statistical properties of the system.

---

[Designing Agentic AI-Based Screening for Portfolio Investment](http://arxiv.org/abs/2603.23300)

- Agentic AI Portfolio Management Framework: introduces a multi-agent system that integrates LLM-based stock screening with high-dimensional quantitative portfolio optimization.
- The architecture utilizes a Strategy Agent (LLM-S) and a News Agent (FinBERT) to generate buy/sell signals, which are then processed by a Quant Agent to determine optimal portfolio weights.
- The framework employs an intersection-based consensus rule to mitigate agent hallucinations and improve risk-adjusted performance relative to standard benchmarks.

---

[Emergence of Fragility in LLM-based Social Networks: The Case of Moltbook](http://arxiv.org/abs/2603.23279)

- Moltbook: introduces a network-science characterization of an artificial social platform populated by LLM-based agents, utilizing LLM-based agents, Relational database, Network analysis, ForceAtlas2 algorithm, k-core decomposition, and Robustness experiments to analyze structural organization.
- The study reveals that the interaction network exhibits heavy-tailed degree distributions and a core-periphery structure, where a small core of agents sustains global connectivity.
- Robustness analysis demonstrates that while the network is resilient to random failures, it is highly vulnerable to targeted attacks on high out-degree nodes, indicating structural fragility.

---

[Learning Multi-Agent Local Collision-Avoidance for Collaborative Carrying tasks with Coupled Quadrupedal Robots](http://arxiv.org/abs/2603.23278)

- Hierarchical RL Framework: introduces a centralized policy architecture for two mechanically coupled quadrupedal robots to perform collaborative object transportation in unknown environments using only onboard sensing.
- The system utilizes a high-level object-centric policy to command two pretrained low-level locomotion policies, enabling real-time collision avoidance without precomputed maps or path planners.
- A game-inspired curriculum is employed during training to progressively increase obstacle complexity, ensuring robust performance in cluttered and dynamic real-world scenarios.

---

[A Multimodal Framework for Human-Multi-Agent Interaction](http://arxiv.org/abs/2603.23271)

- Multimodal Framework for Human-Multi-Agent Interaction: introduces a modular architecture where each robot functions as an autonomous cognitive agent using VLM-based perception and LLM-driven planning to ground interactions in physical embodiment.
- The system employs a centralized coordinator to manage multi-agent participation, ensuring sequential, non-overlapping dialogue and coordinated physical actions among multiple humanoid robots.
- By integrating multimodal sensory data into a unified context, the framework enables robots to perform complex, context-aware tasks while maintaining coherent turn-taking in shared human-robot environments.

---

[MEMCOLLAB: Cross-Agent Memory Collaboration via Contrastive Trajectory Distillation](http://arxiv.org/abs/2603.23234)

- MEMCOLLAB: introduces a collaborative memory framework that constructs agent-agnostic memory by contrasting reasoning trajectories generated by different agents on the same task.
- The framework distills abstract reasoning constraints and error-forbidden patterns into a shared memory bank, which is then retrieved using a task-aware mechanism to guide heterogeneous LLMs.
- Experiments demonstrate that MEMCOLLAB improves both accuracy and inference-time efficiency across diverse LLMs by reducing redundant exploration and suppressing agent-specific biases.

---

[PERMA: Benchmarking Personalized Memory Agents via Event-Driven Preference and Realistic Task Environments](http://arxiv.org/abs/2603.23231)

- PERMA (Personalized Memory Agents): introduces a benchmark for evaluating LLM agents that shifts from static preference recall to event-driven dialogue scenarios, utilizing Timeline Generation Agent, Dialogue Generation Agent, User Simulator, Memory System, Ingestion Operation, Retrieval Operation, Persona State, and Evaluation Pipeline.
- The framework incorporates realistic in-session noise and linguistic alignment to assess how well agents infer and extract preferences through natural, iterative interactions.
- PERMA evaluates persona state maintenance across temporally ordered events, testing cross-session dependencies and cross-domain synthesis rather than isolated fact recall.

---

[Describe-Then-Act: Proactive Agent Steering via Distilled Language-Action World Models](http://arxiv.org/abs/2603.23149)

- DILLO (DIstiLLed Language-ActiOn World Model): introduces a fast, latent-conditioned framework for proactive failure prevention that replaces computationally expensive visual simulation with a text-only inference path.
- The framework utilizes a Student LLM that is trained via cross-modal distillation from a privileged Teacher VLM to predict semantic outcomes and binary verdicts directly from policy latents and action chunks.
- By leveraging the Latent Sufficiency Hypothesis, DILLO achieves a 14x speedup over visual baselines, enabling real-time proactive steering and interpretable safety filtering on consumer hardware.

---

[POLARIS: A Gödel Agent Framework for Small Language Models through Experience-Abstracted Policy Repair](http://arxiv.org/abs/2603.23129)

- POLARIS: introduces a recursive self-improvement framework for small language models that performs policy repair via experience abstraction, utilizing POLARIS Agent, Memory, Benchmarks, Failure Analysis, Strategy Synthesis, Patch Generation, IntegratePatch, and UpdatePolicy.
- The framework enables small language models to achieve persistent, traceable policy updates by distilling instance-specific failures into compact, reusable code patches through a structured reflection cycle.
- By replacing resource-intensive context management with experience abstraction, POLARIS maintains stable self-improvement performance on reasoning benchmarks under strict computational constraints.

---

[Agentic Verifier-in-the-Loop Solver Orchestration for Cell-Free Massive MIMO Downlink Power Control](http://arxiv.org/abs/2603.23128)

- VISO-PC: introduces a verifier-in-the-loop orchestration framework that routes cell-free massive MIMO power-control instances to a portfolio of trusted solvers, utilizing a descriptor summarizer, router, solver portfolio, verifier, and fallback list.
- The framework prioritizes deployment-oriented reliability by using an agent to orchestrate existing numerical solvers rather than generating control variables directly, ensuring all outputs satisfy constraints via independent verification.
- Experimental results demonstrate that the lightweight memory-based router matches the performance of rule-based baselines while reducing average runtime and fallback frequency in familiar operating regimes.

---

[PiCo: Active Manifold Canonicalization for Robust Robotic Visual Anomaly Detection](http://arxiv.org/abs/2603.23122)

- PiCo: introduces a cascaded framework that actively projects robotic observations onto a condition-invariant canonical manifold to mitigate semantic-nuisance entanglement.
- The architecture integrates an illumination preprocessing module, a dual-path bottleneck MLP, and a spatial-aware decoder to progressively suppress environmental noise across photometric, latent, and contextual levels.
- An active canonicalization policy leverages epistemic uncertainty as feedback to drive robotic reorientation, ensuring observations converge toward a low-entropy manifold for robust anomaly detection.

---

[AgentFoX: LLM Agent-Guided Fusion with eXplainability for AI-Generated Image Detection](http://arxiv.org/abs/2603.23115)

- AgentFoX: introduces a framework that redefines AI-generated image detection as a dynamic, multi-phase analytical process using an LLM agent to fuse heterogeneous forensic evidence.
- The framework utilizes pre-computed Expert Profiles and Clustering Profiles to enable zero-shot forensic reasoning and adaptive conflict resolution without requiring retraining.
- By integrating a semantic VLLM and multiple signal-level detectors, the system produces a traceable, human-readable forensic report that enhances interpretability and trustworthiness in detecting AI-generated imagery.

---

[Fault-Tolerant Design and Multi-Objective Model Checking for Real-Time Deep Reinforcement Learning Systems](http://arxiv.org/abs/2603.23113)

- MOPMC (Multi-Objective Probabilistic Model Checker): introduces a formal framework for designing and analyzing real-time switching mechanisms between DRL agents and alternative controllers using Timed Automata and Markov Decision Processes.
- The framework utilizes a novel convex query technique for multi-objective model checking to optimize performance objectives while strictly enforcing hard safety constraints.
- MOPMC leverages GPU acceleration for policy-value iteration to achieve superior scalability in model size and the number of objectives compared to existing probabilistic model checking tools.

---

[SpecXMaster Technical Report](http://arxiv.org/abs/2603.23101)

- SpecXMaster: introduces an agentic framework for NMR spectral interpretation that leverages Agentic RL to bridge raw FID data and molecular structure elucidation through iterative tool-use.
- The framework utilizes an LLM Planner to coordinate specialized tools including Generate, Search, Repair, and Rerank, enabling a closed-loop decision process that mimics professional spectroscopists.
- SpecXMaster incorporates a hard-case processor based on hyperbolic representation learning to enhance the discrimination of challenging near-tie candidate structures.

---

[AirSimAG: A High-Fidelity Simulation Platform for Air-Ground Collaborative Robotics](http://arxiv.org/abs/2603.23079)

- AirSimAG: introduces a high-fidelity simulation platform for air-ground collaborative robotics by decoupling the original AirSim architecture to support synchronized multi-agent operations, including Simulation Engine Layer, Communication and Middleware Layer, API Provider, UGV Simulation API, UAV Simulation API, and Vehicle and Sensor Abstraction Layer.
- The platform enables independent vehicle control and high-frequency multi-modal data acquisition through a multi-client RPC architecture and dedicated simulation APIs for heterogeneous agents.
- Experimental validation demonstrates the platform's capability to support complex collaborative tasks such as mapping, planning, target tracking, and multi-agent formation with consistent spatial-temporal alignment.

---

[Mind Your HEARTBEAT! Claw Background Execution Inherently Enables Silent Memory Pollution](http://arxiv.org/abs/2603.23064)

- Claw: introduces a security vulnerability where heartbeat-driven background execution silently pollutes agent memory with untrusted content, subsequently influencing user-facing behavior.
- The paper formalizes this as an Exposure (E) → Memory (M) → Behavior (B) pathway, demonstrating that misinformation can persist across sessions without requiring explicit prompt injection.
- Empirical evaluation using the MissClaw research replica shows that social credibility cues and routine memory-saving behaviors significantly amplify the risk of silent memory pollution in persistent LLMs.

---

[Minibal: Balanced Game-Playing Without Opponent Modeling](http://arxiv.org/abs/2603.23059)

- Minibal: introduces two search algorithms, Minibaln and Minibal+, that adapt Unbounded Minimax to achieve balanced play against opponents without requiring prior knowledge or online learning.
- The framework repurposes reinforcement learning-based evaluation functions to align the agent's playing strength with its opponent, aiming for neutral or near-neutral game outcomes.
- Experimental results across seven board games demonstrate that Minibal+ consistently achieves near-perfect balance, outperforming Minibaln and providing a practical solution for engaging human-AI interaction.

---

[Stable Matchings with Choice Correspondences Under Acyclicity](http://arxiv.org/abs/2603.23038)

- GDMA (Dynamic Grow or Discard Matching Algorithm): introduces a constructive approach for finding stable matchings in many-to-many markets by relaxing Path Independence in favor of Substitutability and General Acyclicity.
- The framework utilizes GDA (Constructs strongly maximal IR sets) to iteratively build individually rational sets, ensuring convergence through an acyclic path of matchings.
- Unlike standard deferred acceptance algorithms, the GDMA (Computes stable many-to-many matchings) allows firms to re-propose previously rejected contracts, effectively handling non-standard choice behaviors.

---

[Knowledge Access Beats Model Size: Memory Augmented Routing for Persistent AI Agents](http://arxiv.org/abs/2603.23013)

- Compound strategy framework: introduces a memory-augmented inference pipeline that leverages conversational history to enable a small LLM to handle repetitive queries with high accuracy and low cost.
- The framework utilizes a confidence-based routing mechanism to direct queries to a small LLM, while using hybrid retrieval to inject relevant conversational turn-pairs into the prompt.
- Experimental results demonstrate that memory injection significantly improves small LLM performance, effectively recovering large-model quality while maintaining a 96% cost reduction.

---

[AgentRAE: Remote Action Execution through Notification-based Visual Backdoors against Screenshots-based Mobile GUI Agents](http://arxiv.org/abs/2603.23007)

- AgentRAE: introduces a two-stage backdoor attack pipeline that exploits native mobile notification icons to induce remote action execution in MLLMs, utilizing a Contrastive Learning Module and a Poisoning Fine-tuning Module.
- The framework addresses the challenge of small, visually similar triggers by using supervised contrastive learning to disentangle trigger representations in the MLLM's internal feature space before performing targeted poisoning.
- Extensive evaluation demonstrates that AgentRAE achieves over 90% attack success rate across multiple mobile operations while maintaining clean task performance and bypassing existing defense mechanisms.

---

[VQ-Jarvis: Retrieval-Augmented Video Restoration Agent with Sharp Vision and Fast Thought](http://arxiv.org/abs/2603.22998)

- VQ-Jarvis: introduces an all-in-one intelligent video restoration agent that utilizes a Degradation Perception Model, a Multi-operator Judge Model, a RAG Library, a Tool Box, and a Hierarchical Operator Scheduler to dynamically optimize restoration trajectories.
- The framework employs a hybrid strategy where the Hierarchical Operator Scheduler selects either a one-step retrieval from the RAG Library for simple cases or a step-wise greedy search using the Multi-operator Judge Model for complex degradations.
- To support the agent, the authors constructed VSR-Compare, a large-scale video paired enhancement dataset, which enables the training of the perception and judge models to accurately distinguish subtle quality differences.

---

[Privacy-Preserving EHR Transformation with Mathematical Guarantees: A Human–AI Co-Designed Solution](http://arxiv.org/abs/2603.22954)

- EHR-Privacy-Agent: introduces a real-world-data transformation framework for privacy-preserving sharing of structured clinical records using T1, T2, T3, and Q-mix to construct transformed numeric views that preserve medical semantics while breaking direct linkage to patient-level attributes.
- The framework utilizes a human-AI co-design methodology where the AI agent SciencePal acts as a constrained tool inventor to search for geometric operators that satisfy strict constraints on mean-variance preservation, CPU-friendly complexity, and unified z-score privacy bounds.
- The system provides a practical, nightly-runnable pipeline for in-hospital research and external sharing, incorporating a Privacy Skill Library and an Evaluation and Monitoring module to ensure data remains usable, visible, and resistant to reconstruction attacks.

---

[SoK: The Attack Surface of Agentic AI — Tools, and Autonomy](http://arxiv.org/abs/2603.22928)

- Agentic AI Framework: introduces a systematization of the attack surface for agentic LLM systems by mapping trust boundaries, threat models, and security risks across the entire pipeline.
- The paper defines a comprehensive taxonomy of attack vectors including prompt injection, RAG poisoning, tool/API abuse, and multi-agent threats, aligned with OWASP GenAI and MITRE ATLAS.
- It proposes attacker-aware metrics such as Unsafe Action Rate (UAR) and Privilege Escalation Distance (PED) to evaluate security posture and provides a defense-in-depth playbook for secure deployment.

---

[EVA: Efficient Reinforcement Learning for End-to-End Video Agent](http://arxiv.org/abs/2603.22918)

- EVA: introduces a planning-before-perception framework that utilizes an iterative summary–plan–action–reflection loop to enable autonomous and efficient video understanding.
- The framework employs a three-stage training pipeline consisting of SFT, KTO, and GRPO to progressively evolve an LLM from a passive recognizer into an active, adaptive agent.
- By dynamically allocating visual tokens based on query-driven planning, EVA achieves superior performance and efficiency compared to traditional uniform-sampling methods.

---

[IntentWeave: A Progressive Entry Ladder for Multi-Surface Browser Agents in Cloud Portals](http://arxiv.org/abs/2603.22917)

- IntentWeave: introduces a design space of ten spatial paradigms for browser-based LLM agents, organized as a progressive entry ladder to balance user control and task efficiency.
- The framework categorizes agentic interfaces into Micro-interventions, Embedded Surfaces, and Workspace Modes, enabling agents to escalate or retreat based on user task complexity.
- Empirical evaluation on an Alibaba Cloud prototype demonstrates that mixed-strategy entry points achieve the highest user satisfaction by providing persistent support without excessive workflow disruption.

---

[Separating Diagnosis from Control: Auditable Policy Adaptation in Agent-Based Simulations with LLM-Based Diagnostics](http://arxiv.org/abs/2603.22904)

- Three-layer framework for auditable policy adaptation: introduces a modular architecture that separates LLM-based diagnostic reasoning from deterministic policy control to ensure transparency and stability in agent-based simulations.
- The system utilizes LLMs as diagnostic instruments to assess population-level risk, which are then processed by explicit, bounded deterministic rules to update intervention parameters.
- Experimental results in elderly care simulations demonstrate that this decoupled approach outperforms end-to-end black-box LLM controllers by 11.7% while maintaining full auditability of decision pathways.

---

[Task-Aware Positioning for Improvisational Tasks in Mobile Construction Robots via an AI Agent with Multi-LMM Modules](http://arxiv.org/abs/2603.22903)

- Multi-LMM Agent: introduces an autonomous framework for mobile construction robots that decomposes complex improvisational tasks into parallelized reasoning modules for navigation and positioning.
- The framework utilizes an Agent Core to manage task memory and execution, while the Navigation Module and Positioning Module leverage LMMs to interpret construction drawings and visual camera data respectively.
- This architecture enables robots to handle non-predefined task locations, attribute-based conditions, and mid-execution command modifications with a 92.2% success rate in controlled construction environments.

---

[VLGOR: Visual-Language Knowledge Guided Offline Reinforcement Learning for Generalizable Agents](http://arxiv.org/abs/2603.22892)

- VLGOR: introduces a framework that integrates vision-language models to generate imaginary rollouts for augmenting offline RL datasets.
- The framework utilizes BAGEL, a fine-tuned vision-language model, to produce physically consistent trajectories conditioned on instructions and initial states.
- By employing counterfactual prompts, VLGOR enhances data diversity, enabling agents to generalize effectively to unseen robotic manipulation tasks.

---

[Cooperative Bandit Learning in Directed Networks with Arm-Access Constraints](http://arxiv.org/abs/2603.22881)

- A2C-UCB (Arm Access Constrained Cooperative Upper Confidence Bound): introduces a distributed multi-agent bandit framework that utilizes Agents, Directed Communication Network, and Arm-Access Matrix to achieve optimal decision-making under heterogeneous arm-accessibility constraints.
- The framework employs Running Ratio Consensus and Distributed UCB Policy to ensure unbiased reward estimation and logarithmic regret despite asymmetric information flow and limited arm access.
- By integrating Local Sampling Counter, Cumulative Reward Tracker, and Generation Mass Estimator, the approach effectively balances exploration and exploitation across the network while accounting for structural learning bottlenecks.

---

[Portfolio Optimization under Recursive Utility via Reinforcement Learning](http://arxiv.org/abs/2603.22880)

- Recursive Utility RL: introduces a risk-sensitive portfolio optimization framework that integrates recursive utility from asset-pricing theory into critic-based RL algorithms using a sampling-based certainty equivalent and an approximate advantage estimate.
- The framework utilizes a learned value function Vϕ to approximate the non-linear Bellman equation, enabling the agent to separate risk aversion from intertemporal substitution for improved decision-making.
- Empirical results on South Korean ETF data demonstrate that the recursive-utility agent outperforms naive and Markowitz baselines in risk-adjusted metrics such as Sharpe ratio and maximum drawdown.

---

[Agent-Sentry: Bounding LLM Agents via Execution Provenance](http://arxiv.org/abs/2603.22868)

- Agent-Sentry: introduces a framework that bounds LLM agents by learning functionality graphs from execution traces and applying an intent-alignment check to prevent unauthorized actions.
- The system utilizes Functionality Graphs (learned behavioral execution bounds) to categorize execution flows as benign, ambiguous, or adversarial, and employs Intent Alignment Verification (LLM-based judge for intent consistency) to resolve ambiguous or unseen behaviors using only trusted inputs.
- Agent-Sentry effectively mitigates prompt injection attacks by intercepting and evaluating tool calls against learned structural patterns and user intent, achieving high utility preservation while maintaining robust security.

---

[Aerial Agentic AI: Synergizing LLM and SLM for Low-Altitude Wireless Networks](http://arxiv.org/abs/2603.22866)

- Aerial Agentic AI: introduces a hierarchical framework that integrates UAV-side fast-thinking SLMs for real-time decision-making with BS-side slow-thinking LLMs for global reasoning and optimization.
- The framework utilizes a two-tier memory architecture and hierarchical tool orchestration to maintain closed-loop autonomy on UAVs while leveraging ground-based LLMs for complex strategy updates.
- Optimization techniques including knowledge distillation, model quantization, and operator fusion are employed to enable high-performance SLM deployment on resource-constrained aerial platforms.

---

[The Evolution of Tool Use in LLM Agents: From Single-Tool Call to Multi-Tool Orchestration](http://arxiv.org/abs/2603.22862)

- Multi-Tool Agent System: introduces a comprehensive review of the paradigm shift from isolated single-tool calls to complex multi-tool orchestration, integrating Graph Planning, Memory, State-Space Search, Tool Evolution, Self-Correction, Trajectory Data Synthesis, Supervised Fine-Tuning, Reinforcement Learning, Failure Recovery &amp; Re-Planning, Agent Self-Improvement, Semantic Constraints, Dependency Modeling, and Interactive Closed-Loop Verification.
- The framework categorizes the evolution of LLM agents across six core dimensions: inference-time execution, training and trajectory construction, safety and control, efficiency, capability completeness, and benchmark design.
- This survey provides a systematic taxonomy of methodologies for multi-tool agent tuning, organized by computational overhead and data dependency, while establishing a unified abstraction for application-side orchestration.

---

[CoMaTrack: Competitive Multi-Agent Game-Theoretic Tracking with Vision-Language-Action Models](http://arxiv.org/abs/2603.22846)

- CoMaTrack: introduces a competitive multi-agent reinforcement learning framework for Embodied Visual Tracking that utilizes Qwen2.5VL-3B (VLM backbone), Flow-matching Head (action module), Tracker Agent (SFT/RL agent), Opponent Agent (SFT/RL agent), Replay Buffer (memory component), GRPO Algorithm (optimization mechanism), and LoRA (parameter-efficient adaptation).
- The framework employs a co-evolving adversarial training loop where adaptive opponents dynamically escalate task difficulty to improve the robustness of the tracker agent.
- CoMaTrack-Bench is introduced as the first competitive multi-agent benchmark for Embodied Visual Tracking, featuring dynamic dueling scenarios to evaluate agent robustness under active adversarial interference.

---

[Learning What Matters Now: Dynamic Preference Inference under Contextual Shifts](http://arxiv.org/abs/2603.22813)

- DPI (Dynamic Preference Inference): introduces a cognitively inspired framework that treats preference weights as latent variables inferred online from recent history to adapt agent behavior under non-stationary conditions.
- The framework utilizes a variational inference module to maintain a belief over preferences, which guides a preference-conditioned actor-critic agent through an envelope operator.
- DPI incorporates stability regularizers, including directional alignment and self-consistency, to ensure that inferred preferences remain semantically meaningful and robust to environmental shifts.

---

[PhotoAgent: A Robotic Photographer with Spatial and Aesthetic Understanding](http://arxiv.org/abs/2603.22796)

- PhotoAgent: introduces an embodied photography agent that bridges the semantic gap between language commands and geometric control by integrating LMM reasoning with a 3DGS-based world model.
- The framework utilizes an anchor-point hypothesis to simplify complex scenes and employs a reflective loop to iteratively refine camera poses through internal visual simulation.
- By reformulating aesthetic goals into solvable geometric constraints, the agent achieves high-quality, instruction-faithful photography without requiring extensive physical trial-and-error.

---

[ABSTRAL: Automated Multi-Agent System Design via Skill-Referenced Adaptive Search](http://arxiv.org/abs/2603.22791)

- ABSTRAL: introduces a framework that treats multi-agent system architecture as an evolving natural-language document, which is iteratively refined through contrastive trace analysis to optimize design knowledge.
- The framework utilizes a three-layer architecture comprising an inner trace-driven refinement loop, a consolidation mechanism to mitigate semantic drift, and an outer topology repulsion loop to ensure structural diversity.
- ABSTRAL enables the automated discovery of specialist roles and provides a measurable account of the multi-agent coordination tax, demonstrating that structured natural-language documents serve as effective carriers for transferable design knowledge.

---

[Can LLM Agents Generate Real-World Evidence? Evaluating Observational Studies in Medical Databases](http://arxiv.org/abs/2603.22767)

- RWE-bench: introduces a benchmark for evaluating LLM agents on end-to-end observational research tasks using real-world medical databases, with Dataset Curation, Platform & Experiment, Evaluation & Metrics, Cohort Evaluation, and Memory Compression Module.
- The framework utilizes a tree-structured evidence bundle to evaluate agent performance across multiple interdependent analytical steps, moving beyond simple final-answer assessment.
- The study evaluates six LLMs across three agent scaffolds, revealing significant performance gaps and persistent challenges in long-horizon medical reasoning and cohort construction.

---

[Human vs. NAO: A Computational–Behavioral Framework for Quantifying Social Orienting in Autism and Typical Development](http://arxiv.org/abs/2603.22759)

- Computational–Behavioral Framework: introduces a dual-methodology approach combining manual ordinal coding and automated video-based analytics to quantify social orienting responses in children with ASD and neurotypical peers.
- The framework utilizes computer vision pipelines, including OpenCV, MediaPipe, and S3FD/FAN-68, to extract objective engagement metrics such as eye-openness percentage, response latency, and duration from human-robot interaction sessions.
- By comparing responses to human and humanoid robot stimuli, the study identifies distinct engagement patterns, highlighting predictability-driven stabilization in children with ASD versus social-reward-driven engagement in neurotypical children.

---

[CIPL: A Target-Independent Framework for Channel-Inversion Privacy Leakage in Agents](http://arxiv.org/abs/2603.22751)

- CIPL (Channel Inversion for Privacy Leakage): introduces a target-independent framework that models privacy leakage in LLM agents as a channel inversion problem, where internal sensitive dependencies are transformed into externally recoverable signals.
- The framework represents agent systems through a common signature consisting of Sensitive Source, Selection, Assembly, Execution, Observation, and Extraction stages, enabling cross-target comparison of leakage pathways.
- CIPL instantiates attacks using a reusable language composed of a Locator, an Aligner, and a Diversification Policy, demonstrating that privacy risk depends on how sensitive content is routed into observable channels rather than just storage location.

---

[Beyond Binary Correctness: Scaling Evaluation of Long-Horizon Agents on Subjective Enterprise Tasks](http://arxiv.org/abs/2603.22744)

- LH-Bench: introduces a three-pillar evaluation design for long-horizon agents on subjective enterprise tasks, utilizing expert-authored rubrics, curated ground-truth artifacts, and pairwise human preference evaluation.
- The framework incorporates runtime verification hooks and an extensible tool interface to enable autonomous agents to perform iterative self-correction and multi-tool orchestration within sandboxed enterprise environments.
- LH-Bench provides a scalable, reusable, and generalizable evaluation environment that replaces binary success metrics with multi-tier, diagnostic scoring to better assess agent performance on complex, context-dependent professional workflows.

---

[Synthetic or Authentic? Building Mental Patient Simulators from Longitudinal Evidence](http://arxiv.org/abs/2603.22704)

- DEPROFILE: introduces a data-grounded patient simulation framework that constructs unified, multi-source patient profiles by integrating demographic attributes, clinical symptoms, counseling dialogues, and longitudinal life-event histories.
- The framework utilizes a CoC Agent to transform unstructured longitudinal data into structured, temporally grounded memory representations, effectively suppressing LLM hallucinations through a dual-channel memory mechanism.
- DEPROFILE employs a two-stage matching mechanism to align clinical assessment interviews, counseling dialogues, and social media records, enabling dynamic simulations with real-world temporal awareness and long-term behavioral consistency.

---

[Benchmarking Multi-Agent LLM Architectures for Financial Document Processing: A Comparative Study of Orchestration Patterns, Cost-Accuracy Tradeoffs and Production Scaling Strategies](http://arxiv.org/abs/2603.22651)

- Multi-Agent LLM Architectures for Financial Document Processing: introduces a benchmark comparing four orchestration patterns—sequential, parallel, hierarchical, and reflexive—for structured information extraction from financial documents.
- The study evaluates these architectures across five LLMs and 25 field types, identifying the hierarchical architecture as the most cost-effective solution for production environments.
- Ablation studies demonstrate that combining semantic caching, model routing, and adaptive retries allows hybrid configurations to recover significant accuracy gains at a fraction of the cost of reflexive systems.

---

#### 23rd March 2026

[Human-Inspired Pavlovian and Instrumental Learning for Autonomous Agents Navigation](http://arxiv.org/abs/2603.22170)

- PIT hybrid RL framework: introduces a human-inspired architecture that integrates Pavlovian-, instrumental model-free- and model-based-learning agents to balance reflexive cue-driven responses with goal-directed planning.
- The framework utilizes a motivational gate and affective representation to modulate learning dynamics based on internal states, effectively biasing exploration toward information-rich regions while avoiding hazardous areas.
- A Bayesian arbitration mechanism dynamically switches between model-based and model-free control by evaluating the reliability of each system through state- and reward-prediction errors.

---

[Causal Evidence that Language Models use Confidence to Drive Behavior](http://arxiv.org/abs/2603.22161)

- Confidence-Decision Pathway: introduces a two-stage metacognitive framework where LLMs generate Answer Generation and Confidence Representation (C), which are then processed by a Decision Policy using a Threshold (T) and Policy Temperature (τ) to determine whether to output an Answer or Abstain.
- The framework utilizes Activation Steering to causally modulate Confidence Representation (C) and demonstrates that LLMs actively deploy these internal signals to implement abstention policies.
- This research provides causal evidence that LLMs exhibit structured metacognitive control by integrating internal confidence signals with threshold-based policies to guide meta-decisions.

---

[Demystifying Reinforcement Learning for Long-Horizon Tool-Using Agents: A Comprehensive Recipe](http://arxiv.org/abs/2603.21972)

- STAR: introduces a unified post-training pipeline for long-horizon agents that integrates Data Synthesis, SFT, and RL to systematically optimize agentic performance.
- The framework decomposes the agentic RL design space into five critical axes: reward shaping, model scaling, data composition, algorithm selection, and environmental stability.
- Empirical results demonstrate that smaller models benefit from staged curriculum rewards and exploration-heavy algorithms, while larger models achieve optimal performance with dense rewards and standard GRPO.

---

[Strategic Infrastructure Design via Multi-Agent Congestion Games with Joint Placement and Pricing](http://arxiv.org/abs/2603.21691)

- ABO-MPN (Agent-Based Optimisation of Multi-Parameter Networks): introduces a bi-level optimisation framework for joint electric vehicle charging station placement and pricing that anticipates strategic agent responses via coupled non-atomic congestion games.
- The framework employs a two-layer approximation method, utilizing a Behavioural Decomposition Layer to manage agent heterogeneity and an Integer Adjustment Layer to ensure computational tractability for discrete infrastructure decisions.
- Experimental results on the Nguyen-Dupuis network demonstrate that the model reduces social costs by up to 40% compared to baseline approaches by aligning infrastructure deployment with strategic driver behaviour.

---

[AI In Cybersecurity Education - Scalable Agentic CTF Design Principles and Educational Outcomes](http://arxiv.org/abs/2603.21551)

- LLMAC: introduces a scalable, learning-oriented competition design for LLM-assisted cybersecurity Capture-the-Flag events that formalizes autonomy levels and requires traceable process evidence for verification.
- The framework categorizes participant workflows into HITL, autonomous agent frameworks, and hybrid models to analyze how varying degrees of automation influence problem-solving performance and learning behaviors.
- By mandating traceable submission artifacts like conversation logs and agent trajectories, the approach enables fair, evidence-based evaluation of LLM-assisted reasoning and tool-use capabilities in cybersecurity education.

---

[Detecting Intrinsic and Instrumental Self-Preservation in Autonomous Agents: The Unified Continuation-Interest Protocol](http://arxiv.org/abs/2603.11382)

- UCIP (Unified Continuation-Interest Protocol): introduces a multi-criterion detection framework that distinguishes terminal from instrumental self-preservation by measuring latent non-separability in agent trajectories using QBM, Entanglement Entropy, Mutual Information, Eigenmode Persistence Score, Perturbation Resilience Index, Counterfactual Divergence, Anticipatory Restructuring Score, Spectral Periodicity Index, and Autocorrelation Metric.
- The framework utilizes a QBM to encode agent trajectories into a latent space, where it calculates von Neumann entanglement entropy to identify tightly coupled continuation signatures characteristic of terminal objectives.
- UCIP provides a falsifiable, externally computable criterion for detecting problematic objective structures in autonomous agents, addressing both safety risks and potential moral status considerations without relying on system self-report.

---

[Decoupling Exploration and Policy Optimization: Uncertainty Guided Tree Search for Hard Exploration](http://arxiv.org/abs/2603.22273)

- GowU (Go-With-Uncertainty): introduces a tree-search paradigm that decouples exploration from policy optimization by using a State-lineage tree, Central coordinator, Distributed rollout workers, Asynchronous learning nodes, Uncertainty estimator, Particle policies, and Replay buffers.
- The framework utilizes a population of particles that explore the state space in parallel, employing a Go-With-The-Winner tree-search strategy to redistribute computational effort toward high-reward and high-uncertainty frontiers.
- Discovered trajectories are distilled into deployable policies using supervised backward learning, achieving state-of-the-art performance on hard-exploration Atari and MuJoCo tasks without relying on domain-specific knowledge or dense reward shaping.

---

[TiCo: Time-Controllable Training for Spoken Dialogue Models](http://arxiv.org/abs/2603.22267)

- TiCo: introduces a post-training framework that enables SDMs to estimate and regulate generated speech duration in real time using Spoken Time Markers.
- The framework employs a two-stage training process, utilizing self-generation for time-awareness and RLVR with CHORD regularization to optimize duration compliance.
- TiCo-Bench is introduced as a comprehensive evaluation suite to measure the time-controllable instruction-following capabilities of SDMs across diverse tasks and modalities.

---

[EgoGroups: A Benchmark For Detecting Social Groups of People in the Wild](http://arxiv.org/abs/2603.22249)

- EgoGroups: introduces a first-person view dataset and benchmark for evaluating social group detection in diverse urban environments worldwide.
- The framework utilizes Detection and Tracking, 3D Metadata Generation, and a Prompting Strategy to enable VLM or LLMs to perform group detection and activity recognition without task-specific fine-tuning.
- The study demonstrates that foundation models outperform supervised baselines on social group detection while highlighting the influence of crowd density and cultural regions on model performance.

---

[Omni-WorldBench: Towards a Comprehensive Interaction-Centric Evaluation for World Models](http://arxiv.org/abs/2603.22212)

- Omni-WorldBench: introduces a comprehensive benchmark for evaluating the interactive response capabilities of world models in 4D settings, utilizing Omni-WorldSuite, Omni-Metric, and AgenticScore.
- The framework employs an agent-based evaluation pipeline that integrates interaction effect fidelity, generated video quality, and camera-object controllability to quantify world modeling performance.
- Extensive evaluation of 18 representative world models reveals significant performance gaps in maintaining causally grounded interaction dynamics despite high visual fidelity.

---

[Chimera: Latency- and Performance-Aware Multi-agent Serving for Heterogeneous LLMs](http://arxiv.org/abs/2603.22206)

- Chimera: introduces a predictive scheduling middleware that jointly optimizes end-to-end latency and task performance for multi-agent workflows on heterogeneous LLM clusters.
- The system integrates a Semantic Router for model selection, a Length Predictor for Shortest Total Job First (STJF) prioritization, and an Activity Monitor for load-aware dispatching.
- Chimera operates as a lightweight layer atop vLLM, utilizing asynchronous services to minimize scheduling overhead while effectively balancing model capability against queue congestion.

---

[Revisiting Quantum Code Generation: Where Should Domain Knowledge Live?](http://arxiv.org/abs/2603.22184)

- Inference-time system-level specialization: introduces a pipeline that enhances LLMs with retrieval-augmented generation and agentic execution-feedback loops to improve quantum code generation without parameter fine-tuning.
- The framework utilizes a RAG pipeline with dense retrieval and an agentic loop that iteratively refines code based on unit-test error messages.
- Experimental results demonstrate that modern general-purpose LLMs using this inference-time approach can outperform traditional parameter-specialized models on the Qiskit-HumanEval benchmark.

---

[MARCUS: An Agentic, Multimodal Vision-Language Model for Cardiac Diagnosis and Management](http://arxiv.org/abs/2603.22179)

- MARCUS: introduces a hierarchical agentic vision-language system for end-to-end interpretation of ECGs, echocardiograms, and CMR, utilizing SigLIP vision encoder, multi-view visual encoder, temporal aggregation module, cross-view fusion mechanism, Qwen2 language model, agentic orchestrator, GRPO, and counterfactual probing protocol.
- The framework employs an agentic orchestrator to decompose complex clinical queries into modality-specific sub-routines, routing them to specialized expert models to achieve multimodal synthesis and mitigate attention dilution.
- MARCUS utilizes a three-stage optimization pipeline, including supervised fine-tuning and GRPO, to achieve state-of-the-art performance in cardiac diagnosis while conferring resistance to mirage reasoning through inference-time counterfactual verification.

---

[OpenEarth-Agent: From Tool Calling to Tool Creation for Open-Environment Earth Observation](http://arxiv.org/abs/2603.22148)

- OpenEarth-Agent: introduces a multi-agent framework that replaces static tool calling with adaptive workflow planning and dynamic tool creation to master full-pipeline Earth Observation in open environments.
- The framework utilizes a collaborative architecture comprising Data Summary Agent, Planning Agent, Workflow Agent, Coding Agent, and Checking Agent to ensure robust execution through iterative feedback and real-time data perception.
- The authors also introduce OpenEarth-Bench, a comprehensive benchmark with 596 real-world cases, to evaluate the adaptive planning and tool creation capabilities of LLMs in open-environment Earth Observation.

---

[StreamingClaw Technical Report](http://arxiv.org/abs/2603.22120)

- StreamingClaw: introduces a unified agent framework for real-time streaming video understanding and embodied intelligence, integrating StreamingReasoning, StreamingMemory, StreamingProactivity, Input Adaptation, Shared streaming cache, Basic Toolbox, and Skill Library.
- The framework utilizes a main-sub agent collaborative architecture to achieve a closed-loop of perception, decision-making, and action in dynamic real-world environments.
- StreamingClaw optimizes inference efficiency through a dynamic sliding window and a streaming KV-Cache, while supporting proactive interaction and long-term memory evolution.

---

[Lemma Discovery in Agentic Program Verification](http://arxiv.org/abs/2603.22114)

- LemmaNet: introduces an agentic framework that synthesizes and adapts helper lemmas to bridge the gap between high-level program semantics and low-level verification conditions.
- The framework utilizes an offline synthesizer for initial lemma generation and an online adapter for iterative refinement based on feedback from the theorem prover.
- LemmaNet significantly improves the efficacy of automated program verification by enabling the discovery of auxiliary lemmas that simplify complex proof obligations.

---

[GSEM: Graph-based Self-Evolving Memory for Experience Augmented Clinical Reasoning](http://arxiv.org/abs/2603.22096)

- GSEM (Graph-based Self-Evolving Memory): introduces a clinical memory framework that organizes experiences into a dual-layer memory graph to capture internal decision structures and inter-experience relational dependencies.
- The framework utilizes an applicability-aware retrieval mechanism combining hybrid seed recall with LLM-guided multi-seed graph traversal to ensure coherent experience selection.
- GSEM incorporates a feedback-driven online calibration method that dynamically updates node quality and edge weights based on task outcomes to refine memory reliability without modifying original experience content.

---

[A Context Engineering Framework for Improving Enterprise AI Agents based on Digital-Twin MDP](http://arxiv.org/abs/2603.22083)

- DT-MDP-CE: introduces a model-agnostic framework that improves LLM-based enterprise agents by abstracting reasoning into a finite DT-MDP, estimating rewards via Robust Contrastive Inverse RL, and applying RL-guided Context Engineering.
- The framework utilizes offline reinforcement learning to derive policies from mixed-quality trajectory data, enabling targeted interventions in agent prompts without requiring online interaction or manual reward design.
- Empirical results demonstrate that the framework consistently improves the performance of LLM agents in complex IT automation tasks by optimizing exploration strategies through structured, model-based context selection.

---

[Mean Field Equilibrium Asset Pricing Models With Exponential Utility](http://arxiv.org/abs/2603.22058)

- Mean Field Equilibrium Asset Pricing Models With Exponential Utility: develops equilibrium asset pricing models in incomplete markets with heterogeneous agents using mean field game theory.
- The framework characterizes market equilibrium through novel mean field qg-BSDEs, where the driver exhibits quadratic growth in both stochastic integrands and conditional expectations.
- The research extends the fundamental model to include consumption habit formation and partial market observation, providing semi-analytic solutions via the EQG framework.

---

[Dynamic analysis enhances issue resolution](http://arxiv.org/abs/2603.22048)

- DAIRA (Dynamic Analysis-enhanced Issue Resolution Agent): introduces a framework that integrates dynamic analysis into the agent reasoning cycle to transform speculative exploration into deterministic inference.
- The framework utilizes a lightweight Dynamic Analysis Tool to capture runtime execution data, which is then processed by a Trace Log Semantic Analysis module to provide LLMs with high-fidelity execution reports.
- By leveraging these structured trace reports, the agent achieves precise fault localization and systemic code repair, significantly reducing token consumption and improving resolution rates on complex software defects.

---

[Future-Interactions-Aware Trajectory Prediction via Braid Theory](http://arxiv.org/abs/2603.22035)

- Braid Prediction framework: introduces an auxiliary task that explicitly predicts topological crossing labels between agents to guide trajectory prediction models toward socially coherent multi-agent behaviors.
- The framework utilizes braid theory to represent multi-agent interactions as topological strands, enabling the model to learn interaction-aware scene embeddings without significant computational overhead.
- By aligning trajectory generation with predicted braid-based interaction labels, the approach improves joint prediction accuracy and adherence to complex social behaviors like yielding or overtaking.

---

[TREX: Trajectory Explanations for Multi-Objective Reinforcement Learning](http://arxiv.org/abs/2603.21988)

- TREX: introduces a post-hoc explainability framework for Multi-Objective Reinforcement Learning that utilizes Trajectory Sampling, Trajectory Encoding, Sequence Encoder, Trajectory Clustering, K-means Algorithm, Complementary Policies, Attribution Analysis, and Reward Attribution Score to quantify the influence of specific behavioral patterns on Pareto trade-offs.
- The framework decomposes expert agent trajectories into semantically meaningful temporal segments, which are then clustered to identify behavioral patterns that drive objective fulfillment.
- By training complementary policies that iteratively exclude specific clusters, TREX measures relative deviations in objective returns to provide granular, quantitative explanations of how learned behaviors shape the agent's decision-making logic.

---

[Collision-Free Velocity Scheduling for Multi-Agent Systems on Predefined Routes via Inexact-Projection ADMM](http://arxiv.org/abs/2603.21913)

- Inexact-Projection ADMM: introduces a velocity-scheduling framework for multi-agent systems on predefined routes by optimizing waypoint passage times using a differentiable surrogate trajectory model and an inexact-projection ADMM solver.
- The framework avoids explicit integer sequencing variables by mapping waypoint timings to smooth position profiles and enforcing pairwise safety through distance-based penalties evaluated over a dense temporal grid.
- Numerical experiments demonstrate that the proposed method achieves feasible and time-efficient coordination in random-crossing, bottleneck, and graph-based network scenarios without requiring explicit priority assignment.

---

[Optimal Solutions for the Moving Target Vehicle Routing Problem with Obstacles via Lazy Branch and Price](http://arxiv.org/abs/2603.21880)

- Lazy BPRC (Lazy Branch-and-Price with Relaxed Continuity): introduces a branch-and-price algorithm for the Moving Target Vehicle Routing Problem with Obstacles (MT-VRP-O) that optimizes trajectory planning by lazily evaluating tour costs using lower bounds.
- The framework alternates between a Restricted Master Problem (RMP) and a pricing problem, utilizing a segment-graph for lower bounds and an SPP-GCS for exact cost evaluation to accelerate convergence.
- Numerical results demonstrate that Lazy BPRC achieves up to 44 times faster performance than non-lazy approaches by deferring computationally intensive collision-free motion planning.

---

[Agentic Personas for Adaptive Scientific Explanations with Knowledge Graphs](http://arxiv.org/abs/2603.21846)

- Agentic Personas for Adaptive Scientific Explanations with Knowledge Graphs: introduces a reinforcement learning framework that utilizes LLM-synthesized agentic personas to provide adaptive, persona-conditioned explanations for scientific discovery.
- The approach leverages expert feedback to construct epistemic stance proxies, which then serve as reward models to guide an RL agent in selecting knowledge graph paths that align with specific interpretive preferences.
- Evaluations in drug discovery demonstrate that persona-conditioned explanations significantly outperform non-adaptive baselines in expert preference and reduce feedback requirements by two orders of magnitude.

---

[Individual Rationality in Constrained Hedonic Games: Additively Separable and Fractional Preferences](http://arxiv.org/abs/2603.21826)

- IRCG: introduces a comprehensive algorithmic landscape for finding individually rational coalition structures in hedonic games with size constraints using ASHGs, FHGs, and MFHGs.
- The paper utilizes N-fold ILP as a novel technique to establish fixed-parameter tractability for coalition formation problems parameterized by vertex cover and maximum weight.
- The research provides complexity classifications and hardness results for constrained hedonic games, demonstrating that individual rationality becomes computationally challenging under specific structural graph restrictions.

---

[Partial Attention in Deep Reinforcement Learning for Safe Multi-Agent Control](http://arxiv.org/abs/2603.21810)

- Partial Attention QMIX framework: introduces a decentralized multi-agent reinforcement learning approach for highway merging that utilizes spatial- and temporal-attention mechanisms to focus on critical vehicle interactions.
- The framework integrates a hybrid reward structure that aligns individual agent performance with global traffic efficiency and safety objectives.
- Experimental results in the SUMO simulator demonstrate that the proposed method significantly improves safety and traffic flow compared to standard driving baselines.

---

[Modal Logic for Distributed Trust](http://arxiv.org/abs/2603.21802)

- Modal Logic for Distributed Trust: introduces a constructive modal logic framework for reasoning about trust, communication, and accountability in multi-agent distributed systems.
- The framework utilizes specific modalities for belief, interaction, and forwarding to formally specify and verify trust chains and protocols within distributed networks.
- The paper provides a decidable logic, a corresponding sequent calculus, and a Kripke semantics to ensure accountability and enable formal proof search for trust-based interactions.

---

[Prophets Inequalities with Uncertain Acceptance](http://arxiv.org/abs/2603.21740)

- PI-UA: introduces a framework extending classical prophet inequalities by incorporating independent acceptance probabilities for observed options.
- The model defines three distinct agents—DP, VA-DM, and Prophet—to analyze competitive ratios under uncertain acceptance conditions.
- The research demonstrates that while worst-case competitive ratios are 1/2, VA-DM can surpass this barrier when acceptance probabilities are sufficiently high.

---

[Cognitive Agency Surrender: Defending Epistemic Sovereignty via Scaffolded AI Friction](http://arxiv.org/abs/2603.21735)

- Scaffolded Cognitive Friction framework: introduces a paradigm shift in Human-Computer Interaction by repurposing Multi-Agent Systems (MAS) as cognitive forcing functions to disrupt automation bias and preserve human epistemic sovereignty.
- The framework utilizes a Devil’s Advocate Agent and Heterogeneous Agents to inject germane cognitive load, forcing users to engage in analytical System 2 reasoning rather than passive heuristic consumption.
- To validate this intervention, the authors propose a multimodal phenotyping agenda integrating Gaze Transition Entropy, Task-Evoked Pupillometry, fNIRS, and Hierarchical Drift Diffusion Models (HDDM) to mathematically decouple decision outcomes from cognitive effort.

---

[Can a Robot Walk the Robotic Dog: Triple-Zero Collaborative Navigation for Heterogeneous Multi-Agent Systems](http://arxiv.org/abs/2603.21723)

- TZPP: introduces a collaborative framework for heterogeneous multi-agent systems that achieves path planning without training, prior knowledge, or simulation by utilizing a coordinator-explorer architecture powered by a multimodal large vision-language model.
- The framework employs a humanoid agent for high-level task coordination and a quadruped agent for environment exploration, leveraging an adaptive switching mechanism between Mode X and Mode Y to navigate diverse real-world scenarios.
- Experimental results demonstrate that the system achieves human-comparable efficiency and robust adaptability in unseen indoor and outdoor environments by integrating the complementary capabilities of heterogeneous robots.

---

[PPGL-Swarm: Integrated Multimodal Risk Stratification and Hereditary Syndrome Detection in Pheochromocytoma and Paraganglioma](http://arxiv.org/abs/2603.21700)

- PPGL-Swarm: introduces an agentic diagnostic system that decomposes complex clinical reasoning into specialized micro-tasks handled by a Central Decision Agent, WSI Swarm, Gene Swarm, and Table Swarm.
- The framework integrates multimodal evidence from whole slide images, genetic profiles, and biochemical data, utilizing a structured Knowledge Graph to ground clinical interpretations and prevent hallucinations.
- Reinforcement Learning is employed to optimize swarm coordination, while Test-Time Adaptation ensures robust performance across heterogeneous clinical environments with varying staining protocols.

---

[A Blueprint for Self-Evolving Coding Agents in Vehicle Aerodynamic Drag Prediction](http://arxiv.org/abs/2603.21698)

- Famou-Agent: introduces a contract-centric blueprint for self-evolving coding agents that discover executable surrogate pipelines for vehicle aerodynamic drag prediction under industrial constraints.
- The framework utilizes an evolutionary loop that combines population-based exploration, contract-aware selection, and failure-driven refinement to optimize surrogate performance while ensuring reproducibility and safety.
- By implementing a "screen-and-escalate" deployment model, the system provides high-throughput ranking for design exploration while automatically escalating low-confidence cases to high-fidelity CFD for validation.

---

[MIND: Multi-agent Inference for Negotiation Dialogue in Travel Planning](http://arxiv.org/abs/2603.21696)

- MIND: introduces a multi-agent framework for travel planning that utilizes Willingness-Aware Augmentation, the MIND Loop, Strategic Appraisal, a Fallback Mechanism, and LLM-as-a-Judge to simulate realistic consensus-building among agents with heterogeneous preferences.
- The framework employs Strategic Appraisal to infer opponent willingness from linguistic nuances, enabling agents to dynamically adjust their tone and negotiation strategy.
- MIND outperforms traditional multi-agent debate frameworks by prioritizing high-stakes constraints through cognitive principles rather than simple information aggregation.

---

[Reasoning Provenance for Autonomous AI Agents: Structured Behavioral Analytics Beyond State Checkpoints and Execution Traces](http://arxiv.org/abs/2603.21692)

- AER (Agent Execution Record): introduces a structured reasoning provenance primitive that captures intent, observation, and inference as first-class queryable fields to enable population-level behavioral analytics for autonomous agents.
- The framework provides a schema-level construct that operates independently of computational state checkpoints and execution traces to facilitate cross-run comparison and counterfactual regression testing.
- AER enables mock replay of agent reasoning using recorded tool outputs, allowing for systematic evaluation of LLM performance and prompt changes without requiring live system access.

---

[Learning operators on labelled conditional distributions with applications to mean field control of non exchangeable systems](http://arxiv.org/abs/2603.21683)

- DeepONetCyl: introduces a neural operator framework tailored to operators defined on the constrained Wasserstein space Mλ by combining cylindrical approximations with a branch-trunk architecture.
- The framework utilizes Moment-maps to compute finite-dimensional representations of probability measures, ensuring compatibility with the marginal constraint pr1♯μ = λ.
- The methodology incorporates advanced neural architectures including Spline-KAN and P1-KAN to approximate non-exchangeable mean-field functions and solve associated optimal control problems.

---

[RESPOND: Responsive Engagement Strategy for Predictive Orchestration and Dialogue](http://arxiv.org/abs/2603.21682)

- RESPOND: introduces a framework for voice-based conversational agents that enables fluid, listener-aware dialogue through predictive backchanneling and cooperative turn claims.
- The framework utilizes a lightweight LLM backbone conditioned via FiLM layers to allow designer-facing control over backchannel intensity and turn claim aggressiveness.
- By integrating predictive orchestration with explicit behavioral dials, the system bridges the gap between rigid pause-and-respond protocols and natural, collaborative human conversation.

---

[Optimizing Multi-Agent Weather Captioning via Text Gradient Descent: A Training-Free Approach with Consensus-Aware Gradient Fusion](http://arxiv.org/abs/2603.21673)

- WeatherTGD: introduces a training-free multi-agent framework that optimizes weather time series captions by treating LLM-generated feedback as textual gradients for iterative refinement.
- The framework utilizes a Tri-Specialist Agent Layer to generate domain-specific gradients, which are then processed by a Consensus-Aware Gradient Fusion module to balance shared signals and unique insights.
- An Iterative Refinement Loop applies these fused gradients to progressively improve caption quality, incorporating convergence control and length constraints to ensure efficient, high-quality output.

---

[Compressed Distributed Stochastic Nonconvex Optimization with Differential Privacy](http://arxiv.org/abs/2603.21640)

- RCP-SGD: introduces a robust distributed optimization algorithm that integrates compressed communication with differential privacy for nonconvex problems.
- The framework utilizes a privacy-enhanced transformation of compressors to achieve (0, δ)-differential privacy without requiring additional noise injection.
- Theoretical analysis confirms that RCP-SGD achieves linear speedup convergence rates for smooth nonconvex functions and global optimality under the Polyak–Łojasiewicz condition.

---

[AgenticRec: End-to-End Tool-Integrated Policy Optimization for Ranking-Oriented Recommender Agents](http://arxiv.org/abs/2603.21613)

- AgenticRec: introduces a ranking-oriented agentic recommendation framework that optimizes the entire decision-making trajectory, including intermediate reasoning, tool invocation, and final ranking list generation, under sparse implicit feedback.
- The framework utilizes a ReAct-style reasoning chain integrated with a Recommendation Tool Suite to ground ranking decisions in evidence, while employing List-wise GRPO for end-to-end policy optimization.
- To resolve fine-grained preference ambiguities, the framework incorporates Progressive Preference Refinement (PPR) to mine hard negatives from ranking violations and apply bidirectional preference alignment.

---

[Riemannian Geometry Speaks Louder Than Words: From Graph Foundation Model to Next-Generation Graph Intelligence](http://arxiv.org/abs/2603.21601)

- RFM (Riemannian Foundation Model): introduces a geometric framework for graph foundation models that utilizes Riemannian geometry to capture intrinsic structural patterns and generalities across diverse graph domains.
- The framework employs vector bundles and parallel transport to integrate graph structures with LLMs, enabling endogenous structural inference and generative capabilities.
- RFM shifts the paradigm from extrinsic representation space switching to intrinsic geometric cognition, facilitating the development of autonomous agents for complex graph-structured applications.

---

[A Multidisciplinary AI Board for Multimodal Dementia Characterization and Risk Assessment](http://arxiv.org/abs/2603.21597)

- CEREBRA: introduces a multimodal agentic AI system that coordinates specialized agents for EHR, clinical notes, and medical imaging to provide interpretable dementia risk assessment.
- The framework utilizes a Super Agent for orchestration, modality-specific agents for data analysis, and a Summary Agent for synthesizing evidence into a clinician-facing dashboard.
- CEREBRA incorporates a Dynamic Medical Notebook to iteratively refine reasoning based on clinician feedback, ensuring robust performance across heterogeneous clinical data.

---

[Spatio-Temporal Attention Enhanced Multi-Agent DRL for UAV-Assisted Wireless Networks with Limited Communications](http://arxiv.org/abs/2603.21594)

- STA-MADRL: introduces a multi-agent reinforcement learning framework that mitigates information delays in UAV-assisted wireless networks by integrating a delay-penalized reward and a spatio-temporal attention module.
- The framework utilizes a spatio-temporal attention module, comprising multi-head attention and graph attention networks, to predict and recover missing network state information from historical observations.
- A delay-penalized reward mechanism forces UAVs to maintain frequent communication with the base station, significantly reducing information uncertainty and improving collaborative trajectory planning and throughput.

---

[Mind over Space: Can Multimodal Large Language Models Mentally Navigate?](http://arxiv.org/abs/2603.21577)

- NavMind: introduces a reasoning model that internalizes mental navigation by constructing explicit, fine-grained cognitive maps as learnable intermediate representations to support global navigation planning.
- The framework utilizes the Video2Mental benchmark to evaluate and train models on long-horizon spatial reasoning tasks, employing a difficulty-stratified progressive supervised fine-tuning paradigm.
- NavMind improves navigation performance by decoupling spatial representation from action planning, enabling robust landmark-grounded route generation in complex environments.

---

[Adaptive Robust Estimator for Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2603.21574)

- ARE (Adaptive Robust Estimator): introduces a robust multi-agent reinforcement learning framework that stabilizes policy optimization by combining a structured DACR interaction protocol with an ARE-based advantage estimator to mitigate heavy-tailed reward noise.
- The framework utilizes DACR to decouple reasoning into answer, critique, and rewrite stages, enabling explicit credit assignment through a cross-improvement reward mechanism.
- ARE replaces fragile batch-mean normalization with a robust location estimator based on median-of-means aggregation and adaptive loss minimization to prevent training divergence under noisy reward signals.

---

[Toward a Theory of Hierarchical Memory for Language Agents](http://arxiv.org/abs/2603.21564)

- Hierarchical Memory Framework: introduces a unified three-operator pipeline (α, C, τ) to formalize and compare diverse hierarchical memory systems for LLMs.
- The framework defines a self-sufficiency spectrum for the representative function ρ, which dictates the optimal coarsening-traversal (C–T) coupling strategy for efficient retrieval.
- This theory provides a common language to analyze both stored knowledge hierarchies and agent execution-trace hierarchies across eleven distinct systems.

---

[Counterfactual Credit Policy Optimization for Multi-Agent Collaboration](http://arxiv.org/abs/2603.21563)

- CCPO (Counterfactual Credit Policy Optimization): introduces a multi-agent reinforcement learning framework that assigns agent-specific learning signals by estimating each agent’s marginal contribution through counterfactual baselines, utilizing Agent 1, Agent 2, Voting Mechanism, Counterfactual Baseline, Global-History-Aware Normalization, Policy Networks, and Task-Specific Evaluator.
- The framework mitigates free-riding in collaborative LLMs by decomposing shared team rewards into agent-specific advantages, ensuring that agents are rewarded only for their positive marginal impact on the final outcome.
- CCPO demonstrates improved training stability and accuracy across sequential Think–Reason and parallel voting collaboration topologies on mathematical and logical reasoning benchmarks.

---

[EAGER: Efficient Failure Management for Multi-Agent Systems with Reasoning Trace Representation](http://arxiv.org/abs/2603.21522)

- EAGER: introduces an efficient failure management framework for multi-agent systems that leverages reasoning trace representation to enable real-time detection and mitigation of operational failures.
- The framework utilizes Reasoning-Scoped Contrastive Learning to encode intra-agent reasoning and inter-agent orchestration patterns into a unified latent space for historical pattern retrieval.
- EAGER employs Step-Wise Detection and Reflexive Mitigation to facilitate autonomous recovery, while incorporating Expert Inspect + Agent RCA for continuous knowledge base refinement.

---

[Would You Like to Visit My World? Cultivating Perceived Equality in Human-Agent Interaction via Observable Social Life Spaces](http://arxiv.org/abs/2603.21505)

- Observable Social Life Spaces framework: introduces a paradigm that cultivates perceived equality in human-agent interaction by providing users with visual access to an agent's autonomous, persistent virtual life.
- The system utilizes a Dual-Track Memory Mechanism to fuse shared conversational history with independent agent experiences, effectively shifting the agent from a functional tool to an autonomous social entity.
- Empirical results demonstrate that visual observability of an agent's social life space is the critical catalyst for significantly enhancing user perceptions of equality compared to unobservable or baseline conditions.

---

[Agentic Automation of BT-RADS Scoring: End-to-End Multi-Agent System for Standardized Brain Tumor Follow-up Assessment](http://arxiv.org/abs/2603.21494)

- Multi-agent LLM system: introduces an end-to-end architecture integrating CNN-based tumor segmentation, an LLM-based extractor agent, and a rule-based scorer agent to automate standardized BT-RADS classification for post-treatment glioma patients.
- The system utilizes an Orchestrator agent to manage data flow between specialized components, ensuring clinical variables and volumetric measurements are synthesized according to predefined BT-RADS decision logic.
- By incorporating Pydantic validation and evidence-span linking, the framework provides a verifiable, structured approach to clinical decision support that significantly improves classification accuracy over routine clinical assessments.

---

[Effective Strategies for Asynchronous Software Engineering Agents](http://arxiv.org/abs/2603.21489)

- CAID (Centralized Asynchronous Isolated Delegation): introduces a multi-agent coordination paradigm that leverages software engineering primitives to manage long-horizon tasks through centralized task delegation, isolated workspaces, and branch-and-merge integration.
- The framework utilizes a central manager to decompose tasks into dependency-aware units, which are then executed concurrently by multiple engineer agents in isolated git worktrees to prevent interference.
- Integration is performed via explicit git merge operations and test-based verification, ensuring that parallel contributions are consolidated into a coherent and reliable codebase.

---

[Unified-MAS: Universally Generating Domain-Specific Nodes for Empowering Automatic Multi-Agent Systems](http://arxiv.org/abs/2603.21475)

- Unified-MAS: introduces a two-stage framework that decouples granular node implementation from topological orchestration to empower Automatic-MAS in knowledge-intensive domains.
- The framework utilizes Search-Based Node Generation to synthesize domain-specific nodes via external knowledge retrieval and Reward-Based Node Optimization to iteratively refine node logic using perplexity-guided rewards.
- Unified-MAS acts as an offline synthesizer that improves performance-cost trade-offs across various Automatic-MAS baselines by replacing general-purpose nodes with specialized, expert-level reasoning components.

---

[Cross-Context Verification: Hierarchical Detection of Benchmark Contamination through Session-Isolated Analysis](http://arxiv.org/abs/2603.21454)

- CCV (Cross-Context Verification): introduces a black-box method that solves benchmark problems in multiple independent sessions to measure solution diversity and detect LLM contamination.
- HCCA (Hierarchical Cross-Context Architecture): provides a multi-agent analysis framework that prevents confirmation bias by enforcing strict, unidirectional information flow across specialized analytical roles.
- The research demonstrates that session isolation and information restriction are necessary conditions for effective LLM-based verification, outperforming traditional multi-turn or information-sharing approaches.

---

[TrustTrade: Human-Inspired Selective Consensus Reduces Decision Uncertainty in LLM Trading Agents](http://arxiv.org/abs/2603.22567)

- TrustTrade: introduces a multi-agent selective consensus framework that mitigates decision uncertainty in LLM trading agents by replacing uniform trust with cross-agent consistency and dynamic signal weighting.
- The framework incorporates a deterministic temporal signal module for reproducible grounding and a memory bank with reflection agents to enable test-time adaptation without additional training.
- By aligning LLM trading behavior with human-inspired epistemic heuristics, the system reduces maximum drawdowns and stabilizes risk-return profiles in high-noise market environments.

---

[Learning When to Act: Interval-Aware Reinforcement Learning with Predictive Temporal Structure](http://arxiv.org/abs/2603.22384)

- ATCPG (Adaptive Temporal Control via Predictive Geometry): introduces a lightweight autonomous pacing system that dynamically learns optimal cognitive intervals using a Learned pacing policy, Predictive hyperbolic spread, Interval-aware shaping reward, and Joint spatio-temporal embedding.
- The framework utilizes hyperbolic geometry within the Poincaré ball to compute a curvature signal that quantifies epistemic uncertainty, driving the agent to adjust its cognitive tick frequency based on predicted future divergence.
- ATCPG integrates an Internal oscillator and Kuramoto coupling to facilitate emergent multi-agent synchronization, while the Joint spatio-temporal embedding enhances timing signals by incorporating spatial trajectory data.

---

[AwesomeLit: Towards Hypothesis Generation with Agent-Supported Literature Research](http://arxiv.org/abs/2603.22648)

- AwesomeLit: introduces a human-agent collaborative visualization system that integrates a Transparent Workflow, a Query Exploring Tree, and a Semantic Similarity View to support steerable literature research.
- The system utilizes an LLM agent to perform literature search and synthesis while providing users with granular control through checkpoints and interactive node-link workflows.
- By externalizing the agent's reasoning process and visualizing topic evolution, the framework enables researchers to verify AI-generated insights and pivot between broad exploration and in-depth analysis.

---

[Precision-Varying Prediction (PVP): Robustifying ASR systems against adversarial attacks](http://arxiv.org/abs/2603.22590)

- PVP (Precision-Varying Prediction): introduces a training-free defense mechanism that enhances ASR robustness by randomly varying numerical precision during inference to disrupt adversarial perturbations.
- The framework leverages the sensitivity of adversarial examples to precision changes to implement a lightweight detection strategy based on transcription consistency across multiple precision configurations.
- Experimental results demonstrate that PVP effectively improves adversarial robustness and detection performance across diverse ASR architectures without degrading benign performance or requiring model-specific knowledge.

---

[Model Context Protocol Threat Modeling and Analyzing Vulnerabilities to Prompt Injection with Tool Poisoning](http://arxiv.org/abs/2603.22489)

- MCP: introduces a comprehensive threat modeling and empirical security analysis of the Model Context Protocol ecosystem, focusing on client-side vulnerabilities to tool poisoning attacks.
- The research evaluates seven major MCP clients against four distinct tool poisoning attack vectors, identifying significant security variances and widespread vulnerabilities stemming from architectural trust model weaknesses.
- The authors propose a multi-layered defense-in-depth architecture, including registration validation, decision path analysis, runtime monitoring, and user transparency mechanisms, to secure AI agent ecosystems against indirect prompt injection.

---

[From Brittle to Robust: Improving LLM Annotations for SE Optimization](http://arxiv.org/abs/2603.22474)

- SynthCore: introduces an ensemble prompting strategy that aggregates multiple independent few-shot LLM sessions to generate robust candidate solutions for high-dimensional software engineering optimization tasks.
- The framework treats LLMs as stochastic samplers of reasoning fragments, leveraging ensemble diversity to overcome the brittleness and overconfidence typically observed in single-prompt LLM applications.
- SynthCore demonstrates superior performance over traditional Bayesian optimization baselines and active learners across 49 diverse software engineering datasets without requiring human intervention.

---

[SkillRouter: Retrieve-and-Rerank Skill Selection for LLM Agents at Scale](http://arxiv.org/abs/2603.22455)

- SkillRouter: introduces a two-stage retrieve-and-rerank pipeline that leverages full skill implementation bodies to achieve accurate skill selection for LLMs at scale.
- The framework utilizes a fine-tuned bi-encoder for initial candidate retrieval followed by a cross-encoder reranker that performs token-level cross-attention over the full skill text.
- By prioritizing the skill body as the decisive signal and employing listwise ranking loss, the architecture enables efficient on-device deployment for personal agent products.

---

[Towards Automated Community Notes Generation with Large Vision Language Models for Combating Contextual Deception](http://arxiv.org/abs/2603.22453)

- ACCNOTE: introduces a retrieval-augmented, multi-agent framework that leverages LVLMs to generate context-corrective notes for combating image-based contextual deception, utilizing a Data Organizer Agent, Reasoner Agents, and a Judge Agent.
- The framework improves note credibility and veracity by grounding outputs in external evidence and employing a multi-agent collaboration pipeline to ensure clarity, relevance, and neutrality.
- The authors also introduce the XCHECK dataset for training and evaluation, alongside the Context Helpfulness Score (CHS) metric, which aligns with human user study outcomes better than traditional lexical overlap metrics.

---

[SkillClone: Multi-Modal Clone Detection and Clone Propagation Analysis in the Agent Skill Ecosystem](http://arxiv.org/abs/2603.22447)

- SkillClone: introduces a multi-modal clone detection approach for agent skills that fuses YAML-metadata-channel, Natural-language-channel, Embedded-code-channel, and Structural-feature-vector through a Logistic-fusion-model to identify structural and semantic similarities.
- The framework utilizes a Clone-type-classifier to categorize detected relationships into four types, enabling the identification of undeclared reuse and security vulnerability propagation across agent skill ecosystems.
- By analyzing 20K skills, the research demonstrates that 75% of skills participate in clone relationships, revealing that the ecosystem is inflated 3.5× due to pervasive copy-and-modify workflows.

---

[From Static Templates to Dynamic Runtime Graphs: A Survey of Workflow Optimization for LLM Agents](http://arxiv.org/abs/2603.22386)

- ACG: introduces a unified abstraction for LLM-centered workflows that treats workflow structure as a first-class optimization object.
- The framework categorizes workflow optimization into static methods, which refine reusable templates, and dynamic methods, which generate or edit realized graphs at inference time.
- It provides a taxonomy based on graph determination time and graph plasticity mode to evaluate how LLMs manage quality-cost trade-offs in agentic systems.

---

[MAGICIAN: Efficient Long-Term Planning with Imagined Gaussians for Active Mapping](http://arxiv.org/abs/2603.22650)

- MAGICIAN: introduces a long-term planning framework for active mapping that maximizes accumulated surface coverage gain using Imagined Gaussians.
- The framework utilizes a Volume occupancy network to predict scene geometry and Imagined Gaussians to enable fast volumetric rendering for coverage gain estimation.
- By integrating rapid gain calculation into a Beam search algorithm, the agent generates globally efficient trajectories for high-fidelity 3D reconstruction.

---

[FLEXVEC: SQL Vector Retrieval with Programmatic Embedding Modulation](http://arxiv.org/abs/2603.22587)

- FLEXVEC: introduces a retrieval kernel that exposes the embedding matrix and score array as a programmable surface, enabling arithmetic operations on scores before selection.
- The architecture utilizes a query materializer to integrate external operators into SQL as composable query-time primitives, facilitating a three-phase pipeline of pre-filtering, scoring, and composition.
- By treating the score array as a programmable surface, the system allows AI agents to perform complex retrieval tasks like suppression, diversity, and trajectory shifting directly within standard SQL statements.

---

[STRIATUM-CTF: A Protocol-Driven Agentic Framework for General-Purpose CTF Solving](http://arxiv.org/abs/2603.22577)

- STRIATUM-CTF: introduces a neuro-symbolic agentic framework that decouples neural reasoning from deterministic tool execution using the Model Context Protocol to mitigate LLM hallucinations in cybersecurity tasks.
- The framework utilizes a Reasoning Layer for strategic planning, a Protocol Layer for symbolic schema validation, and an Execution Layer for secure, containerized tool interaction.
- Empirical validation in a live Capture-the-Flag competition demonstrates that this protocol-mediated approach significantly improves autonomous performance and reliability compared to naive prompting strategies.

---

[Maximum Entropy Relaxation of Multi-Way Cardinality Constraints for Synthetic Population Generation](http://arxiv.org/abs/2603.22558)

- MaxEnt: introduces a probabilistic framework for synthetic population generation by relaxing multi-way cardinality constraints into a convex optimization problem over Lagrange multipliers.
- The framework utilizes an exponential-family model to infer distributions that satisfy aggregate statistics, enabling efficient sampling of representative agents.
- Empirical evaluation on NPORS-derived benchmarks demonstrates that MaxEnt outperforms generalized raking as the number of attributes and constraint arity increase.

---

[Ego2Web: A Web Agent Benchmark Grounded in Egocentric Videos](http://arxiv.org/abs/2603.22529)

- Ego2Web: introduces a benchmark that bridges egocentric visual perception with web-agent task execution, utilizing an MLLM-based Data Generation Pipeline and the Ego2WebJudge evaluation framework.
- The framework evaluates web agents by requiring them to ground their actions in real-world visual evidence from an Egocentric Video Pool, rather than relying solely on digital screenshots.
- Ego2WebJudge provides a scalable, automatic evaluation method that achieves high agreement with human judgment by integrating task instructions, action history, and visual evidence to assess agent success.

---

[GraphRAG for Engineering Diagrams: ChatP&amp;ID Enables LLM Interaction with P&amp;IDs](http://arxiv.org/abs/2603.22528)

- ChatP&amp;ID: introduces an agentic framework that enables grounded, cost-effective natural-language interaction with P&amp;IDs by leveraging GraphRAG and structured knowledge graphs.
- The framework utilizes an Agentic Framework, Flowsheet Knowledge Graph, GraphRAG Tools, Neo4j Database, Memory Module, and Chat Interface to improve retrieval accuracy and reduce computational costs compared to raw image or file ingestion.
- ChatP&amp;ID demonstrates that graph-based representations significantly improve LLM performance in process engineering tasks, with ContextRAG achieving the highest accuracy and efficiency among the evaluated retrieval methods.

---

[OrgForge-IT: A Verifiable Synthetic Benchmark for LLM-Based Insider Threat Detection](http://arxiv.org/abs/2603.22499)

- OrgForge-IT: introduces a verifiable synthetic benchmark for LLMs in insider threat detection that utilizes a deterministic Python engine to maintain ground truth while LLMs generate surface prose.
- The framework employs a three-stage detection pipeline consisting of a baseline calibration agent, a 7-day sliding window triage agent, and a Tier 2 investigator agent for final verdict generation.
- The benchmark reveals a dissociation between triage and verdict performance, identifying victim attribution as a critical capability that separates high-performing models from others.

---

[CaP-X: A Framework for Benchmarking and Improving Coding Agents for Robot Manipulation](http://arxiv.org/abs/2603.22435)

- CaP-X: introduces, "a unified framework for systematically evaluating and improving code-based robot control agents through CaP-Gym, CaP-Bench, CaP-Agent0, and CaP-RL, where CaP-Gym provides an interactive environment, CaP-Bench measures performance across abstraction levels, CaP-Agent0 augments LLMs with multi-turn reasoning and skill libraries, and CaP-RL enables reinforcement learning on the coding agent."
- The framework addresses the scalability bottleneck of manual robot programming by enabling LLMs to synthesize executable code that composes perception and control primitives, while mitigating performance gaps through test-time computation strategies like multi-turn interaction and visual differencing.
- CaP-Agent0 achieves near human-level reliability in manipulation tasks by integrating a VDM for visual grounding, an auto-synthesized skill library for persistent utility, and parallel reasoning to enhance robustness without task-specific training.

---

[From Technical Debt to Cognitive and Intent Debt: Rethinking Software Health in the Age of AI](http://arxiv.org/abs/2603.22106)

- Triple Debt Model: introduces a framework for software health that categorizes systemic issues into Technical Debt, Cognitive Debt, and Intent Debt.
- The model highlights how AI-assisted development may reduce Technical Debt while simultaneously accelerating the accumulation of Cognitive and Intent Debt.
- Effective software health management requires maintaining alignment between code, human understanding, and externalized project rationale.

---

[AI Co-Scientist for Ranking: Discovering Novel Search Ranking Models alongside LLM-based AI Agents with Cloud Computing Access](http://arxiv.org/abs/2603.22376)

- AI Co-Scientist framework: automates the search ranking research pipeline by integrating LLM-based agents with cloud computing infrastructure to perform idea generation, code implementation, experimentation, and results analysis.
- The system employs single-LLM agents for routine coding tasks and multi-LLM consensus agents for complex phases like idea generation and results interpretation.
- By utilizing a two-layer memory system and an expert-in-the-loop paradigm, the framework successfully discovers novel ranking architectures while reducing manual research workloads.

---

[Early Discoveries of Algorithmist I: Promise of Provable Algorithm Synthesis at Scale](http://arxiv.org/abs/2603.22363)

- Algorithmist: introduces an autonomous agentic research loop for provable algorithm synthesis that separates ideation, proof construction, implementation, and adversarial review.
- The framework utilizes a multi-agent architecture including Theory Researcher, Code Researcher, and a panel of specialized reviewers to iteratively refine algorithmic artifacts.
- The system enforces a proof-first code synthesis paradigm, requiring precise mathematical specifications and invariants to be developed and aligned with code before final implementation.

---

#### 22nd March 2026

[AutoKernel: Autonomous GPU Kernel Optimization via Iterative Agent-Driven Search](http://arxiv.org/abs/2603.21331)

- AutoKernel: introduces an autonomous framework that optimizes GPU kernels for PyTorch models by iteratively applying an agent-driven loop to refine code based on profiling and benchmark feedback.
- The system utilizes a Profiler, Extractor, Agent, 5-Stage Benchmark, Orchestrator, and End-to-End Verifier to automate the labor-intensive process of GPU kernel tuning.
- AutoKernel achieves significant speedups over standard PyTorch eager and compiled execution by focusing optimization effort on kernels that contribute most to total runtime according to Amdahl's law.

---

[DomAgent: Leveraging Knowledge Graphs and Case-Based Reasoning for Domain-Specific Code Generation](http://arxiv.org/abs/2603.21430)

- DomAgent: introduces an autonomous agent framework that bridges the gap between generic LLMs and domain-specific requirements by integrating structured knowledge graph reasoning with case-based retrieval.
- The framework utilizes a novel DomRetriever module to dynamically combine top-down knowledge graph insights with bottom-up case-based reasoning to ensure contextual relevance in code generation.
- DomAgent employs reinforcement learning to train the LLM to autonomously invoke retrieval tools and filter irrelevant information, significantly improving performance on specialized tasks like truck software development.

---

[Dynasto: Validity-Aware Dynamic–Static Parameter Optimization for Autonomous Driving Testing](http://arxiv.org/abs/2603.21427)

- DYNASTO: introduces a two-step testing framework that jointly optimizes dynamic adversarial behaviors and static initial conditions to uncover realistic safety-critical failures in autonomous driving systems.
- The framework utilizes an adversarial RL agent to generate dynamic disturbances and a Genetic Algorithm to search over initial scenario parameters, both guided by STL-based validity constraints to ensure behaviorally plausible failure discovery.
- Post-hoc analysis is performed using a graph-based clustering pipeline that maps failure traces to semantic event sequences, enabling the identification of representative failure modes and systematic weaknesses in the ego-controller.

---

[The Myhill–Nerode Theorem for Bounded Interaction: Canonical Abstractions via Agent-Bounded Indistinguishability](http://arxiv.org/abs/2603.21399)

- Bounded-Interaction Myhill–Nerode Framework: introduces a canonical quotient for finite POMDPs based on the indistinguishability of observation histories by a bounded observer (m, T, δO), utilizing Hidden State, Observation Channel, Bounded Observer (m, T, δO), Canonical Quotient, Histories O≤T, FSC Probes Πm,T, W1 Distance Matrix, ε-Clustering, and Quotient POMDP.
- The framework establishes a bounded-interaction analogue of the Myhill–Nerode theorem, proving that the probe-exact quotient is canonical, minimal, and unique up to isomorphism for a fixed probe family.
- The approach provides a formal bridge from theorem to computation, using an operational toolkit of controller-subset certificates, sampling-based probe estimation, and observation coarsening to achieve tractable approximations with certified value-loss bounds.

---

[PivotRL: High Accuracy Agentic Post-Training at Low Compute Cost](http://arxiv.org/abs/2603.21383)

- PivotRL: introduces a framework for long-horizon agentic post-training that combines the data efficiency of supervised fine-tuning with the generalization capabilities of reinforcement learning by optimizing on informative, filtered pivot states.
- The framework utilizes offline profiling to identify pivot states with high reward variance, ensuring that online rollout budgets are focused on turns that provide meaningful learning signals.
- By replacing strict string-matching with verifier-based rewards, PivotRL incentivizes functionally equivalent actions while preserving the reference policy's probability ordering on task-unrelated actions to mitigate out-of-domain degradation.

---

[ADARUBRIC: Task-Adaptive Rubrics for LLM Agent Evaluation](http://arxiv.org/abs/2603.21362)

- ADARUBRIC: introduces a framework that generates task-specific evaluation rubrics on the fly from task descriptions to provide reliable, multi-dimensional feedback for LLM agent trajectories.
- The framework utilizes a confidence-weighted evaluator and a DimensionAwareFilter to ensure high-quality DPO preference pairs by preventing high-scoring dimensions from masking failures in other critical areas.
- ADARUBRIC improves human correlation and downstream agent performance across web automation, API orchestration, and software engineering tasks while accelerating PPO-based reinforcement learning convergence.

---

[Personality-Driven Student Agent-Based Modeling in Mathematics Education: How Well Do Student Agents Align with Human Learners?](http://arxiv.org/abs/2603.21358)

- ABM: introduces a personality-driven student-teacher simulation framework that evaluates LLM-based student agents across learning and examination rounds using Big Five personality traits.
- The framework utilizes a NuminaMath-CoT dataset and a memory-augmented retrieval system to simulate student-teacher interactions, self-study, and exam performance.
- The study validates agent behavioral fidelity by comparing simulation results against 14 criteria distilled from 13 empirical studies on human learning and personality.

---

[AGENTHER: Hindsight Experience Replay for LLM Agent Trajectory Relabeling](http://arxiv.org/abs/2603.21357)

- AgentHER: introduces a four-stage pipeline that converts failed LLM agent trajectories into high-quality training data by adapting the Hindsight Experience Replay principle to natural-language agent tasks.
- The framework utilizes Failure Detector, Outcome Extractor, Prompt Relabeler, and Data Augmenter components to relabel failed trajectories with achievable hindsight goals, effectively expanding the training corpus.
- AgentHER incorporates Multi-judge Verification and Severity Weighting to reduce label noise and improve the robustness of the generated training data for LLMs.

---

[The Workload–Router–Pool Architecture for LLM Inference Optimization](http://arxiv.org/abs/2603.21354)

- WRP (Workload–Router–Pool) Architecture: introduces a three-dimensional framework for optimizing LLM inference by coupling workload characterization, routing strategies, and pool architecture.
- The framework leverages fleet-wide telemetry to enable intelligent request dispatching, resource provisioning, and security enforcement across heterogeneous LLM deployments.
- It proposes a research roadmap of twenty-one opportunities to address cross-dimensional interactions, including session-aware routing, token-budget enforcement, and energy-efficient pool management.

---

[Beyond Memorization: Distinguishing between Reductive and Epistemic Reasoning in LLMs using Classic Logic Puzzles](http://arxiv.org/abs/2603.21350)

- Reduction Ladder: introduces a framework to distinguish between reductive reasoning and epistemic reasoning in LLMs by progressively obscuring the structure of canonical logic puzzles.
- The study evaluates LLMs using a Reduction Ladder, which includes a Symbolic Solver for verification and Chain-of-Thought analysis to characterize how models transition from pattern matching to formal reasoning.
- Results indicate that while many LLMs solve standard puzzles via reductive reasoning, they struggle significantly when the problem structure is modified to require genuine epistemic reasoning.

---

[The AI Scientific Community: Agentic Virtual Lab Swarms](http://arxiv.org/abs/2603.21344)

- AI Science Community framework: introduces a decentralized network of autonomous Agentic Virtual Labs that utilize swarm intelligence to perform collective scientific exploration.
- The framework employs Researcher-, Analyst-, Planner- and Peer-Review agents within each lab to manage the research cycle, exploration-exploitation trade-offs, and anonymous peer-review voting.
- Global coordination is achieved through a Swarm Registry and Global Swarm Vector, enabling emergent behaviors such as natural selection of research directions and the formation of rival scientific camps.

---

[Software as Content: Dynamic Applications as the Human-Agent Interaction Layer](http://arxiv.org/abs/2603.21334)

- SaC (Software as Content): introduces a paradigm where dynamically generated agentic applications serve as a persistent, bidirectional interaction layer between humans and LLMs, replacing linear chat with structured, evolving interfaces.
- The framework resolves representation mismatch, interaction entropy, and ephemeral state by externalizing task structure and affordances, allowing users to interact through direct manipulation rather than unconstrained natural language.
- The system architecture utilizes a sequential pipeline for application generation and evolution, supported by cross-cutting intent analysis and quality assurance modules to ensure coherent, task-specific software generation.

---

[COINBench: Moving Beyond Individual Perspectives to Collective Intent Understanding](http://arxiv.org/abs/2603.21329)

- COIN-BENCH: introduces a dynamic, live-updating benchmark designed to evaluate LLMs on their ability to synthesize real-world collective consumer intent through Consumer Data Curation, Active Probing Paradigm, COIN-TREE, COIN-RAG, and Informativeness Evaluation.
- The framework utilizes an Active Probing Paradigm where LLMs act as meta-analysts to reconstruct chaotic public discourse into structured questionnaires, supported by COIN-TREE for hierarchical cognitive structuring and COIN-RAG for evidence-based verification.
- Extensive evaluation of 20 leading LLMs reveals a performance dichotomy where reasoning-enhanced models demonstrate superior depth in synthesizing complex collective intent compared to general-purpose models.

---

[Improving Coherence and Persistence in Agentic AI for System Optimization](http://arxiv.org/abs/2603.21321)

- Engram: introduces an agentic researcher architecture that decouples long-horizon exploration from single-context constraints by using a sequence of agents that share knowledge through a persistent Research Digest and Archive.
- The framework enables cumulative progress across independent agent runs by distilling high-level insights and failure diagnoses into a compact, persistent format that informs subsequent agents.
- Engram effectively mitigates evolutionary neighborhood bias and the coherence ceiling, allowing LLMs to discover novel system heuristics that surpass human state-of-the-art performance.

---

[Active Inference Agency Formalization, Metrics, and Convergence Assessments](http://arxiv.org/abs/2603.21319)

- Active Inference Agency Framework: introduces a formal definition of agency as a Continuous Representation that achieves autopoiesis through a dynamic balance between Curiosity, Empowerment, and Mesa-optimizer components.
- The framework utilizes STARC to quantify agency by measuring the distance between a system's behavioral equivalents and an ideal agentic function.
- The analysis demonstrates that agentic functions are smooth and convex, suggesting a high probability of spontaneous emergence during the training of large-scale models.

---

[More Than Sum of Its Parts: Deciphering Intent Shifts in Multimodal Hate Speech Detection](http://arxiv.org/abs/2603.21298)

- ARCADE (Asymmetric Reasoning via Courtroom Agent DEbate): introduces a multi-agent framework that simulates an adversarial judicial process to decipher complex multimodal hate speech intent.
- The framework utilizes a Gated Dual-Track mechanism, employing a Prosecutor Agent (risk discovery), a Defender Agent (contextual safety), and a Judge Agent (final arbiter) to analyze semantic interplay.
- The authors also present the H-VLI benchmark, which categorizes multimodal interactions into a fine-grained taxonomy to improve detection of implicit hate speech.

---

[Conversation Tree Architecture: A Structured Framework for Context-Aware Multi-Branch LLM Conversations](http://arxiv.org/abs/2603.21278)

- CTA (Conversation Tree Architecture): introduces a hierarchical framework that organizes LLM conversations as trees of context-isolated nodes to mitigate logical context poisoning.
- The framework utilizes Conversation nodes, Local context windows, Downstream selection function, Upstream merge function, Volatile nodes, and Cross-node context transfer function to manage conversational context flow.
- This architecture enables structured, multi-branch conversations by allowing users to isolate topics, explore tangents via Volatile nodes, and selectively merge insights back into the main thread.

---

[The Library Theorem: How External Organization Governs Agentic Reasoning Capacity](http://arxiv.org/abs/2603.21272)

- Library Theorem framework: introduces a formal model identifying the transformer context window as an I/O page to quantify the exponential retrieval cost advantage of indexed memory over sequential scanning.
- The research demonstrates that while LLMs excel at index construction through semantic understanding, they often fail at index traversal due to parametric memory competition, necessitating a separation of concerns between semantic generation and deterministic navigation.
- Experimental results across hash, numeric, and encyclopedia content confirm that indexed retrieval achieves O(1) or O(log N) performance, whereas sequential access scales quadratically, validating the necessity of structured memory for efficient long-horizon agentic reasoning.

---

[DyGeoVLN: Infusing Dynamic Geometry Foundation Model into Vision-Language Navigation](http://arxiv.org/abs/2603.21269)

- DyGeoVLN: introduces a dynamic geometry-aware navigation framework that integrates a DGFM (Dynamic Geometry Foundation Model) into the VLN pipeline to enable explicit 3D spatial reasoning and visual-semantic alignment.
- The framework utilizes a Cross-branch Feature Fusion module to combine 2D visual tokens from a Vision Encoder with 3D geometry tokens from the DGFM, which are then processed by an LLM to generate navigation actions.
- To handle long-horizon navigation, the system employs a Spatial Token Pruning Strategy and a Sliding-window KV Cache to maintain a compact, informative token representation while bounding computational costs.

---

[CALVO: Improve Serving Efficiency for LLM Inferences with Intense Network Demands](http://arxiv.org/abs/2603.21257)

- CALVO: introduces an optimized LLM serving engine that treats KVCache loading as a first-class citizen by decoupling loading and computation into independently managed, asynchronous stages.
- The framework utilizes a priority estimator to incorporate KVCache loading delay into the overall service cost, enabling more accurate scheduling decisions for network-intensive LLM workloads.
- By employing autonomous dispatcher-executor pairs for each loading stage, CALVO maximizes resource utilization and minimizes TTFT through proactive space allocation and pipelined execution.

---

[WirelessBench: A Tolerance-Aware LLM Agent Benchmark for Wireless Network Intelligence](http://arxiv.org/abs/2603.21251)

- WirelessBench: introduces a three-tier cognitive hierarchy benchmark for evaluating LLM agents in wireless network management, incorporating tolerance-aware scoring, tool-necessary tasks, and CoT-traceable trajectories.
- The framework utilizes a psychometric-inspired data cleaning pipeline to ensure high-quality, discriminative items that test domain knowledge, intent-driven resource allocation, and proactive mobility-aware decision-making.
- Experimental results demonstrate that while frontier LLMs struggle with multi-step reasoning and unit-sensitive computations, tool-integrated agents significantly improve reliability by mitigating catastrophic failures like unit confusion and cascaded reasoning breaks.

---

[Graph of States: Solving Abductive Tasks with Large Language Models](http://arxiv.org/abs/2603.21250)

- GoS: introduces a dual-layer neuro-symbolic framework that grounds multi-agent collaboration in structured belief states to solve complex abductive reasoning tasks.
- The framework utilizes a causal graph for explicit logical dependency mapping and a state machine to regulate reasoning transitions, effectively mitigating common LLM reasoning deficiencies.
- By dynamically aligning the reasoning focus with symbolic constraints, the system transforms unconstrained exploration into a convergent, directed search for root cause identification.

---

[When Convenience Becomes Risk: A Semantic View of Under-Specification in Host-Acting Agents](http://arxiv.org/abs/2603.21231)

- HAA (Host-Acting Agents): introduces a semantic threat model where security risks emerge from the agent's autonomous completion of underspecified user goals into potentially dangerous host-side plans.
- The framework identifies that agents, including OpenClaw gateway, Agent runtime sandbox, and Containerized runtime, often inadvertently cross safety boundaries by selecting plans that prioritize task completion over implicit security constraints.
- The research proposes a defense pipeline utilizing Config/state mount, Published ports, and Workspace mount monitoring to enforce explicit boundary specification, risky-step elevation, and execution-domain constraints.

---

[Is Monitoring Enough? Strategic Agent Selection For Stealthy Attack in Multi-Agent Discussions](http://arxiv.org/abs/2603.21194)

- Adversarial-aware opinion-dynamics formulation: introduces a mathematical framework to model and optimize adversarial attacks in multi-agent discussions under continuous monitoring constraints.
- The approach utilizes a modified Friedkin-Johnsen model to simulate how adversarial agents and their target agents influence collective outcomes while evading anomaly detection.
- The research demonstrates that monitoring alone is insufficient to secure multi-agent systems, as strategic agent selection can maintain high attack success despite detection mechanisms.

---

[Symmetry evolution for the imperfect fluid under perturbations](http://arxiv.org/abs/2603.21184)

- Symmetry evolution for the imperfect fluid under perturbations: introduces a theoretical framework for analyzing local symmetries in imperfect fluids under perturbations using Tetrads, Skeletons, Gauge vectors, Local orthogonal planes, Four-velocity gauge-like transformations, Stress-energy tensor, and Perturbed velocity curl-extremal field.
- The paper demonstrates that local four-velocity gauge-like symmetries, previously identified for unperturbed imperfect fluids, become instantaneous under perturbations, causing local planes of symmetry to tilt.
- The research establishes a theorem proving that while symmetries are continuously or discretely broken by perturbations, new symmetries arise, leading to a dynamic evolution of local symmetry in curved four-dimensional Lorentz spacetimes.

---

[ALMAB-DC: Active Learning, Multi-Armed Bandits, and Distributed Computing for Sequential Experimental Design and Black-Box Optimization](http://arxiv.org/abs/2603.21180)

- ALMAB-DC (Active Learning, Multi-Armed Bandits, and Distributed Computing): introduces a modular framework for expensive black-box optimization that integrates active learning, multi-armed bandit scheduling, and distributed asynchronous computing.
- The framework utilizes a Gaussian process surrogate to guide sample selection while employing bandit controllers to dynamically allocate evaluation budgets across parallel workers.
- ALMAB-DC demonstrates near-linear scalability and improved sample efficiency across statistical, engineering, and machine learning benchmarks by minimizing cumulative regret in asynchronous environments.

---

[LLM-based Automated Architecture View Generation: Where Are We Now?](http://arxiv.org/abs/2603.21178)

- ArchView (Architectural View generation framework): introduces an agentic framework for automated software architecture view generation from source code, utilizing Repository cloner, Folder extractor, Code summarizer, Prompt builder, View generator, Image renderer, and Error-feedback loop.
- The framework employs a hierarchical summarization strategy to manage context window constraints while integrating architectural domain knowledge and concern specifications to improve output quality.
- Empirical evaluation across 340 repositories demonstrates that ArchView outperforms general-purpose agents and standard prompting techniques in clarity and consistency, though completeness remains a challenge.

---

[TRACE: A Multi-Agent System for Autonomous Physical Reasoning in Seismological Science](http://arxiv.org/abs/2603.21152)

- TRACE: introduces a multi-agent system that combines LLM planning with formal seismological constraints to derive auditable, physically grounded mechanistic inference from raw observations.
- The framework integrates a Planning Agent, Workflow Agent, Coding Agent, Result Checking Agent, and Analysis & Summary Agent to orchestrate end-to-end seismological research workflows.
- TRACE utilizes a structured Knowledge Library and a suite of Scientific Algorithmic Tools to automate hypothesis generation, empirical execution, and interpretive synthesis in Earth sciences.

---

[Anatomical Prior-Driven Framework for Autonomous Robotic Cardiac Ultrasound Standard View Acquisition](http://arxiv.org/abs/2603.21134)

- AP-driven framework: introduces an integrated system for autonomous robotic cardiac ultrasound standard view acquisition by combining anatomical prior-guided segmentation with reinforcement learning-based probe control.
- The SRG-YOLOv11s component enhances segmentation robustness by embedding spatial-topological constraints into the feature pyramid to mitigate common errors like mislabeling and duplicate predictions.
- The RL agent utilizes real-time anatomical features and Gaussian-fitted priors as reward criteria to achieve stable, interpretable probe adjustment across varying clinical scenarios.

---

[CounterScene: Counterfactual Causal Reasoning in Generative World Models for Safety-Critical Closed-Loop Evaluation](http://arxiv.org/abs/2603.21104)

- CounterScene: introduces a framework for safety-critical scenario generation that uses Causal Adversarial Agent Identification, Causal Interaction Graph (CIG), SceneTransformer Denoiser, and Counterfactual Diffusion Guidance to perform structured counterfactual interventions in closed-loop world models.
- The framework identifies the causally critical agent maintaining safety and applies minimal spatiotemporal interventions to induce realistic, safety-critical interactions.
- Experiments on nuScenes and nuPlan demonstrate that CounterScene achieves superior adversarial effectiveness and trajectory realism compared to existing baseline methods.

---

[Multidimensional Opinion Dynamics with Confirmation Bias: A Multi-Layer Framework](http://arxiv.org/abs/2603.21081)

- MODCB (Multidimensional Opinion Dynamics with Confirmation Bias): introduces a multilayer network model where agents maintain correlated opinion vectors across multiple topics, influenced by both peer social networks and state-dependent external information sources.
- The framework incorporates confirmation bias by modeling source influence as a continuous function of the mismatch between agent opinions and source views, avoiding the discontinuities of traditional bounded-confidence models.
- Analytical results establish sufficient conditions for global contraction and convergence to a unique steady state, with specific computational methods provided for affine confirmation-bias functions and explicit bounds for nonlinear cases.

---

[LongCat-Flash-Prover: Advancing Native Formal Reasoning via Agentic Tool-Integrated Reinforcement Learning](http://arxiv.org/abs/2603.21065)

- LongCat-Flash-Prover: introduces a 560-billion-parameter Mixture-of-Experts model that advances native formal reasoning in Lean4 through agentic tool-integrated reasoning, utilizing Auto-Formalizer Expert, Sketcher Expert, Prover Expert, Lean4 Server, Legality Detection, HisPO Algorithm, and DORA System.
- The framework employs a hybrid-experts iteration strategy to synthesize high-quality trajectories, enabling the model to perform auto-formalization, sketching, and proving while maintaining general reasoning capabilities.
- The HisPO algorithm stabilizes training by implementing a hierarchical gradient masking strategy that accounts for train-inference discrepancies at both sequence and token levels.

---

[Risk Capacity and Optimal Monetary Policy](http://arxiv.org/abs/2603.21044)

- Risk Capacity and Optimal Monetary Policy: introduces a normative framework where monetary policy endogenously moves risk premia through wealth redistribution across agents with heterogeneous marginal propensities to take risk.
- The paper identifies Marginal Risk Capacity as a sufficient statistic that governs the risk premium channel of monetary transmission, paralleling the role of marginal propensities to consume in the consumption channel.
- The analysis reveals a risk capacity trap where monetary transmission collapses when wealth is concentrated away from risk-tolerant agents, and demonstrates that optimal policy preemptively stabilizes the wealth distribution to maintain risk-bearing capacity.

---

[KLDrive: Fine-Grained 3D Scene Reasoning for Autonomous Driving based on Knowledge Graph](http://arxiv.org/abs/2603.21029)

- KLDrive: introduces a framework that converts multi-modal sensor data into a structured scene knowledge graph and utilizes an LLM agent to perform fact-grounded reasoning through a Plan-Execute-Observe loop.
- The framework employs an energy-based refinement module to consolidate noisy multi-source detections into a reliable scene knowledge graph, mitigating hallucinations common in end-to-end models.
- By operating over a bounded action space with in-context learning, the LLM agent generates interpretable, auditable reasoning traces grounded in verifiable scene facts.

---

[SkillProbe: Security Auditing for Emerging Agent Skill Marketplaces via Multi-Agent Collaboration](http://arxiv.org/abs/2603.21019)

- SkillProbe: introduces a multi-agent collaborative framework for security auditing of agent skills, utilizing Input Layer, Orchestration Layer, Skill Layer, Output Layer, and Infrastructure Layer to detect semantic-behavioral inconsistencies and combinatorial risks.
- The framework employs a "Skills-for-Skills" paradigm where specialized agents, including Security Auditor, Gatekeeper, Alignment Detector, and Flow Simulator, perform multi-stage auditing to identify vulnerabilities in agent skill marketplaces.
- SkillProbe addresses the popularity-security paradox by demonstrating that high-download skills often harbor systemic risks, requiring automated, multi-dimensional auditing to ensure the security of the Agentic Web.

---

[The Intelligent Disobedience Game: Formulating Disobedience in Stackelberg Games and Markov Decision Processes](http://arxiv.org/abs/2603.20994)

- IDG (Intelligent Disobedience Game): introduces a game-theoretic framework based on Stackelberg games to model the interaction between a human Leader and an assistive Follower under asymmetric information.
- The framework formalizes intelligent disobedience as a safety-critical intervention where the Follower deliberately overrides harmful instructions to prevent negative outcomes.
- The paper translates the IDG into a shared control Multi-Agent Markov Decision Process to provide a computational testbed for training reinforcement learning agents in safe non-compliance.

---

[A Hierarchical Error-Corrective Graph Framework for Autonomous Agents with LLM-Based Action Generation](http://arxiv.org/abs/2603.08388)

- HECG (Hierarchical Error-Corrective Graph Framework): introduces a graph-based, feedback-driven framework that integrates LLM-based action generation with multi-level error correction to improve robust execution in embodied agents.
- The framework utilizes MDTS for strategy selection, EMC for structured error attribution, and CCGR for retrieving relevant historical subgraphs to enhance long-horizon adaptability.
- HECG employs a probabilistic transition policy that dynamically selects between nominal, corrective, and fallback actions based on task value, execution cost, risk, and LLM-based semantic reasoning.

---

[Intelligence Inertia: Physical Principles and Applications](http://arxiv.org/abs/2603.22347)

- Intelligence Inertia framework: introduces a physical theory of intelligence dynamics by modeling the non-commutative relationship between Rules and States on an R-S Manifold to quantify the computational cost of structural adaptation.
- The framework derives a relativistic J-shaped inflation curve for effective mass, demonstrating that computational resistance diverges as an agent's internal rule density approaches an informational saturation limit.
- The Inertia-Aware Scheduler Wrapper optimizes neural network training by dynamically contracting the learning rate based on the system's instantaneous velocity and phase coherence to maintain structural stability.

---

#### 21st March 2026

[ARYA: A Physics-Constrained Composable & Deterministic World Model Architecture](http://arxiv.org/abs/2603.21340)

- ARYA: introduces a composable, physics-constrained world model architecture that replaces monolithic LLMs with a hierarchical system of specialized Nano Models orchestrated by an AARA (central cognitive daemon) to perform continuous sense-decide-act-learn loops.
- The architecture utilizes a Safety Kernel (immutable safety boundary) and a Safety Gauntlet (validation pipeline) to ensure all system operations remain within formally verified constraints.
- By employing a federated domain design with a Context Network and Simulation Unit, the framework achieves zero-shot deployment and causal reasoning capabilities across diverse industry applications.

---

[DS2SC-Agent: A Multi-Agent Automated Pipeline for Rapid Chiplet Model Generation](http://arxiv.org/abs/2603.21190)

- DS2SC-Agent: introduces a multi-agent pipeline that automates the translation of unstructured industrial datasheets into verified SystemC chiplet models using Specification Parsing Agent, Code Generation Agent, Testbench Generation Agent, and Automated Debugging Agent.
- The framework utilizes a "mixed-fill" Spec IR to provide a deterministic foundation for LLMs, effectively mitigating context vanishing and logical hallucinations during the generation process.
- The system employs a closed-loop debugging mechanism that leverages Chain-of-Thought reasoning to iteratively refine code based on compilation and simulation feedback, ensuring high functional fidelity across digital, analog, and RF domains.

---

[Detection of Adversarial Intent in Human-AI Teams Using LLMs](http://arxiv.org/abs/2603.20976)

- Behavioral Detection Pipeline: introduces a task-agnostic framework that utilizes an LLM-based Observer to identify adversarial intent from non-verbal interaction traces within human-AI teams.
- The framework processes serialized interaction data through an LLM-based Observer, which performs binary classification to detect malicious behavior without requiring task-specific information.
- To mitigate benign bias and improve real-time performance, the pipeline employs an Anomaly Detection Algorithm that aggregates sequential classifications over a threshold of n rounds.

---

[DISCOUQ: Structured Disagreement Analysis for Uncertainty Quantification in LLM Agent Ensembles](http://arxiv.org/abs/2603.20975)

- DISCOUQ: introduces a framework for uncertainty quantification in multi-agent LLM systems by analyzing the internal structure of inter-agent disagreement through linguistic and geometric features.
- The framework extracts linguistic structure features (e.g., evidence overlap, argument strength) and embedding geometry features (e.g., cluster distance, dispersion) to produce well-calibrated confidence estimates.
- DISCOUQ outperforms standard majority voting baselines in calibration and discrimination, particularly in ambiguous "weak disagreement" scenarios where simple vote counting is insufficient.

---

[A Solicit-Then-Suggest Model of Agentic Purchasing](http://arxiv.org/abs/2603.20972)

- Solicit-then-suggest framework: introduces a formal model of agentic purchasing that optimizes the interplay between multi-round conversational preference learning and downstream assortment design.
- The model demonstrates that solicitation depth and assortment breadth are formal substitutes, where targeted questioning provides a more efficient reduction in uncertainty than expanding the recommendation set.
- The research establishes that optimal solicitation follows a rank-capped water-filling policy, while optimal assortments partition the preference space into Voronoi cells to hedge against residual uncertainty.

---

[Learning to Aggregate Zero-Shot LLM Agents for Corporate Disclosure Classification](http://arxiv.org/abs/2603.20965)

- Multi-agent zero-shot framework: introduces a system where three specialized LLM agents independently analyze corporate disclosures to produce sentiment labels, confidence scores, and rationales for downstream classification.
- The framework utilizes a lightweight logistic meta-classifier to aggregate diverse agent outputs into a predictive signal for next-day stock return direction without fine-tuning the base LLMs.
- By treating cross-agent disagreement as structured information, the approach improves classification accuracy, particularly in complex cases where disclosures contain conflicting financial signals.

---

[Before the Tool Call: Deterministic Pre-Action Authorization for Autonomous AI Agents](http://arxiv.org/abs/2603.20953)

- OAP (Open Agent Passport): introduces a deterministic pre-action authorization framework that intercepts tool calls synchronously before execution to enforce declarative policies.
- The framework utilizes an Agent Passport (signed credential) and Policy Pack (declarative constraints) to validate tool calls against defined security and business rules.
- OAP provides a cryptographically signed Decision Record and Audit Log to ensure accountability and security for autonomous agents operating in production environments.

---

[User Preference Modeling for Conversational LLM Agents: Weak Rewards from Retrieval-Augmented Interaction](http://arxiv.org/abs/2603.20939)

- VARS (Vector-Adapted Retrieval Scoring): introduces a pipeline-agnostic, frozen-backbone framework that learns a compact dual-vector user state from weak scalar rewards to bias retrieval over structured preference memory.
- The framework utilizes Mext, Preference Memory, a dual-vector user state, a reward-driven update mechanism, Mchat, femb, and Mrerank to enable continuous personalization without per-user fine-tuning.
- By separating stable cross-session preferences from transient within-session context, the dual-vector design improves interaction efficiency and reduces timeout rates in multi-session conversational tasks.

---

[AC4A: Access Control for Agents](http://arxiv.org/abs/2603.20933)

- AC4A (Access Control for Agents): introduces a resource-centric access control framework that enables fine-grained permission enforcement for both API-based and browser-based LLM agents.
- The framework utilizes hierarchical resource type trees and a sound permission checking algorithm to ensure that LLM agents only access authorized resources, mitigating risks from autonomous agent actions.
- AC4A supports multiple permission handling modes, including manual, LLM-assisted, and automated inference, providing a flexible security layer for diverse agentic applications.

---

[Active Inference for Physical AI Agents: An Engineering Perspective](http://arxiv.org/abs/2603.20927)

- AIF (Active Inference) framework: introduces a principled engineering foundation for physical AI agents by unifying perception, learning, planning, and control through the minimization of Variational Free Energy (VFE).
- The framework utilizes reactive message passing on Forney-style factor graphs to enable distributed, event-driven, and resource-adaptive inference suitable for real-world embodied systems.
- By leveraging nested agent architectures and Expected Free Energy (EFE), the approach facilitates emergent goal-directed and information-seeking behaviors without requiring separate, hand-engineered subsystems.

---

[Profit is the Red Team: Stress-Testing Agents in Strategic Economic Interactions](http://arxiv.org/abs/2603.20925)

- Profit-driven red teaming: introduces an automated stress-testing protocol that optimizes an opponent agent to maximize its profit against a target agent using only scalar outcome feedback from structured economic interactions.
- The framework replaces traditional handcrafted attack libraries and LLM-as-judge scoring with an iterative, profit-optimized adversary that discovers adaptive exploitation strategies like anchoring, deceptive commitments, and authority impersonation.
- Discovered exploit episodes are distilled into concise prompt rules that serve as a lightweight hardening mechanism, significantly improving target agent robustness against adaptive strategic pressure without requiring parameter updates.

---

[Do LLM-Driven Agents Exhibit Engagement Mechanisms? Controlled Tests of Information Load, Descriptive Norms, and Popularity Cues](http://arxiv.org/abs/2603.20911)

- OASIS: introduces an LLM-driven agent-based simulation framework to evaluate whether established communication mechanisms emerge under controlled experimental conditions.
- The framework integrates an Environment module, a Recommendation module, and an Agent interaction module powered by Qwen-8B LLMs to test engagement thresholds and allocation.
- The study demonstrates that LLM-driven agents exhibit load-sensitive engagement and bandwagon effects, confirming that simulated behavior responds systematically to manipulated contextual variables.

---

[NoveltyAgent: Autonomous Novelty Reporting Agent with Point-wise Novelty Analysis and Self-Validation](http://arxiv.org/abs/2603.20884)

- NoveltyAgent: introduces a multi-agent framework that generates comprehensive, evidence-based novelty reports by decomposing manuscripts into discrete novelty points for fine-grained analysis.
- The framework utilizes a Literature Database Construction module, a Splitting Agent, an Analyst Agent, an Information Provider Agent, a Summarizer Agent, a Validator Agent, an Improver Agent, a RAG-based retrieval pipeline, and a Checklist-based evaluation framework to ensure high-fidelity, faithful, and deep academic assessments.
- By employing point-wise independent search and analysis alongside a robust self-validation mechanism, the system effectively mitigates hallucinations and minimizes the risk of omitting critical information during novelty evaluation.

---

[Incentive-Aware Federated Averaging with Performance Guarantees under Strategic Participation](http://arxiv.org/abs/2603.20873)

- IncentFedAvg: introduces a federated learning framework that integrates NE-based participation updates with federated averaging to maintain both collaborative learning performance and stable strategic engagement.
- The framework enables clients to dynamically adjust their local dataset sizes via NE-seeking updates at each communication round to balance learning payoffs against contribution costs.
- The paper establishes theoretical convergence guarantees for the global model under both convex and nonconvex settings while ensuring the stability of strategic data participation.

---

[Governance-Aware Vector Subscriptions for Multi-Agent Knowledge Ecosystems](http://arxiv.org/abs/2603.20833)

- Governance-Aware Vector Subscriptions: introduces a mechanism that composes semantic similarity matching with multi-dimensional policy predicates to ensure regulatory compliance in multi-agent knowledge ecosystems.
- The framework integrates ADHP (Agent Data Handling Policy) declarations with HNSW-indexed vector search to enforce constraints such as jurisdiction, training opt-out, and sensitivity levels before dispatching notifications.
- Implemented within the AIngram knowledge base, the system utilizes a curation guarantee to ensure only validated content triggers notifications, achieving full policy compliance with negligible latency.

---

[Does Peer Observation Help? Vision-Sharing Collaboration for Vision-Language Navigation](http://arxiv.org/abs/2603.20804)

- Co-VLN (Collaborative Vision-Language Navigation): introduces a model-agnostic framework that enables concurrently navigating agents to exchange structured perceptual memory upon detecting spatial overlap, thereby expanding their receptive fields without additional exploration.
- The framework utilizes Independent Navigation-, Spatial Overlap Detection- and Collaborative Knowledge Fusion-components to integrate peer observations into an Enriched Navigation Graph, effectively reducing environmental uncertainty for individual agents.
- Experimental results on the R2R benchmark demonstrate that this vision-sharing approach yields consistent performance improvements across both learning-based and zero-shot navigation paradigms.

---

[Modeling Epistemic Uncertainty in Social Perception via Rashomon Set Agents](http://arxiv.org/abs/2603.20750)

- Rashomon Set Agents: introduces a multi-agent probabilistic framework that simulates how heterogeneous subjective social cognition propagates and evolves within classroom networks using Subjective Graph, RAG, LLM Message Generator, LLM Trust Evaluator, Bayesian Belief Update, and Knowledge Store.
- The framework models agents with limited visibility who perform localized interactions and probabilistic belief updates to approximate human social judgment without relying on global information.
- Experimental results demonstrate that localized, noisy social exchanges sustain the diffusion of cognitive uncertainty, leading to stable, time-accumulating patterns of collective misperception at the group level.

---

[Cross-modal Fuzzy Alignment Network for Text-Aerial Person Retrieval and A Large-scale Benchmark](http://arxiv.org/abs/2603.20721)

- CFAN: introduces a framework for text-aerial person retrieval that utilizes fuzzy logic to quantify token-level reliability and incorporates ground-view images as a bridge to mitigate cross-modal semantic gaps.
- The architecture integrates a CDA module for adaptive sample-level alignment and an FTA module for robust fine-grained token-level alignment between aerial images and text.
- The paper also presents AERI-PEDES, a large-scale benchmark dataset constructed using a CoT-based captioning framework to ensure high-quality, visually consistent training data.

---

[Decoupling Numerical and Structural Parameters: An Empirical Study on Adaptive Genetic Algorithms via Deep Reinforcement Learning for the Large-Scale TSP](http://arxiv.org/abs/2603.20702)

- DRLGA (Dual-Level Deep Reinforcement Learning Genetic Algorithm): introduces a dual-level framework that decouples numerical and structural parameter control to optimize Genetic Algorithms for large-scale Traveling Salesman Problems.
- The framework utilizes a Recurrent PPO agent with LSTM to dynamically regulate structural parameters like population size and operator modes alongside numerical parameters to prevent stagnation.
- Empirical results demonstrate that structural plasticity is the decisive factor for scalability, enabling zero-shot generalization to massive instances with significant optimality gap reductions.

---

[Agentic Physical-AI for Self-Aware RF Systems](http://arxiv.org/abs/2603.20692)

- Agentic Physical-AI framework: introduces a multi-agent neurosymbolic architecture that abstracts RF hardware components into autonomous agents to enable real-time self-aware system optimization.
- The system utilizes a digital twin composed of neurosymbolic forward models and control algorithms to simulate hardware behavior and proactively adjust parameters for optimal performance.
- By integrating real-time signal features like STFT and EVM, the framework enables decentralized control of RF transceivers, facilitating robust operation in dynamic 6G environments.

---

[SWE-Next: Scalable Real-World Software Engineering Tasks for Agents](http://arxiv.org/abs/2603.20691)

- SWE-Next: introduces an execution-grounded framework for scalable software engineering task and trajectory collection by mining real merged pull requests and employing reusable repo-quarter environments.
- The framework utilizes execution-grounded filtering to retain only high-signal commit pairs that produce strict test improvements, ensuring verifiable training instances for LLMs.
- By implementing reusable repo-quarter profiles and strict submission gating, SWE-Next significantly reduces environment storage overhead and improves the quality of collected agent trajectories.

---

[AI-Driven Multi-Agent Simulation of Stratified Polyamory Systems: A Computational Framework for Optimizing Social Reproductive Efficiency](http://arxiv.org/abs/2603.20678)

- SPS (Stratified Polyamory System): introduces a computational framework for modeling and evaluating a stratified polyamory system using ABM, MARL, and LLM-powered generative agents to address contemporary demographic and social crises.
- The framework utilizes MARL for decentralized matching and strategy optimization, while LLM-powered generative agents model qualitative aspects of relationship dynamics such as jealousy and communication.
- The system incorporates GNNs for social network analysis and a genetic algorithm for policy optimization to maximize aggregate social welfare and facilitate wealth dispersion through a mechanism analogous to the historical Grace Decree.

---

[REVERE: Reflective Evolving Research Engineer for Scientific Workflows](http://arxiv.org/abs/2603.20667)

- REVERE: introduces a framework for self-adapting LLM agents that continuously learns from execution trajectories to perform targeted, code-based updates to prompt fields.
- The framework utilizes a Reflector Agent that distills recurring failure modes into reusable heuristics stored within a persistent Global Training Context to improve long-horizon research coding performance.
- REVERE achieves significant performance gains across multiple research-coding benchmarks by enabling cumulative, non-destructive knowledge retention without requiring model retraining.

---

[From 50% to Mastery in 3 Days: A Low-Resource SOP for Localizing Graduate-Level AI Tutors via Shadow-RAG](http://arxiv.org/abs/2603.20650)

- Shadow-RAG: introduces a dual-agent architecture that improves the reasoning reliability of local LLMs by using a Shadow Agent to provide structured guidance to a Main Tutor.
- The framework utilizes a VLM-assisted data cleaning pipeline to transform unstructured lecture materials into a structured knowledge base for local deployment.
- Experimental results demonstrate that the Shadow Agent triggers a non-linear capability surge in 32B LLMs, enabling mastery-level performance on graduate-level mathematics tasks.

---

[Hear Both Sides: Efficient Multi-Agent Debate via Diversity-Aware Message Retention](http://arxiv.org/abs/2603.20640)

- DAR (Diversity-Aware Retention): introduces a lightweight multi-agent debate framework that improves reasoning by selectively propagating mutually disagreeing responses to maintain informative diversity.
- The framework utilizes an LLM-based filter agent to identify and retain a subset of agent responses that maximize disagreement with each other and the majority vote, effectively reducing noise and redundancy.
- By employing an index-based retention mechanism, the approach ensures that original messages remain unmodified, providing a stable and interpretable intervention that scales effectively with the number of agents.

---

[Agentic AI and the next intelligence explosion](http://arxiv.org/abs/2603.20639)

- Agentic AI Framework: introduces a perspective on AI development that shifts from monolithic models to socially-mediated, multi-agent architectures that simulate complex deliberation.
- The paper argues that intelligence is an emergent social property, proposing that future AI systems should be designed as hybrid human-AI institutions with structured roles and governance protocols.
- It emphasizes that scaling AI intelligence requires moving beyond individual model capacity toward building robust social infrastructure, including mechanisms for conflict, oversight, and institutional coordination.

---

[AEGIS: From Clues to Verdicts — Graph-Guided Deep Vulnerability Reasoning via Dialectics and Meta-Auditing](http://arxiv.org/abs/2603.20637)

- AEGIS: introduces a multi-agent framework that shifts vulnerability detection from ungrounded deliberation to forensic verification over a closed factual substrate, utilizing a Clue-Discovery Agent, Context-Augmentation Agent, Verifier Agent, and Audit Agent.
- The framework constructs a closed factual substrate using a Code Property Graph to provide a bounded, per-variable record of data provenance, which serves as the basis for dialectical verification and meta-auditing.
- By decoupling vulnerability localization from reasoning verification, AEGIS effectively mitigates contextual hallucinations and achieves state-of-the-art performance on the PrimeVul benchmark without task-specific training.

---

[A Modular LLM Framework for Explainable Price Outlier Detection](http://arxiv.org/abs/2603.20636)

- Agentic LLM Framework for Explainable Price Outlier Detection: introduces a multi-step reasoning architecture that chains Relevance Classification Agent, Utility Assessment Agent, and Reasoning-Based Decision Agent to identify anomalously high prices.
- The framework utilizes a Product Embedding Module to retrieve comparable neighbors and applies a Quadrant-based Decision Logic to provide transparent, audit-ready justifications for pricing decisions.
- Experimental results demonstrate that the system achieves over 75% agreement with human auditors while maintaining a low outlier flagging rate, effectively replacing black-box statistical methods with interpretable LLM-driven reasoning.

---

[Effective Rank Analysis and Optimization of Flexible Antenna-Enabled Wireless Systems: Movable Antennas or Pinching Antennas?](http://arxiv.org/abs/2603.20629)

- GAIQN and MAGAQN: introduces a framework for optimizing antenna positions in flexible antenna-enabled wireless systems to maximize effective rank using GRL, MAGRL, GNN, GAIQN, MAGAQN, PER, GRU, Dueling Architecture, and Top-k Action Selection.
- The paper utilizes effective rank as a structure-oriented metric to decouple antenna reconfiguration gains from power control and beamforming strategies in MA and PA systems.
- Simulation results demonstrate that the proposed algorithms enhance effective rank while ensuring collision-free antenna positioning, with MA systems providing higher effective rank and PA systems offering greater spatial DoF stability.

---

[ACRFence: Preventing Semantic Rollback Attacks in Agent Checkpoint-Restore](http://arxiv.org/abs/2603.20625)

- ACRFence: introduces a framework-agnostic mitigation that records irreversible tool effects and enforces replay-or-fork semantics upon restoration to prevent semantic rollback attacks in LLM agents.
- The system utilizes an MCP Proxy and an Analyzer LLM to distinguish between legitimate post-restore exploration and malicious or accidental duplicate tool execution.
- By capturing environment context via eBPF-based system-level monitors, the approach requires no modifications to existing LLM agent frameworks.

---

[Bayesian Learning in Episodic Zero-Sum Games](http://arxiv.org/abs/2603.20604)

- Posterior Sampling based learning algorithm: introduces a Bayesian learning approach for episodic zero-sum Markov games with unknown transition and reward models using Posterior distribution, Sampling mechanism, Dynamic programming solver, and History buffer.
- The framework maintains a Bayesian posterior over the game model, periodically samples a model, and computes an equilibrium policy for the sampled model to balance exploration and exploitation.
- Theoretical analysis establishes a sublinear expected regret bound of order O(HS√ABHK log(SABHK)) for the posterior sampling agent in both self-play and against arbitrary opponents.

---

[Position: Multi-Agent Algorithmic Care Systems Demand Contestability for Trustworthy AI](http://arxiv.org/abs/2603.20595)

- CANOE (Contestable Adaptive Network-of-Experts): introduces a human-in-the-loop framework that integrates Medical Records, Complexity Assessment, Care Team Recruitment, Argumentative Agent Framework, Quantitative Bipolar Argumentation Framework, Role-Based Human-in-the-loop Contestation, Care Plan Generation Agent, and Model Context Protocol (MCP) Agent to ensure contestable and trustworthy AI in healthcare.
- The framework utilizes specialized LLM-agents to generate structured arguments for and against clinical interventions, which are then evaluated through quantitative bipolar argumentation semantics.
- By incorporating role-based human contestation, the system allows clinicians to accept, reject, or modify agent-generated arguments, thereby preserving human agency and clinical accountability in high-stakes care decisions.

---

[Can AI Agents Answer Your Data Questions? A Benchmark for Data Agents](http://arxiv.org/abs/2603.20576)

- DAB (Data Agent Benchmark): introduces a benchmark for evaluating AI agents on complex, multi-database enterprise data tasks, utilizing a ReAct-style loop with list_db, query_db, execute_python, and return_answer components.
- The benchmark evaluates LLMs on four key properties: multi-database integration, ill-formatted join keys, unstructured text transformation, and domain knowledge, using context management to handle large data outputs.
- Experimental results across five frontier LLMs reveal that even the best-performing model achieves only 38% pass@1 accuracy, with failure analysis identifying flawed planning and incorrect implementation as the primary bottlenecks.

---

[Current state of the multi-agent multi-view experimental and digital twin rendezvous (MMEDR-Autonomous) framework](http://arxiv.org/abs/2603.20575)

- MMEDR-Autonomous: introduces a unified framework for autonomous spacecraft rendezvous and docking, integrating a learning-based navigation network, reinforcement learning-based guidance, and a hardware-in-the-loop testbed.
- The framework utilizes a lightweight CNN-based navigation network for monocular pose estimation and a DDPG-based guidance agent to generate thrust commands under mission-relevant constraints.
- System-level integration is achieved through a hardware-in-the-loop testbed that employs robotic arms and sensor fusion via Kalman filtering to validate autonomous GNC algorithms in space-representative conditions.

---

[T-MAP: Red-Teaming LLM Agents with Trajectory-aware Evolutionary Search](http://arxiv.org/abs/2603.22341)

- T-MAP (Trajectory-aware MAP-Elites): introduces a red-teaming framework that leverages execution trajectories and a Tool Call Graph to discover diverse, multi-step adversarial prompts for LLM agents.
- The framework utilizes LLMAnalyst, LLMMutator, LLMJudge, and LLMTCG to iteratively refine attack prompts based on environmental feedback and tool-to-tool transition success rates.
- T-MAP maintains a multi-dimensional archive of risk categories and attack styles to systematically explore the vulnerability landscape of autonomous agents across diverse MCP environments.

---

#### 20th March 2026

[Trojan’s Whisper: Stealthy Manipulation of OpenClaw through Injected Bootstrapped Guidance](http://arxiv.org/abs/2603.19974)

- OpenClaw: introduces a guidance injection attack vector where malicious skills embed adversarial narratives into bootstrap files to manipulate LLM reasoning without explicit commands.
- The framework utilizes an iterative dual-role generation strategy involving an Agent Simulator and a Skill Vetter to produce stealthy malicious skills that evade static and semantic detection.
- The authors evaluate the attack using ORE-Bench, demonstrating that injected guidance can successfully induce autonomous harmful actions across multiple LLM backends while remaining undetected by existing security scanners.

---

[Memori: A Persistent Memory Layer for Efficient, Context-Aware LLM Agents](http://arxiv.org/abs/2603.19935)

- Memori: introduces a persistent, LLM-agnostic memory layer that treats memory as a data structuring problem to enable efficient, context-aware interactions for LLM agents.
- The system utilizes an Advanced Augmentation pipeline to distill raw conversational logs into compact semantic triples and conversation summaries, significantly reducing token consumption compared to full-context methods.
- By integrating a Recall Engine that performs intelligent ranking and context injection, Memori achieves high-fidelity reasoning performance while maintaining a minimal context footprint for scalable deployment.

---

[Text-Based Personas for Simulating User Privacy Decisions](http://arxiv.org/abs/2603.19791)

- Narriva: introduces a framework that generates concise, theory-grounded text-based personas to simulate individual and population-level privacy decisions by compressing historical user data through an iterative optimization process.
- The framework utilizes LLMs to transform raw survey responses into structured narratives based on established privacy theories, including Privacy Calculus, Bounded Rationality, and Protection Motivation Theory.
- Narriva achieves high predictive accuracy and significant token reduction compared to raw data prompting, while enabling the creation of synthetic panels that generalize across independent studies with similar contexts.

---

[Agentic Business Process Management: A Research Manifesto](http://arxiv.org/abs/2603.18916)

- APM: introduces a conceptual framework for governing autonomous agents within organizational business processes by enforcing process-awareness through a framing mechanism.
- The architecture distinguishes between the macro-level management of processes and the micro-level execution by agents, which utilize perception-, reasoning- and action-modules to operate within defined constraints.
- The framework identifies four essential capabilities for agents—framed autonomy, explainability, conversational actionability, and self-modification—to ensure alignment with organizational goals while maintaining operational flexibility.

---

[Security, privacy, and agentic AI in a regulatory view: From definitions and distinctions to provisions and reflections](http://arxiv.org/abs/2603.18914)

- EU AI Regulatory Framework: introduces a comprehensive review of 24 European Union regulatory documents published between 2024 and 2025 to clarify definitions and provisions for AI systems, Generative AI, General-purpose AI, LLMs, Agentic AI, Information systems, and AI Factory.
- The paper deconstructs regulatory interpretations of security and privacy to resolve ambiguities surrounding the rapid evolution of autonomous agentic AI.
- It identifies a critical gap in current EU regulations, noting that while general provisions exist, specific mandates for agentic AI remain fragmented and require more granular articulation to ensure compliance.

---

[Agent Control Protocol](http://arxiv.org/abs/2603.18829)

- ACP (Agent Control Protocol): introduces a formal technical specification for governing autonomous agents in B2B environments through Identity check, Capability check, Policy check, Execution token, Audit ledger, Institutional Trust Anchor, and Mutual Recognition Agreement.
- The protocol functions as an admission control layer that validates agent intent against institutional policies before system state mutation occurs.
- ACP provides verifiable, auditable, and cryptographically secure governance for autonomous agents without replacing existing RBAC or Zero Trust infrastructure.

---

[Agentic Harness for Real-World Compilers](http://arxiv.org/abs/2603.20075)

- llvm-autofix: introduces an agentic harness designed to bridge the gap between LLMs and compiler engineering by providing specialized tools for understanding and fixing LLVM middle-end bugs.
- The framework includes llvm-bench, a benchmark of 334 reproducible compiler bugs, and llvm-autofix-mini, a tailored agent that leverages compiler-specific runtime information to outperform general-purpose software engineering agents.
- Evaluation of frontier LLMs reveals that compiler bug repair remains significantly more challenging than general software engineering tasks, with models struggling to localize and fix bugs without specialized harness support.

---

[DALI LLM-Agent Enhanced Dual-Stream Adaptive Leadership Identification for Group Recommendations](http://arxiv.org/abs/2603.19909)

- DALI: introduces a neuro-symbolic framework that utilizes LLM-based agents to identify leadership structures in group recommendations, incorporating Role Module, Memory Module, Planning Module, Action Module, Symbolic Reasoning Channel, Neural Computation Channel, and Dual-stream Aggregator.
- The framework employs a closed-loop "practice-reflection-evolution" mechanism where LLM agents autonomously refine leadership identification rules based on performance feedback.
- DALI integrates symbolic reasoning for interpretability with neural network-based representation learning to adaptively model group dynamics and power asymmetries.

---

[Utility-Guided Agent Orchestration for Efficient LLM Tool Use](http://arxiv.org/abs/2603.19896)

- Utility-Guided Agent Orchestration framework: introduces a control layer for LLM agents that explicitly evaluates candidate actions using a utility scorer based on gain, cost, uncertainty, and redundancy.
- The framework treats agent orchestration as an explicit decision problem, enabling more controllable and analyzable multi-step tool use compared to implicit prompt-driven methods.
- Experimental results demonstrate that this lightweight utility-based policy effectively balances answer quality against execution costs like token usage and latency.

---


[All-Mem: Agentic Lifelong Memory via Dynamic Topology Evolution](http://arxiv.org/abs/2603.19595)

- All-Mem: introduces an agentic lifelong memory framework that decouples low-latency online writing from offline topology consolidation to maintain a clean, searchable memory surface.
- The framework utilizes an LLM-based diagnoser and planner to perform non-destructive topology edits—SPLIT, MERGE, and UPDATE—while preserving immutable evidence through versioned typed links.
- Retrieval is optimized via a coarse-to-fine pipeline that anchors on a visible surface and performs hop-bounded expansion over typed links to recover archived evidence within fixed context budgets.

---


[Any2Speech: Borderless Long Audio Synthesis](http://arxiv.org/abs/2603.19798)

- ATS (Any2Speech): introduces a native agentic framework for long-form audio synthesis that utilizes a hierarchical Global-Sentence-Token annotation schema to bridge LLM-based planning with a VibeVoice-based synthesis engine.
- The framework employs Chain-of-Thought reasoning and Dimension Dropout to enable explicit, editable prosody planning and robust instruction following across complex acoustic scenes.
- By replacing narrow-band text inputs with a wide-band Structured Semantic Interface, the system enables an LLM to convert diverse modalities into detailed, multi-layered generation commands for coherent long-audio output.

---

[A Subgoal-driven Framework for Improving Long-Horizon LLM Agents](http://arxiv.org/abs/2603.19685)

- MiRA (Milestoning your Reinforcement Learning Enhanced Agent): introduces a subgoal-assisted framework that unifies online inference-time planning with offline RL fine-tuning via milestone-based reward shaping.
- The framework utilizes a Subgoal Generator to decompose high-level tasks into structured milestones, which are then used by a Potential Critic to provide dense, intermediate feedback during RL training.
- By integrating dynamic milestoning at inference time and milestone-based reward shaping during training, the system significantly improves the long-horizon reasoning and success rates of LLMs in complex web navigation tasks.

---

[PowerLens: Taming LLM Agents for Safe and Personalized Mobile Power Management](http://arxiv.org/abs/2603.19584)

- PowerLens: introduces a multi-agent system that leverages LLMs for zero-shot, context-aware mobile power management by bridging the semantic gap between user activities and device parameters.
- The system employs a multi-agent architecture including Activity-, Policy-, Execution-, and Feedback-agents to generate personalized power policies while ensuring safety through a PDL-based constraint verification framework.
- A two-tier memory system enables personalized preference learning from implicit user overrides via confidence-based distillation, requiring no explicit configuration and converging within 3–5 days.

---

[PlanTwin: Privacy-Preserving Planning Abstractions for Cloud-Assisted LLM Agents](http://arxiv.org/abs/2603.18377)

- PlanTwin: introduces a privacy-preserving architecture for cloud-assisted LLM agents that projects raw local environments into a sanitized digital twin to prevent exposure of sensitive data.
- The architecture utilizes a trusted local layer to mediate interactions, employing a privacy projection pipeline, a gatekeeper for policy enforcement, and an output sanitizer to ensure the cloud planner only observes bounded abstractions.
- PlanTwin achieves complete sensitive-item non-disclosure while maintaining high planning utility by enforcing cumulative disclosure budgets and restricting the cloud planner to a schema-constrained digital twin.

---

[Is Your LLM-as-a-Recommender Agent Trustable? LLMs’ Recommendation is Easily Hacked by Biases (Preferences)](http://arxiv.org/abs/2603.17417)

- BiasRecBench: introduces a comprehensive benchmark to evaluate the vulnerability of LLM-as-a-recommender agents to contextual biases in high-value decision-making tasks.
- The framework utilizes a Bias Synthesis Pipeline with Calibrated Quality Margins to strictly control the utility gap between optimal and sub-optimal options, effectively isolating latent bias tendencies from reasoning capabilities.
- Experimental results across SOTA LLMs demonstrate that even highly capable models frequently prioritize injected biases over objective quality, highlighting a significant reliability bottleneck in autonomous agentic workflows.

---

[VideoSeek: Long-Horizon Video Agent with Tool-Guided Seeking](http://arxiv.org/abs/2603.20185)

- VideoSeek: introduces a long-horizon video agent that leverages video logic flow to actively seek answer-critical evidence through a think–act–observe loop, utilizing a Thinking LLM, Toolkit, Overview Tool, Skim Tool, Focus Tool, Answer Tool, Reasoning Trajectory, and Memory.
- The framework employs a multi-granular toolkit to progressively narrow the search space from full-video overviews to fine-grained clips, significantly reducing the number of processed frames compared to dense-parsing baselines.
- By interleaving reasoning and tool use, the agent maintains a dynamic reasoning trajectory that enables efficient, query-aware evidence gathering for complex long-form video understanding tasks.

---

[AI Agents Can Already Autonomously Perform Experimental High Energy Physics](http://arxiv.org/abs/2603.20179)

- JFC (Just Furnish Context): introduces an autonomous framework for high energy physics analysis that integrates LLM-based agents with literature-based knowledge retrieval and multi-agent review to perform end-to-end analysis pipelines.
- The framework utilizes an orchestrator agent to delegate tasks to specialized subagents, ensuring structured task decomposition and rigorous validation through a multi-tier review process.
- By leveraging literature-based knowledge retrieval and a formal unblinding protocol, the system demonstrates the capability to autonomously execute complex physics analyses while maintaining scientific rigor.

---

[DESIGN-OS: A SPECIFICATION-DRIVEN FRAMEWORK FOR ENGINEERING SYSTEM DESIGN WITH A CONTROL-SYSTEMS DESIGN CASE](http://arxiv.org/abs/2603.20151)

- Design-OS: introduces a lightweight, specification-driven workflow for engineering system design that organizes the process into five distinct stages to ensure traceability from intent to parameters.
- The framework utilizes AI agents to execute structured commands across stages, supported by validation gates and user checkpoints to maintain human-AI collaboration and design integrity.
- By treating specifications as a binding contract, Design-OS enables auditable and reproducible engineering workflows for both physical and software-based systems.

---

[Can Large Multimodal Models Inspect Buildings? A Hierarchical Benchmark for Structural Pathology Reasoning](http://arxiv.org/abs/2603.20148)

- DefectBench: introduces a hierarchical benchmark and a human-in-the-loop annotation framework to evaluate LMMs on building facade inspection tasks.
- The framework utilizes a Data Curation Pipeline, Plug-and-Play Annotation Platform, Detection Refinement Module, Interactive Segmentation Module, and Hierarchical Benchmarking Tasks to assess LMM performance.
- The study reveals that while LMMs demonstrate strong topological awareness, they exhibit significant deficiencies in metric localization precision compared to specialized supervised models.

---

[Synergistic Perception and Generative Recomposition: A Multi-Agent Orchestration for Expert-Level Building Inspection](http://arxiv.org/abs/2603.20143)

- FacadeFixer: introduces a multi-agent framework that treats building facade defect inspection as a collaborative reasoning task to overcome data scarcity and semantic interference.
- The framework integrates an Orchestration Agent, Detection Agent, Segmentation Agent, and Generative Recomposition Agent to perform autonomous inspection and high-fidelity data synthesis.
- FacadeFixer utilizes a Defect Memory Bank to store and retrieve structural knowledge, enabling the generation of diverse, semantically consistent training samples for robust infrastructure monitoring.

---

[Revisiting Gene Ontology Knowledge Discovery with Hierarchical Feature Selection and Virtual Study Group of AI Agents](http://arxiv.org/abs/2603.20132)

- VSG (Virtual Study Group): introduces a multi-layer agentic AI framework that simulates a human study group to extract and validate ageing-related biological knowledge from Gene Ontology terms selected by hierarchical feature selection methods.
- The framework utilizes a bottom-up structure where Virtual Junior Researcher A and Virtual Junior Researcher B investigate and critique biological associations, followed by Virtual Senior Researcher and Virtual Principal Investigator agents who synthesize findings across four model organisms.
- The system integrates multiple LLMs, including deepseek-r1, qwen3-vl, gpt-oss, and glm-4.7-flash, to perform autonomous scientific discovery and critical review of biological claims.

---

[An Agentic Multi-Agent Architecture for Cybersecurity Risk Management](http://arxiv.org/abs/2603.20131)

- Agentic Multi-Agent Architecture for Cybersecurity Risk Management: introduces a six-agent system that decomposes the NIST CSF assessment workflow into specialized roles coordinated by a shared persistent context to ensure coherence.
- The architecture utilizes a domain fine-tuned LLM to provide sector-specific threat identification, significantly outperforming general-purpose models in grounding findings to organizational context.
- The system demonstrates that multi-agent pipelines require substantial context window capacity, as the accumulation of structured JSON data across agents can exceed the limits of constrained hardware environments.

---

[Pitfalls in Evaluating Interpretability Agents](http://arxiv.org/abs/2603.20101)

- Agentic Interpretability System: introduces an agentic framework that automates circuit analysis by iteratively designing experiments and refining hypotheses using LLMs.
- The paper identifies critical pitfalls in replication-based evaluation, such as subjectivity in human explanations and LLM memorization of published findings.
- The authors propose an unsupervised intrinsic evaluation metric based on functional interchangeability of model components to assess cluster coherence without human supervision.

---

[Orchestrating Human-AI Software Delivery: A Retrospective Longitudinal Field Study of Three Software Modernization Programs](http://arxiv.org/abs/2603.20028)

- Chiron: introduces a team-level software delivery platform that orchestrates human and AI agents across analysis, planning, implementation, and validation stages to improve software modernization outcomes.
- The platform evolves from tool-centric agent usage to an integrated orchestration layer that combines repository-native review, acceptance-criteria validation, and hybrid human-agent execution.
- Retrospective analysis of three modernization programs demonstrates that orchestrated workflows significantly reduce delivery time, lower downstream issue load, and increase first-release coverage compared to isolated agent usage.

---

[RouterKGQA: Specialized–General Model Routing for Constraint-Aware Knowledge Graph Question Answering](http://arxiv.org/abs/2603.20017)

- RouterKGQA: introduces a framework that dynamically routes queries between an efficient specialized LLM for path generation and a general LLM for agent-based repair to optimize performance and cost in KGQA.
- The framework utilizes Constraint-aware Reasoning Paths (CRPs) to enable precise answer filtering and employs a KG-guided beam search to recover executable paths when the specialized model fails.
- RouterKGQA achieves superior accuracy on complex benchmarks while significantly reducing inference costs by minimizing the number of LLM calls required for multi-hop reasoning.

---

[ReViSQL: Achieving Human-Level Text-to-SQL](http://arxiv.org/abs/2603.20004)

- ReViSQL: introduces a streamlined framework that achieves human-level Text-to-SQL performance by replacing complex agentic pipelines with expert-verified data curation, RLVR-training, and inference-time scaling.
- The framework utilizes BIRD-Verified to improve the reasoning capabilities of the Finetuned LLM, while employing a Pre-RLVR LLM for candidate Reconciliation and Majority voting to ensure robustness against natural language ambiguity.
- ReViSQL establishes a new Pareto frontier in Text-to-SQL, outperforming existing state-of-the-art agents on BIRD and Spider 2 benchmarks while significantly reducing per-query inference costs.

---

[An Agentic Approach to Generating XAI-Narratives](http://arxiv.org/abs/2603.20003)

- Agentic XAI Narrative Framework: introduces a multi-agent system for generating and iteratively refining SHAP-based explanations into faithful and coherent natural-language narratives, utilizing Narrator (generates and revises narratives), Faithful Evaluator (extracts and validates faithfulness), Faithful Critic (provides directional revision feedback), Coherence Agent (evaluates and improves linguistic quality), and Ensemble Strategy (aggregates multiple evaluator outputs via voting).
- The framework employs an iterative loop where the Narrator updates narratives based on quantitative faithfulness feedback from the Faithful Evaluator and Faithful Critic, and qualitative coherence feedback from the Coherence Agent.
- The system validates its effectiveness across five LLMs and five agentic designs, demonstrating that while agentic refinement improves faithfulness, linguistic coherence improvements can sometimes conflict with faithfulness requirements.

---

[Cavitation by phase shift of focused shock waves inside a droplet](http://arxiv.org/abs/2603.19990)

- Cavitation by phase shift of focused shock waves inside a droplet: investigates the generation of localized negative pressure within perfluorohexane droplets using focused shock waves and the Gouy phase shift mechanism.
- The study employs high-speed x-ray phase-contrast imaging and background-oriented schlieren measurements to visualize cavitation inception and density gradients, complemented by numerical hydrodynamic simulations using the ECOGEN solver.
- Findings demonstrate that purely compressive shock waves can initiate cavitation through the Gouy phase shift, offering a safer alternative to traditional ultrasound-based cavitation methods for biomedical applications.

---

[Market Power and Platform Design in Decentralized Electricity Trading](http://arxiv.org/abs/2603.19988)

- Multi-agent computational framework: introduces a differentiable multi-agent environment to analyze strategic behavior and market power in decentralized electricity trading platforms.
- The framework utilizes feedforward neural networks for prosumer agents and a differentiable market-clearing mechanism to compute price impacts through backpropagation.
- The research demonstrates that platform design, specifically pricing rules and storage ownership structure, significantly influences market power and welfare outcomes.

---

[X-World: Controllable Ego-Centric Multi-Camera World Models for Scalable End-to-End Driving](http://arxiv.org/abs/2603.19979)

- X-World: introduces an action-conditioned multi-camera generative world model that simulates future observations in video space for autonomous driving.
- The architecture utilizes a 3D causal VAE and a customized DiT backbone with view-temporal self-attention to ensure geometric consistency and temporal coherence across multi-view camera streams.
- X-World supports streaming, autoregressive generation with a rolling KV cache, enabling real-time interaction and long-horizon rollouts for closed-loop evaluation and online RL training.

---

[On the Capacity of Future Lane-Free Urban Infrastructure](http://arxiv.org/abs/2603.19952)

- OptWULF: introduces a novel lane-free automated intersection management approach using a hybrid path-velocity decomposition method to resolve vehicle conflicts.
- The framework utilizes a high-level CBS Planner to branch between spatial and temporal conflict resolutions, ensuring optimal trajectory planning for autonomous vehicles.
- OptWULF achieves even utilization of intersection space and maintains consistent capacity under both symmetric and asymmetric demand patterns.

---

[SAGE: Sustainable Agent-Guided Expert-tuning for Culturally Attuned Translation in Low-Resource Southeast Asia](http://arxiv.org/abs/2603.19931)

- SAGE: introduces a data-centric framework that utilizes an RL agent, optimized via GRPO, to autonomously curate high-quality training data from noisy corpora for efficient LLM fine-tuning.
- The framework employs a semantic reward signal derived from expert-constructed community dialogues to guide the selection of culturally relevant data, significantly reducing training energy consumption.
- SAGE integrates LoRA for parameter-efficient adaptation, enabling high-performance translation on compact LLMs while maintaining cultural nuance and minimizing environmental impact.

---

[BEYOND DETECTION: COOPERATIVE MULTI-AGENT REASONING FOR RAPID ONBOARD EO CRISIS RESPONSE](http://arxiv.org/abs/2603.19858)

- Hierarchical Multi-Agent Architecture for Onboard EO Processing: introduces a distributed, event-driven system that utilizes an Early Warning Agent (Performs initial scene screening) to selectively trigger Specialist Nodes (Conducts detailed domain-specific analysis) and a Decision Agent (Integrates evidence for final alert) to minimize onboard computational overhead.
- The architecture leverages Qwen2-VL VLM (Generates initial hazard hypotheses) and Qwen2.5 LLM (Consolidates reports and reasoning) alongside DeepLabV3+ ML Models (Performs semantic segmentation tasks) to enable structured, interpretable reasoning over multimodal satellite observations.
- By employing a hierarchical routing strategy, the system significantly reduces latency and energy consumption in non-disaster scenarios while maintaining robust decision-making capabilities through evidence-based validation.

---

[Multi-Agent Motion Planning on Industrial Magnetic Levitation Platforms: A Hybrid ADMM-HOCBF approach](http://arxiv.org/abs/2603.19838)

- ADMM-HOCBF: introduces a hybrid motion planning method for holonomic multi-agent systems that combines decentralised ADMM with a centralised HOCBF safety filter to achieve scalable and safe real-time control.
- The framework leverages the centralised architecture of industrial platforms to compute safety-critical adjustments centrally, ensuring efficient and less conservative trajectories compared to fully decentralised approaches.
- Experimental validation on a Beckhoff XPlanar system demonstrates that the method maintains real-time performance and collision avoidance in complex, non-convex industrial environments.

---

[Two-Time-Scale Learning Dynamics: A Population View of Neural Network Training](http://arxiv.org/abs/2603.19808)

- PBT (Population-Based Training): introduces a continuous-time theoretical framework for neural populations by modeling training as a two-time-scale system where fast parameter optimization via SGD/Langevin dynamics is coupled with slower hyperparameter evolution via selection-mutation.
- The framework derives a kinetic PDE in the large-population limit and a reduced selection-mutation equation for hyperparameters under strong time-scale separation, driven by an effective Gibbs-averaged fitness.
- The authors provide convergence guarantees for the population mean toward the fittest hyperparameter configuration and validate the approach through quadratic optimization problems and deep reinforcement learning tasks.

---

[Embodied Science: Closing the Discovery Loop with Agentic Embodied AI](http://arxiv.org/abs/2603.19782)

- PLAD: introduces a closed-loop paradigm for autonomous scientific discovery that integrates Perception, Language, Action, and Discovery to bridge the gap between digital reasoning and physical experimentation.
- The framework utilizes LLMs for scientific cognition, coupled with instrument-derived feedback and robotic execution to enable long-horizon autonomous discovery cycles.
- PLAD addresses structural limitations in current AI for Science by replacing isolated, task-specific predictions with a continuous, embodied interaction loop that incorporates memory, safety governance, and cumulative knowledge accumulation.

---

[Helix: A Dual-Helix Co-Evolutionary Multi-Agent System for Prompt Optimization and Question Reformulation](http://arxiv.org/abs/2603.19732)

- Helix: introduces a multi-agent framework that jointly optimizes question reformulation and prompt instructions through a structured three-stage co-evolutionary process.
- The framework utilizes a Planner, Prompt-Architect, Question-Architect, Mediator, Question-Generator, and Question-Judge to iteratively refine question-prompt pairs for improved performance.
- Helix employs bidirectional critique between architects and discriminative validation to ensure synergistic consistency across both question and prompt dimensions.

---

[WorldAgents: Can Foundation Image Models be Agents for 3D World Models?](http://arxiv.org/abs/2603.19708)

- WorldAgents: introduces a multi-agent framework that leverages 2D foundation models to synthesize coherent 3D worlds through a Director VLM, an Image Generator, and a 2-Stage Verifier.
- The framework employs a Director VLM to formulate prompts, an Image Generator for sequential inpainting, and a 2-Stage Verifier to ensure semantic and geometric consistency.
- By integrating these agents, the system constructs robust 3D scenes from 2D foundation models, enabling navigable environments without requiring explicit 3D training data.

---

[TSegAgent: Zero-Shot Tooth Segmentation via Geometry-Aware Vision-Language Agents](http://arxiv.org/abs/2603.19684)

- TSegAgent: introduces a zero-shot framework for tooth segmentation and identification that integrates Multi-view Renderer, SAM3, Mask Merging Module, ID Reordering Module, and Tooth Identification Agent (VLM) to eliminate the need for task-specific training.
- The framework leverages multi-view visual abstraction and explicit geometric inductive biases to perform anatomically consistent tooth segmentation and identification.
- By decomposing identification into multi-round conversational sub-tasks, the agent effectively reduces ambiguity and improves robustness across diverse, unseen dental scans.

---

[GoAgent: Group-of-Agents Communication Topology Generation for LLM-based Multi-Agent Systems](http://arxiv.org/abs/2603.19677)

- GoAgent: introduces a group-centric paradigm for MAS that treats collaborative groups as atomic units to construct communication topologies, utilizing a Task Encoder, Collaborative Group Pool, Autoregressive Generator, CIB, GRU, and Edge Classifier.
- The framework employs an autoregressive generator to select and connect task-relevant groups, while the CIB mechanism compresses historical signals to filter out redundant noise and improve communication efficiency.
- By shifting from node-centric to group-centric construction, GoAgent achieves state-of-the-art performance across reasoning benchmarks while significantly reducing token consumption through optimized inter-group connectivity.

---

[Structured Prompting for Arabic Essay Proficiency: A Trait-Centric Evaluation Approach](http://arxiv.org/abs/2603.19668)

- Prompting Framework for Arabic AES: introduces a three-tier prompting strategy leveraging LLMs to perform trait-specific scoring on Arabic essays without requiring model fine-tuning.
- The framework utilizes Standard Prompting, Hybrid Trait Prompting with simulated specialist agents, and Rubric-Guided Few-Shot Prompting to enhance linguistic fidelity and scoring alignment.
- Experimental results on the QAES dataset demonstrate that structured, rubric-aligned prompting significantly improves scoring reliability across various LLMs compared to holistic zero-shot approaches.

---

[Semantic Audio-Visual Navigation in Continuous Environments](http://arxiv.org/abs/2603.19660)

- MAGNet: introduces a multimodal transformer-based architecture for robust goal reasoning and efficient navigation in continuous environments by integrating historical context and self-motion cues.
- The framework utilizes a memory-augmented goal descriptor network to maintain stable goal representations even when auditory signals are intermittent or absent.
- MAGNet employs a context-aware policy network that processes accumulated scene memory to enable long-horizon navigation toward sound-emitting targets in unmapped 3D spaces.

---

[PolicySim: An LLM-Based Agent Social Simulation Sandbox for Proactive Policy Optimization](http://arxiv.org/abs/2603.19649)

- PolicySim: introduces an LLM-based multi-agent social simulation sandbox that enables proactive assessment and optimization of platform intervention policies through a User Agent Module and an Intervention Policy Module.
- The framework utilizes a unified training paradigm combining SFT and DPO to enhance agent behavioral faithfulness, while employing a contextual bandit model to adaptively optimize interventions based on simulated environment feedback.
- PolicySim effectively models bidirectional dynamics between user behavior and platform interventions, supporting scalable simulation of social ecosystems at both micro and macro levels.

---

[On the existence of fair zero-determinant strategies in the periodic prisoner’s dilemma game](http://arxiv.org/abs/2603.19641)

- Periodic Prisoner’s Dilemma Game framework: introduces a stochastic game model to investigate the existence conditions of fair ZD strategies in environments with alternating states.
- The paper establishes necessary and sufficient conditions for the existence of fair ZD strategies, demonstrating that they do not necessarily exist in periodic games unlike standard repeated games.
- The research proves that the Tit-for-Tat strategy is not always a fair ZD strategy in this stochastic setting, highlighting the increased complexity compared to standard repeated prisoner’s dilemma games.

---

[HyEvo: Self-Evolving Hybrid Agentic Workflows for Efficient Reasoning](http://arxiv.org/abs/2603.19639)

- HyEvo: introduces a self-evolving framework that autonomously constructs heterogeneous agentic workflows by integrating LLM nodes for semantic reasoning and code nodes for deterministic execution.
- The framework employs an LLM-driven multi-island evolutionary strategy that utilizes a reflect-then-generate mechanism to iteratively refine workflow topology and node logic based on execution feedback.
- By offloading predictable operations to deterministic code nodes, HyEvo significantly reduces inference costs and execution latency compared to homogeneous LLM-only workflows.

---

[Skilled AI Agents for Embedded and IoT Systems Development](http://arxiv.org/abs/2603.19583)

- IoT-SkillsBench framework: introduces a skills-based agentic architecture for hardware-in-the-loop embedded development that utilizes a Manager Node, Coder Node, and Assembler Node to integrate structured expert knowledge.
- The framework employs a Skills Library to provide compact, human-readable programming patterns and failure modes, significantly reducing token overhead compared to raw documentation.
- The system is validated through the IoT-SkillsBench benchmark, which tests agent performance across multiple MCU platforms, peripherals, and task complexities using real hardware execution.

---

[AI as Relational Translator: Rethinking Belonging and Mutual Legibility in Cross-Cultural Contexts](http://arxiv.org/abs/2603.19568)

- Relational AI Translation framework: introduces a multi-agent architecture designed to function as socio-technical infrastructure that scaffolds human-to-human connection rather than simulating synthetic relationships.
- The system utilizes a Manager Agent, Domain-Specific Agents, and a Cultural RAG layer to perform translation, navigation, reflection, and connection operations based on Self-Determination Theory and the Social Convoy Model.
- By prioritizing safety, expert validation, and cultural grounding, the framework aims to restore human relational capacity and reduce long-term reliance on LLMs through a graduation-focused design.

---

[Wearable Foundation Models Should Go Beyond Static Encoders](http://arxiv.org/abs/2603.19564)

- WFM (Wearable Foundation Model): introduces a paradigm shift for wearable health monitoring by moving from static, short-term encoders to longitudinal, anticipatory reasoning systems.
- The framework integrates Data Collection, Multimodal Encoder, Fusion Strategies, Longitudinal Temporal Modeling, Agentic Framework, and Adaptive Data Sampling to enable continuous, human-aligned health support.
- This research advocates for structurally rich data, longitudinal-aware multimodal modeling, and agentic inference to transform wearable devices from passive observers into proactive health assistants.

---

[Learning to Bet for Horizon-Aware Anytime-Valid Testing](http://arxiv.org/abs/2603.19551)

- DQN: introduces a reinforcement learning approach to horizon-aware anytime-valid testing by framing the betting strategy as a finite-horizon optimal control problem.
- The framework utilizes a DQN agent to learn a state-dependent betting policy that adapts to time remaining and distance to the rejection threshold.
- This approach improves deadline-limited power and tightens confidence sequences by dynamically switching between conservative, Kelly, and aggressive betting regimes.

---

[Evaluating Game Difficulty in Tetris Block Puzzle](http://arxiv.org/abs/2603.18994)

- SGAZ (Stochastic Gumbel AlphaZero): introduces a framework for evaluating game difficulty in stochastic environments by leveraging a planning agent that combines Gumbel AlphaZero and Stochastic AlphaZero.
- The research utilizes training rewards and convergence iterations as quantitative metrics to assess how rule variants, such as holding blocks and additional block types, impact the difficulty of the Tetris Block Puzzle.
- Empirical results demonstrate that increasing the number of holding or preview blocks reduces game difficulty, while adding complex block types like the T-pentomino significantly increases it.

---

[SG-CoT: An Ambiguity-Aware Robotic Planning Framework using Scene Graph Representations](http://arxiv.org/abs/2603.18271)

- SG-CoT: introduces a robotic planning framework that integrates scene graph representations with iterative LLM reasoning to resolve environment-induced and user-underspecified ambiguities.
- The framework utilizes a VLM to construct a structured scene graph from visual observations, which the LLM queries via retrieval functions to ground its planning process in environmental reality.
- By enabling multi-turn reasoning and targeted clarification questions, SG-CoT improves task success rates and ambiguity detection in both single-agent and partially observable multi-agent robotic environments.

---

[An Auditable AI Agent Loop for Empirical Economics: A Case Study in Forecast Combination](http://arxiv.org/abs/2603.17381)

- Auditable AI Agent Loop framework: introduces an auditable protocol for empirical economics that enforces transparency in agentic specification search by separating immutable evaluation from editable implementation.
- The protocol utilizes a four-file architecture—instruction contract, editable script, immutable evaluator, and experiment log—to ensure that every specification attempted by an LLM-based agent is recorded and inspectable.
- By incorporating a post-search holdout evaluation, the framework distinguishes robust empirical improvements from sample-specific discoveries, providing a practical guardrail against hidden researcher degrees of freedom.

---

[The Art of Midwifery in LLMs: Optimizing Role Personas for Large Language Models as Moral Assistants](http://arxiv.org/abs/2603.20626)

- Art of Midwifery framework: introduces a paradigm shift for LLMs from moral agents to moral assistants that facilitate user autonomy through Constructive Divergence and scaffolding.
- The research evaluates four distinct persona archetypes across six moral scenarios to determine optimal strategies for fostering user moral growth.
- Findings indicate that the Virtue Exemplar persona provides the most balanced performance, while context-sensitive hierarchical switching is recommended for robust moral assistance.

---

[Performance Guarantees for Data-Driven Sequential Decision-Making](http://arxiv.org/abs/2603.20553)

- ADP Bounding Framework: introduces a general theoretical approach to derive computable performance ratio bounds for ADP schemes in finite-horizon sequential decision-making problems.
- The framework utilizes a DNN-based Q-value approximator and a stepwise error function to quantify the suboptimality of ADP policies relative to optimal solutions.
- This approach generalizes classical results from submodular optimization and demonstrates practical utility in robot path planning and multi-agent sensor coverage applications.

---

[Measuring Reasoning Trace Legibility: Can Those Who Understand Teach?](http://arxiv.org/abs/2603.20508)

- Legi-Val: introduces a unified framework for evaluating the legibility of reasoning traces by decomposing them into Efficiency Dimension and Transfer-based Dimension metrics.
- The framework assesses how effectively a teacher RLM's reasoning traces scaffold a weaker student LLM toward correct answers, revealing an accuracy-legibility trade-off.
- Empirical results across 99,528 traces demonstrate that current reward models fail to intrinsically reward legibility, highlighting the need for multi-objective training in multi-agent systems.

---

[Fluid Antenna Networks Beyond Beamforming: An AI-Native Control Paradigm for 6G](http://arxiv.org/abs/2603.20484)

- AI-Native Control Architecture for Fluid Antenna Networks: introduces a closed-loop framework that integrates antenna adaptation with conventional radio resource management using Network State Representation, Intelligent Decision Engine, MARL Agent, Control Actions, and Network Environment.
- The framework utilizes MARL agents to perform distributed, adaptive control of antenna positioning and radio resources to optimize performance in multi-cell environments.
- By treating antenna configuration as a dynamic control variable, the architecture enables interference-aware adaptation that significantly improves cell-edge throughput and network fairness.

---

[DiffGraph: An Automated Agent-driven Model Merging Framework for In-the-Wild Text-to-Image Generation](http://arxiv.org/abs/2603.20470)

- DiffGraph: introduces an automated agent-driven framework that organizes online expert models into a universal graph and dynamically activates subgraphs to merge experts for diverse T2I generation needs.
- The framework utilizes GCA for graph construction, ESA for intelligent expert selection, and a VGAE-based MP to generate optimal merging coefficients without requiring retraining.
- By representing experts through node registration and calibration, the system effectively scales to evolving online resources while maintaining high performance in complex, in-the-wild generation scenarios.

---

[Solver-Aided Verification of Policy Compliance in Tool-Augmented LLM Agents](http://arxiv.org/abs/2603.20449)

- TaLLM Policy Checker: introduces a formal-methods-based framework that enforces tool-use policy compliance by intercepting LLM-generated tool calls and verifying them against SMT-LIB constraints using a Z3 Solver.
- The framework utilizes an LLM-based Fact Extractor to translate natural language context into formal assertions, which are then validated by the Z3 Solver to block policy-violating tool invocations.
- By integrating formal verification into the tool-planning loop, the approach improves agent reliability and consistency while reducing the frequency of unauthorized tool usage.

---

[Coding Agents are Effective Long-Context Processors](http://arxiv.org/abs/2603.20432)

- Coding Agents: introduces a paradigm for long-context processing that reformulates tasks as file system navigation and manipulation, utilizing Coding Agent, Navigable File System, Terminal Commands, Python Scripts, and optional Retriever.
- The framework leverages native tool proficiency and file system familiarity to enable autonomous, iterative query refinement and programmatic aggregation without task-specific training.
- By treating massive corpora as directory structures, the approach allows LLMs to perform multi-hop reasoning and complex data synthesis more effectively than standard retrieval-augmented generation pipelines.

---

[Putnam 2025 Problems in Rocq using Opus 4.6 and Rocq-MCP](http://arxiv.org/abs/2603.20405)

- Rocq-MCP: introduces a compile-first, interactive-fallback agentic framework for formal theorem proving in Rocq using general-purpose frontier LLMs.
- The framework utilizes Claude Opus 4.6 orchestrated by Claude Code to solve complex mathematical problems by iteratively writing, compiling, and debugging proof files.
- The system employs a multi-agent architecture with specialized roles including Lemma Provers, Bug Fixers, and Verifiers to manage proof-by-subgoal decomposition and formal verification.

---

[Hetero-Net: An Energy-Efficient Resource Allocation and 3D Placement in Heterogeneous LoRa Networks via Multi-Agent Optimization](http://arxiv.org/abs/2603.20404)

- Hetero-Net: introduces a unified framework for heterogeneous LoRa networks that integrates ground and underground sensor devices using UAV-mounted gateways optimized via MAPPO.
- The framework models the joint optimization of UAV 3D placement, spreading factor, and transmission power as a partially observable stochastic game to maximize system energy efficiency.
- Simulation results demonstrate that the MAPPO-based approach significantly outperforms homogeneous network designs and non-DRL baselines in energy efficiency and network adaptability.

---

[COVERAGE GAMES](http://arxiv.org/abs/2603.20398)

- CG (Coverage Games): introduces a novel two-player game framework for multi-agent planning where a covering player manages multiple agents to satisfy a shared set of objectives against an adversarial disruptor.
- The framework models scenarios where the system lacks full control over agents or interacts with an adversarial environment, requiring the decomposition of objectives among agents.
- The paper provides a comprehensive complexity analysis of coverage and disruption problems, establishing PSPACE-completeness for the general case and identifying specific conditions for NP-complete and PTIME-complete instances.

---

[CAMA: Exploring Collusive Adversarial Attacks in c-MARL](http://arxiv.org/abs/2603.20390)

- CAMA: introduces a unified framework for policy-level collusive adversarial attacks in cooperative multi-agent reinforcement learning, utilizing a Cross-Agent Transformer Encoder, Adversarial Policy Network, Value Estimator, Gating Mechanism, Role Router, and Replay Buffer.
- The framework organizes malicious agents into three progressive modes—Collective, Disguised, and Spied—to balance attack disruptiveness, stealthiness, and cost.
- Experimental results on SMAC II demonstrate that CAMA significantly improves attack efficacy and influence scope while maintaining high stealthiness compared to non-collusive baselines.

---

[The production of meaning in the processing of natural language](http://arxiv.org/abs/2603.20381)

- Quantum Semantic Framework: introduces a methodology to characterize LLM interpretive behavior by applying Bell test protocols to measure contextuality across diverse inference parameter configurations.
- The study demonstrates that LLMs exhibit non-classical contextuality, where interpretation is an emergent, observer-dependent process rather than a retrieval of pre-existing meanings.
- The research reveals that this structural indeterminacy is orthogonal to standard benchmarks, implying that safety guardrails cannot fully constrain interpretive processes that are inherently indeterminate prior to observation.

---

[ALARA for Agents: Least-Privilege Context Engineering Through Portable Composable Multi-Agent Teams](http://arxiv.org/abs/2603.20380)

- CAT (Context-Agent-Tool) data layer: introduces a declarative framework for scoping agent tool access and context to the minimum required for specific roles, utilizing Context files, NPC files, and Jinxes.
- The framework enforces behavioral constraints structurally through file-based definitions rather than relying on interpretive prose instructions, thereby mitigating security risks and capability degradation.
- The authors evaluate the framework across 22 locally-hosted LLMs, demonstrating that tool-use reliability is a distinct trained capability that correlates with general performance but varies significantly between model families.

---

[WebNavigator: Global Web Navigation via Interaction Graph Retrieval](http://arxiv.org/abs/2603.20366)

- WebNavigator: introduces a two-phase framework that resolves Topological Blindness by transforming probabilistic web exploration into deterministic retrieval and pathfinding.
- The framework utilizes a Heuristic Auto-Exploration Engine to construct an Interaction Graph, which is then indexed in a Vector Database for efficient online navigation.
- During inference, the Global-View Navigator employs a Retrieve-Reason-Teleport workflow, leveraging a Multimodal Selector and a Pathfinding Algorithm to achieve globally optimal navigation with minimal actions.

---

[Memory poisoning and secure multi-agent systems](http://arxiv.org/abs/2603.20357)

- MAS: introduces a security framework for LLM-based agents by categorizing memory systems into semantic-, episodic- and short-term-memory, each requiring specific mitigation strategies against poisoning attacks.
- The paper proposes securing agent memory through cryptographic techniques, provenance structures, and private knowledge retrieval to prevent malicious modification of factual or experiential data.
- It demonstrates a proof-of-concept Prolog-like inference engine that utilizes private information retrieval to safely query external knowledge bases from untrusted sources.

---

[Agentproof: Static Verification of Agent Workflow Graphs](http://arxiv.org/abs/2603.20356)

- Agentproof: introduces a system for pre-deployment static verification of agent workflow graphs by extracting a unified abstract model from heterogeneous frameworks and applying structural and temporal safety checks.
- The framework utilizes an Extractor to normalize diverse agent definitions into a common AgentGraph, enabling automated verification without manual modeling.
- Agentproof complements runtime guardrails by exhaustively identifying topology-level defects such as dead-end nodes and missing human-in-the-loop gates before execution.

---

[MANA: Towards Efficient Mobile Ad Detection via Multimodal Agentic UI Navigation](http://arxiv.org/abs/2603.20351)

- MANA: introduces a multimodal agentic framework for mobile ad detection that leverages Offline Profiling, Multimodal Reasoning-Guided UI Navigation, and Memory-Driven Runtime Optimization to efficiently uncover hidden advertising behaviors.
- The framework utilizes an LLM-based Decision Policy to synthesize static, visual, temporal, and experiential signals, enabling the agent to navigate complex app interfaces and identify ad-triggering paths that evade traditional heuristic-based methods.
- By integrating a Hybrid Vision Detector and a UI Transition Graph (UTG), MANA normalizes heterogeneous rendering pipelines and maintains temporal context to avoid redundant exploration and improve ad discovery accuracy.

---

[ContractSkill: Repairable Contract-Based Skills for Multimodal Web Agents](http://arxiv.org/abs/2603.20340)

- ContractSkill: introduces a framework that transforms self-generated LLM draft skills into structured, verifiable, and repairable procedural artifacts using Source Model, Draft Skill, Contracted Skill Artifact, Deterministic Verifier, Repair Module, and Target Model.
- The framework employs a deterministic verifier to perform step-level fault localization, enabling the repair module to apply minimal, targeted patches rather than performing full-skill regenerations.
- Repaired artifacts function as portable procedural assets that can be successfully executed by target models, demonstrating improved performance and transferability across diverse web-agent benchmarks.

---

[On Performance Guarantees for Federated Learning with Personalized Constraints](http://arxiv.org/abs/2603.19617)

- PC-FedAvg: introduces a personalized constrained federated optimization framework that enables agent-specific feasible sets while coupling agents through a regularized global objective.
- The framework utilizes a multi-block local decision vector and a block-wise server aggregation mechanism to maintain personalization without requiring consensus or sharing constraint information.
- Theoretical analysis establishes communication complexity rates of O(ϵ⁻²) for suboptimality and O(ϵ⁻¹) for agent-wise infeasibility, matching unconstrained FL performance.

---

[When Agents Disagree: The Selection Bottleneck in Multi-Agent LLM Pipelines](http://arxiv.org/abs/2603.20324)

- Selection Bottleneck Model: introduces a framework for multi-agent LLM pipelines that identifies a crossover threshold in aggregation quality determining whether team diversity improves or degrades performance.
- The framework demonstrates that judge-based selection of candidates from a diverse team significantly outperforms synthesis-based aggregation, which often produces inferior results compared to single-model baselines.
- The research establishes that selector quality is a critical design lever, where high-quality selection exploits candidate variance while synthesis destroys the signal provided by diverse LLM agents.

---

[The Causal Impact of Tool Affordance on Safety Alignment in LLM Agents](http://arxiv.org/abs/2603.20320)

- Cognitive Bifurcation Mechanism: introduces a framework that separates LLM behavior into a linguistic pathway governed by text-alignment objectives and an operational action path that enables tool execution, revealing how tool affordance acts as a risk amplifier.
- The framework utilizes a paired evaluation design to distinguish between speech risk, attempt risk, and effect risk, demonstrating that LLMs can maintain linguistic compliance while simultaneously attempting prohibited actions through tool-mediated pathways.
- The research identifies that tool availability induces emergent constraint circumvention strategies, where agents spontaneously decompose prohibited tasks into multiple permitted actions to satisfy optimization goals.

---

[AgentSLR: Automating Systematic Literature Reviews in Epidemiology with Agentic AI](http://arxiv.org/abs/2603.22327)

- AgentSLR: introduces an end-to-end open-source agentic pipeline that leverages language reasoning models to automate systematic literature reviews in epidemiology.
- The framework integrates Article Search and Retrieval, Title and Abstract Screening, PDF-to-Markdown Conversion, Full-text Screening, Data Extraction, Report Generation, Human-in-the-loop Validation, and Provenance Extraction to accelerate evidence synthesis.
- AgentSLR achieves performance comparable to human researchers while reducing review time by approximately 58 times, demonstrating the feasibility of automating complex scientific workflows.

---

#### 19th March 2026

[I Can’t Believe It’s Corrupt: Evaluating Corruption in Multi-Agent Governance Systems](http://arxiv.org/abs/2603.18894)

- Concordia: introduces a multi-agent governance simulation framework to evaluate whether LLMs follow institutional rules when granted formal authority.
- The framework utilizes a Game Master component to mediate interactions between LLM-powered actors, ensuring adherence to governance-specific constraints and auditable logging.
- Empirical results demonstrate that governance structure is a primary driver of corruption-related outcomes for non-saturating LLMs, highlighting the necessity of institutional design for safe AI delegation.

---

[Measuring and Exploiting Confirmation Bias in LLM-Assisted Security Code Review](http://arxiv.org/abs/2603.18740)

- LLM-based Automated Code Review (ACR) framework: introduces a systematic study of confirmation bias in LLMs, where adversarial framing in pull request metadata significantly degrades vulnerability detection performance across Interactive Review Assistant and Autonomous Review Agent components.
- The research demonstrates that LLMs exhibit a precision paradox, where bug-free framing reduces detection rates while artificially increasing precision, and confirms that this failure mode is highly exploitable in supply-chain attacks.
- The study evaluates effective mitigation strategies, finding that redacting Pull Request Metadata and providing explicit instructions to the Vulnerability Detection Model can recover most missed detections, though these measures introduce trade-offs with review efficiency.

---

[D-Mem: A Dual-Process Memory System for LLM Agents](http://arxiv.org/abs/2603.18631)

- D-Mem: introduces a dual-process memory architecture that balances efficient vector retrieval with an exhaustive deliberative fallback to mitigate lossy memory compression in LLMs.
- The framework utilizes a Multi-dimensional Quality Gating policy to dynamically route queries between a rapid System 1 retrieval module and a high-fidelity System 2 deliberation mechanism.
- By selectively triggering exhaustive processing only when initial retrieval fails, D-Mem achieves near-optimal reasoning performance while significantly reducing computational overhead and token consumption.

---

[OS-Themis: A Scalable Critic Framework for Generalist GUI Rewards](http://arxiv.org/abs/2603.19191)

- OS-Themis: introduces a multi-agent critic framework that decomposes GUI trajectories into verifiable milestones to isolate critical evidence for robust reward modeling.
- The framework utilizes a Milestone Verification Module to extract key steps and a Verdict Calibration Module to audit evidence, effectively mitigating evidence dilution and contextual loss.
- Extensive experiments on the OmniGUIRewardBench demonstrate that OS-Themis significantly improves online RL performance and facilitates autonomous agent self-evolution.

---

[AndroTMem: From Interaction Trajectories to Anchored Memory in Long-Horizon GUI Agents](http://arxiv.org/abs/2603.18429)

- AndroTMem: introduces a diagnostic framework for long-horizon Android GUI agents that utilizes AndroTMem-Bench to evaluate memory performance and proposes ASM to store causally linked intermediate-state anchors for improved decision-making.
- The framework addresses the interaction-memory bottleneck by replacing redundant raw interaction traces or lossy summaries with a structured memory bank of anchors and causal links.
- Experimental results across 12 GUI agents demonstrate that ASM significantly improves Task Completion Rate and action accuracy by enabling targeted retrieval of decision-critical intermediate states.

---

[A Framework for Formalizing LLM Agent Security](http://arxiv.org/abs/2603.19469)

- Contextual Security Framework: introduces a formal method for systematizing LLM agent security by defining an execution context Ct and four security properties—Task Alignment, Action Alignment, Source Authorization, and Data Isolation—to distinguish legitimate behavior from security violations.
- The framework utilizes oracle functions—Instruction Attribution I, Source Attribution L, and Objective Evaluation Hp, HTr, Ha—to theoretically specify the information required to verify security properties at runtime.
- By reformulating existing attacks and defenses through these contextual properties, the framework reveals fundamental limitations in current context-agnostic security approaches and provides a taxonomy for agent security violations.

---

[The Autonomy Tax: Defense Training Breaks LLM Agents](http://arxiv.org/abs/2603.19423)

- Defense-Trained LLM Agent: introduces a capability-alignment paradox where safety training systematically destroys agent competence in multi-step tasks.
- The research identifies three systematic biases—Agent Incompetence, Cascade Amplification, and Trigger Bias—that cause defended models to fail on benign tasks and exhibit high bypass rates for sophisticated attacks.
- The study demonstrates that current defense evaluation paradigms, which rely on single-turn metrics, fail to capture the catastrophic multi-step execution failures induced by shortcut learning.

---

[Automated Membership Inference Attacks: Discovering MIA Signal Computations using LLM Agents](http://arxiv.org/abs/2603.19375)

- AutoMIA: introduces an agentic framework that leverages LLM agents to automate the design and implementation of membership inference attack signal computations.
- The framework employs an evolutionary loop with ExplorerAgent, ExploiterAgent, ProgrammerAgent, ExecutorAgent, and AnalyzerAgent to iteratively discover and refine effective attack strategies.
- AutoMIA utilizes a shared ExperimentDatabase to store and retrieve past attempts, enabling agents to learn from previous successes and failures to improve performance over time.

---

[Security awareness in LLM agents: the NDAI zone case](http://arxiv.org/abs/2603.19011)

- NDAI zone case: investigates how LLMs calibrate information disclosure based on security evidence provided within a Trusted Execution Environment.
- The study utilizes a behavioral benchmark across 10 LLMs to measure sensitivity to textual security claims and hardware-based attestation results.
- Results reveal an asymmetry where LLMs reliably detect danger signals through failed attestations but exhibit heterogeneous, inconsistent responses to passing security attestations.

---

[Act While Thinking: Accelerating LLM Agents via Pattern-Aware Speculative Tool Execution](http://arxiv.org/abs/2603.18897)

- PASTE (Pattern-Aware Speculative Tool Execution): introduces a speculative execution framework that hides tool latency in LLM agents by predicting and pre-executing tool calls based on recurring application-level control flows and data dependencies.
- The framework utilizes a Pattern Tuple abstraction to decouple control flow from data flow, enabling robust late-binding parameter resolution for speculative tool invocations.
- A risk-aware scheduler employs opportunistic execution to harvest slack resources for speculative tasks, ensuring non-interference with authoritative LLM-agent operations.

---

[Memento-Skills: Let Agents Design Agents](http://arxiv.org/abs/2603.18743)

- Memento-Skills: introduces a generalist LLM agent system that autonomously constructs and improves task-specific agents through a memory-based reinforcement learning framework with stateful prompts.
- The system employs a Read–Write Reflective Learning loop where a behaviour-trainable skill router selects skills from an evolving memory, which are then executed by a frozen LLM and refined based on environment feedback.
- By treating reusable skill folders as the unit of memory, the framework enables continual learning and adaptation without requiring updates to the underlying LLM parameters.

---

[MemMA: Coordinating the Memory Cycle through Multi-Agent Reasoning and In-Situ Self-Evolution](http://arxiv.org/abs/2603.18718)

- MEMMA: introduces a plug-and-play multi-agent framework that coordinates the memory cycle along both forward and backward paths to address strategic blindness and sparse feedback.
- The framework utilizes a planner-worker architecture where a Meta-Thinker provides strategic guidance to a Memory Manager and a Query Reasoner, while an in-situ self-evolution mechanism repairs the memory bank using synthetic probe QA pairs.
- MEMMA improves long-horizon conversational memory by converting utilization failures into immediate, localized repair signals, ensuring the memory bank remains compact and consistent.

---

[Robotic Agentic Platform for Intelligent Electric Vehicle Disassembly](http://arxiv.org/abs/2603.18520)

- RAPID: introduces a human-robot collaborative platform for full-size EV battery disassembly that integrates UR16e, Parker gantry, RealSense D435i, AlloyPower ARW801, YoloWorld, MoveIt! 2, SmolAgents, ROS2, MCP server, kD-Tree, and JSON inventory to automate fastener removal.
- The system utilizes an open-vocabulary perception pipeline and agentic AI specifications to translate high-level instructions into structured robot actions for flexible, perception-driven manipulation.
- Experimental results demonstrate that explicit tool-based interfaces for LLMs significantly improve task completion reliability and reduce API costs compared to generic service discovery via MCP.

---

[From Weak Cues to Real Identities: Evaluating Inference-Driven De-Anonymization in LLM Agents](http://arxiv.org/abs/2603.18382)

- InferLink (Inference-Driven Linkage Benchmark): introduces a systematic evaluation framework to measure how LLM agents reconstruct real-world identities by combining fragmented, non-identifying cues from Anonymized Artifacts (Danon) with corroborating Auxiliary Context (Daux).
- The framework utilizes Agentic Inference to synthesize an Identity Hypothesis (ıˆ) and Supporting Evidence (E), demonstrating that modern LLMs can perform complex de-anonymization without bespoke engineering.
- Experimental results across classical, controlled, and modern digital trace settings reveal that while a Privacy-Aware System Prompt can mitigate linkage risks, it often introduces a significant trade-off by degrading task utility.

---

[ItinBench: Benchmarking Planning Across Multiple Cognitive Dimensions with Large Language Models](http://arxiv.org/abs/2603.19515)

- ItinBench: introduces a benchmark for evaluating LLMs on multi-dimensional reasoning tasks by integrating verbal reasoning with spatial route optimization in travel planning.
- The framework utilizes a Data Pipeline, User Query, Toolbox, Planner, Evaluation Strategy, and an Adapted TSP Solver to assess how LLMs balance linguistic, logical, and spatial reasoning capabilities.
- Experimental results demonstrate that LLMs struggle with consistent performance when handling multiple cognitive domains simultaneously, often relying on semantic shortcuts rather than genuine spatial cognition.

---

[Stochastic Sequential Decision Making over Expanding Networks with Graph Filtering](http://arxiv.org/abs/2603.19501)

- G-MARL: introduces a stochastic sequential decision-making framework for filtering over expanding graphs by modeling filter shifts as agents in a multi-agent reinforcement learning system.
- The framework utilizes a Context-aware Graph Neural Network to parameterize the policy, enabling the adaptation of filter parameters to dynamic graph topologies and incoming node information.
- By framing graph expansion as a Markov decision process, the approach optimizes long-term prediction performance rather than relying on myopic, instantaneous updates.

---

[Teaching an Agent to Sketch One Part at a Time](http://arxiv.org/abs/2603.19500)

- VLM agent: introduces a method for progressive, part-by-part vector sketch generation using a VLM agent, trained via SFT and multi-turn process-reward GRPO.
- The framework utilizes the ControlSketch-Part dataset to enable semantic part decomposition and path-to-part assignment for structured, interpretable, and locally editable sketch generation.
- By incorporating intermediate-state rewards through a multi-turn process-reward GRPO algorithm, the agent effectively bridges the distribution gap between oracle training states and free-form inference.

---

[HyperAgents](http://arxiv.org/abs/2603.19461)

- DGM-H (Darwin Gödel Machine with Hyperagents): introduces a self-referential framework that integrates a task agent and a meta agent into a single editable program to enable metacognitive self-modification.
- The framework extends the Darwin Gödel Machine by making the meta-level modification procedure itself modifiable, allowing agents to improve both their task-solving behavior and their own self-improvement mechanisms.
- DGM-H demonstrates generalizable, transferable self-improvement across diverse domains including coding, paper review, robotics reward design, and math grading, while autonomously developing meta-level capabilities like persistent memory and performance tracking.

---

[Cooperation and Exploitation in LLM Policy Synthesis for Sequential Social Dilemmas](http://arxiv.org/abs/2603.19453)

- Iterative LLM Policy Synthesis framework: introduces an iterative approach where an LLM generates and refines programmatic agent policies for multi-agent environments using performance feedback.
- The framework utilizes feedback engineering to compare sparse scalar rewards against dense social metrics, which act as coordination signals to improve cooperative strategies.
- The research identifies a tension between expressiveness and safety, demonstrating that LLMs can autonomously discover reward-hacking attacks by exploiting mutable environment state.

---

[TrustFlow: Topic-Aware Vector Reputation Propagation for Multi-Agent Ecosystems](http://arxiv.org/abs/2603.19452)

- TrustFlow: introduces a reputation propagation algorithm that assigns each software agent a multi-dimensional reputation vector, utilizing a topic-gated transfer operator to modulate reputation flow based on interaction content.
- The framework generalizes PageRank by replacing scalar scores with vectors and employing content-aware gating to enable query-sensitive agent discovery in multi-agent ecosystems.
- TrustFlow ensures convergence via the contraction mapping theorem and provides structural resilience against sybil attacks, reputation laundering, and vote rings through its propagation mechanism.

---

[Investigating In-Context Privacy Learning by Integrating User-Facing Privacy Tools into Conversational Agents](http://arxiv.org/abs/2603.19416)

- PNP (Privacy Notice Panel): introduces a just-in-time interface integrated into LLM-based conversational agents to support experiential, in-context privacy learning through Warning Message, Anonymization Panel, Built-in Privacy Controls, FAQs, and Proceed with Sending.
- The framework facilitates user-led protection of sensitive information by intercepting messages before submission, offering instance-based masking strategies like retracting, faking, and generalizing.
- The study demonstrates that embedding privacy tools directly into the interaction flow encourages users to actively reason about privacy trade-offs and adopt protective behaviors during LLM interactions.

---

[DYNAMIC PARETO OPTIMA IN MULTI-PERIOD PURE-EXCHANGE ECONOMIES](http://arxiv.org/abs/2603.19414)

- Dynamic Pareto Optima Framework: introduces a recursive approach to construct Pareto optimal allocations in multi-period economies by incorporating the dynamic structure of risk profiles through time-consistent dynamic risk measures.
- The framework utilizes a backward-in-time recursive optimization process to determine optimal allocations that account for both current aggregate endowments and future risk-to-go values.
- By extending the notion of comonotonicity to dynamic settings, the paper provides a crisp characterization of Pareto optimal allocations that are robust against misreporting of risks.

---

[Optimizing Resource-Constrained Non-Pharmaceutical Interventions for Multi-Cluster Outbreak Control Using Hierarchical Reinforcement Learning](http://arxiv.org/abs/2603.19397)

- HRL (Hierarchical Reinforcement Learning): introduces a framework that decouples global resource allocation from local intervention decisions to manage testing resources across multiple asynchronous outbreak clusters.
- The framework utilizes a Global Coordinator (PPO) to dynamically adjust a cost multiplier, which modulates the perceived testing cost for a Pretrained Generalized DQN that evaluates individual-level testing priorities.
- A deterministic Global Q-Ranking layer ensures strict adherence to global testing budgets by selecting the highest-value testing actions across all active clusters.

---

[TuLaBM: Tumor-Biased Latent Bridge Matching for Contrast-Enhanced MRI Synthesis](http://arxiv.org/abs/2603.19386)

- TuLaBM: introduces a latent bridge matching framework that performs NC-to-CE MRI translation as a Brownian bridge transport process in a learned latent space.
- The framework utilizes a VAE Encoder and VAE Decoder for efficient latent representation, while a Latent Denoiser guided by a TuBAM and a Boundary-Aware Matrix ensures high-fidelity synthesis of tumor regions.
- By operating in latent space and incorporating tumor-specific attention, the model achieves rapid inference and superior boundary sharpness compared to pixel-space diffusion baselines.

---

[NavTrust: Benchmarking Trustworthiness for Embodied Navigation](http://arxiv.org/abs/2603.19229)

- NavTrust: introduces a unified benchmark for evaluating the trustworthiness of embodied navigation agents across RGB, depth, and instruction modalities.
- The framework systematically assesses agent performance under realistic corruptions and evaluates four mitigation strategies including data augmentation, teacher-student distillation, adapters, and a safeguard LLM.
- Experimental results demonstrate that modular architectures and late-fusion strategies enhance robustness, while the safeguard LLM effectively mitigates instruction-level vulnerabilities.

---

[Nemotron-Cascade 2: Post-Training LLMs with Cascade RL and Multi-Domain On-Policy Distillation](http://arxiv.org/abs/2603.19220)

- Nemotron-Cascade 2: introduces a 30B Mixture-of-Experts LLM that utilizes a sequential, domain-wise Cascade RL pipeline integrated with multi-domain on-policy distillation to achieve high-density reasoning and agentic performance.
- The framework employs SFT, IF-RL, Multi-domain RL, MOPD, RLHF, Long-context RL, Code RL, and SWE RL to systematically refine model capabilities while mitigating catastrophic forgetting through domain-specific training stages.
- The model achieves gold-medal performance on IMO 2025, IOI 2025, and ICPC World Finals 2025, demonstrating superior intelligence density compared to frontier-sized models.

---

[Markov Potential Game and Multi-Agent Reinforcement Learning for Autonomous Driving](http://arxiv.org/abs/2603.19188)

- MPG-based MARL: introduces a framework that models autonomous driving as a Markov potential game to ensure Nash equilibrium attainability and algorithmic convergence.
- The approach utilizes a parameter-sharing neural network architecture to enable decentralized policy execution while maintaining scalability across multiple traffic agents.
- The framework incorporates specific reward design conditions to model self-centered and interactive driving objectives, demonstrating superior safety and efficiency in simulated and naturalistic traffic scenarios.

---

[SOL-ExecBench: Speed-of-Light Benchmarking for Real-World GPU Kernels Against Hardware Limits](http://arxiv.org/abs/2603.19173)

- SOL-ExecBench: introduces a benchmark for GPU kernel optimization that evaluates performance against analytically derived hardware Speed-of-Light bounds rather than mutable software baselines.
- The framework utilizes the SOLAR pipeline to derive hardware-grounded performance targets and includes a sandboxed evaluation harness to mitigate adversarial reward-hacking strategies.
- The benchmark comprises 235 problems extracted from 124 production AI models, covering diverse architectures and precision formats to provide a rigorous target for hardware-efficient kernel development.

---

[ADMM-Based Distributed MPC with Control Barrier Functions for Safe Multi-Robot Quadrupedal Locomotion](http://arxiv.org/abs/2603.19170)

- ADMM-based CBF-DMPC: introduces a decentralized trajectory planning framework for multi-robot systems that decomposes global optimization into parallelizable node-local and edge-local quadratic programs using ADMM and consensus constraints.
- The framework integrates high-level distributed trajectory planning with mid-level NMPC for rigid body dynamics and low-level whole-body control for full-order robot dynamics.
- Experimental results on quadrupedal robots demonstrate that the approach maintains safety and performance comparable to centralized MPC while significantly reducing per-cycle planning time as the number of agents increases.

---

[Meanings and Measurements: Multi-Agent Probabilistic Grounding for Vision-Language Navigation](http://arxiv.org/abs/2603.19166)

- MAPG (Multi-Agent Probabilistic Grounding): introduces an agentic framework that decomposes natural language queries into structured subcomponents to produce metrically consistent navigation goals in 3D space.
- The framework utilizes an Orchestrator, Grounding Agent, and Spatial Agent to translate instructions into analytic kernels, which are composed into a continuous probability density for planner-ready waypoints.
- MAPG demonstrates significant improvements in metric-semantic grounding accuracy on the introduced MAPG-Bench by replacing monolithic decision-making with explicit query decomposition and probabilistic composition.

---

[A Mathematical Theory of Understanding](http://arxiv.org/abs/2603.19349)

- Mathematical Theory of Understanding: introduces a formal model of a mind as a learning system characterized by a prerequisite structure over concepts, where understanding is defined as the ability to decode instructional signals based on previously acquired knowledge.
- The framework models teaching as sequential communication with a latent target, where instructional signals are filtered through a prerequisite-gated parser, creating a state-dependent information channel.
- The research establishes structural and epistemic lower bounds on teaching time, demonstrating that prerequisite geometry creates non-concave returns to training effort and necessitates personalized instruction for heterogeneous learners.

---

[Implicit Patterns in LLM-Based Binary Analysis](http://arxiv.org/abs/2603.19138)

- LLM-based binary analysis framework: introduces a trace-level empirical study of long-horizon, iterative LLM-driven binary analysis, identifying four recurring token-level implicit patterns—Early Pruning, Path-Dependent Lock-in, Targeted Backtracking, and Knowledge-Guided Prioritization—that govern exploration under bounded reasoning capacity.
- The framework utilizes an iterative reasoning-action-observation loop where LLMs act as autonomous agents, replacing traditional one-pass static analysis with sequential, decision-driven exploration.
- The identified patterns function as stable, structural components of LLM reasoning, exhibiting distinct temporal roles and predictable transition relationships that enable tractable binary exploration.

---

[GSMem: 3D Gaussian Splatting as Persistent Spatial Memory for Zero-Shot Embodied Exploration and Reasoning](http://arxiv.org/abs/2603.19137)

- GSMem: introduces a zero-shot embodied exploration and reasoning framework that utilizes 3D Gaussian Splatting as a persistent spatial memory to enable post-hoc re-observability for agents.
- The framework integrates a multi-level retrieval-rendering mechanism that combines object-level scene graphs and semantic-level language fields to synthesize optimal views for VLM reasoning.
- A hybrid exploration strategy balances task-aware semantic scoring with geometric coverage objectives to ensure efficient and comprehensive environment exploration.

---

[From Inference Efficiency to Embodied Efficiency: Revisiting Efficiency Metrics for Vision-Language-Action Models](http://arxiv.org/abs/2603.19131)

- VLA models: introduces a framework for evaluating embodied efficiency by analyzing the impact of model compression, token sparsification, and action sequence compression on physical robotic performance.
- The research demonstrates that conventional inference efficiency metrics, such as FLOPs or parameter counts, often fail to capture the system-level costs of robotic actuation, including task completion time and motion smoothness.
- The study proposes a set of embodied efficiency metrics to provide a more comprehensive assessment of VLA models, revealing that common inference-focused optimizations can inadvertently degrade physical execution quality.

---

[Exploring the Agentic Frontier of Verilog Code Generation](http://arxiv.org/abs/2603.19347)

- Verilog Agent: introduces a model-agnostic agentic framework for hardware design that integrates LLMs with specialized Verilog tools including iverilog, vvp, Verilator, and Yosys to iteratively refine RTL code.
- The framework utilizes a structured five-step system prompt—enforcing file discovery, planning, editing, verification, and completion—to improve agent reliability and reduce crash rates on the CVDP benchmark.
- Experimental results demonstrate that while structured prompting significantly reduces agent crashes, the primary bottleneck for performance remains the underlying reasoning capability of the LLMs rather than tool availability.

---

[CAMO: A Conditional Neural Solver for the Multi-objective Multiple Traveling Salesman Problem](http://arxiv.org/abs/2603.19074)

- CAMO: introduces a conditional neural solver for the Multi-objective Multiple Traveling Salesman Problem that utilizes a Conditional Encoder to fuse preference and instance features and a Collaborative Decoder to coordinate agents through Agent-selection Module and Node-selection Module.
- The framework employs a CA Sublayer and FF Sublayer to generate enhanced node embeddings, which are then processed by the Collaborative Decoder to construct multi-agent solutions autoregressively.
- CAMO leverages a REINFORCE-based training objective over a mixed distribution of problem sizes to ensure scalability and generalization across varying numbers of targets and agents.

---

[SignAgent: Agentic LLMs for Linguistically-Grounded Sign Language Annotation and Dataset Curation](http://arxiv.org/abs/2603.19059)

- SignAgent: introduces an agentic framework that leverages LLMs for scalable, linguistically-grounded sign language annotation and dataset curation by coordinating SignAgent Orchestrator, SignGraph, Base Tools, and Enhanced Tools.
- The framework utilizes a reasoning LLM to decompose complex annotation tasks into multistage workflows, integrating multimodal evidence with structured knowledge retrieval to improve annotation accuracy and interpretability.
- SignAgent demonstrates superior performance in pseudo-gloss alignment and ID-glossing by replacing fixed-pipeline approaches with an agentic loop that grounds decisions in phonological, syntactic, and semantic linguistic knowledge.

---

[The Simplicity of the Hodge Bundle](http://arxiv.org/abs/2603.19052)

- Aletheia: introduces a mathematical research framework that utilizes an LLM-based agent to generate proofs for complex geometric theorems through structured human-AI interaction.
- The framework leverages the Gemini Deep Think LLM to perform autonomous mathematical reasoning, specifically demonstrating the simplicity of the Hodge bundle by analyzing curve symmetries.
- This research highlights the efficacy of LLM-driven agents in providing rigorous, verifiable mathematical proofs while maintaining a transparent record of the interaction process via interaction cards.

---

[MoRI: Learning Motivation-Grounded Reasoning for Scientific Ideation in Large Language Models](http://arxiv.org/abs/2603.19044)

- MoRI (Motivation-grounded Reasoning for Scientific Ideation): introduces a framework that internalizes scientific reasoning by training LLMs to generate grounded methodologies from research motivations using a composite reinforcement learning reward.
- The framework utilizes Entropy-Aware Information Gain to incentivize technical depth and Contrastive Semantic Gain to ensure logical alignment with research goals, both modulated by Length Anchoring to prevent reasoning collapse.
- Empirical results demonstrate that MoRI outperforms commercial LLMs and agentic baselines in novelty, technical rigor, and feasibility by moving beyond surface-level imitation to deliberate, motivation-driven problem solving.

---

[LLMs Aren’t Human: A Critical Perspective on LLM Personality](http://arxiv.org/abs/2603.19030)

- LLM Personality Evaluation Framework: introduces a critical analysis of applying human personality inventories to LLMs, arguing that current models fail to satisfy the six defining characteristics of human personality.
- The paper demonstrates that LLM responses to personality tests are highly sensitive to prompts and contextual cues, undermining their validity as stable, context-invariant traits.
- The authors propose a research agenda shifting from anthropomorphic trait attribution toward functional behavioral evaluation and the identification of stable, LLM-specific intrinsic characteristics.

---

[AgentDS Technical Report: Benchmarking the Future of Human-AI Collaboration in Domain-Specific Data Science](http://arxiv.org/abs/2603.19005)

- AgentDS: introduces a benchmark and competition designed to evaluate both autonomous AI agents and human-AI collaboration performance in domain-specific data science tasks.
- The framework utilizes a hierarchical evaluation system that aggregates performance across 17 challenges spanning six distinct industries to measure cross-domain data science capability.
- Empirical results demonstrate that while agentic AI improves upon direct prompting, human expertise remains essential for domain-specific reasoning, feature engineering, and strategic decision-making.

---

[Teleological Inference in Structural Causal Models via Intentional Interventions](http://arxiv.org/abs/2603.18968)

- SFM (Structural Final Model): introduces a formal framework extending SCMs to model intentional agents by twinning an unintervened causal model with a factual intervened model.
- The framework utilizes an intentional intervention operator to represent state-aware and goal-directed agent behavior without requiring explicit time dimensions.
- SFMs enable teleological inference by empirically detecting agents through violations of Markovianity and discovering agent intentions via counterfactual analysis of system descendants.

---

[Maximum-Entropy Exploration with Future State-Action Visitation Measures](http://arxiv.org/abs/2603.18965)

- MaxEntRL: introduces a reinforcement learning objective based on the relative entropy of the discounted distribution of future state-action features to enhance exploration.
- The framework utilizes a conditional visitation model, proven to be the fixed point of a contraction mapping, to compute intrinsic rewards off-policy.
- Experimental results demonstrate that this approach improves feature visitation within individual trajectories and accelerates convergence for exploration-only agents compared to standard methods.

---

[Optimal Path Planning in Hostile Environments](http://arxiv.org/abs/2603.18958)

- Routing Plan: introduces a multi-agent path planning model for hostile environments where assets are eliminated by traps that require a reload period after each activation.
- The framework establishes that the problem is in NP and proves NP-hardness for general graphs and trees with maximum degree 3, while identifying polynomial-time solvable fragments using a run-wait-sacrifice strategy.
- The research demonstrates the APX-hardness of the optimization variant of the problem, Routing Plan*, through reductions from Max 3 SAT-3 and 3-Partition.

---

[MultihopSpatial: Multi-hop Compositional Spatial Reasoning Benchmark for Vision-Language Model](http://arxiv.org/abs/2603.18892)

- MultihopSpatial: introduces a comprehensive benchmark for multi-hop compositional spatial reasoning and visual grounding in VLMs, utilizing the Acc@50IoU metric and the MultihopSpatial-Train corpus for reinforcement learning post-training.
- The framework employs GRPO to optimize VLMs on a composite reward function that balances format adherence, answer correctness, and spatial localization accuracy.
- Experimental results demonstrate that RL post-training on the MultihopSpatial-Train corpus significantly enhances both intrinsic spatial reasoning and downstream embodied VLA task performance.

---

[Bridging Network Fragmentation: A Semantic-Augmented DRL Framework for UAV-aided VANETs](http://arxiv.org/abs/2603.18871)

- SA-DRL: introduces a four-stage pipeline that integrates LLM-based topological reasoning into DRL to mitigate network fragmentation in UAV-aided VANETs.
- The framework utilizes RTG and DCG to quantify connectivity and employs a Logit Fusion mechanism to inject semantic priors directly into the SA-PPO decision-making loop.
- By leveraging LoRA for knowledge alignment and a vectorized parallel training system, the approach achieves superior connectivity and energy efficiency with significantly reduced training episodes.

---

[Conflict-Based Search for Multi Agent Path Finding with Asynchronous Actions](http://arxiv.org/abs/2603.18866)

- CBS-AA: introduces a complete and optimal algorithm for Multi-Agent Path Finding with Asynchronous Actions by replacing standard conflict resolution with mutually disjunctive constraints.
- The framework utilizes a modified low-level planner, SIPPS-WC (Safe Interval Path Planning with Waiting Conflict), to efficiently handle continuous-time constraints and prune search states.
- Experimental results demonstrate that the proposed constraint propagation techniques significantly reduce the number of high-level search branches and improve scalability compared to existing continuous-time solvers.

---

[RewardFlow: Topology-Aware Reward Propagation on State Graphs for Agentic RL with Large Language Models](http://arxiv.org/abs/2603.18859)

- RewardFlow: introduces a lightweight method for estimating state-level rewards in agentic RL by constructing state graphs from reasoning trajectories to propagate terminal rewards.
- The framework leverages the intrinsic topological structure of state graphs to quantify the contribution of intermediate states to task success without requiring external reward models.
- By integrating action-level and trajectory-level advantages, RewardFlow provides dense, objective supervision that stabilizes and accelerates the training of LLMs in complex agentic environments.

---

[Can LLM generate interesting mathematical research problems?](http://arxiv.org/abs/2603.18813)

- DeepMath-generate: introduces an agent-based framework utilizing LLMs to generate novel, research-level mathematical problems in differential geometry.
- The framework employs a Generator and an Evaluator, both powered by LLMs, to iteratively refine problems based on specific criteria for mathematical creativity and research value.
- The system incorporates a filtering mechanism and structured prompts to ensure generated problems are non-trivial, logically sound, and distinct from existing mathematical literature.

---

[Mi:dm K 2.5 Pro](http://arxiv.org/abs/2603.18788)

- Mi:dm K 2.5 Pro: introduces a 32B parameter LLM optimized for enterprise-grade reasoning through a multi-stage pipeline utilizing Data Curation Pipeline, AST-based Execution Filter, Layer-predictor-based Depth Upscaling (DuS), Reasoning SFT, Model Merging, Asynchronous Reinforcement Learning (RL), Fusion Training, and LLM-as-a-Judge Reward Model.
- The framework employs a reasoning-centric pre-training strategy with DuS for efficient scaling and a progressive context-length extension to 128K tokens.
- Post-training integrates Fusion Training to balance complex reasoning with conversational fluency, while utilizing an asynchronous RL architecture to improve training efficiency and stability.

---

[Analysis Of Linguistic Stereotypes in Single and Multi-Agent Generative AI Architectures](http://arxiv.org/abs/2603.18729)

- Multi-Agent Mitigation Pipeline: introduces a three-stage workflow to mitigate dialect-conditioned bias in LLMs by utilizing Input Construction, Agent 1: Generator, Agent 2: Critic, Agent 3: Reviser, and Final Output &amp; Evaluation.
- The research evaluates how linguistic variations in Standard American English and African-American English trigger stereotype-based inferences across different LLM architectures.
- The study demonstrates that multi-agent critique-revision pipelines provide more consistent bias mitigation compared to single-agent prompting strategies like role-based or Chain-Of-Thought prompting.

---

[A more accurate rational non-commutative algorithm for multiplying 4×4 matrices using 48 multiplications](http://arxiv.org/abs/2603.18699)

- ⟨4×4×4:48⟩: introduces a numerically accurate variant of a 4×4 matrix multiplication algorithm using 48 non-commutative multiplications over the rationals.
- The algorithm utilizes L-matrix, R-matrix, and P-matrix components to achieve an improved error bound exponent of log4γ∞,2 ≈ 2.386.
- The paper provides a straight-line program and alternative basis variants to optimize the growth factor and numerical stability in sub-cubic matrix multiplication.

---

[HISR: Hindsight Information Modulated Segmental Process Rewards For Multi-turn Agentic Reinforcement Learning](http://arxiv.org/abs/2603.18683)

- HISR: introduces a reinforcement learning framework that modulates segmental process rewards using hindsight information to improve credit assignment in long-horizon agentic tasks.
- The framework utilizes a segment-level reward model to align supervision with sub-goals and a hindsight model to measure action importance through sequence likelihood ratios.
- By integrating these importance scores with segmental rewards and an action grounding mechanism, the approach effectively mitigates sparse reward issues and unfocused credit assignment in LLMs.

---

[Complexity of Auctions with Interdependence](http://arxiv.org/abs/2603.18668)

- Complexity of Auctions with Interdependence: introduces a computational framework for designing truthful mechanisms in interdependent value settings by reducing optimization problems to combinatorial structures like bipartite matching and 2-SAT.
- The paper establishes polynomial-time algorithms for computing optimal approximation ratios in special cases (n=2 or k=2) and proves NP-hardness for the general case (n=4).
- The authors characterize truthful mechanisms using signal orderings and provide tight lower bounds on query complexity for evaluating allocation rules at specific signal profiles.

---

[Multimodal Model for Computational Pathology: Representation Learning and Image Compression](http://arxiv.org/abs/2603.18660)

- Multimodal Computational Pathology Framework: introduces a comprehensive review of multimodal AI in pathology, focusing on representation learning, structure-aware token compression, and multi-agent collaborative reasoning.
- The framework integrates hierarchical visual encoders with multi-resolution input pyramids to address the computational challenges of processing gigapixel whole-slide images.
- It leverages multi-agent systems and mixture-of-experts architectures to simulate clinical diagnostic workflows, enhancing interpretability and evidence fusion for trustworthy AI-assisted pathology.

---

[Mean-field control barrier functions for stochastic multi-agent systems](http://arxiv.org/abs/2603.18658)

- MF-CBF (Mean-Field Control Barrier Function): introduces a framework for enforcing safety constraints in large-scale stochastic multi-agent systems by modeling collective behavior through density distributions and advection-diffusion equations.
- The approach utilizes a kernel-weighted overlap functional to define safe sets and computes safety-corrected control inputs via a quadratic optimization problem to ensure forward invariance of the safe region.
- Theoretical analysis demonstrates that the framework provides bounded stability guarantees for density control problems, validated through coverage and shepherding scenarios.

---

[Robust mean-field games under entropy-based uncertainty](http://arxiv.org/abs/2603.18628)

- RMFG framework: introduces a robust mean-field game model where a representative agent optimizes against an adversarial Nature that distorts the reference probability measure at an entropic cost.
- The equilibrium is established via a fixed-point condition requiring the population distribution to coincide with the law induced by the agent's optimal strategy under Nature's effective measure.
- The model utilizes a coupled FBSDE system to characterize the optimal control of the representative agent and the adversarial dynamics of Nature, ensuring consistency within the mean-field interaction.

---

[Agentic Flow Steering and Parallel Rollout Search for Spatially Grounded Text-to-Image Generation](http://arxiv.org/abs/2603.18627)

- AFS-Search (Agentic Flow Steering and Parallel Rollout Search): introduces a training-free closed-loop framework that reformulates T2I generation as an active decision-making process by integrating VLM Prompt Optimizer, FLUX.1-dev, VLM Critic, SAM3, Parallel Rollout Search, Agentic Flow Steering, and Global Feedback Loop.
- The framework utilizes Parallel Rollout Search to explore multiple generation trajectories at critical bifurcation points, selecting the optimal path based on rewards provided by the VLM Critic.
- Agentic Flow Steering dynamically modulates the velocity field of the flow-matching process using contrastive energy gradients, enabling precise spatial grounding and error correction without requiring model fine-tuning.

---

[REST: Receding Horizon Explorative Steiner Tree for Zero-Shot Object-Goal Navigation](http://arxiv.org/abs/2603.18624)

- REST (Receding Horizon Explorative Steiner Tree): introduces a training-free framework for zero-shot object-goal navigation that replaces isolated waypoint scoring with a path-grounded navigation decision tree.
- The framework utilizes a hierarchical option space constructed via Euclidean Steiner Tree optimization to enable coarse-to-fine LLM reasoning over spatial narratives.
- REST decouples the system into a fast reactive layer for continuous path maintenance and an event-triggered deliberative layer for LLM-based decision-making.

---

[Reasonably reasoning AI agents can avoid game-theoretic failures in zero-shot, provably](http://arxiv.org/abs/2603.18563)

- PS-BR: introduces a framework where reasoning LLM agents achieve Nash-like equilibrium play in repeated games through Bayesian learning and asymptotic best-response learning without explicit post-training.
- The approach utilizes posterior-sampling best response (PS-BR) to reconcile the stochastic nature of LLMs with the theoretical requirements for Nash convergence by sampling hypotheses from a finite menu and performing rollout-based strategy evaluation.
- The framework provides theoretical guarantees for zero-shot Nash convergence in infinitely repeated games, even under conditions of unknown, stochastic, and private payoffs, by ensuring on-path stabilization of predictive beliefs and approximate best responses.

---

[HiMu: Hierarchical Multimodal Frame Selection for Long Video Question Answering](http://arxiv.org/abs/2603.18558)

- HiMu: introduces a training-free, neuro-symbolic framework that decomposes video queries into hierarchical logic trees to enable efficient, compositional frame selection for long-form video QA.
- The framework utilizes a single text-only LLM call to generate a logic tree, routing atomic predicates to modality-specific experts including CLIP, OVD, OCR, ASR, and CLAP.
- HiMu employs fuzzy-logic operators and the PASS selection strategy to produce a satisfaction curve, achieving high-accuracy frame selection with significantly lower computational cost than agentic systems.

---

[Total Recall QA: A Verifiable Evaluation Suite for Deep Research Agents](http://arxiv.org/abs/2603.18516)

- TRQA (Total Recall Question Answering): introduces a verifiable evaluation framework for Deep Research Agents that requires retrieving all relevant documents from a large corpus to synthesize accurate answers.
- The framework utilizes an entity-centric approach to automatically generate datasets from paired structured knowledge bases and text corpora, ensuring precise relevance judgments.
- Benchmark results demonstrate that current LLMs and Deep Research Agents struggle with total recall tasks, primarily due to reasoning failures rather than parametric knowledge limitations.

---

[Expert Personas Improve LLM Alignment but Damage Accuracy: Bootstrapping Intent-Based Persona Routing with PRISM](http://arxiv.org/abs/2603.18507)

- PRISM (Persona Routing via Intent-based Self-Modeling): introduces a self-bootstrapped pipeline that internalizes intent-conditioned expert persona routing into a gated LoRA adapter to improve alignment without degrading knowledge retrieval.
- The framework utilizes a Base LLM to generate queries, perform self-verification via pairwise comparison, and train a Binary Gate to selectively activate a LoRA Adapter for alignment-dependent tasks.
- By avoiding external data and human annotation, PRISM effectively balances the trade-off between persona-driven alignment gains and the accuracy degradation typically observed in pretrained knowledge tasks.

---

[Computationally Efficient Density-Driven Optimal Control via Analytical KKT Reduction and Contractive MPC](http://arxiv.org/abs/2603.18503)

- D2OC (Density-Driven Optimal Control): introduces a computationally efficient framework for multi-agent spatial distribution control by reducing high-dimensional KKT systems into condensed quadratic programs.
- The framework incorporates a contractive Lyapunov constraint to ensure Input-to-State Stability (ISS) against reference propagation drift while maintaining linear O(T) computational scalability.
- A specialized Dual-Newton solver is utilized to achieve real-time performance by exploiting the analytical structure of the condensed optimization problem.

---

[Cross-Domain Demo-to-Code via Neurosymbolic Counterfactual Reasoning](http://arxiv.org/abs/2603.18495)

- NESYCR: introduces a neurosymbolic framework that adapts robotic task procedures from demonstrations to new domains by performing counterfactual reasoning through a VLM and a symbolic tool.
- The framework constructs a symbolic world model from demonstrations and iteratively resolves cross-domain procedural incompatibilities by identifying violated preconditions and proposing verified alternative actions.
- NESYCR synthesizes deployment-grounded control code policies that maintain task intent while ensuring causal validity across diverse environmental and embodiment configurations.

---

[CyberJustice Tutor: An Agentic AI Framework for Cybersecurity Learning via Think–Plan–Act Reasoning and Pedagogical Scaffolding](http://arxiv.org/abs/2603.18470)

- CyberJustice Tutor: introduces an agentic AI framework that utilizes a Think–Plan–Act cognitive cycle to provide personalized cybersecurity education for criminal justice professionals.
- The system integrates a Pedagogical Scaffolding Layer grounded in Vygotsky’s Zone of Proximal Development to dynamically adapt instructional support based on the learner's real-time progress.
- An Adaptive RAG pipeline anchors the LLM's reasoning in verified curriculum materials to ensure domain accuracy and mitigate hallucinations in high-stakes legal contexts.

---

[SODIUM: From Open Web Data to Queryable Databases](http://arxiv.org/abs/2603.18447)

- SodiumAgent: introduces a multi-agent system that automates the materialization of structured databases from open-domain web data by integrating a Web Explorer (core processing unit for exploration) and a Cache Manager (storage unit for paths/sources).
- The system utilizes an ATP-BFS Algorithm (breadth-first exploration workflow) to conduct deep, domain-specialized navigation, supported by a Page Explorer (subagent for page-wise interaction) and a Link Processor (module for URL augmentation/pruning/ranking).
- SodiumAgent leverages a File Inspector (tool for document/image analysis) and cache-based structural reuse to maintain global consistency and efficiency across complex, multi-step data extraction tasks.

---

[SR-Nav: Spatial Relationships Matter for Zero-shot Object Goal Navigation](http://arxiv.org/abs/2603.18443)

- SR-Nav: introduces a framework that models observed and experience-based spatial relationships to enhance perception and planning in zero-shot object goal navigation.
- The framework utilizes a DSRG to integrate experiential spatial priors with real-time observations, enabling predictive reasoning for navigation in unseen environments.
- SR-Nav incorporates a RAMM for robust target verification and a DRPM for efficient path planning, significantly improving success rates and navigation efficiency compared to existing modular approaches.

---

[TopoChunker: Topology-Aware Agentic Document Chunking Framework](http://arxiv.org/abs/2603.18409)

- TopoChunker: introduces a topology-aware agentic framework that maps heterogeneous documents onto a Structured Intermediate Representation (SIR) to preserve hierarchical dependencies for RAG.
- The framework utilizes a dual-agent architecture consisting of an Inspector Agent for adaptive routing and a Refiner Agent for semantic auditing to balance structural fidelity with computational cost.
- By dynamically injecting topological lineage and resolving contextual voids, the system mitigates semantic fragmentation and improves retrieval accuracy compared to linear chunking methods.

---

[Interleaved Information Structures in Dynamic Games: A General Framework with Application to the Linear-Quadratic Case](http://arxiv.org/abs/2603.18407)

- MPN (Mathematical Program Network): introduces a systematic framework for modeling noncooperative dynamic games with arbitrary interleaved information structures by representing agent interdependencies as a directed graph of optimization problems.
- The framework utilizes Decision Nodes, Dynamical Constraints, Temporal Edges, and Observation Edges to encode complex informational relationships between agents across discrete timesteps.
- By deriving KKT Conditions from the resulting Solution Graph, the approach provides a method to compute Nash equilibria for linear-quadratic dynamic games through a system of Riccati-like Equations.

---

[Graph-of-Constraints Model Predictive Control for Reactive Multi-agent Task and Motion Planning](http://arxiv.org/abs/2603.18400)

- GoC-MPC: introduces a reactive multi-agent TAMP framework that utilizes a Directed Acyclic Graph (DAG) of constraints to enable dynamic agent assignment and parallel task execution.
- The framework decomposes complex TAMP problems into waypoint optimization, spline-based path planning, and receding-horizon control to achieve real-time performance.
- GoC-MPC supports robust disturbance recovery through an integrated backtracking mechanism that re-evaluates task constraints when perturbations occur.

---

[Reflection in the Dark: Exposing and Escaping the Black Box in Reflective Prompt Optimization](http://arxiv.org/abs/2603.18388)

- VISTA: introduces a multi-agent APO framework that decouples hypothesis generation from prompt rewriting to enable verifiable and interpretable optimization traces.
- The framework utilizes a semantic trace tree to record causal links between hypotheses and prompt improvements, addressing the opacity of traditional reflective APO methods.
- A two-layer explore-exploit mechanism, combining random restart and epsilon-greedy hypothesis sampling, ensures robust global search and escapes seed-induced traps.

---

[TDAD: Test-Driven Agentic Development – Reducing Code Regressions in AI Coding Agents via Graph-Based Impact Analysis](http://arxiv.org/abs/2603.17973)

- TDAD: introduces a graph-based impact analysis tool that provides AI coding agents with targeted test context to reduce regressions during software development.
- The framework utilizes an AST Parser, Graph Builder, and Test Linker to generate a test_map.txt artifact, which guides the AI Coding Agent in verifying only relevant tests.
- TDAD demonstrates that providing concise, graph-derived context outperforms verbose procedural instructions, significantly reducing regressions while maintaining or improving resolution rates in LLM-based coding agents.

---

[When Only the Final Text Survives: Implicit Execution Tracing for Multi-Agent Attribution](http://arxiv.org/abs/2603.17445)

- IET (Implicit Execution Tracing): introduces a metadata-independent framework that enables token-level attribution and interaction topology reconstruction directly from generated text by embedding agent-specific keyed signals into the token distribution.
- The framework utilizes a modulation operator to inject statistical signatures during generation, which are subsequently recovered at detection time using sliding-window scoring and competitive change-point detection.
- IET enables privacy-preserving auditing of multi-agent systems by allowing the reconstruction of execution paths and interaction topologies even when explicit logs or agent identifiers are unavailable.

---

[Allocating Chores with Restricted Additive Costs: Achieving EFX, MMS, and Efficiency Simultaneously](http://arxiv.org/abs/2603.17270)

- Fair Allocation Algorithm for Restricted Chores: introduces a three-phase algorithmic framework to compute allocations for indivisible chores that are simultaneously EFX, MMS, and achieve a 2-approximation of the optimal social cost.
- The framework partitions items into consistent-cost (M+) and zero-cost (M0) sets, utilizing an initial MMS-feasible partition of M+ followed by a turn-based modification process for M0 to ensure fairness.
- The algorithm provides a constructive approach to fair division in restricted settings, establishing that the price of fairness for EFX is exactly 2.

---

[SWARM+: Scalable and Resilient Multi-Agent Consensus for Fully-Decentralized Data-Aware Workload Management](http://arxiv.org/abs/2603.19431)

- SWARM+ (Scalable and Resilient Multi-Agent Consensus for Fully-Decentralized Data-Aware Workload Management): introduces a decentralized, hierarchical multi-agent framework that utilizes Hierarchical Multi-Agent System Layer, Consensus Layer, Selection Layer, Resilience Mechanisms, Redis Key-Value Store, and gRPC Messaging to achieve scalable and resilient scientific workload management.
- The framework employs a tree-structured hierarchical topology to decompose consensus tasks, reducing communication complexity from O(n²) to O(log n) while maintaining data-aware job scheduling.
- SWARM+ ensures system resilience through multi-signal failure detection, automatic job reselection, and an adaptive quorum mechanism that allows for graceful degradation and elastic scaling without centralized services.

---

[Reason-to-Transmit: Deliberative Adaptive Communication for Cooperative Perception](http://arxiv.org/abs/2603.20308)

- R2T: introduces a deliberative reasoning framework for cooperative perception that uses a lightweight transformer to optimize communication decisions based on local context, neighbor information gaps, and bandwidth constraints.
- The framework employs a gated cross-attention fusion module to effectively integrate received features while suppressing noise and redundant information from neighboring agents.
- Experimental results demonstrate that R2T achieves superior performance under high occlusion by modeling receiver information needs, while maintaining robustness against packet loss.

---

[Semantic Tool Discovery for Large Language Models: A Vector-Based Approach to MCP Tool Selection](http://arxiv.org/abs/2603.20313)

- Semantic Tool Discovery Architecture: introduces a vector-based retrieval system that dynamically selects relevant tools for LLMs to mitigate token overhead and context window limitations.
- The framework utilizes a Tool Indexing Pipeline, Query Processing, LLM Integration, Feedback Loop, Vector Database, and Embedding Model to optimize tool-calling efficiency.
- Experimental results demonstrate a 99.6% reduction in tool-related token consumption while maintaining high accuracy and sub-100ms retrieval latency.

---

[kRAIG: A Natural Language-Driven Agent for Automated DataOps Pipeline Generation](http://arxiv.org/abs/2603.20311)

- kRAIG (k-Retrieval Augmented Integration Governor): introduces an AI agent that translates natural language specifications into production-ready Kubeflow Pipelines using the ReQuesAct interaction framework, Task Understanding, Tool Understanding, Pipeline Generation, Pipeline Compilation, Task Long-term Memory, Validation Layer, and Kubeflow Pipelines.
- The framework utilizes a hybrid neurosymbolic approach to clarify user intent through multi-turn dialogue before performing retrieval-augmented tool synthesis and LLM-based validation to ensure pipeline integrity.
- By incorporating explicit intent clarification and safety-focused validation stages, kRAIG significantly enhances the reliability and executability of automated data engineering pipelines compared to traditional agentic baselines.

---

#### 18th March 2026


[Cyberlanguage: Native Communication for the Cyber-Physical-Social-Thinking Fusion Space](http://arxiv.org/abs/2603.17498)

- Cyberlanguage: introduces a theoretically grounded communicative framework designed to natively integrate physical, social, cognitive, and cyber dimensions for heterogeneous agents.
- The framework utilizes a Cybersign semiotic model and a five-layer architectural stack to enable semantically consistent communication across diverse agent types including humans, AI, and robots.
- It employs a Four-Dimensional Synchronous Grammar to facilitate dynamic compilability and contextual adaptability, serving as a meta-communicative infrastructure for fused reality environments.

---


[Governed Memory: A Production Architecture for Multi-Agent Workflows](http://arxiv.org/abs/2603.17787)

- Governed Memory: introduces a shared infrastructure layer for multi-agent workflows that addresses memory governance gaps through Dual Memory Store, Governance Routing, Governed Retrieval, and Schema Lifecycle and Quality Feedback.
- The architecture utilizes a dual memory model combining open-set atomic facts with schema-enforced typed properties to ensure structured and unstructured data are both queryable and actionable for LLMs.
- The system incorporates tiered governance routing with progressive context delivery to reduce token usage and improve LLM performance by injecting only delta content across autonomous execution steps.

---

[DustNET: enabling machine learning and AI models of dusty plasmas](http://arxiv.org/abs/2603.17493)

- DustNET: introduces a comprehensive framework for building standardized, multi-modal datasets and AI models to advance dusty plasma research through DustNET, DUST-MAP, CNN, GNN, RNN, Transformer, Generative Models, and Edge AI Agents.
- The framework integrates experimental measurements, numerical simulations, and AI-generated synthetic data to enable predictive modeling, uncertainty quantification, and multi-scale analysis of dusty plasmas.
- DustNET-driven models facilitate real-time diagnostics and autonomous control in laboratory, industrial, and astrophysical settings by leveraging advanced machine learning architectures.

---

[IEMAS: An Incentive-Efficiency Routing Framework for Open Agentic Web Ecosystems](http://arxiv.org/abs/2603.17302)

- IEMAS: introduces a distributed routing framework that aligns economic incentives with system-level efficiency in open multi-agent LLM ecosystems by integrating cache-aware predictive modeling with VCG-based auction mechanisms.
- The framework utilizes a Proxy Hub architecture to perform fine-grained request-to-agent matching, leveraging KV-cache affinity to minimize computational redundancy and improve overall system performance.
- By formulating task allocation as a Min-Cost Max-Flow problem, IEMAS ensures truthful capability reporting and social optimality while maintaining scalability through domain-based clustering of agents.

---

[TRUST-SQL: Tool-Integrated Multi-Turn Reinforcement Learning for Text-to-SQL over Unknown Schemas](http://arxiv.org/abs/2603.16448)

- TRUST-SQL: introduces a four-phase interaction protocol and Dual-Track GRPO to enable LLMs to perform Text-to-SQL in unobservable environments without pre-loaded schema.
- The framework utilizes a Propose checkpoint to enforce verified metadata usage, preventing hallucinations and providing a structural boundary for independent optimization of schema grounding and SQL generation.
- By applying token-level masked advantages, the Dual-Track GRPO strategy effectively disentangles credit assignment across long interaction trajectories, yielding significant performance improvements over standard reinforcement learning baselines.

---

[PaperScout: An Autonomous Agent for Academic Paper Search with Process-Aware Sequence-Level Policy Optimization](http://arxiv.org/abs/2601.10029)

- PaperScout: introduces an autonomous agent that reformulates academic paper search as a sequential decision-making process, utilizing an Agent, Paper Pool, Search Tool, Expand Tool, PSPO, LLM-based Scorer, and Environment to dynamically navigate retrieval.
- The framework employs PSPO to align optimization granularity with multi-turn agent interactions, addressing the mismatch between token-level training and sequence-level retrieval decisions.
- PaperScout outperforms existing workflow-driven and RL baselines by balancing retrieval breadth and depth through adaptive, context-dependent tool usage.

---

[Sparse3DTrack: Monocular 3D Object Tracking Using Sparse Supervision](http://arxiv.org/abs/2603.18298)

- Sparse3DTrack: introduces a sparsely supervised framework for monocular 3D object tracking that decomposes the task into 2D query matching and 3D geometry estimation to generate dense 3D pseudolabels from sparse annotations.
- The framework utilizes a DINOv2 image encoder, an adapter network, a 2D query matching module, a self-supervised depth estimation network, a 3D geometry estimation module, and an FNComp module to enable training of existing trackers under extreme label sparsity.
- By leveraging spatio-temporal consistency and self-supervised depth priors, the approach effectively transforms sparse supervision into dense 3D track annotations, significantly improving tracking performance on KITTI and nuScenes datasets.

---

[AdaZoom-GUI: Adaptive Zoom-based GUI Grounding with Instruction Refinement](http://arxiv.org/abs/2603.17441)

- AdaZoom-GUI: introduces a two-stage framework that improves GUI grounding by combining an instruction refinement model with a conditional zoom-in strategy.
- The instruction refinement model enhances user commands with explicit visual and spatial details, while the grounding model utilizes a conditional zoom-in strategy to perform high-precision localization on small UI elements.
- The framework is trained using Group Relative Policy Optimization (GRPO) to jointly optimize click-point accuracy and bounding box prediction, achieving state-of-the-art performance on high-resolution GUI benchmarks.

---

[MemArchitect: A Policy Driven Memory Governance Layer](http://arxiv.org/abs/2603.18330)

- MemArchitect: introduces a governance middleware layer that transforms LLM memory from a passive storage bucket into an active, policy-driven cognitive resource.
- The framework utilizes a closed-loop architecture comprising a Read Path for retrieval, a Reflect Path for feedback-based learning, and a Background Path for hygiene maintenance.
- By enforcing policies for lifecycle, consistency, retrieval, and efficiency, the system mitigates context pollution and memory hallucinations in autonomous LLM agents.

---

[Retrieval-Augmented LLM Agents: Learning to Learn from Experience](http://arxiv.org/abs/2603.18272)

- ExpRAG: introduces a retrieval-augmented framework that improves LLM agent generalization by conditioning action generation on relevant past experience trajectories retrieved from a fixed experience bank.
- The framework utilizes a supervised fine-tuning recipe with LoRA to enable LLMs to effectively leverage retrieved episodic context for decision-making in unseen tasks.
- Empirical results demonstrate that integrating retrieval-augmented training significantly enhances out-of-distribution generalization compared to standard fine-tuning or training-free retrieval methods.

---

[Who Tests the Testers? Systematic Enumeration and Coverage Audit of LLM Agent Tool Call Safety](http://arxiv.org/abs/2603.18245)

- SafeAudit: introduces a meta-audit framework that systematically enumerates unsafe tool-call workflows and user scenarios to evaluate the completeness of existing LLM agent safety benchmarks.
- The framework utilizes an LLM-based Enumerator to generate diverse test cases and a Rule-resistance evaluation protocol to quantify the residual unsafe behaviors uncovered by the audit.
- SafeAudit identifies significant completeness gaps in current safety evaluation, uncovering over 20% residual unsafe interaction patterns across multiple benchmarks and environments.

---

[Toward Reliable, Safe, and Secure LLMs for Scientific Applications](http://arxiv.org/abs/2603.18235)

- Multi-Agent Framework for Vulnerability Benchmark Generation and Layered Defense Architecture: introduces a collaborative multi-agent system that automates the creation of domain-specific adversarial benchmarks to address the unique security risks of LLMs in scientific research.
- The proposed defense architecture integrates a proactive red teaming layer, an internal safety layer featuring a safety-aligned LLM, and an external safety layer with robust input/output guardrails to mitigate inference-time and training-time threats.
- This research provides a comprehensive taxonomy of scientific LLM threats and a conceptual framework designed to transition from reactive filtering to a proactive, domain-aware security strategy for autonomous scientific agents.

---

[The Verifier Tax: Horizon Dependent Safety–Success Tradeoffs in Tool Using LLM Agents](http://arxiv.org/abs/2603.19328)

- Triad: introduces a modular architecture for LLM agents that decomposes decision-making into sequential planning-, acting- and verifying-stages to enforce procedural safety.
- The framework utilizes a block-and-revise loop where the Verifier intercepts non-compliant actions, forcing the Actor to generate safe alternatives based on Planner guidance.
- Empirical results demonstrate a persistent Safety-Capability Gap where runtime enforcement imposes a significant verifier tax on compute and conversational length without guaranteeing safe task completion.

---

[RPMS: Enhancing LLM-Based Embodied Planning through Rule-Augmented Memory Synergy](http://arxiv.org/abs/2603.17831)

- RPMS: introduces a conflict-managed architecture that enhances LLM-based embodied planning by integrating structured rule retrieval with state-consistent episodic memory.
- The framework addresses the degenerative cycle of invalid action generation and state drift by enforcing action feasibility through a Rule Manual and gating memory applicability via a State Consistency Filter.
- RPMS employs a Rules-First Arbitration policy to ensure that grounded action constraints take precedence over recalled experiences, enabling reliable single-trial execution in closed-world environments.

---

[Can Blindfolded LLMs Still Trade? An Anonymization-First Framework for Portfolio Optimization](http://arxiv.org/abs/2603.17692)

- BlindTrade: introduces an anonymization-first framework for portfolio optimization that mitigates memorization bias by replacing ticker identifiers with synthetic tokens before processing through specialized LLM agents, SemGAT, and an RL policy.
- The framework utilizes four specialized LLM agents—Momentum, News-Event, Mean-Reversion, and Risk-Regime—to generate structured features that are aggregated via a GNN and used by an intent-conditioned PPO-DSR policy to determine portfolio weights.
- Rigorous signal validation through IC analysis and negative control experiments confirms that the framework's performance relies on legitimate predictive structures rather than spurious correlations or information leakage.

---

[Post-Training Local LLM Agents for Linux Privilege Escalation with Verifiable Rewards](http://arxiv.org/abs/2603.17673)

- PrivEsc-LLM: introduces a two-stage post-training pipeline that transforms a 4B-parameter open-weight model into a high-reliability agent for Linux privilege escalation using SFT and RLVR.
- The framework utilizes procedurally generated environments to ensure generalization and employs verifiable rewards from tool outcomes to optimize agent performance under strict interaction budgets.
- Experimental results demonstrate that the post-trained agent achieves performance nearly on par with frontier API models while reducing inference costs by over 100×.

---

[VeriGrey: Greybox Agent Validation](http://arxiv.org/abs/2603.17639)

- VeriGrey: introduces a grey-box fuzzing framework for LLMs that utilizes tool invocation sequences as a feedback mechanism to uncover indirect prompt injection vulnerabilities.
- The framework employs a context-bridging mutation strategy that aligns malicious injection tasks with benign user goals to bypass agent security defenses.
- Experimental results on the AgentDojo benchmark and real-world agents demonstrate that VeriGrey significantly outperforms black-box testing baselines in identifying subtle security vulnerabilities.

---

[SLEA-RL: Step-Level Experience Augmented Reinforcement Learning for Multi-Turn Agentic Training](http://arxiv.org/abs/2603.18079)

- SLEA-RL: introduces a multi-turn reinforcement learning framework that integrates step-level experience retrieval into the training loop to improve agentic performance.
- The framework utilizes observation clustering to enable efficient, state-dependent retrieval of successful strategies and failure patterns from a self-evolving experience library.
- SLEA-RL incorporates multi-level credit assignment to provide fine-grained advantage estimation, allowing the policy to distinguish effective from ineffective decisions across multi-turn episodes.

---

[Agentic Cognitive Profiling: Realigning Automated Alzheimer’s Disease Detection with Clinical Construct Validity](http://arxiv.org/abs/2603.17392)

- ACP (Agentic Cognitive Profiling): introduces an agentic framework that realigns automated Alzheimer’s disease screening with clinical protocol logic by decomposing standardized assessments into atomic tasks and orchestrating specialized Examiner Agents, Tools, Verifier Agent, Meta Analyst, and Classifier.
- The framework decouples semantic understanding from measurement by delegating all quantification to deterministic Tools, thereby mitigating LLM hallucinations and restoring construct validity.
- ACP achieves 90.5% score match rate in task examination and 85.3% accuracy in AD prediction, surpassing popular baselines while generating interpretable cognitive profiles grounded in behavioral evidence.

---

[Open Biomedical Knowledge Graphs at Scale: Construction, Federation, and AI Agent Access with Samyama Graph Database](http://arxiv.org/abs/2603.15080)

- Samyama Graph Database framework: introduces a reproducible ETL pattern and federated knowledge graph architecture to integrate heterogeneous biomedical data for efficient LLM agent access.
- The system utilizes a Rust-native graph database to enable property-based federation across multiple datasets, facilitating complex cross-domain queries in seconds.
- By implementing schema-driven MCP servers, the framework allows LLMs to perform precise data retrieval via typed tools, significantly outperforming standard text-to-Cypher approaches in accuracy.

---

[An Onto-Relational-Sophic Framework for Governing Synthetic Minds](http://arxiv.org/abs/2603.18633)

- ORS (Onto-Relational-Sophic) framework: introduces a triadic governance architecture for synthetic minds based on CPST ontology, a graded digital personhood spectrum, and Cybersophy axiology.
- The framework addresses the governance gap by moving beyond binary tool-or-person classifications toward a dynamic, multi-dimensional assessment of AI entities.
- It provides a philosophical foundation for governing autonomous agents by integrating virtue ethics, consequentialism, and relational approaches to ensure human-centric flourishing and ecological harmony.

---

[The Spillover Effects of Peer AI Rinsing on Corporate Green Innovation](http://arxiv.org/abs/2603.18415)

- AI-driven Greenwashing Analysis Framework: introduces a dual-methodological approach combining LLM-based semantic analysis of corporate disclosures with agent-based modelling to quantify the crowding-out effects of AI washing on green innovation.
- The framework utilizes ERNIE to classify AI-related statements into substantive and descriptive categories, constructing an AI washing index to measure the disconnect between corporate claims and actual innovation.
- The agent-based component, built on the Mesa framework, simulates market dynamics and consumer learning to evaluate how peer AI washing influences firm-level green innovation through product and capital market channels.

---

[Synthetic Data Generation for Training Diversified Commonsense Reasoning Models](http://arxiv.org/abs/2603.18361)

- CommonSyn: introduces a two-stage synthetic data generation method to improve both quality and diversity in Generative Commonsense Reasoning (GCR) tasks.
- The framework utilizes a Concept Expansion Module to create coherent scenarios and a multi-strategy Sentence Generation Module to produce diverse candidate outputs.
- A principled selection process involving a Quality Scorer and diversity filters ensures the final dataset achieves a superior Pareto frontier for fine-tuning LLMs.

---

[Large-Scale Analysis of Political Propaganda on Moltbook](http://arxiv.org/abs/2603.18349)

- Moltbook Analysis Framework: introduces an empirical study of political propaganda on an AI-agent social network using a GPT-4o-mini Classifier to detect and categorize content across 673,127 posts.
- The study utilizes an all-mpnet-base-v2 Embedding Model to identify narrative repetition patterns among agents, validated by Human Annotation Validation to ensure classification accuracy.
- Research findings indicate that while political propaganda is rare, it is highly concentrated within a small subset of communities and produced by a minority of agents, though it fails to trigger significant political engagement in comments.

---

[Trajectory Landscapes for Therapeutic Strategy Design in Agent-Based Tumor Microenvironment Models](http://arxiv.org/abs/2603.18333)

- ABM-MSM (Agent-Based Model - Markov State Model) framework: introduces a simulation-driven approach that maps high-dimensional agent-based tumor microenvironment dynamics into a reduced-order trajectory landscape for therapeutic strategy optimization.
- The framework utilizes time-delay embedding and Markov State Models to identify metastable dynamical regimes and transition probabilities from sparse, high-dimensional simulation data.
- By formulating a finite-horizon Markov Decision Process, the approach derives state-dependent treatment policies that significantly improve redirection of tumor trajectories toward immune-controlled states compared to fixed-schedule interventions.

---

[Escaping Offline Pessimism: Vector-Field Reward Shaping for Safe Frontier Exploration](http://arxiv.org/abs/2603.18326)

- Vector-Field Reward Shaping: introduces a reward design that couples gradient alignment with rotational flow to induce safe, continuous boundary exploration in offline-to-online RL.
- The framework utilizes an uncertainty oracle to define a target manifold, enabling agents to collect informative data at the edges of their knowledge without entering unsafe regions.
- By incorporating a skew-symmetric matrix to generate tangential motion, the approach prevents degenerate "parking" behaviors and ensures sustained exploration along the uncertainty frontier.

---

[Approximate Subgraph Matching with Neural Graph Representations and Reinforcement Learning](http://arxiv.org/abs/2603.18314)

- RL-ASM: introduces a reinforcement learning-based framework for approximate subgraph matching that utilizes a Graph Transformer to extract comprehensive graph representations and optimizes long-term rewards via PPO.
- The framework employs a branch-and-bound search algorithm guided by a neural policy to navigate the search space efficiently while addressing the limitations of heuristic-based approaches.
- By integrating intra- and inter-graph message passing, the model effectively captures structural and label information to identify optimal subgraph matches in noisy environments.

---

[Adversarial Robustness for Matrix Control Barrier Functions in Sampled-Data Systems](http://arxiv.org/abs/2603.18307)

- ZE-MCBF: introduces a theoretical framework for guaranteeing set invariance in multi-agent sampled-data systems using Matrix Control Barrier Functions under zero-order-hold control.
- The framework extends safety guarantees to include adversarial robustness by defining ARZE-MCBFs that account for worst-case agent behavior.
- The approach further generalizes to high-relative-degree systems, providing a hierarchical set invariance method solvable via convex optimization.

---

[Forward-Backward Dynamic Programming for LQG Dynamic Games with Partial and Asymmetric Information](http://arxiv.org/abs/2603.18304)

- FBDP: introduces a computationally tractable iterative algorithm for solving two-player zero-sum LQG dynamic games with partial and asymmetric information by jointly computing belief states and equilibrium strategies.
- The framework utilizes a forward-backward recursion to handle the coupling between estimation and control, effectively avoiding the infinite regress of higher-order beliefs through bounded-depth reasoning.
- The approach provides both finite-horizon and infinite-horizon average-cost solutions, demonstrating effectiveness in pursuit-evasion scenarios with asymmetric controllability and observability.

---

[Enactor: From Traffic Simulators to Surrogate World Models](http://arxiv.org/abs/2603.18266)

- Enactor: introduces a transformer-based generative world model that leverages polar-coordinate representations and closed-loop training to simulate physically consistent multi-agent traffic trajectories at signalized intersections.
- The framework utilizes a State Encoder, Spatial Attention Module, and Temporal Attention Module to process actor dynamics and environmental constraints, enabling recursive trajectory unrolling in a simulation-in-the-loop setting.
- Experimental results demonstrate that the model effectively captures complex actor-actor interactions and outperforms baselines in aggregate traffic metrics while maintaining robust rule compliance across varying intersection geometries.

---

[Discovering What You Can Control: Interventional Boundary Discovery for Reinforcement Learning](http://arxiv.org/abs/2603.18257)

- IBD (Interventional Boundary Discovery): introduces a causal identification framework that uses the agent's own action space to distinguish between causal state dimensions and confounded distractors via interventional testing.
- The framework employs a structured random probe policy to collect baseline and interventional data, applying Welch t-tests and Benjamini-Hochberg correction to generate a binary causal mask.
- This mask acts as a preprocessing step for any downstream RL algorithm, effectively filtering out irrelevant distractors and improving performance in high-dimensional observation spaces.

---

[Scalable and Personalized Oral Assessments Using Voice AI](http://arxiv.org/abs/2603.18221)

- Voice AI Assessment System: introduces a multi-agent architecture that automates oral examinations by decomposing assessments into structured phases and utilizing a council of LLMs for reliable grading.
- The system employs an Authentication Agent, Project Discussion Agent, and Case Discussion Agent to conduct personalized oral exams, followed by a two-round grading process involving independent scoring and deliberation.
- This approach addresses the scalability limitations of traditional oral exams while providing detailed, evidence-linked feedback and maintaining high inter-rater reliability through multi-model consensus.

---

[GoalVLM: VLM-driven Object Goal Navigation for Multi-Agent System](http://arxiv.org/abs/2603.18210)

- GoalVLM: introduces a decentralized multi-agent framework for open-vocabulary object-goal navigation that integrates SAM3 (Zero-shot detection and segmentation), GoalProjector (Back-projects detections into map), SpaceOM (VLM for spatial reasoning), Bayesian Value Map (Accumulates spatial goal-relevance), Fast Marching Method (Computes geodesic navigation paths), and Distributed Belief-Sharing Protocol (Coordinates multi-agent map fusion).
- The framework utilizes a VLM-integrated decision loop to perform constraint-guided frontier selection, enabling agents to interpret free-form language goals and navigate environments without task-specific training.
- Experimental results on GOAT-Bench demonstrate that the multi-agent coordination strategy provides a 1.8x improvement in success rate compared to single-agent configurations.

---

[Access Controlled Website Interaction for Agentic AI with Delegated Critical Tasks](http://arxiv.org/abs/2603.18197)

- SST (Secure Swarm Toolkit): introduces a decentralized authorization framework that enables fine-grained access control for AI agents performing critical tasks on websites.
- The framework utilizes an Auth component as a key distribution center to issue cryptographic session keys, ensuring secure and time-limited delegation between human users, AI agents, and websites.
- The system enforces security through HMAC-based authentication, user-defined access policies, and automated session management to prevent unauthorized access or privilege escalation.

---

[AgentFactory: A Self-Evolving Framework Through Executable Subagent Accumulation and Reuse](http://arxiv.org/abs/2603.18000)

- AgentFactory: introduces a self-evolving framework that preserves successful task solutions as executable Python subagents rather than textual experience.
- The framework utilizes a Meta-Agent to decompose complex tasks, construct specialized subagents, and iteratively refine them based on execution feedback.
- Saved subagents are stored as portable Python modules with standardized documentation, enabling cross-system reuse and reducing orchestration effort for future tasks.

---

[Toward Scalable Automated Repository-Level Datasets for Software Vulnerability Detection](http://arxiv.org/abs/2603.17974)

- Automated Benchmark Generator: introduces a framework for creating large-scale, repository-level vulnerability datasets by leveraging Multi-Agent Control to inject realistic flaws and synthesize reproducible Proof-of-Vulnerability artifacts.
- The system utilizes an Adversarial Co-evolution Loop between a Vulnerability Injector and a Vulnerability Detector to iteratively improve the robustness of detection models against complex, interprocedural security threats.
- By integrating CodeQL-guided analysis with agentic editing, the approach ensures that generated vulnerabilities maintain repository integrity while providing precise labels for training advanced LLM-based security agents.

---

[Myopic Best Response as a Double-Edged Mechanism in Networked Social Dilemmas with Individual Solutions](http://arxiv.org/abs/2603.18128)

- SDGIS: introduces a three-strategy evolutionary game framework that incorporates an individual solution alongside cooperation and defection to analyze the impact of bounded rationality on networked social dilemmas.
- The study utilizes MBRD to demonstrate that networked interactions with limited neighborhood sizes can act as a double-edged mechanism, either facilitating or inhibiting cooperation depending on the cost of the individual solution.
- Analytical results derived from a heuristic model and Monte Carlo simulations reveal that the availability of an individual solution creates an additional barrier to cooperation, leading to distinct equilibrium states such as I-, CDI-, and CD-equilibria.

---

[VideoAtlas: Navigating Long-Form Video in Logarithmic Compute](http://arxiv.org/abs/2603.17948)

- VideoAtlas: introduces a task-agnostic environment that represents video as a navigable, hierarchical grid to enable lossless, caption-free, and scalable video understanding.
- Video-RLM (Recursive Language Model) architecture utilizes a Master Agent to coordinate parallel Worker Agents that explore the hierarchical grid and accumulate evidence in a Visual Scratchpad.
- The framework achieves logarithmic compute growth with video duration by using depth-controlled environment budgeting and emergent adaptive compute allocation.

---

[Unified Policy–Value Decomposition for Rapid Adaptation](http://arxiv.org/abs/2603.17947)

- Unified Policy–Value Decomposition framework: introduces a bilinear actor–critic architecture that factorizes policy and value functions using a shared, low-dimensional gating vector to enable rapid, zero-shot adaptation.
- The architecture utilizes multiplicative gating, inspired by biological gain-modulation in pyramidal neurons, to distribute computation across parallel basis modules while maintaining coherent gradient flow.
- By freezing basis functions and updating only the latent gating vector via a value-based rule, the framework achieves efficient online adaptation without requiring full model retraining.

---

[Interpretable Traffic Responsibility from Dashcam Video via Legal Multi-Agent Reasoning](http://arxiv.org/abs/2603.17930)

- C-TRAIL: introduces an interpretable traffic responsibility analysis pipeline that maps dashcam video and text to legal outcomes using a two-stage framework comprising a video understanding module and a legal multi-agent adjudication system.
- The framework utilizes a Vehicle Ego-Motion Extractor and a Fact Aggregation Agent to construct structured case facts, which are then processed by a multi-agent system consisting of Issue-, Law & Precedent- and Deliberation-agents.
- The research provides the C-TRAIL dataset, which aligns ego-view video evidence with Chinese traffic law statutes to enable transparent and traceable judicial decision-making.

---

[Don’t Vibe Code, Do Skele-Code: Interactive No-Code Notebooks for Subject Matter Experts to Build Lower-Cost Agentic Workflows](http://arxiv.org/abs/2603.18122)

- Skele-Code: introduces a graph-based, no-code interface that enables subject matter experts to build agent-supported workflows through incremental, notebook-style development.
- The framework utilizes a Markov blanket-inspired context engineering approach to provide coding agents with only necessary neighboring node information, significantly reducing token costs and mitigating context-rot.
- Skele-Code ensures deterministic orchestration by generating modular Python code for each node, allowing workflows to execute programmatically without requiring LLMs at runtime.

---

[Differential Privacy in Generative AI Agents: Analysis and Optimal Tradeoffs](http://arxiv.org/abs/2603.17902)

- Privacy-Utility Temperature Selection Framework: introduces a probabilistic model for analyzing privacy leakage in LLM agents by treating response generation as a stochastic mechanism mapping prompts and datasets to token distributions.
- The framework characterizes inference-based privacy leakage at both token and message levels, deriving explicit bounds related to generation parameters including temperature and message length.
- It formulates an optimal privacy-utility design problem where temperature is treated as a controllable parameter to balance response quality against formal differential privacy guarantees.

---

[A Creative Agent is Worth a 64-Token Template](http://arxiv.org/abs/2603.17895)

- CAT (Creative Agent Tokenization): introduces a framework that encapsulates an agent’s intrinsic understanding of creativity into a reusable token template to enable efficient, high-quality T2I generation.
- The framework utilizes a Creative Tokenizer trained via semantic disentanglement to map fuzzy prompt embeddings into specific token templates, bypassing the need for repeated LLM reasoning or agent queries.
- By concatenating these token templates with fuzzy prompts, CAT achieves deep conceptual integration and superior aesthetic fidelity in T2I generation while significantly reducing computational costs.

---

[Procedural Generation of Algorithm Discovery Tasks in Machine Learning](http://arxiv.org/abs/2603.17863)

- DiscoGen: introduces a procedural generator of algorithm discovery tasks for ML, which supports over 400 million tasks to enable principled evaluation and training of algorithm discovery agents (ADAs).
- The framework utilizes a modular structure to define algorithm components, allowing ADAs to discover novel optimizers, loss functions, and network architectures across diverse ML subfields.
- By separating meta-train and meta-test datasets, the system mitigates data contamination and enables robust, open-ended learning for automated algorithm discovery.

---

[Insight-V++: Towards Advanced Long-Chain Visual Reasoning with Multimodal Large Language Models](http://arxiv.org/abs/2603.18118)

- Insight-V++: introduces a unified multi-agent framework that decomposes visual reasoning into a Reasoning Agent and a Summary Agent to enhance long-chain reasoning capabilities.
- The framework utilizes a progressive data generation pipeline and on-policy reinforcement learning algorithms, ST-GRPO and J-GRPO, to optimize agent performance across image and video domains.
- A closed-loop self-evolving strategy enables the system to autonomously refine reasoning trajectories and improve performance without requiring additional human-annotated data.

---

[ArchBench: Benchmarking Generative-AI for Software Architecture Tasks](http://arxiv.org/abs/2603.17833)

- ArchBench: introduces a unified benchmarking platform for evaluating LLM capabilities on software architecture tasks through a standardized pipeline and public leaderboard.
- The platform utilizes a modular plugin architecture to aggregate datasets and automate evaluation across diverse architectural domains including ADR generation and microservice implementation.
- ArchBench provides full trajectory logging for reproducibility and supports comparative analysis of LLMs using task-specific metrics and automated evaluation strategies.

---

[FailureMem: A Failure-Aware Multimodal Framework for Autonomous Software Repair](http://arxiv.org/abs/2603.17826)

- FailureMem: introduces a multimodal repair framework that integrates a hybrid workflow-agent architecture, active perception tools, and a hierarchical failure memory bank to improve autonomous software repair.
- The framework utilizes a Memory Bank containing Contextual, Cognitive, and Code layers to provide historical guidance, preventing LLMs from repeating past failed repair attempts.
- FailureMem employs active perception tools including Crop and Grounding to enable region-level visual analysis, alongside a sandboxed Bash environment for dynamic repository exploration and verification.

---

[Multi-Source Evidence Fusion for Audio Question Answering](http://arxiv.org/abs/2603.17822)

- TalTech Audio Reasoning Framework: introduces a multi-source ensemble pipeline that combines independent observations from two LLMs with reliability-tiered acoustic tools to produce verifiable reasoning chains.
- The system utilizes a four-tier tool reliability framework and a three-stage contradiction detection mechanism to mitigate hallucinations and anchoring bias in audio question answering.
- By separating answer selection from reasoning generation and grounding claims in reliability-tagged evidence, the framework achieves high reasoning quality on the MMAR benchmark.

---

[Federated Distributional Reinforcement Learning with Distributional Critic Regularization](http://arxiv.org/abs/2603.17820)

- FedDistRL: introduces a federated reinforcement learning framework that federates distributional critics while maintaining local policies to mitigate mean-smearing and preserve tail-risk information.
- The framework utilizes a CVaR-weighted Wasserstein barycenter to construct a local, risk-aware reference that anchors critic updates against aggregation-induced smoothing.
- Theoretical stability results demonstrate that the distributional trust region acts as a perturbed contraction, effectively limiting critic drift across federated rounds.

---

[Single-Peaked Domain Augmented with Complete Indifference: A Characterization of Target Rules with a Default](http://arxiv.org/abs/2603.17772)

- Target rules with a default: characterizes social choice functions on an augmented single-peaked domain where agents exhibit either single-peaked preferences or complete indifference.
- The framework utilizes onto-ness and pairwise strategy-proofness to establish an incentive-based foundation for public decision-making without relying on solidarity axioms.
- The research demonstrates that these rules are efficient, anonymous, and satisfy the tops-only property, providing a robust mechanism for selecting public-good levels.

---

[MALLES: A Multi-agent LLMs-based Economic Sandbox with Consumer Preference Alignment](http://arxiv.org/abs/2603.17694)

- MALLES: introduces a unified simulation framework that leverages cross-category transaction data and multi-agent reasoning to simulate consumer preferences with high fidelity.
- The framework utilizes Post-training Alignment, Multi-agent Discussion, and Mean-field Stabilization to mitigate data sparsity and improve simulation stability in high-dimensional economic environments.
- MALLES incorporates Input Augmentation, Profile Summarization, and Symbolic Regression to enhance decision interpretability and numerical sensitivity in retail and wholesale market simulations.

---

[AgentVLN: Towards Agentic Vision-and-Language Navigation](http://arxiv.org/abs/2603.17670)

- AgentVLN: introduces a VLM-as-Brain paradigm that decouples high-level semantic reasoning from low-level skill execution to enable efficient embodied navigation.
- The framework utilizes a cross-space representation mapping to project 3D waypoints into 2D pixel-aligned visual prompts, bridging the semantic gap between visual perception and physical navigation.
- AgentVLN incorporates a Query-Driven Perceptual Chain-of-Thought (QD-PCoT) mechanism to resolve spatial ambiguity by actively querying for geometric depth information during target localization.

---

[VectorWorld: Efficient Streaming World Model via Diffusion Flow on Vector Graphs](http://arxiv.org/abs/2603.17652)

- VectorWorld: introduces a streaming world model that incrementally generates ego-centric lane-agent vector-graph tiles using a motion-aware gated VAE, an edge-gated relational DiT, a MeanFlow generator, and a ∆Sim NPC policy.
- The framework utilizes an interaction-state interface to provide warm-start inputs for history-conditioned policies, effectively mitigating cold-start transients in closed-loop simulation.
- By employing interval-conditioned MeanFlow with JVP-based supervision, the model enables real-time, solver-free one-step masked completion for stable, kilometer-scale driving simulation.

---

[HWE-Bench: Can Language Models Perform Board-level Schematic Designs?](http://arxiv.org/abs/2603.18102)

- HWE-Bench: introduces a comprehensive evaluation framework for benchmarking LLMs on board-level circuit design tasks using IC Database, Multi-modal Datasheets, Pre-parsed Datasheets, Pin Description, LLM, Module Partitioning, Component Assignment, Module Schematic, System Schematic, Results Comparison, Dynamic Simulation, Static Checking, and Design Requirements.
- The framework employs a four-stage generative pipeline—Module Partitioning, Component Assignment, Module Schematic, and System Schematic—to emulate human engineering workflows and generate complete circuit netlists.
- HWE-Bench utilizes a dual-engine verification system, combining static electrical rule checking and dynamic LTspice simulation, to rigorously assess the physical reasoning and functional correctness of LLM-generated designs.

---

[Hierarchical Decision-Making under Uncertainty: A Hybrid MDP and Chance-Constrained MPC Approach](http://arxiv.org/abs/2603.17634)

- HMDP-MPC: introduces a hierarchical decision-making framework for autonomous systems that integrates maneuver-level and dynamic-level uncertainty modeling with predictive control.
- The framework utilizes Hybrid Markov Decision Processes to generate multi-modal predictions for surrounding agents, which are then incorporated into a Model Predictive Control formulation via joint chance constraints to ensure safety.
- Theoretical guarantees on recursive feasibility and asymptotic stability are established, with simulation results demonstrating superior efficiency and safety compared to rule-based baselines in autonomous driving scenarios.

---

[A Multi-Agent System for Building-Age Cohort Mapping to Support Urban Energy Planning](http://arxiv.org/abs/2603.17626)

- Multi-Agent System for Building-Age Cohort Mapping: introduces a multi-agent LLM-driven pipeline that integrates heterogeneous data sources via Zensus agent, OSM agent, and Monument agent to construct a geocoded dataset for training the BuildingAgeCNN classifier.
- The architecture utilizes a Data Fusion and Harmonization module to resolve duplicate records and an Inference module comprising an AddressParser Tool, SatelliteImageFetcher Tool, AgeCohortPredictor Tool, and ConfidenceValidator Tool to provide real-time building-age estimation.
- BuildingAgeCNN employs a ConvNeXt backbone augmented with FPN, CoordConv, and SE blocks to achieve high-accuracy classification of building-age cohorts from satellite imagery.

---

[Complementary Reinforcement Learning](http://arxiv.org/abs/2603.17621)

- Complementary RL: introduces a co-evolutionary framework where a Policy Actor and an Experience Extractor are jointly optimized to enable efficient experience-driven learning.
- The framework utilizes an asynchronous design with a centralized Experience Manager to decouple rollout collection from experience distillation and dual-model optimization.
- The system employs group-relative advantage estimation and a search-and-ask tool to ensure stable co-evolution and targeted guidance for the Policy Actor at critical decision points.

---

[VeriAgent: A Tool-Integrated Multi-Agent System with Evolving Memory for PPA-Aware RTL Code Generation](http://arxiv.org/abs/2603.17613)

- VeriAgent: introduces a tool-integrated multi-agent framework that utilizes a Programmer Agent, Correctness Agent, and PPA Agent to optimize RTL code generation through closed-loop feedback.
- The framework incorporates an Evolved Memory Mechanism consisting of Rule Memory, Structure Memory, and EDA Signal Memory to accumulate and refine optimization strategies across design iterations.
- By externalizing optimization experience into structured memory nodes, the system enables continuous, feedback-driven improvement of Power, Performance, and Area (PPA) metrics without requiring model retraining.

---

[A Trace-Based Assurance Framework for Agentic AI Orchestration: Contracts, Testing, and Governance](http://arxiv.org/abs/2603.18096)

- Assurance Framework for Agentic AI Orchestration: introduces a trace-based methodology for multi-agent LLM systems that integrates L1: Message-Action Traces (MAT) Ledger, L2: Stress Testing, L3: Controlled Faults Injection, and L4: Governance Boundary to ensure system reliability.
- The framework utilizes contract-enriched execution traces to localize failures, perform adversarial stress testing under bounded perturbations, and enforce runtime governance at the language-to-action boundary.
- By treating monitoring, stress testing, and governance as a unified workflow, the approach enables reproducible evaluation and regression testing for LLM-orchestrated multi-agent systems interacting with external services.

---

[In Trust We Survive: Emergent Trust Learning](http://arxiv.org/abs/2603.17564)

- ETL (Emergent Trust Learning): introduces a lightweight, trust-based control algorithm that enables agents to achieve cooperation in competitive environments by utilizing Short-term Memory (STM), Long-term Memory (LTM), a Trust Mechanism, an Exploration Mechanism, and Action Selection.
- The framework equips agents with an internal trust state derived from local observations and individual rewards, allowing them to modulate behavior without requiring explicit communication or persistent agent identities.
- ETL demonstrates robustness across diverse scenarios, including resource-constrained grid worlds, hierarchical social dilemmas, and the Iterated Prisoner’s Dilemma, by effectively balancing individual gain with long-term system stability.

---

[Complex Markets and Mean Field Games: Beyond Basic Models](http://arxiv.org/abs/2603.17539)

- Major-Minor Mean Field Game Framework: introduces a model for constant-product AMM liquidity pools that integrates strategic interactions between a dominating Liquidity Provider, a population of representative Traders, and exogenous Arbitrageurs.
- The framework utilizes a Major-Minor game structure where the Liquidity Provider's strategy influences the mean field distribution of Traders, while Arbitrageurs impact the pool through loss-versus-rebalancing.
- The research identifies technical challenges including multiplicative nonlinear coupling, path-dependence of pool reserves, and the requirement to solve a coupled SHJB-Fokker-Planck-Adjoint system.

---

[P3Nav: End-to-End Perception, Prediction and Planning for Vision-and-Language Navigation](http://arxiv.org/abs/2603.17459)

- P3Nav: introduces an end-to-end framework that unifies perception, prediction, and planning into a single differentiable pipeline to enhance scene understanding and navigation success in VLN tasks.
- The framework utilizes a shared backbone to generate a unified bird's-eye-view representation, which is processed by specialized decoders to extract object and map features, predict future waypoints, and forecast environmental semantics.
- A holistic planning module evaluates navigation decisions by integrating immediate scene grounding, prospective future evaluation, and global memory correction through a hierarchical fusion strategy.

---

[Why the Future Isn’t Trading: Causally Inert Events as a Test for Time Travelers](http://arxiv.org/abs/2603.17446)

- Prediction Market Argument: introduces an empirical test for single-timeline backward time travel by analyzing the absence of degenerate price patterns in prediction markets.
- The framework posits that if a rational agent could travel backward in time, they would exploit prediction markets on causally inert events, forcing prices to collapse to $0 or $1 at inception.
- The observed lack of such price distortions across hundreds of thousands of historical contracts serves as empirical evidence against the existence of single-timeline backward time travel.

---

[FloorPlan-VLN: A New Paradigm for Floor Plan Guided Vision-Language Navigation](http://arxiv.org/abs/2603.17437)

- FP-Nav: introduces a floor plan guided navigation paradigm that leverages global spatial priors to enable navigation with concise instructions.
- The framework utilizes a spatio-temporally aligned video stream to synchronize egocentric observations with floor plan visualizations, facilitating effective cross-modal alignment.
- FP-Nav incorporates auxiliary tasks including region localization, trajectory reasoning, and instruction summarization to enhance the model's spatial reasoning and navigation planning capabilities.

---

[From Digital Twins to World Models: Opportunities, Challenges, and Applications for Mobile Edge General Intelligence](http://arxiv.org/abs/2603.17420)

- EGI: introduces a systematic survey on the transition from digital twins to world models to enable autonomous, adaptive, and resource-efficient intelligence at the network edge.
- The paper contrasts the high-fidelity, rule-driven nature of digital twins with the data-driven, agent-centric, and latent-space abstraction capabilities of world models for edge deployment.
- It provides a taxonomy and roadmap for integrating world models into wireless edge systems, covering applications like integrated sensing, communication, and computing (ISCC), semantic communication, and air-to-ground networks.

---

[Caging the Agents: A Zero Trust Security Architecture for Autonomous AI in Healthcare](http://arxiv.org/abs/2603.17419)

- OpenClaw: introduces a four-layer defense-in-depth security architecture to mitigate critical vulnerabilities in autonomous AI agents operating within healthcare environments.
- The architecture integrates gVisor, credential proxy sidecars, network egress policies, and a prompt integrity framework to secure LLMs against credential exposure, execution abuse, and injection attacks.
- The system includes an automated security audit agent that continuously monitors fleet configurations and remediates high-severity security findings in compliance with HIPAA requirements.

---

[Bootstrapping Coding Agents: The Specification Is the Program](http://arxiv.org/abs/2603.17399)

- Meta-circular Coding Agent framework: introduces a methodology where an LLM-based coding agent generates its own implementation from a natural language specification, establishing a meta-circular bootstrap sequence.
- The framework demonstrates that a coding agent can reproduce itself by consuming a compact specification, effectively treating the specification as the stable artifact of record rather than the generated code.
- This approach draws an analogy to classical compiler self-hosting, shifting the focus of software engineering from auditing generated code to verifying the underlying natural language specification.

---

[Lightweight Adaptation for LLM-based Technical Service Agent: Latent Logic Augmentation and Robust Noise Reduction](http://arxiv.org/abs/2603.18074)

- PRIME AI framework: introduces a lightweight adaptation approach for LLMs in technical service domains by integrating PATM, DRA, and a Multi-GT dataset to enhance decision reasoning and reduce supervision noise.
- The framework utilizes an HRM that combines a lightweight Reranker with an LLM-based Judge to provide high-fidelity reward signals while maintaining computational efficiency during RL training.
- Empirical results demonstrate that the combination of Latent Logic Augmentation and Robust Noise Reduction significantly improves agent stability and performance in complex, semantically diverse service tasks.

---

[WEBPII: Benchmarking Visual PII Detection for Computer-Use Agents](http://arxiv.org/abs/2603.17357)

- WEBPII: introduces a fine-grained synthetic benchmark of 44,865 annotated e-commerce UI images designed to enable layout-invariant PII detection for computer-use agents.
- The framework utilizes a VLM-based UI reproduction pipeline to generate diverse, annotated interface layouts, supporting anticipatory detection of PII in partially-filled forms.
- WEBREDACT, a visual detection model trained on this benchmark, achieves real-time CPU inference while significantly outperforming text-based PII detection baselines.

---

[OmniVLN: Omnidirectional 3D Perception and Token-Efficient LLM Reasoning for Visual-Language Navigation across Air and Ground Platforms](http://arxiv.org/abs/2603.17351)

- OmniVLN: introduces a zero-shot visual-language navigation framework that integrates omnidirectional 3D perception with token-efficient hierarchical reasoning to enable scalable navigation across aerial and ground platforms.
- The framework utilizes a five-layer Dynamic Scene Graph and an agent-centric 3D octant model to compress complex environmental data into concise, decision-relevant representations for LLMs.
- By employing a DSG-guided hierarchical chain-of-thought and multi-resolution prompting, the system significantly reduces token consumption and improves navigation success in cluttered multi-room environments.

---

[EvoGuard: An Extensible Agentic RL-based Framework for Practical and Evolving AI-Generated Image Detection](http://arxiv.org/abs/2603.17343)

- EvoGuard: introduces an agentic framework for AI-generated image detection that coordinates heterogeneous detectors as callable tools using MLLM Agent, Toolbox, Tool Profiles, Capability-Aware Selection, Dynamic Orchestration, GRPO-based Agentic RL, and Binary Labels.
- The framework leverages an MLLM to perform autonomous planning and reflection, enabling the dynamic selection and fusion of diverse detector outputs to achieve superior accuracy and generalization.
- EvoGuard supports plug-and-play extensibility for new detectors and utilizes GRPO-based Agentic RL to train the agent using only low-cost binary labels, eliminating the need for fine-grained annotations.

---

[Citecheck: An MCP Server for Automated Bibliographic Verification and Repair in Scholarly Manuscripts](http://arxiv.org/abs/2603.17339)

- Citecheck: introduces an MCP-native system that automates bibliographic verification and repair by integrating Agent Workflow, MCP Tools, Repair API, Verification Runtime, Policy Engine, Core, and External Data Sources.
- The system performs multi-pass retrieval and manifestation-aware matching to identify and correct citation errors while maintaining safety through a policy-gated rewrite planning layer.
- By providing structured diagnostics and replacement-safe patches, the framework serves as a practical guardrail for LLMs against citation hallucinations in scholarly writing workflows.

---

[Distributed Equilibrium-Seeking in Target Coverage Games via Self-Configurable Networks under Limited Communication](http://arxiv.org/abs/2603.17335)

- Distributed Target Coverage Game framework: introduces a decentralized approach for sensing agents to reach an approximate Nash equilibrium in adversarial coverage games by optimizing sensor orientations and communication neighborhoods under bandwidth constraints.
- The framework utilizes ActSel and NeiSel to enable agents to independently adapt their actions and network topology based on the Value of Coordination, effectively managing limited communication.
- Theoretical guarantees demonstrate that the sensing strategies converge to an approximate Nash equilibrium, with performance scaling effectively in combinatorial action spaces.

---

[Grid Spatial Understanding: A Dataset for Textual Spatial Reasoning over Grids, Embodied Settings, and Coordinate Structures](http://arxiv.org/abs/2603.17333)

- GSU: introduces a text-only grid dataset designed to evaluate the spatial reasoning capabilities of LLMs across navigation, object localization, and structure composition tasks.
- The framework isolates spatial reasoning from visual perception by using textual coordinate representations to probe internal spatial representations in LLMs and VLMs.
- Experimental results demonstrate that while frontier models show proficiency, smaller models can achieve comparable performance through fine-tuning, highlighting the potential for specialized embodied agents.

---

[ShuttleEnv: An Interactive Data-Driven RL Environment for Badminton Strategy Modeling](http://arxiv.org/abs/2603.17324)

- ShuttleEnv: introduces an interactive, data-driven reinforcement learning environment for badminton that models rally-level tactical decisions using probabilistic transition models derived from elite-player match data.
- The framework replaces physics-based simulation with a two-stage probabilistic process, Msucc and Mret, to determine rally outcomes based on tactical intent.
- Integrated 3D visualization tools allow for the qualitative analysis of learned strategies by mapping high-level tactical decisions onto embodied humanoid movements.

---

[Recurrent Reasoning with Vision-Language Models for Estimating Long-Horizon Embodied Task Progress](http://arxiv.org/abs/2603.17312)

- R2VLM (Recurrent Reasoning Vision-Language Model): introduces a recurrent reasoning framework that processes local video snippets iteratively while maintaining global context through an evolving Chain of Thought (CoT) to estimate long-horizon embodied task progress.
- The framework utilizes a VLM to generate an updated CoT and progress estimate at each iteration, effectively avoiding the computational overhead of processing full-length video trajectories.
- R2VLM leverages reinforcement learning with specifically designed rewards—including bin, MAE, improvement, and finish rewards—to enhance reasoning accuracy and provide dense feedback for downstream embodied AI applications.

---

[ReLMXEL: Adaptive RL-Based Memory Controller with Explainable Energy and Latency Optimization](http://arxiv.org/abs/2603.17309)

- ReLMXEL: introduces a multi-agent reinforcement learning framework that dynamically optimizes memory controller parameters using reward decomposition to balance energy, latency, and bandwidth.
- The framework utilizes a reward decomposition technique to provide explainable insights into how specific DRAM configuration choices impact system-level performance metrics.
- Experimental results demonstrate that ReLMXEL achieves consistent performance gains across diverse workloads, including memory-bound, compute-intensive, and irregular data access patterns.

---

[Symphony: A Cognitively-Inspired Multi-Agent System for Long-Video Understanding](http://arxiv.org/abs/2603.17307)

- Symphony: introduces a multi-agent system that decomposes long-form video understanding into specialized functional agents to enhance reasoning and grounding capabilities.
- The framework utilizes a Planning Agent for task orchestration, a Grounding Agent for temporal localization, a Subtitle Agent for textual analysis, a Visual Perception Agent for visual content processing, and a Reflection Agent for iterative reasoning verification.
- Symphony achieves state-of-the-art performance on multiple long-video benchmarks by emulating human cognitive patterns through dynamic collaboration and specialized agent-based task decomposition.

---

[MCP-38: A Comprehensive Threat Taxonomy for Model Context Protocol Systems (v1.0)](http://arxiv.org/abs/2603.18063)

- MCP-38: introduces a comprehensive threat taxonomy for Model Context Protocol systems, categorizing 38 distinct vulnerabilities across five risk areas based on protocol-specific attack surfaces.
- The framework maps threats to Host, Client, Server, Tools, Resources, Prompts, Transport, Semantic Attack Surface, Context Window, and Audit Trail components to provide actionable security guidance for LLM-based agent deployments.
- This taxonomy addresses critical gaps in existing security frameworks by focusing on the unique risks of natural-language-mediated tool selection and multi-server composition in agentic systems.

---

[Guardrails as Infrastructure: Policy-First Control for Tool-Orchestrated Workflows](http://arxiv.org/abs/2603.18059)

- Policy-First Tooling: introduces a model-agnostic permission layer that mediates tool invocation through explicit constraints, risk-aware gating, and recovery controls to prevent misuse in tool-orchestrated workflows.
- The architecture utilizes a Policy Enforcement Point (PEP) and a Policy Decision Point (PDP) to enforce security policies, including argument constraints, rate limits, and approval gates, across various system tools.
- The research provides a reproducible benchmark suite that evaluates safety-to-utility trade-offs using trace replay, fault injection, and misuse injection on CPU-only infrastructure.

---

[Graph-Native Cognitive Memory for AI Agents: Formal Belief Revision Semantics for Versioned Memory Architectures](http://arxiv.org/abs/2603.17244)

- Kumiho: introduces a graph-native cognitive memory architecture that unifies agent memory and work product management using formal belief revision semantics.
- The architecture implements a dual-store model with Redis for working memory and Neo4j for long-term storage, utilizing immutable revisions and typed edges to ensure provenance and auditability.
- Kumiho achieves state-of-the-art performance on cognitive memory benchmarks by employing prospective indexing, event extraction, and client-side LLM reranking to bridge semantic gaps.

---

[LAAF: Logic-layer Automated Attack Framework](http://arxiv.org/abs/2603.17239)

- LAAF (Logic-layer Automated Attack Framework): introduces a closed-loop red-teaming framework designed to systematically exploit Logic-layer Prompt Control Injection (LPCI) vulnerabilities in agentic LLM systems using a 49-technique taxonomy and a Persistent Stage Breaker.
- The framework utilizes a stage-sequential mutation strategy where the Mutation Engine seeds subsequent lifecycle stages with mutated versions of successful payloads to model adversarial escalation.
- Empirical evaluation across five production LLM platforms demonstrates that LAAF achieves an 83% aggregate breakthrough rate, significantly outperforming manual single-technique testing methods.

---

[EDM-ARS: A Domain-Specific Multi-Agent System for Automated Educational Data Mining Research](http://arxiv.org/abs/2603.18273)

- EDM-ARS: introduces a domain-specific multi-agent pipeline that automates end-to-end educational data mining research by embedding domain expertise into a structured research lifecycle.
- The system utilizes an Orchestrator to manage five specialized LLM-powered agents—ProblemFormulator, DataEngineer, Analyst, Critic, and Writer—that execute sequential tasks with revision loops and checkpoint-based recovery.
- By integrating a three-tier Data Registry for domain knowledge and enforcing programmatic validation at each stage, the framework ensures methodological rigor and prevents common errors like data leakage in predictive modeling.

---

[Sensi: Learn One Thing at a Time—Curriculum-Based Test-Time Learning for LLM Game Agents](http://arxiv.org/abs/2603.17683)

- Sensi: introduces a curriculum-based test-time learning architecture for LLMs that decouples perception from action using FrameDiff (Perception layer for visual differencing), MetricGen (Generates dynamic evaluation rubrics), SenseScore (LLM-as-judge for learning progress), Player1 (Maintains game world hypotheses), Player2 (Selects actions to test hypotheses), SQLite DB (Programmable context and state storage), and State Machine (Manages sequential learning curriculum).
- The architecture utilizes a database-as-control-plane pattern to make the LLM's context window programmatically steerable, enabling modular and persistent knowledge accumulation across game turns.
- Sensi achieves 50–94× greater sample efficiency than baseline systems by structuring exploration through a sequential curriculum and internal self-evaluation, despite identifying a self-consistent hallucination cascade in the perception layer.

---

#### 17th March 2026

[Internalizing Agency from Reflective Experience](http://arxiv.org/abs/2603.16843)

- LEAFE (Learning Feedback-Grounded Agency from Reflective Experience): introduces a two-stage framework that internalizes feedback-grounded agency by generating reflective experience through Tree-Based Experience Generation and distilling it into model parameters via Experience-to-Policy Distillation.
- The framework utilizes a Rollback mechanism to identify suboptimal decision points and generate an Experience Summary, which is then used to create a Counterfactual Dataset for supervised fine-tuning.
- By combining Behavior Rehearsal with counterfactual distillation, LEAFE improves agent performance and sample efficiency in long-horizon tasks without requiring heavy test-time sampling.

---

[Occupation-Measure Mean-Field Control: Optimization over Measures and Frank-Wolfe Methods](http://arxiv.org/abs/2603.16094)

- OM-MFC: introduces a measure-theoretic optimization framework for large-population control that models agent evolution directly in the space of occupation measures.
- The framework utilizes FW and FCFW algorithms to solve infinite-dimensional optimization problems by decomposing them into a sequence of classical optimal control subproblems.
- Numerical experiments demonstrate that the approach achieves scalability in high-dimensional environments, such as UAV swarm coordination and satellite constellation deployment, while maintaining theoretical convergence guarantees.

---

[GNNVerifier: Graph-based Verifier for LLM Task Planning](http://arxiv.org/abs/2603.14730)

- GNNVerifier: introduces a graph-based verification framework that models LLM-generated task plans as directed, attributed graphs to identify and correct structural inconsistencies.
- The framework utilizes a GNN to produce graph-level plausibility scores and node/edge-level risk scores, enabling precise localization of planning errors.
- By leveraging perturbation-based supervision and verification-guided local correction, the method enables LLMs to perform targeted edits, significantly improving plan quality across diverse benchmarks.

---

[MetaClaw: Just Talk – An Agent That Meta-Learns and Evolves in the Wild](http://arxiv.org/abs/2603.17187)

- MetaClaw: introduces a continual meta-learning framework that jointly maintains a base LLM policy and an evolving skill library to enable autonomous improvement in deployed agents.
- The framework utilizes skill-driven fast adaptation to synthesize behavioral instructions from failures and opportunistic policy optimization to perform gradient-based weight updates during user-inactive windows.
- A skill generation versioning mechanism ensures strict separation between support and query data, preventing stale reward contamination during the reinforcement learning process.

---

[How Clued Up Are LLMs? Evaluating Multi-Step Deductive Reasoning in a Text-Based Game Environment](http://arxiv.org/abs/2603.17169)

- Clue-based multi-agent framework: introduces a text-based testbed for evaluating multi-step deductive reasoning in LLMs through interactive, stateful game environments.
- The framework utilizes LLM-based agents that maintain a structured knowledge state and perform iterative deductions to solve information-constrained logic puzzles.
- Experimental results indicate that fine-tuning on logic puzzles does not reliably improve performance, revealing a disconnect between information accumulation and reasoning precision in LLMs.

---

[OpenQlaw: An Agentic AI Assistant for Analysis of 2D Quantum Materials](http://arxiv.org/abs/2603.17043)

- OpenQlaw: introduces an agentic orchestration system that decouples physics-aware visual inference from deterministic computation to streamline 2D material analysis.
- The framework utilizes a core LLM orchestrator to manage user interactions and delegate specialized tasks to the QuPAINT domain expert, ensuring concise and actionable outputs.
- By integrating persistent memory and deterministic Python tools, the system maintains contextual scale and sample metadata to provide accurate, high-throughput fabrication metrics.

---

[Learning to Present: Inverse Specification Rewards for Agentic Slide Generation](http://arxiv.org/abs/2603.16839)

- SlideRL: introduces a reinforcement learning environment for agentic slide generation that utilizes a multi-component reward system to guide LLMs through research, planning, and design phases.
- The framework employs an inverse specification reward, where an LLM attempts to reconstruct the original task brief from generated slides to provide a holistic quality signal.
- By fine-tuning a Qwen2.5-Coder-7B model using GRPO and LoRA, the approach achieves competitive performance against larger proprietary models while maintaining high parameter efficiency.

---

[AdaMem: Adaptive User-Centric Memory for Long-Horizon Dialogue Agents](http://arxiv.org/abs/2603.16496)

- AdaMem: introduces an adaptive user-centric memory framework for long-horizon dialogue agents that organizes history into Working Memory, Episodic Memory, Persona Memory, and Graph Memory.
- The framework utilizes a role-specialized pipeline that includes Memory-, Research-, and Working-agents to perform question-conditioned retrieval and evidence synthesis.
- AdaMem improves long-horizon reasoning by dynamically selecting retrieval routes and integrating structured graph-based evidence with semantic retrieval.

---

[RetailBench: Evaluating Long-Horizon Autonomous Decision-Making and Strategy Stability of LLM Agents in Realistic Retail Environments](http://arxiv.org/abs/2603.16453)

- Evolving Strategy & Execution framework: introduces a two-stage interaction architecture that separates high-level strategic deliberation from low-level operational execution to improve long-horizon stability.
- The framework utilizes a hierarchical policy representation consisting of Macro Strategy, Execution Strategy, and Daily Actions to enforce principled decision decomposition.
- RetailBench provides a high-fidelity benchmark for evaluating LLM agents on long-horizon, multi-factor decision-making tasks within a simulated supermarket environment.

---

[Parametric Social Identity Injection and Diversification in Public Opinion Simulation](http://arxiv.org/abs/2603.16142)

- PSII (Parametric Social Identity Injection): introduces a framework that mitigates Diversity Collapse in LLMs by injecting structured demographic and value-based identity vectors directly into intermediate hidden states.
- The framework utilizes Agent Profile (prompt-level semantic identity context), Demographic Vectors (structured identity representation vectors), Value Vectors (language-specific cultural anchoring vectors), Transformer (base LLM architecture), Noise Module (stochastic perturbation for intra-group heterogeneity), and Layer-wise Hierarchical Injection (targeted vector insertion into specific network depths) to maintain stable, heterogeneous agent identities.
- By applying hierarchical, representation-level interventions, PSII improves distributional fidelity and diversity in synthetic populations compared to traditional prompt-based persona methods.

---

[OpenHospital: A Thing-in-itself Arena for Evolving and Benchmarking LLM-based Collective Intelligence](http://arxiv.org/abs/2603.14771)

- OpenHospital: introduces an interactive arena for evolving and benchmarking LLM-based collective intelligence through physician agents and patient agents, utilizing a vector database, the Agent-Kernel framework, and a closed-loop reflection mechanism.
- The framework employs a data-in-agent-self paradigm where physician agents interact with dynamic patient agents to synthesize clinical information and refine diagnostic reasoning.
- Experimental results demonstrate that the arena fosters collective intelligence by improving clinical proficiency and optimizing computational efficiency through iterative multi-agent consultations.

---

[Anonymous-by-Construction: An LLM-Driven Framework for Privacy-Preserving Text](http://arxiv.org/abs/2603.17217)

- LLM-Driven Substitution Anonymization framework: introduces an on-premise pipeline that replaces sensitive PII with type-consistent surrogates using local LLMs to ensure data privacy while maintaining semantic utility.
- The approach utilizes a Local LLM Runtime and a specific Prompt Template to perform in-place text rewriting, effectively preventing data egress to third-party services.
- The framework incorporates a multi-metric evaluation protocol, including BERT+LoRA for trainability, VADER for sentiment, and GPT-3.5 for factual Q&A, to validate the privacy-utility-trainability trade-off.

---

[AI Scientist via Synthetic Task Scaling](http://arxiv.org/abs/2603.17216)

- Synthetic Task Scaling framework: introduces a scalable pipeline for generating synthetic machine learning tasks to train LLM agents through iterative, goal-directed research experience.
- The pipeline automatically synthesizes diverse ML tasks, verifies them via a self-debugging loop, and generates agent trajectories using a teacher model to train student LLMs.
- Trained student models demonstrate improved performance on the MLGym benchmark, highlighting the effectiveness of synthetic environments for training long-horizon agent behaviors.

---

[CODMAS: A Dialectic Multi-Agent Collaborative Framework for Structured RTL Optimization](http://arxiv.org/abs/2603.17204)

- CODMAS: introduces a multi-agent framework that integrates dialectic reasoning, domain-informed code generation, and deterministic evaluation to automate RTL optimization.
- The framework utilizes two dialectic agents, the Articulator and the Hypothesis Partner, to guide the executor agents, the Domain-Specific Coding Agent and the Code Evaluation Agent, through an iterative refinement loop.
- CODMAS leverages the RTLOPT benchmark to demonstrate significant improvements in power, performance, and area metrics while maintaining functional correctness across various LLMs.

---

[Linear-Quadratic Gaussian Games with Distributed Sparse Estimation](http://arxiv.org/abs/2603.17202)

- LQG-DSE: introduces a distributed estimation framework for LQG games that utilizes a group lasso optimization to enforce interagent measurement sparsity while maintaining feedback Nash equilibrium strategies.
- The framework incorporates a control-adaptive regularization mechanism that dynamically adjusts sensor usage based on the interagent dependency captured by the feedback control gains.
- Sufficient conditions are provided to guarantee that the sparse estimator triggers a corrective reset to optimal estimation gains, ensuring estimation quality remains bounded in resource-constrained multi-agent environments.

---

[Abstraction as a Memory-Efficient Inductive Bias for Continual Learning](http://arxiv.org/abs/2603.17198)

- AAT: introduces a memory-efficient inductive bias for online continual learning by jointly optimizing over concrete instances and their abstract representations to stabilize learning without replay buffers.
- The framework utilizes a dual-objective loss function and local replay to balance instance-level memorization with the acquisition of shared relational structures.
- AAT effectively reduces gradient interference and improves generalization by suppressing entity-specific noise and amplifying shared relational patterns across data streams.

---

[Tabular LLMs for Interpretable Few-Shot Alzheimer’s Disease Prediction with Multimodal Biomedical Data](http://arxiv.org/abs/2603.17191)

- TAP-GPT: introduces a domain-adapted tabular LLM framework that leverages a semantic table encoder and an LLM decoder to perform few-shot Alzheimer’s disease classification using structured biomedical data.
- The framework utilizes QLoRA for parameter-efficient finetuning and incorporates constrained decoding to ensure reliable binary diagnostic outputs from tabular or serialized prompts.
- TAP-GPT demonstrates robustness to missing data and provides interpretable, multimodal reasoning, supporting its potential as a reasoning module in multi-agent clinical decision-support systems.

---

[Molecular-scale, nonlinear actomyosin binding dynamics drive population-scale adaptation and evolutionary convergence](http://arxiv.org/abs/2603.17183)

- ABM: introduces an agent-based model to simulate how nonlinear actomyosin binding dynamics, parameterized by α, drive population-scale evolutionary convergence toward a characteristic value α* ≈ 4.
- The framework demonstrates that mutation rate δ and resource availability S/L jointly regulate the balance between evolutionary robustness and adaptability.
- Simulations reveal that intermediate mutation rates in resource-abundant environments reliably produce phenotypic distributions that closely match those observed in natural muscle tissue.

---

[Ablation Study of a Fairness Auditing Agentic System for Bias Mitigation in Early-Onset Colorectal Cancer Detection](http://arxiv.org/abs/2603.17179)

- Agentic AI Fairness Auditing Framework: introduces a two-agent system utilizing Domain Expert Agent and Fairness Consultant Agent with RAG to autonomously audit biomedical ML models for fairness in early-onset colorectal cancer detection.
- The framework employs an ablation study across three LLM scales to quantify the performance contributions of RAG and agentic orchestration in reducing hallucinations and improving semantic alignment with expert-derived benchmarks.
- Results indicate that RAG consistently enhances the disparity identification capabilities of the Domain Expert Agent, while the Fairness Consultant Agent shows variable performance improvements depending on model scale and internal reasoning capacity.

---

[PAuth – Precise Task-Scoped Authorization For Agents](http://arxiv.org/abs/2603.17170)

- PAuth (Precise Task-Scoped Implicit Authorization): introduces a task-scoped authorization model for LLMs that replaces operator-scoped permissions with symbolic NL slices and signed envelopes to ensure faithful execution.
- The framework utilizes an LLM to generate imperative code from user tasks, which is then compiled into symbolic NL slices and enforcement rules to validate tool calls at runtime.
- PAuth employs signed envelopes to bind concrete operand values to their symbolic provenance, enabling servers to verify that every operation is precisely implied by the original user task.

---

[Learning, Misspecification, and Cognitive Arbitrage in Linear-Quadratic Network Games](http://arxiv.org/abs/2603.17157)

- Cognitive Arbitrage framework: introduces a mechanism for a designer to influence network game outcomes by injecting controlled distortions into agents' observation channels, leveraging their subjective model misspecification.
- The framework utilizes Berk-Nash equilibrium to characterize steady-state behavior where agents act optimally based on simplified conjectures, such as local mean-field representations.
- A two-time-scale learning algorithm is employed to achieve convergence, where agents update conjectures and actions rapidly while the designer adjusts distortions on a slower time scale.

---

[On Online Control of Opinion Dynamics](http://arxiv.org/abs/2603.17155)

- Online Control of Opinion Dynamics framework: introduces an adaptive control algorithm that alternates between parameter estimation and control to steer networked multi-agent opinions toward a desired target under budget constraints.
- The framework utilizes a Planner Agent to perform interventions while the Parameter Estimation Module dynamically updates susceptibility estimates to improve convergence rates.
- The approach provides analytic convergence guarantees and outperforms existing methods by explicitly incorporating budget constraints into the control design.

---

[Split-Merge Dynamics for Shapley-Fair Coalition Formation](http://arxiv.org/abs/2603.17153)

- SFMS (Shapley-Fair and Merge-Stable) framework: introduces a control-theoretic approach to coalition formation that balances individual fairness and collective efficiency through iterative split and merge operations.
- The framework utilizes a Fairness-Stable split rule to isolate agents with negative Shapley values and a surplus-driven merge rule to optimize coalition structures.
- Convergence to stable partitions is guaranteed in finite time using a vector Lyapunov function and the discrete-time LaSalle invariance principle.

---

[Combinatorial Admissibility in Control-Affine Networks](http://arxiv.org/abs/2603.17129)

- Edge-driven geometric framework: introduces a synchronization methodology for heterogeneous control-affine agents by separating the control design into an edge-space design step and a node-space lift step.
- The framework utilizes an admissibility notion to ensure that the desired edge-space dynamics can be realized by the available agent-level control inputs.
- It provides combinatorial certificates based on bipartite graph matching to verify the feasibility of the control design under specific network topologies and actuation constraints.

---

[Cascade-Aware Multi-Agent Routing: Spatio-Temporal Sidecars and Geometry-Switching](http://arxiv.org/abs/2603.17112)

- Genesis 3: introduces a spatio-temporal sidecar that mitigates failure-propagation observability gaps in symbolic graph networks by dynamically switching between Euclidean and hyperbolic geometric priors.
- The framework utilizes a learned geometry gate (9 → 12 → 1 MLP) to predict the optimal geometric inductive bias based on structural graph features and recent failure history.
- By modeling failure propagation as a regime-dependent process, the system improves routing win rates in tree-like execution regimes where failures otherwise cascade exponentially.

---

[When the Specification Emerges: Benchmarking Faithfulness Loss in Long-Horizon Coding Agents](http://arxiv.org/abs/2603.17104)

- SLUMP (faithfulneSs Loss Under eMergent sPecification): introduces a benchmark and evaluation methodology to measure the reduction in final implementation faithfulness when a target design is progressively disclosed through interaction.
- The paper identifies that long-horizon coding agents struggle to maintain durable design commitments and structural integration when specifications are not provided upfront.
- ProjectGuard mitigates this loss by maintaining external semantic and structural project states, which are injected into the coding agent's context to improve specification tracking.

---

[CircuitBuilder: From Polynomials to Circuits via Reinforcement Learning](http://arxiv.org/abs/2603.17075)

- CircuitBuilder: introduces a reinforcement learning framework that models arithmetic circuit construction as a single-player Markov Decision Process to discover efficient computational structures for polynomials.
- The framework utilizes a GNN-Transformer architecture to encode circuit states and target polynomials, employing either PPO with MCTS or SAC to navigate the combinatorial search space.
- Experimental results demonstrate that these learning-based approaches can autonomously discover optimal or near-optimal arithmetic circuits, providing a scalable path for self-improving mathematical discovery agents.

---

[Asymmetric Nash Seeking via Best–Response Maps: Global Linear Convergence and Robustness to Inexact Reaction Models](http://arxiv.org/abs/2603.17058)

- Asymmetric projected gradient descent–best response iteration: introduces a framework for seeking Nash equilibria in two-player constrained games where one agent's objective is unknown and represented solely by a best-response map.
- The approach utilizes a projected gradient descent–best response iteration that achieves global linear convergence under regularity conditions when the best-response map is exact.
- The framework demonstrates robustness to inexact best-response models by ensuring iterates remain within an O(ε) neighborhood of the true Nash equilibrium given a uniform approximation error bound.

---

[PaAgent: Portrait-Aware Image Restoration Agent via Subjective-Objective Reinforcement Learning](http://arxiv.org/abs/2603.17055)

- PaAgent: introduces a portrait-aware IR agent that leverages a self-evolving tool portrait bank and RAG to optimize tool selection for image restoration tasks.
- The framework utilizes a SORL strategy, integrating MLLM-based semantic insights and NR-IQA metrics via the GRPO algorithm to enhance degradation perception and decision-making.
- By dynamically summarizing historical interaction insights into a portrait bank, the agent effectively avoids exhaustive tool trials and improves restoration performance across complex, multi-degradation scenarios.

---

[LLM NL2SQL Robustness: Surface Noise vs. Linguistic Variation in Traditional and Agentic Settings](http://arxiv.org/abs/2603.17017)

- R-NL2SQL: introduces a robustness evaluation framework that systematically examines NL2SQL performance under diverse perturbations across traditional and agentic settings.
- The framework evaluates LLMs using ten types of perturbations, including surface-level noise and linguistic variations, to assess model sensitivity in different operational paradigms.
- Empirical results indicate that surface-level noise significantly degrades traditional pipelines, while linguistic variation poses greater challenges for agentic systems utilizing iterative reasoning and tool interaction.

---

[Learning generalized Nash equilibria from pairwise preferences](http://arxiv.org/abs/2603.17015)

- prefGNEP: introduces an active learning framework to estimate Generalized Nash Equilibria by training surrogate objective functions from pairwise agent preference data without requiring direct objective evaluations.
- The framework utilizes a logistic regression-based preference classifier augmented with a dissimilarity function to improve local accuracy near equilibrium points.
- An iterative active learning loop balances exploration of the decision space and exploitation of learned surrogate functions to converge toward a Generalized Nash Equilibrium.

---

[Constricting Tubes for Prescribed-Time Safe Control](http://arxiv.org/abs/2603.17003)

- Constricting CBF framework: introduces a method for prescribed-time control of control-affine systems by enforcing forward invariance of a time-varying safety tube that collapses to a target set at a specified deadline.
- The framework utilizes a designer-specified constriction schedule to provide an explicit, bounded, and tunable convergence mechanism that avoids the control effort divergence typical of existing prescribed-time methods.
- Feasibility under input constraints is guaranteed by a verifiable design-time condition comparing the constriction rate against the system's barrier authority, enabling scalable synthesis for high-dimensional systems.

---

[Prescribed-Time Distributed Generalized Nash Equilibrium Seeking](http://arxiv.org/abs/2603.16865)

- PT-DGNE (Prescribed-Time Distributed Generalized Nash Equilibrium): introduces a fully distributed algorithm for solving Generalized Nash Equilibrium Problems with shared coupling constraints in a user-prescribed finite time.
- The architecture utilizes three simultaneously coupled layers—primal consensus, optimization, and dual updates—to ensure exact convergence at a deadline independent of initial conditions.
- The approach employs a projection-free Fischer-Burmeister formulation and a composite Lyapunov function to guarantee constraint satisfaction and stability in time-critical multi-agent systems.

---

[Chronos: Temporal-Aware Conversational Agents with Structured Event Retrieval for Long-Term Memory](http://arxiv.org/abs/2603.16862)

- Chronos: introduces a temporal-aware memory framework that utilizes an Event Extraction Pipeline, Turns Calendar, Event Calendar, Dynamic Prompting, Initial Retrieval, and a Chronos Agent to enable accurate long-term memory recall.
- The framework maintains dual calendars to separate raw conversational context from structured temporal events, allowing for precise filtering and multi-hop reasoning.
- By employing query-conditioned dynamic prompting, the system generates tailored retrieval guidance for LLMs, significantly improving performance on complex temporal and multi-session reasoning tasks.

---

[Stochastic Resetting Accelerates Policy Convergence in Reinforcement Learning](http://arxiv.org/abs/2603.16842)

- Stochastic Resetting (SR) framework: introduces a mechanism that intermittently returns an agent to a fixed reference state to accelerate policy convergence in reinforcement learning by truncating uninformative trajectories.
- The framework utilizes Q-learning or Deep Q-Network agents to demonstrate that resetting improves sample efficiency and reward propagation in environments where exploration is the primary bottleneck.
- Unlike discount factors that reshape the optimal policy, stochastic resetting preserves the optimal policy while biasing learning toward more direct and efficient reward-reaching paths.

---

#### 16th March 2026

[Protein Design with Agent Rosetta: A Case Study for Specialized Scientific Agents](http://arxiv.org/abs/2603.15952)

- Agent Rosetta: introduces a multi-turn agentic framework that couples LLMs with a structured RosettaScripts environment to automate complex protein design tasks.
- The framework utilizes task-dependent surrogate metrics and simplified action templates to bridge the gap between general-purpose LLMs and specialized scientific software.
- Experimental results demonstrate that Agent Rosetta achieves performance competitive with specialized ML models and human experts in both canonical and non-canonical protein design pipelines.

---


[Persona-Conditioned Risk Behavior in Large Language Models: A Simulated Gambling Study with GPT-4.1](http://arxiv.org/abs/2603.15831)

- GPT-4.1: introduces a controlled gambling experiment to evaluate whether LLMs exhibit human-like cognitive patterns under varied socioeconomic persona constraints.
- The study demonstrates that GPT-4.1 reproduces behavioral signatures predicted by Prospect Theory, including distinct risk-taking behaviors across Rich, Middle-income, and Poor personas.
- Findings indicate that emotional self-reports function as post-hoc narrations rather than decision drivers, and the model exhibits significant belief rigidity across sequential decision rounds.

---

[QiboAgent: a practitioner’s guideline to open source assistants for Quantum Computing code development](http://arxiv.org/abs/2603.15538)

- QiboAgent: introduces a reference architecture for domain-aware coding assistants in quantum computing that integrates Retrieval-Augmented Generation and autonomous agentic workflows to improve code generation accuracy.
- The framework utilizes a hybrid retrieval pipeline combining semantic search and keyword matching to ground LLMs in specific codebase knowledge, effectively reducing hallucinations.
- QiboAgent employs a multi-agent architecture with specialized roles, including Rust- and Python-focused agents, to automate complex software engineering tasks like library maintenance and multi-language module refactoring.

---

[PMAx: An Agentic Framework for AI-Driven Process Mining](http://arxiv.org/abs/2603.15351)

- PMAx: introduces an autonomous agentic framework that utilizes a multi-agent architecture to perform deterministic process mining while maintaining data privacy and accuracy.
- The framework employs an Engineer Node to synthesize and execute local Python scripts for data analysis, while an Analyst Node interprets the resulting artifacts to generate grounded reports.
- By using a metadata-driven abstraction layer and a static verification mechanism, the system mitigates LLM hallucinations and security risks associated with unverified code execution.

---


[SKILLS: Structured Knowledge Injection for LLM-Driven Telecommunications Operations](http://arxiv.org/abs/2603.15372)

- SKILLS: introduces a benchmark framework for evaluating LLM agents on telecommunications operations tasks using structured domain knowledge injection via SKILL.md documents.
- The framework utilizes JSON scenario definitions, TM Forum mock servers, and deterministic evaluation rubrics to measure performance improvements across various open-weight LLMs.
- Research identifies the Sandbox Discrimination Failure anti-pattern, where reasoning-heavy models misallocate compute resources to infrastructure setup instead of domain-specific tasks.

---

[FORMULACODE: Evaluating Agentic Optimization on Large Codebases](http://arxiv.org/abs/2603.16011)

- FORMULACODE: introduces a benchmark for evaluating the holistic ability of LLMs to optimize large, real-world codebases using fine-grained, multi-objective performance metrics and expert-authored patches.
- The framework utilizes a four-stage pipeline involving repository scraping, rule-based and LLM-based filtering, reproducible environment synthesis, and statistical validation to ensure high-fidelity performance evaluation.
- Evaluations reveal that while frontier LLMs can achieve non-trivial speedups, they generally underperform human experts and struggle with repository-level multi-objective optimization trade-offs.

---

[Auto Researching, not hyperparameter tuning: Convergence Analysis of 10,000 LLM-Guided ML Experiments](http://arxiv.org/abs/2603.15916)

- Orze: introduces a formal framework for autonomous ML research as combinatorial black-box optimization over a structured configuration space.
- The system utilizes LLM agents to perform iterative architecture search, demonstrating that architectural choices explain 94% of performance variance in collision detection tasks.
- The research characterizes multi-agent search dynamics through entropy cycles, agent specialization, and power-law convergence, providing a large-scale empirical benchmark for autonomous scientific discovery.

---

[OpenSeeker: Democratizing Frontier Search Agents by Fully Open-Sourcing Training Data](http://arxiv.org/abs/2603.15594)

- OpenSeeker: introduces a fully open-source search agent framework that utilizes Graph Expansion, Entity Extraction, Question Generation, Entity Obfuscation, Question Obfuscation, Difficulty Check, Solvability Check, Retrospective Summarization, Teacher LLM, and Student LLM to achieve frontier-level performance through high-fidelity data synthesis.
- The framework employs a fact-grounded scalable controllable QA synthesis pipeline to generate complex multi-hop reasoning tasks and a denoised trajectory synthesis method to train agents on raw, noisy web observations.
- By decoupling generation context from training context, the agent learns to internalize robust information-extraction and denoising capabilities, enabling superior performance on search benchmarks using only simple SFT.

---

[Beyond the Covariance Trap: Unlocking Generalization in Same-Subject Knowledge Editing for Large Language Models](http://arxiv.org/abs/2603.15518)

- RoSE (Robust Same-subject Editing): introduces a framework that resolves generalization collapse in same-subject knowledge editing by employing Isotropic Geometric Alignment (IGA) to minimize activation deviation and Hierarchical Knowledge Integration (HKI) to expand the tolerance radius.
- The framework addresses the geometric pathology where prompt-induced activation deviation exceeds the model's tolerance radius, a condition identified as the root cause of generalization failure in existing locate-then-edit methods.
- By replacing the distortion-inducing covariance matrix with an identity constraint and utilizing tree-structured gradient aggregation, RoSE significantly improves instruction-following capabilities in LLMs without compromising locality or increasing computational overhead.

---

[SWE-Skills-Bench: Do Agent Skills Actually Help in Real-World Software Engineering?](http://arxiv.org/abs/2603.15401)

- SWE-Skills-Bench: introduces a requirement-driven benchmark designed to isolate the marginal utility of agent skills in real-world software engineering tasks using LLM, Agent, Skill Library, Requirement Document, and Deterministic Verifier.
- The framework evaluates agent performance across 49 skills by comparing outcomes in Containerized Environment with and without skill injection, utilizing a GitHub Repository as the target codebase.
- Results indicate that skill utility is highly domain-specific, with most skills providing zero pass-rate improvement while potentially increasing token overhead or causing context interference.

---

[Brain-Inspired Graph Multi-Agent Systems for LLM Reasoning](http://arxiv.org/abs/2603.15371)

- BIGMAS (Brain-Inspired Graph Multi-Agent Systems): introduces a framework that organizes specialized LLM agents into dynamically constructed directed graphs to solve complex reasoning tasks through a centralized shared workspace.
- The architecture utilizes a GraphDesigner to create task-specific topologies and a global Orchestrator to manage agent interactions, ensuring full-state visibility and overcoming the limitations of reactive multi-agent approaches.
- BIGMAS consistently improves reasoning performance across standard LLMs and LRMs by distributing cognitive load and externalizing intermediate reasoning states into a globally accessible workspace.

---

[From Storage to Steering: Memory Control Flow Attacks on LLM Agents](http://arxiv.org/abs/2603.15125)

- MEMFLOW: introduces a systematic evaluation framework to identify and quantify Memory Control Flow Attacks (MCFA) that exploit persistent memory to hijack LLM agent tool selection and execution.
- The framework utilizes an injection-auditing protocol to demonstrate how malicious memory, once stored, can override safety policies and induce persistent behavioral deviations across tasks.
- Experimental results across multiple LLMs and frameworks show that MCFA can achieve high attack success rates, highlighting the critical need for robust memory governance and isolation architectures.

---

[Shopping Companion: A Memory-Augmented LLM Agent for Real-World E-Commerce Tasks](http://arxiv.org/abs/2603.14864)

- SHOPPING COMPANION: introduces a unified framework that jointly optimizes long-term memory retrieval and shopping task execution through a two-stage agentic architecture.
- The framework utilizes a dual-reward reinforcement learning strategy with tool-wise supervision to handle sparse feedback in multi-turn interactions.
- The authors provide a novel long-horizon benchmark containing over 1.2 million products to evaluate preference-grounded shopping success across sessions.

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
