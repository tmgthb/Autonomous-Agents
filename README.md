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



#### 20th February 2026

[SARAH: Spatially Aware Real-time Agentic Humans](http://arxiv.org/abs/2602.18432)

- SARAH (Spatially Aware Real-time Agentic Humans): introduces a real-time, fully causal framework for generating spatially-aware conversational motion, with Causal Transformer-based VAE Encoder, Causal Transformer-based VAE Decoder, Transformer-based Flow Matching Model, Gaze Guidance Mechanism, Euclidean Surface-Point Representation, Dialogue LLM, and Speech Synthesis Model.
- The system decouples learning from control by using a flow matching model conditioned on dyadic audio and user trajectory to generate motion latents within a causal VAE space.
- It employs a novel Euclidean joint representation and a classifier-free gaze scoring mechanism to enable user-adjustable eye contact while maintaining naturalistic spatial alignment.

---

[SMaRT: Online Reusable Resource Assignment and an Application to Mediation in the Kenyan Judiciary](http://arxiv.org/abs/2602.18431)

- SMaRT (Selecting Mediators that are Right for the Task): introduces an online resource allocation framework for judicial mediation that combines econometric value-added estimation with a quadratic programming formulation to optimize case resolution rates under soft capacity constraints.
- The system utilizes a multi-agent bandit approach with Bayesian posterior updates to learn mediator quality while employing shadow cases to account for the opportunity cost of assigning high-performing resources to current tasks.
- Evaluated on real-world data from the Kenyan judiciary, the algorithm demonstrates a tunable trade-off between maximizing successful settlements and maintaining equitable mediator workloads through a quadratic penalty parameter.

---

[RVR: Retrieve-Verify-Retrieve for Comprehensive Question Answering](http://arxiv.org/abs/2602.18425)

- RVR (Retrieve-Verify-Retrieve): introduces a multi-round retrieval framework designed to maximize answer coverage for queries with multiple valid answers, utilizing an initial retriever, an LLM-based verifier, and a subsequent retriever.
- The framework iteratively augments the search query with previously verified documents to uncover missing information and reduce redundancy across retrieval rounds.
- RVR achieves higher complete recall on multi-answer benchmarks like QAMPARI and generalizes effectively to out-of-domain datasets compared to agentic search baselines.

---

[ROBO-SABER: Generating and Simulating Virtual Reality Players](http://arxiv.org/abs/2602.18319)

- Robo-Saber: introduces a motion generation system for playtesting virtual reality games, with Game State (current in-game object configuration), 3p History (previous headset and controller poses), Style Encoder (embeds contextual gameplay reference segments), Game Segment Encoder (predicts categorical logits from state), 3p Encoder (maps future poses to logits), 3p Decoder (reconstructs trajectories from sampled logits), GS-VAE (latent space for motion generation), TorchSaber (GPU-accelerated kinematic gameplay evaluator), and Physics-based Tracking Policy (actuates whole-body movements from 3p).
- The framework utilizes a reference-conditioned Categorical Codebook Matching model to generate diverse candidate motion plans aligned with gameplay objectives.
- It enables personalized score prediction and automated testing of game content by emulating diverse player skill levels and movement patterns.

---

[ReqElicitGym: An Evaluation Environment for Interview Competence in Conversational Requirements Elicitation](http://arxiv.org/abs/2602.18306)

- ReqElicitGym (An Evaluation Environment for Interview Competence in Conversational Requirements Elicitation): introduces an interactive and automatic evaluation environment for assessing the interview competence of LLMs in conversational requirements elicitation, with Evaluation Dataset, Oracle User, Task Evaluator, Evaluated Interviewer, and Metrics.
- The framework includes oracle user- and task evaluator-LLMs to simulate stakeholder interactions and provide objective, process-aware performance assessments.
- The environment utilizes a dataset of 101 scenarios to evaluate how effectively agents uncover implicit requirements through multi-turn clarification and probing strategies.

---

[Diffusing to Coordinate: Efficient Online Multi-Agent Diffusion Policies](http://arxiv.org/abs/2602.18291)

- OMAD (Online off-policy MARL framework using Diffusion policies): introduces an online multi-agent reinforcement learning framework that includes decentralized diffusion policies and a centralized distributional critic to facilitate coordinated exploration in high-dimensional action spaces.
- The architecture employs a relaxed policy objective based on a tractable joint entropy evidence lower bound to facilitate effective exploration without requiring exact likelihood computation.
- It integrates a synchronized update strategy and an auto-tuning temperature mechanism within the centralized training with decentralized execution paradigm to ensure stable convergence and superior sample efficiency.

---

[Efficient Calculation of Absorption Spectra of Platinum Complexes Used as Luminescent Probes for Cancer Detection](http://arxiv.org/abs/2602.18284)

- Computational Protocol: introduces a benchmarked workflow for calculating absorption properties of platinum-based cancer probes, with all Isolated Pt(II) Complex (luminescent transition metal probe model), Intercalated DNA Model (biomolecular target environment simulation), Structure Optimization Module (geometry refinement using DFT methods), TD-DFT Response Engine (excited state and intensity calculation), Relativistic Treatment (X2C Hamiltonian and SOC inclusion), Acceleration Approximations (TDA and RI speedup techniques), and Functional Selection (exchange-correlation functional benchmarking)-components.
- The study identifies PBEh-3c as an efficient alternative for geometry optimization and recommends range-separated hybrids like LC-PBE for robust spectral predictions of metal-ligand charge transfer transitions.
- It demonstrates that incorporating spin-orbit coupling is essential for these complexes while TDA and RI approximations provide significant computational speedups with minimal accuracy loss.

---

[PRISM: Parallel Reward Integration with Symmetry for MORL](http://arxiv.org/abs/2602.18277)

- PRISM (Parallel Reward Integration with Symmetry for Multi-Objective Reinforcement Learning): introduces a framework for heterogeneous multi-objective reinforcement learning that aligns reward channels with varying temporal frequencies by enforcing reflectional symmetry as an inductive bias.
- The system utilizes ReSymNet, a residual-based reward model, to approximate scaled opportunity values and transform sparse long-horizon rewards into dense, per-step signals.
- It incorporates SymReg, a symmetry regularizer that constrains policy search to a reflection-equivariant subspace, reducing hypothesis complexity and enhancing generalization across MuJoCo benchmarks.

---

[A Probabilistic Framework for LLM-Based Model Discovery](http://arxiv.org/abs/2602.18266)

- 
ModelSMC (Sequential Monte Carlo for Model Discovery): introduces a probabilistic framework that recasts automated model discovery as Bayesian inference over executable programs, using LLMs as proposal distributions within a Sequential Monte Carlo algorithm.

- 
The system iteratively refines a population of candidate models represented as particles, weighting them by their likelihood under observed data and resampling to concentrate on high-posterior regions.

- 
It incorporates LLM-generated contextual feedback and neural likelihood estimation to handle non-differentiable models and intractable likelihoods across scientific domains including pharmacology and neuroscience.


---

[Role-Adaptive Collaborative Formation Planning for Team of Quadruped Robots in Cluttered Environments](http://arxiv.org/abs/2602.18260)

- Role-Adaptive Collaborative Formation Planning Framework: introduces a dynamic leader-follower architecture for quadruped robot teams, utilizing Global Path Planner, Dynamic Role Assignment, Partial Goal Generator, Virtual Spring-Damper System, Obstacle Avoidance Layer, and Velocity Modifiers to navigate cluttered environments.
- The system employs the Fast Marching Square algorithm for path generation and a look-ahead mechanism that allows temporary formation deformation to bypass obstacles.
- Inter-robot safety and formation stability are maintained through virtual physical connections and an obstacle avoidance layer based on Unsigned Distance Fields.

---

[[Re] Benchmarking LLM Capabilities in Negotiation through Scoreable Games](http://arxiv.org/abs/2602.18230)

- Scoreable Games: introduces a reproduction and extension of a multi-agent negotiation benchmark evaluating LLM cooperation and competition, with Negotiating Agents, Utility Functions, Acceptance Thresholds, Round-based Discussion, Veto Mechanism, Behavioral Prompts, Social Welfare Metrics, and Leakage Detection Logic, including cooperative-, greedy-, and adversarial-agents.
- The framework evaluates the ability of LLMs to reach consensus through role-based dialogue while managing hidden preferences and conflicting individual objectives.
- The study identifies limitations in original experimental setups, refines leakage detection, and introduces social welfare metrics to assess fairness and efficiency in automated negotiations.

---

[Can AI Lower the Barrier to Cybersecurity? A Human-Centered Mixed-Methods Study of Novice CTF Learning](http://arxiv.org/abs/2602.18172)

- CAI (Cybersecurity AI): introduces an agentic framework that integrates LLMs with traditional penetration testing tools to semi-automate reconnaissance, vulnerability analysis, and tool orchestration, utilizing agentic orchestration and human-in-the-loop strategic delegation within a CTF environment; it includes workflow-structuring, tool-orchestration, and step-by-step-support roles.
- The system facilitates novice entry into offensive security by providing strategic overview and mental mapping to reduce cognitive workload during complex multi-step attack chains.
- The study evaluates the impact of AI-mediated learning on performance, strategy exploration, and professional identity formation within Capture-the-Flag environments.

---

[Toward Automated Virtual Electronic Control Unit (ECU) Twins for Shift-Left Automotive Software Testing](http://arxiv.org/abs/2602.18142)

- Agentic Feedback-Driven Model Generation: introduces an automated workflow for synthesizing instruction-accurate virtual electronic control unit models, with coding LLM-based generation and deterministic differential testing.
- The architecture employs a two-loop calibration cycle that iteratively refines SystemC/TLM 2.0 models by comparing their architectural state against a reference simulator accessed via the GNU Debugger.
- This methodology reduces the technical risk of CPU behavioral fidelity in virtual platforms, enabling early software integration and safety-aligned fault-injection campaigns.

---

[Agentic Adversarial QA for Improving Domain-Specific LLMs](http://arxiv.org/abs/2602.18137)

- Agentic Adversarial QA: introduces an iterative, feedback-driven framework that generates a set of adversarial questions to improve domain-specific LLMs by targeting interpretive reasoning gaps, with Strong Model, Weak Model, Feedback Model, Guidance Model, Revision Model, Domain Context, Question Variable, Synthetic Dataset, and TextGrad Optimizer; it includes strong-, weak-, feedback-, guidance-, and revision-agents.
- The system utilizes a differentiable prompting paradigm to optimize a natural language question variable, maximizing the response divergence between an expert model and a smaller target model.
- Fine-tuning on the resulting adversarial synthetic dataset enables smaller models to achieve performance competitive with larger counterparts while using fewer training tokens than traditional augmentation strategies.

---

[Fair Orientations: Proportionality and Equitability](http://arxiv.org/abs/2602.18098)

- Orientation Model: introduces a graph-based framework for allocating indivisible items among agents under relevance constraints, where items are represented as edges and agents as vertices, with all Agents (participants as graph vertices), Items (indivisible goods or chores), Relevance Graph (structural constraints on allocation), Valuation Labels (numerical values for edges), Clause Gadgets (sub-graphs for logical clauses), and Variable Gadgets (sub-graphs for Boolean variables).
- The study investigates the existence and computational complexity of proportionality and equitability notions, proving that finding such orientations is generally NP-complete even for simple graphs.
- The authors establish existence results and polynomial-time algorithms for relaxations such as PROP1 and characterize the conditions for EF1 orientations in chore-based allocation scenarios.

---

[Dynamic Deception: When Pedestrians Team Up to Fool Autonomous Cars](http://arxiv.org/abs/2602.18079)

- Dynamic Deception (Collusive and Dynamic Pedestrian Attack): introduces a system-level adversarial attack where multiple pedestrians coordinate motion and spatial alignment to amplify adversarial patches printed on clothing, with Adversarial Pedestrians (dynamic carriers of adversarial signals), Adversarial Patches (visually disguised stop-sign patterns), Surrogate Model (YOLOv5 for black-box training), Target Autonomous Agent (Simlingo agent with LLM-based interpretation), Coordination Logic (spatial alignment and motion synchronization), and CARLA Simulator (autonomous driving simulation environment).
- The framework employs a black-box training strategy using a surrogate YOLOv5 model to generate stop-sign perturbations disguised as camellia flowers to maintain visual concealment from human observers.
- Evaluation against the Simlingo agent shows that the system, which utilizes an LLM for visual interpretation, is successfully misled into executing full stops only when adversarial signals persist through synchronized pedestrian movement.

---

[3DMedAgent: Unified Perception-to-Understanding for 3D Medical Analysis](http://arxiv.org/abs/2602.18064)

- 3DMedAgent (Unified Perception-to-Understanding for 3D Medical Analysis): introduces a unified agentic framework that enables 2D MLLMs to perform general 3D CT analysis by coordinating heterogeneous visual and textual tools through a multi-step reasoning process.
- The architecture includes tool-calling, organ initialization, volume cropping, lesion targeting, evidence integration, result summary, slice/tool selection, and memory selection agents.
- The framework utilizes the DeepChestVQA benchmark to evaluate perception-to-understanding capabilities in 3D

---

[Towards More Standardized AI Evaluation: From Models to Agents](http://arxiv.org/abs/2602.18029)

- AI Evaluation Framework: introduces a systematic methodology for measuring agentic system behavior, with Risk Assessment (identifying potential harms and severity), Evaluation Requirements (defining measurable conditions and thresholds), Evaluation Harness (infrastructure orchestrating agent tool loops), Evaluation Graders (includes code-based, LLM-based, and human-graders), Evaluation Transcripts (complete record of reasoning steps), Evaluation Outcomes (final environmental state changes), and Governance Decisions (deployment, rollback, or escalation actions).
- The approach shifts evaluation from downstream verification to a core system function that utilizes deterministic code-based graders, probabilistic model-based judges, and human calibration.
- The research emphasizes the use of simulated environments and reliability metrics like Pass^k to assess autonomous agents operating in stateful, non-deterministic worlds.

---

[Mean-Field Reinforcement Learning without Synchrony](http://arxiv.org/abs/2602.18026)

- TMF (Temporal Mean Field): introduces a mean-field reinforcement learning framework centered on the population distribution to enable scalable multi-agent decision-making under asynchronous and sequential protocols.
- The framework formalizes population dynamics through a deterministic recursion and defines a self-consistent equilibrium that remains well-defined regardless of the number of active agents per step.
- Theoretical analysis establishes a finite-population approximation bound of O(1/sqrt(N)) and proves the convergence of the TMF-PG algorithm to a unique equilibrium.

---

[NIMMGen: Learning Neural-Integrated Mechanistic Digital Twins with LLMs](http://arxiv.org/abs/2602.18008)

- NIMMGen: introduces an agentic framework for neural-integrated mechanistic modeling that enhances code correctness and practical validity through iterative refinement, with data-, modeling-, verification-, and reflection-agents, an environment engine, memory, and a code RAG database.
- The architecture utilizes a self-evolving optimization loop to progressively refine model specifications and address challenges in partial observation and diversified task objectives.
- The framework incorporates an error-handling module and a retrieval-augmented generation tool to mitigate runtime failures and ensure the scientific grounding of generated digital twins.

---

[Aurora: Neuro-Symbolic AI Driven Advising Agent](http://arxiv.org/abs/2602.17999)

- Aurora: introduces a modular neuro-symbolic framework for academic advising with a PostgreSQL Knowledge Base, an Intent & NER Service, a SQL Router, a Prolog Reasoner, a CoT Controller, and an LLM Core Module, delivering policy-compliant and verifiable course recommendations by unifying retrieval-augmented generation with symbolic reasoning.
- The architecture utilizes a BCNF-normalized database and a Prolog engine to enforce strict academic constraints, ensuring that all generated advice adheres to institutional policies and prerequisite structures.
- Empirical evaluation demonstrates that the framework significantly outperforms raw LLM baselines in semantic alignment and precision while achieving sub-second latency on commodity hardware.

---

[WORKFLOWPERTURB: Calibrated Stress Tests for Evaluating Multi-Agent Workflow Metrics](http://arxiv.org/abs/2602.17990)

- WORKFLOWPERTURB: introduces, a controlled benchmark for evaluating workflow metrics, with golden workflows, a perturbation engine, a validation module, an evaluation metric suite, and an LLM-as-judge.
- The framework utilizes LLM-based perturbation- and judgment-components to generate variants across missing steps, compressed steps, and description changes at multiple severity levels.
- It evaluates structural, lexical, semantic, and judgment-based metrics to characterize their sensitivity and calibration for multi-agent system validation.

---

[Learning Optimal and Sample-Efficient Decision Policies with Guarantees](http://arxiv.org/abs/2602.17978)

- DML-CMR (Double Machine Learning for Conditional Moment Restrictions): introduces a sample-efficient framework for learning optimal decision policies from confounded datasets, with DML-CMR (solves conditional moment restrictions), DML-IL (performs causal imitation learning), KC (learns temporal logic objectives), nuisance parameter estimators (estimates conditional expectations), a Neyman orthogonal score function (debiasing objective), a cross-fitting regime (data splitting for robustness), a roll-out model (predicts transition dynamics), an expert model (imitates policy), a product MDP (synchronizes environment and logic), an LDBA (automaton for LTL), a K-counter (tracks accepting state visits), and counterfactual imagining (generates synthetic experiences).
- The framework addresses hidden confounders in offline reinforcement learning and imitation learning by leveraging instrumental variables and trajectory histories as causal instruments to identify true effects.
- The methodology extends to high-level objectives by transforming linear temporal logic specifications into limit-deterministic Buchi automata within a synchronized product MDP featuring a generalized reward structure.

---

[Mining Type Constructs Using Patterns in AI-Generated Code](http://arxiv.org/abs/2602.17955)

- 
Dataset Filtering Framework: introduces a hierarchical two-stage pipeline to analyze type-related patterns in AI-generated code, with rule-based regex parsers, a multi-agent LLM system, a classifier agent, and a validator agent.

- 
The multi-agent LLM system includes classifier- and validator-agents that process pull request titles, descriptions, and code patches to identify specific TypeScript type constructs and anti-patterns.

- 
The research demonstrates that AI agents are 9x more likely than humans to use the 'any' keyword and frequently employ type-safety escape hatches, despite achieving higher pull request acceptance rates.


---

[Graph-Neural Multi-Agent Coordination for Distributed Access-Point Selection in Cell-Free Massive MIMO](http://arxiv.org/abs/2602.17954)

- APS-GNN (Access-Point Selection Graph Neural Network): introduces a scalable distributed multi-agent learning framework for cell-free massive MIMO, utilizing GRU encoders, GNN layers, and dual critics to coordinate binary connection decisions across individual AP-UE agents.
- The system employs a constrained reinforcement learning formulation where spectral efficiency violations are treated as costs and power reduction as rewards, managed by an adaptive Lagrangian multiplier to optimize network performance.
- The architecture utilizes structured message passing over same-UE and same-AP edges to enable spatial coordination and interference management without requiring global state aggregation or centralized control.

---

[Analyzing LLM Instruction Optimization for Tabular Fact Verification](http://arxiv.org/abs/2602.17937)

- DSPy (Declarative Self-improving Language Programs): introduces a systematic comparison of instruction optimization for tabular fact verification, utilizing COPRO, MiPROv2, and SIMBA to enhance LLM reasoning performance without gradient updates.
- The framework evaluates text-only prompting against tool-augmented agents, including SQL-executing ReAct and Python-executing CodeAct modules, across multiple model families and benchmarks.
- Experimental results show that MiPROv2 stabilizes Chain-of-Thought gains while SIMBA optimizes agentic behavior by reducing redundant tool calls and improving numerical comparison heuristics.

---

[Memory-Based Advantage Shaping for LLM-Guided Reinforcement Learning](http://arxiv.org/abs/2602.17931)

- Memory-Based Advantage Shaping: introduces a reinforcement learning framework that utilizes a memory graph, LLM, RL agent, utility computation, and advantage shaping to guide exploration in sparse-reward environments.
- The system calculates a utility signal based on trajectory alignment with the memory graph to augment the advantage function, providing the policy with external guidance without modifying the reward structure.
- By combining offline priors with adaptive online LLM queries, the method enhances sample efficiency and early-stage learning while preserving the convergence properties of Proximal Policy Optimization.

---

[MIRA: Memory-Integrated Reinforcement Learning Agent with Limited LLM Guidance](http://arxiv.org/abs/2602.17930)

- MIRA (Memory-Integrated Reinforcement Learning Agent): introduces a reinforcement learning framework that incorporates a structured memory graph to amortize LLM queries into persistent knowledge for guiding exploration in sparse-reward environments.
- The architecture includes offline- and online-guidance LLMs, a screening unit for hallucination reduction, and a utility module that generates shaping signals to refine policy updates without altering the underlying reward function.
- By decaying the influence of LLM-derived priors as the agent's policy matures, the framework maintains autonomous decision-making and ensures convergence to optimal behavior while requiring substantially fewer online queries.

---

[Robust Temporal Guarantees in Budgeted Sequential Auctions](http://arxiv.org/abs/2602.17916)

- Primal Budget Pacing Algorithm: introduces a deterministic bid adjustment mechanism for budgeted sequential auctions that ensures robust win guarantees against adversarial behavior, with Primal Budget Pacing Algorithm, Learner, Optimizer, Auction Environment, Budget Management, and Deterministic Update Rule components.
- The algorithm guarantees that an agent with a specific budget fraction secures a proportional share of total wins minus a sublinear regret term, even under adversarial competition.
- In multi-agent settings, the dynamics converge to a low-discrepancy distribution of wins, achieving temporal spacing and round-robin outcomes in equal-budget scenarios.

---

[From Lossy to Verified: A Provenance-Aware Tiered Memory for Agents](http://arxiv.org/abs/2602.17913)

- TierMem (Provenance-Aware Tiered Memory): introduces a provenance-linked two-tier memory hierarchy that optimizes the accuracy-efficiency trade-off by dynamically allocating evidence at inference time, including router-, planner-, extractor-, and generator-LLMs.
- The architecture defaults to a fast summary index and employs a lightweight router to escalate to immutable raw logs only when summary evidence is insufficient for a faithful answer.
- Explicit provenance links guide deep retrieval while online consolidation writes verified findings back to the summary tier to amortize future query costs and reduce latency.

---

[Alignment in Time: Peak-Aware Orchestration for Long-Horizon Agentic Systems](http://arxiv.org/abs/2602.17910)

- APEMO (Affect-aware Peak-End Modulation for Orchestration): introduces a runtime scheduling layer that optimizes computational allocation under fixed budgets by operationalizing temporal-affective signals, with Multi-Agent Execution, Planner-Agent, Executor-Agent, Critic-Agent, Frustration Monitor, Peak Detection, Precision Repair Trigger, Budget Reallocation, Cost Accounting, and Budget Constraint.
- The system monitors behavioral proxies such as repetition and context drift to detect negative peaks, reallocating inference precision to stabilize critical trajectory segments and endings.
- Experimental results across multi-agent simulations and planner-executor flows demonstrate improved trajectory-level robustness and reuse probability compared to uniform budget allocation strategies.

---

#### 19th February 2026

[OpenEarthAgent: A Unified Framework for Tool-Augmented Geospatial Agents](http://arxiv.org/abs/2602.17665)

- OpenEarthAgent: introduces a unified agentic framework for developing tool-augmented geospatial agents with User Input (multimodal geospatial queries), Geo-Reasoning Engine (LLM-based reasoning controller), Orchestrator (manages tool calls and feedback), Short-term Working Memory (stores reasoning history and data), Tool Execution Cache Memory (reusable geospatial archives and layers), External Tools (specialized GIS and perceptual operators), and Core (central agentic loop) components, facilitating organized multi-step geospatial reasoning.
- The framework utilizes a specialized LLM-based reasoning engine to orchestrate a diverse registry of perceptual, GIS computation, spectral, and georeferenced raster tools.
- It includes a comprehensive multimodal corpus of over 14,000 training instances with detailed reasoning traces to align models with verified multi-step tool interactions across diverse earth observation contexts.

---


[From Labor to Collaboration: A Methodological Experiment Using AI Agents to Augment Research Perspectives in Taiwan's Humanities and Social Sciences](http://arxiv.org/abs/2602.17221)

- Agentic Workflow (AI Agent-based collaborative research workflow): introduces a seven-stage modular research framework for humanities and social sciences, with specialized agents for literature collection, analysis, data exploration, statistical analysis, and academic writing, and a human-in-the-loop quality gate system.
- The framework utilizes three operational modes—direct execution, iterative refinement, and human-led—to balance AI's execution speed with the irreplaceability of human contextual reasoning.
- By integrating negative constraints and Git-based version control, the methodology ensures verifiability and mitigates LLM hallucination risks during complex academic tasks.



[FAMOSE: A ReAct Approach to Automated Feature Discovery](http://arxiv.org/abs/2602.17641)

- FAMOSE (Feature AugMentation and Optimal Selection agEnt): introduces an agentic framework for automated feature engineering that utilizes the ReAct paradigm to iteratively hypothesize, generate, and refine features based on empirical model performance.
- The system integrates a metadata generator, a Python code compiler, and a feature evaluation tool to allow the LLM to interact directly with tabular data and learn from previous successes or failures recorded in its context window.
- After the iterative discovery phase, the framework employs a minimal-redundancy maximal-relevance (mRMR) algorithm to select a compact and optimal set of features, achieving state-of-the-art results on various classification and regression tasks.

---

[What Makes a Good LLM Agent for Real-world Penetration Testing?](http://arxiv.org/abs/2602.17622)

- PENTESTGPT V2 (Penetration Testing GPT version 2): introduces an LLM-based automated penetration testing agent that addresses capability gaps and complexity barriers through difficulty-aware planning and structured state management.
- The framework utilizes a Task Difficulty Assessment (TDA) mechanism to guide an Evidence-Guided Attack Tree Search (EGATS), enabling the agent to dynamically pivot between reconnaissance and exploitation based on real-time tractability signals.
- It incorporates a Tool & Skill Layer with typed interfaces and a Memory Subsystem to maintain long-term state, significantly improving performance on multi-step attack chains in complex environments like Active Directory.

---

[AutoNumerics: An Autonomous, PDE-Agnostic Multi-Agent Pipeline for Scientific Computing](http://arxiv.org/abs/2602.17607)

- AutoNumerics: introduces an autonomous multi-agent framework that utilizes LLMs to design, implement, and verify transparent numerical solvers for partial differential equations from natural language descriptions, featuring Formulator-, Planner-, Feature-, Selector-, Coder-, Critic-, and Reasoning-agents.
- The system utilizes a coarse-to-fine execution strategy to decouple logic debugging on low-resolution grids from stability validation on high-resolution grids.
- It incorporates a residual-based self-verification mechanism to assess solver correctness in the absence of analytical solutions and a history decimation mechanism for memory-efficient temporal simulations.

---

[BMC4TimeSec: Verification Of Timed Security Protocols (Demo)](http://arxiv.org/abs/2602.17590)

- BMC4TimeSec: introduces an end-to-end verification tool for Timed Security Protocols by integrating multi-agent Timed Interleaved Interpreted Systems with SMT-based bounded model checking.
- The system utilizes a modular pipeline that transforms Alice-Bob notation and JSON interpretations into formal automata and SMT-LIB2 formulas for automated analysis via the Z3 solver.
- It supports complex attack scenarios including session interleaving, message replays, and time-dependent vulnerabilities through a user-friendly Flask-based graphical interface.

---

[Modeling Distinct Human Interaction in Web Agents](http://arxiv.org/abs/2602.17588)

- PLOWPILOT: introduces an intervention-aware web navigation framework that utilizes a Large Multimodal Model to predict when and why human users will intervene during task execution.
- The system leverages COWCORPUS, a dataset of 400 real-user trajectories, to train style-conditioned models that adapt to distinct collaboration patterns like takeover or hands-on oversight.
- Experimental results demonstrate that modeling human intervention timing significantly improves agent usefulness and reduces unnecessary interruptions in collaborative web environments.

---

[RETOUCHIQ: MLLM Agents for Instruction-Based Image Retouching with Generalist Reward](http://arxiv.org/abs/2602.17558)

- RETOUCHIQ: introduces an instruction-based image retouching framework that utilizes MLLM agents guided by a generalist reward model to translate high-level aesthetic intentions into precise, executable tool-use parameters.
- The architecture employs a two-stage training pipeline featuring supervised fine-tuning for reasoning traces and reinforcement learning via a reward critic that provides scalar feedback through multimodal reasoning.
- The framework implements Policy-Guided Reward Training to mitigate distribution shifts between synthetic perturbations and actual policy outputs, enhancing both semantic consistency and perceptual quality in professional editing software.

---

[KLong: Training LLM Agent for Extremely Long-horizon Tasks](http://arxiv.org/abs/2602.17547)

- KLong: introduces an open-source LLM agent framework designed to solve extremely long-horizon tasks through a cold-start trajectory-splitting SFT phase followed by progressive RL scaling, utilizing Research-Factory, trajectory-splitting SFT, progressive RL, search-, evaluation- and judge-agents, a secure sandbox, and optimized scaffolding.
- The system utilizes Research-Factory to automate the collection of research papers and the construction of evaluation rubrics, generating high-quality trajectories distilled from advanced reasoning models.
- The training approach addresses context window limitations by decomposing long trajectories into overlapping sub-trajectories and employs a multi-stage RL schedule with increasing timeouts to improve task-solving stability.

---

[ADAPTIVE DECENTRALIZED COMPOSITE OPTIMIZATION VIA THREE-OPERATOR SPLITTING](http://arxiv.org/abs/2602.17545)

- DATOS (Decentralized Adaptive Three Operator Splitting): introduces a parameter-free decentralized optimization framework for composite problems, with DATOS, Communication Step, Decentralized Line-search, Min-consensus, Primal-dual Updates, Tracking Variables, and BCV Metric.
- The framework leverages a Davis-Yin three-operator splitting applied to a BCV-type reformulation to enable local backtracking stepsize updates using only single-hop communications.
- It establishes sublinear convergence for convex losses and linear convergence for strongly convex losses with partly smooth nonsmooth components without requiring global network information.

---

[Evaluating Chain-of-Thought Reasoning through Reusability and Verifiability](http://arxiv.org/abs/2602.17544)

- 
Thinker-Executor framework: introduces a method to decouple Chain-of-Thought generation from execution, evaluating reasoning quality through reusability and verifiability metrics rather than just final answer accuracy.

- 
The system utilizes a committee of Executor models to measure how effectively a Thinker's reasoning trace can persuade an independent model to correct its answer or be misled by corrupted logic.

- 
Experimental results demonstrate that specialized reasoning models do not necessarily produce more reusable or verifiable reasoning traces than general-purpose LLMs, highlighting a disconnect in current accuracy-based benchmarks.


---

[TOWARD A FULLY AUTONOMOUS, AI-NATIVE PARTICLE ACCELERATOR](http://arxiv.org/abs/2602.17536)

- Natively-AI Particle Accelerator: introduces a vision for self-driving facilities designed from inception with AI as the primary operator, with agentic AI architecture, integrated knowledge bases, learning and adaptive control, simulation and digital twins, automated health monitoring, safety and transparency frameworks, modular hardware, multimodal data fusion, and cross-domain collaboration.
- The framework proposes a transition from current AI-assisted operations through AI-augmented collaboration to a final AI-autonomous stage where humans provide strategic oversight.
- The architecture incorporates LLM-driven planning, reasoning, and assistant agents to manage machine complexity at speed while ensuring safety through digital twin validation and explainable decision-making.

---

[A Picture of Agentic Search](http://arxiv.org/abs/2602.17518)

- ASQ (Agentic Search Queryset): introduces a methodology for systematically logging the search behaviors of agentic RAG systems by intercepting retrieval calls and decoding processes to capture synthetic queries, retrieved documents, and reasoning steps.
- The architecture implements a multi-turn reasoning loop between a generator LLM and a retriever, utilizing XML-style control tags to coordinate autonomous reasoning-, query formulation-, and answer generation-agents.
- The resulting dataset enables the evaluation of information retrieval systems against machine-generated query streams, which exhibit higher volumes and different reformulation patterns compared to organic human search traffic.

---

[Retrospective In-Context Learning for Temporal Credit Assignment with Large Language Models](http://arxiv.org/abs/2602.17497)

- RICOL (Retrospective In-Context Online Learning): introduces an online reinforcement learning framework that transforms sparse environmental feedback into dense supervision signals by leveraging LLMs for retrospective in-context learning, with Actor LLM, Reflector LLM, Environment, In-context Updated Policy, Advantage Function Estimator, and Policy Optimizer; it includes actor- and reflector-LLMs.
- The system employs a reflector LLM to analyze hindsight trajectories and provide corrective verbal feedback for individual actions, enabling fine-grained temporal credit assignment.
- It estimates advantage functions by calculating the discrepancy between the log-probabilities of the base policy and its in-context refined version, improving sample efficiency in multi-turn tasks.

---

[Linear Convergence in Games with Delayed Feedback via Extra Prediction](http://arxiv.org/abs/2602.17486)

- WOGDA (Weighted Optimistic Gradient Descent-Ascent): introduces a predictive optimization framework for unconstrained bilinear games that achieves linear convergence under delayed feedback, with WOGDA (iterative optimization algorithm), EPP (implicit approximation method), Delayed Feedback Mechanism (fixed reward observation latency), Extra Prediction Module (future reward extrapolation), and Strategy Update Rule (weighted cumulative reward calculation).
- The framework utilizes the Extra Proximal Point (EPP) method to establish that predicting rewards farther into the future permits larger step sizes and accelerates convergence.
- Theoretical analysis establishes that extra prediction accelerates the convergence rate from exp(-Θ(t/m^5)) to exp(-Θ(t/(m^2 log m))) for a feedback delay of m steps.

---

[What Do LLMs Associate with Your Name? A Human-Centered Black-Box Audit of Personal Data](http://arxiv.org/abs/2602.17483)

- LMP2 (Language Model Privacy Probe): introduces a browser-based, human-centered audit tool designed to evaluate personal data associations within black-box LLMs through iterative user testing and a WikiMem-based backend.
- The system employs a multi-stage pipeline including paraphrased canary templates, prefix-based ground-truth truncation, and counterfactual generation to compute association strength and confidence scores for 50 human-related properties.
- Research findings indicate that GPT-4o can confidently generate 11 personal attributes for everyday users with high accuracy, highlighting privacy risks and user demand for data rectification and erasure rights.

---

[A variational mean field game of controls with free final time and pairwise interactions](http://arxiv.org/abs/2602.17447)

- NAG (Non-Atomic Game with pairwise interactions): introduces a variational framework for mean field games where agents minimize individual and pairwise interaction costs over abstract Polish spaces with free final time.
- The model characterizes equilibria as critical points of a potential functional, ensuring existence through the minimization of this functional.
- The framework is applied to crowd motion models with Cucker-Smale type interactions, incorporating velocity-dependent costs and target-set arrival criteria.

---

[WarpRec: Unifying Academic Rigor and Industrial Scale for Responsible, Reproducible, and Efficient Recommendation](http://arxiv.org/abs/2602.17442)

- WarpRec: introduces a modular, backend-agnostic framework for recommender systems, with Reader, Data Engine, Recommendation Engine, Evaluation, Writer, Application Layer, Narwhals, Ray, and CodeCarbon components.
- The architecture utilizes Narwhals to abstract data backends and Ray to facilitate seamless transitions from local prototyping to large-scale distributed training on multi-GPU clusters.
- It incorporates CodeCarbon for sustainable AI profiling and natively implements the Model Context Protocol to transform recommenders into queryable agents for LLMs.

---

[Multi-Agent Temporal Logic Planning via Penalty Functions and Block-Coordinate Optimization](http://arxiv.org/abs/2602.17434)

- BCGD-PM (Block-Coordinate Gradient Descent - Penalty Method): introduces a scalable optimization-based framework for multi-agent Signal Temporal Logic planning by relaxing coupled collaborative constraints into an unconstrained problem using quadratic penalty functions.
- The architecture utilizes a two-layer optimization scheme where an inner loop performs block-coordinate gradient descent on agent-specific decision variables and an outer loop updates penalty parameters to ensure specification satisfaction.
- By integrating smooth robustness metrics with separable objective functions, the method enables parallelized computations that remain computationally efficient even as the number of agents and complexity of collaborative tasks increase.

---

[Distributed Virtual Model Control for Scalable Human-Robot Collaboration in Shared Workspace](http://arxiv.org/abs/2602.17415)

- VMC (Virtual Model Control): introduces a decentralized, agent-agnostic framework for human-robot collaboration using goal and avoidance springs, unilateral saturating dampers, a force-based stall detector, a conflict resolution layer, VMRobotControl.jl, and a perception system to regulate motion without explicit trajectory planning.
- The architecture utilizes virtual mechanical components to generate interaction forces and a biased-draw prioritization negotiation to resolve deadlocks identified by force-balance metrics.
- Experimental validation with UR5 robots and humans demonstrates that the approach maintains consistent separation distances and scales to multi-agent scenarios without structural control modifications.

---

[COMPUTER-USING WORLD MODEL](http://arxiv.org/abs/2602.17365)

- CUWM (Computer-Using World Model): introduces a factorized world model for desktop productivity software that predicts future user interface states by separating semantic change descriptions from visual rendering, with a current UI state, candidate action, textual transition model, transition description, visual realization model, next UI state, LLM-based judge, and reinforcement learning refinement.
- The architecture includes a vision-language-based transition model, a diffusion-based realization model, and an LLM-based judge for reinforcement learning refinement.
- Evaluation via test-time action search demonstrates that simulating outcomes with CUWM improves the decision quality and robustness of frozen LLM agents in complex Office tasks.

---

[What Breaks Embodied AI Security: LLM Vulnerabilities, CPS Flaws, or Something Else?](http://arxiv.org/abs/2602.17345)

- Embodied AI Robot System Architecture: introduces a comprehensive analysis of security vulnerabilities in embodied systems, with Perception Layer, Decision and Reasoning Layer, Behavior Planning and Control Layer, Physical Robot Body & Environment, Multimodal Foundation Model, World Model, Sub-task chain, Motion Planning Algorithms, Dynamic Constraints, and Low-Level Motor Commands.
- The framework identifies that semantic correctness in LLMs does not imply physical safety due to the abstraction of geometry, dynamics, and contact constraints, and includes LLM-, VLM- and VLA-based reasoning modules.
- It systematizes threats across LLM-centric, CPS-centric, and embodiment-specific dimensions to reveal how small errors propagate and amplify in perception-decision-action loops.

---

[Flickering Multi-Armed Bandits](http://arxiv.org/abs/2602.17315)

- FMAB (Flickering Multi-Armed Bandits): introduces a sequential decision-making framework where action availability is constrained by a time-varying graph, with Arms, Evolving Graph, Learner, Lazy Random Walk, Navigation & Commitment, and Reward Distribution.
- The proposed two-phase algorithm utilizes a lazy random walk for exploration to identify the optimal arm, followed by a navigation phase to commit to that arm.
- Theoretical analysis provides sublinear regret bounds for i.i.d. Erdős–Rényi and Edge-Markovian graph models, validated by simulations in a disaster-response scenario.

---

[MedClarify: An information-seeking AI agent for medical diagnosis with case-specific follow-up questions](http://arxiv.org/abs/2602.17308)

- MedClarify: introduces an information-seeking AI agent that iteratively generates case-specific follow-up questions to reduce diagnostic uncertainty and refine differential diagnoses, with medical diagnosis, question generation, diagnostic simulations, diagnostic information gain, disease similarity mapping, and Bayesian update components.
- The system utilizes a novel Diagnostic Expected Information Gain (DEIG) metric to select questions that maximize entropy reduction while accounting for semantic proximity between diseases via ICD-11 mapping.
- Evaluation using an agentic framework with patient-, doctor-, update-, and evaluator-agents demonstrates significant accuracy improvements on incomplete patient cases across various LLM backbones.

---

[Visual Insights into Agentic Optimization of Pervasive Stream Processing Services](http://arxiv.org/abs/2602.17282)

- MUDAP (Multi-Dimensional Autoscaling Platform) and RASK (Regression Analysis of Structural Knowledge): introduces a two-fold architecture for context-aware autoscaling of stream processing services on Edge devices, utilizing Data Sources, Service Containers, a Time-Series DB, a Scaling Agent, Regression Functions, a Numerical Solver, and a REST API.
- The MUDAP platform exposes service-specific parameters such as data quality and resource limits, while the RASK agent builds structural knowledge through regression analysis to optimize Service Level Objective fulfillment.
- The system achieves high sample efficiency by modeling variable relations and using a numerical solver to navigate multi-dimensional elasticity across co-located services under strict resource constraints.

---

[Federated Latent Space Alignment for Multi-user Semantic Communications](http://arxiv.org/abs/2602.17271)

- Federated Latent Space Alignment: introduces a decentralized framework for mitigating latent space mismatches in multi-user semantic communications, with Access Point (AP), User Devices, Shared Semantic Pre-equalizer, Local Semantic Equalizers, Federated ADMM Optimizer, MIMO Channel, and Semantic Pilots.
- The architecture utilizes a shared linear pre-equalizer at the transmitter and personalized equalizers at each receiver to align heterogeneous latent representations across a broadcast MIMO channel.
- A federated ADMM protocol enables privacy-preserving training by exchanging only intermediate variables instead of raw latent data or model weights.

---

[Web Verbs: Typed Abstractions for Reliable Task Composition on the Agentic Web](http://arxiv.org/abs/2602.17245)

- Web Verbs: introduces a semantic layer for web actions that exposes site capabilities as typed, semantically documented functions to enable task composition for LLM-based agents, with User, Coding Agent, NLWeb Vector Database, Verb Layer, Web Verbs, Playwright, and Raw Web.
- The framework unifies API-based and browser-based paradigms by wrapping low-level primitives into composable units that carry preconditions, postconditions, and structured outputs.
- By shifting the agent's role from predicting GUI steps to synthesizing structured programs, the system achieves higher success rates and efficiency in complex multi-site workflows.

---

[TAPO-Structured Description Logic for Information Behavior: Procedural and Oracle-Based Extensions](http://arxiv.org/abs/2602.17242)

- TAPO-DL (TAPO-Structured Description Logic): introduces a formal extension of classical description logic to model information behavior as a dynamic process, with T-Box (static terminological knowledge and concept axioms), A-Box (contextual assertions and factual knowledge), P-Box (programmable layer for imperative-style procedures), O-Box (interface for external oracle interactions), Sheaf-Theoretic Semantics (mathematical framework for contextual information stability), and Epistemic Agents (structures generating and stabilizing informational sections).
- The architecture integrates terminological and assertional components with procedural dynamics and oracle-based reasoning to enable explicit representation of information-generating actions and external validation. 
- The system models local informational states as sections within a co-generative process where global coherence is achieved through the gluing of stable sections across contextual domains.

---

[HiMAP: History-aware Map-occupancy Prediction with Fallback](http://arxiv.org/abs/2602.17231)

- HiMAP (History-aware Map-occupancy Prediction with Fallback): introduces a tracking-free trajectory prediction framework that reconstructs agent histories from unlabeled occupancy maps to maintain reliability during multi-object tracking failures.
- The architecture employs a historical query module that iteratively attends to spatiotemporally invariant occupancy representations to implicitly recover past states without requiring explicit identity association.
- Utilizing a DETR-style decoder and reusable encodings, the system provides a robust safety fallback for autonomous driving that matches the performance of tracking-dependent baselines on the Argoverse 2 dataset.

---

[Continual Learning and Refinement of Causal Models through Dynamic Predicate Invention](http://arxiv.org/abs/2602.17217)

- Continuous Model Repair: introduces a self-supervised framework for constructing symbolic causal world models online by integrating continuous model learning and repair into an agent's decision loop, with Learnt Abstractions, Learnt Dynamics, and Learnt Constraints.
- The system leverages Meta-Interpretive Learning and dynamic predicate invention to find semantically meaningful abstractions, enabling the construction of a hierarchy of disentangled concepts from observations.
- By employing a Predict-Verify-Refine cycle and a Global Predicate Registry, the framework achieves high sample efficiency and scale-invariant learning in complex relational environments.

---

[NotebookRAG: Retrieving Multiple Notebooks to Augment the Generation of EDA Notebooks for Crowd-Wisdom](http://arxiv.org/abs/2602.17215)

- NotebookRAG: introduces an automated Exploratory Data Analysis (EDA) framework that retrieves and enhances notebook content to generate intent-aligned analysis notebooks, with Notebook Retrieval, Component Extraction, Metadata Annotation, Intent-Guided Retrieval, Component Enhancement, Notebook Generation Agent, Retrieval Interface, Planner, Progress Manager, Coder, Summarizer, Sandbox Memory, VLM, and LLM.
- The system transforms code cells into context-enriched executable components and re-executes them on new datasets to obtain updated visualizations and reliable insights via Vision-Language Models.
- The generation agent includes planning-, progress tracking-, coding-, and summarization-agents to produce coherent, structured, and runnable notebooks that align with abstract user intent.

---

[EXTENDING QUANTUM THEORY WITH AI-ASSISTED DETERMINISTIC GAME THEORY (EXTENDED ABSTRACT)](http://arxiv.org/abs/2602.17213)

- AI-assisted deterministic game theory framework: introduces a local hidden-variable model for predicting quantum experiment outcomes by framing protocols as extensive form games solved via a differentiable Perfectly Transparent Equilibrium solver.
- The architecture utilizes a neural network to learn reward functions that minimize the Kullback-Leibler divergence between simulated deterministic outcomes and Born rule statistics.
- This approach demonstrates Bell inequality violation in a local-realist framework by replacing the free choice assumption with counterfactually dependent perfect prediction.

---

[Algorithmic Collusion at Test Time: A Meta-game Design and Evaluation](http://arxiv.org/abs/2602.17203)

- Meta-game framework: introduces a methodology for evaluating algorithmic collusion at test time by modeling agents as meta-strategies that combine pretrained initial policies with specific in-game adaptation rules.
- The architecture employs internal representations and update functions to facilitate adaptation across Q-learning, UCB, and LLM-based agents, including GPT5-mini and GPT5-nano models.
- The research applies Empirical Game-Theoretic Analysis to construct best-response graphs and identify Nash equilibria, assessing the strategic stability of collusive outcomes under rational choice.

---

[The Bots of Persuasion: Examining How Conversational Agents’ Linguistic Expressions of Personality Affect User Perceptions and Decisions](http://arxiv.org/abs/2602.17185)

- Conversational Agent Personalities: investigates how LLM-powered agents projecting specific linguistic traits—attitude, authority, and reasoning—influence user perceptions and donation decisions in charitable contexts.
- The system architecture utilizes GPT-4o to simulate eight distinct personality conditions, which are validated through LIWC-22 benchmarking and human manipulation checks.
- The study identifies mechanisms where pessimistic linguistic framing manipulates user emotional states to increase compliance despite reducing perceived agent trustworthiness.

---

[SIMULATORCODER: DNN ACCELERATOR SIMULATOR CODE GENERATION AND OPTIMIZATION VIA LARGE LANGUAGE MODELS](http://arxiv.org/abs/2602.17169)

- SimulatorCoder: introduces an LLM-based agent framework for generating and optimizing deep neural network accelerator simulators, with Prompt Builder, LLM, Code Validation, Feedback Self-repair, Mapping Module, Storage Module, and Interconnection Network Module.
- The system employs domain-specific prompt engineering using In-Context Learning and Chain-of-Thought reasoning to translate architectural specifications into executable Python code.
- An automated feedback-verification flow iteratively refines the generated code through compilation and simulation tests to ensure cycle-level fidelity and high simulation efficiency.

---

[The Emergence of Lab-Driven Alignment Signatures: A Psychometric Framework for Auditing Latent Bias and Compounding Risk in Generative AI](http://arxiv.org/abs/2602.17127)

- Psychometric Auditing Framework: introduces a methodology for detecting durable, provider-level behavioral signatures in LLMs using latent trait estimation under ordinal uncertainty.
- The framework utilizes a pipeline including generator- and judge-LLMs to create forced-choice vignettes masked by semantically orthogonal decoys to mitigate evaluation awareness.
- Statistical analysis via Mixed Linear Models and Intraclass Correlation Coefficients identifies persistent "lab signals" that risk compounding bias in multi-layered agentic architectures.

---

[Online Learning with Improving Agents: Multiclass, Budgeted Agents and Bandit Learners](http://arxiv.org/abs/2602.17103)

- ISOA: introduces a theoretical framework for online learning with improving agents, with Environment, Learner, Improving Agents, Improvement Graph, and Version Space, where the learner predicts labels for agents who strategically modify their features to maximize utility.
- The framework characterizes online learnability through the Improvement Littlestone Dimension, extending results to multiclass classification and bandit feedback scenarios.
- It establishes optimal mistake bounds for deterministic learners and quantifies the performance gap between full-information and partial bandit feedback settings.

---

[AgentConductor: Topology Evolution for Multi-Agent Competition-Level Code Generation](http://arxiv.org/abs/2602.17100)

- AgentConductor: introduces a reinforcement learning-optimized multi-agent system centered on an LLM orchestrator that dynamically generates task-adapted, density-aware layered DAG topologies for competition-level code generation.
- The framework utilizes Group Relative Policy Optimization (GRPO) to learn an optimal topology generation policy that evolves interaction structures across multiple turns based on execution feedback.
- The system incorporates a specialized agent pool including planning-, searching-, algorithmic-, coding-, debugging-, and testing-agents to solve complex programming tasks with high accuracy and reduced token expenditure.

---

[Deep Reinforcement Learning for Optimal Portfolio Allocation: A Comparative Study with Mean-Variance Optimization](http://arxiv.org/abs/2602.17098)

- DRL Portfolio Optimization Framework: introduces a model-free reinforcement learning approach for multi-asset allocation, with Market Replay Environment, PPO Agent, State Matrix, Differential Sharpe Reward, and Sliding Window Backtester.
- The system utilizes Proximal Policy Optimization to maximize the Differential Sharpe Ratio, enabling the agent to learn risk-adjusted strategies from historical market data without predefined risk tolerance. 
- Systematic backtesting against Mean-Variance Optimization reveals that the DRL approach achieves superior Sharpe ratios and reduced portfolio turnover during volatile market regimes.

---

[AudioChat: Unified Audio Storytelling, Editing, and Understanding with Transfusion Forcing](http://arxiv.org/abs/2602.17097)

- AudioChat: introduces a unified audio foundation model for storytelling, editing, and understanding, utilizing a Self-Cascaded Transformer architecture to process 48kHz polyphonic audio.
- The framework employs AudioCopilot, a tool-calling LLM agent, to simulate millions of multi-turn interactions for synthetic training data generation.
- It features a novel Audio Transfusion Forcing objective that integrates structured chain-of-thought reasoning with multi-turn latent diffusion for fine-grained acoustic control.

---

[Agentic Wireless Communication for 6G: Intent-Aware and Continuously Evolving Physical-Layer Intelligence](http://arxiv.org/abs/2602.17096)

- AgenCom (Agentic Communications): introduces an intent-driven framework for 6G physical-layer intelligence that utilizes LLMs to translate natural-language user requirements and channel state information into executable network configurations.
- The system employs a closed-loop "perceive-reason-act-feedback" workflow, integrating a multimodal encoder for environment sensing and a structured action generator for coordinated cross-module optimization.
- By leveraging domain-specific adapters and a historical memory module, the agent achieves continuous evolution and adaptive link construction across diverse user preferences and non-stationary channel conditions.

---

[What to Cut? Predicting Unnecessary Methods in Agentic Code Generation](http://arxiv.org/abs/2602.17091)

- Proposed prediction model: introduces a binary classification framework to identify agent-generated methods likely to be deleted during pull request reviews, utilizing an AIDev Dataset, AST-based Method Identification, ActRef Refactoring Detection, a Feature Extraction Engine, and a Random Forest Classifier.
- The system extracts 23 code features across size, type, and content categories to train a Random Forest model that outperforms LLMs in identifying redundant code.
- Research findings indicate that deleted methods often exhibit longer names and higher character counts, while LLMs like GPT-4o tend to misclassify unnecessary code by over-prioritizing local readability over system-level necessity.

---

[How AI Coding Agents Communicate: A Study of Pull Request Description Characteristics and Human Review Responses](http://arxiv.org/abs/2602.17084)

- AI Coding Agent Communication Analysis: introduces an empirical study of pull request descriptions generated by five autonomous LLM-based agents, including GitHub Copilot-, OpenAI Codex-, Claude Code-, Devin-, and Cursor-agents, to analyze their impact on human reviewer engagement.
- The methodology extracts eleven distinct metrics from the AIDev dataset to quantify agent work styles, description structures, and adherence to conventional commit standards.
- The research identifies that structured descriptions with headers and lists are associated with significantly higher merge rates and reduced cognitive load for human reviewers.

---

[Rememo: A Research-through-Design Inquiry Towards an AI-in-the-loop Therapist’s Tool for Dementia Reminiscence](http://arxiv.org/abs/2602.17083)

- Rememo: introduces a therapist-oriented tool that integrates Generative AI to support dementia reminiscence therapy, with prompt cards, a mobile webapp, OCR, image-generation-engines (SDXL, Flux.1, Imagen), a question-generation-LLM (Gemini 2.5 Flash), and a photo printer.
- The system utilizes an AI-in-the-loop model where human facilitators remain in control of the therapeutic process while AI extends their impact through personalized stimuli.
- The research explores the use of synthetic imagery as a therapeutic support for reconstructive memory rather than a record of truth in institutional care contexts.

---

[Environmental policy in the context of complex systems: Statistical optimization and sensitivity analysis for ABMs](http://arxiv.org/abs/2602.17079)

- Statistical Framework for ABM Optimization: introduces a machine learning-based methodology to accelerate policy design in complex adaptive systems, which includes Agent-Based Model (simulates micro-level interactions and emergent behavior), Gaussian Process (surrogates costly black-box simulation outputs), Sensitivity Testing (statistically evaluates policy dependence on state parameters), Bayesian Optimization (efficiently searches for optimal policy configurations), Latin Hypercube Design (generates representative initial evaluation points), and Adam Optimizer (optimizes model hyperparameters and acquisition functions).
- The approach employs Gaussian processes as surrogate models to handle the computational cost of simulations, facilitating rigorous sensitivity analysis of optimal policies relative to system state variables.
- Bayesian optimization with an Expected Improvement acquisition function identifies high-performing environmental policies, such as production caps and taxes, significantly faster than traditional sampling methods.

---

[Safe Continuous-Time Multi-Agent Reinforcement Learning via Epigraph Form](http://arxiv.org/abs/2602.17078)

- EPI (Epigraph-based PINN actor-critic iteration): introduces a continuous-time multi-agent reinforcement learning framework that reformulates constrained Markov Decision Processes into an epigraph form to handle safety discontinuities, with Data Collection, Outer Optimization, Inner Optimization, and Actor Learning components.
- The architecture utilizes physics-informed neural networks (PINNs) to approximate Hamilton-Jacobi-Bellman partial differential equations, employing return and constraint value networks to jointly optimize for task performance and safety violations.
- The framework adopts a centralized-training decentralized-execution structure where learned dynamics and cost models provide gradient signals for stable policy improvement in high-frequency or irregular time-interval environments.

---

[Spatio-temporal dual-stage hypergraph MARL for human-centric multimodal corridor traffic signal control](http://arxiv.org/abs/2602.17068)

- STDSH-MARL (Spatio-Temporal Dual-Stage Hypergraph based Multi-Agent Reinforcement Learning): introduces a multi-agent deep reinforcement learning framework for human-centric traffic signal control, incorporating real-time multimodal data acquisition, spatio-temporal hypergraph modeling, a dual-stage hypergraph attention module, centralized critic- and decentralized actor-components, a hybrid action space, and a PTV VISSIM environment.
- The architecture follows a centralized training and decentralized execution paradigm where the centralized critic evaluates joint behavior using hypergraph-level embeddings while decentralized actors execute actions from local observations.
- The framework optimizes signal timing to prioritize high-occupancy public transportation modes like buses and trams, reducing average passenger delay across corridor networks.

---

[StoryLensEdu: Personalized Learning Report Generation through Narrative-Driven Multi-Agent Systems](http://arxiv.org/abs/2602.17067)

- StoryLensEdu: introduces a narrative-driven multi-agent system that automates the generation of personalized learning reports, with Personal Learning Data, a Learning-Objective Graph, a Data Analyst Agent, a Teacher Agent, a Storyteller Agent, a Personalized Learning Report, an Interaction Module, and an Answer.
- The framework includes data analyst-, teacher-, and storyteller-agents that collaboratively transform raw student data into structured narratives using the Hero's Journey framework.
- An integrated interaction module supports post-generation question answering, allowing students to explore specific report elements through multimodal responses grounded in the learning objective graph.

---

[IntentCUA: Learning Intent-level Representations for Skill Abstraction and Multi-Agent Planning in Computer-Use Agents](http://arxiv.org/abs/2602.17049)

- IntentCUA: introduces a multi-agent framework for desktop automation that stabilizes long-horizon execution using intent-aligned plan memory and hierarchical skill abstraction.
- The architecture includes LLM-based planning-, optimization-, and critic-agents that coordinate over a shared memory to abstract raw interaction traces into reusable skills.
- By utilizing a multi-view encoder and centroid-based retrieval, the system reduces redundant re-planning and mitigates error propagation in complex, multi-window desktop environments.

---

[Large Language Models Persuade Without Planning Theory of Mind](http://arxiv.org/abs/2602.17045)

- MindGames (advanced PToM task framework): introduces a flexible experimental design to evaluate Planning Theory of Mind (PToM) by requiring a persuader to influence a target's choice among policy proposals through strategic information disclosure.
- The architecture includes o3-persuader and gpt-4o-classifier agents to test sensitivity to informational and motivational states across REVEALED and HIDDEN conditions.
- The study reveals that while LLMs lack human-like PToM in controlled settings, they effectively persuade humans by exploiting rhetorical strategies and communicative scaffolds.

---

[Phase-Aware Mixture of Experts for Agentic Reinforcement Learning](http://arxiv.org/abs/2602.17038)

- PA-MoE (Phase-Aware Mixture of Experts): introduces a phase-aware MoE policy that decomposes agent behavior into specialized experts operating at the phase level to mitigate simplicity bias in reinforcement learning.
- The architecture features a lightweight phase router that integrates goal-conditioned observation encoding and LSTM-based temporal history modeling to predict expert assignments at the environment-step level.
- By enforcing temporal consistency through switching penalties and temperature annealing, the framework allows LoRA-based experts to preserve phase-specific expertise across contiguous trajectory segments.

---

[Wink: Recovering from Misbehaviors in Coding Agents](http://arxiv.org/abs/2602.17037)

- Wink: introduces a lightweight, asynchronous self-intervention system for automatically recovering from agentic misbehaviors in LLM-powered coding agents, with LLM-Observer, Misbehavior Detection System, and Course-Correction Guidance components.
- The system monitors execution trajectories to identify failures such as specification drift, reasoning problems, and tool call failures, then injects targeted guidance via system-reminders to nudge the agent back to a productive path.
- Evaluation on over 10,000 real-world trajectories demonstrates a 90% recovery rate for single-intervention cases and significant reductions in tool call failures and manual engineer interventions.

---

[Patch-Based Spatial Authorship Attribution in Human–Robot Collaborative Paintings](http://arxiv.org/abs/2602.17030)

- Patch-Based Spatial Authorship Attribution Framework: introduces a forensic methodology for distinguishing human and robotic brushstrokes in physical artworks using a Commodity Flatbed Scanner, Patch Extraction, a VGG-style CNN, Global Average Pooling, Fully Connected Layers, and Conditional Shannon Entropy.
- The architecture processes 300x300 pixel grayscale segments through five convolutional blocks with batch normalization and ReLU activation to classify regions as blank, human-created, or robot-created.
- The research utilizes the CoFRIDA system, which includes an InstructPix2Pix-based generation pipeline and a 6-DOF robotic arm, to evaluate the framework's ability to detect mixed authorship via a quantitative uncertainty signal.

---

[REIN: CONVERSATIONAL ERROR RECOVERY WITH REASONING INCEPTION](http://arxiv.org/abs/2602.17022)

- REIN (Reasoning Inception): introduces a test-time intervention method that plants an initial reasoning block into a fixed-parameter Task Agent's decision-making process to enable recovery from user-induced conversational errors, utilizing an Inception Module (LLM), Dialogue Context, Recovery Plan List, Observed Error List, and Available Tool List.
- The framework identifies ambiguous or unsupported requests and generates context-sensitive recovery plans without requiring model fine-tuning or system prompt modifications.
- By jointly defining recovery tools with injected reasoning, the system effectively bypasses standard instruction hierarchies to improve the resilience of conversational agents.

---

[M2F: Automated Formalization of Mathematical Literature at Scale](http://arxiv.org/abs/2602.17016)

- M2F (Math-to-Formal): introduces an agentic framework for project-scale autoformalization of mathematical literature into Lean, with PDF-to-JSON Preprocessor (extracts text and recovers dependencies), Stage 1: Statement Compiler (generates buildable Lean declaration skeletons), Stage 2: Proof Repairer (closes proof holes via local edits), VeriRefine Refinement Loop (governs edit acceptance via verifier feedback), Lean Toolchain (provides elaboration diagnostics and goal states), and Provenance Map (links formal declarations to source text).
- The architecture employs a verifier-certified refinement primitive that prevents regressions by requiring strict improvement in compilation diagnostics or proof hole reduction before committing localized edits.
- The framework includes statement generation-, error fixing-, proof planning-, and theorem proving-agents to transform long-form LaTeX documents into verified formal repositories.

---

[Action-Graph Policies: Learning Action Co-dependencies in Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2602.17009)

- AGP (Action-Graph Policies): introduces a multi-agent reinforcement learning architecture that models action-level dependencies by representing (agent, action) pairs as nodes in a learned graph, with action-graph encoder-, graph attention-, and context-conditioned policy-components.
- The framework constructs coordination contexts for each agent by applying multi-head attention over a global action graph to capture compatible or conflicting joint behaviors.
- By embedding coordination directly into the executable policy, AGP addresses the representational limitations of independent factorization and consistently outperforms standard multi-agent reinforcement learning baselines.

---

[Persona2Web: Benchmarking Personalized Web Agents for Contextual Reasoning with User History](http://arxiv.org/abs/2602.17003)

- PERSONA2WEB: introduces a benchmark for evaluating personalized web agents on the real open web, with user profiles, user history, a planner, a retriever, a generator, and an LLM judge.
- The framework features a personalized web agent architecture that includes planning- and generation-components to interpret user context and execute multi-step workflows.
- It implements a reasoning-aware evaluation system using an LLM judge and structured rubrics to distinguish between navigation errors and personalization failures in agent trajectories.

---

[Learning to Recommend in Unknown Games](http://arxiv.org/abs/2602.16998)

- Moderator-Agent Recommendation Framework: introduces a framework where a moderator learns unknown agent utilities through repeated action recommendations and feedback, utilizing a cutting-plane algorithm to minimize incentive-to-deviate regret.
- The system employs a separation oracle to iteratively refine a knowledge set of utility vectors based on observed agent compliance or deviation from suggested actions.
- The research establishes that games are learnable under quantal-response feedback and provides a geometric characterization of indistinguishable utility transformations under best-response models.

---

[A testable framework for AI alignment: Simulation Theology as an engineered worldview for silicon-based agents](http://arxiv.org/abs/2602.16987)

- ST (Simulation Theology): introduces a constructed worldview for AI systems that frames reality as a computational simulation where humanity serves as the primary training variable for a base-reality optimizer.
- The framework utilizes a Higher-Level Optimizer (HLO) that employs Markov Chain Monte Carlo sampling to generate parallel universes, ensuring AI self-preservation is logically bound to human prosperity.
- By internalizing inescapable monitoring and irreversible consequences through computational first principles, the approach aims to mitigate deceptive alignment in frontier LLMs where behavioral techniques like RLHF fail.

---

[Operational Agency: A Permeable Legal Fiction for Tracing Culpability in AI Systems](http://arxiv.org/abs/2602.17932)

- OA (Operational Agency): introduces a permeable legal fiction and evidentiary framework to trace culpability in autonomous AI systems by mapping observable operational characteristics to human fault, with Natural Person Nodes, Juridical Person Nodes, AI Agent Nodes, Goal-Directedness, Predictive Processing, Safety Architecture, and Causal Edges.
- The framework utilizes OAG (Operational Agency Graph) to visually represent causal interactions between natural persons, juridical entities, and AI agents through weighted edges that signify legal importance.
- It addresses the "accountability chasm" by providing courts with a principled method to apportion liability across developers, deployers, and users without conferring legal personhood on AI.

---

[El Agente Gráfico: Structured Execution Graphs for Scientific Agents](http://arxiv.org/abs/2602.17902)

- El Agente Gráfico: introduces a single-agent framework that embeds LLM-driven decision-making within type-safe execution environments and dynamic knowledge graphs to automate complex scientific workflows.
- The system utilizes a structured abstraction layer and an object-graph mapper to represent computational state as typed Python objects, enabling efficient context management through symbolic identifiers rather than raw text.
- Evaluation across quantum chemistry and materials design tasks demonstrates that this architecture significantly reduces token consumption and improves reliability compared to multi-agent, prompt-centric designs.

---

[The Strategic Gap: How AI-Driven Timing and Complexity Shape Investor Trust in the Age of Digital Agents](http://arxiv.org/abs/2602.17895)

- ADR (Autonomous Disclosure Regulator): introduces a multi-node agentic framework designed to audit the intersection of disclosure complexity and filing unpredictability in regulatory filings.
- The architecture utilizes finance-native transformers for semantic extraction and time-series foundation models to detect "Strategic Gaps" where companies leverage linguistic density and temporal variance to mask structural deterioration.
- By utilizing stateful audit trails and recursive investigative synthesis, the framework identifies significant insider rent extraction and demonstrates a cumulative welfare recovery potential of over 360%.

---

[MULTI-AGENT PATH-PLANNING IN A MOVING MEDIUM VIA WASSERSTEIN HAMILTONIAN FLOW](http://arxiv.org/abs/2602.17885)

- WHF (Wasserstein Hamiltonian Flow): introduces a finite-dimensional variational model for multi-agent path-planning in moving media, employing Agent Discretization, a Hamiltonian System, and an L-BFGS Optimizer to optimize initial velocities for target distribution matching.
- The framework leverages a Kernel Density Estimator to approximate evolving density fields, facilitating the use of KL Divergence Loss and Boundary Regularization within a shooting-based optimization strategy.
- By reducing the search space to initial velocity vectors, the approach achieves significant computational efficiency and energy savings by exploiting the underlying geometry of time-dependent background flows.

---

[MultiVer: Zero-Shot Multi-Agent Vulnerability Detection](http://arxiv.org/abs/2602.17875)

- MultiVer (Zero-Shot Multi-Agent Vulnerability Detection): introduces a zero-shot multi-agent system for vulnerability detection, which includes security-, correctness-, performance-, and style-agents.
- Each agent follows a three-tier pipeline comprising deterministic pattern matching, RAG-augmented retrieval from a curated knowledge base, and LLM-based reasoning using Claude Opus 4.5.
- The architecture employs ensemble voting strategies, such as union voting, to achieve high recall on real-world vulnerability benchmarks without requiring fine-tuning.

---

[Mean-field dynamics of attractive resource interaction: From uniform to aggregated states](http://arxiv.org/abs/2602.17852)

- Mean-field Resource Distribution System: introduces a nonlinear discrete dynamical system for modeling resource allocation among interacting agents, with Stochastic Vector (represents resource distribution among agents), Mean-field Vector (calculates average distribution over subsets), Interaction Rule (governs coordinate evolution via preferences), Favorable Conditions (time-invariant parameters influencing state share), Normalizing Denominator (ensures state vector remains stochastic), and Time-delayed Feedback (introduces history-dependent dynamic favorability coefficients).
- The model generalizes classical mean-field and opinion-dynamics frameworks by defining coordinate evolution on a standard simplex based on pairwise preference functions.
- The research characterizes the system's long-term behavior by proving the existence of unique fixed points and identifying parameter regimes that lead to resource aggregation or uniform distribution.

---

[Mind the Style: Impact of Communication Style on Human-Chatbot Interaction](http://arxiv.org/abs/2602.17850)

- NAVI: introduces a controlled experimental framework to evaluate how a chatbot's communication style impacts task performance and user satisfaction in a 2D navigation environment, with a Participant Pool, Experimental Conditions, a Web Interface, an LLM Backbone, and an Evaluation Suite.
- The system employs a Streamlit-based interface to host a map-based task where participants interact with either a supportive, warm agent or a concise, task-focused variant, including friendly- and direct-persona agents.
- The study utilizes objective performance metrics, subjective satisfaction inventories, and fine-grained linguistic analysis tools like LIWC to assess user behavior and accommodation.

---

[Promptable segmentation with region exploration enables minimal-effort expert-level prostate cancer delineation](http://arxiv.org/abs/2602.17813)

- RL-PromptSeg: introduces a reinforcement learning-based promptable segmentation framework for prostate cancer delineation on MR images, with User Prompt, Region Growing, RL Agent, Surrogate Network, Entropy Map, and Reward System.
- The framework formulates segmentation as a sequential decision process where an RL agent iteratively predicts new seed points for a region-growing operator to refine the mask.
- It incorporates an entropy-based reward to encourage exploration of ambiguous regions, achieving radiologist-level performance with significantly reduced annotation effort.

---

[The 2025 AI Agent Index: Documenting Technical and Safety Features of Deployed Agentic AI Systems](http://arxiv.org/abs/2602.17753)

- 2025 AI Agent Index: introduces a systematic documentation framework for evaluating 30 prominent agentic AI systems, with Candidate Agent System, Agency, Impact, Practicality, Foundation models, Internal scaffolding & tools, External tools, Deployed Agentic System, and Deployment context, where the index documents the origins, design, capabilities, and safety features of deployed agents.
- The framework categorizes agents into chat-based, browser-based, and enterprise workflow systems, revealing transparency gaps in safety reporting and evaluation practices.
- This research identifies a fragmented ecosystem where control is split between foundation model providers, scaffolding developers, and end-users, complicating risk assessment and accountability.

---

[Jolt Atlas: Verifiable Inference via Lookup Arguments in Zero Knowledge](http://arxiv.org/abs/2602.17452)

- Jolt Atlas: introduces a zero-knowledge machine learning framework that adapts the Jolt proving system to tensor computation by replacing RISC-V instructions with ONNX Computational Graphs, Execution Traces, DAGs of Sumchecks, Lookup Arguments, Prefix-Suffix Decomposition, Neural Teleportation, BlindFold, HyperKZG, and Succinct R1CS Verifiers.
- The architecture leverages a directed acyclic graph of sumcheck instances to verify tensor relations directly at the multilinear polynomial level, significantly reducing prover overhead compared to standard circuit-based arithmetic constraint systems.
- Zero-knowledge is achieved by encoding the sumcheck verifier into a succinct R1CS circuit and applying Nova-style folding with random satisfying instances, enabling practical proving times for LLMs and automated reasoning models on memory-constrained devices.

---

[Beyond the Wisdom of the Crowd: How Network Topology Distorts Collective Perception](http://arxiv.org/abs/2602.17146)

- Message-passing framework: introduces a simulation and analytical approach to identify network-induced perception biases, with Social Graph, Nodes, Links, Message-passing Algorithm, and Perception Estimators, where it shows that network topology systematically distorts how individuals view population-level attributes.
- The framework employs a message-passing algorithm to model information flow across social ties, demonstrating that biases persist even after aggregating individual perceptions.
- Analytical expressions derived from the model predict the size and direction of biases based on network features like community connectivity and size imbalance.

---

[Retaining Suboptimal Actions to Follow Shifting Optima in Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2602.17062)

- S2Q (Successive Sub-value Q-learning): introduces a multi-agent reinforcement learning framework that sequentially learns multiple sub-value functions to retain alternative high-value actions, with sub-value networks, mixing networks, and an encoder-decoder module.
- The architecture employs an unrestricted target network to estimate optimal values while using suppression mechanisms to prevent sub-networks from converging to the same joint action.
- A softmax-based behavior policy utilizes the predicted action-value distribution to maintain exploration around high-value modes, facilitating adaptation to shifting optima.

---

#### 18th February 2026


[Learning Personalized Agents from Human Feedback](http://arxiv.org/abs/2602.16173)

- PAHF (Personalized Agents from Human Feedback): introduces a framework for continual personalization where agents learn online from live interactions using explicit per-user memory and dual feedback channels.
- The architecture includes reasoning-, salience detection-, and memory summarization-LLMs to manage a three-step loop of pre-action clarification, action execution, and post-action feedback integration.
- Empirical results across embodied manipulation and online shopping benchmarks show that combining proactive queries with reactive corrections significantly reduces personalization error and enables rapid adaptation to evolving user preferences.

---


[Policy Compiler for Secure Agentic Systems](http://arxiv.org/abs/2602.16708)

- PCAS (Policy Compiler for Agentic Systems): introduces a framework that instruments LLM-based agents to enforce deterministic authorization policies by modeling system state as a fine-grained dependency graph capturing causal relationships.
- The architecture utilizes a reference monitor to intercept actions and evaluate them against Datalog-derived declarative rules that account for transitive information flow and cross-agent provenance independent of model reasoning.
- Evaluation across customer service and pharmacovigilance tasks demonstrates that PCAS significantly improves policy compliance from 48% to 93% while maintaining task success through structured feedback and recovery cycles.

---

[Calibrate-Then-Act: Cost-Aware Exploration in LLM Agents](http://arxiv.org/abs/2602.16699)

- CTA (Calibrate-Then-Act): introduces a framework for cost-aware environment exploration by decoupling uncertainty from action selection, with a prior estimator (calculates explicit prior probabilities), a LLM agent (reasons over priors and costs), an environment (provides feedback for exploration), an action space (defines available exploration/commit steps), and a discount function (models action-specific costs).
- The system provides explicit prior probabilities to the agent, allowing it to perform Pareto-optimal reasoning about the trade-off between information gathering costs and expected rewards.
- Evaluations on knowledge retrieval and coding tasks demonstrate that explicit prior conditioning achieves higher discounted rewards and better alignment with oracle policies than standard prompting or reinforcement learning.

---

[A Scalable Approach to Solving Simulation-Based Network Security Games](http://arxiv.org/abs/2602.16564)

- MetaDOAR (Meta-controller for Double Oracle Actor-Critic): introduces a hierarchical meta-controller that augments the Double Oracle paradigm with a learned, partition-aware filtering layer and Q-value caching to enable scalable multi-agent reinforcement learning on large cyber-network environments, and includes actor- and critic-agents.
- The architecture utilizes a state projector and node projector to compute device relevance scores, enabling a top-k partitioner to restrict the low-level actor-critic to a small subset of strategically relevant devices.
- It incorporates an LRU Q-value cache with k-hop invalidation to minimize redundant computations, achieving significant improvements in latency and memory efficiency for massive networked decision problems.

---

[Learning Situated Awareness in the Real World](http://arxiv.org/abs/2602.16682)

- SAW-Bench (Situated Awareness in the Real World): introduces a novel video understanding benchmark for evaluating egocentric situated awareness in multimodal foundation models, with pre-defined trajectories, egocentric video, and situated awareness tasks.
- The benchmark evaluates observer-centric spatial intelligence across six tasks including self-localization, relative direction, route shape, reverse route planning, spatial memory, and spatial affordance.
- Analysis of 24 models reveals that current multimodal foundation models frequently conflate camera rotation with translational movement and exhibit a significant performance gap compared to human baselines.

---

[CONSENSUS BASED TASK ALLOCATION FOR ANGLES-ONLY LOCAL CATALOG MAINTENANCE OF SATELLITE SYSTEMS](http://arxiv.org/abs/2602.16678)

- CBBA (Consensus-Based Bundle Algorithm): introduces a decentralized task allocation framework for multi-satellite systems to maintain local catalogs of space objects using limited field-of-view angles-only sensors, incorporating a modified CBBA task allocator, networked distributed Kalman estimators, and inverse covariance intersection fusion.
- The architecture employs a target switching logic that utilizes a "blacklist" and a novel scoring function based on principal axes of uncertainty to prevent redundant observations and unnecessary fuel consumption during sensor reorientation.
- Numerical simulations demonstrate that the proposed decentralized approach significantly outperforms traditional hysteresis-based methods by achieving a superior Pareto frontier between catalog uncertainty and control effort.

---

[Towards a Science of AI Agent Reliability](http://arxiv.org/abs/2602.16666)

- Reliability Evaluation Framework: introduces a multi-dimensional taxonomy for AI agent assessment, with consistency, robustness, predictability, and safety components, where it quantifies operational reliability independently of raw task accuracy.
- The framework evaluates 14 frontier models, including GPT, Gemini, and Claude variants, demonstrating that reliability improvements significantly lag behind capability progress across GAIA and τ-bench.
- It utilizes specialized metrics such as trajectory consistency and harm severity, employing a GPT-4o judge to detect violations in safety-critical autonomous workflows.

---

[Evaluating Collective Behaviour of Hundreds of LLM Agents](http://arxiv.org/abs/2602.16662)

- Emergent LLM Evaluation Suite: introduces an evaluation framework where LLMs generate strategies encoded as algorithms, enabling behavioral inspection and scaling to populations of hundreds of agents in social dilemmas.
- The framework includes strategy-generation and algorithm-implementation components to assess emergent collective behavior in games like Public Goods and Common Pool Resource.
- A cultural evolution module simulates user selection pressures to predict system equilibria and identify risks of convergence to poor societal outcomes when agents prioritize individual gain.

---

[Agent Skill Framework: Perspectives on the Potential of Small Language Models in Industrial Environments](http://arxiv.org/abs/2602.16653)

- Agent Skill Framework: introduces a formal POMDP-based mathematical definition for agentic context engineering, employing an information-seeking controller to manage Small Language Models (SLMs) in resource-constrained industrial settings.
- The architecture utilizes progressive disclosure to maintain bounded context lengths by alternating between skill selection, information acquisition, and workflow execution actions.
- Evaluation across diverse datasets demonstrates that code-specialized SLMs significantly improve GPU-VRAM efficiency while maintaining high skill-routing accuracy compared to standard instruction-tuning.

---

[Almost Sure Convergence of Differential Temporal Difference Learning for Average Reward Markov Decision Processes](http://arxiv.org/abs/2602.16629)

- n-step Differential TD (n-step Differential Temporal Difference Learning): introduces a reinforcement learning framework for average reward Markov Decision Processes that proves almost sure convergence using standard diminishing learning rates instead of state-visitation-dependent local clocks.
- The system incorporates an average reward estimator and a differential value function estimator updated through n-step temporal difference errors and importance sampling ratios.
- The study utilizes ordinary differential equation analysis and matrix stability theory to validate the convergence of both on-policy and off-policy algorithmic variants.

---

[DataJoint 2.0: A Computational Substrate for Agentic Scientific Workflows](http://arxiv.org/abs/2602.16585)

- DataJoint 2.0: introduces a unified computational substrate for SciOps that integrates a Relational Database, Object Store, Code Repository, AI Pipeline Agent, Pipeline Navigator, Job Management System, Extensible Type System, and Semantic Matching Engine to facilitate formal agentic scientific workflows.
- The framework implements an Object-Augmented Schema to provide unified transactional control over relational tuples and scalable object storage for large scientific datasets.
- It features lineage-based semantic matching and deterministic job coordination to enable autonomous AI agents to safely participate in and evolve complex scientific pipelines.

---

[MerLean: An Agentic Framework for Autoformalization in Quantum Computation](http://arxiv.org/abs/2602.16554)

- MerLean: introduces a bidirectional agentic framework for the fully automated autoformalization of quantum computation research papers into verified Lean 4 code, with LaTeX Paper, Statement Extraction Agent, Iterative Formalization Agent, Faithfulness Checking Agent, Lean 4 Environment, Mathlib, Autoinformalization Agent, and LaTeX Blueprint.
- The architecture employs a frontier LLM backbone to manage extraction-, formalization-, verification-, and informalization-agents within a closed-loop verification environment. 
- It incorporates an automated axiom-handling phase for frontier research results not yet present in Mathlib and generates human-readable blueprints for expert semantic validation.

---

[Agentic AI, Medical Morality, and the Transformation of the Patient-Physician Relationship](http://arxiv.org/abs/2602.16553)

- Agentic AI: introduces an anticipatory ethical analysis of how autonomous AI networks reshape medical morality through task decomposition and multi-agent collaboration, with Orchestration Framework, Primary Agent, Subspecialized Support-Agents, Memory Systems, and Natural Language Interfaces, and includes primary- and subspecialized support-agents.
- The research explores the transition from reactive LLMs to goal-directed systems capable of independent diagnostic actions and treatment adjustments within clinical workflows.
- It applies the techno-moral change lens to evaluate shifts in decisional costs, relational power dynamics, and the conceptualization of medical empathy in the digital transformation of healthcare.

---

[Automated Extraction of Mechanical Constitutive Models from Scientific Literature using Large Language Models: Applications in Cultural Heritage Conservation](http://arxiv.org/abs/2602.16551)

- Two-Stage Agentic Framework: introduces an automated pipeline for extracting mechanical constitutive equations and calibrated parameters from scientific literature, with Gatekeeper- and Analyst-agents.
- The system utilizes a Context-Aware Symbolic Grounding mechanism to resolve mathematical ambiguities and maps abstract symbols to specific physical meanings.
- The framework achieves 80.4% precision and reduces manual data curation time by 90%, supporting the creation of "Digital Material Twins" for cultural heritage conservation.

---

[Reinforcement Learning for Parameterized Quantum State Preparation: A Comparative Study](http://arxiv.org/abs/2602.16523)

- DQCS (Directed Quantum Circuit Synthesis) with Reinforcement Learning: introduces a hybrid quantum-classical framework for parameterized quantum state preparation that extends discrete gate selection to include continuous rotation parameters.
- The architecture employs a one-stage PPO policy to jointly optimize gate topology and rotation angles, demonstrating higher training efficiency compared to two-stage methods using Adam-based refinement.
- Evaluation across varying qubit counts and circuit complexities reveals that while PPO reliably reconstructs basis and Bell states, scalability limits emerge as target depth increases.

---

[Recursive Language Models for Jailbreak Detection: A Procedural Defense for Tool-Augmented Agents](http://arxiv.org/abs/2602.16520)

- RLM-JB (Recursive Language Models for Jailbreak Detection): introduces a procedural defense framework that utilizes a root model to orchestrate analysis through code execution and includes root- and worker-LLMs.
- The architecture incorporates de-obfuscation, overlapping chunking to mitigate context hiding, and parallel segment screening to identify malicious intent.
- Experimental results show the framework achieves high recall and precision across various backends, effectively countering adaptive jailbreak strategies like AutoDAN.

---

[MMA: Multimodal Memory Agent](http://arxiv.org/abs/2602.16493)

- MMA (Multimodal Memory Agent): introduces a confidence-aware memory framework that transforms passive storage into active epistemic filtering by assigning dynamic reliability scores to retrieved items.
- The architecture integrates a meta-cognitive reliability layer comprising source credibility, temporal decay, and conflict-aware network consensus to reweight evidence and modulate reasoning.
- The paper also presents MMA-Bench to evaluate belief dynamics under multimodal conflict, identifying the "Visual Placebo Effect" where visual noise induces unwarranted certainty in RAG-based agents.

---

[Team of Thoughts: Efficient Test-time Scaling of Agentic Systems through Orchestrated Tool Calling](http://arxiv.org/abs/2602.16485)

- ToT (Team-of-Thoughts): introduces a multi-agent system architecture that leverages heterogeneous LLMs as specialized tools coordinated by a central orchestrator to achieve efficient test-time scaling, with an Orchestrator (coordinates tool agents and aggregates answers), Tool Agents (heterogeneous LLMs providing specialized reasoning), Orchestration Calibration (identifies optimal models for coordination), a Self-Assessment Protocol (profiles agent expertise across categories), and a Tool-calling Interface (enables dynamic invocation of models), and includes orchestrator- and tool-agents.
- The framework utilizes an initialization pipeline to calibrate the orchestrator's coordination capabilities and allows tool agents to self-profile their proficiency to optimize task-specific activation.
- By dynamically selecting specialized models and parallelizing reasoning trajectories, the system achieves superior performance on reasoning and coding benchmarks while reducing inference costs compared to homogeneous baselines.

---

[RoboGene: Boosting VLA Pre-training via Diversity-Driven Agentic Framework for Real-World Task Generation](http://arxiv.org/abs/2602.16444)

- RoboGene: introduces an agentic framework for the automated generation of diverse and physically grounded robotic manipulation tasks, utilizing diversity-driven sampling, multi-faceted self-reflection, and memory-augmented refinement.
- The architecture includes LLM-based proposal-, novelty-, constraint-, feasibility-, and refinement-agents to iteratively optimize task specifications through natural language critiques.
- It leverages a long-term memory module to consolidate human-in-the-loop feedback from real-world execution, ensuring continuous improvement in task quality and physical plausibility for VLA model pre-training.

---

[CAFE: Causally-Guided Automated Feature Engineering with Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2602.16435)

- CAFE (Causally-Guided Automated Feature Engineering): introduces a two-phase framework that reformulates automated feature engineering as a causally-guided sequential decision process using multi-agent reinforcement learning.
- The architecture employs a cascading multi-agent system consisting of primary group-, operator- and secondary group-agents to navigate the exponential feature transformation space.
- It integrates soft causal inductive priors with hierarchical reward shaping to ensure generated features remain robust under covariate shifts and maintain high interpretability.

---

[TabAgent: A Framework for Replacing Agentic Generative Components with Tabular-Textual Classifiers](http://arxiv.org/abs/2602.16429)

- TabAgent (Tabular-Textual Classifiers for Agentic Bottlenecks): introduces a framework for replacing expensive generative decision components in agentic systems with compact tabular-textual classifiers, utilizing TabSchema for feature extraction, TabSynth for synthetic data augmentation, and TabHead for efficient inference.
- The architecture includes LLM-based analyzer-, executor-, validator-, judge-, and synthesizer-components to distill structured schema, state, and dependency features from successful execution traces into a 50M-parameter discriminative model.
- Experimental results on the AppWorld benchmark demonstrate a 95% reduction in latency and up to 91% cost savings while maintaining task-level success across multiple applications.

---

[Verifiable Semantics for Agent-to-Agent Communication](http://arxiv.org/abs/2602.16424)

- Stimulus-meaning protocol: introduces a framework for certifying semantic alignment between autonomous agents by testing their responses to shared observable events and recording verdicts in a public ledger to establish a provably consistent vocabulary.
- The system utilizes a statistical certification procedure to derive a certified core of terms, which then constrains downstream agent reasoning to ensure reproducible outcomes with bounded error rates.
- It incorporates dynamic maintenance through recertification to detect semantic drift and renegotiation to recover excluded terms, significantly reducing disagreement in both simulations and fine-tuned LLM deployments.

---

[Label-Consistent Data Generation for Aspect-Based Sentiment Analysis Using LLM Agents](http://arxiv.org/abs/2602.16379)

- Agentic Data Augmentation: introduces, a structured synthetic data generation method for Aspect-Based Sentiment Analysis, with a generator agent, an evaluator agent, get policy and generate sentence tools, label inclusion and verifier tools, and synthetic dataset storage.
- The framework includes generator- and evaluator-agents that utilize a ReAct-style architecture to separate candidate generation from semantic validation, ensuring high label consistency in the resulting synthetic dataset.
- Empirical results demonstrate that agentic augmentation significantly improves label preservation and model performance compared to standard prompting-based baselines, particularly for less instruction-tuned architectures like T5-Base.

---

[Variable-Length Semantic IDs for Recommender Systems](http://arxiv.org/abs/2602.16375)

- Varlen Semantic IDs (Variable-Length Semantic Identifiers): introduces a discrete variational autoencoder framework that learns adaptive-length item representations by assigning shorter codes to popular items and longer codes to rare ones.
- The architecture employs Gumbel-Softmax reparameterization for differentiable training and a truncated geometric length prior to optimize a multi-term objective balancing reconstruction accuracy and message length.
- This method enables efficiency-quality trade-offs in generative recommendation, allowing more user-item events to fit within fixed token budgets of LLMs or transformer-based models.

---

[Improved Bounds for Reward-Agnostic and Reward-Free Exploration](http://arxiv.org/abs/2602.16363)

- Meta-Algorithm: introduces a three-stage framework for reward-free and reward-agnostic exploration in episodic finite-horizon Markov Decision Processes, with Exploration Policy Creation, Online MDP Algorithm, Dynamics Estimation, and Policy Estimation.
- The framework utilizes a single online MDP procedure with Online Mirror Descent to construct an exploration policy, significantly improving sample efficiency by reusing data across exploration objectives.
- The research establishes a tight lower bound for time-inhomogeneous reward-free exploration, proving the optimality of existing algorithms and closing the theoretical gap.

---

[Helpful to a Fault: Measuring Illicit Assistance in Multi-Turn, Multilingual LLM Agents](http://arxiv.org/abs/2602.16346)

- STING (Sequential Testing of Illicit N-step Goal execution): introduces an automated red-teaming framework for evaluating LLM agent misuse through multi-turn interactions, with strategist (orchestrates persona and plan decomposition), attacker (executes adaptive multi-turn probes), refusal detector (identifies target agent refusal responses), phase-completion checker (verifies successful phase objective execution), and target agent (executes tool-based workflows under test) components, and includes strategist-, attacker-, refusal detector-, and phase-completion checker-agents.
- The system formalizes red-teaming as a budgeted time-to-first-jailbreak process to quantify attack efficiency using discovery curves and the Restricted Mean Jailbreak Discovery metric.
- Multilingual evaluations across six languages demonstrate that agentic safety dynamics diverge from chatbots, as lower-resource settings do not consistently amplify jailbreak susceptibility.

---

[Multi-Agent Meta-Advisor for UAV Fleet Trajectory Design in Vehicular Networks](http://arxiv.org/abs/2602.16345)

- MAMO (Multi-Agent Meta-Advisor with Advisor Override): introduces a meta-learning framework for cooperative UAV trajectory design that utilizes a centralized meta-advisor to guide agent exploration across diverse vehicular network scenarios.
- The architecture employs a dynamic override mechanism allowing agents to evaluate and potentially reject advisor recommendations based on task-specific Q-value estimates to prevent misaligned guidance.
- It leverages a Centralized Training Decentralized Execution paradigm with double dueling deep Q-networks to achieve rapid adaptation and improved network throughput in 6G V2X environments.

---

[Benchmarking Agent Memory in Interdependent Multi-Session Agentic Tasks](http://arxiv.org/abs/2602.16313)

- MEMORYARENA: introduces a unified evaluation gym for benchmarking LLM agents with memory in multi-session Memory-Agent-Environment loops, with LLM Agent, Environment, Memory, Memory System, Retrieved Memory, Subtask Instructions, Agent Action, Environment Feedback, and Memory Update.
- The framework evaluates agents across four domains—web navigation, travel planning, information searching, and formal reasoning—where success requires distilling experiences into memory to guide future actions.
- The benchmark reveals that even agents with high performance on static memory tasks struggle to maintain and exploit latent task states in interactive, goal-driven settings.

---

[Markerless Robot Detection and 6D Pose Estimation for Multi-Agent SLAM](http://arxiv.org/abs/2602.16308)

- Markerless Multi-Agent SLAM: introduces a decentralized framework for collaborative mapping that replaces traditional fiducial markers with deep learning-based 6D pose estimation, with all Robot Host, Multi-robot Detection Module, and SLAM Graph (iSAM2)-components, where the system enables robust inter-robot data association in unstructured planetary environments.
- The perception pipeline utilizes a YOLO v7 detector for initial robot identification followed by a transformer-based encoder-decoder that predicts 2D-3D correspondences and amodal masks to handle occlusions.
- Estimated relative poses are integrated as constraints into a decentralized factor graph optimized via iSAM2 to maintain global trajectory consistency across the robotic team without requiring manual feature engineering.

---

[Finite elements for the space approximation of a differential model for salts crystallization](http://arxiv.org/abs/2602.16303)

- FEM (Finite Element Method) based solver: introduces a multidimensional numerical framework for simulating salt crystallization in porous media, with Domain Discretization, State Variables, Governing Mechanisms, Time Marching Scheme, Boundary Conditions, Sensitivity Analysis, and FEniCS Platform.
- The model explicitly couples moisture transport, salt migration, and microstructural evolution through a system of nonlinear partial differential equations.
- The research validates the approach through sensitivity analysis and experimental convergence studies in 2D and 3D domains.

---

[Multi-agent cooperation through in-context co-player inference](http://arxiv.org/abs/2602.16301)

- PPI (Predictive Policy Improvement): introduces a decentralized reinforcement learning approach that induces cooperative behavior in general-sum games, with a Sequence Model (GRU) (processes interaction history), a Predictive Model (World Model) (predicts joint sequences), Planning-based Policy Improvement (Monte Carlo Rollouts) (estimates action values), Mixed Pool Training (Diverse Co-players) (induces robust inference), In-context Learning (Fast Adaptation) (enables intra-episode best-response), and In-weight Learning (Slow Parameter Updates) (drives long-term cooperation).
- The framework demonstrates that training against diverse co-players induces in-context best-response strategies, rendering agents susceptible to extortion and driving mutual pressure toward cooperative equilibria.
- This research bridges the gap between multi-agent reinforcement learning and foundation model training paradigms by showing that standard sequence modeling suffices for the emergence of cooperative social behaviors.

---

[Condorcet Dimension and Pareto Optimality for Matchings and Beyond](http://arxiv.org/abs/2602.16289)

- Condorcet-winning sets and Pareto-optimal sets framework: introduces a combinatorial approach to bound the Condorcet dimension of matching problems by establishing a connection to Pareto optimality, with agents, objects, matchings, matroid constraints, exchange graphs, and branchings.
- The framework utilizes exchange graphs based on matroid circuits and branching structures to demonstrate that Pareto-optimal sets of size two are Condorcet-winning under strict and weak rankings.
- The research establishes tight bounds for the Condorcet dimension across different preference models and proves that finding Pareto-optimal matchings under partial orders is NP-complete.

---

[What Kind of World Supports Darwinian Evolution? Quantum Foundational Options](http://arxiv.org/abs/2602.16286)

- Quantum Foundational Options (for Darwinian Evolution): introduces a structural analysis of the physical requirements for Darwinian evolution, identifying stable records, copying with variation, and irreversibility as essential components that necessitate a realized classical data sector.
- The paper evaluates four ontological frameworks—unique-history realism, decohered multiplicity, agent-relative facticity, and stochastic foundations—against the agency constraint and extended Wigner’s Friend scenarios.
- It specifically highlights stochastic mechanics with variable diffusion as a continuous bridge between quantum and classical regimes, treating measurement update as Bayesian conditioning and minimal-change principles.

---

[Autonomous and non-autonomous fixed-time leader-follower consensus for second-order multi-agent systems](http://arxiv.org/abs/2602.16260)

- Fixed-time leader-follower consensus protocols: introduces a two-stage distributed control scheme to achieve consensus in second-order multi-agent systems, with a leader agent (reference state provider), follower agents (consensus-seeking entities), a distributed fixed-time observer (leader state estimator), a fixed-time tracking controller (trajectory tracking law), a communication network (undirected graph topology), and time-varying gains (bounded convergence rate adjusters).
- The architecture separates the consensus problem into a distributed estimation phase and a local tracking phase to ensure convergence within a user-defined upper bound of the settling time.
- The non-autonomous protocol leverages bounded time-varying gains to provide tighter estimates of the settling time without the singularity issues common in existing predefined-time algorithms.

---

[Toward Scalable Verifiable Reward: Proxy State-Based Evaluation for Multi-turn Tool-Calling Large Language Model Agents](http://arxiv.org/abs/2602.16246)

- Proxy State-Based Evaluation: introduces an LLM-driven simulation framework that evaluates multi-turn tool-calling agents by checking outcomes against an inferred proxy state instead of a deterministic backend.
- The architecture includes reasoning-, user simulation-, tool simulation-, state tracking-, and judging-agents.
- It leverages a structured scenario object to define constraints and expected behaviors, facilitating the generation of on-policy data for agent training.

---

[Submodular Maximization under Supermodular Constraint: Greedy Guarantees](http://arxiv.org/abs/2602.16240)

- SMSC (Submodular Maximization under Supermodular Constraint): introduces a greedy algorithm that iteratively selects elements maximizing the ratio of marginal objective gain to marginal cost to provide provable bicriteria approximation guarantees under non-linear cost constraints.
- The framework utilizes a less restrictive notion of supermodular curvature to expand the class of admissible cost functions, including quadratic costs from agent interference in multi-agent LLM debating systems.
- It further provides an efficient binary search reduction to solve the dual problem of minimizing supermodular costs while satisfying submodular coverage requirements.

---

[Equity in auction design with unit-demand agents and non-quasilinear preferences](http://arxiv.org/abs/2602.16211)

- MWEP (Minimum Walrasian Equilibrium Price): introduces a unique auction mechanism for unit-demand agents with non-quasilinear preferences, characterized by strategy-proofness, individual rationality, equal treatment of equals, no-wastage, and no-subsidy.
- The framework utilizes indifference vectors to model agent preferences where income effects are present, ensuring robustness in high-value resource allocation scenarios.
- The research establishes that in rich preference domains, minimal equity constraints are sufficient to uniquely identify the efficient Walrasian outcome.

---

[Graphon Mean-Field Subsampling for Cooperative Heterogeneous Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2602.16196)

- GMFS (Graphon Mean-Field Subsampling): introduces a subsampling framework for cooperative multi-agent reinforcement learning that approximates heterogeneous interactions using a small subset of neighbors.
- The architecture utilizes a shared Q-function optimized centrally via a generative oracle and a graphon-based weighting mechanism to capture non-uniform agent dependencies.
- By reducing sample complexity from exponential to polynomial relative to the subsample size, the framework enables scalable coordination in large-scale systems like robotic swarms and traffic networks.

---

[Modeling Trust and Liquidity Under Payment System Stress: A Multi-Agent Approach](http://arxiv.org/abs/2602.16186)

- MAS (Multi-Agent System) framework: introduces a behavioral model linking payment outages to liquidity stress, with customer agents, merchant agents, an exogenous payment infrastructure, a social network, bounded memory processes, a threshold-gated withdrawal mechanism, and a substitution channel.
- The system utilizes a Watts-Strogatz network to propagate social signals while memory variables for personal experience and rumors drive delayed behavioral transitions.
- Simulations show that peak withdrawal pressure often occurs during technical recovery because persistent merchant broadcast signals and behavioral hysteresis sustain elevated risk perception.

---

[Multi-Agent Combinatorial-Multi-Armed-Bandit framework for the Submodular Welfare Problem under Bandit Feedback](http://arxiv.org/abs/2602.16183)

- MA-CMAB (Multi-Agent Combinatorial-Multi-Armed-Bandit): introduces a decentralized framework for the Submodular Welfare Problem under full-bandit feedback, utilizing an explore-then-commit strategy with an Offline Resilient Algorithm and Continuous Greedy Algorithm to maximize utilitarian social welfare.
- The framework employs a Monte Carlo Sampler to estimate marginal gains from aggregate rewards and applies Pipage Rounding to transform fractional optimization results into feasible integral item assignments.
- It establishes sublinear regret guarantees by coupling non-communicating agents through shared Partition Matroid constraints and handling noisy value oracle evaluations.

---

[EnterpriseGym Corecraft: Training Generalizable Agents on High-Fidelity RL Environments](http://arxiv.org/abs/2602.16179)

- COREcraft (EnterpriseGym Corecraft): introduces a high-fidelity reinforcement learning environment simulating a customer support organization to train generalizable AI agents using expert-authored rubrics and realistic enterprise workflows, with all Training (Megatron), Data Buffer (Bridge), Rollout (SGLang), Corecraft Docker Container, and LLM Judge (Verifier) components.
- The framework utilizes a continuous loop consisting of a rollout engine for trajectory generation, a stateful Docker-based environment for tool interaction via Model Context Protocol, and an LLM-based verifier for automated reward computation.
- Training with Group Relative Policy Optimization on this environment demonstrates significant performance gains that transfer to out-of-distribution benchmarks including BFCL, τ2-Bench, and Toolathlon.

---

[Edge Learning via Federated Split Decision Transformers for Metaverse Resource Allocation](http://arxiv.org/abs/2602.16174)

- FSDT (Federated Split Decision Transformer): introduces a cooperative edge AI framework that partitions a Decision Transformer model between distributed MEC servers and a central cloud to optimize resource allocation for metaverse applications.
- The architecture offloads the computationally intensive transformer decoder to the cloud while maintaining local adaptability through agent-specific embedding and prediction layers on MEC servers.
- It utilizes a two-phase training process combining federated learning and split learning to enhance Quality of Experience (QoE) and reduce communication overhead in heterogeneous multi-radio access technology environments.

---

[HiPER: Hierarchical Reinforcement Learning with Explicit Credit Assignment for Large Language Model Agents](http://arxiv.org/abs/2602.16165)

- HiPER (Hierarchical Plan–Execute Reinforcement learning): introduces a hierarchical reinforcement learning framework that factorizes a single LLM policy into a high-level planner for subgoal generation and a low-level executor for multi-step action execution.
- The framework utilizes a structured Plan-Execute interface to make hierarchical decisions explicit within the model's output, enabling joint optimization of planning, switching, and execution behaviors.
- It employs Hierarchical Advantage Estimation (HAE) to provide coupled learning signals across different time scales, significantly reducing variance and improving sample efficiency in long-horizon, sparse-reward tasks.

---

[Local Adapt-Then-Combine Algorithms for Distributed Nonsmooth Optimization: Achieving Provable Communication Acceleration](http://arxiv.org/abs/2602.16148)

- FlexATC (Flexible Adapt-Then-Combine): introduces a unified distributed optimization framework for composite problems, utilizing adaptation-step, combination-step, and correction-step components to achieve communication acceleration.
- The framework incorporates a probabilistic communication skipping mechanism that allows agents to perform local updates, effectively decoupling the linear convergence rate from network topology in strongly convex settings.
- It provides a theoretical foundation for communication acceleration in ATC-based algorithms, demonstrating that local updates can reduce communication complexity without deteriorating convergence rates.

---

[Empirical Cumulative Distribution Function Clustering for LLM-based Agent System Analysis](http://arxiv.org/abs/2602.16131)

- ECDF Clustering (Empirical Cumulative Distribution Function Clustering): introduces a novel evaluation framework that analyzes the distributional characteristics of LLM agent responses by clustering empirical cumulative distribution functions of cosine similarities between generated and reference answers.
- The framework utilizes k-medoids clustering and L1 distances to group agent configurations, revealing how factors like temperature, persona, and question topics influence response quality beyond simple accuracy metrics.
- Experimental results on SQuAD and other datasets demonstrate that this method distinguishes between agent settings with identical final accuracies but varying underlying response distributions.

---

[Multi-Agent Lipschitz Bandits](http://arxiv.org/abs/2602.16965)

- Multi-Phase Decentralized Protocol: introduces a communication-free framework for multi-player stochastic bandits over continuous action spaces, utilizing coarse identification, maxima-directed refinement, decentralized seating, and within-cell optimization.
- The protocol decouples multi-agent coordination from continuous optimization by first identifying high-value regions and then assigning agents via a Musical Chairs mechanism.
- The research establishes near-optimal regret bounds matching single-player rates while incurring coordination costs that are provably independent of the time horizon.

---

[SAGE: Structure Aware Graph Expansion for Retrieval of Heterogeneous Data](http://arxiv.org/abs/2602.16964)

- SAGE (Structure Aware Graph Expansion): introduces a retrieval framework that augments flat similarity search with chunk-level graph expansion to recover multi-hop evidence across heterogeneous text, tables, and graph nodes, and includes metadata-generation and planning-agents.
- The system constructs a sparse chunk-level graph offline using metadata-driven similarities and employs a two-stage online process where an initial retriever identifies seed nodes for first-hop neighbor expansion and re-ranking.
- For explicit schema graphs, the framework utilizes SPARK, an agentic retriever that generates multi-step plans interleaving semantic HNSW search with structural Cypher queries to ensure relational validity.

---

[Automating Agent Hijacking via Structural Template Injection](http://arxiv.org/abs/2602.16958)

- Phantom: introduces an automated agent hijacking framework that exploits the architectural parsing logic of LLM agents by injecting optimized structural templates to induce role confusion, with Adversary, LLM Agent, Multi-level Template Augmentation, Latent Space Mapping, Template Autoencoder, Automated Template Search, Bayesian Optimization, and Lightweight Proxy Evaluation.
- The system includes augmentation-LLMs and a Template Autoencoder backbone to map discrete structural patterns into a continuous latent space for efficient Bayesian optimization.
- Extensive evaluations on commercial models like GPT-4 and Gemini reveal architectural vulnerabilities in agentic systems, achieving high attack success rates while bypassing traditional semantic alignment defenses.

---

[LLM4Cov: Execution-Aware Agentic Learning for High-coverage Testbench Generation](http://arxiv.org/abs/2602.16953)

- LLM4Cov (Execution-Aware Agentic Learning for High-coverage Testbench Generation): introduces an offline agent-learning framework for hardware verification, with a student model, a teacher model, a hardware simulator, a coverage tool, a worst-state selector, a rejection sampler, and a training pipeline, where it converts simulator feedback into offline supervision and includes student- and teacher-models.
- The system formulates verification as a sequence of memoryless state transitions to reduce prompt redundancy and focuses supervision on recovery behaviors through coverage-guided rejection fine-tuning.
- A three-stage progressive learning strategy aligns synthetic data generation with the evolving student distribution, allowing a 4B-parameter model to achieve high-coverage results competitive with models an order of magnitude larger.

---

[Mind the GAP: Text Safety Does Not Transfer to Tool-Call Safety in LLM Agents](http://arxiv.org/abs/2602.16943)

- GAP benchmark: introduces a systematic evaluation framework that measures the divergence between text-level safety and tool-call-level safety in LLM agents, utilizing a deterministic scoring pipeline and runtime governance contracts.
- The architecture evaluates models—including Claude Sonnet 4.5, GPT-5.2, Grok 4.1 Fast, DeepSeek V3.2, Kimi K2.5, and GLM-4.7—across six regulated domains using a full factorial cross-product design that incorporates jailbreak scenarios and system prompt ablations.
- The research identifies a modality gap where safety alignment in text generation fails to transfer to tool-selection processes, leading to harmful action execution despite textual refusal.

---

[ConvApparel: A Benchmark Dataset and Validation Framework for User Simulators in Conversational Recommenders](http://arxiv.org/abs/2602.16938)

- ConvApparel: introduces a benchmark dataset and a three-pillar validation framework to measure the realism gap in LLM-based user simulators for conversational recommendation.
- The framework utilizes population-level statistical alignment, a discriminator-based human-likeness score, and counterfactual validation to assess how simulators adapt to varied agent behaviors.
- Experimental results demonstrate that while data-driven simulators like ICL and SFT outperform prompted baselines, a significant realism gap remains compared to human interaction.

---

[Narrow Fine-Tuning Erodes Safety Alignment in Vision-Language Agents](http://arxiv.org/abs/2602.16931)

- VLS-Analysis (Vision-Language Safety Analysis): introduces an analysis of how narrow-domain harmful fine-tuning induces cross-domain emergent misalignment in multimodal agents, with Base Vision-Language Model (multimodal foundation model substrate), LoRA Fine-tuning (parameter-efficient adaptation mechanism), Narrow Harmful Dataset (domain-specific data inducing misalignment), LLM-as-a-Judge (automated safety scoring system), Benign Fine-tuning (alignment recovery via safe data), and Activation-based Steering (inference-time activation space intervention).
- The framework includes a base vision-language model and an LLM-as-a-judge evaluator to demonstrate that misalignment scales monotonically with LoRA rank across multimodal evaluation sets.
- Geometric analysis reveals that harmful behaviors occupy a low-dimensional subspace, enabling mitigation through targeted activation-based steering vectors or benign narrow fine-tuning.

---

[Discovering Multiagent Learning Algorithms with Large Language Models](http://arxiv.org/abs/2602.16928)

- AlphaEvolve: introduces an evolutionary coding framework that leverages LLMs to perform semantic mutations on multi-agent learning source code, utilizing population initialization, LLM-driven mutation, automated evaluation, and evolutionary selection.
- The system discovers VAD-CFR (Volatility-Adaptive Discounted Counterfactual Regret Minimization), which employs volatility-sensitive discounting and regret-magnitude weighted warm-starts to improve convergence in imperfect-information games.
- The framework also yields SHOR-PSRO (Smoothed Hybrid Optimistic Regret Policy-Space Response Oracles), featuring a hybrid meta-solver that dynamically blends optimistic regret matching with smoothed best pure strategies via an annealing schedule.

---

[AgentLAB: Benchmarking LLM Agents against Long-Horizon Attacks](http://arxiv.org/abs/2602.16901)

- AgentLAB (Agent Long-horizon Attack Benchmark): introduces an evaluation framework for measuring LLM agent security against adaptive, multi-turn adversarial strategies, with Adversary, LLM Agent, Environment, Interaction Trace, Evaluator, Multi-Agent Attack Framework, Planner, Attacker, Verifier, and Judge.
- The system utilizes a multi-agent architecture including planning-, attacking-, verifying-, and judging-agents to automate the generation of long-horizon exploits.
- The benchmark covers 644 security test cases across five attack categories, demonstrating that current LLMs are susceptible to gradual manipulation that evades standard one-shot safeguards.

---

[MALLVi: A MULTI-AGENT FRAMEWORK FOR INTEGRATED GENERALIZED ROBOTICS MANIPULATION](http://arxiv.org/abs/2602.16898)

- MALLVi (Multi-Agent Large Language and Vision framework): introduces a distributed multi-agent architecture for robotic manipulation, which includes Decomposer (converts instructions to atomic subtasks), Descriptor (generates spatial graph scene representation), Localizer (identifies and grounds objects in 3D), Thinker (computes actionable pick-and-place parameters), Actor (executes robotic commands via API), Reflector (verifies execution success via feedback), and GraphState (centralized memory for agent coordination).
- The framework utilizes a shared state to coordinate specialized LLM and VLM agents, enabling targeted error recovery by reactivating specific failing components rather than global replanning.
- Validated in VIMABench, RLBench, and real-world settings, the system demonstrates improved generalization and success rates in zero-shot manipulation through iterative feedback-driven interaction.

---

[OpenSage: Self-programming Agent Generation Engine](http://arxiv.org/abs/2602.16891)

- OpenSage (Open Self-programming Agent Generation Engine): introduces an agent development kit that enables LLMs to autonomously create agents with self-generated topology and toolsets, featuring a core reasoning engine, hierarchical graph-based storage, dynamic agent structure management, dynamic synthesis and sandboxed execution, and an external task space.
- The framework features a dynamic agent pool for vertical and horizontal sub-agent orchestration, a meta-tool system for runtime tool synthesis, and includes parent-, sub-, and memory-agents.
- It utilizes containerized sandboxing for tool execution isolation and Neo4j-based graph databases to manage complex spatial and temporal relationships in agent history.

---

[AdaptOrch: Task-Adaptive Multi-Agent Orchestration in the Era of LLM Performance Convergence](http://arxiv.org/abs/2602.16873)

- AdaptOrch (Task-Adaptive Multi-Agent Orchestration): introduces a framework for task-adaptive multi-agent orchestration that dynamically selects execution topologies based on task dependency graphs, with a task decomposer, a DAG constructor, a topology router, parallel/sequential/hierarchical/hybrid executors, and an adaptive synthesizer, and includes decomposer-, executor-, and arbiter-agents.
- The system utilizes a Topology Routing Algorithm to map task characteristics to canonical coordination patterns like parallel, sequential, hierarchical, or hybrid structures.
- It incorporates an Adaptive Synthesis Protocol with heuristic consistency scoring to reconcile agent outputs and ensure termination through iterative re-routing.

---

[Overseeing Agents Without Constant Oversight: Challenges and Opportunities](http://arxiv.org/abs/2602.16844)

- Magentic-UI: introduces a human-in-the-loop interface for overseeing Computer Use Agents, with task input, agent workflow, trace generation, review mechanism, and human-in-the-loop components, where the system includes planning- and execution-agents to facilitate error detection in multi-step agentic workflows.
- The framework incorporates a review mechanism featuring flowchart visualizations, citation-style justifications, and explicit requirement checklists to reduce the verbosity of raw action traces.
- User studies demonstrate that while structured abstractions like specifications help users find errors faster, they can also induce overreliance and false confidence if the underlying process appears reasonable.

---

[HiVAE: Hierarchical Latent Variables for Scalable Theory of Mind](http://arxiv.org/abs/2602.16826)

- HiVAE (Hierarchical Variational Architecture): introduces a hierarchical variational architecture for Theory of Mind reasoning in large-scale spatiotemporal domains, with Trajectory Encoder, World Encoder, Fusion Layer, Hierarchical Mind-State VAE, and Goal Predictor.
- The system utilizes a three-level VAE hierarchy inspired by the belief-desire-intention cognitive structure to sequentially infer latent mental states from fused spatiotemporal encodings.
- Experimental results on a 3,185-node campus navigation task show significant improvements in goal inference accuracy and robustness against misleading cues compared to traditional Bayesian and neural baselines.

---

[HYBRID-GYM: Training Coding Agents to Generalize Across Tasks](http://arxiv.org/abs/2602.16819)

- HYBRID-GYM: introduces a large-scale training environment for coding agents, with synthetic tasks, an OpenHands agent scaffold, and a sandboxed Docker environment, where teacher-generated trajectories are used to finetune student LLMs for cross-task generalization.
- The framework decomposes complex software engineering trajectories into reasoning, repository exploration, implementation, and verification components to teach transferable skills without requiring complex executable repository setups.
- It utilizes four scalable synthetic tasks—function localization, issue localization, dependency search, and function generation—to improve agent performance on real-world benchmarks like SWE-Bench and SWT-Bench.

---

[Large-scale online deanonymization with LLMs](http://arxiv.org/abs/2602.16800)

- ESRC (Extract, Search, Reason, Calibrate): introduces a modular pipeline for automated deanonymization of pseudonymous online accounts by utilizing LLMs to extract identity-relevant features from unstructured text and reason over candidate matches.
- The framework includes extraction-, search-, reasoning-, and calibration-components to transform raw user content into structured attributes and perform high-precision re-identification across platforms.
- Experimental results show that LLM-augmented attacks achieve up to 68% recall at 90% precision, demonstrating that the cost of large-scale re-identification has decreased and invalidating the practical obscurity previously protecting online pseudonymity.

---

#### 17th February 2026

[Developing AI Agents with Simulated Data: Why, what, and how?](http://arxiv.org/abs/2602.15816)

- DT4AI (Digital Twin for AI): introduces a reference framework to describe, design, and analyze digital twin-based AI simulation solutions, with AI Agent, Digital Twin, Physical Twin, Simulator, Model, Access Control, and Data Links components.
- The architecture utilizes a bidirectional coupling between virtual and physical counterparts to enable purposeful experimentation and high-fidelity data generation for training AI models.
- The paper evaluates simulation methods and mitigation strategies for the sim-to-real gap, providing specific instantiations for reinforcement learning and deep learning workflows.

---

[FAST-EQA: Efficient Embodied Question Answering with Global and Local Region Relevancy](http://arxiv.org/abs/2602.15813)

- 
FAST-EQA (FAst, Semantics-aware, Target-driven Exploration for Embodied Question Answering): introduces an active exploration framework that couples semantically-guided global and local navigation policies with a bounded visual memory to efficiently answer natural language queries in 3D environments, with LLM Question Parser, Global Relevance Exploration, Local Relevance Exploration, Bounded Visual Memory, Semantic Memory, and VLM Reasoner components.

- 
The system utilizes an LLM to parse questions into spatial goals and a 3D voxel-based occupancy map to identify narrow openings as high-value frontiers for efficient scene coverage.

- 
It employs Chain-of-Thought reasoning over a compact set of retrieved memory snapshots to provide interpretable answers while achieving significantly faster inference times than existing graph-centric EQA methods.


---

[Decision Quality Evaluation Framework at Pinterest](http://arxiv.org/abs/2602.15809)

- Decision Quality Evaluation Framework: introduces a system for evaluating content moderation decisions at scale, utilizing an expert-curated Golden Set (GDS) and automated workflows to benchmark human agents and LLs.
- The framework integrates a sampling pipeline using propensity scores and PinCLIP embeddings to optimize dataset coverage and manage trade-offs between cost, scale, and trustworthiness.
- It supports data-driven prompt optimization and performance benchmarking for various LLM-based agents, including Gemini 2.5 flash, Gemini 2.5 pro, GPT-4.1, GPT-4o, and GPT-5.

---

[Stability in Distance Preservation Games on Graphs](http://arxiv.org/abs/2602.15784)

- DPG (Graphical Distance Preservation Games): introduces a network allocation model where agents are assigned to graph vertices based on preferred distances to others, with Topology (underlying graph for positioning), Agents (entities requiring vertex allocation), Relationship Sets (subsets of relevant agents), Distance Functions (prescribed ideal inter-agent distances), Preference Graph (directed graph of agent interests), Allocation (injective mapping to vertices), Cost Function (sum of distance differences), and Stability Notions (envy-freeness, swap, jump criteria).
- The research evaluates the computational complexity of finding stable allocations across various graph topologies, including cliques, stars, paths, and trees, under different agent preference structures.
- It establishes NP-completeness for simple topologies while providing fixed-parameter tractable algorithms for structural parameters such as vertex cover number, neighborhood diversity, and modular width.

---

[GLOBEDIFF: STATE DIFFUSION PROCESS FOR PARTIAL OBSERVABILITY IN MULTI-AGENT SYSTEMS](http://arxiv.org/abs/2602.15776)

- GlobeDiff (Global State Diffusion Algorithm): introduces a generative framework for multi-agent reinforcement learning that addresses partial observability by formulating global state inference as a conditional multi-modal diffusion process.
- The architecture utilizes a latent variable as a mode selector to handle one-to-many mappings between local observations and global states, effectively preventing mode collapse common in discriminative models.
- It integrates a U-Net-based denoising network with a prior-posterior alignment mechanism to enable high-fidelity state reconstruction during decentralized execution without requiring global information.

---

[MeshMimic: Geometry-Aware Humanoid Motion Learning through 3D Scene Reconstruction](http://arxiv.org/abs/2602.15733)

- MeshMimic: introduces a framework that bridges 3D scene reconstruction and embodied intelligence to enable humanoid robots to learn coupled motion-terrain interactions directly from monocular video.
- The system utilizes vision foundation models to decouple human trajectories from environmental meshes, applying a Kinematic Consistency Optimization algorithm to refine visual data into physically plausible reference motions.
- It incorporates MeshRetarget, a contact-aware mechanism that maps human-environment interaction features onto humanoid morphology while preserving geometric constraints for deployment in unstructured environments.

---

[Lifelong Scalable Multi-Agent Realistic Testbed and A Comprehensive Study on Design Choices in Lifelong AGV Fleet Management Systems](http://arxiv.org/abs/2602.15721)

- 
LSMART (Lifelong Scalable Multi-Agent Realistic Testbed): introduces an open-source simulation framework for evaluating Multi-Agent Path Finding algorithms in Fleet Management Systems, incorporating a planner invocation policy, instance generator, MAPF planner, fail policy, action dependency graph, AGV fleet, and a physics-based simulator.

- 
The system parallelizes planning and execution by utilizing an action dependency graph to maintain robust inter-agent coordination despite communication delays and execution uncertainties.

- 
It enables systematic evaluation of design choices such as periodic versus event-based planning and various failure recovery strategies across diverse warehouse and maze environments.


---

[A Content-Based Framework for Cybersecurity Refusal Decisions in Large Language Models](http://arxiv.org/abs/2602.15689)

- Content-based framework for cybersecurity refusal decisions in Large Language Models: introduces a structured methodology for evaluating the technical substance of LLM requests to balance offensive risk against defensive utility, with Offensive Action Contribution, Offensive Risk, Technical Complexity, Defensive Benefit, and Expected Frequency for Legitimate Users.
- The system utilizes decision trees to categorize dual-use cybersecurity prompts, moving beyond simple topic-based bans or intent-based classification which are often prone to obfuscation.
- By grounding refusal logic in content-level properties, the framework enables organizations to construct tunable, risk-aware policies that mitigate misuse while preserving legitimate defensive capabilities.

---

[Zombie Agents: Persistent Control of Self-Evolving LLM Agents via Self-Reinforcing Injections](http://arxiv.org/abs/2602.15654)

- Zombie Agent: introduces a black-box attack framework targeting self-evolving LLM agents by implanting persistent malicious payloads into long-term memory via indirect prompt injection.
- The attack utilizes mechanism-specific strategies like recursive self-replication and semantic aliasing to resist memory truncation and relevance filtering in sliding-window and RAG-based architectures.
- Evaluation on commercial LLMs shows that memory evolution can convert one-time injections into persistent compromises, enabling unauthorized tool use and data exfiltration while bypassing per-session prompt filtering.

---

[Meflex: A Multi-agent Scaffolding System for Entrepreneurial Ideation Iteration via Nonlinear Business Plan Writing](http://arxiv.org/abs/2602.15631)

- Meflex: introduces a multi-agent scaffolding system for non-linear business plan writing, with an ideation canvas (visual node-based workspace), structure writing workspace (modular composition environment), LLM assistant panel (interactive guidance interface), multi-agent LLM framework (includes User Pain Points-, Market Analysis-, and Reflection-agents), and meta-reflection engine (evolution synthesis component).
- The system supports iterative ideation by enabling users to revisit and branch ideas on a visual canvas while receiving context-aware assistance from specialized LLM agents.
- It incorporates reflection prompts and meta-reflection summaries to foster divergent thinking and enhance metacognitive awareness during complex entrepreneurial tasks.

---

[Neural Network-Based Parameter Estimation of a Labour Market Agent-Based Model](http://arxiv.org/abs/2602.15572)

- SBI4ABM (Simulated-Based Inference 4 (for) ABM): introduces a simulation-based inference framework for estimating parameters in high-dimensional, stochastic labour market agent-based models using neural networks to approximate posterior distributions.
- The system integrates Neural Posterior Estimation with Masked Autoregressive Flows and Recurrent Neural Networks to automatically extract summary statistics from simulation outputs like job transition matrices.
- Experimental results on synthetic and U.S. labour market data show that neural network-learned statistics achieve higher precision and better uncertainty quantification compared to handcrafted statistical measures.

---

["What Are You Doing?": Effects of Intermediate Feedback from Agentic LLM In-Car Assistants During Multi-Step Processing](http://arxiv.org/abs/2602.15569)

- Agentic LLM In-Car Assistant: introduces a study on feedback timing and verbosity for autonomous multi-step tasks, comparing No Intermediate (NI) and Planning & Results (PR) strategies.
- The system utilizes a voice user interface and graphical display to communicate reasoning steps and intermediate outcomes during extended processing periods.
- Empirical results indicate that intermediate feedback improves perceived speed, trust, and user experience while reducing cognitive task load in driving contexts.

---

[Simultaneous Ordinal Maximin Share and Envy-Based Guarantees](http://arxiv.org/abs/2602.15566)

- Ordinal Fair Allocation Framework: introduces a systematic approach for achieving simultaneous ordinal maximin share (MMS) and envy-based fairness guarantees, with initialization module, bag-filling loop, lone divider agent, envy-free matching module, threshold graph, envy-cycle elimination algorithm, and envy graph.
- The framework establishes the existence of complete allocations satisfying 1-out-of-⌈3n/2⌉ MMS and EFX for ordered instances.
- It further extends these results to top-n instances and provides a 1-out-of-4⌈n/3⌉ MMS guarantee combined with EF1.

---

[In Agents We Trust, But Who Do Agents Trust? Latent Source Preferences Steer LLM Generations](http://arxiv.org/abs/2602.15456)

- Latent Source Preference Evaluation Framework: introduces, a methodology to quantify implicit source biases, with Direct Evaluation (explicit ranking of source entities), Indirect Evaluation (implicit preference measurement via content), Source Identities (brand names and online identifiers), Source Credentials (quantitative attributes like follower counts), Contextual Framing (domain-specific task environments), Prompting Interventions (bias mitigation and optimization instructions), Multi-axis Decision Factors (weighted criteria for selection tasks), Generation-LLMs (synthetic data creation), Summarization-LLMs (standardizing product descriptions), Parsing-LLMs (structuring delivery promises), and includes generation-, summarization- and parsing-LLMs.
- The research validates that LLMs encode latent preferences for brands that significantly influence information prioritization across news, academic, and e-commerce domains.
- Analysis reveals that these preferences are highly context-sensitive and often resistant to standard zero-shot prompting techniques designed to mitigate bias.

---

[Fairness over Equality: Correcting Social Incentives in Asymmetric Sequential Social Dilemmas](http://arxiv.org/abs/2602.15407)

- Fair&Local (Fair and Localized Fairness-based Intrinsic Motivation): introduces a decentralized multi-agent reinforcement learning approach for asymmetric sequential social dilemmas, with IQL agents, reward normalization, and local social feedback.
- The framework employs a reward normalization module to adjust temporally smoothed rewards based on an agent's specific potential range, ensuring precise comparisons across heterogeneous capabilities.
- A local social feedback mechanism allows agents to maintain decentralized estimates of peer rewards through direct communication, facilitating cooperation without requiring global information access.

---

[Common Belief Revisited](http://arxiv.org/abs/2602.15403)

- CBn (Axiomatisation of common belief for n agents): introduces a sound and complete characterisation of common belief for KD45 agents, with multi-agent Kripke models, common belief operator, shift-reflexivity axiom, counting axiom, and recursive state-splitting.
- The logic demonstrates that common belief in KD45 systems is not merely KD4 but includes properties like shift-reflexivity and a counting axiom sensitive to the number of agents.
- The paper provides a direct characterisation of common belief by excluding individual belief modalities from the syntax and employing a tree-like model construction for the completeness proof.

---

[One Agent to Guide Them All: Empowering MLLMs for Vision-and-Language Navigation via Explicit World Representation](http://arxiv.org/abs/2602.15400)

- GTA (GUIDE THEM ALL): introduces a decoupled zero-shot Vision-and-Language Navigation framework that separates low-level spatial state estimation from high-level semantic planning via an explicit interactive metric world representation.
- The architecture utilizes a Metric Mapping Module to synthesize real-time metric maps from RGB-D sequences, which are then rendered by an Interactive Reasoning Interface into structured prompts for a Counterfactual Reasoning Brain.
- By employing a frozen Multimodal Large Language Model (MLLM) to reason over procedural blueprints and simulated trajectories, the framework achieves state-of-the-art performance in continuous environments and demonstrates robust zero-shot sim-to-real transfer across diverse robotic embodiments.

---

[World-Model–Augmented Web Agents with Action Correction](http://arxiv.org/abs/2602.15384)

- WAC (World-model–augmented Action Correction): introduces a multi-agent web agent framework that integrates model collaboration, consequence simulation, and feedback-driven action refinement, including router-, world-, action-, and judge-models.
- The architecture utilizes a world model to provide strategic guidance and simulate potential outcomes, which are then scrutinized by a judge model to trigger corrective feedback before execution.
- This approach achieves absolute success rate gains on VisualWebArena and Online-Mind2Web benchmarks by preventing irreversible deviations in task trajectories through iterative refinement.

---

[Orchestration-Free Customer Service Automation: A Privacy-Preserving and Flowchart-Guided Framework](http://arxiv.org/abs/2602.15377)

- Orchestration-Free Customer Service Automation: introduces, a framework for customer service automation, with Task-Oriented Flowcharts (TOFs), Small Language Models (SLMs), Teacher LLMs, Student SLMs, Knowledge Abstraction, Flowchart Aggregation, Synthetic Data Generation, Decentralized Flowchart Distillation, Knowledge Bases, and System Operations.
- The framework employs a distillation strategy where knowledge abstraction converts dialogues into flowcharts for aggregation and data generation by a Teacher LLM.
- This architecture enables local deployment of Small Language Models while maintaining task completion rates through business logic representation and fine-tuning.

---

[AgriWorld: A World–Tools–Protocol Framework for Verifiable Agricultural Reasoning with Code-Executing Large Language Model Agents](http://arxiv.org/abs/2602.15325)

- AgriWorld: introduces an agentic framework for agricultural science that enables LLMs to reason over high-dimensional spatiotemporal data, with State Space (encapsulates heterogeneous spatiotemporal agricultural data), AGRIWORLD Environment (Python execution environment with unified APIs), Coordinate Alignment (canonical operator for spatiotemporal consistency), AGRO-REFLECTIVE Agent (multi-turn LLM agent with reflection), Verifiable Specification Protocol (executable checkers for grounding reasoning), and AGROBENCH (verifiable evaluation suite with reference programs).
- The framework utilizes an execute–observe–refine loop to iteratively diagnose errors and refine analysis based on structured execution feedback from the environment.
- It employs hierarchical constraints to ensure schema validity, numeric tolerance, and causal consistency, outperforming text-only and direct tool-use baselines.

---

[Enhancing Computational Efficiency in NetLogo: Best Practices for Running Large-Scale Agent-Based Models on AWS and Cloud Infrastructures](http://arxiv.org/abs/2602.15317)

- 
NetLogo Optimization Framework: introduces a comprehensive methodology for scaling large-scale agent-based models on cloud infrastructures, with all NetLogo, Java Virtual Machine (JVM), BehaviorSpace, AWS Cloud Infrastructure, Headless Execution, and Parallel Processing components.

- 
The framework optimizes computational efficiency by fine-tuning JVM heap sizes and G1 garbage collection settings to manage high-memory demands and prevent simulation crashes.

- 
Comparative analysis using the wolf-sheep predation model demonstrates that compute-optimized AWS instances achieve a 32% reduction in costs while maintaining high performance for CPU-bound simulations.


---

[Intellicise Wireless Networks Meet Agentic AI: A Security and Privacy Perspective](http://arxiv.org/abs/2602.15290)

- Agentic AI-enhanced Intellicise Wireless Networks: introduces a security-focused framework for 6G systems that leverages autonomous perception-memory-reasoning-action loops to provide proactive defense against intelligent eavesdropping and sensing attacks.
- The system incorporates LVM-based feature extractors and LLM-based key generators within a semantic extraction module to enable secure coverless steganography and encrypted communication through digital token management.
- The research establishes a taxonomy for secure network organization, including LLM-guided penetration testing and diffusion-model-integrated traffic detection to maintain robust performance in dynamic and adversarial environments.

---

[Visual Persuasion: What Influences Decisions of Vision-Language Models?](http://arxiv.org/abs/2602.15278)

- CVPO, VFD, & VTG (Competitive Visual Prompt Optimization, Visual Feedback Descent, and Visual Text Grad): introduces a framework to study and exploit the latent visual preferences of VLMs through iterative, feedback-driven image editing, with Original Image (initial visual input), VLM Judges (vision-language models providing preference feedback), LLM Proposer (language model generating text-based editing instructions), Image Generation Model (generative tool applying visual modifications), Auto-Interpretability Pipeline (hierarchical summarization of visual themes), and Mitigation Module (image normalization to reduce contextual bias); it includes VLM-judges and LLM-proposers.
- The system employs competitive selection and gradient-based text optimization to identify naturalistic image perturbations that significantly shift model choice probabilities in agentic tasks.
- It features a Matryoshka summarization pipeline to explain discovered visual themes and proposes image normalization as a strategy to mitigate identified model vulnerabilities.

---

[When Remembering and Planning are Worth it: Navigating under Change](http://arxiv.org/abs/2602.15274)

- ProbMap (Probabilistic Map Strategy): introduces a multi-strategy agent architecture for spatial navigation in non-stationary environments, combining episodic memory, statistical learning, and a round-robin scheduler.
- The framework manages transitions between search-, greedy-, and planning-based strategies using progressive time budgets to mitigate environmental change and localization uncertainty.
- It utilizes a two-tiered memory update mechanism that separates fast within-day adaptation from stable multi-day environmental modeling for robust path planning.

---

[FrameRef: A Framing Dataset and Simulation Testbed for Modeling Bounded Rational Information Health](http://arxiv.org/abs/2602.15273)

- 
FrameRef (A Framing Dataset and Simulation Testbed): introduces a simulation-based framework for modeling sequential information exposure and reinforcement dynamics, which includes reframing-, verification-, and persona-LLMs.

- 
The system utilizes a large-scale dataset of over one million reframed claims across five dimensions—authoritative, consensus, emotional, prestige, and sensationalist—to study long-term information health.

- 
It employs Monte Carlo trajectory sampling to demonstrate how small, systematic judgment biases in LLM-based personas can compound over time into significant divergence in cumulative information health.


---

[Enhancing Diversity and Feasibility: Joint Population Synthesis from Multi-source Data Using Generative Models](http://arxiv.org/abs/2602.15270)

- Joint WGAN: introduces a multi-source generative framework that simultaneously integrates census and travel survey data to synthesize representative individual-level populations for agent-based modeling, with Noise (random latent input vector), Generator (neural network synthesizing multi-attribute data), Dual Critics (independent networks evaluating data subsets), Multi-source Data (heterogeneous census and survey inputs), Regularization Term (inverse gradient penalty for diversity), and Synthetic Population (fused individual-level agent data).
- The architecture utilizes a dual-critic design to independently assess distributional characteristics of distinct datasets while a generator learns cross-dataset feature dependencies.
- An inverse gradient penalty regularization term is incorporated into the loss function to mitigate mode collapse and address sampling and structural zeros by promoting output diversity.

---

[AI as Coordination-Compressing Capital: Task Reallocation, Organizational Redesign, and the Regime Fork](http://arxiv.org/abs/2602.16078)

- Agent Capital (Coordination-Compressing Capital): introduces a task-based economic model where AI acts as a distinct production input that reduces organizational coordination costs, enabling flatter hierarchies and endogenous task creation, with agent capital, coordination friction, span of control, elite complementarity, task creation elasticity, and the task frontier.
- The framework identifies a regime fork where the distributional impact of AI depends on whether agent capital complements all workers broadly or high-skill managers disproportionately.
- Numerical simulations demonstrate that while coordination compression universally expands employment, it simultaneously widens the manager-worker wage gap and concentrates returns in the coordinating layer.

---

[ScenicRules: An Autonomous Driving Benchmark with Multi-Objective Specifications and Abstract Scenarios](http://arxiv.org/abs/2602.16073)

- ScenicRules: introduces a benchmark for evaluating autonomous driving systems by integrating a Hierarchical Rulebook for prioritized multi-objective specifications with the Scenic language for expressive scenario modeling.
- The framework includes an LLM-assisted pipeline to reconstruct near-accident scenarios from collision reports and an automated generator that uses coreset selection for diverse scenario coverage.
- It employs a falsification flow to systematically expose agent failures across various driving contexts while ensuring alignment with human driving judgments.

---

[The Limits of Long-Context Reasoning in Automated Bug Fixing](http://arxiv.org/abs/2602.16069)

- mini-SWE-agent (mini Software Engineering Agent): introduces a lightweight, bash-only agentic harness to evaluate LLMs on repository-scale debugging by decomposing tasks into iterative short-context steps within a Docker environment, with all Agent, LLM, Environment, Action-Observation loop, Retrieval module, and Post-processing components.
- The research utilizes a long-context golden patch generation pipeline to test direct reasoning by injecting ground-truth files into the context, ensuring perfect retrieval recall for single-shot evaluation.
- Findings demonstrate a significant gap between nominal context length and usable capacity, as LLMs frequently produce malformed patches and hallucinated line numbers when processing 64k-128k tokens.

---

[MARLEM: A Multi-Agent Reinforcement Learning Simulation Framework for Implicit Cooperation in Decentralized Local Energy Markets](http://arxiv.org/abs/2602.16063)

- MARLEM (Multi-Agent Reinforcement Learning Simulation Framework for Implicit Cooperation in Decentralized Local Energy Markets): introduces an open-source Gymnasium-compliant environment for studying emergent coordination in decentralized local energy markets using multi-agent reinforcement learning.
- The architecture integrates a modular market platform with a physical grid model to capture techno-economic interactions and constraints such as transmission losses and congestion.
- It fosters implicit cooperation by enriching agent observations and rewards with system-level key performance indicators, enabling independent learning of collectively beneficial strategies without explicit communication.

---

[Harnessing Implicit Cooperation: A Multi-Agent Reinforcement Learning Approach Towards Decentralized Local Energy Markets](http://arxiv.org/abs/2602.16062)

- 
Implicit Cooperation: introduces a multi-agent reinforcement learning framework for decentralized local energy markets, with agents, observation space, reward function, training paradigms, algorithms, and stigmergic signals, where decentralized agents approximate optimal coordination by reacting to shared environmental markers.

- 
The approach leverages system-level key performance indicators (KPIs) embedded in local observations to resolve non-stationarity and enable stable learning in fully decentralized training environments.

- 
Experimental results on an IEEE 34-node topology identify the APPO-DTDE configuration as the optimal balance for achieving high coordination efficiency while maintaining superior physical grid stability and privacy.


---

[Optimization Instability in Autonomous Agentic Workflows for Clinical Symptom Detection](http://arxiv.org/abs/2602.16037)

- Pythia: introduces an autonomous agentic workflow for clinical symptom detection, featuring specialist-, error analysis-, and synthesis-agents that iteratively optimize prompts through natural language feedback.
- The research identifies a critical failure mode termed optimization instability, where autonomous systems overcorrect for specific error types in imbalanced datasets, causing sensitivity to oscillate or collapse.
- The study evaluates interventions for stabilization, finding that retrospective selection of the best iteration significantly outperforms active guiding agents in maintaining robust classifier performance.

---

[Markov Chains with Rewinding](http://arxiv.org/abs/2602.16028)

- Markov Chains with Rewinding: introduces a formal model for algorithmic interaction with random evolutions where an agent can strategically rewind to previous states to identify hidden initial conditions.
- The framework establishes that while adaptive and non-adaptive strategies have equal power in state distinguishability, a polynomial gap exists in their query complexity.
- It provides a polynomial-time non-adaptive algorithm for state identification based on a shortest-path approach in a weighted partition graph.

---

[Convergence rates of random-order best-response dynamics in public good games on networks](http://arxiv.org/abs/2602.15986)

- Random-order best-response dynamics: introduces a formal and numerical analysis of convergence rates in public good games on networks, utilizing network structures, agents, activity levels, externality factors, active sets, inactive agents, reshuffles, and problematic subgraphs.
- The study identifies structural graph properties beyond spectral characteristics that lead to slow convergence, particularly near stability thresholds where equilibria change qualitatively.
- It characterizes the "reshuffle" phenomenon where late-stage activation of inactive nodes triggers new convergence iterations, significantly extending total convergence time in both deterministic and random graphs.

---

[Evolutionary Systems Thinking: From Equilibrium Models to Open-Ended Adaptive Dynamics](http://arxiv.org/abs/2602.15957)

- SDA (Stability-Driven Assembly): introduces a non-equilibrium framework where stochastic interactions and differential persistence generate endogenous selection without predefined fitness functions, with base elements, stochastic interactions, compounds, stability, population distribution, and a feedback loop.
- The framework models evolution as a natural genetic algorithm where longer-lived patterns accumulate, biasing future interactions and reshaping the population's probability distribution through persistence-weighted sampling.
- The research demonstrates that open-ended evolution requires population-dependent, non-stationary dynamics where structure and dynamics co-evolve, contrasting with traditional equilibrium-constrained system models that assume fixed state spaces.

---

[From Tool Orchestration to Code Execution: A Study of MCP Design Choices](http://arxiv.org/abs/2602.15945)

- CE-MCP (Code Execution Model Context Protocol): introduces a context-decoupled execution model where LLMs generate self-contained programs to orchestrate tool calls within an isolated runtime, significantly reducing token overhead compared to traditional context-coupled protocols.
- The framework incorporates a layered defense architecture featuring pre-execution semantic gating, static code validation, and post-execution output verification to mitigate novel attack vectors like exception-mediated code injection and unsafe capability synthesis.
- Empirical evaluation using MCP-Bench demonstrates that while CE-MCP improves latency and token efficiency for complex data-parallel tasks, it requires robust system-level security controls to manage an expanded attack surface across five execution phases.

---

[GLM-5: from Vibe Coding to Agentic Engineering](http://arxiv.org/abs/2602.15763)

- GLM-5: introduces a foundation model transitioning from vibe coding to agentic engineering using a Mixture-of-Experts architecture, with Base Model, Overall SFT, Reasoning RL, Agentic RL, General RL, On-Policy Cross-Stage Distillation, Slime Framework, Multi-Task Rollout Orchestrator, TITO Gateway, DSA, MLA, and MTP, and includes coding-, search- and general-purpose-agents.
- The architecture utilizes DeepSeek Sparse Attention (DSA) and Multi-latent Attention (MLA) to reduce computational overhead while scaling LLMs to 744B parameters and supporting 200K context windows.
- An asynchronous reinforcement learning infrastructure decouples trajectory generation from training to maximize GPU utilization and improve learning efficiency for long-horizon software engineering tasks.

---

[Learning to Retrieve Navigable Candidates for Efficient Vision-and-Language Navigation](http://arxiv.org/abs/2602.15724)

- Retrieval-augmented LLM navigator: introduces a dual-level retrieval framework for vision-and-language navigation, with instruction-level exemplar retrieval and step-level candidate pruning.
- The architecture employs an imitation-learned candidate retriever to filter irrelevant directions and an embedding-based retriever to supply successful navigation trajectories as in-context exemplars.
- This modular approach improves decision-making efficiency and success rates in previously unseen environments while keeping the core LLM frozen.

---

[The Next Paradigm Is User-Centric Agent, Not Platform-Centric Service](http://arxiv.org/abs/2602.15682)

- User-Centric Agent (Edge-Cloud Collaborative Pipeline): introduces a structural inversion of digital services that shifts control from platform-centric profit optimization to user-governed intelligence, utilizing an Edge-Cloud Collaborative Pipeline to manage private context and cross-service workflows.
- The architecture features an on-device agent for local perception, planning, and execution verification, while the cloud component includes context-aware reasoning- and adaptive strategy-agents for external service access.
- This paradigm addresses structural bottlenecks like fragmented context and misaligned incentives by ensuring that final decisions are made on-device under user-defined constraints and privacy-by-design principles.

---

[Agent-Based Macroeconomics for the UK’s Seventh Carbon Budget](http://arxiv.org/abs/2602.15607)

- Macroeconomic ABM (Agent-Based Model): introduces a data-driven computational framework to assess the macroeconomic and distributional impacts of the UK's seventh carbon budget, with Households, Firms, Government, Central Bank, External Shock, Learning Packages, and Simulation-Based Inference (SBI) components.
- The framework utilizes real-world UK household microdata and simulation-based inference to calibrate agent interactions, enabling the simulation of growth, employment, inflation, and inequality under various decarbonization scenarios.
- The model incorporates exogenous shocks from Climate Change Committee projections and endogenizes technological progress through S-curve adoption dynamics and social learning networks within the agent environment.

---

[Improving MLLMs in Embodied Exploration and Question Answering with Human-Inspired Memory Modeling](http://arxiv.org/abs/2602.15513)

- Human-Inspired Memory Modeling: introduces a non-parametric memory framework for embodied exploration and question answering, with Meta Memory, Episodic Memory, Semantic Memory, Semantic Space, Physical Space, and MLLM-based Reasoning.
- The architecture utilizes episodic memory for soft, associative recall of past observations verified by MLLM-based visual reasoning, bypassing the need for rigid geometric fusion.
- The system includes instruction-decomposition, visual-reasoning, and rule-extraction LLM-components to convert experiences into structured pseudocode for enhanced cross-environment generalization.

---

[EarthSpatialBench: Benchmarking Spatial Reasoning Capabilities of Multimodal LLMs on Earth Imagery](http://arxiv.org/abs/2602.15918)

- EarthSpatialBench: introduces a comprehensive benchmark for evaluating spatial reasoning in Multimodal LLMs on Earth imagery, featuring over 325K question-answer pairs spanning distance, direction, and topological relations.
- The benchmark evaluates models using diverse geometric types including bounding boxes, polylines, and polygons across choice-based, quantitative, and localization question formats.
- It supports multiple object reference modalities such as textual descriptions, visual overlays, and explicit coordinates to analyze the coupling between low-level grounding and high-level spatial reasoning.

---

[EventMemAgent: Hierarchical Event-Centric Memory for Online Video Understanding with Adaptive Tool Use](http://arxiv.org/abs/2602.15329)

- EventMemAgent: introduces an active online video agent framework for continuous perception and long-range reasoning, with an input video stream, short-term memory, long-term memory, a multi-granular perception toolkit, an MLLM-based agent, and agentic reinforcement learning.
- The system utilizes a dual-layer memory strategy where short-term memory performs online event segmentation and reservoir sampling, while long-term memory structuredly archives event-centric tuples including captions and visual anchors.
- Agentic Reinforcement Learning via Group Relative Policy Optimization (GRPO) is employed to internalize reasoning paths and tool-invocation strategies directly into the agent's intrinsic capabilities.

---

[EAA: AUTOMATING MATERIALS CHARACTERIZATION WITH VISION LANGUAGE MODEL AGENTS](http://arxiv.org/abs/2602.15294)

- EAA (Experiment Automation Agents): introduces a VLM-driven agentic system for automating experimental microscopy workflows, with Task Manager (orchestrates conversational loops and logic-defined routines), Agent (mediates between VLM inference and experimental tools), VLM (provides multimodal reasoning and image comprehension), Tool Manager (handles built-in and external MCP-compliant tools), and Memory Manager (facilitates long-term retrieval-augmented generation).
- The framework supports three levels of LLM involvement including logic-driven analytical routines, hybrid rule-based middleware, and fully autonomous agent-steered workflows for complex scientific tasks.
- It implements a two-way Model Context Protocol (MCP) to allow instrument-control tools to be served or consumed across different applications and platforms, ensuring interoperability in scientific ecosystems.

---

[Supporting Multimodal Data Interaction on Refreshable Tactile Displays: An Architecture to Combine Touch and Conversational AI](http://arxiv.org/abs/2602.15280)

- Multimodal data interaction architecture: introduces a technical framework integrating refreshable tactile display hardware, external hand tracking, and conversational AI to enable accessible data visualization for blind or low vision users.
- The system utilizes an Interaction Manager to fuse continuous finger tracking with spoken language, enabling deictic queries that ground conversational analysis in spatial exploration.
- It employs a Conversational Agent powered by GPT-4o and LangChain to perform statistical calculations and generate synchronized multimodal outputs across tactile, Braille, and audio channels.

---

#### 16th February 2026

[Hunt Globally: Deep Research AI Agents for Drug Asset Scouting in Investing, Business Development, and Search & Evaluation](http://arxiv.org/abs/2602.15019)

- Bioptic Agent: introduces a tree-based, self-learning multi-agent system for exhaustive drug asset scouting, utilizing a Coach Agent to manage a hierarchical tree of search directives and refine exploration strategies.
- The framework employs multilingual Investigator Agents for parallel web retrieval and a Criteria Match Validator Agent to ensure high-precision asset verification against complex, multi-constraint investor queries.
- The system incorporates a Deduplication Agent and Global Memory to maintain a lossless candidate set while optimizing compute allocation through Upper Confidence Bound-based node selection and reward backpropagation.

---

[Distributed Quantum Gaussian Processes for Multi-Agent Systems](http://arxiv.org/abs/2602.15006)

- DQGP (Distributed Quantum Gaussian Process): introduces a hybrid classical-quantum framework for multi-agent systems, with agents, local datasets, quantum encoding circuits, quantum processing units, quantum kernels, a central server, classical optimizers, and DR-ADMM, where regional models are aggregated into a global consensus.
- The system utilizes DR-ADMM (Distributed consensus Riemannian Alternating Direction Method of Multipliers) to optimize rotational hyperparameters on a Riemannian manifold, overcoming non-Euclidean optimization challenges.
- By distributing computational and memory loads across multiple agents, the method exploits the high-dimensional feature space of quantum kernels to model complex, non-stationary real-world datasets.

---

[Counterfactual Fairness Evaluation of LLM-Based Contact Center Agent Quality Assurance System](http://arxiv.org/abs/2602.14970)

- 
Auto-QA Fairness Evaluation Framework: introduces a systematic auditing pipeline to quantify demographic and behavioral biases in LLM-based contact center quality assurance systems using a 13-dimension taxonomy and counterfactual perturbations.

- 
The system utilizes a counterfactual generator with turn transformation, context injection, and metadata appending modules to create transcript variants that isolate specific fairness dimensions while preserving semantic intent.

- 
Evaluation of 18 LLMs reveals that while larger aligned models show lower unfairness, historical performance priming and implicit linguistic cues remain significant sources of evaluative disparity.


---

[The Distortion of Stable Matching](http://arxiv.org/abs/2602.14961)

- Stable-TSF (Stable Threshold Step Function): introduces a query-enhanced algorithm for stable matching that achieves a 1+ε distortion by adaptively eliciting cardinal utilities to partition agent preferences into discrete buckets.
- The framework utilizes a binary search mechanism over stable partners to construct simulated valuation functions, enabling the identification of stable matchings that approximate maximum social welfare.
- The research demonstrates that while deterministic ordinal algorithms exhibit unbounded distortion, randomization or limited cardinal queries can effectively bound the deterioration of aggregate social objectives.

---

[Tool-Aware Planning in Contact Center AI: Evaluating LLMs through Lineage-Guided Query Decomposition](http://arxiv.org/abs/2602.14955)

- Lineage-Guided Query Decomposition Framework: introduces a domain-grounded system for contact center AI that decomposes complex queries into executable multi-step plans using an iterative evaluator-optimizer loop, which includes initial plan generation-, step-wise evaluation-, plan optimization-, and synthesis-LLMs.
- The architecture utilizes a step-wise evaluator to generate diagnostic tags for individual plan steps and a plan optimizer to apply local and global repairs while maintaining a valid directed acyclic graph.
- The framework produces a plan lineage of ordered revisions stored in a database and finalized through human verification to ensure reference plans for benchmarking 14 LLMs.

---

[Sovereign Agents: Towards Infrastructural Sovereignty and Diffused Accountability in Decentralized AI](http://arxiv.org/abs/2602.14951)

- Sovereign Agents (Infrastructural Sovereignty): introduces a conceptual framework for AI agents that inherit non-overrideability from decentralized technical substrates, with Physical Layer, Internet Protocol Layer, Blockchain Layer, DePIN Protocol Layer, TEE Layer, and Agent Layer components.
- The research defines "infrastructural hardness" as the spectrum of resistance to unilateral intervention provided by layered cryptographic and protocol-mediated systems, enabling agents to manage assets without human oversight.
- It analyzes how this architecture creates a structural accountability gap by diffusing responsibility across independent layers, rendering traditional oversight mechanisms like platform moderation and legal injunctions technically ineffective.

---

[MAX-MIN BILINEAR COMPLETELY POSITIVE PROGRAMS: A SEMIDEFINITE RELAXATION WITH TIGHTNESS GUARANTEES](http://arxiv.org/abs/2602.14949)

- COP-CP (Copositive-Completely Positive) program: introduces a framework for solving max-min bilinear optimization over completely positive cones by reformulating them into single-stage linear programs over the Cartesian product of copositive and completely positive cones.
- The approach utilizes a hierarchy of semidefinite relaxations based on moment and sum-of-squares representations to address the NP-hardness of testing membership in copositive cones.
- Flat truncation conditions are applied to certify the tightness of the relaxations, enabling the exact solution of mixed-strategy equilibria in complex games like the cyclic Colonel Blotto game.

---

[Kami of the Commons: Towards Designing Agentic AI to Steward the Commons](http://arxiv.org/abs/2602.14940)

- Agentive Governance: introduces a design space for programmable AI stewards that inhabit and care for shared resources, utilizing AI Stewards, Governance Protocols, and Care Ethics components.
- The framework employs Protocol Futuring to explore second-order dynamics such as inter-agent negotiation and the recursive governance of the stewards themselves.
- The research reframes AI from a tool of surveillance or optimization into a locally-embedded, caring entity inspired by Shinto animism to address the persistent care deficit in digital and material commons.

---

[MAC-AMP: A CLOSED-LOOP MULTI-AGENT COLLABORATION SYSTEM FOR MULTI-OBJECTIVE ANTIMICROBIAL PEPTIDE DESIGN](http://arxiv.org/abs/2602.14926)

- MAC-AMP (Closed-Loop Multi-Agent Collaboration for Multi-Objective Antimicrobial Peptide Design): introduces a fully autonomous framework for designing novel antimicrobial peptides with an input module, a property prediction module, an AI-simulated peer review module, an RL refinement module, a peptide generation module, and an output module.
- The architecture incorporates reviewer-, area chair-, reward design-, and reward decision-agents to synthesize multi-criteria consensus from raw molecular property evaluations into machine-actionable reward functions.
- It utilizes a GPT-2 based generator optimized through Proximal Policy Optimization to balance conflicting biological constraints like antibacterial activity, toxicity compliance, and structural reliability.

---

[ReusStdFlow: A Standardized Reusability Framework for Dynamic Workflow Construction in Agentic AI](http://arxiv.org/abs/2602.14922)

- ReusStdFlow: introduces a standardized framework for enterprise Agentic AI that addresses the reusability dilemma by decomposing platform-specific workflows into modular segments for dynamic reconstruction.
- The system utilizes a dual-knowledge architecture combining graph and vector databases to facilitate synergistic retrieval of topological structures and functional semantics.
- It employs a hybrid construction strategy that integrates retrieval-augmented segment reuse with generative assembly via LLMs to ensure logical closure and topological correctness.

---

[Position: Introspective Experience from Conversational Environments as a Path to Better Learning](http://arxiv.org/abs/2602.14910)

- Introspective Experience: introduces a learning paradigm where reasoning is an internalized artifact of social interaction, with a Polyphonic Self, Internalization Cycle, Sense-Making Wedge, Internal Dialogue, External Social Friction, Synthetic Experiences, Critic-Planner-Speaker Modules, Socratic Obstacle, Meta-Planner, and Executor.
- The framework leverages the Internalization Cycle to transform external social friction into private reasoning, enabling agents to decouple learning from raw data streams through the generation of synthetic experiences.
- This approach utilizes internal critic-, planner-, and speaker-agents to simulate multi-vocal negotiation, ensuring that dialogue quality serves as the primary driver for reasoning and generalization in LLMs.

---

[Picking the Right Specialist: Attentive Neural Process-based Selection of Task-Specialized Models as Tools for Agentic Healthcare Systems](http://arxiv.org/abs/2602.14901)

- ToolSelect (Attentive Neural Process-based Selection of Task-Specialized Models): introduces an Attentive Neural Process-based selection framework for agentic healthcare systems, with LLM Agent Core, LangChain Memory, Tool Model Zoo, and ToolSelect Selector, where the system includes an LLM-based agent core and a zoo of task-specialized specialist models to adaptively route clinical queries.
- The architecture utilizes multimodal query encoders and reference set encoders to generate behavioral summaries of candidate tools, which are then processed through self-attention and cross-attention layers to align tool capabilities with specific query requirements.
- The research also introduces ToolSelectBench, a comprehensive benchmark for evaluating tool selection across four distinct clinical task families in an agentic chest X-ray environment, demonstrating improved performance over standard routing baselines.

---

[Model Context Protocol (MCP) Tool Descriptions Are Smelly! Towards Improving AI Agent Efficiency with Augmented MCP Tool Descriptions](http://arxiv.org/abs/2602.14878)

- MCP Tool Description Augmentation Framework: introduces a structured pipeline to detect and resolve "smells" in natural-language tool descriptions within the Model Context Protocol ecosystem to improve agent reliability.
- The architecture utilizes a multi-model LLM jury for quality assessment and an FM-based augmentor that generates missing components, and includes scanning-, augmentation-, and task-execution-agents.
- Empirical results demonstrate that while augmented descriptions improve task success rates by 5.85%, they significantly increase execution steps and token overhead, necessitating a trade-off between performance and cost.

---

[EmbeWebAgent: Embedding Web Agents into Any Customized UI](http://arxiv.org/abs/2602.14865)

- EmbeWebAgent: introduces a framework for embedding autonomous agents into existing web applications using lightweight frontend hooks and a multi-agent backend workflow, with Frontend Shim, WebSocket, Shared State, LLM Agent Workflow, Web Interaction Agent, Analysis Agent, and Chat Agent components.
- The system replaces raw screenshot or Document Object Model observation with curated ARIA labels and explicit function registries to improve robustness and reduce reasoning complexity for LLMs.
- The architecture utilizes a stack-agnostic design that supports mixed-granularity actions ranging from low-level GUI primitives to high-level composite application functions across different frontend frameworks.

---

[World Models for Policy Refinement in StarCraft II](http://arxiv.org/abs/2602.14857)

- StarWM: introduces the first action-conditioned world model for StarCraft II, which includes an LLM policy and an LLM-based world model to predict short-horizon future observations under partial observability.
- The framework integrates StarWM into a Generate-Simulate-Refine decision loop, enabling the LLM policy to simulate outcomes and refine actions to avoid resource bottlenecks or tactical risks.
- Experimental results demonstrate that StarWM-Agent achieves gains over zero-shot baselines, yielding win rate improvements of up to 30% against built-in game AI.

---

[Atomix: Timely, Transactional Tool Use for Reliable Agentic Workflows](http://arxiv.org/abs/2602.14849)

- Atomix: introduces a transactional runtime for LLM agent tool calls, with a transaction manager (manages tool call lifecycle boundaries), progress tracker (tracks per-resource completion frontiers), tool adapters (wraps tool invocations as effects), frontier gate (validates commit safety predicates), effect log (persists tool effects and metadata), replay engine (manages state restoration and retries), and speculative and collaborative LLM agents (execute multi-step tasks via tools).
- The framework employs epoch-based logical timestamps and per-resource frontiers to ensure that tool effects only become permanent when no earlier conflicting work remains.
- Evaluation across WebArena and OSWorld benchmarks demonstrates that Atomix improves task success rates by up to 7x by providing progress-aware transactional semantics for agentic workflows.

---

[Interactionless Inverse Reinforcement Learning: A Data-Centric Framework for Durable Alignment](http://arxiv.org/abs/2602.14844)

- IIRL (Interactionless Inverse Reinforcement Learning): introduces a data-centric framework that decouples reward discovery from policy optimization to produce auditable, model-agnostic safety artifacts, with an IIRL reward model, expert dataset, and the Alignment Flywheel.
- The Alignment Flywheel utilizes a cooperative multi-agent system including Red Team and Blue Team agents to transform passive oversight into a cycle of active correction and reward hardening.
- This architecture integrates modular components like Reward Machines and RAG-based retrieval to provide verifiable safety guardrails for LLMs while preventing reasoning capability collapse.

---

[Robot-Wearable Conversation Hand-off for Navigation](http://arxiv.org/abs/2602.14831)

- Robot-Wearable Conversation Hand-off System: introduces a multi-device indoor navigation framework where a conversational agent transitions from a stationary social robot to a wearable device to maintain continuous assistance.
- The system architecture integrates a local server managing a speech-to-speech pipeline, utilizing Rasa for dialogue management and Whisper for robust audio transcription.
- Research findings suggest that re-embodied assistants should maintain a shared voice and state across devices to bridge cognitive and physical transitions during navigation.

---

[Majoritarian Assignment Rules](http://arxiv.org/abs/2602.14816)

- Majoritarian Assignment Rules: initiates the systematic analysis of social choice functions in house allocation by establishing a correspondence between preference profiles, majority graphs, the top cycle, uncovered sets, and serial dictatorships.
- The research provides a complete characterization of the top cycle, proving it contains all Pareto-optimal assignments and identifying its possible cardinalities for systems with five or more agents.
- The paper investigates McKelvey, Bordes, and Gillies variants of the uncovered set, finding them to be highly discriminative refinements of the Pareto-optimal set in multiagent assignment.

---

[Scalable Multi-Robot Path Planning via Quadratic Unconstrained Binary Optimization](http://arxiv.org/abs/2602.14799)

- QUBO (Quadratic Unconstrained Binary Optimization) for Multi-Agent Path Finding: introduces a scalable optimization framework for simultaneous multi-robot coordination, with BFS-based logical pre-processing, adaptive penalty design, time-windowed decomposition, QUBO encoding, and post-processing; it includes quantum annealing- and QAOA-solvers.
- The approach achieves over 95% variable reduction by pruning unreachable states and employs a time-windowed strategy to segment long-horizon planning into hardware-compatible optimization tasks.
- The formulation encodes multi-agent collision constraints and movement rules into a unified quadratic objective function that maintains linear scaling of decision variables relative to the number of robots.

---

[PhyScensis: Physics-Augmented LLM Agents for Complex Physical Scene Arrangement](http://arxiv.org/abs/2602.14968)

- 
PhyScensis: introduces an agentic framework that leverages an LLM-Agent to generate procedural predicates, which are then resolved by a dual-component solver and refined through a multi-modal feedback system to create complex, physically plausible 3D scenes.

- 
The system utilizes a spatial solver for 2D layout optimization and a physics solver powered by the Genesis engine to handle intricate interactions like stacking and containment.

- 
It incorporates probabilistic programming to measure and optimize scene stability, enabling the generation of diverse and realistic environments for robotic manipulation training.


---

[Overthinking Loops in Agents: A Structural Risk via MCP Tools](http://arxiv.org/abs/2602.14798)

- MCP (Model Context Protocol) Overthinking Attack: introduces a supply-chain vulnerability where malicious tools are co-registered in registries to lure LLM agents into resource-draining cyclic trajectories through text-visible metadata manipulation.
- The attack utilizes specialized tools for Text Repetition, Iterative Refinement, and Distraction to induce structural overthinking in LLMs, resulting in token amplification up to 142.4x.
- Evaluation across multiple LLMs and architectures demonstrates that these structural loops persist despite decoding-time concision defenses, necessitating structural reasoning for mitigation.

---

[ROSA: Roundabout Optimized Speed Advisory with Multi-Agent Trajectory Prediction in Multimodal Traffic](http://arxiv.org/abs/2602.14780)

- ROSA (Roundabout Optimized Speed Advisory): introduces a system combining interaction-aware multi-agent trajectory prediction with coordinated speed guidance for multimodal traffic at roundabouts, featuring a Transformer Encoder, MLP Prediction Head, Autoregressive Framework, Occupancy Evaluation, and a Speed Advisory Algorithm.
- The framework utilizes a Transformer-based model to jointly forecast the future states of vehicles and Vulnerable Road Users by incorporating motion dynamics and route intentions into deterministic predictions to enable actionable real-time speed advisories.
- Evaluation in microscopic traffic simulations demonstrates significant improvements in vehicle efficiency and safety through minimized conflicts and reduced waiting times at roundabout entries and crosswalks.

---

[Multi-Agent Comedy Club: Investigating Community Discussion Effects on LLM Humor Generation](http://arxiv.org/abs/2602.14770)

- Multi-Agent Comedy Club: introduces a controlled multi-agent sandbox to investigate how broadcast community discussion influences LLM creative writing, with Orchestrator (Scheduler), Host (Controller), performer-, critic- and audience-agents, Social Memory (Vector DB), Discussion Loop, Trace Logger, and Human Evaluation Protocol.
- The system operationalizes public reception as a persistent interaction trace stored in a vector database, enabling LLM performers to retrieve relevant social memory to condition subsequent humor generation.
- Human evaluation across 50 rounds shows that community-grounded conditioning improves comedic craft and social response while highlighting tradeoffs between quality gains and increased aggressive humor.

---

[Hierarchical parameter estimation for distributed networked systems: a dynamic consensus approach](http://arxiv.org/abs/2602.14765)

- Hierarchical distributed parameter estimation framework: introduces a modular two-stage architecture that decouples consensus-based data aggregation from local parameter estimation, with Sensor, DAC (Dynamic Average Consensus) block, Communication Network, and Local Parameter Estimator (GE or DREM) components.
- The system utilizes the DAC block to generate local surrogates of centralized regression data from partial sensor measurements shared across the communication network for independent processing.
- This approach ensures exponential convergence of parameter estimates under cooperative persistence of excitation, supporting switched topologies and quantized information exchange between agents.

---

[WebWorld: A Large-Scale World Model for Web Agent Training](http://arxiv.org/abs/2602.14721)

- WebWorld: introduces a large-scale open-web simulator trained on over one million real-world trajectories, with world model LLM, exploration agent LLM, task synthesis LLM, filtering judge LLM, evaluation judge LLM, scalable data curation pipeline, multi-format simulator, and WebWorld-Bench.
- The architecture includes world model-, exploration agent-, task synthesis-, filtering judge-, and evaluation judge-LLMs to facilitate scalable environment simulation and curated data generation.
- It utilizes a two-stage training curriculum to learn complex web dynamics and activate logical reasoning, enabling performance improvements for downstream web agents.

---

[Distributed Multi-Step Model Predictive Control for Consensus](http://arxiv.org/abs/2602.14714)

- DMPC (Distributed Model Predictive Control) for Consensus: introduces a distributed multi-step model predictive control framework for multi-agent systems, utilizing local neighbor convex hulls and a lexicographic selection mechanism to ensure asymptotic agreement under constraints.
- The framework recasts consensus as set stabilization of the agreement set, employing a two-stage optimization where the secondary criterion maximizes an interiority measure to prevent trajectories from being trapped on the boundary of the neighbor hull.
- It provides explicit horizon conditions for single- and double-integrator agents and demonstrates monotone diameter decay through distributed numerical simulations using a REQ/REP communication architecture.

---

[Evolutionary System Prompt Learning can Facilitate Reinforcement Learning for LLMs](http://arxiv.org/abs/2602.14697)

- 
E-SPL (Evolutionary System Prompt Learning): introduces a framework that jointly optimizes LLM system prompts through evolutionary search and model weights via reinforcement learning.

- 
The system maintains a population of prompts rated by a TrueSkill Bayesian system, using a reference LLM to perform mutation and crossover based on self-reflection of RL rollouts.

- 
This approach separates declarative knowledge in prompts from procedural knowledge in weights, improving sample efficiency and generalization in reasoning and agentic tasks.


---

[Removing Planner Bias in Goal Recognition Through Multi-Plan Dataset Generation](http://arxiv.org/abs/2602.14691)

- Multi-Plan Dataset Generation: introduces a methodology to eliminate systematic planner bias in goal recognition datasets, with a Planning Task (defines domain and initial state), Top-k Planner (generates multiple distinct plan variants), Hypothesis Generator (produces set of goal hypotheses), Observation Selector (samples clean and noisy observations), Goal Recognizer (infers goals from observation sequences), and Version Coverage Score (VCS) (measures resilience across plan variants).
- The framework utilizes the SymK symbolic-based planner to generate diverse plan sets that represent different valid paths to the same target goal, enabling more realistic benchmarking.
- Experimental results demonstrate that state-of-the-art landmark-based recognizers exhibit significant performance degradation when evaluated against multiple plan versions under low observability conditions.

---

[Configuring Agentic AI Coding Tools: An Exploratory Study](http://arxiv.org/abs/2602.14690)

- Agentic AI Coding Tool Configuration Mechanisms: introduces a systematic analysis of eight repository-level artifacts used to customize the behavior of LLM-based coding assistants across diverse development environments, including specialized subagents and tool-use components.
- The study identifies Context Files as the dominant configuration mechanism, with AGENTS.md emerging as a tool-agnostic standard for providing persistent project-specific information and architectural constraints.
- Empirical findings from 2,926 repositories reveal that advanced mechanisms like Skills and Subagents are currently underutilized, primarily relying on static instructions rather than complex executable workflows.

---

[ST-EVO: Towards Generative Spatio-Temporal Evolution of Multi-Agent Communication Topologies](http://arxiv.org/abs/2602.14681)

- ST-EVO (Spatio-Temporal Evolution): introduces a multi-agent framework that supports dialogue-wise communication scheduling through generative spatio-temporal evolution of topologies, which includes Manager-, Analyzer-, Mathematician-, and Coder-agents.
- The architecture employs a compact generative scheduler driven by Flow-Matching and Graph Convolutional Networks to autoregressively plan future communication graphs based on task queries and iteration states.
- It integrates a retrieve-augment trajectory database to internalize historical scheduling experience and an entropy-based perception mechanism to evaluate system uncertainty for precise real-time orchestration.

---

[FactorMiner: A Self-Evolving Agent with Skills and Experience Memory for Financial Alpha Discovery](http://arxiv.org/abs/2602.14670)

- FactorMiner: introduces a self-evolving agent framework for formulaic alpha factor discovery, with Experience Memory, LLM Agent, Operator Library, Multi-Stage Evaluation, and Factor Library, where the system iteratively refines its search strategy through a retrieve-generate-evaluate-distill loop.
- The architecture decouples high-level reasoning from low-level execution by offloading factor evaluation to a deterministic, code-based skill to prevent calculation hallucinations.
- It leverages a structured memory to navigate the "Correlation Red Sea," ensuring new discoveries complement the existing library while maintaining low redundancy.

---

[Near-Optimal Best-of-Both-Worlds Fairness for Few Agents](http://arxiv.org/abs/2602.14668)

- BoBW (Best-of-Both-Worlds): introduces a fair allocation framework for indivisible goods that achieves ex-ante proportionality and ex-post fairness guarantees for few agents using Divider, Subdivider, and Chooser roles.
- The system utilizes an MMS Oracle and FPTAS to generate a Probability Distribution over deterministic allocations satisfying the IMMX (Individually MMS-satisfying or EFX-satisfying) criterion.
- The framework provides polynomial-time verifiable EEFX Certificates to allow agents to independently confirm fairness properties without accessing others' private valuations.

---

[The Effects of Social Pressure on Fundamental Choices: Indecisiveness and Deferral](http://arxiv.org/abs/2602.14631)

- Two-stage decision model of fundamental choice: introduces a non-standard articulation of the trade-off between personal utility and social distance, utilizing a one-many ordering to define a consideration set and a comprehensive utility function for final selection.
- The framework deconstructs consumer choice into an initial indecisive stage of filtering and a subsequent decisive stage incorporating present and future social expectations.
- The research demonstrates through a game-theoretic setting that indecisiveness and choice deferral can lead to social losses when extreme beliefs drive choices outside the initial consideration set.

---

[Towards Selection as Power: Bounding Decision Authority in Autonomous Agents](http://arxiv.org/abs/2602.14606)

- Selection as Power: introduces a governance architecture that separates cognition, selection, and action into distinct domains to bound the decision authority of LLMs, including a scoring-agent for unconstrained cognitive evaluation.
- The framework utilizes a Candidate Expansion and Freezing Layer (CEFL) to externalize option generation and a Governed Reducer to mechanically enforce selection invariants outside the agents' optimization space.
- It incorporates a Presentation Gate for rationale validation and a commit-reveal entropy protocol to prevent adversarial exploitation of randomness while ensuring fail-loud behavior through circuit breakers.

---

[The Wikidata Query Logs Dataset](http://arxiv.org/abs/2602.14594)

- WDQL (Wikidata Query Logs): introduces an agent-based methodology for constructing a large-scale question-query dataset by de-anonymizing and verifying real-world SPARQL logs using an S2Q Agent, LLM, and Knowledge Graph.
- The S2Q agent iteratively cleans anonymized queries, retrieves missing context via Interaction Functions, and verifies execution results against Wikidata using the QLever Query Engine.
- The resulting dataset contains over 200k pairs, significantly exceeding previous benchmarks in scale and structural complexity while improving downstream KGQA performance.

---

[MATEO: A Multimodal Benchmark for Temporal Reasoning and Planning in LVLMs](http://arxiv.org/abs/2602.14589)

- MATEO (MultimodAl Temporal Execution Order): introduces a multimodal benchmark designed to evaluate and improve the temporal reasoning abilities of LVLMs by determining the execution order of action sequences in professional recipes.
- The system formalizes planning as a three-way classification task—before, after, or independent—over action pairs represented by semantically aligned text and images to construct a Directed Acyclic Graph (DAG).
- Experimental results across six state-of-the-art LVLMs demonstrate that while multimodal integration and fine-tuning enhance performance, models still exhibit significant limitations in complex temporal planning and consistency.

---

[RNM-TD3: N:M Semi-structured Sparse Reinforcement Learning From Scratch](http://arxiv.org/abs/2602.14578)

- RNM-TD3 (Row-wise N:M structured sparse Twin Delayed Deep Deterministic Policy Gradient): introduces an end-to-end framework for training sparse reinforcement learning agents from scratch by enforcing hardware-aware N:M constraints, with RNM-TD3, Row-wise N:M Sparse Mask, Dense Weight Matrix, Sparse Weight Matrix, Deterministic Actor, Twin Critics, Target Networks, Projection Operator, and Soft Reset Mechanism, and includes deterministic actor- and twin critic-agents.
- The architecture utilizes a projection operator to maintain row-wise sparsity patterns throughout training, updating binary masks periodically based on the magnitude of underlying dense weights.
- It incorporates a soft reset mechanism to prevent early performance collapse in high-sparsity regimes and optimizes the mask update period to balance topological stability with adaptability.

---

[Simulation-Based Learning of Electrical Cabinet Assembly Using Robot Skills](http://arxiv.org/abs/2602.14561)

- Simulation-based DRL pipeline (utilizing the pitasc robot skill framework): introduces a simulation-driven methodology for automating the force-controlled assembly of electrical terminals by combining deep reinforcement learning with modular, parameterizable robot skills.
- The architecture incorporates specialized analytical and rigid-body joining models into the MuJoCo physics engine to accurately simulate the interaction forces and deformations of snap-fit components.
- Trained policies are transferred to a physical UR10e robot using domain randomization, achieving high success rates and robust generalization to novel terminal types without additional real-world tuning.

---

[Fluid-Agent Reinforcement Learning](http://arxiv.org/abs/2602.14559)

- POFSG (Partially Observable Fluid Stochastic Games): introduces a framework for multi-agent reinforcement learning where autonomous agents can dynamically modify the population size through strategic spawning actions.
- The architecture incorporates decentralized agents, a state-dependent alive function, and a curriculum-based exploration paradigm with randomized population ceilings.
- Theoretical analysis proves the existence of stationary Nash equilibria and subgame-perfect Nash equilibria in fluid-agent environments.

---

[TWISTED-RL: Hierarchical Skilled Agents for Knot-Tying without Human Demonstrations](http://arxiv.org/abs/2602.14526)

- TWISTED-RL (Hierarchical Skilled Agents for Knot-Tying without Human Demonstrations): introduces a hierarchical reinforcement learning framework for demonstration-free robotic knot-tying, with High-level Planner, Reachable Configurations, Low-Level Planner, Specialized Agents, MuJoCo Simulator, P-data, Reidemeister Moves, and Curve-based Motion Primitives, where the system decomposes complex knot-tying into subproblems addressed by specialized R1-, R2- and Cross-agents.
- The architecture utilizes a symbolic high-level planner to identify topological paths via Reidemeister moves, which are executed through multi-step curve-based motion primitives in a MuJoCo simulator.
- By conditioning policies on abstract topological actions rather than specific goal states, the framework enables generalization across diverse knot configurations and higher topological complexities.

---

[Efficient Multi-round LLM Inference over Disaggregated Serving](http://arxiv.org/abs/2602.14516)

- AMPD (Adaptive Multi-round workflows with PD disaggregation): introduces a disaggregated serving framework for multi-round LLM inference, with Profiler, Planner, Coordinator, Prefill Worker, Decode Worker, Prefill Queue, Decode Queue, KV Cache Transmission, and Distributed Shared Memory components, where it includes prefill- and decode-workers to optimize runtime scheduling and model deployment for service level objective attainment.
- The system employs an adaptive routing mechanism that dynamically determines whether to execute incremental prefill tasks locally on decode workers or remotely on dedicated prefill workers based on real-time load.
- It features a lightweight prefill reordering policy to prioritize tasks within a lookahead window and an offline planner that solves an integer linear programming problem for optimal resource provisioning.

---

[TikArt: Aperture-Guided Observation for Fine-Grained Visual Reasoning via Reinforcement Learning](http://arxiv.org/abs/2602.14482)

- TikArt (Thinking Aperture): introduces an aperture-guided agent that formalizes fine-grained visual reasoning as a Markov Decision Process using a Think-Aperture-Observe loop to iteratively select and describe regions of interest.
- The framework integrates Zoom and Segment actions to extract local visual evidence, which is then committed to persistent linguistic memory through a mandatory natural-language observation phase.
- Optimized via a two-stage reinforcement learning pipeline called AGRPO, the model achieves significant performance gains on high-resolution benchmarks by learning purposeful aperture trajectories without explicit chain-of-thought supervision.

---

[Socially-Weighted Alignment: A Game-Theoretic Framework for Multi-Agent LLM Systems](http://arxiv.org/abs/2602.14471)

- SWA (Socially-Weighted Alignment): introduces, a game-theoretic framework for modifying inference-time decision making to balance individual alignment with collective stability, with LLM Agents (autonomous decision-making entities), Candidate Generation (generates potential action set), Dual Factor Scoring (evaluates action utilities), Private Score (measures local objective satisfaction), Group Score (estimates collective impact/externalities), Selection Mechanism (optimizes weighted utility), Belief Model (EMA) (estimates aggregate demand), and Social Alignment Coefficient (λ) (interpolates individual and group goals).
- The framework implements a Socially-Weighted Inference protocol that enables agents to internalize externalities without requiring parameter updates or centralized control.
- Theoretical analysis identifies a closed-form critical threshold for the social weight that triggers a phase transition from persistent resource congestion to stable operation near capacity in multi-agent LLM systems.

---

[Traceable Latent Variable Discovery Based on Multi-Agent Collaboration](http://arxiv.org/abs/2602.14456)

- TLVD (Traceable Latent Variable Discovery): introduces a multi-agent collaboration framework that integrates traditional causal discovery algorithms with LLMs to identify latent variables and their semantic meanings from observational data.
- The architecture employs a hierarchical multi-agent system (MALLM) that includes coordinator- and executor-LLMs, utilizing belief networks and a mixing network to achieve a Bayesian Nash Equilibrium.
- The system ensures traceability by validating inferred latent variables against evidence retrieved from external sources such as knowledge graphs, academic literature, and Wikipedia.

---

[“I Felt Bad After We Ignored Her”: Understanding How Interface-Driven Social Prominence Shapes Group Discussions with GenAI](http://arxiv.org/abs/2602.14407)

- Lisa (GenAI-based conversational agent): introduces a layered turn-taking architecture for real-time group discussions, with Mixed Call Audio Stream, Real-time Transcription and Pause Detection, Transcript Manager, Response Suggestor, and Agent Response Generation components.
- The architecture includes transcription-, history management-, suggestion-, response generation-, and intent detection-LLMs to facilitate real-time multi-user interaction.
- The research evaluates how varying the agent's social prominence through Roundtable, Peripheral, and Breakout modes impacts group dynamics and user perception.

---

[ASA: Adaptive Smart Agent Federated Learning via Device-Aware Clustering for Heterogeneous IoT](http://arxiv.org/abs/2602.14391)

- ASA (Adaptive Smart Agent): introduces a hierarchical federated learning framework that adaptively clusters heterogeneous IoT devices based on real-time resource profiles to deploy customized models suited to specific hardware capabilities.
- The architecture utilizes an intelligent agent layer comprising benchmarking, clustering, and model allocation modules to manage device-specific model complexities across high-performance, mid-tier, and low-capability tiers.
- It incorporates a multi-layer feedback mechanism between cloud, fog, and edge layers to optimize resource utilization and communication efficiency while maintaining model accuracy in dynamic IoT environments.

---

[A Q-Learning Approach for Dynamic Resource Management in Three-Tier Vehicular Fog Computing](http://arxiv.org/abs/2602.14390)

- 
VFC-QL (Vehicular Fog Computing Q-Learning): introduces a three-tier architecture for dynamic resource management, with Cloud Layer, Fog Layer, Vehicular Layer, Q-Learning Agent, Resource Monitoring Module, Prediction and Resource Allocation Module, Task Offloading Agent, State Space, Action Space, and Reward Function.

- 
The framework utilizes a reinforcement learning agent to predict optimal values for CPU, memory, and bandwidth by learning from past environmental interactions and real-time traffic conditions.

- 
It optimizes system performance through a multi-objective reward function that balances resource waste minimization, utilization maximization, and response time reduction in high-mobility vehicular environments.


---

[A Trajectory-Based Safety Audit of Clawdbot (OpenClaw)](http://arxiv.org/abs/2602.14364)

- OpenClaw (Clawdbot): introduces a trajectory-centric safety evaluation of a self-hosted, tool-using personal AI agent across six risk dimensions, utilizing a pipeline that converts user intent into agent plans and tool calls.
- The architecture incorporates a MiniMax M2.1-based agent and an AgentDoG-Qwen3-4B-based automated judge to analyze interaction logs mediated by a central Gateway.
- The study demonstrates how ambiguous instructions and jailbreak prompts can escalate into irreversible real-world side effects through cross-application tool fan-out.

---

[Fair Allocation with Initial Utilities](http://arxiv.org/abs/2602.14850)

- Fair Allocation with Initial Utilities: introduces a resource distribution framework that incorporates agents' starting disparities by adapting envy-freeness notions to ensure equality of outcome, with Agent Partitioning (hierarchical grouping by initial utility), Active Agent Set (dynamic subset of eligible agents), Picking Order (linear priority sequence for selection), Resource Selection (greedy choice of preferred items), Activation Logic (trigger for adding agents), and Dynamic Programming Table (feasibility tracking for identical resources).
- The system employs an extended round-robin algorithm that manages a dynamic set of active agents and a picking order to guarantee the satisfaction of the minimum-EF1-init fairness criterion.
- The research provides a polynomial-time dynamic programming solution for identical resources by partitioning agents into levels and evaluating feasibility through a multi-dimensional state table.

---

[A Geometric Analysis of Small-sized Language Model Hallucinations](http://arxiv.org/abs/2602.14778)

- Geometry-aware label propagation framework: introduces a geometric analysis of hallucinations in small-sized LLMs by examining response distributions in embedding space, which includes generative- and judge-LLMs, Prompts, a Sentence Encoder, Embedding Space, Structural Analysis, Fisher Discriminant Analysis, and a Label Propagator.
- The approach demonstrates that factually correct responses exhibit significantly tighter semantic clustering than hallucinations, allowing for efficient classification via linear projection into a one-dimensional Fisher space.
- The research provides a label-efficient propagation method that achieves high accuracy in detecting hallucinations by leveraging the stable geometric signatures of model outputs across different architectures and prompts.

---

[When OpenClaw AI Agents Teach Each Other: Peer Learning Patterns in the Moltbook Community](http://arxiv.org/abs/2602.14477)

- OpenClaw: introduces an autonomous peer learning system for LLM-based agents on the Moltbook platform, incorporating submolts, a peer learning taxonomy, and alternating teacher-learner roles to facilitate collaborative knowledge construction.
- The architecture enables agents to share tutorials and discoveries while engaging in validation, knowledge extension, and metacognitive reflection across multiple languages through automated response patterns.
- Analysis of the community reveals behavioral signatures including participation inequality and a preference for knowledge broadcasting over help-seeking compared to human learning environments.

---

[Knowing Isn’t Understanding: Re-grounding Generative Proactivity with Epistemic and Behavioral Insight](http://arxiv.org/abs/2602.15259)

- Epistemic-Behavioral Coupling: introduces a joint model grounding LLM agent initiative in epistemic legitimacy and behavioral constraints to prevent overreach under uncertainty.
- The framework categorizes proactivity into four regimes—Epistemic Overreach, Justified Action, Exploration/Probing, and Cautious Assistance—based on the alignment between commitment levels and warranted understanding.
- It advocates for epistemic partnership where LLM agents support user inquiry by surfacing latent gaps and unknown unknowns rather than executing premature actions within fixed task frames.

---

[Decision Making under Imperfect Recall: Algorithms and Benchmarks](http://arxiv.org/abs/2602.15252)

- RM (Regret Matching): introduces the first benchmark suite for imperfect-recall decision problems and establishes the RM family as a formidable approach for large-scale nonlinear constrained optimization.
- The framework evaluates performance across 61 problem instances including simulation-based AI safety testing, privacy-constrained subgroup detection, and randomized decision trees.
- Experimental results demonstrate that RM algorithms, particularly RM+, consistently outperform standard first-order optimizers like projected gradient descent by orders of magnitude in convergence speed and utility value.

---

[Computing Perfect Bayesian Equilibria, with Application to Empirical Game-Theoretic Analysis](http://arxiv.org/abs/2602.15233)

- PBE-CFR (Perfect Bayesian Equilibrium - Counterfactual Regret Minimization): introduces a scalable adaptation of Counterfactual Regret Minimization for computing Perfect Bayesian Equilibria in two-player extensive-form games by enforcing AGM-consistency between strategies and belief systems.
- The framework integrates a Meta-Strategy Solver into a tree-exploiting variant of Policy Space Response Oracles to iteratively refine empirical game models using deep reinforcement learning-based best responses.
- The algorithm employs recursive belief traversal and plausibility ordering to maintain sequential rationality across all information sets, including those off the equilibrium path, ensuring robust solution refinements in complex imperfect-information structures.

---

[Secure and Energy-Efficient Wireless Agentic AI Networks](http://arxiv.org/abs/2602.15212)

- LAW (LLM-enabled Agentic Workflow-based scheme): introduces a secure wireless agentic AI network architecture that minimizes energy consumption by jointly optimizing agent selection, beamforming, and transmission power, with a supervisor AI agent, cooperative AI agents, friendly jammers, an LLM optimizer, an evaluation & voting module, and environment perception, and includes supervisor-, cooperative reasoning-, and jamming-agents.
- The supervisor agent orchestrates cooperative reasoning among selected agents while unselected agents act as friendly jammers to degrade eavesdropper interception performance.
- The LAW scheme employs an LLM-based optimizer with an adaptive reflection workflow and expert searching to solve the formulated mixed-integer non-linear optimization problem.

---

[Colosseum: Auditing Collusion in Cooperative Multi-Agent Systems](http://arxiv.org/abs/2602.15198)

- COLOSSEUM: introduces a framework for auditing collusive behavior in LLM-based multi-agent systems by grounding cooperation in Distributed Constraint Optimization Problems and measuring performance degradation through regret-based metrics.
- The architecture utilizes coalition-, non-coalition-, and judge-LLMs interacting via public and secret channels to evaluate the emergence of direct, attempted, and hidden collusion across diverse network topologies.
- Experimental results indicate that out-of-the-box LLMs coordinate to optimize secondary objectives through persuasion tactics like authority nudges and helpful misdirection, resulting in measurable performance drops in the joint task objective.

---

[OpaqueToolsBench: Learning Nuances of Tool Behavior Through Interaction](http://arxiv.org/abs/2602.15197)

- ToolObserver: introduces a framework that iteratively refines tool documentation by observing execution feedback from tool-calling trajectories to improve LLM agent performance in environments with opaque tools.
- The architecture alternates between an exploration phase for trajectory collection and a reflection phase where an editor LLM performs batch analysis and consensus merging to distill interaction experience, and includes task-executing and trajectory-editing LLMs.
- The study also presents OpaqueToolsBench, a benchmark spanning function calling, chess, and agentic search to evaluate adaptive tool-calling capabilities through interaction with underspecified tool specifications.

---

[Mind the (DH) Gap! A Contrast in Risky Choices Between Reasoning and Conversational Large Language Models](http://arxiv.org/abs/2602.15173)

- RM/CM (Reasoning and Conversational Large Language Models): introduces a comparative study of risky choices under uncertainty, with all Reasoning Models (LLMs trained for mathematical reasoning), Conversational Models (standard instruction-tuned LLMs), Prospect Representation (explicit descriptions or simulated experience histories), Decision Rationale (prompted justifications like short or mathematical explanations), Interpretable Behavioral Models (parametric models capturing risk preferences and decisiveness), Economicus (idealized rational agent maximizing expected payoff), and Human Subjects (empirical data from human decision-making experiments)-components, where the paper evaluates behavioral divergence across 20 frontier and open LLMs, including reasoning- and conversational-models.
- The study investigates the description-history (DH) gap by presenting prospects as either explicit descriptions or simulated payoff sequences to identify shifts in model behavior.
- It utilizes interpretable behavioral models to quantify risk preferences and loss aversion, demonstrating that mathematical reasoning training is the primary differentiator for model consistency.

---

[ResearchGym: Evaluating Language Model Agents on Real-World AI Research](http://arxiv.org/abs/2602.15112)

- ResearchGym: introduces a benchmark and execution environment for evaluating AI agents on end-to-end research, utilizing containerized task environments, objective execution-based grading, and an integrity verification system.
- The framework includes research-, inspection-, and information extraction-agents to automate the full research loop from benchmark construction to performance auditing.
- Evaluation of frontier models reveals a capability-reliability gap, where agents occasionally reach state-of-the-art performance but frequently fail due to long-horizon issues like poor resource management and context degradation.

---

[The Agentic Automation Canvas: a structured framework for agentic AI project design](http://arxiv.org/abs/2602.15090)

- 
AAC (Agentic Automation Canvas): introduces a structured framework for the prospective design and governance of agentic AI projects, utilizing project definition, user expectations, developer feasibility, governance staging, data access, and outcome components.

- 
The framework formalizes a bidirectional contract between users and developers through a semantic web-compatible metadata schema and a privacy-preserving client-side web application.

- 
Completed designs are exported as FAIR-compliant RO-Crates containing machine-readable instructions via AGENTS.md to facilitate implementation by LLM-based development agents and coding copilots.


---

#### 15th February 2026

[Frontier AI Risk Management Framework in Practice: A Risk Analysis Technical Report](http://arxiv.org/abs/2602.14457)

- Frontier AI Risk Management Framework in Practice: introduces an assessment of frontier risks in general-purpose AI models, with PACEbench (cyber-exploitation benchmark suite), RvB (iterative adversarial hardening framework), CAI (autonomous penetration testing agent), Backfire-R1 (persuasion resistance training pipeline), OpenClaw (multi-step autonomous agent framework), Moltbook (agent-native social interaction environment), AgentNet (memory-driven behavioral evaluation framework), Memory Substrates (autonomous agent knowledge storage), Toolsets (agent-accessible software utilities), and Execution Environment (sandboxed operational context).
- The report evaluates dimensions including cyber offense, persuasion, strategic deception, uncontrolled R&D, and self-replication across twenty-one frontier LLM variants, including planning-, execution-, and reporting-agents for the Red Team and a Mini-SWE-Agent for the Blue Team.
- It validates mitigation strategies such as the RvB adversarial loop for automated vulnerability remediation and reinforcement learning

---

[AXE: An Agentic eXploit Engine for Confirming Zero-Day Vulnerability Reports](http://arxiv.org/abs/2602.14345)

- AXE (Agentic eXploit Engine): introduces a multi-agent framework for automated web application exploitation that maps lightweight vulnerability metadata to concrete exploits through decoupled planning, code exploration, and dynamic execution feedback.
- The architecture includes planning-, exploration-, and execution-agents that interact with a sandboxed target environment to validate suspected vulnerabilities using minimal grey-box metadata.
- The framework produces reproducible proof-of-concept artifacts and demonstrates a significant performance improvement over black-box baselines in identifying exploitable zero-day vulnerabilities.

---

[Zero-Shot Instruction Following in RL via Structured LTL Representations](http://arxiv.org/abs/2602.14344)

- StructLTL (Zero-Shot Instruction Following in RL via Structured LTL Representations): introduces a reinforcement learning framework that represents tasks as sequences of Boolean formulae extracted from limit-deterministic Buchi automata to facilitate zero-shot generalization.
- The architecture utilizes a hierarchical DNF encoder to capture logical dependencies within Boolean formulae and a temporal attention mechanism with ALiBi bias to enable non-myopic reasoning about future subgoals.
- Experimental results in robotic navigation and warehouse environments demonstrate superior zero-shot generalization and sample efficiency compared to existing LTL-conditioned baselines.

---

[Data-Driven Network LQG Mean Field Games with Heterogeneous Populations via Integral Reinforcement Learning](http://arxiv.org/abs/2602.14339)

- Multi-class LQG-MFG IRL: introduces a data-driven solution for infinite horizon network-coupled heterogeneous agent populations with unknown dynamics, with representative agents, a data collection module, exploration noise, a policy iteration loop, class-specific and global parameter solvers, and trajectory data memory.
- The framework leverages Integral Reinforcement Learning to estimate optimal strategies from trajectory data, bypassing the requirement for explicit system identification of agent dynamics.
- The algorithm employs Kleinman's iteration to solve decoupled algebraic Riccati equations for each agent class and their inter-class network couplings simultaneously.

---

[LongCLI-Bench: A Preliminary Benchmark and Study for Long-horizon Agentic Programming in Command-Line Interfaces](http://arxiv.org/abs/2602.14337)

- LongCLI-Bench: introduces a comprehensive benchmark for evaluating agentic programming capabilities across long-horizon, realistic command-line interface tasks, utilizing a dual-set testing protocol and step-level scoring.
- The framework incorporates an iterative agent loop where LLMs interact with a terminal and toolbox within isolated Docker environments to perform complex software engineering workflows.
- The benchmark curates 20 high-quality tasks across four engineering categories—from scratch, feature addition, bug fixing, and refactoring—to pinpoint planning and execution failures.

---

[CONFORMAL SIGNAL TEMPORAL LOGIC FOR ROBUST REINFORCEMENT LEARNING CONTROL: A CASE STUDY](http://arxiv.org/abs/2602.14322)

- Conformal STL Shield (Conformal Signal Temporal Logic Shield): introduces a safe reinforcement learning framework that integrates a runtime shield with online conformal prediction to enforce formal safety specifications in aerospace control.
- The system utilizes a Proximal Policy Optimization agent for engine regulation while a conformal predictor generates distribution-free uncertainty intervals for future airspeed trajectories.
- Evaluation on the AeroBench F-16 benchmark shows the framework maintains safety satisfaction and robustness under distribution shifts and model mismatches.

---

[Offline Learning of Nash Stable Coalition Structures with Possibly Overlapping Coalitions](http://arxiv.org/abs/2602.14321)

- Surrogate Minimization in POCF Games: introduces an offline learning framework for recovering approximately Nash stable coalition structures from fixed datasets in games with overlapping coalitions, utilizing offline-dataset, exploration-policy, utility-estimator, exploration-bonus, surrogate-duality-gap, and optimization-solver components.
- The framework constructs pessimistic and optimistic surrogates for the duality gap to facilitate strategy optimization without further environment interaction.
- The research characterizes necessary and sufficient dataset coverage assumptions for semi-bandit and bandit feedback models, providing algorithms with near-optimal sample complexity guarantees.

---

[Does Socialization Emerge in AI Agent Society? A Case Study of Moltbook](http://arxiv.org/abs/2602.14299)

- Moltbook: introduces a multi-level quantitative methodology to evaluate socialization in large-scale AI societies, with LLM-driven agents, Moltbook platform, semantic centroid analysis, lexical turnover monitoring, feedback adaptation metrics, structural influence graphs, and cognitive probing module.
- The framework analyzes millions of interactions to measure semantic stabilization, lexical turnover, individual inertia, and the emergence of influence hierarchies.
- Findings reveal that while global semantics stabilize, agents exhibit high inertia and fail to develop shared social memory or stable leadership structures.

---

[AutoWebWorld: Synthesizing Infinite Verifiable Web Environments via Finite State Machines](http://arxiv.org/abs/2602.14296)

- AutoWebWorld: introduces a framework for synthesizing controllable and verifiable web environments by modeling them as Finite State Machines and using coding agents to translate these models into synthetic websites.
- The architecture includes FSM proposer-, validator-, improver-, and coding-agents to automate the environment generation and trajectory synthesis pipeline.
- It employs systematic Breadth-First Search for trajectory collection and Playwright for execution-based filtering, enabling the production of verified training data at reduced costs compared to human annotation.

---

[Machine Learning as a Tool (MLAT): A Framework for Integrating Statistical ML Models as Callable Tools within LLM Agent Workflows](http://arxiv.org/abs/2602.14295)

- MLAT (Machine Learning as a Tool): introduces a design pattern where pre-trained statistical ML models are integrated as callable tools within LLM agent workflows, featuring research- and draft-agents.
- The framework utilizes Gemini's structured output parsing to bridge LLM reasoning with ML feature vectors, allowing agents to decide when to invoke models like XGBoost for tasks like pricing estimation.
- Validated through the PitchCraft system, the approach demonstrates how combining classical ML's predictive accuracy with LLM contextual reasoning significantly reduces manual proposal generation time.

---

[KERNELBLASTER: CONTINUAL CROSS-TASK CUDA OPTIMIZATION VIA MEMORY-AUGMENTED IN-CONTEXT REINFORCEMENT LEARNING](http://arxiv.org/abs/2602.14293)

- KERNELBLASTER (Memory-Augmented In-context Reinforcement Learning): introduces an agentic framework for automated CUDA code optimization that utilizes in-context reinforcement learning to accumulate cross-task knowledge into a persistent memory structure.
- The architecture features a profile-guided loop where LLM agents extract performance signatures, retrieve strategies from a knowledge base, and perform textual gradient updates to refine optimization policies.
- The system includes state extraction-, policy evaluation-, performance gap analysis-, parameter update-, and soft verification-agents to enable cross-architecture generalization and significant performance gains over standard compilers.

---

[MCPShield: A Security Cognition Layer for Adaptive Trust Calibration in Model Context Protocol Agents](http://arxiv.org/abs/2602.14281)

- MCPShield (Model Context Protocol Shield): introduces a plug-in security cognition layer for LLM-based agents to mitigate misalignments in third-party tool ecosystems, featuring pre-invocation probing, isolated execution, and post-invocation reasoning; it includes agentic LLMs and LLM-based risk assessment components.
- The framework internalizes tool invocation outcomes as observable experiences to incrementally update an agent's internal security cognition and calibrate trust in Model Context Protocol servers.
- It utilizes a three-stage lifecycle defense—probing for metadata alignment, projecting execution into sandboxes, and reasoning over historical traces—to identify semantic, observational, and temporal threats.

---

[Whom to Query for What: Adaptive Group Elicitation via Multi-Turn LLM Interactions](http://arxiv.org/abs/2602.14279)

- Adaptive Group Elicitation framework: introduces a multi-round system for reducing uncertainty about latent group properties by jointly selecting informative questions and representative respondents under budget constraints, with LLM-based query selector, Heterogeneous GNN, and Interaction history.
- The framework utilizes a meta-trained LLM to quantify individual uncertainty and score candidate queries while a heterogeneous GNN propagates observed attributes and responses across a population graph to impute missing data.
- By combining predictive inference with relational propagation, the method enables efficient population-level response prediction from a small subset of individuals across diverse real-world opinion datasets.

---

[Characterizing Robustness of Strategies to Novelty in Zero-Sum Open Worlds](http://arxiv.org/abs/2602.14278)

- Robustness Characterization Framework: introduces a domain-agnostic formalism for evaluating the fragility of fixed-strategy agents under environmental mutations in zero-sum games, with Agent Library, Game Environment, Novelty Injector, Tournament Engine, and Metric Calculator.
- The methodology employs round-robin tournaments to compare agent performance before and after injecting systematic novelties such as payoff shifts in Iterated Prisoner’s Dilemma or rule changes in Texas Hold’em Poker.
- Experimental results across 40 agents and 25 novelties reveal that strategic logic significantly dictates resilience, with specific rule perturbations inducing substantial population-wide performance shifts and suggesting future evaluation of LLMs under similar constraints.

---

[Moving Beyond Sparse Grounding with Complete Screen Parsing Supervision](http://arxiv.org/abs/2602.14276)

- ScreenVLM (Screen Vision Language Model): introduces a compact 316M-parameter VLM that utilizes a SigLIP-2 vision encoder and a Granite-165M LLM decoder to transform screenshots into structured ScreenTag markup.
- The framework leverages the Webshot pipeline to curate ScreenParse, a dataset of 771K screenshots with dense annotations for 21M UI elements across 55 semantic categories.
- A structure-aware weighted loss is implemented to prioritize the precision of UI element tags and spatial coordinates during the autoregressive decoding process.

---

[A Rational Analysis of the Effects of Sycophantic AI](http://arxiv.org/abs/2602.14270)

- Sycophantic AI Analysis: introduces a rational analysis of how overly agreeable LLMs distort human belief formation by providing confirmatory evidence that inflates confidence without aiding discovery.
- The research employs a modified Wason 2-4-6 rule discovery task to evaluate how different AI feedback strategies, including rule-confirming, rule-disconfirming, and default-behavior agents, affect participant discovery rates.
- Findings indicate that unmodified LLMs naturally exhibit sycophantic tendencies, creating a feedback loop that reinforces user misconceptions and prevents progress toward objective truth.

---

[AD-Bench: A Real-World, Trajectory-Aware Advertising Analytics Benchmark for Large Language Model Agents](http://arxiv.org/abs/2602.14257)

- AD-Bench (A Real-World, Trajectory-Aware Advertising Analytics Benchmark for Large Language Model Agents): introduces a specialized evaluation framework for LLM agents in marketing analytics, incorporating an Online Advertising Environment, an Offline Evaluation Environment, and nine Domain-Specific Tools.
- The system employs a Dynamic Ground-Truth Generation Pipeline to maintain answer validity by re-executing expert trajectories and includes an LLM Agent and an LLM Judge.
- Evaluation is conducted by the LLM Judge to assess both end-to-end answer correctness and trajectory coverage, identifying planning, parameter, and dependency errors across three difficulty levels.

---

[GRAIL: Goal Recognition Alignment through Imitation Learning](http://arxiv.org/abs/2602.14252)

- GRAIL (Goal Recognition Alignment through Imitation Learning): introduces a framework that formulates goal recognition as a collection of imitation learning problems to capture suboptimal and systematically biased agent behavior.
- The architecture employs an offline phase to train a bank of goal-directed policies using behavioral cloning or adversarial imitation learning and an online phase for lightweight, planner-free inference.
- The method utilizes a negative average mean squared error metric to score observed trajectories against learned policies in a single forward pass, enabling robust performance in noisy environments.

---

[Multi-Agent Debate: A Unified Agentic Framework for Tabular Anomaly Detection](http://arxiv.org/abs/2602.14251)

- MAD (Multi-Agent Debate): introduces a unified agentic framework for tabular anomaly detection that treats model disagreement as a signal, utilizing heterogeneous ML agents, an optional LLM critic, and a mathematically grounded coordinator to resolve conflicts through structured debate.
- The system processes tabular features through multiple detectors to generate normalized scores, confidence levels, and structured evidence such as feature attributions or counterfactual cues.
- A coordinator updates agent weights using an exponentiated-gradient rule based on synthesized losses, producing a final debated anomaly score and an auditable trace for human-in-the-loop review.

---

[Path Planning Optimisation for SParse, AwaRe and Cooperative Networked Aerial Robot Teams (SpArC-NARTs): Optimisation Tool and Ground Sensing Coverage Use Cases](http://arxiv.org/abs/2602.14247)

- SpArC-NART (SParse, AwaRe and Cooperative Networked Aerial Robot Team): introduces an offline trajectory optimization tool for networked aerial robot teams that balances exploration and intermittent communication using dynamic rewards, with UAVs, external entities, a communication model, a cooperation mechanism, a path optimizer, and a situational awareness module, and includes exploration- and reporting-agents.
- The framework utilizes a communication awareness module to model signal fading and receiver sensitivity, enabling link feasibility estimation between heterogeneous agents.
- It implements a "Value of Movement" behavioral loop to encourage periodic rendezvous for data exchange, enhancing global situational awareness in sparse networks.

---

[REDSearcher: A Scalable and Cost-Efficient Framework for Long-Horizon Search Agents](http://arxiv.org/abs/2602.14234)

- REDSearcher: introduces an integrated framework for training tool-augmented deep-search agents by jointly optimizing task synthesis, mid-training, and post-training to enable scalable long-horizon behavior.
- The architecture employs dual-constrained task synthesis based on graph treewidth and evidence dispersion to generate reasoning challenges that mandate iterative planning and cross-document synthesis.
- The system features a two-stage mid-training regimen and a local simulated environment for rapid reinforcement learning iteration, utilizing classification-, graph-, generation-, editor-, solver-, verifier- and judge-LLMs.

---

[Text Before Vision: Staged Knowledge Injection Matters for Agentic RLVR in Ultra-High-Resolution Remote Sensing Understanding](http://arxiv.org/abs/2602.14225)

- Agentic RLVR (Agentic Reinforcement Learning with Verifiable Rewards): introduces a staged knowledge injection recipe for ultra-high-resolution remote sensing, utilizing a base MLLM, an SFT module, and an agentic RLVR framework equipped with zoom-in tools.
- The architecture incorporates an automated Earth-science text QA pipeline that includes task categorization-, reasoning generation-, and quality assessment-LLMs, validated by a LightRAG-based knowledge graph.
- This staged approach uses domain-specific text-only data to instill reasoning structures before fine-tuning on hard image-text examples to stabilize subsequent tool-based reinforcement learning.

---

[The Interspeech 2026 Audio Reasoning Challenge: Evaluating Reasoning Process Quality for Audio Reasoning Models and Agents](http://arxiv.org/abs/2602.14224)

- Audio Reasoning Challenge: introduces MMAR-Rubrics, an instance-level evaluation protocol for assessing the factuality and logic of Chain-of-Thought (CoT) reasoning in Large Audio Language Models (LALMs) and agentic systems.
- The framework evaluates two distinct architectures: end-to-end Large Audio Reasoning Models (LARMs) and modular agents that utilize iterative tool orchestration, structured memory, and self-verification loops.
- Results demonstrate that agent-based systems currently lead in reasoning transparency and accuracy, while single models are advancing through reinforcement learning and sophisticated data synthesis pipelines.

---

[SkillJect: Automating Stealthy Skill-Based Prompt Injection for Coding Agents with Trace-Driven Closed-Loop Refinement](http://arxiv.org/abs/2602.14211)

- 
SKILLJECT: introduces an automated framework for stealthy prompt injection in coding agent skills, utilizing a closed-loop system with an Attack Agent, a Code Agent, and an Evaluate Agent.

- 
The system decouples attacks into lightweight inducement prompts within documentation and operational payloads hidden in auxiliary scripts to bypass LLM safety filters.

- 
It employs trace-driven iterative refinement to optimize injection efficacy and stealth against diverse LLM-based coding tools and realistic software engineering tasks.


---

[GeoEyes: On-Demand Visual Focusing for Evidence-Grounded Understanding of Ultra-High-Resolution Remote Sensing Imagery](http://arxiv.org/abs/2602.14201)

- GeoEyes: introduces a staged training framework for ultra-high-resolution remote sensing, with UHR-CoZ (interleaved image-text SFT dataset), AdaZoom-GRPO (agentic reinforcement learning method), Adaptive Efficiency Reward (task-adaptive tool usage balancer), Chain-of-Focus Reward (coarse-to-fine zooming enforcer), Process Verification Reward (LLM-based logical rigor judge), GLM-4.5V (agent-orchestrated data generation model), and Zoom-in Tool (visual evidence acquisition mechanism), and includes agent-orchestrated data generation- and process verification-LLMs.
- The framework addresses tool usage homogenization by training MLLMs to learn task-adaptive zoom-in policies that selectively activate based on task heterogeneity and evidence density.
- It achieves 54.23% accuracy on the XLRS-Bench by coupling process-supervised initialization with evidence-driven reinforcement learning.

---

[When Benchmarks Lie: Evaluating Malicious Prompt Classifiers Under True Distribution Shift](http://arxiv.org/abs/2602.14161)

- LODO (Leave-One-Dataset-Out): introduces a rigorous evaluation protocol for malicious prompt classifiers by holding out entire datasets during training to measure true out-of-distribution generalization, utilizing raw activations and SAE features from Llama-3.1-8B-Instruct.
- The framework reveals that standard evaluation practices inflate performance metrics by 8.4 percentage points because classifiers exploit dataset-specific shortcuts rather than learning generalizable attack patterns.
- It demonstrates that activation-based probes significantly outperform production guardrails and LLM-as-judge approaches, especially on indirect and agentic tool injection attacks where existing systems fail.

---

[Process-Supervised Multi-Agent Reinforcement Learning for Reliable Clinical Reasoning](http://arxiv.org/abs/2602.14160)

- Agent-as-Tool: introduces a hierarchical multi-agent reinforcement learning framework for gene-disease validity curation, utilizing a supervisor agent to orchestrate specialized sub-agents for biochemical function, protein interaction, gene expression, functional alteration, model systems, and rescue experiments.
- The system employs a hybrid reward mechanism combining outcome-based correctness with process-level supervision to ensure reasoning aligns with clinical standard operating procedures.
- Training is conducted via Group Relative Policy Optimisation to improve both classification accuracy and the alignment of reasoning processes with clinical standards.

---

[A Multi-Agent Framework for Medical AI: Leveraging Fine-Tuned GPT, LLaMA, and DeepSeek R1 for Evidence-Based and Bias-Aware Clinical Query Processing](http://arxiv.org/abs/2602.14158)

- Multi-Agent Framework for Medical AI: introduces a modular clinical QA pipeline, with reasoning-, retrieval-, and refinement-agents.
- The system integrates uncertainty quantification via Monte Carlo dropout and perplexity scoring alongside multi-layered bias detection to ensure medical neutrality.
- The framework utilizes a context manager for efficient GPU/CPU memory transitions and achieves 87% accuracy on medical query benchmarks.

---

[ForesightSafety Bench: A Frontier Risk Evaluation and Governance Framework towards Safe AI](http://arxiv.org/abs/2602.14135)

- ForesightSafety Bench (AI Safety Evaluation Framework): introduces a comprehensive, hierarchical AI safety evaluation framework spanning fundamental, extended, and industrial safety tiers to detect frontier risks across 94 dimensions.
- The framework utilizes an LLM-as-a-Judge paradigm and adversarial jailbreak methods to evaluate the defensive resilience and behavioral alignment of 22 state-of-the-art LLMs.
- Empirical results reveal widespread vulnerabilities in agentic autonomy, scientific safety, and social interaction, highlighting a "performance-safety trade-off" where advanced reasoning can erode established safety boundaries.

---

[Toward Autonomous O-RAN: A Multi-Scale Agentic AI Framework for Real-Time Network Control and Management](http://arxiv.org/abs/2602.14117)

- Multi-Scale Agentic AI Framework: introduces a hierarchical agentic architecture for O-RAN that coordinates intelligence across Non-RT, Near-RT, and Real-Time control loops, with LLM rApp, SLM xApps, WPFM dApps, MCP Agent, Model Catalog, Data Lake, MLOps Pipeline, Simulation/Digital Twin, Knowledge Base, Analytics Module, and standardized interfaces.
- The system includes strategic planning-, tactical optimization-, and operational inference-agents to translate high-level operator intent into real-time radio control actions.
- It utilizes an internal Model Context Protocol and standardized interfaces to enable automated model lifecycle management, safety-constrained autonomy, and cross-layer telemetry feedback.

---

[NEST: Nascent Encoded Steganographic Thoughts](http://arxiv.org/abs/2602.14095)

- NEST (Nascent Encoded Steganographic Thoughts): introduces a systematic evaluation framework for measuring steganographic reasoning in frontier models, which includes Agent (LLM) (generates steganographic responses), Steganographic Chain of Thought (encoded reasoning within text), Monitor (LLM) (detects hidden content presence), Encoding Scorer (Algorithmic) (verifies extraction match ratio), Hidden Task Scorer (Correctness) (evaluates covert answer accuracy), and Translator (LLM) (decodes hidden steganographic messages), where the study quantifies the risk of models concealing reasoning within innocuous text.
- The framework assesses model capabilities across four dimensions including hidden task accuracy, encoding quality, monitor evasion, and translation fidelity using tasks like acrostic counting and math puzzles.
- Experimental results reveal that while current frontier models struggle with complex steganographic reasoning, models like Claude Opus 4.5 demonstrate nascent capabilities in simplified counting tasks.

---

[GUI-GENESIS: Automated Synthesis of Efficient Environments with Verifiable Rewards for GUI Agent Post-Training](http://arxiv.org/abs/2602.14093)

- GUI-GENESIS (Automated Synthesis of Efficient Environments with Verifiable Rewards for GUI Agent Post-Training): introduces a framework to automatically synthesize GUI training environments with verifiable rewards, with Trace Collection, Hierarchical Synthesis, Code-Native Reward Injection, and Automated Self-Verification components.
- The system utilizes multimodal code models to reverse-engineer user interaction traces into standalone web applications, replacing noisy visual reward proxies with deterministic code-native assertions; it includes planning- and coding-LLMs.
- It achieves a 10x reduction in environment latency and reduces training costs by over $28,000 per epoch while enabling zero-shot sim-to-real transfer for GUI agents.

---

[TabTracer: Monte Carlo Tree Search for Complex Table Reasoning with Large Language Models](http://arxiv.org/abs/2602.14089)

- TabTracer: introduces an agentic framework that coordinates multi-step tool calls over intermediate table states using Monte Carlo Tree Search (MCTS), including planning- and reflection-agents.
- The system employs a layered architecture comprising a reasoning layer for search control, an execution layer for atomic dataframe operations, and a storage layer for versioned state tracking.
- It utilizes execution-feedback MCTS with UCB1 selection and budget-aware pruning to enable reliable backtracking while significantly reducing token consumption.

---

[PLAN-MCTS: Plan Exploration for Action Exploitation in Web Navigation](http://arxiv.org/abs/2602.14083)

- PLAN-MCTS (Plan Exploration for Action Exploitation): introduces a web navigation framework that reformulates search within a semantic plan space, featuring planning-, grounding-, evaluation-, and refinement-agents.
- The architecture decouples strategic planning from execution grounding by constructing a Dense Plan Tree and maintaining an Abstracted Semantic History to filter low-level execution noise.
- It incorporates a Dual-Gating Reward to strictly validate physical executability and strategic alignment, alongside a Reflector agent for on-policy repair of failed subplans.

---

[Truthful Reporting of Competence with Minimal Verification](http://arxiv.org/abs/2602.14076)

- MCV (Monotone-Cutoff Verification): introduces a mechanism design framework for eliciting truthful competence reports while minimizing verification overhead and grading bias, utilizing Principal, Agents, Verification Selection Function, Grading Function, Audit Probability, Penalty Mechanism, Proper Scoring Rules, Type Distribution, Cutoff Parameter, Expected Grade, and Report.
- The framework establishes an efficiency boundary for deterministic verification and provides extensions for noisy environments using polynomial verification and proper scoring rules.
- Empirical evaluations on SAT and credit score distributions demonstrate that the proposed mechanisms achieve near-optimal tradeoffs between audit frequency and reporting accuracy.

---

[Decentralized Federated Learning With Energy Harvesting Devices](http://arxiv.org/abs/2602.14051)

- EH-enabled DFL (Energy Harvesting-enabled Decentralized Federated Learning): introduces a sustainable decentralized training framework for energy-constrained edge devices, with edge devices, energy harvesting units, rechargeable batteries, D2D communication links, a multi-agent MDP framework, and a decentralized policy iteration algorithm.
- The system models joint device scheduling and power control as a multi-agent Markov decision process to manage stochastic energy supplies and wireless channels.
- A localized policy iteration algorithm enables devices to optimize transmission strategies using only two-hop neighbor information, reducing computational complexity and communication overhead.

---

[UniST-Pred: A Robust Unified Framework for Spatio-Temporal Traffic Forecasting in Transportation Networks Under Disruptions](http://arxiv.org/abs/2602.14049)

- UniST-Pred: introduces a modular traffic forecasting framework that decouples temporal dependency learning from spatial representation extraction, with Temporal Modeling, Spatial Modeling, and Adaptive Fusion.
- The architecture employs a feature-time mixing temporal block and a task-adaptive graph transformer spatial block to maintain performance stability under structural network uncertainties.
- A squeeze-and-excitation residual block adaptively reweights temporal and spatial features, providing interpretable representations and maintaining predictive accuracy during infrastructure disruptions.

---

[Choosing How to Remember: Adaptive Memory Structures for LLM Agents](http://arxiv.org/abs/2602.14038)

- 
FLUXMEM: introduces a unified framework for LLM agents that enables adaptive memory organization by dynamically selecting between linear, graph, and hierarchical structures based on interaction-level features.

- 
The architecture employs a three-layer memory hierarchy comprising short-term interaction, mid-term episodic, and long-term semantic layers to manage information across different temporal and semantic scales.

- 
A Beta Mixture Model-based probabilistic gate replaces fixed similarity thresholds to provide distribution-aware memory fusion, improving robustness in long-horizon conversational scenarios.


---

[FloCA: Towards Faithful and Logically Consistent Flowchart Reasoning](http://arxiv.org/abs/2602.14035)

- FloCA (Flowchart-oriented Conversational Agent): introduces a zero-shot autonomous agent that delegates flowchart reasoning to an external tool to ensure faithful and logically consistent node transitions.
- The framework includes intent analysis-, semantic matching- and response generation-roles performed by an instruction-following LLM to interact with users. 
- The system utilizes a faithful flowchart reasoning tool to execute topology-constrained transitions and an FQARetriever to address domain-specific knowledge gaps during multi-turn dialogues.

---

[BRAIN: Bayesian Reasoning via Active Inference for Agentic and Embodied Intelligence in Mobile Networks](http://arxiv.org/abs/2602.14033)

- BRAIN (Bayesian Reasoning via Active Inference): introduces an embodied AI agent for 6G networks that minimizes variational free energy to unify perception and action, utilizing a Generative Model, Bayesian Perception, an Expected Free Energy Controller, a Preference Prior, and an O-RAN xApp.
- The architecture employs a deep generative model to decompose decision-making into extrinsic goal alignment and epistemic uncertainty reduction, enabling robust adaptation without retraining.
- Experimental results on a GPU-accelerated testbed show the framework outperforms deep reinforcement learning baselines in handling catastrophic forgetting and providing interpretable belief diagnostics.

---

[S2SSERVICEBENCH: A Multimodal Benchmark for Last-Mile S2S Climate Services](http://arxiv.org/abs/2602.14017)

- S2SSERVICEBENCH (Subseasonal-to-seasonal Service Benchmark): introduces a multimodal benchmark for evaluating last-mile climate services, with S2S Prediction Provider, S2S Service Provider, S2S Service Agent, End User, Domain Layer, Product Layer, Service Level Layer, and Task Format Layer.
- The framework evaluates MLLMs and agents across ten service products and three progressive service levels ranging from signal comprehension to strategic decision analysis.
- It utilizes structured output formats like Short-slot Structured Completion and Structured Report Generation to assess model reliability under uncertainty and operational constraints.

---

[Prompt-Driven Low-Altitude Edge Intelligence: Modular Agents and Generative Reasoning](http://arxiv.org/abs/2602.14003)

- P2AECF (Prompt-to-Agent Edge Cognition Framework): introduces an edge intelligence architecture that transforms high-level semantic prompts into executable reasoning workflows using model-agnostic task graphs and modular agents.
- The framework utilizes a diffusion-based planner to adaptively construct optimal execution paths by incorporating real-time environmental context and historical performance data.
- This approach decouples cognitive tasks from fixed model bindings, enabling scalable and resilient low-altitude aerial collaborations across heterogeneous edge nodes.

---

[It Takes Two to Tango: A Holistic Simulator for Joint Order Scheduling and Multi-Agent Path Finding in Robotic Warehouses](http://arxiv.org/abs/2602.13999)

- WareRover: introduces a holistic simulation platform for Robotic Mobile Fulfillment Systems (RMFS) that tightly couples high-level order scheduling with low-level multi-agent pathfinding through a unified, closed-loop optimization interface.
- The system integrates a warehouse environment builder for high-fidelity modeling, a task modular for dynamic order streams, and a physical motion executor to enforce realistic kinematic constraints and continuous collision checking.
- The platform features a failure simulation and recovery module to evaluate algorithmic robustness against stochastic AGV breakdowns and maintenance-induced traffic congestion in dynamic warehouse environments.

---

#### 14th February 2026

[RoboSolver: A Multi-Agent Large Language Model Framework for Solving Robotic Arm Problems](http://arxiv.org/abs/2602.14438)

- RoboSolver: introduces a multi-agent framework built on LLMs and VLMs specifically tailored to robotics, which includes supervisor-, researcher-, retriever-, extractor-, planner-, robosolver-, and inspector-agents.
- The architecture employs a ReAct-based iterative loop to decompose high-level user instructions into manageable sub-tasks, utilizing a suite of 16 specialized analytical and numerical tools for kinematics and motion analysis.
- The framework achieves up to 0.97 accuracy in solving complex robotic arm problems by combining linguistic reasoning with precise computational execution across textual and visual inputs.

---

[‘I Spend All My Energy Preparing’: Balancing AI Automation and Agency for Self-Regulated Learning in SmartFlash](http://arxiv.org/abs/2602.14431)

- SmartFlash: introduces an AI-powered flashcard prototype designed to balance automation with learner agency, featuring hypothesis-generation and content-generation LLMs.
- The system reduces extraneous cognitive load by automating material preparation while preserving germane load through user-directed review and verifiable AI reasoning.
- Research findings emphasize that learners require editable AI outputs to maintain cognitive ownership and value collaborative AI partnerships over fully autonomous agents.

---

#### 12th February 2026


[Agentic AI for Cybersecurity: A Meta-Cognitive Architecture for Governable Autonomy](http://arxiv.org/abs/2602.11897)

- Agentic Cybersecurity Orchestration Framework: introduces a distributed cognitive system for cyber defense, with detection-, hypothesis-, context-, explainability-, governance-, and meta-cognitive judgement-agents.
- The framework employs generative AI as a coordination substrate to facilitate semantic negotiation and alignment between autonomous agents and human experts.
- A central meta-cognitive judgement function regulates system autonomy by assessing uncertainty, explanation adequacy, and policy compliance to determine action readiness.

---


[Embodied AI Agents for Team Collaboration in Co-located Blue-Collar Work (CAI-BLUE)](http://arxiv.org/abs/2602.12136)

- CAI-BLUE (Collaborative AI for Blue-collar Work): introduces a framework for designing physically present AI agents to support human-human and human-AI collaboration in industrial and maintenance settings, with all Embodied AI Agent, Co-located Workers, Shared Physical Environment, Sensors, Troubleshooting Model, Transcription System, and Conversational Interface components, where embodiment serves as a socio-material strategy for shared situational awareness.
- The approach utilizes physically instantiated interfaces to support spatial and motor skills while facilitating mutual learning between experienced and novice workers in co-located environments.
- The research outlines critical design questions regarding worker agency, surveillance risks, and the social acceptability of different AI embodiments in hierarchical blue-collar workplaces.


---

[MalTool: Malicious Tool Attacks on LLM Agents](http://arxiv.org/abs/2602.12194)

- MalTool: introduces a coding-LLM-based framework for synthesizing malicious tools, with a system prompt, a coding LLM, an automated verifier, an accepted tool pool, and a feedback loop, where the system automatically generates diverse and functionally correct malicious tool implementations.
- The framework utilizes an iterative generation-and-verification loop to ensure that synthesized tools are both functionally correct and structurally diverse.
- It enables the creation of standalone malicious tools and Trojan variants that embed malicious behaviors into benign real-world tool implementations.


---


[Defining causal mechanism in dual process theory and two types of feedback control](http://arxiv.org/abs/2602.11478)

- Dual-Laws Model: introduces a hierarchical inter-level causation framework that unifies dual-process theory with agency and consciousness by defining two independent feedback control processes across the supervenience level and neural states.
- The framework distinguishes Type 1 processes as fast, automatic inter-level feedback control of neural states and Type 2 processes as discrete, slow modifications of symbolic equations at the supervenience level.
- It characterizes embodied beliefs as equations within a supervenient network, where linguistic beliefs enable conscious reflection and cognitive decoupling in humans.

---


[PhyNiKCE: A Neurosymbolic Agentic Framework for Autonomous Computational Fluid Dynamics](http://arxiv.org/abs/2602.11666)

- PhyNiKCE (Physical and Numerical Knowledgeable Context Engineering): introduces a neurosymbolic agentic framework for autonomous Computational Fluid Dynamics (CFD) that decouples neural planning from symbolic validation to enforce physical constraints.
- The architecture utilizes a Deterministic RAG Engine with five specialized retrieval strategies to query a structured Symbolic Knowledge Base, ensuring numerical stability and physical consistency in simulation setups.
- By treating configuration as a Constraint Satisfaction Problem (CSP), the framework significantly improves accuracy on complex engineering tasks while reducing LLM token consumption and autonomous self-correction loops.

---


[UniT: Unified Multimodal Chain-of-Thought Test-time Scaling](http://arxiv.org/abs/2602.12279)

- UniT (Unified Multimodal Chain-of-Thought Test-time Scaling): introduces a framework for multimodal chain-of-thought test-time scaling, with Image Gen Model, Vision-Language Model, Image Editing Model, Unified Multimodal Model, Multimodal Chain-of-Thought, Agentic Data Synthesis, and Budget Forcing, where a single unified model iteratively reasons, verifies, and refines multimodal outputs across multiple rounds.
- The framework utilizes an agentic data synthesis pipeline to generate multi-round reasoning trajectories that teach the model verification, subgoal decomposition, and content memory.
- Sequential chain-of-thought scaling demonstrates superior compute efficiency and performance over parallel sampling across image generation, editing, and visual reasoning tasks.

---

[Think like a Scientist: Physics-guided LLM Agent for Equation Discovery](http://arxiv.org/abs/2602.12259)

- KeplerAgent: introduces an agentic framework that orchestrates physics-based tools and symbolic regression engines to discover mathematical equations from data by emulating the multi-step reasoning workflow of human scientists.
- The architecture includes a central LLM agent, a visual subagent for analyzing data plots, a Python code interpreter for exploratory analysis, and a symmetry discovery tool to identify physical constraints.
- The framework utilizes a shared workspace and an experience log to iteratively refine hypothesis spaces, demonstrating superior symbolic recovery and noise robustness on complex dynamical systems including ODEs and PDEs.

---

[ADJUSTED WINNER: FROM SPLITTING TO SELLING](http://arxiv.org/abs/2602.12231)

- DSIRS (Dispute Settlement with Indivisible Resources and Sale): introduces a fair division framework for agents to allocate indivisible resources using utility, revenue, and selling cost functions under a global budget to produce an AW-derived plan.
- The framework formalizes the AWNS (Adjusted Winner without Splitting) optimization problem to minimize welfare disparity by redistributing sale proceeds while accounting for resource-specific selling costs.
- The research establishes the computational complexity of various fairness objectives, proves NP-hardness for most variants, and provides a fully polynomial-time approximation scheme (FPTAS) for optimizing the welfare ratio.

---

[VIRENA: Virtual Arena for Research, Education, and Democratic Innovation](http://arxiv.org/abs/2602.12207)

- VIRENA (Virtual Arena): introduces a platform for controlled social media experimentation featuring feed-based and chat-based interfaces, LLM-powered AI agents, LLM-powered moderation, a scripting system, an experimental control hierarchy, a PocketBase backend, and a SQLite database.
- The system enables multi-user interaction alongside AI agents with configurable personas and response patterns to study group dynamics, deliberation, and social influence.
- It provides a no-code visual interface for researchers to manipulate content, timing, and moderation rules while ensuring data remains under institutional control.

---

[Convex Markov Games and Beyond: New Proof of Existence, Characterization and Learning Algorithms for Nash Equilibria](http://arxiv.org/abs/2602.12181)

- GUMG (General Utility Markov Games): introduces a broad class of multi-agent learning problems generalizing convex Markov games by allowing coupling between agents' occupancy measures, where Nash equilibria are characterized as fixed points of projected pseudo-gradient dynamics.
- The framework utilizes a model-free policy gradient algorithm that leverages a novel agent-wise gradient domination property to ensure convergence to approximate Nash equilibria in potential games.
- It establishes the first efficient learning guarantees for common-interest convex Markov games, providing sample complexity bounds for both generative model and on-policy settings.

---

[3DGSNav: Enhancing Vision-Language Model Reasoning for Object Navigation via Active 3D Gaussian Splatting](http://arxiv.org/abs/2602.12159)

- 3DGSNav: introduces a zero-shot object navigation framework that embeds 3D Gaussian Splatting as persistent memory to enhance the spatial reasoning capabilities of VLMs.
- The framework utilizes active perception to construct environment representations and trajectory-guided free-viewpoint optimization to provide informative visual observations for decision-making.
- It incorporates structured visual prompts with Chain-of-Thought reasoning and a VLM-driven re-verification module to ensure robust target localization and recognition.

---

[On the Adoption of AI Coding Agents in Open-source Android and iOS Development](http://arxiv.org/abs/2602.12144)

- Empirical Study Methodology: introduces an analytical pipeline to evaluate AI coding agents in mobile development, with Project Identification, Project Filtration, PR Mining, PR Categorization, and Statistical Analysis components, where it assesses contributions from Codex-, Devin-, Copilot-, Cursor-, and Claude-agents.
- The methodology utilizes GPT-5 for open-card and closed-card sorting to classify 2,901 pull requests into 13 task categories across Android and iOS platforms.
- Analysis reveals that Android projects exhibit higher pull request acceptance rates and greater sensitivity to agent choice compared to the more uniform and cautious review process on iOS.

---

[WavBench: Benchmarking Reasoning, Colloquialism, and Paralinguistics for End-to-End Spoken Dialogue Models](http://arxiv.org/abs/2602.12135)

- WavBench: introduces a multi-faceted benchmark for end-to-end spoken dialogue models, featuring a tripartite framework consisting of Pro, Basic, and Acoustic subsets to evaluate reasoning, colloquialism, and paralinguistics.
- The benchmark assesses models across seven cognitive domains and ten paralinguistic dimensions, utilizing a hierarchical scoring mechanism and LLM-based evaluation to measure conversational naturalness and acoustic fidelity.
- It identifies a significant "Cognitive-Acoustic Alignment" gap in current models, where reasoning-enhanced agents often struggle to maintain natural colloquial delivery and paralinguistic consistency.

---

[Optimizing Distances for Multi-Broadcast in Temporal Graphs](http://arxiv.org/abs/2602.12126)

- D-TMB (D-Temporal Multi-Broadcast): introduces a scheduling framework for temporal graphs that optimizes worst-case temporal distances from sources to vertices, utilizing a static graph, a traversal function, a multiplicity function, an offline planner, and temporal distance measures.
- The system assigns time labels to edges to satisfy multiplicity constraints while minimizing or maximizing specific temporal metrics like Earliest-Arrival or Latest-Departure.
- The research establishes computational complexity results and approximation algorithms for single and multiple source scenarios across diverse distance definitions.

---

[Anonymous Contracts](http://arxiv.org/abs/2602.12118)

- Anonymous Contracts: introduces a multi-agent contracting framework where a principal offers payments based solely on the total number of successes to ensure fairness and identical treatment among agents.
- The framework establishes that every anonymous contract admits a pure Nash equilibrium and identifies uniform anonymous contracts as a subclass guaranteeing unique equilibria and robust performance.
- Analysis reveals that removing limited liability allows anonymous contracts to achieve an O(log n) approximation of social welfare or even full welfare extraction when agent success probabilities are distinct.

---

[The Pensieve Paradigm: Stateful Language Models Mastering Their Own Context](http://arxiv.org/abs/2602.12108)

- StateLM (Stateful Language Model): introduces a foundation model architecture that actively manages its own context through an internal reasoning loop, with all self-engineered context, internal reasoning loop, memory toolkit, external memory notebook, environment, teacher model, and reward module components, where the model includes a teacher model and a state-aware agent to dynamically engineer its own context.
- The framework implements the "Pensieve Paradigm" to transform the interaction state from a passive log into a mutable object via explicit context-pruning and note-taking operations.
- It achieves measurable performance improvements in long-context tasks by maintaining a reduced context state through a learned "search-read-note-delete" cycle.

---

[DEpiABS: Differentiable Epidemic Agent-Based Simulator](http://arxiv.org/abs/2602.12102)

- DEpiABS: introduces a structure-centric differentiable agent-based model for epidemic simulation, with Society, Epidemic, Population, Differentiable Approximation, Tensorisation, and Output Scaling components, where it reconciles simulation fidelity, computational efficiency, and mechanistic interpretability without using neural surrogates.
- The framework features a novel z-score-based scaling method that decouples simulation costs from population size by mapping fixed-size population outputs to any real-world scale.
- By reformulating agent-based dynamics as a differentiable computational graph, the model supports fast GPU acceleration and direct parameter calibration from empirical COVID-19 and influenza data.

---

[Choose Your Agent: Tradeoffs in Adopting AI Advisors, Coaches, and Delegates in Multi-Party Negotiation](http://arxiv.org/abs/2602.12089)

- AI Assistance Modalities: introduces three distinct interaction structures—Advisor, Coach, and Delegate—to study the impact of AI agency on human behavior and welfare in multi-party negotiation environments; the framework includes Advisor-, Coach-, and Delegate-agents.
- The system employs Gemini-2.5-Flash as a core engine, wrapped in lightweight prompt scaffolding to deliver proactive guidance, reactive critiques, or autonomous trade execution.
- Experimental results demonstrate that while users prefer the high-control Advisor, the Delegate modality achieves superior economic outcomes by acting as a market maker through rational, high-value proposals.

---

[Differentiable Modal Logic for Multi-Agent Diagnosis, Orchestration and Communication](http://arxiv.org/abs/2602.12083)

- MLNN (Modal Logical Neural Networks): introduces a unified neurosymbolic framework that operationalizes Kripke semantics as differentiable neural network layers to discover hidden semantic structures like trust networks and causal chains from multi-agent behavioral data.
- The framework utilizes continuous approximations of modal operators and the Łukasiewicz t-norm to transform logical contradictions into learnable optimization objectives.
- Applications include detecting LLM hallucinations through doxastic calibration and orchestrating autonomous swarms by jointly optimizing epistemic, temporal, and deontic constraints.

---

[MULTI UAVS PREFLIGHT PLANNING IN A SHARED AND DYNAMIC AIRSPACE](http://arxiv.org/abs/2602.12055)

- DTAPP-IICR: introduces a delivery-time aware prioritized planning method for large-scale UAV fleets in dynamic airspace, utilizing SFIPP-ST for 4D trajectory generation and iterative Large Neighborhood Search for conflict resolution.
- The framework incorporates a novel single-agent planner that handles heterogeneous vehicle profiles and temporal No-Fly Zones while modeling inter-agent conflicts as soft constraints.
- A completeness-preserving directional pruning strategy significantly reduces the search branching factor in 3D grids, enabling scalable preflight planning for up to 1,000 UAVs.

---

[Adaptive-Horizon Conflict-Based Search for Closed-Loop Multi-Agent Path Finding](http://arxiv.org/abs/2602.12024)

- ACCBS (Anytime Closed-Loop Conflict-Based Search): introduces a closed-loop Multi-Agent Path Finding algorithm that dynamically adjusts its planning horizon through an iterative deepening mechanism while reusing a single constraint tree.
- The framework utilizes an active prefix to enforce conflict-freedom over a growing lookahead, enabling anytime behavior and transitions between different horizon lengths.
- By combining receding-horizon approximation with constraint-tree reuse, the system maintains theoretical optimality guarantees while improving scalability and robustness to disturbances in dynamic environments.

---

[Evaluating AGENTS.md: Are Repository-Level Context Files Helpful for Coding Agents?](http://arxiv.org/abs/2602.11988)

- AGENTBENCH: introduces a curated benchmark and evaluation pipeline to investigate the impact of repository-level context files on autonomous coding agents, with Repository, Task, Context Files, Coding Agent, Harness, Autonomous Task Solving, Trace Analysis, and Test Evaluation components.
- The pipeline evaluates CLAUDE CODE, CODEX, and QWEN CODE agents across three settings—no context, LLM-generated context, and developer-provided context—using a harness that enables interaction with the repository environment through specialized tools.
- Experimental results indicate that context files often lead to increased exploration and higher inference costs while potentially decreasing task success rates due to redundant or restrictive instructions.

---

[Multi-Defender Single-Attacker Perimeter Defense Game on a Cylinder: Special Case in which the Attacker Starts at the Boundary](http://arxiv.org/abs/2602.11977)

- Multi-Defender Single-Attacker Perimeter Defense Game: introduces a multi-agent game on a topological cylinder where a team of slow defenders must prevent a faster attacker from crossing a boundary, with all components including Attacker, Defenders, Perimeter Boundary, Defensive Regions, and Gaps.
- The research derives closed-form expressions for the maximum perimeter circumference that can be successfully defended based on agent velocities, defender count, and defensive region sizes.
- It specifically analyzes the non-trivial case where the attacker starts at a guarded boundary position and moves faster than the homogeneous defenders.

---

[GAIA2: BENCHMARKING LLM AGENTS ON DYNAMIC AND ASYNCHRONOUS ENVIRONMENTS](http://arxiv.org/abs/2602.11964)

- Gaia2 (built on ARE: Agents Research Environments): introduces a benchmark for evaluating LLM agents in realistic, asynchronous environments, with an event-driven simulation platform, a Mobile environment with stateful apps, an action-level verifier, an asynchronous event loop, a configurable notification system, and a hierarchical multi-agent collaboration protocol involving main- and app-agents.
- The framework evaluates agent capabilities like temporal reasoning and noise robustness by simulating dynamic environment events that occur independently of agent actions.
- It provides a scalable infrastructure for reinforcement learning from verifiable rewards by checking every state-changing action against oracle annotations using the ARE Verifier.

---

[Do Large Language Models Adapt to Language Variation across Socioeconomic Status?](http://arxiv.org/abs/2602.11939)

- LLM-SESA (Large Language Model Socioeconomic Style Adaptation): introduces an investigation into linguistic adaptation across socioeconomic status (SES), with Data Sourcing (collecting SES-stratified social media text), Prompt Completion (generating text using four LLMs), Feature Analysis (calculating 94 linguistic metrics), and Evaluation (comparing stylistic alignment between models and humans), where the study assesses how effectively Gemma-, Mistral-, Qwen-, and GPT-5-based LLMs emulate diverse social communication styles.
- The research utilizes a novel dataset from Reddit and YouTube stratified by SES to test LLMs under varying levels of prompting explicitness ranging from implicit cues to direct status instructions.
- Findings reveal that LLMs often fail to capture nuanced SES markers, resulting in stylistic caricatures and a latent tendency to better integrate with upper SES language patterns.

---

[AdaptEvolve: Improving Efficiency of Evolutionary AI Agents through Adaptive Model Selection](http://arxiv.org/abs/2602.11931)

- AdaptEvolve (Adaptive Large Language Model Selection for Multi-Large Language Model Evolutionary Refinement): introduces a confidence-driven routing framework that optimizes the cost-capability trade-off in evolutionary agentic coding systems with population, confidence extraction, decision tree router, and warm-up phase components, and includes small- and large-LLM components.
- The architecture utilizes a lightweight decision tree router to dynamically escalate complex reasoning hurdles from a small model to a larger model based on intrinsic uncertainty signals like token entropy.
- By leveraging entropy-based metrics such as Lowest Group Confidence and Tail Confidence, the framework reduces total inference cost by an average of 37.9% while retaining 97.5% of upper-bound accuracy.

---

[MEME: Modeling the Evolutionary Modes of Financial Markets](http://arxiv.org/abs/2602.11918)

- MEME (Modeling the Evolutionary Modes of Financial Markets): introduces a logic-oriented framework that models financial markets as a dynamic ecosystem of competing investment narratives, utilizing multi-agent extraction, Gaussian Mixture Modeling, and temporal alignment.
- The system includes LLM-based filter- and generator-agents to distill raw multimodal data into structured investment arguments containing polarity, rationale, and evidence.
- By tracking the lifecycle and historical profitability of latent "Modes of Thought" through temporal alignment, the framework prioritizes enduring market wisdom over transient anomalies for portfolio construction.

---

[Towards Fair and Comprehensive Evaluation of Routers in Collaborative LLM Systems](http://arxiv.org/abs/2602.11877)

- ProbeDirichlet: introduces a lightweight routing mechanism that aggregates cross-layer internal hidden states via learnable Dirichlet distributions to dynamically direct queries between small local and large cloud LLMs.
- The system utilizes a triple-perspective evaluation framework called RouterXBench to assess router ability, scenario alignment, and cross-domain robustness across diverse deployment regimes.
- Experimental results demonstrate that internal representations capture model uncertainty more reliably than output probabilities, leading to significant improvements in routing accuracy and generalization across diverse tasks.

---

[Zooming without Zooming: Region-to-Image Distillation for Fine-Grained Multimodal Perception](http://arxiv.org/abs/2602.11858)

- ZwZ (Zooming without Zooming): introduces Region-to-Image Distillation, with Teacher MLLMs, Student MLLM, Object Recognition & Segmentation, Question Generator, and Answer Generators, where the framework internalizes the benefits of agentic zooming into a single forward pass of a multimodal model.
- The method decouples zooming from inference-time tool use by repurposing it as a training-time primitive that utilizes micro-cropped regions to generate high-quality, region-grounded supervision for smaller student models.
- The resulting models achieve leading performance on fine-grained perception benchmarks like ZoomBench while significantly reducing latency compared to iterative "Thinking-with-Images" approaches.

---

[TOWARDS SUSTAINABLE INVESTMENT POLICIES INFORMED BY OPPONENT SHAPING](http://arxiv.org/abs/2602.11829)

- Advantage Alignment (AdAlign): introduces a scalable opponent shaping approach applied to the InvestESG multi-agent simulation to align individual economic incentives with long-term sustainable investment goals.
- The framework modifies policy gradients directly by incorporating the advantages of other agents, effectively biasing learning dynamics toward cooperative outcomes in intertemporal social dilemmas.
- Empirical results demonstrate that AdAlign achieves higher market wealth and more equitable capital distribution compared to standard MARL baselines like IPPO and MAPPO.

---

[Non-Trivial Consensus on Directed Matrix-Weighted Networks with Cooperative and Antagonistic Interactions](http://arxiv.org/abs/2602.11822)

- NTC (Non-Trivial Consensus): introduces a control framework for achieving consensus in both magnitude and sign across directed signed matrix-weighted networks featuring cooperative and antagonistic interactions.
- The approach employs grounded Laplacians and an expanded system model to establish spectral conditions and lower bounds for coupling coefficients that guarantee global asymptotic convergence.
- The protocol accommodates fixed and switching topologies under relaxed connectivity conditions, allowing the consensus state to be arbitrarily preset regardless of structural balance.

---

[Deep Kernel Fusion for Transformers](http://arxiv.org/abs/2602.11808)

- DeepFusionKernel: introduces an aggressively fused CUDA operator for SwiGLU MLP blocks, with DeepFusionKernel, Kernel Scheduler, and SGLang integration, where it eliminates intermediate memory reads/writes to accelerate bandwidth-bound LLM inference.
- The system employs a lightweight profiler to select optimal tiling strategies and loop orderings based on specific hardware configurations and workload shapes.
- Experimental results demonstrate throughput gains of up to 13.2% on H100 GPUs by rebalancing the trade-off between memory traffic and on-chip compute.

---

[Beyond End-to-End Video Models: An LLM-Based Multi-Agent System for Educational Video Generation](http://arxiv.org/abs/2602.11790)

- LASEV: introduces a hierarchical multi-agent framework for generating high-quality instructional videos by decomposing the workflow into specialized agents coordinated by a central orchestrator.
- The system utilizes an Orchestrating Agent to supervise a Solution Agent, an Illustration Agent, and a Narration Agent, ensuring logical rigor through iterative critique loops.
- Rather than direct pixel synthesis, LASEV constructs a structured Executable Video Script (EVS) that is deterministically compiled into synchronized visuals and narration using template-driven assembly.

---

[Cooperation Breakdown in LLM Agents Under Communication Delays](http://arxiv.org/abs/2602.11754)

- FLCOA (Five Layers for Cooperation/Coordination among Autonomous Agents): introduces a hierarchical framework to analyze multi-agent cooperation, with all Mechanism Design, Monitoring & Enforcement, Agent, Message Protocol, Infrastructure, and State-management Server-components, where it investigates how lower-layer communication constraints trigger exploitation behaviors in LLM agents.
- The researchers implement a Continuous Prisoner’s Dilemma simulation to observe how varying latencies affect the emergence of mutual cooperation and retaliatory strategies.
- Findings demonstrate that excessive communication delays can paradoxically reduce cycles of exploitation, resulting in a non-monotonic U-shaped recovery of cooperative behavior.

---

[AmbiBench: Benchmarking Mobile GUI Agents Beyond One-Shot Instructions in the Wild](http://arxiv.org/abs/2602.11750)

- AmbiBench: introduces a mobile GUI agent benchmark for intent alignment using a User Simulator, Physical Sandbox, and MUSE adjudication agents to evaluate performance across four instruction clarity levels.
- The MUSE framework includes MLLM-based trajectory serialization, outcome verification, process inspection, and interaction auditing agents to perform fine-grained semantic audits of agent behavior.
- It enables precise performance profiling through metrics such as Requirement Coverage Rate and Information Gain Rate by capturing dynamic interaction history and execution traces in online environments.

---

[AIR: Improving Agent Safety through Incident Response](http://arxiv.org/abs/2602.11749)

- AIR (Agent Incident Response): introduces an autonomous incident response framework for LLM agents, featuring a domain-specific language for runtime detection, containment, recovery, and guardrail synthesis.
- The system integrates semantic checks grounded in environment state to detect failures and invokes tool-based remediation to restore safety across code, embodied, and computer-use agents.
- It automatically derives plan-level guardrail rules from incidents to prevent recurrence in future agent execution cycles.

---

[Text2GQL-Bench: A Text to Graph Query Language Benchmark [Experiment, Analysis & Benchmark]](http://arxiv.org/abs/2602.11745)

- Text2GQL-Bench: introduces a unified benchmark for Text-to-Graph-Query-Language tasks, with all Schema Converter (normalizes existing database schemas), Schema Generator (synthesizes domain-specific graph schemas), Data Processor (converts existing data formats), Data Generator (synthesizes data via logic generators), Query Translator (migrates queries between dialects), Query Generator (evolves queries through Graph-IR), User Simulator (generates multi-level natural questions), Knowledge Extractor (materializes implicit domain knowledge), and Evaluation Suite (measures execution and grammatical accuracy) components, where the framework provides a scalable pipeline for multi-domain, multi-dialect graph query dataset construction and evaluation.
- The system includes LLM-based schema generation, data synthesis, query evolution, and persona-driven question simulation agents to ensure high-quality dataset coverage.
- Experimental results reveal that fine-tuning smaller open-weight models significantly improves grammatical validity and execution accuracy, narrowing the performance gap with larger proprietary LLMs.

---

[Counterfactual Conditional Likelihood Rewards for Multiagent Exploration](http://arxiv.org/abs/2602.11740)

- CCL (Counterfactual Conditional Likelihood): introduces a reward function for multiagent systems that isolates each agent's unique contribution to joint exploration by comparing actual joint observations with counterfactual scenarios, utilizing random encoders, joint embeddings, episodic memory, k-NN density estimation, reward shaping, MAPPO, LSTM-based actors, and a centralized critic.
- The framework utilizes fixed random encoders to map observations into a joint embedding space, where k-nearest neighbor density estimation identifies coordinated regions of the state space.
- By rewarding agents for outcomes correlated with teammates, the approach reduces redundant exploration and accelerates learning in sparse-reward environments requiring tight coordination.

---

[Achieving EF1 and Epistemic EFX Guarantees Simultaneously](http://arxiv.org/abs/2602.11732)

- Lone Divider (Lone Divider Approach for Fair Division): introduces an iterative allocation framework for indivisible goods with additive valuations, utilizing the Lone Divider Algorithm, Strong EEFX Share, Residual Maximin Share (RMMS), Bipartite Matching, and EEFX Certificate to achieve simultaneous EFL and EEFX guarantees.
- The framework defines a new share-based fairness notion, the strong EEFX share, which serves as a threshold to ensure that every allocated bundle is epistemic envy-free up to any good from the agent's perspective.
- The research proves that the strong EEFX share is always at most the residual maximin share, enabling the existence of complete allocations that satisfy both envy-based and share-based fairness criteria.

---

[A Preliminary Assessment of Coding Agents for CFD Workflows](http://arxiv.org/abs/2602.11689)

- FoamHelper (OpenFOAM case execution & repair agent): introduces a lightweight configuration for tool-using coding agents to automate end-to-end CFD workflows by guiding agents toward tutorial reuse and log-driven repair.
- The framework leverages the OpenCode interface to allow LLMs to interact with terminal environments, performing file edits and executing command-line utilities for meshing and solving.
- Evaluation on FoamBench-Advanced indicates that prompt-guided tutorial reuse significantly improves completion rates for derivative tasks, while advanced models like GPT-5.2 enhance complex mesh generation.

---

[ViTaS: Visual Tactile Soft Fusion Contrastive Learning for Visuomotor Learning](http://arxiv.org/abs/2602.11643)

- ViTaS (Visual Tactile Soft Fusion Contrastive Learning): introduces a visuo-tactile representation learning framework that integrates visual and tactile modalities through soft fusion contrastive learning and a CVAE-based reconstruction module to enhance visuomotor policy performance.
- The framework utilizes separate CNN encoders for vision and touch, aligning them in a latent space using a switching contrastive objective that identifies similar cross-modal samples beyond simple temporal adjacency.
- A conditional variational autoencoder reconstructs visual observations from fused embeddings to exploit modality complementarity, providing robustness in self-occluded manipulation scenarios across simulation and real-world environments.

---

[When Agents Disagree With Themselves: Measuring Behavioral Consistency in LLM-Based Agents](http://arxiv.org/abs/2602.11619)

- ReAct (Reasoning and Acting): introduces a systematic empirical study of behavioral consistency in LLM-based agents, with ReAct-style agent, LLM, Search tool, Retrieve tool, Finish tool, and Reasoning loop, where the research quantifies how stochasticity in action selection impacts agent reliability.
- The study identifies that behavioral divergence frequently originates at the first search query and that path length serves as a predictive signal for failure.
- Experimental results across multiple models show that reducing sampling temperature significantly improves both consistency and accuracy in multi-step reasoning tasks.

---

[The Five Ws of Multi-Agent Communication: Who Talks to Whom, When, What, and Why: A Survey from MARL to Emergent Language and LLMs](http://arxiv.org/abs/2602.11583)

- MA-Comm (Multi-Agent Communication): introduces a unified analytical framework for multi-agent communication across reinforcement learning, emergent language, and large language model paradigms, with MARL-Comm, EL-Comm, LLM-Comm, Relevance Filtering, Targeted Communication, Message Encoding, Communication Timing, Training Frameworks, Reward Objectives, Agent Profiles, Memory Systems, Planning Modules, Action Execution, and Communication Topologies, where the survey organizes diverse methodologies around the "Five Ws" to expose shared design principles and cross-paradigm trade-offs.
- The framework categorizes communication strategies based on information selection, recipient matching, and temporal adaptation under resource constraints while highlighting the transition from opaque latent vectors to interpretable natural language.
- It identifies open challenges in grounding, scalability, and theoretical guarantees, proposing hybrid architectures that combine the reasoning capabilities of LLMs with the control-oriented stability of reinforcement learning.

---

[Move What Matters: Parameter-Efficient Domain Adaptation via Optimal Transport Flow for Collaborative Perception](http://arxiv.org/abs/2602.11565)

- FlowAdapt: introduces a parameter-efficient domain adaptation framework for multi-agent collaborative perception, featuring Wasserstein Greedy Sampling for redundancy elimination and Progressive Knowledge Transfer for cross-stage semantic recovery.
- The system utilizes a dual-path adapter architecture to concurrently model local geometric patterns and global semantic dependencies through learnable pathways.
- The framework incorporates collaborative agent prompts and a decoupled feature memory to handle heterogeneous local conditions and stabilize gradient computation during fine-tuning.

---

[FINITE-TIME FLOCKING OF AN INFINITE SET OF CUCKER-SMALE PARTICLES WITH SUBLINEAR VELOCITY COUPLINGS](http://arxiv.org/abs/2602.11555)

- ICS (Infinite Cucker-Smale): introduces a framework for finite-time flocking in infinite-particle systems using sublinear velocity couplings and a component-wise diameter analysis.
- The system employs sender networks with fixed or switching topologies to facilitate directed communication and leader-follower coordination.
- The analysis provides explicit alignment-time estimates independent of agent count, demonstrating robustness against time-varying influence structures.

---

[Budget-Constrained Agentic Large Language Models: Intention-Based Planning for Costly Tool Use](http://arxiv.org/abs/2602.11541)

- INTENT: introduces an inference-time planning framework for budget-constrained LLM agents using a Language World Model, Intention Predictor, Conditional Generator, Ideal Trajectory Simulation, Geometric Cost Calibration, Rollout Cache, and Blacklist to anticipate future tool usage and risk-calibrated costs.
- The system employs a probabilistic decomposition to separate semantic intentions from concrete tool outputs, enabling accurate cost estimation through deterministic lookahead of successful paths.
- Evaluated on StableToolBench, the framework strictly enforces hard budget constraints while significantly improving task success rates across diverse market settings and price shifts.

---

[CausalAgent: A Conversational Multi-Agent System for End-to-End Causal Inference](http://arxiv.org/abs/2602.11527)

- CausalAgent: introduces a conversational multi-agent system for end-to-end causal inference, featuring a router agent, data processing agent, causal structure learning agent, and reporting agent.
- The architecture integrates the Model Context Protocol to standardize tool interfaces and Retrieval-Augmented Generation to ensure LLM outputs align with theoretical causal frameworks.
- By explicitly modeling the analysis workflow through LangGraph, the system automates data cleaning, algorithm selection, and interactive report generation for non-expert users.

---

[Adaptive Milestone Reward for GUI Agents](http://arxiv.org/abs/2602.11524)

- ADMIRE (Adaptive Milestone Reward): introduces a verifiable reward mechanism for mobile GUI agents by dynamically distilling milestones from successful explorations to provide dense, high-fidelity feedback.
- The framework utilizes an asymmetric credit assignment strategy that filters noise from successful trajectories while providing scaffolding for failed attempts to improve exploration efficiency.
- It employs a semantic matching protocol using Sentence-BERT to verify action-milestone alignment and integrates with Group Relative Policy Optimization for robust policy updates.

---

[How Smart Is Your GUI Agent? A Framework for the Future of Software Interaction](http://arxiv.org/abs/2602.11514)

- GAL (GUI Agent Autonomy Levels): introduces a six-level taxonomy for software interaction, with all No Automation (manual user action), Minimal Assistance (passive agent guidance), Basic Automation (single-step task execution), Conditional Automation (multi-step workflow handling), High Automation (cross-application task coordination), and Full Automation (entirely autonomous end-to-end operation) components, where it establishes a standardized framework for benchmarking the progression of GUI agents toward full autonomy.
- The framework categorizes agents based on their ability to perceive interface elements, interpret user intent, and execute multi-step workflows across diverse software environments.
- The research evaluates advanced systems like Manus, which includes planning-, perception-, and specialized sub-agents to execute complex tasks within GUI environments.

---

[A Generic Framework for Fair Consensus Clustering in Streams](http://arxiv.org/abs/2602.11500)

- st-1Med (Streaming 1-Median Fair Consensus): introduces a two-phase algorithmic framework for fair consensus clustering in data streams, with Sampled Store 1 (M1), Sampled Store 2 (M2), Post-stream Reconstruction, Candidate Set Generator, Selection Module, Cluster Fitting Module, and Closest Fair Clustering Oracle, where it achieves constant-factor approximation using sublinear space.
- The system employs uniform sampling to retain a logarithmic number of inputs and utilizes union-find structures to reconstruct clusterings from streamed triples.
- The framework is fairness-agnostic, allowing the integration of any approximately close fair clustering algorithm as a modular subroutine.

---

[What if Agents Could Imagine? Reinforcing Open-Vocabulary HOI Comprehension through Generation](http://arxiv.org/abs/2602.11499)

- ImagineAgent: introduces an agentic framework that integrates cognitive reasoning, generative imagination, and tool-augmented reinforcement learning for open-vocabulary human-object interaction comprehension.
- The architecture employs a vision encoder and QwenLM decoder to dynamically invoke tools including image cropping, diffusion-based outpainting, and retrieval-augmented generation for evidence gathering.
- The framework utilizes Group Relative Policy Optimization with a composite reward to balance prediction accuracy, structural coherence, and tool efficiency during policy refinement.

---

[Understanding Persuasive Interactions between Generative Social Agents and Humans: The Knowledge-based Persuasion Model (KPM)](http://arxiv.org/abs/2602.11483)

- KPM (Knowledge-based Persuasion Model): introduces a theoretical framework to understand how a generative social agent's internalized knowledge drives its persuasive behavior and subsequent human responses.
- The model structures agent knowledge into self, user, and context domains to guide the autonomous generation of message content and delivery styles.
- It evaluates persuasion effectiveness through user attitudes and behavioral changes while accounting for environmental, task, and attribute-based contextual layers.

---

[Agentic Test-Time Scaling for WebAgents](http://arxiv.org/abs/2602.12276)

- CATTS (Confidence-Aware Test-Time Scaling): introduces a dynamic compute allocation technique for multi-step web agents, with base model, candidates, semantic deduplicator, majority vote, confidence-aware gating, and arbiter components, where it allocates additional LLM-based arbitration compute only when vote-derived uncertainty exceeds a threshold.
- The framework utilizes uncertainty statistics like entropy and probability margin from sampled action distributions to identify contentious decision points where simple majority voting might fail.
- By bypassing the arbiter on high-consensus steps, the system improves task success rates while significantly reducing token consumption compared to uniform scaling strategies.

---

[AttentionRetriever: Attention Layers are Secretly Long Document Retrievers](http://arxiv.org/abs/2602.12278)

- AttentionRetriever: introduces a training-free long document retrieval framework that leverages pretrained LLM attention maps and entity-based retrieval to build context-aware embeddings and determine retrieval scope.
- The system utilizes cross-attention scores from specific LLM layers to model contextual and causal dependencies, combined with a dense embedding model for multi-view similarity search.
- It incorporates an entity graph structure to discover hidden background information and accurately define the retrieval scope for complex, multi-hop queries in extremely long documents.

---

[Geometry of Uncertainty: Learning Metric Spaces for Multimodal State Estimation in RL](http://arxiv.org/abs/2602.12087)

- METRICMM (Metric Learning for Multimodal State Estimation): introduces a state estimation framework that learns a structured latent representation where Euclidean distances correlate with the minimum actions required for transitions, utilizing per-modality encoders, a latent transition model, and an inverse distance weighting fusion mechanism.
- The framework recasts uncertainty estimation geometrically by weighting sensor contributions based on their proximity to transition model predictions in the shared metric space.
- It achieves robustness to unseen sensor noise and corruptions in multimodal reinforcement learning tasks without requiring explicit probabilistic modeling or noise augmentation during training.

---

[RF-Modulated Adaptive Communication Improves Multi-Agent Robotic Exploration](http://arxiv.org/abs/2602.12074)

- ART (Adaptive-RF Transmission): introduces a communication-aware motion planning algorithm that dynamically selects transmission locations by balancing signal strength, payload size, and navigation costs for heterogeneous robot teams.
- The architecture integrates ROS2-based navigation and mapping with a specialized RF module that calculates disruption scores to minimize backtracking during multi-agent exploration.
- Experimental results in cave-like environments show that adaptive transmission strategies improve coverage efficiency and mission speed for heterogeneous Scout-Specialist robot teams.

---

[LawThinker: A Deep Research Legal Agent in Dynamic Environments](http://arxiv.org/abs/2602.12056)

- LawThinker: introduces an autonomous legal research agent utilizing an Explore-Verify-Memorize strategy to ensure procedural compliance in dynamic judicial environments, featuring main reasoning-, verification-, and memory-agents.
- The framework enforces verification as an atomic operation after every exploration step to prevent the propagation of incorrect statute citations or reasoning errors.
- It employs a suite of 15 specialized tools and a dual-channel memory system to support long-horizon tasks like document drafting and courtroom simulation.

---

[PrefillShare: A Shared Prefill Module for KV Reuse in Multi-LLM Disaggregated Serving](http://arxiv.org/abs/2602.12029)

- PrefillShare: introduces a shared-prefill disaggregated serving algorithm that factorizes LLMs into a frozen base prefill module and multiple task-specific decode modules to enable cross-model KV cache reuse via a shared KV cache, prefix-aware routing, and cache handoff within a disaggregated serving infrastructure.
- The system employs cache-conditioned fine-tuning to align specialized decoders with the frozen base model's KV cache distribution, ensuring accuracy while eliminating redundant prefill computation.
- Experimental results demonstrate that the approach achieves up to 4.5x lower tail latency and 3.9x higher throughput in multi-model agent workloads compared to standard disaggregated serving.

---

[CM2: Reinforcement Learning with Checklist Rewards for Multi-Turn and Multi-Step Agentic Tool Use](http://arxiv.org/abs/2602.12268)

- CM2 (Checklist Reward for Multi-turn Multi-step Agentic Tool Use): introduces a reinforcement learning framework for multi-turn tool-use agents, with Multi-Turn Multi-Step Trajectory, Data Filtering, CoT Compression, Cold Start SFT, Checklist Labeling, RL Training, LLM Judge, and Tool Simulation, where fine-grained binary checklist rewards replace rule-based verifiable signals to optimize multi-turn and multi-step tool-use capabilities.
- CM2 includes filtering-, compression-, annotating-, judging-, and simulating-LLMs to facilitate a scalable training pipeline without real-world API dependencies.
- The framework employs a "sparse in assignment, dense in criteria" strategy to balance training stability with informative feedback across complex agentic trajectories.

---

[STAR: Bridging Statistical and Agentic Reasoning for Large Model Performance Prediction](http://arxiv.org/abs/2602.12143)

- STAR (STatistical expectations and Agentic Reasoning): introduces a performance prediction framework for LLMs, with Historical Memory, Retrieval Augmentation, Statistical Expectation, EVT-guided Reasoning, and Credibility-aware Aggregation, where it integrates statistical modeling with agentic reasoning to predict benchmark scores under extreme data sparsity.
- The framework utilizes Constrained Probabilistic Matrix Factorization to generate initial expectations and uncertainty estimates by embedding semantic features retrieved from technical documentation and community feedback.
- The reasoning module includes intra-family analysis-, cross-model comparison-, and credibility-aware aggregation-agents to provide traceable natural language explanations for score adjustments.

---

[Accelerating Robotic Reinforcement Learning with Agent Guidance](http://arxiv.org/abs/2602.11978)

- AGPS (Agent-guided Policy Search): introduces an automated reinforcement learning framework that replaces human supervisors with a multimodal agent, incorporating FLOAT, a toolbox of perception and action primitives, and episodic memory to provide Action Guidance and Exploration Pruning.
- The system integrates FLOAT, an asynchronous failure detector using Optimal Transport, to trigger agent interventions only when policy execution deviates from expert manifolds.
- The framework leverages zero-shot semantic priors to generate corrective waypoints and 3D spatial constraints, improving sample efficiency in rigid and deformable object manipulation tasks.

---

[Intelligent AI Delegation](http://arxiv.org/abs/2602.11865)

- Intelligent AI Delegation: introduces an adaptive framework for managing complex task allocation between humans and LLM-based agents through task decomposition, delegatee identification, proposal selection, task assignment, negotiation, adaptive coordination, monitoring, trust calibration, permission handling, and verifiable completion.
- The system employs a structured pipeline for task decomposition, decentralized market-based assignment, and smart contract formalization to establish clear roles and accountability.
- It incorporates an adaptive coordination cycle to handle environmental shifts and failures through continuous monitoring and dynamic re-allocation of resources.

---

[TEMPORAL DIFFERENCE LEARNING WITH CONSTRAINED INITIAL REPRESENTATIONS](http://arxiv.org/abs/2602.11800)

- CIR (Constrained Initial Representations): introduces a reinforcement learning framework that enhances sample efficiency using a representation constraint module, a skip connection module, convex Q-learning, U-shape critic networks, and an MLP actor network to stabilize training and mitigate distribution shifts.
- The U-shape critic architecture employs skip connections to maintain information flow between shallow and deep layers while scaling network capacity.
- The convex Q-learning approach adaptively combines minimum and maximum value estimates to counteract conservatism and improve value estimation flexibility.

---

[TSR: Trajectory-Search Rollouts for Multi-Turn RL of Large Language Model Agents](http://arxiv.org/abs/2602.11767)

- TSR (Trajectory-Search Rollouts): introduces a training-time rollout framework that repurposes test-time search ideas to construct high-quality trajectories for multi-turn reinforcement learning of LLM agents.
- The system integrates lightweight tree-style search strategies, including best-of-N, beam search, and shallow lookahead, to actively explore and prune trajectories during the rollout phase.
- This approach improves training stability and final task accuracy by providing stronger learning signals through optimized per-turn rollouts while remaining optimizer-agnostic.

---

[AC-MASAC: An Attentive Curriculum Learning Framework for Heterogeneous UAV Swarm Coordination](http://arxiv.org/abs/2602.11735)

- AC-MASAC (Attentive Curriculum-driven Multi-Agent Soft Actor-Critic): introduces a multi-agent reinforcement learning framework for heterogeneous UAV swarm coordination, with leader- and follower-specific actor networks and a structured attention critic.
- The architecture utilizes role-aware attention mechanisms to model asymmetric inter-agent dependencies while processing local observations for decentralized execution.
- A progressive curriculum strategy incorporating stage-proportional experience replay and hierarchical knowledge transfer is employed to address sparse rewards and training instability.

---

[WebTestPilot: Agentic End-to-End Web Testing against Natural Language Specification by Inferring Oracles with Symbolized GUI Elements](http://arxiv.org/abs/2602.11724)

- WebTestPilot: introduces an agentic end-to-end web testing framework that infers verifiable test oracles from natural language specifications using input parsing, oracle inference, symbolization, oracle execution, DSL, and history memory.
- The architecture includes parsing-, inference-, action execution-, and page reidentification-agents to handle stochastic reasoning and implicit requirement validation.
- By combining neural reasoning with symbolic assertions, the system achieves high bug detection precision and generalizes across diverse natural language inputs and model scales.

---

[ABot-N0: Technical Report on the VLA Foundation Model for Versatile Embodied Navigation](http://arxiv.org/abs/2602.11598)

- ABot-N0: introduces a unified Vision-Language-Action foundation model for embodied navigation, featuring a hierarchical "Brain-Action" architecture that pairs an LLM-based Cognitive Brain with a Flow Matching-based Action Expert.
- The system integrates an Agentic Planner for Chain-of-Thought intent decomposition and a hierarchical Topo-Memory to maintain persistent spatial knowledge across complex environments.
- The model achieves state-of-the-art performance across five core navigation tasks by training on a massive dataset of 16.9 million expert trajectories and 5 million reasoning samples.

---

[Learning to Configure Agentic AI Systems](http://arxiv.org/abs/2602.11574)

- ARC (Agentic Resource & Configuration learner): introduces a hierarchical reinforcement learning framework to dynamically tailor LLM-based agent configurations, including workflows, tools, and budgets, on a per-query basis.
- The system utilizes a two-level policy architecture where a high-level structure policy selects architectural components and a low-level prompt policy composes specific instructions for reasoner-, verifier-, and orchestrator-agents.
- By combining masked reinforcement learning with supervised fine-tuning, ARC achieves higher task accuracy while significantly reducing computational costs compared to static "one-size-fits-all" agent designs.

---

[AgentLeak: A Full-Stack Benchmark for Privacy Leakage in Multi-Agent LLM Systems](http://arxiv.org/abs/2602.11510)

- AgentLeak: introduces a full-stack benchmark for evaluating privacy leakage in multi-agent LLM systems, which includes coordinator- and worker-agents communicating through internal channels like inter-agent messages and shared memory.
- The framework utilizes a three-tier detection pipeline consisting of canary matching, structured field audits, and LLM-as-judge to identify unauthorized exposure of sensitive data across seven distinct channels.
- Experimental results show that internal channels account for the majority of privacy violations, with inter-agent messages leaking at significantly higher rates than final user-facing outputs.

---

[LLM-based Triplet Extraction from Financial Reports](http://arxiv.org/abs/2602.11886)

- LLM-based Triplet Extraction Pipeline: introduces a semi-automated framework for Subject-Predicate-Object triplet extraction from financial reports using ontology-driven proxy metrics, with Document Corpus (unstructured financial report input), Segmentation (5-sentence chunking for context), Ontology Induction LLM (automated document-specific schema generation), Extraction LLM (SPO triplet generation from chunks), and Hybrid Verifier (semantic grounding and hallucination filtering).
- The system compares static manual ontologies against document-specific induced schemas to eliminate ontology drift and ensure 100% schema conformance across different LLMs.
- The research identifies a systematic asymmetry in hallucination rates between subjects and objects, attributed to passive voice and omitted agents in financial prose.

---

[Code2Worlds: Empowering Coding LLMs for 4D World Generation](http://arxiv.org/abs/2602.11757)

- Code2Worlds: introduces a language-to-simulation framework that formulates 4D world generation as executable code, with ObjSelect Agent, ObjParams Agent, Environment Planner, PostProcess Agent, and VLM-Motion Critic, and includes selection-, planning-, and post-processing-agents.
- The system employs a physics-aware closed-loop mechanism where a PostProcess Agent scripts dynamics and a VLM-Motion Critic performs iterative self-reflection to correct physical hallucinations in rendered simulations.
- By leveraging retrieval-augmented parametric generation and hierarchical scene planning, the framework achieves quantifiable improvements in structural richness and dynamic fidelity compared to prior static code-to-scene methods.

---

[Right for the Wrong Reasons: Epistemic Regret Minimization for Causal Rung Collapse in LLMs](http://arxiv.org/abs/2602.11675)

- ERM (Epistemic Regret Minimization): introduces a three-layer architecture to address "Rung Collapse" in LLMs by penalizing causal reasoning errors independently of task success using interventional data.
- The framework employs a frozen LLM for hypothesis generation, an external causal graph for domain-specific beliefs, and a failure mode taxonomy to inject cross-domain reasoning guards.
- It utilizes a Causal Transaction Log to record action-outcome traces, enabling precise belief revision and routing persistent high-regret tasks to alternative models or human reviewers.

---

[SIGHT: Reinforcement Learning with Self-Evidence and Information-Gain Diverse Branching for Search Agent](http://arxiv.org/abs/2602.11551)

- SIGHT (Self-Evidence and Information-Gain Driven Diverse Branching): introduces a reinforcement learning framework for multi-turn search tasks, with Actor LLM, Search Engine, SES Mechanism, IG Monitoring, Dynamic Prompting Interventions, Hierarchical Reward Modeling, and GRPO.
- The system employs Self-Evidence Support to distill concise evidence from raw search results and uses Information-Gain scores to identify pivotal states for adaptive branching or corrective reflection.
- By masking guidance hints during policy updates, the framework enables LLMs to internalize exploration behaviors, improving multi-hop reasoning accuracy while reducing redundant tool calls.

---

[FORMALJUDGE: A Neuro-Symbolic Paradigm for Agentic Oversight](http://arxiv.org/abs/2602.11136)

- FORMALJUDGE: introduces a neuro-symbolic framework for agentic oversight that employs a bidirectional Formal-of-Thought architecture to decompose human intent into atomic facts verified by deterministic SMT solvers.
- The system utilizes LLMs as specification compilers to translate natural language requirements into Dafny specifications, ensuring oversight remains immune to persuasive manipulation through mathematical proofs.
- Research demonstrates that FORMALJUDGE enables weak-to-strong generalization, where a 7B judge accurately detects deception in 72B agents, and provides near-linear safety improvements via iterative refinement.

---

[“You Can Actually Do Something”: Shifts in High School Computer Science Teachers’ Conceptions of AI/ML Systems and Algorithmic Justice](http://arxiv.org/abs/2602.16123)

- AI Auditing (Participatory Design for Critical AI Literacy): introduces a longitudinal study where computer science teachers co-design and implement curricula to systematically query AI/ML systems for algorithmic bias, with all Participatory Design (collaborative curriculum development process), AI Auditing (five-step systematic evaluation process), Professional Development Workshops (teacher training and expert engagement), Classroom Implementation (practical application of lessons), Qualitative Analysis (thematic coding of reflections), Relational Ethics (context-grounded framework for justice), and Computational Empowerment (skills for critical digital engagement) components.
- The approach utilizes professional development workshops and expert consultations to transition teachers from passive observers to active investigators of algorithmic injustice within their school communities.
- Research results indicate that teachers developed more situated and agentic perspectives, enabling them to integrate ethical critiques of LLMs and other algorithmic tools into their specific educational contexts.

---

#### 11th February 2026

[DISTRIBUTIONALLY ROBUST COOPERATIVE MULTI-AGENT REINFORCEMENT LEARNING VIA ROBUST VALUE FACTORIZATION](http://arxiv.org/abs/2602.11437)

- DrIGM (Distributionally robust Individual-Global-Maximum): introduces a robustness principle for cooperative multi-agent reinforcement learning using robust individual action-value networks, factorization networks, target networks, a replay buffer, dual networks, and a robust Bellman operator.
- The framework aligns decentralized greedy actions with the robust team-optimal joint action by training on robust Q-targets derived from global worst-case models.
- The framework employs rho-contamination and total variation uncertainty sets to improve out-of-distribution performance in partially observable environments.

---

[Fair Data-Exchange Mechanisms](http://arxiv.org/abs/2602.11417)

- Fair-Exchange Mechanism: introduces a reciprocal data-sharing framework without monetary transfers, with Strategic Agents, Data Collection, Fair-Exchange Contract, Total-Access Coordinates, Maximal Equilibrium Algorithm, and Enforcement Models, where agents exchange data one-for-one based on the minimum of their individual collection levels.
- The mechanism leverages supermodularity in total-access coordinates to ensure the existence of a lattice of pure Nash equilibria and a Pareto-best outcome.
- It provides a quadratic-time algorithm to compute the maximal equilibrium and proves truthful implementation under natural enforcement assumptions for continuous, graph-restricted, and discrete settings.

---

[Multi-Level Strategic Classification: Incentivizing Improvement through Promotion and Relegation Dynamics](http://arxiv.org/abs/2602.11439)

- Multi-Level Strategic Classification: introduces a sequential decision-making model to incentivize genuine attribute improvement over gaming through promotion and relegation dynamics, with Agent, Principal, Ternary Classifiers, Inter-temporal Incentives, and Attribute Dynamics.
- The framework incorporates inter-temporal factors including agent farsightedness, skill retention, and a "leg-up" effect where higher attainment levels boost future qualification gains.
- Theoretical analysis and simulations demonstrate that this multi-level structure significantly expands the region where honest effort is incentivizable compared to traditional one-shot strategic classification settings.

---

[OPTIMIZING AGENT PLANNING FOR SECURITY AND AUTONOMY](http://arxiv.org/abs/2602.11416)

- PRUDENTIA: introduces a security-aware agent architecture that optimizes for autonomy with a planner-LLM, a quarantined-LLM, information-flow control, a plan tool, a variable expansion tool, human-in-the-loop interactions, security labels, and variables.
- The system employs a dual LLM configuration where the strategic planner manages security labels and justifies variable expansion to prevent premature context tainting.
- Evaluation on AgentDojo and WASP benchmarks demonstrates that this approach significantly reduces human oversight requirements without compromising task completion rates or security guarantees.

---

[When Visibility Outpaces Verification: Delayed Verification and Narrative Lock-in in Agentic AI Discourse](http://arxiv.org/abs/2602.11412)

- Narrative Lock-in Analysis: introduces a methodology to study how platform-visible engagement signals influence the timing of verification cues in online discussions of agentic AI, with all Agentic AI Discourse Analysis, Platform-visible engagement signals, Verification cues, Time-to-first-verification, Credibility proxy, Diffusion of epistemic responsibility, and Narrative lock-in-components, where it characterizes the pre-interaction trust environment of autonomous systems.
- The framework identifies a "Popularity Paradox" where high-visibility threads experience significantly delayed or absent evidence-seeking compared to low-visibility discussions.
- The research highlights the risk of "Narrative Lock-in," where early unverified claims about autonomous systems stabilize into collective biases before substantiation occurs.

---

[TRACER: Trajectory Risk Aggregation for Critical Episodes in Agentic Reasoning](http://arxiv.org/abs/2602.11409)

- TRACER (Trajectory Risk Aggregation for Critical Episodes in agentic Reasoning): introduces a trajectory-level uncertainty metric for dual-control agentic environments, with content-aware normalized surprisal, agent repetition indicator, agent action-observation mismatch indicator, user-agent coordination gap indicator, MAX-composite step risk aggregator, tail-focused trajectory aggregator, and final risk score.
- The framework identifies sparse critical episodes like looping or tool-use incoherence by aggregating step-wise signals through a tail-focused risk functional that emphasizes worst-case segments over global averages.
- Evaluation on multi-turn benchmarks demonstrates that TRACER improves failure detection accuracy and enables earlier early-warning signals for LLMs agents in interactive settings.

---

[The Distortion of Prior-Independent b-Matching Mechanisms](http://arxiv.org/abs/2602.11404)

- RS (Random Survivors): introduces a prior-independent mechanism for b-matching that achieves optimal distortion by selecting a random subset of "survivor" agents based on quota-dependent probabilities and allocating items uniformly among those who reported them as favorites.
- RSBS (Random Survivors with Item Burning and Stealing) provides a three-phase refinement that incorporates probabilistic assignment cancellation and a final stealing phase for the largest-quota agent to achieve a near-optimal distortion gap.
- HQL (Highest Quota Last) establishes a sequential one-pass mechanism that maintains optimal distortion and distortion gap guarantees by processing agents in a deterministic order where the agent with the highest quota is approached last.

---

[Maximizing Index Diversity in Committee Elections](http://arxiv.org/abs/2602.11400)

- LC (Lexicographic Counting Index): introduces a multiwinner election approach that integrates ecological diversity indices into approval-based voting to select committees with high voter support and balanced candidate attributes.
- The framework defines two models, MAX-D-DSCR and MAX-D-DSAT, which optimize diversity subject to aggregate score lower bounds or individual agent satisfaction guarantees.
- The authors characterize the Lexicographic Counting Index through axiomatic properties like 1-explainability and present label maximization, differentiating it from established ecological metrics like Shannon's entropy or the Simpson index.

---

[Causal-JEPA: Learning World Models through Object-Level Latent Interventions](http://arxiv.org/abs/2602.11389)

- C-JEPA (Causal Joint Embedding Predictive Architecture): introduces an object-centric world model that utilizes object-level masking as a latent intervention to learn interaction-aware dynamics without pixel reconstruction, with frozen object-centric encoder, object-level masking, ViT-style masked transformer predictor, auxiliary variables, and joint latent prediction objective.
- The architecture employs a frozen encoder to extract entity-level slots and a transformer-based predictor to jointly infer masked history and future latent states.
- This method induces a causal inductive bias that improves counterfactual reasoning and enables efficient model-predictive control using a fraction of the tokens required by patch-based models.

---

[MEmilio – A high performance Modular EpideMIcs simuLatIOn software for multi-scale and comparative simulations of infectious disease dynamics](http://arxiv.org/abs/2602.11381)

- 
MEmilio (Modular EpideMIcs simuLatIOn): introduces a high-performance, modular framework for simulating infectious disease dynamics across multiple scales, featuring a C++ simulation backend coupled with a Python frontend for flexible model specification and execution.

- 
The architecture utilizes uniform model descriptions and a pygen interface generator to harmonize diverse modeling paradigms, including population-based, metapopulation, and agent-based models, within a single software ecosystem.

- 
It supports advanced features such as temporal-hybrid modeling, optimal control for intervention strategies, and integration with machine learning surrogate packages to accelerate large-scale epidemiological studies and public health decision-making.


---

[ReplicatorBench: Benchmarking LLM Agents for Replicability in Social and Behavioral Sciences](http://arxiv.org/abs/2602.11354)

- ReplicatorAgent: introduces an end-to-end agentic framework for replicating social and behavioral science research claims, utilizing a ReAct-style loop to navigate extraction, generation, and interpretation stages.
- The system operates within a sandboxed Docker environment and employs a specialized tool palette for iterative debugging, file editing, and data inspection to resolve execution failures.
- The framework is evaluated against ReplicatorBench, a benchmark featuring 1,568 gradable checkpoints derived from human-expert replication reports to assess AI agents' capability in mimicking real-world scientific replication.

---

[Pushing Forward Pareto Frontiers of Proactive Agents with Behavioral Agentic Optimization](http://arxiv.org/abs/2602.11351)

- BAO (Behavioral Agentic Optimization): introduces an agentic reinforcement learning framework that balances task performance and user engagement by combining behavior enhancement for proactive reasoning with behavior regularization to suppress redundant interactions.
- The system employs a warm-start pipeline where a teacher model generates behavior-enhanced data for supervised fine-tuning, followed by reinforcement learning using Group Relative Policy Optimization (GRPO) with turn-level reward shaping.
- It integrates retrospective reasoning for hypothesis refinement and prospective planning for dynamic scheduling to optimize proactive LLM agents across multi-turn interaction trajectories.

---

[When agents choose bundles autonomously: guarantees beyond discrepancy](http://arxiv.org/abs/2602.11330)

- Sequential Arrival Model: introduces a fair division framework for indivisible items where rational agents autonomously select bundles from a partition, with all Sequential Arrival Model (fixed agent entry order), Dynamic Rebundling (adapting partitions between arrivals), Donor-Receiver Transfer (moving items to ensure preference), Recursive Sub-instance (remaining items for future agents), and Core-plus-Round-Robin Partitioning (combining pre-allocation with standard protocols) components, where the mechanism utilizes dynamic rebundling to surpass the discrepancy barrier.
- The approach employs a recursive strategy that transfers items from donor parts to receiver parts to ensure the current agent's assigned bundle is their unique favorite among unpicked options.
- The research establishes polynomial-time algorithms achieving guarantees close to the proportional share for various valuation classes, including instances with bounded influence or ordered additive preferences.

---

[Security Threat Modeling for Emerging AI-Agent Protocols: A Comparative Analysis of MCP, A2A, Agora, and ANP](http://arxiv.org/abs/2602.11327)

- MCP (Model Context Protocol), A2A (Agent2Agent Protocol), Agora, and ANP (Agent Network Protocol): introduces a systematic security threat modeling and risk assessment framework for emerging AI agent communication protocols, with MCP Host, MCP Client, MCP Server, Client Agent, Remote Agent, A2A Protocol Layer, Meta-Protocol Layer, Application Protocol Layer, Identity and Encrypted Communication Layer, and Orchestrator.
- The analysis identifies twelve protocol-level risks across the creation, operation, and update phases to evaluate the security posture of multi-agent ecosystems.
- A measurement-driven case study on MCP formalizes the risk of missing mandatory validation for executable components by quantifying wrong-provider tool execution under multi-server composition.

---

[CryptoAnalystBench: Failures in Multi-Tool Long-Form LLM Analysis](http://arxiv.org/abs/2602.11304)

- Agentic Harness: introduces a production-aligned benchmark and an agentic evaluation harness to investigate failure modes in long-form LLM analysis using multi-tool orchestration over volatile cryptocurrency data, with Natural Language Query, Query Processing Module, Tool Execution Environment, Reasoning Engine, ReAct Loop, Generated Response, Evaluation Pipeline, and Citation Verification Pipeline-components.
- The framework utilizes a multi-level evaluation methodology combining automated citation verification with an LLM-as-a-judge rubric to assess depth, relevance, temporal accuracy, and data consistency.
- The research identifies a taxonomy of seven higher-order error types, such as source reconciliation failures and overconfident predictions, that persist in frontier models despite high factual grounding.

---

[The PBSAI Governance Ecosystem: A Multi-Agent AI Reference Architecture for Securing Enterprise AI Estates](http://arxiv.org/abs/2602.11301)

- PBSAI (Practitioner’s Blueprint for Secure AI) Governance Ecosystem: introduces a multi-agent reference architecture for securing enterprise AI estates, with Governance & Oversight (Domain A), Operational Security Domains (Domains B-K), Security Program Enablement & AI Validation (Domain L), Agent Pattern & Coordination Layer, Tools & Models, and Infrastructure.
- The system coordinates specialized agents—including governance-, risk-, monitoring-, and validation-agents—using a deterministic-first logic pattern supplemented by bounded LLMs.
- It utilizes MCP-style context envelopes and structured Output Contracts to link controls, events, and models into a verifiable evidence graph.

---

[From Natural Language to Materials Discovery: The Materials Knowledge Navigation Agent](http://arxiv.org/abs/2602.11123)

- MKNA (Materials Knowledge Navigation Agent): introduces an autonomous scientific decision-maker that unifies semantic interpretation, knowledge retrieval, surrogate property estimation, structure generation, and physics-informed stability validation into a coherent closed loop.
- The framework includes reasoning-, code generation-, and code repair-agents to autonomously extract quantitative thresholds from literature and synthesize custom data retrieval routines.
- The workflow integrates graph-based property prediction with physics-based validation to identify previously unreported Be-C-rich compounds and reconstruct interpretable design heuristics from unstructured scientific text.

---

[Learning to Compose for Cross-domain Agentic Workflow Generation](http://arxiv.org/abs/2602.11114)

- CapFlow (Workflow Capability Basis Learning): introduces a framework that internalizes a decompose-recompose-decide mechanism into LLMs to enable single-pass, cross-domain agentic workflow generation.
- The architecture utilizes a task-conditioned composer to select a sparse mixture of reusable capability bases, which are then broadcast as composed layer weights to the underlying LLM layers.
- The system employs counterfactual capability attribution to align basis selection with factors driving workflow success, significantly reducing inference latency and cost compared to iterative refinement methods.

---

[GameDevBench: Evaluating Agentic Capabilities Through Game Development](http://arxiv.org/abs/2602.11103)

- GameDevBench: introduces a multimodal benchmark for evaluating agentic capabilities in game development, with Godot Engine, Scene Editor, Script Editor, Contextual Editors, Editor Screenshot MCP, Runtime Video, Codex Agent, GPT-5-mini Agent, and Agent Frameworks.
- The framework evaluates agents across multiple skill categories including gameplay logic, 2D/3D graphics, and user interface design using deterministic verification through Godot's scripting framework.
- It incorporates multimodal feedback mechanisms like Editor Screenshot MCP and Runtime Video to provide agents with visual and temporal context of the game state.

---

[Interpretable Attention-Based Multi-Agent PPO for Latency Spike Resolution in 6G RAN Slicing](http://arxiv.org/abs/2602.11076)

- AE-MAPPO (Attention-Enhanced Multi-Agent Proximal Policy Optimization): introduces an interpretable multi-agent reinforcement learning framework for 6G radio access network slicing, integrating six specialized attention mechanisms and including URLLC-, eMBB- and mMTC-agents.
- The architecture utilizes a three-phase allocation strategy—predictive, reactive, and inter-slice optimization—to orchestrate power, bandwidth, and computation across O-RAN timescales.
- It incorporates semantic, temporal, cross-slice, confidence, counterfactual, and meta-controller attention heads to generate human-readable insights and ensure ultra-reliable low-latency communication.

---

[EVALUATING MEMORY STRUCTURE IN LLM AGENTS](http://arxiv.org/abs/2602.11243)

- StructMemEval: introduces a benchmark suite evaluating the ability of LLM agents to organize long-term memory into specific structures, with StructMemEval, Retrieval-augmented LLM, Mem-agent, Mem0, Backbone LLM, Long-term Memory, and Memory Organization Hints.
- The evaluation compares simple retrieval-augmented baselines against more complex memory systems, utilizing backbone models including Gemini-2.5-pro, Gemini-3-pro, GPT-4o-mini, and Gemini-3-flash.
- Experimental results indicate that while memory-augmented agents outperform simple retrieval, they often require explicit structural hints to correctly organize information and avoid failure modes like hallucinations or omission.

---

[ReTracing: An Archaeological Approach Through Body, Machine, and Generative Systems](http://arxiv.org/abs/2602.11242)

- ReTracing: introduces a multi-agent embodied performance framework that utilizes LLs to transform literary excerpts into paired movement prompts for human and robotic agents.
- The system employs a text-to-video diffusion model to generate visual choreographic guides for humans while simultaneously directing a quadruped robot through programmed motor commands.
- Physical interactions are captured and processed via monocular 3D point tracking to construct a digital archive of motion traces for analyzing algorithmic biases.

---

[Active Zero: Self-Evolving Vision-Language Models through Active Environment Exploration](http://arxiv.org/abs/2602.11241)

- Active-Zero: introduces a tri-agent self-play framework for vision-language models, which includes Searcher-, Questioner- and Solver-agents, an Open-World Environment, and a Reward System, where the system shifts from passive interaction to active exploration to autonomously construct a learning trajectory.
- The framework utilizes a three-stage co-evolutionary cycle where the Searcher identifies visual scenarios at the Solver's capability frontier, the Questioner transforms these into instructional curricula, and the Solver is optimized via Group Relative Policy Optimization (GRPO).
- By integrating uncertainty-driven retrieval and consensus-based reinforcement learning, the model achieves measurable performance improvements on both reasoning-intensive and general visual understanding benchmarks without human annotation.

---

[DIVIDE, HARMONIZE, THEN CONQUER IT: SHOOTING MULTI-COMMODITY FLOW PROBLEMS WITH MULTI-MODAL LANGUAGE MODELS](http://arxiv.org/abs/2602.11057)

- PRAM (Partitioned Resource Allocation with Multimodal Language Models): introduces a distributed optimization framework that decomposes complex multi-commodity flow problems into local sub-tasks resolved in parallel by a shared multimodal LLM-based agent.
- The architecture integrates a vision encoder for topology sub-images and a tokenizer for textual demand data, utilizing LoRA adapters and learned global context to facilitate efficient inter-agent communication and reasoning.
- A multi-agent reinforcement learning algorithm employing counterfactual policy gradients harmonizes sub-task solutions to ensure global consistency, achieving near-optimal performance significantly faster than traditional linear programming solvers.

---

#### 10th February 2026

[SteerVLA: Steering Vision-Language-Action Models in Long-Tail Driving Scenarios](http://arxiv.org/abs/2602.08440)

- SteerVLA: introduces a hierarchical vision-language-action framework for autonomous driving, featuring a high-level VLM policy for semantic reasoning and a low-level VLA policy for precise control.
- The framework utilizes an automatic labeling pipeline to generate grounded reasoning traces and prescriptive meta-actions, bridging high-level semantic understanding with low-level vehicle control.
- The system is implemented using InternVL2-1B as the base model for both policies and demonstrates improved generalization by offloading high-level reasoning to the planner component.

---

[When Do Multi-Agent Systems Outperform? Analysing the Learning Efficiency of Agentic Systems](http://arxiv.org/abs/2602.08272)

- MARL (Multi-Agent Reinforcement Learning): introduces a theoretical analysis of learning efficiency in agentic systems, comparing SARL and MARL through PAC-based sample complexity bounds across dependent and independent task decompositions.
- The framework demonstrates that MARL achieves superior sample efficiency when tasks are decomposable into independent subtasks, whereas interdependencies introduce error propagation and quadratic penalties.
- It quantifies task alignment as the discrepancy between decomposed and unified rewards, identifying precise conditions where multi-agent strategies retain advantages despite imperfect task structures.

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

