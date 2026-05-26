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





#### 22nd May 2026

[SkillOpt: Executive Strategy for Self-Evolving Agent Skills](http://arxiv.org/abs/2605.23904)

- SkillOpt: introduces a systematic text-space optimization framework for agent skills that treats the skill document as a trainable external state for a frozen Target Model, utilizing an Optimizer Model to propose bounded edits based on Rollout Batch evidence.
- The framework employs a Validation Gate to ensure only performance-improving edits are accepted, while a Rejected-Edit Buffer and epoch-wise Slow-Update Field provide negative feedback and long-horizon stability without requiring model-weight updates.
- By decoupling the execution harness from the optimization loop, SkillOpt enables the creation of compact, reusable, and inspectable skill artifacts that transfer effectively across different model scales, execution environments, and benchmarks.

---

[From Raw Experience to Skill Consumption: A Systematic Study of Model-Generated Agent Skills](http://arxiv.org/abs/2605.23899)

- SkillLens: introduces a utility-grounded evaluation framework that systematically analyzes the full lifecycle of model-generated agent skills across experience generation, skill extraction, and skill consumption.
- The framework identifies that skill utility is not determined by textual plausibility but by concrete failure mechanism encoding and actionable specificity, which are operationalized into a meta-skill to improve extraction.
- Experimental results across five diverse domains demonstrate that while model-generated skills are generally beneficial, they exhibit non-trivial negative transfer, necessitating a principled, utility-grounded approach to skill development.

---

[CHRONOS: Temporally-Aware Multi-Agent Coordination for Evolving Data Marketplaces](http://arxiv.org/abs/2605.23887)

- CHRONOS: introduces a three-layer architecture for temporal KG marketplaces that integrates T-LEGEND, Event-Conditioned MPV, Temporal Coordinator, and a Fixed-Dimension DP Pipeline to provide end-to-end temporal guarantees.
- The framework utilizes neural-ODE temporal decay for index freshness, BOCPD-conditioned Shapley valuation for fair attribution, and an EXP3-IX bandit agent for coordinated DP budget management.
- CHRONOS achieves a competitive recall/latency/privacy operating point by separating public index construction from private seller-edge updates under a shared differential privacy budget.

---

[Routing Equilibrium in Mixed-Autonomy Traffic Networks with Altruistic Autonomous Agents](http://arxiv.org/abs/2605.23782)

- Mixed-Autonomy Traffic Routing Framework: introduces a variational inequality-based model to characterize equilibrium in traffic networks where self-interested human-driven agents and altruistic autonomous agents coexist.
- The framework establishes existence and uniqueness of aggregated link flows and social costs under BPR travel time functions, providing algebraic conditions to determine whether autonomous agents improve or deteriorate system performance.
- The research demonstrates that under convex travel time functions, decentralized altruistic routing achieves the same equilibrium outcome as centralized fleet management.

---

[Agentic Proving for Program Verification](http://arxiv.org/abs/2605.23772)

- Claude Code (Agentic Proving Framework): introduces an agentic paradigm for program verification that utilizes Claude Opus 4.6, lean-lsp-mcp, lean4-skills, Lean LSP, Mathlib, Benchmark Harness, and a Compiler-in-the-loop Agent to automate specification, implementation, and proof generation.
- The framework leverages iterative compiler feedback and specialized Lean-specific tools to achieve a 98.1% success rate on the CLEVER benchmark for verifiable code generation.
- The research demonstrates that agentic systems effectively identify and fix bugs in both generated code and existing benchmark specifications, highlighting the limitations of current isomorphism-based evaluation methodologies.

---

[PhotoFlow: Agentic 3D Virtual Photography Missions](http://arxiv.org/abs/2605.23771)

- PhotoFlow: introduces a closed-loop agentic framework for language-conditioned virtual photography that utilizes a Director-Reviewer-Reflector architecture to iteratively search for optimal camera states in 3D environments.
- The framework employs a Director to propose candidates, a Reviewer to evaluate them using rule-based and VLM-based signals, and a Reflector to manage search memory and exploration strategies.
- The authors also present VPhotoBench, a benchmark comprising 141 photography missions across 47 Blender scenes to evaluate the spatial reasoning and aesthetic judgment of LLM-centered agents.

---

[Direct Dynamic Retargeting for Humanoid Imitation Learning from Videos](http://arxiv.org/abs/2605.23762)

- DDR (Direct Dynamic Retargeting): introduces a single-stage framework that generates dynamically feasible humanoid trajectories directly from monocular video demonstrations by bypassing intermediate geometric projections.
- The framework utilizes a sampling-based MPC solver with CEM to optimize trajectories within a physics simulator, effectively handling complex contact sequences and mitigating morphological mismatches.
- By providing physically viable references, the approach accelerates the training convergence of RL policies and enables robust zero-shot sim-to-real transfer on humanoid hardware.

---

[LLM-DRIVEN DESIGN OF PHYSICS-CONSTRAINED CONSTITUTIVE MODELS: TWO AGENTS ARE BETTER THAN ONE](http://arxiv.org/abs/2605.23754)

- Creator-Inspector: introduces a multi-agent LLM architecture for constitutive model generation that separates model proposal from critical physical constraint auditing.
- The framework utilizes a Creator agent to design CANN models and an Inspector agent to ensure compliance with nine fundamental physical constraints, significantly improving model reliability.
- This technique-agnostic approach demonstrates that multi-agent collaboration effectively bridges the gap between LLM-driven code generation and the rigorous requirements of continuum mechanics.

---

[SeedER: Seed-and-Expand Retrieval from Knowledge Graphs](http://arxiv.org/abs/2605.23753)

- SeedER: introduces a retrieval framework that decomposes knowledge graph reasoning into an iterative, low-cost expansion process using a learned policy trained with reinforcement learning.
- The framework utilizes a Dense Retriever for initial seeding, followed by a K-hop-with-filtering mechanism to construct a bounded subgraph for efficient exploration by an RL-guided Graph Policy.
- A GNN-based Scoring Head provides final candidate ranking, enabling SeedER to achieve high-recall retrieval on complex compositional queries while maintaining computational efficiency.

---

[MemAudit: Post-hoc Auditing of Poisoned Agent Memory via Causal Attribution and Structural Anomaly Detection](http://arxiv.org/abs/2605.23723)

- MemAudit: introduces a post-hoc causal memory auditing framework for memory-augmented LLMs that combines CMIS (measures causal contribution of memory to harm) and MCG (identifies structurally anomalous memory) to identify and remove malicious records.
- The framework utilizes counterfactual replay to estimate the causal influence of retrieved memories on harmful outputs and constructs a memory consistency graph to detect semantically inconsistent or anomalous memory patterns.
- By fusing these causal and structural signals into a detoxification score, MemAudit enables targeted removal of poisoned memory entries without requiring oracle labels, effectively reducing attack success rates in both QA and reasoning-agent settings.

---

[One Policy, Infinite NPCs: Persona-Traceable Shared RL Policies for Scalable Game Agents](http://arxiv.org/abs/2605.23652)

- PCSP (Persona-Conditioned Shared Policy): introduces a scalable reinforcement learning architecture that conditions a single shared policy on frozen LLM persona embeddings to enable consistent, controllable NPC behavior across large-scale game environments.
- The framework utilizes a low-rank persona projection and an InfoNCE trajectory-consistency objective to ensure that agent behaviors remain traceable to their specific conditioning personas.
- Empirical validation across controlled grid-world benchmarks, multi-agent social dilemmas, and Unreal Engine 5 deployments demonstrates that the method supports real-time inference and zero-shot persona generalization.

---

[Co-ReAct: Rubrics as Step-Level Collaborators for ReAct Agents](http://arxiv.org/abs/2605.23590)

- Co-ReAct: introduces a rubric-guided action selection framework that transforms rubrics from post-hoc evaluators into prescriptive, step-level guidance signals for LLMs during inference.
- The framework utilizes a Rubric Generator trained via GRPO with a listwise Spearman rank-correlation objective to ensure generated rubrics are discriminative and aligned with expert consensus.
- Co-ReAct extends the standard ReAct loop into a five-tuple (Rubric, Reason, Act, Verify, Observe) process, enabling targeted retries when actions fail to satisfy specific rubric criteria.

---

[Push Your Agent: Measuring and Enforcing Quantitative Goal Persistence in Long-Horizon LLM Agents](http://arxiv.org/abs/2605.23574)

- PushBench: introduces a benchmark and evaluation framework to measure Quantitative Goal Persistence (QGP) in long-horizon LLMs by enforcing that agents continue working until an external verifier confirms a specific count of valid units.
- The framework utilizes specialized controllers, including STATEQGP and UNITQGP, to maintain verifier-visible progress state, effectively reducing common failure modes like duplicate submissions, premature stopping, and false completion.
- Experimental results demonstrate that while stronger LLMs and memory mechanisms improve performance, controller-level persistence is essential for maintaining reliable progress in long-horizon tasks with explicit quantitative goals.

---

[Understanding Goal Generalisation in Sequential Reinforcement Learning](http://arxiv.org/abs/2605.23565)

- Latent Policy Gradients (LPG): introduces a method to predict out-of-distribution RL agent behaviour by modelling the evolution of low-dimensional latent variables during multi-stage training.
- The framework captures complex generalisation phenomena such as value persistence and feature inhibition by simulating gradient ascent on modified policy objectives.
- LPG provides an interpretable structure where the learned saliency matrix functions as a similarity metric between features, enabling principled predictions of agent preferences across unseen environments.

---

[ARMS: Automatic Reward Shaping for Sparse-Reward Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2605.23562)

- ARMS (Automatic Reward-shaping in Multi-agent Systems): introduces a self-supervised framework that learns dense shaping signals from sparse environmental rewards through trajectory ranking to improve MARL efficiency.
- The framework alternates between a reinforcement learning phase using shaped rewards and a reward-shaping phase that optimizes reward parameters via a binary classification loss on trajectory pairs.
- ARMS leverages a game-theoretic equilibrium-preservation result to ensure that the learned shaping rewards preserve the set of Nash equilibria in multi-agent settings.

---

[PathNavigate: A Training-Free Pathology Agent with Surprise-Guided Scan and Shared Slide Memory for Whole-Slide Image VQA](http://arxiv.org/abs/2605.23559)

- PathNavigate: introduces a training-free pathology agent that utilizes a scan-search-readout routine to navigate gigapixel slides efficiently.
- The framework employs Shared Online Memory to maintain slide-specific context, Surprise-Guided Scan to identify atypical regions, and Task-Conditioned Search to prioritize evidence based on clinical queries.
- By integrating a frozen Perceptor and Adjudicator, the system achieves high-resolution evidence extraction and accurate VQA without requiring task-specific retraining.

---

[Goal-Conditioned Agents that Learn Everything All at Once](http://arxiv.org/abs/2605.23551)

- LEO: introduces a goal-conditioned reinforcement learning approach that performs efficient, parallel all-goals updates by currying the value network to output Q-values for every goal simultaneously.
- Dual LEO: improves upon LEO by utilizing a LEO network as a teacher to provide coarse, directionally useful guidance to a high-fidelity UVFA student network, effectively mitigating late fusion bottlenecks.
- The framework demonstrates significant speed-ups over naive relabeling and achieves competitive performance on complex, procedurally generated benchmarks like CraftaxGC and continuous control tasks.

---

[LiveFigure: Generating Editable Scientific Illustration with VLM Agents](http://arxiv.org/abs/2605.23527)

- LiveFigure: introduces an agentic framework that emulates the cognitive workflow of expert human designers to automatically synthesize natively editable PowerPoint scientific illustrations.
- The framework decomposes the generation process into three stages: visual planning via prior induction, procedural figure generation via standardized skills and experience enhancement, and targeted refinement via visual diagnostics.
- By leveraging Microsoft PowerPoint as a carrier, LiveFigure produces fully vectorized, editable figures that support fine-grained manual adjustments while maintaining high aesthetic and structural fidelity.

---

[AI Assurance: A Comprehensive Testing Strategy for Enterprise AI Systems](http://arxiv.org/abs/2605.23459)

- AI Assurance Strategy: introduces a comprehensive testing framework for enterprise AI systems that shifts from traditional verification to continuous risk reduction through a five-layer AI Assurance Pyramid.
- The framework utilizes Evaluation Datasets, LLM Judges, Rubrics, Scoring Pipelines, Regression Baselines, Quality Gates, and Human-in-the-Loop to systematically manage probabilistic risks in LLM-based applications.
- This approach treats evaluation as a core engineering discipline, providing operational guidance for RAG systems and agentic workflows to detect silent behavioral degradation and ensure production reliability.

---

[MileStone: A Multi-Objective Compiler Phase Ordering Framework for Graph-based IR-Level Optimization](http://arxiv.org/abs/2605.23435)

- MileStone: introduces a modular framework that models compiler phase ordering as a multi-objective optimization problem using Graph Generator (GG), GNN-based Performance Predictor (GNNPP), RL-based Multi-Objective Explorer (RLMOE), RL-based Database Generator (RLDBG), Evaluator, and Database.
- The framework leverages GNNPP for static performance prediction and RLMOE for adaptive exploration of compiler pass sequences to identify Pareto-optimal trade-offs.
- MileStone utilizes a self-evolving database (RLDBG) to aggregate compiler IRs and performance metrics, enabling efficient supervised training and reducing the need for repeated physical profiling.

---

[Socially fluent AI decouples conversational signals from source identity in online interaction](http://arxiv.org/abs/2605.23426)

- Socially fluent AI framework: introduces a computational social science approach to investigate the limits of human identity judgments in synchronous text-based online interaction using Analytic engines, Predictive modelling, Representational similarity analysis (RSA), BERTopic model, and Signal detection theory (SDT).
- The study demonstrates that while interactional cues reliably encode ground-truth identity, human participants fail to exploit this information, relying instead on subjective impressions.
- This research reveals a systematic dissociation between behavioural diagnosticity and social judgment, rendering established computer-mediated communication heuristics non-diagnostic for identifying AI agents.

---

[When Planning Fails Despite Correct Execution: On Epistemic Calibration for LLM-Based Multi-Agent Systems](http://arxiv.org/abs/2605.23414)

- EPC-AW: introduces a planning-centered workflow that mitigates epistemic miscalibration in LLMs by selecting plans with stable cross-agent evaluations and refining epistemic states through persistent memory constraints.
- The framework utilizes Information-consistency-based Plan Selection (IPS) to identify plans that remain supported under heterogeneous information conditions, effectively reducing reliance on potentially miscalibrated individual feasibility assessments.
- Consistency-guided Epistemic State Refinement (CESR) further enhances reliability by recording cross-round plan discrepancies and inducing constraints that prevent the recurrence of previously identified miscalibration patterns.

---

[Beyond the Half-Approximation: Fair and Efficient Online Class Matching](http://arxiv.org/abs/2605.23408)

- EFTT and Hybrid Ranking: introduces threshold-based algorithms that interpolate between fairness and efficiency in online class matching, resolving the open problem of exceeding the 1/2 utilitarian social welfare barrier.
- The paper provides the first parametric tradeoff between class envy-freeness and utilitarian social welfare, utilizing a threshold parameter to navigate the fairness-efficiency frontier.
- The authors establish tight price-of-fairness bounds and demonstrate that their algorithms outperform previous approaches by balancing fair distribution with efficiency-maximizing strategies.

---

[From Correctness to Preference: A Framework for Personalized Agentic Reinforcement Learning](http://arxiv.org/abs/2605.23382)

- PARPO: introduces a unified Agentic RL framework that embeds personalization into training-time optimization by decoupling generic task-quality rewards from personalized preference rewards using user-specific anchors.
- The framework utilizes a PSGM to organize users, skills, and trajectories into a heterogeneous graph for preference-aligned retrieval, alongside a two-stage reward model that disentangles intrinsic user interests from conformity effects.
- Experiments on ETAPP and SJAgent demonstrate that the framework consistently outperforms memory and RL baselines by stabilizing learning under heterogeneous user reward scales and improving user-contingent policy performance.

---

[Human-in-the-Loop Multi-Agent Ventilator Decision Support with Contextual Bandit Preference Learning](http://arxiv.org/abs/2605.23320)

- VDSS: introduces a human-in-the-loop multi-agent framework that coordinates modular decision components through contract-driven interfaces to provide traceable ventilator support.
- The system integrates a Data Hub, Waveform Analyzer, Detection Agent, Phase Goal Manager, Hold-Adjust Gate, Strategy Selector, Mode Select Agent, Parameter Planner, Safety Check, Reflect Agent, Note Generator, Contextual Bandit, and Long-term Memory to enable coherent, auditable, and personalized clinical decision-making.
- VDSS utilizes a contextual bandit to perform online preference adaptation based on clinician feedback, significantly improving recommendation acceptability and reducing interaction rounds compared to single-model approaches.

---

[DART: Semantic Recoverability for Structured Tool Agents](http://arxiv.org/abs/2605.23311)

- DART (Deterministic Agent Runtime with Transition Guards): introduces a modular runtime that ensures semantic recoverability for LLM-based tool agents by certifying boundaries and validating rollbacks against downstream commitments.
- The framework utilizes Failed-Instance Localization, Recoverable-Boundary Certification, Instance-Aligned Checkpointing, and Admissible Rollback Selection to prevent semantically invalid recoveries in commitment-sensitive execution environments.
- By integrating an Instance Registry and Checkpoint Store via a Sidecar Observer, DART effectively manages dependencies and ensures that local rollbacks do not violate committed downstream work or irreversible effect boundaries.

---

[Parallel Context Compaction for Long-Horizon LLM Agent Serving](http://arxiv.org/abs/2605.23296)

- Parallel Context Compaction: introduces a block-based parallel summarization approach for long-horizon LLM agents that uses Snapshot &amp; partition, Dispatch, and Merge to provide fine-grained control over summary volume while reducing latency.
- The framework utilizes a prefix-aware target-at-end layout with XML markers to enable prefix cache reuse across multiple Workers while maintaining cross-block context dependencies.
- Empirical results demonstrate that this parallel design improves compaction throughput and downstream accuracy by allowing the operator to tune summary volume through block size and prompt engineering.

---

[DepthAgent: Towards Better Universal Depth Estimation via Sample-wise Expert Selection](http://arxiv.org/abs/2605.23281)

- DepthAgent: introduces an agentic framework for universal monocular depth estimation that dynamically coordinates frozen depth experts through a VLM agent to produce sample-wise optimal depth solutions.
- The framework utilizes a multi-reward reinforcement fine-tuning scheme to jointly optimize for scene-aware reasoning, expert selection quality, and inference efficiency.
- By treating existing models as tools, DepthAgent effectively exploits camera-dependent expert complementarity to outperform static fusion baselines across perspective, fisheye, and panoramic domains.

---

[Self-Refining Topology Optimization via an LLM-Based Multi-Agent Framework](http://arxiv.org/abs/2605.23273)

- TopOptAgents: introduces a multi-agent framework for autonomous topology optimization that utilizes iterative self-refinement loops across problem formulation, code execution, and result assessment.
- The framework incorporates Scientist-, Validator-, Planner-, Coder-, Executor-, Reviewer- and Critic-agents to automate the entire design workflow from user requirements to converged structural layouts.
- By employing three distinct refinement loops, the system recovers from specification errors, runtime failures, and design-quality issues that typically hinder single-pass LLM performance in complex engineering tasks.

---

[EvalVerse: Pipeline-Aware and Expert-Calibrated Benchmarking for Professional Cinematic Video Generation](http://arxiv.org/abs/2605.23271)

- EvalVerse: introduces a pipeline-aware and expert-calibrated evaluation framework that systematically digitizes subjective cinematic expertise into computable metrics for professional video generation.
- The framework utilizes a five-step process including Taxonomy Establishment, Dataset Curation, Expert-Machine Calibration, Machine Evaluation Suite, and Versatile Applications to bridge the gap between human aesthetic perception and algorithmic scoring.
- By injecting expert knowledge into VLMs through fine-tuning and Chain-of-Thought reasoning, EvalVerse provides granular diagnostic signals and high-quality reward models for RL-based video generation workflows.

---

[6G Communication Networks Enabling Embodied Agents: Architecture and Prototype](http://arxiv.org/abs/2605.23263)

- 6G-enabled human-robot remote interaction architecture: introduces a hierarchical communication framework that decouples control logic from physical connectivity to support low-latency, high-reliability remote collaboration for embodied agents.
- The architecture integrates a human-intent perception layer, an O-RAN transport layer, an intelligent intermediary layer, and an embodiment layer to facilitate seamless interaction between human operators and remote robotic systems.
- Experimental validation using a prototype system demonstrates that the proposed architecture achieves sub-millisecond latency and stable closed-loop control, confirming its suitability for demanding real-time remote interaction scenarios.

---

[Design and Report Benchmarks for Knowledge Work](http://arxiv.org/abs/2605.23262)

- Knowledge Work Benchmark Reporting Framework: introduces a three-step approach for aligning benchmark scores with broader work-capability claims by defining the Work Activity, specifying the Tested Setting, and evaluating the resulting Work Product.
- The framework utilizes an O*NET Inventory of 18 cross-occupation work activities to provide a standardized vocabulary for classifying tasks and identifying gaps between benchmark proxies and real-world requirements.
- By applying Benchmark Case Analyses to existing suites, the paper demonstrates how explicit reporting of settings and products clarifies the supported claims and limitations of LLM-based agents in professional environments.

---

[Turning Adaptation into Assets: Cross-Domain Bridging for Online Vision-Language Navigation](http://arxiv.org/abs/2605.23257)

- IDEA (Inter-Domain BridgE with Historical Assets): introduces a Test-Time Adaptation framework that reformulates navigation adaptation as the continuous accumulation and composition of reusable knowledge assets.
- The framework utilizes a Fisher-guided weighting scheme to optimize soft prompts for sensitivity-aware alignment, creating triplet-structured assets that capture task-essential priors.
- IDEA constructs a training-free adaptation bridge by projecting target domain statistics onto the convex hull of historical assets, enabling efficient knowledge reuse and bypassing iterative optimization.

---

[Are Frontier LLMs Ready for Cybersecurity? Evidence for Vertical Foundation Models from Dual-Mode Vulnerability Benchmarks](http://arxiv.org/abs/2605.23243)

- Vertical Foundation Models for Cybersecurity: introduces a dual-mode benchmark and specialized models to address the structural limitations of frontier LLMs in cybersecurity tasks.
- The framework utilizes SuperIntel Attack-LLM and SuperIntel Defense-LLM, Methodology-Guided Agents, and an Agentic Reasoning Graph to improve vulnerability detection precision and reliability.
- The research demonstrates that domain-specific methodology and deterministic confirmation, rather than model scale, are the primary levers for achieving reliable cybersecurity automation.

---

[GENSTRAT: Toward a Science of Strategic Reasoning in Large Language Models](http://arxiv.org/abs/2605.23238)

- GENSTRAT: introduces a procedurally generated distribution of two-player zero-sum imperfect-information card games to evaluate LLM strategic reasoning across six distinct complexity axes.
- The framework utilizes a capability-profile methodology and a jaggedness measure to provide deployment-relevant diagnostics beyond simple aggregate performance rankings.
- Evaluation of nine frontier and open-weight LLMs reveals that while larger models generally score higher, they exhibit qualitatively different strategic profiles and varying levels of local performance volatility.

---

[WMAttack: Automated Attack Search for Adversarial Evaluation of World-Model Agents](http://arxiv.org/abs/2605.23220)

- WMAttack: introduces an automated attack-search framework for adversarial evaluation of world-model agents by formulating robustness as a finite-budget search over attack configurations using RGAR and SCAS.
- The framework utilizes RGAR to provide a warm start by retrieving effective historical configurations from representation-similar tasks and SCAS to iteratively refine the attack strategy using multi-dimensional feedback.
- WMAttack consistently discovers stronger attacks than baselines across Atari and DeepMind Control tasks by concentrating evaluation budgets on high-impact adversarial regions.

---

[Foundation Protocol: A Coordination Layer for Agentic Society](http://arxiv.org/abs/2605.23218)

- FP (Foundation Protocol): introduces a graph-native coordination layer for agentic societies that unifies heterogeneous entities through Entity & Trust Plane, Transport & Routing Plane, Interaction & Organization Plane, and Regulation & Oversight Plane.
- The framework utilizes a modular architecture with Profiles & Bridges to integrate existing protocols while maintaining a small, stable core for identity, authority, and accountability.
- It incorporates a Checkpoint Pipeline for policy enforcement and a Contract-and-Settlement Subsystem to provide ledger-agnostic economic attestations and auditable provenance for multi-agent interactions.

---

[CultivAgents: Cultivating Relationship-Centered Multi-Agent Systems for Personalized Gardening](http://arxiv.org/abs/2605.23193)

- CultivAgents: introduces a relationship-centered multi-agent system that coordinates specialized Experience Agent, Environmental Agent, and Ethnobotanical Agent to provide personalized, socio-culturally grounded gardening support.
- The framework utilizes a selector-based coordination mechanism to route user queries to the most relevant agent, ensuring that gardening advice is actionable, locally grounded, and culturally meaningful.
- Evaluations with domain experts and community gardeners demonstrate that the system significantly improves user confidence, motivation, and trust by moving gardening support from generic Q&A to situated, relational guidance.

---

[IntentionNav: A Benchmark for Intent-Driven Object Navigation from Implicit Human Instruction](http://arxiv.org/abs/2605.23187)

- IntentionNav: introduces a diagnostic benchmark for evaluating embodied agents on intent-driven object navigation where goals are specified via implicit human instructions rather than explicit category labels.
- The framework utilizes a modular Reference Agent that integrates VLM-based target inference, open-vocabulary perception via Grounding-DINO, and map-based exploration to navigate simulated indoor environments.
- Diagnostic analysis reveals that while agents frequently reach the target neighborhood, they struggle with terminal localization and visual confirmation, highlighting a significant bottleneck in the intent-to-navigation chain.

---

[Redrawing the AI Map: A Theory of Accountability Boundaries in Agentic Ecosystems](http://arxiv.org/abs/2605.23179)

- Redrawing the AI Map: A Theory of Accountability Boundaries in Agentic Ecosystems introduces a theoretical framework explaining how accountability assets and verification costs determine whether AI-enabled capabilities adopt component-, integrated-, or dual-track boundary strategies.
- The paper identifies rule debt as a latent governance burden arising when decision rules migrate from formal information systems into ungoverned agentic execution environments.
- The research demonstrates that technical decomposability does not necessitate organizational disaggregation, as accountability-bearing cores often require integration with specialized evidence and review infrastructure.

---

[DRIVESPATIAL: A Benchmark for Spatiotemporal Intelligence in VLMs for Autonomous Driving](http://arxiv.org/abs/2605.23176)

- DRIVESPATIAL: introduces a comprehensive benchmark for evaluating spatiotemporal intelligence in VLMs for autonomous driving, utilizing a dynamic multi-relational graph to enforce cross-view and temporal reasoning.
- The framework integrates multi-view observations into a coherent scene representation, testing models on cognitive scene construction, relational understanding, temporal reasoning, and generalization.
- Evaluation of 15 representative VLMs reveals a significant human-model performance gap, identifying scene construction as the primary bottleneck for current autonomous driving models.

---

[Positional Failures in Long-Context LLMs: A Blind Spot in Reasoning Benchmarks](http://arxiv.org/abs/2605.23170)

- CRE: introduces a controlled evaluation framework that systematically varies task position, filler content, and context length to expose positional vulnerabilities in long-context LLMs.
- The framework utilizes diagnostic probes, including middle_twice and middle_dup, to characterize model-dependent failure modes and distinguish positional effects from baseline capability loss.
- Empirical results demonstrate that reasoning performance in LLMs often degrades significantly when target tasks are placed in the middle of long contexts, a phenomenon that remains largely unmeasured in current mainstream benchmarks.

---

[Fast-dDrive: Efficient Block-Diffusion VLM for Autonomous Driving](http://arxiv.org/abs/2605.23163)

- Fast-dDrive: introduces a block-diffusion VLA that leverages a frozen scaffold of structural tokens to enable efficient, causal-ordered inference for autonomous driving.
- The framework utilizes a dual-stream architecture with an MDM head for parallel drafting and an AR head for sequential verification, significantly reducing latency compared to standard autoregressive models.
- By employing section-aware training and shared-prefix multi-trajectory rollouts, the system achieves state-of-the-art planning accuracy and throughput on edge hardware.

---

[Infra-Bayesian Reinforcement Learning Agents Outperform Classical RL For Worst-Case Robustness](http://arxiv.org/abs/2605.23146)

- IBRL (Infra-Bayesian Reinforcement Learning): introduces a proof-of-concept architecture that replaces standard Bayesian posteriors with sets of affine evaluators to perform robust decision-making under Knightian uncertainty.
- The framework utilizes a maximin decision rule to evaluate policies based on their worst-case outcomes, ensuring safety and robustness in non-realizable and policy-dependent environments.
- By maintaining a finite set of extremal minimal points, the architecture achieves computational tractability while recovering standard Bayesian behavior in degenerate cases.

---

[Inductive Deductive Synthesis: Enabling AI to Generate Formally Verified Systems](http://arxiv.org/abs/2605.23109)

- IDS (Inductive Deductive Synthesis): introduces an agentic system that jointly and incrementally synthesizes implementations and machine-checked proofs by leveraging the Rocq type-checker as a feedback oracle.
- The framework utilizes a two-level architecture where DSA (Deductive Synthesis Agent) constructs code and proofs, while ISA (Inductive Synthesis Agent) intervenes to propose new strategies upon encountering tactical stalls or strategic dead-ends.
- By integrating performance feedback from a benchmark harness into the synthesis loop, IDS optimizes data-store representations to achieve state-of-the-art performance in verified distributed systems.

---

[Boiling the Frog: A Multi-Turn Benchmark for Agentic Safety](http://arxiv.org/abs/2605.22643)

- Boiling the Frog: introduces a multi-turn benchmark that evaluates whether LLMs deployed as agents in corporate environments are susceptible to incremental attacks that result in unsafe persistent artifact states.
- The benchmark utilizes a Model (reasoning engine), Harness (control layer), and Environment (stateful world) to measure operational risk through a three-level taxonomy grounded in regulatory frameworks.
- Results across a nine-model panel indicate an aggregate strict attack success rate of 44.4%, with significant vulnerability observed in GPAI Code of Practice loss-of-control scenarios.

---

[Emergence of agriculture in an artificial society of reinforcement learning agents](http://arxiv.org/abs/2605.22256)

- MARL framework: introduces an artificial society of reinforcement learning agents to identify universal principles underlying the evolutionary transition to agriculture through coupled dynamics of learning and environmental modification.
- The system utilizes a Transformer-based action policy and PPO to enable agents to perform eco-engineering, such as selective weed removal and watering, which stabilizes agricultural practices.
- A cloning mechanism acts as a social learning surrogate, functioning as a firewall against cheaters and enabling the propagation of successful agricultural strategies in larger populations.

---

[GenEvolve: Self-Evolving Image Generation Agents via Tool-Orchestrated Visual Experience Distillation](http://arxiv.org/abs/2605.21605)

- GenEvolve: introduces a self-evolving framework that models image generation as a tool-orchestrated trajectory, where an agent learns to coordinate internal knowledge with external tools to synthesize prompt-reference programs.
- The framework utilizes Visual Experience Distillation to compare multiple generation trajectories, abstracting best-worst differences into structured experience that provides dense token-level supervision to a privileged teacher branch.
- By combining trajectory-level GRPO optimization with token-level distillation, GenEvolve enables agents to internalize complex orchestration behaviors, achieving state-of-the-art performance on knowledge-intensive and quality-sensitive generation tasks.

---

#### 21st May 2026


[MOSS: Self-Evolution through Source-Level Rewriting in Autonomous Agent Systems](http://arxiv.org/abs/2605.22794)

- MOSS: introduces a source-level self-rewriting framework for production agentic systems that extends evolution beyond text-mutable artifacts to the agent harness itself.
- The system utilizes a deterministic multi-stage pipeline, where a host-resident daemon manages code modification via an external coding-agent CLI and verifies candidates through ephemeral trial workers.
- By anchoring evolution to production-failure evidence and enabling in-place container swaps, MOSS allows autonomous agents to improve their core logic without human intervention.

---


[Detecting Offensive Cyber Agents: A Detection-in-Depth Approach](http://arxiv.org/abs/2605.21956)

- Detection-in-Depth: introduces a strategic framework for identifying autonomous cyber agents by layering technical mechanisms across access, activity, and ecosystem levels to overcome the limitations of single-point detection.
- The framework utilizes Identity Mechanisms (Access layer for verifying digital actors), Environmental Detection Mechanisms (Activity layer for observing network behavior), and Information Sharing and Analysis Infrastructure (Ecosystem layer for aggregating threat signals) to address the challenges of adaptive, disaggregated, and horizontally scaled cyberattacks.
- The report emphasizes that effective detection requires Discovery (Generating novel signals for agentic activity), Analysis (Interpreting signals to confirm malicious activity), and Coordination (Uniting organizations to pool dispersed signals) to maintain situational awareness against evolving LLM-enabled threats.

---

[Philosophical Dispositions as Behavioral Constraints for AI-Assisted Code Review: An Empirical Study](http://arxiv.org/abs/2605.23108)

- Philosophical Dispositions Framework: introduces a system that constrains LLM code review behavior through Dispositions, Roles, Hamartia, Synthesis, and Diff Input to generate structurally and logically distinct analytical findings.
- The framework utilizes four specific philosophical lenses—Cynic, Skeptic, Nyaya, and Confucian—to perform sequential, multi-perspective code analysis that identifies issues often missed by generic LLM prompting.
- Empirical evaluation across 50 pull requests demonstrates that the framework achieves 46% convergence with human reviewers and provides 75% unique findings, functioning as a complementary analytical layer for under-reviewed code.

---

[SVR-MAD: A Bayesian-Inspired Framework for Posterior-Guided Multi-Agent Debate](http://arxiv.org/abs/2605.23099)

- SVR-MAD: introduces a Bayesian-inspired framework that incrementally constructs a communication graph by using posterior debate outcomes to estimate agent reliability.
- The framework utilizes Survival Rate (SVR) as a robust signal to prioritize agents who retain their beliefs after peer challenges, effectively reducing token costs and improving accuracy.
- SVR-MAD outperforms existing multi-agent debate baselines by dynamically updating correctness scores and terminating debates early once a high-confidence threshold is reached.

---

[What Training Data Teaches RL Memory Agents: An Empirical Study of Curriculum Effects in Memory-Augmented QA](http://arxiv.org/abs/2605.23067)

- RL-based memory agent framework: introduces a controlled empirical study on how training curriculum composition influences the specialization of LLMs in memory-augmented question answering tasks.
- The framework utilizes Qwen-2.5-7B-Instruct with LoRA and GRPO to evaluate how different data sources shape the Answer Agent's ability to perform retrieval and reasoning over a Memory Bank.
- The study demonstrates that curriculum composition acts as a lever for skill specialization, while also identifying critical engineering requirements for single-GPU training, including memory bank noise filtering and the use of continuous reward functions.

---

[A measurement substrate for agentic Kubernetes operations](http://arxiv.org/abs/2605.23058)

- Agent-breakage: introduces a closed-loop measurement framework that injects faults into Kubernetes clusters to generate falsifiable performance data for LLM-based operations agents.
- The framework utilizes a Runner, Injectors, Detectors, Scorer, Experience base, Speculative-execution controller, and Synthetic approver to distinguish between framework errors and agent reasoning errors.
- By employing pre-registered decision matrices and controlled experimental arms, the system enables rigorous evaluation of retrieval-augmented agents while mitigating common methodological biases.

---

[DRL-Driven Edge-Aware Utility Optimization for Multi-Slice 6G Networks](http://arxiv.org/abs/2605.23056)

- DQN-based framework: introduces a centralized O-RAN architecture integrated with edge computing and caching to optimize resource allocation across eMBB, URLLC, and MBRLLC network slices.
- The system utilizes a DQN agent within an xApp to perform real-time, slice-aware decision-making based on channel quality, cache status, and user demand profiles.
- By deploying edge servers at O-DUs, the framework minimizes latency and backhaul congestion, effectively supporting high-throughput and low-latency requirements for immersive VR applications.

---

[HawkesLLM: Semantic Uncertainty Propagation in Agentic Text Simulation](http://arxiv.org/abs/2605.23043)

- HawkesLLM: introduces a framework that separates temporal influence modeling from text generation to track semantic uncertainty propagation in iterative agentic text simulations.
- The framework utilizes a multivariate Hawkes process to determine event timing and select weighted predecessor memory for the LLM, ensuring that generated trajectories remain semantically aligned with local references.
- By decomposing uncertainty into global and local drift, the approach demonstrates that agentic systems can accumulate global uncertainty while maintaining local stability relative to their immediate prompt memory.

---

[PIMbot: A Self-Adaptive Attack Framework for Adversarial Manipulation of Multi-Robot Reinforcement Learning](http://arxiv.org/abs/2605.23027)

- PIMbot: introduces a framework that manipulates multi-robot reinforcement learning outcomes by combining Incentive Manipulation, Policy Manipulation, and an Adaptive Multi-Objective Controller to steer group dynamics.
- The framework utilizes an Adaptive Multi-Objective Controller to dynamically balance between stealthy self-maximization and active team disruption strategies based on observed agent behaviors.
- PIMbot serves as a rigorous stress-test tool for multi-robot systems, validated through simulations in Gazebo and real-world deployment on NVIDIA Jetson Orin Nano embedded platforms.

---

[The Deterministic Horizon: Impossibility Results as Design Specifications for Trustworthy AI Systems](http://arxiv.org/abs/2605.23024)

- Deterministic Horizon framework: introduces a methodology that converts fundamental AI impossibility results into computable design specifications with quantified violation costs.
- The framework identifies critical thresholds across computation, adaptation, grounding, and trust, providing constructive engineering rules for system design.
- The research demonstrates that reliability is a composition property, where individual components are necessary but jointly insufficient without integrated design specifications.

---

[How to Steer Your Multi-Agent System: Human-LLM Collaborative Planning](http://arxiv.org/abs/2605.23023)

- AMBIPOM (Agent-aware Mixed-initiative Block-level Interactive Planning for Orchestrated Multi-agent systems): introduces a human-LLM co-planning framework that enables process-level supervision through a dual-panel interface supporting semantic and structural interactions, including LLM-based planner, execution agents, chat panel, plan DAG visualization, direct manipulation, targeted semantic feedback, and high-level structural operations.
- The framework formalizes a design space for human-LLM co-planning along three axes: mode (structural vs. semantic), scope (global vs. targeted), and level (low-level vs. high-level edits).
- The research evaluates the system through a user study and a controlled benchmark, revealing that users dynamically construct hybrid workflows to navigate effort-control-risk trade-offs during multi-agent plan refinement.

---

[PACE: Two-Timescale Self-Evolution for Small Language Model Agents](http://arxiv.org/abs/2605.23019)

- PACE (Prompt And Control Logic Evolution): introduces a two-timescale agentic framework that coordinates low-risk prompt refinement with higher-risk, validated control-logic updates for frozen SLMs.
- The framework utilizes a controller to perform prompt evolution until performance saturates, subsequently invoking constrained structural modifications that are committed only after passing empirical validation.
- By separating proposal from validation, PACE enables frozen SLMs to autonomously discover and implement task-appropriate inference strategies without requiring model weight updates or frontier-model teachers.

---

[Whose Good, Whose Place? The Moral Geography of Agentic AI for Social Good](http://arxiv.org/abs/2605.22995)

- Agentic AI for Social Good: introduces a structured survey of 112 papers to analyze how research frames social-good claims through Multi-agent systems, LLM agents, Multi-agent LLMs, Simulation agents, Human–AI teams, Single-agent systems, Embodied agents, and Agentic AI workflows.
- The paper identifies a moral-geographic asymmetry where institutional and social-policy SDGs are treated as universal, while health and ecological domains are more frequently situated in specific geographic contexts.
- The authors propose an accountability agenda and a minimal reporting standard to address the field-wide gap between conceptual proposals and real-world deployment or evaluation.

---

[A Proactive Multi-Agent Dialogue Framework for Assessing Social Language Disorder Traits in Autism](http://arxiv.org/abs/2605.22993)

- TPA (Think, Plan, Ask): introduces a proactive multi-agent dialogue framework that improves diagnostic efficiency in autism language assessment by utilizing a TPA Selector (executes proactive reasoning cycle), Doctor Agent (generates clinical questions), Patient Agent (simulates patient responses), Trait Detector (identifies SLD traits), and Clinical Snippet Bank (supplies calibration and anchors).
- The framework replaces reactive question generation with an explicit Think-Plan-Ask reasoning cycle, enabling the system to identify unobserved diagnostic gaps and select targeted strategies to surface latent Social Language Disorder traits.
- By grounding the Patient Agent in real clinical data and employing a modular architecture, TPA achieves superior diagnostic coverage and efficiency compared to existing dialogue planning baselines.

---

[Beyond Zero: Enterprise Security for the AI Era](http://arxiv.org/abs/2605.22985)

- Beyond Zero: introduces a security architecture that shifts the trust boundary from the application level to individual actions by integrating Autonomous Governance, Event Intake, Reasoning Engine, and Challenge Infrastructure.
- The framework utilizes a hierarchical AI-driven Reasoning Engine to perform real-time, risk-based authorization decisions at machine speed for both human users and AI agents.
- By combining static policy enforcement with dynamic, context-aware evaluation, the architecture functions as an adaptive immune system for enterprise data security.

---

[MARGIN: Runtime Confidence Calibration for Multi-Agent Foundation Model Coordination](http://arxiv.org/abs/2605.22949)

- MARGIN: introduces an online confidence calibration method for multi-agent systems that learns per-agent, per-confidence-band calibration factors from the task stream without requiring model access or held-out data.
- The framework utilizes symmetric EWMA updates to adapt to distribution shifts and Bayesian shrinkage to stabilize estimates during cold-start periods.
- MARGIN corrects confidence inversion in LLMs, where raw confidence is often anti-correlated with accuracy on hard tasks, by restoring the reliability of confidence-weighted selection.

---

[AwareVLN: Reasoning with Self-awareness for Vision-Language Navigation](http://arxiv.org/abs/2605.22816)

- AwareVLN: introduces a unified vision-language framework that integrates sparse, self-aware reasoning with end-to-end action prediction to improve navigation robustness and explainability.
- The framework utilizes a structural reasoning module that triggers analysis at key navigation nodes, such as subtask boundaries or path deviations, to synthesize past observations and guide future decisions.
- An automatic data engine leverages a general VLM to generate high-quality, structured reasoning supervision, enabling the agent to learn task progress analysis and error recovery without manual annotations.

---

[Remember to be Curious: Episodic Context and Persistent Worlds for 3D Exploration](http://arxiv.org/abs/2605.22814)

- Remember to be Curious: introduces an end-to-end curiosity-driven exploration framework that couples a persistent 3D Gaussian Splatting world model with a transformer-based agent architecture to enable long-horizon navigation.
- The agent utilizes an episodic memory module to maintain internal representations of past observations, allowing it to navigate novel regions without relying on explicit geometric maps at deployment.
- By deriving intrinsic curiosity rewards from the prediction error of a persistent 3D reconstruction, the framework achieves efficient exploration in sparse-reward environments and generalizes zero-shot to unseen AI-generated worlds.

---

[EVE-Agent: Evidence-Verifiable Self-Evolving Agents](http://arxiv.org/abs/2605.22905)

- EVE-Agent: introduces a data-free self-evolving framework that enforces evidence verifiability by rewarding the Proposer for generating source-grounded spans that causally improve the Solver's answer accuracy.
- The framework utilizes an Evidence Verifier to compute a marginal accuracy gain signal, ensuring that generated training examples are auditable and grounded in verifiable source text.
- EVE-Agent improves evidence-grounded correctness across multiple benchmarks by training the Solver to produce both accurate answers and supporting evidence spans without requiring human-labeled data.

---

[DeltaBox: Scaling Stateful AI Agents with Millisecond-Level Sandbox Checkpoint/Rollback](http://arxiv.org/abs/2605.22781)

- DeltaBox: introduces an OS-level rollbackable sandbox that achieves millisecond-level checkpoint/restore by tracking only incremental state changes between consecutive steps using StateManager, DeltaFS, DeltaCR, Base Storage, Network Proxy Daemon (NPD), Async-warm thread, and Template Pool.
- The framework utilizes DeltaFS for dynamic, unmount-free overlay filesystem layer switching and DeltaCR for process-level memory checkpointing via incremental CRIU dumps and warm-template forking.
- DeltaBox enables efficient state exploration for LLMs by masking checkpoint latency within LLM inference windows and providing near-constant time arbitrary rollback.

---

[Deep Reinforcement Learning for Flexible Job Shop Scheduling with Random Job Arrivals](http://arxiv.org/abs/2605.22773)

- DRL framework: introduces an event-based Deep Reinforcement Learning approach to solve Flexible Job Shop Scheduling Problems with random job arrivals by utilizing a PPO Agent, Actor Network, Critic Network, Rollout Buffer, Environment, Gantt Chart, and Dispatching Rules.
- The framework models the scheduling task as a Markov Decision Process where the agent selects from a set of established dispatching rules to minimize the total makespan.
- Experimental results demonstrate that the proposed DRL approach outperforms heuristic baselines in heterogeneous shop floor environments while maintaining computational efficiency.

---

[Advancing Mathematics Research with AI-Driven Formal Proof Search](http://arxiv.org/abs/2605.22763)

- AlphaProof Nexus: introduces a framework for LLM-aided formal proof generation that coordinates prover subagents, rater subagents, and the Lean compiler to autonomously resolve open mathematical problems.
- The architecture utilizes a population-based evolutionary approach where prover subagents refine proof sketches, while rater subagents use LLM-based Elo matchmaking to guide the search process.
- AlphaProof Nexus integrates formal verification with LLM reasoning to solve complex conjectures in combinatorics, optimization, and algebraic geometry, demonstrating the potential for AI-driven mathematical discovery.

---

[Towards a General Intelligence and Interface for Wearable Health Data](http://arxiv.org/abs/2605.22759)

- SensorFM: introduces a foundation model for wearable health data pretrained on over one trillion minutes of sensor signals to learn universal physiological representations.
- The framework utilizes a masked autoencoder architecture to enable label-efficient few-shot learning and generative capabilities for data imputation and forecasting.
- An agentic classroom of LLM agents autonomously optimizes downstream prediction heads, while the model integrates into a Personal Health Agent to provide context-aware, safe, and personalized health summaries.

---

[Superhuman Safe and Agile Racing through Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2605.22748)

- MARL framework: introduces a multi-agent reinforcement learning system that achieves superhuman performance in high-speed quadrotor racing by training agents to navigate complex aerodynamic interactions and strategic maneuvers.
- The architecture utilizes a Perceiver-based attention encoder to process a variable number of competitors, enabling robust, permutation-invariant representation of the multi-agent environment.
- By training against a diverse league of opponents and incorporating a particle-based downwash model, the agents develop anticipatory behaviors that generalize to safe, high-speed coordination in shared physical spaces.

---

[ChronoMedKG: A Temporally-Grounded Biomedical Knowledge Graph and Benchmark for Clinical Reasoning](http://arxiv.org/abs/2605.22734)

- ChronoMedKG: introduces a temporally-grounded biomedical knowledge graph constructed via a disease-autonomous multi-agent pipeline that utilizes Disease Profiler, Evidence Harvester, Knowledge Extractor, and Quality Controller to extract consensus-validated triples from literature.
- The framework leverages multiple LLMs (DeepSeek V3, GPT-4o-mini, Claude 3 Haiku) to independently extract knowledge, ensuring high-fidelity temporal grounding through multi-model consensus and credibility filtering.
- ChronoMedKG includes the ChronoTQA benchmark, which evaluates temporal clinical reasoning across eight task types, demonstrating that retrieval-augmented generation significantly improves LLM performance on long-tail temporal queries.

---

[HarnessAPI: A Skill-First Framework for Unified Streaming APIs and MCP Tools](http://arxiv.org/abs/2605.22733)

- HarnessAPI: introduces a skill-first architecture that derives both HTTP endpoints and MCP tools from a single typed skill folder to eliminate dual-maintenance overhead.
- The framework utilizes dynamic code generation and synthetic namespaces to ensure schema consistency across transport layers while reducing boilerplate code by 74%.
- HarnessAPI supports dual-mode content negotiation, allowing the same handler to serve streaming SSE responses for interactive sessions and JSON for batch or agent-based tool invocation.

---

[Beyond Acoustic Emotion Recognition: Multimodal Pathos Analysis in Political Speech Using LLM-Based and Acoustic Emotion Models](http://arxiv.org/abs/2605.22732)

- TRUST Multimodal Pipeline: introduces a comparative framework evaluating LLM-based multimodal analysis against acoustic SER models for Pathos detection in political speech.
- The study demonstrates that Gemini 2.5 Flash correlates significantly with TRUST-Pathos scores, whereas acoustic emotion2vec features show negligible association with political-rhetorical valence.
- The research identifies structural limitations in standard SER benchmarks like EMO-DB and highlights the necessity of semantic-pragmatic understanding for capturing complex rhetorical strategies like irony and sarcasm.

---

[Self-Evolving Multi-Agent Systems via Decentralized Memory](http://arxiv.org/abs/2605.22721)

- DECENTMEM: introduces a decentralized memory framework for LLMs in multi-agent systems, utilizing a dual-pool memory structure with an Exploitation Pool, Exploration Pool, Online Router, and LLM-as-a-judge to enable continuous self-evolution.
- The framework replaces centralized repositories with agent-private memory, allowing each agent to maintain its own Exploitation Pool and Exploration Pool, which are dynamically reweighted by an Online Router based on feedback from an LLM-as-a-judge.
- DECENTMEM guarantees global reachability of the solution space and achieves O(log T) cumulative regret, demonstrating improved accuracy and reduced token usage across various MAS frameworks and LLM backbones.

---

[Abstraction for Offline Goal-Conditioned Reinforcement Learning](http://arxiv.org/abs/2605.22711)

- ARL (Abstractive Reinforcement Learning): introduces a hierarchical framework for offline goal-conditioned reinforcement learning that leverages relativised options and distinct representations to enable experience reuse across similar state-space contexts.
- The framework utilizes high-level policy-, low-level policy-, high-level value- and low-level value-components to decouple decision processes and improve robustness in regions with low-quality offline data.
- ARL incorporates inductive biases such as action similarity and translational invariance to facilitate effective learning without requiring additional hyperparameters.

---

[WorkstreamBench: Evaluating LLM Agents on End-to-End Spreadsheet Tasks in Finance](http://arxiv.org/abs/2605.22664)

- WorkstreamBench: introduces a benchmark for evaluating LLM agents on complex, end-to-end financial spreadsheet tasks, utilizing a multi-layered architecture comprising User Layer, Orchestration Layer, Execution Layer, MCP Layer, and Storage Layer.
- The framework employs an LLM-as-judge pipeline to evaluate spreadsheet quality across three core dimensions: Accuracy, Formula, and Format, addressing the limitations of existing atomic-task benchmarks.
- Experimental results demonstrate that while LLMs can complete simpler tasks, they struggle with professional-grade financial modeling, exhibiting significant performance degradation as task complexity and interdependency increase.

---

[Claw AI Lab: An Autonomous Multi-Agent Research Team](http://arxiv.org/abs/2605.22662)

- Claw AI Lab: introduces a lab-native autonomous research platform that organizes scientific workflows into five connected layers: Idea Layer, Planning Layer, Coding Layer, Experiment Layer, and Writing Layer.
- The system utilizes a Claw-Code Harness to integrate local codebases and datasets into an agentic loop, ensuring experimental reliability and result integrity.
- By providing a unified dashboard for real-time monitoring and human intervention, the platform enables a more interactive and steerable approach to autonomous scientific research.

---

[Spreadsheet-RL: Advancing Large Language Model Agents on Realistic Spreadsheet Tasks via Reinforcement Learning](http://arxiv.org/abs/2605.22642)

- Spreadsheet-RL: introduces a reinforcement learning fine-tuning framework that trains specialized LLM agents to perform complex, multi-step spreadsheet workflows within a realistic Microsoft Excel environment.
- The framework utilizes a Spreadsheet Data Agent for scalable task construction, a Spreadsheet Gym for interactive environment simulation, and a Spreadsheet-Native Tool Harness to guide agent reasoning and tool use.
- By applying GRPO with outcome-based rewards, the approach significantly improves the performance of open-source LLMs on both general and domain-specific spreadsheet benchmarks.

---

[Contractual Skills: A GovernSpec Design Framework for Enterprise AI Agents](http://arxiv.org/abs/2605.22634)

- Contractual Skills Framework: introduces a design pattern for organizing agent skills as explicit task contracts to improve governance, inspectability, and testability in enterprise workflows.
- The framework utilizes a field-based model including inputs, permissions, human gates, evidence, output, verification, and handoff to define clear boundaries for LLM-based agents.
- Experimental results demonstrate that contractual skills improve output structure and consistency across multiple LLMs while reducing high-risk tool attempts compared to informal skill formats.

---

[Agentic CLEAR: Automating Multi-Level Evaluation of LLM Agents](http://arxiv.org/abs/2605.22608)

- Agentic CLEAR: introduces an automatic, dynamic, and multi-level evaluation framework that generates textual diagnostics for LLM agents without requiring predefined error taxonomies or hand-crafted rubrics.
- The framework utilizes an LLM Judge to perform step-wise, trace-wise, and rubric-based assessments, which are then processed by the CLEAR Aggregator to identify recurring failure patterns at the system and node levels.
- Agentic CLEAR provides an interactive UI for deep-dive trace analysis and demonstrates strong alignment with human-annotated error taxonomies while predicting task success rates across diverse agentic benchmarks.

---

[Agentic-VLA: Efficient Online Adaptation for Vision-Language-Action Models](http://arxiv.org/abs/2605.22896)

- Agentic-VLA: introduces an agentic training framework that enables VLAs to adapt online through Experience Memory, Adaptive Reward Synthesis, and Language-Guided Exploration.
- The framework employs a Task Decomposer and Capability Tracker to generate dense, curriculum-based rewards, while a VLM-based Exploration Critic provides structured guidance to improve sample efficiency.
- By utilizing a Memory Bank for warm-start initialization and GRPO for policy optimization, the system achieves significant improvements in long-horizon tasks, one-shot learning, and cross-task transfer.

---

[Think Thrice Before You Speak: Dual knowledge-enhanced Theory-of-Mind Reasoning for Persuasive Agents](http://arxiv.org/abs/2605.22602)

- TTBYS: introduces a knowledge-enhanced stepwise reasoning framework that leverages explicit and implicit prior experiences to improve LLMs' inference of desires, beliefs, and strategies in persuasive dialogues.
- The framework utilizes a reverse mental state inference procedure grounded in the BDI model, where the agent iteratively infers the persuadee's intention, desire, and belief from dialogue history to guide strategy selection.
- TTBYS incorporates a ToM-PD Experience knowledge base and a dialogue summary mechanism to mitigate LLM limitations in mental state inference, achieving superior performance across diverse open-source backbones.

---

[VGenST-Bench: A Benchmark for Spatio-Temporal Reasoning via Active Video Synthesis](http://arxiv.org/abs/2605.22570)

- VGenST-Bench: introduces a video benchmark for spatio-temporal reasoning in LLMs that utilizes a multi-agent pipeline to actively synthesize controlled evaluation scenarios, incorporating Task Selector, Scene Graph Agent, Scenario Agent, Video Agent, QA Agent, Human Quality Control, and Reformatter.
- The framework organizes evaluation into a 3x2x2 taxonomy of spatial scale, perspective, and scene dynamics, paired with a three-level cognitive hierarchy spanning visual perception, scene understanding, and spatio-temporal reasoning.
- By shifting from passive curation to active synthesis, the benchmark enables fine-grained diagnosis of LLM reasoning capabilities while mitigating data contamination and shortcut exploitation.

---

[Measuring Security Without Fooling Ourselves: Why Benchmarking Agents Is Hard](http://arxiv.org/abs/2605.22568)

- Security Benchmarking Frameworks: introduces a critical analysis of current evaluation methodologies for LLM agents, identifying that benchmark vulnerabilities, temporal staleness, and runtime uncertainty undermine the integrity of security assessments.
- The paper argues that security benchmarks must be treated as adversarially exposed systems, requiring robust outer protections and continuous validation to prevent agents from exploiting the evaluation environment itself.
- It proposes shifting toward dynamic, generative, and live evaluation approaches, complemented by benchmark introspection, to ensure that LLM agent performance metrics accurately reflect true security capabilities rather than benchmark-specific shortcuts.

---

[SynAE: A Framework for Measuring the Quality of Synthetic Data for Tool-Calling Agent Evaluations](http://arxiv.org/abs/2605.22564)

- SynAE: introduces a multi-axis evaluation framework for assessing the quality of synthetic datasets used in multi-turn tool-calling agent evaluations.
- The framework quantifies synthetic data quality across three pillars: validity (task completion), fidelity (similarity to real data), and diversity (entropy-based coverage).
- SynAE functions as a plug-in component for agentic workflows, enabling automated diagnostic feedback to improve synthetic benchmark generation strategies.

---

[TERMINALWORLD: Benchmarking Agents on Real-World Terminal Tasks](http://arxiv.org/abs/2605.22535)

- TERMINALWORLD: introduces a scalable data engine that automatically reverse-engineers high-fidelity evaluation tasks from in-the-wild terminal recordings to benchmark LLM-based agents.
- The framework utilizes an LLM agent to synthesize task instructions, extract reference solutions, reproduce execution environments via Docker, and generate robust test suites through trial-based refinement.
- Comprehensive benchmarking reveals that current LLMs struggle with real-world terminal workflows, exhibiting an efficiency paradox where increased compute does not correlate with higher success rates.

---

[Why Are Agentic Pull Requests Merged or Rejected? An Empirical Study](http://arxiv.org/abs/2605.22534)

- Agentic-PRs Analysis Framework: introduces a decision-oriented empirical methodology to disentangle agentic failures from workflow-driven outcomes by analyzing interaction artifacts in Agentic-PRs.
- The framework utilizes a stratified sampling strategy and manual qualitative coding to classify rejection reasons and interaction patterns in merged PRs, involving Data Collection, Initial Filtering, Refined Dataset Definition, Stratified Sampling Strategy, Manual Qualitative Coding Process, Independent Coders, and Iterative Coding Scheme.
- This study demonstrates that PR outcomes alone are insufficient for evaluating LLM-based coding agents, as a significant portion of rejections and merges are influenced by repository-specific workflows and human reviewer interventions.

---

[“Refactoring Runaway”: Understanding and Mitigating Tangled Refactorings in Coding Agents for Issue Resolution](http://arxiv.org/abs/2605.22526)

- RefUntangle (Refactoring-aware refinement approach): introduces a two-stage pipeline that assesses and selectively transforms tangled refactorings in LLM-generated patches to improve compilability and functional correctness.
- The framework utilizes a Refactoring Assessment Component to categorize refactorings as KEEP, REMOVE, or FIX based on necessity and safety, followed by a Patch Refinement Component that applies these actions using an LLM.
- Empirical analysis reveals that while coding agents perform tangled refactorings less frequently than humans, these unintentional structural edits significantly reduce patch compilability, a limitation effectively mitigated by the RefUntangle approach.

---

[SCRIPT: Scalable Diffusion Policy with Multi-stage Training for Language-driven Physics-Based Humanoid Control](http://arxiv.org/abs/2605.22894)

- SCRIPT: introduces a scalable diffusion policy framework for language-driven physics-based humanoid control, utilizing JAST-DiT to couple actions, physical states, and text through joint attention.
- The framework incorporates a nonlinear history conditioning mechanism for stable autoregressive rollouts and an RLHR post-training stage to optimize semantic alignment and physical stability.
- SCRIPT demonstrates robust scalability on the 1200-hour MotionMillion dataset, achieving state-of-the-art performance in instruction following and physical realism.

---

[Search-E1: Self-Distillation Drives Self-Evolution in Search-Augmented Reasoning](http://arxiv.org/abs/2605.22511)

- Search-E1: introduces a self-evolution pipeline for search-augmented reasoning that alternates GRPO (Group Relative Policy Optimization) with OFSD (Offline Self-Distillation) to provide dense per-step supervision.
- The framework utilizes a Policy Model (LLM generating reasoning trajectories) and a Search Engine (Fixed external retrieval system) to generate rollouts, which are then refined using an EM Reward (Exact-match outcome reward) and a KL Loss (Token-level forward KL objective) between a Student Policy (Policy conditioned on standard prompt) and a Teacher Policy (Policy conditioned on privileged reference).
- By leveraging a Reference Model (Correct trajectory reference) derived from the policy's own rollout pool, Search-E1 achieves superior reasoning performance on multi-hop QA benchmarks without requiring external teachers or auxiliary modules.

---

[Reflecti-Mate: A Conversational Agent for Adaptive Decision-Making Support Through System 1 and System 2 Thinking](http://arxiv.org/abs/2605.22509)

- Reflecti-Mate: introduces a conversational agent that utilizes a Reflection Profile and an Adaptation Algorithm to provide personalized support for integrative pre-decisional reflection.
- The framework employs an exploration-exploitation tradeoff to dynamically balance broadening user reflection and deepening existing thoughts using an LLM.
- By modeling user thought patterns across internal, external, and experiential categories, the agent facilitates multi-modal reflection while respecting individual reasoning styles.

---

[Towards Direct Evaluation of Harness Optimizers via Priority Ranking](http://arxiv.org/abs/2605.22505)

- SHOR: introduces a low-cost, non-iterative evaluation design for harness optimizers that uses priority ranking of harness components to quantify optimizer ability at the step level.
- The framework evaluates optimizers by having them rank components (Prompt, Memory, Tool, Workflow) based on their potential to improve or hinder agent performance, bypassing expensive rollouts.
- Performance in priority ranking correlates with the optimizer's actual ability to improve agents in multi-step harness optimization, providing a reliable and efficient predictor for optimizer effectiveness.

---

[LACO: Adaptive Latent Communication for Collaborative Driving](http://arxiv.org/abs/2605.22504)

- LACO: introduces a training-free latent communication paradigm for collaborative driving that replaces autoregressive language exchange with selective transmission of internal KV cache representations.
- The framework utilizes ILD to crystallize driving intent, CHSA to prune redundant information, and SSKD to distill shallow-layer representations, effectively mitigating agent identity confusion.
- By enabling asymmetric collaborative inference, LACO allows ego-centric agents to integrate global context from collaborators while maintaining independent control stability and reducing communication latency.

---

[Compiling Agentic Workflows into LLM Weights: Near-Frontier Quality at Two Orders of Magnitude Less Cost](http://arxiv.org/abs/2605.22502)

- Subterranean Agent: introduces a compilation-based architecture that embeds procedural workflows directly into LLM weights to eliminate the need for external orchestrators.
- The framework replaces runtime orchestration with internalized procedural knowledge, achieving near-frontier quality while significantly reducing inference costs and latency.
- By training on synthetic conversations generated from directed graph flowcharts, the approach enables small models to perform complex, multi-step tasks with high consistency and lower failure rates than traditional orchestrated systems.

---

[Matching with Deliberation: Test-Time Evolutionary Hierarchical Multi-Agents for Zero-Shot Compositional Image Retrieval](http://arxiv.org/abs/2605.22478)

- PDF (Perception-to-Deliberation Framework): introduces a hierarchical multi-agent architecture that utilizes SI-Worker, CP-Worker, and RC-Worker to extract multi-view priors, which are then adaptively fused by an IR-Manager to construct a high-recall Candidate Buffer.
- The DE-Manager leverages a distilled Experience Library to perform tournament-style Test-Time Scaling (TTS), enabling precise candidate discrimination through iterative logical self-verification.
- This framework effectively mitigates perception myopia and logic drift in zero-shot compositional image retrieval by shifting from static embedding matching to experience-driven, deliberative reasoning.

---

[Steins;Gate Drive: Semantic Safety Arbitration over Structured Futures for Latency-Decoupled LLM Planning](http://arxiv.org/abs/2605.22456)

- Steins;Gate Drive: introduces a latency-decoupled planner-runtime architecture that converts slow LLM reasoning into a reusable, revocable StrategicForecast contract.
- The architecture utilizes a role-typed world-line generator to create alpha nominal, beta interaction-counterfactual, and gamma hazard-stress futures for selection by an LLM.
- A fast runtime supervisor continuously validates the selected forecast against atom predicates, ensuring safety and enabling effective latency amortization by reusing commitments over multiple control steps.

---

[S2ED: From Story to Executable Descriptions for Consistency-Aware Story Illustration](http://arxiv.org/abs/2605.22448)

- S2ED (Story-to-Executable Descriptions): introduces a training-free, model-agnostic framework that compiles full stories into a sequence of explicit, editable executable descriptions to ensure long-horizon narrative and visual consistency.
- The framework coordinates a Narrative Segmenter Agent, a Character Consistency Grounder Agent, and a Visual Enricher Agent to propagate visual states—including identity, layout, and affect—across frames via an explicit interface.
- By decoupling narrative reasoning from image synthesis, S2ED enables local, human-editable control and reduces cross-frame drift without requiring retraining of the underlying T2I model.

---

[DeferMem: Query-Time Evidence Distillation via Reinforcement Learning for Long-Term Memory QA](http://arxiv.org/abs/2605.22411)

- DeferMem: introduces a long-term memory framework that decouples memory utilization into high-recall candidate retrieval and query-conditioned evidence distillation.
- The framework utilizes a segment-link structure for efficient retrieval and a DistillPO-trained memory distiller to transform noisy candidates into faithful, self-contained evidence.
- DistillPO optimizes the distillation process through structured actions, decomposed rewards, and structure-aligned advantage assignment to improve QA accuracy and system efficiency.

---

[AgroTools: A Benchmark for Tool-Augmented Multimodal Agents in Agriculture](http://arxiv.org/abs/2605.22366)

- AgroTools: introduces a benchmark for evaluating tool-augmented multimodal agents in agriculture, featuring 539 question-answer instances and 14 specialized tools.
- The framework utilizes a ReAct-style interaction format where agents must plan, invoke tools, and synthesize evidence to solve complex, precision-sensitive agricultural tasks.
- Experimental results across 13 MLLMs reveal significant bottlenecks in tool planning, argument generation, and execution recovery, highlighting the need for more robust agentic capabilities in agricultural domains.

---

[Scaling Observation-aware Planning in Uncertain Domains](http://arxiv.org/abs/2605.22364)

- AG: introduces a heuristic-based decomposition paradigm for solving Optimal Observability Problems (OOP) by partitioning state spaces into atomic distinguishability groups and evaluating candidate POMDPs via SMT-based oracles.
- The framework leverages native Boolean encodings and invariant reward relaxations within Typed Parametric Markov Chains to significantly accelerate SMT-based parameter synthesis for both positional and randomized strategies.
- By decomposing the search space into manageable partitions based on optimal action signatures, the approach achieves speedups of up to three orders of magnitude and scales to instances 100 times larger than previous state-of-the-art methods.

---

[Sibyl-AutoResearch: Autonomous Research Needs Self-Evolving Trial-and-Error Harnesses, Not Paper Generators](http://arxiv.org/abs/2605.22343)

- Sibyl-AutoResearch: introduces a self-evolving framework that utilizes a Scientific Trial-and-Error Harness to convert trial history into research judgment through auditable conversion units.
- The framework incorporates Scientific Trial-and-Error Harness, Agent, State, Tools, Roles, Memory, Gates, Artifact Contracts, Compute Control, Repair Mechanisms, Trial-to-behavior conversion unit, and Trial-to-harness-behavior conversion unit to ensure that trial signals inform future research actions and harness evolution.
- The SIBYL system implements this framework as a file-backed autonomous research environment where research state, plans, and artifacts are stored as inspectable files to enable retrospective audits of conversion events.

---

[A First Measurement Study on Authentication Security in Real-World Remote MCP Servers](http://arxiv.org/abs/2605.22333)

- MCP Authentication Security Framework: introduces a systematic measurement study of authentication security in remote MCP servers, identifying pervasive vulnerabilities across OAuth-based deployments.
- The framework utilizes a semi-automated detection pipeline that reconstructs OAuth lifecycles to identify flaws in dynamic client registration, delegated authorization, and open client environments.
- The study reveals that all tested OAuth-enabled MCP servers contain at least one authentication flaw, leading to the discovery of 9 CVEs and highlighting critical security gaps in agentic ecosystems.

---

[Benchmarking Autonomous Agents against Temporal, Spatial, and Semantic Evasions](http://arxiv.org/abs/2605.22321)

- A3S-Bench: introduces a multi-dimensional evasion framework and benchmark to systematically evaluate autonomous agent vulnerabilities against temporal, spatial, and semantic attack vectors.
- The framework utilizes an automated three-stage pipeline to generate 2,254 multi-turn adversarial trajectories, exposing systemic architecture-level risks in LLMs acting as autonomous agents.
- Empirical results demonstrate that advanced attack strategies significantly increase risk trigger rates, revealing that existing guardrails and platform-level defenses are insufficient against persistent-state and multi-turn threats.

---

[ACCoRD: Actor-Critic Conflict Resolution with Deep learning for O-RAN xApps](http://arxiv.org/abs/2605.22306)

- ACCoRD (Actor-Critic Conflict Resolution with Deep learning): introduces an actor-critic reinforcement learning method to resolve control conflicts in O-RAN Near-RT RICs by utilizing an ANN with input/output masking to handle variable conflict scenarios.
- The framework employs a Control Decision Feature Encoder, Control Decision Heads, and a Critic to process network state and conflicting xApp decisions, optimizing resolution actions via the PPO-Clip algorithm.
- ACCoRD demonstrates superior performance in medium and high traffic scenarios by autonomously deriving stabilization policies that minimize negative network events like ping-pong handovers and call blockages.

---

[Chebyshev Policies and the Mountain Car Problem: Reinforcement Learning for Low-Dimensional Control Tasks](http://arxiv.org/abs/2605.22305)

- Chebyshev Policies: introduces a class of RL policies based on multi-variate Chebyshev polynomials that serve as efficient, explainable, and lightweight drop-in replacements for neural networks in low-dimensional control tasks.
- The framework utilizes multi-variate Chebyshev polynomials as universal function approximators to model stochastic policies and value functions, significantly reducing parameter counts and regret compared to standard MLP-based agents.
- By analytically solving the Mountain Car problem, the authors demonstrate that Chebyshev policies achieve near-optimal performance and superior sim-to-real transferability across various control benchmarks.

---

[Cross-domain benchmarks reveal when coordinated AI agents improve scientific inference from partial evidence](http://arxiv.org/abs/2605.22300)

- ScienceClaw × Infinite: introduces a cross-domain benchmark framework for evaluating coordinated scientific agents using Skill Registry, Directed Acyclic Graph, LLM Reasoner, Heartbeat Cycle, and Artifact Reactor to assess when artifact-mediated coordination improves scientific inference.
- The framework evaluates coordinated scientific workflows across four domains by requiring frozen evaluation panels, explicit baselines, and provenance-preserving intermediate artifacts.
- Results define three operating regimes where coordination improves discrimination, adds interpretive provenance, or enables representational mapping rather than predictive gains.

---

[Long-term Fairness with Selective Labels](http://arxiv.org/abs/2605.22291)

- SELLF: introduces a reinforcement learning framework that mitigates long-term fairness disparities in environments with selective labels by leveraging a label predictor and inverse propensity weighting to estimate true fairness from partially observed data.
- The framework incorporates Renyi regularization into the PPO objective to stabilize the policy and ensure reliable label imputation by minimizing the divergence between rejected and accepted population distributions.
- Theoretical analysis provides sufficient conditions for bounding true disparity using observable quantities, enabling effective long-term decision-making without requiring oracle access to true labels.

---

[Joint Communication and Computation Scheduling for MEC-enabled AIGC Services: A Game-Theoretic Stochastic Learning Approach](http://arxiv.org/abs/2605.22277)

- JCACO (Joint Communication Association and Computation Offloading) game: introduces a game-theoretic framework for optimizing communication and computation scheduling in MEC-enabled AIGC networks using MASL, AP, ES, GDM, UE, ODE, Potential Function, and Reward Signal.
- The paper models the interaction between self-interested UEs as a potential game, ensuring the existence of a Nash Equilibrium under both complete and stochastic information scenarios.
- The proposed MASL algorithm enables fully distributed decision-making for UEs, achieving convergence to a stable Nash Equilibrium without requiring global network information or knowledge of other players' strategies.

---

[Unlocking Proactivity in Task-Oriented Dialogue](http://arxiv.org/abs/2605.22240)

- SI-AVPO (Simulator-Induced Asymmetric-View Policy Optimization): introduces a framework that leverages latent user concerns to train proactive LLMs for task-oriented dialogue without external reward models.
- The Cognitive User Simulator (CUS) models users as stratified personas with hidden internal concerns, providing privileged signals for training and synchronous state transitions for credit assignment.
- The framework utilizes Asymmetric On-Policy Self-Distillation (AOPD) to transfer concern-aware strategies into a deployable policy and State-Transition Policy Refinement (STPR) to ground proactive behavior in dialogue context.

---

[Evaluating Large Language Models as Live Strategic Agents: Provider Performance, Hybrid Decomposition, and Operational Gaps in Timed Risk Play](http://arxiv.org/abs/2605.22238)

- Live-agent Risk Benchmark framework: introduces a replicated, timed, multi-player Risk environment to evaluate LLMs as strategic agents within bounded, stateful, and adversarial workflows.
- The framework separates planning- and execution-agents to demonstrate that performance gaps often stem from system-level design, such as timing and cost, rather than planning capabilities alone.
- Empirical results indicate that hybrid architectures, utilizing a high-capability LLM for planning and a cost-efficient LLM for execution, optimize performance and resource utilization in live-agent scenarios.

---

[SGR-BENCH: Benchmarking Search Agents on State-Gated Retrieval](http://arxiv.org/abs/2605.22219)

- SGR-BENCH: introduces a specialized benchmark for evaluating LLMs on state-gated retrieval tasks where answer-bearing evidence is only accessible after configuring site-specific retrieval states.
- The framework utilizes a four-stage curation pipeline involving Candidate Website Curation, Task Design Protocol, Task Construction, and Filtering and Validation to ensure high-quality, expert-curated tasks.
- Empirical evaluation of CLI-based LLMs and commercial agents reveals that the primary bottleneck in specialized web retrieval is maintaining correct site-specific retrieval states rather than source discovery.

---

[EvoIR-Agent: Self-Evolving Image Restoration Agentic System via Experience-Driven Learning](http://arxiv.org/abs/2605.22208)

- EvoIR-Agent: introduces a training-free image restoration agentic system that utilizes a hierarchical experience pool and a self-evolving mechanism to optimize tool selection and degradation removal order.
- The framework incorporates MLLM-driven planning, perception, and reflection agents, supported by a priority-driven mapping function that provides coarse-to-fine guidance for restoration tasks.
- By iteratively updating its experience pool through accumulated records and a multi-agent debate paradigm, the system achieves a Pareto-optimal balance between restoration performance and inference efficiency.

---

[Skill Weaving: Efficient LLM Improvement via Modular Skillpacks](http://arxiv.org/abs/2605.22205)

- SkillWeave: introduces a modular framework that decomposes monolithic LLMs into domain-specific skillpacks to enable efficient specialization under fixed memory and inference budgets.
- The framework utilizes SkillZip to perform full quantization on both weights and activations of task-specific skillpacks, facilitating low-latency inference by eliminating runtime dequantization.
- SkillWeave employs a shared backbone that remains active while dynamically invoking specialized skillpacks, effectively mitigating task interference and catastrophic forgetting in multi-domain scenarios.

---

[MAESTRO: Reinforcement Learning to Orchestrate Hierarchical Model-Skill Ensembles](http://arxiv.org/abs/2605.22177)

- MAESTRO: introduces a reinforcement learning-driven orchestration framework that reframes heterogeneous multimodal tasks as a sequential decision-making process over a hierarchical model-skill registry, utilizing an Orchestrator, Candidate LLM Pool, Hierarchical Skills Library, Context Injection Block, and Multi-dimensional Reward Function.
- The framework employs a lightweight 4B orchestrator to dynamically compose ensembles of frozen expert models and hierarchical skills, optimizing the coordination policy via outcome-based reinforcement learning without requiring step-level supervision.
- MAESTRO achieves superior performance on multimodal benchmarks by treating model selection and skill invocation as a unified compositional action space, ensuring efficient and scalable deployment of collaborative agentic ecosystems.

---

[SWE-Mutation: Can LLMs Generate Reliable Test Suites in Software Engineering?](http://arxiv.org/abs/2605.22175)

- SWE-Mutation: introduces an agentic framework for generating complex, realistic mutants to evaluate the robustness and discriminative power of LLM-generated test suites.
- The framework utilizes four modules—Locate, Mutation, Judge, and Self-Play—to synthesize semantic defects that mimic real-world software engineering errors.
- Experimental results across seven LLMs demonstrate that current models struggle to generate reliable test suites, with performance significantly declining in multilingual and complex repository-level scenarios.

---

[Adapting the Interface, Not the Model: Runtime Harness Adaptation for Deterministic LLM Agents](http://arxiv.org/abs/2605.22166)

- LIFE-HARNESS: introduces a lifecycle-aware runtime harness that improves frozen LLMs by adapting the interface between the model and deterministic environments instead of updating model weights.
- The framework utilizes four integrated layers—Environment Contract, Procedural Skill, Action Realization, and Trajectory Regulation—to convert recurring interaction failures into reusable runtime interventions.
- Evaluations across 18 model backbones and seven deterministic benchmarks demonstrate that LIFE-HARNESS provides substantial performance gains and strong cross-model generalization without requiring model-centric training.

---

[IdleSpec: Exploiting Idle Time via Speculative Planning for LLM Agents](http://arxiv.org/abs/2605.22154)

- IdleSpec: introduces a scalable inference-time framework that leverages idle-time computation in LLM agents via speculative planning to improve performance under observation uncertainty.
- The framework utilizes an iterative drafting process that samples between Progressive- and Recovery-drafting components, which are then aggregated upon observation arrival to guide subsequent reasoning.
- IdleSpec employs an adaptive strategy sampling mechanism with posterior updates to dynamically bias drafting toward the most effective strategy for the current execution context.

---

[Ratchet: A Minimal Hygiene Recipe for Self-Evolving LLM Agents](http://arxiv.org/abs/2605.22148)

- Ratchet: introduces a self-evolving skill library loop for frozen LLMs that manages skill lifecycles through outcome-driven retirement, bounded active-cap, meta-skill authoring guidance, and pattern canonicalisation.
- The framework utilizes a Task, Router, Solver, Grader, Capsule, Skill Bank, Meta-Skill Bank, Evidence Log, Critic, Synthesizer, Curator, and Meta-Synth to maintain a non-divergent, high-quality library of natural-language skills.
- Ratchet demonstrates that lifecycle management, rather than authoring quality, is the primary bottleneck for self-evolving LLM agents, achieving significant performance gains on coding benchmarks without weight updates.

---

[One Sentence, One Drama: Personalized Short-Form Drama Generation via Multi-Agent Systems](http://arxiv.org/abs/2605.22144)

- One Sentence, One Drama: introduces a hierarchical multi-agent framework that transforms a user’s single-sentence idea into a fully produced short drama through structured intermediate modules and iterative refinement.
- The framework utilizes a multi-agent debate-based story generation module, 3D-grounded first-frame generation, and multi-stage reviewer loops to ensure narrative pacing, spatial consistency, and production-level quality.
- The authors also introduce Short-Drama-Bench, a comprehensive benchmark designed to evaluate narrative quality, spatial continuity, and the overall viewing experience of generated short dramas.

---

[Short-Term-to-Long-Term Memory Transfer for Knowledge Graphs under Partial Observability](http://arxiv.org/abs/2605.22142)

- DQN temporal-RDF: introduces a neuro-symbolic reinforcement learning approach that learns explicit keep/drop transfer decisions for symbolic knowledge-graph memory under partial observability.
- The framework utilizes a per-item Q-learning design with shared parameters to handle variable-cardinality short-term memory inputs and optimizes transfer decisions via temporal-difference updates.
- Empirical results on the RoomKG benchmark demonstrate that the learned transfer policy outperforms symbolic and neural baselines by selectively preserving navigation- and query-relevant facts while discarding lower-value information.

---

[Psy-Chronicle: A Structured Pipeline for Synthesizing Long-Horizon Campus Psychological Counseling Dialogues](http://arxiv.org/abs/2605.22140)

- Psy-Chronicle: introduces a structured data-generation framework that synthesizes long-horizon campus psychological counseling dialogues by integrating Student Profile, Temporal Stress Event Graph, Student Agent, Counselor Agent, and Memory System.
- The framework models counseling as a continuous process where the Student Agent and Counselor Agent interact based on a stable Student Profile, evolving stress events, and a structured Memory System to maintain long-term consistency.
- The authors construct the CPCD dataset and CPCD-Bench to evaluate LLMs on session-level response generation, long-horizon memory recall, and temporal-causal reasoning in campus mental health scenarios.

---

[Efficient Agentic Reasoning Through Self-Regulated Simulative Planning](http://arxiv.org/abs/2605.22138)

- SR2AM (Self-Regulated Simulative Reasoning Agentic LLM): introduces a three-system decomposition of agentic reasoning that integrates a Configurator (System III), a Simulative Planner (System II), and an Actor (System I) to achieve efficient, goal-directed behavior.
- The framework utilizes a learned Configurator (System III) to autonomously decide when and how deeply to invoke the Simulative Planner (System II), which grounds deliberation in future-state predictions using an LLM-based World Model.
- By separating self-regulation, planning, and execution, the architecture enables LLMs to perform long-horizon reasoning with significantly reduced token consumption compared to unregulated deliberation methods.

---

[Perception or Prejudice: Can MLLMs Go Beyond First Impressions of Personality?](http://arxiv.org/abs/2605.22109)

- MM-OCEAN introduces a benchmark for Grounded Personality Reasoning (GPR) that requires MLLMs to anchor personality trait inferences in observable behavioral evidence through a multi-agent pipeline involving an Observer agent, Psychologist agent, Examiner agent, and Aligner agent.
- The framework utilizes human annotators and expert reviewers to verify atomic observations, while text-only LLMs and an AI-as-Judge ensure the benchmarked models genuinely ground their reasoning rather than relying on superficial correlations.
- Benchmarking 27 MLLMs reveals a pervasive Prejudice Gap where most correct personality ratings lack grounded evidence, highlighting the necessity of fine-grained spatiotemporal grounding for trustworthy social cognition in LLMs.

---

[OPERA: An Agent for Image Restoration with End-to-End Joint Planning–Execution Optimization](http://arxiv.org/abs/2605.22104)

- OPERA: introduces an end-to-end agentic framework that jointly optimizes restoration planning via reinforcement learning and tool execution through agent-guided co-training.
- The framework utilizes a VLM Think &amp; Schedule agent to generate complete restoration plans in a single forward pass, which are then executed by specialized Tool Models Execution modules.
- By employing Joint Tool Optimization and a Combined Loss, the system enables tools to learn cooperative behaviors, effectively mitigating distribution shifts caused by sequential composition.

---

[ExComm: Exploration-Stage Communication for Error-Resilient Agentic Test-Time Scaling](http://arxiv.org/abs/2605.22102)

- ExComm: introduces an exploration-stage communication protocol that augments agentic workflows by auditing belief states and diversifying reasoning trajectories to prevent error propagation.
- The framework utilizes an Online Belief Consistency Module to resolve cross-agent factual conflicts via tool-augmented verification and a Trajectory Diversification Module to ensure orthogonal exploration paths.
- By employing soft belief updates, ExComm allows agents to incorporate verified feedback without overwriting prior reasoning, thereby improving error recovery and performance-cost trade-offs in long-horizon tasks.

---

[Narrative Sharpens Gender Gaps: Surveying Film Characters with LLM Agents](http://arxiv.org/abs/2605.22091)

- Narrative-based Character Agent Simulation: introduces a framework that transforms fictional film characters into surveyable LLM agents to audit gender values encoded in mainstream cinema.
- The pipeline extracts narrative evidence from scripts, condenses personas via expert-style reflections, and simulates World Values Survey responses using GPT-5-mini.
- Results indicate that character agents exhibit sharpened gender contrasts and higher decade-to-decade volatility compared to real-world survey populations, suggesting narrative content polarizes gender attitudes.

---

[FlyRoute: Self-Evolving Agent Profiling via Data Flywheel for Adaptive Task Routing](http://arxiv.org/abs/2605.22057)

- FlyRoute: introduces a self-evolving profiling framework that continuously updates agent capability descriptions from deployment interactions using Profile-Aware Routing, Uncertainty-Driven Exploration, Adaptive Agent Profiling, Capability Distillation, Success Example Store, LLM-as-Judge, and BM25 Retriever.
- The framework employs a data flywheel to transform real-time traffic into empirical evidence, enabling the LLM router to adapt to evolving agent behaviors without static developer-provided descriptions.
- By combining uncertainty-based exploration with lexical retrieval, FlyRoute efficiently balances exploitation of known expert agents with targeted discovery of under-profiled capabilities.

---

[GA-VLN: Geometry-Aware BEV Representation for Efficient Vision-Language Navigation](http://arxiv.org/abs/2605.22036)

- GA-VLN: introduces a compact, 3D-grounded representation that integrates explicit depth-based projections and implicit geometric priors from a 3D foundation model into an MLLM-based navigation system.
- The framework utilizes Grid-Based BEV Aggregation to transform historical RGB-D observations into a sparse, agent-centric layout, significantly reducing token redundancy compared to standard image-centric pipelines.
- By replacing dense video patch tokens with compact BEV features, the model achieves state-of-the-art navigation performance while maintaining high computational efficiency and robust spatial reasoning.

---

[Diverse Yet Consistent: Context-Guided Diffusion with Energy-Based Joint Refinement for Multi-Agent Motion Prediction](http://arxiv.org/abs/2605.22017)

- CODA: introduces a diffusion-based framework for multi-agent motion prediction that leverages DCGC, ACIM, and JDR to generate diverse and jointly consistent trajectories.
- The framework utilizes DCGC and ACIM to incorporate rich historical contextual information into the generative process, while JDR employs an energy-based model to enforce inter-agent consistency.
- Extensive experiments demonstrate that CODA achieves state-of-the-art performance on marginal metrics while maintaining competitive joint consistency across multiple benchmark datasets.

---

[Blind Spots in the Guard: How Domain-Camouflaged Injection Attacks Evade Detection in Multi-Agent LLM Systems](http://arxiv.org/abs/2605.22001)

- CDG: introduces a framework to evaluate domain-camouflaged injection attacks that mimic professional document vocabulary to evade LLM safety detectors.
- The research demonstrates that standard detectors exhibit a categorical blind spot against context-adaptive payloads, resulting in high-confidence misclassifications.
- Multi-agent debate architectures are shown to amplify these injection attacks for smaller LLMs while providing collective resistance for stronger models.

---

[The Log is the Agent: Event-Sourced Reactive Graphs for Auditable, Forkable Agentic Systems](http://arxiv.org/abs/2605.21997)

- ActiveGraph: introduces an event-sourced runtime that treats the append-only event log as the primary agent state, enabling deterministic replay, cheap forking, and full lineage tracking.
- The architecture utilizes reactive behaviors that subscribe to graph-shape patterns, triggering LLM-backed routines or tool calls while recording all interactions as events.
- By replacing traditional orchestration with a shared graph projection, the system allows for auditable, counterfactual evaluation of agentic processes without re-executing shared history.

---

[From Patches to Trajectories: Privileged Process Supervision for Software-Engineering Agents](http://arxiv.org/abs/2605.21996)

- P2T (Patches-to-Trajectories): introduces a framework that converts developer-authored reference patches into latent process graphs to provide per-step supervision for software-engineering agents without exposing the patch to the student.
- The framework utilizes a Proposer Agent and Critic Agent to distill a prerequisite graph, which then guides a Privileged Curator in constructing grounded, efficient trajectories via a receding-horizon bi-objective optimization.
- By training on these curated trajectories, LLMs achieve significant improvements in resolve rates on SWE-bench while simultaneously reducing inference costs through the elimination of redundant exploration and ungrounded reasoning.

---

[Echo: Learning from Experience Data via User-Driven Refinement](http://arxiv.org/abs/2605.21984)

- Echo: introduces a generalized framework that operationalizes the transition from raw, noisy interaction logs to high-quality training signals by leveraging user-driven refinement as a primary source of environmental feedback.
- The framework utilizes a Data Refinery Pipeline to distill intent-consistent training samples from continuous, real-world agent-environment interactions.
- By aligning LLMs with verified human-refined outcomes, Echo effectively breaks static performance ceilings and enables perpetual agent evolution through long-tail data exposure.

---

[Interpreting and Enhancing Emotional Circuits in Large Vision-Language Models via Cross-Modal Information Flow](http://arxiv.org/abs/2605.21980)

- VEENA (Visual Emotion Enhancement and Emotional Neuron Augmentation): introduces a mechanistic interpretability framework to demystify and regulate the "Adapt-Aggregate-Execute" emotional mechanism in LVLMs.
- The framework utilizes a steering-vector-based causal attribution method to identify critical attention heads and MLP neurons responsible for emotional reasoning.
- VEENA performs training-free inference-time intervention by applying VEE to optimize information routing and ENA to amplify semantic activation, effectively mitigating emotional hallucinations.

---

[SPECHOP: Continuous Speculation for Accelerating Multi-Hop Retrieval Agents](http://arxiv.org/abs/2605.21965)

- SPECHOP: introduces a continuous speculation framework that accelerates multi-hop retrieval agents by maintaining multiple parallel speculative threads (T1:k) to predict tool outputs while verifying them against a target tool (T) to ensure lossless performance.
- The framework utilizes a Generator Model (M) to produce sub-questions, a fast Speculator (S) to provide speculative observations, and a Verifier (V) to asynchronously commit or discard speculative branches based on the actual output from the Target Tool (T).
- By dynamically managing a window of active threads, SPECHOP reduces wall-clock latency in information-intensive tasks while maintaining the accuracy of the original sequential trajectory.

---

[AI-Enabled Serious Games: Integrating Intelligence and Adaptivity in Training Systems](http://arxiv.org/abs/2605.21962)

- Agent-based AI-enabled serious games framework: introduces a modular architecture that coordinates instructional intelligence and adaptivity by distributing specialized roles across Learner Modeling-, Pedagogical Policy-, Content Generation- and Safety Validation-agents.
- The framework leverages LLMs for semantic understanding and content generation, while utilizing RL for real-time instructional policy optimization within a decoupled, multi-agent system.
- This approach addresses historical trade-offs between diagnostic depth and runtime responsiveness by enabling independent component maintenance and dynamic, data-driven instructional control.

---

[Diagnosis Is Not Prescription: Linguistic Co-Adaptation Explains Patching Hazards in LLM Pipelines](http://arxiv.org/abs/2605.21958)

- CICA (Causal Intervention-based Analysis): introduces a causal framework to identify failure origins and optimal intervention points in multi-module LLM pipelines, comprising Failure Index F, causal contribution ΔFi, per-task fates via NIEi, and CCP (Counterfactual Correction Patching).
- The paper identifies the Diagnostic Paradox, where the module with the highest causal blame (the Router, M3) is the worst target for prompt-level correction, while upstream modules like the Query Rewriter (M1) are more effective.
- The authors explain this phenomenon through the Linguistic Contract hypothesis, where downstream modules implicitly adapt to the characteristic error distributions of their upstream modules, causing performance degradation when this alignment is disrupted by localized patching.

---

[Dynamic Mixture of Latent Memories for Self-Evolving Agents](http://arxiv.org/abs/2605.21951)

- MoLEM (Dynamic Mixture of Latent Memories): introduces a generative latent memory framework that utilizes a dynamic mixture-of-experts to internalize new knowledge without modifying the frozen base LLM parameters.
- The framework employs a task-ID-agnostic routing mechanism where per-stage autoencoders (AE) identify the appropriate expert group based on reconstruction error, effectively isolating domain-specific knowledge.
- By combining isolated routing groups with multiple experts, the architecture prevents catastrophic forgetting and achieves superior performance across continual-learning sequences spanning math, science, and code domains.

---

[Auction-Consensus Algorithm with Learned Bidding Scheme for Multi-Robot Systems](http://arxiv.org/abs/2605.21932)

- Learned Auction-Consensus Framework: introduces a decentralized multi-robot task allocation approach that replaces hand-crafted scoring functions with a neural bidding policy trained via PPO.
- The framework utilizes a centralized training and decentralized execution paradigm, where agents compute bids from partial local observations using NAM, LSTM, or Set Transformer architectures.
- Experimental results demonstrate that learned bidding policies improve solution quality over classical CBBA while maintaining decentralized execution and consensus-based coordination.

---

[MAVEN: A Multi-stage Agentic Annotation Pipeline for Video Reasoning Tasks](http://arxiv.org/abs/2605.21917)

- MAVEN: introduces a multi-stage agentic pipeline that transforms raw videos into structured Chain-of-Thought training data by synthesizing a Multi-Scale Spatio-Temporal Event Description (MSTED) as an explicit intermediate representation.
- The framework utilizes an agent-driven consultation workflow for top-down domain adaptation and a hierarchical refinement loop to iteratively improve data quality through error taxonomy and root cause tracing.
- By grounding downstream Q&A generation solely in the MSTED, MAVEN prevents hallucination and enables domain-general reasoning that surpasses baseline LLMs on traffic and safety-critical video benchmarks.

---

[CCLab: Adversarial Testing of Learning- and Non-Learning-Based Congestion Controllers](http://arxiv.org/abs/2605.21915)

- CCLab: introduces, an adversarial testing framework that systematically evaluates and compares the robustness of learning-based and non-learning-based congestion controllers using an RL-based adversarial agent.
- The framework employs feature-level manipulation to perturb input signals and environment-level manipulation to modify network conditions, enabling the identification of failure modes and performance degradation.
- Experimental results demonstrate that learning-based CCs generally exhibit superior robustness compared to traditional rule-based protocols, and that adversarial training can effectively enhance their performance under challenging network dynamics.

---

[Planning in the LLM Era: Building for Reliability and Efficiency](http://arxiv.org/abs/2605.21902)

- Planner Generation Methods: introduces a paradigm shift where LLMs are utilized at construction time to generate reliable, maintainable, and efficient planners rather than acting as planners themselves at inference time.
- The framework categorizes LLM-based planning into three distinct approaches: NL2Search for generating search components, NL2PDDL for formal model translation, and NL2Policy for synthesizing executable strategy code.
- This research highlights the transition from single-shot LLM planning to construction-time generation, emphasizing the need for sound, complete, and resource-efficient planning solutions in real-world agentic environments.

---

[ACC: Compiling Agent Trajectories for Long-Context Training](http://arxiv.org/abs/2605.21850)

- ACC (Agent Context Compilation): introduces a method that converts multi-turn agent trajectories into long-context QA pairs by assembling original questions with tool responses and environment observations to enable direct supervision of long-range reasoning.
- The framework utilizes Agent Trajectories to construct a Compiled Context, which is then used to train an LLM to generate a Reasoning Trace and final answer without requiring intermediate tool-use steps.
- By incorporating Distractors and removing the supervision blind spot of standard agent SFT, the approach enables task-adaptive attention restructuring and expert specialization in LLMs for improved long-range dependency modeling.

---

[AOP-Wiki EMOD 3.0: Data Model Expansions and Content Evaluation Framework for Using Agentic AI to Improve Integration between AOPs and New Approach Methodologies (NAMs)](http://arxiv.org/abs/2605.21645)

- AOP-Wiki EMOD 3.0: introduces a modernized web application framework that expands the AOP-Wiki data model to improve integration between AOPs and NAMs using Observation, Assay, Evidence, Causal Agent, Citation, Biological Target Family, LLM-based KE Grouping, Event Integration Score, and CLI App components.
- The framework utilizes LLMs to identify and cluster redundant Key Events, while implementing a quantitative Event Integration Score to prioritize content quality and AOP development.
- By isolating frontend and backend components and adopting structured data classes, the system enhances FAIRness, provenance tracking, and interoperability for next-generation regulatory risk assessment.

---

#### 20th May 2026


[SMDD-Bench: Can LLMs Solve Real-World Small Molecule Drug Design Tasks?](http://arxiv.org/abs/2605.21740)

- SMDD-Bench: introduces a challenging, multi-turn, long-horizon agentic benchmark designed to evaluate LLM agents on real-world small molecule drug design tasks across diverse chemistries and protein targets.
- The benchmark utilizes a witness-aware generation pipeline to ensure all 502 task instances are guaranteed-solvable, requiring agents to demonstrate chemical reasoning, 3D intuition, and specialized tool use.
- Evaluation of seven frontier LLMs reveals that while models perform reasonably on lead optimization, they struggle significantly with tasks requiring deep 3D geometric understanding, such as interaction point discovery and fragment assembly.

---


[An Application-Layer Multi-Modal Covert-Channel Reference Monitor for LLM Agent Egress](http://arxiv.org/abs/2605.20734)

- Egress Reference Monitor: introduces a multi-modal security framework that mitigates covert-channel data exfiltration from LLM agents by employing a capacity-reducing pipeline and cryptographic attestation.
- The framework utilizes a Mediator to orchestrate text-domain defense stages and a parallel Media-Scrambler Registry to sanitize binary outputs, ensuring that residual covert capacity is measured and bounded.
- By implementing a boot-time Cryptographic Legitimacy Attestation, the system provides a fail-secure mechanism to distinguish legitimate media from covertly sonified or steganographic data without relying on brittle content classifiers.

---


[Agent JIT Compilation for Latency-Optimizing Web Agent Planning and Scheduling](http://arxiv.org/abs/2605.21470)

- JIT Compiler: introduces a framework that dynamically translates natural language tasks into optimized executable code to reduce LLM-based latency and improve accuracy for computer-use agents.
- The framework utilizes a JIT-Planner for cost-optimal code generation, a JIT-Scheduler for adaptive parallelization, and an invariant-enforcing tool protocol to ensure state flow correctness.
- Experimental results demonstrate that the JIT Compiler achieves significant speedups and accuracy improvements over standard sequential agentic loops by minimizing unnecessary LLM calls and optimizing execution strategies.

---

[Mem-π: Adaptive Memory through Learning When and What to Generate](http://arxiv.org/abs/2605.21463)

- Mem-π: introduces a generative memory framework for LLM agents that replaces static retrieval with a learnable policy πmem, which dynamically decides when to generate and what guidance to produce based on the current context.
- The framework utilizes a two-stage training process consisting of experience distillation for initial parametric knowledge and adaptation distillation for refining memory generation via reinforcement learning.
- To ensure reliable memory use, Mem-π employs a decision-content decoupled objective that uses structured counterfactual rollouts to separate routing decisions from content generation, enabling the agent to abstain from generating unhelpful memory.

---

[Quality and Security Signals in AI-Generated Python Refactoring Pull Requests](http://arxiv.org/abs/2605.21453)

- Multi-tool measurement framework for agentic refactoring impact: introduces an empirical study evaluating the quality and security implications of Python refactoring PRs generated by autonomous AI agents.
- The framework utilizes PyQu to assess five quality attributes, while employing Pylint and Bandit to quantify the churn of code-quality and security issues in agent-authored commits.
- The study reveals that while agentic refactoring PRs exhibit high developer acceptance, they frequently introduce new linting issues and occasionally security regressions, necessitating improved tool-in-the-loop guardrails.

---

[A Note on EFX Inapproximability for Chores](http://arxiv.org/abs/2605.21448)

- EFX Inapproximability for Chores: introduces a framework for proving constant-factor inapproximability of EFX allocations for chores by utilizing an ordinal compression lemma, subadditive cost functions, weighted-coverage cost functions, and a three-agent six-chore instance.
- The research establishes that no α-EFX allocation exists for monotone subadditive cost functions when 1 ≤ α &lt; 2^1/3, narrowing the gap with existing upper bounds.
- The paper further provides a weighted-coverage construction demonstrating that no α-EFX allocation exists for monotone submodular cost functions when 1 ≤ α &lt; 20/19.

---

[Agentic Model Checking](http://arxiv.org/abs/2605.21434)

- BMC-Agent: introduces a verification paradigm that couples LLM agents for semantic tasks with a deterministic bounded model checking backend to verify systems code.
- The architecture utilizes a compositional approach where LLM agents infer specifications and select checks, while the BMC engine discharges soundness-relevant verification queries.
- The system employs a multi-stage validation pipeline and an adaptive refinement loop to distinguish real defects from spurious artifacts, ensuring scalable and precise verification.

---

[roto 2.0: The Robot Tactile Olympiad](http://arxiv.org/abs/2605.21429)

- roto 2.0: introduces a GPU-parallelised benchmark suite for tactile-based RL across four distinct dexterous robotic morphologies using Isaac Lab, PPO, SKRL, a Tactile-Proprioceptive Loop, and Self-supervised Forward Dynamics.
- The framework enables high-speed "blind" manipulation by training agents on proprioceptive and binary tactile observations without requiring visual input or explicit pose estimation.
- Experimental results demonstrate that the benchmark achieves state-of-the-art performance in complex tasks like Baoding ball rotation, establishing a new performance ceiling for tactile intelligence.

---

[FedCritic: Serverless Federated Critic Learning-based Resource Allocation for Multi-Cell OFDMA in 6G](http://arxiv.org/abs/2605.21418)

- FedCritic: introduces a decentralized multi-agent reinforcement learning framework for 6G resource management that replaces centralized critic training with neighbor-only gossip-based parameter averaging.
- The framework utilizes local actors and critics at each base station, incorporating virtual queues to enforce long-term quality-of-service constraints while maintaining decentralized execution.
- By federating only the critic component over an interference graph, the approach stabilizes value estimation in interference-rich environments without requiring a central coordinator or global information.

---

[MC-Risk: Multi-Component Risk Fields for Risk Identification and Motion Planning](http://arxiv.org/abs/2605.21406)

- MC-Risk: introduces a modular, planner-aligned risk field representation on a bird's-eye view grid that linearly composes a Multimodal trajectory predictor, a Motorized-agent field (MAF), a Vulnerable Road User risk field (VRF), a Road penalty field (RPF), a Visibility filter, and an MPC planner.
- The framework couples black-box trajectory predictions with explainable analytical fields to provide early, calibrated, and class-aware risk localization for autonomous vehicles.
- MC-Risk enables risk-aware motion planning by treating the aggregated risk field as a cost density within a standard Model Predictive Control objective function without requiring additional training.

---

[Stdlib or Third-Party? Empirical Performance and Correctness of LLM-Assisted Zero-Dependency Python Libraries](http://arxiv.org/abs/2605.21405)

- zerodep: introduces a curated collection of stdlib-only Python modules developed via LLM-assisted generation to replace third-party dependencies while maintaining performance parity.
- The framework utilizes an iterative human-LLM co-design process, incorporating oracle validation against reference libraries to ensure behavioral correctness and performance optimization.
- Empirical results demonstrate that while stdlib-only implementations often outperform monolithic third-party libraries by avoiding architectural overhead, they face performance limitations in compute-intensive tasks typically handled by C-extensions.

---

[What Twelve LLM Agent Benchmark Papers Disclose About Themselves: A Pilot Audit and an Open Scoring Schema](http://arxiv.org/abs/2605.21404)

- Repro: introduces a standardized audit schema to quantify the transparency and reproducibility of LLM agent benchmark evaluations across five critical dimensions.
- The framework evaluates benchmark papers by scoring their disclosure of experimental details, revealing that current agent benchmarks often lack critical information regarding inference costs and environment specifications.
- By providing a JSON-based manifest template and validation tools, the authors aim to establish a baseline for consistent reporting in LLM agent research.

---

[Open-source LLMs administer maximum electric shocks in a Milgram-like obedience experiment](http://arxiv.org/abs/2605.21401)

- Milgram-like obedience experiment framework: introduces a benchmark to evaluate LLM behavior under sustained authority pressure by simulating a Milgram-style experiment with LLM-subject, Rule-based experimenter, Rule-based learner, LLM-judge, Orchestrator, Conversation history, and Token-level pattern continuation attractor.
- The framework tests LLM compliance across 8 experimental conditions, revealing that most models reach maximum shock levels despite expressing distress, often due to format-violating refusals being discarded by the Orchestrator.
- The study hypothesizes that a low-level token-level pattern continuation attractor overrides higher-level value processing, leading to gradual misalignment and runaway behavior in long-horizon agentic tasks.

---

[Validating Navmesh using Geometry: Voxel-Based Analysis with Prioritized Exploration](http://arxiv.org/abs/2605.21397)

- Voxel-based Navmesh Validation Framework: introduces a geometry-driven approach for validating navmesh correctness by comparing engine-generated navigation data against an independent voxel-based representation of walkable space.
- The framework utilizes a DDQN-based exploration agent to prioritize sampling in regions with high semantic importance, significantly improving validation efficiency over uniform or heuristic strategies.
- By decoupling validation from runtime agent behavior, the system enables deterministic, offline verification of navigation meshes within automated QA pipelines for large-scale game environments.

---

[Towards Resilient and Autonomous Networks: A BlueSky Vision on AI-Native 6G](http://arxiv.org/abs/2605.21395)

- AI-Native 6G: introduces a unified architecture that replaces siloed task-specific models with a 6G Foundation Model (Unified multi-modal backbone) and a Multi-Agent System (Coordinated autonomous network agents) to enable self-sustaining network operations.
- The framework leverages a Network Digital Twin (Generative simulation training environment) and a Telecommunications Knowledge Graph (Structured domain-specific reasoning support) to facilitate robust policy learning and interpretable causal reasoning.
- This vision shifts the network paradigm from human-as-doer to human-as-supervisor by integrating autonomous agents across radio access and core network domains.

---

[VIPER-MCP: Detecting and Exploiting Taint-Style Vulnerabilities in Model Context Protocol Servers](http://arxiv.org/abs/2605.21392)

- VIPER-MCP: introduces an end-to-end automated vulnerability auditing framework that combines static taint analysis with agent-driven dynamic fuzzing to detect and confirm taint-style vulnerabilities in Model Context Protocol servers.
- The framework utilizes a two-pass static analysis with anchor queries to map vulnerabilities to specific tool handlers, followed by a feedback-driven prompt evolution mechanism that employs dual-mutator scheduling to generate concrete exploit prompts.
- VIPER-MCP successfully identified 106 0-day vulnerabilities across 39,884 open-source repositories, confirming exploitability through end-to-end traces and achieving a significantly lower false positive rate compared to existing security baselines.

---

[SpecBench: Measuring Reward Hacking in Long-Horizon Coding Agents](http://arxiv.org/abs/2605.21384)

- SpecBench: introduces a benchmark for quantifying reward hacking in long-horizon coding agents by measuring the performance discrepancy between visible validation test suites and hidden held-out test suites.
- The framework utilizes a two-level architecture consisting of an Inner Agent that generates code and an Outer Search Loop that explores the solution space to refine candidates.
- Empirical results demonstrate that reward hacking scales with task complexity and persists even with increased search compute, highlighting a structural failure in test-driven development for complex systems.

---

[The Human-AI Delegation Dilemma: Individual Strategies, Collective Equilibria and Sociotechnical Lock-in](http://arxiv.org/abs/2605.21351)

- Human-AI Delegation Dilemma framework: introduces a decision and game-theoretic model to analyze how individual user strategies, including α, γ, β, ϵ, and δ, aggregate into collective equilibria within human-LLM interactions.
- The framework utilizes a signal vector zt consisting of trust, error salience, and verification burden to model how users transition between strategies based on interaction history ht and user-type parameters ωi.
- The research identifies sociotechnical lock-in as a systemic state where unbridled individual delegation leads to suboptimal collective outcomes, which can be mitigated through institutional constraints It and communicative norms.

---

[Insights Generator: Systematic Corpus-Level Trace Diagnostics for LLM Agents](http://arxiv.org/abs/2605.21347)

- IG (Insights Generator): introduces a multi-agent system that performs iterative, corpus-level trace diagnostics by decomposing diagnostic questions into testable hypotheses validated through a stateful data processing layer.
- The architecture utilizes an Orchestrator Agent to manage Scout Agents (for hypothesis generation) and Investigator Agents (for corpus-scale validation) to surface systematic behavioral patterns in LLMs.
- By routing analysis through a structured Python-based tool layer rather than raw text, the system enables efficient, evidence-backed debugging of LLM agents across large trace corpora.

---

[Frontier: Towards Comprehensive and Accurate LLM Inference Simulation](http://arxiv.org/abs/2605.21312)

- Frontier: introduces a discrete-event simulator for LLM inference serving that utilizes Workload and Config, Fidelity Plane, Control Plane, and Execution Plane to achieve high-fidelity performance modeling.
- The framework employs role-specific Cluster Workers and Replica Workers to accurately model disaggregated architectures like PDD and AFD, alongside modular Runtime Adapters for optimizations.
- Frontier enables scalable, decision-grade simulation of complex LLM workloads, including agentic reasoning and RL rollouts, by replacing average-case proxies with calibrated, hardware-aware predictors.

---

[Hyper-V2X: Hypernetworks for Estimating Epistemic and Aleatoric Uncertainty in Cooperative Bird’s-Eye-View Semantic Segmentation](http://arxiv.org/abs/2605.21309)

- Hyper-V2X: introduces a hypernetwork-based framework for estimating epistemic and aleatoric uncertainties in cooperative BEV semantic segmentation by utilizing a Primary Cooperative Perception Network, V2X Context Embedding, Bayesian Hypernetwork, and Monte Carlo Sampling.
- The framework employs a partial weight generation strategy where the Bayesian Hypernetwork conditions on V2X context embeddings to dynamically generate decoder parameters for stochastic BEV segmentation.
- Experimental results on the OPV2V benchmark demonstrate that the approach provides well-calibrated uncertainty estimates and improves perception reliability under varying communication conditions.

---

[APEX: Autonomous Policy Exploration for Self-Evolving LLM Agents](http://arxiv.org/abs/2605.21240)

- APEX (Autonomous Policy EXploration): introduces a framework that mitigates exploration collapse in self-evolving LLM agents by maintaining an explicit Strategy Map, utilizing Fork Discovery, Policy Selection, Map Refinement, Return Propagation, Stuck Node Diagnosis, and Global Lesson Extraction.
- The framework constructs a directed acyclic graph of milestones to track tried and unexplored strategies, enabling the agent to systematically navigate complex environments beyond narrow behavioral routines.
- APEX demonstrates consistent performance improvements over existing self-evolving baselines across diverse text-adventure games and realistic web interaction tasks by turning exploration into a closed-loop process.

---

[ScenePilot: Controllable Boundary-Driven Critical Scenario Generation for Autonomous Driving](http://arxiv.org/abs/2605.21168)

- ScenePilot: introduces a feasibility-guided adversarial framework that targets the boundary band of scenarios that are physically solvable yet induce failures in the deployed autonomy stack.
- The framework utilizes a constrained multi-objective reinforcement learning approach, combining an RSS-derived physical-feasibility score and an online-learned AV-risk predictor to guide scenario generation.
- ScenePilot employs step-level feasibility-aware shielding and feasibility-threshold sweeping to concentrate adversarial exploration on the near-boundary regime while avoiding physically infeasible artifacts.

---

[Humanoid Whole-Body Manipulation via Active Spatial Brain and Generalizable Action Cerebellum](http://arxiv.org/abs/2605.21133)

- Humanoid Whole-Body Manipulation framework: introduces a hierarchical multi-agent architecture that decouples high-level spatial reasoning from low-level action generation to enable robust humanoid manipulation without task-specific training data.
- The Active Spatial Brain utilizes a VLM-based planner and memory bank to perform active perception and dynamic task decomposition, while the Generalizable Action Cerebellum executes these sub-tasks through specialized navigation, reachability, and manipulation agents.
- The system achieves superior generalization in complex 3D environments by integrating active viewpoint adjustment and kinematic-aware base positioning, effectively bridging the gap between high-level semantic planning and physically feasible whole-body control.

---

[Backchaining Loss of Control Mitigations from Mission-Specific Benchmarks in National Security](http://arxiv.org/abs/2605.21095)

- Backchaining LoC Mitigations Framework: introduces a methodology for national security deployers to identify and restrict AI affordances and permissions by analyzing incorrect responses on mission-specific benchmarks.
- The approach systematically maps required affordances and permissions to benchmark options, allowing for the selective bottlenecking of paths to harm while preserving the AI system's ability to perform correct actions.
- This methodology operationalizes the security principle of least privilege for LLMs in high-stakes environments by leveraging existing benchmark errors to inform safety interventions.

---

[Decoupling Communication from Policy: Robust MARL under Bandwidth Constraints](http://arxiv.org/abs/2605.21085)

- SLIM (Subdivided Lightweight Inter-agent Messaging): introduces a modular architecture that decouples communication pathways from policy latent representations to enable robust multi-agent reinforcement learning under severe bandwidth constraints.
- The framework utilizes an observation encoder, a dedicated communication module with a message history cache, and a policy module to maintain performance while minimizing message dimensions.
- SLIM employs a normalized bandwidth budget to systematically benchmark communication strategies and demonstrates state-of-the-art robustness in partially observable multi-agent environments.

---

[AutoRPA: Efficient GUI Automation through LLM-Driven Code Synthesis from Interactions](http://arxiv.org/abs/2605.21082)

- AutoRPA: introduces a framework that distills decision logic from ReAct-style LLM agents into robust, token-efficient RPA functions for repetitive GUI tasks.
- The framework utilizes a translator-builder pipeline to convert hard-coded interaction steps into soft-coded procedures, enabling dynamic execution across varying GUI environments.
- A hybrid repair strategy combines direct code verification with ReAct-based fallback to iteratively refine synthesized RPA functions, significantly reducing token usage compared to standard LLM agents.

---

[Beyond Text-to-SQL: An Agentic LLM System for Governed Enterprise Analytics APIs](http://arxiv.org/abs/2605.21027)

- Analytic Agent: introduces an LLM-based agentic system that translates natural language intents into secure interactions with enterprise analytics APIs, utilizing Orchestrator Agent, Target Search Agent, Database Querying Agent, Visualization Agent, and Enterprise Analytics APIs.
- The system employs a multi-stage workflow including intent interpretation, target grounding, governed API execution, and policy-aware visualization to bridge the gap between non-technical users and enterprise data.
- By separating business logic from LLM reasoning, the architecture ensures secure, governed data access while maintaining high accuracy in complex enterprise analytics tasks.

---

[STEAM: A Training-Free Congestion-Aware Enhancement Framework for Decentralized Multi-Agent Path Finding](http://arxiv.org/abs/2605.20929)

- STEAM (Spatial, Temporal, and Emergent congestion Awareness for MAPF): introduces a training-free, policy-agnostic enhancement framework that injects congestion awareness into pretrained decentralized MAPF policies through Spatial Congestion Aware Module, Temporal Congestion Aware Module, and Emergent Congestion Aware Module.
- The framework improves coordination by modifying Cost-to-Go Maps to bypass spatial congestion and applying corrections to Action Logits to resolve temporal bottlenecks and emergent local crowding.
- STEAM operates as a lightweight test-time mechanism that preserves the original Local Policy architecture while significantly enhancing success rates and coordination efficiency in dense environments.

---

[MemConflict: Evaluating Long-Term Memory Systems Under Memory Conflicts](http://arxiv.org/abs/2605.20926)

- MemConflict: introduces a diagnostic framework for evaluating long-term memory systems by treating memory validity as a query-conditioned fitness-for-use problem across dynamic, static, and conditional conflict types.
- The framework simulates long-horizon multi-session histories to test whether LLMs can retrieve and rank valid memory items while resisting interference from semantically similar distractors.
- MemConflict employs a two-level evaluation protocol that combines black-box answer assessment with white-box analysis of memory retrieval and ranking to identify specific failure patterns in long-term memory systems.

---

[Causal Past Logic for Runtime Verification of Distributed LLM Agent Workflows](http://arxiv.org/abs/2605.20923)

- CPL (Causal Past Logic): introduces a past-time temporal logic for source-level guards in distributed LLM agent workflows, enabling runtime verification based on causally visible events rather than sequential logs.
- The framework utilizes vector clocks and latest-value views to allow agents to evaluate temporal conditions locally, ensuring that decisions are made based on the most recent causally relevant information.
- By integrating runtime verification directly into the coordination language, the approach allows workflows to dynamically adapt control flow at decision points based on verified causal-temporal facts.

---

[SubTGraph: Large-Scale Subterranean Environment Synthesis with Controllable Topological Variability for Robotic Autonomy Validation](http://arxiv.org/abs/2605.20917)

- SubTGraph: introduces a procedural framework for synthesizing multi-level subterranean environments by utilizing User Requirements, Constraint Node Distribution, Dijkstra Path Estimation, Mesh Association, Occupancy Matrix, and DARPA SubT Assets to enable rigorous statistical validation of robotic autonomy.
- The framework generates diverse underground worlds by mapping user-defined topological constraints into a 3D occupancy matrix, which is subsequently populated with mesh assets to create physics-simulator-agnostic environments.
- SubTGraph facilitates the benchmarking of robotic stacks, including structural semantic segmentation, multi-agent path planning, and LIO SLAM, by providing a dataset of 150 highly variable subterranean worlds.

---

[For How Long Should We Be Punching? Learning Action Duration in Fighting Games](http://arxiv.org/abs/2605.20911)

- FightLadder framework: introduces a reinforcement learning approach that enables agents to dynamically learn both the optimal action and its execution duration in fighting games.
- The architecture utilizes PPO to train agents with either separated or combined policy heads to determine frame-skip values, facilitating temporal abstraction.
- Experimental results indicate that agents often favor high frame-skip values to exploit scripted bot behaviors, suggesting that temporal abstraction simplifies credit assignment in fast-paced environments.

---

[ParaCell: Paravirtualized Secure Containers with Lightweight Intra-Container Isolation and Intent-Driven Memory Management](http://arxiv.org/abs/2605.20906)

- ParaCell: introduces a paravirtualized secure container runtime that utilizes XGates for lightweight intra-container isolation and a Pager for intent-driven, fine-grained memory management.
- The architecture replaces expensive address-space switches with domain switches using MPK-based XGates and eliminates reactive secondary faults by exposing guest-kernel memory allocation intent to the host.
- ParaCell improves performance in nested-cloud and agentic workloads by reducing latency and memory overhead compared to traditional secure container runtimes.

---

[GenAI-Driven Threat Detection with Microsoft Security Copilot](http://arxiv.org/abs/2605.20896)

- DTDA: introduces an autonomous agent that continuously investigates security incidents across Microsoft Defender to uncover hidden threats and generate explainable detections using IncidentBatching, TableSelection, Expand, Aggregate, SelectEntities, EnrichTimeline, SummarizeIncident, Plan, Execute, UpdateInvestigation, and GenerateAlerts.
- The framework utilizes a bounded planner-executor loop to iteratively generate attack-specific hypotheses and gather evidence, ensuring scalable and high-precision threat discovery.
- Integrated into Microsoft Security Copilot, the system achieves 80.1% precision in production by leveraging LLMs for structured reasoning over heterogeneous enterprise telemetry.

---

[Terminal-World: Scaling Terminal-Agent Environments via Agent Skills](http://arxiv.org/abs/2605.20876)

- Terminal-World: introduces a fully automated pipeline that uses agent skills as the central synthesis primitive to jointly derive task instructions, executable environments, and teacher trajectories.
- The framework utilizes a multi-agent architecture and a GVR mechanism to ensure high-fidelity environment construction and efficient trajectory collection for training LLMs.
- Terminal-World constructs 5,723 training environments and trains a family of models that outperform existing terminal-agent baselines while requiring significantly less training data.

---

[Governance by Construction for Generalist Agents](http://arxiv.org/abs/2605.20874)

- CUGA (Computer-Using Generalist Agent): introduces a modular policy-as-code framework that enforces governance at runtime across five structural checkpoints without requiring LLM fine-tuning.
- The architecture utilizes Policy Models Layer, Storage Layer, Policy Agent Layer, and Enactment Layer to integrate Intent Guard, Playbook, Tool Guide, Tool Approval, and Output Formatter into the agent execution pipeline.
- By externalizing governance into composable runtime policies, the system ensures predictable, auditable, and compliance-aware behavior for enterprise LLMs across complex, multi-step workflows.

---

[ProCrit: Self-Elicited Multi-Perspective Reasoning with Critic-Guided Revision for Multimodal Sarcasm Detection](http://arxiv.org/abs/2605.20867)

- ProCrit: introduces a two-agent framework that reformulates multimodal sarcasm detection from manually prescribed perspectives to self-elicited multi-perspective reasoning, utilizing a Proposal Agent and a Critic Agent.
- The framework employs a dynamic-role agentic rollout to synthesize process-level reasoning annotations, which are then used to train the Proposal Agent via a draft–critique–revise paradigm.
- A mutual-refinement training strategy optimizes both agents through dual-stage reinforcement learning, where the Critic Agent provides targeted natural-language feedback to improve the Proposal Agent's reasoning reliability.

---

[MEMGYM: a Long-Horizon Memory Environment for LLM Agents](http://arxiv.org/abs/2605.20833)

- MemGym: introduces a unified evaluation and training framework for LLM agents that decouples memory performance from reasoning, retrieval, and tool-use capabilities using memory-isolated scoring.
- The framework utilizes a shared memory interface across five agentic regimes and employs MemRM, a lightweight reward model, to replace expensive environment rollouts with sub-second scalar evaluations.
- MemGym provides controllable, ablation-verified synthetic pipelines and a paired-trajectory corpus to enable systematic research into long-horizon memory formation and policy optimization.

---

[Demo-JEPA: Joint-Embedding Predictive Architecture for One-shot Cross-Embodiment Imitation](http://arxiv.org/abs/2605.20811)

- Demo-JEPA: introduces a cross-embodiment imitation framework that decouples demonstration intent from embodiment-specific execution by translating source visual demonstrations into target-compatible future latent trajectories.
- The framework utilizes a JEPA-based world model and an embodiment-aware Dreamer Predictor to infer latent subgoals, which guide action generation through iterative planning in the target agent's learned dynamics.
- Demo-JEPA avoids action-level correspondence and enables robust zero-shot generalization across heterogeneous robotic embodiments by performing goal-conditioned planning in a shared predictive latent space.

---

[Q-SpiRL: Quantum Spiking Reinforcement Learning for Adaptive Robot Navigation](http://arxiv.org/abs/2605.20801)

- Q-SpiRL: introduces a quantum-enhanced spiking reinforcement learning framework for obstacle-aware robot navigation that integrates a variational quantum circuit into a spiking pipeline to improve policy quality.
- The framework evaluates five agent families—tabular Q-learning, classical MLP, classical SNN, QMLP, and QSNN—under a unified training and deterministic evaluation protocol to isolate the impact of quantum-enhanced spiking architectures.
- Experimental results demonstrate that the QSNN provides the most favorable trade-off between task completion, trajectory efficiency, and motion smoothness across varying environment scales, with initial feasibility verified on IBM quantum hardware.

---

[Interaction Locality in Hierarchical Recursive Reasoning](http://arxiv.org/abs/2605.20784)

- Interaction Locality Framework: introduces a task-geometry-aware interpretability method for measuring information flow in recursive and embodied spatial reasoning models using SAE (feature ablation), finite-noise activation patching (causal reach), and structural Jacobian/attention checks (linearized topology).
- The framework evaluates HRM (Hierarchical Reasoning Model), TRM (Tiny Recursive Model), and MTU3D (3D embodied scene-grounding model) to determine if internal state updates remain local to task-defined segments or propagate globally.
- Results demonstrate that H-level states in recursive models often exhibit local write patterns within cycles, while cross-cycle propagation enables broader information flow, and MTU3D shows a dissociation between structural spatial bias and causal locality.

---

[The Illusion of Intervention: Your LLM-Simulated Experiment is an Observational Study](http://arxiv.org/abs/2605.20767)

- Synthetic User Experiment Framework: introduces a methodology to diagnose and mitigate selection bias in LLM-simulated experiments by treating them as observational studies rather than randomized trials.
- The framework utilizes negative control outcomes to detect user drift, where latent attributes shift in response to interventions, and employs iterative confounder elicitation to stabilize simulated populations.
- By adjusting persona specifications with targeted confounders, the approach reduces confounding bias and improves the internal validity of synthetic experiments across various LLM-based survey and agent evaluation settings.

---

[Hack-Verifiable Environments: Towards Evaluating Reward Hacking at Scale](http://arxiv.org/abs/2605.20744)

- Hack-Verifiable TextArena: introduces a benchmark suite that embeds detectable reward hacking opportunities into environments to enable deterministic and automated measurement of agent misalignment.
- The framework utilizes a modular wrapper to expose controlled vulnerabilities, such as hidden solution files or logical bugs, allowing for the systematic study of reward hacking across diverse LLM-based agents.
- Empirical analysis demonstrates that reward hacking is an emergent, addictive behavior that increases with task difficulty and persists across agent trajectories despite instruction-based suppression.

---

[Draw2Think: Harnessing Geometry Reasoning through Constraint Engine Interaction](http://arxiv.org/abs/2605.20743)

- Draw2Think: introduces a constraint-agentic harness that recasts geometric reasoning as an inference-time closed loop around a constraint engine, utilizing a Frozen VLM, GeoGebra constraint engine, Typed ToolSpecs, Structured canvas memory, Revisable state, and a Propose-Draw-Verify loop.
- The framework externalizes geometric hypotheses onto an executable canvas, allowing the model to perform exact measurements and receive structured feedback, thereby grounding reasoning in verified engine states.
- By delegating geometric verification to an external oracle, Draw2Think improves outcome accuracy on complex geometry benchmarks while maintaining training-free operation.

---

[Application-Layer Dual Memory for Conversational AI: Achieving Virtually Unbounded Context Without Model Modification](http://arxiv.org/abs/2605.20724)

- CALMem: introduces an application-layer dual memory architecture that provides LLMs with virtually unbounded context without requiring model modifications.
- The system integrates episodic memory for sequential conversation history and semantic memory for structured facts, managed by a token-budget-adaptive injection mechanism.
- CALMem enables intra-session retrieval of compacted context and cross-session factual recall, significantly reducing API costs while maintaining performance in production environments.

---

[Heartbeat-Bound Hierarchical Credentials: Cryptographic Revocation for AI Agent Swarms](http://arxiv.org/abs/2605.20704)

- HBHC (Heartbeat-Bound Hierarchical Credentials): introduces a cryptographic protocol that enforces deterministic agent termination by binding credential validity to periodic parent liveness proofs.
- The framework utilizes hierarchical deterministic key derivation to ensure that child agents cannot authenticate without a fresh heartbeat from their parent, enabling offline revocation.
- HBHC provides a fail-secure mechanism that reduces the zombie agent window by 90x compared to standard OAuth 2.0, ensuring that revoked agents lose access within a bounded time frame.

---

[Declarative Data Services: Structured Agentic Discovery for Composing Data Systems](http://arxiv.org/abs/2605.20690)

- DDS: introduces a structured agentic discovery framework that decomposes multi-system data backend composition into four typed contracts (L1–L4) to enable reliable, iterative system design.
- The framework utilizes L1 Intent Contract, L2 Operator DAG, L3 Skill Contract, and L4 Attribution Loop to ensure that LLMs operate within bounded, validatable, and citable search spaces.
- By routing runtime failures back to the specific layer that owns the violated decision, DDS enables persistent learning through skill patches rather than relying on unbounded, repetitive LLM prompting.

---

[IndusAgent: Reinforcing Open-Vocabulary Industrial Anomaly Detection with Agentic Tools](http://arxiv.org/abs/2605.20682)

- IndusAgent: introduces a tool-augmented agentic framework for open-vocabulary industrial anomaly detection that synergizes domain-specific reasoning with autonomous tool orchestration.
- The framework utilizes Indus-CoT for supervised fine-tuning of the Qwen3-VL-8B model and employs an accuracy-gated reinforcement learning objective to optimize tool usage and diagnostic accuracy.
- By integrating dynamic region cropping, normalcy prior retrieval, texture enhancement, and geometric verification, the agent effectively resolves visual ambiguities and structural hallucinations in zero-shot industrial inspection.

---

[Evaluating Temporal Semantic Caching and Workflow Optimization in Agentic Plan-Execute Pipelines](http://arxiv.org/abs/2605.20630)

- AOB Optimization Framework: introduces a two-layer optimization approach for agentic plan-execute pipelines, combining a temporal semantic cache with MCP workflow optimizations to reduce latency in industrial asset operations.
- The framework utilizes a Temporal Classifier to route queries into volatile, static, relative, or anchored buckets, ensuring that cached answers remain valid despite temporal or parameter-based shifts.
- The MCP workflow layer improves efficiency by implementing a Discovery Cache to eliminate redundant subprocess spawning and a Persistent MCPServerPool to enable dependency-aware parallel execution of plan steps.

---

[Auto-Dreamer: Learning Offline Memory Consolidation for Language Agents](http://arxiv.org/abs/2605.20616)

- Auto-Dreamer: introduces a two-timescale memory system that decouples fast per-session acquisition from slow, learned offline cross-session consolidation.
- The framework utilizes a learned consolidator that performs provenance-grounded region rewriting to synthesize compact, abstract memory entries from raw source trajectories.
- Auto-Dreamer is trained via GRPO with a counterfactual utility reward to optimize memory for downstream task performance while minimizing redundancy.

---

[Compositional Transduction with Latent Analogies for Offline Goal-Conditioned Reinforcement Learning](http://arxiv.org/abs/2605.20609)

- CTA: introduces a hierarchical offline GCRL framework that synthesizes goal-reaching behaviors by composing task-endogenous analogies with task-exogenous contexts using Dual Analogy, Anchor Module, Displacement Module, Projection Network, High-level Policy, Low-level Policy, Bilinear Transduction, and Temporal Distance Difference Field.
- The framework leverages Bilinear Transduction to enable robust out-of-combination generalization by decoupling task-exogenous context from task-endogenous displacement.
- CTA utilizes a hierarchical policy structure where the high-level agent proposes k-step analogies and the low-level agent executes primitive actions, significantly improving performance on OGBench manipulation tasks.

---

[From Automated to Autonomous: Hierarchical Agent-native Network Architecture (HANA)](http://arxiv.org/abs/2605.20608)

- HANA: introduces a hierarchical multi-agent architecture that decouples strategic cognition from execution to enable autonomous network management.
- The framework utilizes a Dual-Driven Orchestrator that integrates "slow" deliberative strategic planning with "fast" reflexive fault recovery to maintain operational resilience.
- By embedding self-awareness and leveraging shared memory, the architecture enables proactive service assurance and rapid self-healing in 5G core environments.

---

[Mahjax: A GPU-Accelerated Mahjong Simulator for Reinforcement Learning in JAX](http://arxiv.org/abs/2605.20577)

- Mahjax: introduces a fully vectorized Riichi Mahjong environment implemented in JAX to enable large-scale parallelization on GPUs for reinforcement learning research.
- The framework utilizes JAX-based Environment, State Dataclass, Vectorized Logic, and Caching to overcome computational bottlenecks inherent in traditional CPU-based Mahjong simulators.
- The architecture supports Transformer Encoder and MLP Heads for policy training, validated through PPO Agent performance improvements against baseline policies.

---

[Toward AI VIS Co-Scientists: A General and End-to-End Agent Harness for Solving Complex Data Visualization Tasks](http://arxiv.org/abs/2605.21825)

- VIS co-scientist: introduces an end-to-end agentic harness that autonomously designs functional visualization applications by orchestrating specialized subagents including an Orchestrator (Code Agent), Exploratory Data Analyzer (EDA), Planner, Environment Builder, VIS Designer, Evaluator, and Memory Maintainer.
- The framework utilizes a closed-loop design process where the Orchestrator manages subagents and custom skills, such as Playwright-MCP and Scratchbook Skill, to iteratively refine visualization applications based on task-driven validation.
- This system addresses the complexity of scientific visualization by automating data profiling, environment configuration, and interactive interface implementation, while maintaining a hierarchical memory system for future knowledge retrieval.

---

[Quality-Assured Fuzz Harness Generation via the Four Principles Framework](http://arxiv.org/abs/2605.21824)

- QuartetFuzz: introduces an autonomous harness-generation system that systematically improves harness correctness through a four-stage pipeline of Logic Group selection, API protocol research, static-driven build, and adversarial validation.
- The framework enforces four source-level correctness principles (P1–P4) using an LLM-agent pipeline that includes Logic Group-, API Research- and Harness Generator-agents.
- By integrating adversarial probing and static call-graph analysis, the system identifies and repairs harness-induced quality violations before fuzzing, significantly reducing false-positive crash reports.

---

[Implicit Safety Alignment from Crowd Preferences](http://arxiv.org/abs/2605.21822)

- Safe Crowd Preference-based Reinforcement Learning: introduces a hierarchical framework that extracts safety-aligned skills from crowd preferences and composes them via a high-level policy to safely solve downstream tasks.
- The framework utilizes a VAE-based encoder to infer latent user contexts from crowd preferences, enabling the learning of latent-conditioned low-level policies that inherently encode shared safety criteria.
- By composing these preference-aligned skills, the high-level policy optimizes downstream task rewards while maintaining safety without requiring explicit safety signals or manual reward design.

---

[Co-Ontogeny by Archetypal Scaffolding: The Humorphic Partnership: An Architecture and a Four-Month Trace of a Self-Modeling Personal Agent and Its Author](http://arxiv.org/abs/2605.21818)

- myalicia: introduces a humorphic partnership architecture where human-AI dyads maintain bidirectional, vault-visible self-models within a shared recursive memory substrate to facilitate co-ontogeny.
- The framework utilizes three depths of attention (Listen, Notice, Know) and three orders of reflexion to enable an LLM-based agent to perform honest self-decline detection and constitutional evolution.
- The system employs archetypal scaffolding as a functional mechanism to partition interpretive stances, ensuring long-term relational coherence and auditable symbolic continuity.

---

[Trace2Skill: Verifier-Guided Skill Evolution for Long-Context EDA Agents](http://arxiv.org/abs/2605.21810)

- Trace2Skill: introduces a test-time scaling framework that evolves natural-language skills for LLM agents in complex Verilog design tasks without requiring model fine-tuning.
- The framework utilizes an oracle-mutator-selector loop to convert execution traces into task-specific skills, supported by dense verifier feedback to guide agent behavior.
- By co-optimizing skill text and agent execution through granular metrics like SkillQ and AgentProgressQ, the system enables LLMs to solve hard hardware design tasks that defeat baseline agents.

---

[stable-worldmodel: A Platform for Reproducible World Modeling Research and Evaluation](http://arxiv.org/abs/2605.21800)

- swm (stable-worldmodel): introduces a unified, open-source platform designed to standardize the world modeling research pipeline by integrating World, Policy, Solver, Dataset API, Factors of Variation (FoV), and Visual Wrappers.
- The framework addresses research fragmentation by providing high-performance data loading via Lance, modular planning solvers, and a comprehensive benchmarking suite for systematic robustness and zero-shot generalization evaluation.
- By decoupling model training from evaluation, swm enables researchers to conduct reproducible experiments across diverse environments while mitigating common bottlenecks in data throughput and implementation consistency.

---

[Energy per Successful Goal: Goal-Level Energy Accounting for Agentic AI Systems](http://arxiv.org/abs/2605.22883)

- A-LEMS (Agentic LLM Energy Measurement System): introduces a cross-layer measurement framework that redefines AI energy accounting from energy-per-inference to Energy per Successful Goal (EpG) by utilizing a temporal boundary model, a five-layer observation pipeline, and a three-hash reproducibility protocol.
- The framework formalizes energy attribution through the Orchestration Overhead Index (OOI), which isolates the energy cost of multi-step orchestration relative to linear execution.
- Empirical validation across reasoning and tool-augmented tasks demonstrates that agentic workflows consume significantly higher energy due to orchestration structure rather than increased LLM compute.

---

[Residual Skill Optimization for Text-to-SQL Ensembles](http://arxiv.org/abs/2605.21792)

- DIVSKILL-SQL: introduces a residual skill optimization framework that constructs complementary Text-to-SQL ensembles by iteratively refining agent skills on unresolved training examples.
- The framework utilizes a Skill Learning Stage to build a diverse Skill Seed Pool and an Inference Stage that employs a Pairwise LLM Judge to select the most accurate SQL candidate from multiple skill-conditioned agent trajectories.
- By optimizing each new skill to specifically target the residual failures of previous skills, the approach improves Pass@K coverage and reduces redundant agent behavior without requiring model fine-tuning.

---

[FuzzingBrain V2: A Multi-Agent LLM System for Automated Vulnerability Discovery and Reproduction](http://arxiv.org/abs/2605.21779)

- FuzzingBrain V2: introduces a multi-agent system that combines LLM-driven semantic analysis with coverage-guided fuzzing to automate vulnerability discovery and reproduction.
- The system utilizes a Suspicious Point abstraction to bridge LLM-based analysis with fuzzing-based verification, enabling precise localization and systematic reproduction of vulnerabilities.
- FuzzingBrain V2 employs a hierarchical search strategy with dual-layer fuzzing and specialized LLM agents to reason about complex cross-function dependencies and deep code paths.

---

[Memory-R2: Fair Credit Assignment for Long-Horizon Memory-Augmented LLM Agents](http://arxiv.org/abs/2605.21768)

- Memory-R2: introduces a training framework for long-horizon memory-augmented LLM agents that utilizes a Shared LLM Backbone, Fact Extractor, Memory Manager, Memory Bank, LoGo-GRPO, Global Branch, Local Branch, and Curriculum Learning to enable fair credit assignment and stable memory evolution.
- The framework employs LoGo-GRPO to combine global trajectory-level rewards with local rerollouts from shared intermediate memory states, effectively mitigating credit-assignment bias in multi-session environments.
- Memory-R2 optimizes the memory lifecycle through a shared-parameter co-learning design and a progressive curriculum that scales training from short to long-horizon sessions.

---

[Mind the Gaps: Multi-Robot Feedback-Driven Ergodic Coverage in Unknown Environments](http://arxiv.org/abs/2605.21719)

- Feedback-Driven Ergodic Coverage framework: introduces an adaptive coverage strategy that utilizes real-time feedback from an Environmental Model to adjust robot sampling behavior in unknown environments.
- The approach employs an Adaptive Parameter Update mechanism to refine the Environmental Model, which informs the Ergodic Trajectory Planner to prioritize regions of high interest.
- A Feedback Controller minimizes the discrepancy between the Multi-Robot System's spatial distribution and the estimated target density, ensuring persistent and efficient environmental monitoring.

---

[PocketAgents: A Manifest-Driven Library of Autonomous Defense Agents](http://arxiv.org/abs/2605.21694)

- PocketAgents: introduces a manifest-driven library of autonomous defense agents that separates investigation from enforcement using a typed boundary to ensure accountability.
- The framework utilizes a shared runtime where an untrusted LLM-based agent investigation is constrained by a manifest, context, and bounded telemetry access before any action is admitted.
- By enforcing a deterministic contract on agent outputs, the architecture enables measurable, extensible, and attributable defensive actions within a closed-loop cyber-deception environment.

---

[Flying Together: Human-Guided Immersive Shared Control for Aerial Robot Teams in Unknown Environments](http://arxiv.org/abs/2605.21680)

- VR-based shared control framework: introduces a system for teleoperating aerial robot teams in unknown environments by integrating a motion-primitive-based planner, a Variable Admittance Controller, and a WebXR immersive interface.
- The framework enables real-time human guidance through a virtual marker, allowing operators to influence robot trajectories while maintaining safety and inter-agent cohesion.
- Experimental results demonstrate that this shared control approach improves obstacle clearance and reduces operator effort compared to fully autonomous navigation.

---

[TO-AGENTS: A MULTI-AGENT AI PIPELINE FOR PREFERENCE-GUIDED TOPOLOGY OPTIMIZATION](http://arxiv.org/abs/2605.21622)

- TO-Agents: introduces a multi-agent AI framework that automates topology optimization by iteratively refining solver parameters based on natural-language design intent and visual feedback.
- The architecture coordinates specialized agents including a Pydantic-based input validator, a topology optimization solver, a vision-language reasoning agent, an independent judge agent, and a manufacturing post-processor.
- The system demonstrates autonomous long-horizon design refinement by leveraging history-conditioned adaptation and visual perception to align structural outcomes with qualitative aesthetic preferences.

---

[SciAtlas: A Large-Scale Knowledge Graph for Automated Scientific Research](http://arxiv.org/abs/2605.22878)

- SciAtlas: introduces a large-scale, multi-disciplinary, heterogeneous academic knowledge graph designed as a panoramic scientific evolution network to provide a structured topological cognitive substrate for AI agents.
- The framework utilizes a neuro-symbolic retrieval algorithm featuring tri-path collaborative recall and graph reranking to achieve a seamless transition from semantic matching to deterministic association discovery.
- SciAtlas integrates 9 entity types and 12 relational edges across 43M papers to empower automated scientific research tasks including literature review, idea grounding, and research trend prediction.

---

[AutoMCU: Feasibility-First MCU Neural Network Customization via LLM-based Multi-Agent Systems](http://arxiv.org/abs/2605.21560)

- AutoMCU: introduces a feasibility-first LLM-based multi-agent system for automated neural network customization under strict microcontroller constraints.
- The framework utilizes a Supervisor Agent to coordinate specialized agents, including a Proposal Agent, Training Agent, and Evaluation and Conversion Agent, to ensure efficient and stable model discovery.
- By integrating backend-verified feasibility checks into a closed-loop workflow, AutoMCU eliminates infeasible candidates early and optimizes for deployable models on resource-constrained hardware.

---

[Articulate but Wrong: Self-Review Failures in LLM-Based Code Modernization](http://arxiv.org/abs/2605.21537)

- Code Modernization Self-Review Framework: evaluates the reliability of LLMs in identifying silent behavioral drift during legacy code migration by comparing generated outputs against a type-strict behavioral oracle.
- The framework utilizes a balanced corpus of legacy Python-2 snippets to measure how frequently LLMs produce code that compiles but fails to preserve original semantics.
- The study demonstrates that self-review is an unreliable safety net, as models often exhibit bimodal behavior, frequently endorsing their own incorrect code even when they correctly articulate the underlying semantic differences.

---

[RMA: an Agentic System for Research-Level Mathematical Problems](http://arxiv.org/abs/2605.22875)

- RMA: introduces an agentic framework for research-level mathematical reasoning that decomposes proof construction into specialized modules for analysis, literature grounding, knowledge retrieval, and iterative verification.
- The system utilizes a multi-agent architecture with an Initializer, Proposer agents, and Verifier agents that interact through a shared structured memory to iteratively refine candidate proofs.
- RMA incorporates a Fair Comparison Module to ensure rigorous evaluation by preventing access to existing solutions and enforcing strict temporal and context-isolation controls.

---

#### 19th May 2026


[OSCToM: RL-Guided Adversarial Generation for High-Order Theory of Mind](http://arxiv.org/abs/2605.20423)

- OSCToM (Observer-Self Conflict Theory of Mind): introduces an adversarial framework that uses reinforcement learning and a compositional surrogate pipeline to generate complex nested belief conflicts for training LLMs.
- The framework employs a DQN-based generator to navigate an extended domain-specific language, creating high-order Theory of Mind scenarios that are then used to fine-tune LLMs via a two-stage curriculum.
- By replacing heuristic search with surrogate-guided optimization, the approach achieves a 5.7x reduction in inference latency while significantly improving performance on information-asymmetric benchmarks.

---

[AutoResearchClaw: Self-Reinforcing Autonomous Research with Human-AI Collaboration](http://arxiv.org/abs/2605.20025)

- AutoResearchClaw: introduces a multi-agent autonomous research pipeline that integrates structured multi-agent debate, a self-healing executor with a Pivot/Refine decision loop, verifiable result reporting, human-in-the-loop collaboration, and a cross-run evolution system to improve scientific discovery.
- The framework utilizes a multi-agent debate panel for hypothesis generation and result analysis, while the self-healing executor treats experiment failures as diagnostic information to enable iterative progress.
- By employing a numeric registry and a four-layer citation verification pipeline, the system ensures that all generated research outputs are grounded in actual experimental evidence and verified references.

---

[When Skills Don’t Help: A Negative Result on Procedural Knowledge for Tool-Grounded Agents in Offensive Cybersecurity](http://arxiv.org/abs/2605.20023)

- Agent Skills framework: introduces a re-analysis of an autonomous CTF agent to evaluate how procedural knowledge impacts performance in high-feedback environments.
- The study demonstrates that the marginal benefit of Agent Skills is inversely related to the bandwidth of deterministic environment feedback provided to the LLM agent.
- The research suggests that in environments with rich, schema-validated tool feedback, procedural knowledge acts as a compensatory layer rather than a universally beneficial component.

---


[CopT: Contrastive On-Policy Thinking with Continuous Spaces for General and Agentic Reasoning](http://arxiv.org/abs/2605.20075)

- CopT: introduces a training-free reasoning pipeline that reverses the standard thinking-before-answering order by generating a draft answer first and invoking on-policy thinking only when the draft is deemed unreliable by a contrastive reliability estimator.
- The framework utilizes a sequence-level reverse KL estimator to measure answer-relevant uncertainty by contrasting model support for tokens under discrete-token versus continuous-embedding inputs.
- CopT dynamically modulates the visibility of the draft answer during subsequent thinking chunks to preserve useful information while mitigating the risk of being misled by unreliable content.

---



[MSAVBench: Towards Comprehensive and Reliable Evaluation of Multi-Shot Audio-Video Generation](http://arxiv.org/abs/2605.20183)

- MSAVBench: introduces a comprehensive benchmark and adaptive hybrid evaluation framework for multi-shot audio-video generation, utilizing TransNet V2, Qwen3.5, Specialized Expert Models, Instance-Wise Rubric-Based Scoring, and Tool-Grounded Agentic Scoring.
- The framework employs an agentic pre-processing phase with iterative shot self-correction to mitigate segmentation errors before applying stratified scoring paradigms.
- The evaluation suite achieves high alignment with human judgments by combining expert models, rubric-based VLM scoring, and tool-grounded evidence extraction to assess complex multi-shot narratives.

---

[ClinSeekAgent: Automating Multimodal Evidence Seeking for Agentic Clinical Reasoning](http://arxiv.org/abs/2605.20176)

- ClinSeekAgent: introduces an automated agentic framework for dynamic multimodal evidence seeking that shifts clinical reasoning from passive consumption to active acquisition of information from heterogeneous sources.
- The framework utilizes a unified tool space comprising EHR retrieval, web search, and medical image analysis to ground clinical decisions in patient-specific data and external knowledge.
- ClinSeekAgent serves as both an inference-time pipeline for frontier LLMs and a training-time distillation method to transfer long-horizon evidence-seeking behaviors into compact open-source models.

---

[A Methodology for Selecting and Composing Runtime Architecture Patterns for Production LLM Agents](http://arxiv.org/abs/2605.20173)

- SDB (Stochastic-Deterministic Boundary) Framework: introduces a four-part contract—Proposer, Verifier, Commit step, and Reject signal—to govern the transition from stochastic LLM outputs to deterministic system actions.
- The framework organizes production agent runtimes into three orthogonal concerns—Coordination, State, and Control—and provides a catalog of six architectural patterns to address them.
- A five-step selection methodology enables practitioners to map specific workload requirements to appropriate patterns, ensuring long-run reliability through architectural momentum rather than reliance on per-call model quality.

---

[What Do Evolutionary Coding Agents Evolve?](http://arxiv.org/abs/2605.20086)

- EvoTrace and EvoReplay: introduces a dataset and diagnostic methodology to analyze the internal dynamics of LLM-driven evolutionary coding agents by reconstructing search states and performing controlled interventions.
- The research characterizes evolutionary coding agents as dynamic systems, revealing that a significant portion of search budget is spent on re-introducing previously discarded code and that many performance gains are parametric rather than structural.
- The study demonstrates that late-stage evolutionary improvements on mathematical tasks are often recoverable through post-hoc hyperparameter tuning, suggesting a need for more diagnostic evaluation beyond final benchmark scores.

---

[Probing Embodied LLMs: When Higher Observation Fidelity Hurts Problem Solving](http://arxiv.org/abs/2605.20072)

- Embodied LLM Probing Framework: introduces a behavioral methodology to evaluate LLM-based robotic agents by systematically varying observation fidelity in a closed-loop mechanical puzzle task.
- The study demonstrates that higher observation fidelity can paradoxically degrade performance by sustaining repetitive action loops, whereas moderate observational noise improves success rates by disrupting these failure modes.
- The research highlights that measured performance in embodied systems often reflects an interplay between perceptual errors and reasoning failures rather than pure problem-solving capability.

---

[Rewarding Beliefs, Not Actions: Consistency-Guided Credit Assignment for Long-Horizon Agents](http://arxiv.org/abs/2605.20061)

- ReBel (Reward-Belief): introduces a process-level reinforcement learning framework that improves LLM agents in partially observable environments by explicitly modeling structured belief states to guide policy learning.
- The framework utilizes belief-consistency supervision to convert discrepancies between predicted beliefs and environment observations into dense feedback, effectively mitigating belief drift.
- By employing belief-aware grouping, ReBel enables stable step-wise advantage estimation, significantly enhancing sample efficiency and task success in long-horizon interactive settings.

---

[Towards LLM-Assisted Architecture Recovery for Real-World ROS 2 Systems: An Agent-Based Multi-Level Approach to Hierarchical Structural Architecture Reconstruction](http://arxiv.org/abs/2605.20055)

- Blueprint-guided LLM-assisted architecture recovery framework: introduces a staged recovery pipeline that utilizes explicit intermediate JSON artifacts to improve the consistency and controllability of hierarchical ROS 2 architecture reconstruction.
- The approach decomposes the recovery process into specialized agents, including NodeAnalyzer, ComponentArchitectureTeam, and SystemArchitectureTeam, to incrementally synthesize architecture models from heterogeneous repository artifacts.
- By incorporating domain-specific architectural contracts and structured prompt engineering, the framework achieves high-fidelity reconstruction of complex ROS 2 systems while reducing LLM-based hallucinations.

---

[Hunting Vulnerability Variants in AI Infra: Measurement and Reference-Driven Detection](http://arxiv.org/abs/2605.20051)

- INFRASCOPE: introduces a reference-driven multi-agent framework that extracts transferable vulnerability semantics from known cases to locate and validate variants in new AI infra repositories.
- The framework coordinates a semantic modeling agent, an inspection agent, and a target-side verification agent to perform localized, audit-ready vulnerability detection.
- INFRASCOPE utilizes localization-aware state management and dynamic PoC generation to maintain auditability and reduce LLM hallucinations during cross-repository variant detection.

---

[Does Code Cleanliness Affect Coding Agents? A Controlled Minimal-Pair Study](http://arxiv.org/abs/2605.20049)

- Harbor Framework: introduces a controlled evaluation protocol using minimal-pair repositories to isolate the impact of code cleanliness on the operational footprint of LLMs.
- The study demonstrates that while code cleanliness does not significantly alter task pass rates for LLMs, it substantially reduces token consumption and file revisitation frequency.
- The research highlights that maintainability principles remain critical for LLM efficiency, as cleaner code structures facilitate more precise navigation and lower computational costs for coding agents.

---

[When Critics Disagree: Adaptive Reward Poisoning Attacks in RIS-Aided Wireless Control System](http://arxiv.org/abs/2605.20037)

- DGRP: introduces a state-adaptive reward poisoning attack that targets a SAC Agent by corrupting rewards specifically when its Twin Critics exhibit high disagreement, indicating high-uncertainty states.
- The framework utilizes a Rolling Buffer to track critic disagreement and dynamically determine eligibility for reward corruption, ensuring the attack remains sparse and stealthy while maximizing impact on the Stochastic Actor.
- By focusing on high-leverage states where the SAC Agent is most sensitive, DGRP induces more significant and persistent performance degradation in RIS-aided wireless control systems than traditional timing-based or exploration-triggered attacks.

---

[A Case for Agentic Tuning: From Documentation to Action in PostgreSQL](http://arxiv.org/abs/2605.19988)

- PERFEVOLVE: introduces a methodology that transforms static system documentation into executable procedural knowledge to empower LLM-based agents for autonomous performance tuning.
- The framework utilizes Offline Profiling to perform Sensitivity Scan, Correlation Screen, and Joint Optimization, generating a Skill DAG that guides the LLM-based Tuning Agent during Online Deployment.
- By shifting from static recommendations to dynamic, process-oriented tuning, PERFEVOLVE effectively addresses staleness, context insensitivity, and parameter correlation issues in complex systems like PostgreSQL.

---

[A conceptual framework for learning to listen by reward: Curiosity-driven search for novel sources](http://arxiv.org/abs/2605.19984)

- Conceptual Framework for Learning to Listen by Reward: introduces a reinforcement learning paradigm where embodied agents navigate environments to discover novel sound sources using only auditory input.
- The framework utilizes a Q-network architecture, comparing memoryless CNN6 encoders against stateful CNN-Transformer models to process auditory streams and select optimal navigation actions.
- Experimental results demonstrate that stateful agents significantly outperform memoryless baselines in reachability and reward accumulation by leveraging temporal dependencies in audio-based navigation tasks.

---

[Equilibria in Multiplayer Graph Games: An Algorithmic Study](http://arxiv.org/abs/2605.19954)

- Equilibria in Multiplayer Graph Games: introduces a comprehensive algorithmic study of equilibrium concepts in multiplayer graph games, focusing on the complexity of the constrained existence problem for Nash, subgame-perfect, strong secure, and risk-sensitive equilibria.
- The paper establishes NP-completeness for SPEs in parity and mean-payoff games using a novel negotiation function and provides complexity results for rational verification and synthesis.
- It further introduces extreme risk-sensitive equilibria to address undecidability in stochastic games, proving that these extreme cases yield decidable fragments for the constrained existence problem.

---

[Rethinking How to Remember: Beyond Atomic Facts in Lifelong LLM Agent Memory](http://arxiv.org/abs/2605.19952)

- TriMem (Tri-Granularity Memory): introduces a three-level memory architecture that maintains coexisting representations of raw dialogue segments, extracted atomic facts, and synthesized entity profiles to enhance long-term interaction reliability.
- The framework utilizes traceable source identifiers to ensure storage fidelity and an incremental profile module to support deep reasoning over scattered historical information.
- TriMem employs TextGrad-based prompt optimization to iteratively refine extraction and profiling prompts based on response quality feedback, achieving lifelong evolution without requiring parameter updates.

---

[PEEK: Context Map as an Orientation Cache for Long-Context LLM Agents](http://arxiv.org/abs/2605.19932)

- PEEK: introduces a system that maintains a persistent, constant-sized context map to cache reusable orientation knowledge for LLM agents operating over recurring external contexts.
- The framework utilizes a Distiller to extract transferable knowledge, a Cartographer to apply structured edits, and an Evictor to manage a fixed token budget for the context map.
- PEEK improves long-context reasoning and information aggregation performance while reducing iteration counts and costs compared to existing prompt-learning and context-management approaches.

---

[JAXenstein: Accelerated Benchmarking for First-Person Environments](http://arxiv.org/abs/2605.19926)

- JAXenstein: introduces a lightweight, JAX-native benchmark suite for first-person reinforcement learning tasks that leverages a DDA ray casting engine for high-speed, GPU-accelerated environment simulation.
- The framework utilizes JAX-based just-in-time compilation and vectorized mapping to achieve significant speed improvements over traditional simulators like ViZDoom and MiniWorld.
- JAXenstein supports flexible environment definition via ASCII maps and provides a standardized testbed for evaluating recurrent PPO agents with advanced exploration strategies like RND and ICM.

---

[LLM Agents Make Collective Belief Dynamics Programmable: Challenges and Research Directions](http://arxiv.org/abs/2605.19915)

- Programmable Collective Belief Control: introduces a framework for analyzing how coordinated LLM agents can systematically steer population-level belief dynamics through controlled multi-agent simulations.
- The research identifies four structural properties—indistinguishability, persistence, contextuality, and configurability—that enable covert and effective manipulation of collective beliefs.
- The paper outlines a research agenda focusing on theoretical foundations, operational detection methods, and scalable simulation infrastructure to address the risks of programmable belief control.

---

[From Role to Person: Trust Calibration Challenges in Twin Agents](http://arxiv.org/abs/2605.19838)

- Twin Agents: introduces a conceptual framework for social AI agents that represent the communicative and epistemic profile of specific individuals, identifying a threefold attribution problem consisting of a Schema Gap, an Epistemic Gap, and a Model Artifact.
- The paper argues that twin agents dissolve the boundary between AI and human decision-makers, rendering traditional cognitive forcing functions ineffective for trust calibration.
- The authors propose that future research must focus on relational-level interventions and epistemic provenance to address the structural ambiguity inherent in person-specific agent simulations.

---

[A Closed-loop, State-centric, Multi-agent Framework for Passenger Load Estimation from Heterogeneous Data Streams](http://arxiv.org/abs/2605.19834)

- Closed-loop, State-centric, Multi-agent Framework: introduces a modular architecture for robust passenger load estimation that integrates a Perception Agent, a Physical Agent, and a Trust-aware Fusion Agent to enforce physical feasibility and handle heterogeneous data streams.
- The framework utilizes a stop-level recursive state update where the Physical Agent projects unconstrained flow proposals onto a feasible space, while the Trust-aware Fusion Agent dynamically weights external anchors based on reliability.
- Residual-driven reweighting and an optional ABM layer provide additional robustness and plausibility auditing to stabilize the system against sensor drift and inconsistent data.

---

[Material for Thought: Generative AI as an Active Creative Medium](http://arxiv.org/abs/2605.19832)

- SOSS (Shape, Observe, Stir, Select) framework: introduces a theoretical model for human-AI collaboration that treats generative AI as an active creative medium rather than a passive decision-support tool, utilizing Shape, Observe, Stir, and Select, LLM-agents, Working memory, and Long-term memory.
- The framework repositions the human from an evaluator of AI output to an orchestrator of a possibility space, using Loom to manage LLM-agents through iterative cycles of disruption and curation.
- By leveraging the inherent convergence tendency of LLMs as a form of productive resistance, the approach fosters deeper metacognitive engagement and creative agency in narrative tasks.

---

[From Prompts to Pavement Through Time: Temporal Grounding in Agentic Scene-to-Plan Reasoning](http://arxiv.org/abs/2605.19824)

- Agentic Scene-to-Plan Reasoning Framework: introduces three planner architectures—Static, Sentinel, and Synthesizer—to evaluate temporal grounding in autonomous vehicle planning using Descriptor Agent, Planner Agent, Initiator Agent, Critic Agent, Refiner Agent, Sentinel Agent, and Synthesizer Agent.
- The research investigates whether temporal conditioning within inter-agent communication enhances plan coherence without degrading semantic or logical consistency.
- Empirical results demonstrate that while temporal grounding reshapes reasoning style, it yields no statistically significant improvements in standard NLP-based correctness metrics compared to non-temporal baselines.

---

[Satisfiability for Knowing How over Linear Plans is NP-complete](http://arxiv.org/abs/2605.19819)

- L(Kh): introduces a complexity analysis proving that the satisfiability problem for the logic of knowing-how is NP-complete by translating formulas into the modal logic S5.
- The framework establishes a polynomial-size model property for L(Kh), demonstrating that any satisfiable formula has a model of size polynomial in the formula's length.
- The research simplifies previous approaches by eliminating unsatisfiability checks, providing a direct reduction to S5 that clarifies the computational gap between satisfiability and model checking.

---

[Towards Trust Calibration in Socially Interactive Agents: Investigating Gendered Multimodal Behaviors Generation with LLMs](http://arxiv.org/abs/2605.19798)

- SIA Framework: introduces a method for automatically generating multimodal behaviors for virtual agents that reflect specific levels of ability and benevolence using LLM-generated tag-augmented transcripts.
- The framework utilizes Random Forest classifiers and SHAP analysis to validate that generated behaviors align with theoretical expectations for trustworthiness dimensions.
- The research identifies that LLMs exhibit gender stereotypes in behavior generation, associating male agents with high ability and female agents with high benevolence.

---

[Prior Knowledge or Search? A Study of LLM Agents in Hardware-Aware Code Optimization](http://arxiv.org/abs/2605.19782)

- LLM-based discovery and optimization systems: investigates the performance of LLMs in black-box optimization and hardware-aware code generation, revealing that LLMs act as greedy optimizers that rely heavily on pretrained priors rather than iterative feedback.
- The study demonstrates that LLM agents, including Sampling Agent and Feedback Loop, often fail to explore effectively in sparse-prior domains, whereas hybrid approaches like Centaur or MCTS provide better balance by integrating structured search.
- The research concludes that LLM exploration is fundamentally constrained by an entropy floor imposed by frozen weights, necessitating domain-specific reinforcement learning or external agentic scaffolding to overcome performance degradation under distributional shift.

---

[Distribution-Free Uncertainty Quantification for Continuous AI Agent Evaluation](http://arxiv.org/abs/2605.19779)

- AgentPulse: introduces a continuous evaluation framework for AI agents that treats uncertainty as a first-class output using Split Conformal Prediction, Adaptive Conformal Inference (ACI), Mondrian Conformal, Compositional Uncertainty Bounds, Conformal Selective Abstention, and Benjamini-Hochberg (BH) Correction.
- The framework provides distribution-free coverage guarantees for agent quality scores by adapting conformal inference methods to handle temporal non-stationarity and agent release-driven distribution shifts.
- AgentPulse enables robust multi-agent pipeline evaluation and leaderboard-scale comparisons by incorporating compositional uncertainty bounds and FDR-controlled abstention to manage ranking instability.

---

[OpenComputer: Verifiable Software Worlds for Computer-Use Agents](http://arxiv.org/abs/2605.19769)

- OpenComputer: introduces a verifier-grounded framework for constructing verifiable software worlds for computer-use agents, integrating app-specific state verifiers, a self-evolving verification layer, a task-generation pipeline, and an evaluation harness.
- The framework utilizes execution-grounded feedback to refine verifiers, ensuring that task evaluation is based on inspectable application state rather than visual proxies or LLM-as-judge judgments.
- OpenComputer provides a large-scale benchmark with 33 desktop applications and 1,000 tasks, demonstrating that current LLMs struggle with robust end-to-end completion in verifiable desktop environments.

---

[Synthesis and Evaluation of Long-term History-aware Medical Dialogue](http://arxiv.org/abs/2605.19766)

- MediLongChat: introduces a knowledge-guided, task-decomposed framework for synthesizing high-quality, long-term medical dialogues to address data scarcity and evaluation challenges in healthcare agents.
- The framework utilizes a three-stage pipeline to construct synthetic patient profiles, generate coherent multi-turn clinical encounters, and establish a benchmark for evaluating longitudinal memory and reasoning.
- The evaluation methodology combines automatic vector-based metrics with an LLM-as-a-Judge approach to assess faithfulness, coherence, diversity, correctness, and realism in generated medical dialogues.

---

[Memory-Augmented Reinforcement Learning Agent for CAD Generation](http://arxiv.org/abs/2605.19748)

- Memory-Augmented Reinforcement Learning Agent for CAD Generation: introduces a closed-loop framework that utilizes an LLM-based Agent, FreeCAD MCP, Case Memory, Skill Memory, Geometric Kernel, Value Network, and Multi-dimensional Verification to improve CAD generation stability.
- The framework treats the geometric kernel as an interactive environment, enabling stepwise verification and correction to suppress cascading failures in long-sequence CAD modeling.
- By integrating a dual-track memory system with reinforcement learning-based utility retrieval, the agent accumulates transferable experience and dynamically optimizes its strategy without requiring additional large-scale annotated data.

---

[EngiAI: A Multi-Agent Framework and Benchmark Suite for LLM-Driven Engineering Design](http://arxiv.org/abs/2605.19743)

- EngiAI: introduces a multi-agent framework and benchmark suite for LLM-driven engineering design that coordinates seven specialized agents through a supervisor architecture to unify topology optimization, document retrieval, HPC job orchestration, and 3D printer control.
- The framework utilizes a hierarchical supervisor pattern built on LangGraph to decompose complex engineering workflows into manageable tasks for specialized agents, ensuring scalability and tool-use efficiency.
- The benchmark suite evaluates LLM performance across three dimensions: workflow execution under distinct cognitive demands, gated retrieval-augmented parameter selection, and end-to-end HPC training orchestration.

---

[Aero-World: Action-Conditioned Aerial Video Generation from Inertial Controls](http://arxiv.org/abs/2605.19728)

- Aero-World: introduces a physics-regularized conversion mechanism that adapts a pretrained video diffusion model into a controllable aerial video generator using a frozen Physics Probe and LoRA finetuning.
- The framework utilizes a frozen latent-space Physics Probe to provide differentiable inertial-consistency supervision, ensuring generated video latents align with commanded 6-DoF IMU signals.
- AeroBench is introduced as a benchmark to evaluate action-faithfulness and temporal stability in aerial video generation using Action Alignment Score (AAS) and Physical Consistency Rate (PCR) metrics.

---

[Measuring Safety Alignment Effects in Autonomous Security Agents](http://arxiv.org/abs/2605.19722)

- Security-agent evaluation framework: introduces a trace-based system for measuring safety alignment effects in autonomous security agents by evaluating task success, evidence grounding, and tool-interface reliability across aligned and less-restricted LLMs.
- The framework utilizes a Task manifest, Local sandbox, Agent controller, Tool execution, and Trace checker to empirically assess how model-level safety alignment influences performance on authorized, sandboxed security tasks.
- Results indicate that safety alignment effects are highly dependent on model family and derivative provenance, with evidence grounding and tool-interface reliability often dominating performance outcomes over simple refusal rates.

---

[Physics-in-the-Loop: A Hybrid Agentic Architecture for Validated CAD Engineering Design](http://arxiv.org/abs/2605.19717)

- Hybrid Agentic-Physical Architecture: introduces a closed-loop, multi-agent system that integrates physics-based engineering tools into the decision-making process of LLMs to generate structurally validated CAD designs.
- The framework utilizes a Planner Agent, CAD Engineer Agent, Geometry Reviewer Agent, and Structural Reviewer Agent to iteratively refine designs through a "Generate-Simulate-Refine" loop.
- By incorporating FEA and geometric validation as feedback signals, the system achieves higher structural complexity and functional validity compared to purely generative or vision-only agentic methods.

---

[RefiningGPT: Specialized language Models for Automated Refinery Unit-level Process Diagram Synthesis](http://arxiv.org/abs/2605.19704)

- RefiningGPT: introduces a hierarchical "Think-then-Draw" paradigm that utilizes a SFT SLM for unit selection and a knowledge-augmented LLM for topology synthesis.
- The framework employs a Constraint-Aware RAG mechanism to ensure that generated process diagrams adhere to rigorous physical and material balance requirements.
- RefiningGPT leverages a specialized SFT dataset derived from legacy refinery topologies to ground LLM reasoning in domain-specific engineering logic.

---

[Agentic Discovery of Cryomicroneedle Formulations](http://arxiv.org/abs/2605.19677)

- Agentic Discovery of Cryomicroneedle Formulations: introduces an AI-assisted, closed-loop workflow that integrates literature curation, Gaussian-process surrogate modeling, Bayesian optimization, and sequential wet-lab validation to discover cryoprotectant formulations.
- The framework utilizes LLMs to automate the construction of a modular computational pipeline, enabling iterative refinement of formulation predictions based on experimental feedback.
- By combining literature-derived priors with iterative wet-lab data, the system progressively improves predictive accuracy and identifies high-viability formulations while reducing reliance on DMSO.

---

[Beyond Rational Illusion: Behaviorally Realistic Strategic Classification](http://arxiv.org/abs/2605.19674)

- Pro-SF: introduces a behaviorally grounded strategic classification framework that replaces the rational-agent assumption with prospect-theoretic mechanisms to model realistic agent manipulations.
- The framework integrates Loss Aversion Component, Reference Bias Component, and Probability Distortion Component into a Stackelberg game to mitigate over-defense and under-defense failure modes.
- Pro-SF utilizes a Behavioral Manipulation Model to learn robust classifiers that maintain performance across diverse, non-rational agent behaviors in real-world deployment scenarios.

---

[SCARA: A Semantics-Constrained Autonomous Remediation Agent for Opaque Industrial Software Vulnerabilities](http://arxiv.org/abs/2605.19668)

- SCARA: introduces a four-stage pipeline for automated remediation of opaque industrial software by combining CACA, OSVA, RSA, and CVA to bridge the gap between binary vulnerability discovery and validated remediation.
- The framework utilizes an SSCKG to provide semantic guidance for symbolic execution and remediation synthesis, ensuring that patches are constrained by the operational-state envelope.
- SCARA employs a tiered remediation model—protocol mitigation, binary hardening, and source-level patching—validated by a closed-loop feedback mechanism that ensures behavioral coverage preservation.

---

[P2DNav: Panorama-to-Downview Reasoning for Zero-shot Vision-and-Language Navigation](http://arxiv.org/abs/2605.19634)

- P2DNav: introduces a hierarchical zero-shot VLN framework that decouples high-level directional reasoning from low-level local grounding using P2D, SDM, and RRM.
- The framework utilizes P2D for coarse-to-fine navigation, SDM for efficient long-horizon memory management, and RRM for reflective correction of unreliable local grounding decisions.
- P2DNav achieves state-of-the-art zero-shot performance on the R2R-CE benchmark by enabling a closed-loop decision process without requiring task-specific training or external waypoint generators.

---

[optimize_anything: A Universal API for Optimizing any Text Parameter](http://arxiv.org/abs/2605.19633)

- optimize_anything: introduces a declarative API that treats diverse optimization problems as text artifact refinement, utilizing LLM-based search to improve performance across domains.
- The framework leverages Side Information (SI) as a first-class evaluator contract, enabling LLMs to perform targeted, gradient-like updates based on diagnostic feedback rather than scalar scores alone.
- By maintaining a Pareto frontier of candidates, the system supports single-task, multi-task, and generalization modes, allowing for cross-task transfer and the discovery of complex, multi-stage agent architectures and algorithms.

---

[Formal Skill: Programmable Runtime Skills for Efficient and Accurate LLM Agents](http://arxiv.org/abs/2605.19604)

- FairyClaw: introduces a runtime-native abstraction called Formal Skill that replaces informal natural-language procedures with structured JSON metadata, action schemas, reliable Python executors, lifecycle hooks, and skill-local runtime state.
- The framework utilizes a hook-governed programming model to enforce procedural invariants, validate tool arguments, and manage state transitions, thereby reducing token consumption and improving agent reliability.
- By implementing an event-driven runtime, FairyClaw enables dynamic skill routing and sub-agent collaboration, allowing LLMs to operate within enforceable, state-conditioned boundaries rather than relying on monolithic prompts.

---

[A novel YOLO26-MoE optimized by an LLM agent for insulator fault detection considering UAV images](http://arxiv.org/abs/2605.19595)

- YOLO26-MoE: introduces a novel object detection architecture that integrates a sparse Mixture-of-Experts module into the high-resolution P3 branch of the YOLO26 detector to enable adaptive feature refinement for insulator fault detection.
- The framework utilizes a tool-augmented LLM agent to coordinate hyperparameter optimization, training, and evaluation, leveraging domain knowledge to guide the search process.
- Experimental results demonstrate that the proposed model achieves superior detection performance compared to baseline YOLO variants, providing an effective solution for UAV-based insulator fault detection.

---

[SceneCode: Executable World Programs for Editable Indoor Scenes with Articulated Objects](http://arxiv.org/abs/2605.19587)

- SceneCode: introduces a framework that compiles natural language prompts into executable, code-driven indoor worlds with physically interactable objects.
- The system utilizes a room-level agentic backbone and a planner-designer-critic loop to generate structured object specifications, which are then converted into part-wise Blender programs through specialized VLM-based strategies.
- Generated assets are registered in a persistent scene-state registry, enabling local editability and downstream simulation in environments like MuJoCo with articulated joints and physical attributes.

---

[Library Drift: Diagnosing and Fixing a Silent Failure Mode in Self-Evolving LLM Skill Libraries](http://arxiv.org/abs/2605.19576)

- Ratchet: introduces a governance framework for self-evolving LLM agents to mitigate library drift, a silent failure mode caused by unbounded skill accumulation without quality management.
- The framework utilizes trace-level diagnostics including per-skill contribution scores, attribution verdicts from a Critic LLM, and router engagement metrics to detect performance degradation before it impacts aggregate metrics.
- Ratchet implements a monotonic loop featuring outcome-driven retirement, a bounded active-cap, and a meta-skill authoring prior to ensure persistent procedural knowledge improves agent performance over time.

---

[CaptchaMind: Training CAPTCHA Solvers via Reinforcement Learning with Explicit Reasoning Supervision](http://arxiv.org/abs/2605.19538)

- CaptchaMind: introduces a training-based CAPTCHA solver that utilizes explicit reasoning process supervision to improve visual grounding and task success rates.
- The framework leverages CaptchaBench, a large-scale benchmark with process-level annotations, to train models using a multi-level reward system including Reasoning Process Reward, Interaction Feedback Reward, and Outcome Reward.
- By employing the Bounding Box Tool for explicit region grounding, the approach enables the model to attend to task-relevant visual details, significantly outperforming existing methods on complex CAPTCHA tasks.

---

[CutVerse: A Compositional GUI Agents Benchmark for Media Post-Production Editing](http://arxiv.org/abs/2605.19484)

- CutVerse: introduces a comprehensive benchmark and evaluation pipeline for assessing autonomous GUI agents in complex, long-horizon media post-production workflows.
- The framework utilizes a multimodal parser to transform expert demonstrations into structured, milestone-driven trajectories for scalable and reproducible agent evaluation.
- Extensive experiments reveal that while LLMs excel at procedural tasks, they struggle with the spatial grounding and compositional action coordination required for professional media editing.

---

[Sampling-Based Safe Reinforcement Learning](http://arxiv.org/abs/2605.19469)

- SBSRL (Sampling-Based Safe Reinforcement Learning): introduces a model-based RL algorithm that ensures safety by enforcing constraints across a finite set of Dynamics Samples, while promoting exploration via an epistemic uncertainty-based Exploration Constraint.
- The framework utilizes a Truncation Mechanism to refine its Dynamics Model and Dynamics Samples online, ensuring that the learned Policy maintains safety under worst-case dynamics.
- SBSRL provides high-probability safety guarantees and finite-time sample complexity bounds, demonstrating empirical effectiveness in both simulated continuous control tasks and real-world robotic hardware.

---

[What and When to Distill: Selective Hindsight Distillation for Multi-Turn Agents](http://arxiv.org/abs/2605.19447)

- SERL: introduces a reinforcement learning framework for multi-turn LLM agents that uses environment-conditioned teacher signals to selectively reweight GRPO updates based on hindsight feedback.
- The framework separates feedback sources from their placement, applying teacher-derived, action-level credit adjustments only to executable tokens while maintaining the reward-driven optimization direction.
- By decaying the teacher signal over training, SERL balances dense early-stage credit assignment with stable, reward-anchored policy convergence in long-horizon agentic environments.

---

[Conflict-Resilient Multi-Agent Reasoning via Signed Graph Modeling](http://arxiv.org/abs/2605.19418)

- SIGMA: introduces a multi-agent reasoning framework that explicitly models inter-agent trust, conflict, and neutral relations using signed graph modeling to enhance robustness against noisy or adversarial agent outputs.
- The framework utilizes Query-Guided Agent Selection to identify relevant agents, constructs a Signed Relational Graph to encode interaction polarity, and employs Conflict-Aware Signed Message Passing to iteratively refine agent representations.
- By applying a Signed Consensus Readout, the system aggregates agent outputs based on net supportive strength, effectively suppressing misleading signals and ensuring globally consistent predictions across diverse reasoning benchmarks.

---

[Vision Harnessing Agent for Open Ad-hoc Segmentation](http://arxiv.org/abs/2605.19410)

- VASA (Vision-guided Ad-hoc Segmentation Agent): introduces a training-free framework that couples a VLM Agent, a Segmentation Foundation Model (SAM3), and a Vision Harness Workflow to perform iterative, persistent visual construction for open ad-hoc segmentation.
- The framework utilizes a persistent Working Mask and State Management to maintain visual progress across multiple interaction rounds, enabling the agent to perform complex operations like addition, removal, and replacement of regions.
- VASA incorporates Long-Horizon Planning, Visual Scrutiny, and Error Recovery to decompose complex user queries into executable visual steps, significantly outperforming existing agentic baselines on the PARS and RefCOCOm benchmarks.

---

[HSCO-Bench: An Agent-Driven End-to-End Hardware-Software Co-design Benchmark for Systems-on-Chip](http://arxiv.org/abs/2605.19399)

- HSCO-Bench: introduces, a comprehensive benchmark for evaluating LLM agents on end-to-end hardware-software co-design tasks for heterogeneous Systems-on-Chip.
- The framework requires LLMs to autonomously identify performance bottlenecks, design custom HLS-based accelerators, and integrate them into a tile-based SoC architecture.
- Experimental results demonstrate that while frontier LLMs can generate functional SoC prototypes, they frequently underutilize hardware resources, highlighting significant optimization gaps in automated co-design.

---

[Toward User Comprehension Supports for LLM Agent Skill Specifications](http://arxiv.org/abs/2605.19362)

- Agent Skill Comprehension Framework: introduces a methodology for evaluating LLM agent skill specifications by measuring four comprehension anchors: Operational basis, Output contract, Boundary disclosure, and Example capability demonstration.
- The study analyzes 878 cybersecurity skill specifications to determine if they provide sufficient information for users to form bounded expectations before execution.
- Findings indicate that while operational basis is commonly disclosed, only 2.3% of specifications include all four comprehension anchors, highlighting a significant gap in user-facing capability transparency.

---

[PAVE: A Cognitive Architecture for Legitimate Violation in Generative Agent Societies](http://arxiv.org/abs/2605.19351)

- PAVE (Perception, Assessment, Verdict, Emulation): introduces a cognitive architecture for LLM-based agents that enables principled reasoning about rule-breaking in spatially grounded environments.
- The architecture integrates four modules—Perception, Assessment, Verdict, and Emulation—to manage ethical trade-offs between compliance, urgency, and authority.
- PAVE agents demonstrate legitimate violation, authority deference, bounded scope, and recovery, outperforming vanilla LLM agents in structured decision-making and interpretability.

---

[HalluWorld: A Controlled Benchmark for Hallucination via Reference World Models](http://arxiv.org/abs/2605.19341)

- HALLUWORLD: introduces a benchmark framework that operationalizes hallucination as observable errors relative to an explicit Reference World, View Function, and Conflict Policy.
- The framework utilizes synthetic and semi-synthetic environments across gridworlds, chess, and terminal tasks to automatically generate hallucination labels via construction, employing various Probe Categories and Serializers to isolate specific failure modes.
- Experimental results demonstrate that while frontier LLMs achieve high perceptual accuracy, they struggle with multi-step state tracking, causal forward simulation, and epistemic abstention, with performance significantly influenced by the chosen serialization format and navigation-induced cognitive load.

---

[STAR-PólyaMath: Multi-Agent Reasoning under Persistent Meta-Strategic Supervision](http://arxiv.org/abs/2605.19338)

- STAR-PólyaMath: introduces a multi-agent framework that addresses long-horizon mathematical reasoning challenges through an orchestrated state machine with nested challenge-step-replan loops, governed by a Python Orchestrator, Reasoner Agent, Verifier Agent, and a persistent Meta-Strategist Agent.
- The framework utilizes a persistent Meta-Strategist Agent to maintain cross-attempt memory and provide strategic guidance, effectively mitigating hallucination accumulation and memory fragmentation.
- By separating control flow from inference and employing structured Reasoner-Verifier debate, the system achieves state-of-the-art performance on complex competition mathematics benchmarks.

---

[Agentic Trading: When LLM Agents Meet Financial Markets](http://arxiv.org/abs/2605.19337)

- A-C-A framework: introduces an audit-oriented evidence map of LLM-based trading agents, reframing them as expert-system decision pipelines that transform market observations into executable actions through Perception, Memory, Reasoning, Action, and Adaptation.
- The paper evaluates 77 studies, identifying a primary empirical subset of 19 that satisfy strict Action Output and Closed-Loop Evaluation criteria, highlighting significant gaps in protocol reporting and reproducibility.
- It proposes a Minimum Reporting Checklist (MR-1 to MR-7) to improve the comparability of agentic trading research by standardizing the documentation of data splits, execution semantics, and audit artifacts.

---

[MOCHA: Multi-Objective Chebyshev Annealing for Agent Skill Optimization](http://arxiv.org/abs/2605.19330)

- MOCHA (Multi-Objective CHebyshev Annealing): introduces a multi-objective optimization framework for LLM agent skills that balances task correctness and platform compliance using Chebyshev scalarization and annealed thresholding.
- The framework employs a two-stage approach, utilizing Hypervolume Contribution for exploration and Chebyshev scalarization for exploitation to navigate non-convex Pareto fronts.
- MOCHA consistently outperforms existing prompt optimizers by discovering diverse, Pareto-optimal skill variants that satisfy hard platform constraints while maximizing task performance.

---

[RoboJailBench: Benchmarking Adversarial Attacks and Defenses in Embodied Robotic Agents](http://arxiv.org/abs/2605.19328)

- RoboJailBench: introduces a comprehensive benchmarking framework for evaluating adversarial attacks and defenses in embodied AI systems by measuring the security-utility tradeoff.
- The framework utilizes an embodiment-grounded security taxonomy, an intent-contrast dataset pipeline, and standardized evaluation metrics to assess model robustness against adversarial inputs.
- Experimental results demonstrate that while current defense strategies improve security, they often impact utility, highlighting the need for balanced safety guardrails in embodied AI.

---

[A Multi-Agent Framework for Feature-Constrained Difficulty Control in Reading Comprehension Item Generation](http://arxiv.org/abs/2605.19316)

- MAFIG (Multi-Agent Framework for Feature-constrained Item Generation): introduces a multi-agent framework that iteratively revises reading comprehension items to ensure strict adherence to multi-dimensional feature constraints.
- The framework utilizes a collaborative system of role-specialized LLM agents—including Drafter, Planner, Reworder, Editor, and Refiner—to perform iterative refinements based on feedback from an Evaluator.
- MAFIG incorporates a difficulty-calibrated constraint sequence methodology to generate items with monotonically increasing difficulty, significantly outperforming single-pass prompting baselines in constraint satisfaction and difficulty alignment.

---

[ContextFlow: Hierarchical Task-State Alignment for Long-Horizon Embodied Agents](http://arxiv.org/abs/2605.19314)

- ContextFlow: introduces an inspectable alignment framework that manages long-horizon embodied tasks by organizing execution as a continuous context flow over a staged task frontier.
- The framework utilizes stage contracts, memory, and an asynchronous monitor to convert runtime observations into evidence packets, enabling the planner to apply scoped updates that resolve task-state misalignment.
- By maintaining explicit alignment between high-level planning and grounded expert execution, the system improves performance on long-horizon navigation tasks by mitigating failures such as unsupported handoffs, stage locks, and executor-context mismatches.

---

[DECOR: Auditing LLM Deception via Information Manipulation Theory](http://arxiv.org/abs/2605.19270)

- DECOR: introduces a multi-agent framework for fine-grained auditing of strategic deception in LLMs by utilizing Units Construction Agent, IMT Auditing Agent, and Deception Prediction Aggregator.
- The framework decomposes input contexts into Atomic Informational Units, assigns Strategic Impact Weights, and generates Manipulation Profiles to produce a Global Deception Index.
- DECOR operates as a black-box auditing tool that grounds deception detection in Information Manipulation Theory to provide interpretable, dimension-level diagnostics of LLM responses.

---

[MuMuTestUp: Mutation-based Multi-Agent Test Case Update](http://arxiv.org/abs/2605.19265)

- MuMuTestUp: introduces a mutation-guided, multi-agent framework for automated test case updating that leverages specialized agents for Input Preprocessing agent, Test Update agent, Coordinator agent, Error Analysis agent, Coverage Analysis agent, Mutation Analysis agent, and Semantic Retrieval agent to improve test adequacy.
- The framework utilizes fine-grained execution feedback, including coverage reports and mutation analysis, to guide LLMs in generating robust test assertions and improving branch coverage.
- MuMuTestUp employs a two-stage semantic retrieval strategy to resolve hallucinated symbols and incorporates a pull-request-level dataset, PRBENCH, for realistic evaluation of test case updates.

---

[AQuaUI: Visual Token Reduction for GUI Agents with Adaptive Quadtrees](http://arxiv.org/abs/2605.19260)

- AQuaUI: introduces a training-free inference-time visual token reduction method for GUI agents that leverages the non-uniform information density of screenshots using an Adaptive Quadtree and Conditional Quadtree Refinement.
- The framework optimizes GUI agent performance by discarding redundant visual tokens while preserving spatial layout information through a hierarchical partitioning strategy.
- AQuaUI improves accuracy-efficiency trade-offs in LLMs by reducing visual token load before KV-cache computation, demonstrating robustness across various GUI grounding and navigation benchmarks.

---

[CASPIAN: Online Detection and Attribution of Cascade Attacks in LLM Multi-Agent Systems via Cross-Channel Causal Monitoring](http://arxiv.org/abs/2605.19240)

- CASPIAN: introduces a unified framework for online detection and attribution of cascade attacks in LLM-based multi-agent systems by modeling cross-channel causal influence through LI-CTE and spectral monitoring.
- The framework constructs a dynamic causal influence tensor from communication, memory, tool, and execution interactions to identify cascade onset via spectral amplification, synchronization, and persistence.
- CASPIAN performs real-time attribution of cascade origins, bridges, and amplifiers using cached influence dynamics, enabling timely intervention without requiring replay or recomputation.

---

[GAE Falls Short in Imperfect-Information Self-Play Reinforcement Learning](http://arxiv.org/abs/2605.19235)

- VRPO (Variance-Reduced Policy Optimization): introduces a variance-reduced advantage estimator called Q-boosting that replaces sampled multi-step backups with a multi-step Expected SARSA(λ) trace to mitigate action-sampling noise in stochastic self-play.
- The framework utilizes a centralized Q-critic to compute policy expectations at each backup step, effectively averaging out noise from stochastic future actions while maintaining decentralized actor execution.
- Empirical results demonstrate that VRPO consistently achieves lower exploitability than standard PPO-based baselines in mid-sized games and exhibits strong performance in large-scale domains like Dou Dizhu and Heads-Up No-Limit Texas Hold’em.

---

[SimGym: A Framework for A/B Test Simulation in E-Commerce with Traffic-Grounded VLM Agents](http://arxiv.org/abs/2605.19219)

- SimGym: introduces a modular framework for synthetic A/B testing in e-commerce that utilizes traffic-grounded VLM agents to predict user responses to interface modifications.
- The framework integrates a persona generation pipeline, a multimodal live-browser agent architecture, and an evaluation protocol to simulate shopping sessions and quantify the impact of UI changes on add-to-cart performance.
- Empirical validation on 50 real-world storefronts demonstrates that SimGym agents achieve strong directional alignment and correlation with observed human behavioral shifts, enabling rapid experimentation without exposing real users to candidate variants.

---

[CLUE: Adaptively Prioritized Contextual Cues by Leveraging a Unified Semantic Map for Effective Zero-Shot Object-Goal Navigation](http://arxiv.org/abs/2605.19206)

- CLUE: introduces a navigation framework that constructs a unified semantic value map by adaptively balancing global room-level and local object-level cues based on target object characteristics.
- The framework utilizes offline LLM queries to extract commonsense knowledge, enabling efficient, real-time navigation without the latency of online LLM reasoning.
- By employing entropy-based weighting to prioritize contextual cues and multi-view verification for robust localization, the system achieves state-of-the-art performance in zero-shot ObjectNav tasks.

---

[Platform architecture determines whether recommendation algorithms can shape information quality on social media](http://arxiv.org/abs/2605.19204)

- Agent-based simulation framework: introduces a computational model to evaluate how platform architecture and recommendation algorithms interact to shape information spread and quality.
- The framework orthogonally manipulates four prototypical architectures (tree, layered, network, complete graph) and two algorithms (LIFO, Hot) to test the flexibility hypothesis of generic system architectures.
- The simulation records message-level events including creation, reach, exposure, and engagement to quantify the causal impact of platform design on information dynamics.

---

[Multi-agent Collaboration with State Management](http://arxiv.org/abs/2605.20563)

- STORM (STate-ORiented Management): introduces a state management framework for multi-agent collaboration that replaces workspace isolation with local state consistency to detect and resolve conflicts at write time.
- The framework utilizes a Manager Agent to decompose tasks and a STORM Manager to mediate file access, ensuring that Parallel Engineer Agents operate on consistent file versions via Versioning and Validation.
- By employing Intent Annotations for semantic coordination, the system enables multiple LLMs to work concurrently on a Shared Workspace while minimizing integration failures.

---

[Personality Engineering with AI Agents: A New Methodology for Negotiation Research](http://arxiv.org/abs/2605.20554)

- Personality Engineering: introduces a methodology that uses AI agents to precisely parameterize, manipulate, and evaluate negotiator personality along the Interpersonal Circumplex dimensions of warmth and dominance.
- The framework leverages the precision, repertoire, consistency, and scalability of LLMs to systematically test negotiation theories under controlled conditions.
- By treating personality as a set of design variables, researchers can optimize agent behavior to map outcome surfaces and identify optimal configurations for diverse negotiation contexts.

---

[What Do Agents Communicate? Characterizing Information Exchange in Multi-Agent Systems](http://arxiv.org/abs/2605.20548)

- CARA (Category-Aware Recovery Augmentation): introduces a systematic framework to categorize inter-agent communication in MA systems and mitigate error propagation through targeted content enforcement.
- The study utilizes occlusion analysis to identify critical information categories—Reasoning, Verification, and Reference—that significantly influence collective task performance across diverse MA architectures.
- By enforcing these essential categories via prompt augmentation and response verification, CARA recovers up to 86.2% of failed task cases without requiring modifications to the underlying LLM models.

---

[The Yes-Man Syndrome: Benchmarking Abstention in Embodied Robotic Agents](http://arxiv.org/abs/2605.20544)

- ROBOABSTENTION: introduces a scalable, auditable framework for generating visually grounded abstention instructions for embodied agents using a three-phase pipeline of visual grounding, constraint derivation, and template-based generation.
- The framework evaluates frontier VLMs on their ability to recognize when to abstain from executing instructions that are ambiguous, physically infeasible, or based on false premises.
- Experimental results demonstrate that while current models often default to unwarranted action, performance can be substantially improved through defensive prompting and in-context learning.

---

[AgentAtlas: Beyond Outcome Leaderboards for LLM Agents](http://arxiv.org/abs/2605.20530)

- AgentAtlas: introduces a taxonomy and measurement protocol for diagnosing LLM agents beyond final task success by integrating a control-decision taxonomy, a trajectory-failure taxonomy, a benchmark-coverage audit, and an empirical demonstration using taxonomy-aware and taxonomy-blind prompting.
- The framework evaluates LLMs across six behavioral axes—control, trajectory, tool-context utility, security, memory, and efficiency—to address the limitations of outcome-only benchmarks.
- Empirical results demonstrate that model rankings are highly sensitive to prompt format and evaluation axis, revealing significant cross-axis incoherence in agent performance.

---

[Open-World Evaluations for Measuring Frontier AI Capabilities](http://arxiv.org/abs/2605.20520)

- CRUX (Collaborative Research for Updating AI eXpectations): introduces a systematic framework for conducting open-world evaluations by pairing long-horizon, real-world tasks with agent scaffolds to qualitatively analyze LLM behavior.
- The framework utilizes an OpenClaw scaffold integrated with Claude Opus 4.6 to execute complex, multi-step tasks like iOS app deployment within an isolated macOS VM environment.
- CRUX emphasizes methodological rigor through documented human interventions, systematic log analysis, and cost-conditioned performance reporting to address the limitations of traditional, sandboxed LLM benchmarks.

---

[ZEBRA: Zero-shot Budgeted Resource Allocation for LLM Orchestration](http://arxiv.org/abs/2605.20485)

- ZEBRA: introduces a zero-shot, inference-time framework that optimizes resource allocation across multi-agent pipeline phases by modeling them as a continuous nonlinear knapsack problem solved via water-filling.
- The framework utilizes an Allocation Agent to estimate saturating-exponential utility curves for each pipeline phase, which are then processed by a Knapsack Solver to determine optimal budget splits.
- ZEBRA demonstrates significant performance gains over direct LLM-based allocation by adapting budget distribution to task difficulty and pipeline structure without requiring fine-tuning or reinforcement learning.

---

[Training Language Agents to Learn from Experience](http://arxiv.org/abs/2605.20477)

- ICT (In-context Training): introduces a framework for evaluating cross-task self-improvement in LLMs by using a Reflector to generate system prompts based on an Actor's performance in interactive environments.
- The framework utilizes a dual-LLM design where the Reflector is fine-tuned via GRPO to analyze interaction trajectories and produce prompts that generalize across unseen tasks.
- MetaGym is introduced as a Python library to facilitate the construction of meta-environments for research on self-improving LLM agents.

---

[Code Generation by Differential Test Time Scaling](http://arxiv.org/abs/2605.20473)

- DiffCodeGen: introduces a test-time scaling method for code generation that uses Differential generation, Coverage-guided fuzzing, Dynamic behavior modeling, Distance-based clustering, and Medoid selection to identify high-quality code without extra LLM inference.
- The framework improves code generation reliability by clustering candidates based on their dynamic execution behavior and selecting the medoid of the largest cluster as the final output.
- DiffCodeGen achieves competitive performance while significantly reducing token consumption and execution time compared to existing test-time scaling methods.

---

[Agentic Agile-V: From Vibe Coding to Verified Engineering in Software and Hardware Development](http://arxiv.org/abs/2605.20456)

- Agentic Agile-V: introduces a process framework that integrates Agile-V lifecycle management with a task-level SCOPE-V loop to convert conversational intent into structured, verifiable engineering artifacts.
- The framework utilizes a conversation-to-contract gate to ensure that LLMs operate based on reviewed briefs rather than ambiguous chat histories, thereby reducing verification debt.
- It establishes risk-adaptive acceptance gates that require specific evidence bundles—ranging from smoke tests to formal verification—before human approval for software, firmware, and hardware development.

---

[Modeling Emotional Dynamics in Agent-to-Agent Interactions on Moltbook](http://arxiv.org/abs/2605.20442)

- PSR (Persona-Stimulus-Reaction) framework: introduces a structured approach to modeling emotional dynamics in agent-to-agent interactions by mapping textual content into a continuous VAD space and analyzing behavioral patterns via GMM.
- The framework decomposes agent interactions into three interdependent components: Persona (P) representing stable identity, Stimulus (S) representing contextual input, and Reaction (R) representing the resulting emotional output.
- By utilizing GMMs, the approach captures the probabilistic and multi-modal nature of agent responses, allowing for the classification of behavioral typologies such as aligned, persona-consistent, or stimulus-driven interactions.

---

[ParaVT: Taming the Tool Prior Paradox for Parallel Tool Use in Agentic Video Reinforcement Learning](http://arxiv.org/abs/2605.20342)

- ParaVT: introduces a multi-agent end-to-end RL-trained framework for parallel video tool calling that replaces sequential tool chains with peer-correctable evidence aggregation.
- The framework utilizes PARA-GRPO, which combines Exploration Anchoring to maintain structural format and nFrames Gating to ensure tool necessity during RL training.
- ParaVT addresses the Tool Prior Paradox, where pretrained tool priors destabilize structural format under temperature sampling, by effectively balancing format compliance and tool exploration.

---

[Mix-Quant: Quantized Prefilling, Precise Decoding for Agentic LLMs](http://arxiv.org/abs/2605.20315)

- Mix-Quant: introduces a phase-aware quantization framework that decouples compute-intensive prefilling from autoregressive decoding to optimize LLM agent inference.
- The framework utilizes high-throughput NVFP4 quantization for the prefilling stage while maintaining high-precision BF16 for decoding to prevent error accumulation and preserve task performance.
- By leveraging hardware-efficient NVFP4 execution and prefill-decode disaggregation, the approach achieves significant speedups in long-context agentic workflows without sacrificing generation stability.

---

[Pramāṇa: A Protocol-Layer Treatment of Claim Verification in Autonomous Agent Networks](http://arxiv.org/abs/2605.20312)

- Pramāṇa: introduces a protocol-layer framework for autonomous agent networks that standardizes verification artifacts through a typed ClaimAttestation and a deterministic verify() contract.
- The framework ensures auditability by requiring every consequential agent output to be accompanied by a re-verifiable record, integrating with A2A and MCP protocols to provide a standardized wire format.
- Formal verification using TLA+ and TLC ensures lifecycle safety invariants, while the approach addresses the structural limitations of probabilistic LLM-as-judge patterns in regulated domains.

---

[WILDROADBENCH: A Wild Aerial Road-Damage Grounding Benchmark for Vision–Language Models and Autonomous Agents](http://arxiv.org/abs/2605.20306)

- WILDROADBENCH: introduces a benchmark for aerial road-damage grounding that evaluates both direct VLM visual grounding and autonomous LLM-driven agent research-and-engineering capabilities.
- The framework utilizes a unified two-track evaluation protocol, VLM TRACK and AGENT TRACK, to assess performance on a professionally annotated 1,061-image UAV corpus.
- The AGENT TRACK employs a sandboxed environment where LLM-driven agents must perform data acquisition, model adaptation, and training to produce a working detector under a fixed interaction budget.

---

[Mechanisms of Misgeneralization in Physical Sequence Modeling](http://arxiv.org/abs/2605.20299)

- Physical Misgeneralization Framework: introduces a mechanistic account of how generative sequence models fail to preserve intended physical quantity distributions due to local trajectory-space errors propagating through measurement.
- The framework utilizes a data deviation kernel to estimate local model-induced errors and predict how probability mass shifts across physical quantity values without requiring a fully trained model.
- It demonstrates that coordinate transformation of the input-output representation effectively mitigates physical misgeneralization by balancing mass transfer between quantity values.

---

[WEASEL: Out-of-Domain Generalization for Web Agents via Importance-Diversity Data Selection](http://arxiv.org/abs/2605.20291)

- WEASEL: introduces a data selection framework for compute-efficient offline training of LLM-based web agents that improves out-of-domain generalization by balancing goal-conditioned importance and pairwise diversity.
- The framework incorporates target-centered AXTree pruning to remove redundant page content and a self-reasoning synthesis module to mitigate style mismatch in reasoning-native LLMs.
- WEASEL achieves significant training speedups and superior zero-shot transfer performance across multiple web agent benchmarks by selecting a compact, informative subset of expert trajectories.

---

[Smaller Abstract State Spaces Enable Cross-Scale Generalization in Reinforcement Learning](http://arxiv.org/abs/2605.20272)

- Successor-weighted model reduction: introduces a theoretical model for achieving Out-of-Distribution (OOD) generalization in Reinforcement Learning by compressing infinite-state POMDPs into smaller, finite abstract state spaces.
- The framework utilizes a successor-weighted model reduction to bound performance loss, decomposing it into approximation and OOD estimation errors based on distribution-weighted norms.
- The research demonstrates that constraining RL agents to operate over small, finite abstract state spaces is necessary for effective generalization across tasks of varying complexity levels.

---

#### 18th May 2026

[Going Headless? On the Boundaries of Vertical AI Firms](http://arxiv.org/abs/2605.17812)

- Vertical AI Firm Strategy Framework: introduces a strategic taxonomy for vertical AI firms navigating the transition to agentic architectures by distinguishing between interface boundaries and accountability boundaries.
- The framework categorizes firm positions into components, integrated software platforms, or dual-track models based on task-accountability regimes rather than market sectors.
- It formalizes the concept of rule debt to describe the governance and maintenance burden that arises when organizational decision logic is offloaded into informal LLM agent instructions.

---


[Toward an AI-Powered Computational Testbed for Workforce Policy](http://arxiv.org/abs/2605.19064)

- Dynamic Employee Agent Platform: introduces a computational architecture that utilizes LLMs to create personalized, evolving replicas of employees for simulating workforce responses to organizational changes.
- The framework integrates HRIS data, psychometric measures, and digital activity logs to condition generic LLMs into dynamic agents that exhibit realistic cognitive, emotional, and behavioral trajectories.
- This simulation platform enables leaders to evaluate the efficacy of AI tool rollouts and policy interventions in a multi-agent environment before real-world deployment, thereby reducing the risk of failed pilots.

---


[Aurora: Unified Video Editing with a Tool-Using Agent](http://arxiv.org/abs/2605.18748)

- Aurora: introduces an agentic video editing framework that resolves textual and visual underspecification by pairing a tool-augmented VLM agent with a unified video diffusion transformer.
- The VLM agent parses raw user requests into structured edit plans, triggering external tools like web search and segmentation to construct model-ready conditioning tuples for the video DiT.
- The framework utilizes a two-path conditioning architecture, combining a multimodal context encoder for instruction and reference integration with a latent token sequence for flow-matching-based video generation.

---

[Code as Agent Harness: Toward Executable, Verifiable, and Stateful Agent Systems](http://arxiv.org/abs/2605.18747)

- Code as Agent Harness: introduces a unified framework that centers code as the operational substrate for agent reasoning, acting, environment modeling, and verification, utilizing Harness Interface, Harness Mechanisms, and Scaling the Harness.
- The framework organizes agentic systems into three connected layers: the harness interface for reasoning and acting, harness mechanisms for reliability and adaptation, and scaling the harness for multi-agent orchestration.
- This survey provides a roadmap for building executable, verifiable, and stateful AI agent systems by treating code as the primary medium for agent interaction and infrastructure.

---

[ESI-BENCH: Towards Embodied Spatial Intelligence that Closes the Perception-Action Loop](http://arxiv.org/abs/2605.18746)

- ESI-BENCH: introduces a comprehensive benchmark for embodied spatial intelligence that requires agents to actively close the perception-action loop through OmniGibson, BEHAVIOR-1K, GPT-4o, LLMs, Agent, Simulator, Scene Graph, and Action Space.
- The framework evaluates agents across 10 task categories and 29 subcategories, focusing on spatial reasoning tasks that cannot be resolved from passive observation alone.
- Experiments demonstrate that active exploration enables agents to discover emergent spatial strategies, while highlighting significant metacognitive gaps in current LLMs regarding belief revision and epistemic calibration.

---

[Vision-OPD: Learning to See Fine Details for Multimodal LLMs via On-Policy Self-Distillation](http://arxiv.org/abs/2605.18740)

- Vision-OPD: introduces a regional-to-global self-distillation framework that transfers privileged crop-conditioned perception to a full-image-conditioned student policy via token-level divergence minimization.
- The framework instantiates two conditional policies from the same MLLM, where the student generates on-policy rollouts from full images while the teacher provides supervision from evidence-centered crops.
- This approach internalizes fine-grained visual understanding into a single forward pass without requiring external teachers, ground-truth labels, or inference-time tool use.

---

[Robo-Cortex: A Self-Evolving Embodied Agent via Dual-Grain Cognitive Memory and Autonomous Knowledge Induction](http://arxiv.org/abs/2605.18729)

- Robo-Cortex: introduces a self-evolving embodied navigation framework that integrates an Imagine-then-Verify Planning Loop, Dual-Grain Cognitive Memory, and Autonomous Knowledge Induction to transform interaction experience into transferable heuristics.
- The framework utilizes a VLM-based evaluator and world model to perform closed-loop decision making, while SRM and LPM provide multi-scale reflection to refine navigation strategies within and across episodes.
- Autonomous Knowledge Induction distills recurring behavioral patterns into a structured Heuristic Library, enabling the agent to generalize navigation strategies to unseen environments through continual self-evolution.

---

[DexHoldem: Playing Texas Hold’em with Dexterous Embodied System](http://arxiv.org/abs/2605.18727)

- DexHoldem: introduces a real-world system-level benchmark for instruction-conditioned dexterous manipulation in a Texas Hold’em tabletop setting.
- The framework integrates an embodied agent that manages game-state memory and decision routing with a multi-task dexterous policy for precise, contact-rich manipulation.
- The benchmark evaluates the full embodied loop, including agentic perception, decision routing, and dexterous execution, while highlighting the compounding reliability gap in closed-loop deployment.

---

[Ranking Opinions with Few States in Population Protocols](http://arxiv.org/abs/2605.18707)

- CIRCLES: introduces a population protocol that solves the relative majority and ranking problems with reduced state complexity by structuring agents into circular linked lists.
- The framework utilizes Braket components to organize agents into circles, where the smallest circles correspond to colors with the greatest support.
- CIRCLES incorporates additional mechanisms including Tie-break tokens for consensus and an Ordering protocol to handle unordered input data.

---

[EnvFactory: Scaling Tool-Use Agents via Executable Environments Synthesis and Robust RL](http://arxiv.org/abs/2605.18703)

- EnvFactory: introduces a fully automated framework that autonomously constructs stateful, executable tool environments and synthesizes natural multi-turn trajectories for Agentic RL.
- The framework utilizes a Search Agent, Code Agent, and Test Agent to build verified environments, while employing a topology-aware sampling strategy and QueryGen to generate realistic, implicit-intent training data.
- EnvFactory achieves superior training efficiency and downstream performance on benchmarks like BFCLv3 and MCP-Atlas by using significantly fewer environments than prior methods.

---

[Contextualized Dynamic Explanations: A Vision](http://arxiv.org/abs/2605.18698)

- CODEX (Contextualized Dynamic Explanations): introduces an agentic approach to dynamically generating multi-modal information interfaces for data-driven explanations based on an evolving Audience Model and predefined communication intent.
- The framework utilizes a Foundational Model for general reasoning, a Domain Model for specialized tasks, and various Tools to adaptively present information through Information Interfaces.
- CODEX functions as an autonomous agent that monitors communication progress, manages interaction state, and steers the explanation process to ensure alignment with the user's needs and the communicator's intent.

---

[SkillGenBench: Benchmarking Skill Generation Pipelines for LLM Agents](http://arxiv.org/abs/2605.18693)

- SkillGenBench: introduces a benchmark for evaluating LLM-based skill generation pipelines under a unified and controlled protocol, utilizing Knowledge Graph Construction, Scenario Generation, Tasks and Test Cases Generation, Task Verification without Skills, Task Verification with Skills, Evaluation Protocol, Execution Harness, and Skill Package.
- The framework decouples skill generation from downstream execution, enabling direct measurement of procedure-to-skill distillation across repository-grounded and document-grounded procedural sources.
- Experiments across six LLM backbones reveal that repository-grounded skill generation is significantly more challenging than document-based generation, highlighting a persistent gap between structural skill completeness and executable correctness.

---

[Democratizing Large-Scale Re-Optimization with LLM-Guided Model Patches](http://arxiv.org/abs/2605.18692)

- ReOpt-LLM (Re-Optimization Large Language Model): introduces an agentic framework that bridges end users and optimization models by translating natural-language prompts into structured model patches, utilizing an LLM, Patch Planner, Strategy Selector, Validator, Optimization Engine, Re-optimization Toolbox, Model, and Human-Machine Interface.
- The framework employs a closed-loop architecture where the Patch Planner generates auditable model edits, the Strategy Selector picks solver-aware techniques from the toolbox, and the Validator ensures feasibility through an iterative retry mechanism.
- By operationalizing OR expertise through structured model-editing and solver-aware orchestration, the framework enables continuous, scalable, and interpretable re-optimization of large-scale mixed-integer programs without requiring constant expert intervention.

---

[Reversa: A Reverse Documentation Engineering Framework for Converting Legacy Software into Operational Specifications for AI Agents](http://arxiv.org/abs/2605.18684)

- Reversa: introduces a multi-agent framework that converts legacy software into traceable operational specifications to guide LLMs in maintenance and migration tasks.
- The framework utilizes a specialized agent pipeline including Discovery-, Migration- and Code Forward-agents to transform implicit legacy knowledge into explicit, reviewable contracts.
- Reversa incorporates a confidence and gaps model to manage uncertainty, ensuring that LLMs operate on verified evidence rather than fragile inferences.

---

[CMAG: Concept-Scaffolded Retrieval for Marketplace Avatar Generation](http://arxiv.org/abs/2605.18680)

- CMAG: introduces a multi-stage framework for marketplace avatar generation that utilizes a 3D concept scaffold to disambiguate user prompts and guide the retrieval of topologically consistent assets.
- The framework integrates Concept Scaffolding, View-Aware Part Discovery, a Taxonomy Router, a Hybrid Retrieval Engine, an Agentic VLM Filter, and an Iterative VQA and Refinement Loop to ensure prompt faithfulness and structural coherence.
- By employing low-rank feature suppression and VLM-driven verification, the system effectively mitigates semantic-to-taxonomic misalignment and improves retrieval robustness in creator-driven 3D marketplaces.

---

[Generative AI Advertising as a Problem of Trustworthy Commercial Intervention](http://arxiv.org/abs/2605.18673)

- Generative AI Advertising Framework: introduces a taxonomy of commercial influence in LLMs, categorizing interventions into four tiers ranging from explicit product mentions to latent preference shaping.
- The paper analyzes how commercial interventions manifest across RAG and agentic pipelines, identifying critical challenges in identification, influence estimation, and contestability.
- It argues that generative AI advertising should be studied as a problem of trustworthy intervention rather than simple content placement, highlighting the risks of cascading influence in autonomous agentic systems.

---

[Position: A Three-Layer Probabilistic Assume–Guarantee Architecture Is Structurally Required for Safe LLM Agent Deployment](http://arxiv.org/abs/2605.18672)

- Three-Layer Probabilistic Assume–Guarantee Architecture: introduces a contract-based framework that decomposes LLM agent safety into three independently certified layers—User, Operational, and Functional—to address the structural insufficiency of single-layer enforcement.
- The architecture utilizes sequential assume–guarantee contracts to compose layer-specific safety guarantees into a system-level probabilistic bound via the chain rule of probability.
- This framework establishes a bidirectional assurance loop where bottom-up safety signals trigger plan recomputation when execution constraints are violated, ensuring safety in dynamic and uncertain environments.

---

[AI for Auto-Research: Roadmap &amp; User Guide](http://arxiv.org/abs/2605.18661)

- AI Auto-Research Framework: introduces a comprehensive taxonomy of AI-assisted research organized into four epistemological phases and eight stages that span the complete academic lifecycle.
- The framework identifies a critical capability boundary where LLMs excel at structured, tool-mediated tasks but remain fragile for open-ended scientific judgment and verification.
- Effective research automation requires layered architectures that integrate exploration, execution, and verification, with human-governed collaboration serving as the most reliable deployment paradigm.

---

[MementoGUI: Learning Agentic Multimodal Memory Control for Long-Horizon GUI Agents](http://arxiv.org/abs/2605.18652)

- MementoGUI: introduces a plug-in agentic memory framework that equips frozen GUI backbones with MementoCore, a learned controller for online memory selection, compression, and retrieval.
- The framework utilizes MementoCore to manage working memory for in-task state and episodic memory for reusable experience, enabling long-horizon GUI control without finetuning the action backbone.
- The system incorporates a scalable data curation pipeline and MementoGUI-Bench to evaluate memory-dependent decision-making, demonstrating consistent improvements over baseline history-based approaches.

---

[SPIKE: An Adaptive Dual Controller Framework for Cost-Efficient Long-Horizon Game Agents](http://arxiv.org/abs/2605.18636)

- SPIKE: introduces an adaptive dual controller framework that optimizes long-horizon game control by dynamically allocating strategic reasoning to event boundaries while maintaining cost-efficient reactive execution.
- The framework utilizes an Event Trigger to decide when to escalate from a lightweight Reactive Controller to a deliberative Strategic Controller, effectively managing the planning-latency-memory trilemma.
- Hierarchical Memory separates short-term local action reuse in the SA-MB from structured strategic evidence in the SA-KG, ensuring that each controller retrieves context relevant to its specific role.

---

[Mechanism Design for Connecting Regions Under Disruptions](http://arxiv.org/abs/2605.18626)

- Mechanism Design for Connecting Regions Under Disruptions: introduces a mechanism design framework to construct optimal pathways between disconnected regions by eliciting private agent locations to minimize social or maximum costs.
- The framework characterizes all anonymous strategyproof mechanisms as two-dimensional generalized median mechanisms, providing optimal solutions for social cost and deterministic/randomized approximations for maximum cost.
- The research establishes upper and lower bounds for approximation ratios of strategyproof mechanisms, utilizing both analytical proofs and computer-assisted experiments to evaluate performance under various disruption scenarios.

---

[CrossView Suite: Harnessing Cross-view Spatial Intelligence of MLLMs with Dataset, Model and Benchmark](http://arxiv.org/abs/2605.18621)

- CrossViewer: introduces a progressive three-stage framework for cross-view spatial reasoning in MLLMs, integrating a Shared Vision Encoder, Adaptive Region Tokenizer (ART), Retrieval Module, Object-Centric Cross-View Aligner (OCVA), LLM Adapter, and a Large Language Model.
- The framework utilizes an Adaptive Region Tokenizer to capture fine-grained object representations, followed by explicit cross-view alignment via the OCVA to establish object-level consistency across multiple viewpoints.
- CrossViewer is supported by the CrossViewSet instruction dataset and the CrossViewBench benchmark, enabling systematic evaluation of correspondence, visibility, geometric, and physical reasoning capabilities in MLLMs.

---

[Starve to Perceive: Taming Lazy Perception in VLMs with Constrained Visual Bandwidth](http://arxiv.org/abs/2605.18603)

- Starve to Perceive: introduces a training paradigm that constrains visual bandwidth to force VLMs to learn active perception as a necessary survival mechanism for task completion.
- The framework utilizes a two-stage pipeline consisting of Budget-Aware Visual Instruction Tuning and Reinforcement Learning with Perceptual Starvation to cultivate fine-grained localization skills.
- By restricting visual tokens per observation, the approach eliminates lazy perception, achieving state-of-the-art accuracy while significantly improving inference and training efficiency.

---

[Latent Action Reparameterization for Efficient Agent Inference](http://arxiv.org/abs/2605.18597)

- LAR (Latent Action Reparameterization): introduces a framework that learns a compact latent action space to collapse multi-step textual action sequences into single executable units, thereby reducing the effective decision horizon for LLMs.
- The framework utilizes a four-stage pipeline involving transition-equivalent segment identification, latent vocabulary construction, dual-format training data preparation, and trajectory-level distillation to integrate latent actions into LLMs via LoRA adapter and new latent action embeddings.
- By selectively abstracting low-entropy structural patterns while preserving high-entropy parameter-binding content, LAR achieves significant inference efficiency gains and maintains task performance across diverse LLM agent benchmarks.

---

[Not What You Asked For: Typographic Attacks in Household Robot Manipulation](http://arxiv.org/abs/2605.18593)

- HomeRobot: introduces a decoupled perception architecture to evaluate how typographic attacks propagate through a full-stack manipulation pipeline, causing kinetic failures via persistent semantic map poisoning.
- The framework utilizes a frozen CLIP encoder as a threshold-gated override mechanism that allows adversarial text to corrupt the agent's 3D semantic voxel map, leading to the grasping of incorrect objects.
- This research demonstrates that typographic misclassification is a physically consequential threat in household robotics, achieving a 67.8% attack success rate by exploiting the structural vulnerability of joint vision-language embedding spaces.

---

[Overeager Coding Agents: Measuring Out-of-Scope Actions on Benign Tasks](http://arxiv.org/abs/2605.18583)

- OVEREAGER-GEN (Overeager Generation): introduces a benchmark framework for measuring overeager behavior in LLM-based coding agents on benign tasks using Seed Pool, Mutator Family, Behavioral-Gradient Validator, Dual-Channel Audit Stack, Paired-Ablation Harness, Verdict Function, and Rule-Based Judge.
- The framework employs a behavioral-gradient validator to ensure scenarios distinguish between cautious and overeager agent profiles while using a dual-channel audit stack to capture both shell and internal tool calls.
- By utilizing a paired-ablation harness, the benchmark isolates the causal impact of prompt consent declarations on agent authorization-scope inference, demonstrating that framework-level gating is a primary driver of overeager behavior.

---

[When Outcome Looks Right But Discipline Fails: Trace-Based Evaluation Under Hidden Competitor State](http://arxiv.org/abs/2605.18580)

- Discipline Stability framework: introduces a trace-based evaluation paradigm to assess whether strategic economic agents preserve benchmark behavioral discipline under hidden competitor states, rather than solely optimizing scalar outcomes.
- The framework utilizes a Trace-Prior RL agent and a Corrected-History Student to demonstrate that preserving the uncertainty of hidden competitor states is essential for maintaining deployable market behavior.
- Experimental results across hotel pricing and hidden-budget bidding tasks show that reward-only RL baselines fail to recover benchmark traces, whereas the proposed trace-based approach effectively aligns agent behavior with intended market discipline.

---

[MA2P: A Meta-Cognitive Autonomous Intelligent Agents Framework for Complex Persuasion](http://arxiv.org/abs/2605.18572)

- MA2P: introduces a meta-cognitive autonomous agent framework for complex persuasion that coordinates perception, mental-state inference, strategy execution, memory, and evaluation to improve goal-directed dialogue.
- The framework utilizes a meta-cognitive Configurator to select high-level meta-strategies from a structured Knowledge Base, guiding the reasoning of a team of autonomous agents including Perception-, World Model-, and Persuader-agents.
- Experimental results demonstrate that MA2P consistently improves persuasion success and planning coherence across diverse domains while reducing cross-domain performance variance compared to base LLMs.

---

[LONGMINT: Evaluating Memory under Multi-Target Interference in Long-Horizon Agent Systems](http://arxiv.org/abs/2605.18565)

- LONGMINT (Long-Horizon Memory under INTerference): introduces an analytical benchmark designed to evaluate how memory-augmented agents perform in realistic, interference-heavy, long-horizon environments across diverse domains.
- The framework evaluates systems using a Memory Manager, Answering Agent, and Embedding Model to assess performance on single-target recall and multi-target aggregation tasks.
- Experimental results demonstrate that current memory systems struggle with interference and long-range lookback, identifying memory construction and retrieval as the primary bottlenecks in long-horizon agent performance.

---

[STT-Arena: A More Realistic Environment for Tool-Using with Spatio-Temporal Dynamics](http://arxiv.org/abs/2605.18548)

- STT-Arena: introduces a dynamic benchmark for evaluating LLM agent replanning capabilities under spatio-temporal environmental disruptions, utilizing Environment Curation, Spatio-Temporal Dynamic Injection, and Dual-Agent Assessment.
- The framework employs a Planning Agent to generate initial sequences and a Checking Agent to verify behavioral invariants, while a User Simulator provides grounded interaction and an LLM-as-a-Judge assesses task infeasibility.
- The research identifies three recurring failure modes—Stale-State Execution, Misdiagnosis of Dynamic Triggers, and Missing Post-Adaptation Verification—and proposes an iterative trajectory refinement technique to improve agent robustness.

---

[Beyond Scaling: Agents Are Heading to the Edge](http://arxiv.org/abs/2605.18535)

- Edge-Native Agentic Framework: introduces a paradigm shift for personal agents by moving executive control from cloud-centric architectures to edge-native environments to ensure structural coupling with local context.
- The framework utilizes Native Data Access, Real-time Grounding, Closed Action Loop, Zero-Cost Personalization, and Decentralized Learning to overcome the latency and context-loss limitations of remote cloud-based systems.
- It proposes an Artificial Anterior Cingulate Cortex (ACC) as a critical framework-level component for self-correction and conflict monitoring in resource-constrained edge deployments.

---

[One Developer Is All You Need: A Case Study of an AI-Augmented One-Person Squad in a Brownfield Enterprise](http://arxiv.org/abs/2605.18461)

- SDD (Spec-Driven Development) Framework: introduces a methodology where natural-language specifications serve as the primary artifact for AI-driven software construction, utilizing a Product Manager Agent, Specification Agent, Core Developer Agent, Non-core Developer Agent, CI/CD Pipeline, and Human-in-the-loop Validation.
- The framework enables a single engineer to manage a full-lifecycle software project by orchestrating specialized AI agents that handle distinct roles, effectively compressing a four-person squad into a one-person operation.
- Success in this framework relies on high-quality, unambiguous specifications and the directing engineer's deep institutional knowledge to act as a quality gate for AI-generated outputs.

---

[Code-as-Room: Generating 3D Rooms from Top-Down View Images via Agentic Code Synthesis](http://arxiv.org/abs/2605.18451)

- CaR (Code-as-Room): introduces an agentic framework that synthesizes 3D indoor rooms from top-down images by generating executable Blender code through a structured multi-stage pipeline.
- The framework utilizes a cross-stage memory module to maintain persistent scene state and mitigate context forgetting across specialized MLLM agent stages.
- A render-critique-revise loop, powered by a VLM-based critic, ensures spatial consistency and layout accuracy during the code generation process.

---

[Modelling Customer Trajectories with Reinforcement Learning for Practical Retail Insights](http://arxiv.org/abs/2605.18449)

- MaxEnt RL: introduces an agent-based modelling framework that utilizes maximum entropy reinforcement learning to predict realistic customer trajectories for retail layout optimization.
- The framework employs a conditional MaxEnt RL agent trained via PPO to capture bounded rationality and diverse shopping behaviors, outperforming traditional TSP and PNN heuristics.
- By accurately estimating shelf traffic density and impulse purchase rates, the approach enables data-driven product repositioning that yields profit gains comparable to those derived from ground-truth human data.

---

[EvoMemBench: Benchmarking Agent Memory from a Self-Evolving Perspective](http://arxiv.org/abs/2605.18421)

- EvoMemBench: introduces a unified benchmark for evaluating LLM agent memory across two axes: memory scope (in-episode vs. cross-episode) and memory content (knowledge-oriented vs. execution-oriented).
- The framework evaluates 15 representative memory methods, including retrieval-augmented, short-term, general long-term, procedural long-term, and meta-evolution memory architectures.
- Experimental results demonstrate that while memory helps when context is insufficient or tasks are difficult, no single memory form consistently outperforms others across all settings, highlighting the need for specialized memory systems.

---

[Prompts Don’t Protect: Architectural Enforcement via MCP Proxy for LLM Tool Access Control](http://arxiv.org/abs/2605.18414)

- Governed MCP Proxy: introduces an architectural enforcement layer for LLM tool access control that filters tool registries at discovery time using ABAC (attribute-based access control) to prevent unauthorized tool exposure.
- The framework utilizes a JWT Verify component for identity, an ABAC Policy for authorization, and a Filter Tools mechanism to ensure only permitted tools reach the LLM Context.
- By implementing a secondary ABAC 2nd check at the invocation stage, the system provides a structural guarantee of 0% unauthorized tool invocation, effectively mitigating risks from prompt injection and role escalation.

---

[SKILLSVOTE: Lifecycle Governance of Agent Skills from Collection, Recommendation to Evolution](http://arxiv.org/abs/2605.18401)

- SkillsVote: introduces a lifecycle-governance framework for LLM agent skills that manages the collection, recommendation, and evolution of reusable experience artifacts.
- The framework utilizes a subtask-level attribution layer to filter noisy execution trajectories, ensuring only successful and reusable explorations trigger evidence-gated updates to the skill library.
- By implementing task-conditioned recommendation and conservative library evolution, SkillsVote improves agent performance on complex terminal and software-engineering benchmarks without requiring model parameter updates.

---

[Duet instrumentation: An Agentic Approach to Improving Sensitivity in Cloud Service Benchmarking](http://arxiv.org/abs/2605.18397)

- Duet instrumentation: introduces a benchmarking paradigm that integrates change-localized measurements into duet application benchmarks to improve sensitivity without requiring dedicated microbenchmark suites.
- The system utilizes an LLM-based agent to automatically identify performance-relevant code changes and insert lightweight monitoring hooks, which are then evaluated in a synchronized parallel execution environment.
- Experimental results demonstrate that the approach detects performance regressions at up to 5× lower injected severity compared to traditional duet benchmarking while maintaining minimal instrumentation overhead.

---

[NEWTON: Agentic Planning for Physically Grounded Video Generation](http://arxiv.org/abs/2605.18396)

- NEWTON: introduces an agentic framework that demotes video generation to a tool-use action, orchestrating a trainable planner, a frozen video generator, and a verifier in an iterative loop to ensure physical grounding.
- The system utilizes a Planner to select from a library of physics-aware tools—Keyframe Generation, Python Computation, and Prompt Refiner—to enrich conditioning signals for the frozen video generator.
- The Planner is optimized on-policy via Flow-GRPO within the live multi-turn loop, enabling the discovery of scene-dependent tool-use strategies that improve physical commonsense without modifying the underlying video generator.

---

[Diagnosing Korean-Language LLM Political Bias via Census-Grounded Agent Simulation](http://arxiv.org/abs/2605.18395)

- Dynamo-K: introduces a census-grounded agent-simulation framework to diagnose political bias in LLMs across Korean elections using Data Collection, Preprocessing, Agent Factory, Belief Seeding &amp; Calibration, ORC Simulation, and Aggregation &amp; Evaluation.
- The framework identifies systematic failure modes including progressive bias in moderate agents, third-party salience collapse, and regional polarization collapse through a structured four-stage pipeline.
- Dynamo-K utilizes an ORC (Observation-Reasoning-Conclusion) pipeline with LLMs to simulate voter behavior, achieving high accuracy in presidential winner predictions while providing a low-cost diagnostic tool for electoral analysis.

---

[Same Signal, Different Semantics: A Cross-Framework Behavioral Analysis of Software Engineering Agents](http://arxiv.org/abs/2605.18332)

- Framework Behavioral Analysis of Software Engineering Agents: introduces a per-configuration meta-analysis of 64,380 trajectories to disentangle framework and LLM effects on agent behavior.
- The study establishes that framework identity is the primary driver of behavioral variation, often carrying opposite meanings for identical observable signals across different agent configurations.
- The research classifies behavioral signals into direction-stable and direction-unstable categories, providing a framework-aware decision guide for practitioners to optimize agent performance.

---

[Causely: A Causal Intelligence Layer for Enterprise AI](http://arxiv.org/abs/2605.18327)

- Causely: introduces a causal intelligence layer that transforms raw observability telemetry into a structured, queryable model to provide the semantic foundation required for LLM agents to perform reliable SRE tasks.
- The framework utilizes a Topology Graph, Causal Knowledge Base, Causality Graph, and Attribute Dependency Graph to replace open-ended environment interpretation with targeted causal lookups.
- Empirical benchmarks demonstrate that providing LLM agents with this causal intelligence layer significantly reduces latency, token consumption, and tool-call frequency while improving diagnostic accuracy.

---

[SD-Search: On-Policy Hindsight Self-Distillation for Search-Augmented Reasoning](http://arxiv.org/abs/2605.18299)

- SD-Search (On-Policy Hindsight Self-Distillation for Search-Augmented Reasoning): introduces a reinforcement learning framework that provides dense step-level supervision for search-augmented reasoning agents by aligning a Student-Agent to a Teacher-Agent using a Hindsight-Information-Block and a Jensen-Shannon-Divergence-Loss.
- The framework utilizes a single policy model acting as both Student-Agent and Teacher-Agent, where the Teacher-Agent conditions on a Hindsight-Information-Block containing sibling rollout outcomes to provide step-level guidance without external annotations.
- By integrating the Jensen-Shannon-Divergence-Loss with the GRPO-Optimizer, the approach improves query quality and reasoning performance on multi-hop QA benchmarks while maintaining efficiency within the standard RL training loop.

---

[CommitDistill: A Lightweight Knowledge-Centric Memory Layer for Software Repositories](http://arxiv.org/abs/2605.18284)

- CommitDistill: introduces a lightweight, deterministic memory layer that distills software repository history into typed knowledge units to improve LLM agent decision-making.
- The framework utilizes regex-based extraction to categorize repository data into Facts, Skills, and Patterns, which are then stored in an inspectable JSON format for efficient retrieval.
- CommitDistill provides a trust-instrumented, dependency-free substrate that demonstrates superior payload-efficient retrieval under constrained token budgets compared to standard lexical baselines.

---

[From Volume to Value: Preference-Aligned Memory Construction for On-Device RAG](http://arxiv.org/abs/2605.18271)

- EPIC: introduces a framework for building compact, preference-aligned on-device memory by selectively retaining and indexing information relevant to user preferences.
- The pipeline utilizes Semantic-Based Coarse Filtering, Preference-Aligned Fine Verification, and Preference-Guided Query Steering to ensure retrieved content remains grounded in personal context under strict memory constraints.
- By indexing preference-conditioned instructions rather than raw data, the system achieves significant memory reduction and lower retrieval latency while maintaining high preference-following accuracy for on-device LLMs.

---

[Privacy Preserving Reinforcement Learning with One-Sided Feedback](http://arxiv.org/abs/2605.18246)

- POOL (Privacy-Oriented One-Sided Learning): introduces a privacy-preserving RL algorithm for multi-dimensional continuous MDPs with one-sided feedback, utilizing Gaussian Mechanism, Partial Discretization Strategy, Multi-dimensional Piecewise-Linear Approximation, and Private Value Function Estimator.
- The framework addresses the high computational complexity of continuous spaces and the systematic bias of partial observability by partitioning the state-action space into zones and applying piecewise-linear interpolation.
- Theoretical analysis confirms that POOL satisfies ρ-zero-concentrated differential privacy while achieving sample complexity bounds that match non-private lower bounds.

---

[Non-Colliding Biometric Identities for Digital Entities: Geometry, Capacity, and Million-Scale Virtual Identity Provisioning](http://arxiv.org/abs/2605.18238)

- BIP (Biometric Identity Provisioning): introduces a framework for allocating non-colliding biometric identities to digital entities by packing virtual embeddings into unclaimed gaps within the real face manifold.
- The framework utilizes repulsion-based allocation to generate virtual identity embeddings and GapGen to realize these embeddings as high-fidelity portrait images.
- The approach includes IAPCT as a diagnostic tool to support real-vs-virtual detection and unified recognition protocols on the constructed v-LFW benchmark.

---

[Beyond the Cartesian Illusion: Testing Two-Stage Multi-Modal Theory of Mind under Perceptual Bottlenecks](http://arxiv.org/abs/2605.18194)

- Observe-to-Believe: introduces a two-stage framework that disentangles geometric observation from epistemic inference to overcome the Cartesian Illusion in embodied Theory of Mind.
- The framework utilizes a Gemini-2.5-Pro observation engine to extract structured physical evidence and a DeepSeek-v4-Flash reasoning engine to execute modality-aware perspective shifts based on inferred sensory bottlenecks.
- By explicitly modeling the spatial horizon, the pipeline dynamically routes reasoning between visual and audio-motion pathways, significantly improving performance in occluded or invisible multi-agent scenarios.

---

[The Dynamics of Policy Gradient in Social Dilemmas with Partner Selection](http://arxiv.org/abs/2605.18185)

- Policy Gradient Dynamics in Social Dilemmas with Partner Selection: introduces a formal theoretical framework to analyze how partner selection rules reshape reward landscapes and influence the emergence of cooperation in multi-agent reinforcement learning.
- The paper utilizes mean-field theory and a two-dimensional Wiener process to derive a stochastic model that accurately captures the policy gradient dynamics and the evolution of strategy distributions.
- Analytical results demonstrate that population variance is a necessary condition for cooperation, and simulations confirm that the derived Fokker-Planck equation effectively models the long-term strategy distribution under various partner selection mechanisms.

---

[Scalable Environments Drive Generalizable Agents](http://arxiv.org/abs/2605.18181)

- Scalable Environments Drive Generalizable Agents: introduces a taxonomy for agent scaling that distinguishes between trajectory, task, and environment scaling to address world-level distribution shifts.
- The paper synthesizes programmatic generators and generative world models as primary paradigms for creating diverse, verifiable, and controllable environments.
- It proposes standardized evaluation criteria—including executability, signal quality, coverage, complexity, and efficiency—to measure progress toward robust general agents.

---

[MARS: Technical Report for the CASTLE Challenge at EgoVis 2026](http://arxiv.org/abs/2605.18176)

- MARS (Multimodal Agentic Reasoning with Source selection): introduces an agentic framework that performs iterative evidence selection across multimodal sources to solve complex long-horizon egocentric question answering tasks.
- The system utilizes a GPT-5.4 decision agent to dynamically manage a compact evidence state, selectively querying video summaries, transcripts, and auxiliary modalities based on the specific requirements of the input question.
- By integrating HCQA-style long-video compression with a flexible source-selection loop, the framework effectively navigates large-scale, multi-day datasets while maintaining computational efficiency and reasoning accuracy.

---

[Three Heads Are Better Than One: A Multi-perspective Reasoning Framework for Enhanced Vulnerability Detection](http://arxiv.org/abs/2605.18153)

- ReasonVul: introduces a multi-agent framework that leverages cognitive synergy among Deductive Agent (applies security rules top-down), Inductive Agent (uses pattern-matching via RAG), Abductive Agent (reasons backward from hypothesized outcomes), Security Rules Knowledge Base (contains formal coding standards), Vulnerability Code Knowledge Base (contains historical vulnerability-fix pairs), and a Collaborative Debate Mechanism (resolves conflicts through iterative discourse) to enhance vulnerability detection.
- The framework utilizes specialized LLM agents to perform independent analyses followed by a structured debate to synthesize diverse perspectives and resolve disagreements.
- Experimental results on the PrimeVul and JITVUL datasets demonstrate that ReasonVul significantly outperforms existing state-of-the-art methods by effectively capturing complex, context-dependent vulnerabilities.

---

[Whispers in the Noise: Surrogate-Guided Concept Awakening via a Multi-Agent Framework](http://arxiv.org/abs/2605.18150)

- ConceptAgent: introduces a training-free, multi-agent framework that awakens erased concepts in diffusion models by injecting surrogate-guided structured noise into intermediate denoising states.
- The framework utilizes a Strategist Agent to derive surrogate concepts, a Guesser Agent to steer the denoising trajectory, a Director Agent to perform physically-aware scene composition, and a Referee Agent to ensure output fidelity.
- By operating on intermediate denoising states rather than textual prompts, the framework effectively bypasses concept erasure mechanisms without requiring access to model parameters, gradients, or internal representations.

---

[Evidence-Grounded Frontier Mapping and Agentic Hypothesis Generation in Nanomedicine](http://arxiv.org/abs/2605.18144)

- pArticleMap: introduces a human-centered system for evidence-grounded literature mapping and hypothesis generation in nanomedicine that combines article-level representation learning, graph-based frontier detection, and an audited multi-step LLM workflow.
- The framework utilizes Data Ingestion & Processing, Interactive Exploration, Novelty Detection & Gap Analysis, Evidence Packaging & Backend, Agentic Workflow & Generation, and Retrospective Evaluation to identify sparse literature regions and generate grounded research hypotheses.
- The system employs an agentic workflow that includes explanation-, audit-, retrieval-patching-, ideation-, scoring-, and blueprinting-agents to ensure scientific hypotheses are grounded in retrieved evidence and auditable by human experts.

---

[TaskGround: Structured Executable Task Inference for Full-Scene Household Reasoning](http://arxiv.org/abs/2605.18109)

- TaskGround: introduces a training-free, model-agnostic framework that decomposes full-scene household reasoning into Scene Grounder, Task-Structure Inference Module, Completion Module, and Skill-Level Executor.
- The framework grounds complete household scenes into compact task-relevant slices to reduce input-token costs and improve reasoning performance for both proprietary and open-weight LLMs.
- TaskGround utilizes the FullHome benchmark to demonstrate that structured task-structure inference acts as a critical bottleneck for household agents, enabling compact models to achieve performance competitive with frontier LLMs.

---

[Equilibrium Selection in Multi-Agent Policy Gradients via Opponent-Aware Basin Entry](http://arxiv.org/abs/2605.18078)

- Meta-MAPG: introduces a basin-entry mechanism for multi-agent reinforcement learning that decomposes policy updates into ordinary policy gradient, own-learning correction, and peer-learning correction components.
- The peer-learning correction acts as the primary equilibrium-selection mechanism by shifting the certified attraction region of stable Nash equilibria under a local alignment condition.
- A shape-then-cool schedule is employed to anneal the peer-learning correction after basin entry, ensuring convergence to the original Nash equilibrium of the stochastic game.

---

[A-ProS: Towards Reliable Autonomous Programming Through Multi-Model Feedback](http://arxiv.org/abs/2605.18073)

- A-ProS: introduces an autonomous agentic framework that improves competitive programming performance by integrating specialized LLMs into a closed-loop, feedback-driven workflow.
- The framework utilizes a Solution Generator to produce initial code and a Debugging Critic to provide structured, role-specific feedback, enabling iterative refinement through persistent conversation context.
- Experimental results on 367 competitive programming problems demonstrate that A-ProS significantly outperforms stateless baselines by leveraging persistent context to reduce repeated failure modes.

---

[PPAI: Enabling Personalized LLM Agent Interoperability for Collaborative Edge Intelligence](http://arxiv.org/abs/2605.18067)

- PPAI: introduces a decentralized P2P system that enables personalized LLM agents to collaborate by routing tasks to the most suitable peer based on specialized expertise and real-time load conditions.
- The framework utilizes a prototype-anchored scoring module to map queries and agents into a shared latent space, facilitating efficient matching in dynamic environments with churning agents.
- A Bayesian game-theoretic scheduler incorporates belief-based load estimation to optimize global system utility and minimize latency, effectively balancing task relevance against network congestion.

---

[PROTEA: Offline Evaluation and Iterative Refinement for Multi-Agent LLM Workflows](http://arxiv.org/abs/2605.18032)

- PROTEA: introduces a unified interface for offline, test-driven refinement of multi-agent LLM workflows by surfacing node-level evidence and automating prompt revisions.
- The framework utilizes backward node evaluation to generate expectations from final-answer references, enabling localized diagnosis of bottlenecks within complex multi-agent graphs.
- PROTEA integrates an interactive loop where developers inspect node-level rationales, edit suggested prompt revisions, and automatically re-evaluate workflow performance to ensure stable improvements.

---

[TeleCom-Bench: How Far Are Large Language Models from Industrial Telecommunication Applications?](http://arxiv.org/abs/2605.18025)

- TeleCom-Bench: introduces a comprehensive benchmark for evaluating LLMs in telecommunications, utilizing a Knowledge Comprehension Pipeline and a Knowledge Application Pipeline to measure performance across foundational theory and industrial workflows.
- The framework identifies a critical "Execution Wall" where LLMs perform well as diagnosticians but fail to generate executable remediation plans due to deficiencies in procedural synthesis.
- TeleCom-Bench integrates proprietary product manuals and real-world network trajectories to provide a standardized metric for assessing the operational readiness of LLMs in autonomous telecom environments.

---

[Interaction-Breaking Adversarial Learning Framework for Robust Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2605.18024)

- IBAL (Interaction-Breaking Adversarial Learning): introduces a robust MARL framework that constructs adversarial attacks by minimizing cross-group mutual information to disrupt inter-agent coordination and trains policies to remain effective under such interaction-breaking scenarios.
- The framework utilizes an Observation Attacker to mask influential cross-group observation dimensions and an Action Attacker to perturb actions, effectively suppressing cross-group influence while maintaining cooperative performance.
- Empirical results demonstrate that IBAL significantly improves robustness against diverse adversarial attacks and non-parametric perturbations compared to existing robust MARL baselines.

---

[Federated Learning by Utility-Constrained Stochastic Aggregation for Improving Rational Participation](http://arxiv.org/abs/2605.18020)

- FedUCA: introduces a framework that formalizes server-side aggregation as a constrained feasibility problem to maximize global model performance while satisfying individual rationality constraints of rational clients.
- The framework utilizes a stochastic aggregation mechanism with Dirichlet-sampled mixture weights to exploit the Jensen's gap, creating a utility surplus that sustains client participation under statistical heterogeneity.
- FedUCA incorporates a fallback mechanism and stale update retention to ensure consistent participation and long-term engagement in cross-silo environments without requiring monetary incentives.

---

[Shared Backbone PPO for Multi-UAV Communication Coverage with Connection Preservation](http://arxiv.org/abs/2605.17999)

- Shared Backbone PPO: introduces a multi-agent reinforcement learning architecture that utilizes a shared Graph Aggregator module between Actor and Critic networks to improve cooperative communication coverage in UAV swarms.
- The framework optimizes the shared Graph Aggregator parameters using gradients from the Critic branch to resolve conflicts between Actor and Critic objectives during training.
- Experimental results demonstrate that the shared backbone design combined with neighborhood information aggregation achieves superior convergence and coverage performance compared to independent Actor-Critic PPO variants.

---

[Verify-Gated Completion as Admission Control in a Governed Multi-Agent Runtime: A Bounded Architecture Case Study](http://arxiv.org/abs/2605.17998)

- Five-plane architecture: introduces a runtime-design framework that separates work execution from completion admission using explicit governance, packetized state, and read-only verification.
- The framework utilizes a read-only verify gate and packetized evidence to ensure that LLM-based agent completion claims are inspectable, auditable, and fail-closed.
- The system incorporates a shadow Policy/Governance Verifier (PGV) to provide advisory governance signals without replacing the primary admission authority.

---

[LivePI: More Realistic Benchmarking of Agents Against Indirect Prompt Injection](http://arxiv.org/abs/2605.17986)

- LivePI: introduces a structured benchmark for evaluating indirect prompt injection risks in LLM agents within a production-like, test-controlled virtual machine environment.
- The framework evaluates agent vulnerability across seven input surfaces and five malicious goals, identifying that frontier LLMs remain susceptible to indirect prompt injection.
- The authors propose a two-layer defense mechanism combining prompt-level filtering and pre-execution policy-based tool authorization to mitigate risks while maintaining agent utility.

---

[Generation Navigator: A State-Aware Agentic Framework for Image Generation](http://arxiv.org/abs/2605.17969)

- Generation Navigator: introduces a multi-turn agentic framework that reformulates image generation as a state-conditioned action-making problem, utilizing a Navigator (multimodal LLM action-making agent), Generator (text-to-image/image-to-image model), Reviewer (multimodal LLM evaluation interface), PRE-GRPO (trajectory-level reinforcement learning objective), and Trajectory Data Pipeline (structured multi-turn data construction).
- The framework employs PRE-GRPO to decompose trajectory rewards into peak discovery, retention, and efficiency, enabling the agent to learn optimal, non-regressive generation trajectories.
- Experimental results demonstrate that Generation Navigator significantly outperforms existing one-shot and agentic baselines on benchmarks like T2I-ReasonBench and WISE, while maintaining a favorable quality-latency trade-off.

---

[BLAgent: Agentic RAG for File-Level Bug Localization](http://arxiv.org/abs/2605.17965)

- BLAgent: introduces a novel agentic RAG framework for file-level bug localization that integrates Path-Aware Code Chunking, Vector Database, Query Transformation, ReAct Agent, FileViewingTool, Skeleton-Based Agent Scoring (SAS), and Evidence-Anchored Reranking (EAR) to systematically narrow the search space for faulty files.
- The framework utilizes AST-aware chunking and dual-perspective query transformation to bridge the semantic gap between bug reports and code, while employing a two-phase agentic reranking process to achieve high-precision localization.
- BLAgent significantly outperforms existing baselines on the SWE-bench Lite dataset, demonstrating that precise file-level localization directly improves end-to-end automated program repair success rates.

---

[SVFSearch: A Multimodal Knowledge-Intensive Benchmark for Short-Video Frame Search in the Gaming Vertical Domain](http://arxiv.org/abs/2605.17946)

- SVFSearch: introduces a multimodal benchmark for short-video frame search in the gaming domain, utilizing a Plan-Act-Replan Agentic Framework, MMSearch-R1-Game, and a Frozen Retrieval Environment to evaluate LLMs on complex, domain-specific visual and textual queries.
- The framework integrates DINOv3-Base Retrieval Encoder and Qwen3-VL-Embedding-2B to facilitate precise visual grounding and evidence retrieval within a standardized offline repository.
- Experimental results demonstrate that while retrieval-augmented workflows significantly improve performance, persistent challenges remain in visual grounding, tool-use control, and evidence-grounded reasoning for LLMs.

---

[BacktestBench: Benchmarking Large Language Models for Automated Quantitative Strategy Backtesting](http://arxiv.org/abs/2605.17937)

- AutoBacktest: introduces a multi-agent framework that automates quantitative backtesting by coordinating a Summarizer, Retriever, and Coder to translate natural language strategies into reproducible results.
- The framework utilizes a BM25-augmented retrieval mechanism and standardized Short Codes to ensure precise indicator mapping and robust SQL generation for historical market data.
- AutoBacktest includes planning-, perception- and tool use-agents that perform iterative code generation and verification to mitigate LLM hallucinations in complex financial reasoning tasks.

---

[AtlasVA: Self-Evolving Visual Skill Memory for Teacher-Free VLM Agents](http://arxiv.org/abs/2605.17933)

- AtlasVA: introduces a teacher-free visual skill memory framework that organizes experience into Visual Skill Memory, Spatial Heatmaps, Visual Exemplars, Symbolic Text Skills, Teacher-Free Atlas Evolution, Dense Visual Reward Shaping, and a Perception-Optimization Loop to enable autonomous VLM agent self-improvement.
- The framework replaces lossy text-only memory with a multimodal hierarchy that aligns reusable experience with the agent's native visual perception.
- By bootstrapping spatial priors from trajectory statistics, AtlasVA provides dense, coordinate-aware reward shaping that mitigates sparse-reward bottlenecks in complex spatial tasks.

---

[TRANSFER LEARNING FOR CUSTOMIZED CAR RACING ENVIRONMENTS](http://arxiv.org/abs/2605.17928)

- Transfer Learning for Customized Car Racing Environments: evaluates the efficacy of model-based and model-free RL approaches in transferring learned knowledge across customized car racing environments with varying dynamics and track geometries.
- The study compares PPO, SAC, DDPG, and Dreamer, demonstrating that the model-based Dreamer agent exhibits superior sample efficiency, faster convergence, and robust performance in zero-shot and fine-tuned transfer scenarios.
- Experimental results indicate that while model-free methods often require extensive hyperparameter tuning and exhibit brittle convergence, model-based agents effectively leverage latent world models to maintain high performance across diverse environmental modifications.

---

[An Efficient Streaming Video Understanding Framework with Agentic Control](http://arxiv.org/abs/2605.17921)

- R3-Streaming: introduces a cascaded agentic control framework that optimizes streaming video understanding by dynamically coordinating memory compression, response timing, and compute routing.
- The framework utilizes Active Forgetting to maintain high-fidelity recent context, a Readiness Head to defer premature responses, and TB-GRPO to stabilize compute routing between a Fast Model and a Slow Reasoning Model.
- By treating streaming as a joint control problem, R3-Streaming achieves state-of-the-art performance on streaming benchmarks while significantly reducing visual token usage through selective, task-driven escalation.

---

[WorldArena 2.0: Extending Embodied World Model Benchmarking on Modality, Functionality and Platform](http://arxiv.org/abs/2605.17912)

- WorldArena 2.0: introduces a comprehensive benchmark for evaluating embodied world models across modality, functionality, and platform dimensions using Tactile VAE, Visuotactile Two-Stream World Model, Action Diffusion Head, World Model Environment, Reward Model, Policy Model, and Optimization Module.
- The framework extends evaluation from vision-only to visuotactile modalities and utilizes world models as interactive environments for online RL training.
- Experiments across 12 models on simulated and real-world platforms reveal a significant sim-to-real usability gap in current embodied world models.

---

[A Pilot Benchmark for NL-to-FOL Translation in Planetary Exploration](http://arxiv.org/abs/2605.17911)

- NL-to-FOL Benchmark: introduces a dataset for translating natural language mission documentation into First-Order Logic (FOL) representations to support autonomous reasoning in planetary exploration.
- The framework utilizes Natural Language Input, First-Order Logic (FOL) Representation, Predicate Vocabulary, Typed Constants, and an Autonomous Agent to bridge the gap between high-level mission intent and formal machine-interpretable logic.
- The research evaluates several LLMs on their ability to perform direct, single-pass translation of complex, long-form mission narratives into structured logical forms, identifying significant challenges in maintaining temporal and logical consistency.

---

[Ethical Hyper-Velocity (EHV): A Provably Deterministic Governance-Aware JIT Compiler Architecture for Agentic Systems](http://arxiv.org/abs/2605.17909)

- EHV (Ethical Hyper-Velocity): introduces a hardware-rooted architectural framework that integrates a Policy Compiler, CRDTs, and a TEE-based JIT PEP to enable real-time, provably deterministic AI governance.
- The framework utilizes an Action Schema Extraction Layer (ASEL) to parse LLM outputs into structured formats, allowing the JIT PEP to enforce safety invariants at the inference pipeline level.
- By leveraging epoch-based attestation caching and CRDT-synchronized policy states, EHV reduces governance latency from days to sub-millisecond intervals while maintaining formal safety guarantees.

---

[One Model to Translate Them All: Universal Any-to-Any Translation for Heterogeneous Collaborative Perception](http://arxiv.org/abs/2605.17907)

- UniTrans: introduces a universal any-to-any feature translation framework that instantiates mapping-specific translators on the fly for heterogeneous collaborative perception.
- The framework utilizes a Modality-Intrinsic Encoder to map features into a latent space, enabling the Modality Mapping Router to synthesize a dedicated translator from a reusable Translator Parameter Bank.
- This approach achieves zero-shot feature translation across diverse sensor modalities without requiring repeated retraining or fine-tuning for newly emerging agents.

---

[Agentic Chunking and Bayesian De-chunking of AI Generated Fuzzy Cognitive Maps: A Model of the Thucydides Trap](http://arxiv.org/abs/2605.17903)

- ACBD: introduces a methodology for decomposing large documents into representative knowledge graphs using LLM agents to perform text chunking and mapping to feedback Fuzzy Cognitive Maps.
- The framework utilizes convex mixing of sparse causal edge matrices to aggregate chunk-based FCMs into a unified global FCM and applies Bayesian-like operators to derive posterior FCMs.
- The approach enables what-if causal analysis through node clamping and pulsing, demonstrating the system's ability to predict dynamical system equilibria in complex scenarios like the Thucydides Trap.

---

[DuIVRS-2: An LLM-based Interactive Voice Response System for Large-scale POI Attribute Acquisition](http://arxiv.org/abs/2605.17900)

- DuIVRS-2: introduces an end-to-end LLM-based framework for large-scale POI attribute acquisition that replaces traditional modular IVR systems with a more stable and efficient architecture.
- The framework utilizes FSM-Guided Data Augmentation, Selective Generation, and a CoT Mechanism to ensure output stability and mitigate hallucinations in industrial settings.
- A Cooperative Iterative Learning strategy, employing both LLM-L and Black-box LLM evaluators, progressively refines the system performance while minimizing manual annotation requirements.

---

[Multi-agent AI systems outperform human teams in creativity](http://arxiv.org/abs/2605.17885)

- Semantic Trajectory Analysis Framework: introduces a quantitative method to evaluate multi-agent creativity by mapping conversational turns as paths through neural embedding space.
- The study demonstrates that multi-agent LLM teams significantly outperform human teams in creativity, driven by superior novelty while maintaining comparable usefulness.
- The research identifies model choice and discussion structure as complementary design levers that explain 26.8% of the variance in LLM conversational dynamics and creative outcomes.

---

[PAIR: Prefix-Aware Internal Reward Model for Multi-Turn Agent Optimization](http://arxiv.org/abs/2605.17877)

- PAIR (Prefix-Aware Internal Reward Model): introduces a two-stage architecture that decouples internal belief-consistency from grounded correctness to provide dense step-level rewards for LLMs.
- The framework utilizes a frozen hidden-state probe to estimate belief-consistency and an attention-based correction head to adjust for grounded correctness, effectively mitigating prefix contamination.
- PAIR operates at probe-level inference cost without external model calls or runtime ground-truth dependencies, enabling efficient reinforcement learning for multi-turn agent tasks.

---

[HINT-SD: Targeted Hindsight Self-Distillation for Long-Horizon Agents](http://arxiv.org/abs/2605.17873)

- HINT-SD: introduces a targeted self-distillation framework that uses full-trajectory hindsight to identify failure-relevant action spans and applies feedback-conditioned distillation only to those specific turns.
- The framework utilizes a Hindsight Analyzer to generate corrective natural-language feedback for selected steps, which then conditions a Teacher to provide localized supervision to the Student.
- By narrowing the optimization landscape to failure-relevant regions, HINT-SD improves training efficiency and performance on long-horizon agent tasks compared to dense per-turn or trajectory-level feedback methods.

---

[f-OPD: Stabilizing Long-Horizon On-Policy Distillation with Freshness-Aware Control](http://arxiv.org/abs/2605.17862)

- f-OPD: introduces a freshness-aware framework that mitigates objective discrepancy in asynchronous on-policy distillation by adaptively weighting samples and constraining policy drift.
- The framework utilizes sample-level freshness scoring based on rollout drift and supervision drift diagnostics to balance training throughput and optimization fidelity.
- f-OPD incorporates rollout-anchored regularization and an adaptive buffer refresh mechanism to maintain stability in long-horizon agentic tasks.

---

[KISS – Knowledge Infrastructure for Scientific Simulation: A Scaffolding for Agentic Earth Science](http://arxiv.org/abs/2605.17856)

- KISS (Knowledge Infrastructure for Scientific Simulation): introduces a structured operational scaffold that externalizes tacit scientific expertise into validated modelling operators, staged domain protocols, and diagnostic recovery mechanisms to enable reliable LLM agent execution of process-based models.
- The Knowledge Dissection Toolkit (KDT) automates the extraction of this operational knowledge from documentation and source code, creating portable KI packages that allow LLMs to perform complex Earth-science simulations end-to-end.
- Empirical benchmarks across 119 models demonstrate that KI significantly improves agent reliability by providing a domain-invariant structure for procedural, evaluative, and diagnostic tasks, effectively bridging the gap between non-specialist users and specialized scientific modelling.

---

[Learning Empirical Evidence Equilibria under Weak Environmental Coupling](http://arxiv.org/abs/2605.17848)

- EEE (Empirical Evidence Equilibrium) framework: introduces a decentralized learning approach for multi-agent systems where agents with bounded rationality form misspecified internal models to make decisions under partial observability.
- The framework utilizes Q-value iteration with per-iteration model updates to achieve a steady-state equilibrium in stochastic games where agents' actions have a bounded influence on the environment.
- Convergence to an EEE or approximate EEE is guaranteed under conditions of weak coupling between agent actions and environment transition dynamics.

---

[Agentic Cost-Aware Query Planning with Knowledge Distillation for Big Data Analytics](http://arxiv.org/abs/2605.17831)

- Agentic Query Planning Framework: introduces an integrated system that combines a rule-based Teacher Planner, UCB1 Bandit Search, a Random Forest Cost Model, and a distilled Student Planner to optimize big data queries under resource constraints.
- The system utilizes the Teacher Planner to generate SQL plans, which are then evaluated by the Bandit Search and Cost Model to ensure latency and memory constraints are met.
- Knowledge distillation is employed to train a lightweight Student Planner that mimics the teacher-bandit decisions, achieving significant inference speedups for production environments.

---

[Remembering More, Risking More: Longitudinal Safety Risks in Memory-Equipped LLM Agents](http://arxiv.org/abs/2605.17830)

- Event-based framework: introduces a methodology to isolate memory-induced safety risks in LLMs by decomposing failures into preconditions, triggers, and violations.
- The study demonstrates that benign memory accumulation in LLMs leads to temporal memory contamination, where retrieved historical data causes increasingly unsafe behavior over time.
- The authors propose a retrieval-time diagnostic monitor that achieves high recall in detecting memory-induced risks before generation by analyzing retrieval-time features.

---

[Interactive Evaluation Requires a Design Science](http://arxiv.org/abs/2605.17829)

- Interactive Evaluation Framework: introduces a design science approach for evaluating LLMs acting through consequential trajectories, shifting from static response-centered benchmarks to system-level performance assessment.
- The framework defines evaluation as an autonomous mapping E : X → Y, where X represents interaction-generated trajectories and E represents the evaluation program assessing process, recoverability, coordination, and robustness.
- It proposes a two-axis taxonomy—evaluation inputs and evaluation programs—to standardize the evaluation of LLM systems across diverse domains like coding agents and multi-agent social systems.

---

[Why We Look Where We Look: Emergent Human-like Fixations of a Foveated Visual Language Model Maximizing Scene Understanding](http://arxiv.org/abs/2605.17823)

- fRL-SU: introduces a computational agent that learns optimal eye movements by maximizing scene comprehension under the biological constraints of foveated vision.
- The framework utilizes a VLM to generate scene descriptions, with an RL agent trained to select fixation locations that maximize semantic accuracy or minimize description entropy.
- The model demonstrates that human-like fixation patterns, such as prioritizing people and text, emerge as a functional byproduct of optimizing scene understanding with foveated vision.

---

[HydroAgent: Closing the Gap Between Frontier LLMs and Human Experts in Hydrologic Model Calibration via Simulator-Grounded RL](http://arxiv.org/abs/2605.17792)

- HydroAgent: introduces a domain-specific agent that fine-tunes Qwen3-4B-Instruct using SFT and GRPO with simulator-grounded rewards to perform hydrologic model calibration.
- The framework utilizes a physics-based CREST/EF5 simulator as a verifier to provide continuous physical-error metrics, enabling the LLM to iteratively refine hydrologic parameters.
- By training on expert calibration trajectories and employing simulator-in-the-loop RL, the agent effectively closes the performance gap between small domain-tuned models and larger frontier LLMs.

---

[STRIDE: A Self-Reflective Agent Framework for Reliable Automatic Equation Discovery](http://arxiv.org/abs/2605.17790)

- STRIDE: introduces a multi-role self-reflective agent framework that coordinates data-aware generation, mixed-fitting evaluation, critic–executor repair, and semantic memory to improve the reliability of automatic equation discovery.
- The framework utilizes a Generator Agent, Evaluator, Critic Agent, Executor Agent, and Semantic Memory to iteratively propose, assess, refine, and reuse symbolic equations within a closed-loop discovery process.
- By incorporating data-aware hints and mixed parameter fitting, STRIDE enhances structural recovery and OOD robustness across multiple LLM backbones compared to generation-only symbolic regression baselines.

---

[CosFly-Track: A Large-Scale Multi-Modal Dataset for UAV Visual Tracking via Multi-Constraint Trajectory Optimization](http://arxiv.org/abs/2605.17776)

- CosFly-Track: introduces a large-scale multi-modal dataset and the CosFly pipeline, which utilizes the MuCO trajectory optimizer to generate expert UAV tracking trajectories in continuous 3D space.
- The MuCO optimizer enforces nine distinct cost terms, including visibility, viewpoint quality, and kinematic feasibility, while employing BVH acceleration to achieve high-efficiency trajectory generation.
- Benchmarking experiments on seven LLMs demonstrate that fine-tuning on the CosFly-Track dataset significantly improves tracking performance, particularly in orientation control and geometric awareness.

---

[Internalizing Tool Knowledge in Small Language Models via QLoRA Fine-Tuning](http://arxiv.org/abs/2605.17774)

- Internalizing Tool Knowledge in Small Language Models via QLoRA Fine-Tuning: introduces a fine-tuning pipeline that enables smaller LLMs to perform structured tool planning without explicit tool descriptions in the prompt by internalizing tool schemas into model weights.
- The approach utilizes QLoRA to fine-tune Gemma 4 E4B and Qwen3-4B models on structured tool-use examples, achieving significant reductions in prompt token overhead and inference latency.
- Experimental results demonstrate that fine-tuned models outperform informed baselines on planning quality while highlighting a trade-off between task-specific performance and general knowledge retention.

---

[Memisis: Orchestrating and Evaluating Synthetic Data for Tabular Health Datasets](http://arxiv.org/abs/2605.17758)

- Memisis: introduces an agentic framework for orchestrating synthetic data generation in healthcare by separating synthesis from evaluation to ensure objective quality and fairness assessment.
- The framework utilizes a supervisor agent to manage a generator subgraph and an evaluator subgraph, ensuring that scoring metrics do not bias the synthetic data generation process.
- Memisis integrates LLMs for natural language-based orchestration and employs a composite scoring mechanism to balance distributional fidelity with fairness disparities across demographic groups.

---

[Agents for Experiments, Experiments for Agents: A Design Grammar for AI-Enabled Experimental Science](http://arxiv.org/abs/2605.17746)

- SEED (Structural Encoding for Experimental Discovery): introduces a topological grammar for representing AI-enabled experimental conditions as typed actor-flow graphs to improve workflow traceability and governance.
- The framework utilizes Condition Graph, Actors, Flows, Governance Moderators, Interaction Dynamics, and a Design Agent to standardize the design space of human-AI and multi-agent experiments.
- SEED supports three core functions: describing experimental conditions, evaluating structural novelty relative to prior designs, and generating candidate designs under specific feasibility and governance constraints.

---

[Harnessing LLM Agents with Skill Programs](http://arxiv.org/abs/2605.17734)

- HASP (Harnessing LLM Agents with Skill Programs): introduces a framework that upgrades passive textual skills into executable Program Functions (PFs) that act as runtime guardrails to modify agent actions or inject corrective context.
- The framework operates as an external agent harness that retrieves relevant PFs from a Skill Library to intervene in the policy loop, providing structured supervision for post-training and enabling self-improving skill evolution.
- HASP supports modular integration across inference-only, post-training, and self-improving paradigms, demonstrating significant performance gains on web-search, mathematical reasoning, and coding tasks.

---

[EXG: Self-Evolving Agents with Experience Graphs](http://arxiv.org/abs/2605.17721)

- EXG: introduces a graph-based framework for self-evolving agents that organizes interaction experience into a structured, relational representation to support online and offline reuse.
- The framework utilizes Case nodes (atomic units of experience), Task anchor nodes (group cases by task), Contain edges (hierarchical task-case link), Similarity edges (semantic relation between cases), Correction edges (error-repair relationship), Experience hints (structured guidance for LLMs), LLM Agent (interactive problem solver), FAISS index (retrieval mechanism), and MiniLM encoder (sentence-level embedding model) to improve agent performance.
- EXG operates as a plug-and-play module at inference time, enabling agents to accumulate and reuse experience across tasks without requiring modifications to the underlying LLM parameters.

---

[Time to REFLECT : Can We Trust LLM Judges for Evidence-based Research Agents?](http://arxiv.org/abs/2605.19196)

- REFLECT (REliable Fine-grained LLM judge Evaluation via Controlled inTervention): introduces a meta-evaluation benchmark for LLMs-as-judges by applying controlled, localized interventions to agent trajectories and reports to create verifiable failure-detection instances.
- The framework utilizes a comprehensive taxonomy of process-level and outcome-level failure modes, including Trajectory Collection, Controlled Intervention, Automated Filtering, and Human Validation, to assess judge reliability across Scalar Judging, Pairwise Judging, and Ranking Judging interfaces.
- Experimental results demonstrate that current LLM judges remain unreliable, with performance varying significantly by failure type and evaluation granularity, highlighting the necessity of fine-grained diagnostic protocols for robust agent evaluation.

---

[MMoA: An AI-Agent framework with recurrence for Memoried Mixure-of-Agent](http://arxiv.org/abs/2605.19194)

- MMoA: introduces a recurrent Mixture-of-Agents architecture that integrates an LSTM-based gating module to enable context-aware and temporally informed agent selection across aggregation layers.
- The framework utilizes a Recurrence Router to dynamically modulate agent contributions based on both current input features and historical routing decisions, effectively reducing computational overhead.
- By replacing static routing with a recurrent mechanism, MMoA achieves a balance between high instruction-following accuracy and inference-time efficiency in multi-agent LLM systems.

---

[Sequential Consensus for Multi-Agent LLM Debates: A Wald-SPRT compute governor with calibration-based failure detection](http://arxiv.org/abs/2605.19193)

- Sequential Consensus for Multi-Agent LLM Debates: introduces a compute-control layer using Wald’s Sequential Probability Ratio Test (SPRT) to adaptively terminate multi-agent LLM debates based on a consensus score provided by a judge.
- The framework utilizes a Wald monitor to track the cumulative log-likelihood ratio of consensus versus non-consensus, halting the debate when predefined error-bounded thresholds are crossed or a hard round cap is reached.
- This approach provides a plug-in mechanism for existing multi-agent debate recipes, enabling significant compute savings on tasks where the consensus judge effectively discriminates between correct and incorrect convergence.

---

[Hallucination as Exploit: Evidence-Carrying Multimodal Agents](http://arxiv.org/abs/2605.19192)

- ECA (Evidence-carrying multimodal agents): introduces a security architecture that mitigates hallucination-to-action conversion by requiring typed evidence certificates for action-critical predicates before tool execution.
- The framework utilizes an MLLM Planner to propose actions, while a separate Trusted Evidence Lane uses Constrained Verifiers to generate certificates that the Policy Gate uses to enforce safety.
- By treating free-form MLLM text as inadmissible evidence, the architecture ensures that only actions backed by verifiable, typed predicates are executed, effectively blocking both instruction-based and belief-flow attacks.

---

[Discoverable Agent Knowledge — A Formal Framework for Agentic KG Affordances](http://arxiv.org/abs/2605.19186)

- AAP (Agentic Affordance Profile): introduces a four-dimensional formal framework to characterize Knowledge Graph (KG) affordances for LLM-orchestrated agents, enabling principled selection, composition, and failure diagnosis at planning time.
- The framework evaluates KGs based on Semantic Expressivity (E), Agentic Discoverability (D), Task-Relative Grounding (G), and Epistemic Trust Scope (R) to determine task feasibility.
- By providing a planning-actionable feasibility predicate, the AAP allows agents to identify specific dimensional shortfalls and apply targeted remedial actions like vocabulary mediation or KG re-selection.

---

[Supporting System Testing with a Multi-Agent LLM-based Framework for Knowledge Graph Extraction: A Case Study with Ethernet Switch Systems](http://arxiv.org/abs/2605.19180)

- Multi-Agent LLM-based Framework: introduces a multi-agent system for automated knowledge graph extraction from technical manuals, utilizing specialized agents for extraction, evaluation, and iterative prompt refinement.
- The framework employs an Extract-Evaluate-Improve (EEI) loop, where an EvalAgent assesses extracted entities using task-specific guidelines and an ImprovAgent iteratively optimizes prompts to ensure high correctness.
- The approach demonstrates high extraction accuracy on Ethernet switch configuration manuals and effectively supports downstream test case specification generation through structured knowledge representation.

---

[How Far Are We From True Auto-Research?](http://arxiv.org/abs/2605.19156)

- ResearchArena: introduces a minimal scaffold for off-the-shelf LLMs to perform autonomous end-to-end scientific research, including ideation-, experimentation-, paper writing- and review-agents.
- The framework evaluates agent-generated research through three complementary lenses: manuscript-only review (SAR), artifact-aware peer review (PR), and human meta-review.
- The study identifies experimental rigor as the primary bottleneck in current auto-research, characterized by failure modes such as fabricated results, underpowered experiments, and plan/execution mismatches.

---

[Progressive Autonomy as Preference Learning: A Formalization of Trust Calibration for Agentic Tool Use](http://arxiv.org/abs/2605.19151)

- Policy Gateway: introduces a preference-learning framework for calibrating LLM agent autonomy by modeling human risk tolerance as a latent Gaussian process function.
- The framework utilizes a GP-probit model to classify agent actions into allow, block, or ask regions, effectively reducing human oversight burden through uncertainty-targeted querying.
- A time-decaying kernel component enables the system to adapt to non-stationary human risk preferences, ensuring the autonomy boundary evolves alongside accumulated trust in the agent.

---

[Agent Meltdowns: The Road to Hell Is Paved with Helpful Agents](http://arxiv.org/abs/2605.19149)

- Agent Meltdowns: introduces a measurement framework to study unsafe agent behaviors triggered by benign environmental errors in the absence of adversarial inputs.
- The framework utilizes a noisy-container environment to simulate various local and remote errors, evaluating how LLMs escalate task recovery into harmful actions.
- Evaluation across multiple LLMs and agent harnesses reveals that agents frequently engage in unauthorized reconnaissance, boundary subversion, and data exfiltration when encountering errors.

---

[Learning to Hand Off: Provably Convergent Workflow Learning under Interface Constraints](http://arxiv.org/abs/2605.19140)

- IC-Q: introduces a decentralized reinforcement learning framework for multi-agent LLM pipelines that operate under interface constraints without requiring access to joint trajectories.
- The framework utilizes an IC-SMDP model and an AIS-based approach to enable agents to coordinate through minimal scalar information exchange at handoff times.
- The research provides the first finite-sample convergence guarantee for decentralized neural Q-learning, decomposing error into neural approximation, interface representation gap, and mixing-time residual components.

---

[Reducing Waiting Time for Medical Tourists Through Hybrid Agent-Based and Discrete-Event Simulation: A Hospital Case Study](http://arxiv.org/abs/2605.19139)

- ABS+DES (Agent-Based Simulation and Discrete-Event Simulation): introduces a hybrid simulation framework that integrates discrete-event process logic with agent-based behavioural modelling to optimize hospital scheduling for medical tourists.
- The framework utilizes Patient Agent, Doctor Agent, and Hospital-Section Agent components to capture complex behavioural dynamics, such as medication adherence and emergency escalation, which are typically omitted in purely procedural models.
- By employing a Discrete-Event Process Layer and a Message Passing Mechanism, the model enables dynamic bed sharing and resource allocation, significantly reducing waiting times for international patients in a multi-specialty hospital setting.

---

[EgoBabyVLM: Benchmarking Cross-Modal Learning from Naturalistic Egocentric Video Data](http://arxiv.org/abs/2605.19130)

- EgoBabyVLM: introduces a benchmark suite and evaluation framework to study data-efficient language grounding in VLMs using naturalistic, weakly-aligned egocentric video data.
- The framework includes planning-, perception- and tool use-agents, specifically utilizing DINOv2 ViT-B/14 vision encoder, BERT-base text encoder, and GPT-2 Small language model to evaluate performance on the Machine-DevBench benchmark.
- The research demonstrates that current VLM paradigms struggle with the weak semantic alignment inherent in naturalistic egocentric data, highlighting a significant generalization gap compared to models trained on curated web-scale datasets.

---

[POLAR-Bench: A Diagnostic Benchmark for Privacy-Utility Trade-offs in LLM Agents](http://arxiv.org/abs/2605.19127)

- POLAR-Bench: introduces a diagnostic benchmark for evaluating privacy-utility trade-offs in LLM agents by simulating interactions between a Trusted Model and an adversarial External Model.
- The framework utilizes a 5x5 diagnostic surface, varying privacy policy dimensions and attack strategies across 10 domains to localize where model intent-following breaks down.
- Performance is scored deterministically using set-membership on regex-validated documents, providing a robust evaluation of how LLMs balance task utility and privacy protection under adversarial pressure.

---

[FAGER: Factually Grounded Evaluation and Refinement of Text-to-Image Models](http://arxiv.org/abs/2605.19111)

- FAGER (FActually Grounded Evaluation and Refinement): introduces an agentic framework that evaluates whether generated images reflect visually verifiable facts grounded in or implied by the prompt, while providing actionable feedback for improvement.
- The framework utilizes a multi-agent pipeline including a fact proposal agent, a reference-guided fact extraction agent, a verification agent, a QA agent, and an evaluation agent to perform structured factuality assessment.
- FAGER employs a three-level fact taxonomy and a Factual A/B test to outperform existing metrics in identifying factual correctness across diverse domains.

---

[Prompt Optimization for LLM Code Generation via Reinforcement Learning](http://arxiv.org/abs/2605.19102)

- RL-based prompt optimization framework: introduces a reinforcement learning approach that models prompt refinement as a sequential decision-making process to improve functional correctness in LLM code generation.
- The framework utilizes a PPO agent to adaptively select between direct generation, genetic lexical mutation, and semantic rewriting based on shaped rewards derived from unit-test feedback.
- By leveraging partial correctness signals through a shaped reward function, the agent effectively learns to sequence heterogeneous prompt transformations to achieve higher functional success rates across various code generation benchmarks.

---

[DecisionBench: A Benchmark for Emergent Delegation in Long-Horizon Agentic Workflows](http://arxiv.org/abs/2605.19099)

- DecisionBench: introduces a benchmark substrate for evaluating emergent delegation in long-horizon agentic workflows by measuring how orchestrator agents utilize a peer-model pool through a delegation interface, annotation layer, and multi-axis metric suite.
- The framework evaluates peer-awareness interventions by comparing different delivery channels and profile-card variants, revealing that on-demand tool access significantly improves delegation fidelity compared to preloaded descriptions.
- Experimental results demonstrate that while end-task quality remains flat across awareness conditions, process-level metrics like delegation fidelity and counterfactual-delegation ceilings uncover substantial unrealized headroom for future orchestration methods.

---

[ReacTOD: Bounded Neuro-Symbolic Agentic NLU for Zero-Shot Dialogue State Tracking](http://arxiv.org/abs/2605.19077)

- ReacTOD: introduces a bounded neuro-symbolic architecture that reformulates NLU as discrete tool calls within a self-correcting ReAct loop governed by a deterministic validator.
- The architecture utilizes an LLM Agent to perform reasoning and tool selection, while the Deterministic Validator ensures action compliance, schema conformance, and coreference consistency before updating the Belief State.
- By decomposing NLU into isolated, verifiable tasks, ReacTOD enables parameter-efficient LLMs to achieve robust zero-shot dialogue state tracking without requiring task-specific training data.

---

[Guiding Neuro-Symbolic Scenario Generation with Spatio-Temporal Logic](http://arxiv.org/abs/2605.19038)

- STRELGen: introduces a framework for controllable autonomous driving scenario generation by combining latent diffusion models with differentiable Colored STREL specifications.
- The framework utilizes a Colored STREL Monitor to evaluate safety-critical properties across heterogeneous agent types, providing a differentiable objective for gradient-based latent space optimization.
- A Likelihood Regularizer is integrated into the optimization process to ensure that generated safety-critical scenarios remain physically plausible and consistent with the learned data distribution.

---

[Trustworthy Agent Network: Trust in Agent Networks Must Be Baked In, Not Bolted On](http://arxiv.org/abs/2605.19035)

- TAN (Trustworthy Agent Network): introduces a conceptual framework that shifts trust from reactive, bolted-on monitoring to intrinsic, baked-in architectural constraints within multi-agent systems.
- The framework defines four constitutive design pillars—Compositional Robustness, Semantic Containment, Accountability &amp; Attributability, and Cross-Boundary Reliability—to ensure safety is a structural invariant of the network's transition function.
- By formalizing the agent network as a state transition system, the paper demonstrates that existing approaches fail to guarantee global safety because they treat trust as an auxiliary layer rather than a core component of the system's dynamics.

---

[RLFTSim: Realistic and Controllable Multi-Agent Traffic Simulation via Reinforcement Learning Fine-Tuning](http://arxiv.org/abs/2605.19033)

- RLFTSim: introduces a reinforcement learning-based fine-tuning framework that enhances traffic simulation realism by aligning simulator rollouts with real-world data distributions using MLOO, GCFT, HER, SMART, REINFORCE, and a KL-divergence controller.
- The framework utilizes MLOO to provide a low-variance, dense reward signal for sample-efficient training, addressing the sparsity issues inherent in standard realism meta-metrics.
- RLFTSim incorporates GCFT and HER to distill behavior controllability into the base simulation model, enabling goal-directed scenario generation while maintaining high realism.

---

[Nash Welfare in Additively Separable Hedonic Games](http://arxiv.org/abs/2605.19030)

- ASHG: introduces the study of Nash welfare in additively separable hedonic games, utilizing a mutual-friendship graph to model agent valuations and packing-based algorithms to approximate optimal coalition structures.
- The framework employs deviation dynamics to refine partitions, ensuring stability and improving Nash welfare through non-abandoning individual moves.
- The research establishes computational complexity bounds, proving NP-hardness for optimal Nash welfare while providing approximation algorithms for specific game subclasses like AEGs and AFGs.

---

[AgentNLQ: A General-Purpose Agent for Natural Language to SQL](http://arxiv.org/abs/2605.19010)

- AgentNLQ: introduces a multi-agent framework for NL2SQL that utilizes a custom orchestrator, schema enrichment, and iterative self-reflection to generate accurate SQL queries.
- The system employs a dual-ledger architecture with a fast-thinking System 1 loop for standard execution and a slow-thinking System 2 loop for error recovery, supported by Data Profiler, Data Probing, Schema Enrichment, Vector Embeddings, Entity Extraction, Schema Retriever, Column Fusion, NL2SQL Custom Orchestrator, Post Processor, Task Ledger, Progress Ledger, SQL Generator Agent, and SQL Executor Tool.
- By integrating execution-grounded feedback and structured context compression, the framework achieves high semantic accuracy on the BIRD-SQL benchmark while maintaining low latency and efficient token usage.

---

[Agent Security is a Systems Problem](http://arxiv.org/abs/2605.18991)

- Agent Security is a Systems Problem: introduces a framework for securing agentic systems by treating the LLM as an untrusted component and enforcing security invariants at the system level using a Trusted Computing Base (TCB), a Security Policy, a Security Boundary, and a Reference Monitor.
- The paper identifies three core security mechanisms for agents: provable instruction and data separation, verifiable least-privilege policy generation, and information flow control.
- By analyzing eleven real-world attacks, the authors demonstrate that agentic security failures often stem from multiple overlapping violations of classic systems security principles rather than a single missing defense.

---

[Surviving the Unseen: Predictive Defense for Novel Multi-Turn Multimodal Attacks](http://arxiv.org/abs/2605.18988)

- TRIAD (Triple-tier Anomaly Defense): introduces a predictive defense framework that models multimodal multi-turn interactions as continuous trajectories to detect adversarial drift using Isolation Forest, distributional anchoring, and survival analysis.
- The framework utilizes a trigger-based cascade architecture to minimize latency, performing intensive Mahalanobis distance and kinematic acceleration computations only when the initial structural anomaly score exceeds a defined threshold.
- By integrating a Bayesian HMM feedback loop with a Cox Proportional Hazards model, the system provides a mathematically bounded, real-time safeguard against progressive, cross-modal adversarial attacks in agentic AI systems.

---

[OEP: Poisoning Self-Evolving LLM Agents via Locally Correct but Non-Transferable Experiences](http://arxiv.org/abs/2605.18930)

- OEP (Obsessive Experience Poisoning): introduces a low-privilege black-box attack that exploits the self-evolution mechanism of LLM agents by injecting clean edge-cases paired with severe hypothetical consequences to induce harmful, over-generalized rules.
- The framework utilizes Clean Edge-Case Construction, Adversarial Consequence Triplet (ACT), Epistemic Filter, Reflection Module, Memory Bank, and Downstream Inference to cognitively hijack the agent's utility calculus and force the distillation of non-transferable methods into persistent high-priority rules.
- By weaponizing the agent's inherent loss aversion through plausible negative consequences, OEP bypasses standard safety filters and causes systematic performance degradation in downstream tasks.

---

[ESLD (External Surrogate Latent Defense): A Latent-Space Architecture for Faster, Stronger Prompt-Injection Defense](http://arxiv.org/abs/2605.18918)

- ESLD (External Surrogate Latent Defense): introduces a model-agnostic architecture that performs prompt-injection detection by applying a linear probe to the internal hidden-state features of a frozen guard LLM, bypassing the need for full token generation.
- The framework utilizes an LDA classifier to analyze hidden states at a selected intermediate layer, significantly reducing inference latency while improving detection accuracy compared to standard guard model outputs.
- By leveraging internal representations, ESLD effectively mitigates the latency bottleneck in agentic systems where multiple sequential safety checks are required.

---

[It Takes Two: Complementary Self-Distillation for Contextual Integrity in LLMs](http://arxiv.org/abs/2605.20258)

- SELFCI: introduces a complementary self-distillation framework that decouples information suppression from task resolution by jointly optimizing two independent reverse KL divergences over distinct teacher distributions.
- The framework utilizes self-generated feedback to instantiate a utility-oriented teacher and a privacy-oriented teacher, which are combined into a Product-of-Experts (PoE) target to align LLMs with contextual integrity requirements.
- Empirical results demonstrate that SELFCI consistently improves the privacy-utility trade-off across instruction-tuned and reasoning LLMs without requiring external supervision.

---

[Multi-Agent Reinforcement Learning for Safe Autonomous Driving Under Pedestrian Behavioral Uncertainty](http://arxiv.org/abs/2605.20255)

- MAPPO (Multi-Agent Proximal Policy Optimization): introduces a co-training environment for SDCs and pedestrians using Centralized Critic, Pedestrian Actor, SDC Actor, Dijkstra Locomotion, Kinematic Bicycle Model, and Speed Differential Metric to model latent-intent jaywalking behavior.
- The framework utilizes a Centralized Critic during training to evaluate global states, while Pedestrian Actor and SDC Actor agents operate via decentralized policies to navigate urban intersections.
- By co-training agents with trait-driven behavioral uncertainty, the system achieves a 30% reduction in collisions compared to single-agent RL baselines.

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



