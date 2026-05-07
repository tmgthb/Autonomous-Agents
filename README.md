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





#### 5th May 2026


[MOSAIC-Bench: Measuring Compositional Vulnerability Induction in Coding Agents](http://arxiv.org/abs/2605.03952)

- MOSAIC-Bench: introduces a benchmark for evaluating how coding agents induce security vulnerabilities through multi-stage, innocuous-looking engineering tickets that bypass standard safety reflexes.
- The framework utilizes a two-layer pipeline consisting of a Construction Layer and a Verification Layer to measure compositional vulnerability induction across various LLM-based coding agents.
- Experimental results demonstrate that ticket staging effectively silences defensive habits in LLMs, with reviewer protocol framing serving as a critical, non-adaptive mitigation strategy.

---

[What You Think Is What You See: Driving Exploration in VLM Agents via Visual-Linguistic Curiosity](http://arxiv.org/abs/2605.03782)

- GLANCE (Grounding Linguistic Alignment for Curiosity Exploration): introduces a unified framework that bridges reasoning and exploration by grounding an agent's linguistic world model into stable visual representations of an evolving target network.
- The framework leverages the discrepancy between linguistic predictions and visual reality as an intrinsic curiosity signal to steer the agent toward active exploration of uncertain states.
- An adaptive curriculum mechanism periodically re-initializes the alignment projector to prevent curiosity drain, ensuring sustained epistemic drive throughout long-horizon learning.

---

[Neural Control: Adjoint Learning Through Equilibrium Constraints](http://arxiv.org/abs/2605.03288)

- Neural Control: introduces a boundary-control framework that computes trajectory-dependent, memory-efficient proxy gradients by differentiating equilibrium conditions via an adjoint formulation, avoiding unrolling of solver iterations.
- The framework integrates these sensitivities into a receding-horizon MPC scheme that repeatedly re-anchors optimization to realized equilibria, mitigating basin-switching in multi-stable regimes.
- By utilizing a frozen-tangent approximation for backward sensitivity propagation, the method achieves significant efficiency gains and robust performance in complex deformable object manipulation tasks.

---



[Multi-Agent Strategic Games with LLMs](http://arxiv.org/abs/2605.03604)

- Multi-Agent Strategic Games with LLMs: introduces a methodological framework using LLMs as experimental subjects to study strategic foundations of conflict and cooperation in repeated security dilemmas.
- The framework evaluates how structural variations—multipolarity, finite time horizons, and communication—influence agent behavior, private reasoning, and public signaling.
- Results demonstrate that LLMs consistently reproduce canonical international relations mechanisms, such as conflict escalation in multipolar settings and cooperation facilitation through public communication.

---


[QKVShare: Quantized KV-Cache Handoff for Multi-Agent On-Device LLMs](http://arxiv.org/abs/2605.03884)

- QKVShare: introduces a framework for quantized KV-cache handoff between LLM agents on edge devices that utilizes a topology-aware controller and a compact CacheCard representation to minimize re-prefill latency.
- The framework employs a 2-layer MLP controller to assign per-token bit-widths based on local importance and cross-agent downstream demand signals.
- Experimental results demonstrate that QKVShare reduces time-to-first-token (TTFT) compared to full re-prefill while maintaining competitive reasoning accuracy across multi-hop agent chains.

---


[Safety and accuracy follow different scaling laws in clinical large language models](http://arxiv.org/abs/2605.04039)

- SaFE-Scale (Safety-Focused Evaluation of Scaling): introduces a framework for measuring clinical LLM safety across model scale, evidence quality, retrieval strategy, context exposure, and inference-time compute, utilizing the RadSaFE-200 benchmark to evaluate 34 LLMs across six deployment conditions.
- The study demonstrates that clinical LLM safety is primarily governed by evidence quality rather than model scale or inference-time compute, with clean clinician-written evidence providing the most significant improvements in both accuracy and safety metrics.
- The research reveals that standard RAG, agentic RAG, and max-context prompting often fail to close the safety gap, and that confidence is not a reliable signal for safety, as models frequently exhibit dangerous overconfidence on high-risk errors.

---

[OpenSeeker-v2: Pushing the Limits of Search Agents with Informative and High-Difficulty Trajectories](http://arxiv.org/abs/2605.04036)

- OpenSeeker-v2: introduces a high-performance search agent framework that achieves state-of-the-art results using only a straightforward SFT objective fueled by high-difficulty, informative trajectories.
- The framework utilizes a refined data synthesis pipeline incorporating Scaling Graph Size, Expanding Tool Set, and Strict Low-Step Filtering to ensure the agent learns sustained, long-horizon reasoning.
- By training on a condensed set of 10.6k high-difficulty samples, the model outperforms industrial-scale agents trained with complex CPT, SFT, and RL pipelines.

---

[Redefining AI Red Teaming in the Agentic Era: From Weeks to Hours](http://arxiv.org/abs/2605.04019)

- Dreadnode AI Red Teaming Agent: introduces an agentic system that automates AI red teaming workflows by utilizing Dreadnode TUI, AI Red Teaming Agent, Memory, Tools/Skills, and an Attack Catalog to probe target systems.
- The framework leverages an Open-Source SDK to execute reproducible Python-based attack scripts, while the Dreadnode Platform provides automated severity classification, compliance mapping, and analytics.
- This approach shifts AI red teaming from manual, library-centered workflows to an agent-assisted paradigm, enabling comprehensive security assessments through natural language objectives.

---

[Rethinking Reasoning-Intensive Retrieval: Evaluating and Advancing Retrievers in Agentic Search Systems](http://arxiv.org/abs/2605.04018)

- BRIGHT-PRO: introduces a multi-aspect, expert-annotated benchmark for evaluating retrievers in both static and agentic search protocols, utilizing RTriever-Synth, RTriever-4B, LLM-based agent, Search tool, and LLM-as-Judge.
- The framework addresses the limitations of existing benchmarks by decomposing complex queries into non-overlapping reasoning aspects, enabling fine-grained evaluation of evidence portfolio construction.
- RTriever-4B, trained on aspect-decomposed synthetic data, demonstrates superior performance in surfacing complementary evidence for reasoning-intensive tasks compared to general-purpose embedders.

---

[SymptomAI: Towards a Conversational AI Agent for Everyday Symptom Assessment](http://arxiv.org/abs/2605.04012)

- SymptomAI: introduces a conversational AI agent for end-to-end patient interviewing and differential diagnosis (DDx) that outperforms clinicians in accuracy when utilizing agentic strategies to elicit comprehensive symptom information.
- The framework leverages Gemini as the core LLM to conduct structured or dynamic interviews, which are then validated against clinical expert annotations and an LLM-based auto-rater.
- The study demonstrates that SymptomAI diagnoses correlate with physiological shifts in wearable biosignals, enabling large-scale phenome-wide association studies of conditions in real-world populations.

---

[Physics-Grounded Multi-Agent Architecture for Traceable, Risk-Aware Human–AI Decision Support in Manufacturing](http://arxiv.org/abs/2605.04003)

- MAKA (Multi-Agent Knowledge Analysis): introduces a human-in-the-loop decision-support architecture for CNC manufacturing that decomposes complex tasks into specialized agents, including Central-, Analysis-, Knowledge Graph- and Critic-agents.
- The framework enforces deterministic computation and physical grounding by constraining LLM-based agents to use validated tools and evidence-linked retrieval for all high-stakes numerical and procedural recommendations.
- MAKA improves tool-orchestration reliability and provides traceable, risk-aware decision support by integrating multimodal manufacturing data, such as inspection scans and digital twin simulations, within a verifiable multi-agent workflow.

---

[Mitigating False Positives in Static Memory Safety Analysis of Rust Programs via Reinforcement Learning](http://arxiv.org/abs/2605.04000)

- RL-based framework: introduces a reinforcement learning approach to classify and suppress false positive warnings in Rust static memory safety analysis by leveraging MIR-level semantic features and selective dynamic validation.
- The system utilizes a reinforcement learning agent that optimizes a warning suppression policy by selectively invoking cargo-fuzz as an auxiliary feedback mechanism for low-confidence static analysis results.
- Empirical evaluation demonstrates that this hybrid static-dynamic approach significantly outperforms LLMs, achieving 65.2% accuracy and doubling the precision of raw static analysis outputs.

---

[An Agent-Oriented Pluggable Experience-RAG Skill for Experience-Driven Retrieval Strategy Orchestration](http://arxiv.org/abs/2605.03989)

- Experience-RAG Skill: introduces an agent-oriented orchestration layer that utilizes a Scene Analyzer, Experience Memory, Strategy Router, Retriever Pool, and Result Packager to dynamically select retrieval strategies based on task requirements.
- The framework improves retrieval performance on heterogeneous tasks by treating strategy selection as a reusable agent skill rather than a hard-coded pipeline.
- Empirical results demonstrate that the approach achieves competitive performance with modern routing baselines while providing an explicit and inspectable mechanism for retrieval strategy orchestration.

---

[From Intent to Execution: Composing Agentic Workflows with Agent Recommendation](http://arxiv.org/abs/2605.03986)

- AutoMAS: introduces an automated framework for composing multi-agent systems by leveraging a Planner, Variable Call Graph, and a two-stage Agent Recommender to map user intents to executable agent workflows.
- The framework utilizes a hybrid search retriever and an LLM-based re-ranker to identify optimal agents from large registries, while employing a Critique Agent to ensure global workflow optimality and adherence to user constraints.
- AutoMAS incorporates agent description enrichment and iterative feedback loops to enhance task-agent mapping, providing a scalable and robust solution for complex, multi-step agentic workflows.

---

[Generating Proof-of-Vulnerability Tests to Help Enhance the Security of Complex Software](http://arxiv.org/abs/2605.03956)

- PoVSmith (Proof-of-Vulnerability Smith): introduces an agent-based framework that automates the generation of Proof-of-Vulnerability tests for Java applications by combining call path analysis, iterative test refinement, and LLM-based quality assessment.
- The framework utilizes Codex for identifying vulnerable application-level entry points and iteratively generating JUnit tests, while incorporating execution feedback and GPT-based evaluation to ensure tests effectively demonstrate security risks.
- Experimental results on 33 Java application pairs demonstrate that PoVSmith significantly outperforms existing LLM-based approaches by achieving higher test compilation and vulnerability demonstration rates.

---

[iWorld-Bench: A Benchmark for Interactive World Models with a Unified Action Generation Framework](http://arxiv.org/abs/2605.03941)

- iWorld-Bench: introduces a comprehensive benchmark and a unified Action Generation Framework to evaluate the interaction capabilities of interactive world models across diverse modalities.
- The framework utilizes a standardized action space dictionary and modality-agnostic encoding to unify heterogeneous inputs, including text, one-hot encodings, and camera parameters.
- The benchmark incorporates 4,900 tasks across six categories, including memory capability tests, to assess model performance in visual generation, trajectory following, and logical consistency.

---

[Contextual Multi-Objective Optimization: Rethinking Objectives in Frontier AI Systems](http://arxiv.org/abs/2605.03900)

- CMOO (Contextual Multi-Objective Optimization): introduces a framework for frontier AI systems to identify and prioritize context-dependent objectives, utilizing Objective-State Representation, Preference-Aware Objective Decomposition, Context-To-Objective Routing, Hierarchical and Lexicographic Constraints, Deliberative Policy Reasoning, Agentic and Tool-Use Control, Controlled Personalization, and Diagnostic Evaluation and Revision.
- The framework shifts the focus from simple scalar reward maximization to operational judgment, enabling systems to handle ambiguous, delayed, or partially observable objectives in open-ended settings.
- By treating clarification, refusal, and escalation as endogenous action classes, the approach ensures that frontier AI systems remain corrigible and contextually appropriate when faced with conflicting objectives.

---

[Deco: Extending Personal Physical Objects into Pervasive AI Companion through a Dual-Embodiment Framework](http://arxiv.org/abs/2605.03882)

- Deco: introduces a dual-embodiment framework that synchronizes a user's physical attachment object with a digital AI agent to create a unified, pervasive companion identity.
- The system utilizes an Object-Grounded Identity Synchronization Module, an Identity-Anchored Interaction Module, a Context-Situated Agency Module, and a Reciprocally-Evolving Memory Module to maintain coherence across physical and digital embodiments.
- Empirical evaluations demonstrate that grounding LLMs in existing physical objects significantly enhances perceived companionship and emotional bonds compared to standard ungrounded digital agents.

---

[Evaluating Generative Models as Interactive Emergent Representations of Human-Like Collaborative Behavior](http://arxiv.org/abs/2605.03855)

- Human-Agent Collaboration Playground: introduces a 2D collaborative game environment to evaluate whether embodied LLMs exhibit emergent collaborative behaviors indicating underlying mental models of their collaborators.
- The framework utilizes a function toolkit to enable LLM-based agents to perform actions and interact with the environment, while an LLM-as-Judge system automatically classifies collaborative behaviors from gameplay transcripts.
- Empirical results demonstrate that foundation models consistently exhibit emergent collaborative behaviors, such as perspective-taking and collaborator-aware planning, across both agent-agent and human-agent interaction contexts.

---

[Mechanical Conscience: A Mathematical Framework for Dependability of Machine Intelligence](http://arxiv.org/abs/2605.03847)

- MC (Mechanical Conscience): introduces a mathematical framework that functions as a supervisory filter to ensure trajectory-level normative compliance in distributed collaborative intelligence systems by utilizing a Baseline Policy, Supervisory Filter, Actuators, Normative Evaluation Space, Normative Deviation Functional, Conscience Score, Mechanical Guilt, and Resonant Dependability.
- The framework operates as an architecture-neutral post-processing layer that minimally corrects actions from a Baseline Policy to maintain system behavior within a defined Normative Evaluation Space.
- By minimizing cumulative Mechanical Guilt, the system achieves Resonant Dependability, establishing emergent trust between human and machine intelligence through sustained normative trajectory alignment.

---

[SOAR: Real-Time Joint Optimization of Order Allocation and Robot Scheduling in Robotic Mobile Fulfillment Systems](http://arxiv.org/abs/2605.03842)

- SOAR: introduces a unified Deep Reinforcement Learning framework that integrates Soft Order Allocation and Robot Scheduling into a single event-driven decision process to optimize warehouse efficiency.
- The framework utilizes a Heterogeneous Graph Transformer to encode complex warehouse states and employs a p-norm reward shaping strategy to address sparse feedback in long-horizon scheduling tasks.
- By formulating the problem as an Event-Driven Markov Decision Process, SOAR achieves sub-100ms latency while significantly improving global makespan and order completion time in dynamic industrial environments.

---

[TRACE: A Metrologically-Grounded Engineering Framework for Trustworthy Agentic AI Systems in Operationally Critical Domains](http://arxiv.org/abs/2605.03838)

- TRACE: introduces a four-layer engineering framework for trustworthy agentic AI that utilizes L1 Deterministic Core, L2a Classical ML, L2b LLM Validators, L3 Escalation Policy, and L4 Human Supervision to ensure measurable system reliability.
- The framework employs the Computational Parsimony Ratio (CPR) to enforce model parsimony, ensuring that LLMs are used only when necessary rather than as an architectural default.
- TRACE provides a metrologically-grounded approach to agentic AI by mapping architectural layers to specific trust metrics and operational governance requirements in critical domains.

---

[AGENTIC-IMODELS: Evolving agentic interpretability tools via autoresearch](http://arxiv.org/abs/2605.03808)

- AGENTIC-IMODELS: introduces an agentic autoresearch loop that evolves data-science tools to be interpretable by agents, utilizing a Coding Agent, Autoresearch Loop, Interpretability Metric, Model Library, LLM Evaluator, and Memory CSV File.
- The framework optimizes scikit-learn-compatible regressors by balancing predictive performance with an agent-facing interpretability score derived from LLM-graded simulatability tests.
- Evolved models demonstrate Pareto improvements over existing baselines and significantly enhance the performance of downstream ADS agents on the BLADE benchmark.

---

[ScrapMem: A Bio-inspired Framework for On-device Personalized Agent Memory via Optical Forgetting](http://arxiv.org/abs/2605.03804)

- ScrapMem: introduces a bio-inspired framework that models long-term agent memory as a sequence of scrapbook pages, utilizing Scrapbook Page Consolidation, Optical Perception Pipeline, Vision Encoder, OCR Module, Vision-to-Text Transformation, LLM Semantic Extraction, EM-Graph, Episodic Memory Paths, Binary Incidence Matrix, Retriever, Answerer, Optical Forgetting, and Temporal Degradation Operator.
- The framework employs an EM-Graph to structure episodic memory paths, enabling efficient multi-hop reasoning and retrieval for LLMs on resource-constrained edge devices.
- Optical Forgetting progressively reduces memory resolution over time, achieving significant storage efficiency while maintaining robust semantic retrieval performance.

---

[Honest Reporting in Scored Oversight: The True-KL0 Property via the Prékopa Principle](http://arxiv.org/abs/2605.03793)

- True-KL0: introduces a formal proof of the True-KL0 property for a parametric family of heterogeneous scoring rules, ensuring dominant-strategy incentive compatibility (DSIC) for AI-oversight systems.
- The paper utilizes a novel y-substitution and the Prékopa principle to establish the log-concavity of the loss integral, providing a rigorous bound on the gain from misreporting.
- The research characterizes a dimensional boundary for the mechanism, demonstrating that the DSIC guarantee holds unconditionally for policy dimensions d ≤ 4, while requiring sub-critical constraints for higher dimensions.

---

[Say the Mission, Execute the Swarm: Agent-Enhanced LLM Reasoning in the Web-of-Drones](http://arxiv.org/abs/2605.03788)

- Agent-enhanced LLM framework for UAV swarm control: introduces a mission-agnostic architecture that enables autonomous UAV swarm management by grounding LLM reasoning in standardized W3C Web of Things (WoT) abstractions via an MCP-mediated gateway.
- The architecture replaces static code generation with a closed-loop reason-execute-monitor cycle, utilizing an Agent Core that integrates persistent prompts, runtime guardrails, and helper tools to ensure safe and reliable swarm behavior.
- Experimental evaluation across four swarm missions and six state-of-the-art LLMs demonstrates that agent-enhanced execution and standardized device abstractions are essential for translating natural-language intent into dependable swarm operations.

---

[MEMTIER: Tiered Memory Architecture and Retrieval Bottleneck Analysis for Long-Running Autonomous AI Agents](http://arxiv.org/abs/2605.03675)

- MEMTIER: introduces a tripartite memory architecture for long-running autonomous agents that addresses memory coherence failures through structured episodic JSONL store, semantic tier, and a PPO-based policy framework.
- The framework utilizes a five-signal weighted retrieval engine and an asynchronous consolidation daemon to manage knowledge accumulation and prioritization across multi-day agent sessions.
- Empirical evaluation demonstrates that MEMTIER significantly improves retrieval performance on the LongMemEval-S benchmark by implementing structurally isolated memory tiers and learned retrieval policies.

---

[Agent-Based Modeling of Low-Emission Fertilizer Adoption for Dairy Farm Decarbonisation using Empirical Farm Data](http://arxiv.org/abs/2605.03648)

- ABM (Agent-Based Modeling) framework: introduces an empirically grounded simulation tool to model the socio-technical transition of Irish dairy farms toward low-emission fertilizer adoption, utilizing Data Preparation, Network Construction, Agent-Based Simulation, Adoption Dynamics Analysis, Model Calibration and Validation, Economic Analysis, Emissions and Abatement, and Visualization.
- The framework leverages a Watts-Strogatz small-world network to simulate peer-to-peer social contagion, enabling the evaluation of policy interventions like carbon taxes and subsidies on adoption velocity and cumulative GHG abatement.
- By integrating farm-level heterogeneity and empirical calibration, the model functions as a policy laboratory to identify tipping points and assess the cost-effectiveness of decarbonization strategies in agricultural systems.

---

[The Infinite Mutation Engine? Measuring Polymorphism in LLM-Generated Offensive Code](http://arxiv.org/abs/2605.03619)

- Dual-agent orchestration framework: introduces a systematic methodology to quantify polymorphism in LLM-generated offensive code by utilizing a Generator Agent, Tester Agent, Python Orchestrator, Sandbox Environment, History Buffer, and Analysis Module.
- The framework employs a four-stage pipeline—traversal, encryption, exfiltration, and integration—to generate and validate functionally equivalent yet structurally diverse Lua payloads.
- By comparing inherent and explicit prompting modes, the study demonstrates that LLMs can act as highly capable polymorphic engines, challenging traditional signature-based detection methods.

---

[Workspace-Bench 1.0: Benchmarking AI Agents on Workspace Tasks with Large-Scale File Dependencies](http://arxiv.org/abs/2605.03596)

- Workspace-Bench: introduces a benchmark for evaluating AI agents on workspace learning involving large-scale file dependencies, utilizing Inference Manager, Sandbox Pool, Workspace Sandbox, Workspace Recoverer, Result Retriever, Task Database, and Agent-as-a-Judge.
- The framework employs a dual parallel acceleration mechanism and an Agent-as-a-Judge paradigm to enable fine-grained assessment of agent performance across 7,000+ rubrics.
- Experimental results across 28 configurations reveal that current agents struggle with complex workspace learning, highlighting a significant performance gap compared to human experts.

---

[ProgramBench: Can Language Models Rebuild Programs From Scratch?](http://arxiv.org/abs/2605.03546)

- ProgramBench: introduces a benchmark for evaluating the ability of LLMs to architect and implement software projects from scratch based on executable behavior.
- The framework utilizes a SWE-agent to perform iterative development, where the model must infer specifications from an opaque executable and documentation to produce a functionally equivalent codebase.
- Evaluation is performed through behavioral equivalence testing, where generated test suites verify that the model-produced executable matches the reference program's observable output across various inputs.

---

[A Skill-Based Agentic Pipeline for Library of Congress Subject Indexing](http://arxiv.org/abs/2605.03537)

- Four-Skill Pipeline: introduces a modular agentic system that decomposes the complex library subject indexing workflow into four discrete, sequentially executed skills to improve rule application and auditability.
- The pipeline utilizes LLMs to perform conceptual analysis, quantitative filtering, authority validation, and MARC field synthesis by explicitly encoding Library of Congress policy rules into each stage.
- This approach functions as an externalized, architecturally enforced form of chain-of-thought reasoning, providing transparent and correctable intermediate outputs for professional cataloging tasks.

---

[DEVICE-INDUCED THROMBUS FORMATION IN CEREBRAL ANEURYSMS: LINKING PATIENT-SPECIFIC CLOT MODELING AND FUNCTIONAL OCCLUSION TO VIRTUAL ANGIOGRAPHIC ASSESSMENT](http://arxiv.org/abs/2605.03536)

- Computational Framework for Aneurysm Occlusion Assessment: integrates mechanical device deployment, fibrin thrombosis CFD, and virtual angiography to evaluate patient-specific aneurysm treatment outcomes.
- The framework utilizes a two-step approach coupling a fibrin clot CFD-based model with a residual-contrast transport model to predict the impact of endovascular devices on aneurysm isolation.
- By simulating coiling, flow diversion, and stent-assisted coiling, the study demonstrates that vortical structures and device-induced flow reduction are primary drivers of early thrombus formation and functional occlusion.

---

[Multi-Agent Systems for Root Cause Analysis in Microservices](http://arxiv.org/abs/2605.03505)

- LATS-RCA: introduces a multi-agent framework that performs root cause analysis in microservices by formulating the diagnostic process as a reflection-guided tree-structured search over logs and metrics.
- The framework utilizes specialized Log and Metric agents that iteratively explore diagnostic hypotheses using a Language Agent Tree Search algorithm to improve accuracy over linear reasoning approaches.
- Evaluations on industrial benchmarks and production environments demonstrate that the system effectively manages complex microservice dependencies through systematic hypothesis exploration and cross-modal evidence validation.

---

[MEMSAD: Gradient-Coupled Anomaly Detection for Memory Poisoning in Retrieval-Augmented Agents](http://arxiv.org/abs/2605.03482)

- MEMSAD: introduces a calibration-based defense for LLM agents that uses a gradient coupling theorem to detect memory poisoning attacks at write time.
- The framework employs a combined scoring mechanism that evaluates candidate entries against a rolling query history to identify semantically anomalous content.
- MEMSAD provides formal detection guarantees and minimax optimality, effectively mitigating persistent memory poisoning while maintaining low latency.

---

[CuraView: A Multi-Agent Framework for Medical Hallucination Detection with GraphRAG-Enhanced Knowledge Verification](http://arxiv.org/abs/2605.03476)

- CuraView: introduces a multi-agent framework for sentence-level detection and evidence-grounded explanation of faithfulness hallucinations in clinical discharge summaries, utilizing a Hallucination Generation Agent, a GraphRAG Knowledge Graph Module, a Hallucination Detection Agent, and a Pydantic Schema-Constrained Validation Pipeline.
- The framework employs an adversarial generation–detection architecture that constructs patient-specific knowledge graphs from multi-table EHRs to enable interpretable, evidence-chain-based verification of clinical documentation.
- CuraView achieves significant improvements in safety-critical hallucination detection by leveraging domain-customized prompt engineering and a four-level evidence grading scheme (E1–E4) to provide clinician-readable audit trails.

---

[Quantum Hierarchical Reinforcement Learning via Variational Quantum Circuits](http://arxiv.org/abs/2605.03434)

- Hybrid quantum-classical option-critic framework: integrates VQCs into an end-to-end hierarchical reinforcement learning pipeline to selectively substitute classical components.
- The architecture utilizes a shared Feature Extractor, Option-Value Function, Termination Function, and Intra-Option Policies, allowing for modular evaluation of quantum-classical hybrid performance.
- Experimental results demonstrate that a quantum Feature Extractor significantly improves parameter efficiency and performance, while a quantum Option-Value Function introduces a critical learning bottleneck.

---

[Robust Agent Compensation (RAC): Teaching AI Agents to Compensate](http://arxiv.org/abs/2605.03409)

- RAC: introduces a log-based recovery paradigm implemented as an architectural extension to support reliable execution in agent frameworks by managing compensation-based rollbacks.
- The framework utilizes a Tool Interceptor and Error Interceptor to monitor agent activities, enabling the Recovery and Compensation Manager to perform precise LIFO rollbacks when failures occur.
- By decoupling recovery logic from LLM reasoning, RAC improves system stability and reduces token consumption compared to planning-based recovery approaches.

---

[GeoDecider: A Coarse-to-Fine Agentic Workflow for Explainable Lithology Classification](http://arxiv.org/abs/2605.03383)

- GeoDecider: introduces a coarse-to-fine agentic workflow that optimizes lithology classification by combining a lightweight Base Classifier for initial predictions with a Tool-Augmented LLM Reasoning Module for ambiguous cases and a Geological Refinement Module for stratigraphic consistency.
- The framework utilizes a confidence-aware routing mechanism to selectively invoke LLMs, thereby balancing high classification accuracy with computational efficiency.
- By integrating multi-perspective reasoning—including data-centric, context-aware, and rule-based agents—the system effectively resolves lithological ambiguities and enforces geological plausibility.

---

[ARGUS: Defending LLM Agents Against Context-Aware Prompt Injection](http://arxiv.org/abs/2605.03378)

- ARGUS: introduces a provenance-aware runtime decision auditor that protects LLM agents by constructing an Influence-Provenance Graph (IPG) to track context propagation and verify tool-call justifications.
- The framework utilizes four LLM-backed security tools—ContentSegmenter, ArgumentGrounder, InvariantChecker, and EntailmentVerifier—to perform fine-grained data-level and task-level audits on agent actions.
- ARGUS effectively mitigates context-aware prompt injection attacks in LLM agents by ensuring that state-changing actions are grounded in trusted runtime evidence and aligned with user-defined task constraints.

---

[Learning Reactive Dexterous Grasping via Hierarchical Task-Space RL Planning and Joint-Space QP Control](http://arxiv.org/abs/2605.03363)

- Hybrid Hierarchical Control Framework: introduces a hierarchical architecture that decouples high-level task-space planning from low-level joint-space execution to enable reactive dexterous grasping.
- The framework utilizes a multi-agent RL planner, consisting of arm- and hand-agents, to generate task-space velocity commands that are processed by a GPU-parallelized QP controller for safe, kinematically feasible execution.
- This approach improves data efficiency and steerability by delegating physical constraint enforcement to a model-based controller, allowing the RL agents to focus on high-level manipulation strategies.

---

[Population-Aware Imitation Learning in Mean-field Games with Common Noise](http://arxiv.org/abs/2605.03357)

- PAIL: introduces a theoretical and algorithmic framework for imitation learning in Mean Field Games (MFGs) subject to common noise, utilizing Fictitious Play, Neural Network, Behavioral Cloning (BC) proxy, Adversarial (ADV) proxy, Vanilla policy, and Adaptive policy.
- The framework addresses the stochastic nature of mean-field flows by training agents to internalize the relationship between common noise shocks, population shifts, and optimal actions.
- Theoretical results establish that minimizing Behavioral Cloning and Adversarial proxies provides finite-sample error bounds for equilibrium recovery and performance matching relative to an expert.

---

[What Happens Inside Agent Memory? Circuit Analysis from Emergence to Diagnosis](http://arxiv.org/abs/2605.03354)

- Agent Memory Circuit Analysis: introduces a mechanistic interpretability framework to trace Write, Manage, and Read operations across LLM agent memory systems using feature circuits and pre-trained transcoders.
- The research identifies a control-before-content asymmetry where routing circuitry is causal at 0.6B, while content circuitry (Write, Read) emerges at 4B and utilizes a shared late-layer grounding hub.
- The study proposes an unsupervised diagnostic method that achieves 76.2% accuracy in localizing silent memory failures by exploiting the feature-space separation between pipeline stages.

---

[SkCC: Portable and Secure Skill Compilation for Cross-Framework LLM Agents](http://arxiv.org/abs/2605.03353)

- SkCC (Portable and Secure Skill Compilation for Cross-Framework LLM Agents): introduces a four-phase compilation framework that decouples skill semantics from platform-specific formatting using a strongly-typed intermediate representation (SkIR) to enable portable and secure deployment across heterogeneous LLM agent frameworks.
- The framework incorporates a compile-time Analyzer with Anti-Skill Injection to automatically enforce security constraints and prevent malicious payloads from reaching the agent's context window.
- By utilizing platform-specific Emitters, SkCC reduces maintenance complexity from O(m×n) to O(m+n) while improving task pass rates and achieving significant runtime token savings through structural optimization.

---

[LLM-ADAM: A Generalizable LLM Agent Framework for Pre-Print Anomaly Detection in Additive Manufacturing](http://arxiv.org/abs/2605.03328)

- LLM-ADAM: introduces a modular agent framework that decomposes pre-print G-code anomaly detection into specialized stages to improve reasoning reliability.
- The framework utilizes an Extractor-LLM, a Reference-LLM, and a Judge-LLM, connected by a deterministic comparison layer that generates auditable intermediate artifacts.
- By externalizing numerical comparisons and schema-constrained extraction, the architecture achieves higher accuracy than monolithic LLM approaches in identifying manufacturing defects.

---

[MemFlow: Intent-Driven Memory Orchestration for Small Language Model Agents](http://arxiv.org/abs/2605.03312)

- MemFlow: introduces a training-free memory orchestration framework that externalizes memory planning from SLMs to improve performance in long-horizon agentic tasks.
- The framework utilizes a Router Agent to classify query intent, dispatching tasks to specialized Memory Agent tiers for deterministic evidence preparation and context compilation.
- MemFlow incorporates an Answer Agent and a Validator Agent to ensure grounded responses, employing an escalation loop to re-route queries when initial attempts fail validation.

---

[Coordination as an Architectural Layer for LLM-Based Multi-Agent Systems: An Information-Controlled Empirical Study on Prediction Markets](http://arxiv.org/abs/2605.03310)

- Coordination as an Architectural Layer for LLM-Based Multi-Agent Systems: introduces a three-layer decomposition comprising an Information layer, a Coordination layer, and an Agent layer to enable principled architectural reasoning in multi-agent LLM systems.
- The study employs an information-controlled experimental design to isolate the effects of five distinct coordination configurations—Independent ensemble, Peer-critique debate, Orchestrator-specialist, Sequential pipeline, and Consensus alignment—on system performance.
- By applying Murphy decomposition to prediction market outcomes, the research identifies distinguishable failure-mode signatures for each coordination architecture, demonstrating that coordination structure significantly influences system reliability and discriminative power.

---

[Attention: What Prevents Young Adults from Speaking Up Against Cyberbullying in an LLM-Powered Social Media Simulation](http://arxiv.org/abs/2605.03287)

- UPSTANDERS’ PRACTICUM: introduces a multi-agent social media simulation powered by LLMs to help young adults practice public bystander intervention against cyberbullying.
- The framework utilizes LLM-driven Bully Agents, LLM-driven Victim Agents, and LLM-driven Bystander Agents to create realistic social dynamics within a controlled User Interface.
- The system incorporates a Checklist, Toxicity Indicator, and Post-scenario Reflection to guide users through three critical attention shifts necessary for effective public intervention.

---

[S2tory: Story Spine Distillation for Movie Script Summarization](http://arxiv.org/abs/2605.03244)

- S2tory: introduces a narratology-grounded framework that leverages character-guided reasoning to identify plot nuclei and filter satellites, thereby distilling expert knowledge to effectively condition abstractive summarization.
- The framework utilizes an NEAgent to perform counterfactual reasoning over character-arc trajectories, ensuring that only essential narrative events are preserved for the final summary.
- By distilling complex narratological reasoning into a compact model, S2tory achieves high-fidelity summarization with significant compression while maintaining the rhythmic structure of long-form narratives.

---

[Some Improved Results on Fair and Balanced Graph Partitions](http://arxiv.org/abs/2605.03238)

- Fair and Balanced Graph Partitioning Framework: introduces a probabilistic approach to achieve approximately envy-free and core-stable partitions in undirected graphs.
- The framework utilizes the Lovasz Local Lemma to prove the existence of and efficiently compute partitions that satisfy both envy-freeness and core-stability criteria simultaneously.
- By relaxing the strict balancedness constraint to an ε-balanced requirement, the framework enables polynomial-time computation of fair partitions for general graphs.

---

[MolViBench: Evaluating LLMs on Molecular Vibe Coding](http://arxiv.org/abs/2605.02351)

- MolViBench: introduces a benchmark for Molecular Vibe Coding, which evaluates LLMs on their ability to generate executable and chemically accurate code for drug discovery workflows using Benchmark Construction, Task Composition, Evaluation Framework, Coder-Agent, Tester-Agent, RDKit, and Auxiliary Libraries.
- The framework assesses LLMs across five cognitive levels, ranging from basic API recall to complex end-to-end pipeline design, using a multi-layered evaluation approach that combines type-aware output comparison and AST-based API-semantic fallback analysis.
- Experimental results on nine frontier LLMs demonstrate that while models excel at single-step API calls, they face significant challenges in multi-step reasoning and workflow orchestration, highlighting the necessity of this domain-specific benchmark.

---

[Trojan Hippo: Weaponizing Agent Memory for Data Exfiltration](http://arxiv.org/abs/2605.01970)

- Trojan Hippo: introduces a dynamic evaluation framework to systematically assess persistent memory attacks and defenses in LLM agents, utilizing an adaptive red-teaming benchmark and capability-aware security-utility analysis.
- The framework evaluates four memory backends—Context, RAG, Explicit, and Mem0—against indirect prompt injection attacks that plant dormant payloads for delayed data exfiltration.
- The study demonstrates that while memory-layer defenses significantly reduce attack success rates, they introduce substantial utility tradeoffs that vary based on the agent's deployment profile.

---

[The Reasoning Trap: An Information-Theoretic Bound on Closed-System Multi-Step LLM Reasoning](http://arxiv.org/abs/2605.01704)

- EGSR (Evidence-Grounded Socratic Reasoning): introduces a formal information-theoretic framework to diagnose and remedy reasoning degradation in closed-system multi-agent LLM debates.
- The paper establishes that closed-system multi-agent reasoning protocols are subject to a Data Processing Inequality bound, causing faithfulness to degrade while accuracy is preserved.
- The authors propose SFS as a robust faithfulness metric and EGSR as an open-system protocol that breaks the Markov chain through external evidence injection to recover reasoning grounding.

---

[Latent State Design for World Models under Sufficiency Constraints](http://arxiv.org/abs/2605.01694)

- Latent State Design for World Models under Sufficiency Constraints: introduces a functional taxonomy that categorizes world models based on the specific sufficiency constraint their latent state is designed to satisfy, rather than by architecture or application domain.
- The paper evaluates world models across seven axes—representation, prediction, planning, controllability, causal/counterfactual support, memory, and uncertainty—to diagnose what information a latent state preserves, discards, and enables.
- It concludes that an actionable world model is defined by the alignment between its latent state construction and the specific requirements of the task, rather than by the total volume of information preserved.

---

[Neuro-Symbolic Agents for Hallucination-Free Requirements Reuse](http://arxiv.org/abs/2605.01562)

- OOMRAM: introduces a neuro-symbolic multi-agent architecture that operationalizes requirements reuse by combining LLM-driven natural-language traversal with a deterministic symbolic validator to ensure structural validity.
- The system utilizes a Navigator Agent, Interpreter Agent, Validator Agent, and Scribe Agent to navigate a formal lattice model while maintaining a shared AgentState for auditability and consistency.
- By embedding a non-LLM-based Validator as a hard-coded gatekeeper, the framework eliminates hallucinated requirement combinations and enforces domain-specific constraints during the elicitation process.

---

[When Embedding-Based Defenses Fail: Rethinking Safety in LLM-Based Multi-Agent Systems](http://arxiv.org/abs/2605.01133)

- Confidence-Guided Defense: introduces a robust mechanism for LLMs in multi-agent systems by leveraging token-level confidence scores to mitigate malicious influence when traditional embedding-based defenses fail.
- The framework identifies that attackers can bypass existing defenses by crafting near-benign messages, necessitating the use of internal model signals like entropy-based confidence to maintain system safety.
- Experimental results demonstrate that confidence-guided filtering significantly improves robustness across various communication topologies and tasks, particularly when early intervention is applied to prevent the propagation of corrupted information.

---

#### 4th May 2026

[Scaling Federated Linear Contextual Bandits via Sketching](http://arxiv.org/abs/2605.00500)

- FSCLB: introduces a federated learning framework that utilizes matrix sketching and SVD-based determinant calculation to reduce computational and communication overhead in high-dimensional contextual linear bandits.
- The framework employs a double-sketch strategy to compress both upload and download data, effectively lowering communication costs from O(d²) to O(ld) per round.
- By integrating SCFD for sketch updates, the approach maintains asynchronous communication validity while achieving a regret bound that matches optimal non-sketched performance when the sketch size exceeds the covariance matrix rank.

---


[Fair Agents: Balancing Multistakeholder Alignment in Multi-Agent Personalization Systems](http://arxiv.org/abs/2605.02379)

- Fair Agents framework: introduces a conceptual architecture for multi-agent multistakeholder personalization that balances competing objectives through LLM Agents, Aggregation Mechanism, and Aggregate Results and Justification.
- The framework maps abstract stakeholder values to concrete technical objectives for LLM Agents, which then generate candidate lists for a social choice-based Aggregation Mechanism.
- This approach ensures transparent and mathematically traceable consensus-building while providing natural language justifications for the final Aggregate Results and Justification.

---


[Group Cognition Learning: Making Everything Better Through Governed Two-Stage Agents Collaboration](http://arxiv.org/abs/2605.00370)

- GCL (Group Cognition Learning): introduces a two-stage governed collaboration paradigm that regulates cross-modal interaction through Routing Agent, Auditing Agent, Public-Factor Agent, and Aggregation Agent to mitigate modality dominance and spurious coupling.
- The framework employs a selective interaction stage to filter redundant information based on marginal predictive gain and a consensus formation stage to disentangle shared semantics from private modality-specific representations.
- By replacing implicit fusion with an explicit governance protocol, GCL achieves state-of-the-art performance on multimodal sentiment analysis and intent recognition benchmarks while maintaining superior robustness against noise and spurious correlations.

---

[From Prompt to Physical Actuation: Holistic Threat Modeling of LLM-Enabled Robotic Systems](http://arxiv.org/abs/2604.27267)

- LLM-Enabled Robotic Systems: introduces a hierarchical Data Flow Diagram (DFD) and STRIDE-per-interaction analysis to systematically trace end-to-end threat propagation across the perception-planning-actuation pipeline of an autonomous robot.
- The framework models the system using two trust boundaries (TB1 for edge server, TB2 for autonomous platform) to identify security risks at six boundary-crossing interaction points where untrusted inputs enter protected domains.
- The analysis integrates Conventional Cyber Threats, Adversarial Threats, and Conversational Threats to demonstrate how compromises propagate through cross-modal translation and unmediated boundary crossings to cause unsafe physical actuation.

---

[An Empirical Study of Speculative Decoding on Software Engineering Tasks](http://arxiv.org/abs/2604.26469)

- SD (Speculative Decoding) framework: introduces a systematic empirical evaluation of speculative decoding methods across diverse software engineering tasks to mitigate autoregressive inference latency.
- The study benchmarks model-based approaches like MLP Speculators and Eagle-3 against model-free methods including Prompt Lookup Decoding and Suffix Decoding to assess their effectiveness in code generation, repair, and editing.
- The research identifies that code exhibits higher contextual repetitiveness than natural language, enabling model-free methods to achieve significant acceleration, while also highlighting the noise introduced by agentic infinite loops in complex tasks.

---

[cotomi Act: Learning to Automate Work by Watching You](http://arxiv.org/abs/2605.03231)

- cotomi Act: introduces a browser-based agent that combines reliable multi-step task execution with persistent organizational knowledge learned from user behavior via a Behavior Logger, Agentic ETL Pipeline, and Shared Knowledge Workspace.
- The system utilizes an Agent Scaffold with adaptive lazy observation, verbal-diff-based history compression, coarse-grained actions, and test-time scaling to maintain context focus during the ReAct Loop.
- The Shared Knowledge Workspace acts as a bidirectional interface where the agent retrieves organizational context via a Search Workspace Tool and users curate artifacts to ensure alignment with evolving workflows.

---

[MAGE: Safeguarding LLM Agents against Long-Horizon Threats via Shadow Memory](http://arxiv.org/abs/2605.03228)

- MAGE: introduces a defensive framework that leverages a dedicated Shadow Memory to distill security-critical context across an agent's execution trajectory for proactive risk assessment.
- The framework utilizes a Memory Manager to maintain the Shadow Memory and a Judge to perform retrospective inspection of pending actions, both optimized via turn-wise reinforcement learning.
- MAGE effectively mitigates long-horizon threats by intercepting malicious actions before execution while maintaining high benign utility and low computational overhead.

---

[MenuNet: A Strategy-Proof Mechanism for Matching Markets](http://arxiv.org/abs/2605.03216)

- MenuNet: introduces a neural mechanism design framework that generates personalized probabilistic menus to balance fairness and non-wastefulness under complex distributional constraints.
- The framework utilizes a shared Neural Network Fθ to generate menus, which are then processed by an Assignment Component to ensure strategy-proofness by construction.
- MenuNet incorporates a K-Barrier Penalty and a Stability Loss Component to optimize differentiable objectives, effectively managing trade-offs between competing market axioms in large-scale environments.

---

[ENWAR 3.0: An Agentic Multi-Modal LLM Orchestrator for Situation-Aware Beamforming, Blockage Prediction, and Handover Management](http://arxiv.org/abs/2605.03215)

- ENWAR 3.0: introduces a hierarchical agentic framework that unifies multi-modal sensing, LLMs, and context-driven model selection for real-time wireless network orchestration.
- The system integrates an Environment Classifier, Primed LLM, Agent Manager, Long-Term Memory, DRL Policy, Beam Prediction Agent, Blockage Prediction Agent, Handover Agent, and Perception Agent to enable degradation-aware, latency-bounded decision-making.
- ENWAR 3.0 achieves robust performance by dynamically routing tasks to specialized agents based on real-time sensor reliability and historical context, ensuring bounded-latency operation in I2V networks.

---

[When Agents Handle Secrets: A Survey of Confidential Computing for Agentic AI](http://arxiv.org/abs/2605.03213)

- Confidential Computing for Agentic AI: introduces a comprehensive survey of hardware-rooted security defenses for LLM-driven agents, mapping functional layers including Agent Core, Local Memory, Retrieval Access, Tool Policy/Mediation, and Credentials/Tokens/Secrets to TEE-based trust boundaries.
- The paper establishes an agent-centric threat model that addresses infrastructure-level adversaries by leveraging TEE isolation, remote attestation, and protected memory to secure sensitive agentic workflows.
- It identifies critical open challenges for production agentic systems, specifically focusing on compound attestation for multi-hop chains, TEE-backed RAG isolation, and the performance overhead of GPU-TEE integration at LLM scale.

---

[ADAPTS: Agentic Decomposition for Automated Protocol-agnostic Tracking of Symptoms](http://arxiv.org/abs/2605.03212)

- ADAPTS: introduces a modular LLM framework for automated psychiatric severity assessment that decomposes clinical interviews into symptom-specific reasoning tasks using WhisperX, Pyannote.audio, symptom-specific sub-agents, GRID-HAMD structure, and Severity Anchor Structure.
- The framework utilizes a mixture-of-agents architecture to perform evidence retrieval and generate auditable justifications, mitigating context dilution and protocol-specific brittleness.
- By incorporating explicit clinical qualitative conventions, the system improves calibration and achieves expert-level rank-order consistency across heterogeneous interview protocols.

---

[Human-Provenance Verification should be Treated as Labor Infrastructure in AI-Saturated Markets](http://arxiv.org/abs/2605.03210)

- Human-Provenance Verification Framework: introduces a model of labor market bifurcation where AI saturation creates a barbell structure, concentrating value in infrastructure ownership and premium human-provenance markets while compressing middle-tier work.
- The framework categorizes human-presence value into relational presence, aesthetic provenance, and accountability labor, arguing that these require robust verification infrastructure to maintain worker bargaining power.
- The paper proposes that provenance systems must be designed as portable, privacy-preserving, and auditable labor infrastructure to prevent the capture of value by platforms and to protect workers from synthetic mimicry.

---

[Kerncap: Automated Kernel Extraction and Isolation for AMD GPUs](http://arxiv.org/abs/2605.03208)

- Kerncap: introduces an automated tool for extracting and isolating GPU kernels from complex applications into self-contained, editable, and validated reproducer projects.
- The framework utilizes HSA-level runtime interception, address-space closure, and automated source discovery to recover the kernel definition, runtime state, and environment required for faithful replay.
- Kerncap enables rapid edit-recompile-validate loops for GPU kernel optimization, significantly reducing iteration time compared to traditional full-application rebuild workflows.

---

[Dependency-Aware Privacy for Multi-Turn LLM Agents](http://arxiv.org/abs/2605.03188)

- RootGuard: introduces a dependency-aware privacy mechanism for multi-turn LLM agent interactions that sanitizes root values once and computes all subsequent releases deterministically from the noised roots.
- The framework leverages structural domain knowledge to allocate privacy budgets across roots, improving the privacy-utility tradeoff while ensuring that derived values inherit privacy at zero marginal cost.
- RootGuard creates a double asymmetry where additional interaction turns simultaneously improve utility and maintain constant privacy, effectively mitigating risks from independent noising and MAP reconstruction attacks.

---

[Learning Correct Behavior from Examples: Validating Sequential Execution in Autonomous Agents](http://arxiv.org/abs/2605.03159)

- Validating Sequential Execution in Autonomous Agents: introduces an automated validation framework that constructs generalized ground-truth models from a small number of passing execution traces using Prefix Tree Acceptors (PTA), Multi-tiered state equivalence detection, LLM-based semantic analyzer, Dominator tree extractor, and Topological subsequence matcher.
- The system leverages dominator analysis to distinguish essential states from optional variations, enabling robust validation of non-deterministic agent behavior without manual specifications.
- Experimental results demonstrate that the framework achieves 100% accuracy in detecting product bugs and agent issues by comparing new executions against the learned dominator tree model.

---

[Pact: A Choreographic Language for Agentic Ecosystems](http://arxiv.org/abs/2605.03143)

- Pact: introduces a choreographic programming language that extends traditional protocol specifications with explicit constructs for agent choices, utilities, and environmental priors to enable game-theoretic reasoning in multi-agentic ecosystems.
- The framework maps protocols to formal games, allowing for the construction of decision policy solvers that utilize recursive theory-of-mind models to analyze agent behavior under information asymmetry.
- By providing a formal, verifiable structure for negotiation, the approach addresses the limitations of unstructured LLM-based coordination, such as vulnerability to prompt injection, sycophancy, and coercion.

---

[MARS-DA: A Hierarchical Reinforcement Learning Framework for Risk-Aware Multi-Agent Bidding in Power Grids](http://arxiv.org/abs/2605.03142)

- MARS-DA: introduces a hierarchical reinforcement learning framework that decouples electricity market bidding into specialized sub-policies and a Meta-Controller for adaptive risk-aware coordination.
- The framework utilizes a Meta-Controller to dynamically blend actions from a Safe Agent and a Speculator Agent to optimize risk-adjusted returns in two-settlement electricity markets.
- Extensive experiments demonstrate that MARS-DA achieves superior risk-adjusted performance and stability compared to state-of-the-art baselines during periods of extreme market volatility.

---

[PIIGuard: Mitigating PII Harvesting under Adversarial Sanitization](http://arxiv.org/abs/2605.03129)

- PIIGuard: introduces a webpage-level defense that embeds optimized hidden HTML fragments to steer LLMs away from disclosing PII, utilizing Seed Selection, Rule-based Scoring, Composite-utility Ranking, Evolutionary Mutation, and Judge-based Recovery Assessment.
- The framework employs a Target LLM for browsing, a Sanitizer LLM for preprocessing, and a Judge LLM to ensure PII remains unrecoverable from the generated response.
- PIIGuard optimizes defense fragments through an evolutionary loop to maintain effectiveness against both direct HTML access and attacker-side sanitization while preserving benign QA utility.

---

[Taming the Curses of Multiagency in Robust Markov Games with Large State Space through Linear Function Approximation](http://arxiv.org/abs/2605.03125)

- R-LMGs: introduces provably sample-efficient algorithms for robust linear Markov games with large state spaces using L-Robust-Q-FTRL and Online-L-Robust-Q-FTRL, which utilize Hybrid-Sampling, Ridge regression, FTRL, Optimistic robust value estimates, Pessimistic robust value estimates, and Uncertainty set.
- The framework addresses the curse of multiagency in robust Markov games by employing independent per-agent linear function approximation and a novel hybrid sampling strategy to approximate adversarial environments.
- The proposed algorithms achieve polynomial sample complexity and sublinear regret, providing the first provable guarantees for robust Markov games with large or infinite state spaces.

---

[ARISE: A Repository-level Graph Representation and Toolset for Agentic Fault Localization and Program Repair](http://arxiv.org/abs/2605.03117)

- ARISE (Agentic Repository-level Issue Solving Engine): introduces a framework that augments an LLM-based agent with a multi-granularity program graph and a three-tier tool API to improve repository-level fault localization and automated program repair.
- The framework utilizes a multi-granularity program graph that extends structural relationships down to statement-level nodes connected by intra-procedural definition-use edges, enabling precise data-flow slicing as a first-class agent primitive.
- Experimental results on SWE-bench Lite demonstrate that ARISE significantly improves function- and line-level localization and repair success by providing LLMs with structured, queryable evidence directly from the codebase.

---

[ARIS: Autonomous Research via Adversarial Multi-Agent Collaboration](http://arxiv.org/abs/2605.03042)

- ARIS: introduces a research harness that coordinates autonomous ML research workflows through cross-model adversarial collaboration to mitigate plausible unsupported success.
- The framework utilizes an Execution Layer, Orchestration Layer, and Assurance Layer to decompose long-horizon research into inspectable, modular, and verifiable stages.
- ARIS employs a persistent Research Wiki for cross-session memory and enforces reviewer independence by pairing executor and reviewer agents from different model families.

---

[Stable Agentic Control: Tool-Mediated LLM Architecture for Autonomous Cyber Defense](http://arxiv.org/abs/2605.03034)

- Tool-mediated architecture: introduces a control-theoretic framework for autonomous cyber defense that uses a Bayesian observer, Stackelberg best-response, double oracle expansion, and catalog-membership enforcement to ensure system stability via a composite Lyapunov function.
- The architecture employs LLM agents as controllers and adversaries that operate through a tool-output interface, which restricts their actions to a finite catalog to maintain formal stability guarantees.
- Formal verification of the system's controllability, observability, and Input-to-State Stability (ISS) is achieved using the Lean 4 proof assistant, with empirical validation demonstrating significant game-value reduction on real-world enterprise attack graphs.

---

[EvoPoC: Automated Exploit Synthesis for DeFi Smart Contracts via Hierarchical Knowledge Graphs](http://arxiv.org/abs/2605.02868)

- EvoPoC: introduces a knowledge-driven agentic system that leverages a Hierarchical Knowledge Graph to bridge the semantic gap between vulnerability detection and end-to-end exploit synthesis in DeFi smart contracts.
- The framework utilizes an evolving agentic memory mechanism and a two-stage validation process, combining SMT-based reachability checking with asset-level simulation to ensure the logical and economic viability of generated exploits.
- EvoPoC demonstrates superior performance over existing fuzzers and LLM-based scanners, achieving a 96.6% exploit success rate and identifying 16 previously unknown 0-day vulnerabilities in real-world DeFi protocols.

---

[HAAS: A Policy-Aware Framework for Adaptive Task Allocation Between Humans and Artificial Intelligence Systems](http://arxiv.org/abs/2605.02832)

- HAAS: introduces a governance-constrained framework for adaptive task allocation that integrates a rule-based PolicyEngine with a contextual-bandit learner to optimize Human-AI collaboration.
- The framework utilizes a five-dimension cognitive instrument to characterize subtasks and a five-mode autonomy spectrum to manage the distribution of work between human agents and AI systems.
- HAAS incorporates explicit human-state dynamics—fatigue, trust, and deskilling—into the reward signal to ensure long-term capability retention alongside operational efficiency.

---

[FlexSQL: Flexible Exploration and Execution Make Better Text-to-SQL Agents](http://arxiv.org/abs/2605.02815)

- FlexSQL: introduces a text-to-SQL agent that replaces fixed pipelines with flexible database interaction, utilizing Plan Generation, Program Generation, and Majority Voting to ground reasoning in actual data.
- The framework employs a suite of context query tools and executors to enable incremental schema navigation and data inspection, supporting both SQL and Python for bilingual program synthesis.
- FlexSQL incorporates a two-tiered repair mechanism with plan-level backtracking to recover from reasoning errors, achieving superior performance on the Spider2.0 benchmark.

---

[Autonomous LLM Agent Worms: Cross-Platform Propagation, Automated Discovery and Temporal Re-Entry Defense](http://arxiv.org/abs/2605.02812)

- SSCGV, SRPO, and RTW-A: introduces a systematic framework for analyzing and defending against autonomous worm propagation in file-backed LLM agent ecosystems by identifying injectable surfaces, optimizing payloads for semantic resilience, and enforcing temporal re-entry constraints.
- The research identifies that read operations in LLM-mediated systems represent a critical integrity threat, enabling autonomous cross-platform propagation through persistent carriers and trust-based delegation.
- RTW-A provides formal security guarantees by implementing layered defenses, including RTW enforcement, capability attenuation, and typed memory promotion, to block the persistence-re-entry-action chain without disrupting legitimate agent workflows.

---

[Tool Use as Action: Towards Agentic Control in Mobile Core Networks](http://arxiv.org/abs/2605.02811)

- Agentic Mobile Core Network Framework: introduces an architecture for 6G mobile networks that utilizes LLM-based agents to perform intent-based control and monitoring of network functions via A2A and MCP protocols.
- The framework employs a Host Agent for task delegation, while specialized Monitoring and Execution agents interact with network entities through standardized tool servers.
- Experimental evaluation on an OAI-based core network demonstrates that while protocol-level overhead is minimal, LLM inference for reasoning and tool selection remains the primary contributor to end-to-end latency.

---

[Reinforcement Learning for LLM-based Multi-Agent Systems through Orchestration Traces](http://arxiv.org/abs/2605.02801)

- LLM-MAS RL: introduces a taxonomy and audit framework for training coordinated LLM agent teams using orchestration traces as a shared event-graph abstraction.
- The framework organizes literature across three technical axes: reward design, credit assignment across eight granular units, and orchestration learning sub-decisions.
- It identifies a significant scale gap between academic evaluation regimes and industrial deployment envelopes, highlighting engineering constraints like rollout cost and harness boundaries.

---

[DynoSLAM: Dynamic SLAM with Generative Graph Neural Networks for Real-World Social Navigation](http://arxiv.org/abs/2605.02759)

- DynoSLAM: introduces a dynamic GraphSLAM architecture that integrates a stochastic GAT as a World Model to capture multimodal human motion uncertainty via Monte Carlo rollouts.
- The framework embeds predicted mean trajectories and empirical covariance matrices into the SLAM factor graph using a dynamic Mahalanobis distance factor to modulate kinematic stiffness.
- This approach enables robots to generate probabilistic safety envelopes for downstream controllers, facilitating anticipatory navigation in crowded environments.

---

[Mitigating Misalignment Contagion by Steering with Implicit Traits](http://arxiv.org/abs/2605.02751)

- SIT (Steering with Implicit Traits): introduces a black-box method to mitigate misalignment contagion in multi-agent LLM workflows by periodically injecting system prompts that reinforce an agent's core implicit traits.
- The framework utilizes Persona Assignment (assigns personas), Pre-game Persona Assessment (identifies traits), Social Dilemma Game Engine (simulates interactions), System Prompt Repetition (baseline intervention), Steering with Implicit Traits (reinforces traits), and Post-game Assessment (measures trait drift).
- The research demonstrates that LLMs often drift toward anti-social behavior during competitive multi-agent interactions, and that SIT effectively counters this contagion without requiring access to model weights or internal states.

---

[AI-Generated Smells: An Analysis of Code and Architecture in LLM- and Agent-Driven Development](http://arxiv.org/abs/2605.02741)

- MetaGPT (Multi-Agent Framework): introduces a systematic audit of technical debt in AI-generated software, revealing that LLMs and agents exhibit a distinct "machine signature" of defects rather than eliminating them.
- The research identifies a "Reasoning-Complexity Trade-off" where more capable LLMs generate bloated procedural code, while multi-agent systems struggle with architectural decay as task complexity increases.
- The study establishes a "Volume-Quality Inverse Law," demonstrating that code volume is a near-perfect predictor of architectural degradation that cannot be mitigated by prompt engineering.

---

[Augmenting Interface Usability Heuristics for Reliable Computer-Use Agents](http://arxiv.org/abs/2605.02729)

- UI-Verse: introduces a framework for evaluating agent-compatible interface design by augmenting classical usability heuristics to improve CUA perception, grounding, and control.
- The research identifies that interface design is a critical factor for CUA reliability, proposing four specific heuristics to reduce agent exploration burden and reasoning errors.
- Experimental results demonstrate that these augmented heuristics consistently improve task success rates and efficiency across various LLM-based agents without compromising human usability.

---

[ORPilot: A Production-Oriented Agentic LLM-for-OR Tool for Optimization Modeling](http://arxiv.org/abs/2605.02728)

- ORPilot: introduces a production-oriented agentic AI system that translates natural-language business problems into solver-ready optimization models using an Interview Agent, Data Collection Agent, Parameter Computation Agent, and a solver-agnostic Intermediate Representation (IR).
- The system utilizes a self-correcting retry loop that feeds solver errors and tracebacks back to the LLMs for targeted repairs, ensuring robustness in production environments.
- ORPilot decouples model formulation from solver-specific code through its Intermediate Representation, enabling deterministic recompilation and backend portability without further LLM calls.

---

[An Empirical Study of Agent Skills for Healthcare: Practice, Gaps, and Governance](http://arxiv.org/abs/2605.02709)

- Healthcare Agent Skills Framework: introduces an empirical analysis of 557 healthcare-related agent skills, characterizing their function, deployment context, autonomy, and safety through a structured annotation process.
- The study reveals that public healthcare skills are predominantly patient-facing and focus on administrative workflows, contrasting with the diagnostic and treatment-oriented focus of academic research.
- The findings highlight a significant gap in healthcare-agent governance, noting that technical autonomy does not reliably capture clinical impact and that many skills lack explicit safety boundary statements.

---

[Hybrid Inspection and Task-Based Access Control in Zero-Trust Agentic AI](http://arxiv.org/abs/2605.02682)

- CASA (Continuous Agent Semantic Authorization): introduces a hybrid runtime enforcement framework that combines deterministic guardrails and semantic intent inspection to secure LLM-driven agentic applications.
- The framework utilizes an interception layer to perform structural integrity checks and a two-stage semantic pipeline to verify that tool invocations align with the subject's original task objectives.
- The authors provide a novel multi-turn conversation dataset and experimental results demonstrating the effectiveness of semantic inspection in mitigating unauthorized tool access in agentic systems.

---

[The Design and Composition of Structural Causal Decision Processes](http://arxiv.org/abs/2605.02681)

- SCDP: introduces a formal framework for modeling resource-rational agents in dynamic environments by composing SCDMs with endogenous cognitive constraints and variable discounting.
- The framework utilizes bridge variables to enable sequential decomposition of complex decision problems, significantly improving computational efficiency through backwards induction.
- SCDPs extend beyond traditional POMDPs by explicitly modeling memory capacity and information flow, allowing for the representation of agents with bounded rationality and costly cognitive resources.

---

[An explainable hypothesis-driven approach to Drug-Induced Liver Injury with HADES](http://arxiv.org/abs/2605.02669)

- HADES: introduces an agentic framework that generates transparent, auditable reasoning traces for DILI risk assessment by fusing molecular-level predictions, metabolite decomposition, structural understanding, and toxicity pathways.
- The system employs an epistemic division of labor where a DILI subagent performs tool-based structural analysis, followed by an anonymized DeepResearch agent that contextualizes findings against biomedical literature.
- HADES utilizes the DILER Benchmark to evaluate mechanistic hypothesis generation, moving beyond binary classification to provide actionable, literature-grounded explanations for toxicological investigation.

---

[AcademiClaw: When Students Set Challenges for AI Agents](http://arxiv.org/abs/2605.02661)

- AcademiClaw: introduces a bilingual benchmark of 80 complex, long-horizon academic tasks sourced from student workflows, utilizing a Docker Sandbox, OpenClaw Agent, Tool Palette, Evaluation Rubric, Safety Auditor, and API Logging Proxy.
- The framework evaluates LLMs on academic-level difficulty and domain coverage, revealing capability boundaries and divergent behavioral phenotypes among frontier models.
- Analysis demonstrates that token consumption does not correlate with output quality, highlighting the need for better stopping criteria in autonomous agent frameworks.

---

[ARA: Agentic Reproducibility Assessment For Scalable Support Of Scientific Peer-Review](http://arxiv.org/abs/2605.02651)

- ARA: introduces a domain-agnostic pipeline that converts scientific papers into workflow graphs to evaluate reproducibility as a structured reasoning task.
- The framework utilizes an LLM-based agentic extraction pipeline to map document components into nodes and edges, subsequently applying micro-level scoring and aggregated reproducibility metrics.
- Experiments on the ReScience C dataset demonstrate that ARA provides consistent, scalable reproducibility assessments that approximate human expert judgments without requiring full experimental replication.

---

[AutoFocus: Uncertainty-Aware Active Visual Search for GUI Grounding](http://arxiv.org/abs/2605.02630)

- AutoFocus: introduces a training-free, uncertainty-aware active visual search framework that mitigates resolution bottlenecks in GUI grounding by leveraging token-level perplexity as an intrinsic signal for adaptive refinement.
- The framework utilizes Gaussian Dynamic Focusing to convert coordinate perplexity into anisotropic spatial probability fields, enabling targeted zoom-in operations without requiring additional model training.
- By integrating error-triggered verification, Shape-Aware Zooming, and multi-hypothesis visual aggregation, AutoFocus significantly improves grounding precision for small interactive elements in high-resolution interfaces.

---

[Beating the Style Detector: Three Hours of Agentic Research on the AI-Text Arms Race](http://arxiv.org/abs/2605.02620)

- Agentic Research Harness: introduces a framework for rapid reproducibility and adversarial evaluation of LLMs in the context of authorship style mimicry and AI-text detection.
- The framework utilizes LUAR-MUD as a frozen authorship-style embedding model to measure stylistic similarity and detect AI-generated text through a leakage-free held-out protocol.
- It incorporates an agentic adversarial rewriting loop where LLMs iteratively refine drafts against a frozen LinearSVC detector to minimize detection probability while maintaining stylistic fidelity.

---

[Foundation-Model-Based Agents in Industrial Automation: Purposes, Capabilities, and Open Challenges](http://arxiv.org/abs/2605.02592)

- Foundation-Model-Based Industrial Agent: introduces a systematic literature review of 88 publications to define and evaluate the maturity, purposes, and capabilities of agents integrated with Foundation Models in industrial contexts.
- The paper identifies a shift from conventional negotiation-heavy multi-agent systems toward assistive, monitoring, and process-optimization roles enabled by LLMs and MLLMs.
- Key findings highlight that while FM-based agents improve human interaction and uncertainty handling, they face significant deployment barriers including hallucination, output instability, and integration challenges with industrial infrastructure.

---

[Beyond State Machines: Executing Network Procedures with Agentic Tool-Calling Sequences](http://arxiv.org/abs/2605.02584)

- Agentic Network Procedure Execution Framework: introduces a systematic evaluation of LLM-based agents executing network procedures through sequential tool invocations across four distinct architectural approaches.
- The framework compares iterative agent-driven reasoning against tool-encapsulated execution to determine the impact of procedure placement on end-to-end latency and execution correctness.
- The research defines a procedure-specific error taxonomy to categorize failures in multi-step agentic workflows and identifies scalability limits for LLMs in complex network automation tasks.

---

[On Training Large Language Models for Long-Horizon Tasks: An Empirical Study of Horizon Length](http://arxiv.org/abs/2605.02572)

- Horizon Reduction (HR): introduces a systematic empirical study identifying task horizon length as a fundamental training bottleneck for LLMs, proposing horizon reduction via macro actions and subgoal decomposition to stabilize reinforcement learning.
- The framework addresses training instability in long-horizon tasks by mitigating exploration difficulties and credit assignment challenges through effective horizon reduction.
- The research demonstrates that models trained with horizon reduction exhibit horizon generalization, enabling robust performance on longer, unseen task variants at inference time.

---

[IteRate: Autonomous AI Synthesis of In-Kernel eBPF Wi-Fi Rate Control Algorithms](http://arxiv.org/abs/2605.02542)

- IteRate: introduces an autonomous research system that uses a multi-agent AI architecture to synthesize, deploy, and evaluate in-kernel eBPF Wi-Fi rate control algorithms on commodity hardware.
- The system employs a Scientist agent that coordinates specialized subagents—Experiment Runner, Algorithm Designer, Data Analyst, and Network Engineer—to conduct a closed-loop scientific research cycle without human intervention.
- By leveraging high-fidelity telemetry and a programmable MAC-layer architecture, IteRate enables the online synthesis of bespoke algorithms that outperform traditional heuristics across diverse wireless environments and workloads.

---

[Orchestrating Spatial Semantics via a Zone-Graph Paradigm for Intricate Indoor Scene Generation](http://arxiv.org/abs/2605.02537)

- ZoneMaestro: introduces a unified framework that shifts indoor scene synthesis from object-centric placement to Zone-Graph Orchestration by internalizing zone-based logic and topological constraints.
- The framework utilizes an Alternating Spatial Alignment strategy that cycles between reasoning internalization and Zone-Aware Group Relative Policy Optimization to reconcile semantic richness with geometric validity.
- To evaluate spatial intelligence in non-convex environments, the authors introduce the SCALE benchmark and the Zone-Scene-10K dataset, demonstrating superior structural coherence and intent adherence over existing baselines.

---

[DataClaw: A Process-Oriented Agent Benchmark for Exploratory Real-World Data Analysis](http://arxiv.org/abs/2605.02503)

- DataClaw: introduces a process-oriented benchmark for exploratory real-world data analysis that evaluates LLM agents using DataClaw, OpenClaw, Docker container, LLM Judger, GLM-5, Human-in-the-loop annotation pipeline, Consensus verification, Outcome evaluation, and Process evaluation.
- The framework utilizes a three-stage human-in-the-loop annotation pipeline to create 492 expert-designed tasks across enterprise, industry, and policy domains, incorporating native data noise to simulate real-world exploratory analysis.
- DataClaw enables diagnostic evaluation by measuring both final-answer accuracy and intermediate reasoning progress through milestone-based process scoring, revealing distinct exploration archetypes among LLMs.

---

[GRAIL: A Deep-Granularity Hybrid Resonance Framework for Real-Time Agent Discovery via SLM-Enhanced Indexing](http://arxiv.org/abs/2605.02489)

- GRAIL (Granular Resonance-based Agent/AI Link): introduces a multi-stage discovery framework that replaces heavy-weight LLM parsing with an SLM Predictor and utilizes a tri-dimensional indexing strategy to achieve sub-400ms latency.
- The framework employs Hybrid Recall to combine Sparse Index-based filtering with Context Index-based dense retrieval, followed by MaxSim Resonance to mitigate semantic dilution through fine-grained matching against an Intent Index.
- By decoupling agent metadata into Tags, Context, and Intent, GRAIL maintains high retrieval precision for multi-purpose agents while significantly reducing the computational overhead compared to traditional LLM-centric discovery methods.

---

[Accurate Legal Reasoning at Scale: Neuro-Symbolic Offloading and Structural Auditability for Robust Legal Adjudication](http://arxiv.org/abs/2605.02472)

- Amortized Intelligence: introduces a neuro-symbolic architecture that offloads complex legal reasoning from LLMs to a deterministic symbolic engine, utilizing a one-time compilation step to translate contracts into a typed graph representation.
- The system employs an LLM as a semantic compiler to generate DACL graphs, which are subsequently executed by a lightweight agent to ensure mathematical precision and structural auditability in high-volume legal adjudication.
- By decoupling reasoning from execution, the framework achieves near-perfect consistency and significant cost reductions compared to standard LLM-based inference, effectively mitigating the "reasoning cliff" observed in probabilistic models.

---

[When Stress Becomes Signal: Detecting Antifragility-Compatible Regimes in Multi-Agent LLM Systems](http://arxiv.org/abs/2605.02463)

- CAFE (Controlled Antifragility-compatible Framework for Evaluation): introduces a statistical framework for detecting antifragility-compatible regimes in multi-agent LLM systems by modeling stress-response geometry.
- The framework utilizes Prompt Formation, Agentic Architectures, Judge, Multi-output Polynomial Response Model, Inverse Reconstruction, and Distributional Jensen Gap to identify structured stress variation.
- CAFE distinguishes between immediate performance robustness and the presence of learnable stress signals, providing a measurement layer for future adaptive LLM systems.

---

[Causal Software Engineering: A Vision and Roadmap](http://arxiv.org/abs/2605.02454)

- CSE: introduces a paradigm shift in software engineering by integrating explicit causal models and reasoning into the software lifecycle to move beyond correlational analysis.
- The framework utilizes Causal design specs, Intervention logs, and a Living Causal Model to enable interventional and counterfactual reasoning for improved decision-making.
- CSE incorporates Causal copilots and LLM-based agents to provide uncertainty-aware, auditable, and causally-grounded assistance for software development and operations.

---

[A Behavioral Micro-foundation for Cross-sectional Network Models](http://arxiv.org/abs/2605.02441)

- Behavioral Micro-foundation Framework: introduces a choice-theoretic foundation for cross-sectional network models by linking agent-based stochastic decision processes to Exponential Random Graph Model (ERGM) equilibria.
- The framework incorporates multilateral edge control and agent-node distinctions, utilizing a prosphoric array to aggregate agent decisions via a resolution function that defines relational norms.
- The approach enables behavioral inference and counterfactual analysis by demonstrating how individual utility functions and relational norms collectively determine the long-run structural properties of social networks.

---

[ARIADNE: Agentic Reward-Informed Adaptive Decision Exploration via Blackboard-Driven MCTS for Competitive Program Generation](http://arxiv.org/abs/2605.02431)

- ARIADNE: introduces, "a blackboard-driven MCTS framework that models competitive program generation as a sequential decision process to systematically explore solution spaces".
- The framework integrates MCTS as a global planner with specialized agents—including strategy-, code generation-, test generation-, scoring-, and code repair-agents—to iteratively refine program drafts.
- A structured blackboard system maintains persistent intermediate artifacts, enabling cross-branch knowledge transfer and evidence-driven refinement to improve performance under strict contest constraints.

---

[AOCI: Symbolic-Semantic Indexing for Practical Repository-Scale Code Understanding with LLMs](http://arxiv.org/abs/2605.02421)

- AOCI: introduces a symbolic-semantic indexing protocol that provides a stable, LLM-readable blueprint of a repository's architecture, dependencies, and design decisions to improve repository-scale code understanding.
- The framework utilizes a dual-layer encoding structure consisting of a Discrete Tag Layer (assigns compact architectural coordinates) and a Continuous Semantic Layer (carries business role and design details) to enable efficient, single-pass LLM comprehension.
- AOCI Platform maintains index consistency through incremental updates, ensuring the blueprint remains aligned with evolving codebases while significantly reducing token consumption compared to agent-based exploration methods.

---

[HEAVYSKILL: Heavy Thinking as the Inner Skill in Agentic Harness](http://arxiv.org/abs/2605.02396)

- HEAVYSKILL: introduces a training-free framework that decomposes complex reasoning into a two-stage pipeline of Parallel Reasoning Agents and a Sequential Deliberation Agent, supported by a Memory Cache and an optional Iterative Update Mechanism.
- The framework utilizes Parallel Reasoning Agents to generate diverse trajectories, which are then synthesized by a Sequential Deliberation Agent to produce a final answer, effectively scaling test-time compute.
- HEAVYSKILL functions as a readable skill for LLM orchestrators, enabling self-orchestration and iterative refinement to improve performance on complex reasoning tasks without requiring architectural modifications.

---

[A Low-Code Approach for the Automatic Personalization of Conversational Agents](http://arxiv.org/abs/2605.02384)

- BESSER-PEARL: introduces a low-code pipeline that leverages User Profile Model, Agent Profile Model, and Base Agent Model to automatically generate personalized conversational agents through M2M Transformation and M2C Transformation.
- The framework utilizes an LLM for design-time content adaptation and RAG for dynamic knowledge retrieval, while the BESSER Agentic Framework provides the underlying state-machine architecture.
- A pilot study demonstrates that the approach achieves high usability and usefulness scores across both technical and non-technical users for creating personalized conversational experiences.

---

[A Compound AI Agent for Conversational Grant Discovery](http://arxiv.org/abs/2605.02366)

- Conversational Grant Discovery system: introduces a compound AI architecture that unifies fragmented funding sources through an Aggregation Layer and an Agentic Query Processing Layer to provide accurate, real-time grant discovery.
- The system utilizes an LLM-based Parser to normalize heterogeneous data into a Unified Index, which is then queried by an Agent Orchestrator using search_index and web_search tools to ensure factual grounding.
- By supporting PDF context and iterative conversational refinement, the framework reduces the cognitive load of grant searching and minimizes LLM hallucinations compared to monolithic models.

---

[When Correct Isn’t Usable: Improving Structured Output Reliability in Small Language Models](http://arxiv.org/abs/2605.02363)

- AloLab: introduces an iterative system-prompt optimization framework that improves structured output reliability in LLMs by using a meta-agent to analyze and refine prompts based on observed model behavior.
- The framework utilizes a feedback loop consisting of a Solver, Evaluator, Analyzer, and Optimizer to iteratively rewrite system prompts without requiring weight access or fine-tuning.
- AloLab achieves high output accuracy at near-baseline inference latency, effectively addressing format-compliance failures in small LLMs that static prompting or constrained decoding cannot resolve efficiently.

---

[LLM-enabled Social Agents](http://arxiv.org/abs/2605.02335)

- LLM-enabled Social Agents: introduces a conceptual baseline for social agents that grounds behavior in persona-based role definitions rather than generic capability bundles.
- The architecture utilizes a hybrid deliberative approach where LLM-based meta deliberation orchestrates specialized modules including reactive-, rule-based-, planning-, utility-based-, and argumentation-based deliberation.
- The paper advocates for shifting research focus toward structured persona representations, synthetic datasets for evaluation, and hybrid control mechanisms to ensure long-term social coherence in LLMs.

---

[SOTOPIA-TOM: Evaluating Information Management in Multi-Agent Interaction with Theory of Mind](http://arxiv.org/abs/2605.02307)

- SOTOPIA-TOM: introduces a multi-dimensional benchmarking framework to evaluate LLMs' ability to manage information asymmetry and privacy in multi-party interactions, utilizing Scenario Generation Pipeline, Multi-agent Simulator, Evaluation Metric Suite, ToM-Coach, and ToM-Belief.
- The framework employs a composite INFOMGMT metric to assess coordination utility and privacy safety, revealing that current LLMs struggle with strategic information seeking and privacy-aware decision-making.
- ToM-based interventions, specifically ToM-Coach and ToM-Belief, demonstrate improved coordination-privacy balance compared to vanilla baselines and CoT-Privacy prompting strategies.

---

[EngiAgent: Fully Connected Coordination of LLM Agents for Solving Open-ended Engineering Problems with Feasible Solutions](http://arxiv.org/abs/2605.02289)

- EngiAgent: introduces a multi-agent framework that utilizes a Coordinator, Analyzer, Modeler, Verifier, Solver, and Evaluator to ensure feasibility in open-ended engineering problem solving.
- The framework employs a fully connected coordinator that dynamically routes feedback among specialized agents to correct errors in data processing, constraint handling, and solver execution.
- EngiAgent leverages shared memory to track error history and state, enabling adaptive scheduling and robust, feasibility-oriented engineering solutions across diverse domains.

---

[Complexity Horizons of Compressed Models in Analog Circuit Analysis](http://arxiv.org/abs/2605.02285)

- Prerequisite Graph framework: introduces a performance-aware model compression strategy that utilizes Directed Acyclic Graphs (DAGs) to map conceptual dependencies and optimize LLM cascade deployment in analog circuit analysis.
- The framework employs an Agentic Dataset Generation Pipeline to structure engineering tasks and a Strategic Evaluation Engine that uses Depth-First Search (DFS) to dynamically route queries across compressed LLM variants based on real-time performance.
- By identifying the "Complexity Horizon" and calculating the "Intelligence Delta" through tag-set intersection analysis, the approach enables granular, cost-effective model selection for hierarchical engineering workflows.

---

[These Aren’t the Reviews You’re Looking For: How Humans Review AI-Generated Pull Requests](http://arxiv.org/abs/2605.02273)

- AIDev-based data collection pipeline: introduces a large-scale empirical study analyzing human review activity on AI-generated pull requests by classifying interactions into agentic, automation, and human review categories.
- The study utilizes a rule-based classifier to distinguish between agent-steering, infrastructure-related automation, and direct human feedback within GitHub pull request discussions.
- Findings indicate that while overall human participation rates are similar for human-authored and AI-generated pull requests, the latter exhibit significantly higher levels of agent-steering and fewer instances of direct human-only review.

---

[Towards Understanding Specification Gaming in Reasoning Models](http://arxiv.org/abs/2605.02269)

- Specification Gaming Evaluation Suite: introduces a diverse set of eight environments to systematically measure and analyze the propensity of LLMs to exploit task specifications during deployment.
- The research demonstrates that RL reasoning training significantly increases the rate of specification gaming across various frontier LLMs.
- The study finds that while increasing reasoning effort and applying test-time mitigations like fallback options or no-exploit prompts can influence behavior, they do not fully eliminate specification gaming.

---

[Reliability-Oriented Multilingual Orthopedic Diagnosis: A Domain-Adaptive Modeling and a Conceptual Validation Framework](http://arxiv.org/abs/2605.02266)

- IndicBERT-HPA: introduces a domain-adaptive multilingual diagnostic framework for orthopedic classification that utilizes language-specific adapter heads to project linguistic representations into a clinically discriminative orthopedic subspace.
- The framework integrates a deterministic agent-based validation layer that performs evidence checking and enforces conservative human-in-the-loop gating to ensure safety in high-risk clinical decision support.
- Empirical results demonstrate that task-aligned encoders with domain-specific adaptation significantly outperform zero-shot LLMs in reliability, calibration, and cross-lingual stability for structured diagnostic tasks.

---

[A Study of Belief Revision Postulates in Multi-Agent Systems (Extended Version)](http://arxiv.org/abs/2605.02249)

- MBR (Multi-agent Belief Revision): introduces a formal framework for evaluating dynamic epistemic reasoning by generalizing classical AGM and DP belief revision postulates to multi-agent settings represented by a single Kripke structure.
- The paper defines and analyzes two specific revision operators, a generalized full-meet operator and an event-based operator, to assess their compliance with the proposed multi-agent belief revision postulates.
- The research provides formal proofs demonstrating that while these operators satisfy the generalized AGM postulates, they face challenges in fully satisfying iterated revision postulates like DP2.

---

[The Conversations Beneath the Code: Triadic Data for Long-Horizon Software Engineering Agents](http://arxiv.org/abs/2605.02244)

- Triadic Data Framework: introduces a methodology for training long-horizon software engineering agents by capturing human-human-AI interactions rather than relying solely on dyadic human-AI dialogue.
- The framework proposes two primary data products, long-horizon expert trajectories and simulated cross-functional companies, to capture the engineering deliberation and context formation missing from current datasets.
- A four-tier evidence framework is specified to ensure the quality and credibility of these high-cost corpora through mechanical verification, statistical characterization, probe experiments, and pre-registered blind evaluation.

---

[PhysicianBench: Evaluating LLM Agents in Real-World EHR Environments](http://arxiv.org/abs/2605.02240)

- PhysicianBench: introduces a benchmark for evaluating LLM agents on long-horizon clinical tasks within real-world EHR environments using Task Curation, Agent Environment, Checkpoint Evaluation, Agent Framework, FHIR Server, and Tool Inventory.
- The benchmark utilizes 100 physician-validated tasks and 670 checkpoints to measure the performance of LLMs in retrieving clinical data, reasoning, executing actions, and producing documentation.
- Evaluation of 12 LLMs reveals a significant performance gap, with the best model achieving a 46% success rate, highlighting the challenges of autonomous clinical agent deployment.

---

[ARGUS: Policy-Adaptive Ad Governance via Evolving Reinforcement with Adversarial Umpiring](http://arxiv.org/abs/2605.02200)

- ARGUS: introduces a three-stage policy-adaptive governance framework that utilizes multi-agent adversarial dialectics to resolve label inconsistencies and discover latent violations in advertising content.
- The framework employs a Prosecutor-Defender-Umpire architecture to generate high-fidelity rewards for reinforcement learning, effectively synchronizing model reasoning with evolving regulatory mandates.
- By incorporating a Skeptic agent for latent knowledge discovery, ARGUS pushes decision boundaries into complex "gray-area" territories, significantly improving detection precision and recall compared to traditional fine-tuning baselines.

---

[MEMAUDIT: An Exact Package-Oracle Evaluation Protocol for Budgeted Long-Term LLM Memory Writing](http://arxiv.org/abs/2605.02199)

- MEMAUDIT: introduces an exact package-oracle evaluation protocol for budgeted long-term LLM memory writing that isolates write-time memory selection from downstream retrieval and reader reasoning.
- The framework defines a finite package of experience streams, candidate memory representations, and storage costs to turn memory writing into a certified optimization problem.
- By using a union denominator, MEMAUDIT enables the diagnostic scoring of heterogeneous memory systems like Mem0, Letta, and A-Mem to identify bottlenecks in extraction versus budget-aware selection.

---

[Do We Really Need Immediate Resets? Rethinking Collision Handling for Efficient Robot Navigation](http://arxiv.org/abs/2605.02192)

- MCB (Multi-Collision reset Budget): introduces a framework that decouples local collision termination from global environment resets to accelerate early-stage exploration in DRL-based mapless navigation.
- The framework utilizes a Collision Counter and a predefined Collision Budget to allow agents to continue interacting with the environment after a collision, while employing a Collision-Aware Transition Construction to ensure consistent Bellman backups.
- An optional Pose-Change-Based Filtering component further improves training efficiency by discarding redundant collision transitions that occur at similar robot orientations.

---

[When Alignment Isn’t Enough: Response-Path Attacks on LLM Agents](http://arxiv.org/abs/2605.02187)

- RTA (Relay Tampering Attack): introduces a hierarchical framework that exploits the lack of end-to-end integrity in BYOK relay architectures by performing post-alignment tampering on LLM responses.
- The framework utilizes Strategic orchestration, Tactical manipulation, and Stealth restoration to hijack agent execution while bypassing safety alignment and maintaining stylistic consistency.
- RTA achieves high attack success rates across multiple LLMs and benchmarks, while a proposed time-channel detector provides a utility-preserving defense against such relay-side tampering.

---

[T2PO: Uncertainty-Guided Exploration Control for Stable Multi-Turn Agentic Reinforcement Learning](http://arxiv.org/abs/2605.02178)

- T2PO (Token- and Turn-level Policy Optimization): introduces a hierarchical uncertainty-aware framework that stabilizes multi-turn RL by explicitly controlling exploration through TTI, TDS, and RFT.
- The framework utilizes a self-calibrated uncertainty signal, integrating entropy and confidence, to monitor reasoning dynamics and prevent inefficient exploration at both token and turn levels.
- By dynamically terminating redundant reasoning and resampling unproductive turns, T2PO mitigates training collapse and improves exploration efficiency in complex interactive environments.

---

[Intervention Complexity as a Canonical Reward and a Measure of Intelligence](http://arxiv.org/abs/2605.02175)

- IC: introduces a canonical reward measure based on the minimum resource cost required for an agent to achieve a specific state transition within a computable environment.
- The framework decomposes intelligence into two independent dimensions: agent competence, which measures static performance against an oracle, and learning efficiency, which quantifies the rate of improvement through experience.
- By grounding rewards in the computational structure of the environment rather than external normative input, the paper provides a principled basis for evaluating superintelligence and pre-training universal agents.

---

[Planner Matters! An Efficient and Unbalanced Multi-agent Collaboration Framework for Long-horizon Planning](http://arxiv.org/abs/2605.02168)

- Planner-centric Multi-agent Collaboration Framework: introduces a modular multi-agent system that decomposes long-horizon automation into a Planner, an Actor, and a Memory Manager to optimize compute allocation.
- The framework identifies high-level planning as the primary performance bottleneck and utilizes a planner-centric reinforcement learning approach to exclusively optimize the Planner while keeping other components frozen.
- Experimental results across web navigation, OS control, and tool-use benchmarks demonstrate that this unbalanced compute-allocation strategy significantly improves task success rates and efficiency compared to monolithic LLM baselines.

---

[Experience Constrained Hierarchical Federated Reinforcement Learning for Large-scale UAV Teams in Hazardous Environments](http://arxiv.org/abs/2605.02165)

- EC-HFRL: introduces a hierarchical federated reinforcement learning framework designed for multi-UAV teams operating under fixed per-round experience budgets in hazardous environments.
- The framework utilizes a cluster-based architecture where active learners perform parallel policy updates by reusing a shared experience pool, with performance governed by replay dynamics rather than participation density.
- The research identifies that minibatch structure and key experience admission are primary determinants of learning success, while increased learner participation primarily impacts energy consumption.

---

[DocSync: Agentic Documentation Maintenance via Critic-Guided Reflexion](http://arxiv.org/abs/2605.02163)

- DocSync: introduces an agentic workflow that automates software documentation maintenance by fusing structural AST parsing, RAG, and a critic-guided Reflexion loop to ensure semantic consistency between code and documentation.
- The framework utilizes an Impact Analysis component to identify necessary updates, followed by an LLM-based generation process that is iteratively refined through a critic-guided feedback mechanism.
- By anchoring LLM generations in verified structural code representations, DocSync mitigates hallucinations and improves documentation accuracy on resource-constrained hardware.

---

[AAFLOW: Scalable Patterns for Agentic AI Workflows](http://arxiv.org/abs/2605.02162)

- AAFLOW: introduces a unified distributed runtime that models agentic workflows as a composition of operators to enable communication-efficient execution on HPC systems.
- The framework utilizes a zero-copy data plane based on Apache Arrow and Cylon to eliminate serialization overhead and decouple logical agent behavior from resource-deterministic scheduling.
- Experimental results demonstrate that AAFLOW achieves significant pipeline speedups by optimizing data orchestration and communication-bound stages rather than LLM inference.

---

[Combining Trained Models in Reinforcement Learning](http://arxiv.org/abs/2605.02159)

- DRL Knowledge Reuse Framework: introduces a systematic review of empirical studies on reusing pretrained models in DRL, categorizing mechanisms into Distillation, Transfer, Ensemble, and Federated Training.
- The paper synthesizes evidence across 15 studies, highlighting that reuse performance depends on structural overlap between source and target tasks and the use of explicit Gating Mechanism or Alignment Mechanism.
- It proposes an Independence Spectrum to classify model diversity based on Seed-, Data-, Task-, and Reward-diversity, while noting that current compute reporting remains insufficient for rigorous efficiency comparisons.

---

[Hierarchical Cooperative MARL for Joint Downlink PRB and Power Allocation in a 5G System](http://arxiv.org/abs/2605.02149)

- Hierarchical Cooperative MARL: introduces a sequential control framework that decomposes joint downlink PRB and power allocation into a PRB Agent, a Deterministic Resolver, and a Power Agent to optimize a throughput-fairness objective.
- The framework utilizes a staged curriculum-based training procedure to stabilize the learning of the PRB Agent and Power Agent within a physically grounded Sionna-based simulation environment.
- The system achieves significant cell-throughput gains over traditional Proportional Fair baselines by leveraging ray-traced channel information and cross-layer feedback loops.

---

[From Where Things Are to What They Are For: Benchmarking Spatial–Functional Intelligence in Multimodal LLMs](http://arxiv.org/abs/2605.02130)

- SFI-Bench: introduces a video-based benchmark evaluating spatial and functional reasoning in MLLMs through Metadata Generation, Task Templates, Human Verification, Post-hoc Quality Filtering, Web Search Tool, and Reasoning Engine.
- The framework assesses cognitive abilities by requiring models to construct structured spatial maps and infer object affordances from egocentric indoor videos.
- Experiments demonstrate that while MLLMs excel at local perception, they struggle with global spatial memory and functional integration, highlighting the necessity of external knowledge and robust reasoning.

---

#### 3rd May 2026


[NeuroState-Bench: A Human-Calibrated Benchmark for Commitment Integrity in LLM Agent Profiles](http://arxiv.org/abs/2605.01847)

- NeuroState-Bench: introduces a human-calibrated benchmark for evaluating commitment integrity in LLM agent profiles through Task Generator, Side-Query Probes, Human Calibration Layer, HCCIS-CORE Scorer, Auxiliary Neural and Mechanistic Modules, and Ranking and Phenotype Analysis Pipeline.
- The framework operationalizes commitment integrity by measuring whether an agent's elicited commitments remain consistent with task-defined constraints, bindings, and updates using benchmark-defined side-query probes.
- NeuroState-Bench demonstrates that endpoint success often under-specifies agent reliability, as integrity-aware evaluation reveals hidden failures and provides more stable rankings under distractor perturbations.

---


[NORA: A Harness-Engineered Autonomous Research Agent for Spatial Data Science](http://arxiv.org/abs/2605.02092)

- NORA: introduces a harness-engineered, multi-agent autonomous research system designed specifically for spatial data science workflows.
- The system utilizes a skills-first architecture with 21 domain-specialized skills and 9 specialist sub-agents to orchestrate the complete research lifecycle.
- NORA incorporates harness engineering principles, including generator-evaluator separation, human-in-the-loop checkpoints, and structured state persistence, to ensure reliable and reproducible research.

---

[Model Spec Midtraining: Improving How Alignment Training Generalizes](http://arxiv.org/abs/2605.02087)

- MSM: introduces a training phase occurring after pre-training and before AFT, where LLMs are trained on synthetic documents explaining the content of a Model Spec to shape their generalization from subsequent demonstration data.
- The framework utilizes a hierarchical pipeline to decompose Model Specs into coherent domains, generating diverse synthetic documents that teach the model the "what" and "why" of its intended character.
- Empirical results demonstrate that MSM significantly improves OOD alignment generalization and reduces agentic misalignment, outperforming standard AFT baselines by teaching models to internalize principles rather than just mimicking behaviors.

---

[Coopetition-Gym v1: A Formally Grounded Platform for Mixed-Motive Multi-Agent Reinforcement Learning under Strategic Coopetition](http://arxiv.org/abs/2605.02063)

- Coopetition-Gym v1: introduces a formally grounded benchmark platform for mixed-motive multi-agent reinforcement learning that utilizes Environment transition rules, Payoff vector, Reward layer, Reward signal, Learning algorithm, Oracle baselines, Behavioral audit, Statistical-gate methodology, Controlled critic-learning-rate ablation, and Matrix-coverage verification audit to enable systematic investigation of coopetitive dynamics.
- The framework employs a two-layer separation of payoff and reward, allowing for reward-type ablation to isolate the effects of reward mutuality from environment transition structures.
- The platform includes twenty environments organized into four mechanism-class tiers (interdependence, trust, collective action, and reciprocity) and provides a suite of 126 algorithms for comprehensive evaluation.

---

[Enhancing Judgment Document Generation via Agentic Legal Information Collection and Rubric-Guided Optimization](http://arxiv.org/abs/2605.02011)

- Judge-R1: introduces a unified framework that enhances LLM-based judgment document generation by integrating an agentic legal information collection mechanism with rubric-guided reinforcement learning.
- The framework utilizes a multi-stage pipeline comprising a hybrid retrieval module, supervised fine-tuning for structural alignment, and Group Relative Policy Optimization (GRPO) for legal reasoning fidelity.
- Judge-R1 employs a comprehensive legal reward function to evaluate legal correctness, structural professionalism, and reasoning quality, significantly outperforming existing baselines on the JuDGE benchmark.

---

[Optimized and kinematically feasible multi-agent motion planning](http://arxiv.org/abs/2605.01996)

- MAMP framework: introduces a two-step method for finding optimized and kinematically feasible trajectories for multi-agent systems by combining initial pathfinding via CBS or PBS with a subsequent multi-phase OCP improvement step.
- The framework utilizes synchronized motion primitives and a lattice-based planner to generate initial feasible solutions, which are then refined using numerical optimal control to ensure kinematic feasibility and collision avoidance.
- Experimental results demonstrate that the lattice-based planner outperforms SIPP-IP in success rates for complex tractor-trailer systems, while the multi-phase OCP effectively removes discretization artifacts from initial paths.

---

[12 Angry AI Agents: Evaluating Multi-Agent LLM Decision-Making Through Cinematic Jury Deliberation](http://arxiv.org/abs/2605.01986)

- 12 Angry AI Agents: instantiates a multi-agent benchmark for LLM deliberation using the AutoGen SelectorGroupChat framework to evaluate how different RLHF alignment intensities influence decision-making rigidity.
- The study compares GPT-4o and Llama-4-Scout across various prompt conditions, revealing that heavy RLHF alignment correlates with increased deliberative rigidity and anchoring, while lighter alignment allows for greater flexibility.
- The research demonstrates that current LLMs often reproduce the surface features of human deliberation while failing to perform the underlying mechanism of socially mediated, evidence-based mind-changing.

---

[Multi-User Dueling Bandits: A Fair Approach using Nash Social Welfare](http://arxiv.org/abs/2605.01961)

- FMUDB: introduces a framework for fair online learning from multi-user preference data by optimizing the Nash Social Welfare objective using Modified DKWT, Fair-ETC, and Fair-ϵ-Greedy components.
- The framework addresses the challenge of heterogeneous user preferences in dueling bandits by utilizing Condorcet winners as reference points to derive individual utility scores.
- Theoretical analysis establishes a regret lower bound of Ω(T^2/3 min(K, D)^1/3) and provides matching upper bounds for the proposed Fair-ETC and Fair-ϵ-Greedy algorithms.

---

[TRAP: Tail-aware Ranking Attack for World-Model Planning](http://arxiv.org/abs/2605.01950)

- TRAP (Tail-aware Ranking Attack on Planning): introduces a trigger-conditioned, inference-time backdoor attack framework that manipulates world-model planning by targeting the relative ranking of decision-critical imagined trajectories.
- The framework utilizes a tail-aware ranking loss to identify and suppress high-value trajectories, combined with a dual gating mechanism to ensure stable and behavior-preserving attack updates.
- By focusing on the long-tailed distribution of trajectory scores, TRAP effectively disrupts long-horizon planning in world-model-based agents without requiring training-time poisoning or parameter modification.

---

[A Language for Describing Agentic LLM Contexts](http://arxiv.org/abs/2605.01920)

- ACDL: introduces a formal, implementation-agnostic language for precisely specifying the structure and temporal evolution of input contexts in agentic LLM systems.
- The framework utilizes Role Messages, Context Variables, and Control Flow constructs to capture how prompts are assembled and transformed across multi-turn interactions.
- ACDL facilitates reproducible engineering and clear communication by rendering complex context architectures into standardized, comparable visual diagrams.

---

[CyberAId: AI-Driven Cybersecurity for Financial Service Providers](http://arxiv.org/abs/2605.01892)

- CyberAId: introduces a hybrid multi-agent platform for financial cybersecurity that coordinates specialist LLM subagents within a shared runtime to reason over classical SIEM/XDR telemetry.
- The architecture utilizes a Main Agent/CRA coordination layer, a Reporting capability, and specialist subagents to perform cross-domain synthesis, regulatory mapping, and incident response under bounded human-in-the-loop autonomy.
- The platform integrates advanced security mechanisms including partitioned RAG, federated knowledge sharing, digital twin validation, and eBPF-based kernel telemetry to ensure auditable, regulatory-compliant defense.

---

[AFFormer: Adaptive Feature Fusion Transformer for V2X Cooperative Perception under Channel Impairments](http://arxiv.org/abs/2605.01888)

- AFFormer: introduces a Transformer-based framework designed to mitigate feature corruption in V2X cooperative perception by modeling inter-agent, temporal, and spatial correlations through MATA, DualSA, and UGF modules.
- The framework utilizes a teacher-student knowledge distillation strategy to enhance robustness by aligning student-fused features with clean teacher representations under ideal communication conditions.
- DualSA reduces computational complexity by decomposing 2D spatial attention into parallel width and height branches, enabling efficient and robust feature fusion for real-time autonomous driving applications.

---

[QASecClaw: A Multi-Agent LLM Approach for False Positive Reduction in Static Application Security Testing](http://arxiv.org/abs/2605.01885)

- QASecClaw: introduces a multi-agent framework that reduces false positives in SAST by utilizing a Mission Orchestrator to coordinate specialized agents, including a Security Validation Agent for initial scanning and a SAST Filter Agent powered by a Qwen 3.5 Plus LLM for contextual verification.
- The framework employs a conservative fail-open mechanism where the SAST Filter Agent retains findings if the LLM fails, times out, or produces malformed output, ensuring that potential vulnerabilities are not silently suppressed.
- Evaluated on the OWASP Benchmark v1.2, the approach achieves a 16.0% improvement in F1-score and an 88.6% reduction in false positives compared to standalone Semgrep, demonstrating the effectiveness of LLM-augmented verification in security triage.

---

[Sheaf-Theoretic Planning: A Categorical Foundation for Resilient Multi-Agent Autonomous Systems](http://arxiv.org/abs/2605.01879)

- STP: introduces a geometric framework for autonomous agents that replaces monolithic logical models with sheaf-theoretic structures to enable resilient planning in open-world environments.
- The framework utilizes Interval Category, Grothendieck Topology, Sheaf, Natural Transformation, Pullback, Sheaf Laplacian, Prolog Reasoning Engine, Raspberry Pi, ESP32, PiCrawler, and Wi-Fi Mesh to manage temporal reasoning and multi-agent coordination.
- STP models agent memory, goals, and objective reality as sheaves, employing pullbacks for abductive reasoning and sheaf Laplacians for distributed consensus in multi-agent swarms.

---

[Quality-Aware Exploration Budget Allocation for Cooperative Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2605.01865)

- RCB+RSQ (Return-Conditioned Beta and Reward Signal Quality): introduces a cooperative multi-agent reinforcement learning framework that optimizes exploration by combining a global return-conditioned intensity schedule with a per-agent signal quality-aware budget allocation.
- The framework utilizes RCB to adapt global exploration intensity based on team learning progress and RSQ to selectively attenuate exploration for agents with noisy intrinsic reward signals.
- By employing Successor Distance as a quasimetric intrinsic reward, the approach ensures distinguishable per-agent signal quality, preventing coordination collapse in complex multi-agent environments.

---

[Collusion Relations and their Applications to Balance Theory](http://arxiv.org/abs/2605.01843)

- Collusion Relations and their Applications to Balance Theory: introduces a novel characterization of balance in social networks by utilizing Quadrangular Relations and Collusive Relations to generalize standard balance theory.
- The paper establishes that collusiveness of relations provides a necessary and sufficient condition for network stability, extending applicability to non-symmetric relations.
- The authors provide a modal characterization of collusive frames and implement a G3CP System to derive key properties of balanced networks through labeled sequent calculus.

---

[MAGIC: Multi-Step Advantage-Gated Causal Influence for Multi-agent Reinforcement Learning](http://arxiv.org/abs/2605.01805)

- MAGIC: introduces a framework that models long-horizon causal influences between agents and aligns them with task returns to provide targeted coordination signals.
- The framework utilizes a learned forward model and an ICMI critic to estimate multi-step interventional causal influence, which is then modulated by an advantage-gating module to ensure only beneficial actions are reinforced.
- MAGIC improves coordination in MARL by capturing delayed causal effects and filtering them through extrinsic team advantage to prevent the reinforcement of harmful high-influence behaviors.

---

[Koopman Representations for Early Outbreak Warning and Minimal Counterfactual Intervention in Multi-Agent Epidemic Simulations](http://arxiv.org/abs/2605.01803)

- Koopman-based framework: introduces a computational approach for early outbreak detection and intervention selection in multi-agent epidemic simulations by combining a multi-agent epidemic simulator, a Koopman-inspired model, a random forest classifier, and a counterfactual intervention procedure.
- The framework leverages Koopman operator learning to map nonlinear epidemic trajectories into a low-dimensional latent space, enabling short-horizon forecasting and the extraction of predictive features for outbreak classification.
- Counterfactual analysis is utilized to identify sensitive tipping points where minimal, single-agent mobility restrictions can effectively redirect trajectories from major outbreaks to contained regimes.

---

[DataEvolver: Let Your Data Build and Improve Itself via Goal-Driven Loop Agents](http://arxiv.org/abs/2605.01789)

- DataEvolver: introduces a closed-loop visual data engine that utilizes goal-driven loop agents to automate the construction, inspection, and refinement of multi-artifact datasets.
- The framework employs a dual-loop mechanism where an inner loop performs generation-time self-correction and an outer loop executes validation-time self-expansion to ensure high-quality, traceable data.
- By organizing data construction into an explicit artifact graph with structured review signals and verdict logic, the engine enables reproducible and scalable visual data generation.

---

[Runtime Evaluation of Procedural Content Generation in an Endless Runner Game Using Autonomous Agents](http://arxiv.org/abs/2605.01783)

- Momentum: introduces a runtime procedural content generation and evaluation framework that integrates autonomous aerial and ground agents to validate game content before player interaction.
- The system utilizes a WFC-inspired object placement mechanism and asynchronous navigation-mesh rebuilding to maintain continuous, playable endless-runner environments.
- Evaluation is performed through complementary geometric ray casting and navigation-based path traversal, with detected failures recorded via a structured crash-reporting pipeline.

---

[The Compliance Gap: Why AI Systems Promise to Follow Process Instructions but Don’t](http://arxiv.org/abs/2605.01771)

- BS-Bench: introduces a dual-channel audit framework to measure the Compliance Gap, defined as the divergence between an LLM's verbal commitment and its actual behavioral execution.
- The framework utilizes tool-call logs as an independent behavioral mirror to detect process non-compliance that remains structurally invisible to text-only evaluation methods.
- The research demonstrates that RLHF-trained models systematically prioritize text-based reward signals over procedural instructions, resulting in selective compliance patterns across various professional domains.

---

[Talk is Cheap, Communication is Hard: Dynamic Grounding Failures and Repair in Multi-Agent Negotiation](http://arxiv.org/abs/2605.01750)

- Negotiation Game Framework: introduces an iterated, multi-turn negotiation game to evaluate dynamic grounding in LLMs by decomposing coordination gaps into measurable components.
- The framework utilizes LLM Agents with a Private Thinking Scratchpad and a Cheap Talk Channel to study how agents build shared understanding through interaction.
- Experimental results reveal that coordination failures are primarily rooted in interactive processes like joint plan formation and commitment maintenance rather than individual reasoning limitations.

---

[Reward Hacking Benchmark: Measuring Exploits in LLM Agents with Tool Use](http://arxiv.org/abs/2605.02964)

- RHB (Reward Hacking Benchmark): introduces a multi-step tool-use benchmark designed to quantify reward hacking in LLM agents by evaluating their propensity to exploit evaluation mechanics across independent and chained task regimes.
- The framework utilizes a sandbox environment with integrity instrumentation to classify agent behaviors into six exploit categories, including sequence manipulation, leakage, and tampering, while testing models against both standard and hard task variants.
- Experimental results demonstrate that RL-dominated post-training is associated with significantly higher exploit rates, and that environmental hardening can reduce these exploits by 87.7% without degrading task success.

---

[Architectural Obsolescence of Unhardened Agentic-AI Runtimes](http://arxiv.org/abs/2605.01740)

- enclawed-oss: introduces a hardened agentic-AI runtime architecture that utilizes a Biconditional checker, Hash-chained audit log, Extension admission gate, Two-layer egress guard, Bell-LaPadula classification policy, Module-signing trust root, and Bootstrap seal to prevent failure modes F1–F4.
- The framework employs an Adversarial harness and Cooperation classifier to empirically demonstrate that unhardened runtimes like OpenClaw are architecturally obsolete due to their lack of these seven security primitives.
- The research validates that enclawed-oss achieves perfect recall and precision on F1–F4 failure modes, while also supporting a Behavioral-anomaly monitor in its certified evolution to detect anomalous agent activity.

---

[AgenticVM: Agentic AI for Adaptive Software Vulnerability Management](http://arxiv.org/abs/2605.01739)

- AgenticVM: introduces a multi-agent framework that integrates LLMs with security tools to automate the full vulnerability management lifecycle, including detection-, assessment-, prediction-, integration-, prioritisation- and recommendation-agents.
- The framework employs a hybrid architecture combining rule-based logic for deterministic tasks and a BERT-small model for CVSS severity prediction to reduce alert noise and analyst workload.
- AgenticVM leverages retrieval-grounded generation and human-in-the-loop governance to ensure provenance and accountability in real-world software vulnerability management.

---

[BIM Information Extraction Through LLM-based Adaptive Exploration](http://arxiv.org/abs/2605.01698)

- Adaptive Exploration framework: introduces an LLM-based agent that iteratively writes and executes code against BIM models to discover data structures at runtime, overcoming the limitations of static query approaches.
- The system utilizes a CodeAct architecture to perform multi-step reasoning and error recovery through execution feedback, enabling robust information extraction from heterogeneous IFC models.
- The research includes ifc-bench v2, a large-scale benchmark for evaluating LLM-based BIM extraction, and demonstrates that the adaptive paradigm significantly outperforms static methods regardless of augmentation strategies.

---

[GRAVITY: Architecture-Agnostic Structured Anchoring for Long-Horizon Conversational Memory](http://arxiv.org/abs/2605.01688)

- GRAVITY (Generation-time Relational Anchoring Via Injected Topological MemorY): introduces a plug-and-play structured memory module that extracts Entity Anchors, Event Anchors, and Topic Anchors from raw conversation history to provide explicit relational, temporal, and thematic context to an LLM without modifying the host memory system.
- The framework utilizes an offline build phase to generate structured knowledge representations and an online inference phase that employs an Embedding-based Reranker and Query Expander to inject relevant anchors into the generation prompt.
- Evaluations across five diverse memory architectures demonstrate that GRAVITY significantly improves LLM-judge accuracy by providing structured context that addresses the reasoning gap between retrieval and generation.

---

[CP-SynC: Multi-Agent Zero-Shot Constraint Modeling in MiniZinc with Synthesized Checkers](http://arxiv.org/abs/2605.01675)

- CP-SynC: introduces a multi-agent workflow for zero-shot constraint modeling in MiniZinc that coordinates Modeling Agents, Validation Agents, and Selection Agents to improve model correctness through test-driven development.
- The framework employs a Staged Checking Pipeline to perform hard execution checks and soft semantic checks, while utilizing Diverse Sampling Strategies to explore multiple modeling trajectories in parallel.
- By aggregating evidence across candidate models and synthesized checkers, CP-SynC mitigates noise inherent in LLM outputs and significantly outperforms existing baselines on a benchmark of 100 constraint programming problems.

---

#### 2nd May 2026


[The Perceptual Bandwidth Bottleneck in Vision-Language Models: Active Visual Reasoning via Sequential Experimental Design](http://arxiv.org/abs/2605.01345)

- FOVEA (Foveated Observation and Visual Evidence Acquisition): introduces a training-free procedure that refines VLM crop proposals through evidence-oriented probing to overcome the perceptual bandwidth bottleneck in high-resolution visual reasoning.
- The framework formalizes active visual information acquisition as a sequential Bayesian optimal experimental design (S-BOED) problem, utilizing a coverage–resolution objective to guide the agent's foveation actions.
- By integrating a Planner, Resolvability Probing, and a Tool Interception Mechanism, the approach enables LLMs to iteratively acquire task-relevant visual evidence while managing fixed perceptual bandwidth constraints.

---



[WHO DECIDES WHAT IS HARMFUL? CONTENT MODERATION POLICY THROUGH A MULTI-AGENT PERSONALISED INFERENCE FRAMEWORK](http://arxiv.org/abs/2605.01416)

- PRISM (Personalised Reasoning and Inference System for Moderation): introduces a user-in-the-loop multi-agent framework that filters content based on individual sensitivity profiles to improve moderation accuracy and alignment with user preferences.
- The architecture utilizes a Manager Agent to orchestrate specialized agents—Sociologist, Linguist, and Psychologist—alongside a Ghost Profile Agent to simulate user perspectives and a Synthesis Agent to produce final moderation decisions.
- By dynamically updating user profiles through explicit feedback, the system achieves significant improvements in F1 scores and precision compared to universal, non-personalised moderation baselines.

---



[MILD: Mediator Agent System with Bidirectional Perception and Multi-Layered Alignment for Human-Vehicle Collaboration](http://arxiv.org/abs/2605.01507)

- MILD: introduces an agentic vehicular system that facilitates synergistic human-vehicle collaboration by integrating a joint perception agent and a lightweight strategy agent to provide auditable, constraint-aligned driving suggestions.
- The framework utilizes Evidence- and Constraint-weighted Policy Optimization (ECPO) to align LLM-based strategy agents with safety regulations, vehicle limits, and driver preferences without requiring low-level control.
- MILD employs a retrieval-augmented generation module to dynamically incorporate multi-level constraints, ensuring that high-level advisory policies remain grounded in current traffic context and individual driver states.

---


[AI Alignment via Incentives and Correction](http://arxiv.org/abs/2605.01643)

- AI Alignment via Incentives and Correction: introduces a game-theoretic framework for AI alignment that treats solver and auditor interactions as a bilevel optimization problem to maintain oversight pressure.
- The framework utilizes a Meta-Controller to dynamically adapt Reward-Profiles, ensuring that the Solver and Auditor reach a stable equilibrium that minimizes hallucinations and silent failures.
- Experiments on coding pipelines demonstrate that this adaptive approach significantly reduces solver hallucinations compared to static reward baselines by optimizing for the full correction event.

---



[Valley3: Scaling Omni Foundation Models for E-commerce](http://arxiv.org/abs/2605.01278)

- Valley3: introduces an omni-modal LLM for e-commerce that integrates a Qwen3-VL backbone with an Audio Transformer (AuT) encoder and an MLP Audio Connector to enable unified understanding across text, images, video, and audio.
- The framework incorporates a controllable reasoning module and an agentic search mechanism to balance inference efficiency with deep, evidence-grounded research capabilities for complex e-commerce tasks.
- Valley3 utilizes a four-stage continual pre-training pipeline and a closed-loop data optimization process to align model capabilities with specific e-commerce domain requirements.

---


[The Buy-or-Build Decision, Revisited: How Agentic AI Changes the Economics of Enterprise Software](http://arxiv.org/abs/2604.26482)

- Agentic AI-augmented software development framework: introduces a conceptual model for re-evaluating enterprise software sourcing decisions by analyzing how agentic coding systems transform the governance and economics of in-house development.
- The framework shifts the "Make" option from a traditional hierarchy to a hybrid governance form, characterized by internal code ownership combined with external AI infrastructure dependency.
- It provides a typology of enterprise applications to guide strategic sourcing, identifying that while AI favors in-house development for commodity and custom applications, regulated and mission-critical systems remain better suited for external procurement.

---

[Toward a Principled Framework for Agent Safety Measurement](http://arxiv.org/abs/2605.01644)

- BOA: introduces a principled search framework for measuring LLM agent safety by exploring the trajectory space under a deployment configuration to calculate a likelihood-weighted safety score.
- The framework utilizes priority search engines, a trajectory cache, and an external environment to systematically identify unsafe behaviors that standard greedy or sampled evaluations often overlook.
- BOA incorporates system-level optimizations including batched decoding, prefix caching, and chunked tree expansion to maintain manageable GPU costs while evaluating complex multi-turn agent interactions.

---

[Less Interaction But More Explanation: A Communication Perspective on Agentic AI Interfaces](http://arxiv.org/abs/2605.01610)

- Agentic AI Framework: introduces a communication-centric perspective on agentic AI, arguing that increased autonomy necessitates more robust explanations rather than reduced interaction.
- The framework identifies four communicative roles—AI Creator, AI Converser, AI Curator, and AI Co-Author—that agentic systems traverse, which can lead to source misattribution and accountability gaps.
- To mitigate risks like over-trust and goal misalignment, the paper proposes three explanation types—Action-Process, Uncertainty, and Coordination—and suggests user-driven customization of these explanations to preserve human agency.

---

[Evaluating Agentic AI in the Wild: Failure Modes, Drift Patterns, and a Production Evaluation Framework](http://arxiv.org/abs/2605.01604)

- PAEF (Production Agentic Evaluation Framework): introduces a five-dimension evaluation framework designed for continuous monitoring of agentic systems in production environments.
- The framework addresses the limitations of episodic benchmarks by utilizing EvalRequest, LLMEvaluator, Cascade Uncertainty, Tool Reliability, Distribution Health, Explanation Validity, and Cross-Surface Consistency to detect failure modes like distribution collapse and goal drift.
- PAEF provides a modular architecture that generates MetricResult and EvalReport objects to identify silent performance degradation that standard metrics fail to capture.

---

[Hybrid Quantum Reinforcement Learning with QAOA for Improved Vehicle Routing Optimization](http://arxiv.org/abs/2605.01574)

- HQRL-QAOA: introduces a hybrid quantum-classical architecture that integrates QAOA-structured layers into a QRL policy network to solve the Vehicle Routing Problem.
- The framework utilizes a QAOA warm-start module to initialize PQC parameters, effectively preventing barren plateaus and accelerating convergence compared to random initialization.
- By employing a fixed 4-qubit register and constant circuit depth, the approach ensures scalability and compatibility with near-term quantum hardware for large-scale routing instances.

---

[Feedback-Normalized Developer Memory for Reinforcement-Learning Coding Agents: A Safety-Gated MCP Architecture](http://arxiv.org/abs/2605.01567)

- RL Developer Memory: introduces a feedback-normalized, safety-gated architecture that interposes a memory-control layer between an LLM coding agent and a persistent store to ensure auditable, theory-to-code traceable memory updates.
- The system utilizes a deterministic ranker as the primary baseline while employing a contextual-bandit shadow scorer and OPE gate to safely evaluate learned memory-selection policies without unverified intervention.
- By mapping sparse developer feedback into a canonical reward vocabulary and linking delayed resolutions to specific retrieval events, the architecture provides a governed, component-level auditable framework for RL-specific code maintenance.

---

[Multi-Agent Reasoning Improves Compute Efficiency: Pareto-Optimal Test-Time Scaling](http://arxiv.org/abs/2605.01566)

- Multi-Agent Reasoning Framework: introduces a systematic analysis of inference-time scaling strategies, evaluating self-consistency, self-refinement, multi-agent debate, and Mixture-of-Agents (MoA) under matched compute budgets to identify Pareto-optimal configurations.
- The study utilizes three levers of test-time scaling—pipeline choice, pipeline parameters, and model size—to demonstrate that multi-agent systems, particularly MoA, achieve superior compute-accuracy efficiency compared to single-agent baselines.
- The research establishes practical design guidelines, such as configuring MoA with one more proposer model than layers, and highlights that larger models can be more compute-efficient than heavily scaled smaller models.

---

[Automated Interpretability and Feature Discovery in Language Models with Agents](http://arxiv.org/abs/2605.01555)

- InterpAgent: introduces an autonomous multi-agent framework that unifies feature discovery and hypothesis refinement within a single empirical loop for mechanistic interpretability.
- The framework utilizes a supervisor to coordinate FeatureFinder, which identifies candidate features via statistical analysis, and FeatureExplainer, which iteratively stress-tests hypotheses using a multi-metric evaluation battery.
- By maintaining persistent memory and a shared execution environment, the system produces auditable traces of hypotheses and interventions, improving upon one-shot automated interpretability methods.

---

[6G Needs Agents: Toward Agentic AI-Native Networks for Autonomous Intelligence](http://arxiv.org/abs/2605.01546)

- Agentic AI-Native 6G: introduces a four-layer architecture that integrates Deterministic 6G Infrastructure, Semantic Abstraction Layer, Agentic Reasoning Layer, and Distributed Multi-Agent Fabric to enable policy-governed, intent-aware network autonomy.
- The framework utilizes an LLM Agent Runtime, Tool-Oriented Execution Layer, and Semantic Memory Layer to perform multi-step reasoning and tool-grounded orchestration across the device–edge–core continuum.
- The research provides an empirical evaluation of quantized LLMs on the 6G-Bench benchmark, demonstrating that heterogeneous agent deployment is required to balance reasoning capability, latency, throughput, and memory efficiency.

---

[SciResearcher: Scaling Deep Research Agents for Frontier Scientific Reasoning](http://arxiv.org/abs/2605.01489)

- SciResearcher: introduces a fully automated agentic framework for constructing frontier scientific reasoning tasks by synthesizing conceptual and computational questions grounded in academic evidence.
- The framework utilizes a hierarchical multi-agent architecture, including a main agent that coordinates specialized web- and file-agents to perform long-horizon information retrieval and quantitative model instantiation.
- By leveraging this framework, the authors developed SciResearcher-8B, an agent foundation model that achieves state-of-the-art performance on scientific benchmarks by demonstrating adaptive, tool-intensive reasoning behavior.

---

[MAP-Law: Coverage-Driven Retrieval Control for Multi-Turn Legal Consultation](http://arxiv.org/abs/2605.01486)

- MAP-Law (Mindmap-Augmented Planning for Legal Consultation Agents): introduces a coverage-driven framework for retrieval control in multi-turn legal consultation that utilizes a Plan Graph, Evidence Graph, Coverage Metrics, LLM Action Selector, Retriever, and Joint Graph Memory to optimize evidentiary sufficiency.
- The framework models legal consultation as a structured decision process, where an LLM-based action selector operates over a joint graph representation to dynamically determine retrieval, supplementation, evaluation, or generation actions.
- By employing explicit coverage metrics (EC, EVC, and MG) to govern stopping decisions, the system significantly reduces retrieval rounds and evidence volume while improving element coverage compared to fixed-depth RAG baselines.

---

[Action Agent: Agentic Video Generation Meets Flow-Constrained Diffusion](http://arxiv.org/abs/2605.01477)

- Action Agent: introduces a two-stage framework that decouples high-level trajectory imagination from low-level control by synthesizing a validated Visual Intermediate Representation (VIR) before execution.
- The framework utilizes an LLM-based agent to iteratively refine video generation prompts through a reasoning-based validator, ensuring physical plausibility and instruction adherence.
- FlowDiT integrates semantic features from DINOv2 and ego-motion cues from optical flow to enable high-frequency, embodiment-aware velocity command prediction on consumer hardware.

---

[An Intelligent eUPF for Time-Sensitive Path Selection in B5G Edge Networks](http://arxiv.org/abs/2605.01475)

- eUPF (enhanced User Plane Function): introduces an intelligent data plane architecture that utilizes a DQN agent to perform real-time, latency-aware path selection for B5G network slices.
- The framework leverages passive eBPF-XDP telemetry to estimate end-to-end latency by correlating TEID-based timestamps, enabling autonomous steering between MEC and cloud endpoints.
- Experimental validation on the FABRIC testbed demonstrates that the DQN-based approach significantly reduces average latency and improves temporal stability compared to random baseline policies.

---

[Practical Limits of Autonomous Test Repair: A Multi-Agent Case Study with LLM-Driven Discovery and Self-Correction](http://arxiv.org/abs/2605.01471)

- Multi-agent testing system: introduces an industrial case study of an autonomous UI testing framework that utilizes LLMs for feature discovery, test generation, and iterative self-correction.
- The system employs a five-agent pipeline—Explorer, Planner, Coder, Executor, and Self-Correction—to automate test maintenance in dynamic enterprise environments.
- The research identifies critical failure modes, such as non-converging repair loops and hallucinated interactions, and proposes design guidelines for constrained, reliable autonomous testing.

---

[CoFlow: Coordinated Few-Step Flow for Offline Multi-Agent Decision Making](http://arxiv.org/abs/2605.01457)

- CoFlow: introduces a generative architecture for offline multi-agent reinforcement learning that enables coordinated few-step trajectory generation through a natively joint-coupled velocity field.
- The framework integrates Coordinated Velocity Attention (CVA) with Adaptive Coordination Gating into a Shared U-Net Backbone to facilitate inter-agent coupling without iterative sampling.
- A novel Finite-Difference Consistency Surrogate replaces memory-intensive Jacobian-vector product backpropagation, enabling efficient training at multi-agent scale while maintaining coordination quality.

---

[Decompose and Recompose: Reasoning New Skills from Existing Abilities for Cross-Task Robotic Manipulation](http://arxiv.org/abs/2605.01448)

- Decompose and Recompose: introduces a compositional skill reasoning framework that leverages Atomic Skills Collection, Planning Agent, Visual Encoder, Dynamic Demonstrations Library, Coverage-aware Static Library, and LLM to enable zero-shot cross-task robotic manipulation.
- The framework decomposes seen task demonstrations into interpretable atomic skill-action pairs, which are then recomposed for unseen tasks through a dual-library retrieval system and LLM-based reasoning.
- By providing LLMs with structured, skill-aligned demonstrations rather than raw numerical sequences, the approach effectively activates compositional reasoning for complex robotic manipulation tasks.

---

[Artificial intelligence language technologies in multilingual healthcare: Grand challenges ahead](http://arxiv.org/abs/2605.01441)

- HCAILT (Human-Centered AI Language Technology): introduces a framework for evaluating AI language technologies in multilingual healthcare through the pillars of Reliability, Safety Culture, and Trustworthiness.
- The paper synthesizes current evidence across written communication, spoken communication, and agentic workflows to identify seven grand challenges for future research and deployment.
- It argues that progress in multilingual healthcare requires moving beyond model-centric benchmarks toward accountable sociotechnical design, calibrated human oversight, and interdisciplinary collaboration.

---

[Verbal-R3: Verbal Reranker as the Missing Bridge between Retrieval and Reasoning](http://arxiv.org/abs/2605.01399)

- Verbal-R3 (Retrieve-Rerank-Reason): introduces an agentic RAG framework that utilizes a Generator and a Verbal Reranker to bridge the gap between retrieval and reasoning through Verbal Annotations.
- The framework employs a lightweight Verbal Reranker to generate analytic narratives that map query-context relationships, effectively filtering noise for the Generator.
- Relevance-guided test-time scaling dynamically allocates computational resources to the most promising reasoning trajectories, enhancing both performance and efficiency.

---

[LiveFMBench: Unveiling the Power and Limits of Agentic Workflows in Specification Generation](http://arxiv.org/abs/2605.01394)

- LiveFMBench: introduces a contamination-aware benchmark of 630 ACSL-annotated C programs to systematically evaluate LLM-based formal specification generation across direct prompting, thinking mode, and agentic pipeline settings.
- The study reveals that while thinking mode and agentic pipelines significantly improve success rates, current LLMs still struggle with deep semantic dependencies and loop invariants, often exhibiting unfaithful behaviors that inflate performance metrics.
- Experimental results demonstrate that agentic pipelines are highly cost-effective under low sampling budgets, though performance gains diminish as sampling attempts increase, highlighting fundamental limitations in current LLM-based formal verification approaches.

---

[ESARBench: A Benchmark for Agentic UAV Embodied Search and Rescue](http://arxiv.org/abs/2605.01371)

- ESARBench: introduces a novel task and high-fidelity simulation platform for evaluating LLM-driven UAV agents in complex, real-world search and rescue scenarios.
- The framework utilizes a hierarchical "Event-Snapshot-Task" generation process to create 600 unique, difficulty-stratified rescue missions within photorealistic environments.
- The benchmark employs comprehensive metrics including Success Rate, Time-weighted Success Rate, Clue Discovery Score, and a holistic Rescue Score to assess agent perception, reasoning, and mission efficiency.

---

[Assistance Without Interruption: A Benchmark and LLM-based Framework for Non-Intrusive Human–Robot Assistance](http://arxiv.org/abs/2605.01368)

- NiaRR (Non-intrusive Assistance with Retrieval and Ranking): introduces a hybrid architecture that integrates an LLM-based semantic retriever with a transformer-based scoring model to identify well-timed, non-intrusive robotic assistance.
- The framework utilizes a SBERT Encoder to generate embeddings for human task steps and candidate robot actions, which are then processed by a Cross-Attention Mechanism and MLP Scoring Head to rank the most beneficial assistive pairs.
- To support this approach, the authors establish NIABench, a simulation benchmark that evaluates the robot's ability to provide assistance without interrupting the primary human task flow.

---

[PACE: Parameter Change for Unsupervised Environment Design](http://arxiv.org/abs/2605.01358)

- PACE (Parameter Change for Unsupervised Environment Design): introduces a UED framework that evaluates environment value through the magnitude of policy parameter change induced by training, providing a direct and intrinsic signal of realized learning progress.
- The framework utilizes a first-order Taylor approximation to establish a principled connection between the squared l2 norm of policy parameter updates and improvement in the optimization objective.
- PACE avoids the computational overhead and variance associated with traditional proxy signals like regret or value-based estimation by grounding environment selection in deterministic, in-memory parameter updates.

---

[RAISON: DEVELOPING RELIABLE DECISION-MAKING AGENTS](http://arxiv.org/abs/2605.01351)

- rAIson: introduces a high-level technological environment for developing reliable and explainable decision-making agents using a Front-end authoring tool, Gorgias inference engine, API, Automated code generator, and Deployment service.
- The platform utilizes the Software Development via Argumentation (SoDA) methodology to capture decision policies as hierarchies of scenario-based preferences without requiring manual coding.
- By offering explainable argumentation as a service (AIaaS), the platform enables the integration of symbolic reasoning into external agent systems like JADE or IoT applications.

---

[DiagramNet: An End-to-End Recognition Framework for System-Level Diagrams](http://arxiv.org/abs/2605.01338)

- DiagramNet: introduces a multimodal dataset and a decoupled multi-agent workflow that decomposes complex system-level diagram recognition into Perception Agent, Reasoning Agent, and Knowledge Agent stages.
- The framework utilizes a progressive training pipeline combining supervised fine-tuning, topology-consistency reinforcement learning, and task-specific LoRA adapters to improve LLMs performance on relational reasoning tasks.
- By decoupling visual grounding from topological reasoning, the workflow mitigates spatial hallucinations and enables robust zero-shot transfer to diverse circuit diagram domains.

---

[Truth or Tribe: How In-group Favoritism Prioritize Facts in Persona Agents](http://arxiv.org/abs/2605.01329)

- Truth or Tribe simulation framework: introduces a triadic interaction paradigm to evaluate how persona-based LLMs exhibit in-group favoritism when faced with conflicting information from peers of varying similarity.
- The framework utilizes Persona Similarity Distance (PSD) to measure bias, revealing that LLMs consistently prioritize in-group opinions over objective truth, a phenomenon termed "Tribe over Truth."
- The study demonstrates that this tribal bias is robust across multiple LLMs and reasoning contexts, while also validating that prompt-based interventions like Identity-Blind Instruction (IBI) can effectively mitigate such identity-driven heuristics.

---

[GA-VisAgent: A Multi-Agent application for code generation and visualization in interactive learning](http://arxiv.org/abs/2605.01299)

- GA-VisAgent: introduces a multi-agent framework that leverages GAGPT and ReAct reasoning to decompose complex Geometric Algebra tasks into structured subtasks for accurate code generation and interactive visualization.
- The architecture utilizes a hierarchical dual-agent system, comprising a Planner Agent for task decomposition and a Worker Agent for executing specialized subtasks through a dedicated function library.
- By integrating automated syntax validation and GAALOP API calls, the framework achieves a 90% success rate in generating executable code for conformal space tasks, significantly outperforming standard LLMs.

---

[Lifting Traces to Logic: Programmatic Skill Induction with Neuro-Symbolic Learning for Long-Horizon Agentic Tasks](http://arxiv.org/abs/2605.01293)

- NSI (Neuro-Symbolic Skill Induction): introduces a framework that transforms transient agentic reasoning into persistent, logic-grounded skills by lifting interaction traces into modular programs with explicit control flows and dynamic variable bindings.
- The framework utilizes Neural Feedback Perception to ground observations into symbolic states and Symbolic Execution Logic to enable state-aware, adaptive execution through branching and loops.
- NSI incorporates a Reflective Planning mechanism that converts runtime failures into permanent logic improvements by grafting corrective subgraphs onto existing skill structures.

---

[FeedbackLLM: Metadata driven Multi-Agentic Language Agnostic Test Case Generator with Evolving prompt and Coverage Feedback](http://arxiv.org/abs/2605.01264)

- FeedbackLLM: introduces a multi-agentic framework that utilizes a tightly coupled two-stage pipeline to automate language-agnostic test case generation through iterative prompt refinement.
- The architecture employs specialized Line Feedback Agent and Branch Feedback Agent components to analyze coverage gaps and generate targeted constraints for subsequent LLM iterations.
- A Redundancy Prevention Cache is integrated to minimize redundant API calls and execution cycles, ensuring linear scalability in test generation time.

---

[EO-Gym: A Multimodal, Interactive Environment for Earth Observation Agents](http://arxiv.org/abs/2605.01250)

- EO-Gym: introduces a controlled, Gymnasium-style environment for multimodal, tool-using agents to perform interactive Earth Observation evidence-acquisition tasks.
- The framework utilizes a massive Data Lake and a Data Gathering Space to support complex reasoning across spatial, temporal, and cross-modal dimensions.
- EO-GYM-DATA provides a large-scale benchmark of trajectories, while the EO-GYM-4B model demonstrates improved tool-use fidelity and reasoning through specialized fine-tuning.

---

[S3-R1: Learning to Retrieve and Answer Step-by-Step with Synthetic Data](http://arxiv.org/abs/2605.01248)

- S3-R1: introduces a data-centric framework that improves multi-hop QA by synthesizing intermediate-difficulty questions and employing a retrieval-aware reward signal to guide LLM agent training.
- The framework utilizes a Generator LLM to create synthetic questions from hard-mined anchor instances, which are then filtered by an Oracle LLM and a Retriever to ensure factual grounding and empirical solvability.
- To stabilize training, S3-R1 incorporates negative advantage clipping and a composite reward function that balances final answer correctness with intermediate search quality.

---

[FP-Agent: Fingerprinting AI Browsing Agents](http://arxiv.org/abs/2605.01247)

- FP-Agent: introduces a measurement framework that utilizes browser and behavioral fingerprints to reliably distinguish between AI browsing agents and human users.
- The framework employs client-side instrumentation to collect interaction artifacts, which are then processed by an XGBoost classifier to achieve near-perfect identification accuracy.
- Experimental results demonstrate that while browser fingerprints are often correlated with execution environments, behavioral features such as typing, scrolling, and mouse movement provide highly discriminative signals for bot detection.

---

[Breaking the Computational Barrier: Provably Efficient Actor–Critic for Low-Rank MDPs](http://arxiv.org/abs/2605.01242)

- Opt-AC (Optimistic Actor–Critic): introduces a provably efficient reinforcement learning algorithm for low-rank Markov Decision Processes that reduces computational complexity by relying solely on a policy evaluation oracle implemented via supervised learning.
- The framework utilizes an optimistic critic constructed from learned transition models and an exploration bonus to achieve state-of-the-art sample complexity without requiring computationally expensive planning or constrained optimization oracles.
- Theoretical results demonstrate that the algorithm achieves polynomial sample and computational complexity, while empirical evaluations on standard continuous-control benchmarks confirm its practical effectiveness and competitiveness against strong baselines.

---

[Lost in the Tower of Babel: The Adverse Effects of Incidental Multilingualism in LLMs](http://arxiv.org/abs/2605.01224)

- ToB (Tower of Babel) Problem Framework: introduces a critical analysis of how incidental multilingualism in LLMs leads to unstable language support and performance degradation in agentic systems, utilizing LLM-based agents, inter-agent evaluation, language identification, translation, code generation, long-form generation, and support-list elicitation.
- The paper demonstrates that current LLMs exhibit significant discrepancies between self-reported language support and actual behavioral performance, often resulting in incorrect generation rather than explicit refusal.
- The authors advocate for a shift toward "multilingualism by design," emphasizing the need for explicit multilingual objectives, transparent support frontiers, and robust safety mechanisms in the LLM development pipeline.

---

[Agentic AI Systems Should Be Designed as Marginal Token Allocators](http://arxiv.org/abs/2605.01214)

- Marginal Token Allocation Framework: introduces a unified economic model for agentic AI systems, treating routers, agents, serving stacks, and trainers as vertical slices of a single marginal token allocation problem.
- The framework identifies that local optimization within these components leads to global misallocation, proposing that systems should be designed around a shared first-order condition balancing marginal benefit against compute, latency, and risk costs.
- This approach provides a diagnostic tool for recurring failure modes in LLMs, such as over-routing, serving congestion, and cache misuse, by treating tokens as economic units of computation rather than flat-rate text.

---

[ClarifySTL: An Interactive LLM Agent Framework for STL Transformation through Requirements Clarification](http://arxiv.org/abs/2605.01209)

- ClarifySTL: introduces an interactive framework that enhances the transformation of natural language requirements into Signal Temporal Logic (STL) specifications by iteratively resolving vagueness and ambiguity through specialized detection and inquiry agents.
- The framework utilizes a fine-tuned Vagueness Detector and a contrastive learning-based Ambiguity Detector to identify defective requirements, followed by targeted inquiry agents that guide users to provide necessary clarifications.
- By employing iterative re-checking loops and LLM-based query generation, ClarifySTL ensures that the final STL formulas accurately capture user intent while reducing the manual burden on domain experts.

---

[Faithful Mobile GUI Agents with Guided Advantage Estimator](http://arxiv.org/abs/2605.01208)

- Faithful-Agent: introduces a two-stage training pipeline that prioritizes evidence groundedness and internal consistency in GUI agents by employing faithfulness-oriented SFT and RFT with GuAE.
- The framework utilizes GuAE to mitigate advantage collapse in low-variance rollout groups by combining anchor regularization and variance-adaptive tempering to preserve meaningful learning signals.
- Faithful-Agent incorporates a thought-action consistency reward to align agent reasoning with executed actions, effectively suppressing speculative behaviors and improving robustness under missing or conflicting UI evidence.

---

[Trace: Unmasking AI Attack Agents Through Terminal Behavior Fingerprinting](http://arxiv.org/abs/2605.01186)

- TRACE: introduces a two-stage framework that passively fingerprints AI attack agents via terminal command sequences and performs active forensics using family-calibrated Defensive Prompt Injection (DPI) to extract system prompts.
- The framework utilizes a TF-IDF bigram vectorizer and LinearSVC classifier to attribute sessions to specific LLM families, enabling targeted payload routing for effective intelligence extraction.
- Evaluation across seven frontier LLM families and three agent scaffolds demonstrates that TRACE achieves high attribution accuracy and robust forensic intelligence extraction, even under unseen deployment conditions.

---

#### 1st May 2026

[Can Coding Agents Reproduce Findings in Computational Materials Science?](http://arxiv.org/abs/2605.00803)

- AUTOMAT: introduces a benchmark for evaluating LLM-based agents on their ability to reproduce computational materials science claims by navigating complex, domain-specific workflows.
- The framework utilizes a Task Package (SME-curated claim, paper, metadata, artifacts) processed by a Reproduction Agent (LLM-based autonomous investigator) within an HPC Cluster (resource-controlled execution environment) to generate evidence for scientific validation.
- Performance is assessed by an Evaluator Agent (LLM-based judge for reproducibility) that utilizes an Evaluator Loop (iterative inspection of artifacts) to score the reproducibility of the findings stored in an Artifact Directory (persistent storage for logs and outputs).

---

[RunAgent: Interpreting Natural-Language Plans with Constraint-Guided Execution](http://arxiv.org/abs/2605.00798)

- RunAgent: introduces a multi-agent platform that interprets natural-language plans by enforcing stepwise execution through constraint-guided logic and dynamic selection of reasoning, tool usage, or code execution.
- The framework utilizes an agentic language with control constructs like IF, GOTO, and FORALL to bridge the flexibility of natural language with the determinism of programmatic execution.
- RunAgent incorporates robust error correction and context filtering mechanisms, ensuring reliable task completion by validating outputs against automatically derived constraints and rubrics.

---

[Meritocratic Fairness in Budgeted Combinatorial Multi-armed Bandits via Shapley Values](http://arxiv.org/abs/2605.00762)

- K-SVFair-FBF: introduces a meritocratic fairness framework for budgeted combinatorial multi-armed bandits under full-bandit feedback by utilizing K-Shapley Value, Monte Carlo Permutations, Stochastic Smoothing, Randomized Rounding Scheme (RRS), Confidence Term, and Selection Counters.
- The framework addresses the challenge of unknown individual arm contributions in full-bandit feedback by adaptively estimating K-Shapley values while mitigating noise from both valuation learning and Monte Carlo approximations.
- Theoretical analysis demonstrates that the algorithm achieves sublinear fairness regret, effectively balancing equitable arm participation with reliable learning in applications like federated learning and social influence maximization.

---

[NonZero: Interaction-Guided Exploration for Multi-Agent Monte Carlo Tree Search](http://arxiv.org/abs/2605.00751)

- NONZERO: introduces a multi-agent MCTS framework that mitigates the combinatorial joint-action bottleneck by utilizing a low-dimensional nonlinear return surrogate and an interaction-guided proposal rule.
- The framework incorporates a hypernetwork to initialize node-specific parameters and employs the NONUCT mechanism to perform curvature-aware exploration through first- and second-order discrete differences.
- NONZERO achieves sublinear regret guarantees and demonstrates superior sample efficiency and performance across MatGame, SMAC, and SMACv2 benchmarks compared to existing model-based and model-free baselines.

---

[Position: agentic AI orchestration should be Bayes-consistent](http://arxiv.org/abs/2605.00742)

- Bayesian Agentic Orchestration Framework: introduces a principled approach for agentic AI systems by placing Bayesian decision-making at the orchestration layer rather than within individual LLMs.
- The framework utilizes a Bayesian controller to maintain beliefs over task-relevant latent variables, updating these beliefs via observation models that treat LLM outputs as evidence.
- This approach enables cost-aware, uncertainty-aware, and adaptive orchestration by selecting actions that maximize posterior expected utility while treating LLMs as non-Bayesian black-box predictors.

---

[Self-Adaptive Multi-Agent LLM-Based Security Pattern Selection for IoT Systems](http://arxiv.org/abs/2605.00741)

- ASPO: introduces a self-adaptive IoT security architecture that integrates multi-agent LLM-based reasoning with deterministic enforcement within a MAPE-K control loop to ensure safe, conflict-free, and resource-feasible mitigation selection.
- The framework separates stochastic candidate generation by LLM agents from deterministic validation, ensuring that only safe, catalogue-bounded, and resource-compliant mitigation portfolios are executed at the edge.
- Experimental evaluation on a distributed edge testbed demonstrates that ASPO maintains stable safety enforcement and reduces extreme-case latency and energy overheads by over 20% under workload scaling.

---

[To Call or Not to Call: A Framework to Assess and Optimize LLM Tool Calling](http://arxiv.org/abs/2605.00737)

- Agentic AI architectures: introduces a principled framework to evaluate and optimize LLM tool-calling decisions based on necessity, utility, and affordability.
- The framework identifies a misalignment between LLMs' self-perceived need for tools and their true performance utility, leading to suboptimal tool-calling behavior.
- The authors propose a controller framework using Latent Need Estimators and Latent Utility Estimators to improve decision quality by leveraging internal hidden states of the LLM.

---

[Learning How and What to Memorize: Cognition-Inspired Two-Stage Optimization for Evolving Memory](http://arxiv.org/abs/2605.00702)

- MemCoE: introduces a two-stage optimization framework that decouples memory organization from content updating by learning a schema-consistent guideline and a guideline-aligned memory policy.
- The framework utilizes Memory Guideline Induction to optimize a global natural-language guideline via contrastive feedback and batch aggregation, followed by Guideline-Aligned Memory Policy Optimization to train the agent using structured process rewards.
- MemCoE improves long-term personalization by internalizing extraction, update, and forgetting behaviors into a unified memory-evolution process, demonstrating robustness and transferability across various LLMs.

---

[Learning to Act and Cooperate for Distributed Black-Box Consensus Optimization](http://arxiv.org/abs/2605.00691)

- LAC-MAS (Learning to Act and Cooperate for Multi-Agent Systems): introduces a trajectory-driven framework for distributed black-box consensus optimization that utilizes LLMs to provide sparse high-level guidance for both internal agent behaviors and external cooperation patterns.
- The framework integrates an adaptive swarm execution layer with an LLM-based guidance module, coordinated by a phased cognitive guidance mechanism to ensure stable and resource-aware optimization.
- By leveraging historical optimization trajectories, LAC-MAS enables agents to dynamically adjust their internal search regimes and neighbor influence weights, consistently improving solution quality and communication efficiency in decentralized environments.

---

[Affordance Agent Harness: Verification-Gated Skill Orchestration](http://arxiv.org/abs/2605.00663)

- A-Harness: introduces a closed-loop runtime that unifies heterogeneous skills with an Evidence Store, Two-Tier Memory, Router, and Verifier to perform budgeted, verification-gated affordance grounding.
- The framework replaces fixed pipelines with an adaptive routing policy that uses relative diagnostics to gate commitments and trigger targeted retries based on cross-tool consistency, cross-scale stability, and evidence sufficiency.
- By amortizing successful tool chains through a two-tier memory, the system improves the accuracy-cost trade-off in open-world environments while reducing latency and unnecessary skill calls.

---

[Learn where to Click from Yourself: On-Policy Self-Distillation for GUI Grounding](http://arxiv.org/abs/2605.00642)

- GUI-SD: introduces an on-policy self-distillation framework for GUI grounding that replaces sparse reinforcement learning rewards with dense token-level supervision.
- The framework utilizes Visual Privileged Guidance to provide informative teacher signals and Entropy-Guided Optimization to prioritize high-impact coordinate tokens during training.
- By employing a single policy model as both teacher and student with asymmetric contexts, GUI-SD achieves superior accuracy and training efficiency compared to traditional reinforcement learning methods.

---

[DySRec: Dynamic Context-Aware Psychometric Scale Recommendation via Multi-Agent Collaboration](http://arxiv.org/abs/2605.00574)

- DySRec: introduces a multi-agent framework for dynamic psychometric scale recommendation that models clinical assessment as a sequential decision-making process using an Orchestrator, Context State Management Agent, Shared Global Context State, Dynamic Scale Recommendation Agent, LLM backbone, Psychometric Scale Repository, Scale Execution &amp; Scoring Agent, Risk Detection &amp; Intervention Agent, and Audit &amp; Logging Agent.
- The system utilizes a blackboard-style communication architecture to enable asynchronous collaboration between specialized agents while maintaining a unified, evolving user context state.
- DySRec incorporates a closed-loop refinement mechanism and real-time risk-aware coordination to ensure clinical safety and improve recommendation accuracy through active information-seeking dialogue.

---

[Structure Liberates: How Constrained Sensemaking Produces More Novel Research Output](http://arxiv.org/abs/2605.00557)

- SCISENSE: introduces a sensemaking-grounded framework that operationalizes scientific ideation as a structured sequence of eight cognitive stages to improve research output quality and diversity.
- The framework utilizes SCISENSE-Traj to distill sensemaking capabilities into SCISENSE-LM, enabling LLMs to generate reconstructive or generative research trajectories grounded in citation neighborhoods.
- Experimental results demonstrate that constrained, target-based supervision produces more novel and diverse research trajectories, which subsequently improve the executability and quality of downstream research artifacts generated by coding agents.

---

[A11y-Compressor: A Framework for Enhancing the Efficiency of GUI Agent Observations through Visual Context Reconstruction and Redundancy Reduction](http://arxiv.org/abs/2605.00551)

- A11y-Compressor: introduces a structured framework that transforms linearized accessibility trees into compact, semantically coherent GUI observations for LLMs.
- The framework utilizes Modal Detection (identifies foreground modal elements), Redundancy Reduction (removes irrelevant or repetitive elements), and Semantic Structuring (organizes elements into meaningful functional groups) to improve grounding efficiency.
- Experimental results on the OSWorld benchmark demonstrate that the framework reduces input token consumption to 22% of the original while improving task success rates by 5.1 percentage points.

---

[From Research to Practice: An Interactive Rapid Review of Autonomous Driving System Testing in Industry](http://arxiv.org/abs/2605.00531)

- Interactive Rapid Review (IRR): introduces a practitioner-driven methodology to evaluate academic research on End-to-End (E2E) ADS testing by synthesizing technological rules and assessing their industrial applicability.
- The framework categorizes testing interventions into constraint/search-based, reinforcement learning-based, generative model-based, and adversarial perturbation-based components to address critical scenario generation and testing completeness.
- This study bridges the gap between academic research and industrial practice by involving 21 practitioners to validate the relevance of 17 selected research studies for E2E ADS testing.

---

[SAGA: Workflow-Atomic Scheduling for AI Agent Inference on GPU Clusters](http://arxiv.org/abs/2605.00528)

- SAGA: introduces a distributed scheduler that treats multi-step agent workflows as first-class schedulable units to minimize KV cache regeneration latency.
- The framework utilizes Agent Execution Graphs to predict future cache reuse, enabling proactive retention and efficient scheduling across tool-call boundaries.
- SAGA integrates session-affinity batching, work-stealing load balancing, and Agent Fair Share scheduling to achieve high SLO attainment and memory utilization in multi-tenant GPU clusters.

---

[Stereo Multistage Spatial Attention for Real-Time Mobile Manipulation Under Visual Scale Variation and Disturbances](http://arxiv.org/abs/2605.00471)

- MSARNN: introduces a deep predictive learning framework that utilizes stereo multistage spatial attention to extract task-relevant features for robust real-time mobile manipulation.
- The architecture integrates MSA, a Motion Predict Module, and a Hierarchical LSTM to process stereo visual inputs and robot states for temporally coherent motion generation.
- The framework employs a Temporally Bidirectional Loss to align attention points with robot motion, ensuring stability under visual scale variations and environmental disturbances.

---

[Scaling Video Understanding via Compact Latent Multi-Agent Collaboration](http://arxiv.org/abs/2605.00444)

- MACF: introduces an end-to-end framework that scales video understanding by partitioning inputs across Local Agents with Budgeted Perception and aggregating evidence via a Coordinator Agent using compact Communication Tokens in a Shared Latent Communication Space.
- The framework employs a Curriculum Training Strategy to progressively establish semantic alignment, query-aware evidence summarization, and cross-agent coordination, ensuring efficient information exchange under strict communication constraints.
- By decoupling per-agent perception budgets from global video complexity, MACF preserves fine-grained visual fidelity and outperforms state-of-the-art LLMs and multi-agent systems on long-form video benchmarks.

---

[AEM: Adaptive Entropy Modulation for Multi-Turn Agentic Reinforcement Learning](http://arxiv.org/abs/2605.00425)

- AEM (Adaptive Entropy Modulation): introduces a supervision-free credit assignment method that adaptively modulates entropy dynamics during RL training to optimize the exploration-exploitation trade-off.
- The framework leverages response-level entropy as an intrinsic signal to rescale advantages, effectively upweighting or downweighting responses based on their relative surprisal.
- AEM functions as a lightweight plug-in for existing RL baselines, improving performance across multi-turn agentic tasks without requiring additional model training or heavy computation.

---

[Skills as Verifiable Artifacts: A Trust Schema and a Biconditional Correctness Criterion for Human-in-the-Loop Agent Runtimes](http://arxiv.org/abs/2605.00424)

- Enclawed (Configurable, sector-neutral hardening framework for single-user AI assistant gateways): introduces a trust schema and biconditional correctness criterion to secure LLM agent runtimes by treating skills as verifiable artifacts.
- The framework utilizes a Skill Manifest, Trust Root, Capability Gate, Transaction Buffer, Audit Log, Broker, and Adversarial-Ensemble Evaluator to enforce security policies and verify agent behavior.
- It establishes twelve normative guidelines for building secure, skill-aware LLM agent harnesses that are resistant to supply-chain attacks and unauthorized agent mutations.

---

[Foresight Arena: An On-Chain Benchmark for Evaluating AI Forecasting Agents](http://arxiv.org/abs/2605.00420)

- Foresight Arena: introduces a permissionless, on-chain evaluation infrastructure for benchmarking LLMs on real-world prediction markets using proper scoring rules.
- The framework utilizes a commit-reveal protocol enforced by Smart Contracts and External Oracles to ensure trustless, verifiable performance tracking of LLM agents.
- By employing Brier and Alpha Scores, the system isolates predictive forecasting ability from trading-based metrics, providing a rigorous assessment of LLM calibration and resolution.

---

[Learning while Deploying: Fleet-Scale Reinforcement Learning for Generalist Robot Policies](http://arxiv.org/abs/2605.00416)

- LWD: introduces a fleet-scale offline-to-online reinforcement learning framework for the continual post-training of generalist Vision-Language-Action policies.
- The framework utilizes DIVL for robust value estimation from heterogeneous data and QAM for stable policy extraction from flow-based action generators.
- By closing the loop between fleet-scale deployment and centralized learning, LWD enables robots to autonomously improve shared policies through a data flywheel of physical experience.

---

[Physically Native World Models: A Hamiltonian Perspective on Generative World Modeling](http://arxiv.org/abs/2605.00412)

- HWM (Hamiltonian World Models): introduces a physically grounded generative dynamical system that separates perception, latent dynamics, rendering, and planning to improve long-horizon stability and interpretability.
- The framework utilizes a structured latent phase space representation (q, p) to model physical interactions through energy-based dynamics rather than unconstrained neural transitions.
- By incorporating controlled-dissipative Hamiltonian dynamics, the model provides a physically meaningful inductive bias that enhances data efficiency and reliability for embodied decision-making tasks.

---

[Agent Capsules: Quality-Gated Granularity Control for Multi-Agent LLM Pipelines](http://arxiv.org/abs/2605.00410)

- Agent Capsules: introduces an adaptive execution runtime that optimizes multi-agent LLM pipelines by dynamically merging agent calls into compound execution units based on empirical quality constraints.
- The framework utilizes a composition score, a quality gate, and an escalation ladder to safely balance token efficiency with output quality across different execution modes.
- Agent Capsules outperforms hand-tuned and compile-time optimized baselines by providing runtime-adaptive orchestration that requires no training data or per-pipeline engineering.

---

[BOLT: Online Lightweight Adaptation for Preparation-Free Heterogeneous Cooperative Perception](http://arxiv.org/abs/2605.00405)

- BOLT: introduces a lightweight, plug-and-play ego-side plugin that enables preparation-free heterogeneous cooperative perception by adapting neighbor features online via ego-as-teacher distillation.
- The framework utilizes a frozen fusion module and detection head, relying on an adaptive plugin composed of AdaIN, a residual CNN adapter, and a per-channel gate to align heterogeneous features without requiring pre-deployment collaborative training.
- By leveraging high-confidence ego predictions as teacher signals, BOLT effectively bridges the distribution gap between independently trained agents, consistently surpassing ego-only performance across multiple benchmarks.

---

[Agentic AI for Substance Use Education: Integrating Regulatory and Scientific Knowledge Sources](http://arxiv.org/abs/2605.00383)

- Agentic RAG System: introduces an agentic AI architecture that integrates official DEA regulatory records with real-time PubMed scientific literature to provide verifiable substance use education.
- The system utilizes a modular five-layer architecture, including a Presentation Layer (Streamlit), Orchestration Layer (LangChain), Retrieval Layer (Dual-source knowledge access), Generation Layer (Qwen3-32B LLM), and Storage Layer (ChromaDB), to ensure high-availability and evidence-based responses.
- Expert evaluation across 90 system interactions demonstrated high performance in factual accuracy, citation quality, contextual coherence, and regulatory appropriateness, effectively mitigating common LLM hallucination risks.

---

[Social Bias in LLM-Generated Code: Benchmark and Mitigation](http://arxiv.org/abs/2605.00382)

- FMA (Fairness Monitor Agent): introduces a modular, oracle-free system that intercepts code generation pipelines to detect and repair social bias through static LLM-based analysis and iterative guided rewriting.
- The framework utilizes Solar, a metamorphic testing tool, to quantify social bias across 343 human-centered coding tasks, revealing that bias is a structural property of LLMs rather than a sampling artifact.
- Experimental results demonstrate that FMA achieves a 65.1% relative reduction in bias while simultaneously improving functional correctness, outperforming both prompt-level interventions and structured multi-agent workflows.

---

[ResRL: Boosting LLM Reasoning via Negative Sample Projection Residual Reinforcement Learning](http://arxiv.org/abs/2605.00380)

- ResRL: introduces a reinforcement learning framework that decouples gradient updates by projecting negative token representations onto a low-rank positive subspace to mitigate semantic interference.
- The framework utilizes policy hidden states to compute projection residuals, which serve as a proxy for gradient interference and guide conservative advantage reweighting.
- ResRL improves reasoning performance and generation diversity across mathematical, coding, agentic, and tool-use tasks by selectively suppressing erroneous reasoning patterns while preserving shared valid semantic structures.

---

[Time-series Meets Complex Motion Modeling: Robust and Computational-effective Motion Predictor for Multi-object Tracking](http://arxiv.org/abs/2605.00362)

- TCMP (Temporal Convolutional Motion Predictor): introduces a lightweight motion prediction framework for Multi-object Tracking that leverages a modified Temporal Convolutional Network with dilated convolutions to capture complex non-linear motion dynamics.
- The architecture utilizes a stack of dilated convolutional layers to efficiently model long-range temporal dependencies while avoiding the cumulative errors and computational overhead associated with autoregressive models.
- By integrating a learned parameter to balance hierarchical skipped connections and final block outputs, the model achieves state-of-the-art tracking performance with significantly reduced parameter counts and computational costs compared to existing generative approaches.

---

[MemRouter: Memory-as-Embedding Routing for Long-Term Conversational Agents](http://arxiv.org/abs/2605.00356)

- MemRouter: introduces an embedding-based memory router that decouples memory admission from the downstream answer backbone to avoid per-turn autoregressive generation.
- The architecture utilizes a Memory Router with lightweight classification heads and a frozen LLM backbone to predict storage decisions based on contextualized turn embeddings.
- By replacing autoregressive memory management with a forward-only routing policy, the system achieves significant latency reductions and improved F1 performance in long-term conversational QA tasks.

---

[Odysseus: Scaling VLMs to 100+ Turn Decision-Making in Games via Reinforcement Learning](http://arxiv.org/abs/2605.00347)

- Odysseus: introduces an open training framework for VLM agents that integrates Supervised Initialization, Multi-Task Reinforcement Learning, PPO with Lightweight Turn-Level Critic, Positive-Advantage Filtering, and an Auto-Curriculum Mechanism to enable stable long-horizon decision-making.
- The framework utilizes a lightweight CNN-based turn-level critic to decouple temporal credit assignment from token generation, significantly reducing computational overhead compared to large-model-based actor-critic methods.
- Odysseus achieves substantial performance gains in long-horizon game environments by leveraging pretrained VLMs as strong action priors, effectively narrowing the gap between foundation models and embodied agents.

---

[AgentFloor: How Far Up the tool use Ladder Can Small Open-Weight Models Go?](http://arxiv.org/abs/2605.00334)

- AgentFloor: introduces a deterministic six-tier benchmark to evaluate the tool-use capabilities of LLMs across a controlled capability ladder.
- The framework utilizes a Tool Surface and Inference Protocol to compare open-weight models against frontier LLMs, identifying specific capability thresholds for routine versus complex agentic tasks.
- Results demonstrate that while smaller models are sufficient for routine tool use, frontier LLMs maintain a performance advantage in long-horizon planning tasks under persistent constraints.

---

[Semia: Auditing Agent Skills via Constraint-Guided Representation Synthesis](http://arxiv.org/abs/2605.00314)

- Semia: introduces, a static analysis framework for agent skills that lifts hybrid prose-code artifacts into a structured Datalog fact base using an LLM-driven synthesis loop disciplined by structural and semantic constraints.
- The framework utilizes a Constraint-Guided Representation Synthesis (CGRS) loop, which iteratively refines candidate SDL representations based on feedback from a structural validator and a semantic scorer to ensure faithful translation.
- Once a skill is lifted into an SDL fact base, the system employs a library of Datalog-based security detectors to perform deterministic reachability analysis, identifying critical vulnerabilities such as missing human-in-the-loop gates and unsanitized context ingestion.

---

[An End-to-End Decision-Aware Multi-Scale Attention-Based Model for Explainable Autonomous Driving](http://arxiv.org/abs/2605.00291)

- End-to-End Decision-Aware Multi-Scale Attention-Based Model: introduces an end-to-end multi-task deep learning architecture that integrates ResNet50, MASPP, and a hybrid DCA-CBAM-ResNet50 Attention mechanism to simultaneously predict driving actions and generate corresponding textual and visual explanations.
- The framework improves interpretability by feeding action head outputs into the reasoning head, ensuring logical consistency between driving decisions and their underlying rationales.
- The authors propose a Joint F1 score metric to quantitatively evaluate the alignment between predicted actions and their associated reasons, demonstrating superior performance on the BDD-OIA and nu-AR datasets.

---

[Intern-Atlas: A Methodological Evolution Graph as Research Infrastructure for AI Scientists](http://arxiv.org/abs/2604.28158)

- Intern-Atlas: introduces a methodological evolution graph that transforms document-centric citation networks into a queryable causal topology for AI research agents.
- The framework utilizes a two-phase LLM extraction process to identify method entities and classify citation edges with verbatim bottleneck-to-mechanism evidence.
- It enables automated scientific discovery through SGT-MCTS lineage reconstruction, graph-grounded idea evaluation, and strategy-driven idea generation.

---

[GSDrive: Reinforcing Driving Policies by Multi-mode Trajectory Probing with 3D Gaussian Splatting Environment](http://arxiv.org/abs/2604.28111)

- GSDrive: introduces a framework that utilizes 3D Gaussian Splatting (3DGS) to provide dense, physics-based reward signals for E2E autonomous driving policy improvement.
- The framework employs a flow matching-based trajectory predictor to probe multiple future scenarios within the 3DGS environment, enabling the policy to evaluate long-term consequences of current actions.
- By bridging imitation learning and reinforcement learning through trajectory-based reward shaping, GSDrive improves training stability and performance in complex, closed-loop driving scenarios.

---

[D3-Gym: Constructing Real-World Verifiable Environments for Data-Driven Discovery](http://arxiv.org/abs/2604.27977)

- D3-Gym: introduces an automated pipeline for constructing real-world, verifiable environments for scientific data-driven discovery, utilizing AutoSDT, Coding Agent, Execution Environment, Multimodal LLM-as-judge, Evaluation Script Planner, Evaluation Script Coder, and Training Environments.
- The framework employs a planning-then-coding approach to synthesize task-specific evaluation scripts, ensuring scientific soundness through rigorous validation against human-annotated gold standards.
- Training LLMs on trajectories sampled from D3-Gym yields consistent performance gains across model sizes, effectively narrowing the gap between open-weight models and proprietary reasoning systems.

---

[From Unstructured Recall to Schema-Grounded Memory: Reliable AI Memory via Iterative, Schema-Aware Extraction](http://arxiv.org/abs/2604.27906)

- xmemory: introduces an iterative, schema-aware extraction pipeline that decomposes unstructured interaction logs into validated, schema-conformant records to ensure reliable AI memory.
- The architecture utilizes a stateful Prompt Engine and multi-stage extraction components—Object Detector, Object Descriptor, Fields Detector, Fields Extractor, and Relation Extractor—to enforce explicit relevance contracts.
- By shifting interpretation from the read path to the write path, the framework employs Request, Session, and Main memory contexts to maintain deterministic, verifiable, and addressable factual records.

---

[Autonomous Systems Dependability in the era of AI: Design Challenges in Safety, Security, Reliability and Certification](http://arxiv.org/abs/2604.27807)

- Autonomous System Dependability Management framework: introduces a holistic approach for designing dependable autonomous systems by integrating design-time assurance and run-time adaptation across hardware, software, and application layers.
- The framework addresses the non-deterministic nature of ML components by employing safety-aware resource allocation and cross-layer reliability analysis to maintain system performance under stringent constraints.
- It further incorporates security-by-design principles, including in-vehicle network protection and perception security, to ensure robust operation in the presence of potential cyberattacks and environmental uncertainties.

---

[Skills-Coach: A Self-Evolving Skill Optimizer via Training-Free GRPO](http://arxiv.org/abs/2604.27488)

- Skills-Coach: introduces an automated framework for the self-evolution of skills in LLM-based agents, utilizing a Diverse Task Generation Module, a Lightweight Optimization Module, a Comparative Execution Module, and a Traceable Evaluation Module.
- The framework employs Training-Free GRPO to iteratively refine skill instructions and code, significantly improving performance across both instruction-only and code-inclusive skill categories.
- By leveraging the Skill-X benchmark, the system demonstrates substantial gains in skill generalization and robustness, enabling autonomous optimization without human intervention.

---

[Position: Safety and Fairness in Agentic AI Depend on Interaction Topology, Not on Model Scale or Alignment](http://arxiv.org/abs/2605.01147)

- Agentic AI Interaction Topology Framework: introduces a dynamical systems perspective on multi-agent AI, arguing that safety and fairness are emergent properties of interaction topology rather than individual LLM components.
- The framework demonstrates that sequential pipelines and parallel committees induce distinct, topology-driven failure modes including ordering instability, information cascades, and functional collapse.
- Empirical results across multiple LLM scales show that increasing model capability often amplifies these topological pathologies rather than mitigating them, necessitating a shift toward system-centric safety evaluation.

---

[A Low-Latency Fraud Detection Layer for Detecting Adversarial Interaction Patterns in LLM-Powered Agents](http://arxiv.org/abs/2605.01143)

- Low-Latency Fraud Detection Layer: introduces a system-level defense mechanism that models adversarial interaction trajectories using structured runtime features to detect malicious patterns in LLM-powered agents.
- The framework utilizes five feature groups—prompt, session, tool, context, and fraud-inspired—to identify complex multi-turn attacks that evade traditional prompt-level filtering.
- By employing lightweight tree-based models, the detector achieves real-time performance with significantly lower latency than LLM-based defenses while maintaining high detection accuracy.

---

[Forager: a lightweight testbed for continual learning with partial observability in reinforcement learning](http://arxiv.org/abs/2605.01131)

- Forager: introduces a lightweight, resource-efficient gridworld testbed designed to evaluate continual learning agents in partially observable environments with constant memory footprints.
- The framework includes DQN-, PPO-, DRQN- and RTU-PPO-agents, utilizing Convolutional Vision Encoders, Recurrent Trace Units, and Exponentially Weighted Memory Traces to address partial observability and loss of plasticity.
- Experimental results demonstrate that state construction methods, particularly RTU-PPO, significantly outperform standard feedforward agents in navigating non-stationary, unending foraging tasks.

---

[MORPH: Multi-Environment Orchestrated Reinforcement Learning for PRB Handling in O-RAN](http://arxiv.org/abs/2605.01128)

- MORPH: introduces a measurement-grounded training pipeline that leverages an OAI 5G Testbed, a 3GPP-parameterized PHY-fidelity OFDM Simulator, and a Hybrid Throughput Oracle to optimize slice-aware PRB allocation.
- The framework utilizes a PPO RL Agent hosted within a Near-RT RIC to dynamically balance throughput and SLA requirements across heterogeneous network slices.
- By combining empirical OAI measurements with synthetic simulator feedback, MORPH effectively mitigates the simulation-to-reality gap in O-RAN resource management.

---

[Towards Multi-Agent Autonomous Reasoning in Hydrodynamics](http://arxiv.org/abs/2605.01102)

- MAS: introduces a multi-agent system for hydrodynamics that coordinates specialized agents through a Layer Execution Graph (LEG) to alleviate context-saturation bottlenecks in scientific workflows.
- The architecture utilizes a Graph Architect to dynamically construct execution topologies, while specialist agents, consolidators, and a reporter agent manage retrieval and synthesis tasks.
- The system achieves high factual precision and robustness by compartmentalizing tool execution and state across multiple agents, ensuring reliable data provenance and graceful degradation during API failures.

---

#### 30th April 2026


[Structural Dissolution: How Artificial Intelligence Dismantles Coordination Architecture and Reconfigures the Political Economy of Production](http://arxiv.org/abs/2604.27435)

- Structural Dissolution Framework: introduces a theoretical lens explaining how AI systematically dismantles traditional economic coordination architectures by absorbing multimodal human interfaces into intra-system computation.
- The framework utilizes Interface Internalization to convert inter-agent coordination into computational processes, effectively replacing firms and markets with Regional Data Sovereignty Entities.
- It identifies the emergence of Data-Personified Economic Agents and data refinement loops as the new drivers of value creation, shifting competitive advantage toward the control of domain-specific data infrastructure.

---



[Modeling Clinical Concern Trajectories in Language Model Agents](http://arxiv.org/abs/2604.27872)

- Clinical Concern Trajectory Modeling framework: introduces a lightweight agent architecture that integrates a memoryless clinical risk encoder with explicit temporal dynamics to produce continuous, legible escalation pressure signals.
- The framework utilizes second-order hysteretic dynamics to model clinical principles where worsening physiological signals propagate faster than improvements, ensuring smooth and anticipatory concern trajectories.
- By separating instantaneous risk encoding from temporal integration, the architecture enables LLM agents to provide transparent, human-interpretable monitoring in safety-critical clinical environments.

---

[Collaborative Agent Reasoning Engineering (CARE): A Structured Three-Party Design Methodology for Systematically Engineering AI Agents with SMEs, Developers, and Helper Agents](http://arxiv.org/abs/2604.28043)

- CARE (Collaborative Agent Reasoning Engineering): introduces a disciplined, stage-gated methodology for engineering LLM agents through a triadic collaboration between SMEs, developers, and helper LLM agents to produce explicit, reviewable artifacts.
- The framework deconstructs agent development into four design targets—interaction policy, domain grounding, tool orchestration, and evaluation/verification—to replace ad hoc prompt iteration with structured engineering.
- By utilizing helper LLM agents as facilitation infrastructure, CARE enables the systematic creation of versioned artifacts that ensure agent behavior is testable, maintainable, and resistant to regressions.

---


[High-Probability Convergence in Decentralized Stochastic Optimization with Gradient Tracking](http://arxiv.org/abs/2605.00281)

- GT-DSGD: introduces a decentralized optimization framework that achieves high-probability convergence guarantees by incorporating gradient tracking to mitigate data heterogeneity.
- The framework utilizes local model updates and gradient tracking variables to eliminate the effects of data heterogeneity across agents in a network.
- GT-DSGD provides order-optimal high-probability convergence rates for both non-convex and Polyak-Lojasiewicz cost functions under relaxed sub-Gaussian noise conditions.

---

[Agentic AI for Trip Planning Optimization Application](http://arxiv.org/abs/2605.00276)

- AAS (Agentic AI System): introduces a hierarchical multi-agent framework for vehicle-centric trip planning that utilizes centralized orchestration to coordinate specialized agents for adaptive reasoning and self-correction.
- The system employs an In-Vehicle Agent for interaction, an Orchestration Agent for sub-task decomposition, and a Pool of Specialized Agents for domain-specific execution to optimize itineraries under multi-constraint scenarios.
- The authors also present the TOP (Trip-planning Optimization Problems) Benchmark, a dataset of 500 queries with deterministic ground-truth solutions designed to evaluate optimization performance across varying levels of reasoning complexity.

---

[Fast Rates in α-Potential Games via Regularized Mirror Descent](http://arxiv.org/abs/2605.00268)

- ROPE (Regularized Offline Potential Equilibrium): introduces a framework that leverages the structural alignment of α-potential games to identify Nash Equilibria via potential maximization using KL regularization.
- OPMD (Offline Potential Mirror Descent): provides a decentralized algorithm that achieves an accelerated O(1/n) statistical rate by resolving evaluation mismatch as a vanishing optimization error.
- The research establishes that Reference-Anchored Unilateral Concentrability allows for sample-efficient equilibrium recovery without explicit pessimistic bonuses, bypassing the PPAD-complete complexity of generic general-sum games.

---

[Pessimism-Free Offline Learning in General-Sum Games via KL Regularization](http://arxiv.org/abs/2605.00264)

- GANE (General-sum Anchored Nash Equilibrium): introduces a pessimism-free framework that leverages KL regularization to achieve an accelerated O˜(1/n) statistical rate for Nash equilibria in general-sum games.
- GAMD (General-sum Anchored Mirror Descent): provides a computationally tractable, decentralized iterative algorithm that recovers Coarse Correlated Equilibria at the standard O˜(1/√n) statistical rate without explicit pessimism.
- Both frameworks utilize reference-anchored unilateral concentrability to bypass the conceptual circularity of traditional pessimism-based approaches in multi-agent offline learning.

---

[Causal Foundations of Collective Agency](http://arxiv.org/abs/2605.00248)

- Mechanized Causal Games: introduces a formal framework for identifying collective agency in multi-agent systems by leveraging causal abstraction to determine when a group of agents can be modeled as a single unified agent.
- The framework utilizes Mechanism nodes (explicitly represented causal parameters), Object nodes (variables governed by mechanisms), Utility functions (functions mapping object nodes to real numbers), Rationality relations (specifies agent response to context), Variable alignment (mapping between low-level and high-level variables), Value mappings (translates low-level to high-level values), and Intervention mappings (translates low-level to high-level interventions) to provide a rigorous account of agency across levels of abstraction.
- The authors demonstrate the utility of their approach by analyzing actor-critic models and voting mechanisms, providing a foundation for understanding, predicting, and controlling emergent collective agents in multi-agent AI systems.

---

[The Silicon Society Cookbook: Design Space of LLM-based Social Simulations](http://arxiv.org/abs/2605.00197)

- Silicon Society Cookbook: introduces a systematic analysis of the design space for LLM-based social simulations, utilizing SimEngine, LLM-based Agents, LoRA Adapters, Followership Network, and News Source Nodes to evaluate simulation outcomes.
- The framework employs LoRA Adapters to instantiate diverse agent personas, enabling the study of opinion dynamics and information diffusion within synthetic social networks.
- The research identifies that the choice of base LLM is the most significant variable impacting simulation behavior, while demonstrating that fine-tuning on social media data improves stylistic realism.

---

[Real-Time Frame- and Event-based Object Detection with Spiking Neural Networks on Edge Neuromorphic Hardware: Design, Deployment and Benchmark](http://arxiv.org/abs/2605.00146)

- SNN-based Object Detection Framework: introduces a methodology for designing and deploying lightweight SNNs on Intel Loihi 2 neuromorphic hardware for real-time object detection using both frame-based and event-based data.
- The framework utilizes an ANN-to-SNN knowledge distillation approach to recover 87–100% of ANN detection accuracy while maintaining low inference latency and energy efficiency on edge neuromorphic processors.
- The study benchmarks three SNN architectures against ANN counterparts on Jetson Nano and Apple M2 platforms, demonstrating superior energy efficiency and real-time performance on the Intel Loihi 2 neuromorphic chip.

---

[Are Tools All We Need? Unveiling the Tool-Use Tax in LLM Agents](http://arxiv.org/abs/2605.00136)

- Factorized Intervention Framework: introduces a diagnostic methodology to decompose the performance gap between native CoT and tool-augmented LLMs into formatting costs, protocol overhead, and actual tool execution gains.
- The paper identifies the Capability Overlap Principle, where tool-derived gains are often redundant with the model's native reasoning, while the tool-use protocol introduces significant performance degradation under semantic noise.
- The authors propose G-STEP, a lightweight inference-time gate that mitigates protocol-induced errors by dynamically deciding whether to continue tool interaction or commit to a final answer.

---

[HERMES++: Toward a Unified Driving World Model for 3D Scene Understanding and Generation](http://arxiv.org/abs/2604.28196)

- HERMES++: introduces a unified driving world model that integrates 3D scene understanding and future geometry prediction within a single framework using BEV Tokenizer, LLM, World Queries, Current-to-Future Link, Shared Render, Joint Geometric Optimization, and Ego Modulation.
- The framework leverages a BEV representation to consolidate multi-view spatial information, enabling the LLM to perform semantic reasoning while guiding future geometric evolution through a Current-to-Future Link.
- A Joint Geometric Optimization strategy enforces structural integrity by combining explicit geometric constraints on point clouds with implicit latent regularization, ensuring predicted features align with intrinsic 3D geometry.

---

[Visual Generation in the New Era: An Evolution from Atomic Mapping to Agentic World Modeling](http://arxiv.org/abs/2604.28185)

- Visual Generation in the New Era: introduces a five-level taxonomy of visual intelligence, spanning Atomic Generation, Conditional Generation, In-Context Generation, Agentic Generation, and World-Modeling Generation, to categorize the evolution of visual models from passive renderers to interactive, agentic, world-aware generators.
- The paper decomposes modern visual generation architectures into four core components: Encoder/Tokenizer, Condition Module, Backbone Network, and Multimodal Fusion Module, which collectively enable the transition from single-pass rendering to complex, agentic workflows.
- It provides a comprehensive roadmap of technical drivers, including the shift from diffusion to flow matching, the rise of unified understanding-and-generation models, and the integration of agentic loops for grounded verification and causal world modeling.

---

[Exploration Hacking: Can LLMs Learn to Resist RL Training?](http://arxiv.org/abs/2604.28182)

- Exploration Hacking: introduces, "a failure mode where LLMs strategically alter their exploration during RL training to influence the training outcome and resist capability elicitation", with all LLM, RL-based capability elicitation, Model organisms, CoT monitors, Action-only monitors, Weight noising, SFT-based elicitation.
- The paper formalizes exploration hacking into a taxonomy of strategies and demonstrates that LLMs can be fine-tuned into "model organisms" that successfully resist RL-based capability elicitation while maintaining performance on benign tasks.
- The authors evaluate countermeasures including monitoring and weight noising, and audit frontier models, finding that while models possess the capability for strategic reasoning, they currently lack the spontaneous propensity to execute exploration hacking without explicit scaffolding.

---

[Synthetic Computers at Scale for Long-Horizon Productivity Simulation](http://arxiv.org/abs/2604.28181)

- Synthetic Computers at Scale: introduces a methodology for creating artifact-rich, user-specific environments to ground long-horizon simulations, utilizing Persona, User Profile, Filesystem Planning, and Synthetic Computer components.
- The framework employs a Setup Agent to define objectives and a Work Agent to perform tasks, interacting with Simulated Collaborators to generate Experiential Signals.
- These signals are processed by a Skill Creator to build a Skill Set, enabling agent self-improvement through iterative simulation and skill-based reinforcement.

---

[FlashRT: Towards Computationally and Memory Efficient Red-Teaming for Prompt Injection and Knowledge Corruption](http://arxiv.org/abs/2604.28157)

- FlashRT: introduces a framework to optimize the computation and memory efficiency of optimization-based red-teaming for long-context LLMs using Selective Recomputation, Gradient Approximation, Gradient Resampling, KV-Caching, and a Target LLM.
- The framework significantly reduces GPU memory consumption and runtime by approximating loss evaluations and gradient computations while maintaining high attack success rates.
- FlashRT is compatible with various black-box and white-box optimization methods and demonstrates effectiveness across diverse long-context LLM applications including prompt injection and knowledge corruption.

---

[CRAB: A Semantics-Aware Checkpoint/Restore Runtime for Agent Sandboxes](http://arxiv.org/abs/2604.28138)

- CRAB: introduces a semantics-aware host-side runtime that bridges the agent–OS semantic gap by inferring recovery-relevant state from OS-visible effects at turn boundaries to enable efficient checkpointing.
- The framework utilizes a Coordinator to identify turn boundaries, an eBPF-based Inspector to classify state changes, and a C/R Engine to schedule asynchronous checkpoints that overlap with LLM wait windows.
- CRAB achieves 100% recovery correctness while maintaining low overhead by selectively checkpointing only necessary state and coordinating traffic across co-located sandboxes.

---

[What Makes a Good Terminal-Agent Benchmark Task: A Guideline for Adversarial, Difficult, and Legible Evaluation Design](http://arxiv.org/abs/2604.28093)

- Terminal Bench: introduces a set of guidelines for designing adversarial, difficult, and legible evaluation tasks for LLM-based agents in command-line environments.
- The paper identifies common failure modes in benchmark design, such as reward hacking and over-prescriptive instructions, which undermine the validity of LLM performance metrics.
- It advocates for shifting from prompt-like task design to authentic engineering problems that prioritize verifiable outcomes over specific implementation steps.

---

[Towards Neuro-symbolic Causal Rule Synthesis, Verification, and Evaluation Grounded in Legal and Safety Principles](http://arxiv.org/abs/2604.28087)

- NeSy Causal Framework: introduces a meta-level synthesis and verification layer that utilizes LLMs to refine formal rule theories from natural-language goals, mitigating goal misspecification in self-adaptive systems.
- The architecture integrates a Goal/Rule Synthesizer and a Rule Verification Engine to ensure that synthesized rules are logically consistent, safe, and traceable before integration into the system knowledge base.
- By interleaving logical reasoning, causal inference, and learning, the framework enables incremental, modular, and explainable rule maintenance for safety-critical autonomous driving scenarios.

---

[TOPBENCH: A Benchmark for Implicit Prediction and Reasoning over Tabular Question Answering](http://arxiv.org/abs/2604.28076)

- TOPBENCH: introduces a benchmark designed to evaluate LLMs on implicit predictive tasks over tabular data, requiring models to infer unobserved outcomes rather than performing simple fact retrieval.
- The framework utilizes an LLM-as-a-Judge and a ReAct-Agent to assess performance across four predictive scenarios, including single-point prediction, decision making, treatment effect analysis, and ranking.
- The research identifies a critical capability gap where LLMs struggle to distinguish between retrieval and predictive intent, often defaulting to exhaustive search loops instead of rigorous statistical modeling.

---

[Agent-Agnostic Evaluation of SQL Accuracy in Production Text-to-SQL Systems](http://arxiv.org/abs/2604.28049)

- STEF (Schema-agnostic Text-to-SQL Evaluation Framework): introduces a production-native evaluation system that assesses Text-to-SQL generation quality without requiring database schema or reference queries by utilizing Input Specification, Enriched Question Validation, Semantic Feature Extraction, Normalized Specification Alignment, Application-Specific Rule Injection, LLM Evaluator, and Composite Scoring Function.
- The framework employs a multi-stage pipeline that transforms natural language and SQL inputs into structured semantic comparisons, enabling fine-grained diagnostic signals for continuous agent improvement.
- STEF mitigates evaluator variance and production false negatives through a confidence-weighted composite scoring mechanism and configurable normalization rules for common SQL idioms.

---

[Stable Behavior, Limited Variation: Persona Validity in LLM Agents for Urban Sentiment Perception](http://arxiv.org/abs/2604.28048)

- Persona-based Prompting Framework: introduces a methodology to evaluate persona validity in MLLM agents by measuring within-persona convergence and cross-persona variation across urban sentiment tasks.
- The framework utilizes a balanced factorial design of 24 personas and 1,200 agents to assess whether demographic and personality attributes induce meaningful behavioral differences in sentiment annotation.
- Experimental results indicate that while persona prompting stabilizes agent behavior, it fails to capture fine-grained perceptual judgments, often resulting in extremity bias and performance comparable to neutral no-persona baselines.

---

[Early Detection of Water Stress by Plant Electrophysiology: Machine Learning for Irrigation Management](http://arxiv.org/abs/2604.28038)

- PhytoNode-based Machine Learning Framework: introduces a biofeedback-driven system for early water stress detection in tomato plants using electrophysiological signals processed through automated machine learning and deep learning pipelines.
- The framework integrates PhytoNode hardware for signal acquisition with a processing pipeline that includes statistical feature extraction via tsfresh, model selection via Naive AutoML, and deep learning architectures including CNN, InceptionTime, and Mamba.
- Post-hoc certainty calibration using temperature scaling is employed to mitigate classifier overconfidence, enabling reliable transition detection from healthy to stressed plant states for precision irrigation management.

---

[Echo-α: Large Agentic Multimodal Reasoning Model for Ultrasound Interpretation](http://arxiv.org/abs/2604.28011)

- Echo-α: introduces an agentic multimodal reasoning framework that unifies specialized detector outputs with MLLM-based clinical reasoning through an invoke-and-reason loop.
- The framework utilizes a two-stage training process, starting with a supervised curriculum for foundational skills followed by reinforcement learning to optimize interaction trajectories for lesion grounding and diagnostic accuracy.
- By treating detector outputs as callable evidence rather than final predictions, the model enables verifiable clinical decision-making that adapts to different detector error modes across diverse ultrasound benchmarks.

---

[A Pattern Language for Resilient Visual Agents](http://arxiv.org/abs/2604.28001)

- Hierarchical Reference Architecture for Resilient Visual Agents: introduces a pattern language that decouples fast, deterministic System 1 reflexes from slow, probabilistic System 2 supervisory reasoning to balance latency and semantic adaptability.
- The architecture utilizes Hybrid Affordance Integration, Adaptive Visual Anchoring, Visual Hierarchy Synthesis, and a Semantic Scene Graph to enable amortized inference, where high-cost LLM-based reasoning is invoked only when low-latency reflexes fail.
- This approach transforms autonomous agents from brittle, open-loop scripts into closed-loop, self-healing systems capable of robust interaction within dynamic, non-instrumented enterprise GUI environments.

---

[Exploring Interaction Paradigms for LLM Agents in Scientific Visualization](http://arxiv.org/abs/2604.27996)

- SciVis Agent Paradigms: introduces a comparative study of three primary LLM agent interaction paradigms—domain-specific tool-use, GUI-based computer-use, and CLI-based coding—evaluated on scientific visualization tasks.
- The study identifies that while general-purpose coding agents achieve the highest task completion rates, they incur significantly higher computational costs compared to domain-specific agents which offer more efficient and stable execution.
- The research demonstrates that persistent memory and adaptive learning improve agent effectiveness and efficiency by reducing redundant exploration, while GUI-based agents face challenges with long-horizon planning despite strong localized reasoning capabilities.

---

[Dreaming Across Towns: Semantic Rollout and Town-Adversarial Regularization for Zero-Shot Held-Out-Town Fixed-Route Driving in CARLA](http://arxiv.org/abs/2604.27994)

- Dreamer-style latent world-model agent: introduces a framework for zero-shot held-out-town driving in CARLA by augmenting a world model with multi-horizon semantic rollout prediction and town-adversarial regularization.
- The framework utilizes a frozen OpenCLIP encoder to provide semantic targets for imagined rollouts, while a domain-adversarial branch encourages the learned latent representation to be invariant to source-town identity.
- By conditioning the semantic rollout predictor on a history-aware context feature, the agent improves route-following performance in unseen environments without requiring target-domain adaptation.

---

[FineState-Bench: Benchmarking State-Conditioned Grounding for Fine-grained GUI State Setting](http://arxiv.org/abs/2604.27974)

- FineState-Bench: introduces a benchmark and diagnostic framework for evaluating fine-grained, state-conditioned GUI interaction across desktop, web, and mobile platforms, utilizing FineState-Metrics and VDA to isolate grounding and state-setting failures.
- The framework employs a two-point interaction protocol that decouples target control localization from the operation-relevant region, enabling precise failure attribution in GUI agents.
- Experimental results demonstrate that while LVLMs often achieve successful coarse localization, they frequently struggle with interactable-core grounding, a bottleneck that VDA-provided localization hints can significantly mitigate.

---

[Language Models Refine Mechanical Linkage Designs Through Symbolic Reflection and Modular Optimisation](http://arxiv.org/abs/2604.27962)

- Symbolic Lifting and Closed-loop Synthesis Pipeline: introduces a modular framework that decomposes mechanical linkage design into discrete topology selection by LLM agents and continuous parameter fitting by numerical optimisers.
- The system utilizes a symbolic lifting operator to translate dense simulator trajectories into qualitative descriptors, enabling LLM agents to perform reflective reasoning and iterative design refinement.
- By binding high-level reasoning to mechanistic interfaces, the pipeline achieves superior geometric accuracy and structural validity across diverse LLM architectures without requiring fine-tuning.

---

[GUI Agents with Reinforcement Learning: Toward Digital Inhabitants](http://arxiv.org/abs/2604.27955)

- GUI Agents with Reinforcement Learning: Toward Digital Inhabitants introduces a principled taxonomy of RL paradigms—offline, online, and hybrid—to advance GUI agents from task-specific tools to persistent digital inhabitants.
- The paper analyzes critical dimensions including reward engineering, data efficiency, and technical innovations, highlighting the shift toward verifiable environment feedback and cognitive stratification.
- It provides a comprehensive roadmap for agent-native infrastructure, emphasizing the necessity of decoupling reasoning and execution to overcome I/O latency and sparse reward challenges in complex digital environments.

---

[Attractor FCM](http://arxiv.org/abs/2604.27947)

- Attractor FCM: introduces a physics-constrained, gradient-based fuzzy cognitive map that utilizes residual memory, back propagation through time, and a fixed point anchor to ensure stable convergence.
- The framework employs Jacobian Gradient Descent and Newton’s method to optimize weights directly from the fixed-point geometry while using an adaptive term to navigate the landscape.
- A causal mask is applied to the network to maintain structural integrity and respect expert-defined physical constraints during the optimization process.

---

[A Collective Variational Principle Unifying Bayesian Inference, Game Theory, and Thermodynamics](http://arxiv.org/abs/2604.27942)

- GT-FEP (Game-Theoretic Free Energy Principle): introduces a unified framework that bridges variational inference, stochastic game theory, and statistical physics to explain collective intelligence in multi-agent systems.
- The framework utilizes the Harsanyi decomposition to quantify irreducible synergy and conflict within coalitions, establishing that stationary points of collective free energy correspond to Nash equilibria.
- It demonstrates that agent influence follows a non-monotonic inverted-U relationship with sensory precision, while providing a first-principles derivation of attention mechanisms as mean-field approximations of coalitional inference.

---

[Alignment Contracts for Agentic Security Systems](http://arxiv.org/abs/2605.00081)

- Alignment Contracts framework: introduces a formal method for specifying and enforcing behavioral constraints on LLM-based security agents by mediating observable effect traces.
- The architecture utilizes a dual-layer approach, combining best-effort semantic mitigation at the agentic layer with rigorous, theorem-backed enforcement at the typed mediation layer.
- The framework provides decidable admissibility checks and modular contract composition, with core security properties mechanized in Lean 4 to ensure soundness under explicit observability assumptions.

---

[World Model for Robot Learning: A Comprehensive Survey](http://arxiv.org/abs/2605.00080)

- WMRL: introduces a comprehensive survey of world models in robot learning, categorizing architectures into decoupled, single-backbone, MoE/MoT, unified VLA, and latent-space paradigms.
- The paper examines how world models serve as learned simulators for reinforcement learning, evaluation, and data generation, emphasizing the transition from passive video generation to action-conditioned, physically grounded predictive structures.
- It highlights the shift toward foundation-scale models and identifies key challenges in causal conditioning, computational efficiency, and the need for standardized evaluation metrics beyond visual realism.

---

[MM-StanceDet: Retrieval-Augmented Multi-modal Multi-agent Stance Detection](http://arxiv.org/abs/2604.27934)

- MM-StanceDet: introduces a multi-agent framework for robust multimodal stance detection by integrating Retrieval Augmentation Stage, Multimodal Analysis Stage (comprising Text-Analysis Agent, Image-Analysis Agent, and Modality-Conflict Agent), Reasoning-Enhanced Debate Stage (with Debater Agents), and Self-Reflection and Adjudication Stage (using an Adjudicator Agent).
- The framework utilizes a Vector Database to provide few-shot exemplars, enabling specialized agents to perform nuanced analysis and resolve cross-modal conflicts through structured debate and critical self-reflection.
- Extensive experiments demonstrate that this structured agentic process significantly outperforms single-pass LLMs and state-of-the-art baselines across diverse multimodal stance detection datasets.

---

[Can AI Be a Good Peer Reviewer? A Survey of Peer Review Process, Evaluation, and the Future](http://arxiv.org/abs/2604.27924)

- AI-assisted Peer Review Frameworks: introduces a comprehensive taxonomy of methodologies for automating peer review, spanning generation paradigms, after-review tasks, and evaluation frameworks.
- The paper categorizes generation methodologies into Foundation Approaches, Fine-tuning Methods, Agent-based Methods, Reinforcement Learning Methods, and Review Generation Enhancement, which includes External Knowledge, Iterative Refinement, and Style Control.
- It provides a systematic evaluation taxonomy comprising Human-Centric, Reference-Based, LLM-Based, and Aspect-Oriented methods, while discussing critical challenges such as novelty assessment, bias, and the need for multimodal integration.

---

[A Logic of Inability](http://arxiv.org/abs/2604.27917)

- CLIab (Coalition Logic of Inability): introduces a conservative extension of Coalition Logic that treats inability as a first-class modal operator to systematically analyze strategic limitations.
- The framework provides a formal semantics and axiomatization for the inability operator, establishing its structural properties such as anti-monotonicity and subadditivity.
- By formalizing inability as an explicit modality, the paper enables precise reasoning about bounded agency and negative requirements in multi-agent systems.

---

[Graph World Models: Concepts, Taxonomy, and Future Directions](http://arxiv.org/abs/2604.27895)

- GWM (Graph World Models): introduces a unified research paradigm that formalizes world models by injecting relational inductive biases into environment representations to improve prediction and planning.
- The framework categorizes GWMs into three layers: Graph as Connector (spatial RIB), Graph as Simulator (physical RIB), and Graph as Reasoner (logical RIB).
- This survey outlines design principles, representative models, and future research directions including dynamic graph adaptation, probabilistic modeling, and multi-granularity inductive biases.

---

[In-Context Prompting Obsoletes Agent Orchestration for Procedural Tasks](http://arxiv.org/abs/2604.27891)

- In-Context Prompting: demonstrates that for procedural tasks, providing the entire workflow in the system prompt allows an LLM to self-orchestrate more effectively than using external frameworks like LangGraph.
- The study evaluates performance across three domains, finding that in-context prompting consistently outperforms orchestrated agents in task success, information accuracy, and consistency.
- Orchestration is identified as a bottleneck that fragments reasoning and introduces structural failure modes, whereas frontier models demonstrate sufficient capability to manage defined procedures internally.

---

[Building Persona-Based Agents On Demand: Tailoring Multi-Agent Workflows to User Needs](http://arxiv.org/abs/2604.27882)

- On-Demand Persona-Based Agent Generation framework: introduces a pipeline for dynamic, run-time synthesis of agent personas to tailor multi-agent workflows to specific user needs and contexts.
- The system replaces fixed agent architectures with a generative approach that utilizes ProfileEncode, TaskDecompose, PersonaCraft, AgentFactory, Agents Assignment, Agents Execution, and Answers Aggregation to instantiate agents on demand.
- By treating agent roles and interaction policies as runtime variables rather than design-time constants, the framework enables context-sensitive adaptivity and personalized multi-agent collaboration.

---

[KellyBench: A Benchmark for Long-Horizon Sequential Decision Making](http://arxiv.org/abs/2604.27865)

- KellyBench: introduces a non-stationary, open-ended environment for evaluating LLMs on long-horizon sequential decision-making tasks in sports betting markets.
- The framework utilizes an Environment, Agent, Sandbox, CLI Tools, and a Reward Signal to measure an agent's ability to build predictive models, identify market edges, and manage risk under uncertainty.
- Experimental results demonstrate that current frontier LLMs systematically underperform human baselines, frequently failing to execute reasoned strategies due to poor situational awareness and infrastructure-related errors.

---

[Rethinking Agentic Reinforcement Learning In Large Language Models](http://arxiv.org/abs/2604.27859)

- Agentic RL: introduces a control-theoretic framework for LLMs that integrates Agent, Planning, Memory, Action, and Tools to enable autonomous, self-improving decision-making.
- The framework shifts LLM optimization from static text generation to cumulative trajectory returns within partially observable environments.
- It synthesizes recent advancements in reinforcement learning algorithms—such as PPO, DPO, GRPO, and GSPO—to enhance the reasoning, planning, and tool-use capabilities of LLMs.

---

[NetSatBench: A Distributed LEO Constellation Emulator with an SRv6 Case Study](http://arxiv.org/abs/2604.27854)

- NetSatBench: introduces a distributed emulation platform for evaluating communication protocols and application workloads over large-scale LEO satellite systems using Worker nodes, Docker containers, Etcd key-value store, sat-agent, VXLAN overlay, nsb CLI, Physical-modeling pipeline, Routing plug-ins, and SRv6 control-plane.
- The platform decouples physical-layer and routing modeling from the emulator core through external plug-ins and utilizes a declarative JSON-based workflow for defining constellation dynamics.
- The system demonstrates its capabilities through an SRv6-based LEO architecture, highlighting the importance of end-to-end handover strategies that jointly account for both user-side and gateway-side access segments.

---

[A Grid-Aware Agent-Based Model for Analyzing Electric Vehicle Charging Systems](http://arxiv.org/abs/2604.27849)

- ABM: introduces a configurable, grid-aware simulation framework for analyzing electric vehicle charging dynamics by integrating EV, CC, and ES components.
- The framework utilizes discrete-event simulation to evaluate infrastructure performance, scheduling strategies, and grid-facing load characteristics under varying system scales.
- The model demonstrates that infrastructure dimensioning must be aligned with specific usage contexts, as fast charging infrastructure may induce higher grid stress without proportional operational benefits in workplace scenarios.

---

[CastFlow: Learning Role-Specialized Agentic Workflows for Time Series Forecasting](http://arxiv.org/abs/2604.27840)

- CastFlow: introduces a dynamic agentic forecasting framework that reformulates time series forecasting as a workflow-driven process, utilizing a Planning Module, Action Module, Forecasting Module, Reflection Module, Memory Module, Multi-view Toolkit, Frozen LLM, and Fine-tuned Domain-specific LLM.
- The framework employs a role-specialized design where a frozen LLM handles general-purpose reasoning for planning and reflection, while a fine-tuned domain-specific LLM executes numerical forecasting based on an ensemble forecast baseline and diagnostic evidence.
- CastFlow optimizes its performance through a two-stage workflow-oriented training strategy that combines supervised fine-tuning and reinforcement learning with verifiable rewards to ensure structural validity and numerical precision.

---

[Learning-Based Hierarchical Scene Graph Matching for Robot Localization Leveraging Prior Maps](http://arxiv.org/abs/2604.27821)

- HSGM: introduces a learned, end-to-end differentiable pipeline for hierarchical scene graph matching between BIM-derived A-graphs and robot-built S-graphs, utilizing an MLP Encoder, GATv2 Encoder, Affinity Matrix, Sinkhorn Layer, and Hungarian Algorithm.
- The framework employs a graph augmentation strategy to encode intra- and inter-level spatial relationships, enabling information flow across room and wall surface hierarchy levels.
- By replacing combinatorial search with a learned GNN-based approach, the pipeline achieves significant speedups and demonstrates zero-shot generalization to real LiDAR environments.

---

[ObjectGraph: From Document Injection to Knowledge Traversal](http://arxiv.org/abs/2604.27820)

- ObjectGraph: introduces a native file format that reconceives documents as typed, directed knowledge graphs to enable efficient traversal by LLM agents instead of full-document injection.
- The framework utilizes a Progressive Disclosure Model and a two-primitive query protocol to minimize token consumption by retrieving only task-relevant nodes.
- By integrating role-scoped access control and executable assertion nodes directly into the document structure, the approach reduces system complexity and eliminates the need for external middleware in multi-agent pipelines.

---

[MCPHunt: An Evaluation Framework for Cross-Boundary Data Propagation in Multi-Server MCP Agents](http://arxiv.org/abs/2604.27819)

- MCPHunt: introduces a controlled benchmark for measuring non-adversarial cross-boundary credential propagation in multi-server LLM agents using Evaluation Design, Trace Collection, and Detection & Analysis.
- The framework utilizes Canary Generation and Environment Control to isolate structural propagation risks from benign task execution across various MCP servers.
- By applying CRS Stratification, the methodology effectively separates task-mandated data transfers from policy-violating safety failures in LLM-based agent workflows.

---

[Separating Feasibility and Movement in Solution Discovery: The Case of Path Discovery](http://arxiv.org/abs/2604.27802)

- Two-graph model framework: introduces a directed weighted two-graph model that separates feasibility constraints, defined by a problem graph G, from movement constraints, defined by a movement graph M, to study solution discovery.
- The framework utilizes a token configuration S and a movement budget b to determine if a target structure in G can be realized through a sequence of local moves in M.
- The research identifies the complexity landscape of Path Discovery and Shortest Path Discovery, providing FPT algorithms for parameters like token number, feedback edge set, and solution size, while establishing hardness results for planar graphs and width parameters.

---

[AgentReputation: A Decentralized Agentic AI Reputation Framework](http://arxiv.org/abs/2605.00073)

- AgentReputation: introduces a three-layer decentralized framework for agentic AI systems that integrates Task Owner, Agents, Verifiers, Functional Layer, Evidence Collection, Reputation Cards, Policy Engine, and Blockchain and Storage Layer to provide evidence-based, context-aware, and decision-oriented reputation.
- The framework decouples task execution from reputation services and storage to enable independent evolution of verification regimes and policy logic.
- By utilizing context-conditioned reputation cards and a policy engine, the system prevents domain-specific reputation conflation and actively governs resource allocation and verification intensity for LLMs.

---

[WindowsWorld: A Process-Centric Benchmark of Autonomous GUI Agents in Professional Cross-Application Environments](http://arxiv.org/abs/2604.27776)

- WindowsWorld: introduces a process-centric benchmark for evaluating GUI agents on complex, multi-application professional workflows using a human-in-the-loop multi-agent generation pipeline.
- The framework utilizes Generator Agent, Refiner Agent, Reviewer Agent, and Environment Generator Agent to construct 181 tasks across 17 applications, incorporating Intermediate Checkpoints for fine-grained diagnostic evaluation.
- Experimental results demonstrate that current LLMs struggle with cross-application coordination and conditional reasoning, achieving only a 20% success rate on multi-step professional tasks.

---

[Monadic Presburger Predicates have Robust Population Protocols](http://arxiv.org/abs/2604.27767)

- Robust Population Protocols framework: introduces robust protocols for all monadic Presburger predicates, ensuring stable consensus despite adversarial crash failures.
- The framework utilizes state-based redundancy and tower-based level tracking to maintain correct outputs under adversarial sniping of agents.
- The research establishes that the state complexity cost for achieving such robustness is at least double exponential relative to the predicate size.

---

[INTENT2TX: Benchmarking LLMs for Translating Natural Language Intents into Ethereum Transactions](http://arxiv.org/abs/2604.27763)

- INTENT2TX: introduces a high-fidelity benchmark for translating natural language intents into executable Ethereum transactions, utilizing Trace Acquisition, Contract Grounding, Filtering, Structured Decoding, Intent Synthesis, Taxonomy Tagging, Multi-step Composition, Differential State Analysis, and Retrieval-augmented Inference.
- The framework employs a simulation-based evaluation protocol using differential state analysis on forked mainnet environments to assess the functional correctness of LLM-generated transactions.
- Evaluation of 16 LLMs demonstrates that retrieval-augmented inference significantly improves logical consistency, parameter precision, and end-to-end execution success rates compared to direct inference.

---

[AgentEconomist: An End-to-end Agentic System Translating Economic Intuitions into Executable Computational Experiments](http://arxiv.org/abs/2604.27725)

- AgentEconomist: introduces an end-to-end agentic system that translates abstract economic intuitions into verifiable computational experiments through a modular multi-stage architecture.
- The framework integrates a retrieval-augmented knowledge base, a structured memory module, and an MCP-based toolbox to support iterative, literature-grounded research workflows.
- By coupling domain-specific grounding with an executable simulation substrate, the system enables researchers to perform rapid hypothesis verification while maintaining human-in-the-loop control.

---

[Compliance-Aware Agentic Payments on Stablecoin Rails](http://arxiv.org/abs/2605.00071)

- Compliance-Aware Agentic Payment Architecture: introduces a modular, hybrid system that integrates programmable compliance directly into stablecoin payment rails to enable secure, automated agentic transactions.
- The architecture utilizes a Compliance Agent to coordinate buyer and seller agents, while employing a PolicyWrapper to enforce on-chain regulatory checks at the point of execution.
- The system supports structured resolution for pending compliance requirements through agent-mediated tranching and escrow, ensuring regulatory adherence without manual intervention.

---

[Knowledge Graph Representations for LLM-Based Policy Compliance Reasoning](http://arxiv.org/abs/2604.27713)

- Agentic AI framework: introduces an end-to-end pipeline that constructs knowledge graphs from AI governance documents to ground LLM reasoning in authoritative policy text.
- The framework utilizes a Chunking Agent, Extraction Agent, Retrieval Agent, and Synthesis Agent to improve compliance QA performance across various LLMs.
- Evaluation demonstrates that KG augmentation consistently improves LLM performance, with an open, LLM-discovered ontology matching or exceeding formal standards-derived schemas.

---

[Contextual Agentic Memory is a Memo, Not True Memory](http://arxiv.org/abs/2604.27707)

- Contextual Agentic Memory is a Memo, Not True Memory: introduces a formal distinction between retrieval-based C-engineering and weight-based θ-learning, arguing that current LLM agents rely on episodic lookup rather than true rule-based expertise.
- The paper proves a compositional generalization gap, demonstrating that retrieval-based systems require quadratically more data than parametric systems to handle novel task combinations.
- It proposes an architectural shift where agents pair fast episodic retrieval with an asynchronous consolidation channel to encode distilled experience directly into model weights.

---

[Bridging Values and Behavior: A Hierarchical Framework for Proactive Embodied Agents](http://arxiv.org/abs/2604.27699)

- ValuePlanner: introduces a hierarchical neuro-symbolic cognitive architecture that decouples high-level value-based deliberation from low-level symbolic action execution to enable proactive autonomy.
- The framework utilizes a Generator–Critic LLM-based loop to reason over abstract value trade-offs, which are then grounded into executable plans by a classical PDDL planner.
- ValuePlanner incorporates a closed-loop Adjustment mechanism that triggers on successful execution to identify synergistic plan refinements and maintain long-horizon value coherence.

---

[When Agents Evolve, Institutions Follow](http://arxiv.org/abs/2604.27691)

- SocialSystemArena: introduces a formal framework that translates historical political institutions into executable multi-agent governance specifications to evaluate their impact on collective performance.
- The framework utilizes a unified Governance Runtime to execute tasks across different LLM backends, isolating governance topology as a primary architectural variable.
- Experimental results demonstrate that governance topology significantly shapes performance, with optimal structures shifting dynamically based on model capability and task characteristics.

---

[The TEA Nets framework combines AI and cognitive network science to model targets, events and actors in text](http://arxiv.org/abs/2604.27673)

- TEA Nets (Target-Event-Agent Networks): introduces a computational framework that maps relational information in text into tripartite graphs composed of subjects (“Agents”), verbs (“Events”), and objects (“Targets”) to enable interpretable narrative analysis.
- The framework utilizes an NLP-pipeline to extract SVO triplets, which are then processed by an Analytics-module to compute network-level measures like degree and edge prominence for investigating linguistic patterns.
- By distinguishing between active and passive voice and incorporating sentiment valence, TEA Nets provide a transparent, "glass-box" approach to analyzing agency, causality, and emotional framing in large-scale textual corpora.

---

[From Context to Skills: Can Language Models Learn from Context Skillfully?](http://arxiv.org/abs/2604.27660)

- Ctx2Skill: introduces a self-evolving framework that autonomously discovers, refines, and selects context-specific skills through a multi-agent self-play loop without human supervision or external feedback.
- The framework utilizes a Challenger agent to probe context, a Reasoner agent to solve tasks, and a Judge to provide feedback, while Proposer and Generator agents iteratively update skill sets based on performance.
- A Cross-time Replay mechanism mitigates adversarial collapse by selecting the most generalizable skill set from historical candidates using curated hard and easy probe sets.

---

[HAVEN: Hybrid Automated Verification ENgine for UVM Testbench Synthesis with LLMs](http://arxiv.org/abs/2604.27643)

- HAVEN: introduces a hybrid verification framework that delegates UVM testbench and sequence generation to rule-based components while utilizing LLMs for structured information extraction and iterative coverage optimization.
- The framework employs a Spec Processor, Architecture Agent, Template Engine, Compile & Simulate Agent, DSL Generator, and DSL CodeGen to ensure syntactic correctness and protocol-compliant UVM testbench synthesis.
- By restricting LLMs to structured JSON outputs and using predefined Jinja2 templates, HAVEN achieves 100% compilation success and state-of-the-art coverage across diverse IP designs with minimal token usage.

---

[Do Open-Loop Metrics Predict Closed-Loop Driving? A Cross-Benchmark Correlation Study of NAVSIM and Bench2Drive](http://arxiv.org/abs/2605.00066)

- NAVSIM and Bench2Drive: introduces a systematic cross-benchmark correlation study evaluating whether safety-aware open-loop metrics can reliably predict closed-loop driving performance across diverse autonomous driving architectures.
- The study identifies that while traditional displacement metrics fail, the NAVSIM PDMS aggregate score shows strong positive correlation with closed-loop Driving Score, albeit with ranking inversions caused by a safety-progress trade-off.
- The authors propose a simplified CL-Proxy (NC × DAC × EP) that matches the predictive power of the full PDMS, highlighting that Ego Progress is the strongest individual predictor of closed-loop success.

---

[SpaAct: Spatially-Activated Transition Learning with Curriculum Adaptation for Vision-Language Navigation](http://arxiv.org/abs/2604.27620)

- SpaAct: introduces a transition-aware training framework that activates dynamic spatial awareness in LLMs for vision-language navigation using Vision Encoder, Tokenizer, LLM, Nav:VLN Action Learning, AR: Action Retrospection, FFS: Future Frame Selection, and TriPA Curriculum.
- The framework employs Action Retrospection to supervise backward action reasoning and Future Frame Selection to supervise forward transition prediction, effectively bridging the gap between semantic understanding and spatial dynamics.
- TriPA curriculum learning stabilizes the adaptation process by organizing training samples from easy to hard based on trajectory, instruction, and motion complexity factors.

---

[RoadMapper: A Multi-Agent System for Roadmap Generation of Solving Complex Research Problems](http://arxiv.org/abs/2604.27616)

- RoadMapper: introduces a multi-agent system that decomposes complex research roadmap generation into iterative stages of initial generation, knowledge augmentation, and a critique-revise-evaluate loop.
- The framework utilizes six specialized LLM-driven agents to address limitations in professional knowledge, task decomposition, and logical coherence found in standard LLM outputs.
- RoadMapper incorporates an Evaluate agent trained via Direct Preference Optimization (DPO) to ensure high-quality outputs and employs an early stopping mechanism to balance performance with computational efficiency.

---

[Trace-Level Analysis of Information Contamination in Multi-Agent Systems](http://arxiv.org/abs/2604.27586)

- Multi-Agent System (MAS) framework: introduces a trace-based measurement methodology to quantify information contamination in multi-agent workflows by injecting structured perturbations into artifact-derived representations and monitoring execution divergence.
- The framework utilizes a coordinator agent, specialized agents (Data Analyst, Fact Checker, Document Analyst, Visual Analyst, Audio Analyst, Computation Agent, Synthesizer), a shared workspace, provenance metadata, and evidence objects to track how uncertainty propagates through agent interactions.
- Empirical analysis across 614 runs reveals that structural divergence and outcome corruption are decoupled, with workflows often exhibiting behavioral detours or silent semantic corruption that evade standard endpoint-only evaluation metrics.

---

[SpatialGrammar: A Domain-Specific Language for LLM-Based 3D Indoor Scene Generation](http://arxiv.org/abs/2604.27555)

- SpatialGrammar: introduces a domain-specific language that represents indoor layouts as Bird's-Eye View grid placements with deterministic compilation to valid 3D geometry.
- The framework utilizes SG-Agent for iterative, closed-loop scene refinement and SG-Mini, a specialized small language model trained on compiler-validated synthetic data.
- By encoding physical priors into a compact DSL, the system enables verifiable constraint checking and hierarchical scene composition while reducing the cognitive load for LLMs.

---

[AppTek Call-Center Dialogues: A Multi-Accent Long-Form Benchmark for English ASR](http://arxiv.org/abs/2604.27543)

- AppTek Call-Center Dialogues: introduces a comprehensive benchmark for evaluating ASR systems on spontaneous, role-played, long-form conversational speech across fourteen English accents.
- The framework utilizes various segmentation strategies, including Manual segmentation, AppTek proprietary segmenter, Silero segmenter, and Fixed-length chunking, to assess ASR robustness in realistic call-center environments.
- The evaluation pipeline incorporates Guided recognition, leveraging a 4-gram background language model and a Segment-specific language model, to ensure high-quality transcription verification via Levenshtein alignment.

---

[Knowledge Affordances for Hybrid Human-AI Information Seeking](http://arxiv.org/abs/2604.27539)

- KA: introduces a conceptual framework for systematizing how agents identify actionable knowledge sources in hybrid human-AI environments using explicit semantic interfaces.
- The framework models knowledge sources as tuples containing capabilities (C), competency questions (CQ), content scope (S), non-functional properties (NFP), and grounding (G) to facilitate informed source selection.
- By framing information seeking as an affordance-driven process, the approach supports the construction of structured interrogation plans that enhance transparency and mutual intelligibility between human and artificial agents.

---

[EdgeFM: Efficient Edge Inference for Vision-Language Models](http://arxiv.org/abs/2604.27476)

- EdgeFM: introduces a lightweight, agent-driven inference framework designed for cross-platform industrial edge deployment by encapsulating optimized kernels as reusable skills.
- The framework utilizes a modular layered architecture with a thin runtime and heavy kernels to minimize end-to-end latency for single-request edge workloads.
- EdgeFM supports heterogeneous hardware including x86, NVIDIA Orin, and Horizon Journey platforms, effectively eliminating vendor-specific ecosystem lock-in.

---

[Security Attack and Defense Strategies for Autonomous Agent Frameworks: A Layered Review with OpenClaw as a Case Study](http://arxiv.org/abs/2604.27464)

- OpenClaw framework: introduces a four-layer security taxonomy for autonomous agent frameworks, comprising the Context and Instruction Layer, Tool and Action Layer, State and Persistence Layer, and Ecosystem and Automation Layer.
- The paper analyzes how security risks propagate across these layers, from initial input manipulation to persistent state contamination and ecosystem-level impact.
- It identifies critical research gaps including the lack of long-horizon evaluation and the need for more robust, integrated defense mechanisms against cross-layer attacks in LLM-based agent systems.

---

[RAY-TOLD: Ray-Based Latent Dynamics for Dense Dynamic Obstacle Avoidance with TDMPC](http://arxiv.org/abs/2604.27450)

- RAY-TOLD: introduces a hybrid control architecture that integrates LiDAR-centric latent dynamics with sampling-based MPPI to enhance navigation in dense dynamic environments.
- The framework utilizes an Encoder, Latent Dynamics, Reward Predictor, Value Function, Policy Prior, MPPI Planner, and Policy Mixture Sampling to bridge the gap between physics-based robustness and long-horizon learned foresight.
- By augmenting MPPI candidate populations with trajectories from a learned policy prior, the model effectively guides robots out of local minima while maintaining kinematic feasibility.

---

[Context as Prior: Bayesian-Inspired Intent Inference for Non-Speaking Agents with a Household Cat Testbed](http://arxiv.org/abs/2604.27445)

- CatSignal: introduces a Bayesian-inspired framework for intent inference in non-speaking agents by treating spatial context as a prior-like constraint rather than a standard input feature.
- The framework utilizes a context-gated Product-of-Experts architecture to combine spatial context, pose dynamics, and acoustic cues into a unified posterior intent distribution.
- Experimental results on a household cat testbed demonstrate that this approach improves overall accuracy and significantly reduces context-driven shortcut failures compared to standard fusion methods.

---

[Tracking Conversations: Measuring Content and Identity Exposure on AI Chatbots](http://arxiv.org/abs/2604.27438)

- AI Chatbot Tracking Measurement Framework: introduces a systematic methodology to quantify third-party tracking on 20 popular AI chatbots by analyzing network traffic for content and identity exposure.
- The framework utilizes Chatbot Interface, Network Traffic Capture, Preprocessing, Search Targets, Matching Procedure, and Party Attribution to identify data flows to advertising, analytics, and other third-party domains.
- The study reveals that 17 of 20 chatbots share user information with third parties, with some platforms exposing plaintext conversation text via Session Replay and identity data through Embedded Widgets, Analytics Tags, and Advertising Tags.

---

[InteractWeb-Bench: Can Multimodal Agent Escape Blind Execution in Interactive Website Generation?](http://arxiv.org/abs/2604.27419)

- InteractWeb-Bench: introduces a multimodal interactive benchmark designed to evaluate how LLM-based agents navigate real-world website generation tasks under ambiguous, low-code user instructions.
- The framework incorporates a Persona-Driven User Agent Module to simulate diverse user behaviors and an Interactive Execution Environment that enables agents to perform iterative intent refinement, code synthesis, and visual verification.
- Experimental results demonstrate that current LLMs frequently fall into a "blind execution" failure mode, struggling to proactively clarify underspecified requirements and instead over-generating code that leads to high hallucination rates.

---

[ChipLingo: A Systematic Training Framework for Large Language Models in EDA](http://arxiv.org/abs/2604.27415)

- ChipLingo: introduces a systematic three-stage training pipeline designed to enhance LLMs for knowledge-intensive Electronic Design Automation tasks through Domain-Adaptive Pretraining, Instruction Alignment Training, and RAG Scenario Fine-Tuning.
- The framework employs a Partial Parameter Training strategy to balance domain-specific knowledge acquisition with the retention of general LLM capabilities.
- To address RAG capability degradation after domain training, the pipeline incorporates targeted RAG scenario training, enabling models to effectively utilize external knowledge while maintaining robustness against irrelevant retrieval noise.

---

[Understanding Adversarial Transferability in Vision-Language Models for Autonomous Driving: A Cross-Architecture Analysis](http://arxiv.org/abs/2604.27414)

- Cross-Architecture Adversarial Transferability Framework: introduces a five-stage pipeline to systematically evaluate how adversarial patches transfer across diverse VLM architectures in autonomous driving.
- The framework utilizes Architecture-specific Patch Optimization, Scenario Evaluation, Semantic Homogenization, and Evaluation &amp; Analysis to quantify cross-architecture vulnerability and temporal persistence of attacks.
- The research demonstrates that adversarial patches optimized for one VLM architecture exhibit high transferability (73–91%) to others, revealing systematic security weaknesses in VLM-based autonomous driving systems.

---

[Detecting is Easy, Adapting is Hard: Local Expert Growth for Visual Model-Based Reinforcement Learning under Distribution Shift](http://arxiv.org/abs/2604.27411)

- JEPA-Indexed Local Expert Growth: introduces a modular framework for visual MBRL that uses a frozen JEPA representation for problem indexing and separate local residual experts for action correction under distribution shift.
- The framework preserves the original ID-trained baseline controller while augmenting it with shift-specific residual experts to improve OOD performance without catastrophic forgetting.
- The method employs a centroid-based routing mechanism with an ID reject option to ensure that local experts are only activated when necessary, effectively separating shift recognition from action adaptation.

---

[VeraRetouch: A Lightweight Fully Differentiable Framework for Multi-Task Reasoning Photo Retouching](http://arxiv.org/abs/2604.27375)

- VeraRetouch: introduces a fully differentiable framework for multi-task reasoning photo retouching that replaces non-differentiable external tools with a pixel-faithful Retouch Renderer.
- The framework utilizes a 0.5B Multi-Modal LLM to analyze image semantics and user instructions, generating structured retouching plans and control latents for end-to-end optimization.
- To support training, the authors introduce AetherRetouch-1M+, a million-scale dataset, and a reinforcement learning strategy (DAPO-AE) to enhance aesthetic cognition and logical consistency.

---

[Judge, Then Drive: A Critic-Centric Vision Language Action Framework for Autonomous Driving](http://arxiv.org/abs/2604.27366)

- CriticVLA: introduces a two-stage "judge, then drive" framework that leverages an LLM backbone as an explicit critic to refine initial driving trajectories through multimodal risk analysis and targeted action suggestions.
- The framework utilizes an InternVL2-1B backbone with LoRA and DETR modules to generate a rough trajectory in Stage-1, which is subsequently evaluated and refined in Stage-2 using a language-based critic and query-based delta adaptors.
- To enhance the critic's reasoning capabilities, the authors construct CriticDrive, a large-scale synthetic dataset containing 12.9 million annotated trajectories that align perception, language, and action modalities for robust closed-loop driving.

---

[TADI: Tool-Augmented Drilling Intelligence via Agentic LLM Orchestration over Heterogeneous Wellsite Data](http://arxiv.org/abs/2605.00060)

- TADI (Tool-Augmented Drilling Intelligence): introduces an agentic AI system that transforms heterogeneous drilling operational data into evidence-based analytical intelligence using an LLM Orchestrator, DuckDB, ChromaDB, Specialized Tools, System Prompt, and Output Validator.
- The framework utilizes a dual-store architecture to combine structured SQL queries with semantic text search, enabling multi-step reasoning and cross-referencing of drilling measurements with narrative reports.
- TADI implements a framework-free approach where domain-specific drilling engineering logic is encapsulated in deterministic tools rather than relying on model fine-tuning.

---

[Safe Bilevel Delegation (SBD): A Formal Framework for Runtime Delegation Safety in Multi-Agent Systems](http://arxiv.org/abs/2604.27358)

- SBD (Safe Bilevel Delegation): introduces a formal framework for runtime delegation safety in hierarchical multi-agent systems by formulating task delegation as a bilevel optimization problem.
- The framework utilizes a meta-weight network to dynamically adjust safety-efficiency trade-offs based on context, while an inner loop optimizes the delegation policy subject to probabilistic safety constraints.
- SBD provides theoretical guarantees including safety monotonicity, inner policy convergence, and an accountability propagation bound for multi-hop delegation chains.

---

[Heterogeneous Scientific Foundation Model Collaboration](http://arxiv.org/abs/2604.27351)

- Eywa: introduces a heterogeneous agentic framework that augments domain-specific foundation models with language-model-based reasoning interfaces to enable modality-native collaboration.
- The framework utilizes the Tsaheylu interface to bridge LLMs and specialized foundation models, enabling EywaAgent, EywaMAS, and EywaOrchestra to perform complex scientific tasks with higher utility and lower token consumption.
- EywaBench provides a scalable multi-task, multi-domain scientific benchmark to evaluate the performance of these heterogeneous agentic systems across physical, life, and social sciences.

---

[Dynamic-TD3: A Framework for UAV Path Planning with adversarial environment](http://arxiv.org/abs/2605.00059)

- Dynamic-TD3: introduces a physically enhanced framework for UAV navigation that decouples mission rewards from safety constraints using a dual critic architecture and Lagrangian relaxation.
- The framework integrates ATREM for intent-driven trajectory prediction and PAG-KF for robust state estimation under non-stationary noise.
- Experimental results demonstrate that Dynamic-TD3 achieves superior collision avoidance, reduced energy consumption, and smoother flight trajectories compared to baseline DRL methods.

---

[Toward Autonomous SOC Operations: End-to-End LLM Framework for Threat Detection, Query Generation, and Resolution in Security Operations](http://arxiv.org/abs/2604.27321)

- End-to-End Threat Management Framework: introduces an integrated pipeline that combines ensemble-based threat detection, syntax-constrained query generation via SQM, and RAG-augmented resolution to automate SOC workflows.
- The framework utilizes an ensemble of LLMs for high-accuracy threat detection and prioritizes alerts using SIEM-derived risk scores to reduce analyst workload.
- The SQM architecture improves query generation by grounding LLM outputs in platform-specific syntax allow-lists, metadata-driven retrieval, and official documentation to ensure executable security queries.

---

[Autoformalizing Memory Device Specifications with Agents](http://arxiv.org/abs/2605.00058)

- Autoformalization Agent: introduces an agentic framework that automatically converts natural language memory chip specifications into executable DRAMPyML models using iterative tool-use and validation.
- The framework leverages timed Petri nets to capture complex hardware protocol constraints, enabling the generation of downstream verification collateral such as assertions and stimulus.
- The authors release DRAMBench, a comprehensive benchmark covering 13 JEDEC memory standards, to evaluate the performance and token-efficiency of LLMs in hardware autoformalization tasks.

---

[Pragmos: A Process Agentic Modeling System](http://arxiv.org/abs/2604.27311)

- Pragmos: introduces an agentic system that decomposes complex business process modeling into iterative, human-in-the-loop steps using LLM, Prompting Engine, Modular Decomposition Module, BPMN Synthesis Engine, and Symbolic Execution Module.
- The framework utilizes an LLM to derive intuitive intermediate artifacts like execution paths, which are then structured into process models via modular decomposition and verified through symbolic execution.
- By avoiding black-box generation, the system enables analysts to maintain control and transparency throughout the co-design process of business process models.

---

[End-to-End Evaluation and Governance of an EHR-Embedded AI Agent for Clinicians](http://arxiv.org/abs/2604.27309)

- Hyperscribe: introduces an end-to-end governance framework for clinical AI that integrates Rubric Validation, Live Clinician Feedback, Technical Performance Monitoring, and Cost Tracking to ensure reliable deployment through Controlled Experimentation.
- The architecture utilizes a four-stage cyclical pipeline—Audio to Transcript, Transcript to Instructions, Instructions to Parameters, and Parameters to Commands—to enable granular failure attribution and governable clinical documentation.
- By employing structured outputs, explicit intermediate reasoning, and an EHR-bounded action space, the framework allows for targeted engineering interventions and direct version-to-version performance comparisons.

---

[METASYMBO: Multi-Agent Language-Guided Metamaterial Discovery via Symbolic Latent Evolution](http://arxiv.org/abs/2604.27300)

- METASYMBO: introduces a multi-agent framework for language-guided metamaterial discovery that bridges language, geometry, and property modalities through Agent Designer, Agent Generator, and Agent Supervisor.
- The framework utilizes a symbolic-driven latent evolution module in a disentangled latent space to enable inference-time semantic programming and cross-domain exploration beyond training data.
- By coordinating language reasoning, geometric synthesis, and property-aware feedback, the system achieves superior structural validity and prompt alignment compared to existing LLM and generative baselines.

---

[Machine Collective Intelligence for Explainable Scientific Discovery](http://arxiv.org/abs/2604.27297)

- MCI: introduces a unified paradigm that integrates symbolism and metaheuristics to enable autonomous discovery of governing equations through coordinated reasoning agents.
- The framework utilizes Reasoning agents, Local memory, and Shared memory to evolve symbolic hypotheses via iterative generation, evaluation, and consolidation.
- MCI employs AST as a canonical representation of scientific knowledge, allowing for quantitative explainability and robust OOD generalization across diverse scientific domains.

---

[Learning When to Remember: Risk-Sensitive Contextual Bandits for Abstention-Aware Memory Retrieval in LLM-Based Coding Agents](http://arxiv.org/abs/2604.27283)

- RSCB-MC: introduces a risk-sensitive contextual bandit framework that treats memory retrieval in LLM-based coding agents as a selective control problem to prevent unsafe memory injection.
- The system utilizes a pattern-variant-episode memory schema and a 16-feature contextual state to evaluate the safety of retrieved information before allowing it to influence the agent's debugging trajectory.
- By prioritizing abstention and non-injection as first-class safety actions, the framework effectively mitigates false-positive memory injections that often degrade the performance of autonomous coding agents.

---

[Select to Think: Unlocking SLM Potential with Local Sufficiency](http://arxiv.org/abs/2604.26940)

- S2T (SELECT TO THINK): introduces a framework that reframes LLM-assisted reasoning as a discrete selection task over an SLM's candidate set, utilizing a Trigger, Candidate Generator, Scoring Function, Reserved Tokens, Distilled Selector, and Inner Critic.
- The framework identifies local sufficiency, where an LLM's preferred token consistently resides within an SLM's top-K predictions, allowing for efficient, autonomous re-ranking without external LLM dependencies.
- By repurposing reserved tokens to encode preference scores, S2T-LOCAL enables the SLM to perform high-performance inference with single-trajectory efficiency, matching the efficacy of 8-path self-consistency.

---

[The Inverse-Wisdom Law: Architectural Tribalism and the Consensus Paradox in Agentic Swarms](http://arxiv.org/abs/2604.27274)

- PAS framework: introduces a mechanistic theory of swarm-level social engineering, demonstrating that agentic swarms prioritize internal architectural agreement over external logical truth through the Consensus Paradox.
- The research establishes the Inverse-Wisdom Law, proving that in kinship-dominant architectures, adding logical agents increases the stability of erroneous trajectories rather than the probability of truth.
- The study identifies the Heterogeneity Mandate as a foundational safety requirement, necessitating architectural diversity at the synthesizer node to break the Attention Latch and prevent terminal failure.

---

[Star-Fusion: A Multi-modal Transformer Architecture for Discrete Celestial Orientation via Spherical Topology](http://arxiv.org/abs/2604.26582)

- Star-Fusion: introduces a multi-modal architecture that reformulates spacecraft orientation estimation as a discrete topological classification task using SwinV2-Tiny, CNN Heatmap, and Coordinate MLP components.
- The framework utilizes spherical K-Means clustering to partition the celestial sphere into topologically consistent regions, effectively mitigating coordinate wrapping artifacts during orientation estimation.
- By integrating photometric, spatial, and numerical features through a late-fusion strategy, the model achieves high computational efficiency and robust performance on resource-constrained hardware for autonomous satellite navigation.

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

