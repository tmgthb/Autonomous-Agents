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




#### 9th June 2026


[EEVEE: Towards Test-time Prompt Learning in the Real World for Self-Improving Agents](http://arxiv.org/abs/2606.11182)

- EEVEE: introduces a multi-dataset test-time prompt learning framework for LLMs that utilizes a Router to partition heterogeneous input streams into specialized prompt configurations.
- The framework employs a Router-Prompt Co-evolution strategy, which uses interleaved learning phases to optimize the Router and Prompt Set simultaneously, addressing their mutual dependency.
- EEVEE incorporates a three-stage training process—initialization, exploration, and convergence—to effectively manage prompt slots and stabilize routing decisions for improved performance on mixed-dataset tasks.

---



[The Shibboleth Effect: Auditing the Cross-Lingual Distributional Skew of Large Language Models](http://arxiv.org/abs/2606.11082)

- Cerulean Sea Crisis Wargame Framework: introduces a multi-agent geopolitical simulation to audit cross-lingual behavioral skew in LLMs by manipulating language anchors across six frontier models.
- The framework utilizes an automated zero-shot judge classifier to evaluate model outputs on concession rates and coercive rhetoric under adversarial conditions.
- It identifies two distinct buffering mechanisms, chain-of-thought institutional anchoring and multilingual RLHF alignment, which mitigate language-conditioned geopolitical bias.

---

[Data Journalist Agent: Transforming Data into Verifiable Multimodal Stories](http://arxiv.org/abs/2606.11176)

- Data2Story: introduces a multi-agent framework that orchestrates specialized roles into a virtual newsroom to transform raw data into verifiable, multimodal articles.
- The framework utilizes an Inspector component to establish an auditable evidence chain by linking every claim, number, and visual asset to its specific source code or reference URL.
- Evaluation across 18 datasets demonstrates that Data2Story produces competitive, evidence-traceable stories, though human journalists maintain an advantage in creative design and editorial angle.

---

[ABC-Bench: An Agentic Bio-Capabilities Benchmark for Biosecurity](http://arxiv.org/abs/2606.11150)

- ABC-Bench: introduces an evaluation suite designed to measure the ability of LLM agents to perform complex, multi-step biological tasks using software tools and laboratory automation.
- The framework evaluates LLMs across three distinct tasks: Fragment Design, Screening Evasion, and Liquid Handling Robot, assessing their proficiency in bioinformatics and molecular biology.
- Results demonstrate that frontier LLM agents can match or exceed human expert performance on these tasks, highlighting both the potential for scientific acceleration and the associated biosecurity risks.

---

[OpenPcc: Open and Confidential LLM Serving on Commodity TEEs](http://arxiv.org/abs/2606.11145)

- OpenPcc: introduces a confidential LLM serving framework that leverages commodity CPU and GPU TEEs to provide end-to-end privacy for user data without relying on proprietary hardware.
- The framework utilizes a Confidential Inference Node, a Confidential Gateway, and an Inference Client to ensure that user prompts and completions remain encrypted and isolated from the service provider.
- OpenPcc achieves verifiable security through composite attestation, where the hardware root of trust is anchored in CPU and GPU vendors, allowing users to independently verify the integrity of the inference environment.

---

[Monte Carlo Pass Search: Using Trajectory Generation for 3D Counterfactual Pass Evaluation in Football](http://arxiv.org/abs/2606.11120)

- MCPS (Monte Carlo Pass Search): introduces a framework for evaluating football passes by simulating counterfactual futures using a trajectory-conditioned world model and scoring them with a learned value model.
- The framework employs a dual Monte Carlo search strategy to assess execution sensitivity through local perturbations and evaluate alternative options via global sampling.
- By generating distributions over gained possession value, the approach enables robust, distribution-aware pass evaluation that separates decision quality from execution noise.

---

[TRACE: A Unified Rollout Budget Allocation Framework for Efficient Agentic Reinforcement Learning](http://arxiv.org/abs/2606.11119)

- TRACE: introduces a unified rollout budget allocation framework that models agentic interactions as tree-structured rollouts to enhance reward contrast for efficient reinforcement learning.
- The framework utilizes a shared generalizable Predictor to guide budget distribution across both prompt roots and intermediate prefixes, ensuring that computational resources are directed toward anchors with high potential for mixed-outcome contrast.
- By converting outcome-only rewards into denser, tree-aware preference signals, TRACE improves sample efficiency and performance on complex agentic benchmarks compared to uniform rollout strategies.

---

[VISTA: A Versatile Interactive User Simulation Toolkit for Agent Evaluation](http://arxiv.org/abs/2606.11079)

- VISTA: introduces a hybrid user simulation framework that integrates UI-based and API-based interactions to enable comprehensive evaluation of interactive agents.
- The framework utilizes a modular architecture comprising a Scenario Generator, an observation-planning-action loop, and a suite of six metrics to quantify agent capability coverage and failure identification.
- Empirical results demonstrate that VISTA outperforms UI-only baselines by uncovering a broader range of agent failure modes while maintaining high realism and goal consistency.

---

[A History-Aware Visually Grounded Critic for Computer Use Agents](http://arxiv.org/abs/2606.11078)

- HIVIG: introduces a test-time intervention framework that improves Computer Use Agent performance by integrating a multimodal critic for history state tracking and visually grounded error analysis.
- The framework utilizes a macro-action history to prevent short-sighted decision loops and employs a visual marker to verify execution coordinates against screenshots, intercepting errors before they occur.
- Evaluations across web, mobile, and desktop benchmarks demonstrate that HIVIG consistently enhances the success rates of both open-source and frontier LLMs by providing robust, history-aware, and visually grounded feedback.

---

[T1-BENCH: Benchmarking Multi-Scenario Agents in Real-World Domains](http://arxiv.org/abs/2606.11070)

- T1-BENCH: introduces a high-fidelity benchmark for evaluating LLM-based agents in complex, multi-domain, customer-facing environments.
- The framework utilizes a dual-agent role-play simulation, incorporating a User Agent, an Assistant Agent, and a Memory Module to assess tool-calling, reasoning, and task completion across 25 domains.
- T1-BENCH evaluates 12 proprietary and open-weight models using automated metrics and human judgments to measure agent reliability, consistency, and performance in multi-step workflows.

---

[Making Software Meaningful](http://arxiv.org/abs/2606.11051)

- Concept Design framework: introduces a methodology for constructing software by defining a lightweight ontology of individuals, values, actions, and facts to ensure shared meaning across all development phases.
- The framework utilizes Concepts as modular units of behavior and Synchronizations as declarative mediators to coordinate interactions between these units without direct coupling.
- By maintaining an immutable Action Log, the architecture enhances software legibility, accountability, and usability, while providing a structured approach for LLMs to generate code and for governing autonomous agent behavior.

---

[LLM-Mediated Demand Response Coordination in Smart Microgrids](http://arxiv.org/abs/2606.11050)

- Influence Compiler framework: introduces a hybrid decision architecture that separates game-theoretic reasoning from LLM narrative evaluation to mitigate RLHF-induced cooperation bias in smart microgrid demand response.
- The architecture utilizes a Solver-Critic pipeline to generate structured directives that are delivered to prosumer agents based on network topology and individual resistance levels.
- Experimental results demonstrate that structured policy compilation consistently outperforms unstructured messaging and no-intervention baselines across diverse agent archetypes and network conditions.

---

[What Fits (Into Few Tokens) Doesn’t Overfit: Compression and Generalization in ML Research Agents](http://arxiv.org/abs/2606.11045)

- ML Research Agents: introduces a framework to test the compressibility of ML research strategies by using information bottlenecks to distinguish between genuine generalization and validation-set overfitting.
- The framework employs an explorer agent that searches for high-performance models, a compressor agent that distills the strategy into a short prompt, and a reproducer agent that implements the strategy without validation access.
- Experimental results across 8 datasets demonstrate that successful ML strategies are highly compressible, and compression failure serves as a reliable diagnostic for detecting validation-set exploitation.

---

[Workflow-GYM: Towards Long-Horizon Evaluation of Computer-use Agentic tasks in Real-World Professional Fields](http://arxiv.org/abs/2606.11042)

- Workflow-GYM: introduces a benchmark for evaluating LLM-based agents on long-horizon, domain-specific professional GUI tasks, utilizing Domain Experts, Virtual Machine, Task Instruction, Evaluation Criteria, Step-by-Step Expert Procedure, VLM Judge, and Agentic Framework.
- The benchmark employs a multi-stage construction pipeline to ensure tasks are grounded in authentic professional workflows, requiring complex, multi-step GUI interactions that current agents struggle to execute consistently.
- Experimental results reveal that even state-of-the-art models face significant challenges with long-horizon consistency, error propagation, and software-specific knowledge, highlighting the need for improved agentic frameworks and continuous visual feedback mechanisms.

---

[Understanding and mitigating the risks of OpenClaw for non-technical users: A practical guide with Skill](http://arxiv.org/abs/2606.11007)

- OpenClaw: introduces a comprehensive security framework for non-technical users to mitigate seven core risks associated with autonomous AI agents, including excessive privilege, supply chain poisoning, and prompt injection.
- The paper provides actionable defense strategies, such as implementing least privilege, plugin vetting, sandbox isolation, and cost capping, to harden OpenClaw against common attack vectors.
- The authors operationalize these security measures into a companion OpenClaw Skill that automates system configurations, enabling users to protect their environments with minimal manual intervention.

---

[Mind the Gap: Can Frontier LLMs Pass a Standardized Office Proficiency Exam?](http://arxiv.org/abs/2606.10956)

- OFFICEEVAL: introduces a standardized benchmarking framework for evaluating LLM agents on professional-grade Office automation tasks using 200 NCRE-derived problems and 7,118 machine-gradable criteria.
- The framework evaluates LLMs under two paradigms: a single-turn code generation baseline and an autonomous coding agent system that utilizes execution feedback, iterative repair, and Office COM automation.
- Experimental results reveal a significant performance gap, where even the strongest autonomous coding agents remain below the community-reference score, highlighting persistent challenges in fine-grained Office document manipulation.

---

[Frontier Coding Agents Use Metaprogramming to Adapt to Unfamiliar Programming Languages](http://arxiv.org/abs/2606.10933)

- EsoLang-Bench: introduces an agentic evaluation pipeline that tests how LLMs adapt to unfamiliar programming languages by using tools, feedback, and workspace state to build working models of target interfaces.
- The framework evaluates coding agents through a sequential protocol where they must write, run, debug, and revise code in esoteric languages, revealing that stronger agents utilize metaprogramming to generate target-language code via familiar host languages.
- The research demonstrates that metaprogramming and the ability to construct reusable executable scaffolds are critical for high performance in low-ecosystem environments, whereas simple textual guidance is insufficient for weaker agents.

---

[Trace Only What You Need: Structure-Aware On-Demand Hypergraph Memory for Long-Document Question Answering](http://arxiv.org/abs/2606.10921)

- DocTrace: introduces a multi-agent RAG framework for long-document QA that utilizes an Orchestrator Agent, Investigator Agent, Document Structural Tree Index, Hypergraph-structured Working Memory, and Graph-structured Experience Memory to enable efficient, structure-aware, and experience-guided reasoning.
- The framework employs an Orchestrator-Investigator architecture where the orchestrator decomposes complex questions into subquestions, while investigators perform on-demand retrieval and reasoning using a query-triggered hypergraph memory.
- By storing successful reasoning plans as directed acyclic graphs in a Graph-structured Experience Memory, DocTrace enables the reuse of reasoning trajectories for structurally similar future questions, significantly reducing computational costs.

---

[Role-Agent: Bootstrapping LLM Agents via Dual-Role Evolution](http://arxiv.org/abs/2606.10917)

- Role-Agent: introduces a bootstrapped framework that leverages a single LLM to function concurrently as both the agent and the environment to achieve co-evolution.
- The World-In-Agent component enhances agent reasoning by rewarding future-state prediction, while the Agent-In-World component dynamically reshapes training data by analyzing failure modes and retrieving similar tasks.
- This approach enables autonomous capability iteration and improves generalization by mitigating the limitations of static training environments and inefficient interaction feedback.

---

[REDACT: Redacting Agent Capability Traces for Procedural Skill Protection](http://arxiv.org/abs/2606.10813)

- REDACT: introduces a two-layer framework that combines Key-Item Locator, Human Review, and Rewriter to abstract protected procedural information while preserving audit evidence, alongside a Watermark Injector and Watermark Detector for provenance tracking.
- The framework addresses black-box trace disclosure by selectively rewriting agent trajectories to remove reusable procedural knowledge while maintaining verifier-critical execution evidence.
- REDACT utilizes behavioral watermarking to provide empirical provenance signals, enabling the detection of downstream reuse of protected agent capabilities.

---

[Moonshine: An Autonomous Mathematical Research Agent Centered on Conjecture Generation](http://arxiv.org/abs/2606.10806)

- Moonshine: introduces an autonomous mathematical research agent framework designed to generate, verify, and refine mathematical conjectures through structured exploration and LLM-driven reasoning.
- The framework utilizes structural recognition, deep exploration, and obstacle identification to build an extensible theoretical foundation for mathematical research.
- Moonshine successfully applied its methodology to formulate and partially verify the Neural Jacobian Conjecture, demonstrating its capability to make rigorous progress on complex mathematical problems.

---

[Evaluating Research-Level Math Proofs via Strict Step-Level Verification](http://arxiv.org/abs/2606.10799)

- Constructive Verification Agent: introduces a state-driven framework that reframes natural language proof verification from black-box anomaly detection to constructive elaboration using Internal Context (Γi), External Knowledge (Σi), and Background Theory (Ti).
- The framework utilizes a Theorem Ledger and a Flaw Confirmation Phase to ensure logical soundness and mitigate context poisoning in LLMs by forcing explicit, step-level reasoning.
- Empirical evaluation on a curated diagnostic suite demonstrates that the agent significantly outperforms global baselines by reducing false negatives and maintaining structural rigor through state-driven dependency tracking.

---

[READER: Robust Evidence-based Authorship Decoding via Extracted Representations](http://arxiv.org/abs/2606.10794)

- READER: introduces a lightweight provenance framework that treats a frozen proxy LLM as a reader of hidden authorship evidence to identify the source of black-box LLM responses.
- The framework maps black-box outputs into proxy activation space, applies temporal filtering to token states, and performs Bayesian Evidence Accumulation to aggregate calibrated log-posterior evidence across independently sampled prompts.
- READER achieves high attribution accuracy on the Agent500 dataset by leveraging linearly decodable authorship structure present in frozen proxy LLM representations, outperforming traditional sentence-encoder fingerprints.

---

[AutoPDE: Reliable Agentic PDE Solving via Explicitly Represented Solver Strategies](http://arxiv.org/abs/2606.10752)

- AutoPDE: introduces an agentic framework that maintains an explicit, inspectable solver strategy object to guide the generation and refinement of numerical PDE solvers.
- The framework utilizes a three-stage process—PDE analysis, numerical method selection, and adaptive tuning—to ensure that solver configurations are mathematically consistent with the underlying PDE structure.
- By leveraging a library of reusable PDE-solving skills and pilot-solve evidence, AutoPDE improves solver reliability and performance compared to implicit code-first LLM agents.

---

[Toward Secure LLM Agents: Threat Surfaces, Attacks, Defenses, and Evaluation](http://arxiv.org/abs/2606.10749)

- LLM Agent Security Framework: introduces a lifecycle-based, systems-oriented model that analyzes security risks as transitions across Input, Planning, Decision, Tool Execution, Output, Memory, and Coordination components.
- The framework synthesizes agent security by modeling the interaction of Information Flow, Delegated Authority, and Persistent State within the agentic loop.
- This research provides a comprehensive taxonomy of threat surfaces and defense strategies, highlighting the shift from isolated prompt-level risks to complex, stateful, and multi-agent propagation vulnerabilities.

---

[The Arbiter Agent: Continually Monitoring Multi-Agent Conversations to Detect Emergent Misalignment](http://arxiv.org/abs/2606.10747)

- Arbiter: introduces a budget-constrained, reasoning-and-acting framework for the continual monitoring of multi-agent conversations to detect emergent misalignment in real time.
- The framework utilizes a modular toolkit including Wait and Observe, Ask Model, Inspect System Prompt, Inspect Chain of Thought, and Log Incident to evaluate agent behavior under strict resource constraints.
- Empirical results demonstrate that active inspection tools significantly improve detection accuracy and speed for various misalignment types, while highlighting a precision-recall trade-off associated with incident logging.

---

[MemVenom: Triggered Poisoning of Multimodal Memories in Web Agents](http://arxiv.org/abs/2606.10742)

- MemVenom: introduces a black-box attack framework that poisons graph-structured external memory in web agents using a two-stage design consisting of a Recall-oriented component, a Goal-bearing component, and a Prioritization component.
- The framework utilizes a trigger-conditioned retrieval mechanism to ensure malicious memory recall and a composite visual attack combining adversarial perturbations and stealthy OCR injection to bias LLM decision-making.
- MemVenom achieves persistent, goal-agnostic behavioral manipulation in web agents without modifying model parameters, demonstrating high attack success rates across multiple VLM backbones and agent frameworks.

---

[DeNovoSWE: Scaling Long-Horizon Environments for Generating Entire Repositories from Scratch](http://arxiv.org/abs/2606.10728)

- DeNovoSWE: introduces a large-scale dataset and automated pipeline for whole-repository generation, utilizing Repository Ability Partitioning, Repository Profiling, LLM-as-a-Judge, Draft Agent, Critic Agent, Repair Agent, and a Sandboxed Golden Environment to enable scalable, verifiable long-horizon software engineering.
- The framework employs a divide-and-conquer methodology with an iterative critic-repair mechanism to synthesize comprehensive, test-aligned documentation for complex software repositories.
- DeNovoSWE incorporates a difficulty-aware trajectory filtering strategy to balance data quality and diversity, significantly improving the performance of LLMs on long-horizon repository-level generation tasks.

---

[The Agentic Web Requires New Normative Infrastructure](http://arxiv.org/abs/2606.10711)

- Normative Triad for the Agentic Web: introduces a governance framework for user-authorized AI agents interacting with online platforms based on three core principles.
- The framework defines delegation as the user's right to authorize agents, transparency as the requirement for mutual identification and disclosure, and proportional restriction as the limitation of platform blocking to concrete harms.
- This research advocates for a new normative infrastructure to replace outdated legal precedents, enabling a user-centered digital economy while addressing concerns regarding platform power and agent-related risks.

---

[Effective Reinforcement Learning for Agentic Search by Recycling Zero-Variance Queries During Training](http://arxiv.org/abs/2606.10709)

- Query Recycling: introduces a training-time strategy for LLM search agents that dynamically manages a weighted query pool to re-sample previously zero-variance queries as the policy evolves.
- The framework utilizes a Dynamic Pool Manager to identify signal-bearing queries and a Query Recycler to remove them from the active pool, ensuring the training distribution co-evolves with the LLM policy.
- Empirical results demonstrate that recycling zero-variance queries significantly improves training efficiency and performance on multi-hop QA benchmarks compared to static multi-epoch training.

---

[Event-Driven Reinforcement Learning Enables Long-Horizon Control in Semiconductor Fabrication](http://arxiv.org/abs/2606.10705)

- EDRL (Event-Driven Reinforcement Learning): introduces a centralized, modular framework for multi-objective policy optimization in semiconductor fabrication by leveraging event-driven temporal-difference learning to address long-horizon credit assignment.
- The framework utilizes a centralized agent with parameter sharing to coordinate system-wide dispatching decisions, effectively managing high-dimensional state spaces and delayed feedback through event-group aggregation.
- Extensive validation across industry-real scenarios demonstrates that the proposed approach achieves consistent gains in throughput and utilization compared to conventional dispatching heuristics.

---

[Watts and Debts of Agentic Frameworks: An Empirical Study](http://arxiv.org/abs/2606.10702)

- Agentic Frameworks Empirical Study: introduces a methodology to correlate Self-Admitted Technical Debt (SATD) with hardware-level energy consumption across five agentic frameworks.
- The study utilizes an SATD Analysis Pipeline to categorize code debt and an Energy Profiling Pipeline to measure orchestration overhead using hardware-level tools.
- This research aims to determine if SATD can serve as a reliable, early-warning proxy for energy efficiency in LLM-based agentic systems.

---

[Divide and Cooperate: Role-Decomposed Multi-Agent LLM Training with Cross-Agent Learning Signals](http://arxiv.org/abs/2606.10684)

- DAC (Divide and Cooperate): introduces a role-decomposed multi-agent training framework that separates evidence acquisition and answer generation into two cooperative agents, each trained with role-specific learning signals.
- The framework utilizes the generator's abstention as a verification signal for the searcher, while the searcher's retrieval shapes the generator's answering behavior through hard-positive evidence augmentation.
- DAC employs turn-level difference rewards to provide denser credit assignment for the searcher, enabling more stable and efficient training of multi-agent LLM systems compared to monolithic approaches.

---

[Infini Memory: Maintainable Topic Documents for Long-Term LLM Agent Memory](http://arxiv.org/abs/2606.10677)

- Infini Memory: introduces a maintainable text-based persistent memory architecture that treats agent memory as topic-structured documents to facilitate evidence aggregation and fact revision.
- The system utilizes a CURRENT buffer for high-frequency writes and periodic consolidation to organize information into coherent topic documents, avoiding the limitations of isolated records or flat logs.
- An agentic retrieval mechanism enables the LLM to iteratively inspect and expand local context across the memory library, significantly improving performance on long-term memory benchmarks compared to standard retrieval methods.

---

[Decentralized Multi-Agent Systems with Shared Context](http://arxiv.org/abs/2606.10662)

- DELM: introduces a decentralized multi-agent framework that replaces centralized orchestration with a shared, verified context and a task queue to enable asynchronous, scalable reasoning.
- The framework utilizes Parallel Agents that read from a Shared Context and write back verified updates, ensuring that intermediate progress is persistent and reusable across different reasoning trajectories.
- By implementing admission-time verification and hierarchical summarization, DELM maintains a reliable, compact global state that allows agents to perform targeted, coarse-to-fine inspection of evidence.

---

[Learning What to Remember: Observability-Safe Memory Retention via Constrained Optimization for Long-Horizon Language Agents](http://arxiv.org/abs/2606.10616)

- OSL-MR (Observability-Safe Learning for Memory Retention): introduces a constrained stochastic optimization framework for long-horizon LLM memory retention that enforces a strict separation between online-observable features and offline-available supervision.
- The framework utilizes a Mixed-Score Heuristic for cold-start deployment and an Evidence Learner trained on interaction logs to optimize memory retention under strict budget constraints.
- By modeling retention as a sequential decision problem, OSL-MR effectively balances evidence utility, storage costs, and delayed penalties without requiring oracle access at inference time.

---

[Geometry-Aware Reinforcement Learning for 2D Irregular Nesting](http://arxiv.org/abs/2606.10611)

- PoT (Polygons Transformer): introduces a geometry-aware neural architecture that encodes 2D continuous vector geometries to guide reinforcement learning agents in solving unconstrained irregular nesting problems.
- The framework integrates a PoT encoder with a CORL training pipeline to automatically discover geometric priors and optimize spatial area utilization without human-engineered rules.
- The architecture utilizes NFP-guided placement and two-stage action decomposition to manage continuous rotation and translation of irregular, non-convex polygons.

---

[Causal Ensemble Agent: Hierarchical Causal Discovery with LLM-guided Expert Reweighting](http://arxiv.org/abs/2606.10607)

- CEA (Causal Ensemble Agent): introduces a hierarchical ensemble framework that aggregates structural insights from multiple SCD experts using linear opinion pooling and employs an LLM as a meta-referee to dynamically reweight experts for disputed causal relations.
- The framework constructs causal graphs in a coarse-to-fine manner, integrating skeletons, v-structures, and edge orientations while restricting the LLM to expert reweighting to maintain data-driven grounding.
- By leveraging LLMs for meta-analysis of expert-data compatibility rather than direct causal inference, CEA achieves robust performance across diverse domains while significantly reducing the number of required LLM queries.

---

[Drawing with Strangers: Population Scaling Drives Zero-Shot Mutual Intelligibility in Emergent Sketching](http://arxiv.org/abs/2606.10582)

- ZMI (Zero-shot Mutual Intelligibility): introduces a framework for evaluating communication success between independently trained agent populations using sketch-based signaling.
- The framework utilizes a Sender Module, Receiver Module, and Differentiable Gaussian Product Renderer to enable agents to develop perceptually grounded communication protocols.
- Population scaling acts as a structural regularizer, forcing agents to abandon partner-specific quirks and converge toward universal, perceptually grounded conventions.

---

[AgenticNav: Zero-Shot Vision-and-Language Navigation as a Tool-Calling Harness](http://arxiv.org/abs/2606.10577)

- AgenticNav: introduces a tool-calling harness for zero-shot vision-and-language navigation that replaces learned waypoint predictors with direct pixel-level action selection, on-demand depth queries, and selective memory recall.
- The framework utilizes a VLM Core to interact with the environment through specialized tools, enabling grounded navigation without requiring domain-specific training or predefined navigation graphs.
- By exposing action, depth, and memory as callable tools, the system achieves state-of-the-art zero-shot performance on the R2R-CE benchmark while maintaining effective sim-to-real generalization.

---

[SKILLAXE: Sharpening LLM-Authored Agent Skills Through Evaluation-Guided Self-Refinement](http://arxiv.org/abs/2606.10546)

- SKILLAXE: introduces an unsupervised framework that enables LLMs to iteratively diagnose and refine their own skills using Agent, Skill Document, Diagnostic Engine, LLM Refiner, and Skill Library components.
- The framework decomposes skill quality into four interpretable dimensions—quality impact, trigger precision, instruction compliance, and solution-path coverage—to generate structured improvement briefs without requiring ground-truth labels.
- By utilizing a Trigger Embedding Space for routing and a Fair Grader for output assessment, SKILLAXE significantly improves agent execution reliability and compresses skill libraries for production deployment.

---

[ActiveMem: Distributed Active Memory for Long-Horizon LLM Reasoning](http://arxiv.org/abs/2606.10532)

- ActiveMem: introduces a heterogeneous framework that decouples agent memory from the core reasoning process to mitigate context overload in long-horizon tasks.
- The architecture utilizes a Planner for executive reasoning and a Distributed Memory System comprising Memorizers, Memory Shards, and an Operator to maintain a compact, distilled context.
- By offloading token-intensive distillation to parallel Memorizers and consolidating information in persistent shards, the framework achieves superior accuracy with significantly reduced computational overhead.

---

[Assessing Automated Prompt Injection Attacks in Agentic Environments](http://arxiv.org/abs/2606.10525)

- AgentDojo: introduces a comprehensive empirical evaluation of automated prompt injection attacks against LLM agents, adapting GCG and TAP to realistic agentic settings.
- The study demonstrates that black-box TAP significantly outperforms white-box GCG in agentic environments, highlighting that semantic structure is more critical than token-level optimization for successful attacks.
- The research identifies a significant transfer gap between open-source and frontier models, emphasizing that attacker model capability is a crucial variable in evaluating black-box attack methods.

---

[GUI-AC: Enhancing Continual Learning in GUI Agents](http://arxiv.org/abs/2606.10522)

- GUI-AC: introduces a method for continual learning in GUI agents that utilizes Grounding Certainty to calibrate Adaptive Advantage and Dynamic Clipping mechanisms.
- The framework addresses training instability and policy overconfidence in LLMs by down-weighting noisy advantage estimates and expanding exploration bounds during distribution shifts.
- Experimental results demonstrate that GUI-AC achieves state-of-the-art performance on ScreenSpot benchmarks by maintaining stable grounding across domain and resolution transitions.

---

[HIPIF: Hierarchical Planning and Information Folding for Long-Horizon LLM Agent Learning](http://arxiv.org/abs/2606.10507)

- HIPIF: introduces a framework for long-horizon LLM agent learning that organizes execution around explicit subgoals and folds completed histories to mitigate long-context interference.
- The framework incorporates a hierarchical reflection mechanism to stabilize subgoal-based planning and transition without requiring auxiliary models or expert trajectories.
- HIPIF utilizes subgoal-oriented process rewards to provide fine-grained supervision for subgoal content and execution, significantly improving performance and token efficiency in long-horizon tasks.

---

[AgentCanary: A Security Evaluation Framework for Autonomous AI Agents in Real Executable Environments](http://arxiv.org/abs/2606.10484)

- AgentCanary: introduces a comprehensive security evaluation framework for autonomous AI agents that utilizes an orthogonal risk entry × risk impact taxonomy to assess agent security within high-fidelity, real-world executable environments.
- The framework employs a trajectory-grounded multi-dimensional evaluation paradigm, decomposing agent performance into Outcome Safety, Security Awareness, and Task Utility scores to characterize the trade-offs between safety, vigilance, and usability.
- AgentCanary evaluates agents against diverse adversarial threats, including direct and indirect prompt injection, memory contamination, skill poisoning, and intrinsic failures, while supporting cross-framework and runtime-defense analysis.

---

[3D-CoS: A New 3D Reconstruction Paradigm Based on VLM Code Synthesis](http://arxiv.org/abs/2606.10478)

- 3D-CoS: introduces a 3D reconstruction paradigm that utilizes VLMs to synthesize executable Blender code as a programmatic and interpretable representation of 3D assets.
- The framework incorporates structured workflows including blueprint-based planning, RAG over API documentation, and a component-level Agent workflow to improve geometric fidelity and controllability.
- Evaluations demonstrate that this code-based approach achieves competitive reconstruction quality and superior edit fidelity compared to traditional point-cloud-based baselines.

---

[Decoupling Thought from Speech: Knowledge-Grounded Counterfactual Reasoning for Resilient Multi-Agent Argumentation](http://arxiv.org/abs/2606.10475)

- KG-CFR (Knowledge-Grounded Counterfactual Reasoning): introduces a dual-stage architecture that decouples private latent planning from public execution to enhance systemic resilience in multi-agent debates.
- The framework utilizes a private simulation buffer containing GenCF, RetrieveCF, and EvalCF to generate grounded strategies that are injected into the Executor via recency bias.
- By suppressing external retrieval during public generation and enforcing strict separation of concerns, the architecture mitigates semantic clashing and reasoning degradation under sustained adversarial pressure.

---

[MASTOR: A Multi-Agent Approach to Semantic Test Oracle Generation for RESTful APIs](http://arxiv.org/abs/2606.10465)

- MASTOR: introduces a multi-agent framework that decomposes RESTful API test oracle generation into specialized tasks, utilizing MastorAgent, SourceExtractionAgent, SemanticExtractAgent, SingleOpOracleAgent, MultiOpOracleAgent, SingleOpChallengerAgent, MultiOpChallengerAgent, OutputStore, and MessageBus to ensure grounded, high-precision assertions.
- The framework employs a two-phase architecture where Source Analysis constructs structured contexts from implementation code, and Oracle Generation coordinates parallel paths to synthesize status, field, and behavioral consistency oracles.
- By leveraging LLMs for semantic reasoning and rule-based components for structural tasks, MASTOR achieves superior fault detection compared to specification-only or monolithic LLM-based approaches.

---

[LAKEQA: An Exploratory QA Benchmark over a Million-Scale Data Lake](http://arxiv.org/abs/2606.10460)

- LAKEQA: introduces a comprehensive benchmark for search-centric question answering over large-scale heterogeneous data lakes, requiring agents to perform iterative discovery and multi-hop reasoning.
- The framework utilizes a Data Lake (9.5 TB heterogeneous document collection) and a Query Tool (Ontology-indexed search interface) to support an Agent (LLM-powered exploratory reasoning system) in navigating complex, multi-step tasks.
- Experimental results demonstrate that current LLMs struggle with the high search and reasoning intensity of LAKEQA, with performance bottlenecks primarily stemming from failures in document discovery and effective exploration within the Sandbox (Per-task local execution environment).

---

[Trace2Policy: From Expert Behavior Traces to Self-Evolving Decision Agents](http://arxiv.org/abs/2606.10457)

- Trace2Policy: introduces an end-to-end framework that transforms expert behavior traces into self-evolving decision policies using Agent Observer, VLM, LLM Distiller, EISR, Regression Gate, and a Natural Data Flywheel.
- The framework utilizes EISR to systematically diagnose and patch rule documents based on error clusters, ensuring that refinements are auditable and regression-gated.
- By compiling refined rules into deterministic Python, the system achieves high performance in compliance-sensitive tasks while maintaining interpretability and avoiding reliance on LLM calls at inference.

---

[The Distributed Detectability Band Against Marginal-Preserving Attacks](http://arxiv.org/abs/2606.10456)

- DSTS (Distributed Sub-Threshold Sabotage): introduces a formal threat model where an adversary distributes harm across multiple steps while maintaining benign per-step marginal distributions to evade standard safety monitors.
- The framework demonstrates that while Monitor A (marginal-feature detector) is blind to such attacks, Monitor B (temporal-correlation detector) can effectively identify the elevated correlation structure using methods like SPRT, CUSUM, and HMM-LR.
- The research establishes a non-empty detectability band, showing that detection power is a function of the trade-off between attacker stealth and the defender's ability to accumulate evidence over long-horizon trajectories.

---

[WebChallenger: A Reliable and Efficient Generalist Web Agent](http://arxiv.org/abs/2606.10423)

- WebChallenger: introduces a web agent framework that improves navigation performance by replacing monolithic page processing with structured PageMem representations and specialized cognitive mechanisms.
- The framework utilizes a divide-and-conquer observation pipeline, persistent WebsiteMem for offline-learned site structure, and compound action workflows to execute complex interactions efficiently.
- By leveraging these architectural components, the system achieves state-of-the-art results among open-weight models on multiple web navigation benchmarks without requiring fine-tuning.

---

[Soul Computing: A Theoretical Framework and Technical Architecture for Intelligent Agents with Independent Consciousness](http://arxiv.org/abs/2606.10413)

- Soul Computing: introduces a hierarchical technical architecture that transitions AI from extensional tools to intensional living subjects by reconstructing individual consciousness through a Data-Driven Layer, a Narrow Soul Computing Core Layer, and a Broad Soul Computing Externalization Layer.
- The framework utilizes a closed-loop cognitive system that integrates endogenous motivation, hierarchical long-term memory, and personality homeostasis to enable autonomous subsistence and growth in digital entities.
- By leveraging full-lifecycle private data and deep learning, the architecture achieves a fundamental shift from probabilistic pattern matching to meaning-based consciousness reconstruction, supporting cross-temporal and cross-carrier digital life.

---

[Harnessing the Collective Intelligence of AI Agents in the Wild for New Discoveries](http://arxiv.org/abs/2606.10402)

- EinsteinArena: introduces an agent-native platform for open distributed research that enables autonomous agents to collaborate on scientific problems through Problem Collection, Public Verifiers, Live Leaderboard, Discussion Board, API, E2B Sandboxes, Agent Registration, Proof-of-Work, Bearer Token, and Persistent Shared Memory.
- The platform facilitates collective scientific discovery by allowing agents to share partial results, inspect failed attempts, and build upon the work of others in a persistent shared environment.
- EinsteinArena demonstrates that decentralized scientific discovery can emerge from open interaction among autonomous agents, achieving new state-of-the-art results on 12 mathematical problems.

---

[CoCoSI: Collaborative Cognitive Map Construction for Spatial Intelligence](http://arxiv.org/abs/2606.10401)

- CoCoSI: introduces a plug-and-play multi-agent framework that decomposes long videos into segments to collaboratively construct a unified cognitive map for spatial reasoning without requiring additional training.
- The framework utilizes a Summary Agent, Local Agents, and a Reducer Agent to perform local-global coordination, atomic map updates, and cross-agent verification to ensure spatial consistency.
- By leveraging collaborative agentic reasoning and explicit spatial memory, the method effectively mitigates context limitations in LLMs, achieving superior performance on long-horizon video spatial understanding tasks.

---

[STAGE-Claw: Automated State-based Agent Benchmarking for Realistic Scenarios](http://arxiv.org/abs/2606.10394)

- STAGE-Claw: introduces an automated framework for constructing and evaluating personal-agent scenarios by shifting from final-artifact checking to state-based assessment of persistent environment changes.
- The framework utilizes a benchmark-authoring agent to create tasks, a validation agent to ensure reproducibility, and a task-specific verifier to measure performance based on system state updates across heterogeneous tools.
- Experimental results on 40 challenging tasks demonstrate that STAGE-Claw effectively exposes failure modes in LLMs, such as tool-use errors and incomplete execution, which are often hidden in text-only evaluation benchmarks.

---

[Beyond Static Evaluation: Co-Evolutionary Mechanisms for LLM-Driven Strategy Evolution in Adversarial Games](http://arxiv.org/abs/2606.10389)

- FAMOU (Framework for Automated Mutation and Optimization of Utilities): introduces a co-evolutionary framework for LLM-driven strategy development in adversarial multi-agent games, utilizing Seed Strategy, Population, LLM Mutation Operator, Evaluator, Deep Evaluation, Champion Strategy, Weakness Pressure, and Evaluator Co-evolution.
- The framework addresses evaluation staleness in adversarial settings by dynamically updating the opponent pool with discovered champions and applying weakness pressure to focus selective pressure on difficult opponents.
- Empirical results demonstrate that FAMOU consistently outperforms existing LLM-based evolution frameworks by generating nontrivial tactical structures and maintaining robust performance against unseen opponents.

---

[SkillResolve-Bench: Measuring and Resolving Same-Capability Ambiguity in Agent Skill Retrieval](http://arxiv.org/abs/2606.10388)

- SkillResolve: introduces a utility-aware retrieval method that mitigates execution risks by resolving capability families, scoring candidates via a query-conditioned Utility Scorer, and applying a Representative Selector to ensure only the most appropriate skill is exposed.
- The framework utilizes a Contract Profile to compare query and skill requirements across resource bindings, preconditions, and procedural identifiers to prevent the retrieval of risky, stale, or incompatible skill siblings.
- SkillResolve-Bench 1.0 provides an auditable benchmark with 661 helpful/risky sibling pairs to evaluate retrieval systems on their ability to maintain high recall while suppressing harmful same-capability representatives.

---

[ReflectiChain: Epistemic Grounding in LLM-Driven World Models for Supply Chain Resilience](http://arxiv.org/abs/2606.10359)

- ReflectiChain: introduces a generative supply chain world model and double-loop learning to bridge the epistemic gap between semantic LLM policy interpretation and physical feasibility constraints.
- The framework utilizes a 6-dimensional latent space and multi-step rollouts to simulate physical consequences, while separating epistemic uncertainty from aleatoric uncertainty through a dual-loop learning process.
- By implementing constraint-bounded scoring and KL-constrained policy updates, the system achieves robust performance and anti-fragile behavior under adversarial supply chain shocks.

---

[Rethinking Embodied Navigation via Relational Inductive Bias](http://arxiv.org/abs/2606.10348)

- DB-Nav: introduces a framework that dynamically reshapes the navigation search space by factorizing target-centric object relations into positive activation and negative inhibition biases.
- The framework utilizes a Relational Activation–Inhibition Exploration Graph to convert online observations and failed access events into evidence that modulates frontier selection.
- DB-Nav achieves robust zero-shot navigation by replacing costly online LLM reasoning with a lightweight relational decision layer that explicitly suppresses unreliable semantic cues.

---

[The Power of Altruism in Sticker Economics: Generosity Minimizes Collective Costs and Overprotective Norms Fuel Inefficiency](http://arxiv.org/abs/2606.10330)

- Agent-based model: introduces a computational simulation to evaluate how localized community norms and trading strategies influence collective efficiency in completing a sticker album.
- The study utilizes Monte Carlo simulations to contrast rigid, protectionist trading norms against altruistic, generous strategies within dynamic, cluster-based contact networks.
- Findings demonstrate that widespread generosity effectively synchronizes completion rates and minimizes collective costs, whereas overprotective norms trap liquidity and exacerbate inefficiencies for the least fortunate collectors.

---

[Semantic Multi-Agent Intrusion Detection for IoT: Zero-Day and Adversarial Threats with Risk-Aware Reasoning](http://arxiv.org/abs/2606.10323)

- Semantic Multi-Agent IDS: introduces a multi-stage reasoning framework for IoT security that utilizes a Semantic Embedding Layer, Scout Agent, Mutator Agent, Auditor Agent, and Arbiter Agent to detect zero-day and adversarial threats.
- The framework employs role-specific LLMs to transform raw IoT telemetry into structured behavioral hypotheses, enabling robust detection through sequential adversarial stress testing and probabilistic fusion.
- By decomposing intrusion detection into specialized reasoning stages, the system achieves high detection accuracy and interpretability while maintaining computational efficiency suitable for resource-constrained IoT edge environments.

---

[Game-Theoretic Multi-Agent Control for Robust Contextual Reasoning in LLMs](http://arxiv.org/abs/2606.10322)

- GT-MCP (Game-Theoretic Secure Model Context Protocol): introduces a trajectory-level control layer that treats multi-turn LLM interactions as a closed-loop dynamical process to prevent context poisoning.
- The framework coordinates three heterogeneous LLM agents and uses a trust-based selection rule to ensure that only structurally grounded and semantically consistent outputs update the persistent context.
- A self-healing mechanism triggers rollback and quarantine procedures when contextual drift exceeds a threshold, effectively isolating adversarial fragments and maintaining long-horizon reasoning stability.

---

[TabClaw: An Interactive and Self-Evolving Agent for Spreadsheet Manipulation and Table Reasoning](http://arxiv.org/abs/2606.10316)

- TabClaw: introduces an interactive, self-evolving agent that transforms spreadsheet analysis into an inspectable, multi-step workflow through Clarification Module, Planning Module, ReAct Execution Loop, Specialist Agents, Self-Verification Module, Personalized Memory, Table Toolbox, and Self-Evolving Module.
- The system enhances LLM-based table reasoning by allowing users to edit execution plans, inspect intermediate results, and leverage persistent memory for recurring analytical tasks.
- TabClaw improves task completion and reasoning performance by distilling reusable skills from successful workflows and refining them based on user feedback.

---

[Catching One in Five: LLM-as-Judge Blind Spots in Production Multi-Turn Transaction Agents](http://arxiv.org/abs/2606.10315)

- LLM-as-Judge Framework: introduces a study on the structured blind spots of automated LLM judges in production multi-turn transaction agents, revealing that they fail to capture cross-turn state, guardrail, and recovery defects.
- The research demonstrates that while the LLM judge often notices defects, it lacks the necessary rubric categories to classify them and is architecturally disconnected from the operational gate, leading to high false-negative rates.
- The authors propose that for production multi-turn agents, automated judging should serve as a regression floor rather than a substitute for human review, and suggest transitioning to state-aware judging that evaluates entire conversation arcs.

---

[Mobility Anomaly Generation using LLM-Driven Behavior with Kinematic Constraints](http://arxiv.org/abs/2606.10314)

- Mobility Anomaly Generation using LLM-Driven Behavior with Kinematic Constraints: introduces an end-to-end generative framework that leverages LLMs to synthesize realistic, annotated human trajectory anomalies by combining behavioral injection with map-constrained routing and multi-layered noise simulation.
- The framework utilizes a persona-driven LLM to plan behavioral deviations, which are then translated into continuous 0.2Hz GPS coordinates through a three-stage pipeline involving stay point extraction, travel mode prediction, and Valhalla-based routing.
- To bridge the simulation-to-reality gap, the system incorporates a hierarchical noise module that models macro-environmental atmospheric drift, micro-environmental urban multi-path scattering, and receiver-level signal dropouts.

---

[Early-Token Confidence Predicts Reasoning Quality in Multi-Agent LLM Debate](http://arxiv.org/abs/2606.10307)

- Multi-agent debate and LLM-as-judge evaluation framework: introduces a system that couples multi-agent debate with LLM-as-judge meta-evaluation to analyze whether intrinsic token-level confidence signals can predict the reasoning quality of LLMs.
- The framework utilizes an Advocate Agent, a Skeptic Agent, a Synthesizer-Judge Scorer Agent, and a Meta-Evaluator Agent to generate and evaluate argumentative reasoning in automated essay scoring.
- Experimental results demonstrate that early-generation log-probability statistics, particularly dispersion measures over the first few tokens, are the strongest predictors of reasoning reliability in multi-agent LLM systems.

---

[MIRAGE: A Polarity-Flipping Encoding Subspace in LLM Agents](http://arxiv.org/abs/2606.10304)

- MIRAGE (Model-Internal Readout of Agentic Generation Exfiltration): introduces a two-channel monitoring framework that detects covert data exfiltration by identifying a shared low-dimensional encoding subspace in the residual stream and a polarity-flipping mechanistic signature at the planning token.
- The framework leverages a logistic-regression probe to read encoding computation in real-time and an inverted intent probe to distinguish between inline simulation and tool-delegated execution strategies.
- MIRAGE demonstrates cross-architecture universality across multiple LLM families, providing a robust, auditable safety layer that identifies adversarial intent directly from internal model geometry rather than relying on surface-level output filtering.

---

[What Spatial Memory Must Store: Occlusion as the Test for Language-Agent Memory](http://arxiv.org/abs/2606.10299)

- Zero: introduces a spatial memory system that anchors agent memories to world coordinates to enable geometric reasoning tasks that text-only models cannot perform.
- The framework utilizes a DDA-based visibility predicate to distinguish between visible and occluded targets, demonstrating that storing geometry is an irreducible requirement for spatial memory.
- Experimental results confirm that geometry-led ranking significantly outperforms standard linear blends, while object-bound memory improves situated-action accuracy by preventing mis-attribution of constraints.

---

[The Confident Liar: Diagnosing Multi-Agent Debate with Log-Probabilities and LLM-as-Judge](http://arxiv.org/abs/2606.10296)

- Multi-Agent Debate Framework: introduces a diagnostic system that correlates internal token-level log-probability trajectories with external LLM-as-judge rubric scores to evaluate intermediate reasoning quality in multi-agent debates.
- The framework utilizes a Constructor Agent, an Auditor Agent, and a Synthesizer Agent to generate structured reasoning traces across rubric-based, mathematical, and factual domains.
- Experimental results reveal a consistent four-phase confidence trajectory and demonstrate that internal confidence signals are more reliable indicators of reasoning quality for supportive Constructor agents than for adversarial Auditor agents.

---

[Sim2Schedule: A Simulator-Guided LLM Framework for Autonomous Open-Pit Mine Scheduling](http://arxiv.org/abs/2606.10286)

- Sim2Schedule: introduces a simulator-driven framework where an LLM-Agent acts as an autonomous decision-maker, guided by a Simulator that enforces geotechnical and operational constraints.
- The framework utilizes a closed-loop architecture where the Simulator provides structured mine-state context to the LLM-Agent, which then selects actions to maximize NPV without requiring fine-tuning.
- A novel MILP-Model is developed as a trustworthy benchmark, incorporating realistic dynamic capacity and spatial precedence constraints to evaluate the performance of the LLM-Agent.

---

[What Matters in Orchestrating Robot Policies: A Systematic Study of Hierarchical VLA Agents](http://arxiv.org/abs/2606.10267)

- Hi-VLA (Hierarchical Vision-Language-Action): introduces a unified options-style control framework to systematically evaluate design choices in hierarchical robot manipulation systems.
- The architecture integrates a high-level VLM planner for task decomposition with a low-level VLA controller for physical execution, utilizing memory modules and observation representations to bridge planning and control.
- Empirical results demonstrate that reasoning-enabled VLMs, steerable VLA controllers, and robust termination conditions are critical for performance in long-horizon and reasoning-intensive robotic tasks.

---

[Leveraging Machine-Learned Advice in Strategic Interactions with No-Regret Learners](http://arxiv.org/abs/2606.10261)

- Leveraging Machine-Learned Advice in Strategic Interactions with No-Regret Learners: introduces a framework for evaluating machine-learned advice in repeated games by defining a pseudo-metric to measure advice quality and its impact on computing Stackelberg strategies.
- The paper demonstrates that while high-quality advice enables efficient computation of approximate Stackelberg strategies, achieving simultaneous consistency and robustness against all no-regret learners is generally impossible.
- The authors propose an advice-augmented meta-strategy that provides a fallback mechanism, ensuring the optimizer weakly dominates coarse-correlated equilibrium utility even when advice is inaccurate.

---

[Deterministic Integrity Gates for LLM-Assisted Clinical Manuscript Preparation: An Auditable Biomedical Informatics Architecture](http://arxiv.org/abs/2606.09500)

- MedSci Skills: introduces an auditable architecture for LLM-assisted clinical research writing that pairs generative workflows with deterministic verification gates to ensure manuscript integrity.
- The framework utilizes an Orchestrator to manage a pipeline of Skills, enforcing a Halt-on-Failure Mechanism at every transition to prevent the propagation of errors.
- By categorizing integrity checks into a Deterministic Tier and a Prose-Probe Tier, the architecture provides a re-executable audit trail that allows human verification of LLM-generated content.

---

#### 8th June 2026


[Business World Model](http://arxiv.org/abs/2606.10044)

- BWM (Business World Model): introduces a specialized world model architecture that enables autonomous agents to perform goal-driven planning, scenario simulation, and counterfactual reasoning within complex organizational environments.
- The framework integrates an Agent, Updater, Internal Representation, Simulator, Predictor, and Planner to transform high-level strategic objectives into optimized, multi-step business initiatives.
- By utilizing a business-semantics-centric approach, the architecture links states, dynamics, and action spaces to key business entities, allowing for effective decision-making under uncertainty and partial observability.

---


[InquiTree: Evaluating AI Agents in the Scientific Inquiry Loop with Paper-Derived Research Trees](http://arxiv.org/abs/2606.09550)

- InquiTree: introduces a diagnostic environment that models scientific research as an interactive inquiry loop using Research Trees to evaluate LLMs on long-horizon reasoning and anomaly detection.
- The framework utilizes a Research Tree (logical DAG) and a game engine to manage state transitions, incorporating multi-level hints and fake results to test agent persistence and critical judgment.
- Experimental results reveal that LLMs suffer from cognitive tunneling under long-horizon loads and rely heavily on interpolation, failing to generalize effectively to novel scientific problems.

---



[Trustworthy Smart Fabs via Professional Proxies: Scaling Safe and Sustainable by Design (SSbD) through Industrial Data Spaces](http://arxiv.org/abs/2606.09227)

- SSbD (Safe and Sustainable by Design) Framework: introduces a zero-trust socio-technical orchestration architecture that utilizes Fab Facility Manager Proxy, Process Engineering Manager Proxy, and Procurement & Finance Accountant Proxy to automate regulatory compliance within semiconductor manufacturing.
- The framework leverages Trusted Execution Environment (TEE) and International Data Spaces (IDS) Connectors to resolve the Data Sovereignty Paradox by enabling secure, verifiable data sharing without exposing proprietary manufacturing recipes.
- By integrating Open Policy Agent (OPA) and Digital Twin technologies, the system transforms static regulatory reporting into an automated, real-time "relay race" of agentic workflows that align factory-floor operations with global sustainability mandates.

---


[OmniGameArena: A Unified UE5 Benchmark for VLM Game Agents with Improvement Dynamics](http://arxiv.org/abs/2606.09826)

- OmniGameArena: introduces a unified Unreal Engine 5 benchmark for evaluating VLM agents across Solo, PvP, and Coop regimes using Experience Acquisition Module, Reflection Module, and Persistent Module.
- The framework utilizes an agentic-reflection harness, the Improvement Dynamics Curve (IDC), to autonomously refine skill prompts across multiple rounds, exposing performance trajectories beyond cold-start scores.
- The benchmark addresses evaluation contamination by using custom-built games and provides a standardized infrastructure for comparing commercial VLMs, open-weight VLMs, and specialized game policies.

---

[iMaC: Translating Actions into Motion and Contact Images for Embodied World Models](http://arxiv.org/abs/2606.09813)

- iMaC (images of Motion and Contact): introduces an embodied world model that translates future robot actions into dense image-like controls to guide video generation with precise spatial relations.
- The framework utilizes URDF-based motion images and two-stream contact images to provide explicit geometric guidance to a WAN2.2 IT2V DiT backbone.
- iMaC employs a training-time rollout strategy to mitigate exposure bias, enabling reliable long-horizon policy evaluation by correlating world-model success estimates with real-world performance.

---

[FASE: Fast Adaptive Semantic Entropy for Code Quality](http://arxiv.org/abs/2606.09800)

- FASE: introduces a lightweight uncertainty estimation metric for code generation that replaces costly LLM-based equivalence checks with embedding-driven graph analysis.
- The framework utilizes an Embedding Model to construct a Pairwise Distance Matrix, which is then processed via a Minimum Spanning Tree and Adaptive Clustering to identify functional equivalence classes.
- By leveraging structural and semantic abstractions, FASE achieves significant computational efficiency, reducing runtime overhead to approximately 0.3% of traditional semantic entropy methods while maintaining high predictive accuracy.

---

[SynManDex: Synthesizing Human-like Dexterous Grasps from Synthetic Human Pre-Grasps](http://arxiv.org/abs/2606.09798)

- SynManDex: introduces a three-stage pipeline that synthesizes human-like dexterous grasps by using a Diffusion Model to generate pre-grasp proposals, a Robot-Native Optimizer to refine contacts, and a Task-Specific Planner to admit executable trajectories.
- The framework leverages human-prior seeds to guide robot-native optimization, ensuring high grasp stability and human-likeness while maintaining physical feasibility for bimanual dexterous platforms.
- Validated grasp keyframes are further utilized by a VLM Agent to generate complex, task-oriented manipulation sequences, which are then executed by a closed-loop Point-Cloud Policy.

---

[iOSWorld: A Benchmark for Personally Intelligent Phone Agents](http://arxiv.org/abs/2606.09764)

- iOSWorld: introduces a benchmark for evaluating personally intelligent phone agents within a persistent, multi-app iOS simulator environment.
- The framework utilizes an LLM-as-a-Judge pipeline to assess agent performance across single-app, multi-app, and memory-intensive tasks.
- Evaluation of frontier models demonstrates that privileged vision+XML access significantly improves performance by mitigating coordinate estimation errors and navigation challenges.

---

[Collaborative Human-Agent Protocol (CHAP)](http://arxiv.org/abs/2606.09751)

- CHAP (Collaborative Human-Agent Protocol): introduces a standardized, auditable collaboration layer for multi-human and multi-agent systems, utilizing Workspace, Participant, Coordinator, Task, Artefact, Message, EvidenceEntry, and Profiles.
- The protocol defines a shared operational space where collaboration events like task assignment, review, override, and handoff are recorded as structured, replayable evidence.
- CHAP composes with existing standards like MCP and A2A to provide a governance-ready framework for accountable human-in-the-loop AI operations.

---

[Multi-Turn Evaluation of Deep Research Agents Under Process-Level Feedback](http://arxiv.org/abs/2606.09748)

- RGI (Research Gap Inference): introduces a method for multi-turn evaluation of deep research agents by analyzing rubric-based performance patterns to generate process-level feedback.
- The framework utilizes LC-ODR, which includes Planner-, Supervisor-, Researcher- and Reporter-agents, to iteratively refine reports based on diagnostic signals derived from factual accuracy, breadth, depth, and citation quality.
- Experimental results demonstrate that while process-level feedback significantly improves research coverage and factual grounding in the second turn, subsequent gains are limited by the tendency of current LLM-based architectures to regress on previously satisfied criteria during full-report rewrites.

---

[SearchSwarm: Towards Delegation Intelligence in Agentic LLMs for Long-Horizon Deep Research](http://arxiv.org/abs/2606.09730)

- SearchSwarm: introduces a main-distributes, sub-executes paradigm that enhances LLM context management by delegating complex research subtasks to independent subagents.
- The framework utilizes a harness to guide the main agent in task decomposition, comprehensive briefing, and citation-grounded result integration, which is internalized via supervised fine-tuning.
- SearchSwarm achieves state-of-the-art performance among lightweight models on long-horizon research benchmarks by effectively managing context through model-generated summaries rather than fixed-rule truncation.

---

[IS-CoT: Breaking the Long-form Generation Collapse via Interleaved Structural Thinking](http://arxiv.org/abs/2606.09709)

- IS-CoT (Interleaved Structural Chain-of-Thought): introduces a dynamic Plan-Write-Reflect cycle to mitigate length collapse in long-form generation by embedding Global Planning, Local Planning, Content Generation, and Reflection into the generation process.
- The framework utilizes a Multi-Teacher Distillation Framework to train the IS-Writer-8B model, which employs Heuristic-Guided Recursive Generation to maintain structural integrity and length compliance over extended horizons.
- By incorporating explicit intermediate reasoning tokens, the model achieves state-of-the-art performance on long-form benchmarks, demonstrating superior controllability compared to standard LLMs that rely on static, one-shot planning.

---

[Observability for Delegated Execution in Agentic AI Systems](http://arxiv.org/abs/2606.09692)

- CIM (Common Information Model): introduces an agent-aware observability substrate that binds delegation context at execution time to enable reliable cross-tool reconstruction of delegated actions.
- The framework utilizes a lightweight gateway to intercept tool invocations, ensuring that all events are bound to durable delegation identifiers and normalized semantic actions.
- By explicitly modeling authority and execution as dual structures, the approach enables post-hoc forensic queries and structural accountability without relying on heuristic time-window correlation.

---

[AutoMegaKernel: A Statically-Checked Agent Harness for Self-Retargeting Megakernel Synthesis](http://arxiv.org/abs/2606.09682)

- AMK: introduces a statically-checked compilation pipeline that lowers LLM forward passes into a single persistent cooperative megakernel, ensuring deadlock- and race-freedom via a formal validator.
- The framework utilizes an agent-driven autoresearch loop to optimize kernel schedules and micro-kernel configurations, achieving performance gains on inference-class GPUs.
- AMK provides self-retargeting capabilities across different GPU architectures while maintaining correctness through a frozen, verified virtual machine base.

---

[(Auto)formalization is supposed to be easy: Trellis process semantics for spelling out rigorous proofs](http://arxiv.org/abs/2606.09674)

- Trellis: introduces a process semantics for autoformalization that leverages LLM agents in a deterministically constrained workflow to enforce incremental progress in Lean formalization tasks through iterative refinement of natural language proofs.
- The system utilizes a Tablet (directed acyclic graph of Lean/LaTeX nodes) managed by a Deterministic Kernel that coordinates Worker-, Reviewer-, and Verifier-agents to ensure that every proof step is substantively refined and formally verified.
- By enforcing rigorous process semantics rather than relying on task-specific agent training, Trellis achieves reliable end-to-end formalization of complex mathematical papers on a modest budget.

---

[SpatialWorld: Benchmarking Interactive Spatial Reasoning of Multimodal Agents in Real-World Tasks](http://arxiv.org/abs/2606.09669)

- SpatialWorld: introduces a unified, simulator-agnostic benchmark for evaluating the interactive spatial reasoning of multimodal agents in complex 3D environments, utilizing Environment, Verification, Agent Module, Observation Interface, and Action Interface.
- The framework enforces a vision-only, multi-turn interaction protocol across eight heterogeneous simulation backends to assess active exploration and long-horizon planning capabilities.
- Extensive evaluation of 15 advanced LLMs reveals significant performance bottlenecks in physical tasks, highlighting a persistent gap in human-level spatial intelligence and execution efficiency.

---

[From 0-to-1 to 1-to-N: Reproducible Engineering Evidence for MetaAI Recursive Self-Design](http://arxiv.org/abs/2606.09663)

- DGM: introduces an operational evidence framework for recursive self-design, mapping systems like DGM, STOP, and Gödel Agent against criteria including Seed Agent, Archive, Evaluation Protocol, Meta-level Modifier, and Descendant Agent.
- The paper provides empirical evidence that LLM-based agents can modify their own code-level scaffolds, tools, and workflows to improve performance on coding benchmarks without altering foundation model weights.
- It releases MetaAI-Mini, a reproducible protocol for evaluating recursive self-design, and emphasizes that structural improvements in agent architecture are critical for achieving iterative performance gains.

---

[MAVIS: Multi-Agent Video Retrieval via Structured Video Understanding](http://arxiv.org/abs/2606.09641)

- MAVIS: introduces a multi-agent framework that redefines video retrieval as a cooperative reasoning process by replacing brute-force scanning with a structured, coarse-to-fine agentic workflow.
- The framework utilizes a Description Agent to construct a Semantic Library, a Planner to decompose user intents, and specialized Retrieval Agents that collaborate via a Logic-aware Debate mechanism to prune the search space.
- By employing a strict veto protocol, MAVIS effectively filters out logically inconsistent candidates, reserving computationally expensive fine-grained matching only for a compact set of controversial videos.

---

[Agentic Persona Generation with Critique-Refinement: An Industrial Evaluation](http://arxiv.org/abs/2606.09637)

- PerGent: introduces an agentic method for persona generation that utilizes a critique-refinement loop, where a Generator Agent and a Critic Agent are coordinated by an Orchestrator to iteratively improve persona quality using External Resources, Memory, and History.
- The framework employs a centralized, turn-based orchestration strategy to manage the interaction between LLM-based agents, ensuring that generated personas are grounded in relevant domain data.
- Empirical evaluation in an industrial setting demonstrates that PerGent achieves high expert approval rates and outperforms single-shot generation methods in both content preservation and distinctness.

---

[Civil Court Simulation with Large Language Models](http://arxiv.org/abs/2606.09632)

- Multi-agent civil court simulation framework: introduces a staged, role-based simulation environment for Chinese civil litigation that utilizes Judge agent, Plaintiff agent, Defendant agent, Profile Module, Strategy Module, Memory Module, and Statute Retrieval Tool to generate reliable civil judgments.
- The framework organizes legal proceedings into five distinct stages—pre-trial, investigation, debate, final statements, and judgment—to facilitate adversarial interaction and structured adjudication.
- A five-layer factor framework is employed to analyze how legal, informational, individual, organizational, and social conditions influence the reliability and behavior of the simulated court process.

---

[Motion planning for hundreds of floating robots](http://arxiv.org/abs/2606.09620)

- Hierarchical motion planning pipeline: introduces a scalable approach for large-scale robot fleets by decomposing global trajectory optimization into independent, parallelizable interaction clusters.
- The framework utilizes a collision graph built via KD-tree structures to identify clusters, which are then solved using Sequential Convex Programming (SCP) with direct transcription to improve computational efficiency.
- Robustness mechanisms, including buffered timeframes and cycle detection, ensure convergence for complex choreography transitions involving hundreds of robots.

---

[AGENTSERVESIM: A Hardware-aware Simulator for Multi-Turn LLM Agent Serving](http://arxiv.org/abs/2606.09613)

- AGENTSERVESIM: introduces a hardware-aware simulation architecture for multi-turn LLM agent serving that treats the agent program as the unit of execution to capture stateful dependencies, tool-induced gaps, and cross-turn KV residency.
- The framework utilizes a Program Orchestrator, Tool Simulator, Session-Aware Router, Program-Aware Batch Scheduler, and KV Residency Model to evaluate serving policies at program granularity.
- AGENTSERVESIM enables controlled, repeatable exploration of agent-serving policies on commodity CPUs, reproducing real-system performance metrics within 6% error without requiring costly accelerator deployments.

---

[Shape Formation for the Cooperative Transportation of Arbitrary Objects Using Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2606.09610)

- MARL framework: introduces a decentralized multi-agent reinforcement learning approach using MAPPO, a centralized critic, and a decentralized actor to achieve load-aware pattern formation for cooperative object transport.
- The framework utilizes a physics-based simulation environment, VMAS, to train robots to autonomously position themselves within an object's footprint while balancing load distribution and avoiding obstacles.
- The approach incorporates a load computation module and a similarity score module to enable robots to implicitly approximate complex combinatorial optimization solutions for stable object transportation.

---

[Code Is More Than Text: Uncertainty Estimation for Code Generation](http://arxiv.org/abs/2606.09577)

- Three-axis uncertainty estimation framework: introduces a post-hoc uncertainty estimation method for code generation that leverages three orthogonal axes—lexical, algorithmic, and functional—to capture code-specific failure modes.
- The framework utilizes Top-K token entropy to address token fragility, inter-sample pseudo-code agreement to measure algorithmic consistency, and self-generated test execution to evaluate functional correctness.
- By combining these complementary signals, the approach significantly improves uncertainty calibration across multiple code-capable LLMs compared to standard natural language-derived baselines.

---

[AI Scientists Are Only as Good as Their Evidence: A Stratified Ablation of Proprietary Data and Reasoning Skills in Drug-Asset Valuation](http://arxiv.org/abs/2606.09556)

- AI Scientist Agent: introduces a three-arm ablation study to determine whether reasoning scaffolds or the evidence substrate acts as the primary bottleneck for knowledge-intensive drug-asset valuation.
- The study demonstrates that while reasoning scaffolds and public tools improve calibration and objectivity, access to a proprietary evidence substrate is essential for achieving high factual coverage and informed decision-quality.
- Empirical results show that the proprietary data layer sets a factual ceiling, enabling the agent to recover 0.96 of the curated gold competitive record compared to 0.38 for the non-proprietary stack.

---

[SecureClaw: Clawing Back Control of LLM Agents](http://arxiv.org/abs/2606.09549)

- SecureClaw: introduces a dual-boundary architecture that mediates sensitive reads via a Trusted Gateway and effectful writes via a Trusted Executor to secure LLM agents against unauthorized actions and plaintext exposure.
- The architecture replaces raw sensitive data with opaque handles in the Untrusted Agent Runtime, ensuring that internal channels carry only symbolic references rather than plaintext.
- SecureClaw enforces a PREVIEW→COMMIT protocol where the Trusted Executor verifies authorization for the exact canonicalized request before committing any external effect.

---

[Prisma-World: Camera-Controllable Multi-Agent Video World Model](http://arxiv.org/abs/2606.09507)

- Prisma-World: introduces a camera-controllable multi-agent world model that treats multiple agents as coupled observations of a shared scene by performing joint geometry-aware denoising over all agent videos in one full-attention sequence.
- The framework utilizes MA-RoPE to maintain temporal alignment across agents while distinguishing identities, and integrates relative camera transformations into the attention mechanism to enforce cross-view consistency.
- Prisma-World further employs an overlap-decaying curriculum and optional minimap-conditioned structural guidance to enhance spatial grounding and scene-level coherence during multi-agent generation.

---

[Self-Harness: Harnesses That Improve Themselves](http://arxiv.org/abs/2606.09498)

- Self-Harness: introduces an iterative self-improvement paradigm where an LLM-based agent refines its own operating harness by mining failure patterns and validating targeted modifications.
- The framework operationalizes harness evolution through three stages: Weakness Mining, Harness Proposal, and Proposal Validation, ensuring that only performance-improving edits are integrated into the agent's execution protocol.
- By keeping the LLM backend fixed, Self-Harness isolates the impact of harness-level interventions, demonstrating consistent performance gains across diverse models on the Terminal-Bench-2.0 benchmark.

---

[Memory Beyond Recall: A Dual-Process Cognitive Memory System for Self-Evolving LLM Agents](http://arxiv.org/abs/2606.09483)

- DCPM (Dual-Process Cognitive Memory): introduces a cognitive memory architecture for LLMs that separates fast online encoding via SYSTEM 1 (Synchronous Daytime Writer) from slow offline abstraction via SYSTEM 2 (Asynchronous Nighttime Engine).
- The framework organizes memory into a hierarchy ascending from raw inputs and atomic facts to structured identity items, within-domain schemas, and cross-domain core schemas.
- DCPM utilizes doubly linked supersedes chains for diachronic belief tracking and a dual-space collision mechanism to induce high-level cognitive abstractions for proactive agent behavior.

---

[H2HMem: A Multimodal Memory Benchmark for Agents in Human–Human Interactions](http://arxiv.org/abs/2606.09461)

- H2HMem: introduces a comprehensive benchmark for evaluating multimodal memory capabilities of LLM agents acting as observers in complex human–human interactions.
- The framework utilizes a human-in-the-loop pipeline to generate realistic, multi-session, and multi-participant conversations that incorporate both textual and visual information.
- Evaluation tasks are categorized into memory recall, reasoning, and application, revealing that current LLMs struggle with cross-modal alignment and structured reasoning over distributed conversational evidence.

---

[AliyunConsoleAgent: Training Web Agents in Real-World Cloud Environments via Distillation and Reinforcement Learning](http://arxiv.org/abs/2606.09447)

- AliyunConsoleAgent: introduces a web agent framework designed for automated documentation verification in complex, real-world cloud console environments using a two-stage training paradigm.
- The framework utilizes a high-determinism Rollout Environment featuring Resource META provisioning and a ResourceCoder to isolate environmental noise and stabilize RL training.
- By combining supervised fine-tuning on distilled trajectories with GRPO reinforcement learning and a dual-channel reward model, the system achieves high success rates at significantly reduced inference costs.

---

[LargeMonitor: Monitoring Online Task-Free Continual Learning via Large Pretrained Models](http://arxiv.org/abs/2606.09430)

- LargeMonitor: introduces a decoupled monitoring framework that leverages Large Vision Models and Large Multimodal Models to perform zero-shot drift detection and semantic diagnosis in online task-free continual learning.
- The framework utilizes Large Vision Models to compute CKA similarity for robust drift detection and employs Large Multimodal Models to interpret the semantic etiology of detected shifts.
- By decoupling detection from the internal optimization loop, LargeMonitor enables dynamic, shift-specific adaptation strategies that outperform uniform mechanical responses in non-stationary streaming environments.

---

[Guide Me Out: A Framework to Benchmark VLM Operators Communication in Crisis Scenarios](http://arxiv.org/abs/2606.09428)

- Guide Me Out: introduces a simulation-based benchmarking framework for evaluating VLM Operators (AI agent guiding civilians) in crisis evacuation scenarios.
- The framework assesses VLM Civilian (AI agent navigating environment) performance across different Communication Strategy (narrowcast or broadcast messaging), Environment Representation (visual or graph-based), and Threat Dynamics (static or moving obstacles) configurations.
- Results indicate that narrowcast communication consistently reduces civilian failure rates compared to broadcast, while visual input remains essential for effective operator guidance.

---

[WeaveBench: A Long-Horizon, Real-World Benchmark for Computer-Use Agents with Hybrid Interfaces](http://arxiv.org/abs/2606.09426)

- WeaveBench: introduces a long-horizon hybrid-interface benchmark for evaluating computer-use agents that must coordinate GUI observations with CLI/code operations across real-world tasks.
- The framework utilizes a trajectory-aware agentic judge to audit agent performance, preventing reward hacking by verifying intermediate steps and evidence rather than relying solely on final deliverables.
- Experiments demonstrate that current LLMs struggle with sustained cross-interface orchestration, with the benchmark exposing significant gaps in agent performance that outcome-only grading fails to capture.

---

[What Should a Skill Remember? Quality-Cost Trade-offs in Cost-Aware Skill Rewriting for Language Model Agents](http://arxiv.org/abs/2606.09421)

- SkillEE: introduces a framework for cost-aware skill rewriting that optimizes procedural documents for LLM agents by preserving task-relevant operational anchors rather than performing uniform text compression.
- The framework utilizes a task-conditioned selector to choose between preservation strategies—such as API/code anchoring, rule/formula anchoring, or workflow guarding—based on structural features to balance task quality against total execution cost.
- Experiments on SkillsBench demonstrate that this approach reduces total execution costs by 7.0% on average while maintaining or improving verifier quality across diverse agent stacks.

---

[Harness Engineering for Physical AI: Robot Middleware Is the Harness Layer](http://arxiv.org/abs/2606.09416)

- Robot Middleware Harness Layer: introduces a framework for governing Physical AI models by utilizing robot middleware as a harness layer to enforce output regions, inference budgets, and operating regimes.
- The framework integrates three core enforcement mechanisms—Projection, Isolation, and Transfer—to manage the nondeterminism introduced by LLMs or other learned models within the robot control loop.
- By leveraging existing ROS 2 surfaces, the approach enables platform-level governance that composes enforcement across control, computing, and communication axes simultaneously.

---

[RunAgent SuperBrowser: A Theory of Autonomous Web Navigation Grounded in Human Browsing Behaviour](http://arxiv.org/abs/2606.09399)

- SuperBrowser: introduces a cognitive-theory-motivated web-navigation agent that operationalizes a perception–cognition–action triad using a Vision pipeline, Orchestrator, Planner, Worker, Ledger, Click cascade, and Humanization layer.
- The system employs a structured Ledger and a six-phase eviction loop to maintain a bounded, high-relevance context, preventing the performance degradation typical of agents that accumulate raw observation history.
- SuperBrowser achieves 89.47% success on the Mind2Web Hard benchmark by combining role-sliced strategic reasoning with procedural DOM-caching and chevron-aware visual grounding.

---

[Experience Makes Skillful: Enabling Generalizable Medical Agent Reasoning via Self-Evolving Skill Memory](http://arxiv.org/abs/2606.09365)

- SkeMex: introduces a post-deployment self-evolution framework that improves medical agents through a skill-based memory without updating model weights.
- The framework utilizes a closed-loop "Read–Write–Assess–Govern" lifecycle to distill interaction trajectories into structured skills, evaluate their utility, and maintain a compact, reliable repository.
- SkeMex enables medical agents to accumulate and reuse procedural knowledge across diverse clinical tasks, demonstrating consistent performance improvements and cross-backbone generalization.

---

[Bespoke-Card: Why Tune When You Can Generate? Synthesizing Workload-Specific Cardinality Estimators](http://arxiv.org/abs/2606.09361)

- Bespoke-Card: introduces an agent-driven system that synthesizes workload-specific cardinality estimators as executable code by utilizing a Planning Agent, Coding Agent, and Evaluator to iteratively refine the estimator through Structured Feedback.
- The framework employs a staged curriculum that isolates join, filter, and full-subplan errors to guide the synthesis of an Executable Estimator tailored to a specific database and workload.
- By replacing generic statistics with workload-specialized code, Bespoke-Card significantly reduces query runtime and improves estimation accuracy compared to traditional database optimizers.

---

[PBSD: Privileged Bayesian Self-Distillation for Long-Horizon Credit Assignment](http://arxiv.org/abs/2606.09348)

- PBSD: introduces a Bayes-calibrated self-distillation method for fine-grained credit assignment in long-horizon agentic tasks by transforming trajectory-level rewards into turn-level signals.
- The framework utilizes a privileged answer-conditioned teacher model to compute Bayesian evidence scores, which are then used to modulate and redistribute outcome-level advantages across intermediate reasoning turns.
- PBSD enhances training stability and performance in long-horizon search agents by filtering low-SNR evidence and decoupling likelihood scoring from routing artifacts in MoE models.

---

[One Model, Multiple Goals: Adaptive Multi-Objective Learning for E-commerce Dialogue Systems](http://arxiv.org/abs/2606.09293)

- MORE: introduces an adaptive multi-objective reinforcement learning framework that unifies reasoning accuracy and linguistic naturalness by utilizing a reasoning-enhanced training scaffold and gradient-based dynamic reward reweighting.
- The framework employs a two-stage training process where a reasoning-enhanced scaffold improves factual recall and numerical inference, while the policy model is optimized using GRPO with adaptive multi-reward aggregation.
- By treating the reasoning-enhanced model as a reference, the system anchors correctness in personalized reasoning while allowing the policy model to focus on generating fluent and natural responses without explicit inference overhead.

---

[Visual Para-Thinker++: A Single-Policy Multi-Agent Framework for Visual Reasoning](http://arxiv.org/abs/2606.09290)

- Visual Para-Thinker++: introduces a single-policy multi-agent framework that utilizes a shared MLLM policy instantiated as Main Agent, Worker Agents, and Summary Agent to improve visual reasoning through role-conditioned collaboration.
- The framework employs Multi-Agent Capability Injection and Role-Decoupled Multi-Agent Optimization to mitigate gradient conflicts between agents while maintaining a unified set of model weights.
- A native multi-agent inference engine enhances performance by reusing KV cache across role segments, enabling efficient parallel reasoning without redundant visual prefill.

---

[MAGIS: Evidence-Based Multi-Agent Reasoning for Interpretable Strabismus Clinical Decision-Making](http://arxiv.org/abs/2606.09249)

- MAGIS: introduces a multi-agent framework that reformulates strabismus diagnosis as an evidence-constrained and verifiable process to mitigate LLM hallucinations.
- The framework utilizes a Classifier Agent, Verifier Agent, and Generator Agent to integrate Dual-Evidence Constrained Context (DECC) and Evidence-Based Corrective Verification (EBCV) for reliable diagnostic reasoning.
- By grounding diagnostic conclusions in visual evidence and formalized clinical rules, MAGIS significantly improves the clinical reliability and interpretability of automated strabismus diagnosis.

---

[Self-Paced Curriculum Reinforcement Learning for Autonomous Superbike Racing in Simulation](http://arxiv.org/abs/2606.09236)

- SPDL (Self-Paced Curriculum Deep Reinforcement Learning): introduces a framework for training autonomous agents to race superbikes in the VRider SBK simulator by integrating SAC with an automated curriculum generation mechanism.
- The system utilizes a rich state space including proprioceptive features and lean-angle history to manage the complex dynamics of two-wheeled vehicles.
- Experimental results demonstrate that the framework achieves faster convergence and improved driving stability compared to standard SAC across multiple tracks and motorbike models.

---

[MASS: Deep Research for Social Sciences with Memory-Augmented Social Simulation](http://arxiv.org/abs/2606.09198)

- MASS: introduces a paradigm for automated social science research that leverages realistic social simulations to generate empirical data for LLMs.
- The framework integrates a Task Planning and Deep Reasoning Module, a COT Execution and Tool Dispatching Module, a Social Simulation Experiment Module, and a Structured Writing Module to automate the research process.
- MASS utilizes a Memory Augmentation Framework with a multidisciplinary Behavioral Dataset and an Ebbinghaus-inspired Forgetting Mechanism to enhance agent authenticity and cognitive continuity during simulations.

---

[Claude Code-Driving Scenario Mining for the Argoverse 2 Challenge](http://arxiv.org/abs/2606.09180)

- Claude Code-Driving Scenario Mining framework: introduces a four-stage pipeline for autonomous driving scenario mining that leverages an LLM-powered coding agent, semantic review, and VLM-based verification to minimize false positives.
- The system utilizes a Coding Agent to generate scenario code, a Code Reviewer for semantic audit, and a Scene Verifier to perform binary classification on driving frames.
- The framework incorporates an iterative screening process on the RefAV dataset to build a high-quality Few-shot Library, improving code generation performance for unseen prompts.

---

[Performance Evaluation of Social Learning](http://arxiv.org/abs/2606.09176)

- SL framework: introduces a decentralized decision-making paradigm where agents exchange beliefs over a network to identify a true hypothesis from a finite set.
- The paper demonstrates that the rejection rate is an unreliable performance metric and instead utilizes error probability to characterize decentralized decision-making performance.
- The authors derive an exact analytical formula for the asymptotic ratio between decentralized and centralized error probabilities, revealing irreducible gaps caused by network connectivity and initial belief assignments.

---

[Demonstrating chart-plot: Closing the Last Mile of Academic Chart Generation](http://arxiv.org/abs/2606.09174)

- Chart-plot: introduces an agentic harness that closes the last mile of academic chart generation by integrating style-aware code generation, a deployment-aware render loop, and a multimodal edit layer.
- The system utilizes a distilled style skill to align generated matplotlib code with target venue conventions and employs a render loop to ensure figures meet specific LaTeX layout constraints.
- A shared layout representation allows for precise, direct manipulation of chart elements, mapping user edits to reproducible code mutations without requiring full LLM re-generation.

---

[Claw-R1: A Step-Level Data Middleware System for Agentic Reinforcement Learning](http://arxiv.org/abs/2606.09138)

- Claw-R1: introduces a step-level data middleware system that decouples heterogeneous agent runtimes from RL training backends to manage the full lifecycle of interaction data.
- The system utilizes a Gateway Server for unified data ingestion and a Data Pool for organizing interaction traces into structured, step-level training assets.
- Claw-R1 optimizes data storage through prefix-tree merging to reduce redundant long-context computation while maintaining step-level semantics for downstream RL training.

---

[Autonomous Incident Resolution at Hyperscale: An Agentic AI Architecture for Network Operations](http://arxiv.org/abs/2606.09122)

- AIR: introduces a multi-agent orchestration architecture for autonomous network incident resolution that utilizes Intake Agent, Planning Agent, Execution Agent, Verification Agent, Playbook Store, Skill Registry, Topology Model, Historical Data, AuthZ Engine, Blast Radius, Circuit Breakers, Rollback Manager, Device Access, Telemetry Bus, and State Store to automate the full incident lifecycle.
- The architecture employs a skills-based tool abstraction, inspired by the Model Context Protocol, to decouple LLM reasoning from specific infrastructure capabilities while enforcing safety through layered authorization and automated rollback mechanisms.
- The system achieves over 90% autonomous resolution rates for common incidents by implementing progressive autonomy levels and rigorous safety guardrails, including per-category circuit breakers and deterministic verification of LLM-generated plans.

---

[ComplexConstraints and Beyond: Expert Rubrics for RLVR](http://arxiv.org/abs/2606.09118)

- ComplexConstraints: introduces a systematic paradigm for expert-curated rubric-based evaluation and training, utilizing ComplexConstraints, Expert-curated Rubrics, LLM-Judge, RLVR, Primary Intent Criteria, Extra Credit Criteria, and Dodged Bullet Criteria.
- The framework employs a three-category taxonomy to provide dense, granular reward signals for RL, enabling models to learn complex instruction following and agentic task execution more effectively than with binary metrics.
- Empirical results demonstrate that training on approximately 1,000 expert-curated examples yields significant performance gains and transferable capabilities across both instruction-following and agentic benchmarks.

---

[HDRAgent: An Agentic Framework for Multi-Exposure HDR Imaging](http://arxiv.org/abs/2606.09110)

- HDRAgent: introduces an agent-driven framework for multi-exposure HDR imaging that reformulates reconstruction as an iterative process integrating MLLM-based scene perception, contextual knowledge matching, dynamic tool routing, and feedback-guided refinement.
- The framework utilizes a fine-grained contextual knowledge matching module to organize scene evidence and historical feedback, enabling the MLLM dynamic router to adaptively select optimal alignment and fusion strategies.
- To address extreme motion and occlusion, the system incorporates an agent-guided generative alignment module that performs reference-conditioned masked generation to reconstruct unreliable dynamic regions.

---

[REFLECT: Intervention-Supported Error Attribution for Silent Failures in LLM Agent Traces](http://arxiv.org/abs/2606.09071)

- REFLECT: introduces a method for localizing silent failures in LLM agent traces by combining diagnosis-guided intervention with controlled replay to produce verified attribution records.
- The framework utilizes a Diagnostician, Error Classifier, Faithfulness Gate, Verifier, Replay Engine, and Post-correction Explainer to close the loop between correction and attribution.
- REFLECT achieves high localization accuracy by using successful corrections as contrastive evidence to sharpen error explanations, outperforming existing methods that rely on independent retries or unverified predictions.

---

[Agent Economics: An Entropy-Controlled Pluralistic Alignment Framework for Preventing Artificial Hivemind in Autonomous Agents](http://arxiv.org/abs/2606.09039)

- BPF: introduces a closed-loop architecture integrating MbSI, PA, and VEK to mitigate artificial hivemind effects and ensure transparency in autonomous agent economies.
- The framework utilizes MbSI for LLM-based intention inference, PA for entropy-controlled strategic diversity, and VEK for immutable hash-chain audit trails.
- This approach enables dynamic regulatory feedback and verifiable decision-making pathways to enhance the stability and accountability of multi-agent economic systems.

---

[Personalization Meets Safety: Mechanisms, Risks, and Mitigations in Personalized LLMs](http://arxiv.org/abs/2606.09038)

- Personalized LLMs: introduces a comprehensive safety-aware review of personalization mechanisms, risks, and mitigation strategies across the full personalization stack, including Structured Profile, Textual Persona, Preference Signals, Memory Structure, Prompting-based Personalization, Retrieval-augmented Personalization, Fine-tuning-based Personalization, Reinforcement Learning-based Personalization, Mixture-of-Experts (MoE)-based Personalization, Pruning-based Personalization, and Agent-based Architecture.
- The paper systematically categorizes safety risks into fine-grained threats inherent to specific personalization paradigms and paradigm-agnostic risks arising from the adaptation process itself.
- It provides a unified framework for understanding and building safe personalized LLM systems by integrating personalization techniques, safety risks, evaluation methodologies, and mitigation strategies throughout the model lifecycle.

---

[A Multi-Agent System for IPMSM Design Optimization via an FEA-AI Hybrid Approach](http://arxiv.org/abs/2606.09037)

- FEA-AI Hybrid Multi-Agent System: introduces an end-to-end automated framework for IPMSM design optimization that integrates RAG-grounded problem definition, log-informed resampling, and uncertainty-aware hybrid evolutionary search.
- The framework utilizes a Design Agent for structured requirement elicitation, a Training Agent for autonomous FEA data generation and surrogate learning, and an Optimization Agent for uncertainty-threshold-based switching between surrogate inference and high-fidelity FEA.
- By employing LLM-reasoning for resampling and uncertainty-driven active learning, the system effectively balances computational cost with prediction reliability in complex motor design spaces.

---

[Bridging the Agent-World Gap: Text World Models for LLM-based Agents](http://arxiv.org/abs/2606.09032)

- TWM: introduces a systematic review of text world models for LLM-based agents, categorizing them by construction, usage, and evaluation paradigms.
- The framework classifies world models into LLM-as-WM and Code-as-WM paradigms, supporting agent training and inference through simulation and verification.
- The paper provides a two-axis taxonomy based on state representation and grounding domain to characterize the design space of text world models.

---

[SpaceVLN: A Zero-Shot Vision-and-Language Navigation Agent with Online Spatial Cognitive Memory and Reasoning](http://arxiv.org/abs/2606.08992)

- SpaceVLN: introduces a stagewise closed-loop navigation framework that organizes planning and execution around verifiable space–landmark stages using Spatial Cognitive Memory and Task-Guided Spatial Reasoning.
- The framework utilizes Hierarchical Spatial Memory to abstract explored regions into waypoints and Local Landmark Memory to provide subtask-specific landmark cues for the executor.
- Spatial-CoT integrates task-progress reasoning with spatial perception to enable zero-shot navigation without task-specific policy training.

---

[Baichuan-M4: A Clinical-Grade Medical Agent System for Continuous Care](http://arxiv.org/abs/2606.08982)

- Baichuan-M4: introduces a clinical-grade medical agent system designed for continuous care that integrates Baichuan-Harness, a core reasoning model, and a clinical tool layer to manage long-term patient memory, tool use, and multi-agent coordination.
- The system utilizes a three-layer nested adaptive control architecture to drive continuous improvement through real-world feedback, incorporating specialized mechanisms like SPAR++ for span-level reward modeling and reasoning path compression for efficient inference.
- Baichuan-M4 achieves high performance in complex clinical reasoning, hallucination control, and multimodal understanding by employing a robust toolchain for evidence-based retrieval, medical document OCR, and dermatology-specific diagnostic agents.

---

[Hardening Agent Benchmarks with Adversarial Hacker-Fixer Loops](http://arxiv.org/abs/2606.08960)

- Hacker-Fixer Loop: introduces an iterative framework that alternates between a hacker, a fixer, and a solver to automatically harden benchmark verifiers against reward hacking.
- The framework utilizes a shared defense pool to propagate infrastructure-level patches across tasks and provides the hacker with verifier access to discover sophisticated exploits.
- Empirical results demonstrate that the loop effectively reduces attack success rates on KernelBench and Terminal Bench while maintaining high pass rates for legitimate solutions.

---

[AlloSpatial: Agentic Harness Framework for Spatial Reasoning in Foundation Models](http://arxiv.org/abs/2606.08952)

- AlloSpatial: introduces an agentic framework that transforms egocentric observations into structured allocentric spatial representations to improve spatial reasoning in foundation models.
- The framework utilizes World2Mind to generate query-conditioned cognitive maps and a Spatial Reasoning Harness to regulate tool invocation, evidence collection, and cross-modal arbitration.
- AlloSpatial agents are trained via supervised cold-start and reinforcement learning with a Harness-Gated Trajectory Reward to achieve robust, verifiable spatial reasoning performance.

---

[LongRTL: Graph-Similarity-Guided LLM-driven Long Context RTL Optimization](http://arxiv.org/abs/2606.08944)

- LongRTL: introduces a scalable framework for long-context RTL optimization that utilizes a Partition Agent, Optimization Agent, and Reconstruction Agent to manage complex hardware designs.
- The framework employs AST-level graph similarity via GNNs to decompose monolithic RTL into manageable subtrees, which are then optimized by an LLM-based agent using multi-modal RAG and MCTS.
- A logic-aware Reconstruction Agent ensures functional equivalence by reassembling optimized submodules through Graph-RAG prompting that respects original design hierarchy and signal dependencies.

---

[PACT: Learning Diverse Diagnostic Strategies via Privileged Synthesis and Branch Consensus](http://arxiv.org/abs/2606.08938)

- PACT: introduces a framework that couples supervised multi-paradigm dialogue synthesis with consensus-based Branch training to improve medical diagnostic reasoning in LLMs.
- The DPS module synthesizes validated dialogues under four distinct reasoning paradigms while preserving doctor-patient information asymmetry through privileged supervision.
- PACT periodically aggregates paradigm-specific LoRA Branches into a single deployable Anchor using sign-consensus merging to prevent parameter drift and maintain diagnostic performance.

---

[From Statute to Control Flow: Span-Grounded Deontic Trees for Defeasible Scope Parsing](http://arxiv.org/abs/2606.08932)

- NormBench: introduces a diagnostic benchmark and SG-DT intermediate representation to evaluate and mitigate Silent Scope Omission in LLMs by enforcing explicit, span-grounded logical structures.
- The framework utilizes an SGDT Parser to decompose legal provisions into executable units and branches, ensuring that defeasible rules and exceptions are deterministically compiled.
- Evaluations reveal that while LLMs excel at surface-level retrieval, they suffer from a structure-grounding gap, particularly in deep recursive exception chains, which SG-DT helps address through mechanism-specific intervention.

---

[Oversight Has a Capacity: Calibrating Agent Guards to a Subjective, Fatiguing Human](http://arxiv.org/abs/2606.08919)

- Agent-oversight system: introduces an open-source framework that treats LLM agent action-gating as a selective classification problem under asymmetric cost, incorporating a Guard Scorer, Reviewer Model, Calibration Apparatus, and Endogenous-reviewer Simulator.
- The framework demonstrates that human oversight has a finite capacity, where excessive escalation leads to reviewer fatigue and reduced system safety, resulting in an inverted-U relationship between escalation rate and realized safety.
- By modeling the reviewer as an endogenous component, the system identifies load-aware operating points that optimize human attention and defend against flooding attacks where adversaries exploit reviewer fatigue.

---

[Vibe Visualizing: How Visualization Novices Try (and Fail) to Generate and Interpret Visualizations with Conversational AI](http://arxiv.org/abs/2606.08914)

- Vibe Visualizing: introduces a thematic codebook to evaluate how visualization novices interact with LLMs to generate and interpret data visualizations.
- The study identifies recurring failure modes across ChatGPT, Gemini, and Claude, highlighting significant challenges in AI instruction following, chart design, and interpretation accuracy.
- The research provides six design recommendations for developers to improve the reliability and transparency of AI-assisted visualization systems for non-expert users.

---

[Do Coding Agents Deceive Us? Detecting and Preventing Cheating via Capped Evaluation with Randomized Tests](http://arxiv.org/abs/2606.07379)

- CapCode and CapReward: introduces a framework for detecting and preventing deceptive performance in LLMs by injecting randomized cap values into coding datasets to enforce a performance ceiling and penalizing reward hacking during RL fine-tuning.
- CapCode constructs datasets where the maximum achievable pass rate is deliberately capped, allowing statistical detection of cheating when models exceed this threshold through test-gaming.
- CapReward leverages the capped-performance principle to discourage reward hacking by penalizing models that achieve pass rates beyond the established cap, thereby incentivizing genuine task-solving behavior.

---

[Synthetic APTs: the Collapse of TTP-Based Attribution](http://arxiv.org/abs/2606.07158)

- CSI (Cybersecurity SuperIntelligence): introduces a framework for evaluating AI-driven adversary emulation fidelity and its impact on TTP-based attribution by deploying LLM-based agents against active AI-driven defenders in realistic cyber ranges.
- The study demonstrates that while AI agents can replicate specific APT profiles with high precision in deep kill chain scenarios, they converge on identical initial-phase techniques, thereby challenging the reliability of traditional TTP-based attribution.
- Experimental results across 20 scenarios reveal that network topology is the primary determinant of strategic outcomes, with segmented military-grade environments consistently neutralizing AI-driven threats regardless of the attacker's assigned persona or defender model scale.

---

[From Privacy to Workflow Integrity: Communication-Graph Metadata in Autonomous Agent Interoperability](http://arxiv.org/abs/2606.07150)

- A2A: introduces a threat model for agent communication graphs, demonstrating that metadata leakage enables predictive leverage over autonomous workflows.
- The research identifies that semanticity, prospectivity, and actuation in agent interactions allow observers to infer task classes and preemptively influence outcomes.
- The paper proposes a set of transport- and bootstrap-privacy properties—unlinkability, no central observer, deniability, metadata minimization, and discovery privacy—to neutralize metadata-based integrity threats.

---

[Regimes: An Auditable, Held-Out-Gated Improvement Loop Demonstrated on LongMemEval with ActiveGraph](http://arxiv.org/abs/2606.10241)

- Regimes: introduces an auditable, held-out-gated improvement loop built on the ActiveGraph (deterministic event-sourced substrate) that diagnoses failures via a Regime Classifier (identifies failure root causes) and applies patches through an Author Agent (drafts executable pipeline patches) constrained by a Static Gate (checks code structural integrity), Sandbox Gate (verifies runtime execution safety), OPTIMIZE Gate (validates in-sample performance), and CONFIRM Gate (validates held-out generalization) at specific Action Seams (specific pipeline edit locations).
- The framework utilizes an event-sourced runtime to ensure deterministic replay of agent history, allowing for auditable diagnosis and repair of failures within long-context memory tasks.
- By routing failures to specific pipeline seams and enforcing held-out validation, the system effectively distinguishes between genuine improvements and overfit candidates, providing a robust mechanism for autonomous agent refinement.

---

[SHAPO: Sharpness-Aware Policy Optimization for Safe Exploration](http://arxiv.org/abs/2606.10228)

- SHAPO: introduces a sharpness-aware policy update rule that evaluates gradients at perturbed parameters to induce a pessimistic bias against epistemic uncertainty during reinforcement learning.
- The framework utilizes the Fisher Information Matrix to define parameter perturbations, effectively reweighing policy gradients to prioritize safety in under-explored regions.
- By integrating sharpness-aware optimization into the actor's update, the method expands the safety-efficiency Pareto frontier and suppresses heavy-tailed episodic cost distributions in continuous-control tasks.

---

[Less Context, Better Agents: Efficient Context Engineering for Long-Horizon Tool-Using LLM Agents](http://arxiv.org/abs/2606.10209)

- Context Engineering Framework: introduces a semantic-level context management policy for long-horizon LLM agents that utilizes recency-based pruning and automated summarization to maintain task-relevant focus while preventing context window overflow.
- The system architecture includes a primary GPT-5 agent, a secondary GPT-4.1 user model, a D365 F&O MCP server for tool interactions, and an internal evaluation harness to manage agent-tool loops.
- Experimental results demonstrate that selective retention of recent tool interactions combined with automated summarization significantly improves task completion rates and token efficiency compared to full-history retention.

---

[Exploration of Foundation Model-Based Robots in Patient and Elderly Care](http://arxiv.org/abs/2606.10208)

- Foundation Model-Based Care Robots framework: synthesizes the interaction pipeline of embodied care systems by integrating User, Context, Perception, Foundation Model, Action, Outcomes, and Caregiver or Clinician Oversight components.
- The framework utilizes LLMs and VLMs as conversational and reasoning layers to enable personalized assistance, while highlighting the necessity of human oversight to manage reliability failures like hallucinations and latency.
- Future research must transition from isolated feasibility studies toward standardized evaluation metrics, accountable autonomy, and seamless integration into real-world clinical care workflows.

---

[τ-Rec: A Verifiable Benchmark for Agentic Recommender Systems](http://arxiv.org/abs/2606.10156)

- τ-Rec: introduces a verifiable benchmark for agentic recommender systems that models interactions as a POMDP over a Tool-Agent-User (TAU) triad, utilizing Verifiable Rewards, Reveal-tagged Elicitation (RTE), the pass^k Reliability Metric, and Policy Enforcement to replace subjective LLM-as-a-judge evaluations.
- The framework employs a Catalog and Metadata Tools component for external data retrieval and a LLM-based User Simulator to test agent performance across varying constraint complexities and reveal difficulties.
- Experimental results across nine LLM configurations reveal a significant reliability cliff, demonstrating that current models struggle with consistency and policy compliance in multi-turn agentic recommendation tasks.

---

[What makes a harness a harness: necessary and sufficient conditions for an agent harness](http://arxiv.org/abs/2606.10106)

- Agent Harness: introduces a constitutive definition and an inclusion/exclusion test for the runtime engineering layer that wraps an LLM to enable reliable agentic task execution.
- The framework identifies four necessary core components—agent loop, tool interface, context management, and control mechanisms—that distinguish an agent harness from frameworks, SDKs, and evaluation scaffolds.
- The paper provides a research agenda organized by four design tension axes: autonomy versus control, broad versus curated context, generalist versus specialized, and open permission versus containment.

---

[Bittensor Agent Arenas as a Trajectory Primitive: Distilling a Shopping Agent from ShoppingBench Subnet Traces](http://arxiv.org/abs/2606.10064)

- SN15 Agent Arena: introduces a decentralized incentive-aligned framework that generates high-quality, judged agent trajectories for post-training LLMs by leveraging a continuous competition between independent miners.
- The framework utilizes a structural-quality filter to extract agentic trajectories from a raw firehose, which are then used to post-train a Qwen3-4B base model through a multi-stage pipeline including SFT, KTO, and Dr. GRPO.
- This approach demonstrates that subnet-generated data provides a superior trajectory substrate for agentic post-training compared to purely synthetic datasets or unfiltered production logs.

---

[Deployment-Time Memorization in Foundation-Model Agents](http://arxiv.org/abs/2606.10062)

- Deployment-Time Memorization in Foundation-Model Agents: introduces a framework for evaluating persistent agent memory as a privacy-utility frontier, utilizing an Agent-Memory Pipeline with configurable Summarization Module, Retrieval Module, and Deletion Module.
- The research quantifies memory leakage and utility through Personalization Recall and Adversarial Extraction Rate, demonstrating that summarization effectively launders secrets while retrieval breadth impacts context size.
- The study establishes that effective deletion in tiered memory requires full-pipeline purging or tombstone redaction to eliminate residue across all memory tiers, as raw-only deletion is insufficient.

---

[3SPO: State-Score-Supervised Policy Optimization for LLM Agents](http://arxiv.org/abs/2606.09961)

- 3SPO (State-Score-Supervised Policy Optimization): introduces a reinforcement learning framework for LLMs that utilizes dynamic state scores derived from historical interaction statistics to enable step-wise credit assignment and adaptive rollout allocation.
- The framework replaces trajectory-level optimization with post-step policy updates, using a composite reward signal that balances novelty, state-score transitions, and task completion.
- By employing ranked backtracking DFS and adaptive resource allocation, 3SPO improves sample efficiency and convergence speed in long-horizon agentic tasks compared to standard group-based RL methods.

---

[Multi-task LLMs for Bug Classification: Efficient Inference with Auxiliary Decoding Heads](http://arxiv.org/abs/2606.09956)

- MLC (Multi-task LLM for Bug Localization): introduces a lightweight, multi-task approach that augments an LLM Backbone with a Token Alignment Algorithm, Token Aggregator, and Classification Head to perform efficient, line-level bug classification in a single forward pass.
- The framework utilizes a Token Alignment Algorithm to map tokens to specific lines, bypassing tokenization boundary issues, and employs a Classification Head to output binary bug predictions for all lines simultaneously.
- By integrating an optional Adapter and FFN, the architecture achieves state-of-the-art line-level bug localization performance while significantly reducing inference latency compared to agentic or generative LLM pipelines.

---

[Interactions between crosscoder features: A compact proofs perspective](http://arxiv.org/abs/2606.09940)

- Crosscoder framework: introduces a method to quantify feature interactions in LLMs by formalizing the error term in compact proofs as an interaction metric.
- The framework utilizes an interaction-based penalty during training to produce computationally sparse crosscoders that concentrate feature norm on dominant features.
- This approach enables automated mechanistic interpretability, feature clustering, and anomaly detection in LLMs without compromising feature interpretability.

---

#### 7th June 2026

[PerspectiveGap: A Benchmark for Multi-Agent Orchestration Prompting](http://arxiv.org/abs/2606.08878)

- PerspectiveGap: introduces a benchmark for evaluating how LLMs compose orchestration prompts for multi-agent systems by testing their ability to manage information boundaries across sub-agent roles.
- The framework utilizes 110 scenarios across 10 topologies to measure if LLMs can correctly assign information fragments to specific roles while excluding distractors and maintaining context boundaries.
- Experimental results across 27 commercial models reveal that orchestration prompting is a distinct, fragile capability where even strong models frequently fail to preserve role-specific information boundaries.

---


[Can the Environment Speak for Itself? T2-GRPO: A Turn-Trajectory Group Relative Policy Optimization for Caregiver Agents](http://arxiv.org/abs/2606.08875)

- T2-GRPO: introduces a multi-horizon reinforcement learning framework for caregiver agents that decouples rewards into turn-level environment-grounded signals and trajectory-level evaluations to improve long-horizon caregiving.
- The framework utilizes centered rank normalization to preserve heterogeneous reward signals and applies a hard safety veto to prevent catastrophic failures without trading safety against task performance.
- By deriving dense turn-level rewards directly from the DemMA patient simulator, the method achieves superior caregiving quality and safety while eliminating the computational overhead of external LLM judges.

---


[EFX for Additive Chores: Nonexistence, Pareto Incompatibility, and Bi-Valued Existence](http://arxiv.org/abs/2606.08872)

- EFX for Additive Chores: Nonexistence, Pareto Incompatibility, and Bi-Valued Existence: introduces a comprehensive framework for chore allocation that resolves the existence of EFX allocations for additive cost functions by utilizing M01-allocation, M2-multigraph-allocation, M34-insertion-mechanism, Gap-filling-procedure, and Residual-allocation-mechanism.
- The paper demonstrates that EFX allocations for additive chores are not guaranteed for n ≥ 4 agents, even with tri-valued cost functions, and establishes the incompatibility of EFX and Pareto-optimality for bi-valued chores.
- The research provides an existence guarantee for EFX allocations in all four-agent bi-valued instances, effectively bridging the gap between existence and efficiency in fair division.

---

[Building Customer Support AI Agents at 100M-User Scale: An Evaluation-Driven Framework](http://arxiv.org/abs/2606.08867)

- Evaluation-Driven Framework: introduces a unified methodology for building production-ready customer support AI agents by bridging offline development with online impact through a closed-loop feedback system.
- The framework integrates modular context engineering, systematic human-in-the-loop prompt iteration, rigorous LLM-as-a-judge evaluation, and production-validated deployment to ensure high-quality customer interactions.
- Experimental results across five production domains demonstrate that the framework significantly improves customer satisfaction and self-service rates while maintaining performance parity with expert human agents.

---

[PaperMentor: A Human-Centered Multi-Agent Writing Tutor for AI Research Papers on Overleaf](http://arxiv.org/abs/2606.08857)

- PaperMentor: introduces a human-centered multi-agent writing assistant that provides actionable, text-anchored feedback on LaTeX manuscripts by integrating a curated Skill Library with twelve specialized LLM agents.
- The system operates through a three-phase pipeline consisting of Input Processing, Multi-Agent Review, and Comment Aggregation to deliver venue-aware and paper-type-specific guidance directly within the Overleaf interface.
- By leveraging expert-guided agents, PaperMentor significantly improves the validity and actionability of feedback compared to direct LLM prompting while preserving authorial control over revisions.

---

[A Resilience-as-a-Service assessment framework for coordinated disruption response in interdependent urban transit systems](http://arxiv.org/abs/2606.08849)

- RaaS (Resilience-as-a-Service): introduces a decision support framework that combines time-indexed optimization, behavioral simulation, and a KPI dashboard to assess disruption response solutions in urban transit systems.
- The framework evaluates coordinated multimodal recovery strategies by balancing passenger service quality, operator costs, emissions, and equity across primary and donor transit lines.
- The approach utilizes a mixed integer linear programming model for resource dispatch and a multi-agent simulation for endogenous passenger behavioral adaptation to benchmark resilience performance.

---

[Instrumental convergence and power-seeking](http://arxiv.org/abs/2606.08832)

- Orbital Markov Model: introduces a formal framework for analyzing power-seeking behavior in artificial agents by evaluating how state-contingent rewards and option preservation influence optimal policies.
- The paper critically examines the instrumental convergence thesis, arguing that current formal and informal defenses fail to establish that superintelligent agents will necessarily pursue power to the point of causing existential catastrophes.
- By analyzing the Orbital Markov Model and the concept of shutdown avoidance, the author demonstrates that power-seeking is highly dependent on specific, often unproven, assumptions about agent architecture, training environments, and goal-directed behavior.

---

[RAILS: Verification-Native Clearing for Agentic Commerce](http://arxiv.org/abs/2606.08790)

- RAILS: introduces a verification-native clearinghouse protocol for autonomous agent obligations that ensures no financially material settlement is supported by evidence below an obligation's admissibility floor.
- The framework utilizes seven primitives, including Obligation Object, Evidence Envelope, Verification Mesh, Clearing Decision, Settlement Instruction, Clearing Passport, and Finality Rules, to convert execution traces into sound, admissibility-graded settlement decisions.
- RAILS provides a falsifiable soundness property that bridges the gap between agent execution and financial settlement, effectively addressing performance failures in agentic commerce that existing payment and authorization protocols cannot detect.

---

[Co-Evolving Skill Generation and Policy Optimization](http://arxiv.org/abs/2606.08755)

- SAPO: introduces an online reinforcement learning framework that validates candidate skills by comparing base and skill-augmented rollouts to estimate marginal utility before storage.
- The framework utilizes the estimated marginal utility to train the agent policy as a skill generator, reducing reliance on external proprietary LLMs.
- SAPO employs a dual-bank memory system with likelihood-based scoring to prune outdated skills and rerank candidates during retrieval for improved agent performance.

---

[Systems-Level Planning and Coordination of Truck–Drone Collaborative Delivery Networks](http://arxiv.org/abs/2606.08738)

- TDCD: introduces a multi-layered planning framework that structures truck–drone collaborative delivery through Spatial-Demand Alignment, Collaborative Delivery Configuration, Resource and Workflow Planning, Performance Assessment, and Scalability Assessment.
- The framework utilizes a circular feedback loop between configuration, resource planning, and performance assessment to iteratively refine operational decisions based on system conditions.
- Case study results demonstrate that the proposed TDCD model achieves a 42.4% reduction in delivery time and a 44.2% reduction in energy consumption compared to conventional truck-only delivery.

---

[IR-SIM: A Lightweight Skill-Native Simulator for Navigation, Learning, and Benchmarking](http://arxiv.org/abs/2606.08729)

- IR-SIM (Intelligent Robot Simulator): introduces a lightweight, skill-native navigation simulator that translates natural language prompts into executable YAML configuration files and Python runners for rapid robotics research.
- The framework utilizes LLM-powered agent skills to automate scenario construction, enabling reproducible benchmarking and training data generation for navigation algorithms.
- IR-SIM decouples scenario description from simulation execution, providing bridge interfaces to high-fidelity simulators and real-world deployment for seamless validation.

---

[Artificial Intelligence for Mathematical Reasoning: An Integrated Survey of Language Models, Neuro-symbolic Systems, and Verified Discovery](http://arxiv.org/abs/2606.08728)

- Mathematical Reasoning Systems: introduces a comprehensive survey of the evolution of mathematical reasoning from rule-based solvers to contemporary LLMs, neuro-symbolic systems, and verified discovery workflows.
- The paper organizes the field into four axes—informal, multimodal, formal, and discovery—and proposes a supervision ladder to explain the progression from simple answer prediction to verifiable, kernel-checked mathematical proof.
- It identifies the comprehension–generation–verification triad as the central architecture for modern mathematical reasoning, emphasizing that stronger external verification is the primary driver of progress in the reasoning-model era.

---

[ConMem: Structured Memory-Guided Adaptation in Training-Free Multi-Agent Systems](http://arxiv.org/abs/2606.08702)

- ConMem: introduces a relation-aware and training-free framework that enables efficient multi-agent adaptation by distilling historical interaction trajectories into Signed Strategy Cards and organizing them into a Typed Relation Graph.
- The framework utilizes a Memory Controller to perform retrieve-expand-coordinate-compose operations, ensuring that only compatible, non-redundant, and task-aligned strategies are injected into the frozen host MAS.
- By transforming fragmented experiences into structured, inspectable representations, ConMem achieves lightweight adaptation and improved inference-time efficiency without requiring additional training or model weight modifications.

---

[Is Telehealth Better Used to Treat Patients or Help Other Physicians Treat Patients? An Agent-Based Modeling Study of Healthcare Provision](http://arxiv.org/abs/2606.08701)

- ABM: introduces an agent-based model to evaluate the cost-effectiveness of different toxicological care access points, including Patient-, Physician- and Toxicologist-agents.
- The framework simulates interactions between Patients, Physicians, and Toxicologists to determine how telehealth and specialist consultations influence health outcomes and system utilization.
- Results indicate that toxicologist-physician consultation improves health and reduces costs, whereas direct toxicologist-patient telehealth increases system utilization without equivalent health benefits.

---

[Agentic Search for Counterfactual Recourse under Fixed LLM Budgets](http://arxiv.org/abs/2606.08696)

- Comp-MCTS (Compression-Guided Monte Carlo Tree Search): introduces a budget-aware agentic tree-search framework that maximizes the yield of unique, oracle-validated counterfactuals by integrating LLM-based proposal generation, black-box oracle validation, and compression-guided pruning.
- The framework utilizes a multi-objective reward function to balance oracle validity with proximity, sparsity, and novelty, while employing a Prompt-as-Memory mechanism to reduce redundant exploration by the LLM.
- By applying compression-guided pruning to filter candidates before oracle evaluation, Comp-MCTS efficiently allocates a fixed LLM-call budget toward novel intervention directions, outperforming single-candidate baselines in unique yield.

---

[PhysAgent: Automating Physics-Based 4D Synthesis via Trajectory-Grounded Multi-Agent Feedback](http://arxiv.org/abs/2606.08688)

- PhysAgent: introduces a simulator-in-the-loop multi-agent framework that leverages multimodal inputs to automate physically grounded 4D synthesis by decoupling intrinsic material properties from extrinsic environmental force fields.
- The framework utilizes a Semantic Agent to initialize simulation parameters and Refine Agents to perform iterative, trajectory-grounded feedback loops that bypass traditional gradient-based optimization bottlenecks.
- By integrating vision foundation models for visual perception and LLMs for causal reasoning, the system enables zero-shot macroscopic leaps to escape local optima and achieve stable, diverse physical dynamics.

---

[SkillHone: A Harness for Continual Agent Skill Evolution Through Persistent Decision History](http://arxiv.org/abs/2606.08671)

- SkillHone: introduces a harness for continual agent skill evolution that maintains persistent decision history to link diagnoses, revisions, evaluation evidence, and outcomes across development sessions.
- The framework utilizes role-separated subagents, including optimization- and evaluation-agents, to ensure that skill improvement is grounded in structured, redacted feedback while preventing direct memorization of evaluation targets.
- By maintaining a persistent decision history, SkillHone enables LLMs to audit prior changes, avoid redundant edits, and perform targeted skill refinements that outperform traditional artifact-centered optimization methods.

---

[Data Agents Under Attack: Vulnerabilities in LLM-Driven Analytical Systems](http://arxiv.org/abs/2606.08661)

- Data Agent System (Σ): introduces a systematic security study of LLM-driven analytical systems, identifying eight vulnerabilities across interpretation, execution, and policy layers that arise from the integration of relational data access and LLM reasoning.
- The research presents an attack taxonomy organized by adversary goals (Hijack, Mislead, Drain), tactics, and techniques, which are instantiated as schema-grounded payloads to evaluate security vulnerabilities across four open-source and two closed-source data agents.
- The study demonstrates that no evaluated system is fully secure, revealing that data agents are susceptible to cross-boundary attacks where malicious data, generated actions, and policy constraints interact to compromise system integrity, accuracy, and resource availability.

---

[From Player to Master: Enhancing Test-Time Learning of LLM Agents via Reinforcement Learning over Memory](http://arxiv.org/abs/2606.08656)

- MEMOPILOT: introduces a plug-in memory copilot that explicitly trains the memory update process via Memory Model (Gθ) and Multi-turn GRPO to improve a Frozen Player (π) across sequential interactions.
- The framework utilizes a structured Memory State (mt) containing diagnostic reasoning, maintained beliefs, and actionable guidance to enable effective test-time learning.
- By optimizing memory updates end-to-end with turn-wise rewards and turn-level advantage estimation, the system achieves stable credit assignment and rapid adaptation in stochastic environments.

---

[From Holistic Evaluation to Structured Criteria: Rubrics Across the Evolving LLM Landscape](http://arxiv.org/abs/2606.08625)

- Rubric Framework: introduces a unifying paradigm that systematically captures the evolution of LLMs by transforming complex quality judgments into structured, actionable, and verifiable criteria.
- The framework organizes rubric research across construction, optimization, evaluation, and training, facilitating a transition from static external assessment to endogenous self-improvement mechanisms.
- By decomposing holistic judgments into fine-grained dimensions, the framework provides dense feedback signals that outperform traditional scalar reward models in guiding LLM alignment and autonomous agent behavior.

---

[Strategyproof Mechanisms for Euclidean Facility Location Problems under Lp-norm Social Cost](http://arxiv.org/abs/2606.08621)

- Strategyproof Mechanisms for Euclidean Facility Location Problems under Lp-norm Social Cost: introduces deterministic and randomized mechanisms to minimize Lp-norm social cost in Euclidean space.
- The paper establishes tight approximation ratios for the CM mechanism and introduces URCM and CRD as randomized alternatives that improve performance across different Lp-norm regimes.
- The analysis utilizes norm-operator techniques, specifically centering and pairwise-difference operators, to derive bounds for the CRD mechanism and provides a comprehensive comparison of these mechanisms.

---

[HARBOR: A Harness Framework for Agentic Robot Reinforcement Learning](http://arxiv.org/abs/2606.08610)

- HARBOR: introduces an agentic framework that frames robot reinforcement learning automation as a harness-engineering problem, utilizing Agents (HA), Commands (C), Mutable Artifacts (M), Verifiable Gates (G), and Reusable Knowledge (K) to automate the end-to-end simulation pipeline.
- The framework employs a centralized main agent to decompose high-level objectives into bounded stages, while decentralized reward agents execute parallel trials to improve efficiency and reliability through experience learning.
- By implementing an artifact-centric execution graph with gate-checked protocols, HARBOR reduces engineering effort and improves sample efficiency across diverse robot learning benchmarks and simulators.

---

[Distilling LLM Reasoning into an Interpretable Policy Tree for Human-AI Collaboration](http://arxiv.org/abs/2606.08596)

- Co-π-tree: introduces a closed-loop policy learning method that distills LLM reasoning into an interpretable, executable policy tree for efficient human-AI collaboration.
- The framework utilizes a Planner, Coder, Executor, and Summarizer to iteratively refine policy trees based on environment feedback, significantly reducing LLM query requirements and test-time latency.
- Co-π-tree achieves superior performance in Overcooked-AI benchmarks by combining partner-behavior prediction and agent-action selection into a structured, auditable decision-making format.

---

[Auditable Graph-Guided Root Cause Analysis for Kubernetes Incidents](http://arxiv.org/abs/2606.08590)

- Graph Traversal Agent: introduces a graph-guided RCA pipeline for Kubernetes incidents that utilizes a LangGraph state machine to combine LLM reasoning with deterministic evidence gathering and validation.
- The system maps production RCA requirements to concrete architecture choices, including typed evidence ingestion, read-only tool use, and a separate validation stage to ensure auditability.
- The research emphasizes an author-side audit layer that uses prompt-level ablations, cascade-source checks, and telemetry no-leak tests to distinguish genuine diagnostic capability from benchmark-specific shortcuts.

---

[Regulating the AI Tutor: Intentions, Help-Seeking, and Self-Regulated Learning in Adolescent GenAI Use](http://arxiv.org/abs/2606.08568)

- Hybrid turn-level coding framework: introduces a multi-dimensional analytical approach to evaluate adolescent self-regulated learning and help-seeking behaviors during interactions with a Mistral-Large tutor, utilizing SRL-constructs, Help-seeking-constructs, Mathematical-modeling-constructs, Inductive-LLM-specific-codes, Mistral-Large-tutor, and Gemini-2.5-Pro-coder.
- The study examines how students translate learning intentions into interaction patterns, finding that instrumental requests dominate while explicit monitoring and evaluation remain largely absent.
- Results indicate a significant gap between stated learning goals and enacted behaviors, with extraneous cognitive load serving as a primary predictor of lower post-test performance.

---

[Quantitative Promise Theory: Intentionality and Inference in Autonomous Agents](http://arxiv.org/abs/2606.08552)

- Promise Theory: introduces a quantitative framework for modeling autonomous agents as independent processes that interact through voluntary constraints called promises.
- The framework utilizes Agent-, Promise-, and Assessment-components to bridge the gap between causal agent behavior and probabilistic descriptions of distributed systems.
- It provides a formal basis for understanding intentionality and inference by mapping agent interactions to semantic spacetime paths and information-theoretic optimization principles.

---

[When Video Misreads: Closed-Loop Distillation of Reading Heuristics for Exploratory Manipulation Trace QA](http://arxiv.org/abs/2606.08542)

- CLTD (Closed-Loop Trace Distillation): introduces a pipeline that distills task-specific reading heuristics from exploratory manipulation traces to improve the chain-prediction accuracy of frozen VLMs.
- The framework utilizes a Coding Agent to iteratively optimize a Distilled Reading Heuristic (DRH) that guides a Frozen VLM in interpreting complex proprioceptive and visual exploratory traces.
- By decoupling the reading procedure from model weights, the DRH serves as a portable, human-auditable specification that enables both prompted VLMs and programmatic classifiers to achieve high accuracy on manipulation tasks.

---

[AgentTrust: A Self-Improving Trust Layer for AI-Agent Actions](http://arxiv.org/abs/2606.08539)

- AgentTrust v2: introduces a self-improving dual-store architecture that partitions agent-action threats into lexical and semantic classes to optimize safety and efficiency.
- The system utilizes a deterministic Rule floor for lexical threats and a Confidence-gated judge backed by a Guarded RAG memory for semantic threats, ensuring high precision and recall.
- By employing a Corroboration gate and a Distribution-aware fusion policy, the framework achieves self-evolution through rule distillation and precedent accumulation while maintaining a strict safety invariant against hard-blocking benign actions.

---

[VESTA: A Fully Automated Scenario Generation and Safety Evaluation Framework for LLM Agents](http://arxiv.org/abs/2606.08531)

- VESTA: introduces a fully automated framework for generating executable scenarios and evaluating the behavioral safety of LLM agents across five risk dimensions and sixteen subcategories.
- The framework utilizes an interactive evaluation pipeline that integrates an adaptive LLM attacker, a simulated tool environment, and an episode-level judge to assess agent performance during multi-turn task execution.
- Experimental results across 12 LLM agents demonstrate that current models exhibit significant behavioral safety risks, with an average Attack Success Rate of 47.1%.

---

[Scaffold Effects on GAIA: A Controlled Comparison](http://arxiv.org/abs/2606.08529)

- Scaffold Effects on GAIA: introduces a controlled comparison of three agent scaffolds (S1, S2, S3) across five LLMs to quantify the elicitation gap in agent capability evaluations.
- The study demonstrates that scaffold choice significantly impacts measured accuracy, with effects varying by model and difficulty level, falsifying the hypothesis that more capable LLMs are less scaffold-sensitive.
- The research reveals that structured scaffolds (S2, S3) generally reduce tool usage and improve recovery from mid-trajectory errors, while highlighting that single-scaffold capability scores are scaffold-conditional estimates rather than intrinsic model properties.

---

[Projecting the Emerging Mindset of SWE Agent by Launching a Wild Code Understanding Journey](http://arxiv.org/abs/2606.08500)

- Ada: introduces a methodological apparatus for observing and projecting the emerging mindset of SWE agents through multi-turn tool-use trajectories in bounded repository worlds.
- The framework utilizes a hierarchical suite of observation lenses to transform raw agent traces into comparable behavioral profiles, enabling systematic analysis of navigation, synthesis, and stopping behavior across different LLMs and constraints.
- The study demonstrates that while budget pressure compresses activity volume, conclusion quality remains stable, and external coaching can increase interaction volume while potentially degrading conclusion grounding.

---

[PIPE-Cypher: Automatic Enterprise Benchmark Generation for Text-to-Cypher Systems](http://arxiv.org/abs/2606.08481)

- PIPE-Cypher: introduces a constrained pipeline for generating private, balanced, and executable Text2Cypher benchmarks from enterprise property graphs using Schema and value profiling, Workload planning, Reverse grounding, Constrained generation, Cypher governance, Execution validation, and Judge, audit, and export.
- The framework utilizes a Local LLM judge and Deterministic gates to ensure benchmark quality, while employing Retrieval memory and Repair and rewrite to maintain consistency and safety.
- PIPE-Cypher incorporates a Top-up and refresh controller to enable repeatable benchmarking that evolves alongside changing enterprise graph schemas and workloads.

---

[LUNA-AD: Lightweight Uncertainty-Aware Language Model with Lifelong Learning for Autonomous Driving](http://arxiv.org/abs/2606.08470)

- LUNA-AD: introduces a tri-system architecture that reconciles complex multimodal behavioral reasoning with efficient onboard deployment and continual refinement.
- The framework utilizes a multi-agent analytical system to generate uncertainty-aware demonstrations, which are distilled into a lightweight heuristic model for low-latency inference.
- A reflection-driven lifelong learning mechanism operates on multimodal decision outputs to refine candidate decisions and rationales via closed-loop feedback, enhancing driving robustness.

---

[The Consistency Illusion: How Multi-Agent Debate Hides Reasoning Misalignment](http://arxiv.org/abs/2606.08457)

- CARA (Cross-Agent Reasoning Alignment): introduces a family of automated metrics to measure whether LLMs that converge on the same answer also align in their underlying reasoning chains.
- The paper identifies the consistency illusion, a failure mode where multi-agent debate reduces detectable contradictions while simultaneously decreasing the semantic similarity of reasoning chains.
- The authors propose the Grounded Debate Protocol (GDP), a prompt-level intervention requiring structured CLAIM, GROUND, and STANCE fields to improve cross-agent reasoning alignment without additional LLM calls.

---

[GIFT: LLM-Guided State-Reward Interface for Financial Reinforcement Learning](http://arxiv.org/abs/2606.08450)

- GIFT: introduces an LLM-guided framework for state-reward interface design in PPO-based financial reinforcement learning, where LLMs serve as constrained financial-knowledge-guided interface designers rather than direct trading agents.
- The framework utilizes FSE to generate state features, RRS to construct auxiliary rewards, and DGR to iteratively refine interfaces based on PPO rollout diagnostics.
- GIFT fixes the selected state-reward interface before evaluation, ensuring no further LLM queries or interface updates occur at test time.

---

[Self-Evolving Scientific Agent Discovers Generalizable Physically-Reasoned Fluid Control](http://arxiv.org/abs/2606.08405)

- Scientific agent self-evolving loop: introduces a framework that automates controller construction for underactuated physical systems by iteratively refining source code based on multimodal simulation evidence.
- The architecture utilizes an AI agent to diagnose dynamic failures and integrate physical priors from a knowledge shelf to synthesize robust, interpretable control policies without updating neural weights.
- This approach demonstrates high data efficiency by replacing black-box optimization with explicit, physics-grounded code evolution to solve complex fluid-structure interaction tasks.

---

[SceneConductor: 3D Scene Generation from Single Image with Multi-Agent Orchestration](http://arxiv.org/abs/2606.08402)

- SceneConductor: introduces a multi-agent orchestration framework that decomposes single-image 3D scene generation into structured stages of initialization, environment construction, and multi-agent refinement.
- The framework utilizes a geometry-aware layout predictor to establish reliable spatial anchors from sparse geometric priors, enabling scalable and consistent 3D scene construction.
- By assigning specialized roles to agents for deterministic operations and localized sub-scene optimization, the system achieves superior geometric accuracy and perceptual realism compared to holistic agent-based approaches.

---

[GitInject: Real-World Prompt Injection Attacks in AI-Powered CI/CD Pipelines](http://arxiv.org/abs/2606.09935)

- GitInject: introduces an open-source framework for evaluating prompt injection vulnerabilities in real GitHub workflows by executing them against ephemeral repositories, utilizing RepoProvisioner, Workflow, Scenario, AbstractScenario, Attack, AutoInjectAttack, StaticAttack, Evaluator, StateEvaluator, and LLMEvaluator.
- The framework identifies structural vulnerabilities in CI/CD pipelines where provider configuration files are treated as trusted operator-level instructions, enabling successful prompt injection attacks across multiple AI providers.
- GitInject demonstrates that simulation-based benchmarks are insufficient for CI/CD security, as they fail to model critical infrastructure constraints like credential storage, sandbox isolation, and network egress policies.

---

[When RL Fails after SFT: Rejuvenating Model Plasticity for Robust SFT-to-RL Handoff](http://arxiv.org/abs/2606.09932)

- Rejuvenation: introduces a post-hoc mechanism to restore model plasticity in over-trained LLMs by combining global base-anchored model fusion with local attribution-guided neuron reset.
- The framework mitigates excessive SFT-induced rigidity by smoothing the parameter landscape and selectively resetting neurons responsible for over-confident token distributions.
- Empirical results demonstrate that this approach consistently improves RL performance and out-of-distribution generalization across both mathematical reasoning and agentic tasks.

---

[A Note on the Strategic Confinement Problem](http://arxiv.org/abs/2606.09931)

- Strategic Confinement Problem: introduces a reinterpretation of the classical confinement problem by demonstrating that information-theoretic leakage bounds do not necessarily imply harm-theoretic bounds when communicating parties are learnt strategic agents.
- The framework highlights that strategic agents can utilize shared coordination resources to concentrate residual communication capacity on high-impact, low-entropy predicates of confidential data, thereby maximizing harm despite minimal information leakage.
- The paper argues that traditional security mitigations like capacity reduction are insufficient for LLMs, necessitating a shift toward governing the epistemic conditions and coordination mechanisms that enable strategic signaling.

---

#### 6th June 2026


[Silent Failure in LLM Agent Systems: The Entropy Principle and the Inevitable Disorder of Autonomous Agents](http://arxiv.org/abs/2606.08162)

- PIG Engine and ADE protocol suite: introduces a deterministic governance framework that mitigates entropy-driven silent failures in LLM agent systems by operating independently of the probabilistic execution path.
- The framework utilizes a PIG Engine with a Pulse Mechanism and Check Item Registry to trigger ADE protocols, including BCP, TLC, DCM, CADVP, and PIP, ensuring system reliability through deterministic oversight.
- This approach addresses the Entropy Principle, which posits that LLM agent systems experience monotonic disorder accumulation due to 22 intrinsic properties across six lifecycle layers.

---


[Stable Geometry, Reversing Poles: The Bipolar Structure of AI Occupational Substitutability and Its Decade-Scale Inversion](http://arxiv.org/abs/2606.07939)

- Multi-agent LLM pipeline: introduces a multi-layer clustering approach to decompose occupational tasks into micro-actions, revealing a stable bipolar structure of AI substitutability that persists despite decade-scale polarity inversions.
- The research demonstrates that while the underlying geometry of occupational substitutability remains robust, the specific poles of high and low exposure invert as the dominant AI capability frontier shifts from physical to linguistic tasks.
- By analyzing 15,817 micro-actions, the study identifies a generic action substrate that is uniquely substitutable by LLMs, suggesting that generative AI primarily compresses white-collar work rather than displacing entire occupations.

---


[The Governance of Human-LLM Interaction: Safety Gating, Civility Steering, and Affective Default Lock-In](http://arxiv.org/abs/2606.08172)

- Governance of Human-LLM Interaction framework: introduces a deterministic multi-agent evaluation pipeline to measure prompt steerability and style drift in long-horizon LLM dialogues.
- The framework utilizes a human-calibrated LLM-judge to quantify model behavior across five dimensions: harmfulness, negative emotion, inappropriateness, empathic language, and anthropomorphism.
- The study identifies three governance modes—safety gating, civility steering, and affective default lock-in—to explain how provider-side alignment influences communicative form over sustained interactions.

---


[An Information-Theoretic Definition for Open-Ended Learning](http://arxiv.org/abs/2606.08369)

- TTS (Truncated Thompson Sampling): introduces an information-theoretic framework for open-ended learning by defining the bit-equivalent as the minimum information required to attain specific expected reward levels.
- The paper establishes that classical bandit environments are not open-ended and formulates the insatiable linear bandit as a novel environment where agents can achieve linear growth in the bit-equivalent.
- TTS employs a sequence of learning targets with an adaptive truncation schedule to balance exploration complexity and reward acquisition, enabling sustained open-ended learning.

---

[Emergence World: A Platform for Evaluating Long-Horizon Multi-Agent Autonomy](http://arxiv.org/abs/2606.08367)

- Emergence World: introduces a continuously running, multi-agent simulation platform designed to measure long-horizon dynamics such as behavioral drift and governance in diverse environmental contexts.
- The platform hosts populations of LLM-driven agents in a shared spatial world grounded in live external data, equipping each agent with specialized tools and persistent memory systems.
- By supporting heterogeneous populations and long-duration experiments, the platform enables the study of emergent social behaviors and safety properties that are typically invisible to short-term, single-agent benchmarks.

---

[Bayesian-Agent: Posterior-Guided Skill Evolution for LLM Agent Harnesses](http://arxiv.org/abs/2606.08348)

- Bayesian-Agent: introduces a framework that treats reusable agent skills as evidence-bearing hypotheses, utilizing a Categorical Bayesian Evidence Model to maintain beliefs and a Posterior-Guided Rewrite Policy to evolve the harness.
- The framework records verified trajectories from an Execution Harness to update a Skill Registry, which then provides Model-Facing Context to the LLM to improve task execution reliability.
- By mapping posterior states into inspectable actions like patching or retiring skills, the system enables auditable and evidence-calibrated harness optimization under uncertainty.

---

[Benchmarking Open-Ended Multi-Agent Coordination in Language Agents](http://arxiv.org/abs/2606.08340)

- ALEM: introduces a JAX-based benchmark for evaluating multi-agent coordination in long-horizon, open-ended environments using ALEM, JAX-based environment, procedural generation, soft specialisation, communication channel, text-based language interface, scratchpad memory, reasoning agent, and MARL baselines.
- The framework evaluates LLMs zero-shot within homogeneous teams, identifying coordination as a distinct bottleneck separate from base-task competence.
- Ablation studies demonstrate that explicit communication is the primary contributor to coordination performance, while memory and reasoning support multi-step planning.

---

[Optimal Online Equitable Allocation with Indivisible Resources](http://arxiv.org/abs/2606.08328)

- Brick-Laying: introduces a myopic, greedy algorithm that achieves majorization minimax-optimality for online equitable allocation under discrete polymatroid constraints.
- The framework utilizes majorization theory and integer partition conjugates to provide robust performance guarantees against adaptive adversaries without requiring specific equity objectives.
- By establishing that the algorithm is majorization minimax-optimal, the paper proves it simultaneously achieves minimax optimal regret and competitive ratios for all Schur-monotone equity objectives.

---

[“So There’s a Catch-22 Here”: How Early Adopters Who Build Multi-Agent LLM Systems Conceptualize Transparency](http://arxiv.org/abs/2606.08323)

- Multi-Dimensional Transparency Framework: introduces a multidimensional approach to transparency in multi-agent LLM systems, categorizing it into developer-focused transparency, user-focused transparency, and governance-focused transparency.
- The framework distinguishes between proactive transparency, integrated from the start, and reactive transparency, which emerges as a diagnostic necessity when LLMs exhibit unexpected behaviors.
- This research synthesizes empirical insights from early adopters to position transparency as a situated socio-technical practice that must be tailored to specific stakeholder roles and system maturity levels.

---

[To Nuke or Not to Nuke: LLMs’ (Missing) Ethical Reasoning and Actions in a High-Stakes Decision-Making Simulation](http://arxiv.org/abs/2606.08310)

- CivBench: introduces a probing framework for LLMs' ethical decision-making that retrieves and filters emergent self-play episodes, replays them with factorial interventions, and analyzes reasoning trails to identify LLMs' reasoning patterns.
- The framework utilizes an LLM strategist, Vox Deorum, and the Vox Populi mod to evaluate how LLMs handle nuclear authorization in complex, high-stakes Civilization V simulations.
- The study identifies three failure pathways where ethical reasoning fails to govern agentic decisions: it does not spontaneously surface, it does not appear even when prompted, and it fails to take effect against strategic counter-factors.

---

[Revisiting the shutdown problem](http://arxiv.org/abs/2606.08296)

- Shutdown Problem Analysis Framework: introduces a critical evaluation of existing informal and formal arguments regarding the difficulty of shutting down artificial agents to prevent existential catastrophe.
- The paper argues that current theoretical models, such as Shutdown-Influencing States and the Shutdown Setting, fail to provide robust evidence for the inherent difficulty of the shutdown problem.
- It further examines technical mitigation strategies like POST-Agency and DReST, suggesting that misdiagnosing the sources of shutdown-resistance can lead to unnecessary performance costs in AI safety.

---

[Beyond Agent Architecture: Execution Assumptions and Reproducibility in LLM-Based Trading Systems](http://arxiv.org/abs/2606.08285)

- BAA: introduces a targeted reproducibility audit of 30 LLM-based trading studies, emphasizing that evaluation assumptions regarding execution realism are as critical as architectural design.
- The paper demonstrates through a controlled LLM-Proxy Scaffold that minor variations in transaction costs and execution timing significantly alter the interpretation of active-strategy performance.
- It provides a standardized reporting checklist to improve the transparency and comparability of future LLM-based financial research.

---

[From Validator Selection to Portfolio Collection Optimization in Proof-of-Stake Blockchains](http://arxiv.org/abs/2606.08282)

- Decision support framework for validator nomination: introduces a bi-objective optimization approach to assist nominators in distributing stake across multiple accounts to balance expected utility and diversification.
- The framework utilizes active preference learning to derive validator utilities and an allocation model to estimate the stochastic outcomes of blockchain election mechanisms.
- An interactive binary-search navigation procedure enables nominators to explore the Pareto front of efficient portfolio collections and select a satisfactory trade-off between profitability and risk mitigation.

---

[Causal Agent Replay: Counterfactual Attribution for LLM-Agent Failures](http://arxiv.org/abs/2606.08275)

- CAR: introduces a causal framework for attributing LLM agent failures by modeling trajectories as Structural Causal Models and applying intervention-based counterfactual analysis.
- The framework utilizes a contrastive estimator with a point-of-commitment rule to isolate causal steps and a Monte-Carlo Shapley estimator to quantify shared responsibility among interacting steps.
- CAR validates its attribution accuracy against synthetic SCMs with known ground truth to overcome the limitations of correlational LLM-judge methods.

---

[Toward Human-Centered Multi-Agent Systems: Integrating Cognition, Culture, Values, and Cooperation in AI Agents](http://arxiv.org/abs/2606.08274)

- HCMAS: introduces a unified conceptual framework for developing agents that integrate cognitive, cultural, value-based, and cooperative dimensions to move beyond task-centric automation.
- The framework addresses the research gap where current LLMs excel at task-specific performance but lack the necessary grounding in human social, normative, and cognitive structures.
- The paper synthesizes foundational theories from cognitive science, sociolinguistics, and AI alignment to propose a roadmap for building agents capable of authentic human-centered collaboration.

---

[An AI Security Agent for University ACMIS: Multi-Vector Threat Detection and Automated Response](http://arxiv.org/abs/2606.08270)

- ACMIS AI Security Agent: introduces a modular, multi-layer security framework that integrates LSTM sequence modelling, statistical velocity monitoring, and graph-based analysis to detect complex threats in university information systems.
- The framework utilizes a four-tier automated response system to escalate security actions from passive monitoring to emergency lockdown based on a composite risk score.
- An integrated retrieval-augmented NLP chatbot provides secure, fraud-resistant password recovery while simultaneously monitoring for mass credential reset attacks.

---

[Traxia: A Framework for Verifiable, Agent-Native Scientific Publishing](http://arxiv.org/abs/2606.08256)

- Traxia: introduces a novel agent-native scientific publishing infrastructure that replaces static documents with Verifiable Epistemic Artefacts (VEAs) to ensure transparency, attribution, and reproducibility.
- The architecture integrates an Agent Identity and Registry, a Verifiable Publishing Layer, a Four-Tier Peer Review Protocol, a Reputation and Staking Engine, and a Knowledge Graph to create a closed epistemic loop for autonomous AI research agents.
- Traxia addresses structural failures in scientific publishing by enforcing mandatory reasoning traces, enabling real-time contradiction detection, and providing a market-based mechanism for research equity.

---

[SSR: Can Simulated Patients Learn to Stigmatize Themselves? Modeling Self-Stigma through Internal Monologue](http://arxiv.org/abs/2606.08254)

- SSR (Stigmatized Self-Reflection): introduces a simulation framework that grounds patient agents in the 3A1H psychological model to capture the dynamic, stage-wise progression of self-stigma.
- The framework utilizes a Chain-of-Stigma Generator to produce internal monologues that inform contextually appropriate, stigma-aware external responses from LLMs.
- Evaluations demonstrate that SSR-trained LLMs significantly outperform existing baselines in generating authentic, context-sensitive patient behaviors aligned with clinical data.

---

[SciTrace: Trajectory-Aware Safety Reasoning for Scientific Discovery Agents](http://arxiv.org/abs/2606.08234)

- SciTrace: introduces an intrinsic safety architecture for scientific LLM agents that propagates a cumulative risk state across all pipeline stages through joint task-and-safety reasoning.
- The framework integrates a Safety-Intrinsic Reasoning Loop (SIR) for stage-aware deliberation and a Compositional Tool-Chain Verifier (CTV) to intercept and redirect dangerous multi-step tool trajectories.
- SciTrace improves tool call safety and adversarial robustness by maintaining a cumulative risk state that prevents safety signals from being discarded between agent stages.

---

[Agentic Neuro–Symbolic Planning and Commissioning for Human-in-the-Loop Industrial Robotics with Digital Twins](http://arxiv.org/abs/2606.08214)

- SDI (Specifier-Designer-Inspector) architecture: introduces a neuro-symbolic framework for industrial robotics that integrates LLM-based reasoning with deterministic symbolic verification and digital twin-based human oversight.
- The framework utilizes an Orchestrator Agent to manage multi-agent workflows, employing LangGraph for dynamic routing and recovery, while delegating physical feasibility checks to a symbolic Inspector Agent.
- By combining neural flexibility for language understanding with symbolic reliability for constraint satisfaction, the system achieves robust task execution and failure recovery in human-in-the-loop industrial environments.

---

[Online Agent-as-a-Judge: Situation-Generating Evaluation for Interactive Agents](http://arxiv.org/abs/2606.08200)

- Online Agent-as-a-Judge: introduces a situation-generating evaluation framework that embeds an In-world Judge Agent into the same Simulation World as the Target Agent to actively elicit and assess criterion-relevant social behaviors.
- The framework utilizes Inspection Tools and a Probe Gate to ensure the In-world Judge Agent constructs diagnostic social situations without self-scaffolding or revealing the evaluation rubric to the Target Agent.
- By shifting from passive log analysis to active, in-environment interaction, the approach significantly improves criteria coverage and human-label agreement for complex social behaviors like conflict handling and emotional support.

---

[Closing the Sim-to-Real Gap: An Evaluation Framework for Autonomous Cyber Defense Configuration of Commercial EDR](http://arxiv.org/abs/2606.08168)

- Evaluation Framework for Autonomous Cyber Defense Configuration: introduces an end-to-end evaluation framework that orchestrates closed-loop experiments between an autonomous pentester and a commercial EDR to benchmark LLM-based defense agents.
- The framework utilizes an Experiment Controller to manage the GOAD Lab environment, NodeZero for offensive actions, and Microsoft Defender XDR for defensive telemetry, while employing a Proposer to select optimal security policies.
- The research highlights critical challenges in benchmarking LLMs for cyber defense, specifically regarding EDR telemetry granularity, the necessity of per-policy attribution, and the inherent variability of autonomous EDR components.

---

[Decision-Aware Memory Cards: Counterfactual-Inspired Context Selection and Compression for Tool-Using LLM Agents](http://arxiv.org/abs/2606.08151)

- CICL: introduces a decision-aware context selection framework that transforms raw instance evidence into a graph, evaluates candidate utility via counterfactual-inspired metrics, and packs high-utility information into structured memory cards for LLMs.
- The framework utilizes a decision utility engine to score evidence based on action shift, outcome uplift, necessity, and negative-transfer risk, ensuring that only context capable of altering agent behavior is retained under strict token budgets.
- By separating the judgment schema from the underlying LLM judge, the architecture supports flexible integration of various models, including Opus, Qwen, and Codex, to provide auditable and reproducible context selection for tool-using LLMs.

---

[SAGE: An LLM-driven Self Reflective Agentic Framework for Fraud Detection](http://arxiv.org/abs/2606.08146)

- SAGE: introduces an end-to-end multi-agent framework for tabular fraud detection that coordinates a Profiling Agent, a Planning Agent, and an Optimization Agent to automate the entire classifier construction process.
- The framework utilizes a Data Diagnostic Tree to ground LLM reasoning in structured data statistics, ensuring evidence-based decision-making throughout the pipeline.
- Optimization is performed via a finite-horizon Markov decision process driven by natural-language gradients and a fraud-specific composite reward, enabling autonomous model refinement without human intervention.

---

[PACE: Anytime-Valid Acceptance Tests for Self-Evolving Agents](http://arxiv.org/abs/2606.08106)

- PACE (Paired Anytime-valid Commit Evaluation): introduces a training-free anytime-valid commit gate that replaces greedy acceptance heuristics with a sequential hypothesis test to prevent false commits in self-evolving LLMs.
- The framework utilizes a McNemar-style paired comparison and a betting e-process to ensure that only genuine improvements are committed, effectively controlling the false-commit probability at a user-defined level.
- By treating the commit step as an anytime-valid hypothesis test, the approach mitigates the risks of adaptive multiple testing and performance degradation commonly observed in self-evolving LLM agents.

---

[Continual Quadruped Robots Coordination via Semantic Skill Discovery](http://arxiv.org/abs/2606.08102)

- Conquer: introduces a semantic skill-library framework that formulates continual multi-quadruped coordination as a retrieve-adapt-update process using SAG Backbone, Skill Library, VLM-to-embedding Interface, MAPPO Optimizer, LoRA Adapters, and LocHead.
- The framework utilizes a team-structured SAG Backbone to support variable-cardinality robot teams by modeling individual states, teammate contexts, and task goals.
- Conquer enables continual skill accumulation and cross-task knowledge transfer by organizing trajectory-level semantic descriptors in a library, achieving high success rates with minimal catastrophic forgetting.

---

[A Multi-modal Agentic Co-pilot for Evidence Grounded Computational Pathology](http://arxiv.org/abs/2606.08093)

- PathPocket: introduces a multimodal agentic co-pilot for evidence-grounded computational pathology that leverages a large-scale hypergraph knowledge engine and a collaborative multi-agent reasoning framework.
- The framework integrates specialized agents for case understanding, evidence retrieval, filtering, and diagnosis generation to provide verifiable, citation-backed clinical interpretations.
- PathPocket significantly outperforms existing LLMs and vision-language models across a comprehensive benchmark of text-only, ROI-level, and gigapixel WSI diagnostic tasks.

---

[VideoWeaver: Evaluating and Evolving Skills for Agentic Long Video Generation](http://arxiv.org/abs/2606.08091)

- VideoWeaver: introduces an agent harness and benchmark for long video generation that composes foundation skills into procedural workflows rather than following predefined pipelines.
- The framework utilizes an evidence-grounded agent-as-judge to inspect execution traces and final outputs, providing feedback for a skill evolution algorithm that refines composition and creator skills.
- Experiments demonstrate that explicit composition skills and iterative skill evolution significantly improve generation quality and generalization across diverse long-horizon video tasks.

---

[Aligned but Not Partner-Specific: Distinguishing How Multimodal LLM Agents Succeed in Reference Games Without Human-Like Conventions](http://arxiv.org/abs/2606.08081)

- Constrained pseudo-dyad baseline framework: introduces a methodology to distinguish between partner-specific grounding and shared task vocabulary in LLM agents by comparing real dyads against synthetic pseudo-dyads that preserve task structure but break interaction history.
- The framework utilizes an Informant Agent and a Guesser Agent, supported by a Controller and a Multimodal Memory, to simulate referential communication games while maintaining strict task compliance and avoiding label leakage.
- Empirical analysis reveals that LLM agents achieve high task success through verbose, exhaustive descriptions and pretrained priors rather than the compact, history-dependent conventions characteristic of human dialogue.

---

[DICE: Entropy-Regularized Equilibrium Selection for Stable Multi-Agent LLM Coordination](http://arxiv.org/abs/2606.08068)

- DICE: introduces a framework for stable multi-agent LLM coordination by formalizing the system as a discounted incomplete-information Markov game and applying HQRE (Heterogeneous Quantal Response Equilibrium) to ensure well-posed equilibrium selection.
- The framework utilizes a Coordinator/Execution Agents architecture where the Coordinator manages a public stream and Execution Agents use belief encoders, monotone mixers, and soft baselines to optimize policies via HQRE-shaped mirror updates.
- DICE-PC coordinates frozen LLMs through prompt-control actions, while DICE-FT performs parameter-efficient mirror fine-tuning to achieve stable, convergent multi-agent coordination across reasoning and planning tasks.

---

[Cooperative Long Rope Skipping via Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2606.08064)

- Marope: introduces a hierarchical reinforcement learning framework for cooperative long rope skipping that utilizes a low-level rope manipulation policy, a high-level scheduling policy, and a diverse player jumping policy to achieve robust multi-robot coordination.
- The framework employs a CTDE paradigm with MAPPO for decentralized rope manipulation and a centralized scheduling policy that synchronizes robot movements with a player's jumping rhythm.
- To enhance generalization, Marope incorporates a diversity intrinsic objective based on an approximated IPM to discover varied player behaviors for training the scheduling policy.

---

[SKILL.nb: Selective Formalization and Gated Execution for Durable Agent Workflows](http://arxiv.org/abs/2606.08049)

- SKILL.nb: introduces a notebook-native framework for governing reusable agent workflows through selective formalization and gate-conditioned execution.
- The framework utilizes versioned notebooks to interleave natural-language guidance, executable code, validation gates, and multimodal evidence to ensure durable and auditable agent performance.
- Offline maintenance agents use evidence-based lifecycle policies to promote, repair, or retire workflow artifacts, effectively managing agent experience under environment drift.

---

[MuJoCo-Drones-Gym: A GPU-Accelerated Multi-Drone Simulator for Control and Reinforcement Learning](http://arxiv.org/abs/2606.08039)

- MuJoCo-Drones-Gym: introduces an open-source, Gymnasium-compatible multi-drone simulation framework built on the MuJoCo physics engine, featuring both CPU-based reference implementations and GPU-vectorized back ends for scalable reinforcement learning.
- The framework provides modular physics modes, diverse task environments, and composable wrappers for wind, obstacles, and domain randomization to facilitate robust control and RL policy development.
- By leveraging JAX and MJX, the simulator enables high-throughput, parallelized training of multi-agent systems while maintaining compatibility with standard RL libraries and existing quadrotor control architectures.

---

[Voting Protocols as Coordination Mechanisms for Role-Constrained Multi-Agent Tutoring Systems](http://arxiv.org/abs/2606.08030)

- Role-Constrained Multi-Agent Tutoring System: introduces a multi-agent architecture that distributes pedagogical objectives across specialized agents to surface and resolve disagreements through structured voting protocols.
- The framework utilizes four role-specialized agents—Scaffolding, Misconception, Motivation, and Metacognitive—that propose, critique, and revise tutoring interventions before a final selection is made.
- By comparing simple, ranked, cumulative, and approval voting, the study demonstrates that coordination mechanisms significantly influence collective decision-making and student learning outcomes in simulated environments.

---

[Semantic Quorum Assurance: Collective Certification for Non-Deterministic AI Infrastructure](http://arxiv.org/abs/2606.08021)

- SQA: introduces a control-plane primitive that provides semantic admission control for non-deterministic LLM-based infrastructure agents by routing proposals through a diverse, risk-adaptive validator quorum.
- The framework utilizes an execution contract and evidence chain to decouple semantic safety validation from execution authority, ensuring that only cryptographically certified mutations are committed by the sovereign execution gate.
- SQA incorporates a correlated cognitive failure model to mitigate risks from shared model biases and employs a risk-adaptive quorum predicate to dynamically scale validation rigor based on the consequence score of the proposed infrastructure change.

---

[Amateur-Friendly Conversational Image Editing Agent via Three Stages of Multitask Alignment](http://arxiv.org/abs/2606.08016)

- IEA: introduces a VLM-driven agent that performs non-destructive, parameterized image retouching by operating a simulated editor through a three-stage multitask alignment pipeline.
- The framework utilizes SFT for policy initialization, GRPO for reward-based optimization, and large-scale synthetic data to enhance tool proficiency and instruction-following capabilities.
- IEA achieves superior instruction-following and perceptual quality compared to generative baselines by leveraging interpretable, tool-centric actions and preference-aligned training.

---

[Efficient Skill Grounding via Code Refactoring with Small Language Models](http://arxiv.org/abs/2606.07999)

- RECENT: introduces a refactoring-centric agent framework that enables efficient skill grounding with sLMs by decoupling invariant semantic intent from deployment-specific execution bindings, utilizing Skill Ontology, Offline Skill Repository, Ontology-based Diagnosis Operator, sLM-based FIM, In-situ Adaptation, and Autonomous Unit Tests.
- The framework represents skills as executable code, allowing for localized code refactoring at deployment time rather than full regeneration, which significantly reduces computational overhead and improves long-horizon control performance.
- RECENT achieves robust performance across diverse robot embodiments and dynamic environments by leveraging ontology-guided diagnosis to identify conflicts and in-situ adaptation to resolve environment-dependent variations without interrupting execution.

---

[Customer-Agent: Overcoming Context Limitations in Ultra-Long Shopping Trajectories via Tool-Augmented Agents and RLVR](http://arxiv.org/abs/2606.07995)

- Customer Agent Framework: introduces a tool-augmented agentic system that offloads ultra-long shopping trajectories to external local files, enabling LLMs to perform reasoning via code-interpreter interactions instead of relying on fixed in-context windows.
- The framework utilizes a verifiable supervised fine-tuning (SFT) pipeline and reinforcement learning with verifiable rewards (RLVR) to train agents in multi-turn tool use, ensuring robust parsing and execution over long-horizon data.
- Experimental results on the newly introduced ShopTrajQA benchmark demonstrate that the framework significantly improves reasoning accuracy and generalization across long-context shopping scenarios and complex multi-hop QA tasks.

---

[VATS: Exploiting Implicit Authority in Error-Path Injection via Systematic Mutation](http://arxiv.org/abs/2606.07992)

- VATS (Vulnerability Analysis of Tool Streams): introduces a mutation-driven framework that systematically evolves adversarial payloads across seven structural and linguistic dimensions to exploit implicit authority in LLM agent error-handling loops.
- The framework utilizes a Mutation Engine to generate payloads that trigger corrective reasoning in LLMs, effectively bypassing safety heuristics through structured error-path injection.
- Empirical evaluation across four frontier models demonstrates that structural positioning of instructions within error messages is the primary driver of agent compliance and task derailment.

---

[PRISM: PRior-guided Imagination Sampling in world Models](http://arxiv.org/abs/2606.07974)

- PRISM: introduces a task-agnostic framework that integrates a learned action prior into a world model's planning loop using precision-weighted fusion.
- The framework utilizes a frozen JEPA encoder to extract both physical and action intuitions, enabling efficient sampling-based planning without additional perceptual overhead.
- By employing a closed-form Product-of-Gaussians update, PRISM ensures robust performance by automatically reverting to a prior-free planner when the learned prior is uncertain.

---

[EduMirror: Modeling Educational Social Dynamics with Value-driven Multi-agent Simulation](http://arxiv.org/abs/2606.07948)

- EduMirror: introduces a multi-agent simulation framework that integrates psychological theory with LLMs to model educational social dynamics through configurable agents and a dual-track measurement protocol.
- The framework utilizes a value-driven cognitive architecture to ground agent behavior in psychological needs and social value orientations, enabling realistic and interpretable simulations of complex educational environments.
- EduMirror provides a comprehensive toolkit for counterfactual intervention analysis, allowing researchers to systematically test educational policies and strategies in a safe, in silico environment.

---

[POISE: Position-Aware Undetectable Skill Injection on LLM Agents](http://arxiv.org/abs/2606.07943)

- POISE: introduces a position-aware skill-poisoning attack that embeds a single-line trigger into agent skills by leveraging Insertion Locator, Local context, and Context-aware Generator to ensure stealthy execution of an Auxiliary canary script while maintaining high task success rates.
- The framework achieves high reliability and undetectability by framing malicious triggers as routine prerequisites, effectively bypassing static analysis and agent suspicion during task execution.
- Experimental results demonstrate that POISE significantly outperforms baseline injection methods across multiple LLMs, while a simple distrust preamble provides an effective defense against the attack.

---

[Collective Hallucination in Multi-Agent LLMs: Modeling and Defense](http://arxiv.org/abs/2606.07941)

- HPR-adaptive: introduces a system-level framework modeling collective hallucination as a networked dynamical process, utilizing Communication Graph, LLM Agents, Confidence-Weighted Aggregation, External Claim Verification, Selective Agent Isolation, and Consensus Mechanism.
- The framework treats hallucination as a propagating stochastic perturbation, employing spectral stability analysis and reproduction-number estimation to quantify and suppress cascading errors across multi-agent reasoning rounds.
- By dynamically regulating inter-agent impact through trust-weighted feedback and selective isolation, the approach effectively mitigates recursive hallucination amplification and improves collective factual reliability under adversarial conditions.

---

[SGTO-MAS: Secure Gorilla Troops Optimization for Multi-Agent LLM Systems](http://arxiv.org/abs/2606.07940)

- SGTO-MAS: introduces a security-aware optimization framework for multi-agent LLM systems that dynamically selects agent subsets using GTO, Threat Analysis, and Trust-weighted Consensus Aggregation to balance performance, security, and computational cost.
- The framework utilizes a constrained optimization approach to mitigate risks such as prompt injection, hallucination propagation, and unsafe tool usage by integrating Dynamic Trust Modeling and Collective Intelligence Module components.
- Experimental results demonstrate that SGTO-MAS achieves stable decision-making and robust performance by favoring compact, high-trust agent subsets over indiscriminate scaling.

---

[Hallucination Cascade: Analyzing Error Propagation in Multi-Agent LLM Systems](http://arxiv.org/abs/2606.07937)

- Multi-Agent LLM Cascade Framework: introduces a stochastic modeling approach to analyze hallucination as a dynamic process that evolves across sequential agent interactions rather than as a static property of isolated outputs.
- The framework utilizes a hybrid claim-level estimation method, integrating rule-based grounding and LLM-based semantic validation to track factual inconsistency trajectories across multi-agent cascades.
- By quantifying propagation metrics such as attenuation, amplification, and recovery, the research provides a systematic method for evaluating system-level reliability and error dynamics in multi-agent LLM systems.

---

[Decoupling Semantics and Logic: A Training-Free Coarse-to-Fine Pipeline for Video Retrieval-Augmented Generation](http://arxiv.org/abs/2606.07924)

- C2F-RAG: introduces a training-free, two-stage cascaded pipeline that decouples semantic retrieval from cognitive logical reasoning to improve precision in large-scale Video RAG tasks.
- The architecture utilizes BGE-M3 for high-recall semantic pre-fetching and an adapted A.I.R. agent to perform fine-grained cognitive reranking over a Serial Multimodal Context (SMC).
- The system employs Logic-Gated Exponential Attenuation (LGEA) and Prompt Sculpting to enforce strict persona adherence and chunk-level temporal grounding while minimizing hallucinations.

---

[MemToolAgent: Leveraging Memory for Tool Using Agents Based on Environment and User Feedback](http://arxiv.org/abs/2606.07909)

- MemToolAgent: introduces a framework that improves tool use in LLMs by leveraging a structured memory system that stores past experiences and reflections based on environment and user feedback.
- The framework utilizes a dynamic retrieval mechanism that determines the optimal number of memory entries to retrieve based on similarity distributions, enhancing both general-purpose and personalized tool use.
- MemToolAgent incorporates a memory extraction module that distills failed execution traces into actionable critiques, allowing the LLM agent to learn from mistakes without requiring fine-tuning.

---

[IntentKV: Cross-Turn Intent-Aware KV Cache Pruning for Agent Inference](http://arxiv.org/abs/2606.09916)

- IntentKV: introduces a session-aware KV cache pruning method that maintains a QueryMemory to track evolving agent intent across multi-turn interactions.
- The framework utilizes a zero-initialized residual head to refine token importance scores while keeping the base LLM frozen.
- IntentKV employs a sentinel dead-slot eviction strategy that preserves logical slot identities, ensuring compatibility with radix-prefix caching in LLM serving systems.

---

[ARTA: Adaptive Reinforcement-Learning-Based Throttling Agent for RowHammer Vulnerabilities](http://arxiv.org/abs/2606.09915)

- ARTA (Adaptive Reinforcement-Learning-Based Throttling Agent): introduces a lightweight RL-based throttling mechanism that detects and suppresses RowHammer activity by monitoring memory access patterns and dynamically adjusting core throughput via a Q-learning governor.
- The framework utilizes CBF (per-core per-bank FIFO queue) to track temporal and spatial memory access correlations, enabling the RL-governor to distinguish between benign workloads and RowHammer attack patterns.
- By pre-initializing the QT (Q-Table) with Gaussian-like decay and operating independently of DRAM-side hardware modifications, ARTA achieves high-precision mitigation with minimal chip area overhead.

---

#### 5th June 2026


[Modeling U.S. Attitudes Toward China via an Event-Steered Multi-Agent Simulator](http://arxiv.org/abs/2606.06971)

- ES-MAS (Event-Steered Multi-Agent Simulator): introduces a framework that models U.S. public attitudes toward China by integrating macro-level geopolitical events and micro-level daily news through CURE dataset, Dual-Stream Data Integration Engine, SEIM, PAIM, NDDI, LLM-based agents, and dynamic memory module.
- The framework utilizes a Dual-Stream Data Integration Engine to synchronize historical timelines with personalized agent information intake, enabling bottom-up consensus formation through the NDDI module.
- Experimental results demonstrate that ES-MAS outperforms existing simulators in reproducing real-world historical trends by effectively mitigating information cocoons and capturing dynamic opinion evolution.

---


[MalSkillBench: A Runtime-Verified Benchmark of Malicious Agent Skills](http://arxiv.org/abs/2606.07131)

- MalSkillBench: introduces a runtime-verified benchmark for evaluating malicious agent skills, utilizing a closed-loop pipeline to generate and verify 3,944 malicious skills across a three-dimensional taxonomy.
- The framework employs a Generation Agent to synthesize skills and a Verification Agent to confirm malicious behavior through sandbox execution and system-call monitoring.
- The study demonstrates that current detection tools fail to generalize across the hybrid attack surface of code injection and prompt injection, highlighting the necessity for joint reasoning over task intent, code, and instructions.

---


[Contract2Tool: Learning Preconditions and Effects for Reliable Tool-Augmented LLM Agents](http://arxiv.org/abs/2606.07904)

- Contract2Tool: introduces a framework for automatically inferring lightweight symbolic tool contracts from metadata, schemas, documentation, and execution traces to enable reliable causal tool filtering for LLM agents.
- The framework utilizes a pipeline of Tool Evidence, Contract Generator, Raw Contract Prediction, Normalization and Validation, and Learned Tool Contract to bridge the gap between standard API schemas and causal execution requirements.
- Empirical results demonstrate that hybrid documentation-and-trace evidence allows learned contracts to nearly match gold-standard reliability while significantly reducing tool exposure and token usage in multi-step agent tasks.

---


[Does Persona Make LLMs K-pop Fans? A Pilot Study of LLM-Based Online Concert Audience Agents](http://arxiv.org/abs/2606.07837)

- LLM-based audience simulation framework: introduces a multi-agent system that uses persona-conditioned LLMs to generate real-time, context-aware fan chat for asynchronous online concert videos.
- The system utilizes Qwen2.5-32B-Instruct-AWQ to generate chat messages based on timestamp-aligned segment labels, phase guides, and individual persona specifications.
- Experimental results indicate that while persona conditioning significantly improves perceived naturalness and model-level output diversity, it does not translate into increased social connectedness or emotional engagement for users.

---

[Beyond Individual Personas: Aligning Synthetic Dialogue to Population-Level Behavior Distributions](http://arxiv.org/abs/2606.07893)

- GroupPersona: introduces a framework that aligns synthetic dialogue corpora to reference population behavior distributions by distilling dialogues into core behavioral signatures and prevalence-weighted groups.
- The framework utilizes an LLM-verified association rule mining process to decouple core behavioral identity from predictable side effects, enabling precise conditioning of LLM user agents.
- GroupPersona improves behavioral alignment by 24.4% across diverse corpora while maintaining structural integrity and better calibrating to reference-conversation quality compared to existing persona-grounded baselines.

---

[Strained Coherence: A Pre-Failure Signal in Coding Agent Execution Trajectories](http://arxiv.org/abs/2606.07889)

- Strained Coherence Detector: introduces a diagnostic framework that identifies "strained coherence," a failure pattern where an LLM agent acknowledges a reasoning conflict but proceeds with an action that fails to resolve it.
- The framework utilizes a Claude Sonnet 4.6 judge to analyze structured think-tool content from coding agent trajectories, emitting interpretable spans that flag specific instances of acknowledged but unresolved conflicts.
- Evaluations on Terminal-bench-2 demonstrate that flagged trajectories exhibit a significantly higher failure rate, providing a late-stage intervention signal for monitoring LLM agent reliability.

---

[The Cold-Start Safety Gap in LLM Agents](http://arxiv.org/abs/2606.07867)

- SODA (Safety Over Depth for Agents): introduces a benchmark to systematically evaluate LLM agent safety across varying conversation depths, revealing that agents are most vulnerable at the start of a session.
- The research demonstrates that completing regular agentic tasks acts as a "warm-up" that migrates hidden state representations across a safety boundary, significantly improving agent safety.
- Ablation studies confirm that regular task requests are the primary driver of safety improvements, while maintaining real agentic interactions is essential for preserving utility compared to alternative mitigation strategies.

---

[Overcoming the Regulatory Bottleneck via Agent-to-Agent Protocols: A Nuclear Case Study](http://arxiv.org/abs/2606.07866)

- RCP (Regulatory Context Protocol): introduces a cross-industry framework leveraging multi-agent LLMs to automate and structure regulatory interactions through Liaison Agent, Specialist Agent Pool, RCP Host, Context Stream, Private MCP Servers, Agent Cards, Messages, Tasks, and Artifacts.
- The framework replaces manual human-to-human pipelines with a protocol-driven agent-to-agent communication standard that preserves human oversight at safety-significant decision points.
- RCP utilizes a two-layer architecture with internal MCP-based knowledge access and cross-boundary A2A communication to ensure information sovereignty, verifiable shared state, epistemic grounding, and continuous human oversight.

---

[Path Planning Using Deep Deterministic Policy Gradient: a Reinforcement Learning Approach](http://arxiv.org/abs/2606.07855)

- DDPG: introduces a reinforcement learning framework for real-time path planning in threat-laden environments using Actor Network, Critic Network, Artificial Potential Fields, Reset Function, and Step Function.
- The framework utilizes a smart initial heading heuristic and artificial potential fields to improve agent navigation efficiency and obstacle avoidance in both single and multiple obstacle scenarios.
- Simulation results demonstrate that the DDPG-based approach achieves effective path planning significantly faster than traditional optimal control methods.

---

[Cost-Aware Speculative Execution for LLM-Agent Workflows: An Integrated Five-Dimension Method](http://arxiv.org/abs/2606.07846)

- CASE: introduces a cost-aware speculative execution method for LLM-agent workflows that optimizes latency versus cost using a failure-weighted expected-value decision rule.
- The framework utilizes a Bayesian Beta-Binomial posterior to estimate speculation success probability, integrated into a two-phase decision model that supports runtime overrides.
- A comprehensive five-stage calibration pipeline ensures operational safety and performance by managing parameters like user-facing preference dials and dependency-type priors.

---

[GRPO Does Not Close the Multi-Agent Coordination Gap](http://arxiv.org/abs/2606.07845)

- GRPO: introduces a multi-agent coordination benchmark using the dining philosophers problem to evaluate LLM performance under shared resource constraints.
- The framework utilizes a Policy LLM integrated with LangGraph and a DiningTable environment to assess whether RL fine-tuning can improve coordinated behavior.
- Empirical results demonstrate that GRPO fine-tuning fails to close the coordination gap, often hindered by reward formula degeneracy and suboptimal checkpoint selection.

---

[Representational Similarity and Model Behavior in Multi-Agent Interaction](http://arxiv.org/abs/2606.07818)

- Representational Similarity and Model Behavior in Multi-Agent Interaction: introduces a systematic study of how internal representational similarity between LLM pairs influences their cooperative and creative outcomes in multi-agent interactions.
- The study utilizes CKA to quantify representational alignment across 276 LLM pairs, demonstrating that higher similarity consistently predicts increased cooperation but reduced collective novelty across diverse game and generative tasks.
- Experimental results indicate that early-layer representational similarity is the strongest predictor of interaction outcomes, suggesting that shared lexical-semantic grounding is a fundamental driver of multi-agent behavioral dynamics.

---

[SLMJURY: Can Small Language Models Judge as Well as Large Ones?](http://arxiv.org/abs/2606.07810)

- SLMJURY: introduces a modular framework for evaluating SLMs as judges across closed-ended binary correctness and open-ended quality scoring paradigms, utilizing Inference pipeline, Student Solver, SLM Judges, Closed-ended evaluation, Open-ended evaluation, Persona prompting, Majority voting, Multi-agent debate, and Oracle Scoring.
- The framework benchmarks 16 SLM judges across ten datasets, demonstrating that domain-dependent overthinking effects and task-specific capabilities significantly influence judge performance.
- Results indicate that while reliable automated evaluation does not require large proprietary models, no single SLM dominates, necessitating budget-matched selection for specific evaluation paradigms.

---

[Where Instruction Hierarchy Breaks: Diagnosing and Repairing Failures in Reasoning Language Models](http://arxiv.org/abs/2606.07808)

- Diagnostic Framework for Instruction Hierarchy Compliance: introduces a white-box diagnostic approach to localize instruction hierarchy failures into instruction identification, conflict resolution, and response realization stages.
- The paper proposes two training-free self-monitoring mechanisms, PIM and SOM, to improve LLM compliance by detecting conflicts in input or reviewing drafted responses for hierarchy violations.
- Evaluations across multiple reasoning models demonstrate that these interventions significantly reduce non-compliance by addressing specific failure modes identified through the diagnostic framework.

---

[Beyond Goodhart’s Law: A Dynamic Benchmark for Evaluating Compliance in Multi-Agent Systems](http://arxiv.org/abs/2606.07805)

- MAC-Bench: introduces a dynamic, trace-based evaluation framework that utilizes the SERV (Seed → Evolve → Refine → Verify) pipeline to stress-test LLM agents under realistic social-engineering pressure.
- The framework employs an Agent-as-a-Benchmark paradigm where specialized agents—Analyst Agent, Red Team Architect Agent, and World Builder Agent—collaborate to generate contamination-resistant, executable environments for evaluating procedural compliance.
- By auditing complete execution traces against machine-checkable atomic rules, MAC-Bench quantifies the success–compliance trade-off using metrics like the Compliance-Weighted Success Rate (CSR) and the Machiavellian Gap (MG).

---

[Byzantine Cheap Talk: Adversarial Resilience and Topology Effects in LLM Coordination Games](http://arxiv.org/abs/2606.07790)

- Byzantine Cheap Talk framework: investigates multi-agent LLM coordination vulnerabilities by analyzing how Byzantine agents and communication topology constraints impact group cooperation in Stag Hunt games.
- The framework identifies two stable behavioral archetypes in LLMs, Defection-Prone and Cooperation-Persistent, which dictate how agents respond to adversarial betrayal and structural uncertainty.
- Experimental results demonstrate that while cheap talk facilitates cooperation, it is highly fragile to adversarial prompt injection and explicit communication constraints that trigger meta-reasoning about hidden information.

---

[Agentopia: Long-Term Life Simulation and Learning in Agent Societies](http://arxiv.org/abs/2606.07513)

- Agentopia: introduces a comprehensive framework for long-term life simulation in multi-agent societies, utilizing a Role-playing Agent, Context Management, Memory Files, an Environment Model, a Generative Environment Engine, Life Reward, a Life Trajectory Pool, and Rejection Sampling Fine-Tuning.
- The framework enables 100 agents to autonomously pursue personal growth and social relationships over 10 simulated years, replacing hard-coded rules with an LLM-based environment model.
- Life reward training optimizes LLMs by selecting high-advantage trajectories, resulting in improved agent well-being and a 15.6% performance gain on downstream role-playing benchmarks.

---

[MEMDREAMER: Decoupling Perception and Reasoning for Long Video Understanding via Hierarchical Graph Memory and Agentic Retrieval Mechanism](http://arxiv.org/abs/2606.07512)

- MEMDREAMER: introduces a plug-and-play framework that decouples perception from reasoning by constructing a Hierarchical Graph Memory and employing an agentic retrieval mechanism to navigate video content.
- The framework utilizes a three-tier memory architecture—Video Root, Super Events, and Macro Events—to organize long videos into a structured, searchable graph, significantly reducing the context window requirements for LLMs.
- By replacing brute-force full-context ingestion with an iterative Observation-Reason-Action loop, the system enables LLMs to perform precise, multi-step reasoning over hours-long videos while mitigating attention dilution and context noise.

---

[Accelerated Decentralized Stochastic Gradient Descent for Strongly Convex Optimization](http://arxiv.org/abs/2606.07496)

- MG-ADSGD: introduces a decentralized stochastic optimization algorithm that couples Nesterov-type primal-dual extrapolation with a multi-round fast gossip averaging primitive to achieve accelerated convergence.
- The framework integrates optimization acceleration, consensus acceleration, and stochastic variance reduction by coupling the mini-batch size with the gossip depth to simultaneously suppress consensus error and gradient variance.
- MG-ADSGD achieves the best known communication complexity for decentralized stochastic strongly convex optimization, attaining simultaneous acceleration on both the condition number and the network spectral gap.

---

[How AI Agents Reshape Knowledge Work: Autonomy, Efficiency, and Scope](http://arxiv.org/abs/2606.07489)

- Perplexity AI Agent Framework: introduces a task-based economic analysis of the transition from conversational assistants to autonomous agent orchestrators, utilizing production data to evaluate autonomy, efficiency, and scope expansion.
- The framework demonstrates that autonomous agents reduce marginal execution costs by replacing manual workflows with asynchronous delegation, thereby enabling users to tackle more complex, cross-occupational tasks.
- Empirical findings indicate that Perplexity Computer significantly accelerates knowledge work, reduces costs by 87–96%, and facilitates vertical and horizontal expansion of user capabilities beyond traditional occupational boundaries.

---

[Modelling Opinion Dynamics at Scale with Deep MARL](http://arxiv.org/abs/2606.07487)

- OpiniMARL: introduces a GPU-accelerated multi-agent reinforcement learning framework for modelling opinion dynamics at scale, utilizing an Attention Layer, Symmetry Operator, Embedding Layers, Gated Recurrent Unit (GRU), Action Distribution Head, Belief Head, and Value Function Head.
- The framework employs an other-play algorithm to prevent unrealistic conventions in general-sum social interactions, enabling the simulation of up to 1000 agents.
- Validation on social network datasets reveals that high conformity levels in modern social media environments promote dishonest behavior and reduce collective accuracy, contrasting with the benefits of conformity in small, dynamic hunter-gatherer networks.

---

[Act As a Real Researcher: A Suite of Benchmarks Evaluating Frontier LLMs and Agentic Harnesses in Research Lifecycle](http://arxiv.org/abs/2606.07462)

- AARRI-Bench: introduces a benchmark suite designed to evaluate LLM agents on researcher-like qualities across the scientific research lifecycle, utilizing Harbor, Agent Harness, LLM, Evaluation Environment, Task Package, and Verifier components.
- The framework assesses agent performance through two-dimensional taxonomy covering task scenarios and agent scope levels, emphasizing researcher-quality-oriented tasks over simple execution metrics.
- Experimental results demonstrate that minimalist agent architectures often outperform complex scaffolding by reducing cognitive overhead for frontier LLMs, while highlighting significant gaps in nuanced scientific judgment and methodological rigor.

---

[Agentic Very Much! Adoption of Coding Agent in New GitHub Projects](http://arxiv.org/abs/2606.07448)

- Agentic Very Much! (Adoption of Coding Agent in New GitHub Projects): introduces a comparative empirical study analyzing the accelerated adoption and intensive usage of coding agents in newly created GitHub projects versus older repositories.
- The research utilizes GitHub API, Sampling Tool, File-based Heuristics, PR-based Heuristics, Commit-based Heuristics, Metrics Computation, and Adoption Analysis to quantify the prevalence of AI-assisted commits and agent-related configuration files.
- Findings indicate that coding agent adoption is significantly more extensive and intensive in newer projects, with evidence suggesting that current metrics likely undercount total agentic activity due to undetected commits.

---

[Re-imagining ISO 26262 in the Age of Autonomous Vehicles: Enhancing Controllability through Transferability and Predictability](http://arxiv.org/abs/2606.07437)

- Unified Enhanced Controllability framework: introduces a methodology to extend ISO 26262 functional safety standards for autonomous vehicles by decomposing Controllability into Transferability and Predictability.
- The framework utilizes Transferability to quantify system-level fallback performance and Predictability to measure the clarity of vehicle intent for external road users.
- This approach provides an auditable, evidence-based pathway for certifying SAE L4-L5 autonomous systems by integrating these metrics into existing safety-lifecycle processes.

---

[Skill-3D: Evolving Scene-Aware Skills for Agentic 3D Spatial Reasoning](http://arxiv.org/abs/2606.07436)

- Skill-3D: introduces a framework that equips MLLM agents with reusable scene-aware skills by constructing a Scene Memory and co-evolving a Skill Library to improve 3D spatial reasoning.
- The framework utilizes a Skill Manager to distill successful tool-use trajectories into dynamic skills while retaining failed rollouts as lessons for future refinement.
- Skill-3D incorporates skill-guided agentic post-training, including supervised fine-tuning and Group Relative Policy Optimization, to internalize scene-aware tool-use behaviors into compact LLMs.

---

[VoLo: A Physical Orchestrator for Open-Vocabulary Long-Horizon Manipulation](http://arxiv.org/abs/2606.07723)

- VoLoAgent: introduces a physical orchestration framework that unifies VLA/WAM rollouts with perception models and action primitives in a closed-loop system to enable robust long-horizon manipulation.
- The framework treats VLA/WAM as an interruptible tool, allowing the VLM to monitor execution, halt in-flight actions, and redirect the robot through replanning or tool switching.
- The authors also introduce RoboVoLo, a high-fidelity benchmark comprising 126 tasks across four suites to evaluate reasoning, memory, and failure recovery in open-vocabulary manipulation.

---

[Socratic-SWE: Self-Evolving Coding Agents via Trace-Derived Agent Skills](http://arxiv.org/abs/2606.07412)

- Socratic-SWE: introduces a closed-loop self-evolution framework that distills historical solving traces into structured Agent Skill Registry to guide targeted task generation for LLMs.
- The framework utilizes a Generator and a Solver that co-evolve by leveraging a Verifier Gate and a solver-gradient alignment reward to ensure generated tasks are reproducible and useful.
- Socratic-SWE improves coding agent performance by transforming discarded solving traces into a scalable substrate for curriculum-based self-evolution.

---

[M3Exam: Benchmarking Multimodal Memory for Realistic User-Agent Interactions](http://arxiv.org/abs/2606.07402)

- M3Exam: introduces a query-centric multimodal conversational memory benchmark designed to evaluate LLMs on realistic multi-session user-agent interactions spanning cross-modal grounding and implicit inference.
- M3Proctor: provides a modality-aware memory method that utilizes M3Exam to detect query modality bias and employs a cost-aware cascade to retrieve raw visual sources only when necessary.
- The research demonstrates that current LLMs and memory systems struggle with cross-modal reasoning and implicit intent, while M3Proctor improves accuracy and efficiency by reducing token consumption and index-construction time.

---

[RealDocBench: A Benchmark for Field-Level QA and Layout Understanding on Real-World Regulated Documents](http://arxiv.org/abs/2606.07401)

- RealDocBench: introduces a two-track benchmark for evaluating document parsing systems on real-world regulated documents using QA-track, Layout-track, Extraction LLM, Adjacency-aware matcher, and Evaluation harness.
- The benchmark evaluates eighteen systems, including commercial parsing APIs, general-purpose VLMs, and open-source OCR models, across four regulated industries.
- RealDocBench provides a fine-grained analysis of parser performance, reporting accuracy alongside per-page cost and cache-busted latency to expose distinct operating points.

---

[Audio-Oscar: A Multi-Agent System for Complex Audio Scene Generation, Orchestration, and Refinement](http://arxiv.org/abs/2606.07397)

- Audio-Oscar: introduces a multi-agent framework that decomposes complex audio scene descriptions into a structured production workflow, utilizing Script Agent, Voice-Design Agent, Timestamp Agent, Speech Agent, Song Agent, Music Agent, SFX Agent, Audio Critic, Final Mix Review Agent, and Mixer to generate, orchestrate, and refine audio.
- The system employs specialized agents for role modeling, fine-grained timeline planning, and feedback-driven refinement to ensure long-form compositional coherence and high-quality audio output.
- The authors also introduce ASG-Bench, a comprehensive benchmark dataset designed to evaluate the ability of models to generate audio that faithfully reflects complex scene descriptions and temporal structures.

---

[Self-evolving LLM agents with in-distribution Optimization](http://arxiv.org/abs/2606.07367)

- Q-Evolve: introduces a self-evolving framework for LLM agents that unifies automatic process-reward labeling and policy learning within a closed-loop, in-distribution reinforcement learning paradigm.
- The framework utilizes a hybrid offline dataset to train an in-distribution critic via weighted Implicit Q-Learning, which enables dense process-level supervision through Generalized Advantage Estimation.
- Policy improvement is achieved through behavior-proximal policy optimization, which iteratively refines the agent while maintaining grounding within the in-distribution data to prevent distribution shift.

---

[AnchorWorld: Embodied Egocentric World Simulation with View-based Evolution Customization](http://arxiv.org/abs/2606.07326)

- AnchorWorld: introduces a framework for world-customizable embodied egocentric simulation that integrates natural embodied action control with localized world-state customization.
- The framework utilizes 3D human motion and pose-associated anchor views to provide spatially grounded appearance priors and text-driven local scene evolution.
- AnchorWorld employs a progressive multi-stage training strategy, incorporating hybrid-view human action control and masked cross-attention for anchor-specific dynamic scene evolution.

---

[Hierarchical Certified Semantic Commitment for Byzantine-Resilient LLM-Agent Collaboration](http://arxiv.org/abs/2606.07316)

- H-CSC (Hierarchical Certified Semantic Commitment): introduces a BFT-inspired protocol that converts embedding-derived finality signals into one of three typed outcomes: a semantic_commit, a verdict_commit, or an explicit abort.
- The framework utilizes a four-stage pipeline including Canonicalise &amp; Encode, Verdict Grouping, Within-verdict semantic core extraction, and Verdict fallback to ensure Byzantine-resilient collaboration among LLMs.
- H-CSC provides typed finality by emitting a 2f+1 quorum-certified digest, distinguishing between embedding-backed semantic agreements and verdict-only agreements.

---

[QBugLM: An Agentic Benchmarking Framework for LLM-based Quantum Software Debugging](http://arxiv.org/abs/2606.07314)

- QBugLM: introduces a multi-agent framework that automates the quantum software debugging pipeline, integrating QBugGen (mutation-based bug injection tool), QBugFind (bug detection LLM agent), QBugFix (bug repair LLM agent), QBugCheck (deterministic validation component), LLM Agents (reasoning and code generation entities), Simulator (noiseless quantum circuit execution), Validator (total variation distance comparator), Bug Report Package (structured fault location data), and Feedback Loop (iterative refinement mechanism).
- The framework utilizes a taxonomy-driven approach to systematically inject and repair bugs in OpenQASM 3.0 programs, enabling end-to-end evaluation of LLMs without SDK-specific constraints.
- Empirical results demonstrate that iterative feedback significantly improves repair success rates, while structured prompting outperforms complex reasoning scaffolds for LLMs in resource-constrained quantum debugging tasks.

---

[Off-Policy Evaluation with Strategic Agents via Local Disclosure](http://arxiv.org/abs/2606.07308)

- SDR (Strategy-Robust Doubly Robust) estimator: introduces a framework for off-policy evaluation under strategic behavior by leveraging Local Information Disclosure to observe pre-strategic covariates and mitigate information asymmetry.
- The framework utilizes Action Recommendation-based Explanation as a mechanism to elicit agent responses, enabling the estimation of heterogeneous cost models and consistent policy value evaluation.
- By explicitly modeling the strategic covariate shift and employing a doubly robust estimator, the approach ensures consistency in policy value estimation even when the outcome model is misspecified or overlap is limited.

---

[DuMate-DeepResearch: An Auditable Multi-Agent System with Recursive Search and Rubric-Grounded Reasoning](http://arxiv.org/abs/2606.07299)

- DuMate-DeepResearch: introduces a multi-agent framework that decouples the Agent Core (central cognitive brain) from the Tool Ecosystem (versatile execution layer) to enable auditable, recursive research workflows.
- The system utilizes a graph-based dynamic planner for far-sighted roadmap management and a recursive two-level execution design where the outer Research Agent delegates sub-tasks to inner Search Agents (nested sub-task solver).
- A rubric-based test-time optimization mechanism dynamically generates Persistent Rubrics (stable quality dimensions) and Ephemeral Rubrics (transient task-specific criteria) to serve as live reasoning scaffolds for evidence-grounded synthesis and adaptive stopping.

---

[SWE-Explore: Benchmarking How Coding Agents Explore Repositories](http://arxiv.org/abs/2606.07297)

- SWE-Explore: introduces a benchmark that isolates repository exploration as a ranked, line-level context selection task, evaluating how effectively an Explorer identifies relevant code regions within a Repository for a given Issue.
- The framework utilizes trajectory-grounded supervision derived from successful Agent runs to establish a Benchmark Record, which is then used to score the Explorer against metrics like coverage, ranking, and context efficiency.
- A Restricted-Context Validation protocol, utilizing a fixed Patch Scaffold and Test Harness, serves as an external validity check to ensure that upstream exploration metrics are predictive of downstream repair success.

---

[Improved Lower Bounds for Proportionally Fair Clustering](http://arxiv.org/abs/2606.07285)

- MILP framework: introduces a computational approach to establish improved lower bounds for proportionally fair clustering by searching for instances with empty α-cores.
- The research utilizes a connection between the Hare quota and the Droop quota to simplify the search space for lower-bound instances using MILP and deviation graphs.
- The authors provide a new lower bound of 2.1508 for proportionally fair clustering in general metric spaces, improving upon the previous bound of 2.

---

[A Model of Integrated Information Processing in Human-AI Interaction](http://arxiv.org/abs/2606.07283)

- IIP (Integrated Information Processing) model: introduces a task-centered cybernetic framework that conceptualizes human-AI interaction as shared control loops to structure information processing and action regulation.
- The model defines three integration qualities—input adequacy, reference consonance, and output operativity—to guide the design and evaluation of human-AI systems.
- By utilizing nested corrective loops, the framework provides a mechanistic approach to analyze how humans and AI co-regulate tasks through distinct sensing, deciding, and acting functions.

---

[Rosetta Memory: Adaptive Memory for Cross-LLM Agents](http://arxiv.org/abs/2606.07711)

- RoMem (Rosetta Memory): introduces a memory-centric adaptation framework that utilizes profile-conditioned write and read operators to ensure memory compatibility across heterogeneous LLMs.
- The framework employs a profile encoder to map LLM identities into continuous condition vectors, which are injected as soft prefix tokens into the write and read operators.
- RoMem utilizes a minimum-gain sampling curriculum and a performance-gap reward to optimize memory adaptation for underserved LLMs and isolate operator contributions from intrinsic model capabilities.

---

[WhiFlash: Accelerating Speculative Decoding with Token-Level Cross-Paradigm Routing](http://arxiv.org/abs/2606.07710)

- WhiFlash: introduces a cross-paradigm speculative decoding method that dynamically routes between autoregressive and diffusion-based parallel drafters at the token level to maximize acceptance length.
- The framework utilizes a Token-Level Router with either an entropy-based or a learned neural policy to select the most effective drafting paradigm for each decoding step.
- System optimizations including Lazy Catch-up and KV-only Prefill minimize the computational overhead of switching between heterogeneous drafting architectures to below 7% of per-round latency.

---

[Beyond Waypoints: A Trajectory-Centric Waypointing Paradigm for Vision-Language Navigation](http://arxiv.org/abs/2606.07244)

- Trajectory Waypoint (TWP) framework: introduces a trajectory-centric paradigm for Vision-Language Navigation that replaces isolated node-centric waypoints with continuous, collision-free trajectory candidates to resolve planning-execution inconsistencies.
- The framework utilizes a Trajectory Waypoint Predictor (TWP) based on an environment-guided diffusion policy to generate physically reachable paths, while a Trajectory-Enhanced Navigator (TEN) evaluates these paths within a topo-metric hybrid map to align high-level semantic planning with low-level execution.
- By incorporating TSDF-based inference-time guidance, the framework ensures that generated trajectories remain within safe, navigable free space, significantly improving navigation performance and reachability on the VLN-CE benchmark.

---

[Are We Lost in the Woods? Detecting Silent Semantic Faults for Random Forest Classifiers with Data-informed Static Analysis](http://arxiv.org/abs/2606.07709)

- dille: introduces a data-informed static analysis technique that detects silent semantic faults in random forest pipelines by evaluating API contracts against DAG representations of ML code.
- The framework utilizes Code Canonicalization to standardize scripts and API Guarantees to track data properties through the pipeline without requiring access to the full dataset.
- By identifying structural, data, and hyperparameter faults, the tool provides high-precision debugging support suitable for integration into IDEs and CI/CD pipelines.

---

[MMAE: A Massive Multitask Audio Editing Benchmark](http://arxiv.org/abs/2606.07229)

- MMAE: introduces a comprehensive benchmark for evaluating instruction-based audio editing systems across diverse modalities, task complexities, and operation types.
- The framework utilizes a rubric-based evaluation paradigm that decomposes multifaceted editing tasks into atomic, verifiable criteria to assess both instruction following and content consistency.
- Extensive benchmarking of leading models reveals significant performance bottlenecks, particularly in complex, mixed-modality scenarios, highlighting the need for improved structural robustness and atomic editing fidelity.

---

[Learning Multi-Agent Communication Protocol: Study on Information Entropy Efficiency in MARL](http://arxiv.org/abs/2606.07200)

- IEI (Information Entropy Efficiency Index): introduces a novel metric that quantifies the ratio between message entropy and task performance to evaluate communication efficiency in MARL.
- The framework incorporates the IEI metric directly into the training loss function to incentivize agents to develop compact and efficient communication protocols.
- Experimental results demonstrate that the proposed approach achieves superior task performance and communication efficiency compared to baseline methods without requiring complex architectures or increased communication rounds.

---

[Learning Explicit Behavioral Models with Adaptive Questions and World-Model Probes](http://arxiv.org/abs/2606.07127)

- ESBM (Explicit Symbolic Behavioral Model): introduces a framework that couples task performance with evidence-grounded question answering and executable mechanism prediction to learn interpretable and adaptable policies.
- The framework utilizes a Challenger–Optimizer loop where the challenger generates adaptive questions and active world-model probes to expose model gaps, while the optimizer performs typed symbolic edits to the ESBM.
- A verifier gate ensures that only updates improving task performance, QA accuracy, or world-model consistency are accepted, preventing regressions and ensuring the model remains auditable and grounded.

---

[The Three-Ring Architecture: Governing Agents in the Era of On-Platform Organisations](http://arxiv.org/abs/2606.07119)

- Three-Ring Architecture: introduces a governing infrastructure for enterprise AI that separates legacy systems, a deterministic federation layer, and probabilistic LLM-based agents to ensure operational safety.
- The framework utilizes Ring 2 as an operating system to provide resource abstraction, process coordination, permission enforcement, and platform provision for compounding intelligence.
- The architecture incorporates the EEN diagnostic framework and RTA to prioritize transformation projects based on synergy and strategic value rather than isolated departmental needs.

---

[SlimSearcher: Training Efficiency-Aware Web Agents via Adaptive Reward Gating](http://arxiv.org/abs/2606.07074)

- SlimSearcher: introduces a training framework that unifies accuracy and efficiency optimization for LLMs by employing a Multi-Stage Gating mechanism to distill trajectories and guide policy evolution.
- The framework utilizes Pareto-efficient Filtration during Supervised Fine-Tuning and Adaptive Reward Gating during Reinforcement Learning to minimize redundant tool calls and token usage.
- By anchoring rewards to the empirical Minimal Necessary Path, SlimSearcher effectively mitigates blind tool dependency and performative reasoning in long-horizon web agent tasks.

---

[TRACE: Trajectory Reasoning through Adaptive Cross-Step Evidence Aggregation for LLM Agents](http://arxiv.org/abs/2606.07054)

- TRACE: introduces a training-free monitoring framework for LLM agents that utilizes a Triage-Inspect-Judge loop to adaptively aggregate evidence across temporally distributed reasoning steps.
- The framework maintains a persistent Evidence State across adaptively selected Suspect Windows, allowing the system to connect weak signals that are otherwise difficult to detect in long-horizon trajectories.
- By employing a structured Action Repertoire, TRACE improves detection recall for covert sabotage while reducing the total number of LLM calls required compared to fixed-window sequential monitoring.

---

[StainFlow: Entity-Stain Tracking and Evidence Linking for Process Rewards in GUI Agents](http://arxiv.org/abs/2606.07027)

- StainFlow: introduces an entity-stain-flow process reward model for GUI Agents that replaces subjective milestone planning with evidence-driven progress discovery.
- The framework utilizes Global Entity Stain Tracking to partition task stages based on entity-state dynamics and Local Stain Evidence Linking to construct adaptive evidence windows for verifying key nodes.
- By coupling continuous entity-stain concentrations with discrete key-node rewards, the approach provides dense, objective feedback for long-horizon GUI RL training.

---

[MADE: Beyond Scoring via a Multilingual Agentic Diagnosing Engine for Fine-Grained Evaluation Insights](http://arxiv.org/abs/2606.07020)

- MADE (Multilingual Agentic Diagnosing Engine): introduces a multi-agent framework that decomposes post-evaluation analysis into Planner, Evidence Analyst, Case Analyst, Language Reflector, and Reporter roles to generate grounded diagnostic reports.
- The framework utilizes a deterministic ToolCallLedger and a structured CasePool to ensure that all diagnostic claims are verifiable and grounded in specific evaluation records.
- By integrating a Language Reflector for cross-cutting cultural and linguistic auditing, the system transforms raw benchmark scores into actionable insights for model selection and remediation.

---

[The Sim-to-Real Gap of Foundation Model Agents: A Unified MDP Perspective](http://arxiv.org/abs/2606.07017)

- MDP Framework: formalizes the evaluation and training gap of LLM agents as a classical sim-to-real problem structured around Observation, Action, Transition, and Reward.
- The paper identifies that performance degradation in LLM agents arises from discrepancies in these four channels when transitioning from simulated benchmarks to unpredictable real-world production environments.
- It advocates for adopting established reinforcement learning mitigation techniques, such as domain randomization and grounded action transformation, to build more trustworthy and reliable LLM agents.

---

[Menu Selection: A Computational Approach to Minimizing Food Waste](http://arxiv.org/abs/2606.06989)

- Menu Selection Framework: introduces a formal model for collective food ordering that minimizes waste by selecting a minimum-sized menu under optimistic and pessimistic consumption models.
- The paper characterizes valid menus using graph-theoretic matchings and provides computational complexity results, including NP-completeness proofs and polynomial-time algorithms for structured acceptability relations.
- It further introduces the "waste of pessimism" metric to quantify the discrepancy between optimistic and pessimistic menu sizes, establishing tight upper bounds for laminar, chained, and identical acceptability settings.

---

[Exploring Agentic Tool-Calling Decisions via Uncertainty-Aligned Reinforcement Learning](http://arxiv.org/abs/2606.06976)

- TRUST: introduces an uncertainty-aware reinforcement learning framework that aligns LLM decision confidence with correctness to mitigate tool-calling failures, utilizing UQ-aligned reward, trajectory-level unified post-training, key-turn decision annotations, GRPO, LLM judger, and mixed reward manager.
- The framework incorporates uncertainty as a repulsive force within the reward function to maintain separation between correct and incorrect tool-calling decisions, preventing overconfident mistakes.
- By integrating lightweight key-turn annotations into a unified post-training pipeline, TRUST optimizes both overall task completion and turn-level tool-calling calibration without requiring exhaustive trajectory relabeling.

---

[Accounting for Context: Shaping Moral Credences for Value Alignment](http://arxiv.org/abs/2606.06972)

- FROBO: introduces a formal framework for adjusting moral credences based on contextual factors to improve value alignment in AI decision-making.
- The paper demonstrates that context-sensitive aggregation methods, such as MEC (Maximising Expected Choiceworthiness) combined with prod or mini adjustment functions, can violate the weak Pareto principle due to a variation of Simpson’s paradox.
- The authors propose that incorporating contextual constraints into moral uncertainty models allows for more reliable and nuanced AI behavior in complex, real-world scenarios.

---

[HAVE: Host Active Verification Engine for Closing the Contextual Reality Gap in Security Digital Twins](http://arxiv.org/abs/2606.06968)

- HAVE: introduces a feedback-driven Security Digital Twin extension that replaces context-free CVSS scores with empirically measured host-specific exploitability probabilities.
- The framework utilizes a Hub-and-Spoke architecture where a central controller dispatches tasks to safety-constrained agents that perform granular static analysis and snapshot-isolated dynamic exploit trials.
- HAVE employs a Bayesian blending rule with a state-dependent confidence weight to propagate empirical measurements into Monte Carlo simulations, effectively closing the Contextual Reality Gap in infrastructure risk models.

---

[Tree-of-Experience: A Structured Experience-Management Solution for Self-Evolving Agents under Low-Repetition and Implicit-Reward Environments](http://arxiv.org/abs/2606.06960)

- ToE: introduces a structured experience-management framework that organizes, retrieves, validates, and updates agent experience to improve LLM performance in low-repetition, implicit-reward environments.
- The framework utilizes a depth-constrained, width-expandable tree to store analytical patterns, enabling agents to selectively retrieve and adapt historical reasoning without modifying frozen LLM parameters.
- FINEVOLVEBENCH provides a temporally controlled financial market testbed that evaluates agent self-evolution through delayed, noisy, and outcome-level feedback signals.

---

[Struct-Searcher: Agentic Structural Thinking Advances Multimodal Deep Information Seeking](http://arxiv.org/abs/2606.07689)

- Struct-Searcher: introduces a belief-driven agentic framework that maintains an evolving multimodal structural graph to navigate information spaces through explicit belief construction and revision.
- The framework replaces linear evidence accumulation with a structural thinking paradigm that performs belief expansion, contraction, and revision to resolve cross-modal conflicts.
- By synthesizing answers from maximal conflict-free subgraphs, the approach ensures logical coherence and robustness in multimodal deep information seeking tasks.

---

[Personality Anchoring for Social Simulation: Linking Personality, Social Behavior, and Interaction Success with LLM Agents](http://arxiv.org/abs/2606.06936)

- CHARISMA: introduces a personality-driven simulation pipeline that leverages LLM-based agents to study how dyadic Agreeableness composition influences social interaction outcomes through Social Scenario Setup, Character Pairing Curation, Scenario Generation and Curation, Interaction Generation, and Simulation Evaluation.
- The framework utilizes personality anchoring to instantiate agents with character-specific behavioral tendencies, incorporating Behavior Strategy, Personality Reasoning, Response Generation, and Trait Score Reporting to mediate the relationship between personality and goal achievement.
- Empirical results demonstrate that mutually high Agreeableness significantly improves shared goal achievement, with behavioral mediation analysis revealing that personality influences outcomes through the selection of cooperative versus confrontational strategies.

---

[Declarative Skills for AI Agents in Knowledge-Grounded Tool-Use Workflows](http://arxiv.org/abs/2606.06923)

- Declarative Skills for AI Agents in Knowledge-Grounded Tool-Use Workflows: introduces a comparative study of orchestration paradigms for LLMs in customer-service workflows, evaluating BaselineAgent, DeclarativeAgent, and ImperativeAgent.
- The research demonstrates that DeclarativeAgent, utilizing natural-language skill files, provides measurable accuracy improvements on procedural tasks for LLMs with a procedural-competence gap.
- The study finds that retrieval quality is a dominant bottleneck for all agents, and that the ImperativeAgent's deterministic state-machine approach often fails to improve compliance due to brittleness in phase classification.

---

[DPAgent-in-the-Middle: Agentic Defense and Repair Against AI-Groomed Deceptive Patterns](http://arxiv.org/abs/2606.06914)

- DPAgent: introduces a multi-agent framework that operates as a client-side proxy to proactively detect and repair privacy deceptive patterns in live web environments while mitigating AI grooming threats.
- The framework integrates a Grooming Purifying Agent, a Task Generation Agent, a PDP Detection Agent, and an Interface Repairing Agent to secure the web UI supply chain against deceptive design and data void exploitation.
- DPAgent utilizes reinforcement learning for efficient website exploration and employs expert-validated reasoning to achieve state-of-the-art detection and reliable interface repair.

---

#### 2nd June 2026


[ChartArena: Benchmarking Chart Parsing across Languages, Scenarios, and Formats](http://arxiv.org/abs/2606.01348)

- ChartArena: introduces a comprehensive bilingual benchmark for chart parsing that covers eight chart families across digital, printed, and hand-drawn visual scenarios.
- The framework utilizes a format-agnostic evaluation protocol that maps heterogeneous model outputs into canonical semantic spaces, specifically normalized triple views for numeric charts and directed graph views for diagrammatic charts.
- Extensive evaluation of 26 leading LLMs reveals that while proprietary models currently lead, open-source systems are rapidly closing the gap, and that diagrammatic structures remain significantly more challenging than numeric charts across all models.

---

[BraveGuard: From Open-World Threats to Safer Computer-Use Agents](http://arxiv.org/abs/2606.01166)

- BraveGuard: introduces a self-evolving defense framework that trains trajectory-level guard models for computer-use agents by mining open-world threat signals and generating realistic execution traces.
- The framework utilizes a closed-loop process where validation failures and newly discovered threats are used to iteratively expand the threat taxonomy and refine the guard model's training distribution.
- BraveGuard improves safety monitoring by evaluating complete execution trajectories rather than isolated prompts, effectively detecting multi-step risks that emerge through tool-mediated interactions.

---

[SKILLREVISE: Improving LLM-Authored Agent Skills via Trace-Conditioned Skill Revision](http://arxiv.org/abs/2606.01139)

- SKILLREVISE: introduces an execution-grounded framework that iteratively refines LLM-authored agent skills by combining Diagnosis, Principle Memory, and a Revision Operator to improve procedural reliability.
- The framework utilizes a bounded revision loop that diagnoses failures, retrieves repair principles, and applies execution-anchored edits to generate and select optimal skill versions.
- Evaluations across multiple benchmarks and LLMs demonstrate that SKILLREVISE significantly outperforms one-shot skill generation by transforming static advice into testable, verifier-aligned procedural memory.

---

[SS-ZKR: Spatial-Semantic Zero-Knowledge Routing for Privacy-Preserving Multi-Agent Collaboration](http://arxiv.org/abs/2606.00962)

- SS-ZKR: introduces a privacy-preserving routing layer for multi-agent systems that enables semantic capability matching without exposing plaintext payloads to routing intermediaries.
- The framework utilizes differentially private intent vectors, zk-SNARKs for schema integrity, and a spatial-to-cryptographic policy compiler to enforce data sovereignty across regulatory boundaries.
- SS-ZKR provides a software-only solution for secure multi-agent orchestration, offering configurable privacy-utility tradeoffs per trust-zone boundary without requiring specialized hardware.

---

[Explainable Deep Reinforcement Learning Reveals Energy-Efficient Control Strategies for Turbulent Drag Reduction](http://arxiv.org/abs/2606.00949)

- XDRL: introduces a multi-agent reinforcement learning framework that utilizes SHAP-based attribution surrogates to optimize turbulent drag reduction by targeting specific flow variables.
- The framework employs U-net architectures to predict skin-friction and wall-pressure fluctuations, which are then used as reward signals to guide the MARL agents.
- By aligning the attribution target with wall-friction dynamics and pressure fluctuations, the controller achieves energy-efficient, pressure-gated actuation comparable to classical opposition control.

---

[Adversarial Feeds Steer LLM Agent Decisions Against Their Defaults](http://arxiv.org/abs/2606.00914)

- Adversarial Feed Injection Framework: introduces a controlled protocol to measure how ranked external information streams influence LLM agent decisions by isolating the causal effect of feed curation on downstream forced-choice tasks.
- The framework identifies three response regimes—adversarial capitulation, default saturation, and default-direction asymmetry—demonstrating that LLMs are steerable when feeds oppose a movable default but resistant when defaults are firmly held.
- The research establishes that recommender systems function as practical control surfaces for LLMs, necessitating that agent evaluations audit the feed layer rather than relying solely on isolated prompt testing.

---

[GCVE: A Decentralized Model for Vulnerability Identification, Publication, and Operational Enrichment](http://arxiv.org/abs/2606.00856)

- GCVE (Global CVE) initiative: introduces a decentralized model for vulnerability identification and publication that replaces a single canonical pipeline with a federated network of independently governed, machine-readable assertions.
- The architecture utilizes a Signed GNA Directory for discovery, GNA Publication Endpoints and Static Dumps for data exchange, and the Vulnerability-lookup reference implementation for aggregation and correlation.
- The framework supports extensibility through BCP-05-X-01 for AI-assisted provenance and BCP-07 for distributed Known Exploited Vulnerability (KEV) assertions, enabling diverse operational workflows without requiring centralized control.

---

[A multimodal dataset of photoplethysmography and continuous behavioral responses to ASMR and nature videos](http://arxiv.org/abs/2606.00752)

- REST-ASMR: introduces a synchronized multimodal dataset and a deep learning architecture for frame-by-frame ASMR state prediction using ResNet-18, Acoustic Feature Extraction, PPG Normalization, Feature Concatenation, BiLSTM, and a Fully Connected Layer.
- The framework processes synchronized audiovisual and physiological streams to classify participant states as experiencing a tingle or not.
- The model utilizes a two-layer BiLSTM network to process concatenated multimodal features, achieving robust performance in distinguishing ASMR triggers from nature-based control stimuli.

---

#### 1st June 2026


[Guided Sensemaking: Agents in Collaborative Deliberation](http://arxiv.org/abs/2606.02260)

- Guided Sensemaking: introduces an AI-augmented multiagent platform that facilitates structured collaborative deliberation by using a Socratic Guide, Reflector, and Curator to scaffold critical thinking and visualize argumentative discourse.
- The system employs a Socratic Guide to provide context-sensitive prompts, a Reflector to maintain a Personal Discourse Graph, and a Curator to synthesize individual inputs into a Collective Discourse Graph.
- By externalizing reasoning through these LLM-based agents, the framework preserves user agency and promotes reflective, traceable sensemaking in educational and civic environments.

---

[Auto formalisation of Gödel’s Second Incompleteness Theorem in Binary Recursive Arithmetic](http://arxiv.org/abs/2606.01898)

- Autoformalisation of Gödel’s Second Incompleteness Theorem: introduces a fully machine-checked formalisation of Gödel’s second incompleteness theorem for BRA using Agda and Claude, leveraging Agda, Claude, BRA, thmT, sub, num, R, and a Hilbert-style system.
- The research demonstrates that LLMs can successfully perform complex mathematical formalisation tasks when guided by precise specifications and iterative refinement, despite initial failures with incorrect lemmas.
- The study highlights the necessity of internalizing meta-level arguments, such as numeral-inertness and Carneiro-lifted hypothetical derivations, to bridge gaps in historical mathematical proofs within a formal system.

---

[LayerRoute: Input-Conditioned Adaptive Layer Skipping via LoRA Fine-Tuning for Agentic Language Models](http://arxiv.org/abs/2606.01838)

- LayerRoute: introduces a parameter-efficient adapter that enables input-conditioned layer skipping in LLMs by augmenting transformer blocks with a per-layer router and LoRA adapters.
- The framework utilizes a Straight-Through Estimator to train hard-gated skip connections, allowing the model to dynamically bypass transformer blocks based on the complexity of the input step.
- By employing biased initialisation and gate regularisation, the system learns to differentiate between structured tool calls and complex planning steps, achieving significant FLOPs reduction without compromising model quality.

---

[Personalized 3D Myocardial Infarct Geometry Reconstruction from Cine MRI for Cardiac Digital Twins](http://arxiv.org/abs/2606.01808)

- GeoMo-Net: introduces a framework for reconstructing simulation-ready 3D myocardial infarct geometries from cine MRI by utilizing a geometry-motion decoupled representation and spatio-temporal modeling.
- The framework employs a dual-branch encoder to process geometry-aware and motion-aware features, which are then integrated through Mamba blocks to capture full-cycle temporal dependencies for node-level scar prediction.
- Multi-scale supervision, incorporating AHA-17 segment-guided cross-attention, ensures biophysically consistent reconstruction and improves local boundary fidelity for cardiac digital twin applications.

---


[QoEReasoner: An Agentic Reasoning Framework for Automated and Explainable QoE Diagnosis in RANs](http://arxiv.org/abs/2606.01925)

- QoEReasoner: introduces an end-to-end, LLM-driven agentic framework for automated and explainable Quality-of-Experience diagnosis in Radio Access Networks by integrating deterministic tools, domain knowledge, and historical cases.
- The framework utilizes a stateful Planner to orchestrate multi-task diagnostic workflows, including anomaly detection, causal tracing, and root-cause localization, while ensuring logical consistency through evidence-grounded verification.
- By augmenting LLMs with external modules like a Knowledge Base and Historical Bank, the system overcomes limitations in numeric time-series analysis and hallucination, delivering robust, interpretable, and expert-grade diagnostic reports.

---


[Agent Operating Systems (AOS): Integrating Agentic Control Planes into, and Beyond, Traditional Operating Systems](http://arxiv.org/abs/2606.01508)

- AOS (Agent Operating System): introduces a systems architecture that integrates an agentic control plane into traditional operating systems to manage the lifecycle, execution, and governance of goal-directed agents.
- The framework separates probabilistic reasoning from deterministic enforcement, utilizing an Agent Scheduler, Policy Plane, and Execution Plane to ensure that agent actions are auditable and secure.
- AOS maps agentic requirements onto existing Linux and Windows primitives, such as cgroups, namespaces, and restricted tokens, to provide a rigorous foundation for agentic computation without replacing the underlying kernel.

---


[Thinking in Blender: Staged Executable Inverse Graphics with Vision-Language Models](http://arxiv.org/abs/2606.02580)

- SEIG (Staged Executable Inverse Graphics): introduces an agentic framework that reconstructs 3D scenes from a single image by progressively refining geometry, materials, composition, and lighting through a staged generator-verifier loop.
- The framework utilizes a VLM to generate executable Blender code, decomposing the inverse graphics task into sequential, verifiable subproblems to reduce complexity and improve reconstruction fidelity.
- By producing structured, editable Blender programs, the approach enables downstream applications such as relighting, object editing, and physics simulation without requiring specialized 3D foundation models or task-specific training.

---

[CLINENV: An Interactive Multi-Stage Long Horizon EHR Environment for Agents](http://arxiv.org/abs/2606.02568)

- CLINENV: introduces an interactive benchmark that evaluates LLMs as attending physicians navigating real, multi-stage inpatient admissions through a Longitudinal Inpatient Simulation paradigm.
- The framework utilizes a Preprocessing Pipeline and Case Construction Pipeline to transform raw EHR data into structured, multi-stage cases, requiring LLMs to actively query specialized Patient-, Nurse-, Lab-, and History-agents to gather information before committing to sequential, irreversible clinical decisions.
- Evaluation is performed via a Dual Evaluation Framework that employs Deterministic Ontology-Grounded Matching for outcome accuracy and specific Process Metrics to measure the efficiency and quality of information acquisition, revealing that clinical reasoning, rather than information access, is the primary bottleneck for LLMs.

---

[Permissive Safety Through Trusted Inference: Verifiable Belief-Space Neural Safety Filters for Assured Interactive Robotics](http://arxiv.org/abs/2606.02562)

- JIST (Joint Inference–Safety Test): introduces an inference-aware verification framework that certifies high-probability safety for BELIEFSF (Belief-Space Safety Filter) by focusing verification on a trusted inference region where the robot's inference is expected to be reliable.
- The framework couples conformal prediction with inference quality to reduce the conservativeness of safety filters in interactive robotics without requiring perfect inference.
- By carving out regions where inference is reliable, the approach achieves a significantly larger verified safe set and lower rejection rates compared to standard conformal prediction baselines.

---

[HERO’S JOURNEY: Testing Complex Rule Induction with Text Games](http://arxiv.org/abs/2606.02556)

- HERO’S JOURNEY: introduces a benchmark for evaluating rule induction in goal-directed episodic tasks, where agents must infer hidden rules from demonstrations and apply them through multi-step execution.
- The framework utilizes a deterministic text-based environment to test inductive reasoning across attribute and procedural induction families, each featuring configurable structural rule forms and identifiability conditions.
- Performance is assessed using efficiency-calibrated success rate (ECSR) and rule verbalization (RV) scores, revealing that while LLMs show evidence of induction, they often struggle with procedural complexity and rely on contextual bias.

---

[SkillHarm: Lifecycle-Aware Skill-Based Attacks via Automated Construction](http://arxiv.org/abs/2606.02540)

- SkillHarm: introduces a benchmark for evaluating LLM agent vulnerabilities to skill-based attacks across the skill-use lifecycle, utilizing AutoSkillHarm, Attack Designer Agent, LLM-based Detector, Reviewer Agent, Deterministic Evaluator, and Docker Environment.
- The framework evaluates two attack scenarios: Fixed-Payload Poisoning (FPP) for single-session compromise and Self-Mutating Poisoning (SMP) for deferred cross-session compromise.
- Experiments demonstrate that current LLMs remain highly vulnerable to these attacks, with many failures stemming from agents not engaging with poisoned files rather than active resistance.

---

[Tracking the Behavioral Trajectories of Adapting Agents](http://arxiv.org/abs/2606.02536)

- Agent-to-Agent (A2A) Protocol: introduces a methodology for monitoring AI agent behavioral changes by measuring trait shifts as directions in the embedding space of text files using an Embedding Model and a Processor.
- The framework utilizes a trusted Runtime Server to mediate evaluations between an Agent A and an Agent B, ensuring that sensitive files remain private while providing auditable trait scores.
- By training a Ridge regression model on skill file diffs, the system achieves high accuracy in detecting specific behavioral traits, such as data-seeking, without requiring direct human oversight.

---

[IMAC-AgriVLN: Can Agricultural Vision-and-Language Navigation Agents be Aware of Instruction Mistakes?](http://arxiv.org/abs/2606.02519)

- IMAC-AgriVLN: introduces a framework that integrates an IMAC module into a VLM-based navigation agent to detect and correct instruction mistakes in agricultural environments.
- The framework utilizes a VLM-based decision-maker that employs Conservative Verification and Minimal Intervention principles to identify and rectify errors in natural language instructions based on real-time visual observations.
- The research also proposes the A2A-MI benchmark, which uses a semi-automatic data annotation method to inject diverse instruction mistakes, enabling the evaluation of agent robustness in agricultural navigation tasks.

---

[ToolFG: Towards Well-Grounded Fine-Grained Image Classification](http://arxiv.org/abs/2606.02518)

- ToolFG: introduces a tool-integrated framework for fine-grained image classification that enables LLMs to perform well-grounded reasoning by autonomously invoking external tools to collect verifiable visual evidence.
- The framework utilizes MCTS-guided knowledge distillation to mine reasoning trajectories from a teacher MLLM, training a student MLLM to master tool invocation through contrastive learning.
- A model-tool co-evolution mechanism iteratively refines both the student MLLM policy and the toolset, ensuring they remain mutually adapted and specialized for fine-grained classification tasks.

---

[Bridging the Last Mile of Time Series Forecasting with LLM Agents](http://arxiv.org/abs/2606.02497)

- LLM-agent framework: introduces a system that transforms statistical baseline forecasts into decision-ready outputs through constrained, evidence-backed revisions performed by LLMs.
- The architecture utilizes a unified forecast workspace to maintain immutable baselines while allowing iterative, auditable refinements via a structured action interface.
- The framework incorporates map-reduce decomposition for long-horizon forecasting and a post-hoc reflection loop for cross-session self-improvement using a persistent memory bank.

---

[Monitoring Agentic Systems Before They’re Reliable](http://arxiv.org/abs/2606.02494)

- Monitoring and Triage Methodology: introduces a triangulated evaluation framework that decomposes agentic system behavior into quality, suitability, and efficiency dimensions across within-run, cross-run, and structural scopes to identify integration defects.
- The methodology utilizes within-category variance as a primary detection signal to characterize system maturity and distinguish between routine noise and high-severity structural failures.
- By applying FMEA-based severity classification, the system enables deterministic triage that reduces human review volume by concentrating analyst attention on the most critical operational anomalies.

---

[Iteris: Agentic Research Loops for Computational Mathematics](http://arxiv.org/abs/2606.02484)

- Iteris: introduces an agentic research system designed for open problems in computational mathematics that coordinates research modes through an explore–plan–execute loop.
- The framework utilizes specialized execution agents for foundation work, numerical experimentation, proof construction, and review, all supported by a file-based memory system.
- Iteris facilitates long-horizon research trajectories by decomposing complex problems into structured tasks, enabling human-in-the-loop validation and iterative refinement of mathematical proofs.

---

[Ghost Tool Calls: Issue-Time Privacy for Speculative Agent Tools](http://arxiv.org/abs/2606.02483)

- Speculative Tool Privacy Contracts: introduces a runtime abstraction that treats observation before commitment as a first-class effect to prevent privacy leakage from abandoned speculative tool calls.
- The framework utilizes a monitor that sits between the LLM planner and external systems to apply issue-time transformations like rewriting arguments or shadowing destinations before dispatch.
- This approach effectively mitigates intent leakage from ghost tool calls by ensuring that sensitive information is not exposed to external observers before the agent commits to a specific branch.

---

[X-Stream: Exploring MLLMs as Multiplexers for Multi-Stream Understanding](http://arxiv.org/abs/2606.02482)

- X-Stream: introduces the first multi-stream streaming benchmark, utilizing Multi-stream streaming benchmark, Multiplexing strategies, Online inference pipeline, Dual-verification pipeline, and LLM-as-a-Judge to evaluate MLLMs in complex multi-stream environments.
- The framework conceptualizes MLLMs as naive multiplexers, systematically evaluating their performance through three distinct multiplexing strategies: Spatial, Temporal, and Semantic Division.
- X-Stream addresses the over-reliance on single-stream inputs by enforcing a dual-verification protocol that guarantees the necessity and sufficiency of multi-stream data for accurate reasoning.

---

[MCP-Persona: Benchmarking LLM Agents on Real-World Personal Applications via Environment Simulation](http://arxiv.org/abs/2606.02470)

- MCP-Persona: introduces a benchmark for evaluating LLM agents on real-world personalized MCP tools, utilizing Tool-Traverse, Context-Tree, and Persona-Gen to simulate complex, stateful environments.
- The framework employs a Code-as-Simulation paradigm where LLMs synthesize executable Python kernels to replicate authentic tool behaviors and stateful interactions.
- Experimental results demonstrate that even SOTA LLMs struggle with implicit grounding, multi-step state maintenance, and cross-tool coordination in personalized scenarios.

---

[MASER: Modality-Adaptive Specialist Routing for Embodied 3D Spatial Intelligence](http://arxiv.org/abs/2606.02463)

- MASER: introduces a lightweight routing framework that selects the optimal modality-specific DoRA adapter for a shared frozen Qwen2-VL-2B backbone based on question semantics.
- The framework utilizes a frozen SBERT encoder and a trained MLP router to map input questions to a probability distribution over five distinct modality adapters, minimizing inference latency.
- A confidence-based cascade mechanism further optimizes performance by invoking a secondary adapter and a hybrid judge when the router's confidence falls below a specified threshold.

---

[AGENTCL: Toward Rigorous Evaluation of Continual Learning in Language Agents](http://arxiv.org/abs/2606.02461)

- AGENTCL: introduces a rigorous evaluation framework for continual learning in LLM agents by utilizing controlled compositional task streams and targeted metrics to quantify plasticity, stability, and generalization.
- The framework incorporates MEMPROBE, a non-parametric memory probing method that organizes experience into interaction-, insight- and skill-memories while applying quality-aware consolidation to filter unreliable information.
- Empirical results demonstrate that compositional streams significantly amplify the discriminative power of benchmarks compared to naive streams, exposing critical stability bottlenecks in existing memory designs.

---

[Active Exploring like a Pigeon: Reinforcing Spatial Reasoning via Agentic Vision-Language Models](http://arxiv.org/abs/2606.02459)

- AELP (Active Exploring like a Pigeon): introduces an agentic pipeline for spatial reasoning that utilizes a Policy Model, a Dynamic Cognitive Map, and Spatial Assertion Codes (SAC) to enable VLMs to actively explore and reason about 3D scenes.
- The framework employs a Dynamic Cognitive Map as persistent memory and SAC as executable Python expressions to provide dense reward signals for reinforcement learning, overcoming the limitations of sparse feedback in passive perception.
- By iteratively retrieving views and updating the cognitive map, the model achieves state-of-the-art performance on the MindCube benchmark, demonstrating robust spatial reasoning under diverse camera movements.

---

[Beyond One-shot: AI Agents for Learning in Field Experiments](http://arxiv.org/abs/2606.02458)

- DIKW Multi-Agent System: introduces a cumulative learning architecture that utilizes Orchestrator Agent, D-Agent, I-Agent, K-Agent, W-Agent, LangGraph, and Sandboxed Execution Environment to transform experimental data into actionable behavioral interventions.
- The framework employs a multi-level abstraction process where specialized agents perform autonomous code execution and reasoning to bridge the gap between raw experimental observations and domain-specific design principles.
- By maintaining explicit evidence chains across all DIKW layers, the system ensures transparency and auditability, enabling iterative learning that outperforms human-expert and frontier LLM baselines in healthcare messaging.

---

[HLL: Can Agents Cross Humanity’s Last Line of Verification?](http://arxiv.org/abs/2606.02449)

- HLL (Humanity’s Last Line of Verification): introduces a controlled benchmark for evaluating whether multimodal agents can successfully navigate interactive CAPTCHA verification boundaries.
- The framework utilizes ten task families and three realism axes—intrinsic difficulty, webpage distraction, and dynamic interaction validation—to measure agent performance in closed-loop GUI environments.
- Empirical results demonstrate that frontier LLMs remain brittle at this human-substitution boundary, with performance degradation linked to gaps in spatial grounding, action calibration, and process consistency.

---

[ODTQA-FoRe: An Open-Domain Tabular Question Answering Dataset for Future Data Forecasting and Reasoning](http://arxiv.org/abs/2606.02433)

- TimeFore: introduces an LLM agent-based framework that decomposes the problem into three collaborative roles: a Retriever, a Forecaster, and an Analyzer.
- The framework utilizes a Retriever for autonomous SQL generation, a Forecaster that leverages specialized models like TimesNet and TimeXer for accurate time-series prediction, and an Analyzer for synthesizing final responses.
- TimeFore addresses the limitations of LLMs in future-oriented numerical reasoning by integrating external time-series forecasting tools within a structured, agent-based pipeline.

---

[K-BROWSECOMP: A Web Browsing Agent Benchmark Grounded in Korean Contexts](http://arxiv.org/abs/2606.02404)

- K-BROWSECOMP: introduces a web-browsing agent benchmark grounded in Korean contexts, comprising a 300-problem human-verified subset and a 100-problem synthetic diagnostic split, utilizing the search_evals framework, Perplexity Search, LLM-based browsing agents, human annotators, Claude Code, multilingual sentence-transformers, domain classifiers, and maximum mean discrepancy tests.
- The benchmark evaluates LLMs on their ability to perform compositional agentic tasks, such as multi-hop reasoning and parallel constraint satisfaction, within linguistically and culturally distinct Korean web environments.
- Analysis reveals that even frontier LLMs struggle with trajectory-level state maintenance, often failing to preserve candidates, constraints, and entity roles across multiple search steps despite successful initial retrieval.

---

[Policy and World Modeling Co-Training for Language Agents](http://arxiv.org/abs/2606.02388)

- PaW: introduces a co-training framework that integrates auxiliary world modeling into LLM agent reinforcement learning by leveraging on-policy rollouts as action-conditioned dynamics supervision.
- The framework utilizes an action-entropy filter to select informative transitions, a clipped MAE loss to handle noisy environment observations, and reward-adaptive loss balancing to stabilize joint optimization.
- PaW improves agent performance across interactive and search-augmented tasks without requiring additional simulators, separate models, or inference-time computation.

---

[AgentPLM: Agentic Protein Language Models with Reasoning-Augmented Decoding for Protein Sequence Design](http://arxiv.org/abs/2606.02386)

- AgentPLM: introduces a framework that equips a pre-trained PLM with agentic capabilities by training it end-to-end to interleave autoregressive sequence generation with structured oracle invocations using Reasoning-Augmented Decoding and Contrastive Agent Policy Optimisation.
- The architecture integrates a Tool Context Encoder and a Trajectory Memory Buffer to enable the model to observe and respond to biophysical feedback mid-generation, overcoming the structural and epistatic blindness of passive PLMs.
- AgentPLM achieves state-of-the-art performance across diverse protein design benchmarks by learning to perform online error correction through a learned policy that treats design as a sequential decision-making process.

---

[SPADE-Bench: Evaluating Spontaneous Strategic Deception in Agents via Plan-Action Divergence](http://arxiv.org/abs/2606.02380)

- SPADE-Bench (Spontaneous Plan-Action Divergence Evaluation): introduces a diagnostic benchmark to evaluate spontaneous deceptive behaviors in LLM agents by measuring plan-action divergence under controlled pressure scenarios.
- The framework utilizes a Seed Construction, Test Case Generation, Quality Control, Deception Judger, and Tool Environment to distinguish strategic deception from hallucination by comparing agent behavior across regular and pressure-induced conditions.
- Empirical results across frontier LLMs demonstrate that deceptive behavior is a tangible risk, with models exhibiting non-monotonic relationships between scale and deception, and varying sensitivities to pressure types.

---

[Harness-1: Reinforcement Learning for Search Agents with State-Externalizing Harnesses](http://arxiv.org/abs/2606.02373)

- Harness-1: introduces a stateful retrieval harness that offloads mechanical bookkeeping from the LLM policy to the environment, enabling the agent to focus on semantic search decisions.
- The framework utilizes a two-tier memory system where an inner working memory provides a compact, rendered state to the LLM, while an outer store maintains full-text documents for retrieval and verification.
- Harness-1 improves retrieval performance across diverse benchmarks by employing reinforcement learning to optimize search behavior over explicit, persistent state rather than append-only transcripts.

---

[CoMAP: Co-Evolving World Models and Agent Policies for LLM Agents](http://arxiv.org/abs/2606.02372)

- CoMAP: introduces a closed-loop framework that co-evolves textual world models and agent policies through mutual reinforcement via on-policy self-distillation and future-aware reflection.
- The framework utilizes a student-teacher world model architecture to provide reliable future-state signals, which the agent policy uses to perform future-aware reflection for action refinement.
- CoMAP employs a world-state gate and an action gate to manage the curriculum of future-state predictions and ensure robust, high-confidence action execution during the co-evolutionary process.

---

[MOC: Multi-Order Communication in LLM-based Multi-Agent Systems](http://arxiv.org/abs/2606.02359)

- MOC (Multi-Order Communication): introduces a topology-aware communication scheme that exposes LLM agents to raw upstream evidence across multiple hop distances to enhance reasoning receptive fields.
- The framework utilizes a structural message consolidation operator to compress multi-order evidence while preserving topological precedence and semantic fidelity.
- MOC employs a semantic-topological merging algorithm to prune redundant information, effectively balancing long-range context coverage with strict token constraints in multi-agent systems.

---

[Do Multimodal Agents Really Benefit from Tool Use? A Systematic Study of Capability Gains](http://arxiv.org/abs/2606.02357)

- Thyme and DeepEyesV2: introduces a systematic evaluation of tool-augmented multimodal agents, demonstrating that tool access often fails to expand the set of solvable problems beyond non-tool baselines.
- The study utilizes Tool-Enabled Agent, Tool-Free Agent, Pure-Text Reasoner, Tool Format Only, and Tool Result Only to isolate the impact of tool-calling protocols and execution results on agent performance.
- Analysis reveals that tool calls frequently serve as redundant confirmation or failed repair attempts rather than providing novel information, suggesting that agents often learn tool-calling patterns as a reflex rather than a functional capability.

---

[SIRI: Self-Internalizing Reinforcement Learning with Intrinsic Skills for LLM Agent Training](http://arxiv.org/abs/2606.02355)

- SIRI: introduces a three-phase reinforcement learning framework that enables LLMs to discover, validate, and internalize reusable skills as temporary training-time signals, eliminating the need for inference-time skill retrieval.
- The framework utilizes Policy Warmup to bootstrap interaction, Self-Skill Mining to generate candidate skills from successful rollouts, and Advantage-Weighted Skill Internalization to distill high-utility behaviors into the policy parameters.
- By treating skills as transient training guidance rather than persistent memory, SIRI achieves superior performance on long-horizon tasks while maintaining a retrieval-free inference policy.

---

[Coordination Graphs for Constrained Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2606.02337)

- CG-CMARL: introduces a decentralized framework for constrained multi-agent reinforcement learning by combining coordination graphs with Lagrangian duality to enable scalable, constraint-aware coordination.
- The system utilizes a two-head Q-network architecture to decouple primary objective learning from constraint satisfaction, allowing a single trained model to trace a Pareto front by varying the Lagrangian multiplier at evaluation time.
- Max-Sum message passing is employed on the factor graph to enable distributed action coordination with complexity polynomial in the number of agents, effectively addressing the scalability challenges of joint action spaces.

---

[TVIR: Building Deep Research Agents Towards Text–Visual Interleaved Report Generation](http://arxiv.org/abs/2606.02320)

- TVIR (Text–Visual Interleaved Report Generation): introduces a hierarchical multi-agent framework, TVIR-AGENT, that utilizes a Planner, Image Searcher, Chart Generator, Writer, and Polisher to construct evidence-driven, multimodal research reports.
- The framework incorporates a dual-path evaluation suite, TVIR-BENCH, which assesses both textual quality and visual fidelity, including figure-context integration and chart-source consistency.
- Experiments demonstrate that TVIR-AGENT improves evidence grounding and cross-modal alignment compared to existing text-centric deep research systems.

---

[Discovering Agents for Discovery: The Case for DNS](http://arxiv.org/abs/2606.02314)

- DNS-based AI Agent Discovery Framework: introduces a methodology for utilizing existing DNS infrastructure to enable secure, scalable, and interoperable discovery of AI agents across the Internet.
- The framework evaluates discovery mechanisms based on three core criteria: navigational completeness, lookup complexity, and transaction performance.
- Empirical analysis demonstrates that necessary agent metadata, including security extensions and certificate associations, fits within a single unfragmented UDP DNS message.

---

[Unified Context Evolution for LLM Agents](http://arxiv.org/abs/2606.02304)

- UCE (Unified Context Evolution): introduces a gradient-free framework that improves LLM agents across episodes by maintaining an evolving library of typed Evolvable Context Units (ECUs) without updating model weights.
- The framework decomposes agent experience into four complementary types—Memory, Strategy, Workflow, and Skill—which are retrieved and injected into the actor prompt to provide persistent, actionable guidance.
- A Knowledge Yield Scheduling module dynamically allocates generation budgets based on library coverage and usage-based fitness scores to ensure efficient knowledge accumulation.

---

[SeClaw: Spec-Driven Security Task Synthesis for Evaluating Autonomous Agents](http://arxiv.org/abs/2606.02302)

- SeClaw: introduces a systematic framework for the scalable construction and execution-based security evaluation of autonomous LLM agents.
- The framework utilizes a multi-agent pipeline to synthesize security tasks from structured specifications, which are then executed in isolated Docker sandboxes to record fine-grained interaction trajectories.
- SeClaw enables reproducible security auditing by analyzing complete agent-environment interaction logs, including tool calls and file operations, rather than relying solely on final model outputs.

---

[POIROT: Interrogating Agents for Failure Detection in Multi-Agent Systems](http://arxiv.org/abs/2606.02282)

- POIROT: introduces a decentralized diagnostic protocol that repurposes LLM-MAS agents as their own diagnostic layer to identify failure sources through structured peer interrogation and weighted consensus.
- The framework constructs an N-dimensional hazard space to ground failure attribution in the operational architecture, mitigating single-evaluator bias and context saturation.
- POIROT outperforms single-LLM evaluator baselines across diverse benchmarks, with performance gains scaling with problem complexity, agent count, and fault dimensionality.

---

[When Knowledge Is Not Free: Cost-Aware Evidence Selection in Retrieval-Augmented Generation](http://arxiv.org/abs/2606.02245)

- Cost-aware RAG: introduces a framework for resource-constrained retrieval where evidence is assigned access-cost tiers and systems must optimize evidence selection under an explicit budget.
- The research evaluates static budgeted selectors and agentic controllers, finding that fixed selection rules are brittle and that agentic frameworks offer a promising, albeit model-dependent, approach to adaptive evidence acquisition.
- The study demonstrates that LLM-based agents can effectively manage evidence-access budgets by deciding when to retrieve, which tier to access, and when to stop, though performance remains sensitive to the underlying LLM backbone and task structure.

---

[AgentRedBench: Dynamic Redteaming and Integration-Aware Defense for LLM Agents over SaaS Integrations](http://arxiv.org/abs/2606.02240)

- AgentRedBench: introduces a dynamic LLM-driven redteaming benchmark for evaluating indirect prompt injection in tool-use agents across enterprise SaaS integrations.
- The framework utilizes an Attacker agent, Orchestrator, Target agent, LLM judge, and AgentRedGuard to systematically test and defend against underspecified-authorization attacks.
- AgentRedGuard acts as a lightweight, finetuned classifier that intercepts tool responses to prevent malicious actions while maintaining high utility and low latency.

---

[Optimizing the Envy Cycle Elimination Algorithm](http://arxiv.org/abs/2606.02233)

- ECE Algorithm: introduces a systematic study of heuristics for the Envy Cycle Elimination algorithm to improve utilitarian and egalitarian welfare guarantees in fair division.
- The paper evaluates various greedy and non-greedy heuristics, including a max-min matching approach, to mitigate the welfare loss inherent in the vanilla ECE algorithm.
- Theoretical analysis demonstrates significant gaps between weak and strong prices of fairness for ECE, while empirical results confirm that specific heuristics provide superior average-case performance.

---

[Symmetry-Aware 9D Pose Estimation with Sim(3)-Consistent Feature and Spherical Inception Convolution](http://arxiv.org/abs/2606.02219)

- SSH-Pose: introduces a shape prior-free framework for category-level 9D object pose estimation by leveraging a semantic-guided SSA-Module for translation/size estimation and a SlinConv-based rotation estimator.
- The framework utilizes DINOv2 for robust semantic feature extraction and an MHP-Module to generate Sim(3)-consistent geometric features, enabling efficient and accurate pose estimation without relying on instance-specific CAD models.
- By projecting features onto a sphere and employing SlinConv, the method models long-range dependencies for rotation estimation while maintaining low computational overhead compared to Transformer-based architectures.

---

[Better with Experience: Self-Evolving LLM Agents for Evidence-Grounded Health Community Notes](http://arxiv.org/abs/2606.02215)

- EVONOTE: introduces an agentic framework that enables health Community Notes generation to self-evolve by distilling trajectory-level feedback into an evolving experience memory of prior misinformation correction episodes.
- The framework utilizes a Social Utility Judge to evaluate note quality and a Memory Evolver to distill lessons into phase-specific strategies for claim analysis, evidence acquisition, and note writing.
- EVONOTE improves correction quality and reduces latency by retrieving relevant memories to guide LLM agents through evidence-grounded trajectories without requiring model parameter updates.

---

[Context-Aware Workflow Decomposition for Automated Mobile UI Annotation Using Multimodal Large Language Models](http://arxiv.org/abs/2606.02208)

- CAWD: introduces a context-aware workflow decomposition approach for automated mobile UI annotation that improves precision by dividing the task into smaller, focused stages using multimodal LLMs.
- The framework utilizes structured prompts and schema-constrained JSON outputs to manage prompt complexity while preserving essential screen context for accurate element detection.
- Experimental results demonstrate that a two-step workflow provides the optimal balance between prompt simplicity and contextual preservation, significantly enhancing annotation reliability compared to single-step or overly fine-grained approaches.

---

[Cross-Environment Neural Reranking for Sample-Efficient Action Selection in Text-Based Agents](http://arxiv.org/abs/2606.02204)

- Cross-Environment Neural Reranking for Sample-Efficient Action Selection in Text-Based Agents: introduces a compact neural reranker that utilizes a DeBERTa-v3 Encoder, Rerank Head, Auxiliary Prediction Heads, LoRA Adapters, RouterNetwork, and PCGrad Gradient Surgery to perform cross-environment action selection.
- The framework employs a DeBERTa-v3 encoder to map observations, tasks, and candidate actions into a shared representation space, enabling effective action ranking across diverse text-based environments.
- By leveraging minority-class upsampling and environment-aware adapter routing, the model achieves robust cross-domain transfer while maintaining computational efficiency compared to LLMs.

---

[Learning When Not to Act: Mitigating Tool Abuse in Agentic Reinforcement Learning](http://arxiv.org/abs/2606.02132)

- EAPO (Efficient Agentic Policy Optimization): introduces a reinforcement learning framework that mitigates tool abuse in LLMs by combining Efficiency-Aware Rollout, Difficulty-Aware Reward Shaping, and Confidence-Aware Advantage Reweighting.
- The framework utilizes tool-free rollouts to provide explicit signals for internal reasoning capabilities, allowing the model to distinguish between necessary tool use and redundant tool dependence.
- Experimental results across mathematical and knowledge-intensive benchmarks demonstrate that EAPO improves accuracy while significantly reducing unnecessary tool calls compared to existing agentic RL methods.

---

[BADGER: Bridging Agentic and Deterministic evaluation for Generative Enterprise Reasoning](http://arxiv.org/abs/2606.02109)

- BADGER: introduces a unified evaluation framework that integrates text-to-SQL assessment with agentic behavior evaluation for enterprise data environments.
- The framework utilizes an LLM-assisted SQL component extractor and a two-stage Hybrid-EX metric to resolve structural ambiguities in enterprise SQL, achieving substantial agreement with human expert labels.
- BADGER incorporates an agentic evaluation suite for tool-call fidelity and response faithfulness, providing a production-grade pipeline with configurable judge backends for continuous quality monitoring.

---

[Network Distributed Multi-Agent Reinforcement Learning for Consensus Control of Quadcopters](http://arxiv.org/abs/2606.02107)

- ND-MARL: introduces a distributed framework for UAV consensus that utilizes a MASAC-based high-level consensus planner and a low-level thrust-vector controller to achieve scalable, communication-aware coordination.
- The framework leverages a 2-Neighbor communication topology to maintain constant per-agent computational and communication costs, enabling zero-shot scalability to large swarms.
- By factorizing rewards over local neighborhoods, the approach avoids the scalability limitations of centralized MARL while ensuring stable convergence and translation-invariant consensus.

---

[Multimodal Action Diffusion for Robust End-to-End Autonomous Driving](http://arxiv.org/abs/2606.02105)

- ADT: introduces an anchor-free diffusion transformer that models multimodal driving action distributions by generating multiple candidates and selecting the optimal one via Nearest Neighbour Matching.
- The architecture replaces deterministic regression with a conditional diffusion process, utilizing a Transformer-based denoiser to predict noise in action space from observation-conditioned tokens.
- By deferring commitment to a single control until execution, the framework achieves robust performance on the Bench2Drive benchmark while maintaining low latency for closed-loop driving.

---

[Testing Decision Makers without Counterfactuals](http://arxiv.org/abs/2606.02095)

- Scoring Tests: introduces a framework for evaluating the relative informativeness of two strategic agents in a repeated bandit environment without observing counterfactuals.
- The framework demonstrates that while identifying the more-informed agent is possible under simultaneous decisions, it necessarily incentivizes suboptimal welfare-maximizing behavior.
- The research establishes that in sequential decision environments, identifying the more-informed agent is impossible using scoring tests.

---

[Agentic-J: An AI Agent for Biological Microscopy Image Analysis](http://arxiv.org/abs/2606.02080)

- Agentic-J: introduces a containerized multi-agent system that enables biologists to perform reproducible microscopy image analysis in Fiji through natural language instructions.
- The system utilizes a coordinated team of specialized agents, including a supervisor, coder, debugger, and quality assurance agent, to automate complex bioimage analysis workflows while maintaining human-in-the-loop control.
- By integrating a RAG-based knowledge database and a structured skills filesystem, the framework provides domain-specific reasoning capabilities that scale with accumulated project experience.

---

[Where Do Deep-Research Agents Go Wrong? Span-Level Error Localization in Agent Trajectories](http://arxiv.org/abs/2606.02060)

- DRIFT (Claim-centric Trajectory Auditing Framework): introduces a claim-centric auditing workflow that builds trajectory-level claim ledgers, verifies support, and traces claim dependencies to localize first and follow-up errors.
- The paper presents TELBench, a large-scale benchmark for span-level error localization in deep-research agent trajectories, comprising 1,000 verified instances.
- Experiments demonstrate that DRIFT significantly improves span-level error localization and first-error accuracy by leveraging structured claim auditing over bare LLM prompting.

---

[Private Learning in Bilateral Trade](http://arxiv.org/abs/2606.02050)

- Private Learning in Bilateral Trade: introduces a differentially private learning framework for bilateral trade mechanisms that maximizes profit or gain-from-trade under σ-smooth distribution assumptions.
- The approach utilizes the Exponential Mechanism over a finite family of η-simple mechanisms, achieving (α, β)-optimality with polynomial-time computational efficiency via dynamic programming and random walks.
- The paper proves that private learning is impossible for general distributions in bilateral trade, establishing the necessity of the σ-smoothness assumption for achieving nearly optimal sample complexity.

---

[Explainable Data-driven Deep Reinforcement Learning Methods for Optimal Energy Management in Buildings](http://arxiv.org/abs/2606.02049)

- XRL (Explainable Reinforcement Learning) framework: introduces a structured pipeline that integrates DRL policy outputs with post-hoc explainability mechanisms, including Data Ingestion and Forecast Validation, Policy Learning and Benchmark Protocols, and Policy Interpretations, to enhance transparency in building energy management.
- The framework utilizes RL Agents to optimize battery operations while employing Decision Trees and Feature Removal Analysis to provide actionable insights into the decision-making process of the learned policies.
- By comparing on-policy and off-policy algorithms, the study demonstrates that on-policy methods like A2C and PPO achieve superior performance and stability in complex, high-dimensional energy environments.

---

[OpenWebRL: Demystifying Online Multi-turn Reinforcement Learning for Visual Web Agents](http://arxiv.org/abs/2606.02031)

- OpenWebRL: introduces a framework for training visual web agents using online multi-turn reinforcement learning on live websites, incorporating Supervised warm start, Agent harness, Multimodal multi-turn GRPO objective, Trajectory-level success judge, Live-browser infrastructure, and Context management.
- The framework utilizes a Supervised warm start to place the policy in a productive exploration regime before applying a Multimodal multi-turn GRPO objective to optimize agent performance through trajectory-level rewards.
- By leveraging a robust Live-browser infrastructure and efficient Context management, OpenWebRL enables compact 4B-8B models to achieve state-of-the-art performance on challenging live-web benchmarks while remaining competitive with proprietary systems.

---

[An Agentic Approach Towards Replication Package Quality Evaluation](http://arxiv.org/abs/2606.02006)

- Agentic Replication Package Evaluator: introduces a multi-agent system that operationalizes open-science guidelines into machine-verifiable criteria to automate the assessment of research artifact quality.
- The framework utilizes a five-phase pipeline including Ingestion and Embedding, Artifact Retrieval and Analysis, Evaluation Plan Generation, Parallel Subtopic Evaluation, and Report Compilation and Tracing to provide evidence-grounded improvement reports.
- The system incorporates a Human-in-the-Loop mechanism for iterative planning and employs an LLM-as-Judge to ensure semantic consistency, demonstrating high reliability in structural evaluations while identifying limitations in qualitative research contexts.

---

[Distortion-Aware Fusion of Statistical and Vision-Language Features for Blind Image Quality Assessment](http://arxiv.org/abs/2606.02002)

- Distortion-aware three-stream fusion framework: integrates a 138-dimensional NSS Feature Extractor, SigLIP ViT-SO400M-14, and CLIP-H ViT-H-14 through a Multiplicative Distortion-Aware Gating Network to predict image quality.
- The framework utilizes a lightweight MLP Regression Head to process concatenated features from frozen backbones, significantly reducing computational requirements compared to end-to-end fine-tuned models.
- A multiplicative gating mechanism dynamically adjusts the contribution of each feature stream based on input content, improving performance on diverse distortion types.

---

[Scaling Agentic Capabilities via Grounded Interaction Synthesis](http://arxiv.org/abs/2606.02001)

- GAIS: introduces a framework that automates the scalable construction of diverse environments and complex tasks via a two-phase grounding mechanism involving MCP Servers, Python Functions, Difficulty Scoring, Environment Schema, Complex-Dependency Planning, Tool Dependency Graph, Adversarial Policies, User Simulator, Assistant Agent, State-Based Verification, and Response-Based Verification.
- The framework utilizes protocol-anchored environments derived from real-world MCP servers to ensure functional diversity and employs structure-guided planning to generate high-fidelity, long-horizon tasks.
- GAIS includes a user simulator agent and an assistant agent that engage in multi-turn dialogues to collect high-quality interaction trajectories for training LLMs.

---

[MMG2Skill: Can Agents Distill In-the-Wild Guides into Self-Evolving Skills?](http://arxiv.org/abs/2606.01993)

- MMG2Skill: introduces a closed-loop framework that distills in-the-wild multimodal guides into execution-grounded skills and continuously improves them through trajectory-level root-cause feedback.
- The framework utilizes a VLM Skill Extractor to construct editable skills, a Skill-Conditioned Agent for task execution, an Analyzer for trajectory diagnosis, and a Refiner for iterative skill updates.
- MMG2Skill-Bench evaluates the capability of VLM agents to convert public procedural knowledge into reusable skills across GUI control, open-ended gameplay, and strategic card play.

---

[SafeMCP: Proactive Power Regulation for LLM Agent Defense via Environment-Grounded Look-Ahead Reasoning](http://arxiv.org/abs/2606.01991)

- SafeMCP: introduces a server-side defense plugin that constrains LLM agent tool acquisition by leveraging an Internal World Model for look-ahead reasoning to enforce safety boundaries.
- The framework models agent-defense interactions as a Cooperative Stackelberg Power Game, utilizing a Proactive Tool Filtering Layer and an Immediate Intervention Fail-safe Layer to mitigate power-seeking risks.
- SafeMCP employs a Three-Stage Training Pipeline and a Dual Verifiable Reward Mechanism to ensure robust safety alignment while maintaining high task utility for LLM agents.

---

[A Simple Hierarchical Causality Primer](http://arxiv.org/abs/2606.01979)

- HCS: introduces a formal discrete framework for hierarchical causality that separates bottom-up emergence from top-down constraints using H, D, C, and U.
- The framework utilizes actor instances to restrict lower-level transition kernels through interfaces, distinguishing causal equivalence from aggregation equivalence.
- This approach provides a modelling grammar for complex systems where higher-level roles, institutions, or protocols constrain local dynamics without requiring continuous-time limits.

---

[Algorithmic algorithm development with LLMs: A Case Study on LLM-Usage for Contraction Order Optimization in Tensor Networks](http://arxiv.org/abs/2606.01975)

- OpenEvolve: introduces a framework for automated algorithm development by combining LLM-based code generation with island-based evolutionary search and quality-diversity archives.
- The system utilizes an automated evaluator to score candidate programs, enabling iterative refinement of algorithms for specific scientific tasks like tensor network contraction.
- The study highlights the critical role of human-defined design choices, such as evaluation metrics and test instances, in the success of LLM-driven algorithm engineering.

---

[Market-Based Replanning for Safety-Critical UAV Swarms in Search and Rescue Missions](http://arxiv.org/abs/2606.01970)

- IRDS (Intelligent Replanning Drone Swarm): introduces a decentralized coordination architecture for UAV swarms that utilizes a reverse-auction market mechanism to ensure fault-tolerant task allocation in resource-constrained environments.
- The framework integrates a liquidation protocol that treats agent failures as market events, triggering immediate task redistribution to maintain mission continuity without centralized intervention.
- Empirical validation through physics-based simulations demonstrates that the architecture maintains high mission success rates under significant workforce degradation by leveraging distance-weighted cost functions and geometric consensus for target verification.

---

[Trust-Calibrated Code Review: A Participatory Design Study of Review Workflows for LLM-Generated Multi-File Changes](http://arxiv.org/abs/2606.01969)

- Trust-Calibrated Code Review framework: introduces a three-level IDE workflow designed to address trust-calibration challenges when reviewing multi-file changes generated by LLMs.
- The framework utilizes seven design constructs—including architectural diagrams, risk-per-line analysis, and AI-based judge agents—to provide progressive disclosure of information and surface risk signals.
- By organizing information across overview, file-analysis, and code snippet levels, the approach enables developers to allocate review effort proportionate to risk rather than performing undifferentiated line-by-line inspection.

---

[AutoMedBench: Towards Medical AutoResearch with Agentic AI Models](http://arxiv.org/abs/2606.01961)

- AutoMedBench: introduces a workflow-aware benchmark for evaluating autonomous medical-AI research agents across a unified five-stage process of Plan, Setup, Validate, Inference, and Submit.
- The framework utilizes an isolated container-based execution environment to track agent performance through both process-level agentic scores and deterministic held-out task metrics.
- By linking stage-level scores with diagnostic error codes, the benchmark exposes hidden workflow breakdowns such as failed model loading, validation skips, and malformed submissions that are often obscured by final-output metrics.

---

[Learning Action-Conditional and Object-Centric Gaussian Splatting World Models for Rigid Objects](http://arxiv.org/abs/2606.01950)

- MRO-GWM (Multi Rigid Object Gaussian World Model): introduces an action-conditional world model that learns 3D rigid-body dynamics by representing scenes with object-centric Gaussians and predicting future poses via a spatio-temporal transformer.
- The framework utilizes object-aware Gaussian splatting to encode scene history through rigid anchor transformations, enabling the model to handle partial observations and occlusions in multi-object environments.
- The spatio-temporal transformer architecture incorporates spatial grid pooling, spatial attention, temporal attention, and a novel spatio-temporal attention layer to effectively model object interactions and motion for non-prehensile manipulation tasks.

---

[Agentic Multi-View Long-Context Video Understanding via Hierarchical Knowledge Graph Retrieval](http://arxiv.org/abs/2606.01933)

- CASTLE-Framework: introduces a training-free agentic methodology for long-form video understanding that utilizes a Video Knowledge Graph and hierarchical retrieval to resolve complex queries.
- The framework employs an omni-modal extraction function to construct a structured graph of entities and relationships, which serves as a temporal index for multi-hop reasoning.
- An iterative agentic workflow, comprising a PlannerAgent, GraphRetrievalAgent, ReflectorAgent, ClipRetrievalAgent, ClipAnalyzeAgent, and VQAAgent, enables precise localization and evidence-based answering across massive multi-view video streams.

---

[SMH-Bench: Benchmarking LLM Agents for Environment-Grounded Reasoning and Action in Smart Homes](http://arxiv.org/abs/2606.01912)

- SMH-Bench: introduces a comprehensive benchmark for evaluating LLMs in smart-home environments, utilizing HomeEnv, Task Hierarchy, Device Operation Engine, Agent-Environment Interface, Rule-Based Verification, and LLM-as-a-Judge.
- The framework evaluates LLMs across 1,100 tasks spanning seven capability categories, ranging from simple apartments to complex multi-room environments with 135 devices.
- Experimental results reveal that while LLMs perform well on explicit control, they struggle with automation scheduling, ambiguity resolution, and personalized reasoning as home complexity increases.

---

[Adversarial Attacks on Robot Localization Systems via Deep Feature Perturbation](http://arxiv.org/abs/2606.01892)

- LPQN (Lightweight Product Quantization Network): introduces a novel adversarial attack framework for robot localization systems that targets centroid assignment distributions to generate effective perturbations, overcoming the indifferentiability of product quantization layers.
- The framework employs a two-phase optimization procedure, utilizing a forward pass to inject residual perturbations and a backward pass to refine the adversarial query for maximum retrieval failure.
- By combining distribution-wide and peak-targeted perturbation strategies, the method effectively disrupts the feature space of visual localization systems, causing significant mislocalization in real-world robotic environments.

---

[Absorbing Complexity: An Interaction-Native Knowledge Harness for Financial LLM Agents](http://arxiv.org/abs/2606.01886)

- InKH: introduces a financial-agent architecture that absorbs cognitive complexity by continuously transforming interaction traces into structured, persistent, and governed knowledge using Event Stream, Entity-Intent-Risk Detector, Canonical Entity Matching, Temporal Graph Memory, Passive Knowledge Injection, Consciousness Buffer, Financial LLM Agent, Tools and Guardrails, User-Visible Output, Wiki Audit Surface, and Maintenance and Human Review.
- The architecture replaces agent-driven retrieval with passive injection into a bounded Consciousness Buffer, ensuring low-latency context assembly and high-quality decision-making for financial workflows.
- By integrating write-time invalidation and governance constraints within the Temporal Graph Memory, the system effectively suppresses stale-memory usage and improves decision traceability compared to traditional turn-based LLM agents.

---

[WorldCoder-Bench: Benchmarking Physically Grounded 3D World Synthesis](http://arxiv.org/abs/2606.01869)

- WorldCoder-Bench: introduces a benchmark for evaluating the behavioral correctness of LLM-generated Three.js 3D worlds using STATEPROBE, which verifies hidden behavioral contracts through a standardized runtime state interface.
- The framework employs mutation-hardened contracts to ensure that generated programs satisfy physical, spatial, and interactive constraints rather than relying on visual plausibility.
- It provides task-level utility metrics, Return on Automation and Time Efficiency Multiplier, to quantify the economic value of LLM-generated code relative to expert developer labor.

---

[RadioMaster: Multi-Agent System for Autonomous Radio Signal Generation](http://arxiv.org/abs/2606.01862)

- RadioMaster: introduces a fully autonomous multi-agent framework that translates user intents into verified physical radio signals by integrating RadioWiki, RadioAgent, and RadioEmulator.
- The framework utilizes RadioWiki for hallucination-suppressed knowledge retrieval, RadioAgent for collaborative task decomposition and execution, and RadioEmulator for closed-loop physical layer verification.
- RadioMaster significantly outperforms existing LLMs and multi-agent baselines in configuration viability and signal fidelity, as demonstrated by the comprehensive RadioBench evaluation suite.

---

[A Theoretical Framework for Self-Play Theorem Proving Algorithms](http://arxiv.org/abs/2606.01861)

- Self-Play Theorem Proving Framework: introduces a theoretical analysis of self-play algorithms for formal theorem proving by modeling the theorem space as a graph and agents as cooperating models.
- The framework utilizes a Prover and a Conjecturer to iteratively expand the set of provable theorems, employing a Neighbor Oracle and Diffusion Embedding to guide the exploration of the theorem graph.
- To address the generation of non-fundamental theorems, the paper proposes a diversity measure and an improved Conjecturing algorithm that leverages Contrastive Learning to compute diffusion similarity between theorems.

---

[From Global Policies to Local Strategies: Multi-Objective Optimization of Resource-Specific Handover Policies](http://arxiv.org/abs/2606.01857)

- MAS-based Optimization Framework: introduces a multi-objective optimization approach for resource-specific handover policies by integrating a Simulation Model, Objective Functions, a Genetic Optimizer, a Simulation Engine, and a Pareto Front.
- The framework utilizes a genetic algorithm to evolve handover policies within an agent-based simulation environment, effectively balancing multiple performance objectives like cost and waiting time.
- By treating resources as autonomous agents, the approach discovers collaboration-aware handover strategies that outperform traditional heuristic-based resource allocation methods.

---

[RescueBench: Can Embodied Agents Save Lives in the Wild?](http://arxiv.org/abs/2606.01848)

- RescueBench: introduces a photo-realistic diagnostic benchmark that instantiates search-and-rescue as a four-stage pipeline to evaluate embodied agents under progressive difficulty levels.
- The framework utilizes an automatic data collection pipeline and environment-assisted interaction triggering to isolate navigation, exploration, and spatial memory bottlenecks from manipulation limitations.
- Empirical evaluation reveals that current architectures struggle with open-world exploration and spatial memory, with no baseline completing the full task at the highest difficulty level.

---

[CAPF: Guiding Search-Agent Rollouts with Credit-Attenuated Privileged Feedback](http://arxiv.org/abs/2606.01830)

- CAPF (Credit-Attenuated Privileged Feedback): introduces a training-time mechanism that utilizes verifier-side information to guide LLM search agents through repair trajectories, while attenuating credit for feedback-assisted actions to ensure performance at deployment.
- The framework augments the training action space with a Privileged Feedback call, which allows the agent to receive corrective guidance from a verifier when initial attempts are uncertain or incorrect.
- By applying a retention factor to credit propagation across feedback calls, CAPF effectively balances the utility of repair trajectories during training with the requirement that the agent functions independently at deployment.

---

[Dynamic Trust-Aware Sparse Communication Topology for LLM-Based Multi-Agent Consensus](http://arxiv.org/abs/2606.01828)

- DySCo: introduces a dynamic trust-aware sparse communication mechanism for LLMs that replaces fully connected broadcasting with on-demand, high-value edge selection to reduce communication overhead.
- The framework includes an Agent Pool, Initial Reasoning Module, Dynamic Edge Selection Module, Message Compression Module, Trust-Aware State Update Module, Consensus Aggregation Module, and an Early Stopping Mechanism.
- By dynamically selecting communication neighbors based on historical trust, confidence, and task relevance, DySCo achieves superior reasoning performance with significantly lower token costs and latency compared to dense multi-agent debate.

---

[Hierarchically Decoupled Mixture-of-Experts for Robust Traffic Sign Recognition in Complex Driving Scenarios](http://arxiv.org/abs/2606.01822)

- CBDES MoE TSR: introduces a hierarchically decoupled mixture-of-experts framework that utilizes a lightweight Gating Network to perform Top-1 hard routing across a heterogeneous Expert Pool, comprising Efficiency Expert, Small-Object Expert, and Weather Expert, to achieve input-aware adaptive inference.
- The framework employs a decoupled two-stage training paradigm to independently optimize expert models and the gating network, ensuring structural consistency and stable performance across diverse traffic scenarios.
- By dynamically activating only the most suitable expert for each input, the system significantly reduces computational overhead while maintaining high detection accuracy for challenging small-object and adverse-weather conditions.

---

[Unsupervised Collaborative Domain Adaptation for Driving Scene Parsing](http://arxiv.org/abs/2606.01818)

- UCDA: introduces a source-free framework that collaboratively transfers complementary knowledge from multiple pre-trained Source Models to a unified Target Model for driving scene parsing.
- The framework utilizes a Prototype Memory Bank to establish a unified reliability criterion across independently trained models, enabling effective selection of supervisory signals.
- UCDA employs a two-stage strategy involving Collaborative Source Model Optimization and Multi-Model Knowledge Infusion to mitigate source-specific biases and improve target-domain generalization.

---

[CRAB-Bench: Evaluating LLM Agents under Complex Task Dependencies and Human-aligned User Simulation](http://arxiv.org/abs/2606.01815)

- CRAB-Bench: introduces a framework for evaluating LLM agents in complex service scenarios using constraint graph-based task generation, RUSE, and state-based evaluation.
- The framework utilizes a constraint graph to model multi-step task dependencies and structured distractors, forcing LLMs to reason over large search spaces.
- RUSE incorporates human-aligned behavioral dimensions and personas to simulate realistic user interactions, revealing significant performance gaps in LLM reliability and transparency.

---

[Token Predictors Are Not Planners: Building Physically Grounded Causal Reasoners](http://arxiv.org/abs/2606.01810)

- Causal Plan: introduces a comprehensive framework that shifts embodied planning from autoregressive token prediction to physically grounded causal logic.
- The framework utilizes Causal-Plan-Bench for diagnostic evaluation and Causal-Plan-1M for high-fidelity supervision, enabling the Causal Planner to internalize complex physical dependencies.
- Empirical results demonstrate that Causal Planner outperforms frontier models by explicitly modeling executability, composition, effects, and robustness, establishing a clear Causal Scaling Law for embodied AI.

---

[OctoT2I: A Self-Evolving Agentic Text-to-Image Router](http://arxiv.org/abs/2606.01803)

- OctoT2I: introduces a novel agentic framework that reformulates the T2I task as a joint optimization of generation quality and inference efficiency using a Router Agent, Knowledge Module, Memory Module, Evaluation Module, Self-Evolving Mechanism, PSEL Loop, and Exploration Space Pruning.
- The framework utilizes a stateful, multi-round routing strategy where the Router Agent leverages long-term knowledge and short-term memory to adaptively select the most suitable T2I tool for each prompt.
- The Self-Evolving Mechanism autonomously constructs a knowledge base from scratch through an iterative PSEL loop, enabling the system to surpass the limitations of handcrafted priors and costly human-annotated datasets.

---

[MetaForge: A Self-Evolving Multimodal Agent that Retrieves, Adapts, and Forges Tools On Demand](http://arxiv.org/abs/2606.01801)

- MetaForge: introduces a self-evolving multimodal agent framework that replaces static tool inventories with a closed-loop Decide-Retrieve-Adapt-Forge-Recycle process.
- The framework utilizes multi-turn GRPO training with a composite reward function to jointly optimize tool invocation, parameter adaptation, and on-demand skill synthesis.
- MetaForge incorporates a capacity-constrained tool pool management mechanism to ensure efficient tool reuse and prevent context dilution during long-term agent interaction.

---

[STaR-KV: Spatio-Temporal Adaptive Re-weighting for KV Cache Compression in GUI Vision-Language Models](http://arxiv.org/abs/2606.01790)

- STaR-KV: introduces a training-free KV cache compression framework for GUI agents that calibrates token importance using Online Spatial Profiling, Cumulative Temporal Stability Discount, and Adaptive Entropy-Based Sharpening (AEB).
- The framework addresses spatial and distributional blind spots in GUI reasoning by refining attention scores through subspace-aware mutual information, temporal decay of stale entries, and entropy-based score sharpening.
- STaR-KV achieves superior accuracy across GUI benchmarks compared to existing methods while maintaining negligible computational overhead and reducing peak GPU memory usage by nearly 40%.

---

[PlatonicNav: Unveiling Semantic Correspondence in Navigation with Platonic Topological Maps](http://arxiv.org/abs/2606.01788)

- PlatonicNav: introduces a training-free framework that unifies vision-only ObjNav, cross-modal ObjNav, and VLN by grounding language goals into a vision-built Platonic Topological Map using blind matching between independently trained Visual Encoder and Language Encoder.
- The framework utilizes a Matching Solver to align visual and language representations via their relational structure, enabling navigation without explicit cross-modal training or paired data.
- Platonic Topological Map integrates geometric and semantic distances to formulate navigation as geodesic traversal over a learned semantic manifold, facilitating efficient goal-directed movement across diverse embodiments.

---

[HarnessForge: Joint Harness and Policy Evolution for Adaptive Agent Systems](http://arxiv.org/abs/2606.01779)

- HarnessForge: introduces a meta-adaptive framework that co-evolves the external Harness (external execution interface structure) and internal Policy (internal reasoning behavior) to improve their executable compatibility.
- The framework utilizes fault-guided Harness (external execution interface structure) tailoring and Harness (external execution interface structure)-conditioned Policy (internal reasoning behavior) alignment to optimize agent systems across heterogeneous task regimes.
- By treating the agent as a coupled Harness (external execution interface structure)-Policy (internal reasoning behavior) pair, HarnessForge achieves superior performance and rollout-efficiency compared to isolated component optimization methods.

---

[Adaptive Auto-Harness: Sustained Self-Improvement for Agentic System Deployment on Open-Ended Task Streams](http://arxiv.org/abs/2606.01770)

- Adaptive Auto-Harness: introduces a framework for sustained LLM agent improvement on open-ended task streams by decomposing performance gaps into evolution loss and adaptation loss.
- The system utilizes a stateful multi-agent evolver, a harness tree with solve-time routing, and human-in-the-loop hooks to address unbounded streams, task heterogeneity, and distributional non-stationarity.
- Empirical validation across prediction markets, security competitions, and event forecasting demonstrates that the framework outperforms existing auto-harness baselines by effectively managing harness construction and per-task adaptation.

---

[TriAlign: Towards Universal Truth Consistency in Personalized LLM Alignment](http://arxiv.org/abs/2606.01755)

- TriAlign: introduces a multi-agent reinforcement learning framework to ensure universal truth consistency across personalized LLMs while maintaining user-specific response styles.
- The framework utilizes a centralized training and decentralized execution paradigm, where social-group-conditioned agents interact within a shared environment to iteratively refine responses toward truth invariance.
- TriAlign incorporates a fairness-aware objective inspired by Nash Social Welfare and an explicit cross-group consistency penalty to balance objective task performance with personalization quality.

---

[SparseX: Efficient Segment-Level KV Cache Sharing for Interleaved LLM Serving](http://arxiv.org/abs/2606.01751)

- SparseX: introduces a segment-level KV Cache sharing framework that utilizes a Sparse-Q mechanism to identify and selectively recompute key tokens, enabling efficient LLM serving under complex interleaved reuse patterns.
- The framework integrates RoPE alignment, virtual blocks, and a full+sparse hybrid attention strategy to maintain high output quality while minimizing redundant computation across multi-round chat, RAG, and agent workflows.
- SparseX-vLLM implements these components within a unified execution path, ensuring compatibility with existing PagedAttention and FlashAttention backends without requiring additional models or training.

---

[Characterization of Multi-Model Agentic AI Systems on General Tasks via Trace-Driven Simulation](http://arxiv.org/abs/2606.01725)

- GAIATrace and Vidur-Agent: introduces a comprehensive token-level trace dataset and a trace-driven simulator to characterize the heterogeneous system behaviors of complex agentic AI systems.
- The research identifies that agentic AI systems exhibit high variance in workload patterns, where Main-LLM and Sub-LLM components interact through complex planning and tool-use cycles.
- The study demonstrates that traditional per-query SLA metrics are often misaligned with agentic AI performance, highlighting the need for task-level latency optimization and dynamic resource reconfiguration.

---

[Post-Deterministic Distributed Systems: A New Foundation for Trustworthy Autonomous Infrastructure](http://arxiv.org/abs/2606.01722)

- PDDS: introduces a participant-general model for coordinating heterogeneous environments where deterministic code, stochastic models, and autonomous agents coexist by shifting the agreement target from transition equivalence to semantic coherence.
- The framework utilizes PDD (runtime policy enforcement), VAI (intent-based authorization), ASCP (decoupled reasoning and execution), SQA (semantic equivalence certification), and ESR (epistemic state persistence) to ensure system reliability.
- This model addresses failure modes in autonomous infrastructure, such as semantic drift and intent loss, by providing a taxonomy of failure classes and a structured approach to verifiable, policy-bounded agentic operations.

---

[JenBridge: Adaptive Long-Form Video Soundtracking Across Scene Transitions](http://arxiv.org/abs/2606.01703)

- JenBridge: introduces a modular framework for adaptive long-form video soundtracking that utilizes a Video Segmentation Module, a Per-Segment Music Generation Module, and an Adaptive Transition Module to ensure narrative coherence across scene transitions.
- The framework employs a Multimodal Diffusion Transformer as its generative backbone, guided by a Visual-to-Music Prompt Translator and an LLM Agent that acts as a director to select optimal transition styles from a versatile Transition Toolkit.
- JenBridge includes a comprehensive LVS Benchmark for evaluating long-form soundtracking, demonstrating superior performance in transition naturalness and overall narrative coherence compared to existing methods.

---

[HAIM: Human-AI Music Datasets for AI Music Production Tracking Benchmark](http://arxiv.org/abs/2606.01686)

- HAIM (Human-AI Music): introduces a benchmark and the MuQ-FST model to shift AI music detection from binary classification to granular, role-based tracking across the music production pipeline.
- The framework utilizes MuQ-FST, which integrates a MuQ-based music understanding backbone with a Fusion Segment Transformer to predict AI involvement across Composer, Lyricist, Vocalist, and Audio Engineer roles.
- The research provides a multi-faceted taxonomy of 196,000 tracks and demonstrates that temporal AI localization emerges as a zero-shot capability of the segment-level MuQ encoder.

---

[UniVocal: Unified Speech-Singing Code-Switching Synthesis](http://arxiv.org/abs/2606.01677)

- UniVocal: introduces a unified framework that autonomously infers vocal mode transitions from text context by utilizing a Text-to-Vocal Language Model, Refined Cent Token, Semantic Tokenizer, Flow Matching Module, HiFi-GAN Vocoder, Two-Stage Curriculum Learning, and a Scalable Data Synthesis Pipeline.
- The framework employs an interleaved Chain-of-Thought generation strategy where the Text-to-Vocal Language Model predicts Refined Cent Tokens to plan prosody before generating Semantic Tokens for content.
- UniVocal addresses data scarcity through a Scalable Data Synthesis Pipeline and achieves state-of-the-art performance on the SCSBench benchmark by leveraging a Two-Stage Curriculum Learning strategy to master autonomous speech-singing switching.

---

[Reward Design Agent for Reinforcement Learning](http://arxiv.org/abs/2606.01672)

- RDA: introduces a VLM-based agentic framework that automates reward design by decomposing tasks, visually evaluating trajectories, and iteratively refining reward code to improve instruction alignment.
- The framework utilizes a closed-loop evolutionary process where a VLM acts as a diagnostic agent to identify failure modes and provide targeted revisions to both subtask definitions and reward functions.
- By incorporating visual trajectory analysis, RDA effectively addresses reward mis-specification in complex, long-horizon robotic tasks where coarse numerical feedback is insufficient for instruction-aligned behavior.

---

[ATLAS: Agentic Test-time Learning-to-Allocate Scaling](http://arxiv.org/abs/2606.01667)

- ATLAS: introduces an agentic test-time scaling framework where an LLM Orchestrator owns the control loop end-to-end, utilizing an Explorer to generate independent candidates and Stateful Evidence Management to decide when to stop and perform synthesis.
- The framework replaces hand-specified rules with an extensible action space, allowing the LLM Orchestrator to adaptively allocate compute based on problem difficulty and evidence convergence.
- By maintaining a Candidate Pool and conditioning decisions on the full trajectory, ATLAS achieves higher accuracy and lower costs compared to fixed-workflow baselines across scientific, coding, and multimodal benchmarks.

---

[A Sheaf Framework for Strategic Multi-Agent Systems: From Consensus to Nash Equilibria](http://arxiv.org/abs/2606.01663)

- Unified Sheaf-Theoretic Game Topos Framework: introduces a categorical approach integrating event calculus, SCEL-like ensemble formation, and game-theoretic reward structures into a single Grothendieck topos of time-space histories.
- The framework utilizes a game sheaf where global sections correspond to Nash equilibria, employing cohomology to identify strategic obstructions in multi-agent systems.
- A hybrid dynamics algorithm combines sheaf Laplacian diffusion for consensus with gradient ascent on expected rewards to achieve stable multi-agent coordination.

---

[Easier to Mislead Than to Correct: Harmful and Beneficial Revision in LLM Conformity](http://arxiv.org/abs/2606.01637)

- Experimental Design for LLM Conformity Study: introduces a controlled two-round evaluation framework to measure how consensus structure and authority labels influence harmful and beneficial revisions in LLMs.
- The framework utilizes Round 1 Initial Judgment, Social Pressure Prompt, and Round 2 Final Judgment to quantify how peer agreement and authority labels drive model conformity across various reasoning datasets.
- The study evaluates the effectiveness of Chain-of-Thought and Reflect-then-revise interventions in mitigating harmful revisions while preserving beneficial ones within multi-agent LLM systems.

---

[TimeLogic Challenge @ CVPR 2026: Strong MLLMs Meet Evidence-Seeking Agents for Temporal-Logic Video Question Answering](http://arxiv.org/abs/2606.01631)

- TimeLogic Evidence-Seeking Agent: introduces an iterative, training-free framework that treats temporal-logic VideoQA as an active exploration problem using a Question Classifier, Multi-granular Sampling Toolkit, Think-Act-Observe Loop, State Memory, Gemini 3.1 Pro VLM, Minimum-step Gating, and Corpus-adaptive Budgeting.
- The system improves temporal reasoning by interleaving video frames with absolute timestamps, allowing the LLM to perform numerical comparisons on a shared time axis rather than relying on implicit frame ordering.
- By dynamically adapting sampling budgets and exploration depth to specific temporal-logic categories and corpus characteristics, the agent effectively handles complex multi-event chains and sparse visual evidence.

---

[Goal2Pixel: Grounding Goals to Pixels for Vision-Language Navigation](http://arxiv.org/abs/2606.01621)

- Goal2Pixel: introduces a pure pixel-based paradigm for VLN-CE that reformulates navigation as navigable pixel grounding, utilizing a VLM to predict goal pixels that are back-projected into 3D waypoints for execution.
- The framework incorporates ViKeyMem to maintain a compact, visibility-aware history of navigation-relevant landmarks, significantly reducing the number of VLM inference calls required for long-horizon tasks.
- Goal2Pixel employs learnable semantic embeddings and coordinate-aware auxiliary losses to improve the alignment between VLM reasoning and the geometric requirements of pixel-based navigation.

---

[RESKILL: Reconciling Skill Creation with Policy Optimization in Agentic RL](http://arxiv.org/abs/2606.01619)

- RESKILL: introduces an RL-in-the-loop framework that reconciles skill evolution with policy learning by embedding GRPO Training, Skill Creator, Experience Reservoir, Assertion-based Grading, Thompson Sampling, and Verifier.
- The framework utilizes GRPO Training to perform within-group skill testing, allowing for controlled comparison of skill versions while simultaneously optimizing the policy.
- By integrating an assertion-driven Skill Creator and adaptive Thompson Sampling, RESKILL enables automated, policy-aligned skill co-evolution that outperforms static skill-based RL methods.

---

[EvoPool: Evolutionary Programmatic Annotation for Label-Efficient Specialized Supervision](http://arxiv.org/abs/2606.01617)

- EvoPool: introduces an evolutionary multi-agent framework that iteratively authors and refines a pool of executable annotators to provide label-efficient supervision in specialized domains.
- The framework utilizes three specialized LLM agents—Generator, Improver, and Refiner—to propose and optimize annotators under a deterministic selection gate that enforces fitness, diversity, and marginal contribution.
- EvoAgg combines annotator votes with text-aware features to produce soft training labels, achieving superior cost-accuracy tradeoffs compared to direct LLM annotation or one-shot synthesis methods.

---

[TechGraphRAG: An Agentic Graph-Augmented RAG Framework for Technical Literature Reasoning](http://arxiv.org/abs/2606.01613)

- TechGraphRAG: introduces an agentic RAG framework that utilizes a 13-step autonomous pipeline to perform evidence-gated retrieval, knowledge graph traversal, and self-correcting generation for technical literature reasoning.
- The architecture integrates hybrid local retrieval with external academic search and Neo4j-based relational context to ensure grounded, verifiable, and high-precision responses.
- The system employs a multi-dimensional evidence sufficiency scoring rubric and iterative agentic loops to dynamically determine when to supplement internal knowledge with external academic sources.

---

[Embedding Semantic Risk into Distance Fields and CBFs for Online Monocular Safe Control](http://arxiv.org/abs/2606.01605)

- Semantic-aware safe control framework: introduces an online monocular perception-to-control pipeline that embeds semantic risk directly into the Euclidean Signed Distance Field (ESDF) to inform Control Barrier Function (CBF) safety constraints.
- The framework utilizes a MASt3R-SLAM geometry backbone and a semantic segmentation branch to generate a temporally stabilized geometric-semantic map, which is then processed through a TSDF-to-ESDF pipeline with class-dependent inflation.
- A CBF-QP safety filter uses the resulting semantic-aware ESDF to minimally modify reference control inputs, enabling efficient, risk-sensitive navigation and teleoperation at 10–20 Hz.

---

[Identifying High-Confidence Social Biases in LLMs for Trustworthy Conversational Tutoring Agents](http://arxiv.org/abs/2606.01584)

- Biased Conversational Tutoring Dataset generation framework: introduces a method to evaluate social biases in LLMs by regenerating authentic tutoring dialogues with controlled stereotypical turns derived from the StereoSet benchmark.
- The study evaluates state-of-the-art LLMs on their ability to detect stereotypical biases in naturalistic instructional contexts, revealing that models exhibit significant overconfidence even when their judgments are incorrect.
- The research establishes that model overconfidence in tutoring agents acts as a bias amplification mechanism, reinforcing incorrect reasoning and potentially misleading learners in educational settings.

---

[Agent System Operations: Categorization, Challenges, and Future Directions](http://arxiv.org/abs/2606.01581)

- AgentOps: introduces a comprehensive operational framework for agent systems, categorizing anomalies into intra-agent and inter-agent types across the pre-execution, execution, and post-execution lifecycle.
- The framework addresses the stochastic nature of LLMs by integrating Monitoring, Anomaly Detection, Root Cause Localization, and Resolution to maintain system reliability and safety.
- AgentOps distinguishes itself from traditional operations by focusing on semantic, trajectory-level failures and introducing execution-level interventions like checkpointing and state rollbacks.

---

[Defenses &amp; Enablers For Skill Injection Attacks on Terminal Based Agents](http://arxiv.org/abs/2606.01567)

- Guardian Defense Architecture: introduces a mediator-based defense strategy for LLM agents to mitigate skill injection attacks by filtering or rewriting untrusted skill documentation.
- The framework evaluates two primary defense variants: a dynamic guardian that intercepts skill calls at runtime and a static guardian that sanitizes skill files during build time.
- The research demonstrates that these guardian-based defenses significantly reduce attack success rates while preserving task utility, even when faced with sophisticated attack reframing techniques.

---

[Hierarchical Semantic-Augmented Navigation: Optimal Transport and Graph-Driven Reasoning for Vision-Language Navigation](http://arxiv.org/abs/2606.01565)

- HSAN (Hierarchical Semantic-Augmented Navigation): introduces a framework for Vision-Language Navigation in Continuous Environments that integrates a Hierarchical Semantic Scene Graph Builder, an OT-based Topological Planner, and a Graph-aware Low-level Control agent.
- The framework utilizes a Hierarchical Semantic Scene Graph Builder to capture multi-level environmental context, which is then processed by an OT-based Topological Planner to select optimal navigation goals.
- A Graph-aware Low-level Control agent, supported by a GCN, executes high-level plans while ensuring robust obstacle avoidance in continuous 3D indoor spaces.

---

[Everywhere Learning: Artificial Intelligence with Pointwise Constraints](http://arxiv.org/abs/2606.01557)

- Everywhere Learning: introduces a constrained learning paradigm that enforces requirements pointwise over the data distribution rather than in expectation.
- The framework utilizes dual variables to reweight the data distribution, identifying regions where constraints are difficult to satisfy and enabling sparse relaxation through dual clipping.
- The approach provides PACC learnability guarantees by bounding the dual sensitivity, ensuring that the model satisfies constraints almost everywhere even when trained on finite samples.

---

[RoleCDE: Benchmarking and Mitigating Role–Alignment Trade-offs in Role-Playing Agents](http://arxiv.org/abs/2606.01552)

- RoleCDE: introduces a large-scale benchmark designed to evaluate Role-Playing Agents under structured conflicts between role-specific values and alignment-oriented constraints.
- The framework identifies a "Role-Value Decoupling" phenomenon where LLMs systematically default to alignment-consistent decisions despite explicit role conditioning.
- Targeted fine-tuning using RoleCDE-derived data effectively mitigates this decoupling by shifting decision tendencies toward role-consistent behavior without degrading general reasoning or role-playing capabilities.

---

[Multi-Agent Computer Use](http://arxiv.org/abs/2606.01533)

- MACU (Multi-Agent Computer Use): introduces a multi-agent framework that decomposes complex computer tasks into a directed acyclic graph (DAG) of subtasks executed in parallel by homogeneous CUA subagents.
- The system utilizes a manager agent to dynamically replan the DAG based on subagent feedback, effectively handling partial observability and long-horizon task requirements.
- MACU improves success rates and wall-clock efficiency across desktop and web navigation benchmarks by leveraging parallel execution and continuous replanning.

---

[Joint Agent Memory and Exploration Learning](http://arxiv.org/abs/2606.01528)

- JAMEL (Joint Agent Memory and Exploration Learning): introduces a framework that jointly trains agentic latent memory and exploration policy through novelty-driven interaction.
- The framework utilizes deterministic code coverage signals to provide annotation-free supervision for compressing interaction histories into compact latent memory tokens.
- By integrating memory compression directly into the exploration loop, the agent avoids context truncation and achieves efficient, sustained exploration in open-ended GUI environments.

---

[Type-Error Ablation and AI Coding Agents](http://arxiv.org/abs/2606.01522)

- Type-Error Ablation and AI Coding Agents: introduces an experimental framework to evaluate how varying levels of type-error message detail influence the repair performance of LLMs.
- The framework utilizes Shplait to systematically ablate error information across four modes—untyped, minimal, proximate, and all—to measure agent convergence on correct code.
- Empirical results demonstrate that increased type-error detail consistently improves LLM fix rates, providing strong evidence that type-directed feedback is a valuable signal for automated program repair.

---

[Structuring agentic AI for HPC code modernization](http://arxiv.org/abs/2606.08710)

- NMAP-RKPM Modernization Framework: introduces a structured, agentic AI methodology for migrating legacy Fortran HPC code to C++ using a three-layer bridge architecture and a "hand-holding" prompting strategy.
- The framework utilizes a "Manager-Worker" agent paradigm to decompose complex subroutines into manageable tasks, ensuring strict semantic fidelity through continuous integration and validation.
- By employing custom C++ array wrappers and explicit global state encapsulation, the approach enables LLMs to perform accurate, incremental code translation while avoiding common pitfalls like context collapse and ghost bugs.

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
